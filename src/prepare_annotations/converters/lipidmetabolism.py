"""
Convert Lipid Metabolism SQLite data to unified annotation schema.

Outputs three Parquet files:
- annotations.parquet: Variant-level facts
- studies.parquet: Per-study evidence  
- weights.parquet: Curator-defined scoring
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
from eliot import start_action


def normalize_genotype(genotype: str | None) -> list[str] | None:
    """Normalize genotype to alphabetical list of alleles (e.g., 'GA' -> ['A', 'G'])."""
    if genotype is None:
        return None
    return sorted(list(genotype))


def derive_state_from_weight(weight: float | None) -> str:
    """Derive semantic state from weight value."""
    if weight is None:
        return "neutral"
    if weight > 0:
        return "protective"
    elif weight < 0:
        return "risk"
    return "neutral"


def convert_lipidmetabolism_annotations(db_path: Path) -> pl.LazyFrame:
    """
    Convert Lipid Metabolism to annotations.parquet format.
    
    Schema: rsid, module, gene, phenotype, category
    """
    with start_action(action_type="convert_lipidmetabolism_annotations", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        # Get unique rsid -> gene mappings from rsids table
        query = """
        SELECT DISTINCT
            rsid,
            gene
        FROM rsids
        WHERE rsid IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("lipidmetabolism").alias("module"),
            pl.lit("lipid_metabolism").alias("phenotype"),
            pl.lit("lipids").alias("category"),  # Default category for this module
        ).select("rsid", "module", "gene", "phenotype", "category")


def convert_lipidmetabolism_studies(db_path: Path) -> pl.LazyFrame:
    """
    Convert Lipid Metabolism to studies.parquet format.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    
    Uses the rsids table which has pmids, population, p_value, and rsid_conclusion.
    """
    with start_action(action_type="convert_lipidmetabolism_studies", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        # The rsids table contains study information
        query = """
        SELECT DISTINCT
            rsid,
            pmids as pmid,
            population,
            p_value,
            rsid_conclusion as conclusion
        FROM rsids
        WHERE rsid IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("lipidmetabolism").alias("module"),
            pl.lit(None).cast(pl.Utf8).alias("study_design"),
        ).select("rsid", "module", "pmid", "population", "p_value", "conclusion", "study_design")


def convert_lipidmetabolism_weights(
    db_path: Path,
    curator: str = "just-dna-seq",
    method: str = "literature_review",
) -> pl.LazyFrame:
    """
    Convert Lipid Metabolism to weights.parquet format.
    
    Schema: rsid, genotype, module, weight, state, priority, conclusion, curator, method
    
    The weight table already has genotype column, so we just need to normalize it.
    """
    with start_action(
        action_type="convert_lipidmetabolism_weights",
        db_path=str(db_path),
    ):
        conn = sqlite3.connect(db_path)
        
        # Load weights from weight table
        query = """
        SELECT 
            rsid,
            genotype,
            weight,
            state,
            genotype_specific_conclusion as conclusion
        FROM weight
        WHERE rsid IS NOT NULL
        """
        weights_raw = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        # Normalize genotype and convert weight to float
        normalized = weights_raw.with_columns(
            # Normalize genotype alphabetically as a list
            pl.when(pl.col("genotype").str.len_chars() == 2)
            .then(
                pl.when(pl.col("genotype").str.slice(0, 1) > pl.col("genotype").str.slice(1, 1))
                .then(
                    pl.concat_list([
                        pl.col("genotype").str.slice(1, 1),
                        pl.col("genotype").str.slice(0, 1)
                    ])
                )
                .otherwise(
                    pl.concat_list([
                        pl.col("genotype").str.slice(0, 1),
                        pl.col("genotype").str.slice(1, 1)
                    ])
                )
            )
            .otherwise(
                # Fallback: split and handle non-empty parts if not 2 chars
                pl.col("genotype").str.split("").list.slice(1, -1)
            )
            .alias("genotype"),
            # Convert weight to float (stored as TEXT in SQLite)
            pl.col("weight").cast(pl.Float64, strict=False).alias("weight"),
            # Add module, curator, method
            pl.lit("lipidmetabolism").alias("module"),
            pl.lit(curator).alias("curator"),
            pl.lit(method).alias("method"),
            pl.lit(None).cast(pl.Utf8).alias("priority"),
        )
        
        # Derive state from weight if not already present
        result = normalized.with_columns(
            pl.when(pl.col("state").is_not_null() & (pl.col("state") != ""))
            .then(pl.col("state"))
            .when(pl.col("weight") > 0)
            .then(pl.lit("protective"))
            .when(pl.col("weight") < 0)
            .then(pl.lit("risk"))
            .otherwise(pl.lit("neutral"))
            .alias("state"),
        )
        
        # Deduplicate by (rsid, genotype, module)
        result = (
            result
            .group_by(["rsid", "genotype", "module"])
            .agg([
                pl.col("weight").first(),
                pl.col("state").first(),
                pl.col("priority").first(),
                pl.col("conclusion").first(),
                pl.col("curator").first(),
                pl.col("method").first(),
            ])
            .select(
                "rsid",
                "genotype",
                "module",
                "weight",
                "state",
                "priority",
                "conclusion",
                "curator",
                "method",
            )
        )
        
        return result


def convert_lipidmetabolism(
    db_path: Path,
    output_dir: Path,
    curator: str = "just-dna-seq",
    method: str = "literature_review",
) -> dict[str, Path]:
    """
    Convert Lipid Metabolism SQLite to unified annotation schema.
    
    Outputs three Parquet files to output_dir:
    - annotations.parquet
    - studies.parquet
    - weights.parquet
    
    Args:
        db_path: Path to Lipid Metabolism SQLite database
        output_dir: Directory for output Parquet files
        curator: Curator name for weights
        method: Curation method for weights
        
    Returns:
        Dictionary mapping table names to output paths
    """
    with start_action(
        action_type="convert_lipidmetabolism",
        db_path=str(db_path),
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # Convert annotations
        annotations = convert_lipidmetabolism_annotations(db_path)
        annotations_path = output_dir / "annotations.parquet"
        annotations.sink_parquet(annotations_path, engine="streaming")
        outputs["annotations"] = annotations_path
        
        # Convert studies
        studies = convert_lipidmetabolism_studies(db_path)
        studies_path = output_dir / "studies.parquet"
        studies.sink_parquet(studies_path, engine="streaming")
        outputs["studies"] = studies_path
        
        # Convert weights
        weights = convert_lipidmetabolism_weights(
            db_path,
            curator=curator,
            method=method,
        )
        weights_path = output_dir / "weights.parquet"
        weights.sink_parquet(weights_path, engine="streaming")
        outputs["weights"] = weights_path
        
        return outputs

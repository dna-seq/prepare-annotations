"""
Convert Superhuman SQLite data to unified annotation schema.

This module has qualitative annotations (superability, adverse_effects) rather than
numeric weights. The weight column will be NULL but the state will be derived from
the presence of superability vs adverse_effects.

Outputs three Parquet files:
- annotations.parquet: Variant-level facts
- studies.parquet: Per-study evidence  
- weights.parquet: Curator-defined scoring (with NULL weights)
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


def convert_superhuman_annotations(db_path: Path) -> pl.LazyFrame:
    """
    Convert Superhuman to annotations.parquet format.
    
    Schema: rsid, module, gene, phenotype, category
    
    Uses superability as the phenotype/category.
    """
    with start_action(action_type="convert_superhuman_annotations", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT DISTINCT
            rsid,
            gene,
            superability
        FROM superhuman
        WHERE rsid IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("superhuman").alias("module"),
            pl.lit("elite_performance").alias("phenotype"),
            pl.col("superability").alias("category"),
        ).select("rsid", "module", "gene", "phenotype", "category")


def convert_superhuman_studies(db_path: Path) -> pl.LazyFrame:
    """
    Convert Superhuman to studies.parquet format.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    
    The references column contains literature references.
    """
    with start_action(action_type="convert_superhuman_studies", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT DISTINCT
            rsid,
            "references" as pmid
        FROM superhuman
        WHERE rsid IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("superhuman").alias("module"),
            pl.lit(None).cast(pl.Utf8).alias("population"),
            pl.lit(None).cast(pl.Utf8).alias("p_value"),
            pl.lit(None).cast(pl.Utf8).alias("conclusion"),
            pl.lit(None).cast(pl.Utf8).alias("study_design"),
        ).select("rsid", "module", "pmid", "population", "p_value", "conclusion", "study_design")


def convert_superhuman_weights(
    db_path: Path,
    curator: str = "just-dna-seq",
    method: str = "literature_review",
) -> pl.LazyFrame:
    """
    Convert Superhuman to weights.parquet format.
    
    Schema: rsid, genotype, module, weight, state, priority, conclusion, curator, method
    
    This module has no numeric weights. State is derived from:
    - "protective" if superability is present
    - "risk" if adverse_effects is present  
    - "neutral" otherwise
    
    Conclusion combines superability and adverse_effects.
    """
    with start_action(
        action_type="convert_superhuman_weights",
        db_path=str(db_path),
    ):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            rsid,
            genotype,
            superability,
            adverse_effects
        FROM superhuman
        WHERE rsid IS NOT NULL AND genotype IS NOT NULL
        """
        weights_raw = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        # Normalize genotype and derive state
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
            # No numeric weight for this module
            pl.lit(None).cast(pl.Float64).alias("weight"),
            # Derive state from superability/adverse_effects
            pl.when(pl.col("superability").is_not_null() & (pl.col("superability") != ""))
            .then(pl.lit("protective"))
            .when(pl.col("adverse_effects").is_not_null() & (pl.col("adverse_effects") != ""))
            .then(pl.lit("risk"))
            .otherwise(pl.lit("neutral"))
            .alias("state"),
            # Combine superability and adverse_effects as conclusion
            pl.when(
                (pl.col("superability").is_not_null() & (pl.col("superability") != "")) &
                (pl.col("adverse_effects").is_not_null() & (pl.col("adverse_effects") != ""))
            )
            .then(pl.col("superability") + " | Adverse: " + pl.col("adverse_effects"))
            .when(pl.col("superability").is_not_null() & (pl.col("superability") != ""))
            .then(pl.col("superability"))
            .when(pl.col("adverse_effects").is_not_null() & (pl.col("adverse_effects") != ""))
            .then(pl.lit("Adverse: ") + pl.col("adverse_effects"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("conclusion"),
            # Add module, curator, method
            pl.lit("superhuman").alias("module"),
            pl.lit(curator).alias("curator"),
            pl.lit(method).alias("method"),
            pl.lit(None).cast(pl.Utf8).alias("priority"),
        )
        
        # Deduplicate by (rsid, genotype, module)
        result = (
            normalized
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


def convert_superhuman(
    db_path: Path,
    output_dir: Path,
    curator: str = "just-dna-seq",
    method: str = "literature_review",
) -> dict[str, Path]:
    """
    Convert Superhuman SQLite to unified annotation schema.
    
    Outputs three Parquet files to output_dir:
    - annotations.parquet
    - studies.parquet
    - weights.parquet
    
    Args:
        db_path: Path to Superhuman SQLite database
        output_dir: Directory for output Parquet files
        curator: Curator name for weights
        method: Curation method for weights
        
    Returns:
        Dictionary mapping table names to output paths
    """
    with start_action(
        action_type="convert_superhuman",
        db_path=str(db_path),
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # Convert annotations
        annotations = convert_superhuman_annotations(db_path)
        annotations_path = output_dir / "annotations.parquet"
        annotations.collect().write_parquet(annotations_path)
        outputs["annotations"] = annotations_path
        
        # Convert studies
        studies = convert_superhuman_studies(db_path)
        studies_path = output_dir / "studies.parquet"
        studies.collect().write_parquet(studies_path)
        outputs["studies"] = studies_path
        
        # Convert weights
        weights = convert_superhuman_weights(
            db_path,
            curator=curator,
            method=method,
        )
        weights_path = output_dir / "weights.parquet"
        weights.collect().write_parquet(weights_path)
        outputs["weights"] = weights_path
        
        return outputs

"""
Convert Coronary Disease SQLite data to unified annotation schema.

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


def normalize_genotype(genotype: str | None) -> str | None:
    """Normalize genotype to alphabetical order (e.g., 'GA' -> 'AG')."""
    if genotype is None or len(genotype) != 2:
        return genotype
    return "".join(sorted(genotype))


def convert_coronary_annotations(db_path: Path) -> pl.LazyFrame:
    """
    Convert Coronary Disease to annotations.parquet format.
    
    Schema: rsid, module, gene, phenotype, category
    """
    with start_action(action_type="convert_coronary_annotations", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT DISTINCT
            rsID as rsid,
            Gene as gene
        FROM coronary_disease
        WHERE rsID IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("coronary").alias("module"),
            pl.lit("coronary_disease").alias("phenotype"),
            pl.lit("cardiovascular").alias("category"),
        ).select("rsid", "module", "gene", "phenotype", "category")


def convert_coronary_studies(db_path: Path) -> pl.LazyFrame:
    """
    Convert Coronary Disease to studies.parquet format.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    """
    with start_action(action_type="convert_coronary_studies", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT DISTINCT
            rsID as rsid,
            PMID as pmid,
            Population as population,
            P_value as p_value,
            Conclusion as conclusion,
            GWAS_study_design as study_design
        FROM coronary_disease
        WHERE rsID IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("coronary").alias("module"),
        ).select("rsid", "module", "pmid", "population", "p_value", "conclusion", "study_design")


def convert_coronary_weights(
    db_path: Path,
    curator: str = "just-dna-seq",
    method: str = "gwas_literature",
) -> pl.LazyFrame:
    """
    Convert Coronary Disease to weights.parquet format.
    
    Schema: rsid, genotype, module, weight, state, priority, conclusion, curator, method
    """
    with start_action(
        action_type="convert_coronary_weights",
        db_path=str(db_path),
    ):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            rsID as rsid,
            Genotype as genotype,
            Weight as weight,
            state,
            Conclusion as conclusion
        FROM coronary_disease
        WHERE rsID IS NOT NULL AND Genotype IS NOT NULL
        """
        weights_raw = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        # Normalize genotype and convert weight to float
        normalized = weights_raw.with_columns(
            # Normalize genotype alphabetically
            pl.when(pl.col("genotype").str.len_chars() == 2)
            .then(
                pl.when(pl.col("genotype").str.slice(0, 1) > pl.col("genotype").str.slice(1, 1))
                .then(pl.col("genotype").str.slice(1, 1) + pl.col("genotype").str.slice(0, 1))
                .otherwise(pl.col("genotype"))
            )
            .otherwise(pl.col("genotype"))
            .alias("genotype"),
            # Convert weight to float (stored as TEXT in SQLite)
            pl.col("weight").cast(pl.Float64, strict=False).alias("weight"),
            # Add module, curator, method
            pl.lit("coronary").alias("module"),
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


def convert_coronary(
    db_path: Path,
    output_dir: Path,
    curator: str = "just-dna-seq",
    method: str = "gwas_literature",
) -> dict[str, Path]:
    """
    Convert Coronary Disease SQLite to unified annotation schema.
    
    Outputs three Parquet files to output_dir:
    - annotations.parquet
    - studies.parquet
    - weights.parquet
    
    Args:
        db_path: Path to Coronary Disease SQLite database
        output_dir: Directory for output Parquet files
        curator: Curator name for weights
        method: Curation method for weights
        
    Returns:
        Dictionary mapping table names to output paths
    """
    with start_action(
        action_type="convert_coronary",
        db_path=str(db_path),
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # Convert annotations
        annotations = convert_coronary_annotations(db_path)
        annotations_path = output_dir / "annotations.parquet"
        annotations.collect().write_parquet(annotations_path)
        outputs["annotations"] = annotations_path
        
        # Convert studies
        studies = convert_coronary_studies(db_path)
        studies_path = output_dir / "studies.parquet"
        studies.collect().write_parquet(studies_path)
        outputs["studies"] = studies_path
        
        # Convert weights
        weights = convert_coronary_weights(
            db_path,
            curator=curator,
            method=method,
        )
        weights_path = output_dir / "weights.parquet"
        weights.collect().write_parquet(weights_path)
        outputs["weights"] = weights_path
        
        return outputs

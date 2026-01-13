"""
Convert VO2max SQLite data to unified annotation schema.

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


def convert_vo2max_annotations(db_path: Path) -> pl.LazyFrame:
    """
    Convert VO2max to annotations.parquet format.
    
    Schema: rsid, module, gene, phenotype, category
    """
    with start_action(action_type="convert_vo2max_annotations", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        # Get unique rsid -> gene mappings from rsid table
        query = """
        SELECT DISTINCT
            rsid,
            gene
        FROM rsid
        WHERE rsid IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("vo2max").alias("module"),
            pl.lit("athletic_performance").alias("phenotype"),
            pl.lit("vo2max").alias("category"),
        ).select("rsid", "module", "gene", "phenotype", "category")


def convert_vo2max_studies(db_path: Path) -> pl.LazyFrame:
    """
    Convert VO2max to studies.parquet format.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    
    Uses the rsid table which has pmids, population, p_value, and rsid_conclusion.
    """
    with start_action(action_type="convert_vo2max_studies", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT DISTINCT
            rsid,
            pmids as pmid,
            population,
            p_value,
            rsid_conclusion as conclusion
        FROM rsid
        WHERE rsid IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("vo2max").alias("module"),
            pl.lit(None).cast(pl.Utf8).alias("study_design"),
        ).select("rsid", "module", "pmid", "population", "p_value", "conclusion", "study_design")


def convert_vo2max_weights(
    db_path: Path,
    curator: str = "just-dna-seq",
    method: str = "literature_review",
) -> pl.LazyFrame:
    """
    Convert VO2max to weights.parquet format.
    
    Schema: rsid, genotype, module, weight, state, priority, conclusion, curator, method
    
    The genotype_weights table has the weight data.
    Note: Column is 'rsID' not 'rsid' in genotype_weights table.
    """
    with start_action(
        action_type="convert_vo2max_weights",
        db_path=str(db_path),
    ):
        conn = sqlite3.connect(db_path)
        
        # Load weights - note rsID column name
        query = """
        SELECT 
            rsID as rsid,
            genotype,
            weight,
            state,
            genotype_specific_conclusion as conclusion
        FROM genotype_weights
        WHERE rsID IS NOT NULL
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
            pl.lit("vo2max").alias("module"),
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


def convert_vo2max(
    db_path: Path,
    output_dir: Path,
    curator: str = "just-dna-seq",
    method: str = "literature_review",
) -> dict[str, Path]:
    """
    Convert VO2max SQLite to unified annotation schema.
    
    Outputs three Parquet files to output_dir:
    - annotations.parquet
    - studies.parquet
    - weights.parquet
    
    Args:
        db_path: Path to VO2max SQLite database
        output_dir: Directory for output Parquet files
        curator: Curator name for weights
        method: Curation method for weights
        
    Returns:
        Dictionary mapping table names to output paths
    """
    with start_action(
        action_type="convert_vo2max",
        db_path=str(db_path),
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # Convert annotations
        annotations = convert_vo2max_annotations(db_path)
        annotations_path = output_dir / "annotations.parquet"
        annotations.collect().write_parquet(annotations_path)
        outputs["annotations"] = annotations_path
        
        # Convert studies
        studies = convert_vo2max_studies(db_path)
        studies_path = output_dir / "studies.parquet"
        studies.collect().write_parquet(studies_path)
        outputs["studies"] = studies_path
        
        # Convert weights
        weights = convert_vo2max_weights(
            db_path,
            curator=curator,
            method=method,
        )
        weights_path = output_dir / "weights.parquet"
        weights.collect().write_parquet(weights_path)
        outputs["weights"] = weights_path
        
        return outputs

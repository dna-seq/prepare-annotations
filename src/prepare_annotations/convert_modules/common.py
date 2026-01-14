from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl


def convert_longevitymap_data(
    db_path: Path,
    ensembl_cache: Path | None = None,
) -> pl.LazyFrame:
    """
    Converts an OakVar module SQLite database to an extended weights lazy Polars frame.
    Optionally joins with Ensembl genome data from cache.

    Args:
        db_path: Path to the SQLite database
        ensembl_cache: Path to the Ensembl cache directory containing parquet files.
                      If provided, joins with genome data and filters by matching alleles.

    Returns:
        A Polars LazyFrame containing the extended weights, optionally with genome data
    """
    conn = sqlite3.connect(db_path)
    # Get weights from allele_weights and categories
    weights_query = """
    SELECT rsid, allele, state, zygosity, weight, priority, categories.name
    FROM allele_weights
    JOIN categories ON categories.id = allele_weights.category_id
    """
    weights = pl.read_database(weights_query, connection=conn).lazy()

    # Get variants and populations
    variants_query = """
    SELECT variant.identifier as rsid, variant.study_design, variant.conclusions, variant.association,
           variant.gender, variant.quickref, variant.quickyear, variant.quickpubmed,
           population.name as population_name
    FROM variant
    JOIN population ON variant.population_id = population.id
    """
    variants = (
        pl.read_database(variants_query, connection=conn)
        .with_columns((pl.col("association") == "significant").alias("is_significant"))
        .drop("association")
        .lazy()
    )

    conn.close()
    
    # Join weights and variants to create extended weights
    extended_weights = weights.join(variants, on="rsid")
    
    # If ensembl_cache is provided, join with genome data
    if ensembl_cache is not None:
        if not ensembl_cache.exists():
            print(f"WARNING: Ensembl cache directory does not exist: {ensembl_cache}")
            print("Returning extended weights without genome data")
            return extended_weights
        
        parquet_pattern = str(ensembl_cache / "*.parquet")
        # Load genome data from all parquet files in the cache
        genome = pl.scan_parquet(parquet_pattern, low_memory=True)
        
        # Join with extended weights and filter by matching alleles
        return (
            genome.rename({"id": "rsid"})
            .join(extended_weights, on="rsid")
            .filter(pl.col("alts").list.contains(pl.col("allele")))
        )
    
    return extended_weights

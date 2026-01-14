"""
Convert Drugs TSV data to unified annotation schema.

The drugs module uses a TSV file from PharmGKB with pharmacogenomic associations.

Outputs three Parquet files:
- annotations.parquet: Variant-level facts
- studies.parquet: Per-study evidence  
- weights.parquet: Curator-defined scoring
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from eliot import start_action


def extract_rsid(variant_str: str | None) -> str | None:
    """Extract rsid from Variant/Haplotypes string."""
    if variant_str is None:
        return None
    # The variant column may contain rsid directly or need extraction
    # Most entries are like "rs9839376"
    if variant_str.startswith("rs"):
        return variant_str.split(",")[0].strip()  # Take first rsid if multiple
    return variant_str


def convert_drugs_annotations(tsv_path: Path) -> pl.LazyFrame:
    """
    Convert Drugs TSV to annotations.parquet format.
    
    Schema: rsid, module, gene, phenotype, category
    
    Uses Drug(s) as category and Phenotype Category as phenotype.
    """
    with start_action(action_type="convert_drugs_annotations", tsv_path=str(tsv_path)):
        df = pl.read_csv(tsv_path, separator="\t").lazy()
        
        result = df.select(
            pl.col("Variant/Haplotypes").alias("rsid"),
            pl.lit("drugs").alias("module"),
            pl.lit(None).cast(pl.Utf8).alias("gene"),  # Gene not in this dataset
            pl.col("Phenotype Category").alias("phenotype"),
            pl.col("Drug(s)").alias("category"),
        ).filter(
            pl.col("rsid").is_not_null() & pl.col("rsid").str.starts_with("rs")
        ).unique(subset=["rsid", "module", "category"])
        
        return result


def convert_drugs_studies(tsv_path: Path) -> pl.LazyFrame:
    """
    Convert Drugs TSV to studies.parquet format.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    
    Uses Sentence as conclusion and P Value from the TSV.
    """
    with start_action(action_type="convert_drugs_studies", tsv_path=str(tsv_path)):
        df = pl.read_csv(tsv_path, separator="\t").lazy()
        
        result = df.select(
            pl.col("Variant/Haplotypes").alias("rsid"),
            pl.lit("drugs").alias("module"),
            pl.lit(None).cast(pl.Utf8).alias("pmid"),  # PMID not directly in this dataset
            pl.lit(None).cast(pl.Utf8).alias("population"),
            pl.col("P Value").cast(pl.Utf8).alias("p_value"),
            pl.col("Sentence").alias("conclusion"),
            pl.lit(None).cast(pl.Utf8).alias("study_design"),
        ).filter(
            pl.col("rsid").is_not_null() & pl.col("rsid").str.starts_with("rs")
        ).unique(subset=["rsid", "module", "conclusion"])
        
        return result


def convert_drugs_weights(
    tsv_path: Path,
    curator: str = "PharmGKB",
    method: str = "pharmacogenomics_db",
) -> pl.LazyFrame:
    """
    Convert Drugs TSV to weights.parquet format.
    
    Schema: rsid, genotype, module, weight, state, priority, conclusion, curator, method
    
    This module doesn't have explicit weights. State is derived from Significance field.
    Genotype is not available in this dataset, so we use a placeholder.
    """
    with start_action(
        action_type="convert_drugs_weights",
        tsv_path=str(tsv_path),
    ):
        df = pl.read_csv(tsv_path, separator="\t").lazy()
        
        result = df.select(
            pl.col("Variant/Haplotypes").alias("rsid"),
            pl.concat_list([pl.lit("?"), pl.lit("?")]).alias("genotype"),  # Genotype not in dataset
            pl.lit("drugs").alias("module"),
            pl.lit(None).cast(pl.Float64).alias("weight"),
            # Derive state from Significance
            pl.when(pl.col("Significance").str.to_lowercase() == "yes")
            .then(pl.lit("significant"))
            .otherwise(pl.lit("not_significant"))
            .alias("state"),
            pl.lit(None).cast(pl.Utf8).alias("priority"),
            # Combine drug and sentence as conclusion
            (pl.col("Drug(s)") + ": " + pl.col("Sentence")).alias("conclusion"),
            pl.lit(curator).alias("curator"),
            pl.lit(method).alias("method"),
        ).filter(
            pl.col("rsid").is_not_null() & pl.col("rsid").str.starts_with("rs")
        ).unique(subset=["rsid", "genotype", "module", "conclusion"])
        
        return result


def convert_drugs(
    tsv_path: Path,
    output_dir: Path,
    curator: str = "PharmGKB",
    method: str = "pharmacogenomics_db",
) -> dict[str, Path]:
    """
    Convert Drugs TSV to unified annotation schema.
    
    Outputs three Parquet files to output_dir:
    - annotations.parquet
    - studies.parquet
    - weights.parquet
    
    Args:
        tsv_path: Path to Drugs TSV file (annotation_tab.tsv)
        output_dir: Directory for output Parquet files
        curator: Curator name for weights
        method: Curation method for weights
        
    Returns:
        Dictionary mapping table names to output paths
    """
    with start_action(
        action_type="convert_drugs",
        tsv_path=str(tsv_path),
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # Convert annotations
        annotations = convert_drugs_annotations(tsv_path)
        annotations_path = output_dir / "annotations.parquet"
        annotations.collect().write_parquet(annotations_path)
        outputs["annotations"] = annotations_path
        
        # Convert studies
        studies = convert_drugs_studies(tsv_path)
        studies_path = output_dir / "studies.parquet"
        studies.collect().write_parquet(studies_path)
        outputs["studies"] = studies_path
        
        # Convert weights
        weights = convert_drugs_weights(
            tsv_path,
            curator=curator,
            method=method,
        )
        weights_path = output_dir / "weights.parquet"
        weights.collect().write_parquet(weights_path)
        outputs["weights"] = weights_path
        
        return outputs

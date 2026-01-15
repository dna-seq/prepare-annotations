"""
Standalone preparation API for genomic data.

This module provides the PreparationPipelines class - a high-level API for
downloading, converting, and uploading genomic data without requiring Dagster.

Usage:
    from prepare_annotations.pipelines import PreparationPipelines
    
    # Download and convert Ensembl data
    result = PreparationPipelines.download_ensembl(species="homo_sapiens")
    
    # Download ClinVar
    result = PreparationPipelines.download_clinvar(assembly="GRCh38_ensembl")
    
    # Upload to HuggingFace
    result = PreparationPipelines.upload_dataset_to_hf("ensembl")

For Dagster-based pipelines with full lineage tracking, use:
    from prepare_annotations.definitions import defs
"""
# Re-export from the implementation module
from prepare_annotations.pipelines.logic import (
    PreparationPipelines,
    split_parquets,
    compute_rsid_coordinates,
    prepare_vcf_source,
)

# Also export the Dagster defs for convenience
from prepare_annotations.pipelines.definitions import defs

__all__ = [
    "PreparationPipelines",
    "split_parquets",
    "compute_rsid_coordinates", 
    "prepare_vcf_source",
    "defs",
]

"""
Dagster assets for genomic data preparation.

This module contains Dagster asset definitions organized by domain:
- ensembl: Ensembl VCF download and conversion assets
- modules: OakVar module conversion assets
"""
from prepare_annotations.assets.ensembl import (
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
    ENSEMBL_VCF_PARTITIONS,
    download_retry_policy,
)
from prepare_annotations.assets.modules import (
    ensembl_variations_source,
    longevitymap_sqlite,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
    module_assets,
)

# All assets for easy import
all_assets = [
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
    ensembl_variations_source,
    longevitymap_sqlite,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
]

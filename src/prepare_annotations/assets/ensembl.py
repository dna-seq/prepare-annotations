"""
Dagster assets for Ensembl VCF preparation pipelines.

Re-exports from the legacy location for backward compatibility.
Assets are defined in pipelines/ensembl_assets.py during migration.
"""
# Re-export from legacy location during migration
from prepare_annotations.pipelines.ensembl_assets import (
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
    ENSEMBL_VCF_PARTITIONS,
    download_retry_policy,
)

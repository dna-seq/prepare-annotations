"""
Dagster Definitions for genomic data preparation pipelines.

This module exports the main Definitions object that Dagster uses
to discover all assets, jobs, and resources.

Features:
- Parallel downloads with configurable concurrency
- Retry policies with exponential backoff
- Checksum verification and resumable downloads
- Splitting is OFF by default (legacy feature)

Usage:
    # Launch Dagster development server
    uv run dg dev

    # Materialize specific assets
    uv run dg asset materialize --select ensembl_vcf_urls

    # Materialize the default pipeline (download + convert, no splitting)
    uv run dg asset materialize --select ensembl_vcf_urls ensembl_vcf_files ensembl_parquet_files
"""

from dagster import Definitions, define_asset_job, AssetSelection

from prepare_annotations.pipelines_dagster.ensembl_assets import (
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_files,
    ensembl_parquet_files,
    ensembl_hf_upload,
)


# ============================================================================
# JOBS: Named execution paths through the asset graph
# Short names for easy CLI usage: prepare, download, convert, upload, full
# ============================================================================

# Default job: download -> convert (NO splitting by default)
prepare_job = define_asset_job(
    name="prepare",
    description="Download and convert Ensembl VCF data to Parquet (no splitting).",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_files,
        ensembl_parquet_files,
    ),
)

# Job to just download VCF files (parallel with retries)
download_job = define_asset_job(
    name="download",
    description="Download Ensembl VCF files from FTP (parallel, resumable).",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_files,
    ),
)

# Job to convert only (assumes VCF files already downloaded)
convert_job = define_asset_job(
    name="convert",
    description="Convert downloaded VCF files to Parquet format.",
    selection=AssetSelection.assets(
        ensembl_parquet_files,
    ),
)

# Job to upload to HuggingFace Hub
upload_job = define_asset_job(
    name="upload",
    description="Upload Ensembl parquet files to HuggingFace Hub.",
    selection=AssetSelection.assets(
        ensembl_hf_upload,
    ),
)

# Full pipeline: download → convert → upload
full_job = define_asset_job(
    name="full",
    description="Complete pipeline: download, convert, and upload to HuggingFace.",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_files,
        ensembl_parquet_files,
        ensembl_hf_upload,
    ),
)


# ============================================================================
# DEFINITIONS: Main export for Dagster
# ============================================================================

defs = Definitions(
    assets=[
        ensembl_ftp_source,
        ensembl_vcf_urls,
        ensembl_vcf_files,
        ensembl_parquet_files,
        ensembl_hf_upload,
    ],
    jobs=[
        prepare_job,
        download_job,
        convert_job,
        upload_job,
        full_job,
    ],
)

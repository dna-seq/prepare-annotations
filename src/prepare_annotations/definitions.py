"""
Dagster Definitions for genomic data preparation pipelines.

This module exports the main Definitions object that Dagster uses
to discover all assets, jobs, and resources.

Features:
- Dynamic partitioning based on FTP file discovery
- Per-file lineage tracking in Dagster UI
- Retry policies with exponential backoff
- Checksum verification and resumable downloads
- Concurrency limits to prevent OOM (respects PREPARE_ANNOTATIONS_* env vars)
- OakVar module conversion with Ensembl genotype resolution

Concurrency Control:
    The following environment variables control parallelism:
    - PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS: Max concurrent VCF downloads (default: 4)
    - PREPARE_ANNOTATIONS_PARQUET_WORKERS: Max concurrent parquet conversions (default: 2)

Usage:
    # Launch Dagster development server
    uv run dagster dev -m prepare_annotations

    # Materialize VCF URL discovery (registers partitions)
    uv run dagster asset materialize --select ensembl_vcf_urls

    # Materialize all VCF file downloads (all partitions)
    uv run dagster asset materialize --select ensembl_vcf_file

    # Materialize specific partition
    uv run dagster asset materialize --select ensembl_vcf_file --partition homo_sapiens.vcf.gz
    
    # Materialize longevitymap module with Ensembl genotype resolution
    uv run dagster asset materialize --select ensembl_variations_source longevitymap_weights
"""

import os

from dagster import Definitions, define_asset_job, AssetSelection

from prepare_annotations.assets import (
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
    ensembl_variations_source,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
)
from prepare_annotations.dagster_io_managers import (
    ensembl_cache_io_manager,
    huggingface_upload_io_manager,
)


def get_download_concurrency_limit() -> int:
    """Get max concurrent VCF downloads from env or default."""
    env_value = os.getenv("PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS")
    if env_value:
        return max(1, int(env_value))
    return 4  # Default: 4 concurrent downloads


def get_parquet_concurrency_limit() -> int:
    """Get max concurrent parquet conversions from env or default."""
    env_value = os.getenv("PREPARE_ANNOTATIONS_PARQUET_WORKERS")
    if env_value:
        return max(1, int(env_value))
    return 2  # Default: 2 concurrent conversions (memory-intensive)


# ============================================================================
# JOBS: Named execution paths through the asset graph
# Short names for easy CLI usage: prepare, download, convert, upload, full
# ============================================================================

# Default job: discover -> download -> convert (per-file partitioned)
prepare_job = define_asset_job(
    name="prepare",
    description="Discover, download and convert Ensembl VCF data to Parquet (per-file partitioned).",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
    ),
)

# Job to just download VCF files (partitioned with retries)
download_job = define_asset_job(
    name="download",
    description="Download Ensembl VCF files from FTP (per-file partitioned, resumable).",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_file,
    ),
)

# Job to convert only (assumes VCF files already downloaded)
convert_job = define_asset_job(
    name="convert",
    description="Convert downloaded VCF files to Parquet format (per-file partitioned).",
    selection=AssetSelection.assets(
        ensembl_parquet_file,
    ),
)

# Job to collect and upload to HuggingFace Hub
upload_job = define_asset_job(
    name="upload",
    description="Collect and upload Ensembl parquet files to HuggingFace Hub.",
    selection=AssetSelection.assets(
        ensembl_all_parquet_files,
        ensembl_hf_upload,
    ),
)

# Full pipeline: discover → download → convert → collect → upload
full_job = define_asset_job(
    name="full",
    description="Complete pipeline: discover, download, convert, and upload to HuggingFace.",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
    ),
)


# ============================================================================
# MODULE CONVERSION JOBS
# ============================================================================

# Job to convert LongevityMap module with Ensembl genotype resolution
longevitymap_job = define_asset_job(
    name="longevitymap",
    description="Convert LongevityMap module to unified schema with Ensembl genotype resolution.",
    selection=AssetSelection.assets(
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
    ),
)

# Job to create full LongevityMap with Ensembl join
longevitymap_full_job = define_asset_job(
    name="longevitymap_full",
    description="Convert LongevityMap and join with full Ensembl variation data.",
    selection=AssetSelection.assets(
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_with_ensembl,
    ),
)

# Job to upload LongevityMap to HuggingFace Hub
longevitymap_upload_job = define_asset_job(
    name="longevitymap_upload",
    description="Upload LongevityMap module to just-dna-seq/annotators on HuggingFace.",
    selection=AssetSelection.assets(
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_hf_upload,
    ),
)


# ============================================================================
# DEFINITIONS: Main export for Dagster
# ============================================================================

defs = Definitions(
    assets=[
        # Ensembl VCF pipeline assets (partitioned per-file)
        ensembl_ftp_source,
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
        # Module conversion assets
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_with_ensembl,
        longevitymap_hf_upload,
    ],
    jobs=[
        # Ensembl VCF pipeline jobs
        prepare_job,
        download_job,
        convert_job,
        upload_job,
        full_job,
        # Module conversion jobs
        longevitymap_job,
        longevitymap_full_job,
        longevitymap_upload_job,
    ],
    resources={
        "io_manager": ensembl_cache_io_manager,
        "hf_upload_io_manager": huggingface_upload_io_manager,
    },
)

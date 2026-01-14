"""
Dagster-based pipelines for genomic data preparation.

This module provides an alternative to the Prefect-based pipelines,
with proper asset lineage tracking for all intermediate and final files,
including HuggingFace downloads and uploads.

Key features:
- **Parallel downloads** with configurable concurrency (max_concurrent_downloads)
- **Retry policies** with exponential backoff for fault tolerance
- **Checksum verification** with BSD sum (CHECKSUMS file)
- **Resumable downloads** via fsspec filecache
- File-based assets with full lineage tracking
- Memory-efficient processing with LazyFrames (sink_parquet streaming)
- Caching at the Dagster level (re-materialization only when needed)

Asset Graph (Ensembl Pipeline):
    ensembl_ftp_source (external)
           ↓
    ensembl_vcf_urls (discovery)
           ↓
    ensembl_vcf_files (parallel download with retries)
           ↓
    ensembl_parquet_files (streaming conversion)
           ↓
    ensembl_hf_upload (HuggingFace Hub, optional)

Usage:
    # Launch the Dagster webserver
    uv run dg dev

    # Run the default pipeline (download + convert, no splitting)
    uv run prepare-annotations dagster-prepare

    # Materialize specific asset
    uv run dg asset materialize --select ensembl_vcf_urls

    # Run a job
    uv run dg launch --job ensembl_prepare_job
"""

from prepare_annotations.pipelines_dagster.definitions import defs

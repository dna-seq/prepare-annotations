"""
Dagster-based pipelines for genomic data preparation.

This module provides Dagster pipelines with proper asset lineage tracking
for all intermediate and final files, including HuggingFace downloads and uploads.

Key features:
- **Dynamic partitioning** based on FTP file discovery
- **Per-file lineage tracking** in Dagster UI (each VCF/parquet file is visible)
- **Retry policies** with exponential backoff for fault tolerance
- **Checksum verification** with BSD sum (CHECKSUMS file)
- **Resumable downloads** via fsspec filecache
- Memory-efficient processing with LazyFrames (sink_parquet streaming)
- Caching at the Dagster level (re-materialization only when needed)

Asset Graph (Ensembl Pipeline):
    ensembl_ftp_source (external)
           ↓
    ensembl_vcf_urls (discovery, registers dynamic partitions)
           ↓
    ensembl_vcf_file (per-file download, dynamically partitioned)
           ↓
    ensembl_parquet_file (per-file conversion, dynamically partitioned)
           ↓
    ensembl_all_parquet_files (collector for upload)
           ↓
    ensembl_hf_upload (HuggingFace Hub)

Module Conversion Assets:
    ensembl_variations_source (local cache or HuggingFace)
           ↓
    longevitymap_weights (with Ensembl genotype resolution)
           ↓
    longevitymap_with_ensembl (enriched with Ensembl data)

Usage:
    # Launch the Dagster webserver
    uv run dg dev

    # Materialize VCF URL discovery (registers partitions)
    uv run dg asset materialize --select ensembl_vcf_urls

    # Materialize all VCF file downloads (all partitions)
    uv run dg asset materialize --select ensembl_vcf_file

    # Materialize specific file partition
    uv run dg asset materialize --select ensembl_vcf_file --partition homo_sapiens.vcf.gz

    # Run a job
    uv run dg launch --job prepare
    
    # Run longevitymap conversion
    uv run dg launch --job longevitymap
"""
from prepare_annotations.pipelines.definitions import defs
from prepare_annotations.pipelines.logic import PreparationPipelines

__all__ = ["defs", "PreparationPipelines"]

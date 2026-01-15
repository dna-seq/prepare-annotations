"""
Download utilities for genomic data sources.

This module provides robust downloaders for:
- VCF files from Ensembl, ClinVar, dbSNP, gnomAD
- Reference genome FASTA files from Ensembl

Features:
- Resumable downloads with fsspec filecache
- Retry logic with exponential backoff
- Checksum verification (BSD sum format)
- Progress logging
- S3 and HTTP/HTTPS support
"""
# VCF downloader exports
from prepare_annotations.downloaders.vcf import (
    ChecksumInfo,
    parse_checksums_file,
    download_checksums,
    compute_checksum,
    verify_checksum,
    list_paths,
    download_path,
    convert_to_parquet,
    validate_downloads_and_parquet,
)

# Genome downloader exports
from prepare_annotations.downloaders.genome import (
    GenomeType,
    MaskingType,
    get_default_ensembl_cache_dir,
    get_default_genome_cache_dir,
    get_ensembl_fasta_url,
    find_genome_file,
    download_ensembl_genome,
    list_available_genomes,
    download_all_chromosomes,
    gunzip_to_fasta,
    write_fai_for_fasta,
    ensure_uncompressed_fasta_with_fai,
)

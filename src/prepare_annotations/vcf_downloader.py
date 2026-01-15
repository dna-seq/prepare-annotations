"""
VCF file downloader.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.downloaders.vcf
"""
# Re-export from downloaders for backward compatibility
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
    EliotDownloadCallback,
    RETRYABLE_STATUS,
)

# Also export the AnnotatedLazyFrame type for backward compat
from prepare_annotations.core.io import AnnotatedLazyFrame

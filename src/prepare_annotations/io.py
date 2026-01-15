"""
File I/O utilities for VCF and Parquet.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.core.io
"""
# Re-export from core for backward compatibility
from prepare_annotations.core.io import (
    read_vcf_file,
    vcf_to_parquet,
    merge_parquet_files,
    is_parquet,
    get_info_fields,
    get_default_threads,
    _strip_vcf_suffix,
    _default_parquet_path,
    AnnotatedResult,
    AnnotatedLazyFrame,
    SaveParquet,
)

"""
Core utilities for prepare-annotations.

This module provides foundational utilities used across the package:
- File I/O operations (VCF, Parquet)
- Pydantic models for results
- Path helpers and resource locations
- Runtime environment and profiling
- Variant splitting utilities
- Configuration helpers
"""
from prepare_annotations.core.io import (
    read_vcf_file,
    vcf_to_parquet,
    merge_parquet_files,
    is_parquet,
    get_info_fields,
    get_default_threads,
    AnnotatedResult,
    AnnotatedLazyFrame,
    SaveParquet,
)
from prepare_annotations.core.models import (
    ResourceReport,
    ModuleDependency,
    ModuleManifest,
    SplitResult,
    PreparationResult,
    SingleUploadResult,
    BatchUploadResult,
    RSIDCoordinateResult,
)
from prepare_annotations.core.paths import (
    ROOT_DIR,
    DATA_DIR,
    INPUT_DIR,
    INTERIM_DIR,
    OUTPUT_DIR,
    MODULES_DIR,
    MODULES_OUTPUT_DIR,
    LOGS_DIR,
    TEMPORARY_DIR,
    CACHE_DIR,
    ENSEMBL_CACHE_DIR,
    VCF_CACHE_DIR,
    get_cache_dir,
    get_ensembl_cache,
    get_ensembl_variations_cache,
    get_ensembl_genome_cache,
    find_ensembl_genome_fasta,
    list_ensembl_variation_parquets,
    list_ensembl_genome_fastas,
    get_default_cache_dir,
    get_default_input_dir,
    get_default_interim_dir,
    get_default_output_dir,
    get_output_dir,
    get_default_ensembl_cache_dir,
)
from prepare_annotations.core.runtime import (
    load_env,
    resource_tracker,
    resolve_worker_counts,
    is_port_in_use,
)
from prepare_annotations.core.config import (
    get_default_workers,
    get_parquet_workers,
    get_download_workers,
    get_profile_enabled,
)
from prepare_annotations.core.splitter import (
    split_variants_by_tsa,
    validate_split_outputs,
)

"""
Dagster configuration classes for prepare-annotations.

Re-exports from pipelines.configs for backward compatibility.
"""
from prepare_annotations.pipelines.configs import (
    get_default_duckdb_memory_limit,
    EnsemblDownloadConfig,
    ParquetConversionConfig,
    HuggingFaceUploadConfig,
    DuckDBConfig,
    EnsemblSourceConfig,
    LongevityMapConfig,
    AnnotatorsUploadConfig,
)

"""
Dagster I/O managers for prepare-annotations.

Re-exports from pipelines.io_managers for backward compatibility.
"""
from prepare_annotations.pipelines.io_managers import (
    ensembl_cache_io_manager,
    huggingface_upload_io_manager,
)

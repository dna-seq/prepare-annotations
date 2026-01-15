"""
Configuration helpers.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.core.config
"""
# Re-export from core for backward compatibility
from prepare_annotations.core.config import (
    get_default_workers,
    get_parquet_workers,
    get_download_workers,
    get_profile_enabled,
)

"""
Variant splitting utilities.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.core.splitter
"""
# Re-export from core for backward compatibility
from prepare_annotations.core.splitter import (
    split_variants_by_tsa,
    validate_split_outputs,
)

"""
Runtime environment utilities.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.core.runtime
"""
# Re-export from core for backward compatibility
from prepare_annotations.core.runtime import (
    load_env,
    resource_tracker,
    resolve_worker_counts,
    is_port_in_use,
)
from prepare_annotations.core.models import ResourceReport

"""
Pydantic models for prepare-annotations.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.core.models
"""
# Re-export from core for backward compatibility
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

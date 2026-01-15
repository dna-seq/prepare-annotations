"""
Dagster assets for OakVar module conversion.

Re-exports from the legacy location for backward compatibility.
Assets are defined in pipelines/module_assets.py during migration.
"""
# Re-export from legacy location during migration
from prepare_annotations.pipelines.module_assets import (
    ensembl_variations_source,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
    module_assets,
)

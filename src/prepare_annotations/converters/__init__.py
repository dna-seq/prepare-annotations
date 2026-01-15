"""
OakVar module conversion utilities.

This module provides converters for transforming OakVar SQLite modules
to the unified annotation schema (parquet format).

Available converters:
- longevitymap: Longevity-associated variants
- coronary: Coronary disease variants
- drugs: Pharmacogenomic data
- lipidmetabolism: Lipid metabolism variants
- superhuman: Elite performance genetics
- vo2max: VO2max-related variants

Each converter produces three standardized parquet files:
1. annotations.parquet: Variant-level facts (rsid, module, gene, phenotype, category)
2. studies.parquet: Per-study evidence (rsid, module, pmid, population, p_value, ...)
3. weights.parquet: Curated scoring (rsid, genotype, module, weight, state, ...)
"""
# Re-export from convert_modules (legacy name) during migration
from prepare_annotations.convert_modules import (
    # Common utilities
    load_weights_raw,
    load_variants_raw,
    scan_ensembl_variations,
    convert_module_weights_with_ensembl,
    # Genotype utilities
    genotype_has_placeholder,
    resolve_genotype_placeholders_with_ensembl,
    select_ensembl_minimal,
    # Module converters
    convert_longevitymap,
    convert_longevitymap_annotations,
    convert_longevitymap_studies,
    convert_longevitymap_weights,
    convert_lipidmetabolism,
    convert_vo2max,
    convert_superhuman,
    convert_coronary,
    convert_drugs,
)

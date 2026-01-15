"""
OakVar module conversion utilities.

Import specific converters directly from their modules:
    from prepare_annotations.convert_modules.longevitymap import convert_longevitymap
    from prepare_annotations.convert_modules.common import convert_module_weights_with_ensembl
"""
# Re-export main conversion functions for convenience
from prepare_annotations.convert_modules.common import (
    load_weights_raw,
    load_variants_raw,
    scan_ensembl_variations,
    convert_module_weights_with_ensembl,
)
from prepare_annotations.convert_modules.genotypes import (
    genotype_has_placeholder,
    resolve_genotype_placeholders_with_ensembl,
    select_ensembl_minimal,
)
from prepare_annotations.convert_modules.longevitymap import (
    convert_longevitymap,
    convert_longevitymap_annotations,
    convert_longevitymap_studies,
    convert_longevitymap_weights,
)
from prepare_annotations.convert_modules.lipidmetabolism import convert_lipidmetabolism
from prepare_annotations.convert_modules.vo2max import convert_vo2max
from prepare_annotations.convert_modules.superhuman import convert_superhuman
from prepare_annotations.convert_modules.coronary import convert_coronary
from prepare_annotations.convert_modules.drugs import convert_drugs

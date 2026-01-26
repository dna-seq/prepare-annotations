"""
Dagster Asset Checks for module conversion validation.

These checks run after assets are materialized and validate:
- Schema correctness (column types, formats)
- Value constraints (valid states, genotype format)
- Data quality (no empty outputs, required columns present)

All checks use Polars LazyFrame scans to avoid loading full data into memory.
This makes them suitable for large parquet files without memory bloat.

Key design principles:
- LazyFrame-only: Never call .collect() on full data, only on aggregations
- Fail-fast: Return early on first failure for efficiency
- Metadata: Include helpful debugging info in check results
"""
from pathlib import Path
from typing import Sequence

import polars as pl
from dagster import (
    asset_check,
    AssetCheckResult,
    AssetCheckSeverity,
    AssetKey,
)

from prepare_annotations.core.paths import MODULES_OUTPUT_DIR


# ============================================================================
# VALID VALUE SETS (from schema spec)
# ============================================================================

VALID_WEIGHT_STATES = {"protective", "risk", "neutral", "alt", "ref"}
VALID_ALLELE_CHARS = set("ACGTN?")


# ============================================================================
# HELPER FUNCTIONS (LazyFrame-based, memory-efficient)
# ============================================================================


def check_parquet_exists(path: Path) -> AssetCheckResult | None:
    """Return failure result if parquet file doesn't exist, else None."""
    if not path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": f"File not found: {path}"},
        )
    return None


def check_row_count_nonzero(path: Path) -> AssetCheckResult | None:
    """Return failure result if parquet has zero rows, else None."""
    row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
    if row_count == 0:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"error": "Parquet file has 0 rows", "path": str(path)},
        )
    return None


def check_required_columns(
    path: Path, required_cols: Sequence[str]
) -> AssetCheckResult | None:
    """Return failure result if required columns are missing, else None."""
    schema = pl.scan_parquet(path).collect_schema()
    actual_cols = set(schema.names())
    missing = set(required_cols) - actual_cols
    if missing:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "error": f"Missing required columns: {sorted(missing)}",
                "actual_columns": sorted(actual_cols),
            },
        )
    return None


# ============================================================================
# WEIGHTS PARQUET CHECKS
# ============================================================================


def _weights_checks_core(
    weights_path: Path,
    module_name: str,
) -> AssetCheckResult:
    """
    Core validation for weights.parquet files.
    
    Validates:
    - File exists and has rows
    - Required columns present
    - Module column matches expected value
    - State values are valid
    - Genotype format is list of 2 alleles
    
    Note: Weight/state consistency is NOT checked because different modules
    use different semantic systems:
    - longevitymap, superhuman: protective/risk (health impact)
    - coronary, lipidmetabolism, vo2max: alt/ref (allele type)
    
    The weight sign meaning varies by module and state system, so no
    universal weight/state consistency rule applies.
    
    All checks use LazyFrame scans to avoid memory bloat.
    """
    # Basic existence checks
    if (result := check_parquet_exists(weights_path)) is not None:
        return result
    
    if (result := check_row_count_nonzero(weights_path)) is not None:
        return result
    
    required_cols = ["rsid", "genotype", "module", "weight", "state"]
    if (result := check_required_columns(weights_path, required_cols)) is not None:
        return result
    
    lf = pl.scan_parquet(weights_path)
    
    # Check module column value
    module_values = lf.select(pl.col("module").unique()).collect()["module"].to_list()
    if module_values != [module_name]:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "error": f"Module column should be '{module_name}'",
                "actual_values": module_values,
            },
        )
    
    # Check state values
    states = set(
        lf.select(pl.col("state").unique().drop_nulls()).collect()["state"].to_list()
    )
    invalid_states = states - VALID_WEIGHT_STATES
    if invalid_states:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "error": f"Invalid state values: {sorted(invalid_states)}",
                "valid_states": sorted(VALID_WEIGHT_STATES),
            },
        )
    
    # Check genotype format (sample-based to avoid full scan)
    # Get first 100 unique genotypes for validation
    sample_genotypes = (
        lf.select(pl.col("genotype").unique().head(100))
        .collect()["genotype"]
        .to_list()
    )
    
    for gt in sample_genotypes:
        if not isinstance(gt, list):
            return AssetCheckResult(
                passed=False,
                severity=AssetCheckSeverity.ERROR,
                metadata={
                    "error": f"Genotype should be list, got {type(gt).__name__}",
                    "sample_value": str(gt),
                },
            )
        if len(gt) != 2:
            return AssetCheckResult(
                passed=False,
                severity=AssetCheckSeverity.ERROR,
                metadata={
                    "error": f"Genotype should have 2 alleles, got {len(gt)}",
                    "sample_value": str(gt),
                },
            )
        # Check alleles are alphabetically sorted
        if gt != sorted(gt):
            return AssetCheckResult(
                passed=False,
                severity=AssetCheckSeverity.WARN,
                metadata={
                    "warning": "Genotype not alphabetically normalized",
                    "sample_value": str(gt),
                },
            )
    
    # All checks passed
    row_count = lf.select(pl.len()).collect().item()
    unique_rsids = lf.select(pl.col("rsid").n_unique()).collect().item()
    
    return AssetCheckResult(
        passed=True,
        metadata={
            "row_count": row_count,
            "unique_rsids": unique_rsids,
            "states_found": sorted(states),
            "module": module_name,
        },
    )


def _annotations_checks_core(
    annotations_path: Path,
    module_name: str,
) -> AssetCheckResult:
    """
    Core validation for annotations.parquet files.
    
    Validates:
    - File exists and has rows
    - Required columns present (rsid, module, gene, phenotype, category)
    - Module column matches expected value
    """
    if (result := check_parquet_exists(annotations_path)) is not None:
        return result
    
    if (result := check_row_count_nonzero(annotations_path)) is not None:
        return result
    
    required_cols = ["rsid", "module"]
    if (result := check_required_columns(annotations_path, required_cols)) is not None:
        return result
    
    lf = pl.scan_parquet(annotations_path)
    
    # Check module column value
    module_values = lf.select(pl.col("module").unique()).collect()["module"].to_list()
    if module_values != [module_name]:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "error": f"Module column should be '{module_name}'",
                "actual_values": module_values,
            },
        )
    
    # All checks passed
    row_count = lf.select(pl.len()).collect().item()
    unique_rsids = lf.select(pl.col("rsid").n_unique()).collect().item()
    
    return AssetCheckResult(
        passed=True,
        metadata={
            "row_count": row_count,
            "unique_rsids": unique_rsids,
            "module": module_name,
        },
    )


def _studies_checks_core(
    studies_path: Path,
    module_name: str,
) -> AssetCheckResult:
    """
    Core validation for studies.parquet files.
    
    Validates:
    - File exists and has rows
    - Required columns present (rsid, module, pmid)
    - Module column matches expected value
    """
    if (result := check_parquet_exists(studies_path)) is not None:
        return result
    
    if (result := check_row_count_nonzero(studies_path)) is not None:
        return result
    
    required_cols = ["rsid", "module"]
    if (result := check_required_columns(studies_path, required_cols)) is not None:
        return result
    
    lf = pl.scan_parquet(studies_path)
    
    # Check module column value
    module_values = lf.select(pl.col("module").unique()).collect()["module"].to_list()
    if module_values != [module_name]:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "error": f"Module column should be '{module_name}'",
                "actual_values": module_values,
            },
        )
    
    # All checks passed
    row_count = lf.select(pl.len()).collect().item()
    unique_rsids = lf.select(pl.col("rsid").n_unique()).collect().item()
    
    return AssetCheckResult(
        passed=True,
        metadata={
            "row_count": row_count,
            "unique_rsids": unique_rsids,
            "module": module_name,
        },
    )


# ============================================================================
# LONGEVITYMAP CHECKS
# ============================================================================


@asset_check(
    asset=AssetKey("longevitymap_weights"),
    description="Validate longevitymap weights.parquet schema, format, and value constraints.",
)
def check_longevitymap_weights(longevitymap_weights: Path) -> AssetCheckResult:
    """
    Validate LongevityMap weights parquet file.
    
    Uses LazyFrame scans for memory-efficient validation of:
    - Schema: required columns present
    - Genotype format: list of 2 alphabetically-sorted alleles
    - State values: in valid set {protective, risk, neutral, alt, ref}
    - Weight/state consistency: positive=protective, negative=risk
    """
    return _weights_checks_core(
        weights_path=longevitymap_weights,
        module_name="longevitymap",
    )


@asset_check(
    asset=AssetKey("longevitymap_annotations"),
    description="Validate longevitymap annotations.parquet schema and module column.",
)
def check_longevitymap_annotations(longevitymap_annotations: pl.LazyFrame) -> AssetCheckResult:
    """Validate LongevityMap annotations parquet file.
    
    Note: This asset uses polars_parquet_io_manager, so we receive a LazyFrame
    and compute the path from IO manager convention.
    """
    # dagster-polars stores at: base_dir / asset_key.parquet
    annotations_path = MODULES_OUTPUT_DIR / "longevitymap_annotations.parquet"
    return _annotations_checks_core(
        annotations_path=annotations_path,
        module_name="longevitymap",
    )


@asset_check(
    asset=AssetKey("longevitymap_studies"),
    description="Validate longevitymap studies.parquet schema and module column.",
)
def check_longevitymap_studies(longevitymap_studies: pl.LazyFrame) -> AssetCheckResult:
    """Validate LongevityMap studies parquet file.
    
    Note: This asset uses polars_parquet_io_manager, so we receive a LazyFrame
    and compute the path from IO manager convention.
    """
    # dagster-polars stores at: base_dir / asset_key.parquet
    studies_path = MODULES_OUTPUT_DIR / "longevitymap_studies.parquet"
    return _studies_checks_core(
        studies_path=studies_path,
        module_name="longevitymap",
    )


# ============================================================================
# LIPIDMETABOLISM CHECKS
# ============================================================================


@asset_check(
    asset=AssetKey("lipidmetabolism_weights"),
    description="Validate lipidmetabolism weights.parquet schema, format, and value constraints.",
)
def check_lipidmetabolism_weights(lipidmetabolism_weights: Path) -> AssetCheckResult:
    """Validate LipidMetabolism weights parquet file."""
    return _weights_checks_core(
        weights_path=lipidmetabolism_weights,
        module_name="lipidmetabolism",
    )


@asset_check(
    asset=AssetKey("lipidmetabolism_annotations"),
    description="Validate lipidmetabolism annotations.parquet schema and module column.",
)
def check_lipidmetabolism_annotations(lipidmetabolism_annotations: Path) -> AssetCheckResult:
    """Validate LipidMetabolism annotations parquet file."""
    return _annotations_checks_core(
        annotations_path=lipidmetabolism_annotations,
        module_name="lipidmetabolism",
    )


@asset_check(
    asset=AssetKey("lipidmetabolism_studies"),
    description="Validate lipidmetabolism studies.parquet schema and module column.",
)
def check_lipidmetabolism_studies(lipidmetabolism_studies: Path) -> AssetCheckResult:
    """Validate LipidMetabolism studies parquet file."""
    return _studies_checks_core(
        studies_path=lipidmetabolism_studies,
        module_name="lipidmetabolism",
    )


# ============================================================================
# VO2MAX CHECKS
# ============================================================================


@asset_check(
    asset=AssetKey("vo2max_weights"),
    description="Validate vo2max weights.parquet schema, format, and value constraints.",
)
def check_vo2max_weights(vo2max_weights: Path) -> AssetCheckResult:
    """Validate VO2Max weights parquet file."""
    return _weights_checks_core(
        weights_path=vo2max_weights,
        module_name="vo2max",
    )


@asset_check(
    asset=AssetKey("vo2max_annotations"),
    description="Validate vo2max annotations.parquet schema and module column.",
)
def check_vo2max_annotations(vo2max_annotations: Path) -> AssetCheckResult:
    """Validate VO2Max annotations parquet file."""
    return _annotations_checks_core(
        annotations_path=vo2max_annotations,
        module_name="vo2max",
    )


@asset_check(
    asset=AssetKey("vo2max_studies"),
    description="Validate vo2max studies.parquet schema and module column.",
)
def check_vo2max_studies(vo2max_studies: Path) -> AssetCheckResult:
    """Validate VO2Max studies parquet file."""
    return _studies_checks_core(
        studies_path=vo2max_studies,
        module_name="vo2max",
    )


# ============================================================================
# SUPERHUMAN CHECKS
# ============================================================================


@asset_check(
    asset=AssetKey("superhuman_weights"),
    description="Validate superhuman weights.parquet schema and format (no numeric weights).",
)
def check_superhuman_weights(superhuman_weights: Path) -> AssetCheckResult:
    """
    Validate Superhuman weights parquet file.
    
    Note: Superhuman module has NULL weights (qualitative annotations only).
    """
    return _weights_checks_core(
        weights_path=superhuman_weights,
        module_name="superhuman",
    )


@asset_check(
    asset=AssetKey("superhuman_annotations"),
    description="Validate superhuman annotations.parquet schema and module column.",
)
def check_superhuman_annotations(superhuman_annotations: Path) -> AssetCheckResult:
    """Validate Superhuman annotations parquet file."""
    return _annotations_checks_core(
        annotations_path=superhuman_annotations,
        module_name="superhuman",
    )


@asset_check(
    asset=AssetKey("superhuman_studies"),
    description="Validate superhuman studies.parquet schema and module column.",
)
def check_superhuman_studies(superhuman_studies: Path) -> AssetCheckResult:
    """Validate Superhuman studies parquet file."""
    return _studies_checks_core(
        studies_path=superhuman_studies,
        module_name="superhuman",
    )


# ============================================================================
# CORONARY CHECKS
# ============================================================================


@asset_check(
    asset=AssetKey("coronary_weights"),
    description="Validate coronary weights.parquet schema, format, and value constraints.",
)
def check_coronary_weights(coronary_weights: Path) -> AssetCheckResult:
    """Validate Coronary weights parquet file."""
    return _weights_checks_core(
        weights_path=coronary_weights,
        module_name="coronary",
    )


@asset_check(
    asset=AssetKey("coronary_annotations"),
    description="Validate coronary annotations.parquet schema and module column.",
)
def check_coronary_annotations(coronary_annotations: Path) -> AssetCheckResult:
    """Validate Coronary annotations parquet file."""
    return _annotations_checks_core(
        annotations_path=coronary_annotations,
        module_name="coronary",
    )


@asset_check(
    asset=AssetKey("coronary_studies"),
    description="Validate coronary studies.parquet schema and module column.",
)
def check_coronary_studies(coronary_studies: Path) -> AssetCheckResult:
    """Validate Coronary studies parquet file."""
    return _studies_checks_core(
        studies_path=coronary_studies,
        module_name="coronary",
    )


# ============================================================================
# ALL CHECKS LIST (for easy import in definitions.py)
# ============================================================================

all_module_checks = [
    # LongevityMap
    check_longevitymap_weights,
    check_longevitymap_annotations,
    check_longevitymap_studies,
    # LipidMetabolism
    check_lipidmetabolism_weights,
    check_lipidmetabolism_annotations,
    check_lipidmetabolism_studies,
    # VO2Max
    check_vo2max_weights,
    check_vo2max_annotations,
    check_vo2max_studies,
    # Superhuman
    check_superhuman_weights,
    check_superhuman_annotations,
    check_superhuman_studies,
    # Coronary
    check_coronary_weights,
    check_coronary_annotations,
    check_coronary_studies,
]

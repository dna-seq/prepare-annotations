"""
Prepare Annotations - Genomic data preparation pipelines.

This package provides tools for downloading, converting, and uploading
genomic annotation data (Ensembl, ClinVar, dbSNP, gnomAD) and converting
OakVar modules to unified annotation schema.

Main entry points:
- Dagster definitions: `prepare_annotations.definitions.defs`
- Standalone API: `prepare_annotations.pipelines.PreparationPipelines`
- CLI: `uv run prepare-annotations` or `uv run prepare`

Project Structure:
- core/: Core utilities (I/O, models, paths, runtime)
- downloaders/: VCF and genome download utilities
- huggingface/: HuggingFace Hub integration
- assets/: Dagster assets
- converters/: OakVar module converters
- definitions.py: Main Dagster definitions
- pipelines.py: Standalone preparation API
"""


def hello() -> str:
    return "Hello from prepare-annotations!"


# Lazy imports for commonly used items
def __getattr__(name: str):
    """Lazy imports for easy access to main entry points."""
    if name == "defs":
        from prepare_annotations.definitions import defs
        return defs
    if name == "PreparationPipelines":
        from prepare_annotations.pipelines import PreparationPipelines
        return PreparationPipelines
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

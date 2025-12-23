# Prepare Annotations Agent Guidelines

This repository is dedicated to the preparation of genomic annotation data (Ensembl, ClinVar, dbSNP, gnomAD, etc.).

## Repository Layout (uv package)

- `src/prepare_annotations/`: Core logic and CLI.
  - `preparation/`: Source-specific preparation pipelines (Prefect-based).
  - `vortex/`: Vortex data conversion utilities.
  - `cli.py`: Main Typer CLI entrypoint.
  - `io.py`: VCF/Parquet I/O utilities.
  - `runtime.py`: Execution environment and profiling.
  - `models.py`: Pydantic models for results.
- `dataset_cards/`: Markdown templates for Hugging Face dataset cards.
- `tests/`: Unit and integration tests.

## Coding Standards

- **Type hints**: Mandatory for all Python code.
- **Pathlib**: Always use for all file paths.
- **Polars**: Prefer over Pandas for performance.
- **Prefect**: Used for workflow orchestration and parallel execution.
- **Eliot**: Used for structured logging and action tracking.
- **Typer**: Mandatory for CLI tools.
- **Pydantic 2**: Mandatory for data classes.

## Commands

- `uv run prepare-annotations ensembl`: Download and prepare Ensembl variations.
- `uv run prepare-annotations clinvar`: Download and prepare ClinVar data.

## Deployment

Datasets are typically uploaded to the `just-dna-seq` organization on Hugging Face Hub.


# Prepare Annotations Agent Guidelines

This repository is dedicated to the preparation of genomic annotation data (Ensembl, ClinVar, dbSNP, gnomAD, etc.) and conversion of OakVar modules from the [dna-seq GitHub organization](https://github.com/orgs/dna-seq/repositories).

## Repository Layout (uv package)

- `src/prepare_annotations/`: Core logic and CLI.
  - `preparation/`: Source-specific preparation pipelines (Prefect-based).
    - `pipelines.py`: Main flow and pipeline definitions.
    - `oakvar/`: OakVar module management and conversion.
      - `modules.py`: CLI for downloading and managing OakVar modules.
      - `convert_longevitymap.py`: LongevityMap conversion to unified schema.
      - `convert_module.py`: Generic module conversion utilities.
  - `vortex/`: Vortex data conversion utilities.
  - `cli.py`: Main Typer CLI entrypoint.
  - `io.py`: VCF/Parquet I/O utilities.
  - `runtime.py`: Execution environment and profiling.
  - `models.py`: Pydantic models for results.
- `dataset_cards/`: Markdown templates for Hugging Face dataset cards.
- `tests/`: Unit and integration tests.
  - `conftest.py`: Shared fixtures including OakVar module download helpers.
  - `test_longevitymap_module.py`: Comprehensive validation of longevitymap conversion.

## Coding Standards

- **Type hints**: Mandatory for all Python code.
- **Pathlib**: Always use for all file paths.
- **Polars**: Prefer over Pandas for performance.
- **Prefect**: Used for workflow orchestration and parallel execution.
- **Eliot**: Used for structured logging and action tracking.
- **Typer**: Mandatory for CLI tools.
- **Pydantic 2**: Mandatory for data classes.

## Commands

### Main Genomic Data Pipelines

- `uv run prepare-annotations ensembl`: Download and prepare Ensembl variations.
- `uv run prepare-annotations clinvar`: Download and prepare ClinVar data.
- `uv run prepare-annotations dbsnp`: Download and prepare dbSNP data.
- `uv run prepare-annotations gnomad`: Download and prepare gnomAD data.

### OakVar Module Management

- `uv run modules data --repo dna-seq/just_longevitymap`: Download module data files.
- `uv run modules clone --repo dna-seq/just_longevitymap`: Clone full module repository.
- `uv run modules convert-longevitymap`: Convert LongevityMap to unified annotation schema.

### Unified Annotation Schema

The module conversion produces three standardized parquet files:

1. **annotations.parquet**: Variant-level facts
   - Schema: `rsid, module, gene, phenotype, category`
   - Links variants to genes and phenotype categories

2. **studies.parquet**: Per-study evidence
   - Schema: `rsid, module, pmid, population, p_value, conclusion, study_design`
   - Scientific evidence from publications

3. **weights.parquet**: Curator-defined scoring
   - Schema: `rsid, genotype, module, weight, state, priority, conclusion, curator, method`
   - Curated weight assignments for variant impact
   - State: `protective`, `risk`, or `neutral`
   - Genotype: Normalized (e.g., `CT`, `TT`, `AA`)

### Available Modules

Modules from https://github.com/orgs/dna-seq/repositories:
- `just_longevitymap`: Longevity-associated variants
- `just_pathogenic`: Pathogenic variant annotations
- `just_cancer`: Cancer-associated genes
- `just_coronary`: Coronary disease variants
- `just_vo2max`: VO2max-related variants
- `just_lipidmetabolism`: Lipid metabolism variants
- `just_prs`: Polygenic risk score data
- `just_drugs`: Pharmacogenomic data
- `just_superhuman`: Elite performance genetics

## Deployment

Datasets are typically uploaded to the `just-dna-seq` organization on Hugging Face Hub.

## Testing

### Test Philosophy

- **Integration tests**: Use real data, no mocking unless necessary
- **Auto-download**: Tests automatically download required data from GitHub
- **Validation**: Comprehensive checks ensuring data integrity during conversion

### Running Tests

```bash
# Run all tests (excluding large downloads)
uv run pytest

# Run specific module tests
uv run pytest tests/test_longevitymap_module.py -v

# Run with verbose output
uv run pytest -vvv
```

### Test Fixtures

The `conftest.py` provides shared fixtures for OakVar module testing:

- `ensure_oakvar_module_data()`: Downloads module data if not present
- `download_oakvar_module_data()`: Directly downloads from GitHub repositories

These fixtures are automatically used by test modules to ensure data availability.

### Example: LongevityMap Validation

The `test_longevitymap_module.py` includes 47 tests validating:

1. **Weights Table** (1043 rows, 528 variants)
   - Row counts match between SQLite and Parquet
   - Weight values preserved (sum, min, max, mean)
   - Per-rsid weight sums match
   - Negative (risk) weights correctly identified

2. **APOE Variants** (Critical longevity markers)
   - rs7412 (APOE e2): protective weights
   - rs429358 (APOE e4): risk weights

3. **Schema Transformations**
   - Genotype format (het → CT, hom → TT)
   - State values (protective/risk)
   - Module column correctness

4. **Studies & Annotations**
   - All PMIDs preserved (270 unique)
   - Categories preserved (12 categories)
   - Populations preserved (81 populations)

Tests automatically:
1. Download SQLite from `dna-seq/just_longevitymap` if missing
2. Convert to parquet if needed
3. Validate data integrity

# Prepare Annotations Agent Guidelines

This repository is dedicated to the preparation of genomic annotation data (Ensembl, ClinVar, dbSNP, gnomAD, etc.) and conversion of OakVar modules from the [dna-seq GitHub organization](https://github.com/orgs/dna-seq/repositories).

## Repository Layout (uv package)

The package follows Dagster best practices with utilities organized in subpackages:

- `src/prepare_annotations/`: Main package
  - `definitions.py`: **Main Dagster definitions** (assets, jobs, resources)
  - `pipelines.py`: **Standalone API** (PreparationPipelines class)
  - `cli.py`: Main Typer CLI entrypoint
  
  - `core/`: Core utilities
    - `io.py`: VCF/Parquet I/O utilities
    - `models.py`: Pydantic models for results
    - `paths.py`: Path helpers and resource locations
    - `runtime.py`: Execution environment and profiling
    - `config.py`: Configuration helpers
    - `splitter.py`: Variant splitting by type
  
  - `assets/`: Dagster assets
    - `ensembl.py`: Ensembl VCF pipeline assets
    - `modules.py`: OakVar module conversion assets
  
  - `downloaders/`: Download utilities
    - `vcf.py`: VCF download with retry/resume
    - `genome.py`: Ensembl genome FASTA download
  
  - `huggingface/`: HuggingFace Hub integration
    - `uploader.py`: Upload utilities
    - `dataset_cards.py`: Dataset card templates
  
  - `converters/`: OakVar module converters
    - `longevitymap.py`, `coronary.py`, `drugs.py`, etc.
    - `common.py`: Shared conversion utilities

- `dataset_cards/`: Markdown templates for Hugging Face dataset cards
- `tests/`: Unit and integration tests

## Coding Standards

- **Type hints**: Mandatory for all Python code.
- **Pathlib**: Always use for all file paths.
- **Polars**: Prefer over Pandas for performance.
- **Dagster**: Primary tool for workflow orchestration and parallel execution.
- **Eliot**: Used for structured logging and action tracking.
- **Typer**: Mandatory for CLI tools.
- **Pydantic 2**: Mandatory for data classes.
- **Avoid __all__**: Avoid __init__.py with __all__ as it confuses where things are located.

## Import Guidelines

For new code, use the organized subpackages:

```python
# Dagster definitions
from prepare_annotations.definitions import defs

# Standalone API
from prepare_annotations.pipelines import PreparationPipelines

# Assets
from prepare_annotations.assets import ensembl_vcf_urls, longevitymap_weights

# Core utilities
from prepare_annotations.core.io import read_vcf_file, vcf_to_parquet
from prepare_annotations.core.models import PreparationResult
from prepare_annotations.core.paths import get_cache_dir, LOGS_DIR

# Downloaders
from prepare_annotations.downloaders.vcf import download_path, list_paths
from prepare_annotations.downloaders.genome import download_ensembl_genome

# HuggingFace
from prepare_annotations.huggingface.uploader import upload_parquet_to_hf
from prepare_annotations.huggingface.dataset_cards import generate_ensembl_card

# Converters
from prepare_annotations.converters import convert_longevitymap
```

## Dagster Guide (Agents)

These pipelines are Dagster-first. Follow these rules to avoid the issues we already hit:

### 1) Use modern API (no legacy CLI)

- Do not use `dagster job execute` or other deprecated CLI for orchestration.
- Prefer Python API: `materialize()` for assets, `execute_job()` for non-partitioned jobs.
- If CLI is needed, use Dagster dev server only (`uv run dagster dev -m prepare_annotations`).

### 2) Dynamic partitions must be explicit

- Use dynamic partitions whenever the upstream file list is external or changing (FTP, HTTP, HF repo, etc.).
- The discovery asset must register partitions via `DynamicPartitionsDefinition`.
- Partitioned assets must use `context.partition_key` and must not be run without a partition key.

### 3) Materialize with full asset list + selection

When using `materialize()`:

- Always pass the full asset graph (all upstream assets).
- Use `selection=["asset_name"]` to run a single asset.
- Do not pass config for assets not present in the selection.

Example pattern:

- `materialize(assets=all_assets, selection=["download_asset"], partition_key="...")`

### 4) IO manager must resolve partition paths

Partitioned assets need file-level input paths:

- Inputs must resolve to concrete files for each partition key
- If an IO manager returns a directory for a partitioned asset, downstream processing will crash

### 5) Collector must depend on partitioned outputs

Collector assets must declare deps on the partitioned asset to ensure correct lineage and ordering.

### 6) Filter temporary outputs before upload

Do not upload temp files. Filter out:

- files starting with `tmp`
- files ending with `.tmp.parquet` or `.parquet.tmp`
- dotfiles

### 7) Concurrency and memory safety

- Use download parallelism (I/O bound), but keep conversion limited.
- Concurrency should be enforced via Dagster tag limits in `dagster.yaml`.

### 8) Dagster home

- `DAGSTER_HOME` is `data/interim/dagster`
- Always set it for runs so UI and API share the same instance.

### Primary Dagster Pipelines (Recommended)

- `uv run dagster-ensembl`: Run the full Ensembl pipeline (download, convert, upload).
- `uv run dagster-ensembl ui`: Launch Dagster UI for Ensembl pipelines.
- `uv run dagster-ui`: General Dagster development server entrypoint.

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

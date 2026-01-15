# Prepare Annotations

A dedicated toolkit for downloading, processing, and preparing genomic annotation datasets.

## Features

- **Dagster-based Pipelines (Primary)**: Software-Defined Assets (SDA) with full lineage tracking, parallel execution, and automated Hugging Face uploads.
- **Support for multiple sources**:
  - **Ensembl**: Human genetic variations.
  - **ClinVar**: Clinical variant data.
  - **dbSNP**: Single Nucleotide Polymorphism database.
  - **gnomAD**: Genome Aggregation Database.
- **OakVar Module Management**: Download and convert data from [dna-seq](https://github.com/orgs/dna-seq/repositories) OakVar modules.
- **VCF to Parquet**: Efficient conversion of large VCF files to columnar format using `polars-bio`.
- **Hugging Face Hub Integration**: Direct upload of processed datasets with automatic dataset card generation.

## Installation

This project uses `uv` for dependency management.

```bash
git clone https://github.com/dna-seq/prepare-annotations.git
cd prepare-annotations
uv sync
```

## Usage

### 🔷 Dagster Pipelines

The primary way to run pipelines is using Dagster. This provides parallel execution, resumable downloads, and integrated Hugging Face uploads.

![Dagster Pipeline Lineage](images/pipelines.jpg)

#### Ensembl Pipeline

```bash
# Run the full pipeline (download → convert → upload)
uv run dagster-ensembl

# Start the Dagster UI for monitoring and interactive execution
uv run dagster-ensembl ui

# Run for a specific species
uv run dagster-ensembl run --species mus_musculus
```

#### Other Dagster Commands

```bash
# List all available assets
uv run dagster-ui assets

# Materialize specific assets
uv run dagster-ui materialize ensembl_vcf_urls
```

### OakVar Module Management

The `modules` command manages OakVar modules from the [dna-seq GitHub organization](https://github.com/orgs/dna-seq/repositories).

#### Download Module Data

Download data files (SQLite databases, etc.) from module repositories:

```bash
# Download longevitymap data
uv run modules data --repo dna-seq/just_longevitymap

# Download other modules
uv run modules data --repo dna-seq/just_pathogenic
uv run modules data --repo dna-seq/just_cancer
uv run modules data --repo dna-seq/just_coronary
uv run modules data --repo dna-seq/just_vo2max
uv run modules data --repo dna-seq/just_lipidmetabolism

# Download with specific extensions
uv run modules data --ext .parquet --ext .csv

# Download to custom directory
uv run modules data --output-dir /path/to/output
```

#### Clone Full Module Repository

Clone entire module repositories:

```bash
# Clone longevitymap module
uv run modules clone --repo dna-seq/just_longevitymap

# Clone to specific directory
uv run modules clone --repo dna-seq/just_pathogenic --output-dir ./modules/
```

#### Convert Module Data

Convert OakVar module data to unified annotation schema:

```bash
# Convert LongevityMap to unified schema (3 parquet files)
uv run modules convert-longevitymap

# With custom paths
uv run modules convert-longevitymap \
  --db-path data/modules/just_longevitymap/longevitymap.sqlite \
  --output-dir data/output/modules/longevitymap \
  --curator "Olga Borysova" \
  --method "literature_review"
```

The conversion produces three parquet files:
- **annotations.parquet**: Variant-level facts (rsid, module, gene, phenotype, category)
- **studies.parquet**: Per-study evidence (rsid, module, pmid, population, conclusion, study_design)
- **weights.parquet**: Curator-defined scoring (rsid, genotype, module, weight, state, priority, curator, method)

### Available Modules

The following modules are available from the [dna-seq organization](https://github.com/orgs/dna-seq/repositories):

- **just_longevitymap**: Longevity-associated variants
- **just_pathogenic**: Pathogenic variant annotations
- **just_cancer**: Cancer-associated genes
- **just_coronary**: Coronary disease variants
- **just_vo2max**: VO2max-related variants
- **just_lipidmetabolism**: Lipid metabolism variants
- **just_prs**: Polygenic risk score data
- **just_drugs**: Pharmacogenomic data
- **just_superhuman**: Elite performance genetics

## Package Structure

The package follows Dagster best practices with utilities organized in subpackages:

```
src/prepare_annotations/
├── definitions.py          # Main Dagster definitions (assets, jobs, resources)
├── pipelines.py            # Standalone API (PreparationPipelines)
├── cli.py                  # Typer CLI entrypoint
│
├── core/                   # Core utilities
│   ├── io.py               # VCF/Parquet I/O
│   ├── models.py           # Pydantic models
│   ├── paths.py            # Path helpers
│   └── runtime.py          # Profiling, environment
│
├── assets/                 # Dagster assets
│   ├── ensembl.py          # Ensembl VCF pipeline
│   └── modules.py          # OakVar module conversion
│
├── downloaders/            # Download utilities
│   ├── vcf.py              # VCF download
│   └── genome.py           # Genome FASTA download
│
├── huggingface/            # HuggingFace Hub integration
│   ├── uploader.py         # Upload utilities
│   └── dataset_cards.py    # Dataset card templates
│
└── converters/             # OakVar module converters
```

### Import Examples

```python
# Dagster definitions
from prepare_annotations.definitions import defs

# Standalone API
from prepare_annotations.pipelines import PreparationPipelines

# Core utilities
from prepare_annotations.core.io import read_vcf_file
from prepare_annotations.core.paths import get_cache_dir

# Downloaders
from prepare_annotations.downloaders.vcf import download_path

# HuggingFace
from prepare_annotations.huggingface.uploader import upload_parquet_to_hf
```

## Development

See [AGENTS.md](AGENTS.md) for development guidelines and repository layout.

### Running Tests

The project includes comprehensive test suites with automatic data download:

```bash
# Run all tests (excluding large downloads)
uv run pytest

# Run specific test file
uv run pytest tests/test_longevitymap_module.py -v

# Run with all markers (including large downloads)
uv run pytest -m ""
```

#### Test Features

- **Auto-download**: Tests automatically download required data from GitHub if not present
- **Integration tests**: Real data validation (no mocking unless necessary)
- **Module validation**: Comprehensive validation of converted module data

Example test modules:
- `test_longevitymap_module.py`: 47 tests validating longevitymap conversion accuracy
  - Validates weights table preservation (1043 rows, 528 variants)
  - Verifies APOE variant weights (rs7412, rs429358)
  - Tests schema transformations
  - Validates studies and annotations tables

The tests will automatically:
1. Download SQLite data from `dna-seq/just_longevitymap` if missing
2. Convert to unified parquet schema if needed
3. Run comprehensive validation checks

### Data Directories

```
data/
├── modules/                    # Downloaded module data
│   └── just_longevitymap/
│       └── longevitymap.sqlite
└── output/                     # Converted/processed data
    └── modules/
        └── longevitymap/
            ├── annotations.parquet
            ├── studies.parquet
            └── weights.parquet
```

## License

Apache 2.0

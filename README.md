# Prepare Annotations

A dedicated toolkit for downloading, processing, and preparing genomic annotation datasets.

## Features

- **Prefect-based Pipelines**: robust workflows for data preparation.
- **Support for multiple sources**:
  - **Ensembl**: Human genetic variations.
  - **ClinVar**: Clinical variant data.
  - **dbSNP**: Single Nucleotide Polymorphism database.
  - **gnomAD**: Genome Aggregation Database.
- **OakVar Module Management**: Download and convert data from [dna-seq](https://github.com/orgs/dna-seq/repositories) OakVar modules.
- **VCF to Parquet**: Efficient conversion of large VCF files to columnar format.
- **Variant Splitting**: Splitting variants by type (SNV, Indel, etc.) for optimized annotation.
- **Hugging Face Hub Integration**: Direct upload of processed datasets with automatic dataset card generation.

## Installation

This project uses `uv` for dependency management.

```bash
git clone https://github.com/dna-seq/prepare-annotations.git
cd prepare-annotations
uv sync
```

## Usage

### Main Genomic Data Pipeline

The `prepare-annotations` command handles large-scale genomic data downloads and processing.

```bash
# Show version
uv run prepare-annotations version

# Download and process Ensembl variations
uv run prepare-annotations ensembl --split --upload

# Download and process ClinVar data
uv run prepare-annotations clinvar --split --upload

# Download and process dbSNP data
uv run prepare-annotations dbsnp --build GRCh38 --split

# Download and process gnomAD data
uv run prepare-annotations gnomad --version v4 --split
```

#### Main Pipeline Options

- `--dest-dir`: Destination directory for downloads.
- `--split`: Split downloaded files by variant type.
- `--upload`: Upload results to Hugging Face Hub.
- `--repo-id`: Custom Hugging Face repository ID.

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

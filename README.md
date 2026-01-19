# Prepare Annotations

A dedicated toolkit for downloading, processing, and preparing genomic annotation datasets (Ensembl, ClinVar, dbSNP, gnomAD) using **Dagster** for robust, parallel, and observable pipelines.

## 🔷 Why Dagster?

Genomic data preparation is complex, involving multi-GB downloads and multi-step transformations. We use Dagster to provide:

- **Software-Defined Assets (SDA)**: Instead of just running "tasks", we define **Assets** (like a Parquet file). Dagster understands the dependencies between assets and only runs what is necessary.
- **Lineage & Observability**: You can visualize exactly which source VCF produced which output Parquet file. If a file looks wrong, you can trace it back to its source.
- **Dynamic Partitioning**: We discover files on remote servers (like Ensembl FTP) and create a "partition" for each. This allows fine-grained progress tracking and the ability to retry only failed files.
- **Parallelism & Concurrency**: Safe parallel execution with configurable limits to avoid overloading source servers or local system resources.
- **Self-Documenting**: The Dagster UI provides a live, interactive map of your data pipeline and its current state.

## Installation

This project uses `uv` for dependency management.

```bash
git clone https://github.com/dna-seq/prepare-annotations.git
cd prepare-annotations
uv sync
```

## Usage

### Running Pipelines

The primary entry points are `dagster-ensembl` for running jobs and `dagster-ui` for the web interface.

![Dagster Pipeline Lineage](images/pipelines.jpg)

```bash
# Run the full Ensembl pipeline (download → convert → upload)
uv run dagster-ensembl

# Run the LongevityMap pipeline (convert → join with Ensembl → upload)
uv run prepare longevitymap

# Start the Dagster UI for monitoring and lineage visualization
uv run dagster-ui

# Run for a specific species
uv run dagster-ensembl --species mus_musculus
```

### Advanced Operations

Use the `prepare` command for more granular control:

```bash
# List all available assets and jobs
uv run prepare assets
uv run prepare jobs

# Materialize specific assets
uv run prepare materialize ensembl_vcf_urls
uv run prepare materialize ensembl_vcf_file --partition homo_sapiens.vcf.gz
```

### OakVar Module Management

The `modules` command manages OakVar modules from the [dna-seq GitHub organization](https://github.com/orgs/dna-seq/repositories).

```bash
# Download data files from a module
uv run modules data --repo dna-seq/just_longevitymap

# Convert module data to unified schema
uv run modules convert-longevitymap
```

## Package Structure

The package follows Dagster best practices with utilities organized in subpackages:

```
src/prepare_annotations/
├── definitions.py          # Main Dagster definitions (assets, jobs, resources)
├── pipelines.py            # Standalone API for ClinVar, dbSNP, gnomAD (non-Dagster)
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

## Testing

```bash
# Run all tests (excluding large downloads)
uv run pytest

# Run specific module tests
uv run pytest tests/test_longevitymap_module.py -v
```

## License

Apache 2.0

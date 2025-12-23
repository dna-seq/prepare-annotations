# Prepare Annotations

A dedicated toolkit for downloading, processing, and preparing genomic annotation datasets.

## Features

- **Prefect-based Pipelines**: robust workflows for data preparation.
- **Support for multiple sources**:
  - **Ensembl**: Human genetic variations.
  - **ClinVar**: Clinical variant data.
  - **dbSNP**: Single Nucleotide Polymorphism database.
  - **gnomAD**: Genome Aggregation Database.
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

### Command Line Interface

The main entry point is the `prepare-annotations` command.

```bash
# Show version
uv run prepare-annotations version

# Download and process Ensembl variations
uv run prepare-annotations ensembl --split --upload

# Download and process ClinVar data
uv run prepare-annotations clinvar --split --upload
```

### Options

- `--dest-dir`: Destination directory for downloads.
- `--split`: Split downloaded files by variant type.
- `--upload`: Upload results to Hugging Face Hub.
- `--repo-id`: Custom Hugging Face repository ID.

## Development

See [AGENTS.md](AGENTS.md) for development guidelines and repository layout.

### Running Tests

```bash
uv run python -m pytest
```

## License

Apache 2.0

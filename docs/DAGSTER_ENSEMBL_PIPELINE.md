### Dagster Ensembl pipeline (prepare-annotations)

This repo includes a **Dagster** implementation of the Ensembl preparation pipeline as a parallel alternative to the Prefect flows.

The Dagster implementation lives under:
- `src/prepare_annotations/pipelines_dagster/`

It is intentionally **file/directory based**: each asset materializes a concrete on-disk artifact (a JSON manifest, a directory of VCFs, a directory of Parquet files, etc.). This makes lineage inspectable and keeps memory usage predictable.

---

### Key Features

- **Parallel downloads**: Configurable concurrent downloads (`max_concurrent_downloads`, default: 4)
- **Retry policies**: Exponential backoff retry policy on download failures (max 3 retries)
- **Checksum verification**: BSD sum checksum validation using CHECKSUMS file from Ensembl FTP
- **Resumable downloads**: fsspec filecache-based resumption for interrupted transfers
- Uploads directly from the non-split Parquet directory (no legacy TSA splitting in Dagster)

---

### What it does

The default pipeline prepares Ensembl VCFs into Parquet format:
- **Discover** remote VCF URLs on the Ensembl FTP (per species)
- **Download** the VCFs in parallel (with retries, resume, and optional CHECKSUMS verification)
- **Convert** VCF → Parquet (streaming write; no full in-memory collect)
- **Optional**: upload the Parquet dataset to HuggingFace Hub

---

### Asset graph / lineage

#### Mermaid diagram

```mermaid
flowchart TD
  A[ensembl_ftp_source<br/>(external)] --> B[ensembl_vcf_urls<br/>vcf_urls.json]
  B --> C[ensembl_vcf_files<br/>vcf/ directory<br/>(parallel downloads)]
  C --> D[ensembl_parquet_files<br/>species dir (*.parquet)]
  D --> F[ensembl_hf_upload<br/>(optional)]
```

#### ASCII diagram (fallback)

```
ensembl_ftp_source  (external)
        |
        v
ensembl_vcf_urls    (vcf_urls.json)
        |
        v
ensembl_vcf_files   (vcf/ directory, parallel downloads with retries)
        |
        v
ensembl_parquet_files  (species directory with *.parquet)
        |
        v
ensembl_hf_upload   (optional)
```

---

### On-disk layout (default)

Paths are resolved via `src/prepare_annotations/pipelines_dagster/resources.py`.

By default the pipeline writes to your user cache (same convention as other Just DNA tooling):
- Base cache dir: `~/.cache/just-dna-pipelines/` (or `JUST_DNA_PIPELINES_CACHE_DIR`)

For Ensembl:
- `~/.cache/just-dna-pipelines/ensembl/{species}/`
  - `vcf_urls.json` (URL manifest)
  - `vcf/` (downloaded `.vcf.gz` files)
  - `*.parquet` (per-chromosome conversions, e.g. `homo_sapiens-chr1.parquet`)

---

### Memory/performance model

- **Parallel downloads** via `ThreadPoolExecutor` with configurable concurrency (`max_concurrent_downloads`).
- **Retry policy** with exponential backoff (30s initial delay, up to 3 retries) at the Dagster asset level.
- **Resumable downloads** via fsspec filecache (interrupted downloads resume from where they left off).
- **Checksum verification** using BSD sum (`CHECKSUMS` file from Ensembl FTP); corrupted files are automatically re-downloaded.
- **VCF → Parquet** uses `polars-bio` scanning and `LazyFrame.sink_parquet(...)` to stream to disk.
- Dagster assets return **Paths** (manifest files / directories), not large Python lists, to avoid passing large in-memory objects between steps.

---

### How to run

#### Run via CLI (recommended)

Run the **full pipeline** (download → convert → upload):

```bash
uv run dagster-ensembl
```

This is equivalent to:

```bash
uv run dagster-ensembl run --job full
```

Run for a different species:

```bash
uv run dagster-ensembl run --species mus_musculus
```

Run specific jobs:

```bash
uv run dagster-ensembl run --job prepare   # download + convert (no upload)
uv run dagster-ensembl run --job download  # download only
uv run dagster-ensembl run --job convert   # convert only
uv run dagster-ensembl run --job upload    # upload only
```

List available jobs:

```bash
uv run dagster-ensembl jobs
```

#### Run via Dagster UI

Start the web interface for interactive execution:

```bash
uv run dagster-ensembl ui
```

Then materialize assets / jobs from the UI.

---

### Jobs provided

Jobs are defined in `src/prepare_annotations/pipelines_dagster/definitions.py`:

| Job | Description |
|-----|-------------|
| `full` | Complete pipeline: download → convert → upload **(default)** |
| `prepare` | Download and convert to Parquet (no upload) |
| `download` | Download VCF files only (parallel with retries) |
| `convert` | Convert VCF to Parquet (assumes VCFs downloaded) |
| `upload` | Upload to HuggingFace Hub (assumes parquet exists) |

---

### Configuration options

Key configuration parameters (set via Dagster config):

**EnsemblDownloadConfig:**
- `species`: Species name (default: `homo_sapiens`)
- `max_concurrent_downloads`: Maximum parallel downloads (default: `4`)
- `verify_checksums`: Whether to verify checksums (default: `True`)
- `retries`: Number of retry attempts per file (default: `10`)
- `connect_timeout`: Connection timeout in seconds (default: `10.0`)
- `sock_read_timeout`: Socket read timeout in seconds (default: `120.0`)

### HuggingFace upload lineage

The upload asset (`ensembl_hf_upload`) depends on the parquet directory output (`ensembl_parquet_files`). In the Dagster UI, this makes it straightforward to answer:
- "Which local dataset was uploaded?"
- "When did we last upload, and what was uploaded vs skipped?"

Uploads are executed using the existing uploader implementation:
- `prepare_annotations.preparation.huggingface_uploader.upload_parquet_to_hf`


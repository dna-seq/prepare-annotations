"""
Dagster assets for OakVar module conversion with Ensembl genotype resolution.

These assets handle the conversion of OakVar modules (e.g., longevitymap) to
unified annotation schema with proper genotype expansion based on zygosity.

Key features:
- DuckDB-powered joins for memory efficiency on Ensembl-scale data
- Ensembl data sourced from local cache OR HuggingFace Hub
- Proper heterozygous genotype resolution using Ensembl ref alleles
"""
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import polars as pl
from dagster import (
    asset,
    AssetExecutionContext,
    Output,
    MetadataValue,
)
from eliot import start_action

from prepare_annotations.core.paths import (
    get_default_ensembl_cache_dir,
    MODULES_DIR,
    MODULES_OUTPUT_DIR,
)
from prepare_annotations.pipelines.configs import (
    EnsemblSourceConfig,
    LongevityMapConfig,
    AnnotatorsUploadConfig,
    DuckDBConfig,
)
from prepare_annotations.converters import convert_module_weights_with_ensembl
from prepare_annotations.converters import (
    convert_longevitymap_annotations,
    convert_longevitymap_studies,
)


# ============================================================================
# ENSEMBL SOURCE ASSET
# ============================================================================


def resolve_ensembl_local_cache(
    config: EnsemblSourceConfig,
    logger,
) -> Optional[Path]:
    """Resolve the local Ensembl cache directory if available."""
    if config.local_cache_path:
        local_path = Path(config.local_cache_path)
        if local_path.exists():
            return local_path
        logger.warning(f"Specified local cache not found: {local_path}")

    if config.prefer_local:
        default_cache = get_default_ensembl_cache_dir(config.species)
        if default_cache.exists():
            parquet_files = list(default_cache.glob(f"{config.species}-chr*.parquet"))
            if parquet_files:
                return default_cache
    return None


def resolve_ensembl_source_path(
    config: EnsemblSourceConfig,
    logger,
) -> tuple[str, str]:
    """Resolve the Ensembl source path and label."""
    local_path = resolve_ensembl_local_cache(config, logger)
    if local_path is not None:
        return "local_cache", str(local_path)
    return "huggingface", f"hf://datasets/{config.hf_repo}"


def resolve_ensembl_parquet_files_from_source(source_path: str) -> list[str]:
    """Resolve a list of parquet files from a local or HuggingFace source path."""
    if source_path.startswith("hf://datasets/"):
        from huggingface_hub import HfApi

        repo_id = source_path.split("hf://datasets/", 1)[1].strip("/")
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = [
            f for f in repo_files if f.startswith("data/") and f.endswith(".parquet")
        ]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in HuggingFace repo {repo_id}")
        return [
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
            for f in parquet_files
        ]

    source = Path(source_path)
    if source.is_file():
        return [str(source)]
    if not source.exists():
        raise FileNotFoundError(f"Ensembl source not found: {source}")

    parquet_files = sorted(source.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {source}")
    return [str(p) for p in parquet_files]

@asset(
    description="Ensembl variation data for genotype resolution. "
                "Uses local Dagster pipeline cache if available, otherwise downloads from HuggingFace.",
    compute_kind="data_source",
    io_manager_key="io_manager",
    metadata={
        "format": "parquet",
        "storage": "cache",
    },
)
def ensembl_variations_source(
    context: AssetExecutionContext,
    config: EnsemblSourceConfig,
) -> Output[str]:
    """
    Resolve the Ensembl variations source path.
    
    Priority order:
    1. If local_cache_path is specified and exists, use it
    2. If prefer_local=True and default cache exists, use it
    3. Otherwise, stream from HuggingFace Hub
    
    Returns a source path (local cache dir or HuggingFace dataset URI).
    """
    logger = context.log
    
    with start_action(action_type="load_ensembl_variations") as action:
        source_used, source_path = resolve_ensembl_source_path(config, logger)
        if source_used == "local_cache":
            logger.info(f"Using local Ensembl cache: {source_path}")
            action.log(message_type="info", source="local", path=source_path)
        else:
            logger.info(f"Using HuggingFace Ensembl source: {source_path}")
            action.log(message_type="info", source="huggingface", repo=config.hf_repo)
    
    return Output(
        source_path,
        metadata={
            "source_type": MetadataValue.text(source_used),
            "source_path": MetadataValue.text(source_path),
            "species": MetadataValue.text(config.species),
        },
    )


# ============================================================================
# LONGEVITYMAP ASSETS
# ============================================================================

def get_longevitymap_db_path(config: LongevityMapConfig) -> Path:
    """Resolve the LongevityMap database path."""
    if config.db_path:
        return Path(config.db_path)
    
    # Default location
    default_path = MODULES_DIR / "just_longevitymap" / "longevitymap.sqlite"
    if default_path.exists():
        return default_path
    
    # Try alternate location
    alt_path = MODULES_DIR / "longevitymap" / "longevitymap.sqlite"
    if alt_path.exists():
        return alt_path
    
    raise FileNotFoundError(
        f"LongevityMap database not found. Tried:\n"
        f"  - {default_path}\n"
        f"  - {alt_path}\n"
        f"Use config.db_path to specify the location."
    )


def get_longevitymap_output_dir(config: LongevityMapConfig) -> Path:
    """Resolve the output directory for LongevityMap conversion."""
    if config.output_dir:
        return Path(config.output_dir)
    return MODULES_OUTPUT_DIR / config.module_name


@asset(
    description="LongevityMap annotations converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="io_manager",
    metadata={
        "schema": "rsid, module, gene, phenotype, category",
        "format": "parquet",
    },
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def longevitymap_annotations(
    context: AssetExecutionContext,
    config: LongevityMapConfig,
) -> Output[Path]:
    """
    Convert LongevityMap to annotations.parquet.
    
    Schema: rsid, module, gene, phenotype, category
    """
    logger = context.log
    
    db_path = get_longevitymap_db_path(config)
    output_dir = get_longevitymap_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.parquet"
    
    logger.info(f"Converting annotations from {db_path}")
    
    with start_action(action_type="convert_longevitymap_annotations", db_path=str(db_path)):
        annotations = convert_longevitymap_annotations(db_path)
        annotations.collect().write_parquet(output_path)
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} annotations to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="LongevityMap studies converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="io_manager",
    metadata={
        "schema": "rsid, module, pmid, population, p_value, conclusion, study_design",
        "format": "parquet",
    },
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def longevitymap_studies(
    context: AssetExecutionContext,
    config: LongevityMapConfig,
) -> Output[Path]:
    """
    Convert LongevityMap to studies.parquet.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    """
    logger = context.log
    
    db_path = get_longevitymap_db_path(config)
    output_dir = get_longevitymap_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "studies.parquet"
    
    logger.info(f"Converting studies from {db_path}")
    
    with start_action(action_type="convert_longevitymap_studies", db_path=str(db_path)):
        studies = convert_longevitymap_studies(db_path)
        studies.collect().write_parquet(output_path)
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} studies to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="LongevityMap weights converted to unified schema with Ensembl genotype resolution.",
    compute_kind="conversion",
    io_manager_key="io_manager",
    metadata={
        "schema": "rsid, genotype, module, weight, state, priority, conclusion, curator, method",
        "format": "parquet",
    },
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def longevitymap_weights(
    context: AssetExecutionContext,
    ensembl_variations_source: str,
    config: LongevityMapConfig,
) -> Output[Path]:
    """
    Convert LongevityMap to weights.parquet with Ensembl genotype resolution.
    
    Schema: rsid, genotype (list[str]), module, weight, state, priority, conclusion, curator, method
    
    Genotype expansion logic:
    - Homozygous (hom): allele "C" -> ["C", "C"]
    - Heterozygous (het) + spec: allele "CT" -> ["C", "T"]
    - Heterozygous (het) + alt: allele "C" + Ensembl ref -> ["C", "T"] (ref from Ensembl)
    """
    logger = context.log
    
    db_path = get_longevitymap_db_path(config)
    output_dir = get_longevitymap_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weights.parquet"
    
    logger.info(f"Converting weights from {db_path} with Ensembl genotype resolution")
    
    with start_action(
        action_type="convert_longevitymap_weights_with_ensembl",
        db_path=str(db_path),
    ) as action:
        # Use the common conversion function with Ensembl
        weights = convert_module_weights_with_ensembl(
            db_path=db_path,
            ensembl_source=ensembl_variations_source,
            module_name=config.module_name,
            curator=config.curator,
            method=config.method,
        )
        
        # Collect and write
        weights.collect().write_parquet(output_path)
        action.log(message_type="info", step="weights_written", path=str(output_path))
    
    # Get stats
    stats = pl.scan_parquet(output_path).select([
        pl.len().alias("row_count"),
        pl.col("rsid").n_unique().alias("unique_rsids"),
        pl.col("state").value_counts().alias("state_counts"),
    ]).collect()
    
    row_count = stats["row_count"][0]
    unique_rsids = stats["unique_rsids"][0]
    
    logger.info(f"Wrote {row_count} weights ({unique_rsids} unique variants) to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "row_count": MetadataValue.int(row_count),
            "unique_rsids": MetadataValue.int(unique_rsids),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
            "curator": MetadataValue.text(config.curator),
        },
    )


@asset(
    description="LongevityMap weights joined with Ensembl variation data for annotation.",
    compute_kind="join",
    io_manager_key="io_manager",
    metadata={
        "format": "parquet",
        "join_type": "inner",
    },
    op_tags={"dagster/concurrency_key": "ensembl_join"},
)
def longevitymap_with_ensembl(
    context: AssetExecutionContext,
    ensembl_variations_source: str,
    longevitymap_weights: Path,
    config: LongevityMapConfig,
) -> Output[Path]:
    """
    Join LongevityMap weights with full Ensembl variation data.
    
    This creates an enriched dataset with:
    - All LongevityMap weight columns
    - Ensembl variant info (chrom, start, end, ref, alt, etc.)
    
    The join is on rsid, matching genotype alleles with Ensembl alts.
    For heterozygous variants:
    - Joins each weight row with Ensembl rows where the curated allele is in alts
    
    This enables downstream analysis like:
    - Chromosome distribution of longevity variants
    - Clinical significance from ClinVar flags
    - Population frequencies
    """
    logger = context.log
    
    output_dir = get_longevitymap_output_dir(config)
    output_path = output_dir / "longevitymap_ensembl_joined.parquet"
    
    logger.info("Joining LongevityMap weights with Ensembl variations")
    
    with start_action(action_type="join_longevitymap_ensembl") as action:
        ensembl_files = resolve_ensembl_parquet_files_from_source(ensembl_variations_source)
        row_count = join_longevitymap_with_ensembl_duckdb(
            weights_path=Path(longevitymap_weights),
            ensembl_files=ensembl_files,
            output_path=output_path,
            duckdb_config=DuckDBConfig(),
        )
        action.log(
            message_type="info",
            step="joined_written",
            path=str(output_path),
            row_count=row_count,
        )
    
    # Get stats
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    
    logger.info(f"Wrote {row_count} joined rows to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
        },
    )


def _duckdb_quote_path(value: str) -> str:
    return value.replace("'", "''")


def join_longevitymap_with_ensembl_duckdb(
    *,
    weights_path: Path,
    ensembl_files: Sequence[str],
    output_path: Path,
    duckdb_config: DuckDBConfig,
) -> int:
    """Join LongevityMap weights with Ensembl data using DuckDB."""
    if not ensembl_files:
        raise FileNotFoundError("No Ensembl parquet files provided for join")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_sql = _duckdb_quote_path(str(weights_path))
    output_sql = _duckdb_quote_path(str(output_path))
    ensembl_list_sql = "[" + ", ".join(f"'{_duckdb_quote_path(p)}'" for p in ensembl_files) + "]"

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{duckdb_config.get_memory_limit()}'")
    Path(duckdb_config.temp_directory).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{_duckdb_quote_path(duckdb_config.temp_directory)}'")
    con.execute("SET preserve_insertion_order = false")

    if any(p.startswith(("http://", "https://")) for p in ensembl_files):
        try:
            con.execute("LOAD httpfs")
        except duckdb.Error as exc:
            raise RuntimeError(
                "DuckDB httpfs extension is required to read remote parquet files. "
                "Please ensure httpfs is available or use a local Ensembl cache."
            ) from exc

    con.execute(
        f"""
        COPY (
            WITH weights AS (
                SELECT * FROM read_parquet('{weights_sql}')
            ),
            ensembl AS (
                SELECT
                    id,
                    chrom,
                    start,
                    "end",
                    ref,
                    alts,
                    "ClinVar_202502" AS clinvar,
                    "CLIN_pathogenic" AS pathogenic,
                    "CLIN_benign" AS benign,
                    "CLIN_likely_pathogenic" AS likely_pathogenic,
                    "CLIN_likely_benign" AS likely_benign
                FROM read_parquet({ensembl_list_sql})
            ),
            weights_exploded AS (
                SELECT w.*, allele
                FROM weights w, UNNEST(w.genotype) AS allele
            ),
            joined AS (
                SELECT
                    w.*,
                    e.chrom,
                    e.start,
                    e."end",
                    e.ref,
                    e.alts,
                    e.clinvar,
                    e.pathogenic,
                    e.benign,
                    e.likely_pathogenic,
                    e.likely_benign
                FROM weights_exploded w
                JOIN ensembl e ON e.id = w.rsid
                WHERE w.allele = e.ref
                   OR list_contains(CASE WHEN e.alts IS NULL THEN [] ELSE e.alts END, w.allele)
            )
            SELECT
                rsid,
                genotype,
                module,
                weight,
                state,
                priority,
                conclusion,
                curator,
                method,
                chrom,
                start,
                "end",
                ref,
                ANY_VALUE(alts) AS alts,
                ANY_VALUE(clinvar) AS clinvar,
                ANY_VALUE(pathogenic) AS pathogenic,
                ANY_VALUE(benign) AS benign,
                ANY_VALUE(likely_pathogenic) AS likely_pathogenic,
                ANY_VALUE(likely_benign) AS likely_benign
            FROM joined
            GROUP BY
                rsid,
                genotype,
                module,
                weight,
                state,
                priority,
                conclusion,
                curator,
                method,
                chrom,
                start,
                "end",
                ref
        ) TO '{output_sql}' (
            FORMAT 'PARQUET',
            COMPRESSION 'ZSTD',
            COMPRESSION_LEVEL 14
        )
        """
    )

    row_count = con.execute(f"SELECT count(*) FROM read_parquet('{output_sql}')").fetchone()[0]
    con.close()
    return int(row_count)


# ============================================================================
# HUGGINGFACE UPLOAD ASSET
# ============================================================================

@asset(
    description="Upload LongevityMap module parquet files to HuggingFace Hub.",
    compute_kind="upload",
    io_manager_key="hf_upload_io_manager",
    metadata={
        "destination": "HuggingFace Hub",
        "repo": "just-dna-seq/annotators",
        "storage": "remote",
    },
)
def longevitymap_hf_upload(
    context: AssetExecutionContext,
    longevitymap_annotations: Path,
    longevitymap_studies: Path,
    longevitymap_weights: Path,
    config: AnnotatorsUploadConfig,
) -> Output[dict]:
    """
    Upload LongevityMap parquet files to HuggingFace Hub.
    
    Files are uploaded to:
      just-dna-seq/annotators/data/longevitymap/
        - annotations.parquet
        - studies.parquet  
        - weights.parquet
    
    Uses batch upload for efficiency (single commit for all files).
    Only uploads files that differ in size from remote versions.
    """
    from prepare_annotations.huggingface.uploader import upload_files_batch
    from prepare_annotations.core.models import SingleUploadResult
    
    logger = context.log
    
    # Collect all parquet files
    parquet_files = [
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
    ]
    
    # Create paths in repo (data/longevitymap/filename.parquet)
    path_in_repos = [
        f"{config.path_prefix}/longevitymap/{f.name}"
        for f in parquet_files
    ]
    
    logger.info(f"Uploading {len(parquet_files)} files to {config.repo_id}")
    for f, p in zip(parquet_files, path_in_repos):
        logger.info(f"  {f.name} -> {p}")
    
    with start_action(
        action_type="upload_longevitymap_to_hf",
        repo_id=config.repo_id,
        num_files=len(parquet_files),
    ) as action:
        # Generate dataset card content
        dataset_card = _generate_annotators_card([
            {"name": "longevitymap", "files": parquet_files}
        ])
        
        result = upload_files_batch(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            path_in_repos=path_in_repos,
            repo_type="dataset",
            token=config.token,
            commit_message="Update longevitymap module",
            dataset_card_content=dataset_card,
        )
        
        action.log(
            message_type="success",
            uploaded=result.num_uploaded,
            skipped=result.num_skipped,
        )
    
    logger.info(f"Upload complete: {result.num_uploaded} uploaded, {result.num_skipped} skipped")
    
    return Output(
        {
            "repo_id": config.repo_id,
            "num_uploaded": result.num_uploaded,
            "num_skipped": result.num_skipped,
            "files": [r.path_in_repo for r in result.uploaded_files],
        },
        metadata={
            "repo_id": MetadataValue.text(config.repo_id),
            "num_uploaded": MetadataValue.int(result.num_uploaded),
            "num_skipped": MetadataValue.int(result.num_skipped),
            "url": MetadataValue.url(f"https://huggingface.co/datasets/{config.repo_id}"),
        },
    )


def _generate_annotators_card(modules: list[dict]) -> str:
    """Generate dataset card for annotators repository."""
    module_names = [m["name"] for m in modules]
    
    total_files = sum(len(m["files"]) for m in modules)
    total_size_mb = sum(
        f.stat().st_size for m in modules for f in m["files"]
    ) / (1024 * 1024)
    
    return f'''---
license: mit
tags:
  - biology
  - genetics
  - genomics
  - variants
  - annotation
  - longevity
  - pharmacogenomics
language:
  - en
size_categories:
  - 1K<n<10K
---

# Genomic Variant Annotators

Curated genomic variant annotation modules from the [DNA-seq](https://github.com/dna-seq) project.

## Overview

This dataset contains pre-computed annotation data for genetic variants, organized by module:

| Module | Description | Files |
|--------|-------------|-------|
| longevitymap | Longevity-associated variants | annotations.parquet, studies.parquet, weights.parquet |

## Schema

### annotations.parquet
Variant-level facts linking rsIDs to genes and phenotypes.
- `rsid`: dbSNP reference ID
- `module`: Source module name
- `gene`: Associated gene symbol
- `phenotype`: Associated phenotype/trait
- `category`: Functional category

### studies.parquet
Per-study evidence from scientific publications.
- `rsid`: dbSNP reference ID
- `module`: Source module name
- `pmid`: PubMed ID
- `population`: Study population
- `p_value`: Statistical significance
- `conclusion`: Study conclusion
- `study_design`: Type of study

### weights.parquet
Curator-defined scoring for variant impact.
- `rsid`: dbSNP reference ID
- `genotype`: Genotype as list[str] (e.g., ["C", "T"])
- `module`: Source module name
- `weight`: Numeric weight
- `state`: "protective", "risk", or "neutral"
- `priority`: Priority level
- `conclusion`: Curator conclusion
- `curator`: Curator name
- `method`: Curation method

## Usage

```python
import polars as pl

# Load from HuggingFace
weights = pl.read_parquet("hf://datasets/just-dna-seq/annotators/data/longevitymap/weights.parquet")
studies = pl.read_parquet("hf://datasets/just-dna-seq/annotators/data/longevitymap/studies.parquet")
annotations = pl.read_parquet("hf://datasets/just-dna-seq/annotators/data/longevitymap/annotations.parquet")
```

## Statistics

- **Modules**: {len(module_names)} ({", ".join(module_names)})
- **Total files**: {total_files}
- **Total size**: {total_size_mb:.2f} MB

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use this data, please cite the original sources:
- LongevityMap: [https://longevitymap.org/](https://longevitymap.org/)
'''


# ============================================================================
# EXPORT ALL ASSETS
# ============================================================================

module_assets = [
    ensembl_variations_source,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
]

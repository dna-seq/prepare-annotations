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
from eliot import start_action, log_message
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from prepare_annotations.core.runtime import resource_tracker
from prepare_annotations.core.paths import (
    get_default_ensembl_cache_dir,
    MODULES_DIR,
    MODULES_OUTPUT_DIR,
)
from prepare_annotations.converters.common import (
    check_huggingface_auth,
    configure_duckdb_for_hf,
)
import yaml
from prepare_annotations.core.models import ModuleMetadata

# ============================================================================
# MODULE METADATA MAPPING
# ============================================================================

MODULE_METADATA_MAP = {
    "longevitymap": ModuleMetadata(
        name="Longevitymap postagregator",
        description="Longevity map postagregator for longevity report.",
        image_url="/store/remotelogo?module=just_longevitymap&store=ov"
    ),
    "lipidmetabolism": ModuleMetadata(
        name="Lipid metabolism postaggregator",
        description="Postaggregator for Longevity2reporter. It deppends on annotators dbsnp, longevitymap, clinvar, omim, ncbigene, pubmed, gnomad.",
        image_url="/store/remotelogo?module=just_lipidmetabolism&store=ov"
    ),
    "vo2max": ModuleMetadata(
        name="VO2max postaggregator",
        description="Postaggregator for Longevity2reporter. It deppends on annotators dbsnp, longevitymap, clinvar, omim, ncbigene, pubmed, gnomad.",
        image_url="/store/remotelogo?module=just_vo2max&store=ov"
    ),
    "superhuman": ModuleMetadata(
        name="Superhumangenes postaggregator",
        description="Postaggregator for superhumangenes reporter. It deppends on annotators dbsnp, longevitymap, clinvar, omim, ncbigene, pubmed, gnomad.",
        image_url="/store/remotelogo?module=just_superhuman&store=ov"
    ),
    "coronary": ModuleMetadata(
        name="Coronary disease postagregator",
        description="Coronary disease risks postagregator for longevity report.",
        image_url="/store/remotelogo?module=just_coronary&store=ov"
    ),
    "drugs": ModuleMetadata(
        name="Longevity drugs postagregator",
        description="Drugs genetic specific postagregator for longevity report.",
        image_url="/store/remotelogo?module=just_drugs&store=ov"
    ),
    "cancer": ModuleMetadata(
        name="Cancer postagregator",
        description="Cancer risks postagregator for longevity report.",
        image_url="/store/remotelogo?module=just_cancer&store=ov"
    ),
    "cardio": ModuleMetadata(
        name="Cardio postagregator",
        description="Cardio risks postagregator for longevity report.",
        image_url="/store/remotelogo?module=just_cardio&store=ov"
    ),
    "prs": ModuleMetadata(
        name="Prs postagregator",
        description="Pologenic Risk Score (PRS) postagregator for longevity report.",
        image_url="/store/remotelogo?module=just_prs&store=ov"
    ),
    "thrombophilia": ModuleMetadata(
        name="Thrombophilia risks postaggregator",
        description="Postaggregator for Longevity2reporter. It deppends on annotators dbsnp, longevitymap, clinvar, omim, ncbigene, pubmed, gnomad.",
        image_url="/store/remotelogo?module=just_thrombophilia&store=ov"
    )
}


def _get_module_metadata_yaml(module_name: str) -> Optional[str]:
    """Get metadata YAML content for a module if available."""
    metadata = MODULE_METADATA_MAP.get(module_name)
    if metadata:
        return yaml.dump(metadata.model_dump(), sort_keys=False)
    return None


def _get_module_icon_path(module_name: str) -> Optional[Path]:
    """Get the local path to the module icon if available."""
    icon_path = Path("data/logos") / f"{module_name}.jpg"
    if icon_path.exists():
        return icon_path
    return None


from prepare_annotations.core.io import polars_schema_to_table_schema
from prepare_annotations.core.dagster_configs import (
    EnsemblSourceConfig,
    LongevityMapSourceConfig,
    LongevityMapConfig,
    LipidMetabolismSourceConfig,
    LipidMetabolismConfig,
    VO2MaxSourceConfig,
    VO2MaxConfig,
    SuperhumanSourceConfig,
    SuperhumanConfig,
    CoronarySourceConfig,
    CoronaryConfig,
    DrugsSourceConfig,
    DrugsConfig,
    AnnotatorsUploadConfig,
    DuckDBConfig,
)
from prepare_annotations.converters import (
    convert_module_weights_with_ensembl,
    convert_longevitymap_annotations,
    convert_longevitymap_studies,
    convert_lipidmetabolism_annotations,
    convert_lipidmetabolism_studies,
    convert_lipidmetabolism_weights,
    convert_vo2max_annotations,
    convert_vo2max_studies,
    convert_vo2max_weights,
    convert_superhuman_annotations,
    convert_superhuman_studies,
    convert_superhuman_weights,
    convert_coronary_annotations,
    convert_coronary_studies,
    convert_coronary_weights,
    convert_drugs_annotations,
    convert_drugs_studies,
    convert_drugs_weights,
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
# LONGEVITYMAP SOURCE ASSET
# ============================================================================

def _download_file_with_progress(url: str, output_path: Path, logger) -> Path:
    """Download a file from URL with progress logging."""
    import httpx
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                    pct = (downloaded / total) * 100
                    logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB ({pct:.1f}%)")
    
    logger.info(f"Download complete: {output_path}")
    return output_path


@asset(
    description="LongevityMap SQLite database downloaded from GitHub.",
    compute_kind="download",
    io_manager_key="module_io_manager",
    metadata={
        "format": "sqlite",
        "source": "github",
        "repo": "dna-seq/just_longevitymap",
    },
)
def longevitymap_sqlite(
    context: AssetExecutionContext,
    config: LongevityMapSourceConfig,
) -> Output[Path]:
    """
    Download LongevityMap SQLite database from GitHub.
    
    Source: https://github.com/dna-seq/just_longevitymap/blob/master/data/longevitymap.sqlite
    
    The file is cached locally and only re-downloaded if force_download=True.
    """
    logger = context.log
    
    # Output path in modules directory
    output_path = MODULES_DIR / "just_longevitymap" / "longevitymap.sqlite"
    
    # Check if already exists
    if output_path.exists() and not config.force_download:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Using cached SQLite: {output_path} ({size_mb:.2f} MB)")
        return Output(
            output_path,
            metadata={
                "source": MetadataValue.text("cached"),
                "path": MetadataValue.path(str(output_path)),
                "size_mb": MetadataValue.float(size_mb),
            },
        )
    
    # Download from GitHub
    url = config.download_url
    logger.info(f"Downloading LongevityMap from {url}")
    
    with start_action(
        action_type="download_longevitymap_sqlite",
        url=url,
        output_path=str(output_path),
    ) as action:
        _download_file_with_progress(url, output_path, logger)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        action.log(message_type="success", size_mb=size_mb)
    
    return Output(
        output_path,
        metadata={
            "source": MetadataValue.text("github"),
            "url": MetadataValue.url(url),
            "path": MetadataValue.path(str(output_path)),
            "size_mb": MetadataValue.float(size_mb),
        },
    )


# ============================================================================
# LONGEVITYMAP ASSETS
# ============================================================================


def get_longevitymap_output_dir(config: LongevityMapConfig) -> Path:
    """Resolve the output directory for LongevityMap conversion."""
    if config.output_dir:
        return Path(config.output_dir)
    return MODULES_OUTPUT_DIR / config.module_name


@asset(
    description="LongevityMap annotations converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="polars_parquet_io_manager",
    metadata={
        "schema": "rsid, module, gene, phenotype, category",
        "format": "parquet",
        "compression": "zstd",
        "compression_level": 14,
    },
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def longevitymap_annotations(
    context: AssetExecutionContext,
    longevitymap_sqlite: Path,
) -> pl.LazyFrame:
    """
    Convert LongevityMap to annotations.parquet.
    
    Schema: rsid, module, gene, phenotype, category
    
    Uses dagster-polars PolarsParquetIOManager for:
    - Automatic sink_parquet (streaming, memory-efficient)
    - Schema metadata in Dagster UI
    - Sample data preview
    """
    logger = context.log
    logger.info(f"Converting annotations from {longevitymap_sqlite}")
    
    with start_action(action_type="convert_longevitymap_annotations", db_path=str(longevitymap_sqlite)):
        return convert_longevitymap_annotations(longevitymap_sqlite)


@asset(
    description="LongevityMap studies converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="polars_parquet_io_manager",
    metadata={
        "schema": "rsid, module, pmid, population, p_value, conclusion, study_design",
        "format": "parquet",
        "compression": "zstd",
        "compression_level": 14,
    },
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def longevitymap_studies(
    context: AssetExecutionContext,
    longevitymap_sqlite: Path,
) -> pl.LazyFrame:
    """
    Convert LongevityMap to studies.parquet.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    
    Uses dagster-polars PolarsParquetIOManager for:
    - Automatic sink_parquet (streaming, memory-efficient)
    - Schema metadata in Dagster UI
    - Sample data preview
    """
    logger = context.log
    logger.info(f"Converting studies from {longevitymap_sqlite}")
    
    with start_action(action_type="convert_longevitymap_studies", db_path=str(longevitymap_sqlite)):
        return convert_longevitymap_studies(longevitymap_sqlite)


@asset(
    description="LongevityMap weights converted to unified schema with Ensembl genotype resolution.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={
        "schema": "rsid, genotype, module, weight, state, priority, conclusion, curator, method",
        "format": "parquet",
    },
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def longevitymap_weights(
    context: AssetExecutionContext,
    longevitymap_sqlite: Path,
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
    
    output_dir = get_longevitymap_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weights.parquet"
    
    logger.info(f"Converting weights from {longevitymap_sqlite} with Ensembl genotype resolution")
    
    with resource_tracker("longevitymap_weights", context=context):
        with start_action(
            action_type="convert_longevitymap_weights_with_ensembl",
            db_path=str(longevitymap_sqlite),
        ) as action:
            # Use the common conversion function with Ensembl
            weights = convert_module_weights_with_ensembl(
                db_path=longevitymap_sqlite,
                ensembl_source=ensembl_variations_source,
                module_name=config.module_name,
                curator=config.curator,
                method=config.method,
            )
            
            # Collect and write
            weights.sink_parquet(output_path, engine="streaming")
            action.log(message_type="info", step="weights_written", path=str(output_path))
    
    # Get stats (keep lightweight to avoid high memory usage)
    stats = pl.scan_parquet(output_path).select([
        pl.len().alias("row_count"),
    ]).collect()
    
    row_count = stats["row_count"][0]
    
    logger.info(f"Wrote {row_count} weights to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
            "curator": MetadataValue.text(config.curator),
        },
    )


@asset(
    description="LongevityMap weights joined with Ensembl variation data for annotation.",
    compute_kind="join",
    io_manager_key="module_io_manager",
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
    
    with resource_tracker("longevitymap_with_ensembl", context=context):
        with start_action(action_type="join_longevitymap_ensembl") as action:
            ensembl_files = resolve_ensembl_parquet_files_from_source(ensembl_variations_source)
            row_count = join_weights_with_ensembl_duckdb(
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
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
        },
    )


def _duckdb_quote_path(value: str) -> str:
    return value.replace("'", "''")


def join_weights_with_ensembl_duckdb(
    *,
    weights_path: Path,
    ensembl_files: Sequence[str],
    output_path: Path,
    duckdb_config: DuckDBConfig,
) -> int:
    """Join module weights with Ensembl data using DuckDB."""
    if not ensembl_files:
        raise FileNotFoundError("No Ensembl parquet files provided for join")

    # Check HuggingFace authentication
    check_huggingface_auth()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_sql = _duckdb_quote_path(str(weights_path))
    output_sql = _duckdb_quote_path(str(output_path))
    ensembl_list_sql = "[" + ", ".join(f"'{_duckdb_quote_path(p)}'" for p in ensembl_files) + "]"

    con = duckdb.connect()
    
    # Configure DuckDB with rate limiting for HuggingFace (use 2 connections to avoid rate limiting)
    configure_duckdb_for_hf(
        con, 
        memory_limit=duckdb_config.get_memory_limit(), 
        temp_directory=Path(duckdb_config.temp_directory), 
        max_connections=2
    )

    if any(p.startswith(("http://", "https://")) for p in ensembl_files):
        # Avoid relying on ~/.duckdb (can be missing/unwritable on some systems).
        from prepare_annotations.core.paths import get_default_cache_dir

        ext_dir = get_default_cache_dir("duckdb") / "extensions"
        ext_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET extension_directory = '{_duckdb_quote_path(str(ext_dir))}'")
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")

    query = f"""
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
                SELECT 
                    *,
                    UNNEST(genotype) AS allele,
                    -- Compute complement for strand normalization (A<->T, C<->G)
                    UNNEST(list_transform(genotype, a -> 
                        CASE a 
                            WHEN 'A' THEN 'T' 
                            WHEN 'T' THEN 'A' 
                            WHEN 'C' THEN 'G' 
                            WHEN 'G' THEN 'C' 
                            ELSE a 
                        END
                    )) AS allele_complement
                FROM weights
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
                   OR list_contains(COALESCE(e.alts, CAST([] AS VARCHAR[])), w.allele)
                   -- Strand normalization: also match complement alleles
                   OR w.allele_complement = e.ref
                   OR list_contains(COALESCE(e.alts, CAST([] AS VARCHAR[])), w.allele_complement)
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
    
    # Execute query with retry logic for HTTP 429 errors
    # Use longer wait times: 10s, 20s, 40s, 80s, 160s (capped at 120s)
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=5, min=10, max=120),
        reraise=True
    )
    def execute_with_retry():
        """Execute DuckDB query with retry logic for rate limiting."""
        try:
            con.execute(query)
            log_message(
                message_type="duckdb_join_success",
                num_files=len(ensembl_files)
            )
        except Exception as e:
            error_msg = str(e)
            # Check if this is an HTTP 429 error that should be retried
            if "HTTP 429" in error_msg or "Too Many Requests" in error_msg or "HTTPException" in str(type(e)):
                log_message(
                    message_type="duckdb_http_429_retry",
                    error=error_msg,
                    error_type=str(type(e)),
                    retry_wait="exponential backoff (10s -> 120s)"
                )
                raise  # Trigger retry
            else:
                # For non-429 errors, log and re-raise without retry
                log_message(
                    message_type="duckdb_join_error",
                    error=error_msg,
                    error_type=str(type(e))
                )
                raise
    
    try:
        execute_with_retry()
        row_count = con.execute(f"SELECT count(*) FROM read_parquet('{output_sql}')").fetchone()[0]
    finally:
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
    longevitymap_annotations: pl.LazyFrame,
    longevitymap_studies: pl.LazyFrame,
    longevitymap_with_ensembl: Path,
    config: AnnotatorsUploadConfig,
) -> Output[dict]:
    """
    Upload LongevityMap parquet files to HuggingFace Hub.
    
    Files are uploaded to:
      just-dna-seq/annotators/data/longevitymap/
        - annotations.parquet
        - studies.parquet  
        - weights.parquet (joined with Ensembl)
    
    Uses batch upload for efficiency (single commit for all files).
    Only uploads files that differ in size from remote versions.
    
    Note: annotations and studies are LazyFrame inputs from dagster-polars IO manager,
    the file paths are computed from the IO manager's base_dir convention.
    """
    from prepare_annotations.huggingface.uploader import upload_files_batch
    
    logger = context.log
    
    # For LazyFrame inputs from dagster-polars, compute paths from IO manager convention
    # dagster-polars stores at: base_dir / asset_key.parquet
    annotations_path = MODULES_OUTPUT_DIR / "longevitymap_annotations.parquet"
    studies_path = MODULES_OUTPUT_DIR / "longevitymap_studies.parquet"
    
    # Verify the files exist (they should, since dagster-polars wrote them)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")
    if not studies_path.exists():
        raise FileNotFoundError(f"Studies file not found at {studies_path}")
    
    # longevitymap_with_ensembl is still Path-based (uses DuckDB)
    parquet_files = [
        annotations_path,
        studies_path,
        longevitymap_with_ensembl,
    ]
    
    # Create paths in repo (data/longevitymap/filename.parquet)
    path_in_repos = [
        f"{config.path_prefix}/longevitymap/annotations.parquet",
        f"{config.path_prefix}/longevitymap/studies.parquet",
        f"{config.path_prefix}/longevitymap/weights.parquet",
    ]
    
    logger.info(f"Uploading {len(parquet_files)} files to {config.repo_id}")
    for f, p in zip(parquet_files, path_in_repos):
        logger.info(f"  {f} -> {p}")
    
    with start_action(
        action_type="upload_longevitymap_to_hf",
        repo_id=config.repo_id,
        num_files=len(parquet_files),
    ) as action:
        # Generate dataset card content
        dataset_card = _generate_annotators_card([
            {"name": "longevitymap", "files": parquet_files}
        ])
        
        # Generate metadata YAML
        metadata_yaml = _get_module_metadata_yaml("longevitymap")
        
        # Get icon path
        icon_path = _get_module_icon_path("just_longevitymap")
        
        result = upload_files_batch(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            path_in_repos=path_in_repos,
            repo_type="dataset",
            token=config.token,
            commit_message="Update longevitymap module",
            dataset_card_content=dataset_card,
            metadata_yaml_content=metadata_yaml,
            metadata_yaml_path_in_repo=f"{config.path_prefix}/longevitymap/metadata.yaml",
            icon_path=icon_path,
            icon_path_in_repo=f"{config.path_prefix}/longevitymap/logo.jpg",
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
# LIPIDMETABOLISM ASSETS
# ============================================================================


@asset(
    description="LipidMetabolism SQLite database downloaded from GitHub.",
    compute_kind="download",
    io_manager_key="module_io_manager",
    metadata={
        "format": "sqlite",
        "source": "github",
        "repo": "dna-seq/just_lipidmetabolism",
    },
)
def lipidmetabolism_sqlite(
    context: AssetExecutionContext,
    config: LipidMetabolismSourceConfig,
) -> Output[Path]:
    """Download LipidMetabolism SQLite database from GitHub."""
    logger = context.log
    output_path = MODULES_DIR / "just_lipidmetabolism" / "lipid_metabolism.sqlite"
    
    if output_path.exists() and not config.force_download:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Using cached SQLite: {output_path} ({size_mb:.2f} MB)")
        return Output(
            output_path,
            metadata={
                "source": MetadataValue.text("cached"),
                "path": MetadataValue.path(str(output_path)),
                "size_mb": MetadataValue.float(size_mb),
            },
        )
    
    url = config.download_url
    logger.info(f"Downloading LipidMetabolism from {url}")
    
    with start_action(action_type="download_lipidmetabolism_sqlite", url=url):
        _download_file_with_progress(url, output_path, logger)
        size_mb = output_path.stat().st_size / (1024 * 1024)
    
    return Output(
        output_path,
        metadata={
            "source": MetadataValue.text("github"),
            "url": MetadataValue.url(url),
            "path": MetadataValue.path(str(output_path)),
            "size_mb": MetadataValue.float(size_mb),
        },
    )


def get_lipidmetabolism_output_dir(config: LipidMetabolismConfig) -> Path:
    """Resolve the output directory for LipidMetabolism conversion."""
    if config.output_dir:
        return Path(config.output_dir)
    return MODULES_OUTPUT_DIR / config.module_name


@asset(
    description="LipidMetabolism annotations converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, gene, phenotype, category", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def lipidmetabolism_annotations(
    context: AssetExecutionContext,
    lipidmetabolism_sqlite: Path,
    config: LipidMetabolismConfig,
) -> Output[Path]:
    """Convert LipidMetabolism to annotations.parquet."""
    logger = context.log
    output_dir = get_lipidmetabolism_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.parquet"
    
    logger.info(f"Converting annotations from {lipidmetabolism_sqlite}")
    
    with start_action(action_type="convert_lipidmetabolism_annotations"):
        annotations = convert_lipidmetabolism_annotations(lipidmetabolism_sqlite)
        annotations.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} annotations to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="LipidMetabolism studies converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, pmid, population, p_value, conclusion, study_design", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def lipidmetabolism_studies(
    context: AssetExecutionContext,
    lipidmetabolism_sqlite: Path,
    config: LipidMetabolismConfig,
) -> Output[Path]:
    """Convert LipidMetabolism to studies.parquet."""
    logger = context.log
    output_dir = get_lipidmetabolism_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "studies.parquet"
    
    logger.info(f"Converting studies from {lipidmetabolism_sqlite}")
    
    with start_action(action_type="convert_lipidmetabolism_studies"):
        studies = convert_lipidmetabolism_studies(lipidmetabolism_sqlite)
        studies.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} studies to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="LipidMetabolism weights converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, genotype, module, weight, state, priority, conclusion, curator, method", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def lipidmetabolism_weights(
    context: AssetExecutionContext,
    lipidmetabolism_sqlite: Path,
    config: LipidMetabolismConfig,
) -> Output[Path]:
    """Convert LipidMetabolism to weights.parquet."""
    logger = context.log
    output_dir = get_lipidmetabolism_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weights.parquet"
    
    logger.info(f"Converting weights from {lipidmetabolism_sqlite}")
    
    with resource_tracker("lipidmetabolism_weights", context=context):
        with start_action(action_type="convert_lipidmetabolism_weights"):
            weights = convert_lipidmetabolism_weights(
                lipidmetabolism_sqlite,
                curator=config.curator,
                method=config.method,
            )
            weights.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} weights to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
            "curator": MetadataValue.text(config.curator),
        },
    )


@asset(
    description="LipidMetabolism weights joined with Ensembl variation data.",
    compute_kind="join",
    io_manager_key="module_io_manager",
    metadata={"format": "parquet", "join_type": "inner"},
    op_tags={"dagster/concurrency_key": "ensembl_join"},
)
def lipidmetabolism_with_ensembl(
    context: AssetExecutionContext,
    ensembl_variations_source: str,
    lipidmetabolism_weights: Path,
    config: LipidMetabolismConfig,
) -> Output[Path]:
    """Join LipidMetabolism weights with Ensembl variation data for chromosomal positions and ClinVar."""
    logger = context.log
    output_dir = get_lipidmetabolism_output_dir(config)
    output_path = output_dir / "lipidmetabolism_ensembl_joined.parquet"
    
    logger.info("Joining LipidMetabolism weights with Ensembl variations")
    
    with start_action(action_type="join_lipidmetabolism_ensembl") as action:
        ensembl_files = resolve_ensembl_parquet_files_from_source(ensembl_variations_source)
        row_count = join_weights_with_ensembl_duckdb(
            weights_path=Path(lipidmetabolism_weights),
            ensembl_files=ensembl_files,
            output_path=output_path,
            duckdb_config=DuckDBConfig(),
        )
        action.log(message_type="info", step="joined_written", path=str(output_path), row_count=row_count)
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} joined rows to {output_path}")
    
    return Output(output_path, metadata={"dagster/column_schema": polars_schema_to_table_schema(output_path), "row_count": MetadataValue.int(row_count), "output_path": MetadataValue.path(str(output_path))})


@asset(
    description="Upload LipidMetabolism module parquet files to HuggingFace Hub.",
    compute_kind="upload",
    io_manager_key="hf_upload_io_manager",
    metadata={"destination": "HuggingFace Hub", "repo": "just-dna-seq/annotators"},
)
def lipidmetabolism_hf_upload(
    context: AssetExecutionContext,
    lipidmetabolism_annotations: Path,
    lipidmetabolism_studies: Path,
    lipidmetabolism_with_ensembl: Path,
    config: AnnotatorsUploadConfig,
) -> Output[dict]:
    """Upload LipidMetabolism parquet files to HuggingFace Hub."""
    from prepare_annotations.huggingface.uploader import upload_files_batch
    
    logger = context.log
    
    parquet_files = [lipidmetabolism_annotations, lipidmetabolism_studies, lipidmetabolism_with_ensembl]
    path_in_repos = [
        f"{config.path_prefix}/lipidmetabolism/annotations.parquet",
        f"{config.path_prefix}/lipidmetabolism/studies.parquet",
        f"{config.path_prefix}/lipidmetabolism/weights.parquet",
    ]
    
    logger.info(f"Uploading {len(parquet_files)} files to {config.repo_id}")
    
    with start_action(action_type="upload_lipidmetabolism_to_hf", repo_id=config.repo_id):
        # Generate metadata YAML
        metadata_yaml = _get_module_metadata_yaml("lipidmetabolism")
        
        # Get icon path
        icon_path = _get_module_icon_path("just_lipidmetabolism")

        result = upload_files_batch(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            path_in_repos=path_in_repos,
            repo_type="dataset",
            token=config.token,
            commit_message="Update lipidmetabolism module",
            metadata_yaml_content=metadata_yaml,
            metadata_yaml_path_in_repo=f"{config.path_prefix}/lipidmetabolism/metadata.yaml",
            icon_path=icon_path,
            icon_path_in_repo=f"{config.path_prefix}/lipidmetabolism/logo.jpg",
        )
    
    logger.info(f"Upload complete: {result.num_uploaded} uploaded, {result.num_skipped} skipped")
    
    return Output(
        {"repo_id": config.repo_id, "num_uploaded": result.num_uploaded, "num_skipped": result.num_skipped},
        metadata={
            "repo_id": MetadataValue.text(config.repo_id),
            "num_uploaded": MetadataValue.int(result.num_uploaded),
            "num_skipped": MetadataValue.int(result.num_skipped),
            "url": MetadataValue.url(f"https://huggingface.co/datasets/{config.repo_id}"),
        },
    )


# ============================================================================
# VO2MAX ASSETS
# ============================================================================


@asset(
    description="VO2Max SQLite database downloaded from GitHub.",
    compute_kind="download",
    io_manager_key="module_io_manager",
    metadata={"format": "sqlite", "source": "github", "repo": "dna-seq/just_vo2max"},
)
def vo2max_sqlite(
    context: AssetExecutionContext,
    config: VO2MaxSourceConfig,
) -> Output[Path]:
    """Download VO2Max SQLite database from GitHub."""
    logger = context.log
    output_path = MODULES_DIR / "just_vo2max" / "vo2max.sqlite"
    
    if output_path.exists() and not config.force_download:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Using cached SQLite: {output_path} ({size_mb:.2f} MB)")
        return Output(
            output_path,
            metadata={
                "source": MetadataValue.text("cached"),
                "path": MetadataValue.path(str(output_path)),
                "size_mb": MetadataValue.float(size_mb),
            },
        )
    
    url = config.download_url
    logger.info(f"Downloading VO2Max from {url}")
    
    with start_action(action_type="download_vo2max_sqlite", url=url):
        _download_file_with_progress(url, output_path, logger)
        size_mb = output_path.stat().st_size / (1024 * 1024)
    
    return Output(
        output_path,
        metadata={
            "source": MetadataValue.text("github"),
            "url": MetadataValue.url(url),
            "path": MetadataValue.path(str(output_path)),
            "size_mb": MetadataValue.float(size_mb),
        },
    )


def get_vo2max_output_dir(config: VO2MaxConfig) -> Path:
    """Resolve the output directory for VO2Max conversion."""
    if config.output_dir:
        return Path(config.output_dir)
    return MODULES_OUTPUT_DIR / config.module_name


@asset(
    description="VO2Max annotations converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, gene, phenotype, category", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def vo2max_annotations(
    context: AssetExecutionContext,
    vo2max_sqlite: Path,
    config: VO2MaxConfig,
) -> Output[Path]:
    """Convert VO2Max to annotations.parquet."""
    logger = context.log
    output_dir = get_vo2max_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.parquet"
    
    logger.info(f"Converting annotations from {vo2max_sqlite}")
    
    with start_action(action_type="convert_vo2max_annotations"):
        annotations = convert_vo2max_annotations(vo2max_sqlite)
        annotations.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} annotations to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="VO2Max studies converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, pmid, population, p_value, conclusion, study_design", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def vo2max_studies(
    context: AssetExecutionContext,
    vo2max_sqlite: Path,
    config: VO2MaxConfig,
) -> Output[Path]:
    """Convert VO2Max to studies.parquet."""
    logger = context.log
    output_dir = get_vo2max_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "studies.parquet"
    
    logger.info(f"Converting studies from {vo2max_sqlite}")
    
    with start_action(action_type="convert_vo2max_studies"):
        studies = convert_vo2max_studies(vo2max_sqlite)
        studies.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} studies to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="VO2Max weights converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, genotype, module, weight, state, priority, conclusion, curator, method", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def vo2max_weights(
    context: AssetExecutionContext,
    vo2max_sqlite: Path,
    config: VO2MaxConfig,
) -> Output[Path]:
    """Convert VO2Max to weights.parquet."""
    logger = context.log
    output_dir = get_vo2max_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weights.parquet"
    
    logger.info(f"Converting weights from {vo2max_sqlite}")
    
    with resource_tracker("vo2max_weights", context=context):
        with start_action(action_type="convert_vo2max_weights"):
            weights = convert_vo2max_weights(
                vo2max_sqlite,
                curator=config.curator,
                method=config.method,
            )
            weights.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} weights to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
            "curator": MetadataValue.text(config.curator),
        },
    )


@asset(
    description="VO2Max weights joined with Ensembl variation data.",
    compute_kind="join",
    io_manager_key="module_io_manager",
    metadata={"format": "parquet", "join_type": "inner"},
    op_tags={"dagster/concurrency_key": "ensembl_join"},
)
def vo2max_with_ensembl(
    context: AssetExecutionContext,
    ensembl_variations_source: str,
    vo2max_weights: Path,
    config: VO2MaxConfig,
) -> Output[Path]:
    """Join VO2Max weights with Ensembl variation data for chromosomal positions and ClinVar."""
    logger = context.log
    output_dir = get_vo2max_output_dir(config)
    output_path = output_dir / "vo2max_ensembl_joined.parquet"
    
    logger.info("Joining VO2Max weights with Ensembl variations")
    
    with start_action(action_type="join_vo2max_ensembl") as action:
        ensembl_files = resolve_ensembl_parquet_files_from_source(ensembl_variations_source)
        row_count = join_weights_with_ensembl_duckdb(
            weights_path=Path(vo2max_weights),
            ensembl_files=ensembl_files,
            output_path=output_path,
            duckdb_config=DuckDBConfig(),
        )
        action.log(message_type="info", step="joined_written", path=str(output_path), row_count=row_count)
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} joined rows to {output_path}")
    
    return Output(output_path, metadata={"dagster/column_schema": polars_schema_to_table_schema(output_path), "row_count": MetadataValue.int(row_count), "output_path": MetadataValue.path(str(output_path))})


@asset(
    description="Upload VO2Max module parquet files to HuggingFace Hub.",
    compute_kind="upload",
    io_manager_key="hf_upload_io_manager",
    metadata={"destination": "HuggingFace Hub", "repo": "just-dna-seq/annotators"},
)
def vo2max_hf_upload(
    context: AssetExecutionContext,
    vo2max_annotations: Path,
    vo2max_studies: Path,
    vo2max_with_ensembl: Path,
    config: AnnotatorsUploadConfig,
) -> Output[dict]:
    """Upload VO2Max parquet files to HuggingFace Hub."""
    from prepare_annotations.huggingface.uploader import upload_files_batch
    
    logger = context.log
    
    parquet_files = [vo2max_annotations, vo2max_studies, vo2max_with_ensembl]
    path_in_repos = [
        f"{config.path_prefix}/vo2max/annotations.parquet",
        f"{config.path_prefix}/vo2max/studies.parquet",
        f"{config.path_prefix}/vo2max/weights.parquet",
    ]
    
    logger.info(f"Uploading {len(parquet_files)} files to {config.repo_id}")
    
    with start_action(action_type="upload_vo2max_to_hf", repo_id=config.repo_id):
        # Generate metadata YAML
        metadata_yaml = _get_module_metadata_yaml("vo2max")
        
        # Get icon path
        icon_path = _get_module_icon_path("just_vo2max")

        result = upload_files_batch(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            path_in_repos=path_in_repos,
            repo_type="dataset",
            token=config.token,
            commit_message="Update vo2max module",
            metadata_yaml_content=metadata_yaml,
            metadata_yaml_path_in_repo=f"{config.path_prefix}/vo2max/metadata.yaml",
            icon_path=icon_path,
            icon_path_in_repo=f"{config.path_prefix}/vo2max/logo.jpg",
        )
    
    logger.info(f"Upload complete: {result.num_uploaded} uploaded, {result.num_skipped} skipped")
    
    return Output(
        {"repo_id": config.repo_id, "num_uploaded": result.num_uploaded, "num_skipped": result.num_skipped},
        metadata={
            "repo_id": MetadataValue.text(config.repo_id),
            "num_uploaded": MetadataValue.int(result.num_uploaded),
            "num_skipped": MetadataValue.int(result.num_skipped),
            "url": MetadataValue.url(f"https://huggingface.co/datasets/{config.repo_id}"),
        },
    )


# ============================================================================
# SUPERHUMAN ASSETS
# ============================================================================


@asset(
    description="Superhuman SQLite database downloaded from GitHub.",
    compute_kind="download",
    io_manager_key="module_io_manager",
    metadata={"format": "sqlite", "source": "github", "repo": "dna-seq/just_superhuman"},
)
def superhuman_sqlite(
    context: AssetExecutionContext,
    config: SuperhumanSourceConfig,
) -> Output[Path]:
    """Download Superhuman SQLite database from GitHub."""
    logger = context.log
    output_path = MODULES_DIR / "just_superhuman" / "superhuman.sqlite"
    
    if output_path.exists() and not config.force_download:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Using cached SQLite: {output_path} ({size_mb:.2f} MB)")
        return Output(
            output_path,
            metadata={
                "source": MetadataValue.text("cached"),
                "path": MetadataValue.path(str(output_path)),
                "size_mb": MetadataValue.float(size_mb),
            },
        )
    
    url = config.download_url
    logger.info(f"Downloading Superhuman from {url}")
    
    with start_action(action_type="download_superhuman_sqlite", url=url):
        _download_file_with_progress(url, output_path, logger)
        size_mb = output_path.stat().st_size / (1024 * 1024)
    
    return Output(
        output_path,
        metadata={
            "source": MetadataValue.text("github"),
            "url": MetadataValue.url(url),
            "path": MetadataValue.path(str(output_path)),
            "size_mb": MetadataValue.float(size_mb),
        },
    )


def get_superhuman_output_dir(config: SuperhumanConfig) -> Path:
    """Resolve the output directory for Superhuman conversion."""
    if config.output_dir:
        return Path(config.output_dir)
    return MODULES_OUTPUT_DIR / config.module_name


@asset(
    description="Superhuman annotations converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, gene, phenotype, category", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def superhuman_annotations(
    context: AssetExecutionContext,
    superhuman_sqlite: Path,
    config: SuperhumanConfig,
) -> Output[Path]:
    """Convert Superhuman to annotations.parquet."""
    logger = context.log
    output_dir = get_superhuman_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.parquet"
    
    logger.info(f"Converting annotations from {superhuman_sqlite}")
    
    with start_action(action_type="convert_superhuman_annotations"):
        annotations = convert_superhuman_annotations(superhuman_sqlite)
        annotations.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} annotations to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="Superhuman studies converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, pmid, population, p_value, conclusion, study_design", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def superhuman_studies(
    context: AssetExecutionContext,
    superhuman_sqlite: Path,
    config: SuperhumanConfig,
) -> Output[Path]:
    """Convert Superhuman to studies.parquet."""
    logger = context.log
    output_dir = get_superhuman_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "studies.parquet"
    
    logger.info(f"Converting studies from {superhuman_sqlite}")
    
    with start_action(action_type="convert_superhuman_studies"):
        studies = convert_superhuman_studies(superhuman_sqlite)
        studies.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} studies to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="Superhuman weights converted to unified schema (qualitative, no numeric weights).",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, genotype, module, weight, state, priority, conclusion, curator, method", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def superhuman_weights(
    context: AssetExecutionContext,
    superhuman_sqlite: Path,
    config: SuperhumanConfig,
) -> Output[Path]:
    """Convert Superhuman to weights.parquet (qualitative annotations, NULL weights)."""
    logger = context.log
    output_dir = get_superhuman_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weights.parquet"
    
    logger.info(f"Converting weights from {superhuman_sqlite}")
    
    with resource_tracker("superhuman_weights", context=context):
        with start_action(action_type="convert_superhuman_weights"):
            weights = convert_superhuman_weights(
                superhuman_sqlite,
                curator=config.curator,
                method=config.method,
            )
            weights.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} weights to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
            "curator": MetadataValue.text(config.curator),
        },
    )


@asset(
    description="Superhuman weights joined with Ensembl variation data.",
    compute_kind="join",
    io_manager_key="module_io_manager",
    metadata={"format": "parquet", "join_type": "inner"},
    op_tags={"dagster/concurrency_key": "ensembl_join"},
)
def superhuman_with_ensembl(
    context: AssetExecutionContext,
    ensembl_variations_source: str,
    superhuman_weights: Path,
    config: SuperhumanConfig,
) -> Output[Path]:
    """Join Superhuman weights with Ensembl variation data for chromosomal positions and ClinVar."""
    logger = context.log
    output_dir = get_superhuman_output_dir(config)
    output_path = output_dir / "superhuman_ensembl_joined.parquet"
    
    logger.info("Joining Superhuman weights with Ensembl variations")
    
    with start_action(action_type="join_superhuman_ensembl") as action:
        ensembl_files = resolve_ensembl_parquet_files_from_source(ensembl_variations_source)
        row_count = join_weights_with_ensembl_duckdb(
            weights_path=Path(superhuman_weights),
            ensembl_files=ensembl_files,
            output_path=output_path,
            duckdb_config=DuckDBConfig(),
        )
        action.log(message_type="info", step="joined_written", path=str(output_path), row_count=row_count)
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} joined rows to {output_path}")
    
    return Output(output_path, metadata={"dagster/column_schema": polars_schema_to_table_schema(output_path), "row_count": MetadataValue.int(row_count), "output_path": MetadataValue.path(str(output_path))})


@asset(
    description="Upload Superhuman module parquet files to HuggingFace Hub.",
    compute_kind="upload",
    io_manager_key="hf_upload_io_manager",
    metadata={"destination": "HuggingFace Hub", "repo": "just-dna-seq/annotators"},
)
def superhuman_hf_upload(
    context: AssetExecutionContext,
    superhuman_annotations: Path,
    superhuman_studies: Path,
    superhuman_with_ensembl: Path,
    config: AnnotatorsUploadConfig,
) -> Output[dict]:
    """Upload Superhuman parquet files to HuggingFace Hub."""
    from prepare_annotations.huggingface.uploader import upload_files_batch
    
    logger = context.log
    
    parquet_files = [superhuman_annotations, superhuman_studies, superhuman_with_ensembl]
    path_in_repos = [
        f"{config.path_prefix}/superhuman/annotations.parquet",
        f"{config.path_prefix}/superhuman/studies.parquet",
        f"{config.path_prefix}/superhuman/weights.parquet",
    ]
    
    logger.info(f"Uploading {len(parquet_files)} files to {config.repo_id}")
    
    with start_action(action_type="upload_superhuman_to_hf", repo_id=config.repo_id):
        # Generate metadata YAML
        metadata_yaml = _get_module_metadata_yaml("superhuman")
        
        # Get icon path
        icon_path = _get_module_icon_path("just_superhuman")

        result = upload_files_batch(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            path_in_repos=path_in_repos,
            repo_type="dataset",
            token=config.token,
            commit_message="Update superhuman module",
            metadata_yaml_content=metadata_yaml,
            metadata_yaml_path_in_repo=f"{config.path_prefix}/superhuman/metadata.yaml",
            icon_path=icon_path,
            icon_path_in_repo=f"{config.path_prefix}/superhuman/logo.jpg",
        )
    
    logger.info(f"Upload complete: {result.num_uploaded} uploaded, {result.num_skipped} skipped")
    
    return Output(
        {"repo_id": config.repo_id, "num_uploaded": result.num_uploaded, "num_skipped": result.num_skipped},
        metadata={
            "repo_id": MetadataValue.text(config.repo_id),
            "num_uploaded": MetadataValue.int(result.num_uploaded),
            "num_skipped": MetadataValue.int(result.num_skipped),
            "url": MetadataValue.url(f"https://huggingface.co/datasets/{config.repo_id}"),
        },
    )


# ============================================================================
# CORONARY ASSETS
# ============================================================================


@asset(
    description="Coronary SQLite database downloaded from GitHub.",
    compute_kind="download",
    io_manager_key="module_io_manager",
    metadata={"format": "sqlite", "source": "github", "repo": "dna-seq/just_coronary"},
)
def coronary_sqlite(
    context: AssetExecutionContext,
    config: CoronarySourceConfig,
) -> Output[Path]:
    """Download Coronary SQLite database from GitHub."""
    logger = context.log
    output_path = MODULES_DIR / "just_coronary" / "coronary.sqlite"
    
    if output_path.exists() and not config.force_download:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Using cached SQLite: {output_path} ({size_mb:.2f} MB)")
        return Output(
            output_path,
            metadata={
                "source": MetadataValue.text("cached"),
                "path": MetadataValue.path(str(output_path)),
                "size_mb": MetadataValue.float(size_mb),
            },
        )
    
    url = config.download_url
    logger.info(f"Downloading Coronary from {url}")
    
    with start_action(action_type="download_coronary_sqlite", url=url):
        _download_file_with_progress(url, output_path, logger)
        size_mb = output_path.stat().st_size / (1024 * 1024)
    
    return Output(
        output_path,
        metadata={
            "source": MetadataValue.text("github"),
            "url": MetadataValue.url(url),
            "path": MetadataValue.path(str(output_path)),
            "size_mb": MetadataValue.float(size_mb),
        },
    )


def get_coronary_output_dir(config: CoronaryConfig) -> Path:
    """Resolve the output directory for Coronary conversion."""
    if config.output_dir:
        return Path(config.output_dir)
    return MODULES_OUTPUT_DIR / config.module_name


@asset(
    description="Coronary annotations converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, gene, phenotype, category", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def coronary_annotations(
    context: AssetExecutionContext,
    coronary_sqlite: Path,
    config: CoronaryConfig,
) -> Output[Path]:
    """Convert Coronary to annotations.parquet."""
    logger = context.log
    output_dir = get_coronary_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.parquet"
    
    logger.info(f"Converting annotations from {coronary_sqlite}")
    
    with start_action(action_type="convert_coronary_annotations"):
        annotations = convert_coronary_annotations(coronary_sqlite)
        annotations.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} annotations to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="Coronary studies converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, module, pmid, population, p_value, conclusion, study_design", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def coronary_studies(
    context: AssetExecutionContext,
    coronary_sqlite: Path,
    config: CoronaryConfig,
) -> Output[Path]:
    """Convert Coronary to studies.parquet."""
    logger = context.log
    output_dir = get_coronary_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "studies.parquet"
    
    logger.info(f"Converting studies from {coronary_sqlite}")
    
    with start_action(action_type="convert_coronary_studies"):
        studies = convert_coronary_studies(coronary_sqlite)
        studies.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} studies to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
        },
    )


@asset(
    description="Coronary weights converted to unified schema.",
    compute_kind="conversion",
    io_manager_key="module_io_manager",
    metadata={"schema": "rsid, genotype, module, weight, state, priority, conclusion, curator, method", "format": "parquet"},
    op_tags={"dagster/concurrency_key": "module_conversion"},
)
def coronary_weights(
    context: AssetExecutionContext,
    coronary_sqlite: Path,
    config: CoronaryConfig,
) -> Output[Path]:
    """Convert Coronary to weights.parquet."""
    logger = context.log
    output_dir = get_coronary_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weights.parquet"
    
    logger.info(f"Converting weights from {coronary_sqlite}")
    
    with resource_tracker("coronary_weights", context=context):
        with start_action(action_type="convert_coronary_weights"):
            weights = convert_coronary_weights(
                coronary_sqlite,
                curator=config.curator,
                method=config.method,
            )
            weights.sink_parquet(output_path, engine="streaming")
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} weights to {output_path}")
    
    return Output(
        output_path,
        metadata={
            "dagster/column_schema": polars_schema_to_table_schema(output_path),
            "row_count": MetadataValue.int(row_count),
            "output_path": MetadataValue.path(str(output_path)),
            "module": MetadataValue.text(config.module_name),
            "curator": MetadataValue.text(config.curator),
        },
    )


@asset(
    description="Coronary weights joined with Ensembl variation data.",
    compute_kind="join",
    io_manager_key="module_io_manager",
    metadata={"format": "parquet", "join_type": "inner"},
    op_tags={"dagster/concurrency_key": "ensembl_join"},
)
def coronary_with_ensembl(
    context: AssetExecutionContext,
    ensembl_variations_source: str,
    coronary_weights: Path,
    config: CoronaryConfig,
) -> Output[Path]:
    """Join Coronary weights with Ensembl variation data for chromosomal positions and ClinVar."""
    logger = context.log
    output_dir = get_coronary_output_dir(config)
    output_path = output_dir / "coronary_ensembl_joined.parquet"
    
    logger.info("Joining Coronary weights with Ensembl variations")
    
    with start_action(action_type="join_coronary_ensembl") as action:
        ensembl_files = resolve_ensembl_parquet_files_from_source(ensembl_variations_source)
        row_count = join_weights_with_ensembl_duckdb(
            weights_path=Path(coronary_weights),
            ensembl_files=ensembl_files,
            output_path=output_path,
            duckdb_config=DuckDBConfig(),
        )
        action.log(message_type="info", step="joined_written", path=str(output_path), row_count=row_count)
    
    row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count} joined rows to {output_path}")
    
    return Output(output_path, metadata={"dagster/column_schema": polars_schema_to_table_schema(output_path), "row_count": MetadataValue.int(row_count), "output_path": MetadataValue.path(str(output_path))})


@asset(
    description="Upload Coronary module parquet files to HuggingFace Hub.",
    compute_kind="upload",
    io_manager_key="hf_upload_io_manager",
    metadata={"destination": "HuggingFace Hub", "repo": "just-dna-seq/annotators"},
)
def coronary_hf_upload(
    context: AssetExecutionContext,
    coronary_annotations: Path,
    coronary_studies: Path,
    coronary_with_ensembl: Path,
    config: AnnotatorsUploadConfig,
) -> Output[dict]:
    """Upload Coronary parquet files to HuggingFace Hub."""
    from prepare_annotations.huggingface.uploader import upload_files_batch
    
    logger = context.log
    
    parquet_files = [coronary_annotations, coronary_studies, coronary_with_ensembl]
    path_in_repos = [
        f"{config.path_prefix}/coronary/annotations.parquet",
        f"{config.path_prefix}/coronary/studies.parquet",
        f"{config.path_prefix}/coronary/weights.parquet",
    ]
    
    logger.info(f"Uploading {len(parquet_files)} files to {config.repo_id}")
    
    with start_action(action_type="upload_coronary_to_hf", repo_id=config.repo_id):
        # Generate metadata YAML
        metadata_yaml = _get_module_metadata_yaml("coronary")
        
        # Get icon path
        icon_path = _get_module_icon_path("just_coronary")

        result = upload_files_batch(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            path_in_repos=path_in_repos,
            repo_type="dataset",
            token=config.token,
            commit_message="Update coronary module",
            metadata_yaml_content=metadata_yaml,
            metadata_yaml_path_in_repo=f"{config.path_prefix}/coronary/metadata.yaml",
            icon_path=icon_path,
            icon_path_in_repo=f"{config.path_prefix}/coronary/logo.jpg",
        )
    
    logger.info(f"Upload complete: {result.num_uploaded} uploaded, {result.num_skipped} skipped")
    
    return Output(
        {"repo_id": config.repo_id, "num_uploaded": result.num_uploaded, "num_skipped": result.num_skipped},
        metadata={
            "repo_id": MetadataValue.text(config.repo_id),
            "num_uploaded": MetadataValue.int(result.num_uploaded),
            "num_skipped": MetadataValue.int(result.num_skipped),
            "url": MetadataValue.url(f"https://huggingface.co/datasets/{config.repo_id}"),
        },
    )


# ============================================================================
# EXPORT ALL ASSETS
# ============================================================================

module_assets = [
    ensembl_variations_source,
    # LongevityMap
    longevitymap_sqlite,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
    # LipidMetabolism
    lipidmetabolism_sqlite,
    lipidmetabolism_annotations,
    lipidmetabolism_studies,
    lipidmetabolism_weights,
    lipidmetabolism_with_ensembl,
    lipidmetabolism_hf_upload,
    # VO2Max
    vo2max_sqlite,
    vo2max_annotations,
    vo2max_studies,
    vo2max_weights,
    vo2max_with_ensembl,
    vo2max_hf_upload,
    # Superhuman
    superhuman_sqlite,
    superhuman_annotations,
    superhuman_studies,
    superhuman_weights,
    superhuman_with_ensembl,
    superhuman_hf_upload,
    # Coronary
    coronary_sqlite,
    coronary_annotations,
    coronary_studies,
    coronary_weights,
    coronary_with_ensembl,
    coronary_hf_upload,
]

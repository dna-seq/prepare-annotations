"""
Dagster assets for Ensembl VCF preparation pipelines.

Assets represent persistent data products with full lineage tracking:
- ensembl_ftp_source: External asset for Ensembl FTP server
- ensembl_vcf_urls: Local manifest file (JSON) with available VCF URLs
- ensembl_vcf_file: Per-file VCF download (dynamically partitioned)
- ensembl_parquet_file: Per-file parquet conversion (dynamically partitioned)
- ensembl_all_parquet_files: Collector asset for all parquet files
- ensembl_hf_upload: Upload to HuggingFace Hub

Features:
- Dynamic partitioning based on FTP file discovery
- Per-file lineage tracking in Dagster UI
- Dagster retry policies for fault tolerance
- Checksum verification with BSD sum
- Resumable downloads via fsspec filecache
"""

import json
from pathlib import Path
from typing import Dict

from dagster import (
    asset,
    AssetExecutionContext,
    Output,
    MetadataValue,
    AssetSpec,
    RetryPolicy,
    Backoff,
    DynamicPartitionsDefinition,
)
from eliot import start_action

from prepare_annotations.pipelines.configs import (
    EnsemblDownloadConfig,
    ParquetConversionConfig,
    HuggingFaceUploadConfig,
)
from prepare_annotations.core.paths import (
    get_default_ensembl_cache_dir,
    get_ensembl_species_url,
    get_ensembl_vcf_pattern,
)
from prepare_annotations.core.io import _default_parquet_path
from prepare_annotations.core.runtime import resource_tracker


# Retry policy for download operations - exponential backoff
download_retry_policy = RetryPolicy(
    max_retries=3,
    delay=30,  # 30 seconds initial delay
    backoff=Backoff.EXPONENTIAL,
)

# Dynamic partitions for per-file VCF downloads - registered at runtime by ensembl_vcf_urls
ENSEMBL_VCF_PARTITIONS = DynamicPartitionsDefinition(name="ensembl_vcf_file")


def _is_uploadable_parquet(path: Path) -> bool:
    if not path.name.endswith(".parquet"):
        return False
    lowered = path.name.lower()
    if lowered.startswith("."):
        return False
    if lowered.startswith("tmp"):
        return False
    if lowered.endswith(".tmp.parquet") or lowered.endswith(".parquet.tmp"):
        return False
    return True


# ============================================================================
# EXTERNAL SOURCE ASSET - ENSEMBL FTP SERVER
# ============================================================================

ensembl_ftp_source = AssetSpec(
    key="ensembl_ftp_source",
    description="Remote Ensembl variation VCF files on Ensembl FTP server. "
                "This is the external source of truth for Ensembl variation data.",
    metadata={
        "source": "Ensembl FTP",
        "base_url": "https://ftp.ensembl.org/pub/current_variation/vcf/",
        "type": "external_dataset",
        "url": "https://ftp.ensembl.org/pub/current_variation/vcf/",
    },
)


# ============================================================================
# ASSET: VCF URLS (Discovery) - Registers dynamic partitions
# ============================================================================

@asset(
    description="Discovered VCF file URLs from Ensembl FTP server. Registers dynamic partitions for per-file downloads.",
    compute_kind="discovery",
    deps=[ensembl_ftp_source],
    io_manager_key="io_manager",
    metadata={
        "source": "Ensembl FTP",
        "storage": "cache",
    },
)
def ensembl_vcf_urls(
    context: AssetExecutionContext,
    config: EnsemblDownloadConfig,
) -> Output[Path]:
    """
    Discover available VCF file URLs from Ensembl FTP.
    
    This asset lists the remote VCF files and saves the URLs for downstream
    download assets. Registers dynamic partitions for per-file processing.
    """
    from prepare_annotations.downloaders.vcf import list_paths
    
    logger = context.log
    
    # Construct URL for species
    species_url = get_ensembl_species_url(config.species, config.base_url)
    pattern = config.pattern or get_ensembl_vcf_pattern(config.species)
    
    logger.info(f"Discovering VCF files from: {species_url}")
    logger.info(f"Pattern: {pattern}")
    
    with start_action(action_type="dagster_ensembl_vcf_urls", url=species_url, pattern=pattern) as action:
        urls = list_paths(url=species_url, pattern=pattern)
        action.log(message_type="info", step="urls_discovered", count=len(urls))
    
    # Save URLs to cache for reference
    cache_dir = get_default_ensembl_cache_dir(config.species)
    cache_dir.mkdir(parents=True, exist_ok=True)
    urls_file = cache_dir / "vcf_urls.json"
    urls_file.write_text(json.dumps(urls, indent=2))

    # Register dynamic partitions for each file
    filenames = [url.rsplit("/", 1)[-1] for url in urls]
    existing_partitions = set(
        context.instance.get_dynamic_partitions(ENSEMBL_VCF_PARTITIONS.name)
    )
    new_partitions = sorted(set(filenames) - existing_partitions)
    if new_partitions:
        context.instance.add_dynamic_partitions(
            ENSEMBL_VCF_PARTITIONS.name,
            new_partitions,
        )
        logger.info(f"Registered {len(new_partitions)} new dynamic partitions")
    
    logger.info(f"Found {len(urls)} VCF files, {len(filenames)} partitions total")
    
    return Output(
        urls_file,
        metadata={
            "num_urls": MetadataValue.int(len(urls)),
            "species": MetadataValue.text(config.species),
            "pattern": MetadataValue.text(pattern),
            "urls_file": MetadataValue.path(str(urls_file)),
            "sample_urls": MetadataValue.json(urls[:5] if urls else []),
            "num_partitions": MetadataValue.int(len(filenames)),
            "new_partitions": MetadataValue.int(len(new_partitions)),
        }
    )


# ============================================================================
# ASSET: VCF FILE (Per-file Download, Dynamically Partitioned)
# ============================================================================

@asset(
    description="Download a single Ensembl VCF file (dynamically partitioned by filename).",
    compute_kind="download",
    partitions_def=ENSEMBL_VCF_PARTITIONS,
    io_manager_key="io_manager",
    metadata={
        "source": "Ensembl FTP",
        "format": "vcf.gz",
        "storage": "cache",
    },
    retry_policy=download_retry_policy,
    # Concurrency key limits how many downloads can run in parallel
    # Configure limit via DAGSTER_CONCURRENCY_KEYS env or dagster.yaml
    op_tags={"dagster/concurrency_key": "ensembl_vcf_download"},
)
def ensembl_vcf_file(
    context: AssetExecutionContext,
    ensembl_vcf_urls: Path,
    config: EnsemblDownloadConfig,
) -> Output[Path]:
    """
    Download a single VCF file from Ensembl FTP.
    
    Each partition corresponds to one VCF file discovered by ensembl_vcf_urls.
    Uses checksum verification and resumable downloads.
    """
    from prepare_annotations.downloaders.vcf import (
        download_path,
        download_checksums,
        ChecksumInfo,
    )
    
    logger = context.log
    partition_filename = context.partition_key

    urls_file = Path(ensembl_vcf_urls)
    if not urls_file.exists():
        raise FileNotFoundError(f"VCF URL manifest not found: {urls_file}")

    urls: list[str] = json.loads(urls_file.read_text())
    url_lookup = {url.rsplit("/", 1)[-1]: url for url in urls}
    
    if partition_filename not in url_lookup:
        raise KeyError(f"Partition filename not found in manifest: {partition_filename}")

    vcf_dir = urls_file.parent / "vcf"
    vcf_dir.mkdir(parents=True, exist_ok=True)
    url = url_lookup[partition_filename]

    # Try to get checksums for verification
    checksums: Dict[str, ChecksumInfo] = {}
    if config.verify_checksums:
        species_url = get_ensembl_species_url(config.species, config.base_url)
        try:
            checksums = download_checksums(species_url)
        except FileNotFoundError:
            logger.warning("No CHECKSUMS file found - skipping checksum verification")
        except Exception as e:
            logger.warning(f"Failed to download checksums: {e}")

    logger.info(f"Downloading {partition_filename} from {url}")

    with start_action(action_type="dagster_download_vcf", url=url, filename=partition_filename) as action:
        local_path = download_path(
            url=url,
            name="ensembl",
            dest_dir=vcf_dir,
            check_files=True,
            http_max_pool=config.http_max_pool,
            connect_timeout=config.connect_timeout,
            sock_read_timeout=config.sock_read_timeout,
            retries=config.retries,
            expected_checksum=checksums.get(partition_filename),
        )
        file_size = local_path.stat().st_size
        action.log(
            message_type="info",
            step="downloaded",
            path=str(local_path),
            size_mb=round(file_size / (1024 * 1024), 2),
        )

    logger.info(f"Downloaded {partition_filename}: {file_size / (1024*1024):.1f} MB")

    return Output(
        local_path,
        metadata={
            "filename": MetadataValue.text(partition_filename),
            "vcf_path": MetadataValue.path(str(local_path)),
            "file_size_mb": MetadataValue.float(round(file_size / (1024 * 1024), 2)),
            "checksums_verified": MetadataValue.bool(config.verify_checksums and partition_filename in checksums),
        },
    )


# ============================================================================
# ASSET: PARQUET FILE (Per-file Conversion, Dynamically Partitioned)
# ============================================================================

@asset(
    description="Convert a single Ensembl VCF file to Parquet format (dynamically partitioned).",
    compute_kind="conversion",
    partitions_def=ENSEMBL_VCF_PARTITIONS,
    io_manager_key="io_manager",
    metadata={
        "format": "parquet",
        "compression": "zstd",
        "storage": "cache",
    },
    # Concurrency key limits how many of these can run in parallel
    # Configure limit via DAGSTER_CONCURRENCY_KEYS env or dagster.yaml
    op_tags={"dagster/concurrency_key": "ensembl_parquet_conversion"},
)
def ensembl_parquet_file(
    context: AssetExecutionContext,
    ensembl_vcf_file: Path,
    config: ParquetConversionConfig,
) -> Output[Path]:
    """
    Convert a single VCF file to Parquet format using polars-bio.
    
    Each partition corresponds to one VCF file. Uses streaming conversion
    via sink_parquet for memory efficiency.
    """
    from prepare_annotations.core.io import vcf_to_parquet
    
    logger = context.log
    partition_key = context.partition_key

    vcf_path = Path(ensembl_vcf_file)
    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    # Output parquet to species_dir (parent of vcf/)
    species_dir = vcf_path.parent.parent
    parquet_path = species_dir / _default_parquet_path(vcf_path).name

    logger.info(f"Converting {vcf_path.name} to {parquet_path.name}")

    with start_action(action_type="dagster_convert_parquet", vcf=str(vcf_path), partition=partition_key) as action:
        # Check if already converted and up-to-date
        if parquet_path.exists() and not config.force_convert:
            vcf_mtime = vcf_path.stat().st_mtime
            pq_mtime = parquet_path.stat().st_mtime
            if pq_mtime >= vcf_mtime:
                logger.info(f"Skipping {vcf_path.name} - parquet exists and is up-to-date")
                file_size = parquet_path.stat().st_size
                action.log(message_type="info", step="skipped", reason="up_to_date")
                return Output(
                    parquet_path,
                    metadata={
                        "filename": MetadataValue.text(partition_key),
                        "parquet_path": MetadataValue.path(str(parquet_path)),
                        "file_size_mb": MetadataValue.float(round(file_size / (1024 * 1024), 2)),
                        "skipped": MetadataValue.bool(True),
                    },
                )

        # Convert to parquet
        lazy_frame, result_path = vcf_to_parquet(
            vcf_path=vcf_path,
            parquet_path=parquet_path,
            overwrite=config.force_convert,
            compression=config.compression,
            compression_level=config.compression_level,
            alts_list=config.alts_list,
            thread_num=config.get_threads(),
        )
        _ = lazy_frame  # LazyFrame is not used further
        file_size = result_path.stat().st_size
        action.log(message_type="info", step="converted", path=str(result_path), size_mb=round(file_size / (1024*1024), 2))

    logger.info(f"Converted {vcf_path.name} -> {result_path.name}: {file_size / (1024*1024):.1f} MB")

    return Output(
        result_path,
        metadata={
            "filename": MetadataValue.text(partition_key),
            "parquet_path": MetadataValue.path(str(result_path)),
            "file_size_mb": MetadataValue.float(round(file_size / (1024 * 1024), 2)),
            "skipped": MetadataValue.bool(False),
            "compression": MetadataValue.text(config.compression),
        },
    )


# ============================================================================
# ASSET: ALL PARQUET FILES (Collector for HF Upload)
# ============================================================================

@asset(
    description="Collect all converted Ensembl parquet files for upload.",
    compute_kind="collector",
    io_manager_key="io_manager",
    deps=[ensembl_parquet_file],
    metadata={
        "format": "parquet",
        "storage": "cache",
    },
)
def ensembl_all_parquet_files(
    context: AssetExecutionContext,
    ensembl_vcf_urls: Path,
) -> Output[Path]:
    """
    Collect all parquet files that have been converted.
    
    This is a non-partitioned asset that scans the species directory
    for all parquet files. Used as dependency for HuggingFace upload.
    """
    logger = context.log
    
    urls_file = Path(ensembl_vcf_urls)
    species_dir = urls_file.parent
    
    # Find all parquet files
    parquet_files = sorted(
        p for p in species_dir.glob("*.parquet") if _is_uploadable_parquet(p)
    )
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {species_dir}. Run ensembl_parquet_file first.")
    
    total_size = sum(f.stat().st_size for f in parquet_files)
    total_size_gb = total_size / (1024**3)
    
    logger.info(f"Found {len(parquet_files)} parquet files ({total_size_gb:.2f} GB)")
    
    return Output(
        species_dir,
        metadata={
            "num_files": MetadataValue.int(len(parquet_files)),
            "total_size_gb": MetadataValue.float(round(total_size_gb, 2)),
            "species_dir": MetadataValue.path(str(species_dir)),
            "files": MetadataValue.json([f.name for f in parquet_files[:10]]),  # First 10
        },
    )


# ============================================================================
# ASSET: HUGGINGFACE UPLOAD
# ============================================================================

@asset(
    description="Upload Ensembl parquet files to HuggingFace Hub.",
    compute_kind="upload",
    io_manager_key="hf_upload_io_manager",
    metadata={
        "destination": "HuggingFace Hub",
        "storage": "remote",
    },
)
def ensembl_hf_upload(
    context: AssetExecutionContext,
    ensembl_all_parquet_files: Path,
    config: HuggingFaceUploadConfig,
) -> Output[dict]:
    """
    Upload Ensembl parquet files to HuggingFace Hub.
    
    Uses batch upload for efficiency (single commit for all files).
    Only uploads files that differ in size from remote versions.
    """
    from prepare_annotations.huggingface.uploader import upload_parquet_to_hf
    from prepare_annotations.huggingface.dataset_cards import generate_ensembl_card
    
    logger = context.log

    species_dir = Path(ensembl_all_parquet_files)
    if not species_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {species_dir}")
    
    logger.info(f"Uploading from {species_dir} to {config.repo_id}")
    
    # Find parquet files
    parquet_files = sorted(
        p for p in species_dir.glob(config.pattern) if p.is_file() and _is_uploadable_parquet(p)
    )
    
    if not parquet_files:
        all_parquets = list(species_dir.rglob("*.parquet"))
        if all_parquets:
            logger.warning(
                f"No parquet files found with pattern '{config.pattern}' in {species_dir}, "
                f"but found {len(all_parquets)} parquet files recursively."
            )
        raise ValueError(f"No parquet files found in {species_dir} with pattern '{config.pattern}'")
    
    logger.info(f"Found {len(parquet_files)} parquet files to upload")
    
    # Calculate stats for dataset card
    total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
    
    # Generate dataset card
    dataset_card = generate_ensembl_card(
        num_files=len(parquet_files),
        total_size_gb=total_size_gb,
        variant_types=None,
    )
    
    with start_action(
        action_type="dagster_hf_upload",
        repo_id=config.repo_id,
        num_files=len(parquet_files),
    ) as action:
        result = upload_parquet_to_hf(
            parquet_files=parquet_files,
            repo_id=config.repo_id,
            token=config.token,
            path_prefix=config.path_prefix,
            source_dir=species_dir,
            dataset_card_content=dataset_card,
        )
        action.log(
            message_type="info",
            step="upload_complete",
            uploaded=result.num_uploaded,
            skipped=result.num_skipped,
        )
    
    logger.info(f"Upload complete: {result.num_uploaded} uploaded, {result.num_skipped} skipped")
    
    return Output(
        {
            "repo_id": config.repo_id,
            "num_uploaded": result.num_uploaded,
            "num_skipped": result.num_skipped,
            "total_files": len(parquet_files),
        },
        metadata={
            "repo_id": MetadataValue.text(config.repo_id),
            "num_uploaded": MetadataValue.int(result.num_uploaded),
            "num_skipped": MetadataValue.int(result.num_skipped),
            "total_files": MetadataValue.int(len(parquet_files)),
            "total_size_gb": MetadataValue.float(round(total_size_gb, 2)),
        }
    )

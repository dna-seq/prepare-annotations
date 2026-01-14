"""
Dagster assets for Ensembl VCF preparation pipelines.

Assets represent persistent data products with full lineage tracking:
- ensembl_ftp_source: External asset for Ensembl FTP server
- ensembl_vcf_urls: Local manifest file (JSON) with available VCF URLs
- ensembl_vcf_files: Local directory containing downloaded VCF files
- ensembl_parquet_files: Local directory containing converted Parquet files
- ensembl_hf_upload: Upload to HuggingFace Hub

Features:
- Parallel downloads with configurable concurrency (max_concurrent_downloads)
- Dagster retry policies for fault tolerance
- Checksum verification with BSD sum
- Resumable downloads via fsspec filecache
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from dagster import (
    asset,
    AssetExecutionContext,
    Output,
    MetadataValue,
    AssetSpec,
    AssetIn,
    RetryPolicy,
    Backoff,
)
from eliot import start_action

from prepare_annotations.pipelines_dagster.configs import (
    EnsemblDownloadConfig,
    ParquetConversionConfig,
    HuggingFaceUploadConfig,
)
from prepare_annotations.pipelines_dagster.resources import (
    get_default_ensembl_cache_dir,
    get_ensembl_vcf_dir,
    get_ensembl_parquet_dir,
    get_ensembl_species_url,
    get_ensembl_vcf_pattern,
)
from prepare_annotations.io import _default_parquet_path
from prepare_annotations.runtime import resource_tracker


# Retry policy for download operations - exponential backoff
download_retry_policy = RetryPolicy(
    max_retries=3,
    delay=30,  # 30 seconds initial delay
    backoff=Backoff.EXPONENTIAL,
)


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
# ASSET: VCF URLS (Discovery)
# ============================================================================

@asset(
    description="Discovered VCF file URLs from Ensembl FTP server.",
    compute_kind="discovery",
    deps=[ensembl_ftp_source],
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
    download assets. Uses the existing vcf_downloader.list_paths function.
    """
    from prepare_annotations.vcf_downloader import list_paths
    
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
    
    logger.info(f"Found {len(urls)} VCF files")
    
    return Output(
        urls_file,
        metadata={
            "num_urls": MetadataValue.int(len(urls)),
            "species": MetadataValue.text(config.species),
            "pattern": MetadataValue.text(pattern),
            "urls_file": MetadataValue.path(str(urls_file)),
            "sample_urls": MetadataValue.json(urls[:5] if urls else []),
        }
    )


# ============================================================================
# ASSET: VCF FILES (Downloads) - Parallel with retry policy
# ============================================================================

@asset(
    description="Downloaded Ensembl VCF files from FTP server with parallel downloads.",
    compute_kind="download",
    metadata={
        "source": "Ensembl FTP",
        "format": "vcf.gz",
        "storage": "cache",
    },
    retry_policy=download_retry_policy,
)
def ensembl_vcf_files(
    context: AssetExecutionContext,
    ensembl_vcf_urls: Path,
    config: EnsemblDownloadConfig,
) -> Output[Path]:
    """
    Download VCF files from Ensembl FTP in parallel.
    
    Uses the existing vcf_downloader.download_path function with:
    - Parallel downloads via ThreadPoolExecutor (configurable concurrency)
    - Checksum verification with BSD sum
    - Resumable downloads via fsspec filecache
    - Dagster retry policy for fault tolerance
    """
    from prepare_annotations.vcf_downloader import (
        download_path,
        download_checksums,
        ChecksumInfo,
    )
    
    logger = context.log

    urls_file = Path(ensembl_vcf_urls)
    if not urls_file.exists():
        raise FileNotFoundError(f"VCF URL manifest not found: {urls_file}")

    urls: list[str] = json.loads(urls_file.read_text())

    # Derive species cache dir from the manifest location (keeps downstream independent of config/species)
    vcf_dir = urls_file.parent / "vcf"
    vcf_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(urls)} VCF files to {vcf_dir} (max {config.max_concurrent_downloads} parallel)")
    
    # Try to download checksums for verification
    checksums: Dict[str, ChecksumInfo] = {}
    if config.verify_checksums:
        species_url = get_ensembl_species_url(config.species, config.base_url)
        try:
            checksums = download_checksums(species_url)
            logger.info(f"Loaded checksums for {len(checksums)} files")
        except FileNotFoundError:
            logger.warning("No CHECKSUMS file found - skipping checksum verification")
        except Exception as e:
            logger.warning(f"Failed to download checksums: {e}")

    def download_single_file(url: str) -> tuple[str, Path, int]:
        """Download a single VCF file with checksum verification."""
        filename = url.rsplit("/", 1)[-1]
        expected_checksum = checksums.get(filename) if config.verify_checksums else None
        
        with start_action(action_type="dagster_download_vcf", url=url) as action:
            local_path = download_path(
                url=url,
                name="ensembl",
                dest_dir=vcf_dir,
                check_files=True,
                http_max_pool=config.http_max_pool,
                connect_timeout=config.connect_timeout,
                sock_read_timeout=config.sock_read_timeout,
                retries=config.retries,
                expected_checksum=expected_checksum,
            )
            file_size = local_path.stat().st_size
            action.log(message_type="info", step="downloaded", path=str(local_path), size_mb=round(file_size / (1024*1024), 2))
        
        return filename, local_path, file_size

    # Download files in parallel using ThreadPoolExecutor with resource tracking
    total_size = 0
    downloaded_files: list[Path] = []
    failed_downloads: list[tuple[str, str]] = []
    
    with resource_tracker("Parallel VCF Downloads") as tracker:
        with ThreadPoolExecutor(max_workers=config.max_concurrent_downloads) as executor:
            futures = {executor.submit(download_single_file, url): url for url in urls}
            
            for future in as_completed(futures):
                url = futures[future]
                try:
                    filename, local_path, file_size = future.result()
                    total_size += file_size
                    downloaded_files.append(local_path)
                    logger.info(f"Downloaded: {filename} ({file_size / (1024*1024):.1f} MB)")
                except Exception as e:
                    failed_downloads.append((url, str(e)))
                    logger.error(f"Failed to download {url}: {e}")
    
    # Raise error if any downloads failed
    if failed_downloads:
        error_msg = "\n".join([f"  - {url}: {err}" for url, err in failed_downloads])
        raise RuntimeError(f"Failed to download {len(failed_downloads)} files:\n{error_msg}")
    
    total_size_gb = total_size / (1024**3)
    
    # Build metadata including resource tracking if available
    metadata_dict = {
        "num_files": MetadataValue.int(len(urls)),
        "total_size_gb": MetadataValue.float(round(total_size_gb, 2)),
        "vcf_dir": MetadataValue.path(str(vcf_dir)),
        "species": MetadataValue.text(config.species),
        "checksums_verified": MetadataValue.bool(config.verify_checksums and len(checksums) > 0),
        "max_concurrent_downloads": MetadataValue.int(config.max_concurrent_downloads),
    }
    
    # Add resource metrics if available
    if "report" in tracker:
        report = tracker["report"]
        metadata_dict["download_duration_sec"] = MetadataValue.float(round(report.duration, 2))
        metadata_dict["download_peak_memory_mb"] = MetadataValue.float(round(report.peak_memory_mb, 2))
    
    return Output(vcf_dir, metadata=metadata_dict)


# ============================================================================
# ASSET: PARQUET FILES (Conversion)
# ============================================================================

@asset(
    description="Ensembl VCF files converted to Parquet format.",
    compute_kind="conversion",
    metadata={
        "format": "parquet",
        "compression": "zstd",
        "storage": "cache",
    },
)
def ensembl_parquet_files(
    context: AssetExecutionContext,
    ensembl_vcf_files: Path,
    config: ParquetConversionConfig,
) -> Output[Path]:
    """
    Convert VCF files to Parquet format using polars-bio.
    
    Uses streaming conversion via sink_parquet for memory efficiency.
    Skips files that already have a valid parquet conversion.
    """
    from prepare_annotations.io import vcf_to_parquet, is_parquet
    
    logger = context.log

    vcf_dir = Path(ensembl_vcf_files)
    if not vcf_dir.exists():
        raise FileNotFoundError(f"VCF directory not found: {vcf_dir}")

    # Filter to only VCF files (not index files)
    vcf_files = sorted(
        f
        for f in vcf_dir.glob("*.vcf.gz")
        if not f.name.endswith((".tbi", ".csi"))
    )
    
    logger.info(f"Converting {len(vcf_files)} VCF files to Parquet")
    
    total_size = 0
    converted_count = 0
    skipped_count = 0

    species_dir = vcf_dir.parent

    with resource_tracker("VCF to Parquet Conversion") as tracker:
        for vcf_path in vcf_files:
            # Determine parquet path using the same naming as Prefect pipeline
            # This ensures consistent naming: homo_sapiens-chr1.vcf.gz -> homo_sapiens-chr1.parquet
            parquet_path = species_dir / _default_parquet_path(vcf_path).name
            
            with start_action(action_type="dagster_convert_parquet", vcf=str(vcf_path)) as action:
                # Skip if already exists and not forcing
                if parquet_path.exists() and not config.force_convert:
                    vcf_mtime = vcf_path.stat().st_mtime
                    pq_mtime = parquet_path.stat().st_mtime
                    
                    if pq_mtime >= vcf_mtime:
                        logger.info(f"Skipping {vcf_path.name} - parquet exists and is up-to-date")
                        total_size += parquet_path.stat().st_size
                        action.log(message_type="info", step="skipped", reason="up_to_date")
                        skipped_count += 1
                        continue
                
                # Convert to parquet
                lazy_frame, result_path = vcf_to_parquet(
                    vcf_path=vcf_path,
                    parquet_path=parquet_path,
                    overwrite=config.force_convert,
                    compression=config.compression,
                    compression_level=config.compression_level,
                    alts_list=config.alts_list,
                )
                action.log(message_type="info", step="converted", path=str(result_path))
            
            total_size += result_path.stat().st_size
            converted_count += 1
            logger.info(f"Converted: {vcf_path.name} -> {result_path.name}")
    
    total_size_gb = total_size / (1024**3)
    
    # Build metadata including resource tracking if available
    metadata_dict = {
        "num_files": MetadataValue.int(len(vcf_files)),
        "converted_count": MetadataValue.int(converted_count),
        "skipped_count": MetadataValue.int(skipped_count),
        "total_size_gb": MetadataValue.float(round(total_size_gb, 2)),
        "compression": MetadataValue.text(config.compression),
        "compression_level": MetadataValue.int(config.compression_level),
        "species_dir": MetadataValue.path(str(species_dir)),
    }
    
    # Add resource metrics if available
    if "report" in tracker:
        report = tracker["report"]
        metadata_dict["conversion_duration_sec"] = MetadataValue.float(round(report.duration, 2))
        metadata_dict["conversion_peak_memory_mb"] = MetadataValue.float(round(report.peak_memory_mb, 2))
    
    return Output(species_dir, metadata=metadata_dict)


# ============================================================================
# ASSET: HUGGINGFACE UPLOAD
# ============================================================================

@asset(
    description="Upload Ensembl parquet files to HuggingFace Hub.",
    compute_kind="upload",
    metadata={
        "destination": "HuggingFace Hub",
        "storage": "remote",
    },
)
def ensembl_hf_upload(
    context: AssetExecutionContext,
    ensembl_parquet_files: Path,
    config: HuggingFaceUploadConfig,
) -> Output[dict]:
    """
    Upload Ensembl parquet files to HuggingFace Hub.
    
    Uses batch upload for efficiency (single commit for all files).
    Only uploads files that differ in size from remote versions.
    """
    from prepare_annotations.huggingface_uploader import (
        upload_parquet_to_hf,
    )
    from prepare_annotations.dataset_card_generator import generate_ensembl_card
    
    logger = context.log

    species_dir = Path(ensembl_parquet_files)
    if not species_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {species_dir}")
    
    logger.info(f"Uploading from {species_dir} to {config.repo_id}")
    
    # Upload ONLY the non-split parquet files (do not recurse into legacy folders like 'splitted_variants/').
    parquet_files = sorted(p for p in species_dir.glob(config.pattern) if p.is_file())
    
    if not parquet_files:
        # Check if files might be in a different location
        all_parquets = list(species_dir.rglob("*.parquet"))
        if all_parquets:
            logger.warning(
                f"No parquet files found with pattern '{config.pattern}' in {species_dir}, "
                f"but found {len(all_parquets)} parquet files recursively. "
                f"Consider using pattern='**/*.parquet' if files are in subdirectories."
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

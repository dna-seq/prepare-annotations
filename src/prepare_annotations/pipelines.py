"""Prefect-based preparation pipelines for genomic data sources."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import duckdb
import polars as pl
from eliot import start_action
from prefect import task, flow, get_run_logger
from platformdirs import user_cache_dir
from pycomfort.logging import to_nice_stdout, to_nice_file

from prepare_annotations.io import is_parquet, _default_parquet_path
from prepare_annotations.paths import (
    get_cache_dir,
    get_default_cache_dir,
    get_default_input_dir,
    get_default_interim_dir,
    get_default_output_dir,
)
from prepare_annotations.runtime import prefect_flow_run
from prepare_annotations.models import PreparationResult, SplitResult, BatchUploadResult, RSIDCoordinateResult
from prepare_annotations.vcf_downloader import (
    list_paths,
    download_path,
    convert_to_parquet,
    validate_downloads_and_parquet,
    download_checksums,
    ChecksumInfo,
)
from prepare_annotations.huggingface_uploader import (
    collect_parquet_files,
    upload_parquet_to_hf,
)
from prepare_annotations.dataset_card_generator import (
    generate_clinvar_card,
    generate_ensembl_card,
    generate_dbsnp_card,
    generate_dbsnp_t2t_card,
    generate_gnomad_card,
)

# Prefect tasks for preparation steps with retries for resilience
list_paths_task = task(list_paths, name="List Remote Paths", retries=3, retry_delay_seconds=10)
download_path_task = task(download_path, name="Download Path", retries=5, retry_delay_seconds=30)
convert_to_parquet_task = task(convert_to_parquet, name="Convert to Parquet", retries=2, retry_delay_seconds=10)
validate_task = task(validate_downloads_and_parquet, name="Validate Downloads", retries=3, retry_delay_seconds=10)

@task(name="Split Parquet Files")
def split_parquets_task(
    parquet_paths: List[Path],
    explode_snv_alt: bool = False,
    write_to: Optional[Path] = None,
) -> SplitResult:
    """Split parquet files by variant type (TSA)."""
    from prepare_annotations.vcf_parquet_splitter import split_variants_by_tsa
    
    results = {}
    for p in parquet_paths:
        split_dict = split_variants_by_tsa(
            parquet_path=p,
            explode_snv_alt=explode_snv_alt,
            write_to=write_to,
        )
        for k, v in split_dict.items():
            if k not in results:
                results[k] = []
            if isinstance(v, list):
                results[k].extend(v)
            else:
                results[k].append(v)
    return SplitResult(split_variants_dict=results)


@task(name="Compute rsID Coordinates")
def compute_rsid_coordinates_task(
    input_dir: Path,
    output_path: Path,
    memory_fraction: float = 0.8,
    output_dataset: bool = False,
    force: bool = False,
    compression_level: int = 14,
) -> RSIDCoordinateResult:
    """Compute rsID coordinates from split parquet files using DuckDB streaming.
    
    This function reads from split variant folders (SNV, deletion, insertion, etc.) and 
    preserves the tsa (variant type) column from the original data.
    
    Key optimizations:
    - Processes each parquet FILE individually (files are per-chromosome, no cross-file duplicates)
    - DISTINCT only within each file, then streams directly to output
    - Dynamic memory limit based on available system memory
    - No collect() or in-memory aggregation
    - Sorting by chromosome and start position for better compression (30-50% reduction)
    - ZSTD compression level default 14 for balanced speed and size reduction
    - Caching: per-chromosome coordinate chunks are stored in the cache dir to avoid re-computation
    
    Args:
        input_dir: Directory containing variant type subdirectories
        output_path: Path for output parquet file (default) or output directory (if output_dataset=True)
        memory_fraction: Fraction of available memory to use (0.0-1.0)
        output_dataset: If True, write a directory of per-chromosome parquet chunks and skip the final merge step
        force: If True, re-compute all chunks even if they already exist
        compression_level: ZSTD compression level for the final output
    """
    import psutil
    
    if output_dataset:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # For single-file output, write to a sibling temp file and atomically replace the final output
        # on success. This prevents leaving a truncated/corrupted parquet if the task is interrupted.
        tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
        if tmp_output_path.exists():
            tmp_output_path.unlink()
    
    # Find all variant type subdirectories (exclude hidden directories like .cache)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}. Have you run the splitting step?")
        
    variant_type_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not variant_type_dirs:
        raise ValueError(f"No variant type subdirectories found in {input_dir}. The input directory should contain subfolders for each variant type (e.g., SNV, deletion).")
    
    # Dynamic memory calculation
    mem = psutil.virtual_memory()
    # Be more conservative with memory (0.6 fraction) as DuckDB needs overhead
    memory_limit_bytes = int(mem.available * min(memory_fraction, 0.6))
    memory_limit_gb = memory_limit_bytes / (1024 ** 3)
    memory_limit_str = f"{max(2, int(memory_limit_gb))}GB"
    
    def chrom_sort_key(filename: str) -> tuple[int, str]:
        import re
        # Extract chromosome from filename like homo_sapiens-chr21.vcf.parquet
        match = re.search(r'chr([0-9]+|[XYMTm]+)', filename)
        if not match:
            return (1000, filename)
        chrom = match.group(1).upper()
        if chrom.isdigit():
            # Keep numeric ordering via the first tuple element, but preserve a useful label for logs/filenames.
            return (int(chrom), chrom)
        mapping = {"X": 100, "Y": 101, "MT": 102, "M": 102}
        return (mapping.get(chrom, 200), chrom)

    # Create a stable directory for per-chromosome chunks and DuckDB temp spill files.
    import tempfile
    from pathlib import Path
    
    # We use a sequential loop to ensure memory safety. 
    # By processing one chromosome at a time, we cap RAM usage to ~1-2GB.
    chunk_dir: Path
    duckdb_temp_dir: Path
    tmp_dir_ctx: Any = None
    duckdb_tmp_ctx: Any = None

    if output_dataset:
        # Persist chunks directly into the output directory to avoid an extra read+rewrite merge pass.
        chunk_dir = output_path
    else:
        # Use a stable directory in the cache for chunks to allow re-use on restart.
        # We name it 'rsid_coordinates' as it's the standard name for this dataset.
        chunk_dir = input_dir.parent / "rsid_coordinates"
        chunk_dir.mkdir(parents=True, exist_ok=True)
    
    duckdb_tmp_ctx = tempfile.TemporaryDirectory(dir="/tmp", prefix="ensembl_rsids_duckdb_")

    try:
        duckdb_temp_dir = Path(duckdb_tmp_ctx.__enter__())
        
        with start_action(action_type="duckdb_compute_rsids", input_dir=str(input_dir), 
                          variant_types=[d.name for d in variant_type_dirs],
                          memory_limit=memory_limit_str,
                          available_memory_gb=round(mem.available / (1024**3), 1)) as action:
            
            # Create DuckDB connection
            con = duckdb.connect()
            con.execute(f"SET memory_limit = '{memory_limit_str}'")
            con.execute(f"SET temp_directory = '{duckdb_temp_dir}'")
            # Avoid insertion-order preservation: reduces memory pressure for large scans/sorts.
            con.execute("SET preserve_insertion_order = false")
            
            # Group files by chromosome
            from collections import defaultdict
            chrom_groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
            for variant_dir in variant_type_dirs:
                for pq_file in variant_dir.glob("*.parquet"):
                    if not is_parquet(pq_file):
                        continue
                    key = chrom_sort_key(pq_file.name)
                    chrom_groups[key].append(pq_file)

            sorted_keys = sorted(chrom_groups.keys())
            chunk_files = []
            
            for key in sorted_keys:
                order, chrom_name = key
                files = chrom_groups[key]
                
                # Use a descriptive stem based on the first file in the group (e.g. homo_sapiens-chr1)
                first_name = files[0].name
                stem = first_name
                for suffix in [".vcf.parquet", ".parquet"]:
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        break
                
                # We name it after the chromosome/source to keep it clean. 
                # The merge order is preserved via the Python loop, so we don't need a numeric prefix.
                chunk_out = chunk_dir / f"{stem}_rsid_coordinates.parquet"
                
                # Check if this chunk already exists and is valid (skip if not forced)
                if not force and chunk_out.exists() and chunk_out.stat().st_size > 0:
                    try:
                        # Quick validation: can we scan it?
                        pl.scan_parquet(chunk_out).select(pl.len()).collect()
                        action.log(message_type="skipping_chromosome", chromosome=chrom_name, reason="already_exists")
                        chunk_files.append(chunk_out)
                        continue
                    except Exception:
                        action.log(message_type="recomputing_chromosome", chromosome=chrom_name, reason="corrupted_file")
                
                action.log(message_type="processing_chromosome", chromosome=chrom_name, files=len(files))
                
                # Get column names for the first file to check TSA case
                # (Some VCFs use TSA instead of tsa)
                first_file = files[0]
                cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{first_file}')").fetchall()
                col_names = [c[0] for c in cols_info]
                tsa_field = "TSA" if "TSA" in col_names and "tsa" not in col_names else "tsa"
                
                # Subqueries for this chromosome
                subqueries = [
                    f"SELECT chrom, start, \"end\", id, {tsa_field} as tsa FROM read_parquet('{f}')"
                    for f in files
                ]
                
                # Deduplicate and sort JUST this chromosome. 
                # Since data is per-chromosome, this is perfectly safe and memory-efficient.
                # We use a large ROW_GROUP_SIZE (1M) for better compression and scanning performance.
                con.execute(f"""
                    COPY (
                        SELECT DISTINCT chrom, start, "end", id, tsa 
                        FROM ( {' UNION ALL '.join(subqueries)} )
                        ORDER BY start
                     ) TO '{chunk_out}' (
                         FORMAT 'PARQUET', 
                         COMPRESSION 'ZSTD', 
                         COMPRESSION_LEVEL {compression_level}, 
                         ROW_GROUP_SIZE 5000000
                     )
                """)
                # Get the row count of the produced chunk
                chunk_count = con.execute(f"SELECT count(*) FROM read_parquet('{chunk_out}')").fetchone()[0]
                action.log(message_type="chromosome_completed", chromosome=chrom_name, count=chunk_count, size_mb=round(chunk_out.stat().st_size / (1024**2), 2))
                
                chunk_files.append(chunk_out)

            if not output_dataset:
                action.log(message_type="info", step="merging_chunks_pyarrow", total_chunks=len(chunk_files))
                
                from prepare_annotations.io import merge_parquet_files
                
                # Merge chunks using the new pyarrow-based utility
                merge_parquet_files(
                    input_files=chunk_files,
                    output_path=output_path,
                    compression="zstd",
                    compression_level=compression_level,
                )
            con.close()
            
            action.log(message_type="completed", output_path=str(output_path))
    finally:
        if duckdb_tmp_ctx is not None:
            duckdb_tmp_ctx.__exit__(None, None, None)
        if not output_dataset:
            # Best-effort cleanup if an exception occurred before the atomic replace.
            try:
                if 'tmp_output_path' in locals() and tmp_output_path.exists():
                    tmp_output_path.unlink()
            except OSError:
                pass
    
    # Get count for the result using Polars lazy scan (memory efficient)
    scan_target = str(output_path / "*.parquet") if output_dataset else str(output_path)
    count = pl.scan_parquet(scan_target).select(pl.len()).collect().item()
    
    return RSIDCoordinateResult(output_path=output_path, count=count)


@flow(name="Compute Ensembl rsID Coordinates")
def ensembl_rsid_coords_flow(
    input_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    memory_fraction: float = 0.8,
    auto_split: bool = True,
    explode_snv_alt: bool = False,
    output_dataset: bool = False,
    force: bool = False,
    compression_level: int = 14,
    profile: bool = True,
) -> RSIDCoordinateResult:
    """Flow to compute rsID coordinates from Ensembl split parquet files.
    
    Args:
        input_dir: Directory containing splitted_variants subdirectories
        output_path: Path for output parquet file
        memory_fraction: Fraction of available memory to use (0.0-1.0), default 0.8
        auto_split: If True and split files don't exist, automatically run split first
        explode_snv_alt: Whether to explode SNV ALT column during auto-splitting
        output_dataset: If True, write a directory of per-chromosome parquet chunks
        force: If True, re-compute all chunks even if they already exist
        compression_level: ZSTD compression level for the final output
        profile: Enable profiling
    """
    
    if input_dir is None:
        # Look for splitted_variants directly in the cache folder
        cache_dir = get_default_cache_dir("ensembl")
        input_dir = cache_dir / "splitted_variants"
    
    if output_path is None:
        cache_dir = get_default_cache_dir("ensembl")
        output_path = cache_dir / ("rsid_coordinates" if output_dataset else "rsid_coordinates.parquet")
    
    # Check if split files exist
    if not input_dir.exists() or not any(input_dir.iterdir()):
        if auto_split:
            get_run_logger().warning(
                f"Split files not found in {input_dir}. Running split operation first..."
            )
            # Find the parquet files to split, avoiding duplicates like foo.vcf.parquet vs foo.parquet
            cache_dir = input_dir.parent
            all_parquets = [p for p in cache_dir.glob("*.parquet") if is_parquet(p)]
            
            # Group by base name (e.g. 'homo_sapiens-chr2') to detect duplicates
            by_base: Dict[str, Path] = {}
            for p in all_parquets:
                name = p.name
                if name == "rsid_coordinates.parquet":
                    continue
                    
                # Determine base name by stripping extensions
                if name.endswith(".vcf.parquet"):
                    base = name[:-12]
                elif name.endswith(".parquet"):
                    base = name[:-8]
                else:
                    base = name
                
                # If duplicate exists, prefer the one without .vcf. in its name
                if base not in by_base or (".vcf." in by_base[base].name and ".vcf." not in name):
                    by_base[base] = p
            
            parquet_files = list(by_base.values())
            
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found in {cache_dir} to split. "
                    f"Please run the download/conversion step first:\n"
                    f"  uv run prepare-annotations ensembl"
                )
            
            # Run the split operation
            split_parquets_task(
                parquet_paths=parquet_files,
                explode_snv_alt=explode_snv_alt,
                write_to=input_dir
            )
        else:
            raise FileNotFoundError(
                f"Split files not found in {input_dir}. "
                f"Please run the split operation first:\n"
                f"  uv run prepare-annotations split {cache_dir}/*.parquet"
            )
        
    with prefect_flow_run("Ensembl rsID Coordinates", profile=profile):
        return compute_rsid_coordinates_task(
            input_dir=input_dir, 
            output_path=output_path,
            memory_fraction=memory_fraction,
            output_dataset=output_dataset,
            force=force,
            compression_level=compression_level,
        )


@flow(name="Prepare VCF Source")
def prepare_vcf_source_flow(
    url: str,
    pattern: Optional[str] = None,
    name: str = "downloads",
    dest_dir: Optional[str | Path] = None,
    vcf_dir: Optional[str | Path] = None,
    parquet_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    compression: str = "zstd",
    compression_level: Optional[int] = None,
    download_progress_interval_seconds: Optional[float] = None,
    s3_max_pool: Optional[int] = None,
    s3_block_size: Optional[int] = None,
    http_max_pool: Optional[int] = None,
    http_chunk_size: Optional[int] = None,
    connect_timeout: Optional[float] = None,
    sock_read_timeout: Optional[float] = None,
    retries: Optional[int] = None,
    verify_checksums: bool = True,
    profile: bool = True,
) -> PreparationResult:
    """Generic flow to download, convert, and optionally split VCF data.
    
    All files are stored in a flat structure under the cache folder (default) or 
    in specific subdirectories if provided:
    - VCF files: {vcf_dir} or {cache_dir}/{name}/
    - Parquet files: {parquet_dir} or {cache_dir}/{name}/
    - Split variants: {cache_dir}/{name}/splitted_variants/
    
    Args:
        url: Base URL to download from
        pattern: Regex pattern to filter files
        name: Name for cache directory
        dest_dir: Base destination directory
        vcf_dir: Specific directory for VCF files
        parquet_dir: Specific directory for Parquet files
        with_splitting: Whether to split by variant type
        explode_snv_alt: Whether to explode SNV ALT column
        compression: Compression algorithm for Parquet
        compression_level: Compression level
        download_progress_interval_seconds: Interval for progress logging
        s3_max_pool: S3 connection pool size (default: 50)
        s3_block_size: S3 block size for reads
        http_max_pool: HTTP/HTTPS connection pool size (default: 20)
        http_chunk_size: HTTP/HTTPS chunk size for reads
        connect_timeout: Connection timeout in seconds (default: 10s)
        sock_read_timeout: Socket read timeout in seconds (default: 120s)
        retries: Number of retry attempts (default: 10)
        verify_checksums: Whether to verify file checksums (default: True).
            If True, attempts to download CHECKSUMS file from the URL and
            verifies each downloaded file. Corrupted files are re-downloaded.
        profile: Whether to track resource usage
    """
    logger = get_run_logger()
    
    # All files go to the same cache directory by default
    cache_path = Path(dest_dir) if dest_dir else get_default_cache_dir(name)
    
    # Resolve specific directories: default to vcf/ subfolder, while parquets go directly to dest_dir
    vcf_path = Path(vcf_dir) if vcf_dir else cache_path / "vcf"
    parquet_path = Path(parquet_dir) if parquet_dir else cache_path
    
    vcf_path.mkdir(parents=True, exist_ok=True)
    parquet_path.mkdir(parents=True, exist_ok=True)
    
    # Clean up any leftover temporary files from previous failed runs
    for tmp_file in parquet_path.glob("*.tmp.parquet"):
        try:
            logger.info(f"Removing leftover temporary file: {tmp_file}")
            tmp_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove {tmp_file}: {e}")

    with prefect_flow_run(f"Prepare {name}", profile=profile):
        # 2. List paths
        urls = list_paths_task(url=url, pattern=pattern)
        
        # 2.5. Try to download checksums for verification
        checksums: dict[str, ChecksumInfo] = {}
        if verify_checksums:
            try:
                checksums = download_checksums(url)
                logger.info(f"Loaded checksums for {len(checksums)} files from {url}CHECKSUMS")
            except FileNotFoundError:
                logger.warning(f"No CHECKSUMS file found at {url} - skipping checksum verification")
            except Exception as e:
                logger.warning(f"Failed to download checksums from {url}: {e} - skipping checksum verification")
        
        # 3. Download files in parallel to vcf_path
        vcf_local_futures = []
        for u in urls:
            # Get expected checksum for this file if available
            filename = u.rsplit("/", 1)[-1]
            expected_checksum = checksums.get(filename) if verify_checksums else None
            
            vcf_local_futures.append(
                download_path_task.submit(
                    url=u,
                    name=name,
                    dest_dir=vcf_path,
                    progress_interval_seconds=download_progress_interval_seconds,
                    s3_max_pool=s3_max_pool,
                    s3_block_size=s3_block_size,
                    http_max_pool=http_max_pool,
                    http_chunk_size=http_chunk_size,
                    connect_timeout=connect_timeout,
                    sock_read_timeout=sock_read_timeout,
                    retries=retries,
                    expected_checksum=expected_checksum,
                )
            )
        vcf_locals = [f.result() for f in vcf_local_futures]
            
        # 4. Convert to parquet in parallel (to parquet_path)
        conversion_futures = []
        for vcf_p in vcf_locals:
            # Determine specific parquet path based on resolved parquet_path
            from prepare_annotations.io import _default_parquet_path
            p_path = parquet_path / _default_parquet_path(vcf_p).name
            
            conversion_futures.append(
                convert_to_parquet_task.submit(
                    vcf_path=vcf_p, 
                    parquet_path=p_path,
                    compression=compression,
                    compression_level=compression_level,
                    alts_list=alts_list,
                )
            )
        vcf_parquet_paths = [f.result()[1] for f in conversion_futures]
            
        # 5. Validate
        validate_task(urls=urls, vcf_local=vcf_locals, vcf_parquet_path=vcf_parquet_paths)
        
        split_dict = None
        
        # Keep only real parquet conversions (e.g. skip .tbi/.csi index files).
        parquet_only_paths = [p for p in vcf_parquet_paths if is_parquet(p)]

        # 6. Optional splitting
        if with_splitting:
            if not parquet_only_paths:
                logger.warning(
                    "Splitting requested, but no .parquet VCF conversions were produced "
                    "(likely downloaded only an index file). Skipping splitting."
                )
            else:
                split_dir = cache_path / "splitted_variants"
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Clean up any leftover temporary files in split directory
                for tmp_file in split_dir.rglob("*.tmp.parquet"):
                    try:
                        logger.info(f"Removing leftover temporary file in split dir: {tmp_file}")
                        tmp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove {tmp_file}: {e}")

                split_result = split_parquets_task(
                    parquet_paths=parquet_only_paths,
                    explode_snv_alt=explode_snv_alt,
                    write_to=split_dir
                )
                split_dict = split_result.split_variants_dict
            
        # Final cleanup: Remove .fsspec_cache if it exists and is empty
        fsspec_cache = cache_path / ".fsspec_cache"
        if fsspec_cache.exists():
            try:
                # Only remove if empty or contains only empty directories
                if not any(fsspec_cache.iterdir()):
                    fsspec_cache.rmdir()
                else:
                    # Optional: remove it entirely if we're sure it's just leftovers
                    import shutil
                    shutil.rmtree(fsspec_cache, ignore_errors=True)
            except Exception:
                pass

        return PreparationResult(
            urls=urls,
            vcf_local=vcf_locals,
            vcf_parquet_path=vcf_parquet_paths,
            split_variants_dict=split_dict
        )


@flow(name="Prepare dbSNP")
def prepare_dbsnp_flow(
    build: str = "GRCh38",
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for dbSNP preparation."""
    if build == "GRCh38":
        base_url = "https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/"
        pattern = r"GCF_000001405\.40\.gz$"
    elif build == "GRCh37":
        base_url = "https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/"
        pattern = r"GCF_000001405\.25\.gz$"
    else:
        raise ValueError(f"Unsupported build: {build}")
        
    return prepare_vcf_source_flow(
        url=base_url,
        pattern=pattern,
        name=f"dbsnp_{build.lower()}",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        profile=profile,
    )


@flow(name="Prepare dbSNP T2T")
def prepare_dbsnp_t2t_flow(
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    compression: str = "zstd",
    compression_level: int = 14,
    progress_interval_seconds: float = 60.0,
    s3_max_pool: Optional[int] = None,
    s3_block_size: Optional[int] = None,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for dbSNP T2T preparation."""
    # S3 path for T2T dbSNP from human-pangenomics
    s3_url = "s3://human-pangenomics/T2T/CHM13/assemblies/annotation/liftover/"
    # Pattern to match both VCF and its index
    pattern = r"chm13v2.0_dbSNPv155.vcf.gz(.tbi)?$"
    
    return prepare_vcf_source_flow(
        url=s3_url,
        pattern=pattern,
        name="dbsnp_t2t",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        compression=compression,
        compression_level=compression_level,
        download_progress_interval_seconds=progress_interval_seconds,
        s3_max_pool=s3_max_pool,
        s3_block_size=s3_block_size,
        profile=profile,
    )


@flow(name="Prepare gnomAD")
def prepare_gnomad_flow(
    version: str = "v4",
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for gnomAD preparation."""
    if version == "v4":
        base_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.0/vcf/"
        pattern = r"gnomad\.v4\.0\..+\.vcf\.bgz$"
    elif version == "v3":
        base_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1.2/vcf/"
        pattern = r"gnomad\.v3\.1\.2\..+\.vcf\.bgz$"
    else:
        raise ValueError(f"Unsupported version: {version}")
        
    return prepare_vcf_source_flow(
        url=base_url,
        pattern=pattern,
        name=f"gnomad_{version}",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        profile=profile,
    )

@flow(name="Prepare ClinVar")
def prepare_clinvar_flow(
    url: Optional[str] = None,
    pattern: Optional[str] = None,
    dest_dir: Optional[str | Path] = None,
    vcf_dir: Optional[str | Path] = None,
    parquet_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    profile: bool = True,
    assembly: str = "GRCh38_ensembl",
) -> PreparationResult:
    """Prefect flow for ClinVar preparation.
    
    ClinVar is human-only. Use assembly parameter to select source:
    - GRCh38_ensembl: Ensembl VEP-annotated ClinVar (includes consequence annotations)
    - GRCh38: NCBI ClinVar for GRCh38
    - GRCh37: NCBI ClinVar for GRCh37
    """
    effective_url = url
    effective_pattern = pattern
    effective_dest = dest_dir
    
    if "ensembl" in assembly.lower() and not effective_url:
        # Use Ensembl VEP-annotated ClinVar from vep/ folder
        ensembl_species_url = f"https://ftp.ensembl.org/pub/current_variation/vcf/homo_sapiens/"
        try:
            # Check for vep directory
            vep_url = f"{ensembl_species_url}vep/"
            vep_folders = list_paths(vep_url, file_only=False)
            # Filter for folders that contain 'clinvar' and look like Ensembl's dated structure
            # e.g. 115-clinvar20250907
            clinvar_folders = [f for f in vep_folders if "clinvar" in f.lower()]
            if clinvar_folders:
                # Pick the latest one by name (usually they contain dates)
                latest_folder = sorted(clinvar_folders)[-1]
                if not latest_folder.endswith("/"):
                    latest_folder += "/"
                effective_url = latest_folder
                effective_pattern = r"clinvar.*\.vcf\.gz$"
                
                # Use clinvar cache directory
                if not effective_dest:
                    effective_dest = get_default_cache_dir("clinvar")
                
                get_run_logger().info(f"Detected Ensembl ClinVar in {effective_url}")
        except Exception as e:
            get_run_logger().warning(f"Could not detect Ensembl ClinVar {e}")

    if not effective_url:
        # Default to NCBI ClinVar if no URL provided/detected
        effective_url = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_{assembly}/"
        effective_pattern = effective_pattern or r"clinvar\.vcf\.gz$"

    return prepare_vcf_source_flow(
        url=effective_url,
        pattern=effective_pattern,
        name="clinvar",
        dest_dir=effective_dest,
        vcf_dir=vcf_dir,
        parquet_dir=parquet_dir,
        with_splitting=with_splitting,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        profile=profile,
    )

@flow(name="Prepare Ensembl")
def prepare_ensembl_flow(
    species: str = "homo_sapiens",
    dest_dir: Optional[str | Path] = None,
    vcf_dir: Optional[str | Path] = None,
    parquet_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    pattern: Optional[str] = None,
    http_max_pool: Optional[int] = None,
    http_chunk_size: Optional[int] = None,
    connect_timeout: Optional[float] = None,
    sock_read_timeout: Optional[float] = None,
    retries: Optional[int] = None,
    verify_checksums: bool = True,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for Ensembl preparation.
    
    Args:
        species: Species name (e.g., homo_sapiens, mus_musculus)
        dest_dir: Base destination directory
        vcf_dir: Specific directory for VCF files
        parquet_dir: Specific directory for Parquet files
        with_splitting: Whether to split by variant type
        explode_snv_alt: Whether to explode SNV ALT column
        alts_list: Whether to add a list of alternative alleles as 'alts' column
        pattern: Regex pattern to filter files
        http_max_pool: HTTP connection pool size (default: 20)
        http_chunk_size: HTTP chunk size for reads
        connect_timeout: Connection timeout in seconds (default: 10s)
        sock_read_timeout: Socket read timeout in seconds (default: 120s)
        retries: Number of retry attempts (default: 10)
        verify_checksums: Whether to verify file checksums (default: True)
        profile: Whether to track resource usage
    """
    # Base directory for Ensembl variations
    base_dir = Path(dest_dir) if dest_dir else get_default_cache_dir("ensembl")
    
    # Species-specific subfolder
    species_dir = base_dir / species
    
    # Construct URL based on species
    url = f"https://ftp.ensembl.org/pub/current_variation/vcf/{species}/"
    
    # Default pattern also depends on species
    default_pattern = rf"{species}-chr([^.]+)\.vcf\.gz$"
    
    return prepare_vcf_source_flow(
        url=url,
        pattern=pattern or default_pattern,
        name="ensembl",
        dest_dir=species_dir,
        vcf_dir=vcf_dir,
        parquet_dir=parquet_dir,
        with_splitting=with_splitting,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        http_max_pool=http_max_pool,
        http_chunk_size=http_chunk_size,
        connect_timeout=connect_timeout,
        sock_read_timeout=sock_read_timeout,
        retries=retries,
        verify_checksums=verify_checksums,
        profile=profile,
    )


class PreparationPipelines:
    """Pipelines for preparing genomic data from various sources using Prefect.
    
    This class provides static methods for:
    - Downloading, converting, and splitting VCF data
    - Uploading processed data to Hugging Face Hub
    """

    @staticmethod
    def _setup_logging(name: str, log: bool = True) -> None:
        """Setup logging for a pipeline execution."""
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / f"{name}.json", log_dir / f"{name}.log")

    @staticmethod
    def download_clinvar(
        assembly: str = "GRCh38_ensembl",
        url: Optional[str] = None,
        pattern: Optional[str] = None,
        dest_dir: Optional[Path] = None,
        vcf_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download ClinVar VCF files and convert to parquet using Prefect.
        
        Args:
            assembly: Genome assembly and source. Options:
                - GRCh38_ensembl: Ensembl VEP-annotated ClinVar for GRCh38 (default)
                - GRCh38: NCBI ClinVar for GRCh38
                - GRCh37: NCBI ClinVar for GRCh37
            url: Custom URL (overrides assembly-based URL)
            pattern: Regex pattern to filter files
            dest_dir: Base destination directory
            vcf_dir: Specific directory for VCF files
            parquet_dir: Specific directory for Parquet files
            with_splitting: Whether to split by variant type
            explode_snv_alt: Whether to explode SNV ALT column
            alts_list: Whether to add a list of alternative alleles as 'alts' column
            log: Whether to enable logging
            profile: Whether to track resource usage
        """
        PreparationPipelines._setup_logging("download_clinvar", log)
        return prepare_clinvar_flow(
            url=url,
            pattern=pattern,
            dest_dir=dest_dir,
            vcf_dir=vcf_dir,
            parquet_dir=parquet_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            assembly=assembly,
            profile=profile
        )

    @staticmethod
    def download_ensembl(
        species: str = "homo_sapiens",
        dest_dir: Optional[Path] = None,
        vcf_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
        pattern: Optional[str] = None,
        url: Optional[str] = None,
        http_max_pool: Optional[int] = None,
        http_chunk_size: Optional[int] = None,
        connect_timeout: Optional[float] = None,
        sock_read_timeout: Optional[float] = None,
        retries: Optional[int] = None,
        verify_checksums: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download Ensembl VCF files and convert to parquet using Prefect.
        
        Args:
            species: Species name (e.g., homo_sapiens, mus_musculus)
            dest_dir: Base destination directory
            vcf_dir: Specific directory for VCF files
            parquet_dir: Specific directory for Parquet files
            with_splitting: Whether to split by variant type
            explode_snv_alt: Whether to explode SNV ALT column
            alts_list: Whether to add a list of alternative alleles as 'alts' column
            log: Whether to enable logging
            pattern: Regex pattern to filter files
            url: Custom base URL (overrides default Ensembl URL)
            http_max_pool: HTTP connection pool size (default: 20)
            http_chunk_size: HTTP chunk size for reads
            connect_timeout: Connection timeout in seconds (default: 10s)
            sock_read_timeout: Socket read timeout in seconds (default: 120s)
            retries: Number of retry attempts (default: 10)
            verify_checksums: Whether to verify file checksums (default: True)
            profile: Whether to track resource usage
        """
        PreparationPipelines._setup_logging("download_ensembl", log)
        
        # If custom URL provided, we use the generic source flow
        if url:
            # We still respect the vcf/parquet subfolder structure if dest_dir or species is provided
            base_dir = Path(dest_dir) if dest_dir else get_default_cache_dir("ensembl_custom")
            species_dir = base_dir / species
            
            return prepare_vcf_source_flow(
                url=url,
                pattern=pattern,
                name="ensembl_custom",
                dest_dir=species_dir,
                vcf_dir=vcf_dir,
                parquet_dir=parquet_dir,
                with_splitting=with_splitting,
                explode_snv_alt=explode_snv_alt,
                alts_list=alts_list,
                http_max_pool=http_max_pool,
                http_chunk_size=http_chunk_size,
                connect_timeout=connect_timeout,
                sock_read_timeout=sock_read_timeout,
                retries=retries,
                verify_checksums=verify_checksums,
                profile=profile
            )
            
        return prepare_ensembl_flow(
            species=species,
            dest_dir=dest_dir,
            vcf_dir=vcf_dir,
            parquet_dir=parquet_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            pattern=pattern,
            http_max_pool=http_max_pool,
            http_chunk_size=http_chunk_size,
            connect_timeout=connect_timeout,
            sock_read_timeout=sock_read_timeout,
            retries=retries,
            verify_checksums=verify_checksums,
            profile=profile
        )
    
    @staticmethod
    def download_dbsnp(
        dest_dir: Optional[Path] = None,
        build: str = "GRCh38",
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download dbSNP VCF files and convert to parquet using Prefect."""
        PreparationPipelines._setup_logging(f"download_dbsnp_{build.lower()}", log)
        return prepare_dbsnp_flow(
            build=build,
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            profile=profile
        )
    
    @staticmethod
    def download_dbsnp_t2t(
        dest_dir: Optional[Path] = None,
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        compression: str = "zstd",
        compression_level: int = 14,
        progress_interval_seconds: float = 60.0,
        s3_max_pool: Optional[int] = None,
        s3_block_size: Optional[int] = None,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download dbSNP T2T VCF files and convert to parquet using Prefect."""
        PreparationPipelines._setup_logging("download_dbsnp_t2t", log)
        return prepare_dbsnp_t2t_flow(
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            compression=compression,
            compression_level=compression_level,
            progress_interval_seconds=progress_interval_seconds,
            s3_max_pool=s3_max_pool,
            s3_block_size=s3_block_size,
            profile=profile
        )

    @staticmethod
    def download_gnomad(
        dest_dir: Optional[Path] = None,
        version: str = "v4",
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download gnomAD VCF files and convert to parquet using Prefect."""
        PreparationPipelines._setup_logging(f"download_gnomad_{version}", log)
        return prepare_gnomad_flow(
            version=version,
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            profile=profile
        )
    
    @staticmethod
    def split_existing_parquets(
        parquet_files: list[Path] | Path,
        explode_snv_alt: bool = False,
        write_to: Optional[Path] = None,
        log: bool = True,
        profile: bool = True,
    ) -> SplitResult:
        """Quick function to split existing parquet files by variant type using Prefect."""
        if isinstance(parquet_files, Path):
            parquet_files = [parquet_files]
            
        PreparationPipelines._setup_logging("split_parquets", log)
            
        with prefect_flow_run("Split Existing Parquets", profile=profile):
            return split_parquets_task(
                parquet_paths=parquet_files,
                explode_snv_alt=explode_snv_alt,
                write_to=write_to
            )
    
    @staticmethod
    def upload_dataset_to_hf(
        dataset_type: str,
        source_dir: Optional[Path] = None,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Generic method to upload parquet files to Hugging Face Hub."""
        # Map dataset types to default repos and card generators
        config = {
            "clinvar": {
                "repo": "just-dna-seq/clinvar",
                "card": generate_clinvar_card,
                "default_source": get_default_output_dir("clinvar")
            },
            "ensembl": {
                "repo": "just-dna-seq/ensembl_variations",
                "card": generate_ensembl_card,
                "default_source": get_default_output_dir("ensembl")
            },
            "dbsnp": {
                "repo": "just-dna-seq/dbsnp",
                "card": generate_dbsnp_card,
                "default_source": get_default_output_dir("dbsnp_grch38")
            },
            "dbsnp_t2t": {
                "repo": "just-dna-seq/dbsnp_t2t",
                "card": generate_dbsnp_t2t_card,
                "default_source": get_default_output_dir("dbsnp_t2t")
            },
            "gnomad": {
                "repo": "just-dna-seq/gnomad",
                "card": generate_gnomad_card,
                "default_source": get_default_output_dir("gnomad_v4")
            }
        }
        
        ds_cfg = config.get(dataset_type.lower())
        if not ds_cfg:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        repo_id = repo_id or ds_cfg["repo"]
        if source_dir is None:
            source_dir = ds_cfg["default_source"]
            # Fallback for dbsnp/gnomad versions if needed
            if not source_dir.exists() and dataset_type.lower() == "dbsnp":
                source_dir = get_default_output_dir("dbsnp_grch37")
            elif not source_dir.exists() and dataset_type.lower() == "gnomad":
                source_dir = get_default_output_dir("gnomad_v3")
                
            if (source_dir / "splitted_variants").exists():
                source_dir = source_dir / "splitted_variants"
        
        PreparationPipelines._setup_logging(f"upload_{dataset_type.lower()}", log)
        
        with start_action(action_type=f"upload_{dataset_type}_to_hf", source_dir=str(source_dir), repo_id=repo_id, pattern=pattern):
            parquet_files = collect_parquet_files(source_dir, pattern=pattern)
            
            variant_types = set()
            for f in parquet_files:
                try:
                    rel_p = f.relative_to(source_dir)
                    if len(rel_p.parts) > 1:
                        variant_types.add(rel_p.parts[0])
                except ValueError:
                    continue
            
            total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            dataset_card = ds_cfg["card"](len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
            
            return upload_parquet_to_hf(
                parquet_files=parquet_files,
                repo_id=repo_id,
                token=token,
                path_prefix=path_prefix,
                source_dir=source_dir,
                dataset_card_content=dataset_card,
            )

    @staticmethod
    def upload_clinvar_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/clinvar",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload ClinVar parquet files to Hugging Face Hub (deprecated, use upload_dataset_to_hf)."""
        return PreparationPipelines.upload_dataset_to_hf("clinvar", source_dir, repo_id, token, pattern, path_prefix, log)

    @staticmethod
    def upload_ensembl_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/ensembl_variations",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload Ensembl variation parquet files to Hugging Face Hub (deprecated, use upload_dataset_to_hf)."""
        return PreparationPipelines.upload_dataset_to_hf("ensembl", source_dir, repo_id, token, pattern, path_prefix, log)

    @staticmethod
    def upload_dbsnp_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/dbsnp",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload dbSNP parquet files to Hugging Face Hub (deprecated, use upload_dataset_to_hf)."""
        return PreparationPipelines.upload_dataset_to_hf("dbsnp", source_dir, repo_id, token, pattern, path_prefix, log)

    @staticmethod
    def upload_gnomad_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/gnomad",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload gnomAD parquet files to Hugging Face Hub (deprecated, use upload_dataset_to_hf)."""
        return PreparationPipelines.upload_dataset_to_hf("gnomad", source_dir, repo_id, token, pattern, path_prefix, log)

    @staticmethod
    def compute_ensembl_rsid_coords(
        input_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        memory_fraction: float = 0.8,
        auto_split: bool = True,
        explode_snv_alt: bool = False,
        output_dataset: bool = False,
        force: bool = False,
        compression_level: int = 14,
        log: bool = True,
        profile: bool = True,
    ) -> RSIDCoordinateResult:
        """Compute rsID coordinates from Ensembl parquet files.
        
        Args:
            input_dir: Directory containing splitted_variants subdirectories
            output_path: Path for output parquet file
            memory_fraction: Fraction of available memory to use (0.0-1.0)
            auto_split: If True and split files don't exist, automatically run split first
            explode_snv_alt: Whether to explode SNV ALT column during auto-splitting
            output_dataset: If True, write a directory of per-chromosome parquet chunks
            force: If True, re-compute all chunks even if they already exist
            compression_level: ZSTD compression level for the final output
            log: Enable logging
            profile: Enable profiling
        """
        PreparationPipelines._setup_logging("ensembl_rsid_coords", log)
        return ensembl_rsid_coords_flow(
            input_dir=input_dir,
            output_path=output_path,
            memory_fraction=memory_fraction,
            auto_split=auto_split,
            explode_snv_alt=explode_snv_alt,
            output_dataset=output_dataset,
            force=force,
            compression_level=compression_level,
            profile=profile
        )


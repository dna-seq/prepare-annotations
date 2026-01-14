import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any

import aiohttp
import fsspec
import polars as pl
from aiohttp import ClientResponseError, ClientTimeout
from eliot import start_action
from fsspec.exceptions import FSTimeoutError
from fsspec.callbacks import Callback
from platformdirs import user_cache_dir
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential


from prepare_annotations.paths import get_cache_dir
from prepare_annotations.io import AnnotatedLazyFrame, vcf_to_parquet

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class ChecksumInfo:
    """BSD sum checksum information for a file."""
    checksum: int  # 16-bit CRC checksum
    blocks: int    # Size in 1024-byte blocks
    filename: str


def parse_checksums_file(content: str) -> dict[str, ChecksumInfo]:
    """Parse Ensembl/BSD CHECKSUMS file format.
    
    Format: checksum blocks filename
    Example: 29888 168645 homo_sapiens-chr21.vcf.gz
    
    Args:
        content: Raw content of the CHECKSUMS file
        
    Returns:
        Dictionary mapping filename to ChecksumInfo
    """
    result = {}
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3:
            checksum = int(parts[0])
            blocks = int(parts[1])
            filename = parts[2]
            result[filename] = ChecksumInfo(checksum=checksum, blocks=blocks, filename=filename)
    return result


def download_checksums(base_url: str, checksums_filename: str = "CHECKSUMS") -> dict[str, ChecksumInfo]:
    """Download and parse a CHECKSUMS file from a URL.
    
    Args:
        base_url: Base URL (directory) containing the CHECKSUMS file
        checksums_filename: Name of the checksums file (default: CHECKSUMS)
        
    Returns:
        Dictionary mapping filename to ChecksumInfo
        
    Raises:
        FileNotFoundError: If CHECKSUMS file doesn't exist at the URL
    """
    with start_action(action_type="download_checksums", base_url=base_url) as action:
        # Ensure base_url ends with /
        if not base_url.endswith('/'):
            base_url = base_url + '/'
        
        checksums_url = base_url + checksums_filename
        
        storage_options = {}
        if base_url.startswith("s3://"):
            storage_options["anon"] = True
        
        fs, path = fsspec.core.url_to_fs(checksums_url, **storage_options)
        
        try:
            with fs.open(path, 'r') as f:
                content = f.read()
            
            checksums = parse_checksums_file(content)
            action.log(
                message_type="info",
                step="checksums_loaded",
                file_count=len(checksums),
                url=checksums_url
            )
            return checksums
            
        except FileNotFoundError:
            action.log(
                message_type="warning",
                step="checksums_not_found",
                url=checksums_url
            )
            raise
        except Exception as e:
            action.log(
                message_type="warning",
                step="checksums_download_failed",
                url=checksums_url,
                error=str(e)
            )
            raise


def compute_checksum(file_path: Path) -> tuple[int, int]:
    """Compute BSD sum checksum for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (checksum, block_count)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If sum command fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    result = subprocess.run(
        ["sum", str(file_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"sum command failed: {result.stderr}")
    
    parts = result.stdout.split()
    return int(parts[0]), int(parts[1])


def verify_checksum(
    file_path: Path, 
    expected: ChecksumInfo,
    action: Optional[Any] = None
) -> tuple[bool, str]:
    """Verify a file's checksum against expected value.
    
    Args:
        file_path: Path to the file to verify
        expected: Expected checksum info
        action: Optional Eliot action for logging
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        actual_checksum, actual_blocks = compute_checksum(file_path)
        
        if actual_checksum != expected.checksum or actual_blocks != expected.blocks:
            msg = (
                f"Checksum mismatch for {file_path.name}: "
                f"got {actual_checksum}/{actual_blocks}, "
                f"expected {expected.checksum}/{expected.blocks}"
            )
            if action:
                action.log(
                    message_type="warning",
                    step="checksum_mismatch",
                    file=str(file_path),
                    actual_checksum=actual_checksum,
                    actual_blocks=actual_blocks,
                    expected_checksum=expected.checksum,
                    expected_blocks=expected.blocks
                )
            return False, msg
        
        if action:
            action.log(
                message_type="info",
                step="checksum_verified",
                file=str(file_path),
                checksum=actual_checksum,
                blocks=actual_blocks
            )
        return True, "OK"
        
    except Exception as e:
        msg = f"Checksum verification failed for {file_path.name}: {e}"
        if action:
            action.log(
                message_type="error",
                step="checksum_error",
                file=str(file_path),
                error=str(e)
            )
        return False, msg


class EliotDownloadCallback(Callback):
    """
    A callback for fsspec that logs progress to Eliot at intervals.
    """
    def __init__(self, action: Any, url: str, log_interval: float = 60.0):
        super().__init__()
        self.action = action
        self.url = url
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.start_time = self.last_log_time
        self.remote_size = None

    def set_size(self, size: int | None) -> None:
        self.remote_size = size

    def relative_update(self, inc: int = 1) -> None:
        self.value += inc
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress(current_time)
            self.last_log_time = current_time

    def _log_progress(self, current_time: float) -> None:
        total_bytes = self.value
        mb_downloaded = total_bytes / (1024 * 1024)
        elapsed = current_time - self.start_time
        
        # We don't have existing_size easily here if get_file is used, 
        # so speed_mbps is based on total download in this session.
        speed_mbps = (mb_downloaded) / elapsed if elapsed > 0 else 0
        
        progress_percent: float | None = None
        eta_seconds: float | None = None
        
        if self.remote_size and self.remote_size > 0:
            progress_percent = (total_bytes / self.remote_size) * 100.0
            if total_bytes > 0 and elapsed > 0:
                bytes_per_sec = total_bytes / elapsed
                remaining = max(0, self.remote_size - total_bytes)
                eta_seconds = remaining / bytes_per_sec if bytes_per_sec > 0 else None
        
        self.action.log(
            message_type="download_progress",
            url=self.url,
            mb_downloaded=round(mb_downloaded, 2),
            mb_total=round(self.remote_size / (1024 * 1024), 2) if self.remote_size else None,
            progress_percent=round(progress_percent, 2) if progress_percent is not None else None,
            speed_mbps=round(speed_mbps, 2),
            eta_seconds=round(eta_seconds, 1) if eta_seconds is not None else None,
            elapsed_seconds=round(elapsed, 1)
        )


def _retryable_http_error(exc: BaseException) -> bool:
    # Network/client-level errors and timeouts are retryable
    if isinstance(exc, (aiohttp.ClientError, FSTimeoutError, TimeoutError, OSError)):
        return True
    # Response-level status codes that are commonly transient
    if isinstance(exc, ClientResponseError):
        return exc.status in RETRYABLE_STATUS
    # mmap cache errors are retryable (blockcache corruption)
    if isinstance(exc, ValueError) and "mmap length is greater than file size" in str(exc):
        return True
    return False


def list_paths(url: str, pattern: str | None = None, file_only: bool = True) -> list[str]:
    storage_options = {}
    if url.startswith("s3://"):
        storage_options["anon"] = True
        
    fs, path = fsspec.core.url_to_fs(url, **storage_options)  # infer filesystem and strip protocol
    paths = fs.glob(path) if any(ch in path for ch in "*?[]") else [
        e["name"] for e in fs.ls(path, detail=True) if (not file_only) or e.get("type") == "file"
    ]
    # fsspec fs.ls might return full paths or relative paths depending on the implementation.
    # For S3, it usually returns the full path without the protocol.
    # We ensure we return the full URL including protocol for consistency.
    protocol = url.split("://")[0] + "://" if "://" in url else ""
    
    # Filter by pattern if provided
    if pattern:
        rx = re.compile(pattern)
        paths = [p for p in paths if rx.search(p.rsplit("/", 1)[-1])]
    
    # Ensure protocol is present
    result = []
    for p in paths:
        if "://" not in p:
            result.append(f"{protocol}{p}")
        else:
            result.append(p)
            
    return result

def download_path(
    url: str,
    name: str | Path = "downloads",
    dest_dir: Path | None = None,
    cache_storage: Path | None = None,
    check_files: bool = True,
    expiry_time: int | float | None = 7 * 24 * 3600,  # 7 days
    timeout: float | None = None,  # seconds
    connect_timeout: float | None = 10.0,
    sock_read_timeout: float | None = 120.0,
    retries: int = 10,  # Increased default retries for large files
    use_blockcache: bool = True,
    chunk_size: int = 8 * 1024 * 1024,  # 8MB
    resume: bool = True,  # Enable resuming by default
    progress_interval_seconds: float | None = None,
    s3_max_pool: int | None = None,
    s3_block_size: int | None = None,
    http_max_pool: int | None = None,
    http_chunk_size: int | None = None,
    expected_checksum: ChecksumInfo | None = None,
) -> Path:
    """
    Robust HTTP/HTTPS/S3 downloader with fsspec filecache + Tenacity retry/backoff
    and atomic finalization to dest; resumable via filecache's persisted ranges.
    
    Args:
        url: URL to download from (supports http://, https://, s3://, ftp://)
        name: Name for the download cache directory
        dest_dir: Destination directory for the downloaded file
        cache_storage: Directory for fsspec cache storage
        check_files: Whether to check if files already exist
        expiry_time: Cache expiry time in seconds (default: 7 days)
        timeout: Total timeout in seconds (default: 24h from env var)
        connect_timeout: Connection timeout in seconds (default: 10s)
        sock_read_timeout: Socket read timeout in seconds (default: 120s)
        retries: Number of retry attempts (default: 10)
        use_blockcache: Whether to use block caching (kept for compatibility)
        chunk_size: Chunk size for downloads in bytes (default: 8MB)
        resume: Whether to resume interrupted downloads (default: True)
        progress_interval_seconds: Interval for progress logging
        s3_max_pool: S3 connection pool size (default: 50)
        s3_block_size: S3 block size for reads
        http_max_pool: HTTP/HTTPS connection pool size (default: 20)
        http_chunk_size: HTTP/HTTPS chunk size for reads
        expected_checksum: Optional ChecksumInfo to verify file integrity.
            If provided:
            - Before download: existing files are verified; mismatches trigger re-download
            - After download: downloaded files are verified; failures raise an error
        
    Returns:
        Path to the downloaded file
        
    Raises:
        ValueError: If checksum verification fails after download
    """
    timeout = float(os.getenv("PREPARE_ANNOTATIONS_DOWNLOAD_TIMEOUT", 86400.0) if timeout is None else timeout) # Default 24h for dbSNP

    user_cache_path = get_cache_dir()
    # Track whether the caller explicitly provided dest_dir to reflect it in logs
    dest_dir_was_provided = dest_dir is not None
    if dest_dir is None:
        dest_dir = user_cache_path / name if isinstance(name, str) else Path(name)

    if cache_storage is None:
        # Always store fsspec cache in a hidden subdirectory to avoid cluttering the main folder
        cache_storage = dest_dir / ".fsspec_cache"

    dest_dir = Path(dest_dir)
    cache_storage = Path(cache_storage)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cache_storage.mkdir(parents=True, exist_ok=True)

    local = dest_dir / url.rsplit("/", 1)[-1]
    tmp = local.with_suffix(local.suffix + ".part")

    # Use filecache instead of blockcache to avoid mmap issues with concurrent/parallel downloads
    # blockcache can fail with "mmap length is greater than file size" when cache files are being created
    # For now, always use filecache as it's more reliable for parallel downloads
    cache_proto = "filecache"
    chained_url = f"{cache_proto}::{url}"

    # Target-specific options for the underlying filesystem (s3, http, etc.)
    target_options = {}
    if url.startswith("s3://"):
        target_options["anon"] = True
        # Speed up S3 by increasing pool size and using requester pays if needed
        # We also allow env-var overrides for advanced tuning
        s3_max_pool = s3_max_pool or int(os.getenv("PREPARE_ANNOTATIONS_S3_MAX_POOL", "50"))
        target_options["config_kwargs"] = {"max_pool_connections": s3_max_pool}
    elif url.startswith(("http://", "https://")):
        # Configure HTTP/HTTPS client with timeouts
        client_timeout = ClientTimeout(total=timeout, connect=connect_timeout, sock_read=sock_read_timeout)
        target_options["client_kwargs"] = {"timeout": client_timeout}
        
        # Note: Connection pooling for HTTP/HTTPS is handled internally by fsspec/aiohttp
        # The http_max_pool parameter is accepted for API compatibility with S3,
        # but fsspec's HTTPFileSystem manages connections automatically based on
        # the event loop and session handling. Unlike S3's boto3, we cannot
        # easily configure the pool size externally without event loop issues.
        
        # Optional: configure chunk size for HTTP reads if specified
        if http_chunk_size is not None:
            target_options["block_size"] = http_chunk_size

    # Options for the caching filesystem itself
    # For chained protocols like filecache::s3://, we use TOP-LEVEL protocol keys only
    # (not nested target_options) to avoid duplicate kwargs errors
    storage_options = {
        cache_proto: {
            "cache_storage": str(cache_storage),
            "check_files": check_files,   # filecache-only; harmless for blockcache
            "expiry_time": expiry_time,   # filecache-only; harmless for blockcache
        }
    }
    
    # Add protocol-specific options at top level for fsspec chaining
    if url.startswith("s3://"):
        storage_options["s3"] = target_options
    elif url.startswith(("http://", "https://")):
        http_key = "https" if url.startswith("https://") else "http"
        storage_options[http_key] = target_options

    def _clear_cache_for_url() -> None:

        """Clear any cached files for this specific URL."""
        import hashlib
        # fsspec uses the URL to generate cache filenames
        url_hash = hashlib.md5(url.encode()).hexdigest()
        # Try to find and remove any cache files related to this URL
        for pattern in [f"*{url_hash}*", f"*{url.rsplit('/', 1)[-1]}*"]:
            cache_files = list(cache_storage.glob(pattern))
            for cf in cache_files:
                try:
                    if cf.is_file():
                        cf.unlink()
                except Exception:
                    pass

    def _download_to_tmp_with_retry(destination_tmp: Path) -> None:
        retryer = Retrying(
            retry=retry_if_exception(_retryable_http_error),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=60.0),
            stop=stop_after_attempt(retries),
            reraise=True,
        )

        # Check if we should resume - keep the .part file if resume=True
        if not resume and destination_tmp.exists():
            destination_tmp.unlink()

        # Get progress logging interval:
        # - explicit arg wins (lets pipelines tune per dataset)
        # - fallback to env var for global tuning
        # - default 60 seconds
        log_interval = (
            float(progress_interval_seconds)
            if progress_interval_seconds is not None
            else float(os.getenv("PREPARE_ANNOTATIONS_PROGRESS_INTERVAL", "60.0"))
        )

        for attempt in retryer:
            with attempt:
                existing_size = destination_tmp.stat().st_size if destination_tmp.exists() else 0
                
                # Use the target_options we built earlier for url_to_fs
                fs, path = fsspec.core.url_to_fs(url, **target_options)
                
                # Best-effort remote size for percent/ETA (may be unavailable for some backends)
                remote_size: int | None = None
                try:
                    info = fs.info(path)
                    if isinstance(info, dict):
                        size_val = info.get("size")
                        if size_val is not None:
                            remote_size = int(size_val)
                except Exception:
                    remote_size = None

                # Use fs.get_file (fastest) if not resuming.
                # fs.get_file for S3 is multi-threaded and much faster than manual streaming.
                try:
                    # If no existing data, try fs.get_file (fastest) 
                    if existing_size == 0:
                        callback = EliotDownloadCallback(action, url, log_interval=log_interval)
                        callback.set_size(remote_size)
                        
                        # Use fs.get for S3 specifically if possible as it's highly optimized for concurrency
                        if url.startswith("s3://"):
                            action.log(message_type="info", step="fast_s3_download_start", url=url)
                            fs.get(path, str(destination_tmp), callback=callback)
                        else:
                            fs.get_file(path, str(destination_tmp), callback=callback)
                        
                        total_bytes = destination_tmp.stat().st_size
                    else:
                        # Use streaming with fsspec for resumable downloads
                        raise NotImplementedError("Force streaming path for resume")
                        
                except (AttributeError, NotImplementedError, TypeError):
                    # Streaming path - supports resuming and progress logging
                    # IMPORTANT: Stream directly from the underlying fs (S3/HTTP), NOT through
                    # filecache, because filecache downloads the ENTIRE file before returning
                    # a file handle, which defeats progress logging.
                    mode = "ab" if existing_size > 0 else "wb"
                    
                    # For S3, a larger block_size is crucial for sequential read speed in the manual loop
                    open_kwargs = {}
                    if url.startswith("s3://"):
                        s3_block_size_val = s3_block_size or int(os.getenv("PREPARE_ANNOTATIONS_S3_BLOCK_SIZE", str(50 * 1024 * 1024))) # Default 50MB
                        open_kwargs["block_size"] = s3_block_size_val
                        action.log(message_type="info", step="streaming_download_start", url=url, resume=(existing_size > 0), block_size=s3_block_size_val)

                    with fs.open(path, mode="rb", **open_kwargs) as src:
                        if existing_size > 0:
                            src.seek(existing_size)
                        
                        with open(destination_tmp, mode) as dst:
                            # Progress tracking with rate limiting
                            start_time = time.time()
                            last_log_time = start_time
                            total_bytes = existing_size
                            
                            while True:
                                data = src.read(chunk_size)
                                if not data:
                                    break
                                dst.write(data)
                                total_bytes += len(data)
                                
                                # Rate-limited progress logging
                                current_time = time.time()
                                if current_time - last_log_time >= log_interval:
                                    mb_downloaded = total_bytes / (1024 * 1024)
                                    elapsed = current_time - start_time
                                    session_bytes = total_bytes - existing_size
                                    speed_mbps = (session_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                                    progress_percent: float | None = None
                                    eta_seconds: float | None = None
                                    if remote_size and remote_size > 0:
                                        progress_percent = (total_bytes / remote_size) * 100.0
                                        if session_bytes > 0 and elapsed > 0:
                                            bytes_per_sec = session_bytes / elapsed
                                            remaining = max(0, remote_size - total_bytes)
                                            eta_seconds = remaining / bytes_per_sec if bytes_per_sec > 0 else None
                                    
                                    action.log(
                                        message_type="download_progress",
                                        url=url,
                                        mb_downloaded=round(mb_downloaded, 2),
                                        mb_total=round(remote_size / (1024 * 1024), 2) if remote_size else None,
                                        progress_percent=round(progress_percent, 2) if progress_percent is not None else None,
                                        speed_mbps=round(speed_mbps, 2),
                                        eta_seconds=round(eta_seconds, 1) if eta_seconds is not None else None,
                                        elapsed_seconds=round(elapsed, 1)
                                    )
                                    last_log_time = current_time
                
                # Log completion with final stats
                if destination_tmp.exists():
                    total_mb = destination_tmp.stat().st_size / (1024 * 1024)
                    action.log(
                        message_type="download_complete",
                        url=url,
                        mb_total=round(total_mb, 2),
                        progress_percent=100
                    )

    action_kwargs = {"action_type": "download_path", "url": url, "dest": str(local)}
    if dest_dir_was_provided:
        action_kwargs["dest_dir"] = str(dest_dir)

    with start_action(**action_kwargs) as action:
        action.log(message_type="info", step="start_download", url=url, timeout=timeout)

        # Get filesystem and remote info early for size comparison
        fs, path = fsspec.core.url_to_fs(url, **target_options)
        
        remote_size: Optional[int] = None
        try:
            info = fs.info(path)
            if isinstance(info, dict):
                remote_size = info.get("size")
        except Exception:
            pass

        # Smart skip: If the final file exists and passes verification, skip
        if local.exists():
            local_size = local.stat().st_size
            if not check_files: # User said "don't even check remote", just trust existence
                action.log(message_type="info", step="skip_existing_file", path=str(local), reason="check_files_disabled")
                return local
            
            # If we have a checksum, verify it for existing files
            if expected_checksum is not None:
                is_valid, msg = verify_checksum(local, expected_checksum, action)
                if is_valid:
                    action.log(
                        message_type="info", 
                        step="skip_existing_file", 
                        path=str(local), 
                        reason="checksum_verified"
                    )
                    return local
                else:
                    # Checksum mismatch - delete corrupted file and re-download
                    action.log(
                        message_type="warning",
                        step="checksum_mismatch_redownload",
                        path=str(local),
                        message=msg
                    )
                    local.unlink()
                    # Also clear any partial .part file
                    if tmp.exists():
                        tmp.unlink()
            else:
                # No checksum available, fall back to size check
                if remote_size is not None and local_size == remote_size:
                    action.log(message_type="info", step="skip_existing_file", path=str(local), reason="size_matches_remote")
                    return local
                elif remote_size is None:
                    # If we can't get remote size but file exists, we might want to skip if we're not forced
                    action.log(message_type="warning", step="remote_size_unknown", path=str(local), reason="assuming_file_complete")
                    return local

        # Clear any corrupted cache files before attempting download only if not resuming
        if not resume or not tmp.exists():
            _clear_cache_for_url()

        # Stream from cached reader to tmp, then atomically move into dest_dir
        _download_to_tmp_with_retry(tmp)
        tmp.replace(local)

        # Cleanup fsspec cache for this URL after successful download
        try:
            _clear_cache_for_url()
            # If the cache storage is now empty (except for potentially other downloads), 
            # we don't remove it here as other parallel downloads might be using it.
            # But we've cleaned up the hex files for THIS specific download.
        except Exception as e:
            action.log(message_type="warning", step="cleanup_cache_failed", error=str(e))

        # Post-download checksum verification
        if expected_checksum is not None:
            is_valid, msg = verify_checksum(local, expected_checksum, action)
            if not is_valid:
                # Delete the corrupted download
                local.unlink()
                raise ValueError(
                    f"Downloaded file failed checksum verification: {msg}. "
                    f"The file has been deleted. Please retry the download."
                )
            action.log(
                message_type="info",
                step="post_download_checksum_verified",
                path=str(local),
                checksum=expected_checksum.checksum,
                blocks=expected_checksum.blocks
            )

        action.log(message_type="info", step="download_finished", final_path=str(local))
        return local
        
def convert_to_parquet(
    vcf_path: Path, 
    parquet_path: Optional[Path] = None, 
    overwrite: bool = False,
    compression: str = "zstd",
    compression_level: Optional[int] = None,
    alts_list: bool = True,
) -> AnnotatedLazyFrame:
    """Convert a VCF file to Parquet using io utilities.

    If the provided path does not look like a VCF (e.g., it's an index file like
    .tbi), the function skips conversion and returns the original path.
    """
    with start_action(action_type="convert_to_parquet", vcf_path=str(vcf_path)) as action:
        
        
        suffixes = vcf_path.suffixes
        is_index = suffixes and suffixes[-1] in [".tbi", ".csi", ".idx"]
        
        # A file is a VCF if it has .vcf in its suffixes, 
        # OR if it's a compressed file (.gz/.bgz) and we have reason to believe it's a VCF.
        # For dbSNP, NCBI often uses GCF_...gz without the .vcf part.
        is_vcf = not is_index and (
            (".vcf" in suffixes) or 
            (suffixes and suffixes[-1] in [".gz", ".bgz"] and (vcf_path.name.startswith("GCF_") or vcf_path.name.startswith("GCA_") or "dbsnp" in vcf_path.name.lower()))
        )

        if not is_vcf:
            action.log(message_type="info", step="skip_non_vcf", path=str(vcf_path))
            # Return empty LazyFrame and the original path for non-VCF files
            empty_lazy = pl.LazyFrame()
            return empty_lazy, vcf_path

        lazy_frame, parquet_path = vcf_to_parquet(
            vcf_path=vcf_path, 
            parquet_path=parquet_path, 
            overwrite=overwrite,
            compression=compression,
            compression_level=compression_level,
            alts_list=alts_list,
        )
        action.log(
            message_type="info",
            step="conversion_complete",
            parquet_path=str(parquet_path),
        )
        return lazy_frame, parquet_path


def validate_downloads_and_parquet(
    urls: list[str],
    vcf_local: list[Path],
    vcf_parquet_path: list[Path],
    check_integrity: bool = True,
) -> tuple[list[str], list[Path], list[Path]]:
    """
    Reduce-style validator to ensure that for every discovered URL we have a
    downloaded local file, and for every downloaded VCF we produced a parquet.

    Also validates the file sizes of downloaded files against expected sizes if available.
    If check_integrity is True, also performs basic file format validation (e.g. gzip check).

    Returns the validated lists to keep them available to downstream steps.
    """
    with start_action(action_type="validate_downloads_and_parquet") as action:
        # Normalize singletons to lists
        if isinstance(vcf_local, Path):
            vcf_local_list = [vcf_local]
        else:
            vcf_local_list = list(vcf_local)

        if isinstance(vcf_parquet_path, Path):
            parquet_list = [vcf_parquet_path]
        else:
            parquet_list = list(vcf_parquet_path)

        # Filter out non-VCF entries from conversion stage (e.g., .tbi/.csi) which return empty LazyFrame and original path
        # We consider a file a VCF if name contains .vcf and is not an index
        def _is_vcf(p: Path) -> bool:
            suffixes = p.suffixes
            if not suffixes:
                return False
            is_index = suffixes and suffixes[-1] in [".tbi", ".csi", ".idx"]
            
            return not is_index and (
                (".vcf" in suffixes) or 
                (suffixes[-1] in [".gz", ".bgz"] and (p.name.startswith("GCF_") or p.name.startswith("GCA_") or "dbsnp" in p.name.lower()))
            )

        vcf_files = [p for p in vcf_local_list if _is_vcf(p)]

        # Basic existence checks
        missing_locals = [p for p in vcf_local_list if not Path(p).exists()]
        if missing_locals:
            action.log(message_type="error", missing_locals=[str(p) for p in missing_locals])
            raise FileNotFoundError(f"Missing downloaded files: {missing_locals}")

        # Ensure each URL has a corresponding local file by filename
        local_by_name = {Path(p).name: Path(p) for p in vcf_local_list}
        missing_for_urls: list[tuple[str, str]] = []  # (url, expected_filename)
        for url in urls:
            expected_name = url.rsplit("/", 1)[-1]
            if expected_name not in local_by_name:
                missing_for_urls.append((url, expected_name))
        if missing_for_urls:
            action.log(
                message_type="error",
                missing_by_url=[{"url": u, "expected": n} for (u, n) in missing_for_urls],
            )
            raise FileNotFoundError(
                f"No local files matching URLs: {[n for (_, n) in missing_for_urls]}"
            )

        missing_parquet = [p for p in parquet_list if not Path(p).exists()]
        # parquet_list may be empty if all inputs were non-VCF (e.g., only indexes matched); that's acceptable.
        if parquet_list and missing_parquet:
            action.log(message_type="error", missing_parquet=[str(p) for p in missing_parquet])
            raise FileNotFoundError(f"Missing parquet files: {missing_parquet}")

        # Heuristic: number of parquet files should be >= number of VCF files converted
        # (some steps may legitimately skip or deduplicate; we do a soft check/logging)
        if parquet_list and len(parquet_list) < len(vcf_files):
            action.log(
                message_type="warning",
                reason="fewer_parquets_than_vcfs",
                vcfs=len(vcf_files),
                parquets=len(parquet_list),
            )

        # Validate file sizes and integrity
        size_mismatches = []
        integrity_failures = []
        
        for url in urls:
            expected_name = url.rsplit("/", 1)[-1]
            local_path = local_by_name.get(expected_name)
            if local_path is None:
                continue
                
            # 1. Size check
            try:
                # Use storage options for S3 if needed
                storage_options = {}
                if url.startswith("s3://"):
                    storage_options["anon"] = True
                fs, path = fsspec.core.url_to_fs(url, **storage_options)
                
                remote_size = None
                remote_etag = None
                try:
                    info = fs.info(path)
                    remote_size = info.get("size") if isinstance(info, dict) else None
                    remote_etag = info.get("ETag") or info.get("etag") if isinstance(info, dict) else None
                except Exception:
                    if hasattr(fs, "size"):
                        remote_size = fs.size(path)
                
                if remote_size is not None:
                    local_size = local_path.stat().st_size
                    if local_size != remote_size:
                        size_mismatches.append({
                            "url": url,
                            "local_path": str(local_path),
                            "local_size": int(local_size),
                            "remote_size": int(remote_size),
                        })
                
                # 2. Integrity check (if enabled)
                if check_integrity and local_path.exists():
                    # Basic check: is it an HTML error page instead of binary?
                    try:
                        with open(local_path, "rb") as f:
                            head = f.read(1024)
                            if b"<!DOCTYPE html>" in head or b"<html" in head.lower():
                                integrity_failures.append({
                                    "url": url,
                                    "path": str(local_path),
                                    "reason": "File appears to be an HTML page (likely a download error/403/404)"
                                })
                                continue
                    except Exception:
                        pass

                    # Gzip check for .gz files
                    if local_path.suffix == ".gz":
                        try:
                            import gzip
                            # For very large files, we only check the header and tail for speed
                            # But we can at least try to read the first block
                            with gzip.open(local_path, "rb") as gf:
                                gf.read(1024)
                        except Exception as e:
                            integrity_failures.append({
                                "url": url,
                                "path": str(local_path),
                                "reason": f"Gzip integrity check failed: {str(e)}"
                            })

            except Exception as e:
                action.log(message_type="warning", reason="validation_failed", url=url, error=str(e))

        if size_mismatches or integrity_failures:
            error_msg = []
            if size_mismatches:
                action.log(message_type="error", size_mismatches=size_mismatches)
                for m in size_mismatches:
                    error_msg.append(f"{m['local_path']}: size mismatch (local {m['local_size']} vs remote {m['remote_size']})")
            
            if integrity_failures:
                action.log(message_type="error", integrity_failures=integrity_failures)
                for f in integrity_failures:
                    error_msg.append(f"{f['path']}: integrity failure ({f['reason']})")
            
            raise ValueError("\n".join(error_msg))

        action.log(
            message_type="info",
            urls_count=len(urls),
            local_count=len(vcf_local_list),
            parquet_count=len(parquet_list),
        )

        # Return validated outputs to keep pipeline continuity
        return urls, vcf_local_list, parquet_list



if __name__ == "__main__":
    from prepare_annotations.pipelines import PreparationPipelines
    
    print("Running validation for Ensembl downloads...")
    # Note: validate_ensembl might be legacy or missing in the current Pipelines class
    if hasattr(PreparationPipelines, "validate_ensembl"):
        validation_results = PreparationPipelines.validate_ensembl()
        print(f"Validation completed. Results: {validation_results}")
    else:
        print("Warning: PreparationPipelines.validate_ensembl not found.")

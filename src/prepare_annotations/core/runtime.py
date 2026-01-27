"""
Runtime environment and profiling utilities.
"""
from __future__ import annotations

import os
import time
import re
import psutil
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Any

from eliot import start_action, add_destination
from dotenv import find_dotenv, load_dotenv

from prepare_annotations.core.config import get_default_workers, get_parquet_workers
from prepare_annotations.core.models import ResourceReport


def load_env(override: bool = False) -> Optional[str]:
    """
    Search for .env file in the current directory and its parents.
    """
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=override)
        return env_path
    return None


def _log_resource_report_to_dagster(report: ResourceReport, context: Any) -> None:
    """Log resource report to Dagster context."""
    from dagster import MetadataValue
    
    # Build metadata dict with standard keys for charting
    metadata = {
        "duration_sec": MetadataValue.float(round(report.duration, 2)),
        "cpu_percent": MetadataValue.float(round(report.cpu_usage_percent, 1)),
        "peak_memory_mb": MetadataValue.float(round(report.peak_memory_mb, 2)),
        "memory_delta_mb": MetadataValue.float(round(report.memory_delta_mb, 2)),
    }
    
    # Add output metadata to asset materialization (enables charts in UI)
    if hasattr(context, 'add_output_metadata'):
        context.add_output_metadata(metadata)
    
    # Log to run logs for visibility (Dagster 1.12.x doesn't support metadata param in log.info)
    context.log.info(
        f"📊 Resource Report [{report.name}]: Duration: {report.duration:.2f}s, "
        f"CPU: {report.cpu_usage_percent:.1f}%, Peak RAM: {report.peak_memory_mb:.2f}MB"
    )


@contextmanager
def resource_tracker(name: str = "resource_usage", context: Optional[Any] = None):
    """Context manager to track execution time, CPU and peak memory usage.
    
    Args:
        name: Name for the resource tracking report
        context: Dagster AssetExecutionContext (pass this to enable UI charts)
    
    Automatically logs to:
    - Dagster: logs resource metrics as asset metadata (enables charts in UI)
    - Eliot: logs structured resource report (when Dagster context not provided)
    """
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss
    
    # Start CPU tracking
    process.cpu_percent(interval=None)
    
    data = {"name": name, "start_time": start_time, "start_mem": start_mem}
    yield data
    
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss
    cpu_usage = process.cpu_percent(interval=None)
    
    duration = end_time - start_time
    cpu_usage_percent = cpu_usage
    memory_delta = end_mem - start_mem
    peak_memory_mb = max(start_mem, end_mem) / (1024 * 1024)
    memory_delta_mb = (end_mem - start_mem) / (1024 * 1024)

    report = ResourceReport(
        name=name,
        duration=duration,
        cpu_usage_percent=cpu_usage_percent,
        peak_memory_mb=peak_memory_mb,
        memory_delta_mb=memory_delta_mb,
        start_time=start_time,
        end_time=end_time,
        start_mem=start_mem,
        end_mem=end_mem,
        memory_delta=memory_delta
    )
    
    # Store report in the data dict so calling code can access it
    data["report"] = report

    # Log to Dagster if context provided, otherwise use Eliot
    if context is not None:
        _log_resource_report_to_dagster(report, context)
    else:
        with start_action(
            action_type="resource_report",
            name=report.name,
            duration_sec=round(report.duration, 2),
            cpu_percent=round(report.cpu_usage_percent, 1),
            peak_memory_mb=round(report.peak_memory_mb, 2),
            memory_delta_mb=round(report.memory_delta_mb, 2),
        ):
            pass


def resolve_worker_counts(
    download_workers: Optional[int] = None,
    workers: Optional[int] = None,
    parquet_workers: Optional[int] = None,
) -> tuple[int, int, int]:
    """Resolve worker counts from parameters or environment."""
    # Load .env if present (does not override existing env vars)
    env_path = load_env(override=False)
    if env_path:
        with start_action(action_type="load_env", env_path=env_path):
            pass

    env_dl = os.getenv("PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS")
    env_workers = os.getenv("PREPARE_ANNOTATIONS_WORKERS")
    env_parquet = os.getenv("PREPARE_ANNOTATIONS_PARQUET_WORKERS")

    resolved_download = (
        int(os.getenv("PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS", os.cpu_count() or 1))
        if download_workers is None
        else max(1, int(download_workers))
    )
    resolved_workers = get_default_workers() if workers is None else max(1, int(workers))
    resolved_parquet = get_parquet_workers() if parquet_workers is None else max(1, int(parquet_workers))

    with start_action(
        action_type="resolve_worker_counts",
        PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS=env_dl,
        PREPARE_ANNOTATIONS_WORKERS=env_workers,
        PREPARE_ANNOTATIONS_PARQUET_WORKERS=env_parquet,
        resolved_download=resolved_download,
        resolved_workers=resolved_workers,
        resolved_parquet=resolved_parquet,
    ):
        pass
    return resolved_download, resolved_workers, resolved_parquet


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is in use on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

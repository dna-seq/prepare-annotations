from __future__ import annotations

import os
import time
import re
import psutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Any
from prepare_annotations.models import ResourceReport

from eliot import start_action
from dotenv import find_dotenv, load_dotenv

from prepare_annotations.config import get_default_workers, get_parquet_workers


def load_env(override: bool = False) -> Optional[str]:
    """
    Search for .env file in the current directory and its parents.
    """
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=override)
        return env_path
    return None


@contextmanager
def resource_tracker(name: str = "resource_usage"):
    """Context manager to track execution time, CPU and peak memory usage."""
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

    # Log to Prefect if available
    try:
        from prefect import get_run_logger
        from prefect.artifacts import create_markdown_artifact
        try:
            logger = get_run_logger()
            logger.info(
                f"Resource Report [{name}]: Duration: {report.duration:.2f}s, "
                f"CPU: {report.cpu_usage_percent:.1f}%, Peak RAM: {report.peak_memory_mb:.2f}MB"
            )
            
            # Clean name for artifact key (must be lowercase, alphanumeric and hyphens)
            clean_key = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
            if not clean_key:
                clean_key = "resource-usage"
            
            create_markdown_artifact(
                key=f"{clean_key}-resources",
                markdown=f"""# Resource Report: {name}
| Metric | Value |
| :--- | :--- |
| **Duration** | {report.duration:.2f}s |
| **CPU Usage** | {report.cpu_usage_percent:.1f}% |
| **Peak Memory** | {report.peak_memory_mb:.2f} MB |
| **Memory Delta** | {report.memory_delta_mb:+.2f} MB |
""",
                description=f"Resource usage metrics for {name}"
            )
        except Exception:
            # Not in a prefect context or logger not available
            pass
    except ImportError:
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


def setup_prefect_api() -> bool:
    """Setup Prefect API connection if environment variables are provided."""
    api_url = os.getenv("PREFECT_API_URL")
    if api_url:
        print(f"🚀 Prefect configured for server at: {api_url}")
        return True
    else:
        print("💡 Prefect running in ephemeral (standalone) mode.")
        return False


@contextmanager
def prefect_flow_run(name: str, profile: bool = True):
    """Context manager for running a Prefect flow with optional resource tracking."""
    setup_prefect_api()
    if profile:
        with resource_tracker(name) as tracker:
            yield tracker
    else:
        yield {}


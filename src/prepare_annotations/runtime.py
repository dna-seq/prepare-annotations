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
from prepare_annotations.models import ResourceReport

from eliot import start_action, add_destination
from dotenv import find_dotenv, load_dotenv

from prepare_annotations.config import get_default_workers, get_parquet_workers


_PREFECT_LOGGING_SETUP = False

def setup_prefect_logging():
    """Hook Eliot logging to Prefect logger to show logs in Prefect UI.
    
    This adds a destination to Eliot that forwards structured logs to Prefect's logger.
    Eliot will continue to write to all existing destinations (file, stdout, etc.)
    
    Key considerations:
    - Eliot emits multiple messages per action (start, success/failure)
    - Each message is a dictionary with various fields
    - Destinations must not raise exceptions (we catch all errors)
    - This is additive - existing destinations remain active
    """
    global _PREFECT_LOGGING_SETUP
    if _PREFECT_LOGGING_SETUP:
        return
    
    try:
        from prefect import get_run_logger
        
        def prefect_destination(message: Dict[str, Any]):
            """Forward Eliot messages to Prefect logger.
            
            Message structure varies:
            - action_type: present for action start/success/failure
            - action_status: "started", "succeeded", "failed"
            - message_type: present for log messages within actions
            - task_level: list showing nesting level [1], [1,1], [1,1,2] etc.
            
            Note: Prefect UI doesn't support tree rendering like to_nice_file/to_nice_stdout.
            We use indentation to hint at hierarchy, but it's not a true tree view.
            """
            try:
                # Only try to log if we are in a prefect context
                logger = get_run_logger()
                
                # Calculate indentation based on task nesting level
                task_level = message.get("task_level", [])
                # Use 2 spaces per nesting level, but cap at 3 levels to avoid excessive indentation
                indent = "  " * min(len(task_level) - 1, 3) if len(task_level) > 1 else ""
                
                # Eliot action messages (start/success/failure of an action)
                action_type = message.get("action_type")
                action_status = message.get("action_status")
                
                if action_type and action_status:
                    # Only log action completion, skip "started" to reduce noise
                    if action_status in ("succeeded", "failed"):
                        status_emoji = "✅" if action_status == "succeeded" else "❌"
                        logger.info(f"{indent}{status_emoji} {action_type}")
                    return
                
                # Eliot log messages (messages within an action)
                msg_type = message.get("message_type")
                if msg_type:
                    # Extract relevant fields for logging
                    # Remove internal eliot fields
                    display_msg = {k: v for k, v in message.items() 
                                 if k not in ("message_type", "task_uuid", "task_level", "timestamp")}
                    
                    if len(display_msg) == 0:
                        logger.info(f"{indent}📝 {msg_type}")
                    elif len(display_msg) == 1 and "message" in display_msg:
                        logger.info(f"{indent}📝 {msg_type}: {display_msg['message']}")
                    else:
                        # Log with structured data
                        msg_str = ", ".join(f"{k}={v}" for k, v in display_msg.items())
                        logger.info(f"{indent}📝 {msg_type}: {msg_str}")
                        
            except Exception:
                # Critical: Eliot destinations must NEVER raise exceptions
                # If we're not in a prefect context, silently ignore
                pass

        add_destination(prefect_destination)
        _PREFECT_LOGGING_SETUP = True
    except ImportError:
        # Prefect not installed, skip setup
        pass



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
    """Context manager to track execution time, CPU and peak memory usage.
    
    Automatically logs to:
    - Dagster: logs resource metrics as asset metadata
    - Prefect: logs resource metrics and creates markdown artifacts
    - Eliot: logs structured resource report
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

    # Log to Dagster if available (check this first as it's more specific)
    dagster_logged = False
    try:
        from dagster import get_dagster_logger, MetadataValue
        # This will fail if not in a dagster context
        logger = get_dagster_logger()
        logger.info(
            f"📊 Resource Report [{name}]: Duration: {report.duration:.2f}s, "
            f"CPU: {report.cpu_usage_percent:.1f}%, Peak RAM: {report.peak_memory_mb:.2f}MB"
        )
        dagster_logged = True
        
        # Try to add metadata to the current op run
        try:
            from dagster import get_dagster_context
            context = get_dagster_context()
            
            # Clean name for metadata key
            clean_key = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            if not clean_key:
                clean_key = "resource_usage"
            
            # Log metadata directly to the current op
            context.log.info(
                "resource_metrics",
                metadata={
                    f"{clean_key}_duration_sec": MetadataValue.float(round(report.duration, 2)),
                    f"{clean_key}_cpu_percent": MetadataValue.float(round(report.cpu_usage_percent, 1)),
                    f"{clean_key}_peak_memory_mb": MetadataValue.float(round(report.peak_memory_mb, 2)),
                    f"{clean_key}_memory_delta_mb": MetadataValue.float(round(report.memory_delta_mb, 2)),
                }
            )
        except Exception:
            # Context not available or metadata logging failed
            pass
            
    except ImportError:
        pass
    except Exception:
        # Not in a Dagster context
        pass

    # Log to Prefect if available and not already logged to Dagster
    if not dagster_logged:
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


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is in use on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def setup_prefect_api() -> tuple[bool, Optional[str]]:
    """Setup Prefect API connection if environment variables are provided.
    
    If no server is configured and it's not running locally, it attempts to
    start it in the background to provide the UI.
    
    Returns:
        Tuple of (is_server_mode, ui_url)
    """
    api_url = os.getenv("PREFECT_API_URL")
    if api_url:
        # Extract base URL for UI (remove /api suffix if present)
        ui_url = api_url.replace("/api", "")
        print(f"🚀 Prefect configured for server at: {api_url}")
        print(f"📊 Prefect UI: {ui_url}")
        return True, ui_url
    else:
        # Ephemeral mode - use local SQLite database
        ui_url = "http://127.0.0.1:4200"
        if not is_port_in_use("127.0.0.1", 4200):
            print("🌀 Prefect server not found. Starting it in the background to provide UI...")
            try:
                # Use start_new_session to ensure it survives the parent process
                subprocess.Popen(
                    ["prefect", "server", "start"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                # Give it a tiny bit of time to start the process
                time.sleep(0.5)
                print(f"📊 Prefect UI will be available at: {ui_url}")
            except Exception as e:
                print(f"⚠️ Could not start Prefect server automatically: {e}")
                print(f"📊 Prefect UI: {ui_url} (run 'prefect server start' manually)")
        else:
            print(f"🚀 Prefect UI is already running at: {ui_url}")
            
        return False, ui_url


@contextmanager
def prefect_flow_run(name: str, profile: bool = True):
    """Context manager for running a Prefect flow with optional resource tracking."""
    setup_prefect_logging()
    is_server, ui_url = setup_prefect_api()
    if profile:
        with resource_tracker(name) as tracker:
            tracker["prefect_ui_url"] = ui_url
            tracker["prefect_server_mode"] = is_server
            yield tracker
    else:
        yield {"prefect_ui_url": ui_url, "prefect_server_mode": is_server}


#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for genobear tests.
"""

import subprocess
import pytest
import tempfile
import shutil
import os
from pathlib import Path
import pooch
from prepare_annotations.resources import MODULES_DIR
from pycomfort.logging import to_nice_stdout


# =============================================================================
# OakVar Module Data Download Helpers
# =============================================================================

# Mapping of module names to their GitHub repositories
OAKVAR_MODULE_REPOS: dict[str, str] = {
    "just_longevitymap": "dna-seq/just_longevitymap",
    "just_pathogenic": "dna-seq/just_pathogenic",
    "just_cancer": "dna-seq/just_cancer",
    "just_coronary": "dna-seq/just_coronary",
    "just_vo2max": "dna-seq/just_vo2max",
    "just_lipidmetabolism": "dna-seq/just_lipidmetabolism",
    "just_prs": "dna-seq/just_prs",
    "just_drugs": "dna-seq/just_drugs",
    "just_superhuman": "dna-seq/just_superhuman",
}


def download_oakvar_module_data(
    module_name: str,
    output_dir: Path | None = None,
    extensions: list[str] | None = None,
) -> Path:
    """
    Download data files from an OakVar module repository.
    
    Uses the `modules data` CLI command to clone the repository and extract
    data files with the specified extensions.
    
    Args:
        module_name: Name of the module (e.g., "just_longevitymap")
        output_dir: Output directory for downloaded files.
                    Defaults to data/modules/{module_name}/
        extensions: File extensions to download (default: [".sqlite", ".sqlite3", ".db", ".tsv"])
    
    Returns:
        Path to the output directory
        
    Raises:
        ValueError: If module_name is not in OAKVAR_MODULE_REPOS
        subprocess.CalledProcessError: If download fails
    """
    if module_name not in OAKVAR_MODULE_REPOS:
        raise ValueError(
            f"Unknown module: {module_name}. "
            f"Known modules: {list(OAKVAR_MODULE_REPOS.keys())}"
        )
    
    repo = OAKVAR_MODULE_REPOS[module_name]
    
    if output_dir is None:
        output_dir = MODULES_DIR / module_name
    
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "uv", "run", "modules", "data",
        "--repo", repo,
        "--output-dir", str(output_dir),
    ]
    
    if extensions:
        for ext in extensions:
            cmd.extend(["--ext", ext])
    else:
        # Default extensions include TSV for drugs module
        for ext in [".sqlite", ".sqlite3", ".db", ".tsv"]:
            cmd.extend(["--ext", ext])
    
    subprocess.run(cmd, check=True, capture_output=False)
    
    return output_dir


def ensure_oakvar_module_data(
    module_name: str,
    output_dir: Path | None = None,
    expected_file: str | None = None,
) -> Path:
    """
    Ensure OakVar module data exists, downloading if necessary.
    
    Args:
        module_name: Name of the module (e.g., "just_longevitymap")
        output_dir: Output directory for downloaded files.
                    Defaults to data/modules/{module_name}/
        expected_file: Optional filename to check for existence.
                       If provided, checks if this file exists.
                       
    Returns:
        Path to the output directory
    """
    if output_dir is None:
        output_dir = MODULES_DIR / module_name
    
    # Check if data already exists
    needs_download = False
    if not output_dir.exists():
        needs_download = True
    elif expected_file:
        expected_path = output_dir / expected_file
        if not expected_path.exists():
            needs_download = True
    
    if needs_download:
        download_oakvar_module_data(module_name, output_dir)
    
    return output_dir


def pytest_addoption(parser):
    """Add CLI flags.

    By default we keep and reuse the shared pooch cache across test runs.
    Pass --no-shared-pooch-cache to force temporary, cleaned caches.
    """
    parser.addoption(
        "--no-shared-pooch-cache",
        action="store_true",
        default=False,
        help=(
            "Use temporary per-test pooch caches and clean them after tests. "
            "Defaults to False (shared pooch cache is reused)."
        ),
    )
    parser.addoption(
        "--clean-cache",
        action="store_true",
        default=False,
        help="Clean the cache directory before running tests"
    )


@pytest.fixture(scope="session")
def use_shared_pooch_cache(request) -> bool:
    """Whether to use the shared pooch cache across tests (default True)."""
    return not request.config.getoption("--no-shared-pooch-cache")


@pytest.fixture(scope="session")
def shared_pooch_cache_dir() -> Path:
    """Path to the shared pooch cache used by downloaders by default."""
    return Path(pooch.os_cache("ensembl_variation"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists across tests in a session."""
    temp_dir = tempfile.mkdtemp(prefix="genobear_test_data_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for a single test."""
    temp_dir = tempfile.mkdtemp(prefix="genobear_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "large_download: marks tests that download large files (multi-GB)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "download: marks tests that perform downloads"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their characteristics."""
    for item in items:
        # Mark tests with 'large' in name as potentially slow
        if 'large' in item.name.lower():
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def enable_eliot_stdout():
    """Ensure Eliot logs are pretty-printed to stdout during the test session."""
    to_nice_stdout()
    yield

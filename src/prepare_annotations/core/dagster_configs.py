"""
Dagster configuration classes for genomic data preparation pipelines.

Configs are Pydantic-based settings that can be passed to assets at runtime.
"""

import os
from typing import Optional

import psutil
from dagster import Config


def get_default_duckdb_memory_limit() -> str:
    """
    Calculate a sensible DuckDB memory limit based on available system memory.
    
    Strategy:
    - Use 60% of available RAM for DuckDB (conservative for large VCF processing)
    - Minimum: 4GB
    - Maximum: 64GB
    """
    available_gb = psutil.virtual_memory().available / (1024**3)
    duckdb_gb = int(available_gb * 0.6)
    duckdb_gb = max(4, min(duckdb_gb, 64))
    return f"{duckdb_gb}GB"


class EnsemblDownloadConfig(Config):
    """Configuration for downloading Ensembl VCF files."""
    
    species: str = "homo_sapiens"
    base_url: str = "https://ftp.ensembl.org/pub/current_variation/vcf/"
    pattern: Optional[str] = None  # Regex pattern for filtering files
    cache_dir: Optional[str] = None  # Override cache directory
    verify_checksums: bool = True
    http_max_pool: int = 20
    connect_timeout: float = 10.0
    sock_read_timeout: float = 120.0
    retries: int = 10
    force_download: bool = False
    max_concurrent_downloads: int = 4  # Maximum parallel downloads


class ParquetConversionConfig(Config):
    """Configuration for VCF to Parquet conversion."""
    
    compression: str = "zstd"
    compression_level: int = 14
    alts_list: bool = True  # Add list of alternative alleles as 'alts' column
    force_convert: bool = False
    max_concurrent_conversions: Optional[int] = None  # Auto-detect if None
    threads: Optional[int] = None  # Auto-detect if None

    def get_max_concurrent_conversions(self) -> int:
        """Get max concurrent conversions, using auto-detection if not explicitly set."""
        if self.max_concurrent_conversions:
            return self.max_concurrent_conversions
        env_value = os.getenv("PREPARE_ANNOTATIONS_PARQUET_WORKERS")
        if env_value:
            return max(1, int(env_value))
        # Since it's CPU-bound but Polars is multi-threaded, we don't want too many concurrent files
        # Default to 2 when no explicit value or env override is available
        return 2

    def get_threads(self) -> int:
        """Get thread count per conversion, using auto-detection if not explicitly set."""
        if self.threads:
            return self.threads
        cpu_count = psutil.cpu_count(logical=True) or 4
        # If we run 2 concurrent conversions, we can give each ~50% of CPUs
        # But Polars is good at sharing threads, so 0.75 of total is often fine too
        # as they won't both be at 100% all the time.
        return max(2, min(int(cpu_count * 0.5), 16))


class HuggingFaceUploadConfig(Config):
    """Configuration for HuggingFace Hub uploads."""
    
    repo_id: str = "just-dna-seq/ensembl_variations"
    token: Optional[str] = None  # Uses HF_TOKEN env var if None
    path_prefix: str = "data"
    # Pattern to find parquet files. Direct files only by default (no vcf/ or splitted_variants/).
    # Note: *.parquet matches immediate children only. For recursive, use **/*.parquet.
    pattern: str = "*.parquet"


class DuckDBConfig(Config):
    """DuckDB processing configuration for memory-efficient operations."""
    
    memory_limit: Optional[str] = None  # Auto-detect if None
    threads: Optional[int] = None  # Auto-detect if None
    temp_directory: str = "/tmp/duckdb_ensembl"
    preserve_insertion_order: bool = False
    
    def get_memory_limit(self) -> str:
        """Get memory limit, using auto-detection if not explicitly set."""
        return self.memory_limit or get_default_duckdb_memory_limit()
    
    def get_threads(self) -> int:
        """Get thread count, using auto-detection if not explicitly set."""
        if self.threads:
            return self.threads
        cpu_count = psutil.cpu_count(logical=True) or 4
        return max(2, min(int(cpu_count * 0.75), 16))


class EnsemblSourceConfig(Config):
    """Configuration for Ensembl variation data source."""
    
    # Path to local Ensembl cache. If None, tries default cache location.
    local_cache_path: Optional[str] = None
    # HuggingFace dataset repo for Ensembl data
    hf_repo: str = "just-dna-seq/ensembl_variations"
    # Species for file pattern matching
    species: str = "homo_sapiens"
    # If True, prefer local cache over HuggingFace
    prefer_local: bool = True


class LongevityMapSourceConfig(Config):
    """Configuration for downloading LongevityMap SQLite from GitHub."""
    
    # GitHub repository in owner/repo format
    github_repo: str = "dna-seq/just_longevitymap"
    # Path to the SQLite file in the repository
    file_path: str = "data/longevitymap.sqlite"
    # Branch or tag to download from
    ref: str = "master"
    # Force re-download even if file exists
    force_download: bool = False
    
    @property
    def download_url(self) -> str:
        """Get the raw GitHub download URL."""
        return f"https://github.com/{self.github_repo}/raw/{self.ref}/{self.file_path}"


class LongevityMapConfig(Config):
    """Configuration for LongevityMap module conversion."""
    
    # Module name in output
    module_name: str = "longevitymap"
    # Curator name
    curator: str = "Olga Borysova"
    # Curation method
    method: str = "literature_review"
    # Output directory. If None, uses default modules output.
    output_dir: Optional[str] = None


class AnnotatorsUploadConfig(Config):
    """Configuration for uploading annotator modules to HuggingFace Hub."""
    
    # HuggingFace repository ID for annotators
    repo_id: str = "just-dna-seq/annotators"
    # HuggingFace API token. Uses HF_TOKEN env var if None.
    token: Optional[str] = None
    # Path prefix in the repo (module folders go under this)
    path_prefix: str = "data"

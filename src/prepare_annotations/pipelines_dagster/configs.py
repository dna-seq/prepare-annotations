"""
Dagster configuration classes for genomic data preparation pipelines.

Configs are Pydantic-based settings that can be passed to assets at runtime.
"""

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

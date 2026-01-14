"""
Path helpers and resource utilities for Dagster pipelines.

These provide shared utility functions for determining cache, input, 
and output directories. They mirror the patterns from just-dna-lite.
"""

import os
from pathlib import Path

from platformdirs import user_cache_dir


def get_cache_dir() -> Path:
    """Get the root cache directory for all annotations.
    
    Priority:
    1. JUST_DNA_PIPELINES_CACHE_DIR environment variable
    2. System-specific user cache directory (~/.cache/just-dna-pipelines)
    """
    env_cache = os.getenv("JUST_DNA_PIPELINES_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return Path(user_cache_dir(appname="just-dna-pipelines"))


def get_default_ensembl_cache_dir(species: str = "homo_sapiens") -> Path:
    """Get the default cache directory for Ensembl VCF data."""
    return get_cache_dir() / "ensembl" / species


def get_ensembl_vcf_dir(species: str = "homo_sapiens") -> Path:
    """Get the VCF download directory for Ensembl."""
    return get_default_ensembl_cache_dir(species) / "vcf"


def get_ensembl_parquet_dir(species: str = "homo_sapiens") -> Path:
    """Get the Parquet conversion directory for Ensembl."""
    return get_default_ensembl_cache_dir(species)


def get_ensembl_split_dir(species: str = "homo_sapiens") -> Path:
    """Get the split variants directory for Ensembl."""
    return get_default_ensembl_cache_dir(species) / "splitted_variants"


def get_ensembl_rsid_coords_path(species: str = "homo_sapiens") -> Path:
    """Get the rsID coordinates output path."""
    return get_default_ensembl_cache_dir(species) / "rsid_coordinates.parquet"


def get_ensembl_species_url(
    species: str = "homo_sapiens",
    base_url: str = "https://ftp.ensembl.org/pub/current_variation/vcf/"
) -> str:
    """Get the full URL for a species."""
    return f"{base_url}{species}/"


def get_ensembl_vcf_pattern(species: str = "homo_sapiens") -> str:
    """Get the regex pattern for VCF files."""
    return rf"{species}-chr([^.]+)\.vcf\.gz$"


def get_output_dir() -> Path:
    """Get the output directory for uploads and final data."""
    env_output = os.getenv("JUST_DNA_PIPELINES_OUTPUT_DIR")
    if env_output:
        return Path(env_output)
    return Path("data") / "output"

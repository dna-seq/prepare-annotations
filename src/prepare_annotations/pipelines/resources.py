"""
Path helpers and resource utilities for Dagster pipelines.

These provide shared utility functions for determining cache, input, 
and output directories. They mirror the patterns from just-dna-lite.
"""

import os
from pathlib import Path

from prepare_annotations.resources import get_cache_dir, get_default_ensembl_cache_dir, get_output_dir


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


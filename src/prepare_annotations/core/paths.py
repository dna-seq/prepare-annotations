"""
Path helpers and resource utilities for prepare-annotations.

Mirrors the patterns from just-dna-lite for consistency across repositories.
"""
import os
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir


# Root of the project (where pyproject.toml usually is)
# Assuming this file is at src/prepare_annotations/core/paths.py
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

# Project-relative directories (local to the repository)
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "output"

MODULES_DIR = DATA_DIR / "modules"
MODULES_OUTPUT_DIR = OUTPUT_DIR / "modules"

LOGS_DIR = ROOT_DIR / "logs"
TEMPORARY_DIR = ROOT_DIR / "tmp"


def get_cache_dir() -> Path:
    """
    Get the root cache directory for all annotations.

    Priority:
        1. JUST_DNA_PIPELINES_CACHE_DIR environment variable
        2. System-specific user cache directory (e.g., ~/.cache/just-dna-pipelines)

    Returns:
        Path to the cache directory
    """
    env_cache = os.getenv("JUST_DNA_PIPELINES_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return Path(user_cache_dir(appname="just-dna-pipelines"))


CACHE_DIR = get_cache_dir()

# Common cache subdirectories
ENSEMBL_CACHE_DIR = CACHE_DIR / "ensembl"
VCF_CACHE_DIR = CACHE_DIR / "vcf"


def get_ensembl_cache(species: str = "homo_sapiens") -> Path:
    """
    Get the Ensembl cache directory for a species.

    Structure: ~/.cache/just-dna-pipelines/ensembl/{species}/
    """
    return ENSEMBL_CACHE_DIR / species


def get_ensembl_variations_cache(species: str = "homo_sapiens") -> Path:
    """
    Get the Ensembl variations (parquet) cache directory for a species.

    This is where per-chromosome parquet files are stored after VCF conversion.
    Files are named: {species}-chr{N}.parquet
    """
    return get_ensembl_cache(species)


def get_ensembl_genome_cache(species: str = "homo_sapiens") -> Path:
    """
    Get the Ensembl genome (FASTA) cache directory for a species.

    Structure: ~/.cache/just-dna-pipelines/ensembl/{species}/fasta/dna/
    """
    return get_ensembl_cache(species) / "fasta" / "dna"


def find_ensembl_genome_fasta(
    species: str = "homo_sapiens",
    genome_type: str = "primary_assembly",
    chromosome: Optional[str] = None,
) -> Optional[Path]:
    """
    Find a downloaded Ensembl genome FASTA file.
    """
    genome_dir = get_ensembl_genome_cache(species)
    if not genome_dir.exists():
        return None

    if genome_type == "chromosome" or chromosome is not None:
        if chromosome is None:
            raise ValueError("chromosome argument required when genome_type='chromosome'")
        # Pattern: *.dna.chromosome.{chr}.fa.gz
        pattern = f"*.dna.chromosome.{chromosome}.fa.gz"
    elif genome_type == "primary_assembly":
        pattern = "*.dna.primary_assembly.fa.gz"
    elif genome_type == "toplevel":
        pattern = "*.dna.toplevel.fa.gz"
    else:
        raise ValueError(f"Unknown genome_type: {genome_type}")

    matches = list(genome_dir.glob(pattern))
    return matches[0] if matches else None


def list_ensembl_variation_parquets(species: str = "homo_sapiens") -> list[Path]:
    """
    List all Ensembl variation parquet files for a species.
    """
    cache = get_ensembl_variations_cache(species)
    if not cache.exists():
        return []
    return sorted(cache.glob(f"{species}-chr*.parquet"))


def list_ensembl_genome_fastas(species: str = "homo_sapiens") -> list[Path]:
    """
    List all downloaded Ensembl genome FASTA files for a species.
    """
    genome_dir = get_ensembl_genome_cache(species)
    if not genome_dir.exists():
        return []
    return sorted(genome_dir.glob("*.fa.gz"))


def get_default_cache_dir(name: str) -> Path:
    """
    Get the default cache directory for a data source.
    Ensures the directory exists.
    """
    path = CACHE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir() -> Path:
    """Get the output directory for uploads and final data."""
    env_output = os.getenv("JUST_DNA_PIPELINES_OUTPUT_DIR")
    if env_output:
        return Path(env_output)
    return OUTPUT_DIR


# Alias for compatibility with just-dna-lite naming convention
def get_default_ensembl_cache_dir(species: str = "homo_sapiens") -> Path:
    """Get the default cache directory for Ensembl variation data."""
    return get_ensembl_variations_cache(species)


# Additional path helpers for Dagster pipelines
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

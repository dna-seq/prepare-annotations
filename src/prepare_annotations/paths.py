"""
Path utilities and cache directory helpers for prepare-annotations.

Cache Structure:
    ~/.cache/just-dna-pipelines/
    ├── ensembl/
    │   └── {species}/
    │       ├── *.parquet           # Ensembl variation parquets (per chromosome)
    │       ├── vcf/                # Downloaded VCF files
    │       ├── fasta/
    │       │   └── dna/            # Genome FASTA files
    │       │       ├── *.primary_assembly.fa.gz
    │       │       └── *.chromosome.*.fa.gz
    │       └── splitted_variants/  # Split variant files
    ├── clinvar/
    ├── dbsnp/
    └── vcf/
"""

from pathlib import Path
import os
from typing import Optional

from platformdirs import user_cache_dir


# Root of the project (where pyproject.toml usually is)
# Assuming this file is at src/prepare_annotations/paths.py
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

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


# =============================================================================
# Ensembl Cache Helpers
# =============================================================================


def get_ensembl_cache(species: str = "homo_sapiens") -> Path:
    """
    Get the Ensembl cache directory for a species.
    
    Structure: ~/.cache/just-dna-pipelines/ensembl/{species}/
    
    Args:
        species: Species name (e.g., "homo_sapiens", "mus_musculus")
        
    Returns:
        Path to the species cache directory
    """
    return ENSEMBL_CACHE_DIR / species


def get_ensembl_variations_cache(species: str = "homo_sapiens") -> Path:
    """
    Get the Ensembl variations (parquet) cache directory for a species.
    
    This is where per-chromosome parquet files are stored after VCF conversion.
    Files are named: {species}-chr{N}.parquet
    
    Args:
        species: Species name
        
    Returns:
        Path to the variations cache (same as species cache root)
    """
    return get_ensembl_cache(species)


def get_ensembl_genome_cache(species: str = "homo_sapiens") -> Path:
    """
    Get the Ensembl genome (FASTA) cache directory for a species.
    
    Structure: ~/.cache/just-dna-pipelines/ensembl/{species}/fasta/dna/
    
    Files include:
        - {Species}.{Assembly}.dna.primary_assembly.fa.gz
        - {Species}.{Assembly}.dna.chromosome.{N}.fa.gz
    
    Args:
        species: Species name
        
    Returns:
        Path to the genome FASTA directory
    """
    return get_ensembl_cache(species) / "fasta" / "dna"


def find_ensembl_genome_fasta(
    species: str = "homo_sapiens",
    genome_type: str = "primary_assembly",
    chromosome: Optional[str] = None,
) -> Optional[Path]:
    """
    Find a downloaded Ensembl genome FASTA file.
    
    Args:
        species: Species name
        genome_type: Type of genome file ("primary_assembly", "toplevel", or "chromosome")
        chromosome: Chromosome name when genome_type is "chromosome" (e.g., "21", "X")
        
    Returns:
        Path to the FASTA file if found, None otherwise
        
    Examples:
        >>> find_ensembl_genome_fasta()  # Primary assembly
        >>> find_ensembl_genome_fasta(chromosome="21")  # Chromosome 21
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
    
    Args:
        species: Species name
        
    Returns:
        Sorted list of parquet file paths (e.g., homo_sapiens-chr1.parquet, ...)
    """
    cache = get_ensembl_variations_cache(species)
    if not cache.exists():
        return []
    return sorted(cache.glob(f"{species}-chr*.parquet"))


def list_ensembl_genome_fastas(species: str = "homo_sapiens") -> list[Path]:
    """
    List all downloaded Ensembl genome FASTA files for a species.
    
    Args:
        species: Species name
        
    Returns:
        Sorted list of FASTA file paths
    """
    genome_dir = get_ensembl_genome_cache(species)
    if not genome_dir.exists():
        return []
    return sorted(genome_dir.glob("*.fa.gz"))


# =============================================================================
# Generic Cache Helpers
# =============================================================================


def get_default_cache_dir(name: str) -> Path:
    """
    Get the default cache directory for a data source.
    Ensures the directory exists.
    
    Args:
        name: Subdirectory name (e.g., "clinvar", "dbsnp")
        
    Returns:
        Path to the cache subdirectory
    """
    path = CACHE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_input_dir(name: str) -> Path:
    """
    Get the default destination directory for downloads.
    
    Note: For backwards compatibility. New code should use get_default_cache_dir.
    Downloads are stored directly in the cache folder now.
    """
    return get_default_cache_dir(name)


def get_default_interim_dir(name: str) -> Path:
    """
    Get the default directory for intermediate files.
    
    Note: For backwards compatibility. Intermediate parquet files are stored
    directly in the cache folder now.
    """
    return get_default_cache_dir(name)


def get_default_output_dir(name: str) -> Path:
    """
    Get the default directory for final output files.
    
    Note: For backwards compatibility. Output files (like splitted_variants)
    are stored directly in the cache folder now.
    """
    return get_default_cache_dir(name)

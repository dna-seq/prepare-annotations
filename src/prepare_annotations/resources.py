"""
Resource utilities and path helpers.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.core.paths
"""
# Re-export from core for backward compatibility
from prepare_annotations.core.paths import (
    ROOT_DIR,
    DATA_DIR,
    INPUT_DIR,
    INTERIM_DIR,
    OUTPUT_DIR,
    MODULES_DIR,
    MODULES_OUTPUT_DIR,
    LOGS_DIR,
    TEMPORARY_DIR,
    CACHE_DIR,
    ENSEMBL_CACHE_DIR,
    VCF_CACHE_DIR,
    get_cache_dir,
    get_ensembl_cache,
    get_ensembl_variations_cache,
    get_ensembl_genome_cache,
    find_ensembl_genome_fasta,
    list_ensembl_variation_parquets,
    list_ensembl_genome_fastas,
    get_default_cache_dir,
    get_default_input_dir,
    get_default_interim_dir,
    get_default_output_dir,
    get_output_dir,
    get_default_ensembl_cache_dir,
    get_ensembl_vcf_dir,
    get_ensembl_parquet_dir,
    get_ensembl_split_dir,
    get_ensembl_rsid_coords_path,
    get_ensembl_species_url,
    get_ensembl_vcf_pattern,
)

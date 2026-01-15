"""
Ensembl genome FASTA downloader.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.downloaders.genome
"""
# Re-export from downloaders for backward compatibility
from prepare_annotations.downloaders.genome import (
    GenomeType,
    MaskingType,
    ENSEMBL_FTP_BASE,
    ENSEMBL_HTTP_BASE,
    FASTA_COPY_BUFFER_BYTES,
    gunzip_to_fasta,
    write_fai_for_fasta,
    ensure_uncompressed_fasta_with_fai,
    get_default_ensembl_cache_dir,
    get_default_genome_cache_dir,
    get_ensembl_fasta_url,
    find_genome_file,
    download_ensembl_genome,
    list_available_genomes,
    download_all_chromosomes,
)

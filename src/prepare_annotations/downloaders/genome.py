"""
Ensembl genome FASTA downloader.

Downloads reference genomes from the Ensembl FTP server.

FTP structure:
- Base: ftp://ftp.ensembl.org/pub/
- Current release: ftp://ftp.ensembl.org/pub/current_fasta/{species}/dna/
- Specific release: ftp://ftp.ensembl.org/pub/release-{N}/fasta/{species}/dna/

File naming conventions:
- Primary assembly: {Species}.{Assembly}.dna.primary_assembly.fa.gz
- Toplevel (all sequences): {Species}.{Assembly}.dna.toplevel.fa.gz
- Per-chromosome: {Species}.{Assembly}.dna.chromosome.{chr}.fa.gz
- Softmasked: {Species}.{Assembly}.dna_sm.*.fa.gz
- Repeat masked: {Species}.{Assembly}.dna_rm.*.fa.gz
"""

from enum import Enum
import gzip
import shutil
from pathlib import Path
from typing import Optional

from eliot import start_action
from platformdirs import user_cache_dir

from prepare_annotations.downloaders.vcf import download_path, list_paths


# Ensembl FTP URLs
ENSEMBL_FTP_BASE = "ftp://ftp.ensembl.org/pub"
ENSEMBL_HTTP_BASE = "https://ftp.ensembl.org/pub"  # HTTP mirror (often faster)


class GenomeType(str, Enum):
    """Type of genome sequence to download."""
    
    PRIMARY_ASSEMBLY = "primary_assembly"  # Main chromosomes without alt/patch sequences
    TOPLEVEL = "toplevel"  # All sequences including alts and patches
    CHROMOSOME = "chromosome"  # Individual chromosome files


class MaskingType(str, Enum):
    """DNA masking type."""
    
    UNMASKED = "dna"  # Unmasked genomic DNA
    SOFTMASKED = "dna_sm"  # Soft-masked (repeats in lowercase)
    REPEATMASKED = "dna_rm"  # Repeat-masked (repeats as N)


FASTA_COPY_BUFFER_BYTES = 8 * 1024 * 1024  # 8MB


def gunzip_to_fasta(
    fasta_gz_path: Path,
    fasta_path: Path,
    *,
    overwrite: bool,
) -> Path:
    """Decompress a gzipped FASTA to a plain FASTA file (streaming)."""
    fasta_gz_path = Path(fasta_gz_path)
    fasta_path = Path(fasta_path)

    if fasta_path.exists() and not overwrite:
        return fasta_path

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fasta_gz_path, "rb") as src, fasta_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=FASTA_COPY_BUFFER_BYTES)
    return fasta_path


def write_fai_for_fasta(
    fasta_path: Path,
    fai_path: Optional[Path] = None,
    *,
    overwrite: bool,
) -> Path:
    """
    Create a FASTA index (.fai) for an uncompressed FASTA file.

    The index format matches htslib's faidx (seqname, length, offset, linebases, linewidth).
    """
    fasta_path = Path(fasta_path)
    if fai_path is None:
        fai_path = Path(str(fasta_path) + ".fai")
    else:
        fai_path = Path(fai_path)

    if fai_path.exists() and not overwrite:
        return fai_path

    fai_path.parent.mkdir(parents=True, exist_ok=True)

    current_name: str | None = None
    current_length = 0
    current_offset = 0
    current_line_bases = 0
    current_line_width = 0
    saw_sequence_line = False

    with fasta_path.open("rb") as f, fai_path.open("w", encoding="utf-8") as out:
        while True:
            line_offset = f.tell()
            line = f.readline()
            if not line:
                break

            if line.startswith(b">"):
                if current_name is not None:
                    out.write(
                        f"{current_name}\t{current_length}\t{current_offset}\t"
                        f"{current_line_bases}\t{current_line_width}\n"
                    )
                header = line[1:].strip()
                current_name = header.decode("utf-8").split()[0]
                current_length = 0
                current_offset = 0
                current_line_bases = 0
                current_line_width = 0
                saw_sequence_line = False
                continue

            if current_name is None:
                continue

            if line in (b"\n", b"\r\n"):
                continue

            stripped = line.rstrip(b"\r\n")
            if not saw_sequence_line:
                current_offset = line_offset
                current_line_bases = len(stripped)
                current_line_width = len(line)
                saw_sequence_line = True

            current_length += len(stripped)

        if current_name is not None:
            out.write(
                f"{current_name}\t{current_length}\t{current_offset}\t"
                f"{current_line_bases}\t{current_line_width}\n"
            )

    return fai_path


def ensure_uncompressed_fasta_with_fai(
    fasta_gz_path: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """
    Ensure an uncompressed FASTA and its .fai exist next to a downloaded *.fa.gz.

    Returns:
        (fasta_path, fai_path)
    """
    fasta_gz_path = Path(fasta_gz_path)
    if fasta_gz_path.suffix != ".gz":
        fasta_path = fasta_gz_path
    else:
        fasta_path = fasta_gz_path.with_suffix("")

    if fasta_gz_path.suffix == ".gz":
        gunzip_to_fasta(fasta_gz_path, fasta_path, overwrite=overwrite)

    fai_path = write_fai_for_fasta(fasta_path, overwrite=overwrite)
    return fasta_path, fai_path


def get_default_ensembl_cache_dir() -> Path:
    """Get the default cache directory for Ensembl data."""
    return Path(user_cache_dir(appname="just-dna-pipelines")) / "ensembl"


def get_default_genome_cache_dir(species: str = "homo_sapiens") -> Path:
    """Get the default cache directory for genome downloads.
    
    Uses consistent structure: .cache/just-dna-pipelines/ensembl/{species}/fasta/dna/
    This mirrors the Ensembl FTP structure.
    """
    return get_default_ensembl_cache_dir() / species / "fasta" / "dna"


def get_ensembl_fasta_url(
    species: str = "homo_sapiens",
    release: Optional[int] = None,
    use_http: bool = True,
) -> str:
    """
    Construct the Ensembl FASTA directory URL.
    
    Args:
        species: Species name (lowercase with underscore, e.g., "homo_sapiens")
        release: Ensembl release number. If None, uses "current_fasta" for latest.
        use_http: Use HTTP mirror instead of FTP (usually faster)
        
    Returns:
        URL to the FASTA directory
    """
    base = ENSEMBL_HTTP_BASE if use_http else ENSEMBL_FTP_BASE
    
    if release is None:
        return f"{base}/current_fasta/{species}/dna/"
    else:
        return f"{base}/release-{release}/fasta/{species}/dna/"


def find_genome_file(
    species: str = "homo_sapiens",
    genome_type: GenomeType = GenomeType.PRIMARY_ASSEMBLY,
    masking: MaskingType = MaskingType.UNMASKED,
    release: Optional[int] = None,
    chromosome: Optional[str] = None,
    use_http: bool = True,
) -> str:
    """
    Find the URL for a specific genome file on Ensembl FTP.
    
    Args:
        species: Species name (e.g., "homo_sapiens")
        genome_type: Type of genome assembly to download
        masking: DNA masking type
        release: Ensembl release number (None for latest)
        chromosome: Chromosome number/name for CHROMOSOME type (e.g., "1", "X", "MT")
        use_http: Use HTTP mirror instead of FTP
        
    Returns:
        URL to the genome FASTA file
        
    Raises:
        ValueError: If no matching file is found
    """
    with start_action(
        action_type="find_genome_file",
        species=species,
        genome_type=genome_type.value,
        masking=masking.value,
        release=release,
        chromosome=chromosome,
    ) as action:
        base_url = get_ensembl_fasta_url(species, release, use_http)
        
        masking_str = masking.value
        
        if genome_type == GenomeType.CHROMOSOME:
            if chromosome is None:
                raise ValueError("chromosome argument required for GenomeType.CHROMOSOME")
            pattern = rf"\.{masking_str}\.chromosome\.{chromosome}\.fa\.gz$"
        elif genome_type == GenomeType.PRIMARY_ASSEMBLY:
            pattern = rf"\.{masking_str}\.primary_assembly\.fa\.gz$"
        elif genome_type == GenomeType.TOPLEVEL:
            pattern = rf"\.{masking_str}\.toplevel\.fa\.gz$"
        else:
            raise ValueError(f"Unknown genome type: {genome_type}")
        
        files = list_paths(base_url, pattern=pattern)
        
        if not files:
            action.log(
                message_type="error",
                step="no_matching_file",
                base_url=base_url,
                pattern=pattern,
            )
            raise ValueError(
                f"No genome file found matching pattern '{pattern}' at {base_url}"
            )
        
        if len(files) > 1:
            action.log(
                message_type="warning",
                step="multiple_matches",
                files=files,
            )
        
        file_url = files[0]
        action.log(message_type="info", step="found_file", url=file_url)
        return file_url


def download_ensembl_genome(
    species: str = "homo_sapiens",
    genome_type: GenomeType = GenomeType.PRIMARY_ASSEMBLY,
    masking: MaskingType = MaskingType.UNMASKED,
    release: Optional[int] = None,
    chromosome: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    create_fai: bool = True,
    use_http: bool = True,
) -> Path:
    """
    Download an Ensembl genome FASTA file.
    
    Args:
        species: Species name (e.g., "homo_sapiens", "mus_musculus")
        genome_type: Type of assembly (PRIMARY_ASSEMBLY, TOPLEVEL, or CHROMOSOME)
        masking: DNA masking type (UNMASKED, SOFTMASKED, REPEATMASKED)
        release: Ensembl release number (None for latest)
        chromosome: Chromosome name for CHROMOSOME type
        cache_dir: Local directory to store the downloaded file
        force_download: Whether to force download even if cache exists
        use_http: Use HTTP mirror instead of FTP (usually faster)
        
    Returns:
        Path to the downloaded genome FASTA file
    """
    if cache_dir is None:
        cache_dir = get_default_genome_cache_dir(species)
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(
        action_type="download_ensembl_genome",
        species=species,
        genome_type=genome_type.value,
        masking=masking.value,
        release=release,
        chromosome=chromosome,
        cache_dir=str(cache_dir),
    ) as action:
        file_url = find_genome_file(
            species=species,
            genome_type=genome_type,
            masking=masking,
            release=release,
            chromosome=chromosome,
            use_http=use_http,
        )
        
        filename = file_url.rsplit("/", 1)[-1]
        local_path = cache_dir / filename
        
        if local_path.exists() and not force_download:
            action.log(
                message_type="info",
                step="using_cache",
                path=str(local_path),
            )
            return local_path
        
        action.log(message_type="info", step="downloading", url=file_url)
        
        downloaded = download_path(
            url=file_url,
            dest_dir=cache_dir,
            check_files=True,
            resume=True,
        )
        
        if create_fai:
            fasta_path, fai_path = ensure_uncompressed_fasta_with_fai(
                downloaded, overwrite=force_download
            )
            action.log(
                message_type="info",
                step="fasta_index_ready",
                fasta_path=str(fasta_path),
                fai_path=str(fai_path),
            )

        action.log(
            message_type="info",
            step="download_complete",
            path=str(downloaded),
        )
        return downloaded


def list_available_genomes(
    species: str = "homo_sapiens",
    release: Optional[int] = None,
    use_http: bool = True,
) -> list[str]:
    """
    List all available genome FASTA files for a species.
    
    Args:
        species: Species name
        release: Ensembl release number (None for latest)
        use_http: Use HTTP mirror
        
    Returns:
        List of available file URLs
    """
    base_url = get_ensembl_fasta_url(species, release, use_http)
    
    with start_action(
        action_type="list_available_genomes",
        species=species,
        release=release,
        base_url=base_url,
    ) as action:
        files = list_paths(base_url, pattern=r"\.fa\.gz$")
        action.log(message_type="info", step="found_files", count=len(files))
        return files


def download_all_chromosomes(
    species: str = "homo_sapiens",
    masking: MaskingType = MaskingType.UNMASKED,
    release: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    create_fai: bool = True,
    use_http: bool = True,
    chromosomes: Optional[list[str]] = None,
) -> list[Path]:
    """
    Download all individual chromosome FASTA files.
    
    Args:
        species: Species name
        masking: DNA masking type
        release: Ensembl release number (None for latest)
        cache_dir: Local directory for downloads
        force_download: Whether to force re-download
        use_http: Use HTTP mirror
        chromosomes: Specific chromosomes to download (e.g., ["1", "2", "X"]).
                    If None, downloads all available chromosomes.
        
    Returns:
        List of paths to downloaded chromosome files
    """
    if cache_dir is None:
        cache_dir = get_default_genome_cache_dir(species) / "chromosomes"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(
        action_type="download_all_chromosomes",
        species=species,
        masking=masking.value,
        release=release,
        cache_dir=str(cache_dir),
    ) as action:
        base_url = get_ensembl_fasta_url(species, release, use_http)
        
        masking_str = masking.value
        pattern = rf"\.{masking_str}\.chromosome\.\w+\.fa\.gz$"
        files = list_paths(base_url, pattern=pattern)
        
        if chromosomes is not None:
            chr_set = set(chromosomes)
            filtered = []
            for f in files:
                parts = f.rsplit("/", 1)[-1].split(".")
                for i, part in enumerate(parts):
                    if part == "chromosome" and i + 1 < len(parts):
                        chr_name = parts[i + 1]
                        if chr_name in chr_set:
                            filtered.append(f)
                        break
            files = filtered
        
        action.log(
            message_type="info",
            step="found_chromosome_files",
            count=len(files),
        )
        
        downloaded: list[Path] = []
        for file_url in files:
            filename = file_url.rsplit("/", 1)[-1]
            local_path = cache_dir / filename
            
            if local_path.exists() and not force_download:
                action.log(
                    message_type="info",
                    step="using_cache",
                    file=filename,
                )
                downloaded.append(local_path)
            else:
                action.log(message_type="info", step="downloading", file=filename)
                path = download_path(
                    url=file_url,
                    dest_dir=cache_dir,
                    check_files=True,
                    resume=True,
                )
                downloaded.append(path)

            if create_fai:
                fasta_path, fai_path = ensure_uncompressed_fasta_with_fai(
                    downloaded[-1], overwrite=force_download
                )
                action.log(
                    message_type="info",
                    step="chromosome_fasta_index_ready",
                    file=downloaded[-1].name,
                    fasta_path=str(fasta_path),
                    fai_path=str(fai_path),
                )
        
        action.log(
            message_type="info",
            step="all_chromosomes_complete",
            total=len(downloaded),
        )
        return downloaded

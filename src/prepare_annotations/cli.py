"""
Prepare Annotations CLI - Modern pipeline-based data preparation.

This module provides a CLI interface using the Pipelines class for better
parallelization, caching, and pipeline composition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from eliot import start_action
from prepare_annotations.preparation.huggingface_uploader import collect_parquet_files
from prepare_annotations.preparation.dataset_card_generator import (
    generate_ensembl_card, 
    generate_clinvar_card,
    generate_dbsnp_card,
    generate_dbsnp_t2t_card,
    generate_gnomad_card,
)
from huggingface_hub import HfApi

from prepare_annotations.runtime import load_env

logs = Path("logs") if Path("logs").exists() else Path.cwd() / "logs"

load_env()

# Set POLARS_VERBOSE from env if not already set (default: 0 for clean output)
if "POLARS_VERBOSE" not in os.environ:
    os.environ["POLARS_VERBOSE"] = "0"

# Set POLARS_ENGINE_AFFINITY to streaming by default for memory efficiency
if "POLARS_ENGINE_AFFINITY" not in os.environ:
    os.environ["POLARS_ENGINE_AFFINITY"] = "streaming"

# Set POLARS_LOW_MEMORY to enable low memory mode by default
if "POLARS_LOW_MEMORY" not in os.environ:
    os.environ["POLARS_LOW_MEMORY"] = "1"

from prepare_annotations.preparation.pipelines import (
    PreparationPipelines,
    get_default_input_dir,
    get_default_output_dir,
)
from pycomfort.logging import to_nice_file, to_nice_stdout

# Create the main CLI app
app = typer.Typer(
    name="prepare-annotations",
    help="Modern Genomic Data Pipeline Tool (using Pipelines class)",
    rich_markup_mode="rich",
    no_args_is_help=True
)

console = Console()


def run_pipeline(
    name: str,
    pipeline_func,
    log: bool = True,
    upload: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    dest_dir: Optional[str] = None,
    split: bool = False,
    **pipeline_kwargs
):
    """Helper to run a preparation pipeline with consistent logging and upload handling."""
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / f"prepare_{name}.json", logs / f"prepare_{name}.log")
        to_nice_stdout()

    # Show Prefect UI information upfront
    from prepare_annotations.runtime import setup_prefect_api
    is_server, ui_url = setup_prefect_api()
    console.print(f"\n[bold cyan]🌊 Prefect UI:[/bold cyan] [link={ui_url}]{ui_url}[/link]")
    if not is_server:
        console.print("[yellow]💡 Tip: Run 'prefect server start' in another terminal to access the UI[/yellow]\n")
    else:
        console.print("[green]✅ Connected to Prefect server[/green]\n")

    with start_action(action_type=f"prepare_{name}_command") as action:
        action.log(
            message_type="info",
            dest_dir=dest_dir,
            split=split,
            upload=upload,
            **pipeline_kwargs
        )
        
        effective_dest = dest_dir if dest_dir else get_default_input_dir(name)
        console.print(f"📁 Destination: [bold blue]{effective_dest}[/bold blue]")
        console.print(f"🔄 Splitting: [bold blue]{split}[/bold blue]")
        
        console.print("🚀 Starting pipeline execution...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task_id = progress.add_task("Running pipeline...", total=None)
            
            results = pipeline_func(
                dest_dir=Path(dest_dir) if dest_dir else None,
                with_splitting=split,
                log=log,
                **pipeline_kwargs
            )
            
            progress.update(task_id, description="✅ Pipeline completed")
            
        console.print("\n✅ Pipeline execution completed!")
        console.print(f"[dim]View details at: {ui_url}[/dim]")
        
        if results.vcf_parquet_path:
            console.print(f"📦 Converted {len(results.vcf_parquet_path)} parquet files")
        
        if results.split_variants_dict:
            console.print(f"🔀 Split variants into {len(results.split_variants_dict)} categories")
            
        action.log(message_type="success", result_keys=list(results.model_dump().keys()))
        
        if upload:
            console.print(f"\n🔄 Starting upload to Hugging Face...")
            console.print(f"📦 Repository: [bold cyan]{repo_id}[/bold cyan]")
            
            upload_source_dir = Path(dest_dir) if dest_dir else get_default_output_dir(name)
            if split:
                upload_source_dir = upload_source_dir / "splitted_variants"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task_id = progress.add_task("Uploading files...", total=None)
                
                upload_results = PreparationPipelines.upload_dataset_to_hf(
                    dataset_type=name,
                    source_dir=upload_source_dir,
                    repo_id=repo_id,
                    token=token,
                    log=log,
                )
                
                progress.update(task_id, description="✅ Upload completed")
            
            console.print(f"\n📊 Upload Summary:")
            console.print(f"  - Total files: [bold]{len(upload_results.uploaded_files)}[/bold]")
            console.print(f"  - Uploaded: [bold green]{upload_results.num_uploaded}[/bold green]")
            console.print(f"  - Skipped: [bold yellow]{upload_results.num_skipped}[/bold yellow]")


@app.command()
def ensembl(
    species: str = typer.Option(
        "homo_sapiens",
        "--species",
        help="Species name to download (e.g., homo_sapiens, mus_musculus). Default: homo_sapiens"
    ),
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Base destination directory. If not specified, uses standard cache directory with species subfolder."
    ),
    vcf_dir: Optional[str] = typer.Option(
        None,
        "--vcf-dir",
        help="Optional specific directory for VCF downloads. Defaults to <dest_dir>/<species>/vcf"
    ),
    parquet_dir: Optional[str] = typer.Option(
        None,
        "--parquet-dir",
        help="Optional specific directory for Parquet files. Defaults to <dest_dir>/<species> (root of species folder)"
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files by variant type (TSA)"
    ),
    explode_snv_alt: bool = typer.Option(
        False,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants during splitting"
    ),
    alts_list: bool = typer.Option(
        True,
        "--alts-list/--no-alts-list",
        help="Add a list of alternative alleles as 'alts' column"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
    pattern: Optional[str] = typer.Option(
        None,
        "--pattern",
        help="Regex pattern to filter files. Examples: 'chr(21|22)' for chr21&22, 'chr2[12]' for chr21&22, 'chr(X|Y)' for sex chromosomes. Default: all chromosomes"
    ),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        help="Base URL for Ensembl data (default: https://ftp.ensembl.org/pub/current_variation/vcf/homo_sapiens/)"
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="Upload parquet files to Hugging Face Hub after processing"
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/ensembl_variations",
        "--repo-id",
        help="Hugging Face repository ID for upload"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token (uses HF_TOKEN env var if not provided)"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track and display resource usage (time and memory)"
    ),
    http_max_pool: Optional[int] = typer.Option(
        None,
        "--http-max-pool",
        help="HTTP connection pool size (for API consistency; actual pooling managed by fsspec). Mainly useful for timeout and retry tuning."
    ),
    http_chunk_size: Optional[int] = typer.Option(
        None,
        "--http-chunk-size",
        help="HTTP chunk size in bytes for reading data. Larger values may improve throughput."
    ),
    connect_timeout: Optional[float] = typer.Option(
        None,
        "--connect-timeout",
        help="Connection timeout in seconds (default: 10). Increase for slow/unreliable networks."
    ),
    sock_read_timeout: Optional[float] = typer.Option(
        None,
        "--sock-read-timeout",
        help="Socket read timeout in seconds (default: 120). Increase for very large files or slow connections."
    ),
    retries: Optional[int] = typer.Option(
        None,
        "--retries",
        help="Number of retry attempts for failed downloads (default: 10). Increase for unreliable networks."
    ),
    verify_checksums: bool = typer.Option(
        True,
        "--verify-checksums/--no-verify-checksums",
        help="Verify file checksums after download using CHECKSUMS file from source. Detects and re-downloads corrupted files."
    ),
):
    """
    Download Ensembl variation VCF files and convert them to Parquet format.
    
    This command downloads VCF files from Ensembl, converts them to Parquet,
    and optionally splits them by variant type (SNV, insertion, deletion, etc.).
    
    Performance tuning options:
    - Use --http-chunk-size to optimize chunk size for your network
    - Adjust --connect-timeout and --sock-read-timeout for network conditions
    - Increase --retries for unreliable connections
    
    Note: HTTP connection pooling is managed automatically by fsspec/aiohttp.
    The --http-max-pool parameter is kept for API consistency but has limited effect.
    
    Environment variables:
    - PREPARE_ANNOTATIONS_DOWNLOAD_TIMEOUT: Override default total timeout
    """
    # run the pipeline
    run_pipeline(
        name="ensembl",
        pipeline_func=PreparationPipelines.download_ensembl,
        log=log,
        upload=upload,
        repo_id=repo_id,
        token=token,
        dest_dir=dest_dir,
        vcf_dir=vcf_dir,
        parquet_dir=parquet_dir,
        split=split,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        species=species,
        pattern=pattern,
        url=url,
        http_max_pool=http_max_pool,
        http_chunk_size=http_chunk_size,
        connect_timeout=connect_timeout,
        sock_read_timeout=sock_read_timeout,
        retries=retries,
        verify_checksums=verify_checksums,
        profile=profile
    )


@app.command()
def split(
    parquet_files: List[Path] = typer.Argument(..., help="Parquet files to split"),
    output_dir: Optional[str] = typer.Option(
        None, 
        "--output-dir", "-o", 
        help="Output directory for split files. Defaults to 'splitted_variants' next to each input file."
    ),
    explode_snv_alt: bool = typer.Option(
        False, 
        "--explode-snv-alt/--no-explode-snv-alt", 
        help="Explode ALT column for SNV variants"
    ),
    log: bool = typer.Option(True, "--log/--no-log", help="Enable logging"),
    profile: bool = typer.Option(True, "--profile/--no-profile", help="Track resource usage")
):
    """
    Split existing parquet files by variant type (TSA).
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "split_parquets.json", logs / "split_parquets.log")
        to_nice_stdout()
        
    from prepare_annotations.runtime import setup_prefect_api
    _, ui_url = setup_prefect_api()
    console.print(f"\n[bold cyan]🌊 Prefect UI:[/bold cyan] [link={ui_url}]{ui_url}[/link]")
    
    with start_action(action_type="split_command", num_files=len(parquet_files)) as action:
        console.print(f"🚀 Splitting {len(parquet_files)} parquet files...")
        
        results = PreparationPipelines.split_existing_parquets(
            parquet_files=parquet_files,
            explode_snv_alt=explode_snv_alt,
            write_to=Path(output_dir) if output_dir else None,
            log=log,
            profile=profile
        )
        
        console.print(f"\n✅ Splitting completed!")
        console.print(f"🔀 Split into {len(results.split_variants_dict)} variant types")
        for tsa, files in results.split_variants_dict.items():
            console.print(f"  - [bold]{tsa}[/bold]: {len(files)} files")
            
        action.log(message_type="success", num_types=len(results.split_variants_dict))



@app.command()
def index_rsids(
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir",
        help="Input directory containing split variant folders (SNV, deletion, etc.). Defaults to split variants in standard cache directory."
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output-path",
        help="Output path for the rsID coordinates parquet file. Defaults to standard output directory."
    ),
    memory_fraction: float = typer.Option(
        0.8,
        "--memory-fraction",
        help="Fraction of available system memory to use (0.0-1.0). Automatically detects available RAM."
    ),
    as_dataset: bool = typer.Option(
        False,
        "--as-dataset/--single-file",
        help="Write a directory of per-chromosome parquet chunks and skip the final merge (much faster).",
    ),
    explode_snv_alt: bool = typer.Option(
        False,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants during auto-splitting"
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Re-compute all chunks even if they already exist in the cache."
    ),
    compression_level: int = typer.Option(
        14,
        "--compression-level",
        help="ZSTD compression level (default: 14)"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track resource usage"
    ),
):
    """
    Compute rsID coordinates (chrom, start, end, id, variant_type) from Ensembl parquet files.
    
    Uses DuckDB's streaming processing with dynamic memory limits based on available RAM.
    Files are processed per-chromosome (no cross-file duplicates), enabling efficient streaming.
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "ensembl_rsid_coords.json", logs / "ensembl_rsid_coords.log")
        to_nice_stdout()
    
    # Show Prefect UI information
    from prepare_annotations.runtime import setup_prefect_api
    is_server, ui_url = setup_prefect_api()
    console.print(f"\n[bold cyan]🌊 Prefect UI:[/bold cyan] [link={ui_url}]{ui_url}[/link]")
    if not is_server:
        console.print("[yellow]💡 Tip: Run 'prefect server start' in another terminal to access the UI[/yellow]\n")
    else:
        console.print("[green]✅ Connected to Prefect server[/green]\n")
        
    with start_action(action_type="ensembl_rsid_coords_command") as action:
        console.print("🚀 Starting rsID coordinate computation...")
        
        results = PreparationPipelines.compute_ensembl_rsid_coords(
            input_dir=Path(input_dir) if input_dir else None,
            output_path=Path(output_path) if output_path else None,
            memory_fraction=memory_fraction,
            output_dataset=as_dataset,
            explode_snv_alt=explode_snv_alt,
            force=force,
            compression_level=compression_level,
            log=log,
            profile=profile,
        )
        
        console.print(f"\n✅ Computation completed!")
        console.print(f"[dim]View details at: {ui_url}[/dim]")
        console.print(f"📦 Output: [bold cyan]{results.output_path}[/bold cyan]")
        console.print(f"📊 Count: [bold]{results.count}[/bold]")
        
        action.log(message_type="success", output_path=str(results.output_path), count=results.count)


@app.command()
def clinvar(
    assembly: str = typer.Option(
        "GRCh38_ensembl",
        "--assembly",
        help="Genome assembly and source: GRCh38_ensembl (default), GRCh38, or GRCh37. Ensembl suffix uses Ensembl VEP-annotated ClinVar, otherwise uses NCBI."
    ),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        help="Custom URL to download ClinVar VCF from. Overrides default NCBI and Ensembl URLs."
    ),
    pattern: Optional[str] = typer.Option(
        None,
        "--pattern",
        help="Regex pattern to filter ClinVar files."
    ),
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads. If not specified, uses standard cache directory."
    ),
    vcf_dir: Optional[str] = typer.Option(
        None,
        "--vcf-dir",
        help="Optional specific directory for VCF downloads. Defaults to <dest_dir>/vcf"
    ),
    parquet_dir: Optional[str] = typer.Option(
        None,
        "--parquet-dir",
        help="Optional specific directory for Parquet files. Defaults to <dest_dir>"
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files by variant type (TSA)"
    ),
    explode_snv_alt: bool = typer.Option(
        False,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants during splitting"
    ),
    alts_list: bool = typer.Option(
        True,
        "--alts-list/--no-alts-list",
        help="Add a list of alternative alleles as 'alts' column"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="Upload parquet files to Hugging Face Hub after processing"
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/clinvar",
        "--repo-id",
        help="Hugging Face repository ID for upload"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token (uses HF_TOKEN env var if not provided)"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track and display resource usage (time and memory)"
    ),
):
    """
    Download ClinVar VCF files and convert them to Parquet format.
    
    This command downloads VCF files from ClinVar (NCBI or Ensembl based on assembly), 
    converts them to Parquet, and optionally splits them by variant type (SNV, insertion, deletion, etc.).
    
    ClinVar is human-only. Use --assembly to select between:
    - GRCh38_ensembl: Ensembl VEP-annotated ClinVar (default, includes consequence annotations)
    - GRCh38: NCBI ClinVar for GRCh38
    - GRCh37: NCBI ClinVar for GRCh37
    """
    run_pipeline(
        name="clinvar",
        pipeline_func=PreparationPipelines.download_clinvar,
        log=log,
        upload=upload,
        repo_id=repo_id,
        token=token,
        dest_dir=dest_dir,
        vcf_dir=vcf_dir,
        parquet_dir=parquet_dir,
        split=split,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        assembly=assembly,
        url=url,
        pattern=pattern,
        profile=profile
    )


@app.command()
def dbsnp(
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads. If not specified, uses standard cache directory."
    ),
    build: str = typer.Option(
        "GRCh38",
        "--build",
        help="Genome build (GRCh38 or GRCh37)"
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files by variant type"
    ),
    explode_snv_alt: bool = typer.Option(
        False,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants during splitting"
    ),
    alts_list: bool = typer.Option(
        True,
        "--alts-list/--no-alts-list",
        help="Add a list of alternative alleles as 'alts' column"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging"
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="Upload results to Hugging Face Hub"
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/dbsnp",
        "--repo-id",
        help="Hugging Face repository ID"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track resource usage"
    ),
):
    """Download and prepare dbSNP data."""
    run_pipeline(
        name="dbsnp",
        pipeline_func=PreparationPipelines.download_dbsnp,
        log=log,
        upload=upload,
        repo_id=repo_id,
        token=token,
        dest_dir=dest_dir,
        split=split,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        build=build,
        profile=profile
    )


@app.command()
def dbsnp_t2t(
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads."
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files"
    ),
    explode_snv_alt: bool = typer.Option(
        False,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants during splitting"
    ),
    alts_list: bool = typer.Option(
        True,
        "--alts-list/--no-alts-list",
        help="Add a list of alternative alleles as 'alts' column"
    ),
    compression: str = typer.Option(
        "zstd",
        "--compression",
        help="Parquet compression type (default: zstd)"
    ),
    compression_level: int = typer.Option(
        14,
        "--compression-level",
        help="Parquet compression level (default: 14 for zstd)"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging"
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="Upload results to Hugging Face Hub"
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/dbsnp_t2t",
        "--repo-id",
        help="Hugging Face repository ID"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track resource usage"
    ),
    progress_interval_seconds: float = typer.Option(
        60.0,
        "--progress-interval-seconds",
        help="Download progress update interval in seconds (default: 60 for dbSNP T2T)",
    ),
    s3_max_pool: Optional[int] = typer.Option(
        None,
        "--s3-max-pool",
        help="Max concurrent S3 connections (default: 50). Increase for faster downloads."
    ),
    s3_block_size: Optional[int] = typer.Option(
        None,
        "--s3-block-size",
        help="S3 read block size in bytes (default: 50MB). Larger blocks can improve sequential speed."
    ),
):
    """Download and prepare dbSNP T2T (CHM13) data."""
    run_pipeline(
        name="dbsnp_t2t",
        pipeline_func=PreparationPipelines.download_dbsnp_t2t,
        log=log,
        upload=upload,
        repo_id=repo_id,
        token=token,
        dest_dir=dest_dir,
        split=split,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        compression=compression,
        compression_level=compression_level,
        progress_interval_seconds=progress_interval_seconds,
        s3_max_pool=s3_max_pool,
        s3_block_size=s3_block_size,
        profile=profile,
    )


@app.command()
def gnomad(
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads."
    ),
    version: str = typer.Option(
        "v4",
        "--version",
        help="gnomAD version (v4 or v3)"
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files"
    ),
    explode_snv_alt: bool = typer.Option(
        False,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants during splitting"
    ),
    alts_list: bool = typer.Option(
        True,
        "--alts-list/--no-alts-list",
        help="Add a list of alternative alleles as 'alts' column"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging"
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="Upload results to Hugging Face Hub"
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/gnomad",
        "--repo-id",
        help="Hugging Face repository ID"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track resource usage"
    ),
):
    """Download and prepare gnomAD data."""
    run_pipeline(
        name="gnomad",
        pipeline_func=PreparationPipelines.download_gnomad,
        log=log,
        upload=upload,
        repo_id=repo_id,
        token=token,
        dest_dir=dest_dir,
        split=split,
        explode_snv_alt=explode_snv_alt,
        alts_list=alts_list,
        version=version,
        profile=profile
    )


@app.command()
def upload_clinvar(
    source_dir: Optional[str] = typer.Option(None, "--source-dir"),
    repo_id: str = typer.Option("just-dna-seq/clinvar", "--repo-id"),
    token: Optional[str] = typer.Option(None, "--token"),
    pattern: str = typer.Option("**/*.parquet", "--pattern"),
    path_prefix: str = typer.Option("data", "--path-prefix"),
    log: bool = typer.Option(True, "--log/--no-log"),
):
    """Upload ClinVar parquet files to Hugging Face Hub."""
    PreparationPipelines.upload_dataset_to_hf(
        dataset_type="clinvar",
        source_dir=Path(source_dir) if source_dir else None,
        repo_id=repo_id,
        token=token,
        pattern=pattern,
        path_prefix=path_prefix,
        log=log,
    )


@app.command()
def upload_ensembl(
    source_dir: Optional[str] = typer.Option(None, "--source-dir"),
    repo_id: str = typer.Option("just-dna-seq/ensembl_variations", "--repo-id"),
    token: Optional[str] = typer.Option(None, "--token"),
    pattern: str = typer.Option("**/*.parquet", "--pattern"),
    path_prefix: str = typer.Option("data", "--path-prefix"),
    log: bool = typer.Option(True, "--log/--no-log"),
):
    """Upload Ensembl variation parquet files to Hugging Face Hub."""
    PreparationPipelines.upload_dataset_to_hf(
        dataset_type="ensembl",
        source_dir=Path(source_dir) if source_dir else None,
        repo_id=repo_id,
        token=token,
        pattern=pattern,
        path_prefix=path_prefix,
        log=log,
    )


@app.command()
def upload_dbsnp(
    source_dir: Optional[str] = typer.Option(None, "--source-dir"),
    repo_id: str = typer.Option("just-dna-seq/dbsnp", "--repo-id"),
    token: Optional[str] = typer.Option(None, "--token"),
    pattern: str = typer.Option("**/*.parquet", "--pattern"),
    path_prefix: str = typer.Option("data", "--path-prefix"),
    log: bool = typer.Option(True, "--log/--no-log"),
):
    """Upload dbSNP parquet files to Hugging Face Hub."""
    PreparationPipelines.upload_dataset_to_hf(
        dataset_type="dbsnp",
        source_dir=Path(source_dir) if source_dir else None,
        repo_id=repo_id,
        token=token,
        pattern=pattern,
        path_prefix=path_prefix,
        log=log,
    )


@app.command()
def upload_gnomad(
    source_dir: Optional[str] = typer.Option(None, "--source-dir"),
    repo_id: str = typer.Option("just-dna-seq/gnomad", "--repo-id"),
    token: Optional[str] = typer.Option(None, "--token"),
    pattern: str = typer.Option("**/*.parquet", "--pattern"),
    path_prefix: str = typer.Option("data", "--path-prefix"),
    log: bool = typer.Option(True, "--log/--no-log"),
):
    """Upload gnomAD parquet files to Hugging Face Hub."""
    PreparationPipelines.upload_dataset_to_hf(
        dataset_type="gnomad",
        source_dir=Path(source_dir) if source_dir else None,
        repo_id=repo_id,
        token=token,
        pattern=pattern,
        path_prefix=path_prefix,
        log=log,
    )




@app.command()
def update_card(
    dataset: str = typer.Argument(..., help="Dataset name (ensembl, clinvar, dbsnp, gnomad)"),
    source_dir: Optional[str] = typer.Option(None, "--source-dir", help="Source directory to analyze for stats"),
    repo_id: Optional[str] = typer.Option(None, "--repo-id", help="Hugging Face repository ID"),
    token: Optional[str] = typer.Option(None, "--token", help="Hugging Face API token"),
):
    """Update the dataset card (README.md) for a dataset on Hugging Face Hub."""
    to_nice_stdout()
    
    # Configuration map for datasets
    config = {
        "ensembl": {
            "repo": "just-dna-seq/ensembl_variations",
            "source": get_default_output_dir("ensembl"),
            "card_gen": generate_ensembl_card
        },
        "clinvar": {
            "repo": "just-dna-seq/clinvar",
            "source": get_default_output_dir("clinvar"),
            "card_gen": generate_clinvar_card
        },
        "dbsnp": {
            "repo": "just-dna-seq/dbsnp",
            "source": get_default_output_dir("dbsnp_grch38"),
            "card_gen": generate_dbsnp_card
        },
        "dbsnp_t2t": {
            "repo": "just-dna-seq/dbsnp_t2t",
            "source": get_default_output_dir("dbsnp_t2t"),
            "card_gen": generate_dbsnp_t2t_card
        },
        "gnomad": {
            "repo": "just-dna-seq/gnomad",
            "source": get_default_output_dir("gnomad_v4"),
            "card_gen": generate_gnomad_card
        }
    }
    
    ds_cfg = config.get(dataset.lower())
    if not ds_cfg:
        console.print(f"[bold red]Error:[/bold red] Unknown dataset '{dataset}'. Please provide --repo-id and --source-dir if it's a custom dataset.")
        raise typer.Exit(1)

    repo_id = repo_id or ds_cfg["repo"]
    source_dir = source_dir or ds_cfg["source"]
    
    # Check if splitted_variants exists
    source_path = Path(source_dir)
    if (source_path / "splitted_variants").exists():
        source_path = source_path / "splitted_variants"
    
    if not source_path.exists():
        # Try fallback for dbsnp/gnomad versions
        if dataset.lower() == "dbsnp":
            source_path = get_default_output_dir("dbsnp_grch37")
        elif dataset.lower() == "gnomad":
            source_path = get_default_output_dir("gnomad_v3")
            
        if (source_path / "splitted_variants").exists():
            source_path = source_path / "splitted_variants"

    if not source_path.exists():
        console.print(f"[bold red]Error:[/bold red] Source directory '{source_path}' does not exist.")
        raise typer.Exit(1)
        
    parquet_files = collect_parquet_files(source_path)
    
    if not parquet_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No parquet files found in {source_path}")
        return

    variant_types = {f.relative_to(source_path).parts[0] for f in parquet_files if len(f.relative_to(source_path).parts) > 1}
    total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
    
    # Generate card using the mapped generator
    card_content = ds_cfg["card_gen"](len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
        
    # Upload only the README.md
    api = HfApi(token=token)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write(card_content)
        tmp_path = tmp.name
        
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset card for {dataset}",
        )
        console.print(f"✅ Dataset card updated for [bold cyan]{repo_id}[/bold cyan]")
    finally:
        os.unlink(tmp_path)



@app.command()
def version():
    """Show version information."""
    try:
        import importlib.metadata
        v = importlib.metadata.version("prepare-annotations")
        console.print(f"prepare-annotations version: [bold green]{v}[/bold green]")
    except importlib.metadata.PackageNotFoundError:
        console.print("prepare-annotations version: [yellow]development[/yellow]")


if __name__ == "__main__":
    app()

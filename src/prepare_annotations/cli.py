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

from prepare_annotations.preparation.runners import PreparationPipelines
from pycomfort.logging import to_nice_file, to_nice_stdout

# Create the main CLI app
app = typer.Typer(
    name="prepare-annotations",
    help="Modern Genomic Data Pipeline Tool (using Pipelines class)",
    rich_markup_mode="rich",
    no_args_is_help=True
)

console = Console()


@app.command()
def ensembl(
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads. If not specified, uses data/input/ensembl_variations."
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files by variant type (TSA)"
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
    explode_snv_alt: bool = typer.Option(
        True,
        "--explode-snv-alt/--no-explode-snv-alt",
        help="Explode ALT column for SNV variants when splitting"
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
):
    """
    Download Ensembl variation VCF files using the Pipelines approach.
    
    Downloads VCF files from Ensembl FTP, converts them to parquet, and optionally
    splits them by variant type. Can also upload results directly to Hugging Face Hub.
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "prepare_ensembl.json", logs / "prepare_ensembl.log")
        to_nice_stdout()
    
    with start_action(action_type="prepare_ensembl_command") as action:
        action.log(
            message_type="info",
            dest_dir=dest_dir,
            pattern=pattern,
            split=split,
            upload=upload
        )
        
        effective_dest = dest_dir if dest_dir else "data/input/ensembl_variations"
        console.print(f"📁 Destination: [bold blue]{effective_dest}[/bold blue]")
        console.print(f"🔄 Splitting: [bold blue]{split}[/bold blue]")
        
        console.print("🚀 Starting pipeline execution...")
        
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Running pipeline...", total=None)
        
        results = PreparationPipelines.download_ensembl(
            dest_dir=Path(dest_dir) if dest_dir else None,
            with_splitting=split,
            log=log,
            pattern=pattern,
            url=url,
            profile=profile,
        )
        
        progress.update(task, description="✅ Pipeline completed")
        
        console.print("\n✅ Pipeline execution completed!")
        
        if results.vcf_parquet_path:
            parquet_files = results.vcf_parquet_path
            console.print(f"📦 Converted {len(parquet_files) if isinstance(parquet_files, list) else 1} parquet files")
        
        if results.split_variants_dict:
            split_dict = results.split_variants_dict
            console.print(f"🔀 Split variants into {len(split_dict)} categories")
        
        action.log(message_type="success", result_keys=list(results.model_dump().keys()))
        
        if upload:
            console.print("\n🔄 Starting upload to Hugging Face...")
            console.print(f"📦 Repository: [bold cyan]{repo_id}[/bold cyan]")
            
            # Default upload source is data/output/ensembl_variations
            upload_source_dir = Path(dest_dir) if dest_dir else Path("data/output/ensembl_variations")
            if split:
                upload_source_dir = upload_source_dir / "splitted_variants"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Uploading files...", total=None)
                
                upload_results = PreparationPipelines.upload_ensembl_to_hf(
                    source_dir=upload_source_dir,
                    repo_id=repo_id,
                    token=token,
                    log=log,
                )
                
                progress.update(task, description="✅ Upload completed")
            
            console.print(f"\n📊 Upload Summary:")
            console.print(f"  - Total files: [bold]{len(upload_results.uploaded_files)}[/bold]")
            console.print(f"  - Uploaded: [bold green]{upload_results.num_uploaded}[/bold green]")
            console.print(f"  - Skipped: [bold yellow]{upload_results.num_skipped}[/bold yellow]")


@app.command()
def clinvar(
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads. If not specified, uses data/input/clinvar."
    ),
    split: bool = typer.Option(
        False,
        "--split/--no-split",
        help="Split downloaded parquet files by variant type (TSA)"
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
    Download ClinVar VCF files using the Pipelines approach.
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "prepare_clinvar.json", logs / "prepare_clinvar.log")
        to_nice_stdout()
    
    with start_action(action_type="prepare_clinvar_command") as action:
        action.log(
            message_type="info",
            dest_dir=dest_dir,
            split=split,
            upload=upload
        )
        
        console.print("🔧 Setting up ClinVar pipeline...")
        console.print("🚀 Executing pipeline...")
        
        results = PreparationPipelines.download_clinvar(
            dest_dir=Path(dest_dir) if dest_dir else None,
            with_splitting=split,
            log=log,
            profile=profile,
        )

        console.print("✅ ClinVar download completed!")
        action.log(message_type="success", result_keys=list(results.model_dump().keys()))
        
        if upload:
            console.print("\n🔄 Starting upload to Hugging Face...")
            console.print(f"📦 Repository: [bold cyan]{repo_id}[/bold cyan]")
            
            upload_source_dir = Path(dest_dir) if dest_dir else Path("data/output/clinvar")
            if split:
                upload_source_dir = upload_source_dir / "splitted_variants"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Uploading files...", total=None)
                
                upload_results = PreparationPipelines.upload_clinvar_to_hf(
                    source_dir=upload_source_dir,
                    repo_id=repo_id,
                    token=token,
                    log=log,
                )
                
                progress.update(task, description="✅ Upload completed")
            
            console.print(f"\n📊 Upload Summary:")
            console.print(f"  - Total files: [bold]{len(upload_results.uploaded_files)}[/bold]")
            console.print(f"  - Uploaded: [bold green]{upload_results.num_uploaded}[/bold green]")
            console.print(f"  - Skipped: [bold yellow]{upload_results.num_skipped}[/bold yellow]")


@app.command()
def upload_clinvar(
    source_dir: Optional[str] = typer.Option(
        None,
        "--source-dir",
        help="Source directory containing parquet files. If not specified, uses data/output/clinvar."
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/clinvar",
        "--repo-id",
        help="Hugging Face repository ID"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token. If not provided, uses HF_TOKEN environment variable."
    ),
    pattern: str = typer.Option(
        "**/*.parquet",
        "--pattern",
        help="Glob pattern for finding parquet files"
    ),
    path_prefix: str = typer.Option(
        "data",
        "--path-prefix",
        help="Prefix for paths in the repository"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track and display resource usage (time and memory)"
    ),
):
    """
    Upload ClinVar parquet files to Hugging Face Hub.
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "upload_clinvar.json", logs / "upload_clinvar.log")
        to_nice_stdout()
    
    with start_action(action_type="upload_clinvar_command") as action:
        console.print("🚀 Starting upload...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Uploading files...", total=None)
            
            results = PreparationPipelines.upload_clinvar_to_hf(
                source_dir=Path(source_dir) if source_dir else None,
                repo_id=repo_id,
                token=token,
                pattern=pattern,
                path_prefix=path_prefix,
                log=log,
            )
            
            progress.update(task, description="✅ Upload completed")
        
        console.print(f"\n📊 Summary: {results.num_uploaded} uploaded, {results.num_skipped} skipped")


@app.command()
def upload_ensembl(
    source_dir: Optional[str] = typer.Option(
        None,
        "--source-dir",
        help="Source directory containing parquet files. If not specified, uses data/output/ensembl_variations."
    ),
    repo_id: str = typer.Option(
        "just-dna-seq/ensembl_variations",
        "--repo-id",
        help="Hugging Face repository ID"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="Hugging Face API token. If not provided, uses HF_TOKEN environment variable."
    ),
    pattern: str = typer.Option(
        "**/*.parquet",
        "--pattern",
        help="Glob pattern for finding parquet files"
    ),
    path_prefix: str = typer.Option(
        "data",
        "--path-prefix",
        help="Prefix for paths in the repository"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Track and display resource usage (time and memory)"
    ),
):
    """
    Upload Ensembl variation parquet files to Hugging Face Hub.
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "upload_ensembl.json", logs / "upload_ensembl.log")
        to_nice_stdout()
    
    with start_action(action_type="upload_ensembl_command") as action:
        console.print("🚀 Starting upload...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Uploading files...", total=None)
            
            results = PreparationPipelines.upload_ensembl_to_hf(
                source_dir=Path(source_dir) if source_dir else None,
                repo_id=repo_id,
                token=token,
                pattern=pattern,
                path_prefix=path_prefix,
                log=log,
            )
            
            progress.update(task, description="✅ Upload completed")
        
        console.print(f"\n📊 Summary: {results.num_uploaded} uploaded, {results.num_skipped} skipped")


@app.command()
def dbsnp(
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads. If not specified, uses data/input/dbsnp."
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
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / f"prepare_dbsnp_{build.lower()}.json", logs / f"prepare_dbsnp_{build.lower()}.log")
        to_nice_stdout()
    
    with start_action(action_type="prepare_dbsnp_command", build=build) as action:
        results = PreparationPipelines.download_dbsnp(
            dest_dir=Path(dest_dir) if dest_dir else None,
            build=build,
            with_splitting=split,
            log=log,
            profile=profile,
        )
        
        if upload:
            upload_source_dir = Path(dest_dir) if dest_dir else Path(f"data/output/dbsnp_{build.lower()}")
            if split:
                upload_source_dir = upload_source_dir / "splitted_variants"
                
            PreparationPipelines.upload_dbsnp_to_hf(
                source_dir=upload_source_dir,
                repo_id=repo_id,
                token=token,
                log=log,
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
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / f"prepare_gnomad_{version}.json", logs / f"prepare_gnomad_{version}.log")
        to_nice_stdout()
    
    with start_action(action_type="prepare_gnomad_command", version=version) as action:
        results = PreparationPipelines.download_gnomad(
            dest_dir=Path(dest_dir) if dest_dir else None,
            version=version,
            with_splitting=split,
            log=log,
            profile=profile,
        )
        
        if upload:
            upload_source_dir = Path(dest_dir) if dest_dir else Path(f"data/output/gnomad_{version}")
            if split:
                upload_source_dir = upload_source_dir / "splitted_variants"
                
            PreparationPipelines.upload_gnomad_to_hf(
                source_dir=upload_source_dir,
                repo_id=repo_id,
                token=token,
                log=log,
            )


@app.command()
def upload_dbsnp(
    source_dir: Optional[str] = typer.Option(None, "--source-dir"),
    repo_id: str = typer.Option("just-dna-seq/dbsnp", "--repo-id"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Upload dbSNP parquet files to Hugging Face Hub."""
    PreparationPipelines.upload_dbsnp_to_hf(
        source_dir=Path(source_dir) if source_dir else None,
        repo_id=repo_id,
        token=token,
    )


@app.command()
def upload_gnomad(
    source_dir: Optional[str] = typer.Option(None, "--source-dir"),
    repo_id: str = typer.Option("just-dna-seq/gnomad", "--repo-id"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Upload gnomAD parquet files to Hugging Face Hub."""
    PreparationPipelines.upload_gnomad_to_hf(
        source_dir=Path(source_dir) if source_dir else None,
        repo_id=repo_id,
        token=token,
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
    
    # Resolve repo_id if not provided
    if repo_id is None:
        repo_map = {
            "ensembl": "just-dna-seq/ensembl_variations",
            "clinvar": "just-dna-seq/clinvar",
            "dbsnp": "just-dna-seq/dbsnp",
            "gnomad": "just-dna-seq/gnomad",
        }
        repo_id = repo_map.get(dataset.lower())
        if not repo_id:
            console.print(f"[bold red]Error:[/bold red] Unknown dataset '{dataset}'. Please provide --repo-id.")
            raise typer.Exit(1)

    # Resolve source_dir if not provided
    if source_dir is None:
        source_map = {
            "ensembl": "data/output/ensembl_variations",
            "clinvar": "data/output/clinvar",
            "dbsnp": "data/output/dbsnp_grch38",
            "gnomad": "data/output/gnomad_v4",
        }
        source_dir = source_map.get(dataset.lower())
        if source_dir:
             # Check if splitted_variants exists
             s_path = Path(source_dir) / "splitted_variants"
             if s_path.exists():
                 source_dir = str(s_path)
    
    if not source_dir or not Path(source_dir).exists():
        console.print(f"[bold red]Error:[/bold red] Source directory '{source_dir}' does not exist.")
        raise typer.Exit(1)
        
    source_path = Path(source_dir)
    parquet_files = collect_parquet_files(source_path)
    
    if not parquet_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No parquet files found in {source_dir}")
        return

    variant_types = {f.relative_to(source_path).parts[0] for f in parquet_files if len(f.relative_to(source_path).parts) > 1}
    total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
    
    # Generate card based on dataset type
    if dataset.lower() == "ensembl":
        card_content = generate_ensembl_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
    elif dataset.lower() == "clinvar":
        card_content = generate_clinvar_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
    elif dataset.lower() == "dbsnp":
        card_content = generate_dbsnp_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
    elif dataset.lower() == "gnomad":
        card_content = generate_gnomad_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
    else:
        console.print(f"[bold red]Error:[/bold red] No card generator for dataset '{dataset}'")
        raise typer.Exit(1)
        
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

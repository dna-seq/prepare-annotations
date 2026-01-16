"""
Prepare Annotations CLI - Modern pipeline-based data preparation.

This module provides a CLI interface using the Pipelines class for better
parallelization, caching, and pipeline composition.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from eliot import start_action
from prepare_annotations.huggingface.uploader import collect_parquet_files
from prepare_annotations.huggingface.dataset_cards import (
    generate_ensembl_card, 
    generate_clinvar_card,
    generate_dbsnp_card,
    generate_dbsnp_t2t_card,
    generate_gnomad_card,
)
from huggingface_hub import HfApi

from prepare_annotations.core.runtime import load_env
from prepare_annotations.core.paths import LOGS_DIR, get_default_cache_dir, get_output_dir

logs = LOGS_DIR

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

from prepare_annotations.pipelines import PreparationPipelines
from pycomfort.logging import to_nice_file, to_nice_stdout

# Create the main CLI app
app = typer.Typer(
    name="prepare-annotations",
    help="Modern Genomic Data Pipeline Tool (using Pipelines class)",
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Minimal Dagster-only CLI (used by `uv run prepare`)
dagster_app = typer.Typer(
    name="prepare",
    help="Dagster pipelines and UI for prepare-annotations",
    rich_markup_mode="rich",
    no_args_is_help=True,
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

    # Show Dagster UI information upfront
    console.print(f"\n[bold cyan]🔷 Dagster UI:[/bold cyan] [link=http://127.0.0.1:3000]http://127.0.0.1:3000[/link]")
    console.print("[yellow]💡 Tip: Run 'dagster-ui' or 'prepare dagster ui' to start the UI[/yellow]\n")

    with start_action(action_type=f"prepare_{name}_command") as action:
        action.log(
            message_type="info",
            dest_dir=dest_dir,
            split=split,
            upload=upload,
            **pipeline_kwargs
        )
        
        effective_dest = dest_dir if dest_dir else get_default_cache_dir(name)
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
        
        if results.vcf_parquet_path:
            console.print(f"📦 Converted {len(results.vcf_parquet_path)} parquet files")
        
        if results.split_variants_dict:
            console.print(f"🔀 Split variants into {len(results.split_variants_dict)} categories")
            
        action.log(message_type="success", result_keys=list(results.model_dump().keys()))
        
        if upload:
            console.print(f"\n🔄 Starting upload to Hugging Face...")
            console.print(f"📦 Repository: [bold cyan]{repo_id}[/bold cyan]")
            
            upload_source_dir = Path(dest_dir) if dest_dir else get_default_cache_dir(name)
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
    job: str = typer.Option(
        "full",
        "--job", "-j",
        help="Job name: full (default), prepare, download, convert, upload"
    ),
    species: str = typer.Option(
        "homo_sapiens",
        "--species",
        help="Species name to download (e.g., homo_sapiens, mus_musculus). Default: homo_sapiens"
    ),
):
    """
    Run Ensembl pipeline using Dagster (default: full pipeline with upload).
    
    Jobs: full (default), prepare, download, convert, upload
    
    Starts Dagster UI in background for monitoring.
    """
    _dagster_run_ensembl(job_name=job, species=species)


def _dagster_run_ensembl(
    job_name: str = "full",
    species: str = "homo_sapiens",
):
    """
    Run Ensembl Dagster pipeline using the Python API.
    
    For partitioned assets, this:
    1. Materializes ensembl_vcf_urls (discovers files, registers partitions)
    2. Runs backfill for all partitions of ensembl_vcf_file and ensembl_parquet_file
    3. Materializes the collector and upload assets
    """
    import json
    import os
    from dagster import DagsterInstance, materialize
    
    # Set up DAGSTER_HOME
    dagster_home = _get_dagster_home()
    _ensure_dagster_config(dagster_home)
    os.environ["DAGSTER_HOME"] = str(dagster_home)
    
    console.print(f"\n[bold cyan]🔷 Running Dagster Pipeline: {job_name}[/bold cyan]")
    console.print(f"   Species: [bold blue]{species}[/bold blue]")
    console.print(f"   Dagster home: {dagster_home}")
    
    # Start UI in background if not running
    _ensure_dagster_ui_running(force_restart=True)
    
    console.print("\n🚀 Executing pipeline using Dagster Python API...\n")
    console.print("   Monitor progress at: http://127.0.0.1:3000\n")

    # Import assets and definitions
    from prepare_annotations.assets import (
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
        ENSEMBL_VCF_PARTITIONS,
    )
    from prepare_annotations.core.dagster_io_managers import (
        ensembl_cache_io_manager,
        huggingface_upload_io_manager,
    )
    
    instance = DagsterInstance.get()
    
    resources = {
        "io_manager": ensembl_cache_io_manager,
        "hf_upload_io_manager": huggingface_upload_io_manager,
    }
    
    def _run_config_for_assets(asset_names: list[str]) -> dict:
        ops_config: dict[str, dict] = {}
        if "ensembl_vcf_urls" in asset_names:
            ops_config["ensembl_vcf_urls"] = {"config": {"species": species}}
        if "ensembl_vcf_file" in asset_names:
            ops_config["ensembl_vcf_file"] = {"config": {"species": species}}
        if "ensembl_parquet_file" in asset_names:
            # Parquet conversion does not accept species config.
            ops_config["ensembl_parquet_file"] = {"config": {}}
        return {"ops": ops_config} if ops_config else {}

    def _get_max_download_workers() -> int:
        env_value = os.getenv("PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS")
        if env_value:
            return max(1, int(env_value))
        return 4

    all_assets = [
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
    ]
    
    # Step 1: Materialize ensembl_vcf_urls (discovers files, registers partitions)
    console.print("[bold]Step 1:[/bold] Discovering VCF files from Ensembl FTP...")
    result = materialize(
        assets=all_assets,
        selection=["ensembl_vcf_urls"],
        resources=resources,
        run_config=_run_config_for_assets(["ensembl_vcf_urls"]),
        instance=instance,
    )
    if not result.success:
        console.print("[bold red]❌ Failed to discover VCF URLs![/bold red]")
        raise typer.Exit(1)
    console.print("[green]✓ VCF URLs discovered[/green]")
    
    # Get registered partitions
    partition_keys = list(instance.get_dynamic_partitions(ENSEMBL_VCF_PARTITIONS.name))
    console.print(f"   Found [bold]{len(partition_keys)}[/bold] partitions")
    
    if job_name in ("download", "prepare", "full"):
        # Step 2: Download VCF files (parallel downloader + sequential lineage materialization)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from prepare_annotations.core.dagster_configs import EnsemblDownloadConfig
        from prepare_annotations.core.paths import (
            get_default_ensembl_cache_dir,
            get_ensembl_species_url,
        )
        from prepare_annotations.downloaders.vcf import (
            download_path,
            download_checksums,
            ChecksumInfo,
        )

        max_workers = _get_max_download_workers()
        console.print(
            f"\n[bold]Step 2:[/bold] Downloading {len(partition_keys)} VCF files "
            f"with {max_workers} parallel workers..."
        )

        download_config = EnsemblDownloadConfig(species=species)
        cache_dir = get_default_ensembl_cache_dir(species)
        urls_file = cache_dir / "vcf_urls.json"
        urls = json.loads(urls_file.read_text())
        vcf_dir = cache_dir / "vcf"
        vcf_dir.mkdir(parents=True, exist_ok=True)

        existing_files = {p.name for p in vcf_dir.glob("*.vcf.gz")}
        missing_urls = [u for u in urls if u.rsplit("/", 1)[-1] not in existing_files]

        checksums: dict[str, ChecksumInfo] = {}
        if download_config.verify_checksums:
            species_url = get_ensembl_species_url(
                download_config.species, download_config.base_url
            )
            try:
                checksums = download_checksums(species_url)
            except FileNotFoundError:
                console.print("[yellow]No CHECKSUMS file found - skipping checksum verification[/yellow]")
            except Exception as exc:
                console.print(f"[yellow]Failed to download checksums: {exc}[/yellow]")

        def _download_url(url: str) -> tuple[str, bool]:
            filename = url.rsplit("/", 1)[-1]
            try:
                _ = download_path(
                    url=url,
                    name="ensembl",
                    dest_dir=vcf_dir,
                    check_files=True,
                    http_max_pool=download_config.http_max_pool,
                    connect_timeout=download_config.connect_timeout,
                    sock_read_timeout=download_config.sock_read_timeout,
                    retries=download_config.retries,
                    expected_checksum=checksums.get(filename),
                )
                return filename, True
            except Exception:
                return filename, False

        if missing_urls:
            failures: list[str] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_url, url): url for url in missing_urls}
                completed = 0
                for future in as_completed(futures):
                    url = futures[future]
                    completed += 1
                    filename = url.rsplit("/", 1)[-1]
                    console.print(f"   [{completed}/{len(missing_urls)}] {filename}...")
                    _, success = future.result()
                    if not success:
                        failures.append(filename)

            if failures:
                console.print(
                    f"[bold red]❌ Failed to download {len(failures)} files.[/bold red]"
                )
                for filename in failures:
                    console.print(f"   - {filename}")
                raise typer.Exit(1)
        else:
            console.print("   No missing VCF files to download.")

        console.print("[green]✓ VCF files downloaded[/green]")

        console.print("[bold]Registering VCF partitions in Dagster...[/bold]")
        for i, partition_key in enumerate(partition_keys, 1):
            console.print(f"   [{i}/{len(partition_keys)}] {partition_key}...")
            result = materialize(
                assets=all_assets,
                selection=["ensembl_vcf_file"],
                resources=resources,
                run_config=_run_config_for_assets(["ensembl_vcf_file"]),
                instance=instance,
                partition_key=partition_key,
            )
            if not result.success:
                console.print(f"[bold red]❌ Failed to register {partition_key}![/bold red]")
                raise typer.Exit(1)
        console.print("[green]✓ All VCF partitions registered[/green]")
    
    if job_name in ("convert", "prepare", "full"):
        # Step 3: Convert to Parquet (partitioned)
        console.print(f"\n[bold]Step 3:[/bold] Converting {len(partition_keys)} VCF files to Parquet...")
        for i, partition_key in enumerate(partition_keys, 1):
            console.print(f"   [{i}/{len(partition_keys)}] {partition_key}...")
            result = materialize(
                assets=all_assets,
                selection=["ensembl_parquet_file"],
                resources=resources,
                run_config=_run_config_for_assets(["ensembl_parquet_file"]),
                instance=instance,
                partition_key=partition_key,
            )
            if not result.success:
                console.print(f"[bold red]❌ Failed to convert {partition_key}![/bold red]")
                raise typer.Exit(1)
        console.print("[green]✓ All files converted to Parquet[/green]")
    
    if job_name in ("upload", "full"):
        # Step 4: Collect and upload
        console.print("\n[bold]Step 4:[/bold] Collecting parquet files...")
        result = materialize(
            assets=all_assets,
            selection=["ensembl_all_parquet_files"],
            resources=resources,
            instance=instance,
        )
        if not result.success:
            console.print("[bold red]❌ Failed to collect parquet files![/bold red]")
            raise typer.Exit(1)
        console.print("[green]✓ Parquet files collected[/green]")
        
        console.print("\n[bold]Step 5:[/bold] Uploading to HuggingFace Hub...")
        result = materialize(
            assets=all_assets,
            selection=["ensembl_hf_upload"],
            resources=resources,
            instance=instance,
        )
        if not result.success:
            console.print("[bold red]❌ Failed to upload to HuggingFace![/bold red]")
            raise typer.Exit(1)
        console.print("[green]✓ Upload complete[/green]")
    
    console.print(f"\n[bold green]✅ Pipeline '{job_name}' completed successfully![/bold green]")


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
        
    with start_action(action_type="split_command", num_files=len(parquet_files)) as action:
        console.print(f"🚀 Splitting {len(parquet_files)} parquet files...")
        
        results = PreparationPipelines.split_existing_parquets(
            parquet_files=parquet_files,
            explode_snv_alt=explode_snv_alt,
            write_to=Path(output_dir) if output_dir else None,
            log=log,
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
    
    # Show Dagster UI information
    console.print(f"\n[bold cyan]🔷 Dagster UI:[/bold cyan] [link=http://127.0.0.1:3000]http://127.0.0.1:3000[/link]")
    
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
        )
        
        console.print(f"\n✅ Computation completed!")
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
def genome(
    species: str = typer.Option(
        "homo_sapiens",
        "--species",
        help="Species name (e.g., homo_sapiens, mus_musculus)"
    ),
    genome_type: str = typer.Option(
        "primary_assembly",
        "--type",
        help="Genome type: primary_assembly (default), toplevel, or chromosome"
    ),
    masking: str = typer.Option(
        "dna",
        "--masking",
        help="DNA masking: dna (unmasked, default), dna_sm (soft-masked), dna_rm (repeat-masked)"
    ),
    release: Optional[int] = typer.Option(
        None,
        "--release",
        help="Ensembl release number (e.g., 114). If not specified, uses latest release."
    ),
    chromosome: Optional[str] = typer.Option(
        None,
        "--chromosome",
        help="Chromosome name for --type=chromosome (e.g., 1, 21, X, MT)"
    ),
    all_chromosomes: bool = typer.Option(
        False,
        "--all-chromosomes",
        help="Download all individual chromosome files instead of primary assembly"
    ),
    chromosomes: Optional[str] = typer.Option(
        None,
        "--chromosomes",
        help="Comma-separated list of chromosomes to download (e.g., '1,2,3,X,Y'). Only used with --all-chromosomes."
    ),
    dest_dir: Optional[str] = typer.Option(
        None,
        "--dest-dir",
        help="Destination directory for downloads. If not specified, uses standard cache directory."
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Force re-download even if files exist"
    ),
    use_ftp: bool = typer.Option(
        False,
        "--ftp/--http",
        help="Use FTP instead of HTTP (HTTP is usually faster)"
    ),
    list_available: bool = typer.Option(
        False,
        "--list",
        help="List available genome files instead of downloading"
    ),
    index: bool = typer.Option(
        True,
        "--index/--no-index",
        help="Create an uncompressed .fa and .fa.fai index for random-access tools",
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging"
    ),
):
    """
    Download Ensembl reference genome FASTA files.
    
    This command downloads genome sequences from the Ensembl FTP server.
    By default, it downloads the primary assembly (main chromosomes without
    alternative/patch sequences).
    
    Examples:
    
        # Download latest human primary assembly
        uv run prepare-annotations genome
        
        # Download specific Ensembl release
        uv run prepare-annotations genome --release 114
        
        # Download soft-masked toplevel genome
        uv run prepare-annotations genome --type toplevel --masking dna_sm
        
        # Download a single chromosome
        uv run prepare-annotations genome --type chromosome --chromosome 21
        
        # Download all chromosomes separately
        uv run prepare-annotations genome --all-chromosomes
        
        # Download specific chromosomes
        uv run prepare-annotations genome --all-chromosomes --chromosomes "1,2,X,Y"
        
        # List available files
        uv run prepare-annotations genome --list
        
        # Download mouse genome
        uv run prepare-annotations genome --species mus_musculus
    """
    from prepare_annotations.downloaders.genome import (
        GenomeType,
        MaskingType,
        download_ensembl_genome,
        download_all_chromosomes,
        list_available_genomes,
    )
    
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "download_genome.json", logs / "download_genome.log")
        to_nice_stdout()
    
    with start_action(action_type="download_genome_command") as action:
        # Parse enums
        try:
            genome_type_enum = GenomeType(genome_type)
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Invalid genome type '{genome_type}'. Valid options: primary_assembly, toplevel, chromosome")
            raise typer.Exit(1)
        
        try:
            masking_enum = MaskingType(masking)
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Invalid masking type '{masking}'. Valid options: dna, dna_sm, dna_rm")
            raise typer.Exit(1)
        
        use_http = not use_ftp
        cache_dir = Path(dest_dir) if dest_dir else None
        
        action.log(
            message_type="info",
            species=species,
            genome_type=genome_type,
            masking=masking,
            release=release,
            chromosome=chromosome,
            all_chromosomes=all_chromosomes,
            use_http=use_http,
        )
        
        # List mode
        if list_available:
            console.print(f"\n📋 Available genome files for [bold cyan]{species}[/bold cyan]")
            if release:
                console.print(f"   Release: [bold]{release}[/bold]")
            else:
                console.print("   Release: [bold]latest[/bold]")
            console.print()
            
            files = list_available_genomes(species, release, use_http)
            for f in sorted(files):
                filename = f.rsplit("/", 1)[-1]
                console.print(f"  • {filename}")
            
            console.print(f"\n  Total: [bold]{len(files)}[/bold] files")
            return
        
        # Validation
        if genome_type_enum == GenomeType.CHROMOSOME and not chromosome and not all_chromosomes:
            console.print("[bold red]Error:[/bold red] --chromosome is required when --type=chromosome (unless using --all-chromosomes)")
            raise typer.Exit(1)
        
        console.print(f"\n🧬 Downloading Ensembl genome")
        console.print(f"   Species: [bold cyan]{species}[/bold cyan]")
        console.print(f"   Type: [bold]{genome_type}[/bold]")
        console.print(f"   Masking: [bold]{masking}[/bold]")
        if release:
            console.print(f"   Release: [bold]{release}[/bold]")
        else:
            console.print("   Release: [bold]latest[/bold]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            if all_chromosomes:
                task_id = progress.add_task("Downloading chromosomes...", total=None)
                
                chr_list = None
                if chromosomes:
                    chr_list = [c.strip() for c in chromosomes.split(",")]
                
                downloaded = download_all_chromosomes(
                    species=species,
                    masking=masking_enum,
                    release=release,
                    cache_dir=cache_dir,
                    force_download=force,
                    create_fai=index,
                    use_http=use_http,
                    chromosomes=chr_list,
                )
                
                progress.update(task_id, description="✅ Downloads completed")
                
                console.print(f"\n✅ Downloaded {len(downloaded)} chromosome files:")
                for p in sorted(downloaded):
                    size_mb = p.stat().st_size / (1024 ** 2)
                    console.print(f"   📁 {p.name} ({size_mb:.1f} MB)")
            else:
                task_id = progress.add_task("Downloading genome...", total=None)
                
                downloaded = download_ensembl_genome(
                    species=species,
                    genome_type=genome_type_enum,
                    masking=masking_enum,
                    release=release,
                    chromosome=chromosome,
                    cache_dir=cache_dir,
                    force_download=force,
                    create_fai=index,
                    use_http=use_http,
                )
                
                progress.update(task_id, description="✅ Download completed")
                
                size_mb = downloaded.stat().st_size / (1024 ** 2)
                console.print(f"\n✅ Downloaded: [bold cyan]{downloaded}[/bold cyan]")
                console.print(f"   Size: [bold]{size_mb:.1f} MB[/bold]")
        
        action.log(message_type="success", step="genome_download_complete")


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
            "source": get_default_cache_dir("ensembl"),
            "card_gen": generate_ensembl_card
        },
        "clinvar": {
            "repo": "just-dna-seq/clinvar",
            "source": get_default_cache_dir("clinvar"),
            "card_gen": generate_clinvar_card
        },
        "dbsnp": {
            "repo": "just-dna-seq/dbsnp",
            "source": get_default_cache_dir("dbsnp_grch38"),
            "card_gen": generate_dbsnp_card
        },
        "dbsnp_t2t": {
            "repo": "just-dna-seq/dbsnp_t2t",
            "source": get_default_cache_dir("dbsnp_t2t"),
            "card_gen": generate_dbsnp_t2t_card
        },
        "gnomad": {
            "repo": "just-dna-seq/gnomad",
            "source": get_default_cache_dir("gnomad_v4"),
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
            source_path = get_default_cache_dir("dbsnp_grch37")
        elif dataset.lower() == "gnomad":
            source_path = get_default_cache_dir("gnomad_v3")
            
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



# ============================================================================
# DAGSTER COMMANDS
# ============================================================================

# ============================================================================
# DAGSTER INTEGRATION
# ============================================================================


def _get_dagster_home() -> Path:
    """Get or create DAGSTER_HOME directory (resolved relative to repo root).

    Mirrors the `just-dna-lite` approach:
    - If DAGSTER_HOME is set and is relative, interpret it relative to ROOT_DIR.
    - If DAGSTER_HOME is not set, default to ROOT_DIR/data/interim/dagster.
    - Always set DAGSTER_HOME to an absolute path to avoid Dagster writing into CWD.
    """

    def _resolve_dagster_home(root: Path, raw: str | None) -> Path:
        value = raw or "data/interim/dagster"
        p = Path(value)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p

    # Import at call-time so tests (and callers) can override core.paths.ROOT_DIR if needed.
    from prepare_annotations.core import paths as _resources

    dagster_home_path = _resolve_dagster_home(_resources.ROOT_DIR, os.environ.get("DAGSTER_HOME"))
    dagster_home_path.mkdir(parents=True, exist_ok=True)
    os.environ["DAGSTER_HOME"] = str(dagster_home_path)
    return dagster_home_path


def _kill_port_owner(port: int, host: str = "127.0.0.1") -> None:
    """Kill the process listening on host:port (best-effort)."""
    import signal
    import socket
    import subprocess

    # If nothing is listening, do nothing.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) != 0:
            return

    # Prefer lsof, fallback to fuser.
    result = subprocess.run(
        ["lsof", "-t", "-n", "-P", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = [p for p in result.stdout.strip().split() if p]
    if not pids:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            # fuser output: "3000/tcp:  1234 5678"
            tail = result.stdout.split(":")[-1].strip()
            pids = [p for p in tail.split() if p]

    for pid_str in pids:
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue


def _ensure_dagster_config(dagster_home: Path) -> None:
    """Create dagster.yaml if it doesn't exist."""
    config_file = dagster_home / "dagster.yaml"
    if config_file.exists():
        return
    
    dagster_home.mkdir(parents=True, exist_ok=True)
    config_content = """# Dagster instance configuration
auto_materialize:
  enabled: true
  minimum_interval_seconds: 60

telemetry:
  enabled: false
"""
    config_file.write_text(config_content, encoding="utf-8")
    console.print(f"   [green]✔[/green] Created Dagster config at {config_file}")


def _start_dagster_ui_background(port: int = 3000, host: str = "127.0.0.1"):
    """Start Dagster UI in background."""
    import subprocess
    import sys
    
    dagster_home = _get_dagster_home()
    _ensure_dagster_config(dagster_home)
    
    env = os.environ.copy()
    env["DAGSTER_HOME"] = str(dagster_home)
    
    cmd = [
        sys.executable, "-m", "dagster", "dev",
        "-m", "prepare_annotations.pipelines",
        "-p", str(port),
        "-h", host,
    ]
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )


def _ensure_dagster_ui_running(
    port: int = 3000,
    host: str = "127.0.0.1",
    *,
    force_restart: bool = True,
) -> bool:
    """Ensure Dagster UI is running (and points at our DAGSTER_HOME).

    We default to force-restarting any existing server on the port to avoid
    the common "UI is running but shows no runs" problem caused by mismatched
    DAGSTER_HOME between the UI process and the CLI process.
    """
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex((host, port)) == 0
    
    if is_running:
        if not force_restart:
            console.print(f"   [green]✔[/green] Dagster UI running at http://{host}:{port}")
            return False

        console.print(
            "   [yellow]↻[/yellow] Restarting Dagster UI to ensure it uses the same DAGSTER_HOME..."
        )
        _kill_port_owner(port=port, host=host)
    
    console.print(f"   [yellow]⏳[/yellow] Starting Dagster UI in background...")
    _start_dagster_ui_background(port, host)
    
    for _ in range(20):
        time.sleep(0.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                console.print(f"   [green]✔[/green] Dagster UI started at http://{host}:{port}")
                return True
    
    console.print(f"   [yellow]![/yellow] UI may still be starting...")
    return True


@app.command(name="ui")
def dagster_ui(
    port: int = typer.Option(3000, "--port", "-p", help="Port for Dagster webserver"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for Dagster webserver"),
):
    """
    Start Dagster web UI for interactive job execution and lineage visualization.
    """
    import subprocess
    import sys
    
    dagster_home = _get_dagster_home()
    _ensure_dagster_config(dagster_home)
    os.environ["DAGSTER_HOME"] = str(dagster_home)

    console.print("\n[bold cyan]🔷 Starting Dagster UI[/bold cyan]")
    console.print(f"   URL: http://{host}:{port}")
    console.print(f"   Dagster home: {dagster_home}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
    
    cmd = [
        sys.executable, "-m", "dagster", "dev",
        "-m", "prepare_annotations.pipelines",
        "-p", str(port),
        "-h", host,
    ]
    
    subprocess.run(cmd)


@app.command(name="materialize")
def dagster_materialize(
    assets: List[str] = typer.Argument(
        ...,
        help="Asset names to materialize (e.g., ensembl_vcf_urls ensembl_vcf_file)"
    ),
    partition: Optional[str] = typer.Option(
        None,
        "--partition", "-p",
        help="Partition key for partitioned assets"
    ),
):
    """
    Materialize specific Dagster assets using the Python API.
    
    Examples:
        # Materialize VCF URL discovery
        uv run prepare materialize ensembl_vcf_urls
        
        # Materialize a specific partition
        uv run prepare materialize ensembl_vcf_file -p homo_sapiens.vcf.gz
    """
    import os
    from dagster import DagsterInstance, materialize
    
    dagster_home = _get_dagster_home()
    _ensure_dagster_config(dagster_home)
    os.environ["DAGSTER_HOME"] = str(dagster_home)
    
    console.print(f"\n[bold cyan]🔷 Materializing Assets[/bold cyan]")
    console.print(f"   Assets: {', '.join(assets)}")
    if partition:
        console.print(f"   Partition: {partition}")
    console.print()
    
    # Import all assets to build the asset map
    from prepare_annotations.assets import (
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_with_ensembl,
        longevitymap_hf_upload,
    )
    from prepare_annotations.core.dagster_io_managers import (
        ensembl_cache_io_manager,
        huggingface_upload_io_manager,
    )
    
    asset_map = {
        "ensembl_vcf_urls": ensembl_vcf_urls,
        "ensembl_vcf_file": ensembl_vcf_file,
        "ensembl_parquet_file": ensembl_parquet_file,
        "ensembl_all_parquet_files": ensembl_all_parquet_files,
        "ensembl_hf_upload": ensembl_hf_upload,
        "ensembl_variations_source": ensembl_variations_source,
        "longevitymap_annotations": longevitymap_annotations,
        "longevitymap_studies": longevitymap_studies,
        "longevitymap_weights": longevitymap_weights,
        "longevitymap_with_ensembl": longevitymap_with_ensembl,
        "longevitymap_hf_upload": longevitymap_hf_upload,
    }
    
    # Resolve asset objects
    asset_objs = []
    for asset_name in assets:
        if asset_name not in asset_map:
            console.print(f"[bold red]❌ Unknown asset: {asset_name}[/bold red]")
            console.print(f"Available: {', '.join(asset_map.keys())}")
            raise typer.Exit(1)
        asset_objs.append(asset_map[asset_name])
    
    instance = DagsterInstance.get()
    resources = {
        "io_manager": ensembl_cache_io_manager,
        "hf_upload_io_manager": huggingface_upload_io_manager,
    }
    
    materialize_kwargs = {
        "assets": list(asset_map.values()),
        "selection": assets,
        "resources": resources,
        "instance": instance,
    }
    if partition:
        materialize_kwargs["partition_key"] = partition
    
    result = materialize(**materialize_kwargs)
    
    if result.success:
        console.print(f"\n[bold green]✅ Assets materialized successfully![/bold green]")
    else:
        console.print(f"\n[bold red]❌ Materialization failed![/bold red]")
        raise typer.Exit(1)


@app.command(name="job")
def dagster_job(
    job_name: str = typer.Argument(
        ...,
        help="Job: full, prepare, download, convert, upload, longevitymap"
    ),
    species: str = typer.Option(
        "homo_sapiens",
        "--species", "-s",
        help="Species for Ensembl jobs"
    ),
):
    """
    Execute a Dagster job by name using Python API.
    
    Ensembl jobs: full, prepare, download, convert, upload
    Module jobs: longevitymap, longevitymap_full, longevitymap_upload
    """
    ensembl_jobs = {"full", "prepare", "download", "convert", "upload"}
    
    if job_name in ensembl_jobs:
        # Use the proper partitioned asset handling
        _dagster_run_ensembl(job_name=job_name, species=species)
    else:
        # For non-partitioned jobs, use execute_job
        import os
        from dagster import DagsterInstance, execute_job
        
        dagster_home = _get_dagster_home()
        _ensure_dagster_config(dagster_home)
        os.environ["DAGSTER_HOME"] = str(dagster_home)
        
        from prepare_annotations.definitions import defs
        
        console.print(f"\n[bold cyan]🔷 Executing Job: {job_name}[/bold cyan]")
        
        job_def = defs.resolve_job_def(job_name)
        instance = DagsterInstance.get()
        
        result = execute_job(
            job_def,
            instance=instance,
            raise_on_error=True,
        )
        
        if result.success:
            console.print(f"\n[bold green]✅ Job '{job_name}' completed successfully![/bold green]")
        else:
            console.print(f"\n[bold red]❌ Job '{job_name}' failed![/bold red]")
            raise typer.Exit(1)


@app.command(name="assets")
def dagster_list_assets():
    """List all available Dagster assets."""
    console.print("\n[bold cyan]🔷 Available Dagster Assets[/bold cyan]\n")
    
    assets = [
        ("ensembl_ftp_source", "External", "Ensembl FTP server (source of truth)"),
        ("ensembl_vcf_urls", "Discovery", "Discovered VCF file URLs from Ensembl FTP"),
        ("ensembl_vcf_file", "Download", "Per-file VCF download (dynamically partitioned)"),
        ("ensembl_parquet_file", "Conversion", "Per-file VCF to Parquet conversion (dynamically partitioned)"),
        ("ensembl_all_parquet_files", "Collector", "Collect all parquet files for upload"),
        ("ensembl_hf_upload", "Upload", "Upload to HuggingFace Hub"),
    ]
    
    for name, kind, description in assets:
        console.print(f"  [bold green]{name}[/bold green] [{kind}]")
        console.print(f"      {description}")
        console.print()


@app.command(name="jobs")
def dagster_list_jobs():
    """List all available Dagster jobs."""
    console.print("\n[bold cyan]🔷 Available Dagster Jobs[/bold cyan]\n")
    
    jobs = [
        ("full", "Complete Ensembl pipeline: download → convert → upload"),
        ("prepare", "Ensembl: download and convert to Parquet (no splitting)"),
        ("download", "Ensembl: download VCF files from FTP"),
        ("convert", "Ensembl: convert VCF to Parquet"),
        ("upload", "Ensembl: upload to HuggingFace Hub"),
        ("longevitymap", "LongevityMap: convert to unified schema with Ensembl genotype resolution"),
        ("longevitymap_full", "LongevityMap: convert + join with full Ensembl data"),
        ("longevitymap_upload", "LongevityMap: convert + upload to just-dna-seq/annotators"),
    ]
    
    for name, description in jobs:
        console.print(f"  [bold green]{name}[/bold green]")
        console.print(f"      {description}")
        console.print()


@app.command(name="longevitymap")
def dagster_run_longevitymap(
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to longevitymap SQLite database"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for parquet files"
    ),
    ensembl_cache: Optional[str] = typer.Option(
        None,
        "--ensembl-cache",
        help="Path to local Ensembl cache. If not found, downloads from HuggingFace."
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run longevitymap_full job (includes Ensembl join)"
    ),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Upload to HuggingFace Hub (just-dna-seq/annotators)"
    ),
):
    """
    Convert LongevityMap to unified annotation schema with proper genotype expansion.
    
    This uses the new Dagster-based conversion which:
    - Expands homozygous variants: "C" -> ["C", "C"]
    - Expands heterozygous variants with Ensembl: "C" (het) -> ["C", "T"], ["C", "G"], etc.
    - Produces list[str] genotypes for parquet compatibility
    
    Ensembl data is sourced from:
    1. Local cache (if available from prior Ensembl pipeline run)
    2. HuggingFace Hub (just-dna-seq/ensembl_variations)
    
    Use --upload to upload results to just-dna-seq/annotators on HuggingFace.
    """
    from dagster import DagsterInstance
    
    dagster_home = _get_dagster_home()
    _ensure_dagster_config(dagster_home)
    
    # Determine which job to run based on options
    if upload:
        job_name = "longevitymap_upload"
    elif full:
        job_name = "longevitymap_full"
    else:
        job_name = "longevitymap"
    
    console.print(f"\n[bold cyan]🔷 Running Dagster Job: {job_name}[/bold cyan]")
    if db_path:
        console.print(f"   Database: [bold blue]{db_path}[/bold blue]")
    if output_dir:
        console.print(f"   Output: [bold blue]{output_dir}[/bold blue]")
    if ensembl_cache:
        console.print(f"   Ensembl cache: [bold blue]{ensembl_cache}[/bold blue]")
    console.print(f"   Dagster home: {dagster_home}")
    
    # Start UI in background if not running
    _ensure_dagster_ui_running(force_restart=False)
    
    console.print("\n🚀 Executing pipeline...\n")
    console.print("   Monitor progress at: http://127.0.0.1:3000\n")
    
    from prepare_annotations.definitions import defs
    
    job = defs.resolve_job_def(job_name)
    
    # Build run config
    run_config: dict = {"ops": {}}
    
    # Add config for longevitymap assets
    lm_config: dict = {}
    if db_path:
        lm_config["db_path"] = db_path
    if output_dir:
        lm_config["output_dir"] = output_dir
    
    if lm_config:
        run_config["ops"]["longevitymap_annotations"] = {"config": lm_config}
        run_config["ops"]["longevitymap_studies"] = {"config": lm_config}
        run_config["ops"]["longevitymap_weights"] = {"config": lm_config}
        if full:
            run_config["ops"]["longevitymap_with_ensembl"] = {"config": lm_config}
    
    # Add ensembl source config
    ensembl_config: dict = {}
    if ensembl_cache:
        ensembl_config["local_cache_path"] = ensembl_cache
    if ensembl_config:
        run_config["ops"]["ensembl_variations_source"] = {"config": ensembl_config}
    
    with DagsterInstance.get() as instance:
        result = job.execute_in_process(
            instance=instance,
            run_config=run_config if run_config["ops"] else None,
        )
    
    if result.success:
        console.print(f"\n[bold green]✅ Job '{job_name}' completed successfully![/bold green]")
        console.print("\nOutput files:")
        console.print("  - annotations.parquet: Variant-level facts")
        console.print("  - studies.parquet: Per-study evidence")
        console.print("  - weights.parquet: Genotype weights with Ensembl resolution")
        if full:
            console.print("  - longevitymap_ensembl_joined.parquet: Enriched with Ensembl data")
    else:
        console.print(f"\n[bold red]❌ Job '{job_name}' failed![/bold red]")
        raise typer.Exit(1)


def _register_dagster_commands() -> None:
    dagster_app.command()(ensembl)
    dagster_app.command(name="ui")(dagster_ui)
    dagster_app.command(name="materialize")(dagster_materialize)
    dagster_app.command(name="job")(dagster_job)
    dagster_app.command(name="assets")(dagster_list_assets)
    dagster_app.command(name="jobs")(dagster_list_jobs)
    dagster_app.command(name="longevitymap")(dagster_run_longevitymap)


_register_dagster_commands()


@app.command()
def version():
    """Show version information."""
    try:
        import importlib.metadata
        v = importlib.metadata.version("prepare-annotations")
        console.print(f"prepare-annotations version: [bold green]{v}[/bold green]")
    except importlib.metadata.PackageNotFoundError:
        console.print("prepare-annotations version: [yellow]development[/yellow]")


def _run_ensembl_cli():
    """Standalone entrypoint for dagster-ensembl script."""
    from prepare_annotations.cli import dagster_app
    import sys
    # Insert 'ensembl' as the first argument if not already there
    if len(sys.argv) > 1 and sys.argv[1] not in ["ensembl", "--help"]:
        sys.argv.insert(1, "ensembl")
    elif len(sys.argv) == 1:
        sys.argv.append("ensembl")
    dagster_app()


def _run_ui_cli():
    """Standalone entrypoint for dagster-ui script."""
    from prepare_annotations.cli import dagster_app
    import sys
    # Insert 'ui' as the first argument if not already there
    if len(sys.argv) > 1 and sys.argv[1] not in ["ui", "--help"]:
        sys.argv.insert(1, "ui")
    elif len(sys.argv) == 1:
        sys.argv.append("ui")
    dagster_app()


if __name__ == "__main__":
    app()

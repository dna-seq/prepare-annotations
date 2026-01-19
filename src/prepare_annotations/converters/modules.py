"""
Just DNA Pipelines Modules CLI - Manage OakVar modules from GitHub repositories.

This module provides a CLI interface for cloning and downloading data from OakVar modules.

Note: Module conversion to parquet format is now handled by Dagster assets.
Use `uv run prepare <module_name>` for conversion workflows (e.g., `uv run prepare longevitymap`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, FileSizeColumn, TimeRemainingColumn
from rich.table import Table
from eliot import start_action

from prepare_annotations.core.paths import LOGS_DIR, MODULES_DIR, MODULES_OUTPUT_DIR, get_cache_dir
from prepare_annotations.core.runtime import load_env
from prepare_annotations.core.models import ModuleManifest, ModuleDependency

logs = LOGS_DIR

load_env()

from pycomfort.logging import to_nice_file, to_nice_stdout

# Create the main CLI app
app = typer.Typer(
    name="modules",
    help="Manage OakVar modules from GitHub repositories",
    rich_markup_mode="rich",
    no_args_is_help=True
)

console = Console()


def normalize_repo_url(repo: str) -> str:
    """
    Normalize repository identifier to a full GitHub URL.
    
    Args:
        repo: Repository identifier (e.g., "dna-seq/just_longevitymap" or full URL)
        
    Returns:
        Full GitHub URL
    """
    if repo.startswith("http://") or repo.startswith("https://"):
        return repo
    elif repo.startswith("git@"):
        return repo
    elif "/" in repo:
        # Assume GitHub format: owner/repo
        return f"https://github.com/{repo}.git"
    else:
        raise ValueError(f"Invalid repository format: {repo}. Use 'owner/repo' or full URL")


def get_repo_name_from_url(repo_url: str) -> str:
    """
    Extract repository name from URL.
    
    Args:
        repo_url: Repository URL
        
    Returns:
        Repository name (owner/repo)
    """
    if repo_url.startswith("https://github.com/"):
        repo_path = repo_url.replace("https://github.com/", "").replace(".git", "")
        return repo_path
    elif repo_url.startswith("git@github.com:"):
        repo_path = repo_url.replace("git@github.com:", "").replace(".git", "")
        return repo_path
    else:
        # Fallback: use last part of URL
        return repo_url.split("/")[-1].replace(".git", "")


def clone_or_update_repo(repo_url: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Clone or update a git repository.
    
    Args:
        repo_url: Repository URL
        cache_dir: Cache directory for repositories. If None, uses platformdirs cache.
        
    Returns:
        Path to the cloned repository
        
    Raises:
        typer.Exit: If git operations fail
    """
    repo_name = get_repo_name_from_url(repo_url)
    
    if cache_dir is None:
        from pooch import os_cache
        cache_base = Path(os_cache("just-dna-pipelines"))
    else:
        cache_base = Path(cache_dir)
    
    repos_dir = cache_base / "repositories"
    repos_dir.mkdir(parents=True, exist_ok=True)
    
    repo_path = repos_dir / repo_name.replace("/", "_")
    
    if repo_path.exists() and (repo_path / ".git").exists():
        # Repository exists, try to update
        with start_action(action_type="update_repo", repo=repo_url, path=str(repo_path)):
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                # If pull fails, continue with existing version
                console.print(f"[yellow]Warning: Could not update repository: {e.stderr}[/yellow]")
    else:
        # Clone repository
        with start_action(action_type="clone_repo", repo=repo_url, path=str(repo_path)):
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error: Failed to clone repository: {e.stderr}[/red]")
                raise typer.Exit(code=1)
            except FileNotFoundError:
                console.print("[red]Error: git is not installed or not in PATH[/red]")
                raise typer.Exit(code=1)
    
    if not repo_path.exists():
        console.print(f"[red]Error: Repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(code=1)
    
    return repo_path


def load_module_manifest(module_dir: Path) -> Optional[ModuleManifest]:
    """Load module.yaml manifest if it exists."""
    manifest_path = module_dir / "module.yaml"
    if not manifest_path.exists():
        return None
    
    try:
        with open(manifest_path, "r") as f:
            data = yaml.safe_load(f)
            return ModuleManifest(**data)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load manifest at {manifest_path}: {e}[/yellow]")
        return None


def install_module_recursive(repo_url: str, dest_dir: Path, installed_urls: Optional[set[str]] = None):
    """Clone a module and its dependencies recursively."""
    if installed_urls is None:
        installed_urls = set()
    
    if repo_url in installed_urls:
        return
    
    installed_urls.add(repo_url)
    
    repo_name = get_repo_name_from_url(repo_url)
    module_dest = dest_dir / repo_name.split("/")[-1]
    
    # Clone current repo
    if not (module_dest.exists() and (module_dest / ".git").exists()):
        console.print(f"[bold cyan]Cloning {repo_url} to {module_dest}...[/bold cyan]")
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(module_dest)],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error: Failed to clone {repo_url}: {e.stderr}[/red]")
            return
    else:
        console.print(f"[yellow]Module already exists at: {module_dest}[/yellow]")

    # Check for dependencies in manifest
    manifest = load_module_manifest(module_dest)
    if manifest and manifest.dependencies:
        console.print(f"[bold blue]Installing {len(manifest.dependencies)} dependencies for {manifest.name}...[/bold blue]")
        for dep in manifest.dependencies:
            install_module_recursive(dep.url, dest_dir, installed_urls)


def find_files_by_extension(directory: Path, extensions: list[str], recursive: bool = True) -> list[Path]:
    """
    Find all files with the given extensions in the given directory.
    
    Args:
        directory: Directory to search in
        extensions: List of extensions to search for (e.g., [".sqlite", ".db"])
        recursive: Whether to search recursively
        
    Returns:
        List of paths to found files
    """
    found_files: list[Path] = []
    
    if not directory.exists():
        return found_files
    
    if recursive:
        for ext in extensions:
            found_files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            found_files.extend(directory.glob(f"*{ext}"))
    
    # Remove duplicates and sort
    return sorted(set(found_files))


@app.command()
def data(
    repo: Optional[str] = typer.Option(
        "dna-seq/just_longevitymap",
        "--repo",
        "-r",
        help="GitHub repository (owner/repo format or full URL). Default: dna-seq/just_longevitymap"
    ),
    extensions: list[str] = typer.Option(
        [".sqlite", ".sqlite3", ".db", ".tsv"],
        "--ext",
        "-e",
        help="File extensions to search for. Default: .sqlite, .sqlite3, .db, .tsv"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for downloaded files. Default: standard cache directory modules/reponame/"
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Cache directory for cloned repositories. Default: platform-specific cache"
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Search recursively in subdirectories"
    ),
    log: bool = typer.Option(
        False,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
):
    """
    Download data files from an OakVar module repository.
    
    Downloads files with specified extensions from the repository's data folder.
    By default, it searches for SQLite files (.sqlite, .sqlite3, .db).
    
    Examples:
    
        # Download SQLite files from default repository (dna-seq/just_longevitymap)
        modules data
        
        # Download specific file types
        modules data --ext .parquet --ext .csv
        
        # Download from a specific repository
        modules data --repo owner/repo-name
        
        # Download to a specific directory
        modules data --output-dir /path/to/output
        
        # Download non-recursively (only top-level files)
        modules data --no-recursive
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "modules_data.json", logs / "modules_data.log")
        to_nice_stdout()
    
    with start_action(action_type="modules_data_command", repo=repo, output_dir=output_dir, recursive=recursive) as action:
        # Clone or update repository and use its data folder
        repo_url = normalize_repo_url(repo)
        repo_name = get_repo_name_from_url(repo_url)
        console.print(f"[bold cyan]Cloning/updating repository: {repo}[/bold cyan]")
        
        cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else None
        repo_path = clone_or_update_repo(repo_url, cache_path)
        search_dir = repo_path / "data"
        
        if not search_dir.exists():
            console.print(f"[yellow]Warning: data folder does not exist in repository: {search_dir}[/yellow]")
            console.print(f"[yellow]Searching in repository root instead: {repo_path}[/yellow]")
            search_dir = repo_path
        
        console.print(f"[bold cyan]Searching for files with extensions {extensions} in: {search_dir}[/bold cyan]")
        
        found_files = find_files_by_extension(search_dir, extensions, recursive=recursive)
        
        if not found_files:
            console.print(f"[yellow]No files with extensions {extensions} found.[/yellow]")
            action.log(message_type="info", files_found=0)
            return
        
        # Determine output directory
        if output_dir:
            dest_dir = Path(output_dir).expanduser().resolve()
        else:
            # Default: data/modules/reponame/
            repo_basename = repo_name.split("/")[-1]  # Get just the repo name, not owner/repo
            dest_dir = (get_cache_dir() / "modules" / repo_basename).resolve()
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold cyan]Downloading to: {dest_dir}[/bold cyan]")
        
        # Download files with progress
        downloaded_files: list[Path] = []
        total_size = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            FileSizeColumn(),
            TextColumn("/"),
            FileSizeColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Downloading files...", total=sum(f.stat().st_size for f in found_files if f.exists()))
            
            for file_path in found_files:
                try:
                    file_size = file_path.stat().st_size
                    
                    # Preserve relative path structure from search directory
                    relative_path = file_path.relative_to(search_dir)
                    dest_file = dest_dir / relative_path
                    
                    # Skip if file already exists (same path)
                    if dest_file.exists():
                        # Check if it's the same file (same filesystem) or just skip if exists
                        try:
                            if dest_file.samefile(file_path):
                                console.print(f"[yellow]Skipping (already exists): {relative_path}[/yellow]")
                                continue
                        except OSError:
                            # Different filesystem, check if sizes match
                            if dest_file.stat().st_size == file_size:
                                console.print(f"[yellow]Skipping (already exists): {relative_path}[/yellow]")
                                continue
                    
                    # Create parent directories if needed
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(file_path, dest_file)
                    downloaded_files.append(dest_file)
                    total_size += file_size
                    progress.update(task, advance=file_size)
                    
                except OSError as e:
                    console.print(f"[red]Error copying {file_path.name}: {e}[/red]")
                    action.log(message_type="error", file=str(file_path), error=str(e))
        
        # Display results
        table = Table(title="Downloaded Files", show_header=True, header_style="bold magenta")
        table.add_column("File Name", style="cyan", no_wrap=False)
        table.add_column("Size", justify="right", style="green")
        table.add_column("Location", style="yellow")
        
        for file_path in downloaded_files:
            try:
                size = file_path.stat().st_size
                size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                # Get relative path if possible, otherwise use absolute
                try:
                    resolved_path = file_path.resolve()
                    location = str(resolved_path.relative_to(Path.cwd().resolve()))
                except ValueError:
                    # If not in subpath, use the path as-is
                    location = str(file_path)
                table.add_row(
                    file_path.name,
                    size_str,
                    location
                )
            except OSError:
                table.add_row(file_path.name, "N/A", str(file_path))
        
        console.print(table)
        console.print(f"\n[bold green]Downloaded {len(downloaded_files)} file(s)[/bold green]")
        console.print(f"[bold green]Total size: {total_size / (1024*1024):.2f} MB[/bold green]")
        console.print(f"[bold green]Saved to: {dest_dir}[/bold green]")
        
        action.log(
            message_type="success",
            files_downloaded=len(downloaded_files),
            total_size_bytes=total_size,
            output_directory=str(dest_dir)
        )


@app.command()
def clone(
    repo: Optional[str] = typer.Option(
        "dna-seq/just_longevitymap",
        "--repo",
        "-r",
        help="GitHub repository (owner/repo format or full URL). Default: dna-seq/just_longevitymap"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for cloned repository. Default: standard cache directory modules/reponame/"
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Cache directory for cloned repositories. Default: platform-specific cache"
    ),
    log: bool = typer.Option(
        False,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
):
    """
    Clone a full OakVar module repository.
    
    Clones the entire repository to the specified directory.
    
    Examples:
    
        # Clone default repository (dna-seq/just_longevitymap)
        modules clone
        
        # Clone a specific repository
        modules clone --repo owner/repo-name
        
        # Clone to a specific directory
        modules clone --output-dir /path/to/output
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "modules_clone.json", logs / "modules_clone.log")
        to_nice_stdout()
    
    with start_action(action_type="modules_clone_command", repo=repo, output_dir=output_dir) as action:
        repo_url = normalize_repo_url(repo)
        repo_name = get_repo_name_from_url(repo_url)
        console.print(f"[bold cyan]Cloning repository: {repo}[/bold cyan]")
        
        # Determine output directory
        if output_dir:
            dest_dir = Path(output_dir).expanduser().resolve()
        else:
            # Default: data/modules/reponame/
            repo_basename = repo_name.split("/")[-1]  # Get just the repo name, not owner/repo
            dest_dir = (get_cache_dir() / "modules" / repo_basename).resolve()
        
        # Check if destination already exists
        if dest_dir.exists() and (dest_dir / ".git").exists():
            console.print(f"[yellow]Repository already exists at: {dest_dir}[/yellow]")
            console.print("[yellow]Use 'git pull' to update, or remove the directory to re-clone[/yellow]")
            action.log(message_type="info", already_exists=True, path=str(dest_dir))
            return
        
        # Clone repository directly to destination
        with start_action(action_type="clone_repo", repo=repo_url, path=str(dest_dir)):
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(dest_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error: Failed to clone repository: {e.stderr}[/red]")
                action.log(message_type="error", error=str(e))
                raise typer.Exit(code=1)
            except FileNotFoundError:
                console.print("[red]Error: git is not installed or not in PATH[/red]")
                raise typer.Exit(code=1)
        
        console.print(f"[bold green]Repository cloned successfully to: {dest_dir}[/bold green]")
        action.log(
            message_type="success",
            output_directory=str(dest_dir),
            repo=repo
        )


@app.command()
def install(
    repo: str = typer.Argument(
        ...,
        help="GitHub repository (owner/repo format or full URL)."
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Root directory for modules. Default: standard cache directory modules/"
    ),
    log: bool = typer.Option(
        False,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
):
    """
    Install a module and all its dependencies recursively.
    
    Example:
        modules install dna-seq/longevity-app
    """
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "modules_install.json", logs / "modules_install.log")
        to_nice_stdout()
    
    with start_action(action_type="modules_install_command", repo=repo, output_dir=output_dir) as action:
        repo_url = normalize_repo_url(repo)
        
        # Determine root modules directory
        if output_dir:
            dest_dir = Path(output_dir).expanduser().resolve()
        else:
            dest_dir = get_cache_dir() / "modules"
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        install_module_recursive(repo_url, dest_dir)
        
        console.print(f"\n[bold green]Installation complete. Check {dest_dir} for results.[/bold green]")
        console.print("[bold yellow]Tip: If this is a Dagster module, reload your workspace to see changes.[/bold yellow]")
        
        action.log(
            message_type="success",
            repo=repo,
            output_directory=str(dest_dir)
        )


@app.command()
def convert_drugs(
    tsv_path: Optional[str] = typer.Option(
        None,
        "--tsv-path",
        help="Path to Drugs TSV file. Defaults to data/modules/just_drugs/annotation_tab.tsv"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for parquet files. Defaults to data/output/modules/drugs/"
    ),
    curator: str = typer.Option(
        "PharmGKB",
        "--curator",
        help="Curator name for weights provenance"
    ),
    method: str = typer.Option(
        "pharmacogenomics_db",
        "--method",
        help="Curation method for weights provenance"
    ),
    log: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Enable detailed logging to files"
    ),
):
    """
    Convert Drugs (PharmGKB) to unified annotation schema (three parquet files).
    
    Note: This module uses TSV format and may not have complete genotype information.
    
    Note: This is the only convert command here because drugs is not yet in Dagster.
    For other modules, use the Dagster-based commands:
        uv run prepare longevitymap --convert-only
        uv run prepare lipidmetabolism --convert-only
        uv run prepare vo2max --convert-only
        uv run prepare superhuman --convert-only
        uv run prepare coronary --convert-only
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from prepare_annotations.converters.drugs import convert_drugs as do_convert
    
    if log:
        logs.mkdir(exist_ok=True, parents=True)
        to_nice_file(logs / "convert_drugs.json", logs / "convert_drugs.log")
        to_nice_stdout()
    
    with start_action(action_type="convert_drugs_command") as action:
        if tsv_path is None:
            tsv_path_resolved = MODULES_DIR / "just_drugs" / "annotation_tab.tsv"
        else:
            tsv_path_resolved = Path(tsv_path)
        
        if output_dir is None:
            output_dir_resolved = MODULES_OUTPUT_DIR / "drugs"
        else:
            output_dir_resolved = Path(output_dir)
        
        if not tsv_path_resolved.exists():
            console.print(f"[bold red]Error:[/bold red] TSV file not found: {tsv_path_resolved}")
            raise typer.Exit(1)
        
        console.print(f"📁 TSV File: [bold blue]{tsv_path_resolved}[/bold blue]")
        console.print(f"📦 Output: [bold blue]{output_dir_resolved}[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Converting Drugs to unified schema...", total=None)
            outputs = do_convert(
                tsv_path=tsv_path_resolved,
                output_dir=output_dir_resolved,
                curator=curator,
                method=method,
            )
        
        console.print(f"\n✅ Conversion completed!")
        for name, path in outputs.items():
            file_size_mb = path.stat().st_size / (1024 ** 2)
            console.print(f"  - [bold cyan]{name}.parquet[/bold cyan]: {file_size_mb:.2f} MB")
        
        action.log(message_type="success", outputs={k: str(v) for k, v in outputs.items()})

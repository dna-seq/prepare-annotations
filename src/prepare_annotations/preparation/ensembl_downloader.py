"""
Ensembl annotation downloader.
"""

from pathlib import Path
from typing import Optional, List
from eliot import start_action
from huggingface_hub import snapshot_download
from platformdirs import user_cache_dir

def get_default_ensembl_cache_dir() -> Path:
    """Get the default cache directory for Ensembl annotations."""
    return Path(user_cache_dir(appname="just-dna-pipelines")) / "ensembl"

def download_ensembl_annotations(
    repo_id: str = "just-dna-seq/ensembl_variations",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
) -> Path:
    """
    Download Ensembl variation annotations from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Local directory to store the dataset
        force_download: Whether to force download even if cache exists
        token: HuggingFace API token
        allow_patterns: List of patterns to download (e.g. ["data/**/*.parquet"])
        
    Returns:
        Path to the downloaded dataset
    """
    if cache_dir is None:
        cache_dir = get_default_ensembl_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        
    with start_action(action_type="download_ensembl_annotations", repo_id=repo_id, cache_dir=str(cache_dir)) as action:
        if cache_dir.exists() and not force_download:
            parquet_files = list(cache_dir.rglob("*.parquet"))
            if parquet_files:
                action.log(message_type="info", step="using_cache", num_files=len(parquet_files))
                return cache_dir
                
        if allow_patterns is None:
            allow_patterns = ["data/**/*.parquet"]
            
        action.log(message_type="info", step="downloading", repo_id=repo_id)
        
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=allow_patterns,
        )
        
        return Path(downloaded_path)


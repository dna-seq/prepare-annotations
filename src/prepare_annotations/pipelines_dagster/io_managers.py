"""
Dagster IO Managers for genomic data preparation pipelines.

IO Managers handle the persistence and loading of asset data.
They determine where assets are stored and how they are loaded.
"""

from pathlib import Path
from typing import Any

from dagster import IOManager, io_manager, InputContext, OutputContext

from prepare_annotations.pipelines_dagster.resources import (
    get_cache_dir,
    get_default_ensembl_cache_dir,
    get_output_dir,
)


class EnsemblCacheIOManager(IOManager):
    """
    IO Manager for Ensembl VCF and Parquet assets stored in the cache folder.
    
    All Ensembl data lives in:
    ~/.cache/just-dna-pipelines/ensembl/{species}/
    
    This allows:
    - Data persistence across Dagster restarts
    - Sharing cache across projects
    - Lazy materialization (skip if exists)
    """
    
    def __init__(self, species: str = "homo_sapiens"):
        self.species = species
    
    def _get_asset_path(self, asset_key: str) -> Path:
        """Get the cache path for a given asset."""
        base = get_default_ensembl_cache_dir(self.species)
        
        if asset_key == "ensembl_vcf_urls":
            return base / "vcf_urls.json"
        elif asset_key == "ensembl_vcf_files":
            return base / "vcf"
        elif asset_key == "ensembl_parquet_files":
            return base
        
        return base / asset_key
    
    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Asset was materialized - data already on disk, just log."""
        if isinstance(obj, Path):
            context.log.info(f"Ensembl asset stored at: {obj}")
        elif isinstance(obj, list):
            context.log.info(f"Ensembl asset stored {len(obj)} items")
        else:
            context.log.info(f"Ensembl asset materialized: {type(obj).__name__}")
    
    def load_input(self, context: InputContext) -> Path:
        """Load asset by returning its cache path."""
        asset_key = context.upstream_output.asset_key.to_user_string() if context.upstream_output else "unknown"
        cache_path = self._get_asset_path(asset_key)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Ensembl cache not found at {cache_path}. "
                f"Materialize the {asset_key} asset first."
            )
        
        context.log.info(f"Loading Ensembl data from cache: {cache_path}")
        return cache_path


class HuggingFaceUploadIOManager(IOManager):
    """
    IO Manager for HuggingFace upload results.
    
    Tracks upload operations and their metadata without storing large data locally.
    """
    
    def _get_upload_log_path(self, asset_key: str) -> Path:
        """Get the path for upload logs."""
        return get_output_dir() / "uploads" / f"{asset_key}.json"
    
    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Upload completed - log the result."""
        context.log.info(f"HuggingFace upload completed: {obj}")
    
    def load_input(self, context: InputContext) -> dict:
        """Load upload metadata."""
        asset_key = context.upstream_output.asset_key.to_user_string() if context.upstream_output else "unknown"
        log_path = self._get_upload_log_path(asset_key)
        
        if log_path.exists():
            import json
            return json.loads(log_path.read_text())
        
        return {"asset_key": asset_key, "status": "not_found"}


@io_manager
def ensembl_cache_io_manager() -> EnsemblCacheIOManager:
    """IO manager for Ensembl VCF/Parquet assets in cache folder."""
    return EnsemblCacheIOManager()


@io_manager
def huggingface_upload_io_manager() -> HuggingFaceUploadIOManager:
    """IO manager for HuggingFace upload tracking."""
    return HuggingFaceUploadIOManager()

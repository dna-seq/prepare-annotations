"""
Dagster IO Managers for genomic data preparation pipelines.

IO Managers handle the persistence and loading of asset data.
They determine where assets are stored and how they are loaded.
"""

from pathlib import Path
from typing import Any

from dagster import IOManager, io_manager, InputContext, OutputContext

from prepare_annotations.core.io import _default_parquet_path
from prepare_annotations.core.paths import (
    get_default_ensembl_cache_dir,
    get_output_dir,
    MODULES_DIR,
    MODULES_OUTPUT_DIR,
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
        elif asset_key == "ensembl_vcf_file":
            # Partitioned asset - path is per-file in vcf/
            return base / "vcf"
        elif asset_key == "ensembl_parquet_file":
            # Partitioned asset - parquet files in species_dir
            return base
        elif asset_key == "ensembl_all_parquet_files":
            # Collector asset - species directory containing all parquet files
            return base
        
        return base / asset_key
    
    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Asset was materialized - data already on disk, just log."""
        asset_key = context.asset_key.to_user_string()
        if asset_key == "ensembl_variations_source":
            cache_path = self._get_asset_path(asset_key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(str(obj))
            context.log.info(f"Ensembl source recorded at: {cache_path}")
            return
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

        if asset_key == "ensembl_vcf_file" and context.partition_key:
            cache_path = cache_path / context.partition_key
        elif asset_key == "ensembl_parquet_file" and context.partition_key:
            parquet_name = _default_parquet_path(Path(context.partition_key)).name
            cache_path = cache_path / parquet_name
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Ensembl cache not found at {cache_path}. "
                f"Materialize the {asset_key} asset first."
            )
        if asset_key == "ensembl_variations_source":
            context.log.info(f"Loading Ensembl source from cache: {cache_path}")
            return cache_path.read_text().strip()

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


class ModuleIOManager(IOManager):
    """
    IO Manager for OakVar module assets.
    
    Handles paths for:
    - Module SQLite databases in data/modules/just_{module}/
    - Converted parquet files in data/output/modules/{module}/
    """
    
    def _get_asset_path(self, asset_key: str) -> Path:
        """Resolve the path for a module asset based on its key."""
        if asset_key.endswith("_sqlite"):
            module = asset_key.replace("_sqlite", "")
            return MODULES_DIR / f"just_{module}" / f"{module}.sqlite"
        
        # Handle longevitymap_with_ensembl specifically
        if asset_key == "longevitymap_with_ensembl":
            return MODULES_OUTPUT_DIR / "longevitymap" / "longevitymap_ensembl_joined.parquet"
            
        # Standard module assets: {module}_{type} (e.g., longevitymap_annotations)
        parts = asset_key.split("_")
        if len(parts) >= 2:
            module = parts[0]
            type_name = "_".join(parts[1:])
            return MODULES_OUTPUT_DIR / module / f"{type_name}.parquet"
            
        return MODULES_OUTPUT_DIR / asset_key

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Log where the asset was stored."""
        if isinstance(obj, Path):
            context.log.info(f"Module asset stored at: {obj}")
        else:
            context.log.info(f"Module asset materialized: {type(obj).__name__}")

    def load_input(self, context: InputContext) -> Any:
        """Load asset by returning its expected path."""
        asset_key = context.upstream_output.asset_key.to_user_string() if context.upstream_output else "unknown"
        path = self._get_asset_path(asset_key)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Module asset not found at {path}. "
                f"Materialize the {asset_key} asset first."
            )
            
        context.log.info(f"Loading module data from: {path}")
        return path


@io_manager
def ensembl_cache_io_manager() -> EnsemblCacheIOManager:
    """IO manager for Ensembl VCF/Parquet assets in cache folder."""
    return EnsemblCacheIOManager()


@io_manager
def module_io_manager() -> ModuleIOManager:
    """IO manager for OakVar module assets."""
    return ModuleIOManager()


@io_manager
def huggingface_upload_io_manager() -> HuggingFaceUploadIOManager:
    """IO manager for HuggingFace upload tracking."""
    return HuggingFaceUploadIOManager()

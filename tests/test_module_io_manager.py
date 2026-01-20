"""
Tests for ModuleIOManager path registry.

The IO manager now properly stores the paths returned by assets in a registry,
instead of trying to guess/reconstruct paths from asset keys.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prepare_annotations.core.dagster_io_managers import ModuleIOManager


@pytest.fixture
def io_manager(tmp_path: Path) -> ModuleIOManager:
    """Create an IO manager with a temp registry path."""
    manager = ModuleIOManager()
    manager._registry_path = tmp_path / ".asset_paths.json"
    return manager


class TestModuleIOManagerRegistry:
    """Tests for the path registry functionality."""

    def test_handle_output_stores_path_in_registry(
        self,
        io_manager: ModuleIOManager,
        tmp_path: Path,
    ) -> None:
        """handle_output should store the returned Path in the registry."""
        # Create a mock output context
        context = MagicMock()
        context.asset_key.to_user_string.return_value = "coronary_with_ensembl"
        
        # The asset returns this path
        asset_output_path = tmp_path / "coronary" / "coronary_ensembl_joined.parquet"
        asset_output_path.parent.mkdir(parents=True, exist_ok=True)
        asset_output_path.touch()
        
        # Handle the output
        io_manager.handle_output(context, asset_output_path)
        
        # Registry should contain the path
        registry = io_manager._load_registry()
        assert "coronary_with_ensembl" in registry
        assert registry["coronary_with_ensembl"] == str(asset_output_path)

    def test_load_input_retrieves_path_from_registry(
        self,
        io_manager: ModuleIOManager,
        tmp_path: Path,
    ) -> None:
        """load_input should retrieve the path from registry."""
        # Create a file
        asset_path = tmp_path / "longevitymap" / "annotations.parquet"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.touch()
        
        # Manually store in registry (simulating prior handle_output)
        registry = {"longevitymap_annotations": str(asset_path)}
        io_manager._save_registry(registry)
        
        # Create mock input context
        context = MagicMock()
        context.upstream_output.asset_key.to_user_string.return_value = "longevitymap_annotations"
        
        # Load should return the path
        result = io_manager.load_input(context)
        assert result == asset_path

    def test_load_input_raises_if_not_in_registry(
        self,
        io_manager: ModuleIOManager,
    ) -> None:
        """load_input should raise FileNotFoundError if asset not in registry."""
        context = MagicMock()
        context.upstream_output.asset_key.to_user_string.return_value = "unknown_asset"
        
        with pytest.raises(FileNotFoundError, match="not found in registry"):
            io_manager.load_input(context)

    def test_registry_persists_across_instances(
        self,
        tmp_path: Path,
    ) -> None:
        """Registry should persist to disk and be readable by new instances."""
        registry_path = tmp_path / ".asset_paths.json"
        
        # First instance stores a path
        manager1 = ModuleIOManager()
        manager1._registry_path = registry_path
        
        asset_path = tmp_path / "vo2max" / "weights.parquet"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.touch()
        
        context = MagicMock()
        context.asset_key.to_user_string.return_value = "vo2max_weights"
        manager1.handle_output(context, asset_path)
        
        # New instance should be able to load
        manager2 = ModuleIOManager()
        manager2._registry_path = registry_path
        
        context2 = MagicMock()
        context2.upstream_output.asset_key.to_user_string.return_value = "vo2max_weights"
        
        result = manager2.load_input(context2)
        assert result == asset_path

    def test_multiple_assets_stored_in_registry(
        self,
        io_manager: ModuleIOManager,
        tmp_path: Path,
    ) -> None:
        """Multiple assets should be stored in the same registry."""
        # Store multiple assets
        assets = {
            "coronary_annotations": tmp_path / "coronary" / "annotations.parquet",
            "coronary_studies": tmp_path / "coronary" / "studies.parquet",
            "coronary_with_ensembl": tmp_path / "coronary" / "coronary_ensembl_joined.parquet",
        }
        
        for asset_key, path in assets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            
            context = MagicMock()
            context.asset_key.to_user_string.return_value = asset_key
            io_manager.handle_output(context, path)
        
        # All should be in registry
        registry = io_manager._load_registry()
        assert len(registry) == 3
        for asset_key, path in assets.items():
            assert registry[asset_key] == str(path)


class TestModuleIOManagerNonPathOutputs:
    """Tests for non-Path outputs (e.g., dicts from upload assets)."""

    def test_handle_output_ignores_non_path_objects(
        self,
        io_manager: ModuleIOManager,
    ) -> None:
        """Non-Path outputs should not be stored in registry."""
        context = MagicMock()
        context.asset_key.to_user_string.return_value = "coronary_hf_upload"
        
        # Upload assets return dicts, not Paths
        upload_result = {"repo_id": "just-dna-seq/annotators", "num_uploaded": 3}
        io_manager.handle_output(context, upload_result)
        
        # Should not be in registry
        registry = io_manager._load_registry()
        assert "coronary_hf_upload" not in registry

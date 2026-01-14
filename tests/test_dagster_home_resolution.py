from __future__ import annotations

import os
from pathlib import Path


def test_dagster_home_is_absolute_and_root_relative(monkeypatch, tmp_path: Path) -> None:
    # Simulate repo root by monkeypatching prepare_annotations.paths.ROOT_DIR at runtime.
    # We don't want to rely on the real workspace layout in unit tests.
    import prepare_annotations.paths as paths
    import prepare_annotations.cli as cli

    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path, raising=True)

    # Ensure env is clean
    monkeypatch.delenv("DAGSTER_HOME", raising=False)

    dagster_home = cli._get_dagster_home()
    assert dagster_home.is_absolute()
    assert dagster_home == (tmp_path / "data" / "interim" / "dagster").resolve()
    assert os.environ["DAGSTER_HOME"] == str(dagster_home)


def test_relative_dagster_home_env_is_resolved_against_root(monkeypatch, tmp_path: Path) -> None:
    import prepare_annotations.paths as paths
    import prepare_annotations.cli as cli

    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path, raising=True)
    monkeypatch.setenv("DAGSTER_HOME", "data/interim/dagster_custom")

    dagster_home = cli._get_dagster_home()
    assert dagster_home.is_absolute()
    assert dagster_home == (tmp_path / "data" / "interim" / "dagster_custom").resolve()
    assert os.environ["DAGSTER_HOME"] == str(dagster_home)

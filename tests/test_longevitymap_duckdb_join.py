from __future__ import annotations

from pathlib import Path

import polars as pl

from prepare_annotations.pipelines.configs import DuckDBConfig
from prepare_annotations.pipelines.module_assets import (
    join_longevitymap_with_ensembl_duckdb,
)


def _write_ensembl_parquet(path: Path) -> None:
    df = pl.DataFrame(
        {
            "id": ["rs1", "rs2", "rs3"],
            "chrom": ["1", "1", "2"],
            "start": [100, 200, 300],
            "end": [101, 201, 301],
            "ref": ["A", "C", "G"],
            "alts": [["G"], ["T"], ["A", "C"]],
            "ClinVar_202502": ["pathogenic", None, None],
            "CLIN_pathogenic": [True, False, False],
            "CLIN_benign": [False, True, False],
            "CLIN_likely_pathogenic": [False, False, False],
            "CLIN_likely_benign": [False, False, False],
        }
    )
    df.write_parquet(path)


def _write_weights_parquet(path: Path) -> None:
    df = pl.DataFrame(
        {
            "rsid": ["rs1", "rs2", "rs3"],
            "genotype": [["A", "G"], ["C", "C"], ["T", "T"]],
            "module": ["longevitymap", "longevitymap", "longevitymap"],
            "weight": [1.2, -0.5, 0.7],
            "state": ["protective", "risk", "protective"],
            "priority": [1, 2, 3],
            "conclusion": [None, None, None],
            "curator": ["test", "test", "test"],
            "method": ["unit_test", "unit_test", "unit_test"],
        }
    )
    df.write_parquet(path)


def test_duckdb_longevitymap_join_filters_by_alleles(temp_dir: Path) -> None:
    ensembl_path = temp_dir / "ensembl.parquet"
    weights_path = temp_dir / "weights.parquet"
    output_path = temp_dir / "joined.parquet"

    _write_ensembl_parquet(ensembl_path)
    _write_weights_parquet(weights_path)

    row_count = join_longevitymap_with_ensembl_duckdb(
        weights_path=weights_path,
        ensembl_files=[str(ensembl_path)],
        output_path=output_path,
        duckdb_config=DuckDBConfig(),
    )

    result = pl.read_parquet(output_path)

    assert row_count == result.height
    assert result.filter(pl.col("rsid") == "rs3").is_empty()

    rs1_genotype = result.filter(pl.col("rsid") == "rs1").select("genotype").item()
    rs2_genotype = result.filter(pl.col("rsid") == "rs2").select("genotype").item()

    assert sorted(rs1_genotype) == ["A", "G"]
    assert sorted(rs2_genotype) == ["C", "C"]

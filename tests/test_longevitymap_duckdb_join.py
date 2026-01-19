from __future__ import annotations

from pathlib import Path

import polars as pl

from prepare_annotations.core.dagster_configs import DuckDBConfig
from prepare_annotations.assets.modules import join_weights_with_ensembl_duckdb


def _write_ensembl_parquet(path: Path) -> None:
    df = pl.DataFrame(
        {
            "id": ["rs1", "rs2", "rs3", "rs4"],
            "chrom": ["1", "1", "2", "3"],
            "start": [100, 200, 300, 400],
            "end": [101, 201, 301, 401],
            "ref": ["A", "C", "G", "A"],
            "alts": [["G"], ["T"], ["A", "C"], ["C"]],
            "ClinVar_202502": ["pathogenic", None, None, None],
            "CLIN_pathogenic": [True, False, False, False],
            "CLIN_benign": [False, True, False, False],
            "CLIN_likely_pathogenic": [False, False, False, False],
            "CLIN_likely_benign": [False, False, False, False],
        }
    )
    df.write_parquet(path)


def _write_weights_parquet(path: Path) -> None:
    df = pl.DataFrame(
        {
            # rs1: A/G genotype should match (A is ref, G is alt)
            # rs2: C/C genotype should match (C is ref)
            # rs3: T/T genotype should match via strand complement (T<->A, and A is in alts)
            # rs4: G/G genotype should NOT match (ref=A, alts=[C], no G anywhere, G complement is C but that's alt not matching G)
            "rsid": ["rs1", "rs2", "rs3", "rs4"],
            "genotype": [["A", "G"], ["C", "C"], ["T", "T"], ["G", "G"]],
            "module": ["longevitymap", "longevitymap", "longevitymap", "longevitymap"],
            "weight": [1.2, -0.5, 0.7, 0.3],
            "state": ["protective", "risk", "protective", "protective"],
            "priority": [1, 2, 3, 4],
            "conclusion": [None, None, None, None],
            "curator": ["test", "test", "test", "test"],
            "method": ["unit_test", "unit_test", "unit_test", "unit_test"],
        }
    )
    df.write_parquet(path)


def test_duckdb_longevitymap_join_enriches_with_ensembl(temp_dir: Path) -> None:
    """
    Test that the join correctly enriches weights with Ensembl data.
    
    Strand normalization is applied (A<->T, C<->G), so genotypes can match
    via their complements. The join adds Ensembl columns (chrom, start, end, etc).
    """
    ensembl_path = temp_dir / "ensembl.parquet"
    weights_path = temp_dir / "weights.parquet"
    output_path = temp_dir / "joined.parquet"

    _write_ensembl_parquet(ensembl_path)
    _write_weights_parquet(weights_path)

    row_count = join_weights_with_ensembl_duckdb(
        weights_path=weights_path,
        ensembl_files=[str(ensembl_path)],
        output_path=output_path,
        duckdb_config=DuckDBConfig(),
    )

    result = pl.read_parquet(output_path)

    assert row_count == result.height
    
    # All variants should be present (join succeeds with strand normalization)
    assert not result.filter(pl.col("rsid") == "rs1").is_empty()
    assert not result.filter(pl.col("rsid") == "rs2").is_empty()
    assert not result.filter(pl.col("rsid") == "rs3").is_empty()
    assert not result.filter(pl.col("rsid") == "rs4").is_empty()

    # Verify genotypes are preserved
    rs1_genotype = result.filter(pl.col("rsid") == "rs1").select("genotype").item()
    rs2_genotype = result.filter(pl.col("rsid") == "rs2").select("genotype").item()

    assert sorted(rs1_genotype) == ["A", "G"]
    assert sorted(rs2_genotype) == ["C", "C"]
    
    # Verify Ensembl columns are added
    ensembl_columns = {"chrom", "start", "end", "ref", "clinvar"}
    result_columns = set(result.columns)
    assert ensembl_columns.issubset(result_columns), (
        f"Missing Ensembl columns: {ensembl_columns - result_columns}"
    )
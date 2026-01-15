#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/vo2max parquet files
correctly preserve data from the original vo2max.sqlite database.

This test module will automatically download the vo2max data from
GitHub if it doesn't exist locally.
"""

import subprocess
import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data
from prepare_annotations.resources import MODULES_DIR, MODULES_OUTPUT_DIR


# Paths to data files
SQLITE_PATH = MODULES_DIR / "just_vo2max" / "vo2max.sqlite"
PARQUET_DIR = MODULES_OUTPUT_DIR / "vo2max"


@pytest.fixture(scope="module")
def ensure_vo2max_data() -> Path:
    """
    Ensure vo2max SQLite data exists, downloading if necessary.
    """
    ensure_oakvar_module_data(
        module_name="just_vo2max",
        output_dir=SQLITE_PATH.parent,
        expected_file="vo2max.sqlite",
    )
    
    if not SQLITE_PATH.exists():
        pytest.skip(f"Failed to download vo2max data: {SQLITE_PATH}")
    
    return SQLITE_PATH


@pytest.fixture(scope="module")
def ensure_vo2max_parquet(ensure_vo2max_data: Path) -> Path:
    """
    Ensure vo2max parquet files exist, converting if necessary.
    """
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            [
                "uv", "run", "modules", "convert-vo2max",
                "--db-path", str(ensure_vo2max_data),
                "--output-dir", str(PARQUET_DIR),
                "--no-log",
            ],
            check=True,
            capture_output=False,
        )
    
    if not weights_path.exists():
        pytest.skip(f"Failed to convert vo2max data: {weights_path}")
    
    return PARQUET_DIR


@pytest.fixture(scope="module")
def sqlite_connection(ensure_vo2max_data: Path):
    """Create a SQLite connection for the test module."""
    conn = sqlite3.connect(ensure_vo2max_data)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def weights_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load weights data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            rsID as rsid,
            genotype,
            weight,
            state,
            genotype_specific_conclusion as conclusion
        FROM genotype_weights
        """,
        sqlite_connection,
    )


@pytest.fixture(scope="module")
def weights_parquet(ensure_vo2max_parquet: Path) -> pl.DataFrame:
    """Load weights data from parquet."""
    parquet_file = ensure_vo2max_parquet / "weights.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def annotations_parquet(ensure_vo2max_parquet: Path) -> pl.DataFrame:
    """Load annotations data from parquet."""
    parquet_file = ensure_vo2max_parquet / "annotations.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_parquet(ensure_vo2max_parquet: Path) -> pl.DataFrame:
    """Load studies data from parquet."""
    parquet_file = ensure_vo2max_parquet / "studies.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def rsids_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load rsids data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            rsid,
            gene,
            pmids,
            population,
            p_value,
            rsid_conclusion
        FROM rsid
        """,
        sqlite_connection,
    )


class TestWeightsRowCount:
    """Test that weights table row counts match."""

    def test_total_row_count_reasonable(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify parquet has reasonable row count (may deduplicate)."""
        assert len(weights_parquet) > 0
        assert len(weights_parquet) <= len(weights_sqlite)

    def test_unique_rsid_count_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify unique rsid count is identical."""
        sqlite_unique = weights_sqlite["rsid"].n_unique()
        parquet_unique = weights_parquet["rsid"].n_unique()
        assert sqlite_unique == parquet_unique, (
            f"Unique rsid count mismatch: SQLite has {sqlite_unique}, "
            f"Parquet has {parquet_unique}"
        )


class TestWeightValues:
    """Test that weight values are correctly preserved."""

    def test_weight_min_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify minimum weight is identical."""
        sqlite_min = weights_sqlite["weight"].cast(pl.Float64, strict=False).min()
        parquet_min = weights_parquet["weight"].min()
        assert sqlite_min == parquet_min

    def test_weight_max_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify maximum weight is identical."""
        sqlite_max = weights_sqlite["weight"].cast(pl.Float64, strict=False).max()
        parquet_max = weights_parquet["weight"].max()
        assert sqlite_max == parquet_max


class TestAnnotationsTable:
    """Test that annotations data is correctly derived."""

    def test_annotations_have_gene(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify annotations have gene information."""
        assert "gene" in annotations_parquet.columns
        # At least some genes should be present
        non_null_genes = annotations_parquet["gene"].drop_nulls()
        assert len(non_null_genes) > 0

    def test_module_column_correct(
        self, weights_parquet: pl.DataFrame, annotations_parquet: pl.DataFrame
    ):
        """Verify module column is 'vo2max'."""
        assert all(weights_parquet["module"] == "vo2max")
        assert all(annotations_parquet["module"] == "vo2max")

    def test_phenotype_is_athletic_performance(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify phenotype is set correctly."""
        assert all(annotations_parquet["phenotype"] == "athletic_performance")


class TestStudiesTable:
    """Test that studies data is correctly preserved."""

    def test_all_rsids_preserved(
        self, rsids_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify all unique rsids from SQLite are in parquet."""
        sqlite_rsids = set(rsids_sqlite["rsid"].unique().drop_nulls().to_list())
        parquet_rsids = set(studies_parquet["rsid"].unique().drop_nulls().to_list())

        assert sqlite_rsids == parquet_rsids


class TestSchemaTransformation:
    """Test that schema transformations are correct."""

    def test_genotype_format_correct(self, weights_parquet: pl.DataFrame):
        """Verify genotype column has correct format."""
        genotypes = weights_parquet["genotype"].unique().drop_nulls().to_list()

        for gt in genotypes:
            if gt and len(gt) == 2:
                assert gt.isalpha(), f"Genotype should be alphabetic: {gt}"

    def test_state_values_valid(self, weights_parquet: pl.DataFrame):
        """Verify state column has valid values."""
        valid_states = {"protective", "risk", "neutral", "alt", "ref"}
        states = set(weights_parquet["state"].unique().drop_nulls().to_list())

        invalid = states - valid_states
        assert len(invalid) == 0, f"Invalid state values: {invalid}"

    def test_curator_and_method_present(self, weights_parquet: pl.DataFrame):
        """Verify curator and method columns are present and populated."""
        assert "curator" in weights_parquet.columns
        assert "method" in weights_parquet.columns
        assert all(weights_parquet["curator"] == "just-dna-seq")
        assert all(weights_parquet["method"] == "literature_review")

    def test_genotype_normalized(self, weights_parquet: pl.DataFrame):
        """Verify genotypes are alphabetically normalized."""
        genotypes = weights_parquet["genotype"].unique().drop_nulls().to_list()
        
        for gt in genotypes:
            if gt and len(gt) == 2:
                # Genotype should be in alphabetical order
                assert gt == "".join(sorted(gt)), f"Genotype not normalized: {gt}"

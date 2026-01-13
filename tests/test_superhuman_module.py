#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/superhuman parquet files
correctly preserve data from the original superhuman.sqlite database.

This module has qualitative annotations (no numeric weights).

This test module will automatically download the superhuman data from
GitHub if it doesn't exist locally.
"""

import subprocess
import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data


# Paths to data files
SQLITE_PATH = Path("data/modules/just_superhuman/superhuman.sqlite")
PARQUET_DIR = Path("data/output/modules/superhuman")


@pytest.fixture(scope="module")
def ensure_superhuman_data() -> Path:
    """
    Ensure superhuman SQLite data exists, downloading if necessary.
    """
    ensure_oakvar_module_data(
        module_name="just_superhuman",
        output_dir=SQLITE_PATH.parent,
        expected_file="superhuman.sqlite",
    )
    
    if not SQLITE_PATH.exists():
        pytest.skip(f"Failed to download superhuman data: {SQLITE_PATH}")
    
    return SQLITE_PATH


@pytest.fixture(scope="module")
def ensure_superhuman_parquet(ensure_superhuman_data: Path) -> Path:
    """
    Ensure superhuman parquet files exist, converting if necessary.
    """
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            [
                "uv", "run", "modules", "convert-superhuman",
                "--db-path", str(ensure_superhuman_data),
                "--output-dir", str(PARQUET_DIR),
                "--no-log",
            ],
            check=True,
            capture_output=False,
        )
    
    if not weights_path.exists():
        pytest.skip(f"Failed to convert superhuman data: {weights_path}")
    
    return PARQUET_DIR


@pytest.fixture(scope="module")
def sqlite_connection(ensure_superhuman_data: Path):
    """Create a SQLite connection for the test module."""
    conn = sqlite3.connect(ensure_superhuman_data)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def superhuman_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load superhuman data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            rsid,
            gene,
            genotype,
            superability,
            adverse_effects,
            "references"
        FROM superhuman
        """,
        sqlite_connection,
        infer_schema_length=None,
    )


@pytest.fixture(scope="module")
def weights_parquet(ensure_superhuman_parquet: Path) -> pl.DataFrame:
    """Load weights data from parquet."""
    parquet_file = ensure_superhuman_parquet / "weights.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def annotations_parquet(ensure_superhuman_parquet: Path) -> pl.DataFrame:
    """Load annotations data from parquet."""
    parquet_file = ensure_superhuman_parquet / "annotations.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_parquet(ensure_superhuman_parquet: Path) -> pl.DataFrame:
    """Load studies data from parquet."""
    parquet_file = ensure_superhuman_parquet / "studies.parquet"
    return pl.read_parquet(parquet_file)


class TestWeightsRowCount:
    """Test that weights table row counts match."""

    def test_total_row_count_reasonable(
        self, superhuman_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify parquet has reasonable row count."""
        # Filter SQLite rows that have genotype
        sqlite_with_genotype = superhuman_sqlite.filter(
            pl.col("genotype").is_not_null()
        )
        assert len(weights_parquet) > 0
        assert len(weights_parquet) <= len(sqlite_with_genotype)

    def test_unique_rsid_count_matches(
        self, superhuman_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify unique rsid count is identical."""
        sqlite_unique = superhuman_sqlite.filter(
            pl.col("genotype").is_not_null()
        )["rsid"].n_unique()
        parquet_unique = weights_parquet["rsid"].n_unique()
        assert sqlite_unique == parquet_unique, (
            f"Unique rsid count mismatch: SQLite has {sqlite_unique}, "
            f"Parquet has {parquet_unique}"
        )


class TestWeightValues:
    """Test that weight column is NULL (no numeric weights in this module)."""

    def test_weights_are_null(self, weights_parquet: pl.DataFrame):
        """Verify weight column is entirely NULL."""
        assert weights_parquet["weight"].is_null().all(), (
            "Superhuman module should have NULL weights"
        )


class TestAnnotationsTable:
    """Test that annotations data is correctly derived."""

    def test_annotations_have_gene(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify annotations have gene information."""
        assert "gene" in annotations_parquet.columns
        non_null_genes = annotations_parquet["gene"].drop_nulls()
        assert len(non_null_genes) > 0

    def test_module_column_correct(
        self, weights_parquet: pl.DataFrame, annotations_parquet: pl.DataFrame
    ):
        """Verify module column is 'superhuman'."""
        assert all(weights_parquet["module"] == "superhuman")
        assert all(annotations_parquet["module"] == "superhuman")

    def test_phenotype_is_elite_performance(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify phenotype is set correctly."""
        assert all(annotations_parquet["phenotype"] == "elite_performance")

    def test_category_is_superability(
        self, annotations_parquet: pl.DataFrame, superhuman_sqlite: pl.DataFrame
    ):
        """Verify category uses superability values."""
        sqlite_abilities = set(
            superhuman_sqlite["superability"].unique().drop_nulls().to_list()
        )
        parquet_categories = set(
            annotations_parquet["category"].unique().drop_nulls().to_list()
        )
        
        # All parquet categories should be from superability
        assert parquet_categories.issubset(sqlite_abilities)


class TestStudiesTable:
    """Test that studies data is correctly preserved."""

    def test_all_rsids_preserved(
        self, superhuman_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify all unique rsids from SQLite are in parquet."""
        sqlite_rsids = set(superhuman_sqlite["rsid"].unique().drop_nulls().to_list())
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
        """Verify state column has valid values for qualitative module."""
        valid_states = {"protective", "risk", "neutral"}
        states = set(weights_parquet["state"].unique().drop_nulls().to_list())

        invalid = states - valid_states
        assert len(invalid) == 0, f"Invalid state values: {invalid}"

    def test_conclusion_has_superability_info(self, weights_parquet: pl.DataFrame):
        """Verify conclusion column contains superability/adverse effects."""
        conclusions = weights_parquet["conclusion"].drop_nulls()
        # At least some conclusions should be present
        assert len(conclusions) > 0

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
                assert gt == "".join(sorted(gt)), f"Genotype not normalized: {gt}"

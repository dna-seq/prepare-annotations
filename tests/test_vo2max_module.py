#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/vo2max parquet files
correctly preserve data from the original vo2max.sqlite database.

This test module will automatically download the vo2max data from
GitHub if it doesn't exist locally.
"""

import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data
from prepare_annotations.core.paths import MODULES_DIR, MODULES_OUTPUT_DIR


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
    
    Uses the direct converter function instead of CLI subprocess.
    """
    from prepare_annotations.converters.vo2max import convert_vo2max
    
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        convert_vo2max(
            db_path=ensure_vo2max_data,
            output_dir=PARQUET_DIR,
            curator="just-dna-seq",
            method="literature_review",
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
        self, annotations_parquet: pl.DataFrame, rsids_sqlite: pl.DataFrame
    ):
        """Verify annotations have gene information."""
        sqlite_genes = set(rsids_sqlite["gene"].unique().drop_nulls().to_list())
        parquet_genes = set(annotations_parquet["gene"].unique().drop_nulls().to_list())

        if sqlite_genes:
            assert parquet_genes, "Expected gene values in annotations.parquet"
        assert parquet_genes.issubset(sqlite_genes), (
            "annotations.parquet genes should come from rsid table"
        )


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
        """Verify genotype column has correct format (list of 2 alleles)."""
        genotypes = weights_parquet["genotype"].unique().to_list()

        for gt in genotypes:
            assert isinstance(gt, list), f"Genotype should be a list, got {type(gt)}"
            assert len(gt) == 2, f"Invalid genotype format (expected 2 alleles): {gt}"
            for allele in gt:
                assert allele.isalpha() or allele == "?", f"Allele should be alphabetic or '?', got: {allele}"

    def test_state_values_valid(self, weights_parquet: pl.DataFrame):
        """Verify state column has valid values."""
        valid_states = {"protective", "risk", "neutral", "alt", "ref"}
        states = set(weights_parquet["state"].unique().drop_nulls().to_list())

        invalid = states - valid_states
        assert len(invalid) == 0, f"Invalid state values: {invalid}"

    def test_genotype_normalized(self, weights_parquet: pl.DataFrame):
        """Verify genotypes are alphabetically normalized (list of alleles)."""
        genotypes = weights_parquet["genotype"].unique().to_list()
        
        for gt in genotypes:
            if gt and len(gt) == 2:
                # For list genotypes, check elements are sorted
                assert gt == sorted(gt), f"Genotype not normalized: {gt}"

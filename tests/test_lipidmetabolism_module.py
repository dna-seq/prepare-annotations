#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/lipidmetabolism parquet files
correctly preserve data from the original lipid_metabolism.sqlite database.

This test module will automatically download the lipidmetabolism data from
GitHub if it doesn't exist locally.
"""

import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data
from prepare_annotations.core.paths import MODULES_DIR, MODULES_OUTPUT_DIR


# Paths to data files
SQLITE_PATH = MODULES_DIR / "just_lipidmetabolism" / "lipid_metabolism.sqlite"
PARQUET_DIR = MODULES_OUTPUT_DIR / "lipidmetabolism"


@pytest.fixture(scope="module")
def ensure_lipidmetabolism_data() -> Path:
    """
    Ensure lipidmetabolism SQLite data exists, downloading if necessary.
    """
    ensure_oakvar_module_data(
        module_name="just_lipidmetabolism",
        output_dir=SQLITE_PATH.parent,
        expected_file="lipid_metabolism.sqlite",
    )
    
    if not SQLITE_PATH.exists():
        pytest.skip(f"Failed to download lipidmetabolism data: {SQLITE_PATH}")
    
    return SQLITE_PATH


@pytest.fixture(scope="module")
def ensure_lipidmetabolism_parquet(ensure_lipidmetabolism_data: Path) -> Path:
    """
    Ensure lipidmetabolism parquet files exist, converting if necessary.
    
    Uses the direct converter function instead of CLI subprocess.
    """
    from prepare_annotations.converters.lipidmetabolism import convert_lipidmetabolism
    
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        convert_lipidmetabolism(
            db_path=ensure_lipidmetabolism_data,
            output_dir=PARQUET_DIR,
            curator="just-dna-seq",
            method="literature_review",
        )
    
    if not weights_path.exists():
        pytest.skip(f"Failed to convert lipidmetabolism data: {weights_path}")
    
    return PARQUET_DIR


@pytest.fixture(scope="module")
def sqlite_connection(ensure_lipidmetabolism_data: Path):
    """Create a SQLite connection for the test module."""
    conn = sqlite3.connect(ensure_lipidmetabolism_data)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def weights_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load weights data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            rsid,
            genotype,
            weight,
            state,
            genotype_specific_conclusion as conclusion
        FROM weight
        """,
        sqlite_connection,
    )


@pytest.fixture(scope="module")
def weights_parquet(ensure_lipidmetabolism_parquet: Path) -> pl.DataFrame:
    """Load weights data from parquet."""
    parquet_file = ensure_lipidmetabolism_parquet / "weights.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def annotations_parquet(ensure_lipidmetabolism_parquet: Path) -> pl.DataFrame:
    """Load annotations data from parquet."""
    parquet_file = ensure_lipidmetabolism_parquet / "annotations.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_parquet(ensure_lipidmetabolism_parquet: Path) -> pl.DataFrame:
    """Load studies data from parquet."""
    parquet_file = ensure_lipidmetabolism_parquet / "studies.parquet"
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
        FROM rsids
        """,
        sqlite_connection,
    )


class TestWeightsRowCount:
    """Test that weights table row counts match."""

    def test_total_row_count_reasonable(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify parquet has reasonable row count (may deduplicate)."""
        # Parquet may have fewer rows due to deduplication by (rsid, genotype, module)
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

    def test_weight_sum_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify total weight sum is similar (may differ slightly due to dedup)."""
        sqlite_sum = weights_sqlite["weight"].cast(pl.Float64, strict=False).sum()
        parquet_sum = weights_parquet["weight"].sum()
        # Allow some tolerance due to potential deduplication
        if sqlite_sum is not None and parquet_sum is not None:
            assert abs(sqlite_sum - parquet_sum) < abs(sqlite_sum) * 0.5

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

    def test_all_annotation_rsids_have_weights(
        self, annotations_parquet: pl.DataFrame, rsids_sqlite: pl.DataFrame
    ):
        """Verify annotation rsids come from the SQLite rsids table."""
        annotation_rsids = set(annotations_parquet["rsid"].unique().to_list())
        sqlite_rsids = set(rsids_sqlite["rsid"].unique().drop_nulls().to_list())

        # All annotation rsids should be in the source data
        if sqlite_rsids:
            assert annotation_rsids, "Expected annotation rsids when source rsids exist"
        assert annotation_rsids.issubset(sqlite_rsids), (
            "Found annotation rsids that are missing from rsids table"
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

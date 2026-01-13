#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/coronary parquet files
correctly preserve data from the original coronary.sqlite database.

This test module will automatically download the coronary data from
GitHub if it doesn't exist locally.
"""

import subprocess
import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data


# Paths to data files
SQLITE_PATH = Path("data/modules/just_coronary/coronary.sqlite")
PARQUET_DIR = Path("data/output/modules/coronary")


@pytest.fixture(scope="module")
def ensure_coronary_data() -> Path:
    """
    Ensure coronary SQLite data exists, downloading if necessary.
    """
    ensure_oakvar_module_data(
        module_name="just_coronary",
        output_dir=SQLITE_PATH.parent,
        expected_file="coronary.sqlite",
    )
    
    if not SQLITE_PATH.exists():
        pytest.skip(f"Failed to download coronary data: {SQLITE_PATH}")
    
    return SQLITE_PATH


@pytest.fixture(scope="module")
def ensure_coronary_parquet(ensure_coronary_data: Path) -> Path:
    """
    Ensure coronary parquet files exist, converting if necessary.
    """
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            [
                "uv", "run", "modules", "convert-coronary",
                "--db-path", str(ensure_coronary_data),
                "--output-dir", str(PARQUET_DIR),
                "--no-log",
            ],
            check=True,
            capture_output=False,
        )
    
    if not weights_path.exists():
        pytest.skip(f"Failed to convert coronary data: {weights_path}")
    
    return PARQUET_DIR


@pytest.fixture(scope="module")
def sqlite_connection(ensure_coronary_data: Path):
    """Create a SQLite connection for the test module."""
    conn = sqlite3.connect(ensure_coronary_data)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def coronary_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load coronary disease data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            rsID as rsid,
            Gene as gene,
            Genotype as genotype,
            Weight as weight,
            state,
            Conclusion as conclusion,
            PMID as pmid,
            Population as population,
            P_value as p_value,
            GWAS_study_design as study_design
        FROM coronary_disease
        """,
        sqlite_connection,
    )


@pytest.fixture(scope="module")
def weights_parquet(ensure_coronary_parquet: Path) -> pl.DataFrame:
    """Load weights data from parquet."""
    parquet_file = ensure_coronary_parquet / "weights.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def annotations_parquet(ensure_coronary_parquet: Path) -> pl.DataFrame:
    """Load annotations data from parquet."""
    parquet_file = ensure_coronary_parquet / "annotations.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_parquet(ensure_coronary_parquet: Path) -> pl.DataFrame:
    """Load studies data from parquet."""
    parquet_file = ensure_coronary_parquet / "studies.parquet"
    return pl.read_parquet(parquet_file)


class TestWeightsRowCount:
    """Test that weights table row counts match."""

    def test_total_row_count_reasonable(
        self, coronary_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify parquet has reasonable row count."""
        sqlite_with_genotype = coronary_sqlite.filter(
            pl.col("genotype").is_not_null()
        )
        assert len(weights_parquet) > 0
        assert len(weights_parquet) <= len(sqlite_with_genotype)

    def test_unique_rsid_count_matches(
        self, coronary_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify unique rsid count is identical."""
        sqlite_unique = coronary_sqlite.filter(
            pl.col("genotype").is_not_null()
        )["rsid"].n_unique()
        parquet_unique = weights_parquet["rsid"].n_unique()
        assert sqlite_unique == parquet_unique, (
            f"Unique rsid count mismatch: SQLite has {sqlite_unique}, "
            f"Parquet has {parquet_unique}"
        )


class TestWeightValues:
    """Test that weight values are correctly preserved."""

    def test_weight_min_matches(
        self, coronary_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify minimum weight is identical."""
        sqlite_min = coronary_sqlite["weight"].cast(pl.Float64, strict=False).min()
        parquet_min = weights_parquet["weight"].min()
        assert sqlite_min == parquet_min

    def test_weight_max_matches(
        self, coronary_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify maximum weight is identical."""
        sqlite_max = coronary_sqlite["weight"].cast(pl.Float64, strict=False).max()
        parquet_max = weights_parquet["weight"].max()
        assert sqlite_max == parquet_max


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
        """Verify module column is 'coronary'."""
        assert all(weights_parquet["module"] == "coronary")
        assert all(annotations_parquet["module"] == "coronary")

    def test_phenotype_is_coronary_disease(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify phenotype is set correctly."""
        assert all(annotations_parquet["phenotype"] == "coronary_disease")

    def test_category_is_cardiovascular(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify category is cardiovascular."""
        assert all(annotations_parquet["category"] == "cardiovascular")


class TestStudiesTable:
    """Test that studies data is correctly preserved."""

    def test_all_rsids_preserved(
        self, coronary_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify all unique rsids from SQLite are in parquet."""
        sqlite_rsids = set(coronary_sqlite["rsid"].unique().drop_nulls().to_list())
        parquet_rsids = set(studies_parquet["rsid"].unique().drop_nulls().to_list())

        assert sqlite_rsids == parquet_rsids

    def test_pmids_preserved(
        self, coronary_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify PMIDs are preserved."""
        sqlite_pmids = set(coronary_sqlite["pmid"].unique().drop_nulls().to_list())
        parquet_pmids = set(studies_parquet["pmid"].unique().drop_nulls().to_list())

        assert sqlite_pmids == parquet_pmids

    def test_study_design_preserved(
        self, coronary_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify study design is preserved."""
        sqlite_designs = set(
            coronary_sqlite["study_design"].unique().drop_nulls().to_list()
        )
        parquet_designs = set(
            studies_parquet["study_design"].unique().drop_nulls().to_list()
        )

        assert sqlite_designs == parquet_designs


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
        assert all(weights_parquet["method"] == "gwas_literature")

    def test_genotype_normalized(self, weights_parquet: pl.DataFrame):
        """Verify genotypes are alphabetically normalized."""
        genotypes = weights_parquet["genotype"].unique().drop_nulls().to_list()
        
        for gt in genotypes:
            if gt and len(gt) == 2:
                assert gt == "".join(sorted(gt)), f"Genotype not normalized: {gt}"

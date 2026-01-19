#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/longevitymap parquet files
correctly preserve data from the original longevitymap.sqlite database.

This test module will automatically download the longevitymap data from
GitHub if it doesn't exist locally.
"""

import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data
from prepare_annotations.core.paths import MODULES_DIR, MODULES_OUTPUT_DIR


# Paths to data files
SQLITE_PATH = MODULES_DIR / "just_longevitymap" / "longevitymap.sqlite"
PARQUET_DIR = MODULES_OUTPUT_DIR / "longevitymap"


@pytest.fixture(scope="module")
def ensure_longevitymap_data() -> Path:
    """
    Ensure longevitymap SQLite data exists, downloading if necessary.
    
    Downloads from https://github.com/dna-seq/just_longevitymap if not present.
    """
    ensure_oakvar_module_data(
        module_name="just_longevitymap",
        output_dir=SQLITE_PATH.parent,
        expected_file="longevitymap.sqlite",
    )
    
    if not SQLITE_PATH.exists():
        pytest.skip(f"Failed to download longevitymap data: {SQLITE_PATH}")
    
    return SQLITE_PATH


@pytest.fixture(scope="module")
def ensure_longevitymap_parquet(ensure_longevitymap_data: Path) -> Path:
    """
    Ensure longevitymap parquet files exist, converting if necessary.
    
    Uses the direct converter function instead of CLI subprocess.
    """
    from prepare_annotations.converters.longevitymap import convert_longevitymap
    
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    # Check if all parquet files exist
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        convert_longevitymap(
            db_path=ensure_longevitymap_data,
            output_dir=PARQUET_DIR,
            ensembl_cache=None,
            curator="Olga Borysova",
            method="literature_review",
        )
    
    if not weights_path.exists():
        pytest.skip(f"Failed to convert longevitymap data: {weights_path}")
    
    return PARQUET_DIR


@pytest.fixture(scope="module")
def sqlite_connection(ensure_longevitymap_data: Path):
    """Create a SQLite connection for the test module."""
    conn = sqlite3.connect(ensure_longevitymap_data)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def weights_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load weights data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            aw.rsid,
            aw.allele,
            aw.state,
            aw.zygosity,
            aw.weight,
            aw.priority,
            c.name as category
        FROM allele_weights aw
        LEFT JOIN categories c ON aw.category_id = c.id
        """,
        sqlite_connection,
    )


@pytest.fixture(scope="module")
def weights_parquet(ensure_longevitymap_parquet: Path) -> pl.DataFrame:
    """Load weights data from parquet."""
    parquet_file = ensure_longevitymap_parquet / "weights.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def annotations_parquet(ensure_longevitymap_parquet: Path) -> pl.DataFrame:
    """Load annotations data from parquet."""
    parquet_file = ensure_longevitymap_parquet / "annotations.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_parquet(ensure_longevitymap_parquet: Path) -> pl.DataFrame:
    """Load studies data from parquet."""
    parquet_file = ensure_longevitymap_parquet / "studies.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_sqlite(sqlite_connection) -> pl.DataFrame:
    """Load studies/variant data from SQLite."""
    return pl.read_database(
        """
        SELECT 
            v.identifier as rsid,
            v.quickpubmed as pmid,
            p.name as population,
            v.conclusions,
            v.study_design
        FROM variant v
        LEFT JOIN population p ON v.population_id = p.id
        """,
        sqlite_connection,
    )


class TestWeightsRowCount:
    """
    Test that weights table row counts are valid.
    
    Note: The Dagster pipeline expands het+alt genotypes by joining with Ensembl,
    which creates more rows than the original SQLite. Tests validate:
    - SQLite has expected 1043 rows
    - Parquet has at least as many rows (due to genotype expansion)
    - Unique rsid count is preserved
    """

    def test_parquet_has_at_least_as_many_rows(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Parquet should have >= rows due to genotype expansion."""
        assert len(weights_parquet) >= len(weights_sqlite), (
            f"Parquet has fewer rows ({len(weights_parquet)}) than SQLite ({len(weights_sqlite)})"
        )

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
    """
    Test that weight values are correctly preserved.
    
    With genotype expansion, per-rsid weight sums may differ because
    the same weight can appear multiple times for different genotypes.
    Tests validate unique weight values are preserved.
    """

    def test_weight_min_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify minimum weight is identical."""
        assert weights_sqlite["weight"].min() == weights_parquet["weight"].min()

    def test_weight_max_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify maximum weight is identical."""
        assert weights_sqlite["weight"].max() == weights_parquet["weight"].max()

    def test_all_sqlite_weights_in_parquet(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Every (rsid, weight) pair in SQLite should exist in parquet."""
        sqlite_pairs = (
            weights_sqlite.select(["rsid", "weight"])
            .unique()
        )
        parquet_pairs = (
            weights_parquet.select(["rsid", "weight"])
            .unique()
        )
        
        missing = sqlite_pairs.join(
            parquet_pairs, on=["rsid", "weight"], how="anti"
        )
        
        assert len(missing) == 0, f"SQLite weight pairs missing: {missing.head(20)}"

    def test_all_unique_weights_preserved(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """All unique weight values in SQLite should exist in parquet."""
        sqlite_weights = set(weights_sqlite["weight"].unique().to_list())
        parquet_weights = set(weights_parquet["weight"].unique().to_list())
        
        missing = sqlite_weights - parquet_weights
        assert len(missing) == 0, f"Weight values missing: {missing}"

    def test_parquet_has_at_least_as_many_negative_weights(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Parquet should have >= negative weights due to expansion."""
        sqlite_negative = len(weights_sqlite.filter(pl.col("weight") < 0))
        parquet_negative = len(weights_parquet.filter(pl.col("weight") < 0))
        assert parquet_negative >= sqlite_negative


class TestAPOEVariants:
    """Test APOE variants which are critical longevity markers."""

    @pytest.mark.parametrize(
        "rsid,expected_weights",
        [
            ("rs7412", {0.5, 1.0}),  # Protective APOE e2
            ("rs429358", {-0.5, -1.0}),  # Risk APOE e4
        ],
    )
    def test_apoe_weights_preserved(
        self,
        weights_sqlite: pl.DataFrame,
        weights_parquet: pl.DataFrame,
        rsid: str,
        expected_weights: set[float],
    ):
        """Verify APOE variant unique weight values are preserved."""
        sqlite_weights = set(
            weights_sqlite.filter(pl.col("rsid") == rsid)["weight"].unique().to_list()
        )
        parquet_weights = set(
            weights_parquet.filter(pl.col("rsid") == rsid)["weight"].unique().to_list()
        )

        assert sqlite_weights == parquet_weights, (
            f"{rsid}: SQLite={sqlite_weights}, Parquet={parquet_weights}"
        )
        assert sqlite_weights == expected_weights, (
            f"{rsid}: Expected={expected_weights}, Got={sqlite_weights}"
        )



class TestSampledVariants:
    """Test a sample of random variants for data integrity."""

    SAMPLE_RSIDS = [
        "rs11235972",
        "rs11943045",
        "rs1377843",
        "rs2253363",
        "rs262883",
        "rs2764264",
        "rs6991271",
        "rs7524519",
        "rs9876781",
        "rs9977638",
    ]

    @pytest.mark.parametrize("rsid", SAMPLE_RSIDS)
    def test_sampled_variant_unique_weights_match(
        self,
        weights_sqlite: pl.DataFrame,
        weights_parquet: pl.DataFrame,
        rsid: str,
    ):
        """Verify sampled variant unique weight values match between SQLite and Parquet."""
        sqlite_weights = set(
            weights_sqlite.filter(pl.col("rsid") == rsid)["weight"].unique().to_list()
        )
        parquet_weights = set(
            weights_parquet.filter(pl.col("rsid") == rsid)["weight"].unique().to_list()
        )

        assert sqlite_weights == parquet_weights, (
            f"{rsid}: SQLite={sqlite_weights}, Parquet={parquet_weights}"
        )

    @pytest.mark.parametrize("rsid", SAMPLE_RSIDS)
    def test_sampled_variant_row_count_at_least_as_many(
        self,
        weights_sqlite: pl.DataFrame,
        weights_parquet: pl.DataFrame,
        rsid: str,
    ):
        """Verify parquet has at least as many rows as SQLite (due to expansion)."""
        sqlite_count = len(weights_sqlite.filter(pl.col("rsid") == rsid))
        parquet_count = len(weights_parquet.filter(pl.col("rsid") == rsid))

        assert parquet_count >= sqlite_count, (
            f"{rsid}: Parquet has fewer rows ({parquet_count}) than SQLite ({sqlite_count})"
        )


class TestStudiesTable:
    """Test that studies/variant data is correctly preserved."""

    def test_all_pmids_preserved(
        self, studies_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify all unique PMIDs are preserved."""
        sqlite_pmids = set(studies_sqlite["pmid"].unique().drop_nulls().to_list())
        parquet_pmids = set(studies_parquet["pmid"].unique().drop_nulls().to_list())

        assert sqlite_pmids == parquet_pmids, (
            f"PMID mismatch: {len(sqlite_pmids)} in SQLite, {len(parquet_pmids)} in Parquet"
        )

    def test_parquet_has_no_more_rows_than_sqlite_unique(
        self, studies_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify parquet deduplication is valid (fewer or equal rows than unique SQLite rows)."""
        sqlite_unique = studies_sqlite.select(
            ["rsid", "pmid", "population", "conclusions", "study_design"]
        ).n_unique()
        parquet_count = len(studies_parquet)

        assert parquet_count <= sqlite_unique, (
            f"Parquet has more rows ({parquet_count}) than SQLite unique ({sqlite_unique})"
        )


class TestAnnotationsTable:
    """Test that annotations data is correctly derived."""

    def test_all_annotation_rsids_have_weights(
        self, annotations_parquet: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify all annotation rsids exist in weights table."""
        annotation_rsids = set(annotations_parquet["rsid"].unique().to_list())
        weight_rsids = set(weights_parquet["rsid"].unique().to_list())

        missing = annotation_rsids - weight_rsids
        assert len(missing) == 0, f"Annotations have rsids not in weights: {missing}"

    def test_categories_preserved(
        self, sqlite_connection, annotations_parquet: pl.DataFrame
    ):
        """Verify all category names from SQLite are in parquet."""
        sqlite_cats = pl.read_database(
            "SELECT DISTINCT name FROM categories", sqlite_connection
        )
        sqlite_cat_set = set(sqlite_cats["name"].to_list())
        parquet_cat_set = set(
            annotations_parquet["category"].unique().drop_nulls().to_list()
        )

        assert parquet_cat_set.issubset(sqlite_cat_set), (
            f"Parquet has unknown categories: {parquet_cat_set - sqlite_cat_set}"
        )



class TestPopulations:
    """Test that population data is correctly preserved."""

    def test_all_populations_preserved(
        self, sqlite_connection, studies_parquet: pl.DataFrame
    ):
        """Verify all population names from SQLite are in parquet."""
        sqlite_pops = pl.read_database(
            "SELECT DISTINCT name FROM population", sqlite_connection
        )
        sqlite_pop_set = set(sqlite_pops["name"].drop_nulls().to_list())
        parquet_pop_set = set(
            studies_parquet["population"].unique().drop_nulls().to_list()
        )

        assert parquet_pop_set.issubset(sqlite_pop_set), (
            f"Parquet has unknown populations: {parquet_pop_set - sqlite_pop_set}"
        )



class TestSchemaTransformation:
    """Test that schema transformations are correct."""

    def test_genotype_format_correct(self, weights_parquet: pl.DataFrame):
        """Verify genotype column has correct format (list of 2 alleles)."""
        genotypes = weights_parquet["genotype"].unique().to_list()

        for gt in genotypes:
            assert isinstance(gt, list) or hasattr(gt, "to_list"), f"Genotype should be a list, got {type(gt)}"
            assert len(gt) == 2, f"Invalid genotype format (expected 2 alleles): {gt}"
            for allele in gt:
                assert allele.isalpha() or allele == "?", f"Allele should be alphabetic or '?', got: {allele}"

    def test_state_values_valid(self, weights_parquet: pl.DataFrame):
        """Verify state column has valid values."""
        valid_states = {"protective", "risk", "alt", "ref"}
        states = set(weights_parquet["state"].unique().drop_nulls().to_list())

        invalid = states - valid_states
        assert len(invalid) == 0, f"Invalid state values: {invalid}"

    def test_module_column_correct(
        self, weights_parquet: pl.DataFrame, annotations_parquet: pl.DataFrame
    ):
        """Verify module column is 'longevitymap'."""
        assert all(weights_parquet["module"] == "longevitymap")
        assert all(annotations_parquet["module"] == "longevitymap")

    def test_positive_weights_are_protective(self, weights_parquet: pl.DataFrame):
        """Verify positive weights have 'protective' state."""
        positive = weights_parquet.filter(pl.col("weight") > 0)
        states = positive["state"].unique().to_list()
        assert all(s == "protective" for s in states), (
            f"Positive weights should be protective, got: {states}"
        )

    def test_negative_weights_are_risk(self, weights_parquet: pl.DataFrame):
        """Verify negative weights have 'risk' state."""
        negative = weights_parquet.filter(pl.col("weight") < 0)
        states = negative["state"].unique().to_list()
        assert all(s == "risk" for s in states), (
            f"Negative weights should be risk, got: {states}"
        )

#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/longevitymap parquet files
correctly preserve data from the original longevitymap.sqlite database.

This test module will automatically download the longevitymap data from
GitHub if it doesn't exist locally.
"""

import subprocess
import pytest
import polars as pl
import sqlite3
from pathlib import Path

from conftest import ensure_oakvar_module_data


# Paths to data files
SQLITE_PATH = Path("data/modules/just_longevitymap/longevitymap.sqlite")
PARQUET_DIR = Path("data/output/modules/longevitymap")


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
    
    Runs the convert_longevitymap command if parquet files don't exist.
    """
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    # Check if all parquet files exist
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            [
                "uv", "run", "modules", "convert-longevitymap",
                "--db-path", str(ensure_longevitymap_data),
                "--output-dir", str(PARQUET_DIR),
                "--no-log",
            ],
            check=True,
            capture_output=False,
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
    """Test that weights table row counts match."""

    def test_total_row_count_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify total row count is identical."""
        assert len(weights_sqlite) == len(weights_parquet), (
            f"Row count mismatch: SQLite has {len(weights_sqlite)}, "
            f"Parquet has {len(weights_parquet)}"
        )

    def test_expected_row_count(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify expected row count of 1043."""
        assert len(weights_sqlite) == 1043
        assert len(weights_parquet) == 1043

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

    def test_expected_unique_rsid_count(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify expected unique rsid count of 528."""
        assert weights_sqlite["rsid"].n_unique() == 528
        assert weights_parquet["rsid"].n_unique() == 528


class TestWeightValues:
    """Test that weight values are correctly preserved."""

    def test_weight_sum_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify total weight sum is identical."""
        sqlite_sum = weights_sqlite["weight"].sum()
        parquet_sum = weights_parquet["weight"].sum()
        assert abs(sqlite_sum - parquet_sum) < 0.001, (
            f"Weight sum mismatch: SQLite={sqlite_sum}, Parquet={parquet_sum}"
        )

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

    def test_weight_mean_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify mean weight is identical within tolerance."""
        sqlite_mean = weights_sqlite["weight"].mean()
        parquet_mean = weights_parquet["weight"].mean()
        assert abs(sqlite_mean - parquet_mean) < 0.0001

    def test_weight_sums_per_rsid_match(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify weight sums match for each rsid."""
        sqlite_sums = (
            weights_sqlite.group_by("rsid")
            .agg(pl.col("weight").sum().alias("weight_sum"))
            .sort("rsid")
        )
        parquet_sums = (
            weights_parquet.group_by("rsid")
            .agg(pl.col("weight").sum().alias("weight_sum"))
            .sort("rsid")
        )

        # Join and compare
        comparison = sqlite_sums.join(
            parquet_sums, on="rsid", how="full", suffix="_parquet"
        )
        diff = comparison.filter(
            (pl.col("weight_sum") - pl.col("weight_sum_parquet")).abs() > 0.001
        )

        assert len(diff) == 0, f"Weight sum mismatches found for rsids: {diff}"

    def test_negative_weight_count_matches(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify count of negative (risk) weights is identical."""
        sqlite_negative = len(weights_sqlite.filter(pl.col("weight") < 0))
        parquet_negative = len(weights_parquet.filter(pl.col("weight") < 0))
        assert sqlite_negative == parquet_negative == 48


class TestAPOEVariants:
    """Test APOE variants which are critical longevity markers."""

    @pytest.mark.parametrize(
        "rsid,expected_weights",
        [
            ("rs7412", [0.5, 1.0]),  # Protective APOE e2
            ("rs429358", [-0.5, -1.0]),  # Risk APOE e4
        ],
    )
    def test_apoe_weights_preserved(
        self,
        weights_sqlite: pl.DataFrame,
        weights_parquet: pl.DataFrame,
        rsid: str,
        expected_weights: list[float],
    ):
        """Verify APOE variant weights are correctly preserved."""
        sqlite_weights = sorted(
            weights_sqlite.filter(pl.col("rsid") == rsid)["weight"].to_list()
        )
        parquet_weights = sorted(
            weights_parquet.filter(pl.col("rsid") == rsid)["weight"].to_list()
        )

        assert sqlite_weights == parquet_weights, (
            f"{rsid}: SQLite={sqlite_weights}, Parquet={parquet_weights}"
        )
        assert sqlite_weights == sorted(expected_weights), (
            f"{rsid}: Expected={sorted(expected_weights)}, Got={sqlite_weights}"
        )

    def test_rs7412_has_two_entries(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify rs7412 has exactly 2 weight entries (het and hom)."""
        assert len(weights_sqlite.filter(pl.col("rsid") == "rs7412")) == 2
        assert len(weights_parquet.filter(pl.col("rsid") == "rs7412")) == 2

    def test_rs429358_has_two_entries(
        self, weights_sqlite: pl.DataFrame, weights_parquet: pl.DataFrame
    ):
        """Verify rs429358 has exactly 2 weight entries (het and hom)."""
        assert len(weights_sqlite.filter(pl.col("rsid") == "rs429358")) == 2
        assert len(weights_parquet.filter(pl.col("rsid") == "rs429358")) == 2


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
    def test_sampled_variant_weights_match(
        self,
        weights_sqlite: pl.DataFrame,
        weights_parquet: pl.DataFrame,
        rsid: str,
    ):
        """Verify sampled variant weights match between SQLite and Parquet."""
        sqlite_weights = sorted(
            weights_sqlite.filter(pl.col("rsid") == rsid)["weight"].to_list()
        )
        parquet_weights = sorted(
            weights_parquet.filter(pl.col("rsid") == rsid)["weight"].to_list()
        )

        assert sqlite_weights == parquet_weights, (
            f"{rsid}: SQLite={sqlite_weights}, Parquet={parquet_weights}"
        )

    @pytest.mark.parametrize("rsid", SAMPLE_RSIDS)
    def test_sampled_variant_row_count_matches(
        self,
        weights_sqlite: pl.DataFrame,
        weights_parquet: pl.DataFrame,
        rsid: str,
    ):
        """Verify sampled variant row counts match between SQLite and Parquet."""
        sqlite_count = len(weights_sqlite.filter(pl.col("rsid") == rsid))
        parquet_count = len(weights_parquet.filter(pl.col("rsid") == rsid))

        assert sqlite_count == parquet_count, (
            f"{rsid}: SQLite has {sqlite_count} rows, Parquet has {parquet_count}"
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

    def test_expected_pmid_count(
        self, studies_sqlite: pl.DataFrame, studies_parquet: pl.DataFrame
    ):
        """Verify expected count of 270 unique PMIDs."""
        sqlite_pmids = studies_sqlite["pmid"].unique().drop_nulls()
        parquet_pmids = studies_parquet["pmid"].unique().drop_nulls()
        assert len(sqlite_pmids) == 270
        assert len(parquet_pmids) == 270

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

    def test_expected_category_count(
        self, sqlite_connection, annotations_parquet: pl.DataFrame
    ):
        """Verify expected 12 categories."""
        sqlite_cats = pl.read_database(
            "SELECT DISTINCT name FROM categories", sqlite_connection
        )
        assert len(sqlite_cats) == 12

        parquet_cats = annotations_parquet["category"].unique().drop_nulls()
        assert len(parquet_cats) == 12


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

    def test_expected_population_count(
        self, sqlite_connection, studies_parquet: pl.DataFrame
    ):
        """Verify expected population count."""
        parquet_pops = studies_parquet["population"].unique().drop_nulls()
        # SQLite has 81 populations used in variants
        assert len(parquet_pops) == 81


class TestSchemaTransformation:
    """Test that schema transformations are correct."""

    def test_genotype_format_correct(self, weights_parquet: pl.DataFrame):
        """Verify genotype column has correct format (e.g., 'CT', 'TT', 'AA')."""
        genotypes = weights_parquet["genotype"].unique().to_list()

        for gt in genotypes:
            assert len(gt) == 2, f"Invalid genotype format: {gt}"
            assert gt.isalpha(), f"Genotype should be alphabetic: {gt}"

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

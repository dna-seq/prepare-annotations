#!/usr/bin/env python3
"""
Tests to validate that data/output/modules/drugs parquet files
correctly preserve data from the original annotation_tab.tsv file.

This test module will automatically download the drugs data from
GitHub if it doesn't exist locally.
"""

import subprocess
import pytest
import polars as pl
from pathlib import Path

from conftest import ensure_oakvar_module_data
from prepare_annotations.core.paths import MODULES_DIR, MODULES_OUTPUT_DIR


# Paths to data files
TSV_PATH = MODULES_DIR / "just_drugs" / "annotation_tab.tsv"
PARQUET_DIR = MODULES_OUTPUT_DIR / "drugs"


@pytest.fixture(scope="module")
def ensure_drugs_data() -> Path:
    """
    Ensure drugs TSV data exists, downloading if necessary.
    """
    ensure_oakvar_module_data(
        module_name="just_drugs",
        output_dir=TSV_PATH.parent,
        expected_file="annotation_tab.tsv",
    )
    
    if not TSV_PATH.exists():
        pytest.skip(f"Failed to download drugs data: {TSV_PATH}")
    
    return TSV_PATH


@pytest.fixture(scope="module")
def ensure_drugs_parquet(ensure_drugs_data: Path) -> Path:
    """
    Ensure drugs parquet files exist, converting if necessary.
    """
    weights_path = PARQUET_DIR / "weights.parquet"
    annotations_path = PARQUET_DIR / "annotations.parquet"
    studies_path = PARQUET_DIR / "studies.parquet"
    
    if not (weights_path.exists() and annotations_path.exists() and studies_path.exists()):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            [
                "uv", "run", "modules", "convert-drugs",
                "--tsv-path", str(ensure_drugs_data),
                "--output-dir", str(PARQUET_DIR),
                "--no-log",
            ],
            check=True,
            capture_output=False,
        )
    
    if not weights_path.exists():
        pytest.skip(f"Failed to convert drugs data: {weights_path}")
    
    return PARQUET_DIR


@pytest.fixture(scope="module")
def drugs_tsv(ensure_drugs_data: Path) -> pl.DataFrame:
    """Load drugs data from TSV."""
    return pl.read_csv(ensure_drugs_data, separator="\t")


@pytest.fixture(scope="module")
def weights_parquet(ensure_drugs_parquet: Path) -> pl.DataFrame:
    """Load weights data from parquet."""
    parquet_file = ensure_drugs_parquet / "weights.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def annotations_parquet(ensure_drugs_parquet: Path) -> pl.DataFrame:
    """Load annotations data from parquet."""
    parquet_file = ensure_drugs_parquet / "annotations.parquet"
    return pl.read_parquet(parquet_file)


@pytest.fixture(scope="module")
def studies_parquet(ensure_drugs_parquet: Path) -> pl.DataFrame:
    """Load studies data from parquet."""
    parquet_file = ensure_drugs_parquet / "studies.parquet"
    return pl.read_parquet(parquet_file)


class TestRsidFiltering:
    """Test that only valid rsids are included."""

    @pytest.mark.parametrize("fixture_name", ["annotations_parquet", "weights_parquet", "studies_parquet"])
    def test_all_rsids_start_with_rs(self, fixture_name: str, request: pytest.FixtureRequest):
        """Verify all rsids in output tables start with 'rs'."""
        df = request.getfixturevalue(fixture_name)
        invalid_rsids = [r for r in df["rsid"].to_list() if not r.startswith("rs")]
        assert len(invalid_rsids) == 0, f"Invalid rsids in {fixture_name}: {invalid_rsids[:5]}"


class TestAnnotationsTable:
    """Test that annotations data is correctly derived."""

    def test_category_has_drug_names(
        self, annotations_parquet: pl.DataFrame, drugs_tsv: pl.DataFrame
    ):
        """Verify category contains drug names."""
        # Filter TSV to rsids only
        tsv_rsids = drugs_tsv.filter(
            pl.col("Variant/Haplotypes").str.starts_with("rs")
        )
        tsv_drugs = set(tsv_rsids["Drug(s)"].unique().drop_nulls().to_list())
        parquet_categories = set(
            annotations_parquet["category"].unique().drop_nulls().to_list()
        )
        
        # Categories should be subset of drugs from TSV
        if tsv_drugs:
            assert parquet_categories, "Expected drug categories in annotations.parquet"
        assert parquet_categories.issubset(tsv_drugs)

    def test_phenotype_has_values(
        self, annotations_parquet: pl.DataFrame, drugs_tsv: pl.DataFrame
    ):
        """Verify phenotype has values (from Phenotype Category)."""
        tsv_rsids = drugs_tsv.filter(
            pl.col("Variant/Haplotypes").is_not_null()
            & pl.col("Variant/Haplotypes").str.starts_with("rs")
        )
        tsv_phenotypes = set(
            tsv_rsids["Phenotype Category"].unique().drop_nulls().to_list()
        )
        parquet_phenotypes = set(
            annotations_parquet["phenotype"].unique().drop_nulls().to_list()
        )

        if tsv_phenotypes:
            assert parquet_phenotypes, "Expected phenotype values in annotations.parquet"
        assert parquet_phenotypes.issubset(tsv_phenotypes), (
            "annotations.parquet phenotypes should come from TSV 'Phenotype Category'"
        )


class TestStudiesTable:
    """Test that studies data is correctly preserved."""

    def test_conclusion_has_sentences(
        self, studies_parquet: pl.DataFrame, drugs_tsv: pl.DataFrame
    ):
        """Verify conclusions contain sentences from TSV."""
        tsv_rsids = drugs_tsv.filter(
            pl.col("Variant/Haplotypes").is_not_null()
            & pl.col("Variant/Haplotypes").str.starts_with("rs")
        )
        tsv_sentences = set(
            tsv_rsids["Sentence"].unique().drop_nulls().to_list()
        )
        parquet_conclusions = set(
            studies_parquet["conclusion"].unique().drop_nulls().to_list()
        )

        if tsv_sentences:
            assert parquet_conclusions, "Expected conclusion values in studies.parquet"
        assert parquet_conclusions.issubset(tsv_sentences), (
            "studies.parquet conclusions should come from TSV 'Sentence'"
        )

    def test_p_values_preserved(
        self, studies_parquet: pl.DataFrame, drugs_tsv: pl.DataFrame
    ):
        """Verify P values are preserved where available."""
        tsv_rsids = drugs_tsv.filter(
            pl.col("Variant/Haplotypes").is_not_null()
            & pl.col("Variant/Haplotypes").str.starts_with("rs")
        )
        tsv_p_values = (
            tsv_rsids["P Value"]
            .drop_nulls()
            .cast(pl.Utf8, strict=False)
            .unique()
            .to_list()
        )
        parquet_p_values = (
            studies_parquet["p_value"]
            .drop_nulls()
            .cast(pl.Utf8, strict=False)
            .unique()
            .to_list()
        )

        if tsv_p_values:
            assert parquet_p_values, "Expected non-empty p_value values in studies.parquet"
            assert set(parquet_p_values).issubset(set(tsv_p_values)), (
                "studies.parquet p_value entries should come from TSV 'P Value' column"
            )


class TestWeightsTable:
    """Test that weights data is correctly derived."""

    def test_weights_are_null(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify weight column is NULL (no numeric weights in this module)."""
        assert weights_parquet["weight"].is_null().all()

    def test_state_is_significance_based(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify state is derived from Significance field."""
        valid_states = {"significant", "not_significant"}
        states = set(weights_parquet["state"].unique().drop_nulls().to_list())
        
        invalid = states - valid_states
        assert len(invalid) == 0, f"Invalid state values: {invalid}"

    def test_genotype_is_placeholder(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify genotype uses placeholder (no genotype data in TSV)."""
        genotypes = weights_parquet["genotype"].unique().to_list()
        assert "??" in genotypes


class TestSchemaTransformation:
    """Test that schema transformations are correct."""

    def test_conclusion_has_drug_info(
        self, weights_parquet: pl.DataFrame, drugs_tsv: pl.DataFrame
    ):
        """Verify conclusion combines drug name and sentence."""
        expected_conclusions = set(
            drugs_tsv.filter(
                pl.col("Variant/Haplotypes").is_not_null()
                & pl.col("Variant/Haplotypes").str.starts_with("rs")
            )
            .select(
                (pl.col("Drug(s)") + ": " + pl.col("Sentence")).alias("conclusion")
            )["conclusion"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        parquet_conclusions = set(
            weights_parquet["conclusion"].drop_nulls().unique().to_list()
        )
        if expected_conclusions:
            assert parquet_conclusions, "Expected conclusions derived from TSV data"
        assert parquet_conclusions.issubset(expected_conclusions)
        assert all(": " in c for c in parquet_conclusions)

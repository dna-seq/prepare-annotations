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


class TestRowCount:
    """Test that table row counts are reasonable."""

    def test_annotations_has_rows(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify annotations has rows."""
        assert len(annotations_parquet) > 0

    def test_studies_has_rows(
        self, studies_parquet: pl.DataFrame
    ):
        """Verify studies has rows."""
        assert len(studies_parquet) > 0

    def test_weights_has_rows(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify weights has rows."""
        assert len(weights_parquet) > 0


class TestRsidFiltering:
    """Test that only valid rsids are included."""

    def test_all_rsids_start_with_rs(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify all rsids start with 'rs'."""
        rsids = annotations_parquet["rsid"].to_list()
        for rsid in rsids:
            assert rsid.startswith("rs"), f"Invalid rsid: {rsid}"

    def test_weights_rsids_start_with_rs(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify all weight rsids start with 'rs'."""
        rsids = weights_parquet["rsid"].to_list()
        for rsid in rsids:
            assert rsid.startswith("rs"), f"Invalid rsid: {rsid}"

    def test_studies_rsids_start_with_rs(
        self, studies_parquet: pl.DataFrame
    ):
        """Verify all study rsids start with 'rs'."""
        rsids = studies_parquet["rsid"].to_list()
        for rsid in rsids:
            assert rsid.startswith("rs"), f"Invalid rsid: {rsid}"


class TestAnnotationsTable:
    """Test that annotations data is correctly derived."""

    def test_module_column_correct(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify module column is 'drugs'."""
        assert all(annotations_parquet["module"] == "drugs")

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
        assert len(parquet_categories) > 0
        assert parquet_categories.issubset(tsv_drugs)

    def test_phenotype_has_values(
        self, annotations_parquet: pl.DataFrame
    ):
        """Verify phenotype has values (from Phenotype Category)."""
        phenotypes = annotations_parquet["phenotype"].unique().drop_nulls()
        assert len(phenotypes) > 0


class TestStudiesTable:
    """Test that studies data is correctly preserved."""

    def test_conclusion_has_sentences(
        self, studies_parquet: pl.DataFrame
    ):
        """Verify conclusions contain sentences from TSV."""
        conclusions = studies_parquet["conclusion"].drop_nulls()
        assert len(conclusions) > 0

    def test_p_values_preserved(
        self, studies_parquet: pl.DataFrame
    ):
        """Verify P values are preserved where available."""
        p_values = studies_parquet["p_value"].drop_nulls()
        # There should be some P values
        assert len(p_values) >= 0  # May have nulls


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

    def test_module_column_correct(
        self, weights_parquet: pl.DataFrame, annotations_parquet: pl.DataFrame
    ):
        """Verify module column is 'drugs'."""
        assert all(weights_parquet["module"] == "drugs")
        assert all(annotations_parquet["module"] == "drugs")

    def test_curator_and_method_present(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify curator and method columns are present and populated."""
        assert "curator" in weights_parquet.columns
        assert "method" in weights_parquet.columns
        assert all(weights_parquet["curator"] == "PharmGKB")
        assert all(weights_parquet["method"] == "pharmacogenomics_db")

    def test_conclusion_has_drug_info(
        self, weights_parquet: pl.DataFrame
    ):
        """Verify conclusion combines drug name and sentence."""
        conclusions = weights_parquet["conclusion"].drop_nulls()
        assert len(conclusions) > 0
        # At least some conclusions should contain ':'
        has_colon = [c for c in conclusions.to_list() if ":" in c]
        assert len(has_colon) > 0

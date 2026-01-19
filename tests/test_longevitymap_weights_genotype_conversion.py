#!/usr/bin/env python3
"""
Tests to validate that weights are correctly preserved when converting from
legacy het/hom format (SQLite) to new genotype-based format (Parquet).

Legacy format (SQLite allele_weights):
- rsid: variant ID
- allele: effect allele (single char for het/hom, or 2-char for spec)
- state: 'alt' or 'spec' 
- zygosity: 'het' or 'hom'
- weight: numeric value

New format (weights.parquet):
- rsid: variant ID
- genotype: list of 2 alleles normalized alphabetically, e.g., ['A', 'G']
- weight: same numeric value

This test validates the transformation is lossless.
"""

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from conftest import ensure_oakvar_module_data
from prepare_annotations.core.paths import MODULES_DIR, MODULES_OUTPUT_DIR


SQLITE_PATH = MODULES_DIR / "just_longevitymap" / "longevitymap.sqlite"
PARQUET_DIR = MODULES_OUTPUT_DIR / "longevitymap"
WEIGHTS_PARQUET = PARQUET_DIR / "weights.parquet"
JOINED_PARQUET = PARQUET_DIR / "longevitymap_ensembl_joined.parquet"


@pytest.fixture(scope="module")
def ensure_longevitymap_data() -> Path:
    """Ensure longevitymap SQLite data exists."""
    ensure_oakvar_module_data(
        module_name="just_longevitymap",
        output_dir=SQLITE_PATH.parent,
        expected_file="longevitymap.sqlite",
    )
    if not SQLITE_PATH.exists():
        pytest.skip(f"Failed to download longevitymap data: {SQLITE_PATH}")
    return SQLITE_PATH


@pytest.fixture(scope="module")
def sqlite_weights(ensure_longevitymap_data: Path) -> pl.DataFrame:
    """Load legacy weights from SQLite."""
    conn = sqlite3.connect(ensure_longevitymap_data)
    df = pl.read_database(
        """
        SELECT 
            aw.rsid,
            aw.allele,
            aw.state as allele_state,
            aw.zygosity,
            aw.weight,
            aw.priority
        FROM allele_weights aw
        """,
        conn,
    )
    conn.close()
    return df


@pytest.fixture(scope="module")
def parquet_weights() -> pl.DataFrame:
    """Load weights from parquet."""
    if not WEIGHTS_PARQUET.exists():
        pytest.skip(f"weights.parquet not found: {WEIGHTS_PARQUET}")
    return pl.read_parquet(WEIGHTS_PARQUET)


@pytest.fixture(scope="module")
def joined_weights() -> pl.DataFrame:
    """Load weights from joined parquet (if exists)."""
    if not JOINED_PARQUET.exists():
        pytest.skip(f"longevitymap_ensembl_joined.parquet not found: {JOINED_PARQUET}")
    return pl.read_parquet(JOINED_PARQUET)


def reconstruct_genotype_from_legacy(row: dict) -> str | None:
    """
    Reconstruct what the genotype should be from legacy format.
    
    Returns a 2-char string (unsorted) representing the genotype.
    """
    allele = row["allele"]
    zygosity = row["zygosity"]
    allele_state = row["allele_state"]
    
    if zygosity == "hom":
        # Homozygous: both alleles are the same
        return allele + allele
    elif zygosity == "het":
        if allele_state == "spec":
            # Special case: allele already contains full genotype
            return allele
        else:
            # alt state: need ref from VCF - return None as we can't know
            return None
    return None


def normalize_genotype(gt: str) -> list[str]:
    """Normalize genotype to sorted list."""
    return sorted(list(gt))


class TestWeightValuePreservation:
    """
    Tests that weight values are preserved in conversion.
    
    Note: The Dagster pipeline expands het+alt variants to all possible genotypes
    by joining with Ensembl data. This creates MORE rows than the original SQLite.
    Tests validate that:
    1. Unique rsid count is preserved
    2. Weight min/max are preserved
    3. Each (rsid, weight) pair in SQLite exists in parquet
    """

    def test_unique_rsid_count_preserved(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """Number of unique rsids should be preserved."""
        assert sqlite_weights["rsid"].n_unique() == parquet_weights["rsid"].n_unique()

    def test_weight_min_max_preserved(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """Weight min and max should be preserved."""
        assert sqlite_weights["weight"].min() == parquet_weights["weight"].min()
        assert sqlite_weights["weight"].max() == parquet_weights["weight"].max()

    def test_all_sqlite_weights_present_in_parquet(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """
        Every unique (rsid, weight) pair in SQLite should have at least one
        matching entry in parquet. The genotype may differ due to expansion.
        """
        # Get unique (rsid, weight) pairs from SQLite
        sqlite_pairs = (
            sqlite_weights.select(["rsid", "weight"])
            .unique()
            .sort(["rsid", "weight"])
        )
        
        # Get unique (rsid, weight) pairs from parquet
        parquet_pairs = (
            parquet_weights.select(["rsid", "weight"])
            .unique()
            .sort(["rsid", "weight"])
        )
        
        # Find any SQLite pairs missing from parquet
        missing = sqlite_pairs.join(
            parquet_pairs, on=["rsid", "weight"], how="anti"
        )
        
        assert len(missing) == 0, (
            f"SQLite weight pairs missing from parquet: {missing.head(20)}"
        )

    def test_parquet_has_at_least_as_many_rows(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """
        Parquet should have >= rows due to genotype expansion.
        
        het+alt variants get expanded to multiple genotypes when joined with Ensembl.
        """
        assert len(parquet_weights) >= len(sqlite_weights), (
            f"Parquet has fewer rows ({len(parquet_weights)}) than SQLite ({len(sqlite_weights)})"
        )

    def test_all_unique_weights_preserved(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """All unique weight values in SQLite should exist in parquet."""
        sqlite_unique_weights = set(sqlite_weights["weight"].unique().to_list())
        parquet_unique_weights = set(parquet_weights["weight"].unique().to_list())
        
        missing = sqlite_unique_weights - parquet_unique_weights
        assert len(missing) == 0, (
            f"Weight values missing from parquet: {missing}"
        )


class TestZygosityToGenotypeConversion:
    """Tests that het/hom conversion to genotype is correct."""

    def test_homozygous_genotypes_have_matching_alleles(
        self, parquet_weights: pl.DataFrame
    ):
        """
        All homozygous genotypes should have two identical alleles.
        
        Example: ['T', 'T'] or ['C', 'C'] - both alleles must match.
        """
        genotypes = parquet_weights["genotype"].to_list()
        
        hom_genotypes = [
            gt for gt in genotypes 
            if len(gt) == 2 and gt[0] == gt[1]
        ]
        
        # Verify all hom genotypes have valid alleles
        for gt in hom_genotypes:
            assert gt[0].isalpha() or gt[0] == "?", f"Invalid allele in hom genotype: {gt}"

    def test_heterozygous_genotypes_have_different_alleles(
        self, parquet_weights: pl.DataFrame
    ):
        """
        All heterozygous genotypes should have two different alleles.
        
        Example: ['C', 'T'] - alleles must differ.
        """
        genotypes = parquet_weights["genotype"].to_list()
        
        het_genotypes = [
            gt for gt in genotypes 
            if len(gt) == 2 and gt[0] != gt[1]
        ]
        
        # Verify all het genotypes have valid alleles
        for gt in het_genotypes:
            assert gt[0].isalpha() or gt[0] == "?", f"Invalid allele in het genotype: {gt}"
            assert gt[1].isalpha() or gt[1] == "?", f"Invalid allele in het genotype: {gt}"

    def test_homozygous_count_matches(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """
        Homozygous count should match between formats (no expansion for hom).
        
        SQLite: zygosity = 'hom'
        Parquet: genotype where gt[0] == gt[1]
        """
        sqlite_hom_count = len(sqlite_weights.filter(pl.col("zygosity") == "hom"))
        
        genotypes = parquet_weights["genotype"].to_list()
        parquet_hom_count = sum(1 for gt in genotypes if len(gt) == 2 and gt[0] == gt[1])
        
        assert sqlite_hom_count == parquet_hom_count, (
            f"Hom count mismatch: SQLite={sqlite_hom_count}, Parquet={parquet_hom_count}"
        )

    def test_heterozygous_count_at_least_as_many(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """
        Het count in parquet should be >= SQLite due to genotype expansion.
        
        het+alt variants expand to multiple genotypes when joined with Ensembl.
        """
        sqlite_het_count = len(sqlite_weights.filter(pl.col("zygosity") == "het"))
        
        genotypes = parquet_weights["genotype"].to_list()
        parquet_het_count = sum(1 for gt in genotypes if len(gt) == 2 and gt[0] != gt[1])
        
        assert parquet_het_count >= sqlite_het_count, (
            f"Het count too low: SQLite={sqlite_het_count}, Parquet={parquet_het_count}"
        )

    def test_all_genotypes_are_normalized(
        self, parquet_weights: pl.DataFrame
    ):
        """
        All genotypes should be normalized (sorted alphabetically).
        
        Valid: ['A', 'G'], ['C', 'T']
        Invalid: ['G', 'A'], ['T', 'C']
        """
        genotypes = parquet_weights["genotype"].to_list()
        
        not_normalized = []
        for gt in genotypes:
            if len(gt) == 2:
                if gt != sorted(gt):
                    not_normalized.append(gt)
        
        assert len(not_normalized) == 0, (
            f"Found {len(not_normalized)} non-normalized genotypes: {not_normalized[:10]}"
        )

    def test_het_alt_alleles_present_in_genotypes(
        self, sqlite_weights: pl.DataFrame, parquet_weights: pl.DataFrame
    ):
        """
        For het+alt variants, the specified alt allele should appear in the genotype.
        
        We group by rsid and verify that for each het+alt entry, the alt allele
        appears in at least one of the genotypes for that rsid.
        """
        het_alt_sqlite = sqlite_weights.filter(
            (pl.col("zygosity") == "het") & (pl.col("allele_state") == "alt")
        )
        
        # Group by rsid to get all alt alleles per variant
        rsid_alt_alleles = (
            het_alt_sqlite
            .group_by("rsid")
            .agg(pl.col("allele").alias("alt_alleles"))
        )
        
        missing_alleles = []
        
        for row in rsid_alt_alleles.iter_rows(named=True):
            rsid = row["rsid"]
            alt_alleles = set(row["alt_alleles"])
            
            # Get all genotypes for this rsid
            parquet_variant = parquet_weights.filter(pl.col("rsid") == rsid)
            genotypes = parquet_variant["genotype"].to_list()
            
            # Flatten all alleles in genotypes
            all_parquet_alleles = set()
            for gt in genotypes:
                all_parquet_alleles.update(gt)
            
            # Check if alt alleles are present (or placeholder '?')
            missing = alt_alleles - all_parquet_alleles - {"?"}
            if missing and "?" not in all_parquet_alleles:
                missing_alleles.append({
                    "rsid": rsid,
                    "expected_alt": list(alt_alleles),
                    "found": list(all_parquet_alleles),
                })
        
        assert len(missing_alleles) == 0, (
            f"Alt alleles missing from genotypes: {missing_alleles[:10]}"
        )


class TestJoinedWeightsPreservation:
    """Tests that weights are preserved in the Ensembl-joined parquet."""

    def test_joined_weight_values_match_original(
        self, parquet_weights: pl.DataFrame, joined_weights: pl.DataFrame
    ):
        """
        The joined file should have matching (rsid, genotype, weight) tuples.
        
        For each (rsid, genotype, weight) in joined, it should exist in weights.
        """
        # Get unique (rsid, genotype, weight) tuples
        weights_tuples = (
            parquet_weights.select(["rsid", "genotype", "weight"])
            .unique()
        )
        joined_tuples = (
            joined_weights.select(["rsid", "genotype", "weight"])
            .unique()
        )
        
        # Find any joined entries not in weights
        extra_in_joined = joined_tuples.join(
            weights_tuples, on=["rsid", "genotype", "weight"], how="anti"
        )
        
        assert len(extra_in_joined) == 0, (
            f"Joined file has entries not in weights: {extra_in_joined.head(10)}"
        )

    def test_joined_file_has_ensembl_columns(
        self, joined_weights: pl.DataFrame
    ):
        """Joined file should have Ensembl annotation columns."""
        expected_columns = {"chrom", "start", "end", "ref"}
        actual_columns = set(joined_weights.columns)
        
        missing = expected_columns - actual_columns
        assert len(missing) == 0, f"Missing Ensembl columns: {missing}"

    def test_joined_rsids_are_subset_of_original(
        self, parquet_weights: pl.DataFrame, joined_weights: pl.DataFrame
    ):
        """Joined rsids should be a subset of original weights rsids."""
        parquet_rsids = set(parquet_weights["rsid"].unique().to_list())
        joined_rsids = set(joined_weights["rsid"].unique().to_list())
        
        extra_rsids = joined_rsids - parquet_rsids
        assert len(extra_rsids) == 0, (
            f"Joined file has rsids not in weights: {list(extra_rsids)[:10]}"
        )


class TestAPOEWeightsPreservation:
    """Specific tests for critical APOE longevity variants."""

    @pytest.mark.parametrize(
        "rsid,expected_weights,expected_genotypes",
        [
            # rs7412: APOE e2 (protective)
            ("rs7412", {0.5, 1.0}, [["C", "T"], ["T", "T"]]),
            # rs429358: APOE e4 (risk)
            ("rs429358", {-0.5, -1.0}, [["C", "T"], ["C", "C"]]),
        ],
    )
    def test_apoe_weights_and_genotypes_preserved(
        self,
        sqlite_weights: pl.DataFrame,
        parquet_weights: pl.DataFrame,
        rsid: str,
        expected_weights: set[float],
        expected_genotypes: list[list[str]],
    ):
        """
        Validate APOE variant weights and genotypes are correctly converted.
        
        Due to genotype expansion, parquet may have more entries but should
        contain all the expected weights and genotypes.
        """
        # Check SQLite weights
        sqlite_variant = sqlite_weights.filter(pl.col("rsid") == rsid)
        sqlite_weight_set = set(sqlite_variant["weight"].unique().to_list())
        
        # Check parquet weights
        parquet_variant = parquet_weights.filter(pl.col("rsid") == rsid)
        parquet_weight_set = set(parquet_variant["weight"].unique().to_list())
        
        # SQLite weights should be subset of parquet (parquet may have more due to expansion)
        assert sqlite_weight_set == parquet_weight_set, (
            f"{rsid}: SQLite weights={sqlite_weight_set}, "
            f"Parquet weights={parquet_weight_set}"
        )
        assert expected_weights == sqlite_weight_set, (
            f"{rsid}: Expected weights={expected_weights}, "
            f"got={sqlite_weight_set}"
        )
        
        # Expected genotypes should be present in parquet
        parquet_genotypes = [
            sorted(gt) for gt in parquet_variant["genotype"].to_list()
        ]
        for expected_gt in expected_genotypes:
            assert sorted(expected_gt) in parquet_genotypes, (
                f"{rsid}: Expected genotype {expected_gt} not found in {parquet_genotypes}"
            )

    def test_apoe_in_joined_file(
        self, joined_weights: pl.DataFrame
    ):
        """APOE variants should be present in joined file."""
        apoe_variants = ["rs7412", "rs429358"]
        joined_rsids = set(joined_weights["rsid"].unique().to_list())
        
        for rsid in apoe_variants:
            assert rsid in joined_rsids, (
                f"APOE variant {rsid} missing from joined file"
            )


class TestWeightStateConsistency:
    """Tests that state (protective/risk) is consistent with weight sign."""

    def test_positive_weights_are_protective(
        self, parquet_weights: pl.DataFrame
    ):
        """Positive weights should have 'protective' state."""
        positive = parquet_weights.filter(pl.col("weight") > 0)
        states = positive["state"].unique().to_list()
        
        assert states == ["protective"], (
            f"Expected only 'protective' for positive weights, got: {states}"
        )

    def test_negative_weights_are_risk(
        self, parquet_weights: pl.DataFrame
    ):
        """Negative weights should have 'risk' state."""
        negative = parquet_weights.filter(pl.col("weight") < 0)
        states = negative["state"].unique().to_list()
        
        assert states == ["risk"], (
            f"Expected only 'risk' for negative weights, got: {states}"
        )

    def test_state_consistent_between_files(
        self, parquet_weights: pl.DataFrame, joined_weights: pl.DataFrame
    ):
        """State should be consistent between weights and joined files."""
        # Sample comparison
        sample_rsids = parquet_weights["rsid"].unique().head(50).to_list()
        
        for rsid in sample_rsids:
            parquet_row = parquet_weights.filter(pl.col("rsid") == rsid)
            joined_row = joined_weights.filter(pl.col("rsid") == rsid)
            
            if len(joined_row) > 0:
                # Compare first matching row
                parquet_state = parquet_row["state"][0]
                joined_state = joined_row["state"][0]
                
                assert parquet_state == joined_state, (
                    f"{rsid}: state mismatch - weights={parquet_state}, "
                    f"joined={joined_state}"
                )

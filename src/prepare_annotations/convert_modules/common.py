"""
Common utilities for OakVar module conversion with Ensembl genotype expansion.

This module provides lazy Polars operations for:
- Loading module data from SQLite databases
- Expanding genotypes based on zygosity (homo/hetero)
- Joining with Ensembl variation data for het genotype construction

All operations are lazy (LazyFrame) for memory efficiency on laptops.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl


def load_weights_raw(db_path: Path) -> pl.LazyFrame:
    """
    Load raw allele weights from OakVar module SQLite database.
    
    Returns a lazy frame with columns:
    - rsid: Variant identifier
    - allele: The allele of interest
    - state: Allele state (alt, spec, etc.)
    - zygosity: hom (homozygous) or het (heterozygous)
    - weight: Numeric weight score
    - priority: Priority value
    - name: Category name
    """
    conn = sqlite3.connect(db_path)
    weights_query = """
    SELECT rsid, allele, state, zygosity, weight, priority, categories.name
    FROM allele_weights
    JOIN categories ON categories.id = allele_weights.category_id
    """
    weights = pl.read_database(weights_query, connection=conn).lazy()
    conn.close()
    return weights


def load_variants_raw(db_path: Path) -> pl.LazyFrame:
    """
    Load raw variants with population info from OakVar module SQLite database.
    
    Returns a lazy frame with columns:
    - rsid: Variant identifier
    - study_design, conclusions, association, gender, quickref, quickyear, quickpubmed
    - population_name: Population name
    - is_significant: Boolean derived from association
    """
    conn = sqlite3.connect(db_path)
    variants_query = """
    SELECT variant.identifier as rsid, variant.study_design, variant.conclusions, variant.association,
           variant.gender, variant.quickref, variant.quickyear, variant.quickpubmed,
           population.name as population_name
    FROM variant
    JOIN population ON variant.population_id = population.id
    """
    variants = (
        pl.read_database(variants_query, connection=conn)
        .with_columns((pl.col("association") == "significant").alias("is_significant"))
        .drop("association")
        .lazy()
    )
    conn.close()
    return variants


def expand_homozygous_genotypes(
    weights: pl.LazyFrame,
    *,
    allele_col: str = "allele",
    zygosity_col: str = "zygosity",
) -> pl.LazyFrame:
    """
    Expand homozygous variants to genotype list[str].
    
    For homozygous "C", genotype = ["C", "C"]
    
    This is a simple case - no Ensembl join needed.
    Genotype is stored as list[str] for parquet compatibility.
    
    Args:
        weights: LazyFrame with allele and zygosity columns
        allele_col: Name of allele column
        zygosity_col: Name of zygosity column
        
    Returns:
        LazyFrame filtered to hom variants with 'genotype' column (list[str])
    """
    return (
        weights
        .filter(pl.col(zygosity_col) == "hom")
        .with_columns(
            pl.concat_list([
                pl.col(allele_col),
                pl.col(allele_col)
            ]).alias("genotype")
        )
    )


def expand_heterozygous_spec_genotypes(
    weights: pl.LazyFrame,
    *,
    allele_col: str = "allele",
    zygosity_col: str = "zygosity",
    allele_state_col: str = "state",
) -> pl.LazyFrame:
    """
    Expand heterozygous variants with state='spec' to genotype list[str].
    
    For het+spec, the allele column already contains the full 2-char genotype
    (e.g., "CT" means ["C", "T"]).
    
    Genotype is stored as list[str] for parquet compatibility.
    
    Args:
        weights: LazyFrame with allele, zygosity, and state columns
        allele_col: Name of allele column
        zygosity_col: Name of zygosity column
        allele_state_col: Name of allele state column
        
    Returns:
        LazyFrame filtered to het+spec variants with 'genotype' column (list[str])
    """
    return (
        weights
        .filter(
            (pl.col(zygosity_col) == "het") & 
            (pl.col(allele_state_col) == "spec")
        )
        .with_columns(
            # Split 2-char genotype into list of single chars
            pl.when(pl.col(allele_col).str.len_chars() == 2)
            .then(
                pl.concat_list([
                    pl.col(allele_col).str.slice(0, 1),
                    pl.col(allele_col).str.slice(1, 1)
                ]).list.sort()  # Normalize order
            )
            .otherwise(
                # Fallback: split string into chars
                pl.col(allele_col).str.split("").list.slice(1).list.head(
                    pl.col(allele_col).str.len_chars()
                ).list.sort()
            )
            .alias("genotype")
        )
    )


def expand_heterozygous_with_ensembl(
    weights: pl.LazyFrame,
    ensembl: pl.LazyFrame,
    *,
    allele_col: str = "allele",
    zygosity_col: str = "zygosity",
    allele_state_col: str = "state",
    rsid_col: str = "rsid",
    ensembl_id_col: str = "id",
    ref_col: str = "ref",
    alts_col: str = "alts",
) -> pl.LazyFrame:
    """
    Expand heterozygous variants (state='alt') by joining with Ensembl.
    
    For het on allele "C", the other allele is anything BUT "C":
    - Could be ref (if ref != "C")
    - Could be any alt allele that isn't "C"
    
    This expands to MULTIPLE rows - one for each valid genotype pairing.
    
    Example: If het allele is "C" and Ensembl has ref="T", alts=["C", "G"]:
    - Valid other alleles: "T" (ref), "G" (alt != "C")
    - NOT "C" (can't pair with itself in het)
    - Creates genotypes: ["C", "T"], ["C", "G"]
    
    Genotype is stored as list[str] for parquet compatibility.
    
    All operations are lazy for memory efficiency.
    
    Args:
        weights: LazyFrame with allele, zygosity, state columns
        ensembl: LazyFrame with Ensembl variation data (id, ref, alts)
        allele_col: Name of curated allele column in weights
        zygosity_col: Name of zygosity column
        allele_state_col: Name of allele state column
        rsid_col: RSID column in weights
        ensembl_id_col: ID column in Ensembl
        ref_col: Reference allele column in Ensembl
        alts_col: Alternate alleles column in Ensembl (list[str])
        
    Returns:
        LazyFrame with expanded genotypes (potentially multiple rows per input)
    """
    # Filter to het+alt variants only
    het_weights = weights.filter(
        (pl.col(zygosity_col) == "het") & 
        (pl.col(allele_state_col) == "alt")
    )
    
    # Get original columns for preservation
    weight_cols = het_weights.collect_schema().names()
    
    # Join with minimal Ensembl columns (memory efficient)
    joined = het_weights.join(
        ensembl.select(
            pl.col(ensembl_id_col).alias(rsid_col),
            pl.col(ref_col),
            pl.col(alts_col),
        ),
        on=rsid_col,
        how="left",
    )
    
    # Build list of all possible "other" alleles: [ref] + alts
    # Then filter out the curated allele itself
    with_other_alleles = joined.with_columns(
        # Combine ref with alts into single list of all possible alleles
        pl.concat_list([
            pl.col(ref_col),  # ref as single-element start
            pl.col(alts_col).fill_null([]),  # alts list (handle null)
        ]).alias("__all_alleles")
    )
    
    # Explode to get one row per possible "other" allele
    exploded = with_other_alleles.explode("__all_alleles")
    
    # Filter: other allele must NOT equal the curated allele
    filtered = exploded.filter(
        pl.col("__all_alleles").is_not_null() &
        (pl.col("__all_alleles") != pl.col(allele_col))
    )
    
    # Build genotype as sorted list[str]
    with_genotype = filtered.with_columns(
        pl.concat_list([
            pl.col(allele_col),
            pl.col("__all_alleles")
        ]).list.sort().alias("genotype")
    )
    
    # Select only original columns plus genotype
    return with_genotype.select([
        *[pl.col(c) for c in weight_cols],
        pl.col("genotype"),
    ])


def expand_all_genotypes_with_ensembl(
    weights: pl.LazyFrame,
    ensembl: pl.LazyFrame,
    *,
    allele_col: str = "allele",
    zygosity_col: str = "zygosity",
    allele_state_col: str = "state",
    rsid_col: str = "rsid",
    ensembl_id_col: str = "id",
    ref_col: str = "ref",
    alts_col: str = "alts",
) -> pl.LazyFrame:
    """
    Expand all weight variants to genotype list[str] based on zygosity.
    
    Combines:
    1. Homozygous: allele "C" -> ["C", "C"]
    2. Het+spec: allele "CT" -> ["C", "T"] 
    3. Het+alt: allele "C" + Ensembl -> ["C", "T"], ["C", "G"], etc.
    
    This is the main genotype expansion function. It handles all cases
    and produces multiple rows for het+alt variants where multiple
    valid genotype pairings exist.
    
    All operations are lazy for memory efficiency on laptops.
    
    Args:
        weights: LazyFrame from load_weights_raw()
        ensembl: LazyFrame with Ensembl variation data
        
    Returns:
        LazyFrame with 'genotype' column (list[str]), potentially more rows
        than input due to het expansion
    """
    # Process each zygosity case separately, then union
    hom = expand_homozygous_genotypes(
        weights,
        allele_col=allele_col,
        zygosity_col=zygosity_col,
    )
    
    het_spec = expand_heterozygous_spec_genotypes(
        weights,
        allele_col=allele_col,
        zygosity_col=zygosity_col,
        allele_state_col=allele_state_col,
    )
    
    het_alt = expand_heterozygous_with_ensembl(
        weights,
        ensembl,
        allele_col=allele_col,
        zygosity_col=zygosity_col,
        allele_state_col=allele_state_col,
        rsid_col=rsid_col,
        ensembl_id_col=ensembl_id_col,
        ref_col=ref_col,
        alts_col=alts_col,
    )
    
    # Union all cases
    return pl.concat([hom, het_spec, het_alt], how="diagonal_relaxed")


def scan_ensembl_variations(
    source: Path | str,
    *,
    species: str = "homo_sapiens",
    chromosomes: list[str] | None = None,
) -> pl.LazyFrame:
    """
    Lazily scan Ensembl variation parquet files.
    
    Can handle either:
    - Local cache directory with per-chromosome parquet files
    - HuggingFace dataset path (hf://datasets/just-dna-seq/ensembl_variations)
    
    Uses low_memory=True for memory efficiency on laptops.
    
    Args:
        source: Path to local directory or HuggingFace dataset path
        species: Species name (used for file naming pattern)
        chromosomes: Optional list of chromosomes to load (e.g., ["1", "2", "X"])
                    If None, loads all chromosomes.
        
    Returns:
        Lazy Polars frame with Ensembl variation data
    """
    source_str = str(source)
    
    if source_str.startswith("hf://"):
        # HuggingFace dataset - use fsspec/hf protocol
        patterns = [f"{source_str}/data/*.parquet"]
        return pl.scan_parquet(patterns, low_memory=True)
    else:
        # Local directory
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Ensembl source not found: {source_path}")
        
        if chromosomes:
            # Load specific chromosomes
            files = []
            for chrom in chromosomes:
                pattern = f"{species}-chr{chrom}.parquet"
                matches = list(source_path.glob(pattern))
                files.extend(matches)
            if not files:
                raise FileNotFoundError(
                    f"No parquet files found for chromosomes {chromosomes} in {source_path}"
                )
            return pl.scan_parquet([str(f) for f in files], low_memory=True)
        else:
            # Load all chromosome files
            pattern = f"{species}-chr*.parquet"
            files = list(source_path.glob(pattern))
            if not files:
                # Try generic pattern
                files = list(source_path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found in {source_path}")
            return pl.scan_parquet([str(f) for f in sorted(files)], low_memory=True)


def get_ensembl_for_genotype_resolution(
    ensembl: pl.LazyFrame,
    *,
    id_col: str = "id",
    ref_col: str = "ref",
    alts_col: str = "alts",
) -> pl.LazyFrame:
    """
    Select minimal columns from Ensembl needed for genotype resolution.
    
    This reduces memory usage when joining large Ensembl datasets.
    Only id, ref, and alts are needed for genotype expansion.
    
    Args:
        ensembl: Full Ensembl LazyFrame
        id_col: RSID column name
        ref_col: Reference allele column name
        alts_col: Alternative alleles column name (list[str])
        
    Returns:
        LazyFrame with only id, ref, alts columns
    """
    return ensembl.select(
        pl.col(id_col),
        pl.col(ref_col),
        pl.col(alts_col),
    )


def convert_module_weights_with_ensembl(
    db_path: Path,
    ensembl_source: Path | str | pl.LazyFrame,
    *,
    module_name: str,
    curator: str = "unknown",
    method: str = "literature_review",
    species: str = "homo_sapiens",
) -> pl.LazyFrame:
    """
    Convert OakVar module weights to unified schema with Ensembl genotype resolution.
    
    This is the main entry point for converting module weights with proper
    genotype expansion based on zygosity and Ensembl data.
    
    Schema output:
    - rsid: Variant identifier
    - genotype: list[str] with two alleles, sorted alphabetically
    - module: Module name
    - weight: Numeric weight score
    - state: protective, risk, or neutral (derived from weight sign)
    - priority: Priority value
    - conclusion: Optional conclusion text (null for most modules)
    - curator: Curator name
    - method: Curation method
    
    All operations are lazy for memory efficiency on laptops.
    
    Args:
        db_path: Path to OakVar module SQLite database
        ensembl_source: Either:
            - Path to local Ensembl cache directory
            - HuggingFace path (hf://datasets/just-dna-seq/ensembl_variations)
            - Pre-loaded Ensembl LazyFrame
        module_name: Name for the module column
        curator: Curator name
        method: Curation method
        species: Species name (for local file pattern)
        
    Returns:
        LazyFrame with unified weight schema
    """
    # Load raw weights
    weights = load_weights_raw(db_path)
    rsids = (
        weights
        .select(pl.col("rsid").unique())
        .collect()
        .get_column("rsid")
        .to_list()
    )
    
    # Load or use provided Ensembl data
    if isinstance(ensembl_source, pl.LazyFrame):
        ensembl = ensembl_source
    else:
        ensembl = scan_ensembl_variations(ensembl_source, species=species)
    ensembl = ensembl.filter(pl.col("id").is_in(rsids))
    
    # Get minimal Ensembl columns for joining (memory efficient)
    ensembl_minimal = get_ensembl_for_genotype_resolution(ensembl)
    
    # Expand all genotypes with proper het handling
    with_genotype = expand_all_genotypes_with_ensembl(
        weights,
        ensembl_minimal,
    )
    
    # Derive state from weight and add module metadata
    result = with_genotype.with_columns(
        # Derive state from weight sign
        pl.when(pl.col("weight") > 0)
        .then(pl.lit("protective"))
        .when(pl.col("weight") < 0)
        .then(pl.lit("risk"))
        .otherwise(pl.lit("neutral"))
        .alias("state"),
        # Add metadata columns
        pl.lit(module_name).alias("module"),
        pl.lit(curator).alias("curator"),
        pl.lit(method).alias("method"),
        # Null conclusion (module-level, not genotype-specific)
        pl.lit(None).cast(pl.Utf8).alias("conclusion"),
    )
    
    # Deduplicate by (rsid, genotype, module) and select final schema
    # Note: genotype is list[str], so dedup works on list equality
    return (
        result
        .group_by(["rsid", "genotype", "module"])
        .agg([
            pl.col("weight").first(),
            pl.col("state").first(),
            pl.col("priority").first(),
            pl.col("conclusion").first(),
            pl.col("curator").first(),
            pl.col("method").first(),
        ])
        .select(
            "rsid",
            "genotype",
            "module",
            "weight",
            "state",
            "priority",
            "conclusion",
            "curator",
            "method",
        )
    )

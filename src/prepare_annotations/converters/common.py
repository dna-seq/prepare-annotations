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
import time
from pathlib import Path
from typing import Optional

import polars as pl
from eliot import log_message
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def check_huggingface_auth() -> Optional[dict]:
    """
    Check HuggingFace authentication status programmatically and log the result.
    
    Returns:
        Dictionary with user info if authenticated, None otherwise
    """
    try:
        from huggingface_hub import HfApi, whoami
        
        api = HfApi()
        user_info = whoami()
        
        log_message(
            message_type="huggingface_auth_check",
            username=user_info.get("name", "unknown"),
            orgs=user_info.get("orgs", []),
            authenticated=True
        )
        
        return user_info
    except Exception as e:
        log_message(
            message_type="huggingface_auth_check",
            authenticated=False,
            error=str(e)
        )
        return None


def configure_duckdb_for_hf(con, memory_limit: str = "4GB", temp_directory: Path = Path("data/interim/duckdb"), max_connections: int = 2) -> None:
    """
    Configure DuckDB connection with rate-limiting for HuggingFace HTTP requests.
    
    Args:
        con: DuckDB connection
        memory_limit: Memory limit string (e.g., '4GB')
        temp_directory: Temporary directory for DuckDB
        max_connections: Maximum number of simultaneous HTTP connections (default: 2)
    """
    # Basic configuration
    con.execute(f"SET memory_limit = '{memory_limit}'")
    Path(temp_directory).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{str(temp_directory).replace(chr(39), chr(39)+chr(39))}'")
    con.execute("SET preserve_insertion_order = false")
    
    # Limit parallelism to avoid HTTP 429 from HuggingFace
    # These settings control how many concurrent HTTP requests DuckDB makes
    con.execute(f"SET threads = {max_connections}")  # Limit worker threads (controls HTTP parallelism)
    con.execute(f"SET http_timeout = 180000")  # 180 second timeout (3 minutes)
    con.execute(f"SET http_retries = 5")  # More retries for DuckDB's internal retry logic
    con.execute(f"SET http_retry_wait_ms = 5000")  # Wait 5 seconds between DuckDB's internal retries
    
    log_message(
        message_type="duckdb_config",
        memory_limit=memory_limit,
        temp_directory=str(temp_directory),
        max_connections=max_connections,
        threads=max_connections,
        note="Aggressive rate limiting to avoid HuggingFace HTTP 429"
    )


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
    memory_limit: str | None = None,
    temp_directory: str = "/tmp/duckdb_module_conversion",
) -> pl.LazyFrame:
    """
    Convert OakVar module weights to unified schema with Ensembl genotype resolution.
    
    Uses DuckDB for memory-efficient joins - no manual RSID filtering needed.
    DuckDB's optimizer handles the join pushdown automatically and spills to disk
    if memory is tight.
    
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
    
    Args:
        db_path: Path to OakVar module SQLite database
        ensembl_source: Either:
            - Path to local Ensembl cache directory
            - HuggingFace path (hf://datasets/just-dna-seq/ensembl_variations)
            - Pre-loaded Ensembl LazyFrame (will be materialized to temp parquet)
        module_name: Name for the module column
        curator: Curator name
        method: Curation method
        species: Species name (for local file pattern)
        memory_limit: DuckDB memory limit (e.g., "8GB"). Auto-detected if None.
        temp_directory: Directory for DuckDB temp files
        
    Returns:
        LazyFrame with unified weight schema
    """
    import duckdb
    import tempfile
    
    # Check HuggingFace authentication and log status
    user_info = check_huggingface_auth()
    if user_info:
        log_message(
            message_type="ensembl_conversion_start",
            module=module_name,
            hf_user=user_info.get("name", "unknown"),
            hf_orgs=user_info.get("orgs", [])
        )
    else:
        log_message(
            message_type="ensembl_conversion_start",
            module=module_name,
            hf_authenticated=False,
            warning="Running without HuggingFace authentication - may encounter rate limits"
        )
    
    # Resolve Ensembl parquet files
    if isinstance(ensembl_source, pl.LazyFrame):
        # Materialize LazyFrame to temp parquet for DuckDB
        temp_dir = Path(tempfile.mkdtemp(prefix="ensembl_"))
        temp_parquet = temp_dir / "ensembl_temp.parquet"
        ensembl_source.sink_parquet(temp_parquet, engine="streaming")
        ensembl_files = [str(temp_parquet)]
    else:
        ensembl_files = _resolve_ensembl_parquet_files(ensembl_source, species)
    
    # Auto-detect memory limit if not specified
    if memory_limit is None:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        memory_gb = max(4, min(int(available_gb * 0.6), 64))
        memory_limit = f"{memory_gb}GB"
    
    # Build DuckDB query
    db_sql = str(db_path).replace("'", "''")
    ensembl_list_sql = "[" + ", ".join(f"'{p.replace(chr(39), chr(39)+chr(39))}'" for p in ensembl_files) + "]"
    module_sql = module_name.replace("'", "''")
    curator_sql = curator.replace("'", "''")
    method_sql = method.replace("'", "''")
    
    con = duckdb.connect()
    
    # Configure DuckDB with rate limiting for HuggingFace (use 2 connections to avoid rate limiting)
    configure_duckdb_for_hf(con, memory_limit=memory_limit, temp_directory=Path(temp_directory), max_connections=2)
    
    # Load httpfs if we have remote files
    if any(p.startswith(("http://", "https://")) for p in ensembl_files):
        # Avoid relying on ~/.duckdb (can be missing/unwritable on some systems).
        from prepare_annotations.core.paths import get_default_cache_dir

        ext_dir = get_default_cache_dir("duckdb") / "extensions"
        ext_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET extension_directory = '{str(ext_dir).replace(chr(39), chr(39)+chr(39))}'")
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
    
    # Single SQL query handles everything:
    # 1. Reads weights from SQLite with variant conclusions
    # 2. Joins with Ensembl (DuckDB optimizes this automatically)
    # 3. Expands genotypes based on zygosity
    # 4. Derives state from weight sign
    query = f"""
    WITH weights_raw AS (
        SELECT 
            aw.rsid, 
            aw.allele, 
            aw.state AS allele_state, 
            aw.zygosity, 
            aw.weight, 
            aw.priority,
            -- Get one conclusion per rsid to avoid row multiplication
            (SELECT conclusions FROM sqlite_scan('{db_sql}', 'variant') WHERE identifier = aw.rsid LIMIT 1) AS conclusion
        FROM sqlite_scan('{db_sql}', 'allele_weights') aw
    ),
    ensembl AS (
        SELECT id, ref, alts
        FROM read_parquet({ensembl_list_sql})
    ),
    -- Homozygous: allele "C" -> ["C", "C"]
    hom_genotypes AS (
        SELECT 
            rsid,
            list_sort([allele, allele]) AS genotype,
            weight,
            priority,
            conclusion
        FROM weights_raw
        WHERE zygosity = 'hom'
    ),
    -- Heterozygous + spec: allele already contains full genotype "CT" -> ["C", "T"]
    het_spec_genotypes AS (
        SELECT
            rsid,
            list_sort([substr(allele, 1, 1), substr(allele, 2, 1)]) AS genotype,
            weight,
            priority,
            conclusion
        FROM weights_raw
        WHERE zygosity = 'het' AND allele_state = 'spec' AND length(allele) = 2
    ),
    -- Heterozygous + alt: join with Ensembl to get all possible other alleles
    -- Includes strand normalization: try both original allele and its complement
    het_alt_with_ensembl AS (
        SELECT 
            w.rsid,
            w.allele AS curated_allele,
            -- Compute complement for strand normalization (A<->T, C<->G)
            CASE w.allele 
                WHEN 'A' THEN 'T' 
                WHEN 'T' THEN 'A' 
                WHEN 'C' THEN 'G' 
                WHEN 'G' THEN 'C' 
                ELSE w.allele 
            END AS curated_allele_complement,
            w.weight,
            w.priority,
            w.conclusion,
            e.ref,
            e.alts,
            -- Determine if we matched via complement (for correct genotype construction)
            CASE 
                WHEN w.allele = e.ref OR list_contains(COALESCE(e.alts, []), w.allele) THEN FALSE
                ELSE TRUE
            END AS used_complement
        FROM weights_raw w
        JOIN ensembl e ON e.id = w.rsid
        WHERE w.zygosity = 'het' AND w.allele_state = 'alt'
          -- Match either original allele or its complement
          AND (
              w.allele = e.ref 
              OR list_contains(COALESCE(e.alts, []), w.allele)
              OR CASE w.allele WHEN 'A' THEN 'T' WHEN 'T' THEN 'A' WHEN 'C' THEN 'G' WHEN 'G' THEN 'C' ELSE w.allele END = e.ref
              OR list_contains(COALESCE(e.alts, []), CASE w.allele WHEN 'A' THEN 'T' WHEN 'T' THEN 'A' WHEN 'C' THEN 'G' WHEN 'G' THEN 'C' ELSE w.allele END)
          )
    ),
    -- Expand het+alt: curated allele pairs with each valid other allele
    -- When complement was used, use complemented allele in genotype
    het_alt_expanded AS (
        SELECT
            rsid,
            list_sort([
                CASE WHEN used_complement THEN curated_allele_complement ELSE curated_allele END, 
                other_allele
            ]) AS genotype,
            weight,
            priority,
            conclusion
        FROM (
            -- Pair with ref if ref != effective curated allele
            SELECT rsid, curated_allele, curated_allele_complement, used_complement, ref AS other_allele, weight, priority, conclusion
            FROM het_alt_with_ensembl
            WHERE ref != CASE WHEN used_complement THEN curated_allele_complement ELSE curated_allele END
            UNION ALL
            -- Pair with each alt that != effective curated allele
            SELECT h.rsid, h.curated_allele, h.curated_allele_complement, h.used_complement, alt.alt AS other_allele, h.weight, h.priority, h.conclusion
            FROM het_alt_with_ensembl h, UNNEST(COALESCE(h.alts, [])) AS alt(alt)
            WHERE alt.alt != CASE WHEN h.used_complement THEN h.curated_allele_complement ELSE h.curated_allele END
        )
    ),
    -- Combine all genotype sources
    all_genotypes AS (
        SELECT * FROM hom_genotypes
        UNION ALL
        SELECT * FROM het_spec_genotypes
        UNION ALL
        SELECT * FROM het_alt_expanded
    ),
    -- Deduplicate and add metadata (use FIRST_VALUE for conclusion)
    deduplicated AS (
        SELECT DISTINCT
            rsid,
            genotype,
            weight,
            priority,
            conclusion
        FROM all_genotypes
    )
    SELECT
        rsid,
        genotype,
        '{module_sql}' AS module,
        weight,
        CASE
            WHEN weight > 0 THEN 'protective'
            WHEN weight < 0 THEN 'risk'
            ELSE 'neutral'
        END AS state,
        priority,
        conclusion,
        '{curator_sql}' AS curator,
        '{method_sql}' AS method
    FROM deduplicated
    """
    
    # Execute query with retry logic for HTTP 429 errors
    # Use longer wait times: 10s, 20s, 40s, 80s, 160s (capped at 120s)
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=5, min=10, max=120),
        reraise=True
    )
    def execute_with_retry():
        """Execute DuckDB query with retry logic for rate limiting."""
        try:
            result = con.execute(query).pl()
            log_message(
                message_type="duckdb_query_success",
                module=module_name,
                num_files=len(ensembl_files)
            )
            return result
        except Exception as e:
            error_msg = str(e)
            # Check if this is an HTTP 429 error that should be retried
            if "HTTP 429" in error_msg or "Too Many Requests" in error_msg or "HTTPException" in str(type(e)):
                log_message(
                    message_type="duckdb_http_429_retry",
                    module=module_name,
                    error=error_msg,
                    error_type=str(type(e)),
                    retry_wait="exponential backoff (10s -> 120s)"
                )
                raise  # Trigger retry
            else:
                # For non-429 errors, log and re-raise without retry
                log_message(
                    message_type="duckdb_query_error",
                    module=module_name,
                    error=error_msg,
                    error_type=str(type(e))
                )
                raise
    
    try:
        result_df = execute_with_retry()
    finally:
        con.close()
    
    return result_df.lazy()


def _resolve_ensembl_parquet_files(source: Path | str, species: str) -> list[str]:
    """Resolve Ensembl parquet file paths from a source path or HuggingFace URI."""
    source_str = str(source)
    
    if source_str.startswith("hf://datasets/"):
        from huggingface_hub import HfApi
        repo_id = source_str.split("hf://datasets/", 1)[1].strip("/")
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = [f for f in repo_files if f.startswith("data/") and f.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in HuggingFace repo {repo_id}")
        return [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in parquet_files]
    
    source_path = Path(source_str)
    if source_path.is_file():
        return [str(source_path)]
    if not source_path.exists():
        raise FileNotFoundError(f"Ensembl source not found: {source_path}")
    
    # Try species-specific pattern first
    parquet_files = sorted(source_path.glob(f"{species}-chr*.parquet"))
    if not parquet_files:
        parquet_files = sorted(source_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {source_path}")
    return [str(p) for p in parquet_files]

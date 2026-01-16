"""
Convert LongevityMap SQLite data to unified annotation schema.

Outputs three Parquet files:
- annotations.parquet: Variant-level facts
- studies.parquet: Per-study evidence  
- weights.parquet: Curator-defined scoring
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
from eliot import start_action


def normalize_genotype(genotype: str | None) -> list[str] | None:
    """Normalize genotype to alphabetical list of alleles (e.g., 'GA' -> ['A', 'G'])."""
    if genotype is None:
        return None
    return sorted(list(genotype))


def derive_state_from_weight(weight: float | None) -> str:
    """Derive semantic state from weight value."""
    if weight is None:
        return "neutral"
    if weight > 0:
        return "protective"
    elif weight < 0:
        return "risk"
    return "neutral"


def load_longevitymap_raw(db_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Load raw data from LongevityMap SQLite database.
    
    Returns:
        Tuple of (weights_raw, variants_raw, categories)
    """
    conn = sqlite3.connect(db_path)
    
    # Load allele weights with category names
    weights_query = """
    SELECT 
        aw.rsid,
        aw.allele,
        aw.state as allele_state,
        aw.zygosity,
        aw.weight,
        aw.priority,
        c.name as category
    FROM allele_weights aw
    JOIN categories c ON c.id = aw.category_id
    """
    weights_raw = pl.read_database(weights_query, connection=conn).lazy()
    
    # Load variants with gene and population info
    variants_query = """
    SELECT 
        v.identifier as rsid,
        g.symbol as gene,
        v.study_design,
        v.conclusions,
        v.association,
        v.quickpubmed as pmid,
        p.name as population
    FROM variant v
    LEFT JOIN gene g ON v.gene_id = g.id
    JOIN population p ON v.population_id = p.id
    """
    variants_raw = pl.read_database(variants_query, connection=conn).lazy()
    
    conn.close()
    
    return weights_raw, variants_raw


def convert_longevitymap_annotations(
    db_path: Path,
) -> pl.LazyFrame:
    """
    Convert LongevityMap to annotations.parquet format.
    
    Schema: rsid, module, gene, phenotype, category
    """
    with start_action(action_type="convert_longevitymap_annotations", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        # Get unique rsid -> gene, category mappings
        query = """
        SELECT DISTINCT
            v.identifier as rsid,
            g.symbol as gene,
            c.name as category
        FROM variant v
        LEFT JOIN gene g ON v.gene_id = g.id
        JOIN allele_weights aw ON aw.rsid = v.identifier
        JOIN categories c ON c.id = aw.category_id
        WHERE v.identifier IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("longevitymap").alias("module"),
            pl.lit("longevity").alias("phenotype"),
        ).select("rsid", "module", "gene", "phenotype", "category")


def convert_longevitymap_studies(
    db_path: Path,
) -> pl.LazyFrame:
    """
    Convert LongevityMap to studies.parquet format.
    
    Schema: rsid, module, pmid, population, p_value, conclusion, study_design
    """
    with start_action(action_type="convert_longevitymap_studies", db_path=str(db_path)):
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT DISTINCT
            v.identifier as rsid,
            v.quickpubmed as pmid,
            p.name as population,
            v.conclusions as conclusion,
            v.study_design
        FROM variant v
        JOIN population p ON v.population_id = p.id
        WHERE v.identifier IS NOT NULL
        """
        df = pl.read_database(query, connection=conn).lazy()
        conn.close()
        
        return df.with_columns(
            pl.lit("longevitymap").alias("module"),
            pl.lit(None).cast(pl.Utf8).alias("p_value"),
        ).select("rsid", "module", "pmid", "population", "p_value", "conclusion", "study_design")


def convert_longevitymap_weights(
    db_path: Path,
    ensembl_cache: Path | None = None,
    curator: str = "Olga Borysova",
    method: str = "literature_review",
) -> pl.LazyFrame:
    """
    Convert LongevityMap to weights.parquet format.
    
    Schema: rsid, genotype, module, weight, state, priority, conclusion, curator, method
    
    Args:
        db_path: Path to LongevityMap SQLite database
        ensembl_cache: Path to Ensembl parquet cache (needed for het genotype construction)
        curator: Curator name
        method: Curation method
    """
    with start_action(
        action_type="convert_longevitymap_weights",
        db_path=str(db_path),
        ensembl_cache=str(ensembl_cache) if ensembl_cache else None,
    ):
        conn = sqlite3.connect(db_path)
        
        # Load weights with variant conclusions
        weights_query = """
        SELECT 
            aw.rsid,
            aw.allele,
            aw.state as allele_state,
            aw.zygosity,
            aw.weight,
            aw.priority,
            v.conclusions as conclusion
        FROM allele_weights aw
        LEFT JOIN variant v ON v.identifier = aw.rsid
        """
        weights_raw = pl.read_database(weights_query, connection=conn).lazy()
        conn.close()
        
        # For homozygous: genotype = allele + allele
        # For heterozygous with state='spec': allele already contains full genotype
        # For heterozygous with state='alt': need ref from VCF
        
        # Handle hom and spec cases first (no VCF needed)
        hom_weights = weights_raw.filter(pl.col("zygosity") == "hom").with_columns(
            (pl.col("allele") + pl.col("allele")).alias("genotype_raw")
        )
        
        spec_weights = weights_raw.filter(
            (pl.col("zygosity") == "het") & (pl.col("allele_state") == "spec")
        ).with_columns(
            pl.col("allele").alias("genotype_raw")  # Already contains full genotype
        )
        
        # For het + alt state: need VCF ref
        het_alt_weights = weights_raw.filter(
            (pl.col("zygosity") == "het") & (pl.col("allele_state") == "alt")
        )
        
        if ensembl_cache is not None and ensembl_cache.exists():
            # Join with Ensembl to get ref allele
            genome = pl.scan_parquet(str(ensembl_cache / "*.parquet"), low_memory=True)
            
            het_with_ref = (
                het_alt_weights
                .join(
                    genome.select(pl.col("id").alias("rsid"), "ref"),
                    on="rsid",
                    how="left"
                )
                .with_columns(
                    (pl.col("ref") + pl.col("allele")).alias("genotype_raw")
                )
                .drop("ref")
            )
        else:
            # Without VCF, we can't construct het genotypes properly
            # Use allele + "?" as placeholder
            het_with_ref = het_alt_weights.with_columns(
                (pl.col("allele") + pl.lit("?")).alias("genotype_raw")
            )
        
        # Combine all weight types
        combined = pl.concat([hom_weights, spec_weights, het_with_ref])
        
        # Normalize genotype and derive state
        normalized = combined.with_columns(
            # Normalize genotype alphabetically as a list
            pl.when(pl.col("genotype_raw").str.len_chars() == 2)
            .then(
                pl.when(pl.col("genotype_raw").str.slice(0, 1) > pl.col("genotype_raw").str.slice(1, 1))
                .then(
                    pl.concat_list([
                        pl.col("genotype_raw").str.slice(1, 1),
                        pl.col("genotype_raw").str.slice(0, 1)
                    ])
                )
                .otherwise(
                    pl.concat_list([
                        pl.col("genotype_raw").str.slice(0, 1),
                        pl.col("genotype_raw").str.slice(1, 1)
                    ])
                )
            )
            .otherwise(
                # Fallback: split and handle non-empty parts if not 2 chars
                pl.col("genotype_raw").str.split("").list.slice(1, -1)
            )
            .alias("genotype"),
            # Derive state from weight sign
            pl.when(pl.col("weight") > 0)
            .then(pl.lit("protective"))
            .when(pl.col("weight") < 0)
            .then(pl.lit("risk"))
            .otherwise(pl.lit("neutral"))
            .alias("state"),
            # Add module, curator, method
            pl.lit("longevitymap").alias("module"),
            pl.lit(curator).alias("curator"),
            pl.lit(method).alias("method"),
        )
        
        # Deduplicate by (rsid, genotype, module) - keep first occurrence
        # Note: conclusion in weights should be genotype-specific, but longevitymap
        # only has study-level conclusions. We keep NULL for consistency with schema.
        result = (
            normalized
            .group_by(["rsid", "genotype", "module"])
            .agg([
                pl.col("weight").first(),
                pl.col("state").first(),
                pl.col("priority").first(),
                # Set conclusion to NULL as longevitymap doesn't have genotype-specific conclusions
                # Study conclusions are in studies.parquet
                pl.lit(None).cast(pl.Utf8).alias("conclusion"),
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
        
        return result


def convert_longevitymap(
    db_path: Path,
    output_dir: Path,
    ensembl_cache: Path | None = None,
    curator: str = "Olga Borysova",
    method: str = "literature_review",
) -> dict[str, Path]:
    """
    Convert LongevityMap SQLite to unified annotation schema.
    
    Outputs three Parquet files to output_dir:
    - annotations.parquet
    - studies.parquet
    - weights.parquet
    
    Args:
        db_path: Path to LongevityMap SQLite database
        output_dir: Directory for output Parquet files
        ensembl_cache: Path to Ensembl parquet cache (for genotype construction)
        curator: Curator name for weights
        method: Curation method for weights
        
    Returns:
        Dictionary mapping table names to output paths
    """
    with start_action(
        action_type="convert_longevitymap",
        db_path=str(db_path),
        output_dir=str(output_dir),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # Convert annotations
        annotations = convert_longevitymap_annotations(db_path)
        annotations_path = output_dir / "annotations.parquet"
        annotations.sink_parquet(annotations_path, engine="streaming")
        outputs["annotations"] = annotations_path
        
        # Convert studies
        studies = convert_longevitymap_studies(db_path)
        studies_path = output_dir / "studies.parquet"
        studies.sink_parquet(studies_path, engine="streaming")
        outputs["studies"] = studies_path
        
        # Convert weights
        weights = convert_longevitymap_weights(
            db_path,
            ensembl_cache=ensembl_cache,
            curator=curator,
            method=method,
        )
        weights_path = output_dir / "weights.parquet"
        weights.sink_parquet(weights_path, engine="streaming")
        outputs["weights"] = weights_path
        
        return outputs

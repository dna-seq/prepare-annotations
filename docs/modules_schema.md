# Unified Genomic Annotation Schema

This document describes the unified schema for genomic annotation modules used in the just-dna-seq project.

## Overview

The schema consists of **three Parquet tables** that work together with VCF variant data:

1. **annotations.parquet** — Variant-level facts (what a variant is associated with)
2. **studies.parquet** — Per-study evidence (literature references and study details)
3. **weights.parquet** — Curator-defined scoring (genotype-specific weights)

All tables join with VCF data on `rsid` to provide genomic coordinates and allele information.

---

## VCF Data (Join Source)

The annotation tables are designed to join with VCF/Parquet variant data:

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | String | Chromosome |
| `start` | Int64 | Start position (0-based) |
| `end` | Int64 | End position |
| `id` | String | Variant identifier (rsid) — **JOIN KEY** |
| `ref` | String | Reference allele |
| `alt` | String | Alternative allele |
| `qual` | Float64 | Quality score |
| `filter` | String | Filter status |
| `END` | Int64 | End position (VCF INFO field) |

> **Note**: The `id` field in VCF corresponds to `rsid` in annotation tables.

---

## Table 1: annotations.parquet

**Purpose**: Store variant-level facts — what each variant is associated with.

**Granularity**: One row per `(rsid, module)` combination.

### Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `rsid` | String | ✓ | Variant identifier (e.g., "rs7412") |
| `module` | String | ✓ | Source module name |
| `gene` | String | | Curated gene symbol (may differ from VCF annotations) |
| `phenotype` | String | | Trait or phenotype affected |
| `category` | String | | Category within the module |

### Primary Key

`(rsid, module)`

### Notes

- **gene**: This is the curator-assigned gene, which may differ from automatic VCF annotations (e.g., VEP). Useful when curators have specific gene associations.
- **phenotype**: Describes the trait or condition the variant is associated with (e.g., "longevity", "lipid_metabolism", "coronary_disease").
- **category**: Module-specific categorization (e.g., "lipids", "insulin" for longevitymap; drug names for drugs module).

---

## Table 2: studies.parquet

**Purpose**: Store per-study evidence — literature references and study details.

**Granularity**: One row per `(rsid, module, pmid)` combination.

### Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `rsid` | String | ✓ | Variant identifier |
| `module` | String | ✓ | Source module name |
| `pmid` | String | | PubMed ID |
| `population` | String | | Study population (e.g., "European", "Asian") |
| `p_value` | String | | Statistical significance (kept as string due to mixed formats) |
| `conclusion` | String | | Study-specific conclusion text |
| `study_design` | String | | Study design description |

### Primary Key

`(rsid, module, pmid)`

### Notes

- **pmid**: May need parsing from legacy data. Some sources store as `"[PMID 12345]; [PMID 67890]"`, others as plain `"12345"`.
- **p_value**: Stored as string because formats vary (e.g., `"2.00E-28"`, `"< 0.05"`, `"= 0.017"`, `"1 x 10-7"`).
- **Multiple studies per variant**: This is expected and normal. A single rsid can have many studies across different populations.

---

## Table 3: weights.parquet

**Purpose**: Store curator-defined scoring — genotype-specific weights for variant interpretation.

**Granularity**: One row per `(rsid, genotype, module)` combination.

### Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `rsid` | String | ✓ | Variant identifier |
| `genotype` | List[String] | ✓ | Genotype this weight applies to (NORMALIZED alphabetically) |
| `module` | String | ✓ | Source module name |
| `weight` | Float64 | | Numeric weight/score (nullable) |
| `state` | String | | Effect direction: risk/protective/neutral/significant |
| `priority` | String | | Priority level (module-specific) |
| `conclusion` | String | | Genotype-specific text description |
| `curator` | String | ✓ | Curator organization |
| `method` | String | ✓ | Curation method |

### Primary Key

`(rsid, genotype, module)`

### Notes

- **genotype**: Normalized to alphabetical order (e.g., `"AG"` not `"GA"`, `"CT"` not `"TC"`). This ensures consistent matching regardless of source format.
- **weight**: Nullable because some modules (e.g., superhuman) have qualitative annotations without numeric scores.
- **state**: Semantic effect direction only. Values: `risk`, `protective`, `neutral`, `significant`, `not_significant`. Does NOT include `ref`/`alt` (computed at query time).
- **curator/method**: Required for provenance tracking.

---

## Computable Fields

The following fields are **not stored** but can be computed at query time:

| Field | Formula | Description |
|-------|---------|-------------|
| `zygosity` | `"hom" if genotype[0] == genotype[1] else "het"` | Homozygous or heterozygous |
| `allele_type` | `"ref" if effect_allele == vcf.ref else "alt"` | Whether the allele is reference or alternative |
| `is_homozygous` | `genotype[0] == genotype[1]` | Boolean for homozygosity |

### Polars Example

```python
import polars as pl

weights = pl.scan_parquet("weights.parquet")

# Add computed zygosity
weights_with_zygosity = weights.with_columns(
    pl.when(pl.col("genotype").list.get(0) == pl.col("genotype").list.get(1))
    .then(pl.lit("hom"))
    .otherwise(pl.lit("het"))
    .alias("zygosity")
)
```

---

## Genotype Normalization

Genotypes are stored in **alphabetical order** for consistent matching:

| Original | Normalized | Zygosity |
|----------|------------|----------|
| `"GA"` | `["A", "G"]` | het |
| `"TC"` | `["C", "T"]` | het |
| `"AA"` | `["A", "A"]` | hom |
| `"GG"` | `["G", "G"]` | hom |

### Normalization Function

```python
def normalize_genotype(genotype: str) -> list[str]:
    """Normalize genotype to alphabetical list of alleles."""
    if genotype is None:
        return None
    return sorted(list(genotype))
```

### Polars Normalization

```python
df = df.with_columns(
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
        pl.col("genotype_raw").str.split("").list.slice(1, -1)
    )
    .alias("genotype")
)
```

---

## State Values

The `state` field in weights uses these semantic values:

| Value | Description |
|-------|-------------|
| `risk` | Increases disease/negative outcome risk |
| `protective` | Decreases disease/negative outcome risk |
| `neutral` | No significant effect (typically weight ≈ 0) |
| `significant` | Statistically significant (used by drugs module) |
| `not_significant` | Not statistically significant |

> **Note**: `ref` and `alt` are **not** stored as state values. Whether an allele is reference or alternative can be computed by comparing with VCF data.

---

## Curator Metadata

Each weight entry includes provenance information:

| Module | Curator | Method |
|--------|---------|--------|
| `longevitymap` | LongevityMap | literature_review |
| `lipidmetabolism` | just-dna-seq | literature_review |
| `vo2max` | just-dna-seq | literature_review |
| `superhuman` | just-dna-seq | literature_review |
| `coronary` | just-dna-seq | gwas_literature |
| `drugs` | PharmGKB | pharmacogenomics_db |

---

## Query Patterns

### Basic Join with VCF

```python
import polars as pl

# Load data
vcf = pl.scan_parquet("vcf/*.parquet")
annotations = pl.scan_parquet("annotations.parquet")

# Join annotations with VCF
annotated = (
    vcf
    .rename({"id": "rsid"})
    .join(annotations, on="rsid")
)
```

### Get Weights for User Genotypes

```python
# User's genotype data (from VCF)
user_genotypes = pl.DataFrame({
    "rsid": ["rs7412", "rs429358"],
    "user_genotype": ["CT", "TC"]  # Will be normalized
})

# Normalize user genotypes
user_genotypes = user_genotypes.with_columns(
    pl.col("user_genotype").map_elements(
        lambda g: "".join(sorted(g)) if g and len(g) == 2 else g,
        return_dtype=pl.Utf8
    ).alias("user_genotype")
)

# Load weights
weights = pl.scan_parquet("weights.parquet")

# Join to get applicable weights
scored = user_genotypes.join(
    weights.collect(),
    left_on=["rsid", "user_genotype"],
    right_on=["rsid", "genotype"]
)
```

### Get All Studies for a Variant

```python
studies = pl.scan_parquet("studies.parquet")

# All studies for rs7412
rs7412_studies = (
    studies
    .filter(pl.col("rsid") == "rs7412")
    .collect()
)
```

### Full Pipeline

```python
import polars as pl

# Load all data
vcf = pl.scan_parquet("vcf/*.parquet").rename({"id": "rsid"})
annotations = pl.scan_parquet("annotations.parquet")
studies = pl.scan_parquet("studies.parquet")
weights = pl.scan_parquet("weights.parquet")

# User's normalized genotypes
user_genotypes = pl.DataFrame({
    "rsid": ["rs7412"],
    "genotype": ["CT"]
})

# Get full annotation with weights
result = (
    user_genotypes
    .join(vcf.collect(), on="rsid")
    .join(annotations.collect(), on="rsid")
    .join(
        weights.collect(),
        on=["rsid", "genotype", "module"],
        how="left"
    )
    .with_columns(
        # Add computed zygosity
        pl.when(pl.col("genotype").str.slice(0, 1) == pl.col("genotype").str.slice(1, 1))
        .then(pl.lit("hom"))
        .otherwise(pl.lit("het"))
        .alias("zygosity")
    )
)
```

---

## File Organization

```
modules/
├── annotations.parquet      # 5 columns - variant facts
├── studies.parquet          # 7 columns - study evidence
├── weights.parquet          # 8 columns - curator scores
└── by_module/               # Optional: split by module
    ├── longevitymap/
    │   ├── annotations.parquet
    │   ├── studies.parquet
    │   └── weights.parquet
    ├── lipidmetabolism/
    │   └── ...
    └── ...
```

---

## Schema Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    VCF DATA (join source)                        │
├─────────────────────────────────────────────────────────────────┤
│  chrom, start, end, id (rsid), ref, alt, qual, filter, END      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ JOIN ON rsid
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODULE ANNOTATION TABLES                      │
└─────────────────────────────────────────────────────────────────┘

┌───────────────────────────┐
│   annotations.parquet     │
│   (variant-level facts)   │
├───────────────────────────┤
│ PK: rsid, module          │◄──────────────────────┐
├───────────────────────────┤                       │
│   gene                    │                       │
│   phenotype               │                       │
│   category                │                       │
└───────────────────────────┘                       │
           │                                        │
           │ 1:N                                    │
           ▼                                        │
┌───────────────────────────┐                       │
│    studies.parquet        │                       │
│   (per-study evidence)    │                       │
├───────────────────────────┤                       │
│ FK: rsid, module          │───────────────────────┤
│ PK: + pmid                │                       │
├───────────────────────────┤                       │
│   population              │                       │
│   p_value                 │                       │
│   conclusion              │                       │
│   study_design            │                       │
└───────────────────────────┘                       │
                                                    │
┌───────────────────────────┐                       │
│    weights.parquet        │                       │
│  (curator-defined scores) │                       │
├───────────────────────────┤                       │
│ FK: rsid                  │───────────────────────┘
│ PK: + genotype, module    │
├───────────────────────────┤
│   weight                  │
│   state                   │
│   priority                │
│   conclusion              │
├───────────────────────────┤
│   curator                 │
│   method                  │
└───────────────────────────┘

  COMPUTABLE (not stored):
    zygosity    ← from genotype
    allele_type ← from comparing with VCF ref/alt
```

---

## Migration from Legacy Modules

The following legacy modules can be converted to this schema:

| Legacy Module | Annotations | Studies | Weights | Notes |
|---------------|-------------|---------|---------|-------|
| just_longevitymap | ✓ | ✓ | ✓ | Full conversion |
| just_lipidmetabolism | ✓ | ✓ | ✓ | Full conversion |
| just_vo2max | ✓ | ✓ | ✓ | Full conversion |
| just_superhuman | ✓ | ✓ | ✓ | No numeric weights (NULL) |
| just_coronary | ✓ | ✓ | ✓ | Full conversion |
| just_drugs | ✓ | ✓ | ✓ | Missing some allele info |

---

## Design Decisions

### Why three tables?

1. **Separation of concerns**: Facts, evidence, and scoring are distinct concepts
2. **Different update frequencies**: Weights may update independently of study evidence
3. **Query efficiency**: Most queries need only one or two tables

### Why normalize genotypes?

Different sources use different conventions (`"AG"` vs `"GA"`). Alphabetical normalization ensures consistent matching.

### Why store p_value as string?

P-values come in many formats (`"2.00E-28"`, `"< 0.05"`, `"1 x 10-7"`). Storing as string preserves original precision and format.

### Why not store zygosity?

Zygosity is trivially computable from genotype (`genotype[0] == genotype[1]`). Storing it would be redundant.

### Why not store allele_type (ref/alt)?

This requires VCF data to compute. Since we always join with VCF, it can be computed at query time.

---

## Version History

- **v1.0** (2026-01): Initial unified schema design

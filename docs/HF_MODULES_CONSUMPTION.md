# Hugging Face Module Consumption Guide

This document explains how genomic annotation modules are organized in the [just-dna-seq/annotators](https://huggingface.co/datasets/just-dna-seq/annotators) Hugging Face repository and how to consume them in Dagster pipelines.

---

## Repository Organization

The repository follows a structured layout where each annotation module has its own directory under `data/`. Each module directory contains three standardized Parquet files.

### Directory Structure
```
just-dna-seq/annotators/
├── data/
│   ├── longevitymap/
│   │   ├── annotations.parquet
│   │   ├── studies.parquet
│   │   └── weights.parquet
│   ├── lipidmetabolism/
│   │   ├── annotations.parquet
│   │   ├── studies.parquet
│   │   └── weights.parquet
│   ├── vo2max/
│   │   └── ...
│   ├── superhuman/
│   │   └── ...
│   └── coronary/
│       └── ...
└── README.md
```

### Available Modules
- `longevitymap`: Longevity-associated variants
- `lipidmetabolism`: Lipid metabolism and cardiovascular risk
- `vo2max`: Athletic performance and VO2max
- `superhuman`: Elite performance and rare beneficial variants
- `coronary`: Coronary artery disease associations
- `drugs`: Pharmacogenomic annotations (PharmGKB)

---

## Module Formats & Detailed Schema

Each module provides three tables designed to be joined with VCF data on `rsid`. Genotypes in the weights table are normalized to alphabetical allele lists.

### 1. annotations.parquet (Variant Facts)
**Purpose**: Store variant-level facts — what each variant is associated with.
**Granularity**: One row per `(rsid, module)` combination.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `rsid` | String | ✓ | Variant identifier (e.g., "rs7412") |
| `module` | String | ✓ | Source module name |
| `gene` | String | | Curated gene symbol (may differ from VEP) |
| `phenotype` | String | | Trait or phenotype affected |
| `category` | String | | Category within the module |

### 2. studies.parquet (Literature Evidence)
**Purpose**: Store per-study evidence — literature references and study details.
**Granularity**: One row per `(rsid, module, pmid)` combination.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `rsid` | String | ✓ | Variant identifier |
| `module` | String | ✓ | Source module name |
| `pmid` | String | | PubMed ID |
| `population` | String | | Study population (e.g., "European", "Asian") |
| `p_value` | String | | Statistical significance (kept as string) |
| `conclusion` | String | | Study-specific conclusion text |
| `study_design` | String | | Study design description |

### 3. weights.parquet (Curator Scores)
**Purpose**: Store curator-defined scoring — genotype-specific weights for variant interpretation.
**Granularity**: One row per `(rsid, genotype, module)` combination.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `rsid` | String | ✓ | Variant identifier |
| `genotype` | List[String] | ✓ | Normalized genotype (alphabetical allele list) |
| `module` | String | ✓ | Source module name |
| `weight` | Float64 | | Numeric weight/score |
| `state` | String | | Effect direction: `risk`, `protective`, `neutral` |
| `priority` | String | | Priority level (module-specific) |
| `conclusion` | String | | Genotype-specific text description |
| `curator` | String | ✓ | Curator organization |
| `method` | String | ✓ | Curation method |

---

## Data Conventions

### Genotype Normalization
Genotypes are stored in **alphabetical order** for consistent matching:
- `"GA"` → `["A", "G"]`
- `"TC"` → `["C", "T"]`
- `"AA"` → `["A", "A"]`

#### Polars Normalization Example
```python
import polars as pl

def normalize_genotypes(df: pl.LazyFrame, col_name: str = "genotype") -> pl.LazyFrame:
    """Normalize genotype strings (e.g. 'AG') to sorted lists (e.g. ['A', 'G'])."""
    return df.with_columns(
        pl.col(col_name)
        .str.split("")
        .list.slice(1, -1)  # Remove empty strings from split
        .list.sort()
        .alias(col_name)
    )
```

### State Values
The `state` field uses standardized semantic values:
- `risk`: Increases disease/negative outcome risk
- `protective`: Decreases disease/negative outcome risk
- `neutral`: No significant effect
- `significant`: Statistically significant (used by drugs module)

### Computable Fields
The following fields can be computed at query time and are **not stored**:
- `zygosity`: `"hom" if genotype[0] == genotype[1] else "het"`
- `allele_type`: `"ref" if allele == vcf.ref else "alt"` (requires join with VCF)
- `is_homozygous`: `genotype[0] == genotype[1]`

#### Polars Computable Fields Example
```python
# Add computed zygosity
df = df.with_columns(
    pl.when(pl.col("genotype").list.get(0) == pl.col("genotype").list.get(1))
    .then(pl.lit("hom"))
    .otherwise(pl.lit("het"))
    .alias("zygosity")
)
```

---

## Dagster Consumption

To use these modules in a Dagster pipeline, you should define them as `SourceAsset`s or use `AssetSpec`.

### 1. Direct Polars Reading
Polars supports reading directly from Hugging Face using the `hf://` protocol.

```python
import polars as pl

# Load any module's weights
weights = pl.read_parquet("hf://datasets/just-dna-seq/annotators/data/longevitymap/weights.parquet")
```

### 2. Defining Source Assets
In Dagster, you can define these as external sources that your pipeline depends on.

```python
from dagster import SourceAsset, AssetKey

def create_hf_module_asset(module: str, table: str) -> SourceAsset:
    return SourceAsset(
        key=AssetKey(["huggingface", module, table]),
        description=f"{module} {table} from Hugging Face",
        metadata={
            "path": f"hf://datasets/just-dna-seq/annotators/data/{module}/{table}.parquet",
            "format": "parquet"
        }
    )

# Example: LongevityMap Weights
longevity_weights = create_hf_module_asset("longevitymap", "weights")
```

### 3. Consumption in Assets
You can then use these source assets as dependencies in your downstream assets.

```python
from dagster import asset
import polars as pl

@asset(deps=[longevity_weights])
def annotated_vcf(user_vcf: pl.LazyFrame):
    # Load weights from HF
    weights = pl.scan_parquet("hf://datasets/just-dna-seq/annotators/data/longevitymap/weights.parquet")
    
    # Join with user VCF on rsid and normalized genotype
    return user_vcf.join(
        weights,
        left_on=["rsid", "normalized_genotype"],
        right_on=["rsid", "genotype"],
        how="left"
    )
```

### 4. Dynamic Asset Factory (Recommended)
If you need to consume many modules, use a factory pattern to generate the definitions.

```python
MODULES = ["longevitymap", "lipidmetabolism", "vo2max", "superhuman", "coronary"]
TABLES = ["annotations", "studies", "weights"]

hf_assets = [
    create_hf_module_asset(m, t) 
    for m in MODULES for t in TABLES
]
```

---

## Best Practices

1. **Lazy Loading**: Use `pl.scan_parquet()` instead of `pl.read_parquet()` when possible to enable predicate pushdown and efficient joins.
2. **Genotype Normalization**: Always ensure the user's VCF genotypes are normalized (alphabetically sorted list of alleles) before joining with `weights.parquet`.
3. **Caching**: If you perform many runs, consider using a local cache or a `Dagster IO Manager` that mirrors the HF data to local storage to avoid redundant network requests.

"""
Dataset card generator for HuggingFace Hub.

This module generates README.md dataset cards for genomic datasets.
Supports both template-based generation (from .md files) and programmatic generation.
"""

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


def load_template(template_name: str) -> Optional[str]:
    """
    Load a dataset card template from the dataset_cards directory.
    
    Args:
        template_name: Name of the template file (e.g., 'ensembl_card_template.md')
        
    Returns:
        Template content as string, or None if not found
    """
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent.parent
    template_path = project_root / "dataset_cards" / template_name
    
    if template_path.exists():
        return template_path.read_text()
    
    return None


def render_template(
    template_content: str,
    variables: Dict[str, str]
) -> str:
    """
    Render a template by replacing {{variable}} placeholders.
    
    Args:
        template_content: Template string with {{placeholders}}
        variables: Dictionary of variable names to values
        
    Returns:
        Rendered template string
    """
    result = template_content
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


def generate_ensembl_card(
    num_files: int,
    total_size_gb: float,
    variant_types: Optional[List[str]] = None,
    version: Optional[str] = None,
    use_template: bool = True,
) -> str:
    """
    Generate a dataset card for Ensembl variations dataset.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    variant_types_section = ""
    if variant_types:
        variant_list = "\n".join(f"- `{vt}`" for vt in sorted(variant_types))
        variant_types_section = f"""
## Dataset Structure

Parquet files are stored directly under `data/` (no per-type subfolders).
Variant types are encoded in the `TSA` column, with common values:

{variant_list}
"""
    
    if use_template:
        template = load_template("ensembl_card_template.md")
        if template:
            variables = {
                "update_date": date_str,
                "num_files": str(num_files),
                "total_size_gb": f"{total_size_gb:.1f}",
                "variant_types_section": variant_types_section,
                "current_year": str(current_year),
            }
            return render_template(template, variables)
    
    version_info = f"Version: {version}" if version else f"Updated: {date_str}"
    
    card = f"""---
language:
  - en
license: apache-2.0
task_categories:
  - tabular-classification
tags:
  - biology
  - genomics
  - variant-annotation
  - ensembl
  - vcf
  - variants
  - parquet
  - bioinformatics
pretty_name: Ensembl Variations (Parquet)
size_categories:
  - 10G<n<100G
---

# Ensembl Variations (Parquet Format)

This dataset contains Ensembl human genetic variations converted to Parquet format for fast and efficient VCF annotation.

## Dataset Description

- **Purpose**: Fast annotation of VCF files with Ensembl variation data
- **Format**: Apache Parquet (columnar storage)
- **Source**: [Ensembl Variation Database](https://www.ensembl.org/info/genome/variation/)
- **{version_info}**
- **Total Files**: {num_files}
- **Total Size**: ~{total_size_gb:.1f} GB
{variant_types_section}

## Usage

### With Polars (Recommended)

```python
import polars as pl

# Load variants for chromosome 21
df = pl.scan_parquet("hf://datasets/just-dna-seq/ensembl_variations/data/homo_sapiens-chr21.parquet")

# Filter variants by position
variants = df.filter(
    (pl.col("POS") >= 10000000) & (pl.col("POS") <= 20000000)
).collect()

print(variants)
```

## License

This dataset is released under Apache 2.0 license. The original Ensembl data is available under their terms of use.

## Maintenance

This dataset is maintained by the GenoBear project.
- GitHub: [https://github.com/dna-seq/just-dna-lite](https://github.com/dna-seq/just-dna-lite)
- HuggingFace: [https://huggingface.co/just-dna-seq](https://huggingface.co/just-dna-seq)
"""
    
    return card


def generate_clinvar_card(
    num_files: int,
    total_size_gb: float,
    variant_types: Optional[List[str]] = None,
    version: Optional[str] = None,
    use_template: bool = True,
) -> str:
    """
    Generate a dataset card for ClinVar dataset.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    variant_types_section = ""
    if variant_types:
        variant_list = "\n".join(f"- `{vt}`" for vt in sorted(variant_types))
        variant_types_section = f"""
## Dataset Structure

Parquet files are stored directly under `data/` (no per-type subfolders).
Variant types are encoded in the `CLNVC` column, with common values:

{variant_list}
"""
    
    if use_template:
        template = load_template("clinvar_card_template.md")
        if template:
            variables = {
                "update_date": date_str,
                "num_files": str(num_files),
                "total_size_gb": f"{total_size_gb:.1f}",
                "variant_types_section": variant_types_section,
                "current_year": str(current_year),
            }
            return render_template(template, variables)
    
    version_info = f"Version: {version}" if version else f"Updated: {date_str}"
    
    card = f"""---
language:
  - en
license: other
task_categories:
  - tabular-classification
tags:
  - biology
  - genomics
  - variant-annotation
  - clinical-genomics
  - clinvar
  - clinical-variants
  - vcf
  - parquet
  - bioinformatics
  - pathogenicity
pretty_name: ClinVar (Parquet Format)
size_categories:
  - 1G<n<10G
---

# ClinVar (Parquet Format)

This dataset contains ClinVar clinical variant data converted to Parquet format.

## Dataset Description

- **Purpose**: Fast clinical annotation of VCF files
- **Format**: Apache Parquet (columnar storage)
- **Source**: [ClinVar Database](https://www.ncbi.nlm.nih.gov/clinvar/)
- **{version_info}**
- **Total Files**: {num_files}
- **Total Size**: ~{total_size_gb:.1f} GB
{variant_types_section}

## Clinical Use Disclaimer

⚠️ **Important**: This data is for research purposes only.

## Maintenance

This dataset is maintained by the GenoBear project.
"""
    
    return card


def generate_dbsnp_card(
    num_files: int,
    total_size_gb: float,
    variant_types: Optional[List[str]] = None,
    version: Optional[str] = None,
    use_template: bool = True,
) -> str:
    """
    Generate a dataset card for dbSNP dataset.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    variant_types_section = ""
    if variant_types:
        variant_list = "\n".join(f"- `{vt}`" for vt in sorted(variant_types))
        variant_types_section = f"""
## Dataset Structure

Variant types are encoded in the data columns (e.g. `TSA`), with common values:

{variant_list}
"""
    
    version_info = f"Version: {version}" if version else f"Updated: {date_str}"
    
    card = f"""---
language:
  - en
license: other
task_categories:
  - tabular-classification
tags:
  - biology
  - genomics
  - variant-annotation
  - dbsnp
  - rsid
  - vcf
  - parquet
  - bioinformatics
pretty_name: dbSNP (Parquet Format)
size_categories:
  - 100G<n<1T
---

# dbSNP (Parquet Format)

This dataset contains dbSNP variant data converted to Parquet format.

## Dataset Description

- **Purpose**: Fast rsID lookup and variant annotation
- **Format**: Apache Parquet (columnar storage)
- **Source**: [dbSNP Database](https://www.ncbi.nlm.nih.gov/snp/)
- **{version_info}**
- **Total Files**: {num_files}
- **Total Size**: ~{total_size_gb:.1f} GB
{variant_types_section}

## Maintenance

This dataset is maintained by the GenoBear project.
"""
    return card


def generate_dbsnp_t2t_card(
    num_files: int,
    total_size_gb: float,
    variant_types: Optional[List[str]] = None,
    version: Optional[str] = None,
    use_template: bool = True,
) -> str:
    """
    Generate a dataset card for dbSNP T2T (CHM13) dataset.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    card = f"""---
language:
  - en
license: other
task_categories:
  - tabular-classification
tags:
  - biology
  - genomics
  - variant-annotation
  - dbsnp
  - t2t
  - chm13
  - rsid
  - vcf
  - parquet
  - bioinformatics
pretty_name: dbSNP T2T CHM13 (Parquet Format)
size_categories:
  - 10G<n<100G
---

# dbSNP T2T CHM13 (Parquet Format)

This dataset contains dbSNP variant data lifted over to the T2T-CHM13 v2.0 assembly.

## Dataset Description

- **Purpose**: Fast rsID lookup for T2T-CHM13 assembly
- **Format**: Apache Parquet (columnar storage)
- **Source**: [T2T-CHM13 Assemblies](https://github.com/marbl/CHM13) / [dbSNP](https://www.ncbi.nlm.nih.gov/snp/)
- **Updated**: {date_str}
- **Total Files**: {num_files}
- **Total Size**: ~{total_size_gb:.1f} GB

## Maintenance

This dataset is maintained by the GenoBear project.
"""
    return card


def generate_gnomad_card(
    num_files: int,
    total_size_gb: float,
    variant_types: Optional[List[str]] = None,
    version: Optional[str] = None,
    use_template: bool = True,
) -> str:
    """
    Generate a dataset card for gnomAD dataset.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    variant_types_section = ""
    if variant_types:
        variant_list = "\n".join(f"- `{vt}`" for vt in sorted(variant_types))
        variant_types_section = f"""
## Dataset Structure

Variant types are encoded in the data columns, with common values:

{variant_list}
"""
    
    version_info = f"Version: {version}" if version else f"Updated: {date_str}"
    
    card = f"""---
language:
  - en
license: odc-by
task_categories:
  - tabular-classification
tags:
  - biology
  - genomics
  - variant-annotation
  - gnomad
  - allele-frequency
  - vcf
  - parquet
  - bioinformatics
pretty_name: gnomAD (Parquet Format)
size_categories:
  - 1T<n<10T
---

# gnomAD (Parquet Format)

This dataset contains gnomAD variant data converted to Parquet format.

## Dataset Description

- **Purpose**: Population allele frequency annotation
- **Format**: Apache Parquet (columnar storage)
- **Source**: [gnomAD](https://gnomad.broadinstitute.org/)
- **{version_info}**
- **Total Files**: {num_files}
- **Total Size**: ~{total_size_gb:.1f} GB
{variant_types_section}

## Maintenance

This dataset is maintained by the GenoBear project.
"""
    return card


def save_dataset_card(
    card_content: str,
    output_path: Path
) -> Path:
    """
    Save dataset card to a file.
    
    Args:
        card_content: Markdown content of the dataset card
        output_path: Path where to save the README.md file
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card_content)
    return output_path

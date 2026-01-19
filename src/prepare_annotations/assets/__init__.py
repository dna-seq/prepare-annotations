"""
Dagster assets for genomic data preparation.

This module contains Dagster asset definitions organized by domain:
- ensembl: Ensembl VCF download and conversion assets
- modules: OakVar module conversion assets
"""
from prepare_annotations.assets.ensembl import (
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
    ENSEMBL_VCF_PARTITIONS,
    download_retry_policy,
)
from prepare_annotations.assets.modules import (
    ensembl_variations_source,
    # LongevityMap
    longevitymap_sqlite,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
    # LipidMetabolism
    lipidmetabolism_sqlite,
    lipidmetabolism_annotations,
    lipidmetabolism_studies,
    lipidmetabolism_weights,
    lipidmetabolism_with_ensembl,
    lipidmetabolism_hf_upload,
    # VO2Max
    vo2max_sqlite,
    vo2max_annotations,
    vo2max_studies,
    vo2max_weights,
    vo2max_with_ensembl,
    vo2max_hf_upload,
    # Superhuman
    superhuman_sqlite,
    superhuman_annotations,
    superhuman_studies,
    superhuman_weights,
    superhuman_with_ensembl,
    superhuman_hf_upload,
    # Coronary
    coronary_sqlite,
    coronary_annotations,
    coronary_studies,
    coronary_weights,
    coronary_with_ensembl,
    coronary_hf_upload,
    # All module assets
    module_assets,
)

# All assets for easy import
all_assets = [
    # Ensembl VCF pipeline
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
    # Module conversion
    ensembl_variations_source,
    # LongevityMap
    longevitymap_sqlite,
    longevitymap_annotations,
    longevitymap_studies,
    longevitymap_weights,
    longevitymap_with_ensembl,
    longevitymap_hf_upload,
    # LipidMetabolism
    lipidmetabolism_sqlite,
    lipidmetabolism_annotations,
    lipidmetabolism_studies,
    lipidmetabolism_weights,
    lipidmetabolism_with_ensembl,
    lipidmetabolism_hf_upload,
    # VO2Max
    vo2max_sqlite,
    vo2max_annotations,
    vo2max_studies,
    vo2max_weights,
    vo2max_with_ensembl,
    vo2max_hf_upload,
    # Superhuman
    superhuman_sqlite,
    superhuman_annotations,
    superhuman_studies,
    superhuman_weights,
    superhuman_with_ensembl,
    superhuman_hf_upload,
    # Coronary
    coronary_sqlite,
    coronary_annotations,
    coronary_studies,
    coronary_weights,
    coronary_with_ensembl,
    coronary_hf_upload,
]

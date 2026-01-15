"""
Hugging Face Hub integration for prepare-annotations.

This module provides utilities for:
- Uploading parquet datasets to Hugging Face Hub
- Generating dataset cards (README.md)
- Smart upload (skip unchanged files based on size)
"""
from prepare_annotations.huggingface.uploader import (
    upload_to_hf_if_changed,
    collect_parquet_files,
    upload_files_batch,
    upload_parquet_to_hf,
)
from prepare_annotations.huggingface.dataset_cards import (
    load_template,
    render_template,
    generate_ensembl_card,
    generate_clinvar_card,
    generate_dbsnp_card,
    generate_dbsnp_t2t_card,
    generate_gnomad_card,
    save_dataset_card,
)

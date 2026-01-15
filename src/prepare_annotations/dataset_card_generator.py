"""
Dataset card generator for HuggingFace Hub.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.huggingface.dataset_cards
"""
# Re-export from huggingface for backward compatibility
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

"""
Hugging Face dataset uploader.

NOTE: This module is kept for backward compatibility.
New code should use: prepare_annotations.huggingface.uploader
"""
# Re-export from huggingface for backward compatibility
from prepare_annotations.huggingface.uploader import (
    upload_to_hf_if_changed,
    collect_parquet_files,
    upload_files_batch,
    upload_parquet_to_hf,
)
from prepare_annotations.core.models import SingleUploadResult, BatchUploadResult

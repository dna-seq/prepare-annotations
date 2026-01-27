"""
Dagster Definitions for genomic data preparation pipelines.

This module exports the main Definitions object that Dagster uses
to discover all assets, jobs, and resources.

Features:
- Dynamic partitioning based on FTP file discovery
- Per-file lineage tracking in Dagster UI
- Retry policies with exponential backoff
- Checksum verification and resumable downloads
- Concurrency limits to prevent OOM (respects PREPARE_ANNOTATIONS_* env vars)
- OakVar module conversion with Ensembl genotype resolution

Concurrency Control:
    The following environment variables control parallelism:
    - PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS: Max concurrent VCF downloads (default: 4)
    - PREPARE_ANNOTATIONS_PARQUET_WORKERS: Max concurrent parquet conversions (default: 2)
"""

import os

import warnings

from dagster import (
    Definitions,
    define_asset_job,
    AssetSelection,
    success_hook,
    HookContext,
)

# Suppress dagster-polars internal warning about "extension" field shadowing
# This is a known issue in dagster-polars, not our code
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Field name.*extension.*shadows", category=UserWarning)
    from dagster_polars import PolarsParquetIOManager

from prepare_annotations.core.paths import MODULES_OUTPUT_DIR
from prepare_annotations.assets import (
    ensembl_ftp_source,
    ensembl_vcf_urls,
    ensembl_vcf_file,
    ensembl_parquet_file,
    ensembl_all_parquet_files,
    ensembl_hf_upload,
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
    # Asset checks (LazyFrame-based, memory-efficient validation)
    all_module_checks,
)
from prepare_annotations.core.dagster_io_managers import (
    ensembl_cache_io_manager,
    module_io_manager,
    huggingface_upload_io_manager,
)


@success_hook
def resource_summary_hook(context: HookContext) -> None:
    """
    Success hook that logs aggregated resource metrics for the entire run.
    
    This provides run-level visibility into:
    - Total duration across all assets
    - Maximum peak memory (bottleneck identification)
    - Average CPU usage
    
    Appears in the run logs at the end of successful runs.
    """
    # Get all events from this run
    run_id = context.run_id
    instance = context.instance
    
    # Query materialization events for this run (Dagster 1.12.x compatible)
    from dagster import DagsterEventType
    
    # Use all_logs instead of get_event_records (EventRecordsFilter doesn't have run_ids in 1.12.x)
    log_entries = instance.all_logs(run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION)
    
    # Extract resource metrics from asset materializations
    total_duration = 0.0
    max_peak_memory = 0.0
    total_cpu = 0.0
    asset_count = 0
    asset_metrics: list[dict] = []
    
    for entry in log_entries:
        # EventLogEntry.asset_materialization returns Optional[AssetMaterialization] directly
        mat = entry.asset_materialization
        if mat is not None:
            metadata = mat.metadata or {}
            
            duration = metadata.get("duration_sec")
            peak_mem = metadata.get("peak_memory_mb")
            cpu = metadata.get("cpu_percent")
            
            if duration is not None or peak_mem is not None:
                asset_name = mat.asset_key.to_user_string()
                asset_info = {"asset": asset_name}
                
                if duration is not None:
                    dur_val = duration.value if hasattr(duration, 'value') else float(duration)
                    total_duration += dur_val
                    asset_info["duration_sec"] = dur_val
                
                if peak_mem is not None:
                    mem_val = peak_mem.value if hasattr(peak_mem, 'value') else float(peak_mem)
                    max_peak_memory = max(max_peak_memory, mem_val)
                    asset_info["peak_memory_mb"] = mem_val
                
                if cpu is not None:
                    cpu_val = cpu.value if hasattr(cpu, 'value') else float(cpu)
                    total_cpu += cpu_val
                    asset_info["cpu_percent"] = cpu_val
                
                asset_metrics.append(asset_info)
                asset_count += 1
    
    if asset_count == 0:
        context.log.info("No resource metrics found in this run")
        return
    
    avg_cpu = total_cpu / asset_count if asset_count > 0 else 0.0
    
    # Sort by peak memory to identify bottlenecks
    sorted_by_memory = sorted(asset_metrics, key=lambda x: x.get("peak_memory_mb", 0), reverse=True)
    top_memory_assets = sorted_by_memory[:3]
    
    # Log summary
    context.log.info(
        f"📊 RUN RESOURCE SUMMARY\n"
        f"  Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)\n"
        f"  Max Peak Memory: {max_peak_memory:.1f} MB\n"
        f"  Average CPU: {avg_cpu:.1f}%\n"
        f"  Assets with metrics: {asset_count}\n"
        f"  Top memory consumers:\n" +
        "\n".join(f"    - {a['asset']}: {a.get('peak_memory_mb', 0):.1f} MB" for a in top_memory_assets)
    )


def get_download_concurrency_limit() -> int:
    """Get max concurrent VCF downloads from env or default."""
    env_value = os.getenv("PREPARE_ANNOTATIONS_DOWNLOAD_WORKERS")
    if env_value:
        return max(1, int(env_value))
    return 4  # Default: 4 concurrent downloads


def get_parquet_concurrency_limit() -> int:
    """Get max concurrent parquet conversions from env or default."""
    env_value = os.getenv("PREPARE_ANNOTATIONS_PARQUET_WORKERS")
    if env_value:
        return max(1, int(env_value))
    return 2  # Default: 2 concurrent conversions (memory-intensive)


# ============================================================================
# JOBS: Named execution paths through the asset graph
# Short names for easy CLI usage: prepare, download, convert, upload, full
# ============================================================================

# Common hooks for all jobs
_job_hooks = {resource_summary_hook}

# Default job: discover -> download -> convert (per-file partitioned)
prepare_job = define_asset_job(
    name="prepare",
    description="Discover, download and convert Ensembl VCF data to Parquet (per-file partitioned).",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
    ),
    hooks=_job_hooks,
)

# Job to just download VCF files (partitioned with retries)
download_job = define_asset_job(
    name="download",
    description="Download Ensembl VCF files from FTP (per-file partitioned, resumable).",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_file,
    ),
    hooks=_job_hooks,
)

# Job to convert only (assumes VCF files already downloaded)
convert_job = define_asset_job(
    name="convert",
    description="Convert downloaded VCF files to Parquet format (per-file partitioned).",
    selection=AssetSelection.assets(
        ensembl_parquet_file,
    ),
    hooks=_job_hooks,
)

# Job to collect and upload to HuggingFace Hub
upload_job = define_asset_job(
    name="upload",
    description="Collect and upload Ensembl parquet files to HuggingFace Hub.",
    selection=AssetSelection.assets(
        ensembl_all_parquet_files,
        ensembl_hf_upload,
    ),
    hooks=_job_hooks,
)

# Full pipeline: discover → download → convert → collect → upload
full_job = define_asset_job(
    name="full",
    description="Complete pipeline: discover, download, convert, and upload to HuggingFace.",
    selection=AssetSelection.assets(
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
    ),
    hooks=_job_hooks,
)


# ============================================================================
# MODULE CONVERSION JOBS
# ============================================================================

# Per-module output asset lists (for check association via checks_for_assets)
_longevitymap_output_assets = [
    longevitymap_annotations, longevitymap_studies, longevitymap_weights
]
_lipidmetabolism_output_assets = [
    lipidmetabolism_annotations, lipidmetabolism_studies, lipidmetabolism_weights
]
_vo2max_output_assets = [
    vo2max_annotations, vo2max_studies, vo2max_weights
]
_superhuman_output_assets = [
    superhuman_annotations, superhuman_studies, superhuman_weights
]
_coronary_output_assets = [
    coronary_annotations, coronary_studies, coronary_weights
]

# Convert only job: no join, no upload
longevitymap_convert_job = define_asset_job(
    name="longevitymap_convert",
    description="LongevityMap: convert to unified schema only (no join, no upload). Runs checks.",
    selection=AssetSelection.assets(
        longevitymap_sqlite,
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
    ) | AssetSelection.checks_for_assets(*_longevitymap_output_assets),
    hooks=_job_hooks,
)

# Full job: convert + join with Ensembl (no upload)
longevitymap_full_job = define_asset_job(
    name="longevitymap_full",
    description="LongevityMap: convert + join with full Ensembl data (no upload). Runs checks.",
    selection=AssetSelection.assets(
        longevitymap_sqlite,
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_with_ensembl,
    ) | AssetSelection.checks_for_assets(*_longevitymap_output_assets),
    hooks=_job_hooks,
)

# Default job: full pipeline with upload to HuggingFace
longevitymap_job = define_asset_job(
    name="longevitymap",
    description="LongevityMap: full pipeline - convert, join with Ensembl, upload to HuggingFace. Runs checks.",
    selection=AssetSelection.assets(
        longevitymap_sqlite,
        ensembl_variations_source,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_with_ensembl,
        longevitymap_hf_upload,
    ) | AssetSelection.checks_for_assets(*_longevitymap_output_assets),
    hooks=_job_hooks,
)


# ============================================================================
# LIPIDMETABOLISM JOBS
# ============================================================================

lipidmetabolism_convert_job = define_asset_job(
    name="lipidmetabolism_convert",
    description="LipidMetabolism: convert to unified schema only (no Ensembl join, no upload). Runs checks.",
    selection=AssetSelection.assets(
        lipidmetabolism_sqlite,
        lipidmetabolism_annotations,
        lipidmetabolism_studies,
        lipidmetabolism_weights,
    ) | AssetSelection.checks_for_assets(*_lipidmetabolism_output_assets),
    hooks=_job_hooks,
)

lipidmetabolism_full_job = define_asset_job(
    name="lipidmetabolism_full",
    description="LipidMetabolism: convert + join with Ensembl (no upload). Runs checks.",
    selection=AssetSelection.assets(
        lipidmetabolism_sqlite,
        ensembl_variations_source,
        lipidmetabolism_annotations,
        lipidmetabolism_studies,
        lipidmetabolism_weights,
        lipidmetabolism_with_ensembl,
    ) | AssetSelection.checks_for_assets(*_lipidmetabolism_output_assets),
    hooks=_job_hooks,
)

lipidmetabolism_job = define_asset_job(
    name="lipidmetabolism",
    description="LipidMetabolism: full pipeline - convert, join with Ensembl, upload to HuggingFace. Runs checks.",
    selection=AssetSelection.assets(
        lipidmetabolism_sqlite,
        ensembl_variations_source,
        lipidmetabolism_annotations,
        lipidmetabolism_studies,
        lipidmetabolism_weights,
        lipidmetabolism_with_ensembl,
        lipidmetabolism_hf_upload,
    ) | AssetSelection.checks_for_assets(*_lipidmetabolism_output_assets),
    hooks=_job_hooks,
)


# ============================================================================
# VO2MAX JOBS
# ============================================================================

vo2max_convert_job = define_asset_job(
    name="vo2max_convert",
    description="VO2Max: convert to unified schema only (no Ensembl join, no upload). Runs checks.",
    selection=AssetSelection.assets(
        vo2max_sqlite,
        vo2max_annotations,
        vo2max_studies,
        vo2max_weights,
    ) | AssetSelection.checks_for_assets(*_vo2max_output_assets),
    hooks=_job_hooks,
)

vo2max_full_job = define_asset_job(
    name="vo2max_full",
    description="VO2Max: convert + join with Ensembl (no upload). Runs checks.",
    selection=AssetSelection.assets(
        vo2max_sqlite,
        ensembl_variations_source,
        vo2max_annotations,
        vo2max_studies,
        vo2max_weights,
        vo2max_with_ensembl,
    ) | AssetSelection.checks_for_assets(*_vo2max_output_assets),
    hooks=_job_hooks,
)

vo2max_job = define_asset_job(
    name="vo2max",
    description="VO2Max: full pipeline - convert, join with Ensembl, upload to HuggingFace. Runs checks.",
    selection=AssetSelection.assets(
        vo2max_sqlite,
        ensembl_variations_source,
        vo2max_annotations,
        vo2max_studies,
        vo2max_weights,
        vo2max_with_ensembl,
        vo2max_hf_upload,
    ) | AssetSelection.checks_for_assets(*_vo2max_output_assets),
    hooks=_job_hooks,
)


# ============================================================================
# SUPERHUMAN JOBS
# ============================================================================

superhuman_convert_job = define_asset_job(
    name="superhuman_convert",
    description="Superhuman: convert to unified schema only (no Ensembl join, no upload). Runs checks.",
    selection=AssetSelection.assets(
        superhuman_sqlite,
        superhuman_annotations,
        superhuman_studies,
        superhuman_weights,
    ) | AssetSelection.checks_for_assets(*_superhuman_output_assets),
    hooks=_job_hooks,
)

superhuman_full_job = define_asset_job(
    name="superhuman_full",
    description="Superhuman: convert + join with Ensembl (no upload). Runs checks.",
    selection=AssetSelection.assets(
        superhuman_sqlite,
        ensembl_variations_source,
        superhuman_annotations,
        superhuman_studies,
        superhuman_weights,
        superhuman_with_ensembl,
    ) | AssetSelection.checks_for_assets(*_superhuman_output_assets),
    hooks=_job_hooks,
)

superhuman_job = define_asset_job(
    name="superhuman",
    description="Superhuman: full pipeline - convert, join with Ensembl, upload to HuggingFace. Runs checks.",
    selection=AssetSelection.assets(
        superhuman_sqlite,
        ensembl_variations_source,
        superhuman_annotations,
        superhuman_studies,
        superhuman_weights,
        superhuman_with_ensembl,
        superhuman_hf_upload,
    ) | AssetSelection.checks_for_assets(*_superhuman_output_assets),
    hooks=_job_hooks,
)


# ============================================================================
# CORONARY JOBS
# ============================================================================

coronary_convert_job = define_asset_job(
    name="coronary_convert",
    description="Coronary: convert to unified schema only (no Ensembl join, no upload). Runs checks.",
    selection=AssetSelection.assets(
        coronary_sqlite,
        coronary_annotations,
        coronary_studies,
        coronary_weights,
    ) | AssetSelection.checks_for_assets(*_coronary_output_assets),
    hooks=_job_hooks,
)

coronary_full_job = define_asset_job(
    name="coronary_full",
    description="Coronary: convert + join with Ensembl (no upload). Runs checks.",
    selection=AssetSelection.assets(
        coronary_sqlite,
        ensembl_variations_source,
        coronary_annotations,
        coronary_studies,
        coronary_weights,
        coronary_with_ensembl,
    ) | AssetSelection.checks_for_assets(*_coronary_output_assets),
    hooks=_job_hooks,
)

coronary_job = define_asset_job(
    name="coronary",
    description="Coronary: full pipeline - convert, join with Ensembl, upload to HuggingFace. Runs checks.",
    selection=AssetSelection.assets(
        coronary_sqlite,
        ensembl_variations_source,
        coronary_annotations,
        coronary_studies,
        coronary_weights,
        coronary_with_ensembl,
        coronary_hf_upload,
    ) | AssetSelection.checks_for_assets(*_coronary_output_assets),
    hooks=_job_hooks,
)


# ============================================================================
# ALL MODULES JOB
# ============================================================================

# All module output assets (for check association via checks_for_assets)
_all_module_output_assets = (
    _longevitymap_output_assets
    + _lipidmetabolism_output_assets
    + _vo2max_output_assets
    + _superhuman_output_assets
    + _coronary_output_assets
)

all_modules_convert_job = define_asset_job(
    name="all_modules_convert",
    description="Convert all annotation modules to unified schema (no Ensembl join, no upload). Runs asset checks.",
    selection=AssetSelection.assets(
        # LongevityMap
        longevitymap_sqlite,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        # LipidMetabolism
        lipidmetabolism_sqlite,
        lipidmetabolism_annotations,
        lipidmetabolism_studies,
        lipidmetabolism_weights,
        # VO2Max
        vo2max_sqlite,
        vo2max_annotations,
        vo2max_studies,
        vo2max_weights,
        # Superhuman
        superhuman_sqlite,
        superhuman_annotations,
        superhuman_studies,
        superhuman_weights,
        # Coronary
        coronary_sqlite,
        coronary_annotations,
        coronary_studies,
        coronary_weights,
    ) | AssetSelection.checks_for_assets(*_all_module_output_assets),
    hooks=_job_hooks,
)

all_modules_full_job = define_asset_job(
    name="all_modules_full",
    description="All modules: convert + join with Ensembl (no upload). Runs asset checks.",
    selection=AssetSelection.assets(
        ensembl_variations_source,
        # LongevityMap
        longevitymap_sqlite,
        longevitymap_annotations,
        longevitymap_studies,
        longevitymap_weights,
        longevitymap_with_ensembl,
        # LipidMetabolism
        lipidmetabolism_sqlite,
        lipidmetabolism_annotations,
        lipidmetabolism_studies,
        lipidmetabolism_weights,
        lipidmetabolism_with_ensembl,
        # VO2Max
        vo2max_sqlite,
        vo2max_annotations,
        vo2max_studies,
        vo2max_weights,
        vo2max_with_ensembl,
        # Superhuman
        superhuman_sqlite,
        superhuman_annotations,
        superhuman_studies,
        superhuman_weights,
        superhuman_with_ensembl,
        # Coronary
        coronary_sqlite,
        coronary_annotations,
        coronary_studies,
        coronary_weights,
        coronary_with_ensembl,
    ) | AssetSelection.checks_for_assets(*_all_module_output_assets),
    hooks=_job_hooks,
)

all_modules_job = define_asset_job(
    name="all_modules",
    description="All modules: full pipeline - convert, join with Ensembl, upload all to HuggingFace. Runs asset checks.",
    selection=AssetSelection.assets(
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
    ) | AssetSelection.checks_for_assets(*_all_module_output_assets),
    hooks=_job_hooks,
)


# ============================================================================
# DEFINITIONS: Main export for Dagster
# ============================================================================

defs = Definitions(
    assets=[
        # Ensembl VCF pipeline assets (partitioned per-file)
        ensembl_ftp_source,
        ensembl_vcf_urls,
        ensembl_vcf_file,
        ensembl_parquet_file,
        ensembl_all_parquet_files,
        ensembl_hf_upload,
        # Module conversion assets
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
    ],
    jobs=[
        # Ensembl VCF pipeline jobs
        prepare_job,
        download_job,
        convert_job,
        upload_job,
        full_job,
        # LongevityMap jobs
        longevitymap_job,  # Default: full pipeline with upload
        longevitymap_full_job,  # Convert + join (no upload)
        longevitymap_convert_job,  # Convert only (no join, no upload)
        # LipidMetabolism jobs
        lipidmetabolism_job,  # Full pipeline with upload
        lipidmetabolism_full_job,  # Convert + join (no upload)
        lipidmetabolism_convert_job,  # Convert only (no join, no upload)
        # VO2Max jobs
        vo2max_job,  # Full pipeline with upload
        vo2max_full_job,  # Convert + join (no upload)
        vo2max_convert_job,  # Convert only (no join, no upload)
        # Superhuman jobs
        superhuman_job,  # Full pipeline with upload
        superhuman_full_job,  # Convert + join (no upload)
        superhuman_convert_job,  # Convert only (no join, no upload)
        # Coronary jobs
        coronary_job,  # Full pipeline with upload
        coronary_full_job,  # Convert + join (no upload)
        coronary_convert_job,  # Convert only (no join, no upload)
        # All modules jobs
        all_modules_job,  # Full pipeline with upload
        all_modules_full_job,  # Convert + join (no upload)
        all_modules_convert_job,  # Convert only (no join, no upload)
    ],
    resources={
        "io_manager": ensembl_cache_io_manager,
        "module_io_manager": module_io_manager,
        "hf_upload_io_manager": huggingface_upload_io_manager,
        # Polars IO manager for LazyFrame assets - stores to MODULES_OUTPUT_DIR
        "polars_parquet_io_manager": PolarsParquetIOManager(
            base_dir=str(MODULES_OUTPUT_DIR),
            cloud_storage_options=None,
        ),
    },
    asset_checks=all_module_checks,
)

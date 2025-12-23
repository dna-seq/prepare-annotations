"""Prefect-based preparation pipelines for genomic data sources."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from eliot import start_action
from prefect import task, flow, get_run_logger
import polars as pl
from platformdirs import user_cache_dir
from pycomfort.logging import to_nice_stdout, to_nice_file

from prepare_annotations.runtime import prefect_flow_run
from prepare_annotations.models import PreparationResult, SplitResult, BatchUploadResult
from prepare_annotations.preparation.vcf_downloader import (
    list_paths,
    download_path,
    convert_to_parquet,
    validate_downloads_and_parquet,
)
from prepare_annotations.preparation.huggingface_uploader import (
    collect_parquet_files,
    upload_parquet_to_hf,
)
from prepare_annotations.preparation.dataset_card_generator import (
    generate_clinvar_card,
    generate_ensembl_card,
    generate_dbsnp_card,
    generate_gnomad_card,
)

# Prefect tasks for preparation steps
list_paths_task = task(list_paths, name="List Remote Paths")
download_path_task = task(download_path, name="Download Path")
convert_to_parquet_task = task(convert_to_parquet, name="Convert to Parquet")
validate_task = task(validate_downloads_and_parquet, name="Validate Downloads")

@task(name="Split Parquet Files")
def split_parquets_task(
    parquet_paths: List[Path],
    explode_snv_alt: bool = True,
    write_to: Optional[Path] = None,
) -> SplitResult:
    """Split parquet files by variant type (TSA)."""
    from prepare_annotations.preparation.vcf_parquet_splitter import split_variants_by_tsa
    
    results = {}
    for p in parquet_paths:
        split_dict = split_variants_by_tsa(
            parquet_path=p,
            explode_snv_alt=explode_snv_alt,
            write_to=write_to,
        )
        for k, v in split_dict.items():
            if k not in results:
                results[k] = []
            if isinstance(v, list):
                results[k].extend(v)
            else:
                results[k].append(v)
    return SplitResult(split_variants_dict=results)


def get_default_input_dir(name: str) -> Path:
    """Get the default destination directory for downloads."""
    root_dir = Path("data") / "input" / name
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def get_default_interim_dir(name: str) -> Path:
    """Get the default directory for intermediate files."""
    root_dir = Path("data") / "interim" / name
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def get_default_output_dir(name: str) -> Path:
    """Get the default directory for final output files."""
    root_dir = Path("data") / "output" / name
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


@flow(name="Prepare VCF Source")
def prepare_vcf_source_flow(
    url: str,
    pattern: Optional[str] = None,
    name: str = "downloads",
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = True,
    profile: bool = True,
) -> PreparationResult:
    """Generic flow to download, convert, and optionally split VCF data."""
    logger = get_run_logger()
    
    # 1. Resolve directories based on data pattern
    input_path = Path(dest_dir) if dest_dir else get_default_input_dir(name)
    interim_path = get_default_interim_dir(name)
    output_path = get_default_output_dir(name)
    
    with prefect_flow_run(f"Prepare {name}", profile=profile):
        # 2. List paths
        urls = list_paths_task(url=url, pattern=pattern)
        
        # 3. Download files in parallel to data/input
        vcf_local_futures = [
            download_path_task.submit(url=u, name=name, dest_dir=input_path)
            for u in urls
        ]
        vcf_locals = [f.result() for f in vcf_local_futures]
            
        # 4. Convert to parquet in parallel to data/interim
        conversion_futures = [
            convert_to_parquet_task.submit(
                vcf_path=vcf_p, 
                parquet_path=interim_path / vcf_p.with_suffix(".parquet").name
            )
            for vcf_p in vcf_locals
        ]
        vcf_parquet_paths = [f.result()[1] for f in conversion_futures]
            
        # 5. Validate
        validate_task(urls=urls, vcf_local=vcf_locals, vcf_parquet_path=vcf_parquet_paths)
        
        split_dict = None
        # 6. Optional splitting to data/output
        if with_splitting:
            split_result = split_parquets_task(
                parquet_paths=vcf_parquet_paths,
                explode_snv_alt=explode_snv_alt,
                write_to=output_path / "splitted_variants"
            )
            split_dict = split_result.split_variants_dict
            
        return PreparationResult(
            urls=urls,
            vcf_local=vcf_locals,
            vcf_parquet_path=vcf_parquet_paths,
            split_variants_dict=split_dict
        )


@flow(name="Prepare dbSNP")
def prepare_dbsnp_flow(
    build: str = "GRCh38",
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for dbSNP preparation."""
    if build == "GRCh38":
        base_url = "https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/"
        pattern = r"GCF_000001405\.40\.gz$"
    elif build == "GRCh37":
        base_url = "https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/"
        pattern = r"GCF_000001405\.25\.gz$"
    else:
        raise ValueError(f"Unsupported build: {build}")
        
    return prepare_vcf_source_flow(
        url=base_url,
        pattern=pattern,
        name=f"dbsnp_{build.lower()}",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        profile=profile,
    )


@flow(name="Prepare gnomAD")
def prepare_gnomad_flow(
    version: str = "v4",
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for gnomAD preparation."""
    if version == "v4":
        base_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.0/vcf/"
        pattern = r"gnomad\.v4\.0\..+\.vcf\.bgz$"
    elif version == "v3":
        base_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1.2/vcf/"
        pattern = r"gnomad\.v3\.1\.2\..+\.vcf\.bgz$"
    else:
        raise ValueError(f"Unsupported version: {version}")
        
    return prepare_vcf_source_flow(
        url=base_url,
        pattern=pattern,
        name=f"gnomad_{version}",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        profile=profile,
    )

@flow(name="Prepare ClinVar")
def prepare_clinvar_flow(
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for ClinVar preparation."""
    return prepare_vcf_source_flow(
        url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/",
        pattern=r"clinvar\.vcf\.gz$",
        name="clinvar",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        profile=profile,
    )

@flow(name="Prepare Ensembl")
def prepare_ensembl_flow(
    dest_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    pattern: Optional[str] = None,
    profile: bool = True,
) -> PreparationResult:
    """Prefect flow for Ensembl preparation."""
    return prepare_vcf_source_flow(
        url="https://ftp.ensembl.org/pub/current_variation/vcf/homo_sapiens/",
        pattern=pattern or r"homo_sapiens-chr([^.]+)\.vcf\.gz$",
        name="ensembl_variations",
        dest_dir=dest_dir,
        with_splitting=with_splitting,
        profile=profile,
    )


class PreparationPipelines:
    """Pipelines for preparing genomic data from various sources using Prefect.
    
    This class provides static methods for:
    - Downloading, converting, and splitting VCF data
    - Uploading processed data to Hugging Face Hub
    """
    
    @staticmethod
    def download_clinvar(
        dest_dir: Optional[Path] = None,
        with_splitting: bool = False,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download ClinVar VCF files and convert to parquet using Prefect."""
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "download_clinvar.json", log_dir / "download_clinvar.log")
        
        return prepare_clinvar_flow(
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            profile=profile
        )
    
    @staticmethod
    def download_ensembl(
        dest_dir: Optional[Path] = None,
        with_splitting: bool = False,
        log: bool = True,
        pattern: Optional[str] = None,
        url: Optional[str] = None,
        profile: bool = True,
    ) -> PreparationResult:
        """Download Ensembl VCF files and convert to parquet using Prefect."""
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "download_ensembl.json", log_dir / "download_ensembl.log")
        
        # If custom URL provided, we use the generic source flow
        if url:
            return prepare_vcf_source_flow(
                url=url,
                pattern=pattern,
                name="ensembl_custom",
                dest_dir=dest_dir,
                with_splitting=with_splitting,
                profile=profile
            )
            
        return prepare_ensembl_flow(
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            pattern=pattern,
            profile=profile
        )
    
    @staticmethod
    def download_dbsnp(
        dest_dir: Optional[Path] = None,
        build: str = "GRCh38",
        with_splitting: bool = False,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download dbSNP VCF files and convert to parquet using Prefect."""
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "download_dbsnp.json", log_dir / "download_dbsnp.log")
        
        return prepare_dbsnp_flow(
            build=build,
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            profile=profile
        )
    
    @staticmethod
    def download_gnomad(
        dest_dir: Optional[Path] = None,
        version: str = "v4",
        with_splitting: bool = False,
        log: bool = True,
        profile: bool = True,
    ) -> PreparationResult:
        """Download gnomAD VCF files and convert to parquet using Prefect."""
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "download_gnomad.json", log_dir / "download_gnomad.log")
        
        return prepare_gnomad_flow(
            version=version,
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            profile=profile
        )
    
    @staticmethod
    def split_existing_parquets(
        parquet_files: list[Path] | Path,
        explode_snv_alt: bool = True,
        write_to: Optional[Path] = None,
        log: bool = True,
        profile: bool = True,
    ) -> SplitResult:
        """Quick function to split existing parquet files by variant type using Prefect."""
        if isinstance(parquet_files, Path):
            parquet_files = [parquet_files]
            
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "split_parquets.json", log_dir / "split_parquets.log")
            
        with prefect_flow_run("Split Existing Parquets", profile=profile):
            result = split_parquets_task(
                parquet_paths=parquet_files,
                explode_snv_alt=explode_snv_alt,
                write_to=write_to
            )
            return result
    
    @staticmethod
    def upload_clinvar_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/clinvar",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload ClinVar parquet files to Hugging Face Hub."""
        if source_dir is None:
            source_dir = Path("data/output/clinvar")
            if (source_dir / "splitted_variants").exists():
                source_dir = source_dir / "splitted_variants"
        else:
            source_dir = Path(source_dir)
        
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "upload_clinvar.json", log_dir / "upload_clinvar.log")
        
        with start_action(action_type="upload_clinvar_to_hf", source_dir=str(source_dir), repo_id=repo_id, pattern=pattern):
            parquet_files = collect_parquet_files(source_dir, pattern=pattern)
            if not parquet_files:
                return BatchUploadResult(uploaded_files=[], num_uploaded=0, num_skipped=0)
            
            variant_types = {f.relative_to(source_dir).parts[0] for f in parquet_files if len(f.relative_to(source_dir).parts) > 1}
            total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            dataset_card = generate_clinvar_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
            
            return upload_parquet_to_hf(
                parquet_files=parquet_files,
                repo_id=repo_id,
                token=token,
                path_prefix=path_prefix,
                source_dir=source_dir,
                dataset_card_content=dataset_card,
            )

    @staticmethod
    def upload_ensembl_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/ensembl_variations",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload Ensembl variation parquet files to Hugging Face Hub."""
        if source_dir is None:
            source_dir = Path("data/output/ensembl_variations")
            if (source_dir / "splitted_variants").exists():
                source_dir = source_dir / "splitted_variants"
        else:
            source_dir = Path(source_dir)
        
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "upload_ensembl.json", log_dir / "upload_ensembl.log")
        
        with start_action(action_type="upload_ensembl_to_hf", source_dir=str(source_dir), repo_id=repo_id, pattern=pattern):
            parquet_files = collect_parquet_files(source_dir, pattern=pattern)
            if not parquet_files:
                return BatchUploadResult(uploaded_files=[], num_uploaded=0, num_skipped=0)
            
            variant_types = {f.relative_to(source_dir).parts[0] for f in parquet_files if len(f.relative_to(source_dir).parts) > 1}
            total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            dataset_card = generate_ensembl_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
            
            return upload_parquet_to_hf(
                parquet_files=parquet_files,
                repo_id=repo_id,
                token=token,
                path_prefix=path_prefix,
                source_dir=source_dir,
                dataset_card_content=dataset_card,
            )

    @staticmethod
    def upload_dbsnp_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/dbsnp",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload dbSNP parquet files to Hugging Face Hub."""
        if source_dir is None:
            source_dir = Path("data/output/dbsnp_grch38")
            if not source_dir.exists():
                 source_dir = Path("data/output/dbsnp_grch37")
            if (source_dir / "splitted_variants").exists():
                source_dir = source_dir / "splitted_variants"
        else:
            source_dir = Path(source_dir)
        
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "upload_dbsnp.json", log_dir / "upload_dbsnp.log")
        
        with start_action(action_type="upload_dbsnp_to_hf", source_dir=str(source_dir), repo_id=repo_id, pattern=pattern):
            parquet_files = collect_parquet_files(source_dir, pattern=pattern)
            if not parquet_files:
                return BatchUploadResult(uploaded_files=[], num_uploaded=0, num_skipped=0)
            
            variant_types = {f.relative_to(source_dir).parts[0] for f in parquet_files if len(f.relative_to(source_dir).parts) > 1}
            total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            dataset_card = generate_dbsnp_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
            
            return upload_parquet_to_hf(
                parquet_files=parquet_files,
                repo_id=repo_id,
                token=token,
                path_prefix=path_prefix,
                source_dir=source_dir,
                dataset_card_content=dataset_card,
            )

    @staticmethod
    def upload_gnomad_to_hf(
        source_dir: Optional[Path] = None,
        repo_id: str = "just-dna-seq/gnomad",
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        """Upload gnomAD parquet files to Hugging Face Hub."""
        if source_dir is None:
            source_dir = Path("data/output/gnomad_v4")
            if not source_dir.exists():
                 source_dir = Path("data/output/gnomad_v3")
            if (source_dir / "splitted_variants").exists():
                source_dir = source_dir / "splitted_variants"
        else:
            source_dir = Path(source_dir)
        
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / "upload_gnomad.json", log_dir / "upload_gnomad.log")
        
        with start_action(action_type="upload_gnomad_to_hf", source_dir=str(source_dir), repo_id=repo_id, pattern=pattern):
            parquet_files = collect_parquet_files(source_dir, pattern=pattern)
            if not parquet_files:
                return BatchUploadResult(uploaded_files=[], num_uploaded=0, num_skipped=0)
            
            variant_types = {f.relative_to(source_dir).parts[0] for f in parquet_files if len(f.relative_to(source_dir).parts) > 1}
            total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            dataset_card = generate_gnomad_card(len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
            
            return upload_parquet_to_hf(
                parquet_files=parquet_files,
                repo_id=repo_id,
                token=token,
                path_prefix=path_prefix,
                source_dir=source_dir,
                dataset_card_content=dataset_card,
            )

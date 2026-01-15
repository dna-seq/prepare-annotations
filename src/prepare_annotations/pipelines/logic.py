"""Core preparation logic for genomic data sources (Prefect-free)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import duckdb
import polars as pl
from eliot import start_action
from platformdirs import user_cache_dir
from pycomfort.logging import to_nice_stdout, to_nice_file

from prepare_annotations.io import is_parquet, _default_parquet_path
from prepare_annotations.resources import (
    get_cache_dir,
    get_default_cache_dir,
    get_default_input_dir,
    get_default_interim_dir,
    get_default_output_dir,
)
from prepare_annotations.models import PreparationResult, SplitResult, BatchUploadResult, RSIDCoordinateResult
from prepare_annotations.vcf_downloader import (
    list_paths,
    download_path,
    convert_to_parquet,
    validate_downloads_and_parquet,
    download_checksums,
    ChecksumInfo,
)
from prepare_annotations.huggingface_uploader import (
    collect_parquet_files,
    upload_parquet_to_hf,
)
from prepare_annotations.dataset_card_generator import (
    generate_clinvar_card,
    generate_ensembl_card,
    generate_dbsnp_card,
    generate_dbsnp_t2t_card,
    generate_gnomad_card,
)


def split_parquets(
    parquet_paths: List[Path],
    explode_snv_alt: bool = False,
    write_to: Optional[Path] = None,
) -> SplitResult:
    """Split parquet files by variant type (TSA)."""
    from prepare_annotations.vcf_parquet_splitter import split_variants_by_tsa
    
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


def compute_rsid_coordinates(
    input_dir: Path,
    output_path: Path,
    memory_fraction: float = 0.8,
    output_dataset: bool = False,
    force: bool = False,
    compression_level: int = 14,
) -> RSIDCoordinateResult:
    """Compute rsID coordinates from split parquet files using DuckDB streaming."""
    import psutil
    
    if output_dataset:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
        if tmp_output_path.exists():
            tmp_output_path.unlink()
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}. Have you run the splitting step?")
        
    variant_type_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not variant_type_dirs:
        raise ValueError(f"No variant type subdirectories found in {input_dir}.")
    
    mem = psutil.virtual_memory()
    memory_limit_bytes = int(mem.available * min(memory_fraction, 0.6))
    memory_limit_gb = memory_limit_bytes / (1024 ** 3)
    memory_limit_str = f"{max(2, int(memory_limit_gb))}GB"
    
    def chrom_sort_key(filename: str) -> tuple[int, str]:
        import re
        match = re.search(r'chr([0-9]+|[XYMTm]+)', filename)
        if not match:
            return (1000, filename)
        chrom = match.group(1).upper()
        if chrom.isdigit():
            return (int(chrom), chrom)
        mapping = {"X": 100, "Y": 101, "MT": 102, "M": 102}
        return (mapping.get(chrom, 200), chrom)

    import tempfile
    chunk_dir: Path
    duckdb_tmp_ctx = tempfile.TemporaryDirectory(dir="/tmp", prefix="ensembl_rsids_duckdb_")

    if output_dataset:
        chunk_dir = output_path
    else:
        chunk_dir = input_dir.parent / "rsid_coordinates"
        chunk_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        duckdb_temp_dir = Path(duckdb_tmp_ctx.__enter__())
        
        with start_action(action_type="duckdb_compute_rsids", input_dir=str(input_dir), 
                          variant_types=[d.name for d in variant_type_dirs],
                          memory_limit=memory_limit_str,
                          available_memory_gb=round(mem.available / (1024**3), 1)) as action:
            
            con = duckdb.connect()
            con.execute(f"SET memory_limit = '{memory_limit_str}'")
            con.execute(f"SET temp_directory = '{duckdb_temp_dir}'")
            con.execute("SET preserve_insertion_order = false")
            
            from collections import defaultdict
            chrom_groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
            for variant_dir in variant_type_dirs:
                for pq_file in variant_dir.glob("*.parquet"):
                    if not is_parquet(pq_file):
                        continue
                    key = chrom_sort_key(pq_file.name)
                    chrom_groups[key].append(pq_file)

            sorted_keys = sorted(chrom_groups.keys())
            chunk_files = []
            
            for key in sorted_keys:
                order, chrom_name = key
                files = chrom_groups[key]
                
                first_name = files[0].name
                stem = first_name
                for suffix in [".vcf.parquet", ".parquet"]:
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        break
                
                chunk_out = chunk_dir / f"{stem}_rsid_coordinates.parquet"
                
                if not force and chunk_out.exists() and chunk_out.stat().st_size > 0:
                    try:
                        pl.scan_parquet(chunk_out).select(pl.len()).collect()
                        action.log(message_type="skipping_chromosome", chromosome=chrom_name, reason="already_exists")
                        chunk_files.append(chunk_out)
                        continue
                    except Exception:
                        action.log(message_type="recomputing_chromosome", chromosome=chrom_name, reason="corrupted_file")
                
                action.log(message_type="processing_chromosome", chromosome=chrom_name, files=len(files))
                
                first_file = files[0]
                cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{first_file}')").fetchall()
                col_names = [c[0] for c in cols_info]
                tsa_field = "TSA" if "TSA" in col_names and "tsa" not in col_names else "tsa"
                
                subqueries = [
                    f"SELECT chrom, start, \"end\", id, {tsa_field} as tsa FROM read_parquet('{f}')"
                    for f in files
                ]
                
                con.execute(f"""
                    COPY (
                        SELECT DISTINCT chrom, start, "end", id, tsa 
                        FROM ( {' UNION ALL '.join(subqueries)} )
                        ORDER BY start
                     ) TO '{chunk_out}' (
                         FORMAT 'PARQUET', 
                         COMPRESSION 'ZSTD', 
                         COMPRESSION_LEVEL {compression_level}, 
                         ROW_GROUP_SIZE 5000000
                     )
                """)
                chunk_count = con.execute(f"SELECT count(*) FROM read_parquet('{chunk_out}')").fetchone()[0]
                action.log(message_type="chromosome_completed", chromosome=chrom_name, count=chunk_count, size_mb=round(chunk_out.stat().st_size / (1024**2), 2))
                
                chunk_files.append(chunk_out)

            if not output_dataset:
                action.log(message_type="info", step="merging_chunks_pyarrow", total_chunks=len(chunk_files))
                from prepare_annotations.io import merge_parquet_files
                merge_parquet_files(
                    input_files=chunk_files,
                    output_path=output_path,
                    compression="zstd",
                    compression_level=compression_level,
                )
            con.close()
            action.log(message_type="completed", output_path=str(output_path))
    finally:
        if duckdb_tmp_ctx is not None:
            duckdb_tmp_ctx.__exit__(None, None, None)
        if not output_dataset:
            try:
                if 'tmp_output_path' in locals() and tmp_output_path.exists():
                    tmp_output_path.unlink()
            except OSError:
                pass
    
    scan_target = str(output_path / "*.parquet") if output_dataset else str(output_path)
    count = pl.scan_parquet(scan_target).select(pl.len()).collect().item()
    return RSIDCoordinateResult(output_path=output_path, count=count)


def prepare_vcf_source(
    url: str,
    pattern: Optional[str] = None,
    name: str = "downloads",
    dest_dir: Optional[str | Path] = None,
    vcf_dir: Optional[str | Path] = None,
    parquet_dir: Optional[str | Path] = None,
    with_splitting: bool = False,
    explode_snv_alt: bool = False,
    alts_list: bool = True,
    compression: str = "zstd",
    compression_level: Optional[int] = None,
    download_progress_interval_seconds: Optional[float] = None,
    s3_max_pool: Optional[int] = None,
    s3_block_size: Optional[int] = None,
    http_max_pool: Optional[int] = None,
    http_chunk_size: Optional[int] = None,
    connect_timeout: Optional[float] = None,
    sock_read_timeout: Optional[float] = None,
    retries: Optional[int] = None,
    verify_checksums: bool = True,
) -> PreparationResult:
    """Generic logic to download, convert, and optionally split VCF data."""
    cache_path = Path(dest_dir) if dest_dir else get_default_cache_dir(name)
    vcf_path = Path(vcf_dir) if vcf_dir else cache_path / "vcf"
    parquet_path = Path(parquet_dir) if parquet_dir else cache_path
    
    vcf_path.mkdir(parents=True, exist_ok=True)
    parquet_path.mkdir(parents=True, exist_ok=True)
    
    for tmp_file in parquet_path.glob("*.tmp.parquet"):
        try:
            tmp_file.unlink()
        except Exception:
            pass

    urls = list_paths(url=url, pattern=pattern)
    
    checksums: dict[str, ChecksumInfo] = {}
    if verify_checksums:
        try:
            checksums = download_checksums(url)
        except Exception:
            pass
    
    vcf_locals = []
    for u in urls:
        filename = u.rsplit("/", 1)[-1]
        expected_checksum = checksums.get(filename) if verify_checksums else None
        
        vcf_locals.append(
            download_path(
                url=u,
                name=name,
                dest_dir=vcf_path,
                progress_interval_seconds=download_progress_interval_seconds,
                s3_max_pool=s3_max_pool,
                s3_block_size=s3_block_size,
                http_max_pool=http_max_pool,
                http_chunk_size=http_chunk_size,
                connect_timeout=connect_timeout,
                sock_read_timeout=sock_read_timeout,
                retries=retries,
                expected_checksum=expected_checksum,
            )
        )
            
    vcf_parquet_paths = []
    for vcf_p in vcf_locals:
        p_path = parquet_path / _default_parquet_path(vcf_p).name
        _, result_path = convert_to_parquet(
            vcf_path=vcf_p, 
            parquet_path=p_path,
            compression=compression,
            compression_level=compression_level,
            alts_list=alts_list,
        )
        vcf_parquet_paths.append(result_path)
            
    validate_downloads_and_parquet(urls=urls, vcf_local=vcf_locals, vcf_parquet_path=vcf_parquet_paths)
    
    split_dict = None
    parquet_only_paths = [p for p in vcf_parquet_paths if is_parquet(p)]

    if with_splitting and parquet_only_paths:
        split_dir = cache_path / "splitted_variants"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for tmp_file in split_dir.rglob("*.tmp.parquet"):
            try:
                tmp_file.unlink()
            except Exception:
                pass

        split_result = split_parquets(
            parquet_paths=parquet_only_paths,
            explode_snv_alt=explode_snv_alt,
            write_to=split_dir
        )
        split_dict = split_result.split_variants_dict
            
    fsspec_cache = cache_path / ".fsspec_cache"
    if fsspec_cache.exists():
        shutil.rmtree(fsspec_cache, ignore_errors=True)

    return PreparationResult(
        urls=urls,
        vcf_local=vcf_locals,
        vcf_parquet_path=vcf_parquet_paths,
        split_variants_dict=split_dict
    )


class PreparationPipelines:
    """Pipelines for preparing genomic data from various sources (Prefect-free)."""

    @staticmethod
    def _setup_logging(name: str, log: bool = True) -> None:
        if log:
            to_nice_stdout()
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            to_nice_file(log_dir / f"{name}.json", log_dir / f"{name}.log")

    @staticmethod
    def download_clinvar(
        assembly: str = "GRCh38_ensembl",
        url: Optional[str] = None,
        pattern: Optional[str] = None,
        dest_dir: Optional[Path] = None,
        vcf_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
    ) -> PreparationResult:
        PreparationPipelines._setup_logging("download_clinvar", log)
        
        effective_url = url
        effective_pattern = pattern
        effective_dest = dest_dir
        
        if "ensembl" in assembly.lower() and not effective_url:
            ensembl_species_url = f"https://ftp.ensembl.org/pub/current_variation/vcf/homo_sapiens/"
            try:
                vep_url = f"{ensembl_species_url}vep/"
                vep_folders = list_paths(vep_url, file_only=False)
                clinvar_folders = [f for f in vep_folders if "clinvar" in f.lower()]
                if clinvar_folders:
                    latest_folder = sorted(clinvar_folders)[-1]
                    if not latest_folder.endswith("/"):
                        latest_folder += "/"
                    effective_url = latest_folder
                    effective_pattern = r"clinvar.*\.vcf\.gz$"
                    if not effective_dest:
                        effective_dest = get_default_cache_dir("clinvar")
            except Exception:
                pass

        if not effective_url:
            effective_url = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_{assembly}/"
            effective_pattern = effective_pattern or r"clinvar\.vcf\.gz$"

        return prepare_vcf_source(
            url=effective_url,
            pattern=effective_pattern,
            name="clinvar",
            dest_dir=effective_dest,
            vcf_dir=vcf_dir,
            parquet_dir=parquet_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
        )

    @staticmethod
    def download_ensembl(
        species: str = "homo_sapiens",
        dest_dir: Optional[Path] = None,
        vcf_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
        pattern: Optional[str] = None,
        url: Optional[str] = None,
        http_max_pool: Optional[int] = None,
        http_chunk_size: Optional[int] = None,
        connect_timeout: Optional[float] = None,
        sock_read_timeout: Optional[float] = None,
        retries: Optional[int] = None,
        verify_checksums: bool = True,
    ) -> PreparationResult:
        PreparationPipelines._setup_logging("download_ensembl", log)
        
        if url:
            base_dir = Path(dest_dir) if dest_dir else get_default_cache_dir("ensembl_custom")
            species_dir = base_dir / species
            return prepare_vcf_source(
                url=url,
                pattern=pattern,
                name="ensembl_custom",
                dest_dir=species_dir,
                vcf_dir=vcf_dir,
                parquet_dir=parquet_dir,
                with_splitting=with_splitting,
                explode_snv_alt=explode_snv_alt,
                alts_list=alts_list,
                http_max_pool=http_max_pool,
                http_chunk_size=http_chunk_size,
                connect_timeout=connect_timeout,
                sock_read_timeout=sock_read_timeout,
                retries=retries,
                verify_checksums=verify_checksums,
            )
            
        base_dir = Path(dest_dir) if dest_dir else get_default_cache_dir("ensembl")
        species_dir = base_dir / species
        url = f"https://ftp.ensembl.org/pub/current_variation/vcf/{species}/"
        default_pattern = rf"{species}-chr([^.]+)\.vcf\.gz$"
        
        return prepare_vcf_source(
            url=url,
            pattern=pattern or default_pattern,
            name="ensembl",
            dest_dir=species_dir,
            vcf_dir=vcf_dir,
            parquet_dir=parquet_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            http_max_pool=http_max_pool,
            http_chunk_size=http_chunk_size,
            connect_timeout=connect_timeout,
            sock_read_timeout=sock_read_timeout,
            retries=retries,
            verify_checksums=verify_checksums,
        )
    
    @staticmethod
    def download_dbsnp(
        dest_dir: Optional[Path] = None,
        build: str = "GRCh38",
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
    ) -> PreparationResult:
        PreparationPipelines._setup_logging(f"download_dbsnp_{build.lower()}", log)
        if build == "GRCh38":
            base_url = "https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/"
            pattern = r"GCF_000001405\.40\.gz$"
        elif build == "GRCh37":
            base_url = "https://ftp.ncbi.nlm.nih.gov/snp/latest_release/VCF/"
            pattern = r"GCF_000001405\.25\.gz$"
        else:
            raise ValueError(f"Unsupported build: {build}")
            
        return prepare_vcf_source(
            url=base_url,
            pattern=pattern,
            name=f"dbsnp_{build.lower()}",
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
        )
    
    @staticmethod
    def download_dbsnp_t2t(
        dest_dir: Optional[Path] = None,
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        compression: str = "zstd",
        compression_level: int = 14,
        progress_interval_seconds: float = 60.0,
        s3_max_pool: Optional[int] = None,
        s3_block_size: Optional[int] = None,
        log: bool = True,
    ) -> PreparationResult:
        PreparationPipelines._setup_logging("download_dbsnp_t2t", log)
        s3_url = "s3://human-pangenomics/T2T/CHM13/assemblies/annotation/liftover/"
        pattern = r"chm13v2.0_dbSNPv155.vcf.gz(.tbi)?$"
        
        return prepare_vcf_source(
            url=s3_url,
            pattern=pattern,
            name="dbsnp_t2t",
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
            compression=compression,
            compression_level=compression_level,
            download_progress_interval_seconds=progress_interval_seconds,
            s3_max_pool=s3_max_pool,
            s3_block_size=s3_block_size,
        )

    @staticmethod
    def download_gnomad(
        dest_dir: Optional[Path] = None,
        version: str = "v4",
        with_splitting: bool = False,
        explode_snv_alt: bool = False,
        alts_list: bool = True,
        log: bool = True,
    ) -> PreparationResult:
        PreparationPipelines._setup_logging(f"download_gnomad_{version}", log)
        if version == "v4":
            base_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.0/vcf/"
            pattern = r"gnomad\.v4\.0\..+\.vcf\.bgz$"
        elif version == "v3":
            base_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/3.1.2/vcf/"
            pattern = r"gnomad\.v3\.1\.2\..+\.vcf\.bgz$"
        else:
            raise ValueError(f"Unsupported version: {version}")
            
        return prepare_vcf_source(
            url=base_url,
            pattern=pattern,
            name=f"gnomad_{version}",
            dest_dir=dest_dir,
            with_splitting=with_splitting,
            explode_snv_alt=explode_snv_alt,
            alts_list=alts_list,
        )
    
    @staticmethod
    def split_existing_parquets(
        parquet_files: list[Path] | Path,
        explode_snv_alt: bool = False,
        write_to: Optional[Path] = None,
        log: bool = True,
    ) -> SplitResult:
        if isinstance(parquet_files, Path):
            parquet_files = [parquet_files]
        PreparationPipelines._setup_logging("split_parquets", log)
        return split_parquets(
            parquet_paths=parquet_files,
            explode_snv_alt=explode_snv_alt,
            write_to=write_to
        )
    
    @staticmethod
    def upload_dataset_to_hf(
        dataset_type: str,
        source_dir: Optional[Path] = None,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        pattern: str = "**/*.parquet",
        path_prefix: str = "data",
        log: bool = True,
    ) -> BatchUploadResult:
        config = {
            "clinvar": {
                "repo": "just-dna-seq/clinvar",
                "card": generate_clinvar_card,
                "default_source": get_default_output_dir("clinvar")
            },
            "ensembl": {
                "repo": "just-dna-seq/ensembl_variations",
                "card": generate_ensembl_card,
                "default_source": get_default_output_dir("ensembl")
            },
            "dbsnp": {
                "repo": "just-dna-seq/dbsnp",
                "card": generate_dbsnp_card,
                "default_source": get_default_output_dir("dbsnp_grch38")
            },
            "dbsnp_t2t": {
                "repo": "just-dna-seq/dbsnp_t2t",
                "card": generate_dbsnp_t2t_card,
                "default_source": get_default_output_dir("dbsnp_t2t")
            },
            "gnomad": {
                "repo": "just-dna-seq/gnomad",
                "card": generate_gnomad_card,
                "default_source": get_default_output_dir("gnomad_v4")
            }
        }
        
        ds_cfg = config.get(dataset_type.lower())
        if not ds_cfg:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        repo_id = repo_id or ds_cfg["repo"]
        if source_dir is None:
            source_dir = ds_cfg["default_source"]
            if not source_dir.exists() and dataset_type.lower() == "dbsnp":
                source_dir = get_default_output_dir("dbsnp_grch37")
            elif not source_dir.exists() and dataset_type.lower() == "gnomad":
                source_dir = get_default_output_dir("gnomad_v3")
                
            if (source_dir / "splitted_variants").exists():
                source_dir = source_dir / "splitted_variants"
        
        PreparationPipelines._setup_logging(f"upload_{dataset_type.lower()}", log)
        
        with start_action(action_type=f"upload_{dataset_type}_to_hf", source_dir=str(source_dir), repo_id=repo_id, pattern=pattern):
            parquet_files = collect_parquet_files(source_dir, pattern=pattern)
            variant_types = set()
            for f in parquet_files:
                try:
                    rel_p = f.relative_to(source_dir)
                    if len(rel_p.parts) > 1:
                        variant_types.add(rel_p.parts[0])
                except ValueError:
                    continue
            
            total_size_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            dataset_card = ds_cfg["card"](len(parquet_files), total_size_gb, list(variant_types) if variant_types else None)
            
            return upload_parquet_to_hf(
                parquet_files=parquet_files,
                repo_id=repo_id,
                token=token,
                path_prefix=path_prefix,
                source_dir=source_dir,
                dataset_card_content=dataset_card,
            )

    @staticmethod
    def compute_ensembl_rsid_coords(
        input_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        memory_fraction: float = 0.8,
        auto_split: bool = True,
        explode_snv_alt: bool = False,
        output_dataset: bool = False,
        force: bool = False,
        compression_level: int = 14,
        log: bool = True,
    ) -> RSIDCoordinateResult:
        PreparationPipelines._setup_logging("ensembl_rsid_coords", log)
        
        if input_dir is None:
            cache_dir = get_default_cache_dir("ensembl")
            input_dir = cache_dir / "splitted_variants"
        
        if output_path is None:
            cache_dir = get_default_cache_dir("ensembl")
            output_path = cache_dir / ("rsid_coordinates" if output_dataset else "rsid_coordinates.parquet")
        
        if not input_dir.exists() or not any(input_dir.iterdir()):
            if auto_split:
                cache_dir = input_dir.parent
                all_parquets = [p for p in cache_dir.glob("*.parquet") if is_parquet(p)]
                by_base: Dict[str, Path] = {}
                for p in all_parquets:
                    name = p.name
                    if name == "rsid_coordinates.parquet":
                        continue
                    base = name
                    for suffix in [".vcf.parquet", ".parquet"]:
                        if base.endswith(suffix):
                            base = base[:-len(suffix)]
                            break
                    if base not in by_base or (".vcf." in by_base[base].name and ".vcf." not in name):
                        by_base[base] = p
                
                parquet_files = list(by_base.values())
                if not parquet_files:
                    raise FileNotFoundError(f"No parquet files found in {cache_dir} to split.")
                
                split_parquets(
                    parquet_paths=parquet_files,
                    explode_snv_alt=explode_snv_alt,
                    write_to=input_dir
                )
            else:
                raise FileNotFoundError(f"Split files not found in {input_dir}.")
            
        return compute_rsid_coordinates(
            input_dir=input_dir, 
            output_path=output_path,
            memory_fraction=memory_fraction,
            output_dataset=output_dataset,
            force=force,
            compression_level=compression_level,
        )

from pathlib import Path
from typing import Union, Optional, Literal, Tuple, TypeVar
import polars as pl
import polars_bio as pb
from eliot import start_action
import os
import pooch
import tempfile

# Type variable for generic data types
T = TypeVar('T')

# Generic type for results that include both data and the file path where it's saved
AnnotatedResult = Tuple[T, Path]

# Specific type alias for LazyFrame results with their parquet paths
AnnotatedLazyFrame = AnnotatedResult[pl.LazyFrame]

# Type alias describing how parquet saving is configured:
# - None: do not save
# - "auto": save next to the input, replacing .vcf/.vcf.gz with .parquet
# - Path: save to the provided absolute/relative path
SaveParquet = Union[Path, Literal["auto"], None]

SaveVortex = Union[Path, Literal["auto"], None]

def _default_parquet_path(vcf_path: Path) -> Path:
    """Generate default parquet path next to VCF file."""
    # Remove .vcf or .vcf.gz extension before adding .parquet
    if vcf_path.suffixes == ['.vcf', '.gz']:
        # Handle .vcf.gz files
        return vcf_path.with_suffix('').with_suffix('.parquet')
    elif vcf_path.suffix == '.vcf':
        # Handle .vcf files
        return vcf_path.with_suffix('.parquet')
    else:
        # Fallback for other file types
        return vcf_path.with_suffix('.parquet')


def _default_vortex_path(vcf_path: Path) -> Path:
    """Generate default vortex path next to VCF file."""
    # Remove .vcf or .vcf.gz extension before adding .vortex
    if vcf_path.suffixes == ['.vcf', '.gz']:
        # Handle .vcf.gz files
        return vcf_path.with_suffix('').with_suffix('.vortex')
    elif vcf_path.suffix == '.vcf':
        # Handle .vcf files
        return vcf_path.with_suffix('.vortex')
    else:
        # Fallback for other file types
        return vcf_path.with_suffix('.vortex')


def resolve_prepare_annotations_subfolder(subdir_name: str, base: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a subfolder path for prepare_annotations data storage.
    
    This function provides a consistent way to resolve data storage paths across
    the prepare_annotations package, respecting the PREPARE_ANNOTATIONS_FOLDER environment variable.
    
    Args:
        subdir_name: Name of the subdirectory to create/use
        base: Base directory path. If None, uses PREPARE_ANNOTATIONS_FOLDER env var or defaults to "prepare_annotations"
        
    Returns:
        Absolute path to the resolved subfolder
        
    Examples:
        >>> # Using default base (PREPARE_ANNOTATIONS_FOLDER env var or "prepare_annotations")
        >>> resolve_prepare_annotations_subfolder("downloads")
        PosixPath('/home/user/.cache/prepare_annotations/downloads')
        
        >>> # Using custom base
        >>> resolve_prepare_annotations_subfolder("vcf_files", "/tmp/myproject")
        PosixPath('/tmp/myproject/vcf_files')
    """
    if base is None:
        base = os.getenv("PREPARE_ANNOTATIONS_FOLDER", "prepare_annotations")
    
    base_path = Path(base)
    
    # If it's just a name (no path separators), use OS cache
    if len(base_path.parts) == 1 and not base_path.is_absolute():
        cache_path = Path(pooch.os_cache(str(base_path))) / subdir_name
    else:
        cache_path = base_path / subdir_name
        
    return cache_path.resolve()


def get_info_fields(vcf_path: str) -> list[str]:
    """
    Extract INFO field names from a VCF file header by parsing the header directly.
    
    Args:
        vcf_path: Path to the VCF file
        
    Returns:
        List of INFO field names found in the header
    """
    with start_action(action_type="get_info_fields", vcf_path=vcf_path) as action:
        try:
            import gzip
            info_fields = []
            
            # Determine if file is gzipped
            open_func = gzip.open if vcf_path.endswith('.gz') else open
            mode = 'rt' if vcf_path.endswith('.gz') else 'r'
            
            with open_func(vcf_path, mode) as f:
                for line in f:
                    line = line.strip()
                    # Stop when we reach the data (non-header lines)
                    if not line.startswith('#'):
                        break
                    # Look for INFO field definitions
                    if line.startswith('##INFO=<ID='):
                        # Extract the ID from ##INFO=<ID=FIELD_NAME,...>
                        start_idx = line.find('ID=') + 3
                        end_idx = line.find(',', start_idx)
                        if end_idx == -1:  # In case there's no comma after ID
                            end_idx = line.find('>', start_idx)
                        if end_idx > start_idx:
                            field_name = line[start_idx:end_idx]
                            info_fields.append(field_name)
            
            action.log(
                message_type="info",
                step="info_fields_extracted",
                count=len(info_fields),
                fields=info_fields
            )
            return info_fields
            
        except Exception as e:
            action.log(
                message_type="error",
                step="info_fields_extraction_failed",
                error=str(e)
            )
            # Return empty list if extraction fails
            return []


def is_parquet(path: Union[str, Path]) -> bool:
    """Check if a path is a parquet file, excluding temporary files."""
    p = Path(path)
    return p.suffix == ".parquet" and not p.name.endswith(".tmp.parquet")


def read_vcf_file(
    file_path: Union[str, Path],
    info_fields: Union[list[str], None] = None,
    thread_num: Optional[int] = None,
    save_parquet: SaveParquet = "auto",
    save_vortex: SaveVortex = None,
    engine: str = "auto",
    compression: str = "zstd",
    compression_level: Optional[int] = 14,
    alts_list: bool = True
) -> pl.LazyFrame:
    """
    Read a VCF file using polars-bio, automatically handling gzipped files.

    Args:
        file_path: Path to the VCF file (can be .vcf or .vcf.gz)
        info_fields: The fields to read from the INFO column.
        thread_num: The number of threads to use for reading the VCF file. Used **only** for parallel decompression of BGZF blocks. Works only for **local** files.
        save_parquet: Controls saving to parquet.
            - None: do not save
            - "auto": save next to the input, replacing .vcf/.vcf.gz with .parquet
            - Path: save to the provided location
        save_vortex: Controls saving to Vortex format.
            - None: do not save
            - "auto": if parquet saving is enabled, saves next to the parquet with .vortex extension;
              otherwise saves next to the input, replacing .vcf/.vcf.gz with .vortex
            - Path: save to the provided location
        engine: Parquet engine to use for sinking ("auto" or "arrow")
        compression: Compression type for parquet (e.g., "zstd", "snappy")
        compression_level: Compression level for parquet (e.g., 14 for zstd)
        Polars LazyFrame or DataFrame containing the VCF data.
        When saving to parquet, returns a LazyFrame that reads from the parquet file
        (preserving lazy evaluation while ensuring temp files are cleaned up).
        When save_parquet=None, returns the original LazyFrame from polars-bio.

    Note:
        VCF reader uses **1-based** coordinate system for the `start` and `end` columns.
        When saving to parquet, the function creates the parquet file next to the original VCF
        (for "auto") or at the provided path.
    """
    with start_action(
        action_type="read_vcf_file",
        file_path=str(file_path),
        save_parquet=str(save_parquet) if save_parquet else None
    ) as action:
        file_path = Path(file_path)

        # If it's already a parquet file, just scan it
        if is_parquet(file_path):
            action.log(message_type="info", step="detected_parquet", path=str(file_path))
            return pl.scan_parquet(str(file_path))

        # Prepare kwargs for pb.scan_vcf using locals(), excluding non-scan_vcf args
        vcf_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ("file_path", "save_parquet", "save_vortex", "action", "engine", "compression", "compression_level", "alts_list")
        }

        # Resolve parquet path decision early
        if isinstance(save_parquet, Path):
            parquet_path: Optional[Path] = save_parquet
        elif save_parquet == "auto":
            parquet_path = _default_parquet_path(file_path)
        else:
            parquet_path = None

        # Resolve vortex path decision early (can depend on parquet path)
        if isinstance(save_vortex, Path):
            vortex_path: Optional[Path] = save_vortex
        elif save_vortex == "auto":
            vortex_path = parquet_path.with_suffix(".vortex") if parquet_path is not None else _default_vortex_path(file_path)
        else:
            vortex_path = None

        action.log(
            message_type="info",
            step="reading_vcf",
            parquet_path=str(parquet_path) if parquet_path else None,
            vortex_path=str(vortex_path) if vortex_path else None,
        )

        # Let polars-bio handle compression autodetection and any VCF format issues
        vcf_kwargs["info_fields"] = get_info_fields(str(file_path)) if info_fields is None else info_fields

        result = pb.scan_vcf(str(file_path), **vcf_kwargs)
        if alts_list:
            # 1. Define the transformation
            result = result.with_columns(alts=pl.col("alt").str.split("|"))
            
            # 2. Reorder: Find where 'alt' is and slice the column names
            cols = result.collect_schema().names()
            if "alt" in cols:
                idx = cols.index("alt") + 1
                # Move the last column ("alts") to the position right after "alt"
                new_order = cols[:idx] + ["alts"] + [c for c in cols[idx:] if c != "alts"]
                result = result.select(new_order)

        

        action.log(
            message_type="info",
            step="vcf_read_complete",
            result_type=type(result).__name__,
            rows=result.height if hasattr(result, 'height') else 'unknown'
        )

        # Save parquet if requested (and/or needed for vortex)
        effective_parquet_path: Optional[Path] = parquet_path
        temp_parquet_path: Optional[Path] = None

        if effective_parquet_path is None and vortex_path is not None:
            # Need a parquet file for vortex conversion but user didn't request saving parquet.
            # Write a temporary parquet into prepare_annotations cache and remove it after conversion.
            cache_dir = resolve_prepare_annotations_subfolder("tmp")
            cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=f"{file_path.stem}_", suffix=".parquet", dir=str(cache_dir))
            os.close(fd)
            temp_parquet_path = Path(tmp_name)
            effective_parquet_path = temp_parquet_path

        if effective_parquet_path is not None:
            with start_action(
                action_type="save_parquet",
                parquet_path=str(effective_parquet_path),
                is_temporary=temp_parquet_path is not None,
                compression=compression,
                compression_level=compression_level,
            ) as save_action:
                # Use a temporary file for atomic write to avoid corrupted files if interrupted
                actual_path = Path(effective_parquet_path)
                tmp_path = actual_path.with_suffix(".tmp.parquet")
                
                if isinstance(result, pl.LazyFrame):
                    # Stream directly to parquet without collecting into memory
                    result.sink_parquet(
                        str(tmp_path), 
                        engine=engine,
                        compression=compression,
                        compression_level=compression_level,
                    )
                else:
                    # DataFrame path (rare here) – write directly
                    result.write_parquet(
                        str(tmp_path),
                        compression=compression,
                        compression_level=compression_level,
                    )
                
                # Atomic replace
                tmp_path.replace(actual_path)

                save_action.log(
                    message_type="info",
                    step="parquet_saved",
                    parquet_path=str(effective_parquet_path),
                )

        # Save vortex if requested
        if vortex_path is not None:
            from prepare_annotations.vortex.parquet_to_vortex import parquet_to_vortex as _parquet_to_vortex

            with start_action(
                action_type="save_vortex",
                parquet_path=str(effective_parquet_path) if effective_parquet_path else None,
                vortex_path=str(vortex_path),
            ) as vortex_action:
                vortex_output_path = _parquet_to_vortex(
                    parquet_path=effective_parquet_path if effective_parquet_path is not None else file_path,
                    vortex_path=vortex_path,
                    overwrite=False,
                )
                vortex_action.log(
                    message_type="info",
                    step="vortex_saved",
                    vortex_path=str(vortex_output_path),
                )

        # Cleanup temp parquet if we created one for vortex-only conversion
        if temp_parquet_path is not None:
            temp_parquet_path.unlink(missing_ok=True)

        # If we saved parquet (even temporarily), return a LazyFrame scanning it only when it's a user-visible parquet.
        if parquet_path is not None:
            return pl.scan_parquet(str(parquet_path))

        return result


def merge_parquet_files(
    input_files: list[Path],
    output_path: Path,
    compression: str = "zstd",
    compression_level: int = 14
) -> Path:
    """
    Merge multiple parquet files into one using pyarrow.
    This is memory-efficient as it processes files one by one and 
    streams row groups.
    
    Args:
        input_files: List of paths to parquet files to merge
        output_path: Path to the final merged parquet file
        compression: Compression codec to use
        compression_level: Compression level
        row_group_size: Target number of rows per row group in the output
        
    Returns:
        Path to the merged file
    """
    import pyarrow.parquet as pq
    from eliot import start_action
    
    with start_action(action_type="merge_parquet_files", count=len(input_files), output=str(output_path)) as action:
        if not input_files:
            raise ValueError("No input files provided for merging")
            
        # Read schema from the first file
        schema = pq.read_schema(input_files[0])
        
        # Use a temporary file for atomic write
        tmp_output = output_path.with_suffix(".merge_tmp.parquet")
        
        try:
            with pq.ParquetWriter(
                str(tmp_output), 
                schema=schema, 
                compression=compression, 
                compression_level=compression_level,
                # We let the writer handle row group sizes based on data
            ) as writer:
                for f in input_files:
                    action.log(message_type="info", step="merging_file", file=str(f))
                    pf = pq.ParquetFile(f)
                    # Stream row groups to keep memory usage low
                    for i in range(pf.num_row_groups):
                        writer.write_table(pf.read_row_group(i))
            
            # Atomic swap
            tmp_output.replace(output_path)
            return output_path
            
        finally:
            if tmp_output.exists():
                tmp_output.unlink()


def vcf_to_parquet(
    vcf_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    info_fields: Union[list[str], None] = None,
    overwrite: bool = False,
    compression: str = "zstd",
    compression_level: Optional[int] = 14,
    alts_list: bool = True
) -> AnnotatedLazyFrame:
    """
    Read a VCF file and save it to Parquet format, returning both the path and LazyFrame.
    
    This function is a convenience wrapper around read_vcf_file that focuses specifically
    on VCF to Parquet conversion, ensuring the output is always saved and returning
    both the path to the created Parquet file and a LazyFrame for immediate data access.
    
    Args:
        vcf_path: Path to the input VCF file (can be .vcf or .vcf.gz)
        parquet_path: Path where to save the Parquet file. If None, saves next to VCF with .parquet extension
        info_fields: The fields to read from the INFO column. If None, reads all available fields
        overwrite: Whether to overwrite existing Parquet file (default False)
        compression: Compression type for parquet (e.g., "zstd", "snappy")
        compression_level: Compression level for parquet (e.g., 14 for zstd)
        alts_list: Whether to add a list of alternative alleles as 'alts' column
        
    Returns:
        AnnotatedLazyFrame: A tuple containing:
            - LazyFrame reading from the Parquet file for immediate data access  
            - Path to the created Parquet file
        
    Raises:
        FileExistsError: If parquet_path exists and overwrite=False
        
    Example:
        >>> vcf_path = Path("data/variants.vcf.gz")
        >>> lazy_df, parquet_path = vcf_to_parquet(vcf_path)
        >>> print(f"Converted to: {parquet_path}")
        >>> print(f"Shape: {lazy_df.select(pl.len()).collect().item()}")
        
        >>> # Custom output location
        >>> df, custom_path = vcf_to_parquet(vcf_path, "output/my_variants.parquet")
        >>> # Immediate data access
        >>> variant_count = df.select(pl.len()).collect().item()
    """
    with start_action(
        action_type="vcf_to_parquet",
        vcf_path=str(vcf_path),
        parquet_path=str(parquet_path) if parquet_path else None,
        overwrite=overwrite
    ) as action:
        vcf_path = Path(vcf_path)
        
        # Determine output path
        if parquet_path is None:
            output_path = _default_parquet_path(vcf_path)
        else:
            output_path = Path(parquet_path)
        
        action.log(
            message_type="info",
            step="output_path_determined",
            output_path=str(output_path)
        )
        
        # If parquet already exists and overwrite is False, reuse it
        # UNLESS the VCF is newer than the parquet (which suggests we should re-convert)
        if output_path.exists() and not overwrite:
            vcf_mtime = vcf_path.stat().st_mtime
            pq_mtime = output_path.stat().st_mtime
            
            if pq_mtime >= vcf_mtime:
                action.log(
                    message_type="info",
                    step="reusing_existing_parquet",
                    path=str(output_path),
                    vcf_mtime=vcf_mtime,
                    pq_mtime=pq_mtime
                )
                return pl.scan_parquet(str(output_path)), output_path
            else:
                action.log(
                    message_type="info",
                    step="parquet_outdated",
                    path=str(output_path),
                    vcf_mtime=vcf_mtime,
                    pq_mtime=pq_mtime,
                    reason="VCF is newer than existing parquet"
                )
        
        # Use read_vcf_file with forced parquet saving
        lazy_frame = read_vcf_file(
            file_path=vcf_path,
            info_fields=info_fields,
            save_parquet=output_path,
            compression=compression,
            compression_level=compression_level,
            alts_list=alts_list
        )
        
        action.log(
            message_type="info",
            step="conversion_complete",
            output_path=str(output_path)
        )
        
        return lazy_frame, output_path
    







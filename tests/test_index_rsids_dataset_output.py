from __future__ import annotations

from pathlib import Path

import polars as pl

from prepare_annotations.pipelines import compute_rsid_coordinates_task


def _write_test_parquet(path: Path, chrom: str) -> None:
    df = pl.DataFrame(
        {
            "chrom": [chrom, chrom, chrom],
            "start": [100, 100, 200],
            "end": [101, 101, 201],
            "id": ["rs1", "rs1", "rs2"],
            "tsa": ["SNV", "SNV", "SNV"],
        }
    )
    df.write_parquet(path)


def _tail4(path: Path) -> bytes:
    with path.open("rb") as f:
        f.seek(-4, 2)
        return f.read(4)


def test_index_rsids_as_dataset_writes_chunks_and_counts(temp_dir: Path) -> None:
    input_dir = temp_dir / "splitted_variants"
    (input_dir / "SNV").mkdir(parents=True, exist_ok=True)
    (input_dir / "deletion").mkdir(parents=True, exist_ok=True)

    # Two chromosomes, two variant-type dirs. Total rows before DISTINCT:
    # - chr1: 3 rows (2 duplicates for rs1@100 + rs2@200)
    # - chr2: 3 rows (same pattern)
    # Across dirs duplicates also exist, but DISTINCT happens per chromosome group.
    _write_test_parquet(input_dir / "SNV" / "homo_sapiens-chr1.vcf.parquet", "1")
    _write_test_parquet(input_dir / "deletion" / "homo_sapiens-chr1.vcf.parquet", "1")
    _write_test_parquet(input_dir / "SNV" / "homo_sapiens-chr2.vcf.parquet", "2")
    _write_test_parquet(input_dir / "deletion" / "homo_sapiens-chr2.vcf.parquet", "2")

    output_dir = temp_dir / "rsid_coordinates"

    # Prefect task: call underlying function directly for unit test simplicity.
    result = compute_rsid_coordinates_task.fn(
        input_dir=input_dir,
        output_path=output_dir,
        memory_fraction=0.1,
        output_dataset=True,
    )

    assert result.output_path == output_dir
    assert output_dir.is_dir()
    chunk_files = sorted(output_dir.glob("*_rsid_coordinates.parquet"))
    assert len(chunk_files) == 2

    # After DISTINCT per chromosome group, each chromosome should have 2 unique rows.
    # Total should be 4.
    assert result.count == 4


def test_index_rsids_single_file_is_valid_parquet_and_counts(temp_dir: Path) -> None:
    input_dir = temp_dir / "splitted_variants"
    (input_dir / "SNV").mkdir(parents=True, exist_ok=True)
    (input_dir / "deletion").mkdir(parents=True, exist_ok=True)

    _write_test_parquet(input_dir / "SNV" / "homo_sapiens-chr1.vcf.parquet", "1")
    _write_test_parquet(input_dir / "deletion" / "homo_sapiens-chr1.vcf.parquet", "1")
    _write_test_parquet(input_dir / "SNV" / "homo_sapiens-chr2.vcf.parquet", "2")
    _write_test_parquet(input_dir / "deletion" / "homo_sapiens-chr2.vcf.parquet", "2")

    output_file = temp_dir / "rsid_coordinates.parquet"

    result = compute_rsid_coordinates_task.fn(
        input_dir=input_dir,
        output_path=output_file,
        memory_fraction=0.1,
        output_dataset=False,
    )

    assert result.output_path == output_file
    assert output_file.is_file()
    # Parquet files must start and end with magic bytes PAR1.
    with output_file.open("rb") as f:
        assert f.read(4) == b"PAR1"
    assert _tail4(output_file) == b"PAR1"

    # Same semantics as dataset test: 2 unique rows per chromosome -> 4 total.
    assert result.count == 4


from __future__ import annotations

import gzip
from pathlib import Path

from prepare_annotations.genome_downloader import ensure_uncompressed_fasta_with_fai


def test_ensure_uncompressed_fasta_with_fai_creates_fa_and_fai(tmp_path: Path) -> None:
    fasta_text = (
        ">chr1\n"
        "ACGTACGT\n"
        "ACGT\n"
        ">chr2 some description\n"
        "NNNN\n"
    )

    fasta_gz = tmp_path / "test.fa.gz"
    with gzip.open(fasta_gz, "wb") as f:
        f.write(fasta_text.encode("utf-8"))

    fasta_path, fai_path = ensure_uncompressed_fasta_with_fai(fasta_gz, overwrite=False)

    assert fasta_path.exists()
    assert fai_path.exists()
    assert fasta_path.name == "test.fa"
    assert fai_path.name == "test.fa.fai"

    lines = fai_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # chr1 length = 8 + 4 = 12
    chr1 = lines[0].split("\t")
    assert chr1[0] == "chr1"
    assert int(chr1[1]) == 12
    assert int(chr1[2]) > 0  # offset to first base
    assert int(chr1[3]) == 8  # first sequence line bases
    assert int(chr1[4]) in (9, 10)  # includes newline, allow \n or \r\n

    # chr2 length = 4
    chr2 = lines[1].split("\t")
    assert chr2[0] == "chr2"
    assert int(chr2[1]) == 4

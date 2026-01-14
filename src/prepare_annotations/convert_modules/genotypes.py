from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def genotype_has_placeholder(
    *,
    genotype_col: str = "genotype",
    placeholder: str = "?",
) -> pl.Expr:
    """
    Lazy Polars expression: True if the genotype list contains the placeholder allele.

    Assumes `genotype_col` is a `list[str]` (or null).
    """
    return pl.col(genotype_col).list.contains(placeholder)


def resolve_genotype_placeholders_with_ensembl(
    weights: pl.LazyFrame,
    ensembl: pl.LazyFrame,
    *,
    rsid_col: str = "rsid",
    ensembl_id_col: str = "id",
    genotype_col: str = "genotype",
    ref_col: str = "ref",
    alts_col: str = "alts",
    placeholder: str = "?",
    keep_ensembl_cols: bool = False,
    add_allele_valid: bool = False,
) -> pl.LazyFrame:
    """
    Resolve '?' placeholder alleles in a `list[str]` genotype by joining Ensembl and
    replacing '?' with the per-variant `ref` base.

    This is **fully lazy** and **vectorized** (no Python UDFs). It uses the safe
    explode → replace → group_by pattern so replacement can reference `ref`/`alts`.

    Args:
        weights: LazyFrame containing at least `rsid_col` and `genotype_col` (list[str]).
        ensembl: LazyFrame containing `ensembl_id_col`, `ref_col`, `alts_col` (alts is list[str]).
        rsid_col: RSID column name in `weights`.
        ensembl_id_col: Variant id column name in `ensembl` (typically "id").
        genotype_col: Genotype column name in `weights` (list[str]).
        ref_col: Reference allele column in `ensembl` (typically "ref").
        alts_col: Alternate alleles column in `ensembl` (typically "alts", list[str]).
        placeholder: Placeholder allele to replace (default "?").
        keep_ensembl_cols: If True, keep `ref_col` and `alts_col` columns in output.
        add_allele_valid: If True, add boolean column `allele_valid`:
            - True: each genotype allele equals `ref` or is contained in `alts`
            - Null: Ensembl row missing for that rsid (ref/alts null)

    Returns:
        LazyFrame with same columns as `weights` (plus optional `allele_valid` and/or Ensembl cols),
        where `genotype_col` has placeholders replaced and sorted.
    """
    weight_cols = weights.collect_schema().names()
    if rsid_col not in weight_cols:
        raise ValueError(f"weights is missing required column: {rsid_col!r}")
    if genotype_col not in weight_cols:
        raise ValueError(f"weights is missing required column: {genotype_col!r}")

    ensembl_cols = ensembl.collect_schema().names()
    for required in (ensembl_id_col, ref_col, alts_col):
        if required not in ensembl_cols:
            raise ValueError(f"ensembl is missing required column: {required!r}")

    joined = weights.join(
        ensembl.select(
            pl.col(ensembl_id_col).alias(rsid_col),
            pl.col(ref_col),
            pl.col(alts_col),
        ),
        on=rsid_col,
        how="left",
    )

    # We use row index to re-aggregate back to original row granularity after explode.
    exploded = (
        joined.with_row_index("__row_idx")
        .explode(genotype_col)
        .with_columns(
            pl.when(pl.col(genotype_col) == placeholder)
            .then(
                pl.when(pl.col(ref_col).is_not_null())
                .then(pl.col(ref_col))
                .otherwise(pl.lit(placeholder))
            )
            .otherwise(pl.col(genotype_col))
            .alias(genotype_col)
        )
    )

    if add_allele_valid:
        exploded = exploded.with_columns(
            pl.when(pl.col(ref_col).is_null() & pl.col(alts_col).is_null())
            .then(pl.lit(None).cast(pl.Boolean))
            .otherwise(
                (pl.col(genotype_col) == pl.col(ref_col))
                | pl.col(alts_col).list.contains(pl.col(genotype_col))
            )
            .alias("__allele_valid")
        )

    aggs: list[pl.Expr] = [
        *[pl.col(c).first() for c in weight_cols if c != genotype_col],
        pl.col(genotype_col).sort(),
    ]
    if add_allele_valid:
        aggs.append(pl.col("__allele_valid").all().alias("allele_valid"))
    if keep_ensembl_cols:
        # re-attach ref/alts (they are constant per rsid; keep the first)
        aggs.extend([pl.col(ref_col).first(), pl.col(alts_col).first()])

    result = exploded.group_by("__row_idx").agg(aggs).drop("__row_idx")
    if add_allele_valid:
        result = result.drop("__allele_valid")
    if not keep_ensembl_cols:
        # Remove join-only cols if they slipped in via "first()" (when user had same names already,
        # they are part of weight_cols and we should not drop them).
        drop_cols: list[str] = []
        if ref_col in result.collect_schema().names() and ref_col not in weight_cols:
            drop_cols.append(ref_col)
        if alts_col in result.collect_schema().names() and alts_col not in weight_cols:
            drop_cols.append(alts_col)
        if drop_cols:
            result = result.drop(drop_cols)

    return result


def select_ensembl_minimal(
    ensembl: pl.LazyFrame,
    *,
    ensembl_id_col: str = "id",
    ref_col: str = "ref",
    alts_col: str = "alts",
    extra_cols: Iterable[str] = (),
) -> pl.LazyFrame:
    """
    Helper to select Ensembl columns needed for genotype resolution, optionally
    with additional columns (e.g., for debugging/inspection).
    """
    cols = [ensembl_id_col, ref_col, alts_col, *extra_cols]
    return ensembl.select([pl.col(c) for c in cols])


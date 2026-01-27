# Prepare Annotations Agent Guidelines

This repository is dedicated to the preparation of genomic annotation data (Ensembl, ClinVar, dbSNP, gnomAD, etc.) and conversion of OakVar modules from the [dna-seq GitHub organization](https://github.com/orgs/dna-seq/repositories).

---

## General Standards (Reusable)

### Coding Standards

- **Type hints**: Mandatory for all Python code.
- **Pathlib**: Always use for all file paths.
- **Polars**: Prefer over Pandas for performance.
- **Dagster**: Primary tool for workflow orchestration and parallel execution.
- **Eliot**: Used for structured logging and action tracking.
- **Typer**: Mandatory for CLI tools.
- **Pydantic 2**: Mandatory for data classes.
- **Avoid __all__**: Avoid __init__.py with __all__ as it confuses where things are located.

### Dagster Best Practices (Universal)

#### Asset Return Types

| Asset Returns | IO Manager | Use Case |
|---------------|------------|----------|
| `pl.LazyFrame` | `polars_parquet_io_manager` | Small parquet, schema visibility |
| `Path` | Custom IO manager | Large data, DuckDB joins, file uploads |
| `dict` | Default | API responses, upload results |

#### Key Rules

- **dagster-polars**: Use `PolarsParquetIOManager` for `LazyFrame` assets → automatic schema/row count in UI
- **Path assets**: Add `"dagster/column_schema": polars_schema_to_table_schema(path)` for schema visibility
- **Asset checks**: Use `@asset_check` for validation; include via `AssetSelection.checks_for_assets(...)`
- **Streaming**: Use `lazy_frame.sink_parquet()`, never `.collect().write_parquet()` on large data
- **DuckDB**: Use for large joins (out-of-core); set `memory_limit` and `temp_directory`
- **Concurrency**: Use `op_tags={"dagster/concurrency_key": "name"}` to limit parallel execution

#### Resource Tracking (MANDATORY)

**Always track CPU and RAM consumption** for all compute-heavy assets using `resource_tracker`:

```python
from prepare_annotations.core.runtime import resource_tracker

@asset
def my_asset(context: AssetExecutionContext) -> Output[Path]:
    with resource_tracker("my_asset", context=context):
        # ... compute-heavy code ...
        pass
```

**Important:** Always pass `context=context` to enable Dagster UI charts. Without it, metrics only go to Eliot logs.

This automatically logs to Dagster UI:
- `duration_sec`: Execution time in seconds
- `cpu_percent`: CPU usage percentage
- `peak_memory_mb`: Peak RAM usage in MB
- `memory_delta_mb`: Memory change during execution

These metrics appear in the **Asset Details** page and can be plotted over time for performance monitoring.

#### Run-Level Resource Summaries (MANDATORY)

All jobs must include the `resource_summary_hook` to provide aggregated resource metrics at the run level:

```python
from prepare_annotations.definitions import resource_summary_hook

my_job = define_asset_job(
    name="my_job",
    selection=AssetSelection.assets(...),
    hooks={resource_summary_hook},  # Note: must be a set, not a list
)
```

This hook logs a summary at the end of each successful run:
- **Total Duration**: Sum of all asset durations
- **Max Peak Memory**: Highest memory usage (bottleneck identification)
- **Top memory consumers**: Lists the 3 most memory-intensive assets

This helps users with limited RAM identify potential trouble spots before running pipelines.

#### Dagster Version Notes (1.12.x)

**API differences from newer versions:**
- `get_dagster_context()` does NOT exist in Dagster 1.12.x - you must pass `context` explicitly to functions that need it
- `context.log.info()` does NOT accept a `metadata` keyword argument - use `context.add_output_metadata()` separately
- `EventRecordsFilter` does NOT have `run_ids` parameter - use `instance.all_logs(run_id, of_type=...)` instead
- For asset materializations, use `EventLogEntry.asset_materialization` (returns `Optional[AssetMaterialization]`), not `DagsterEvent.asset_materialization`
- `hooks` parameter in `define_asset_job` must be a `set`, not a list: `hooks={my_hook}`
- Use `defs.resolve_all_asset_specs()` instead of deprecated `defs.get_all_asset_specs()`

**External monitoring integrations** (no built-in CPU/memory middleware):
- `dagster-prometheus`: Push custom metrics to Prometheus Pushgateway (beta)
- `dagster-datadog`: Publish metrics to Datadog via DogStatsD (beta)

These allow publishing custom metrics but don't provide automatic system resource tracking. Our `resource_tracker` fills this gap.

#### Dynamic Partitions Pattern

1. Create partition def: `PARTS = DynamicPartitionsDefinition(name="files")`
2. Discovery asset registers partitions: `context.instance.add_dynamic_partitions(PARTS.name, keys)`
3. Partitioned assets use: `partitions_def=PARTS`, access `context.partition_key`
4. Collector depends on partitioned output via `deps=[partitioned_asset]`, scans filesystem for results

#### Execution

- **Python API only**: `defs.resolve_job_def(name)` + `job.execute_in_process(instance=instance)`
- **Same DAGSTER_HOME** for UI and execution: `dagster dev -m module.definitions`
- **All assets in `Definitions(assets=[...])`** for lineage visibility in UI

#### Anti-Patterns

- `dagster job execute` CLI (deprecated)
- Hardcoded asset names; use `defs.get_all_asset_specs()`
- Config for unselected assets (validation errors)
- Suspended jobs holding DuckDB file locks

### Test Generation Guidelines (Universal)

- **Real data + ground truth**: Use actual source data, auto-download if needed, and compute expected values at runtime.
- **Deterministic coverage**: Use fixed seeds or explicit filters; include representative and edge cases.
- **Meaningful assertions**: Prefer relationships and aggregates over existence-only checks.

#### What to Validate

- **Counts & aggregates**: Row counts, sums/min/max/means, distinct counts, and distributions.
- **Joins**: Pre/post counts, key coverage, cardinality expectations, nulls introduced by outer joins, and a few spot-checks.
- **Transformations**: Round-trip survival, subset/superset semantics, value mapping, key preservation.
- **Data quality**: Format/range checks, outliers, malformed entries, duplicates, referential integrity.

#### Avoiding LLM "Reward Hacking" in Tests

- **Runtime ground truth**: Query source data at test time instead of hardcoding expectations.
- **Seeded sampling**: Validate random records with a fixed seed, not just known examples.
- **Negative & boundary tests**: Ensure invalid inputs fail; probe min/max, empty, unicode.
- **Derived assertions**: Test relationships (e.g., input vs output counts), not magic numbers.
- **Allow expected failures**: Use `pytest.mark.xfail` for known data quality issues with a clear reason.

#### Test Structure Best Practices

- **Parameterize over duplicate**: If testing the same logic on multiple outputs, use `@pytest.mark.parametrize` instead of copy-pasting tests.
- **Set equality over counts**: Prefer `assert set_a == set_b` over `assert len(set_a) == 270` - set comparison catches both missing and extra values.
- **Delete redundant tests**: If test A (e.g., set equality) fully covers test B (e.g., count check), keep only test A.
- **Domain constants are OK**: Hardcoding expected enum values or well-known constants from specs is fine; hardcoding row counts or unique counts derived from data inspection is not.

#### Verifying Bug-Catching Claims

When claiming a test "would have caught" a bug, **demonstrate it**:

1. **Isolate the buggy logic** in a test or script
2. **Run it and show failure** against correct expectations
3. **Then show the fix passes** the same test

Never claim "tests would have caught this" without running the buggy code against the test.

#### Anti-Patterns to Avoid

- Testing only "happy path" with trivial data
- Hardcoding expected values that drift from source (use derived ground truth)
- Mocking data transformations instead of running real pipelines
- Ignoring edge cases (nulls, empty strings, boundary values, unicode, malformed data)
- **Claiming tests "would catch bugs" without demonstrating failure on buggy code**

**Meaningless Tests to Avoid** (common AI-generated anti-patterns):

```python
# BAD: Existence-only checks as the sole validation
assert "name" in df.columns
assert len(df) > 0

# BAD: Hardcoded counts derived from data inspection
assert len(source_ids) == 270  # will break when source changes

# BAD: Redundant with set equality test
assert len(output_cats) == 12  # already covered by subset check

# ACCEPTABLE: Required columns as prerequisites
required_cols = {"id", "name", "value"}
assert required_cols.issubset(df.columns)

# GOOD: Set equality from source data
source_ids = set(source_df["id"].unique().drop_nulls().to_list())
output_ids = set(output_df["id"].unique().drop_nulls().to_list())
assert source_ids == output_ids

# GOOD: Domain knowledge constants (from spec, not data inspection)
assert valid_states == {"active", "inactive", "pending"}  # from API spec
```

---

## Project-Specific: prepare-annotations

### Key Entry Points

- `src/prepare_annotations/definitions.py`: Main Dagster definitions (assets, jobs, resources)
- `src/prepare_annotations/cli.py`: Typer CLI - run `uv run prepare --help`
- `src/prepare_annotations/assets/`: Dagster assets (ensembl.py, modules.py)
- `src/prepare_annotations/converters/`: OakVar module converters

### CLI Commands

```bash
uv run prepare longevitymap          # Full pipeline: convert + Ensembl join + upload
uv run prepare longevitymap --convert-only  # Convert only
uv run dagster-ui                    # Launch Dagster UI
uv run modules data --repo dna-seq/just_longevitymap  # Download module data
```

### Unified Annotation Schema

Module conversion produces three standardized parquet files:

1. **annotations.parquet**: `rsid, module, gene, phenotype, category`
2. **studies.parquet**: `rsid, module, pmid, population, p_value, conclusion, study_design`
3. **weights.parquet**: `rsid, genotype, module, weight, state, priority, conclusion, curator, method`
   - State: `protective`, `risk`, or `neutral`
   - Genotype: List of 2 alleles, alphabetically sorted

### Available Modules

Modules from https://github.com/orgs/dna-seq/repositories:
- `just_longevitymap`, `just_coronary`, `just_vo2max`, `just_lipidmetabolism`
- `just_superhuman`, `just_drugs`, `just_pathogenic`, `just_cancer`, `just_prs`

### Testing

```bash
uv run pytest                                    # Run all tests
uv run pytest tests/test_longevitymap_module.py  # Specific module
uv run pytest -v --tb=short                      # Verbose with short traceback
```

### Deployment

Datasets uploaded to `just-dna-seq` organization on HuggingFace Hub.

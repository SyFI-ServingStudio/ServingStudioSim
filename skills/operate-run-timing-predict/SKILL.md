---
name: operate-run-timing-predict
description: >-
  Run offline timing-predict for explicit iteration, attention, or FFN batch
  shapes. Not for scheduler or workload simulation.
---

# Run Timing-Predict

Use timing-predict to cost explicit batch shapes without running the
discrete-event simulator. It builds the selected model CostTree and evaluates
each case once. There is no request trace, scheduler, simulation clock, or SLO.

Read `launcher/timing_predict.py` and `simulator/src/timing_predict.rs` when the
config or cases grammar is unclear.

## 1. Choose the selector

A config contains exactly one outer `arch` selector:

| Target | Selector | Result |
|---|---|---|
| Unified or PD worker | `iter` | Full attention + FFN iteration |
| AFD attention pool | `attn` | Attention-side cost |
| AFD FFN pool | `ffn` | FFN-side section costs |

Run both `attn` and `ffn` configs to represent an AFD deployment. Their
cross-pool handoff is not a CostTree leaf and is not included. In-tree MoE
dispatch and combine communication remains included in the FFN cost.

## 2. Write the config and cases

Start from the nearest `presets/predict_*.json` file. Set:

- one typed `arch` selector and its model parameters;
- `gpu`;
- a fresh `log_dir`;
- `cases_file`, resolved relative to the config file.

The cases file is a top-level JSON array:

- `iter` and `attn`: each case contains `groups`, one per expected attention-DP
  group. A group may contain `prefill_chunk_pairs` and either exact
  `decode_kv_lens` or `decode_count` plus `average_decode_length`.
- `ffn`: each case contains `tokens_per_group`, one value per expected DP group.

Do not mix exact decode lengths with the average-length shorthand. The number of
groups must match the selected architecture; use the validation error rather
than guessing the parallel degree.

## 3. Run the launcher

Run from the repository root:

```bash
uv run python -m launcher timing-predict presets/<config>.json
```

Multiple configs may be passed in one call, for example the two AFD halves.
Supported flags are `--build-type <build-type>` (default `release`) and
`--no-analyze`. Timing-predict does not support simulation-run flags such as
`--dry-run` or `--override`.

Do not pin GPUs by default. A warm `profiling/profile.db` needs no GPU. On a cold
cache, timing-predict asks the profiling layer to JIT-fill the rows it actually
uses, and that layer selects idle GPUs. Do not precompute a guessed row list or
restrict `CUDA_VISIBLE_DEVICES` unless the user or machine policy requires it.

If prediction or JIT profiling fails, preserve the exact error and stop. Do not
report totals from an incomplete run. Use `operate-profile-existing-kernel` only
when diagnosing a specific registered kernel row.

## 4. Check the result

A successful prediction exits zero and writes `prediction.meta.json` plus its
cost artifacts under `log_dir`. Important outputs include:

- `prediction.cases.json`: snapshotted cases;
- `raw/cost_log/worker_predict_0.parquet` and its manifest;
- `reports/iter_breakdown.ans`: per-case CostTree totals and leaves when analysis
  was enabled;
- trace and plot artifacts produced by the normal analysis path.

A timing-predict directory is explicitly cataloged as `timing_predict`; it does
not contain simulation `run_meta.json`, requests, throughput, or SLO results.
Use `operate-use-analyzer` to read user-visible prediction values after analysis
completes.

Report the selector, config and cases paths, command, `log_dir`, per-case totals,
and any profiling failure. For AFD, report attention and FFN results separately.

## Boundaries

- Workload/scheduler simulation: `operate-run-simulation`.
- Querying or filling one registered kernel: `operate-profile-existing-kernel`.

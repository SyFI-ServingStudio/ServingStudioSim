---
name: operate-run-timing-predict
description: Use when the user wants to run VibeSim's offline timing-predict mode — per-building-block cost prediction for explicit batch shapes WITHOUT the discrete-event sim (no scheduler/clock/trace, no workload). Covers the three arch selectors (iter / attn / ffn), the PD→iter and AFD→attn+ffn mapping, the minimal predict config + cases-file grammar, and the launcher entry `python -m launcher timing-predict`. NOT for a real deployment run from a workload trace (that is operate-run-simulation).
---

# Run VibeSim Timing-Predict

Offline **per-building-block timing prediction**. Same family as `dry-run` /
`build-cache-only` / `kernel-query`: it does **not** run the discrete-event sim
(no scheduler, no clock, no request trace, no pools/workload). It takes a batch
of explicit batch shapes ("cases") and, for each, evaluates the compiled
`CostTree(s)` once — predicting the cost of one iteration (or one AFD half)
directly. A warm `profile.db` means **no GPU is needed**; a cold one JIT-profiles
the missing rows on demand (needs a matching GPU on an idle device).

This is the run *workflow* for the predictor. The authoritative code is
`launcher/timing_predict.py` (launcher entry) and `simulator/src/timing_predict.rs`
(the `arch` selectors, config, and cases grammar) — defer to those for edge cases.

Repo root / logs root / launcher: same as `operate-run-simulation`
(`/m-coriander/coriander/kanzhu/VibeSim_workspace/main`, `.../logs`, run under `uv`).

## When NOT to use this

- A **real deployment run** from a workload trace with pools/rates/SLOs →
  `operate-run-simulation`. That path has a `RunConfig`, sweeps, `--dry-run`,
  `--override`, and dated-dir log rewriting. Timing-predict has **none** of those.
- Filling / querying `profile.db` rows for one kernel → `operate-profile-existing-kernel`.

## The one decision: which arch selector

A predict config selects **exactly one** arch family via the outer `arch` key.
Map the user's deployment to selectors:

| Deployment | Selector(s) | Why |
|---|---|---|
| **PD** (prefill/decode disagg) or **unified/colocated** | `iter` (once) | Both PD workers and a unified worker each run a **whole** attn+ffn iteration. Cost one `eval_iter`; give prefill-only and/or decode-only groups in the cases to model each phase. |
| **AFD** (attn/ffn disagg) | `attn` **and** `ffn` (run twice) | The two pools run different code. Predict each side **independently** — one config for `attn`, one for `ffn`, in one launcher call. |

`iter` → one row/case (`section="iter"`). `attn` → one `attn_cost`/case. `ffn` →
the per-section building blocks of one iteration (`prologue`, `pre_attn`,
`post_attn`, `post_attn_last`, `epilogue`) → **5 rows per case**.

> **AFD caveat.** The cross-pool attn↔ffn handoff is a `GpuCluster` transfer, not
> a cost-tree leaf, and is **out of scope** here — timing-predict costs the two
> compute sides only. (The MoE EP dispatch/combine comm *is* an in-tree leaf inside
> the ffn `post_attn` section and is counted automatically.)

## The config (minimal — NOT a RunConfig)

One `arch` selector + `gpu` + `log_dir` + `cases_file`. No workload / pools / io /
sweep. The inner selector reuses the run-side arch grammar (internally tagged on
`type`). Canonical templates live in `presets/predict_qwen3_235b_{iter,attn,ffn}.json`.

```jsonc
// iter (→ PD / unified)
{ "arch": { "iter": { "type": "qwen3_moe_dp_attn_ep_ffn",
              "model_config": "model/config/qwen3_235b.json",
              "attn_tp_size": 4, "ep_size": 8, "hp_size": 1, "nvl_num_gpu": 8, "fp8": false } },
  "gpu": "NVIDIA H200", "log_dir": "logs/<exp>", "cases_file": "<name>_cases.json" }

// attn (→ AFD attn side)
{ "arch": { "attn": { "type": "qwen3_attn_tp", "model_config": "...", "attn_tp_size": 4, "fp8": false } }, ... }

// ffn (→ AFD ffn side)
{ "arch": { "ffn": { "type": "qwen3_ffn_moe", "model_config": "...",
              "attn_tp_size": 4, "ep_size": 8, "nvl_num_gpu": 8, "routing": "uniform", "fp8": false } }, ... }
```

`cases_file` is resolved **relative to the config file's directory**. Easiest
robust layout: author the config (and its cases file, if new) under `presets/`
next to each other, and just point `log_dir` at a dated experiment dir. The
launcher snapshots both the config and its cases file into `log_dir` for
provenance regardless.

## The cases file (grammar differs by arch)

A top-level **array** of cases.

- **`iter` / `attn`** — each case is `{ "groups": [ <group>, ... ] }`, one group
  per attention-DP shard. Each group:
  - `prefill_chunk_pairs`: list of `[prefix_len, append_len]` (fresh prefill has
    `prefix_len = 0`; a chunked-prefill continuation has `prefix_len > 0`).
  - decode is **EITHER** `decode_kv_lens: [k, ...]` (exact per-request KV lengths)
    **XOR** `decode_count` + `average_decode_length` (uniform shorthand) — never both.
  - omit both prefill and decode fields for an empty side (e.g. a decode-only or
    prefill-only iteration).
- **`ffn`** — each case is `FfnArchInput` itself: `{ "tokens_per_group": [t0, t1, ...] }`
  (token counts only, one entry per DP group). It rejects the attention vocabulary
  the ffn cost never reads.

**Group / entry count must match the model's expected DP degree.** `attn` wants
exactly **one** group (one model instance = one DP shard); `iter` / `ffn` want
`num_(attn_)dp_groups` entries. A mismatch is a hard error
(`case has N group(s) but the model expects M`) — read the message and fix the
count, don't guess a formula.

## Workflow

1. Pick the selector(s) from the table above (PD → 1 iter config; AFD → 1 attn +
   1 ffn config).
2. Name the experiment `YYYYMMDD_N_<short-name>` and pick a free per-day index by
   scanning `logs/` (same rule as `operate-run-simulation`; ask if unnamed).
   For AFD, use one index for both halves (e.g. `..._1_predict_afd_attn` +
   `..._1_predict_afd_ffn`).
3. Copy a `predict_qwen3_235b_*` template into `presets/` (or edit one), set its
   `arch` params + `gpu`, set `log_dir` to `logs/<exp>` (there is **no** launcher
   log_dir rewrite / sweep expansion — edit the field directly), and point
   `cases_file` at your cases. Author or adjust the cases file per the grammar above.
4. Run (the launcher builds the release + analyzer binaries and warms the cache
   itself). Pin an idle GPU in case of a cold-cache JIT fill:
   ```bash
   cd /m-coriander/coriander/kanzhu/VibeSim_workspace/main
   # PD:
   CUDA_VISIBLE_DEVICES=<idle> uv run python -m launcher timing-predict presets/<pd_iter>.json
   # AFD (both halves in one call — multiple configs are accepted):
   CUDA_VISIBLE_DEVICES=<idle> uv run python -m launcher timing-predict presets/<afd_attn>.json presets/<afd_ffn>.json
   ```
   `uv run` is mandatory (PyO3 3.12 venv pin; see memory `vibesim_pyo3_build_python_pin`).
   Flags: `--build-type <t>` (default `release`), `--no-analyze` (skip the analyzer
   pass). There is **no** `--dry-run` / `--override` — those are run-only.
5. Read the results (below) and report.

## Output

Each `log_dir` gets the **same artifacts a real run writes** (one worker,
`iter_id` = case index):
- `raw/cost_log/worker_predict_0.parquet` + `raw/cost_manifest/worker_predict_0.json`
  — one row per case (iter/attn) or per section (ffn), byte-compatible with a run.
- `reports/iter_breakdown.ans` — the human-readable cost tree (per-case total +
  per-leaf breakdown); the predict-only report (a real run's thousands of iters
  would make it enormous). This is where the headline per-iteration µs numbers are.
- `traces/<exp>.pftrace.gz` (Perfetto), `plots/` (batch_scatter, kernel tflops/gbps,
  utilization), and a snapshot of the config + cases file.

Request/throughput/SLO analyzer subjects self-skip on a predict dir (no requests);
cost subjects apply. Read `iter_breakdown.ans` with ANSI stripped
(`sed -r 's/\x1b\[[0-9;]*m//g'`) to pull the `total: N us` line per case.

After running, report: the selector(s) used, the experiment name(s) / `log_dir`(s),
the launcher command, per-case predicted totals, and the path to `iter_breakdown.ans`.

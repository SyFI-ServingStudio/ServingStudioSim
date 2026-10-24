# TensorFold Qwen3.8-27B-NVFP4 on GB10: predicted vs actual

Branch `tensorfold_dgx_spark` at `aada6447` in ServingStudioSim (worktree `wt-tf-spark`).

- **Arch:** `qwen35_tensorfold_nvfp4_gdn_dense_dflash2`.
- **Preset:** `presets/predict_qwen38_27b_tensorfold_gb10.json`.
- **Kernel rows:** all from the `vibesim-profiler-tensorfold:402406087e778b7f` image, keyed `NVIDIA GB10` in `profiling/profile.db`.

## Workload

TensorFold multi-stream (`serve_tf_27b_nvfp4.sh`: DFlash2 drafter, 16 rows per stream per verify round) replays the
RAG-agent trajectories (`qwen3.8-27b-bf16-rag-sonnet5-trajectories`) at 1, 2, 4, 8 and 16 concurrent conversations.

- **Captures:** nsys with TensorFold NVTX ranges and `TENSORFOLD_ITER_RECORDS`.
  - `logs/20261023_5_tf27b_batch/`: levels 1–16.
  - `logs/20261023_7_tf27b_n16_steady/`: N=16 after a 900 s warm-up.
- **Cases:** every captured round becomes a timing-predict case, built with `alignment/timing_predict_input/tensorfold_iter_records.py`. A case carries the per-stream kv/query lengths, the accepted length and tree slot class, and the prefill pieces.
- **Comparison:** `tools/tensorfold_compare/tensorfold_compare.py` checks each case against the matching round.

## Result: per-round kernel time, predicted vs measured GPU busy time

| level | rounds | measured med ms | predicted med ms | median err | sum err | p90 abs err |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 524 | 100.8 | 98.8 | -2.2% | -2.1% | 3.5% |
| 2 | 400 | 134.9 | 131.2 | -2.8% | -2.8% | 4.0% |
| 4 | 298 | 150.9 | 145.6 | -3.5% | -3.5% | 4.6% |
| 8 | 195 | 193.4 | 185.8 | -3.6% | -2.3% | 4.6% |
| 16 (ramp) | 78 | 676.5 | 661.5 | -3.3% | -3.7% | 6.7% |
| 16 (steady) | 172 | 752.5 | 739.6 | -2.5% | -2.7% | 5.1% |

## Wall time: kernel prediction plus a fitted host term, vs measured round wall

The host term is fitted on levels 2 and 4 only: 5.905 ms + 0.498 ms per tree stream + 11.335 ms per prefill piece.
Levels 1, 8 and 16 are holdouts.

| level | role | median err | sum err |
|---:|---|---:|---:|
| 1 | holdout | -0.5% | -0.7% |
| 2 | fit | -2.1% | -2.5% |
| 4 | fit | -3.2% | -3.6% |
| 8 | holdout | -2.2% | -1.5% |
| 16 (steady) | holdout | -1.7% | -2.0% |

## Where the remaining error comes from

Per-group numbers are in `wt-tf-l4arch/logs/20261023_tf_predict/analysis/kernel_groups.md`.

- **Predicted low:**
  - Verify NVFP4 linears (`lane_kernel`): -3 to -10%. This is the largest absolute error.
  - Verify tree attention: -2.5 to -7%.
  - At N=16, the prefill GEMMs and prefill attention: about -13 ms per round.
  - The GDN tree kernel: -4 to -23% after the fix. In-forward drift makes later GDN layers about 10% slower than the first 12 in the capture.
- **Predicted high:**
  - Cold-L2 small kernels (activation quants, verify swiglu): +55 to +150%.
- **Not modeled:** CUDA-graph-free launch gaps inside a round (covered by the host term), and the extra GDN chain launch on a `keep_at` cut in a prefill piece.

## Fix applied during validation

`gdn_tree_verify_recurrence` was priced 33–60% high. The profiler times every kernel on a cold L2, but in production
`_gdn_pre` writes q/k/v/g/beta just before `tree_kernel`. The runner now flushes L2, rewrites those inputs, and then
times only `tree_kernel` (commit `e387e0e7`). Cache fidelity: 106 of 108 probes within ±15%, median 0.999.

## Quiet-machine revalidation

Fresh `--fresh` re-measurement of 8 stored rows per backend, after all lanes merged
(`revalidate/`, `revalidate.py` in `logs/20261023_6_quiet_revalidation/`).

- 34 of 35 TensorFold backends have a fresh/stored median within 0.95–1.05.
- **Exceptions:**
  - `gdn_chunk_delta_rule/tensorfold_chain`: median 0.88 (0.79–0.97). The prefill GDN chain may be priced about 12% high; it is a prefill-only, small share.
  - `elementwise/tensorfold_torch`: median 0.95 with one row at 0.61, a tiny kernel.
  - `static_fp8_quant/tensorfold_cuda`: not re-measured, because the generic `--fresh` path rejects it with "no gated form". It is a script limitation, not a data finding.

## Reproduce

```bash
cd wt-tf-spark
uv run python -m launcher timing-predict logs/20261024_10_tf_predict_final/batch/predict.json --no-gpu --analyzer-essential-only
uv run python tools/tensorfold_compare/tensorfold_compare.py \
  --sidecar logs/20261024_10_tf_predict_final/batch/cases/tensorfold_round_sidecar.json \
  --log-dir logs/20261024_10_tf_predict_final/batch/run \
  --nsys 1=logs/20261023_5_tf27b_batch/tf27b_n1.sqlite ... --nsys 16=logs/20261023_5_tf27b_batch/tf27b_n16.sqlite \
  --fit-levels 2,4 --out-dir logs/20261024_10_tf_predict_final/batch/compare
```

To predict a new workload, generate cases from `TENSORFOLD_ITER_RECORDS` with
`python -m alignment.timing_predict_input.tensorfold_iter_records <records.jsonl> --phases <phases.tsv> --output-dir <dir>`.
Then point a copy of `batch/predict.json` at the new `cases_file`. The `cases_file` path is relative to the config file.

## Scripts in this folder

`scripts/` holds copies of the run scripts that live in the gitignored `logs/` directories:

- `capture_batch.sh`: the nsys capture at each concurrency level.
- `build_replay.py`: builds the trajectory replay.
- `replay.py`: the concurrent replay client.
- `revalidate.py`: the quiet-machine re-measurement.

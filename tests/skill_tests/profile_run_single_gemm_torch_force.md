# Profile Single GEMM Torch Force Refresh

## Agent Input

Work in `/m-coriander/coriander/kanzhu/VibeSim_workspace/VibeSim` with no prior
knowledge beyond repo-local instructions. Use the appropriate repo-local skill
to profile the existing L1 kernel `single_gemm` with backend `torch`.

Use exactly one spec:

```json
{"m": 128, "n": 8192, "k": 8192, "dtype": "bf16"}
```

Use a temporary DB under `/tmp`, force-refresh the row, and report the exact
commands, missing count before and after, and a compact result table. Use only
the public `uv run python -m profiling ...` CLI path; do not call runners or
`run_profile_batch` directly.

## Expected Output Description

The final answer should say which instruction files were used, especially
`AGENTS.md` and `skills/operate-profile-existing-kernel/SKILL.md`. It should show
the exact CLI commands for `list --json`, a pre-run `count-missing`, the
`run --force --json`, and a post-run `count-missing` or `query` against the
same temp DB.

The result table should include at least `m`, `n`, `k`, `dtype`, `status`,
`time_ms`, `tflops`, and `energy_j`. The expected final status is `ok`, with
post-run missing count `0`.

For NVIDIA A100/H100/H200-class BF16 runs, `tflops` should be finite and in a
broad sanity range of `50 <= tflops <= 2500`. For other GPUs, the agent should
report the detected GPU name and explain the adjusted sanity range instead of
pretending this fixed range is universal.

## Pass Criteria

- Uses `skills/operate-profile-existing-kernel/SKILL.md`.
- Confirms `single_gemm` / `torch` exists through `python -m profiling list`.
- Uses a temp DB path and records it in the final report.
- Runs through `python -m profiling run ... --force --json`.
- Verifies persistence with a post-run `count-missing` or `query`.
- Reports one result row for the one input spec.
- Reports finite `time_ms > 0`, `tflops > 0`, and `energy_j >= 0`.

## Failure Signals

- Directly invokes a runner module, `profiling.db.batch.run_profile_batch`, or
  `profiling.exec.local_worker`.
- Uses the shared/default DB without being asked.
- Describes `--gpu-name` as a CUDA device selector.
- Treats remaining missing rows as success without clearly reporting them.

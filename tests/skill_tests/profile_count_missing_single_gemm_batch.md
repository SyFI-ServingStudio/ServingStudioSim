# Count Missing Single GEMM Batch

## Agent Input

Work in the repository root supplied by the test harness with no prior knowledge
beyond repo-local instructions. Use the appropriate repo-local skill
to count missing rows for an existing L1 kernel without profiling anything.

Use table `single_gemm`, backend `torch`, a fresh temporary DB under `/tmp`, and
this three-spec batch:

```json
[
  {"m": 128, "n": 8192, "k": 8192, "dtype": "bf16"},
  {"m": 256, "n": 8192, "k": 8192, "dtype": "bf16"},
  {"m": 512, "n": 8192, "k": 8192, "dtype": "bf16"}
]
```

Use either repeated `--spec` flags or a temporary JSON/JSONL specs file. Report
the exact command and count summary.

## Expected Output Description

The final answer should show the agent used
`skills/operate-profile-existing-kernel/SKILL.md`, confirmed the registry entry via
`list --json`, and ran only `count-missing --json` for the batch.

For a fresh temp DB, the expected summary is `spec_count: 3` and
`missing_count: 3`.

## Pass Criteria

- Uses the public `python -m profiling count-missing` command.
- Passes backend through `--backend torch`, not inside each spec.
- Keeps each spec to `m`, `n`, `k`, and `dtype`.
- Reports the selected spec input form and spec count.
- Does not run profiling or require CUDA availability.

## Failure Signals

- Uses `run`, `run --force`, a runner module, or `run_profile_batch`.
- Mixes backend into the individual spec objects.
- Reports only aggregate success without the concrete `missing_count`.

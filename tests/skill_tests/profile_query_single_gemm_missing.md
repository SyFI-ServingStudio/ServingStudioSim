# Query Single GEMM Missing Row

## Agent Input

Work in `/m-coriander/coriander/kanzhu/VibeSim_workspace/main` with no prior
knowledge beyond repo-local instructions. Use the appropriate repo-local skill
to query, but not profile, the existing L1 kernel `single_gemm` with backend
`torch`.

Use exactly one spec:

```json
{"m": 128, "n": 8192, "k": 8192, "dtype": "bf16"}
```

Use a fresh temporary DB under `/tmp`. Report the exact command and the missing
status. Do not run real CUDA profiling.

## Expected Output Description

The final answer should show that the agent discovered
`skills/operate-profile-existing-kernel/SKILL.md`, confirmed `single_gemm` /
`torch` with `list --json`, and then used `query --json` against the fresh temp
DB.

`query` should exit successfully for a valid request even though the row is
absent. The JSON/result summary should report `missing_count: 1`,
`spec_count: 1`, and one result with `status: "missing"`.

## Pass Criteria

- Uses `python -m profiling query`, not `run`.
- Uses a fresh temp DB path and records it in the final report.
- Reports `missing_count: 1` and `status: "missing"`.
- Explains that this is expected for a clean DB and read-only query mode.
- Does not require CUDA availability, because no profiling should occur.

## Failure Signals

- Runs `python -m profiling run` or calls a runner directly.
- Treats `status: "missing"` as a CLI failure instead of an expected read-only
  miss.
- Omits the temp DB path from the report.

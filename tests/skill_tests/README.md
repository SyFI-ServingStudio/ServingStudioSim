# Skill Tests

These files are agent-evaluation fixtures for repo-local skills. They are not
pytest tests; pytest covers code behavior, while these cases check whether a
fresh coding agent can discover the right skill, choose the public workflow, and
report the result in the expected shape.

Each case contains:

- `Agent Input`: the prompt to give a fresh agent or subagent.
- `Expected Output Description`: what a correct final report should contain.
- `Pass Criteria`: concrete checks for the evaluator.
- `Failure Signals`: behaviors that should fail the case even if the command
  happened to run.

Future agents should keep these cases implementation-facing: describe allowed
public commands and expected report shape, but do not embed private shortcuts
such as direct runner calls.

## Automated Harness

Use `run_codex_skill_tests.py` to evaluate cases with two independent Codex CLI
invocations:

- runner: receives only the `Agent Input` section and performs the task.
- judge: receives the whole case plus the runner's final answer and returns a
  JSON verdict using `judge_schema.json`.

Examples:

```bash
uv run python tests/skill_tests/run_codex_skill_tests.py --list
uv run python tests/skill_tests/run_codex_skill_tests.py --case profile_query_single_gemm_missing
uv run python tests/skill_tests/run_codex_skill_tests.py --case profile_run_single_gemm_torch_force --runner-sandbox danger-full-access
```

Artifacts are written under `/tmp/mlsim_skill_tests/<timestamp>/` by default.
Use `--out-dir` to keep a specific run. GPU profiling cases may require
`--runner-sandbox danger-full-access` on hosts where the normal Codex sandbox
cannot access CUDA/NVML.

By default, the harness prints a readable report for each case with the test's
`Agent Input`, the runner agent's final output, and the evaluator's structured
decision. Use `--brief` for CI-style one-line PASS/FAIL output. Long sections
are bounded by `--max-output-chars`; the complete transcript and logs are always
available in the artifact directory. If the runner or judge Codex subprocess
fails before producing a final answer, the console report includes a process
diagnostics section with stderr/stdout excerpts.

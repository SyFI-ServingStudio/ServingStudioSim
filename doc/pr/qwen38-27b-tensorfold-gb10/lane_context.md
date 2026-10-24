# Shared context for TensorFold kernel lanes (GB10 / DGX Spark)

Read this before your lane brief. It is the same for every lane.

## Goal

ServingStudio Sim needs measured L1 kernel timings for **TensorFold** serving `Qwen3.8-27B-NVFP4` with the DFlash2
drafter on one **NVIDIA GB10** (DGX Spark: sm_121, aarch64, unified 128 GB memory). The timings feed a future
TensorFold arch for timing prediction. The production engine config is
`/home/yilegu/airbox/perf_study/serve/serve_tf_27b_nvfp4.sh`: `--parallel 16 --context 131072`, up to 16 concurrent
requests, 16 rows per stream per verify round, prefill steps of at most 1024 rows while others decode and 4096 rows
otherwise.

## Evidence

All of it is under `/home/yilegu/ServingStudio/wt-tf-spark/`:

- `doc/pr/qwen38-27b-tensorfold-gb10/split.md`: the kernel split and the verdict for every op. Your lane implements
  one row.
- `notes.md`: TensorFold engine facts, with source anchors.
- `logs/20261023_4_tf27b_nsys/`: batch-1 nsys capture. `kernel_inventory.json` has kernel time by NVTX phase;
  `tf27b_formal.sqlite` is the export, read-only.
- `logs/20261023_5_tf27b_batch/`: captures at 1/2/4/8/16 concurrent RAG-agent conversations (in progress).

TensorFold source is the submodule in **your** worktree: `alignment/profiler/tensorfold/src/tensorfold/`, commit
6469730a (perf-study). Never edit it.

Kernel entry points you can call with synthetic tensors:
- `tensorfold.cuda.nvfp4.checkpoint`:
  - `quant4`, `quant8`, `matmul` (decode `lane`), `prompt` (prefill `gemm_ck`), `matmul_group`, `mlp_prompt`;
  - raw ops via `_ext()`;
  - `Fp4Linear.from_checkpoint` / `Fp8Linear.from_checkpoint` in `nvfp4/linear.py`.
- `tensorfold.cuda.kernels.prefill_attention.attention`
- `tensorfold.cuda.kernels.attention`: `plan`, `attention`
- `tensorfold.cuda.kernels.gdn`
- `tensorfold.families.qwen3_5.cuda.glue`

Microbench examples that build real objects: `/home/yilegu/airbox/perf_study/bench/tf_mm_bench.py`, `tf_verify_gap.py`.
The real checkpoint config is `/home/yilegu/airbox/perf_study/models/Qwen3.8-27B-NVFP4/config.json`. Do not load
weights; random data of the right layout is enough.

## Environment

- Shell: `source /home/yilegu/ServingStudio/.env; export PATH="$HOME/.cargo/bin:$HOME/.local/opt/node/bin:$HOME/.local/opt/mold/bin:$HOME/.local/opt/protoc/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"`.
  Always use `uv run ...` inside your worktree. Read the worktree's `CLAUDE.md` and `AGENTS.md`.
- `tensorfold_env` (registered in `profiling/exec/env.py`) is the submodule's `.venv`, symlinked: Torch
  2.14.1+cu130, Triton 3.8.0. Your backends set `subprocess_env="tensorfold_env"`. TensorFold JIT-compiles its CUDA
  extensions on first call; the first call can take minutes.
- Runner development: `VIBESIM_PROFILE_RUNTIME=host` runs workers on that host interpreter. It is for debugging,
  not for rows you keep.
- Rows you keep come from the `tensorfold` profiler image:
  - `uv run python -m profiling.container.images build tensorfold`. The tag does not depend on runner code, so every
    lane uses the same image.
  - Only the `tf-gemm` lane builds it. Other lanes wait until `docker image inspect $(uv run python -m
    profiling.container.images tag tensorfold)` succeeds, and use host runtime for smoke until then.
- GPU key: `--gpu-name "NVIDIA GB10"`.

## GPU rules: one GB10, shared by every lane

- Wrap every command that touches the GPU in `flock /home/yilegu/ServingStudio/tmp/gpu.lock <command>`. That covers
  profiling runs, tests that use CUDA and TensorFold JIT builds. A batch capture holds the lock until about 08:05
  UTC; your first GPU command simply waits.
- Do the non-GPU work first: registration, Torch reference, Rust wiring, CPU tests.
- Never set `VIBESIM_PROFILE_GPUS` or `CUDA_VISIBLE_DEVICES`.
- **The GB10's CPU and GPU share one power budget.** Heavy CPU work moves the GPU clock between 1.9 and 2.5 GHz,
  which skews another lane's measurements by up to about 20% (seen by the tf-gdnchain lane). Heavy CPU work means
  cargo build/test, `just test-*`, `uv sync`, docker builds and big pytest runs. Run all of it under the same
  `flock /home/yilegu/ServingStudio/tmp/gpu.lock` as GPU work.
- GB10 timings also drift 5-10% between sessions. Before keeping a grid, re-measure a few grid points in the same
  session as your fidelity probes.
- Memory is unified with the CPU. Keep a single profiling submission's memory use moderate (no 30 GB tensors).
  `MemAvailable` must stay above 10 GB.
- Keep each lock hold short: one complete `run --specs` submission per hold, then release.

## Workflow and conventions

- You own one kernel row end to end. Run `skills/top-add-kernel/SKILL.md` from your worktree: Python profiling
  (`orchestrator-add-kernel-to-python-profile`, `impl-register-kernel`), then Rust wiring
  (`orchestrator-wire-kernel-to-rust`, `impl-wire-kernel-to-rust`, `impl-validate-kernel-cache`). Follow their "Things
  to verify" lists. Also read `top-add-new-arch` "Modeling rules":
  - row keys hold model geometry and dtypes only;
  - grids are generic and span the full deployment range;
  - no artificial caps.
- The operation contract in your brief is already confirmed by the user, so skip asking them about Task A.1. Do
  return to the orchestrator (finish and report) for any decision your brief does not settle:
  - a new args field or a different kind or backend name;
  - a shape the kind cannot express;
  - a production boundary that differs from the brief.
- Time the production public callable with `Timer.cupti`. Allocation, JIT compile, correctness checks and the
  Torch oracle stay outside the timed closure. Check correctness against an independent Torch reference outside
  timing.
- DB rows:
  - Smoke goes to a temp DB (`$TMPDIR/<lane>/profile.db`).
  - The final grid fill goes to your worktree's `profiling/profile.db` (skip-worktree, already a copy of the shared
    DB). Do not commit `profile.db`; the orchestrator merges it.
- Commits: on your own branch only, never push. Use `git -c user.name=yilegu -c
  user.email=yilegu@cs.washington.edu commit`. End messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Keep notes in your worktree's `progress.md`; do not commit it.

## Report back

1. Commit hashes and files changed.
2. Tests run, with pass/fail counts.
3. Smoke evidence: correctness error vs the Torch reference, plus a few `time_ms` values with their `KernelArgs`.
4. Grid filled: spec count, missing before/after, image tag used.
5. Rust `kernel-query` evidence and cache-fidelity numbers.
6. Every deviation from the brief and every open question.

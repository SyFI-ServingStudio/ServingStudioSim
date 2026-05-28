# MLSim — main

Implementation tree for MLSim. Layout follows `docs/file_structure.md`
(symlinked to the design docs in the legacy MoESim repo via `../ref/`).

See `../README.md` for workspace-level context (ref/, worktrees, milestones).

## Quick start

```bash
# Rust simulator
cargo check                      # verify scaffold compiles
cargo run -- list-params         # emit deployment schema JSON
cargo run -- run <run_config.yaml>

# Python side
uv sync                          # install default dev + profiling deps
uv run ruff check .              # lint

# Python profiling stack (Torch/Triton/NVML; needed for real CUDA profiling)
uv run python -c "import torch, triton, pynvml; print(torch.__version__)"
uv run python -m profiling list
uv run python -m profiling count-missing single_gemm --backend torch --gpu-name H100 --spec '{"m":4096,"n":8192,"k":8192,"dtype":"bf16"}'
uv run python -m profiling run single_gemm --backend torch --force --specs specs.json --db /tmp/profile.db
```

`dev`, `profiling`, `launcher`, and `analyze` are default uv groups for this
repo, so use plain `uv run ...` for tests, lint, profiler, launcher, and renderer
entry points. The execution backend selects `main/.venv/bin/python` but does not
install missing packages at profile time. The initial lockfile tracks the
reference CUDA 12.8-era stack
(`torch 2.10.x`, `triton 3.6.x`) until the cluster driver/runtime target is
validated for a newer stack.

## Layout

- `simulator/src/` — Rust crate (L1 timing → L7 sim/log/schema/deployment).
  Top-level `main.rs` is the clap binary entry; `lib.rs` re-exports layer
  modules.
- `profiling/` — Python L1a runners + L1b db + exec backend.
- `launcher/` — Python L7-α launcher.
- `analyzer/` — Rust `analyze` binary + Python plot renderer.
- `model/config/`, `gpu/`, `trace/`, `tests/` — data + tests.
- `docs/` → `../ref/next_gen_design/` (symlink; design is read-only here).

## First milestone

Llama3-8B dense, local single-server, no parallel, single-round trace.
See `../README.md` for the layer-by-layer slice. Build order is L1 → L2 → L3
→ L4 → L5 → L6 → L7, with each layer's vertical slice driven by the next.

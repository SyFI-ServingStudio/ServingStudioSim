# MLSim — main

Implementation tree for MLSim. Layout follows `docs/file_structure.md`
(symlinked to the design docs in the legacy MoESim repo via `../ref/`).

See `../README.md` for workspace-level context (ref/, worktrees, milestones).

## Quick start

```bash
# Rust simulator
cargo check                      # verify scaffold compiles
cargo run -- list-params         # CLI entry, prints stub
cargo run -- run                 # main run path (stub)

# Python side
uv sync                          # install deps (currently none)
ruff check .                     # lint
```

## Layout

- `simulator/src/` — Rust crate (L1 timing → L7 sim/log/schema/deployment).
  Top-level `main.rs` is the clap binary entry; `lib.rs` re-exports layer
  modules.
- `profiling/` — Python L1a runners + L1b db + exec backend.
- `launcher/` — Python L7-α launcher.
- `analyze/` — Python analyzer, grouped by subject
  (worker / pool / lifecycle / profile_db / cross / common).
- `model/config/`, `gpu/`, `trace/`, `tests/` — data + tests.
- `docs/` → `../ref/next_gen_design/` (symlink; design is read-only here).

## First milestone

Llama3-8B dense, local single-server, no parallel, single-round trace.
See `../README.md` for the layer-by-layer slice. Build order is L1 → L2 → L3
→ L4 → L5 → L6 → L7, with each layer's vertical slice driven by the next.

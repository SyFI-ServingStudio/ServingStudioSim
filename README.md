# VibeSim Core

VibeSim is a discrete-event simulator and performance-modeling toolkit for LLM
serving systems. This repository contains the Rust simulator, Rust Analyzer,
Python launcher, kernel profiler, model/deployment definitions, and the
repository-local Agent skills that operate them.

For the complete browser + Agent + Analyzer deployment, use the
[`VibeSimWorkspace`](https://github.com/serendipity-zk/VibeSimWorkspace)
meta-repository. It pins compatible revisions of this repository, VibeSimAgent,
and VibeSimUI and provides the tested `reproduce.md` and root `justfile`.

## Requirements

- Python 3.12 managed by `uv`
- Rust stable
- NVIDIA driver/CUDA stack for profiling and GPU-backed predictions
- A writable, large-capacity `TMPDIR`

Do not run Python or install packages with the system interpreter. Use
`uv run ...` and `uv add ...` from the repository root.

## Build and inspect

```bash
uv sync
cargo build --release

# Rust-authoritative deployment schema and Launcher-rendered human view
cargo run --release -- list-params
uv run python -m launcher list-params --human

# Launcher and Analyzer surfaces
uv run python -m launcher --help
cargo run -p analyzer --release -- --help
```

The launcher is the supported run boundary. It builds/discovers the simulator
schema, validates presets, expands sweep/compound/variant axes, prebuilds kernel
cache rows, records provenance, starts simulations, and triggers requested
Analyzer subjects.

```bash
# Inspect the expanded plan before launching
uv run python -m launcher presets/unified_aime.yaml --dry-run

# Launch the preset
uv run python -m launcher presets/unified_aime.yaml

# Offline timing and existing-kernel profiling entry points
uv run python -m launcher timing-predict --help
uv run python -m launcher kernel-profile --help
```

Use a task-scoped directory under `$TMPDIR` for scratch databases or generated
inputs. Production and managed runs write durable artifacts beneath their
declared workspace `logs/` roots.

## Analyzer

Analyzer is the read-only resource authority for simulation runs and sweeps,
timing predictions, kernel profiles, kernel measurements, rendered plots, and
GPU hardware limits.

```bash
cargo build -p analyzer --release
target/release/analyze serve \
  --bind 127.0.0.1:8787 \
  --workspace-registry ../agent-workspaces/registry.json
```

The workspace registry is maintained by VibeSimAgent. Analyzer discovers
approved workspace roots through that registry; it does not own conversation or
job-lifecycle state.

## Repository map

```text
simulator/        Rust L1-L7 simulator and deployment schema
profiling/        Python kernel registry, runners, execution backend, and cache
launcher/         preset, sweep, managed-run, timing-predict, and profile CLI
analyzer/         Rust read API and Python plot renderer
alignment/        vLLM-under-nsys profiling and measured-vs-simulated reconciliation
model/            model configurations, and model/work/ the independent
                  theoretical-minimum FLOP/byte labeler
gpu/              hardware specification catalog
trace/            trace generators and checked-in samples
presets/          runnable deployment/prediction configurations
skills/           canonical VibeSim Agent workflows
tests/            CPU, GPU, binary, database, and Agent test tiers
doc/              current architecture and detailed design
old-doc/          archived legacy design symlink
```

Start with [`doc/README.md`](doc/README.md) and
[`doc/architecture.md`](doc/architecture.md). Operational details live beside
their implementation, especially [`launcher/README.md`](launcher/README.md) and
[`profiling/README.md`](profiling/README.md).

## Validation

The repository test tiers intentionally separate CPU-only checks from GPU,
built-binary, warm-database, and Agent-runtime checks. Use the repository
`dev-run-tests` skill or inspect `just --list` before selecting a tier.

For focused changes, format and test only the touched files. Do not run
workspace-wide formatters over unrelated worktrees or generated artifacts.

## Licensing

This project is source-available under a dual community licensing
model.

You may use it under whichever of the following licenses applies to
your use:

- **PolyForm Noncommercial License 1.0.0** — for noncommercial,
  research, educational, and other uses permitted by that license.
- **PolyForm Internal Use License 1.0.0** — for internal business use,
  including internal use and modification by commercial organizations.

The two are alternatives; you do not need to satisfy both.

Uses not permitted by either community license, including external
commercial productization and redistribution, require a separate
commercial license.

See [`LICENSING.md`](LICENSING.md) for details and
[`COMMERCIAL-LICENSING.md`](COMMERCIAL-LICENSING.md) for commercial licensing
information. Third-party components remain under their own licenses — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Contributors keep their
copyright and grant the rights in [`CLA.md`](CLA.md); see
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

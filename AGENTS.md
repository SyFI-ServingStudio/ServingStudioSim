# ServingStudio Sim — agent notes

**The project rules for this repo live in [`CLAUDE.md`](CLAUDE.md). Read it
first.** It covers the environment (`just sync`, everything under `uv`), how to
run a simulation (`python -m launcher <preset>.json`), and the test tiers.

This file exists separately because Codex reads `AGENTS.md` while Claude Code
reads `CLAUDE.md`. It is also the mount target a managed-agent container writes
its generated instructions onto, so keep it a short pointer rather than a second
copy of the rules — a duplicated copy would drift out of sync with `CLAUDE.md`
and there would be no way to tell which one was stale.

Workspace-wide rules that sit above this checkout (doc ground truth, worktree
layout, formatter scope, L1 conventions) are in `../AGENTS.md`.

## Orientation

- `doc/README.md` — the current design record: the seven-layer stack, per-layer
  documents, invariants, and the analyzer contract.
- `skills/skill-of-skills/SKILL.md` — the repo-local skill tree. Enter a workflow
  through the highest matching skill (`top-*` → `orchestrator-*` → `impl-*`).
- Per-module `README.md` files next to the code are the finest-grained reference
  and win over `doc/` when they disagree.

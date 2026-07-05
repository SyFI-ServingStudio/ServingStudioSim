---
name: dev-orchestrate-parallel-subagents
description: "Use when you are about to launch two or more subagents that will each MODIFY the same repository concurrently, such as several independent feature/kernel additions in one fan-out. Mandates git worktree isolation per agent so concurrent edits never race on shared files, every agent can run its task skill unmodified and self-validate end to end, and the orchestrator integrates afterward."
---

# Dev Orchestrate Parallel Subagents

When you fan out **2+ subagents that each write to the same repo**, give each its
own **git worktree**. Do NOT run concurrent writing agents in one shared working
tree.

This is an *orchestration* concern, not a task concern: keep it out of the
task-specific skills (they should describe the normal single-agent flow). The
isolation lives here.

## Why worktrees

- **Shared-file races.** Concurrent read-modify-write on shared files (module
  barrels, `mod.rs`, `__init__.py`, lockfiles, registries) clobber each other —
  the last writer silently drops the others' edits.
- **Flaky whole-repo validation.** `cargo test` / `pytest` over a tree that
  several agents are mid-writing produces false failures.
- **Clean skills + clean roles.** With isolation, each agent runs its task skill
  unmodified (wire barrels, build, test, all in its own copy), self-validates end
  to end, and you never hand-do implementer work. The task skill stays
  single-agent; "don't stomp siblings" stops being its problem.

## How to launch

1. One worktree per agent: Agent tool `isolation: "worktree"` (or
   `git worktree add` and point the agent at it). One worktree = one branch = one
   agent.
2. Tell each agent it OWNS its worktree: it MAY edit shared barrels and run full
   validation freely — no sibling can stomp it. (Drop any "defer barrels / don't
   run cargo" constraints — those were only needed for shared-tree runs.)
3. Each agent writes a report at `agent-trace/<topic>.md` (see
   [[agent-trace-workflow]]); pass the path in the brief.

## Reuse the heavy envs (don't rebuild per worktree)

A fresh worktree has no gitignored build state (`.venv`, `target/`). Avoid
multi-GB rebuilds:

- **Python / uv:** the uv cache already holds every wheel (the main `.venv` was
  synced from the same lock). When the cache and the worktree share a filesystem,
  `uv sync` / `uv run` **hardlinks/clones** from the cache — no download, no copy,
  near-instant. If they are on different filesystems (copy fallback), either set
  `UV_CACHE_DIR` onto the worktree's filesystem, or set
  `UV_PROJECT_ENVIRONMENT=<main>/.venv` so `uv run` reuses the main env directly
  (zero work; valid because the worktree's `pyproject`/lock are identical).
- **Rust / cargo:** optionally `CARGO_TARGET_DIR=<main>/target` to share build
  artifacts — concurrent `cargo` then serializes on the build lock (correct, just
  not parallel at the compile step). Otherwise each worktree compiles fresh.

## Integration (orchestrator, after agents finish)

1. Review each worktree's diff + its `agent-trace` report.
2. Merge: new files are disjoint; the shared barrels get a one-line addition per
   agent → take the **union** of those lines (a trivial conflict to resolve).
3. Run **one** consolidated validation on the merged tree: whole-repo build +
   tests + any **cross-agent invariant** that only holds on the union (e.g. a
   registry validator that sees all agents' entries only after merge — each
   isolated worktree validated against just its own + baseline).
4. Fill the Review + Feedback sections of each `agent-trace` file.
5. Clean up worktrees (Agent `isolation: "worktree"` auto-removes ones with no
   changes; remove the rest once merged).

## When NOT to isolate

- A single writing agent (no concurrency) — use the normal tree.
- Read-only / research agents — no writes, no races.
- Agents provably on fully disjoint files with zero shared-file edits — but
  barrels / lockfiles usually betray hidden sharing; when unsure, isolate.

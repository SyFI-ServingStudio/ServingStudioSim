---
name: dev-create-worktree
description: Use when the user asks to create, adopt, or set up a git worktree for VibeSim development, including converting an existing agent checkout into a wt-topic tree. Creates a sibling worktree from current main and provisions required working-copy artifacts. NOT for the parallel-writing-subagent isolation flow (that is dev-orchestrate-parallel-subagents).
---

# Create an VibeSim worktree

Spin up an isolated git worktree for a piece of VibeSim work AND stage the test
inputs that a bare `git worktree add` leaves behind, so the new tree can run
`uv run python -m launcher …` and `just test-*` without first re-profiling
kernels or hunting for trace files.

## Why this skill exists

`git worktree add` only materializes **committed** files. Three things a run/test
needs are therefore missing or unsafe in a fresh or adopted worktree:

1. **`profiling/profile.db`** — tracked, but the primary tree's *working copy* is
   almost always richer than the committed snapshot (every GPU run JIT-fills new
   kernel rows into it without committing). The worktree would get the stale
   committed `profile.db` and re-profile on the next GPU run — slow and needs a
   GPU. Copy the working copy over.
2. **Untracked traces** — large workload CSVs are deliberately **not**
   committed, so they never reach a new worktree: `trace/aime_long.csv` (~5 MB)
   and the full-dataset TraceLab traces `trace/tracelab_reported.csv` /
   `trace/tracelab_preserving.csv` (~23 MB each). Presets that reference them
   (`presets/*_aime*.yaml`) fail without them. Copy them over. Their committed
   `*.manifest.json` sidecars say which policy produced each file, so a worktree
   missing the CSV still records what it is supposed to contain.
3. **Checkout-bound profiler environments** — an editable install records the
   absolute source checkout, while vLLM's precompiled CUDA extensions are
   materialized as untracked `.so` files beside that source. Copying or moving
   `.venv` can therefore leave the interpreter in `wt-topic` importing Python
   from `main/`, or leave `wt-topic` source without `_C`, `_moe_C`,
   `_flashmla_C`, and the FlashAttention extensions. Rebind the environment to
   the new checkout before any alignment run.

The top-level `.venv/`, `target/`, and `__pycache__` are git-ignored and rebuilt
on demand. Do **not** copy them: a copied environment may retain absolute
editable-install paths, and a copied `target/` may carry stale incremental
state. The nested vLLM profiler environment is not created by an ordinary
top-level `uv run`; provision it explicitly in step 4 when the worktree will
run alignment.

## Convention

Worktrees are **siblings of `main/`**, named `wt-<topic>/` — never nested inside
`main/`. The workspace root holds `main/` and every `wt-*/` next to it:

```
<workspace-root>/
├── main/          ← primary tree (source of the working profile.db + traces)
├── wt-<topic>/    ← what this skill creates
└── …
```

Set these once:

```bash
WORKSPACE_ROOT=<directory-containing-main>
MAIN_WORKTREE=$WORKSPACE_ROOT/main
WORKTREE_TOPIC=<topic>              # short kebab, e.g. kv-cache-logging
NEW_WORKTREE=$WORKSPACE_ROOT/wt-$WORKTREE_TOPIC
BRANCH_NAME=$WORKTREE_TOPIC         # or a name the user gave
```

## Steps

### 1. Create the worktree off the current `main/` HEAD

Branch from whatever `main/` currently has checked out (the code you explored),
NOT from `master`/`origin` — the active mainline branch here is usually an
`afd-*` / feature branch, and its committed line numbers are what any plan was
written against.

```bash
BASE_BRANCH=$(git -C "$MAIN_WORKTREE" rev-parse --abbrev-ref HEAD)
git -C "$MAIN_WORKTREE" worktree add -b "$BRANCH_NAME" "$NEW_WORKTREE" "$BASE_BRANCH"
```

If `$BRANCH_NAME` already exists, drop `-b` and pass the branch as the last arg
instead. Confirm with `git -C "$MAIN_WORKTREE" worktree list`.

### 2. Provision the working-copy `profile.db` (rich, but git-invisible)

Copy the primary tree's working `profile.db`, then mark it `skip-worktree` in
the new worktree so it stays present for runs but never shows as modified and is
never swept into a `git commit -am`:

```bash
cp "$MAIN_WORKTREE/profiling/profile.db" "$NEW_WORKTREE/profiling/profile.db"
git -C "$NEW_WORKTREE" update-index --skip-worktree profiling/profile.db
```

(`skip-worktree` is the key: `profile.db` is a tracked binary, so without this
the copy would leave the worktree permanently dirty and risk an accidental
commit of the primary tree's kernel cache.)

### 3. Provision untracked traces

`rsync` the whole `trace/` dir — tracked files already match, so this only adds
the untracked ones (`aime_long.csv`, the `tracelab_*.csv` pair, any others).
Untracked files are not swept by `git commit -am`, so no extra guarding is
needed:

```bash
rsync -a "$MAIN_WORKTREE/trace/" "$NEW_WORKTREE/trace/"
```

(For an unusually large trace set you may symlink instead — `ln -s
"$MAIN_WORKTREE/trace/<file>" "$NEW_WORKTREE/trace/<file>"` — but copy is the robust default and
keeps the worktree self-contained.)

### 4. Qualify checkout-local profiler environments

Only when alignment is in scope, follow the environment setup in
`alignment/profiler/README.md` from inside the new worktree. Never copy or reuse
another checkout's `.venv` or native `.so` files, and never point `PYTHONPATH`
at another checkout.

Before calling the worktree ready, verify that the profiler interpreter and its
required Python and native modules resolve under `$NEW_WORKTREE`, and run the
owner-documented package, driver, and representative model-path checks. If any
check fails, report that alignment setup is incomplete; do not substitute a
different checkout's binaries.

### 5. Report + first-run note

Tell the user the worktree path, its branch, and that the **first** `uv run` /
`just test-*` inside it will `uv sync` + build the release binary (a few minutes,
one-time). All later commands are fast. Every command must run **from inside
`$NEW_WORKTREE`** and under `uv` (see `CLAUDE.md` env rules). If step 4 applied,
also report the exact vLLM version and the checkout path printed by its probe.

## Verify (optional but recommended)

Cheap CPU check that the tree is wired up:

```bash
cd "$NEW_WORKTREE" && just test-cpu          # Rust --lib + mocked pytest, no GPU
```

Or a dry-run of an aime preset to confirm the trace resolved:

```bash
cd "$NEW_WORKTREE" && uv run python -m launcher presets/unified_aime.yaml --dry-run
```

If the dry-run errors on a missing `trace/aime_long.csv`, step 3 did not land.

## Relationship to other skills

- Once the worktree is ready, use `operate-run-simulation` to launch sims and
  `dev-run-tests` for the test tiers — both assume the artifacts this skill staged.
- For importing a coherent change from `main/` into an existing worktree, use
  `dev-present-changes-for-review`; do not copy selected files to imitate a
  rebase.
- For isolating **multiple concurrent writing subagents**, use
  `dev-orchestrate-parallel-subagents` instead; this skill is for a single
  developer/agent worktree.

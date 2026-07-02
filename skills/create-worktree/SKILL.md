---
name: create-worktree
description: Use when the user asks to create or set up a git worktree for MLSim development (e.g. "start a worktree", "make a wt-<topic> worktree", "work on X in a worktree"). Creates a sibling `wt-<topic>/` worktree off the current `main/` HEAD on a fresh branch, then provisions the untracked / working-copy test artifacts that `git worktree add` does NOT carry — the working-copy `profile.db` kernel cost cache (marked `skip-worktree` so it stays rich yet git-invisible) and untracked traces such as `trace/aime_long.csv` — so simulations and tests run immediately without re-profiling on a GPU. NOT for the parallel-writing-subagent isolation flow (that is `orchestrate-parallel-subagents`).
---

# Create an MLSim worktree (with test artifacts)

Spin up an isolated git worktree for a piece of MLSim work AND stage the test
inputs that a bare `git worktree add` leaves behind, so the new tree can run
`uv run python -m launcher …` and `just test-*` without first re-profiling
kernels or hunting for trace files.

## Why this skill exists

`git worktree add` only materializes **committed** files. Two things a run/test
needs are therefore missing in a fresh worktree:

1. **`profiling/profile.db`** — tracked, but the primary tree's *working copy* is
   almost always richer than the committed snapshot (every GPU run JIT-fills new
   kernel rows into it without committing). The worktree would get the stale
   committed `profile.db` and re-profile on the next GPU run — slow and needs a
   GPU. Copy the working copy over.
2. **Untracked traces** — large workload CSVs like `trace/aime_long.csv`
   (~5 MB) are deliberately **not** committed, so they never reach a new
   worktree. Presets that reference them (`presets/*_aime*.yaml`) fail without
   them. Copy them over.

Everything else a run needs (`.venv/`, `target/`, `Cargo.lock`, `__pycache__`)
is git-ignored and rebuilt on demand: the first `uv run` / launcher call does
`uv sync` and builds the release binary. Do **not** copy those — a copied
`target/` can carry stale incremental state across a different source tree.

## Convention (see memory `mlsim-worktree-convention`)

Worktrees are **siblings of `main/`**, named `wt-<topic>/` — never nested inside
`main/`. The workspace root holds `main/` and every `wt-*/` next to it:

```
/m-coriander/coriander/kanzhu/MLSim_workspace/
├── main/          ← primary tree (source of the working profile.db + traces)
├── wt-<topic>/    ← what this skill creates
└── …
```

Set these once:

```bash
WS=/m-coriander/coriander/kanzhu/MLSim_workspace
MAIN=$WS/main
TOPIC=<topic>              # short kebab, e.g. kv-cache-logging
WT=$WS/wt-$TOPIC
BRANCH=$TOPIC              # or a name the user gave
```

## Steps

### 1. Create the worktree off the current `main/` HEAD

Branch from whatever `main/` currently has checked out (the code you explored),
NOT from `master`/`origin` — the active mainline branch here is usually an
`afd-*` / feature branch, and its committed line numbers are what any plan was
written against.

```bash
BASE=$(git -C "$MAIN" rev-parse --abbrev-ref HEAD)   # current main/ branch
git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$BASE"
```

If `$BRANCH` already exists, drop `-b` and pass the branch as the last arg
instead. Confirm with `git -C "$MAIN" worktree list`.

### 2. Provision the working-copy `profile.db` (rich, but git-invisible)

Copy the primary tree's working `profile.db`, then mark it `skip-worktree` in
the new worktree so it stays present for runs but never shows as modified and is
never swept into a `git commit -am`:

```bash
cp "$MAIN/profiling/profile.db" "$WT/profiling/profile.db"
git -C "$WT" update-index --skip-worktree profiling/profile.db
```

(`skip-worktree` is the key: `profile.db` is a tracked binary, so without this
the copy would leave the worktree permanently dirty and risk an accidental
commit of the primary tree's kernel cache.)

### 3. Provision untracked traces

`rsync` the whole `trace/` dir — tracked files already match, so this only adds
the untracked ones (`aime_long.csv`, any others). Untracked files are not swept
by `git commit -am`, so no extra guarding is needed:

```bash
rsync -a "$MAIN/trace/" "$WT/trace/"
```

(For an unusually large trace set you may symlink instead — `ln -s
"$MAIN/trace/<file>" "$WT/trace/<file>"` — but copy is the robust default and
keeps the worktree self-contained.)

### 4. Report + first-run note

Tell the user the worktree path, its branch, and that the **first** `uv run` /
`just test-*` inside it will `uv sync` + build the release binary (a few minutes,
one-time). All later commands are fast. Every command must run **from inside
`$WT`** and under `uv` (see `CLAUDE.md` env rules).

## Verify (optional but recommended)

Cheap CPU check that the tree is wired up:

```bash
cd "$WT" && just test-cpu          # Rust --lib + mocked pytest, no GPU
```

Or a dry-run of an aime preset to confirm the trace resolved:

```bash
cd "$WT" && uv run python -m launcher presets/unified_aime.yaml --dry-run
```

If the dry-run errors on a missing `trace/aime_long.csv`, step 3 did not land.

## Relationship to other skills

- Once the worktree is ready, use `run-simulation` to launch sims and
  `run-tests` for the test tiers — both assume the artifacts this skill staged.
- For isolating **multiple concurrent writing subagents**, use
  `orchestrate-parallel-subagents` instead; this skill is for a single
  developer/agent worktree.

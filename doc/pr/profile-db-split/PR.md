# feat(profiling): store the profile DB as one SQLite file per kind

## Purpose

`profiling/profile.db` held every kind table in one SQLite file. On `main` it is
99.9 MB, at GitHub's 100 MB file limit; #138 adds Kimi rows and has to move it to
Git LFS. Every commit that touched a row also stored a full new copy of the file,
and two branches that touched any rows conflicted on the whole file.

The profile DB is now a directory, `profiling/profile_db/<kind>.db`. No LFS is
needed: the largest file is `mega_moe.db` at 20 MB. A commit stores new
blobs only for the kinds it changed, and conflicts and diffs stay with those kinds.

- **Layout** (`profiling/db/layout.py`, the only module that knows it).
  - Each file holds one kind table, the `_profile_run` rows its rows reference,
    and `_db_metadata`, so it reads on its own: every existing JOIN is a kind
    table with `_profile_run`, and no query needs `ATTACH`.
  - `run_key` is a content hash and runs are insert-only, so a run copied into
    several files is the same row in each.
  - `VIBESIM_PROFILE_DB` now names a directory.
- **Readers and writers.**
  - `Table` opens its kind's file.
  - The read-only connection cache keys by file, with an LRU cap of 128 and the
    same identity check as before. A build that alternates kinds keeps its
    connections instead of reopening one per switch.
  - `perf_api`, `public_api`, achievable peaks, `gpu/measure_sustained.py`,
    remote nodes and the profile-DB lease all use the directory.
  - Any reader given an old single-file DB refuses it and names `split-db`.
- **Tools.**
  - `kernel-profile split-db FILE --output DIR` (new) converts a single-file
    DB: rows are copied as they are, `id` included, then checked against the
    source in both directions. It reports orphan `run_key`s and does not repair
    them.
  - `migrate-db`, the provenance audit and metadata work file by file.
  - `merge-db` merges directories kind by kind and accepts a single-file DB as
    either input; the output is published only when there are no conflicts.
- **Data.** `main`'s `profile.db` (75ffa185, sha256 `57d6bb59…`) split
  without changes:
  - 145 files holding 220,220 rows;
  - 7 empty tables get no file;
  - 78 runs that no row references are not copied;
  - 3,636 rows reference 8 `run_key`s that have no `_profile_run` row in
    `main`'s file. They are kept as they are, and a separate issue will cover
    them.
- **Skills and docs.** Every command that copied, merged or committed the single
  file now works on the directory: worktree setup, model finalization, profiling
  and the parallel-agent skills. Also updated: the module READMEs, `L1.md`, and
  the PR template.

## Test Plan

- `CUDA_VISIBLE_DEVICES="" just test-cpu`
- `just test-presets`: every public preset member builds against the
  directory and finds every row its kernels read.
- `kernel-profile split-db` of `main`'s file: the built-in check compares each
  kind file with its source table in both directions.
- `kernel-profile merge-db profiling/profile_db <shared single-file pool>`:
  directory against a legacy file at full size.
- `ruff check` / `ruff format --check` on the changed Python files.
- Not run:
  - `just test-gpu`;
  - a real `--remote` job (`test_remote_pieces.py` covers the copy and merge);
  - a timing-predict comparison of old code and file against new code and
    directory over every preset.

## Test Result

- `test-cpu`: pytest 5890 passed, 36 skipped; Rust 1926 passed.
- `test-presets`: 9 passed, 1 skipped.
- `split-db`: 145 files, 220,220 rows, every file equal to its source table.
  Total 103.0 MB, against 99.9 MB for the single file.
- `merge-db` against the shared pool (a single file holding `main`'s rows plus
  local ones), 18 s:
  - 91,275 rows inserted, 220,220 duplicates, 0 conflicts, across 148 kinds;
  - every kind's insert count equals what a single-file merge of the same two
    DBs inserted. That merge's report also counted 9 `_profile_run` rows; the
    new report lists kind tables only.
- `ruff`: clean, except one E501 in `profiling/cli.py:447` that is on `main`
  unchanged.
- The four energy-window tests in `test_l1_single_gemm.py` failed on `main`
  under `CUDA_VISIBLE_DEVICES=""`, because the fake torch has no device UUID.
  They now unmask the variable, in a commit of their own.

## Dependencies

- **#138 (Kimi-K3.1-Pro).** It moves `profile.db` to Git LFS. After this PR it
  should drop the LFS move and fold its rows into `profiling/profile_db/`:
  `kernel-profile merge-db profiling/profile_db <its profile.db> --output <dir>`
  takes the single file as it is.
- **Working copies and worktrees.** Shared working copies and existing
  worktrees still hold a single `profile.db`. Convert them once with
  `kernel-profile split-db`, or fold them with `merge-db`.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

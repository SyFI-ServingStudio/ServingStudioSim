---
name: dev-finalize-model-addition
description: >-
  Use when a new model's arch is built and aligned and the user wants it landed on main: check the alignment
  results, publish presets and captures, finish profiling, merge main, review, draft and open the PR. Not for
  building the arch (top-add-new-arch) or only walking a diff (dev-present-changes-for-review).
---

# Finalize a Model Addition

This skill runs after `top-add-new-arch` Phase 5. Work through the steps in order. Each step ends with a
check; do not start the next step until that check passes. Stop and ask the user where a step says so.
Uploading captures, pushing, and opening the PR are public and hard to undo, so each one needs the user's
go-ahead.

## 1. Check the alignment results

Two kinds of evidence must exist and be reasonable:

- **diverse_100.** Run the `trace/diverse_100.csv` capture through Check 1 (kernel) and Check 3 (E2E from
  the `workload_metrics` pass), per `top-align-with-framework`.
- **Campaign cases.** Take these from the model's pack under `presets/alignment/<pack>/` (the 15 standard
  operating points). Run at least one high-concurrency case, one prefill-heavy case, one decode-only case,
  and one long-context case, and cover `chunk_size` 2048, 4096 and 8192 (see `operate-run-alignment`).
  Run more if the GPU budget allows. Produce the table with
  `alignment-campaign compare --measured --markdown`.

"Reasonable" means one of two things. The case is within the tolerances in the pack's `acceptance.yaml`,
or its miss has a stated, checked cause. Either way, mapping coverage must meet the `map_coverage_pct` floor
(97% in current packs) and no large measured kernel may be unmapped. Write the result table and each miss's cause in `logs/<experiment>/summary.md`.

When a miss has no cause yet, return to `top-align-with-framework`. Any change to the cost model, a
calibration constant, or the acceptance bounds needs the user's approval. Do not tune a number to close a
gap.

## 2. Maintain the presets

The public site reads these files. Add or update them for the new model:

- `presets/public/<checkpoint>/<arch type>.yaml`, the arch presets;
- `presets/public_sim/<checkpoint>/<name>.yaml`, the deployment presets;
- the checkpoint's entry in `model/catalog.yaml` and the arch's entry in `model/arch_catalog.yaml`.

`public_api/README.md` defines both formats. Start from the closest existing model's files.

Do not put experiment files under `presets/`. Move every case config, alignment `simulation.yaml`, and
timing-predict case file to `logs/<experiment>/` before the PR. The one exception is the alignment pack in
`presets/alignment/<pack>/`, because `alignment-campaign` reads packs from that directory. To find strays,
run `git diff --name-only origin/main... -- presets/` and `git status --short presets/`.

Check: `just test-presets` passes.

## 3. Upload the captures

Each public workload row names its routing capture by an `hf://` path pinned to a commit of the Hugging Face
dataset `UW-SyFI/servingstudio-workload`. Upload every capture the new presets read. For a MoE model that is
at least the diverse_100 token corpus (`manifest.json`, `routes.u16`). For a speculative model it also
includes the `trace.csv` that carries the measured per-request acceptance. A dense model without speculation
has nothing to upload; go to step 4.

1. Lay out each capture as the dataset README describes:
   `<model>/<engine>[_<proposer>_k<k>]/<workload>/capture/<YYYYMMDD>/`, with `provenance.json` and
   `trace.csv`. Add a row for it to the README table.
2. Never overwrite a data file. A second capture of the same key goes in a new date directory, and its
   `provenance.json` names the first capture in `replicate_of`.
3. Show the user the file list, sizes, and the README row. Upload only after they confirm. Put the capture
   directory and the README edit in one `huggingface_hub.HfApi().create_commit(repo_id=..., repo_type="dataset",
   ...)`. Record the returned commit sha.
4. Set the preset path to `hf://datasets/UW-SyFI/servingstudio-workload@<40-hex sha>/<capture path>/manifest.json`.

Check: `hf_hub_download(..., revision=<sha>)` returns files whose sha256 matches the local copies, and
`just test-presets` passes. It fails when a captured workload row's path is not pinned to a sha, or
when a MoE member states no routing.

## 4. Finish kernel profiling

`just test-presets` builds every public preset member against the shipped `profiling/profile.db`.
`test_every_member_is_measured` lists each member with missing rows. Fill those rows on GPU through
`operate-profile-existing-kernel` (Slurm, per `run-gpu-job`) and repeat until the test passes.

Check: `just test-presets` passes with no missing rows.

## 5. Merge main and review

1. Run `git fetch origin && git merge origin/main` on the feature branch. Resolve every conflict.
2. Fold the local `profiling/profile.db` into main's copy. The file is `skip-worktree`, so the rows step 4
   filled are not committed and `git status` does not show them. Never resolve it by choosing one side.
   - Write main's copy out with `git show origin/main:profiling/profile.db > $TMPDIR/main.db`.
   - Upgrade both copies in place with `uv run python -m launcher kernel-profile migrate-db <db>`.
   - Merge them with `uv run python -m launcher kernel-profile merge-db $TMPDIR/main.db profiling/profile.db
     --output $TMPDIR/merged.db --report $TMPDIR/merge.json`. Read the report, then copy the merged file over
     `profiling/profile.db`.
   - Commit the database as its own commit: `git update-index --no-skip-worktree profiling/profile.db`, then
     `git add` and `git commit`, then `git update-index --skip-worktree profiling/profile.db`.
3. Read every resolved hunk against both sides with `git show --remerge-diff <merge commit>`. A file-level
   check misses a wrong hunk.
4. Run `just test-cpu` and `just test-presets`. Run `just test-gpu` through Slurm.
5. Review each changed core file with `dev-file-design-review`. Fix the findings on the feature branch and
   rerun the tests.

Check: the merge is committed, every test tier passes, and the review has no open findings.

## 6. Draft the PR

Write `doc/pr/<feature>/PR.md` and `doc/pr/<feature>/README.md`. Follow `doc/pr/dsv41-flash/`.

- `PR.md` has these sections: Purpose, Test Plan, Test Result, Dependencies, Contribution licensing.
- `README.md` holds the evidence. Include the deployment modeled, how each model feature is modeled, the
  kernels, Check 1, the simulation, changes that reach other models, modeling choices, known gaps, and how
  to reproduce the results.
- `PR.md`'s Test Result carries the performance numbers from step 1 as tables, not as ranges:
  - for each deployment, the diverse_100 Check 1 table (signed median per phase, coverage, multiplier) and
    the E2E table (TTFT, TPOT, E2E, throughput);
  - the campaign table with one row per case run: kernel abs and signed error, coverage, E2E, output tps,
    TPOT, and TTFT.

  Paste the campaign table from the generated `alignment-campaign compare --markdown` output, and commit
  that output as `doc/pr/<feature>/alignment_matrix.md` beside the metrics JSON. Name each table's source
  file and never retype a number.
- List every out-of-tolerance item with its cause, and every change that alters other models' predictions.

Stop: show the user the draft. Before showing it, check that every case in the pack's run appears as a row
in `PR.md`.

## 7. Create the review worktree

The user reviews the whole change from main as unstaged edits in their own editor. Build a worktree on
main whose working copy equals the feature branch:

```bash
git worktree add -b <feature>-review ../wt-<feature>-review origin/main
cd ../wt-<feature>-review
git restore --source=<feature> --worktree -- .
git diff --name-only --diff-filter=A origin/main <feature> | xargs -r git add -N
```

`git restore --source` changes only the working copy and leaves the index at main. `git add -N` marks new
files intent-to-add so that `git diff` shows them. Record the tree in the root `worktree.md`.

Check: `git diff --stat` in the review tree matches `git diff --stat origin/main <feature>` in the feature
tree. Then walk the user through the change with `dev-present-changes-for-review`.

## 8. Apply the user's feedback to both branches

Make every fix on the feature branch, then refresh the review tree. The user may comment or edit the review
tree directly.

- **Edits made in the review tree.** Run `git diff <feature>` there. It shows exactly the user's edits.
  Apply the output to the feature tree with `git apply`, then commit it on the feature branch.
- **Fixes from comments.** Commit them on the feature branch.
- **Refresh.** In the review tree, run `git restore --source=<feature> --worktree -- .` and repeat the
  `git add -N` line. `git diff <feature>` there must now be empty.
- **Main moved.** Merge it into the feature branch (step 5). In the review tree, confirm that
  `git diff <feature>` is empty, then run `git reset --hard origin/main` and repeat the step 7 commands.

Rerun the tests that the fixes affect. Update the PR draft whenever a number or a listed change moves.

## 9. Open the PR

Wait until the user approves the review and the draft. Then:

```bash
git push -u origin <feature>
gh pr create --base main --head <feature> --title "<PR.md title>" --body-file doc/pr/<feature>/PR.md
```

Keep `goal.md`, `progress.md`, `notes.md`, and `logs/` out of the commits. After the PR merges, update the
root ServingStudio submodule pointer in its own commit. Retire the review worktree and the feature
worktree under `old-wt/` as the root `CLAUDE.md` describes.

---
name: dev-parallelize-arch-addition
description: >-
  Use in single-agent mode when one session runs a `top-add-new-arch` build by
  launching its own parallel subagents: how to cut the work into lanes and when the
  serial steps run. Not for the orchestrator–implementer loop, and not for one
  writing agent's worktree (that is dev-orchestrate-parallel-subagents).
---

# Parallelize an Arch Addition

**Scope.** This applies only when one agent session does the work itself and
may launch subagents: an interactive CLI session, or ServingStudio Agent's
`agent_mode: single`. It does not apply to the `orchestrated` mode, where the
orchestrator must delegate to the implementer and never launches subagents;
there, `top-add-kernel`'s one-kernel-per-task delegation is the unit.

In this skill you are the coordinator. You write the design, brief the lanes,
merge their branches, and decide what runs next. You do not write kernels, worklets, or the
arch yourself. Every lane is one subagent in its own worktree, per
`dev-orchestrate-parallel-subagents`.

The capture drives the plan. The measured kernel sequence and each kernel's
share of iteration time decide the lanes, their order, and how each worklet's
paths are checked. The capture starts first and runs in the background while
everything else is built.

## The plan

| Stage | Lanes | Runs in parallel with | Done when |
|---|---|---|---|
| 0. Start | **capture**: `operate-run-alignment` profile phase for diverse_100 (nsys, workload, and `token_corpus` for MoE), then the pack's case passes. **source map**: read the framework's model code and write which kernel each module launches. | each other | diverse_100 capture parsed; source map written |
| 1. Design | you: rank kernel families by measured share, write the design (below) | capture case passes | user agreed to the design decisions |
| 2. L1 kernels | one lane per kernel family (for example GDN, PLE, HC, MoE, QSA, glue), each running `top-add-kernel` | capture | rows in `profile.db`, cache fidelity checked, `just test-cpu` green |
| 3. L3 worklets | one lane per mechanism, branched from the merged kernel branch, each running `impl-compose-worklet` | `model.work` label lane (`impl-add-model-work-label`) | worklet unit tests green |
| 4. L4 arch | **one** lane: `impl-compose-arch` + `impl-wire-new-arch`, then a timing-predict dry run | nothing that edits its files | arch builds, tests green, dry run lists every missing row |
| 5. Fill and label | one batched profiling pass for all missing rows; location map; pack labels and `ARCH_TYPE` | each other | Check 1 runs on diverse_100 |

Stages 2–3 are where the parallelism pays. Stage 4 stays serial: one agent
owns the arch, the dispatch files, and the README tables.

## The design document

Write it in `logs/<date>_arch_design/design.md` after the capture and the
source map exist. It holds:

- the deployment, copied from the framework recipe the user chose;
- the per-op decision table from `top-split-model-into-kernels`, sorted by
  measured share;
- each decision the user must make, numbered (D1, D2, …), with your
  recommendation. Stop and get answers before briefing any lane;
- a **task table**: one row per lane with the files it creates, the shared
  files it may edit, and what it waits for. No two lanes create the same file.
  Each lane adds exactly one line to a shared module list such as `mod.rs`, so
  merge conflicts there are a union of lines.

Write one common brief per stage (for example `kernel_brief.md`,
`compose_brief.md`) with the rules every lane follows. Each lane prompt then
names its row in the task table, the brief, and its worktree.

## Running the lanes

- **Kernel lanes profile on GPU through Slurm.** Short kernel profiling goes to
  the priority partition; follow `run-gpu-job`. Rows come from the serving
  framework's own environment when the framework's torch build differs from the
  simulator's (`torch_linear_vllm` is the precedent: same runner, different
  `subprocess_env`).
- **A lane that finds the design wrong reports it with source or capture
  evidence.** It does not silently change another lane's scope. You decide, and
  ask the user when the change is a modeling decision.
- **Collect interface notes for the arch lane.** When a worklet lane reports,
  copy every fact the arch must respect into `logs/<date>_arch_design/arch_notes.md`:
  which leaf owns which norm or copy, input field meanings, backend names,
  deviations from the design. The arch lane reads this file first. Send it any
  note that arrives after it started.
- **Check reports against each other.** Two lanes can describe the same pass
  differently (for example one assumes a cache is shared across draft passes,
  another found it is not). Resolve it from the framework source and send the
  fix to the lane that was wrong.
- **Batch the missing rows.** Do not profile a gap as soon as you suspect it.
  The arch dry run lists every missing row at once; profile them in one Slurm
  submission.
- **Stop finished agents.** After a lane reports and you merged its branch,
  check it has no Slurm job or tmux session left, then stop it.

## Merging

Merge each lane into the feature branch as it finishes, in any order:

1. `git merge --no-ff <lane branch>`. Resolve the module-list conflict by
   keeping both sides' lines.
2. Fold the lane's `profile.db` into the feature tree's with
   `launcher kernel-profile merge-db` (never pick one side), and commit the
   database as its own commit. `dev-finalize-model-addition` step 5 has the
   commands. Check the report shows 0 conflicts.
3. Build (`cargo check`) after each merge. Run `just test-cpu` once all lanes
   of a stage are in.
4. Do not merge into the tree while the arch lane is editing it; wait for its
   commit.

Record every lane in the root `worktree.md` and archive each finished lane
worktree under `old-wt/`.

## When it is done

The plan is done when Check 1 runs on diverse_100 with the arch, every missing
row is filled, and the `model.work` location map exists. Continue with
`top-align-with-framework`, then `dev-finalize-model-addition`.

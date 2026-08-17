---
name: dev-present-changes-for-review
description: >-
  Guide a user-paced review of a branch or diff: inventory first, then one
  coherent change at a time. Not a correctness or design audit of one file.
---

# Present Changes for Review

The goal is to make **the user's own review** fast and thorough. You are the
guide, not the judge: you inventory the whole changeset, categorize it, and lay
it out in the order a human should read it — isolated bits first, then each big
feature traced from its entry point down to the leaves, every hop anchored to
`file:line`. You do **not** decide whether the code is correct (that is
`dev-file-design-review`); you make sure the reviewer never has to reconstruct scope
or control flow themselves.

Output is a written walkthrough the user reads next to the diff. Do not modify
code.

## Pacing — one unit per round, user-gated

This is a **multi-round conversation**, not a single dump. A reviewer can only
hold one section at a time; flooding them defeats the purpose.

- **Round 1 — map only.** Present *just* Step 1 (the scope table + module
  categorization) and the proposed **overall review order**: the ordered list of
  units to come — the isolated-changes batch, then each coupled chunk by name.
  Give the table of contents, **not** the contents. Then **stop and ask** which
  unit to start with (default: the isolated batch first).
- **Round 2+ — one unit at a time.** On the user's go-ahead, deliver exactly one
  unit as a full Step 2/3 walkthrough (the isolated batch, or one coupled chunk
  entry→leaf). Then **stop**. End each round by naming what the next unit would
  be and waiting.
- **Never advance unprompted.** Do not roll into the next chunk, and do not
  pre-expand later units, until the user asks. If the user comments on the unit
  under review, engage on that before moving on.

The Step 1–3 material below defines *what each round contains*; this pacing
defines *how many rounds and when*.

## Workflow

### Step 1 — Inventory the whole changeset and categorize by module

1. **Fix the diff scope.** Establish the base explicitly and say which you used:
   - a feature branch → `git merge-base HEAD main` (compare `<base>...HEAD`);
   - review including uncommitted WIP → also fold in `git diff` / `git status`.
   Confirm with the user if the base is ambiguous.
2. **Pull the file list + churn:** `git diff --stat <base>` (or `<base>...HEAD`).
   If there are logical commits, note them (`git log --oneline <base>..HEAD`) —
   they are a hint, not necessarily the best reading order.
   Also inspect `git status --short`, the staged diff, untracked generated
   sources, and nested repositories/submodules. A working tree that passes tests
   can still produce an incomplete commit when a required file is untracked or
   lives behind another Git boundary. Check tracked binary handoffs such as
   `profiling/profile.db` for `skip-worktree`; status silence is not proof that
   their working copy equals the committed copy.
3. **Bucket every changed file by module/layer.** For VibeSim map each path to:
   - **Launcher / user entry** (`launcher/`) — CLI flags, schema, sweep,
     validation: the invocation surface the user drives.
   - **Deployment** (`simulator/src/deployment/`) — config + per-pool flow build.
   - **Simulator core, by layer** — L4 arch (`arch/`), L3 worklet (`worklet/`),
     L2 op, L1 timing/kernels (`timing/kernels/`), bridge (`timing/bridge/`),
     the eval engine; and derive macros (`*-derive/`).
   - **Analyzer** — Rust compute (`analyzer/rust/src/breakdown/…`) + Python render.
   - **Profiling / capability** (`profiling/`).
   - **Tests** (`tests/`), **docs** (`doc/`, `README*`), **skills / memory**.
4. **Emit a scope table:** module → files → churn (+/−) → one line on what that
   module's change accomplishes. This is the reviewer's map before any detail.
5. **List but set aside noise** (generated code, lockfiles, pure-format churn,
   moves) so it is transparent it was skipped, not silently dropped.

### Step 2 — Isolated changes first, then large chunks top-down

1. **Separate isolated changes from coupled chunks.**
   - *Isolated*: a hunk that stands alone — a one-line fix, an additive
     `#[serde(default)]` field, a new independent test, a doc tweak. It adds no
     new caller/callee edge into the rest of the diff. **Present these first**,
     each as one line: what + why + `file:line`. The reviewer clears them fast
     and sets them aside.
   - *Coupled chunk*: a change threaded through several files toward one
     behavior/feature (a new call path or data flow). These are the substance —
     one top-down walkthrough each.
2. **Walk each coupled chunk top-down: invocation → leaf.**
   - Start at the **entry point** — where the user or caller triggers it: the CLI
     flag, preset key, public API, or message.
   - Trace **downward** through every layer the data/control crosses: entry →
     parse/validate → plumbing/threading → the leaf that does the work. Present
     in that order, **not** alphabetical or git order — the reader should follow
     the data as it moves.
   - At each hop, name the next file/function it calls into
     (“→ continues in `engine.rs:106`”), and call out any single choke point or
     injection site.
   - Note a design decision inline where a hunk embodies one (“keyed on role
     NAME not shape, so it survives a tp/ep sweep”).
   - For a big chunk, give the **spine first** (3–5 bullets: entry → … → leaf),
     then the file-by-file detail — so the reviewer holds the shape before the
     specifics.

### Step 3 — File-by-file, line-anchored, detailed

For every isolated item and every step of every chunk, go file by file:

1. **Anchor each hunk to `path:line-range`** *and* the enclosing symbol
   (function / struct / test name) — line numbers drift as the reviewer or you
   edit, the symbol name is the stable fallback. Recompute line numbers from the
   current tree right before writing (`git diff <base> -- <file>`, then read the
   file); do not trust remembered numbers.
2. **Say what the hunk does and its role in the flow** — how it connects to the
   hunk above and below it in the walkthrough.
3. **Point at what to scrutinize** — the risky or subtle part the reviewer should
   look hardest at (an ordering, a fallback, an early return, an invariant).
4. Keep it detailed enough that the user can open the file at that line and know
   what they are looking at without re-deriving context.

### Close with a reading order + review status

- Give the recommended sequence: isolated items → chunk 1 (entry→leaf) →
  chunk 2 → … → tests → docs.
- State the validation already run (which test suites, green/red) so the reviewer
  knows the safety net that exists.
- Distinguish committed, staged, unstaged, untracked, and nested-repository work;
  verify the exact staged or archived snapshot before calling it self-contained.
- Flag open questions / intentional design deviations for explicit sign-off
  (see memory `feedback_surface_design_deviations`).

## Output

The walkthrough is delivered **across rounds** (see Pacing), not in one message:

- **Round 1 (map):** scope (base ref, file count, total churn) + module table
  (module → files → churn → purpose) + the proposed review order (isolated batch,
  then each coupled chunk by name) + validation status already run. Ends by
  asking which unit to start with. No per-file detail yet.
- **Each later round (one unit):** either the isolated-changes list (one line +
  `file:line` each) or one coupled chunk — a 3–5 bullet spine, then top-down
  file-by-file guidance with `file:line` + symbol anchors, entry point first,
  leaf last. Ends by naming the next unit and waiting.
- **Open questions / design deviations** surface in the round where they arise,
  for the user's explicit sign-off.

Do not modify code; if the user then wants commits re-sliced to match the
walkthrough, offer that as a separate step.

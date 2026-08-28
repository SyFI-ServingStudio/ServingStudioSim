---
name: dev-clean-reimplement-for-merge
description: >-
  Use when a feature is substantially complete and accepted, but its development
  branch is too iterative or messy to merge directly. Reimplement the accepted
  behavior cleanly from current master while preserving equivalence and following
  each VibeSim layer's established patterns.
---

# Clean Reimplementation For Merge

Prepare finished work for master by rebuilding it from a clean master-based
branch. Treat the development branch as behavioral evidence, not as the source
layout to preserve.

Do not use this workflow for ordinary in-progress refactoring or a small focused
fix. Use `dev-file-design-review` for one file and
`dev-present-changes-for-review` when the user only wants a guided diff review.

## 1. Freeze The Accepted Behavior

Before writing code, record what the development branch must preserve:

- user-visible behavior and supported configurations;
- public schemas, output fields, persisted data, and error behavior;
- representative commands, tests, profiles, or Analyzer results that prove it;
- explicit bug fixes or behavior changes that should differ from the old branch;
- unresolved experiments, temporary instrumentation, and abandoned approaches
  that must not be carried forward.

Use commits and diffs to discover behavior, but read the final callers, outputs,
and evidence too. Intermediate commits may contradict one another. Do not treat
the final textual diff as the specification.

Keep the original branch and artifacts intact until the clean branch passes the
equivalence checks.

## 2. Start From Current Master

Create a fresh branch and worktree from the current remote master commit. Record
the exact base SHA. Refresh the remote ref when authorized; otherwise state how
current the local ref is.

Use `dev-create-worktree` for worktree layout and required working-copy
artifacts, with one override: this workflow must branch from the verified master
SHA, not from the development branch currently checked out. Do not copy whole
source files or cherry-pick the messy feature series as a shortcut. An isolated
commit may be cherry-picked only after confirming that every hunk already has
the clean shape intended for master.

Run the relevant baseline checks on master before implementation. A pre-existing
failure must be recorded rather than attributed to the reimplementation.

## 3. Map Each Behavior To Its Owner

For each accepted behavior, identify its documented layer and the nearest
existing implementation with the same responsibility. Read the applicable
architecture/design document, owner trait or schema, registry, direct callers,
and focused tests.

Prefer a simple, locally elegant design. Use nearby implementations to discover
the layer's invariants, not as templates that must be copied. Before changing an
established construction shape, determine what its ownership, instance scope,
lifecycle, partitioning, registration, and failure semantics accomplish. A
different design is welcome when it is simpler and preserves the required
invariants; make the difference deliberate and test the semantic boundary it
changes. Do not diverge merely because those constraints were overlooked.

Build a short behavior-to-owner map before implementation. It must expose cases
where the development branch put convenience logic in the wrong layer.

## 4. Reimplement In Small Complete Units

Implement one coherent behavior at a time, from its entry point through its
owner and observable result. After every unit, perform this loop before starting
the next one:

1. **Necessity pass:** inspect every added line. Remove speculative options,
   compatibility aliases with no required caller, repeated validation, temporary
   plumbing, needless state, and abstractions used only once without simplifying
   the code.
2. **Convention pass:** compare the unit with nearby implementations in the same
   layer and identify the invariants behind their shape. Preserve those
   invariants, while allowing a cleaner structure when the difference is
   intentional, simpler, and behaviorally justified.
3. **Duplication pass:** search registries, dispatch arms, builders, kernels,
   workers, admission policies, schemas, and helpers before adding another one.
   Extend or reuse the established owner when semantics match. Never register the
   same kernel twice or maintain two admission paths for the same behavior.
4. **Test-value pass:** add or retain tests only when they protect observable
   behavior, an independently derived invariant, a real regression, integration
   wiring, or a meaningful failure path. Remove tests that merely restate an
   implementation formula, mirror private control flow, assert constants copied
   from the code, or duplicate stronger coverage elsewhere.
5. **Focused validation:** run the narrowest meaningful formatter, compile, and
   behavioral checks for that unit. Fix failures before expanding the diff.

This loop is part of implementation, not a final cleanup phase. Small units make
unnecessary code and accidental parallel implementations visible while they are
still cheap to remove.

## 5. Prove Equivalence

After all units are complete, compare the clean branch with the frozen behavior,
not merely with the old source diff.

- Run the same representative inputs on both branches where practical.
- Compare structured outputs with structured tools and use tolerances only where
  the metric is genuinely nondeterministic.
- Run focused regression tests and the broader tier selected through
  `dev-run-tests` according to the changed surface.
- Confirm that persisted measurements and generated artifacts were preserved
  intentionally rather than hidden by worktree state.
- Review the complete master-to-clean diff with
  `dev-present-changes-for-review` and apply `dev-file-design-review` to the
  highest-risk owner files.

For a performance-sensitive kernel or timing unit, textual equivalence and one
successful smoke are not sufficient performance evidence. Select a few
representative points already present in the accepted profile database, covering
the important small/medium/large or otherwise behavior-changing axes. Re-profile
those exact identities through the clean public entry point on the same GPU SKU,
backend/runtime, and timing boundary. Write every new measurement to a temporary
database, never the shared profile database. Report both values and relative
deltas; repeat enough points or trials to distinguish ordinary measurement noise
from a material regression, and investigate material differences before calling
the unit equivalent.

When delegation is available and independent implementation units remain, hand
this performance-equivalence check to a subagent as soon as the unit is runnable.
The subagent must treat source and the accepted profile database as read-only,
write only temporary measurement artifacts, and return the exact selected specs,
commands, measurements, and deltas. The main agent should continue with the next
independent unit while verification runs, but must incorporate the report before
declaring the performance-sensitive unit complete.

Every behavioral difference needs one of three dispositions: an explicit bug
fix, an approved intentional change, or a defect to fix before merge. Refactoring
alone is not a reason for changed behavior.

## Completion

The branch is ready to propose for master only when:

- the behavior-to-owner map is fully implemented;
- no accepted behavior depends on code left only in the old branch;
- duplication and test-value passes were completed per implementation unit;
- the final diff follows current master conventions and contains no temporary
  migration scaffolding without an active consumer;
- equivalence evidence and intentional differences are recorded;
- relevant tests pass, with baseline failures and unavailable tiers stated;
- commits are reviewable units that explain behavior, not the chronology of the
  original experimentation.

Report the master base SHA, clean branch/worktree, behavior equivalence results,
intentional differences, validation performed, and any remaining merge risk.

---
name: skill-of-skills
description: "Use when adding, renaming, reorganizing, or choosing among repo-local MLSim skills under main/skills. Defines the skill hierarchy, naming conventions, and parent/child relationships between top-level, orchestrator-level, and implement-level skills."
---

# Skill Of Skills

This is the map for repo-local skills. Keep these skills in `main/skills/`; the
workspace `.codex/skills` path should point here rather than becoming a separate
source of truth.

## Naming Levels

- `top-*`: top-level entry points for major user requests. A top skill may
  complete the request by sequencing lower-level skills, but should stay concise
  and route work rather than duplicate implementation details.
- `orchestrator-*`: planning and verification roles. An orchestrator classifies
  intent, writes implementer briefs, defines completion evidence, and verifies
  returned work. It should not contain detailed implementation contracts that
  belong in an implement skill.
- `impl-*`: leaf implementation roles. An implement skill owns a concrete write
  scope or investigation scope, states the files/contracts to follow, and lists
  the tests or smoke commands that prove completion.
- `operate-*`: operating roles for existing systems. An operate skill may query,
  run, profile, validate, or update existing data/artifacts, but should not add a
  new implementation contract.
- `dev-*`: developer-workflow roles. A dev skill supports repository work such
  as worktrees, tests, file review, or presenting diffs for human review.

Folder name and frontmatter `name` must match exactly.

## Current Tree

```text
top-add-kernel - add an L1 kernel end to end
├── orchestrator-add-kernel-to-python-profile - plan and verify Python profiling
│   ├── impl-explore-kernel-source - find a framework source and wrapper plan
│   └── impl-register-kernel - register a Python profiler kind or backend
└── orchestrator-wire-kernel-to-rust - plan and verify Rust timing/cache wiring
    ├── impl-wire-kernel-to-rust - implement KernelSpec and cache wiring
    └── impl-validate-kernel-cache - measure Rust cache fidelity

operate-run-simulation - run simulations from presets
operate-gpu-spec - query or update the GPU spec catalog
operate-profile-sim-speed - profile simulator wallclock speed
operate-profile-existing-kernel - query or fill registered profiler rows

dev-create-worktree - create an MLSim development worktree
dev-orchestrate-parallel-subagents - isolate concurrent writing subagents
dev-run-tests - select and run MLSim test tiers
dev-file-design-review - review one file against docs and contracts
dev-present-changes-for-review - organize a diff for human review
dev-lookup-transformers-model - locate local Transformers model code
```

## Updating The Tree

When adding or renaming a skill:

- put the skill under `main/skills/<skill-name>/SKILL.md`;
- choose the prefix by role, not by implementation language;
- update this tree if the skill becomes part of a routed workflow;
- update parent skill descriptions so agents can discover the relationship from
  metadata alone;
- run the skill validator for each changed skill;
- search for stale names with `rg "<old-skill-name>" main/skills`.

If a lower-level skill starts duplicating a parent, move the duplicated details
down to the leaf and let the parent point to it. If a top skill grows into a
checklist, split it into orchestrator and implement skills.

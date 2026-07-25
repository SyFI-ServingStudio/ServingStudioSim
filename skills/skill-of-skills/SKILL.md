---
name: skill-of-skills
description: "Use when adding, renaming, reorganizing, or choosing among repo-local VibeSim skills under main/skills. Defines the skill hierarchy, naming conventions, and parent/child relationships between top-level, orchestrator-level, and implement-level skills."
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
top-add-new-arch - add a whole model architecture end to end (explore → split → build L1–L4)
├── top-split-model-into-kernels - explore + break the forward into the kernel/op sequence
├── top-add-kernel - create each new L1 kernel (L1-Python impl-register-kernel + L1-Rust impl-wire-kernel-to-rust)
├── impl-compose-op - compose kernels into L2 ops
├── impl-compose-worklet - compose ops into L3 sync-section worklets
├── impl-compose-arch - wire worklets into the L4 model_arch cost file
└── impl-wire-new-arch - integrate the arch into dispatch/build/timing-predict/deployment (Phase 3, before Validate)

top-add-kernel - add an L1 kernel end to end
├── orchestrator-add-kernel-to-python-profile - plan and verify Python profiling
│   ├── dev-explore-kernel - search vLLM/SGLang/FlashInfer for a source to wrap (shared)
│   └── impl-register-kernel - register a Python profiler kind or backend
└── orchestrator-wire-kernel-to-rust - plan and verify Rust timing/cache wiring
    ├── impl-wire-kernel-to-rust - implement KernelSpec and cache wiring
    └── impl-validate-kernel-cache - measure Rust cache fidelity

top-explore-models - understand a new model or checkpoint from HF/public sources
├── dev-calculate-kv-cache-capacity - calculate KV cache bytes and capacity
└── dev-lookup-transformers-model - inspect local Transformers/Torch semantics

top-split-model-into-kernels - break a model forward into the VibeSim kernel sequence
├── top-explore-models - step 1: establish the architecture (entry above)
├── dev-lookup-transformers-model - what math each op computes (shared)
└── dev-explore-kernel - whether a real fused kernel exists in the ecosystem (shared)

top-align-with-framework - evaluate VibeSim↔framework alignment quality (kernel-only deviation + missing-chunk coverage, GPU duty cycle, TTFT/TPOT)
├── operate-run-alignment - run the phased alignment pipeline and label folded kernels (Step 0, below)
├── impl-validate-kernel-cache - fix a wrong-shape kernel cost surfaced by Check 1
└── top-add-kernel - add/repair a kernel whose backend the sim mismodels

top-compose-real-framework-from-sim - actively build a real serving framework from VibeSim evidence in a Tick (sim) / Tock (one measured trial) / Probe (attribute and decide) loop; the orchestrator owns the workflow and delegates only actual code writing
├── top-explore-models - establish exact checkpoint architecture and support requirements
├── top-add-new-arch - add missing VibeSim L1–L4 support before selecting a real-code trial
├── top-add-kernel - add missing measured kernel/backend support
├── operate-run-simulation - produce the comparable serving-workload target and analyzer artifacts
├── operate-run-timing-predict - compare exact fixed-shape building-block candidates
├── operate-profile-serving-run - capture and attribute a real serving profile (the Probe step)
└── dev-llm-serving - implement the frozen trial in the real framework using the routed serving reference library

operate-run-simulation - run deployment simulations from presets (DES, workload trace)
operate-run-timing-predict - offline per-building-block cost prediction (no DES; iter=PD, attn+ffn=AFD)
operate-run-alignment - run the phased measured VibeSim-to-vLLM alignment pipeline (profile, timing-predict, kernel-align, sim with auto-injected multiplier, e2e-align) and label folded kernel positions (evaluate the result via top-align-with-framework)
operate-gpu-spec - query or update the GPU spec catalog
operate-profile-sim-speed - profile simulator wallclock speed
operate-profile-serving-run - capture a comparable bounded profile of a real serving process and attribute its wall time to named engine phases (NVTX readiness + instrumentation contract, node-level CUDA-graph tracing, nsys SQLite aggregation)
operate-profile-existing-kernel - query or fill registered profiler rows

dev-create-worktree - create an VibeSim development worktree
dev-orchestrate-parallel-subagents - isolate concurrent writing subagents
dev-run-tests - select and run VibeSim test tiers
dev-file-design-review - review one file against docs and contracts
dev-present-changes-for-review - organize a diff for human review
dev-llm-serving - implement or review real LLM/multimodal serving framework code using the copied models/algorithms/backends/frameworks/hardware/engines/tooling reference library
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

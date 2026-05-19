---
name: file-design-review
description: Use when asked to check, review, tidy, or validate a specific source file against repo design docs, behavioral correctness, naming quality, future-agent-facing comments, and readable code order. Produces either a focused patch plus validation results, or doc-backed findings/questions when code changes are not justified.
---

# File Design Review

Use this skill for a focused review of one file or a small tightly-coupled set
of files. The review has five required dimensions:

1. Alignment with design docs.
2. Behavioral correctness.
3. Naming quality, especially public parameters and extension hooks.
4. Future-agent-facing comments.
5. Readable code order.

## Workflow

1. **Read the file and local context**
   - Read the target file fully.
   - Read nearby owner files when they affect the contract: package
     `__init__`, registry/barrel files, tests, trait/interface definitions, and
     direct callers/callees.
   - Use `rg`/`rg --files` first for search.

2. **Find the design contract before judging**
   - Search `README*`, `docs/`, and repo-local `AGENTS.md` when present.
   - For MLSim, map the file path to the layer in `docs/file_structure.md`,
     then read the relevant `docs/detailed_design/<layer>/design.md` section.
   - Record the concrete doc anchor that justifies each nontrivial change.
   - If no doc guideline exists for a proposed change, ask the user or label it
     explicitly as a non-doc-backed recommendation.

3. **Check design alignment**
   - Verify layer ownership: no DB work in runners, no runner imports in Rust
     kernels, no execution-backend details leaking into facade/API layers, etc.
   - Verify public API shape, data ownership, field names, registry references,
     environment/subprocess behavior, and side-effect boundaries against docs.
   - Prefer moving behavior to the documented owner over adding convenience code
     in the wrong layer.

4. **Check correctness**
   - Look for import cycles, eager imports of heavy/optional dependencies,
     validation after side effects, incorrect fallback evaluation, bad ordering
     of error checks, lossy coercion, missing edge cases, and persistence bugs.
   - Confirm failure paths preserve documented semantics.
   - Add or update focused tests when behavior changes.

5. **Check naming quality**
   - Prefer names that describe the role at the call site, not just the data
     shape. For callables, use names that reveal they are callables, such as
     `*_fn` when that matches local style.
   - Check public parameters, dataclass fields, registry hooks, generated
     symbols, and private helper names for consistency.
   - Rename vague terms when the surrounding design has a more precise owner or
     concept. Update docs/tests/callers together so terminology does not split.

6. **Check comments for future agents**
   - Add short comments where they clarify ownership, invariants, generation
     patterns, extension rules, or why an import/ordering choice avoids a trap.
   - Do not narrate obvious code.
   - Prefer module or helper comments over scattered line comments when the
     invariant applies to a whole file.

7. **Check code order**
   - Optimize for first-read flow:
     module docstring -> imports -> constants/type aliases -> data records ->
     public entry points -> public helpers -> private implementation helpers.
   - Keep the main path above supporting details.
   - Avoid interleaving public and private helpers unless it substantially
     improves locality.
   - Keep ownership groups together: schema normalization, scheduling,
     persistence, and execution boundaries should be visually separable.

8. **Patch only when justified**
   - If a fix is doc-backed and low-risk, implement it.
   - If a change is stylistic only, keep it small and avoid churn.
   - If design docs and code disagree, update the doc only when the existing
     implementation is clearly the intended contract; otherwise ask.

9. **Validate before finishing**
   - Run the narrowest meaningful checks: formatter/linter/type or compile
     checks, plus focused tests for changed behavior.
   - Run broader tests when the file is shared infra or public API.
   - Before final response, re-check completeness and correctness against all
     required dimensions.

## Output

Lead with what changed or the highest-risk findings. Include:

- Doc anchors used.
- Files changed.
- Validation commands and results.
- Any remaining non-doc-backed assumptions or risks.

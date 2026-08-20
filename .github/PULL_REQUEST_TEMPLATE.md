<!-- markdownlint-disable -->

## Purpose

<!-- What this changes and why. Link the issue it resolves, e.g. "Fixes #7". -->

## Test Plan

<!-- The exact commands. Say which tier ran and which did not — several tiers need a GPU, a
built binary, or an agent runtime, and "tests pass" is ambiguous without that. -->

## Test Result

<!-- Paste the output. For anything that changes a modeled number, show before and after. -->

---
<details>
<summary>Checklist</summary>

- [ ] **Doc-backed.** A design doc, invariant, or per-module README supports the name,
      location, API shape, and behavior — or the PR explains why it deviates.
- [ ] **Formatting radius matches the change.** `git status --short` shows no files the change
      did not intend to touch. Rust was formatted per file
      (`rustfmt --edition 2021 --config skip_children=true <file.rs>`), not with
      `cargo fmt --all`; non-Rust formatters got an explicit file list, not the repo root.
- [ ] **`profiling/profile.db`:** if it changed, rows were merged **row-wise** — never by
      copying the file over another checkout's. State which tables and how many rows were
      added, and on which GPU they were measured.
- [ ] **Unmeasured shapes still fail.** A shape, backend, or parallelism degree with no
      measured rows produces a missing/unsupported cache identity; it does not borrow a
      neighbor's timing.
- [ ] **Goldens.** Per-GPU throughput / sim-speed goldens were re-recorded if this moves them,
      and the GPU type is named.
- [ ] Docs and per-module READMEs updated where the change makes them wrong.

</details>

<!-- markdownlint-disable -->
<!-- Title: conventional commit, as the repo already does — type(scope): summary
     types  feat · fix · perf · refactor · test · docs · data · build · ci · chore
     scopes l1 · profiling · timing · worker · sim · analyzer · alignment · trace ·
            launcher · frontend · skills · <model, e.g. qwen36>
     `data(...)` is for measured rows, not code — see a178766.

     Large architectural change, or one that alters a layer contract or an entry in
     doc/invariants.md? Open an [RFC] issue first and link it here, rather than
     opening the design discussion inside a diff. -->

## Purpose

<!-- What this changes and why. Link the issue it resolves, e.g. "Fixes #7". -->

## Test Plan

<!-- The exact commands, and which tier ran: `just test-cpu` is the default gate,
     `just test-gpu` needs a CUDA device, `just test-bench` / `test-agent` are opt-in.
     Say which tiers you did NOT run — "tests pass" is ambiguous without that. -->

## Test Result

<!-- Paste the output. For anything that changes a modeled number, show before and after. -->

## Contribution licensing

<!-- These are the author's acknowledgements. They are NOT a CLA signature: this repo has
     no CLA bot and records nothing automatically. A maintainer must confirm recorded CLA
     acceptance separately before merging outside work. See CONTRIBUTING.md. -->

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here (project, URL, version, license — or "none"):

---
<details>
<summary>Checklist</summary>

- [ ] **Doc-backed — name the file.** The doc, invariant, or per-module README that
      supports this name / location / API shape / behavior:
      `<path>`. If the change deviates from it, say so here instead.
- [ ] **Formatting radius matches the change.** `git status --short` shows no files the
      change did not intend to touch. Rust was formatted per file
      (`rustfmt --edition 2021 --config skip_children=true <file.rs>`), not with
      `cargo fmt --all`; non-Rust formatters got an explicit file list, not the repo root.
- [ ] **`profiling/profile.db`:** if it changed, rows were merged **row-wise** — never by
      copying the file over another checkout's. State which tables, how many rows, and on
      which GPU they were measured.
- [ ] **Unmeasured shapes still fail.** A shape, backend, or parallelism degree with no
      measured rows produces a missing/unsupported cache identity; it does not borrow a
      neighbor's timing.
- [ ] **Goldens.** Per-GPU throughput / sim-speed goldens were re-recorded
      (`just update-golden`) if this moves them, and the GPU type is named.

Strike through anything genuinely N/A rather than ticking it — a row of blind checkmarks
tells a reviewer less than an honest `~~n/a, docs only~~`.

</details>

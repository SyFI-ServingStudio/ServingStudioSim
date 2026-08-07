# `alignment/labeling` — the manual step, made checkable

Every alignment phase is a command except one. Between `alignment parse` and
`analyze kernel-align` a person decides, for each measured kernel, which modelled
operation it is. That decision sets the quality of every number downstream, and
both ways of getting it wrong are silent:

* an **unlabeled** kernel is time the comparison never sees — it lands in
  `unmapped_measured_kernels` and shows up only as a coverage percentage;
* a **mislabeled** kernel is time charged to the wrong operation — it produces a
  deviation that reads exactly like a cost-model bug. One CUTLASS tile in this
  repository's GLM-5.2 run is `attention.q_absorb` inside a decoder layer and
  the `lm_head` in postprocess; labeled by name alone it inflated q_absorb's
  measured time by 88 % and manufactured a "the model under-predicts by 2×"
  finding that did not exist.

This package is the machinery that makes both visible.

```bash
python -m alignment label coverage <analysis_kernel/reports/alignment_iteration_report.json>
python -m alignment label walk     <kernel_sequences_labeled.json> [--phase forward] [--name FRAGMENT] [--unmapped]
python -m alignment label slots    <timing_predict/raw/cost_manifest/worker_predict_0.json> <slot prefix>
python -m alignment label check    <kernel_sequences_labeled.json>
python -m alignment label apply    <kernel_sequences_labeled.json> <cost manifest> <rules.json> [--dry-run]
python -m alignment label transfer <labeled inventory> <re-parsed inventory> <output> [--dry-run]
```

`coverage` needs an analyzer report and so runs after `analyze kernel-align`;
the rest read only the inventory and the cost manifest. The loop is: read
coverage, walk the positions the unmapped time sits at, write rules, apply,
re-analyze.

## The four views

**`coverage`** pairs the two unmapped lists. The analyzer already reports both,
but it reports measured kernels one row per folded position, so a kernel that
runs at 225 positions never shows its total anywhere. Aggregated by name and
printed beside the unmapped simulated slots, an unlabeled subsystem is obvious:
its kernels and its slots are both at the top of their lists, describing the
same work.

**`walk`** flattens the folded program — phase → sequence → segment → repeat
body — into program order, carrying each position's folded coordinates and its
two neighbours. It is what a labeling decision is actually made against.

**`slots`** prints one layer's simulated slots in compile order, which is the
other half of the same read: a decoder layer's measured kernels and its slots
are the same list twice.

**`check`** is the mechanical half of review. Three findings, each one a defect
that reached a published number at least once:

| finding | why it matters |
| --- | --- |
| a name whose two operations are separated only by position | nothing in the file states the distinction; one role may be charged the other's time |
| one name, one operation, positions in two phases | the lm_head signature above |
| one operation with two different label bodies | the launcher rejects the file but names only the operation, not the two labels that disagree |
| a label with no `cross_rank` | the analyzer defaults it to independent silently, and the omission surfaces far downstream as a schema error in the UI |

A kernel with no `label` key at all is *not* a finding — that is what a fresh
inventory out of `alignment parse` looks like, and `coverage` is where it shows.

## Rules

`apply` runs a JSON rule file over the inventory, so a labeling pass is a
reviewable artifact rather than a script that ran once:

```json
{"rules": [
  {"name": "nvjet_tst_256x8_64x6_2x1_v_bz_TNT", "operation": "main.lm_head",
   "type": "ffn", "role": "lm_head", "slot_suffixes": ["lm_head"],
   "phase": "postprocess", "overwrite": true,
   "note": "the same tile is q_absorb inside a decoder layer"}
]}
```

A rule matches on the three evidence keys `operate-run-alignment` allows a name
to be disambiguated by, and no others: `name` (a fragment, always required),
`after` (the nearest mapped operation before it in the same segment body),
`after_name` (the immediately preceding kernel's name, for where that neighbour
is itself unlabeled), and `phase`. Rules are tried in file order and the first
match wins. A rule never overwrites an existing mapping unless it says
`overwrite`, so the order of a rule file cannot silently change an earlier
decision; a position already carrying the rule's own operation is reported as
*confirmed*.

Slots are claimed by suffix because one logical slot exists once per layer
variant (dense, initial-shared, cycle-full-index, cycle-shared) and a measured
kernel covers all of them. Suffixes resolve against the run's own cost manifest,
so a rule naming a slot the model no longer emits fails loudly.

Re-applying a rule file to the inventory it produced is the file's own test:
`logs/20260803_8_glm52_fp8_dp8_ep8_alignment_ctx8k_out1k_c64/labeling_rules.json`
re-runs to 241 confirmations, 28 identical rewrites, no conflicts, and a
byte-identical inventory.

## Transfer

Re-parsing a capture — after a parser fix, or with a different window — rebuilds
the inventory from the same kernels, so a finished labeling pass is still
correct; only the occurrence bookkeeping (which device, which step) changes.
`transfer` moves the labels across, and refuses unless the two documents are the
*same program*: identical phases, sequence ids, segment structure, kernel names
and categories at every position. Structural equality, never similarity — a
fuzzy transfer charges one kernel's label to another, which is the exact failure
this package exists to prevent. When they differ it names the first position
that disagrees.

The GLM-5.2 wall-clock re-parse moved all 1,232 positions with zero structural
differences, which is also the check that the re-parse changed only bookkeeping.

## What this package will not do

It will not guess. Bookkeeping, allocation, fill, copy, launch prep and sampling
stay unmapped, and a measured subsystem with no simulated slot at all — the MoE
dispatch preparation kernels in the GLM-5.2 run, 749 ms of it — stays unmapped
too, because that is a gap in the cost model and hiding it inside a neighbouring
operation's label would make it disappear from both sides of the comparison.

# Trainium2 database publication

The tracked `profiling/profile.db` includes 122 existing stock-model measurements
under the hardware key `AWS Trainium2 LNC2`, using the same SQLite database and
row-aware merge workflow as other hardware. This publication adds no new
hardware measurements and does not convert per-op trace activity into cost rows.

| Table | Backend | Rows | Measured max-model-length values |
| --- | --- | ---: | --- |
| `neuron_llama_forward` | `vllm_neuron` | 37 | 512, 1024, 2048, 4096, 8192 |
| `neuron_llama_region` | `vllm_neuron_fx_regions` | 42 | 512, 1024, 2048, 8192 |
| `neuron_llama_segmented_forward` | `vllm_neuron_segmented` | 31 | 16384, 32768 (segment size 2048) |
| `neuron_llama_segment` | `vllm_neuron_collective_segments` | 12 | 512 |

These are measured subsets, not complete Cartesian grids. Consumers still need
measurements for missing shapes. The standalone NKI tables introduced in PR #105
are intentionally excluded: their historical JSON evidence is not promoted to
shared cost rows while the numerical-validation and production-shape review
limitations remain unresolved.

## Sources and preservation

The canonical context-ladder database supplied the first three tables. A copy
of the collective-segment database supplied only `neuron_llama_segment`; its
older whole-forward and region reruns were excluded in favor of the ladder.
The source databases were preserved. SHA-256 hashes of the merge inputs:

- Tracked baseline at `7f54832ba36fc71d1de8243997f92572d5296432`:
  `b9ddf978ee56d5bacca9011e6c94f3172f9c7b08893042e7d80c535b9ce4c272`.
- Context-ladder SQLite backup:
  `1beac2067588742bd2df9dab2b3254836fc907bd189a0cbff44f78ecd86582bb`.
- Segment-only SQLite copy:
  `85ca331b26ab172373541f446ee0dc4d636d664588904fc9771529adc09de1bf`.

Both imports used `launcher kernel-profile merge-db`. Original timestamps,
profiler revisions, runtime metadata, verification flags and measurement values
are retained; publication does not restamp old measurements with the import
commit or upgrade their verification flags. The original profiler revisions
refer to the development history, rather than the cleaned PR's ancestry.

## Validation

- Both row-aware merges reported zero conflicts.
- SQLite `PRAGMA integrity_check` returned `ok`.
- SQL set comparisons verified that every existing row in all 82 baseline
  tables was preserved, including all 166,922 existing measurement rows.
- All 122 added timings are positive and have `is_outlier = 0`.
- Read-only public `launcher kernel-profile query` calls using every imported
  spec returned 122 successful results and zero missing rows across the four
  table/backend pairs.

Query validation used `--gpu-name 'AWS Trainium2 LNC2' --db <candidate.db>
--specs <table-specs.json> --json`. No profiling or device allocation was needed
for this publication. The applicable workflow is
`skills/operate-profile-existing-kernel/SKILL.md`; database behavior is documented
in `profiling/README.md` and `doc/detailed_design/L1.md`.

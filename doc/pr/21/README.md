# PR 21 alignment figures

This directory contains a small, review-oriented subset of the Analyzer output
used to validate the GLM-5.2 NVFP4 B200 model wiring. The full generated
experiment trees remain untracked.

| Figure | Evidence |
|---|---|
| `kernel_alignment_case14.png` | Per-iteration kernel timing for case 14, near-full long decode at concurrency 21 |
| `workload_alignment_case14.png` | Full-trace scheduled-KV workload for the same case, comparing vLLM and ServingStudio Sim |
| `e2e_cdf_case15.png` | End-to-end request-latency CDF for case 15, KV-overcommit mixed churn at concurrency 14 |

The figures were copied without modification from
`logs/20260827_1_glm52_nvfp4_long_context_alignment/cases/`.

`alignment_matrix.md` holds the full fifteen-case table those figures are drawn
from. It is generated, not transcribed — regenerate it with:

```bash
uv run python -m launcher alignment-campaign extract \
  --pack presets/alignment/glm52_nvfp4_b200 --runs <case_roots> --out /tmp/metrics.json
uv run python -m launcher alignment-campaign compare \
  --pack presets/alignment/glm52_nvfp4_b200 --measured /tmp/metrics.json --markdown
```

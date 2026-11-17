# trn2-config-search (GLM-5.3-Flash FP8, InferenceX 8k1k)

Simulator-only search over the glm5-trn2 seed engine's layouts on one trn2.48xlarge (two replicas of
32 LNC2 ranks), at InferenceX 8k1k fixed concurrency C in {4, ..., 256}. Results and the answer:
`doc/pr/trn2-glm53-flash/CONFIG_SEARCH.md`.

```bash
# every feasible (attn_tp, chunk, dmax) at layer regions; what-ifs (block regions, host scale)
uv run --no-sync python tools/trn2-config-search/search.py --stage main --out <dir>
uv run --no-sync python tools/trn2-config-search/search.py --stage whatif --out <dir>
# one run as an InferenceX point (+ the per-kind step breakdown)
uv run --no-sync python tools/trn2-config-search/inferencex.py <run_dir> --conc 16 --breakdown
```

- `search.py`: builds one launcher sweep (compound `point` = layout x C) from
  `presets/trn2/glm53_flash_fp8_trn2_8k1k.yaml`, drops layouts whose FP8 weights + preallocated
  pools exceed a core (`memory_gb`, the arch's HBM check), runs it with `--no-gpu --no-analyze`,
  and writes `results_<stage>.json` (one InferenceX point per run) and `breakdown_<stage>.json`
  (the Pareto front's step breakdowns).
- `inferencex.py`: drops the first 2C completions (warm-up); tok/s per chip over the measured
  window, interactivity = 1000 / median TPOT, median TTFT and E2E.

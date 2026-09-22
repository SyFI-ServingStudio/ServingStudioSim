# slime fixed-B sweep — measured-vs-simulated

Everything needed to reproduce the comparison between `presets/slime_B_sweep_sgl_fa3.yaml`
and the recorded slime arms. Findings live in the worktree's `progress.md`;
this directory is only the plumbing.

## Layout

| | |
| --- | --- |
| `extract_engine_windows.py` | **runs on coriander** — per-engine `inference` windows for every (arm, rollout), from each arm's `trace.json`. The ground truth for both metrics. |
| `extract_firings.py` | **runs on coriander** — every threshold firing with its timestamp and train group, from `run.log.gz`. Gives both the firing instant and the abort loop's duration. |
| `extract_decode_batch.py` | **runs on coriander** — SGLang's own `Decode batch` telemetry for one arm: running requests, batch KV, generation throughput. |
| `extract_inference_times.py` | **runs on coriander** — per-rollout `inference_time_s` from each arm's `report.json`. Redundant with the engine windows; kept because it is one file read instead of a 4 MB trace parse. |
| `find_alignment_multipliers.py` | scans an alignment run tree for `recommended_gpu_time_multiplier`. Used to establish that llama3-8b TP1 never had one derived. |
| `analyze_sweep.py` | runs locally — the whole comparison and all three figures. |
| `data/` | the extracted measured data, so the analysis reruns without touching coriander. |

## Reproducing

```bash
# the extractors run where the arms are; land them in our own worktree there,
# not in /tmp, so a rerun months from now finds the same code next to the data.
REMOTE=/m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep

# once, or whenever the measured arms change
for s in extract_engine_windows extract_firings; do
  scp tools/slime-b-sweep/$s.py coriander:$REMOTE/
done
ssh coriander 'python3 $REMOTE/extract_engine_windows.py' > tools/slime-b-sweep/data/all_arms.csv
ssh coriander 'python3 $REMOTE/extract_firings.py'        > tools/slime-b-sweep/data/fires.csv

# per simulated sweep
uv run python -m launcher presets/slime_B_sweep_sgl_fa3.yaml
uv run python tools/slime-b-sweep/analyze_sweep.py logs/<sweep_dir>
```

`analyze_sweep.py` refuses a cell whose parquet is short of 1024 rows. That
check is not paranoia: a Rust panic leaves a partial parquet and no
`summary.json`, and the truncated cell reads as a *fast* one — the
double-migration panic showed up as a rollout that finished 30% early, which
looks like a result rather than a crash.

## Two things the measured data will mislead you about

**Engine windows end at idle, not at release.** An engine's `inference` slice
stops when that engine runs dry; its GPU is not handed back to training until
its train-group partner is also done. Compare against the simulator's per-worker
*activity* windows, not against block release times — the latter inflates the
low-B arms by about 9% and manufactures a systematic error that is not there.

**`gen throughput` is an interval average.** The same log line's
`#running-req` and `#token` are instantaneous. Matching them per point gives
r = 0.54; bucket by `#token` and compare medians instead. This data can pin the
*level* of the cost model and cannot settle whether framework overhead is
constant per iteration or proportional to GPU time.

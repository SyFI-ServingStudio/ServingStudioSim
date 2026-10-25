---
name: operate-tune-trn2-with-engine
description: Use when asked to tune or validate serving configs for Qwen3.8-Flash-Next BF16 on a trn2.48xlarge against the real flashnext-neuron engine - pick configs from the simulator's config search, record blind predictions, run the engine with the same knobs, and check predicted vs measured (tools/trn2-tune-loop). For a simulator-only search use operate-search-trn2-configs; for re-fitting the arch to engine logs use tools/trn2-engine-align.
---

# Tune Trainium2 serving configs with the engine in the loop

The tool is `tools/trn2-tune-loop/` (its README has the steps, files and knob map). The
first loop's results, the headroom of the best measured config and the guard rails'
evidence are in `doc/pr/trn2-flashnext/TUNE_LOOP.md`; the alignment it rests on is
`ALIGNMENT.md`, the search it draws from `CONFIG_SEARCH.md`. Read all three first.

## Rules

- **Predict before you run.** `run` refuses without `predictions.json`. Commit the
  predictions (or copy them under `doc/pr/trn2-flashnext/tune/`) before the first engine
  run, so the comparison stays blind.
- **The device runs only through the engine's docker scripts** (`scripts/slice_run.sh`,
  which `run` writes for you): one container per 32-core replica, comm ports 61000 /
  63000 (keep >= 1000 apart from any other process group), the launcher's
  `NEURON_RT_DISABLE_EXECUTION_BARRIER=1`. Check `sudo docker ps` for `slice-*` first;
  `run --execute` refuses while one is up. After killing a tmux session also
  `sudo docker kill slice-<port>`.
- Run `run --execute` in a tmux session named `tune-<name>` and record session, command
  and log in the engine's `progress.md`.
- **Compiles are long** (block granularity: 10-15 min per prefill shape; a new layout or
  dispatch recompiles everything, ~0.5-1.5 h). The persistent cache
  (`~/neuron-cache/libtorch` + `nki-intermediate-cache`) is shared; order candidates so
  shapes are reused (same layout and granularity first), but do not drop a candidate
  only because it compiles.
- Never change engine code to make a candidate run unless it is a config-only change
  (a flag, a default); document it in the engine's `notes.md`.

## Guard rails

- **Re-align when the verdict is `realign`**: an error above ~15% on duration, out
  tok/s or TPOT p50, or the duration ranking flipping on a pair more than 3% apart.
  Add the new runs to `tools/trn2-engine-align/engine_align.RUNS`, run `align.py replay`
  and `e2e`, fix the composition, then search again. TTFT p50 on W64 only warns: the
  Sim runs ~5-15% low on it (first-use stalls).
- **Sim-only knobs** cannot be validated: `step_forwards fused`, `spmd_padding false`
  (idle-DP padding skip), `max_prefill_chunks_per_partition > 1`, `kind_time_scale` (a
  faster `moe_cte`), `host_time_scale`, `moe_a2a_capacity_factor` without drop-safe
  routing, P/D, 64-core replicas, EP < 32, `gdn_tp != attn_tp`, any `*_impls` list. They
  are what-ifs for engine work, not configs. `propose` refuses them.
- **Not modeled**: a chunk budget above the largest prefill bucket when prompts are
  longer than that bucket (the engine caps chunks at the bucket, the Sim does not).
  `propose` refuses it; a budget above the bucket with prompts that fit one bucket
  (W64 at 4096 with a 2048 bucket) is fine.
- **Engine HBM**: block prefill over 8192 gathered MoE rows (attn_dp x largest prefill
  bucket) fails NEFF load with `Allocation Failure`. The Sim's HBM check does not see
  it, and `propose` refuses it.
- **Uncertain inputs.**
  - **MTP acceptance** is an input.
    - Engine k=1 rates: 0.65-0.72 on random prompts, 0.87 on chat prompts.
    - A uniform `--accept-rate` predicts W64 10-15% short, because duration follows the
      slowest requests.
    - Also predict a second set with `--tag <t> --accept-from <earlier k-matched engine
      runs>` (per-request draws; use the median of ~3 draws).
    - `compare --resim` re-simulates with each run's own acceptance, which separates
      input error from model error.
  - **The all-to-all cost** (`moe_a2a_pack_us_per_row` 0.8) fits dp4 x T=2048 but is
    ~3.4x too low at dp8 x T=512 (`TUNE_LOOP.md`). Treat every a2a prediction as
    untrusted until it is re-fitted.
- **First-use stalls**: the engine's first 3 steps after warm-up stall 3 s at TP8 and
  10 s at TP4, which the Sim does not model. `compare` gates on durations without them
  and reports the raw ranking beside it; say so when the two differ.
- Run-to-run noise on the engine is ~3-5% (host load); differences below that are
  ties, not rankings. Two replicas of one run differ by 2-5% in duration.
- Do not run Sim batches or tests while an engine run is in its timed phase (after
  `warm-up` in the serve log): host load moves decode steps 5-10%.
- tmux `has-session -t name` matches prefixes; use `-t "=name"` in wait loops.

## Steps

From the Sim worktree root:

1. `source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH` and
   `T="uv run --no-sync python tools/trn2-tune-loop/tune.py"`.
2. Search or reuse a search (`operate-search-trn2-configs`); `collect` must have
   written `results.jsonl`.
3. `$T propose <name> --exp <search exp> --workload <label> --top N [--variant
   'label:knob=v,...'] [--contrast 'label:knob=v,...'] [--accept-rate A]`. Use the MTP
   workload label (`fixed_2k_512-a0.67`) for MTP rows. Include one contrast the Sim
   predicts to lose, so the ranking is tested, not only the winner.
4. `$T predict <name>` and `$T predict <name> --tag acc --accept-from <earlier runs>`;
   read both `predictions*.json`; commit them (copy under `doc/pr/trn2-flashnext/tune/`).
5. `$T run <name>` and read `engine/<label>.sh`; then in tmux `tune-<name>`:
   `$T run <name> --execute` (one candidate at a time; rerun resumes).
6. `$T compare <name> --resim` and `$T compare <name> --tag acc`; read `comparison*.md`.
   A failed candidate (see its `logs/engine/serve_<run>_r0.log`) is a finding: add a
   guard to `engine_blockers` when it is an engine limit.
7. Record the table, the verdict and the ranking in `TUNE_LOOP.md` (or a new dated
   section), and the best measured config's R0-R7 ladder from `predictions.json`
   with the top optimization targets: `$T headroom <name> <best> --accept-from <its own
   run files> --tag meas` (R0-R7 ladder + stage-4 what-ifs on its exact knobs).

## Done when

Every candidate is `finished` in `runs.json` or listed with why it failed, the
comparison and verdict are recorded, and a `realign` verdict has either been acted on
(alignment re-run) or is stated as the open issue.

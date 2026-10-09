# Stock Trainium2 alignment

This pack reproduces one full 32-layer Llama 3.1 8B BF16 stock vLLM Neuron
configuration: TP4 on one physical Trainium2 chip with four LNC2 units,
context 512, decode buckets `[1,16]`, 16 resident requests, and 6,782 KV blocks
of 32 tokens. It preserves the existing chunked-prefill worker, separate
prefill priority and disabled prefix cache. CUDA and custom NxDI are separate
engines.

The native and clean passes each run sixteen simultaneous 504+8 requests. The
pool has eight 9,973-token segments; req-frontend's circular ordinal offset
selects the same eight accepted museum prompts twice. Only each segment's
first 504 tokens are consumed. Neutral filler is excluded from corpus claims.
The pack stores pool/tokenizer hashes and trace invariants. Generated request
IDs include the case slug; their prompt tokens and order match the accepted
capture. The engine's corpus preflight compares every prompt hash. There is
no frontend warmup; stock startup warms the configured graphs before capture.

## Host configuration and rendering

Copy `presets/alignment/hosts/example.yaml` to an ignored local host profile.
Set `checkpoints.llama31_8b` to the local checkpoint, `text_corpus` to the pool
generated below, `device_roles.primary` to one physical chip index, and the
port according to the workspace's port convention. Set `fork_python` empty;
stock execution uses the pinned image. Add this host-only mapping using your
local paths and the image digest recorded in `campaign.yaml`:

```yaml
neuron_server:
  cache_path: /path/to/accepted-stock-cache
  image: sha256:<immutable-image-digest>
  docker_host: unix:///path/to/private/docker.sock
  req_frontend_binary: /path/to/session_runner
  accepted_forward_path: /path/to/accepted-forward-run
  docker_command: [sudo, -n, docker]
  python_executable: /opt/conda/bin/python
```

The image, checkpoint, compiled cache, private socket and accepted accuracy
provenance must already be available locally. Fresh captures require the forward
producer's `binary-provenance.json`, verify pinned checkpoint values once before
startup, and bind warm-loaded NEFF bytes to the numerical run. Historical captures
without byte receipts retain graph-key-only provenance. See the stock adapter's
[migration recipe](../../../alignment/neuron/README.md#stock-vllm-neuron-tp4).
The HTTP capture does not
produce a new full-logit accuracy validation. Host profiles and generated
traces remain ignored; the pack carries no host paths. `gpu_memory_utilization`
and `capture_seconds` are required generic case metadata; stock execution
uses its fixed KV block count and bounded request completion.

From the simulator repository, after sourcing the workspace `.env`:

```bash
uv run python -m alignment.neuron.vllm_corpus --model /path/to/checkpoint --output /path/to/corpus
uv run python -m launcher alignment-campaign check --pack llama31_8b_bf16_trainium2_tp4
uv run python -m launcher alignment-campaign render --pack llama31_8b_bf16_trainium2_tp4 --host presets/alignment/hosts/trainium2_stock.local.yaml --out-root logs/<new-run>
uv run python -m launcher alignment profile logs/<new-run>/01_stock_16x504_8/profile_native.yaml --dry-run
uv run python -m launcher alignment profile logs/<new-run>/01_stock_16x504_8/profile_clean.yaml --dry-run
```

Static pack checking accepts nonexistent host paths for portability. Actual
profile preflight checks the local runtime paths. Each subsequent campaign
invocation runs one phase: `profile_native`, `profile_clean`, `timing_predict`,
`analysis_kernel`, `simulation`, then `analysis_e2e`, with labeling after
timing-predict. Inspect each completed phase before the next. No command pushes
artifacts or requires publishing a model.

The canonical input builder is `engine_text`, using the `vllm_neuron_text`
capture adapter and the stock whole-forward architecture. One labeled forward
owns all layers, collectives, head and sampling; the native timing uses the
physical chip/core union, with no TP-rank sum or layer multiplier. Repeated
graph occurrences retain the exact repeat folding policy.

## Separate duty calibration

The rendered baseline always uses `gpu_time_multiplier: 1.0`. The campaign
does not inject calibration. The previously adopted factor
`1.0619761644688026` came from the original native duty-cycle report before
the fresh server timing comparison; `calibration.json` binds its exact report,
hash and field. It accounts for measured gaps and is not a kernel timing refit.
Keep that experiment separate with the existing launcher's explicit overrides:

```bash
uv run python -m launcher logs/<new-run>/01_stock_16x504_8/simulation.yaml --build-type release --override pools.main.groups.0.worker.gpu_time_multiplier=1.0619761644688026 --override io.log_dir=logs/<new-run>/01_stock_16x504_8/calibrated/simulation --dry-run
```

Remove only `--dry-run` to execute after inspecting the plan. Analyze that
simulation in a separate E2E config with the same clean capture. Preserve the
neutral simulation and all raw captures. No new factor is derived from E2E.

## Saved evidence judgment

The policy was selected **after** the saved evidence. It copies every generic
default from `glm53_flash_fp8_b200_tp4_ep4/acceptance.yaml` unchanged, with no
case exceptions. This is not prospective preregistration. Original pack-less
UNJUDGED comparisons remain intact; fresh extractions and policy comparisons
belong in a separate output directory. For each existing run root:

```bash
uv run python -m launcher alignment-campaign extract --runs logs/20261009_2_trainium_stock_server_timings --out /path/to/review/original-metrics.json
uv run python -m launcher alignment-campaign compare --pack llama31_8b_bf16_trainium2_tp4 --measured /path/to/review/original-metrics.json --json
uv run python -m launcher alignment-campaign extract --runs logs/20261009_2_trainium_stock_server_timings/calibrated --out /path/to/review/calibrated-metrics.json
uv run python -m launcher alignment-campaign compare --pack llama31_8b_bf16_trainium2_tp4 --measured /path/to/review/calibrated-metrics.json --json
```

Pack-less extraction retains the historical run names; comparison supplies this
pack's default policy without relabeling those runs or recording a golden.
All thirteen metrics are available. The neutral baseline **fails** E2E
`-7.9327%` against `7%`, and output throughput `+8.6690%` against `8%`.
Its kernel mean absolute error is `0.7461%`, coverage `100%`, server TTFT error
`-2.0760%`, and server TPOT error `-4.2137%`.

The previously duty-calibrated simulation passes all thirteen inherited limits:
E2E `-2.2056%`, output throughput `+2.3050%`, server TTFT `+4.0602%`, server
TPOT `+1.7237%` and iteration cycle `+5.7745%`. Kernel timing and coverage use
the same native capture. This bounded judgment does not remove the retained
R6/R7 partition warning, certify HTTP full-logit accuracy, or validate other
batch/context configurations or additive component composition.

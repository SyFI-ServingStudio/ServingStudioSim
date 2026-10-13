"""Write campaign.yaml for Qwen3.8-Flash-Next-NVFP4 (MTP k=3) on one B200 (TP1) under the vLLM fork 892da082.

The twenty operating points of `qwen38_27b_nvfp4_b200_tp1` keep their index, purpose and coverage: the fifteen
GLM-5.2 points plus cases 04/07 at chunk 4096 and 8192 and case 10 at chunk 8192. Concurrency and lengths are
re-derived from this model's limits below (`c x mean terminal blocks / KV pool` for the KV cases, the knee as
0.8 x the case-10 saturated rate, lengths as fractions of the context limit). Every case states its derivation in
`provenance.derivation`; `CASES` below is the one place it is computed.

Serving config (user decision, after the vLLM recipe for Qwen3.8-Flash-Next and NVIDIA's B200 recipe):

- One B200, TP1, no EP/DP, `--language-model-only`, ModelOpt MIXED_PRECISION (`--quantization modelopt_mixed`: the
  fork resolves the checkpoint's `quant_algo: MIXED_PRECISION` to it whatever `--quantization modelopt*` is passed,
  vllm/config/model.py:1381-1402 + vllm/model_executor/layers/quantization/modelopt.py:1568-1574; naming it keeps
  the launch record literal). Routed experts NVFP4 (FlashInfer TRT-LLM MoE), MTP experts 128x128 block FP8,
  everything else BF16.
- PLE n-gram table on the host: `--engram-config '{"cpu_offload": true}'` (~51 GB pinned host memory per server).
- Native MTP, `{"method":"mtp","num_speculative_tokens":3}` (recipe default): one MTP layer run three times
  (vLLM's single-module AutoRegressiveSpeculator). Every case drafts 3.
- FP8 KV (`--kv-cache-dtype fp8`), FP8 QSA indexer cache (`--attention-config '{"indexer_kv_dtype":"fp8"}'`; the
  fork's Qwen4Exp indexer defaults to bf16, vllm/models/qwen4_exp/nvidia/indexer_qsa.py:151), FP32 GDN SSM state,
  prefix caching, gpu_memory_utilization 0.90, max_model_len 131072 (diverse_100's longest request is 131071).
- `--max-num-seqs 256` (the recipe's value). vLLM's B200 default 1024 does not start (smoke jobs 6579, 6595): Model
  Runner V2's CUDA-graph memory profiling allocates a minimal KV cache of min(max_num_seqs, max capture size) blocks
  (vllm/v1/worker/gpu/cudagraph_utils.py `_init_minimal_kv_cache_for_profiling`); with this model's ~44 MB blocks
  that is ~43 GiB, which is still held when the real KV cache is allocated (118.65 GiB allocated at the 77.29 GiB KV
  allocation -> OOM). At 256 the start succeeds (job 6600). No case offers more than 256 concurrent requests.
- FlashInfer autotune stays on (vLLM default). The recipe's `--no-enable-flashinfer-autotune` belongs to its
  TP8 + EP MTP command; at TP1 the autotuner runs and caches, and a pass logging "No tuned config" is discarded.
- `api_server_count: 4`; Model Runner V2 pinned by `VLLM_USE_V2_MODEL_RUNNER=1` (already vLLM's choice for this
  config, vllm/config/vllm.py:697-746; routed-expert capture exists only in V2's AuxOutput connector).
- CUDA graphs FULL_AND_PIECEWISE with `max_cudagraph_capture_size` = the chunk and an explicit list:
  vLLM's own sizes up to 2048 ([1, 2, 4], 8..248 step 8, 256..2048 step 16 = 147) then every 128 up to the chunk
  (162 / 194 sizes at chunk 4096 / 8192). Under Model Runner V2 the PIECEWISE (mixed) graphs use the list as
  given; FULL decode graphs round each size up to a multiple of the verify width 4 and keep those <= the chunk
  and <= max_num_seqs x 4 (vllm/v1/worker/gpu/cudagraph_utils.py:248-330). V1's
  `adjust_cudagraph_sizes_for_spec_decode` rounding does not run under V2 (vllm/config/compilation.py:1492-1510).
- Passes run req-frontend's warmup, and the job wrapper points every JIT cache at one persistent directory.

ARCH_TYPE is a placeholder: the simulator arch for this model is not named yet.

Run from the repo root: uv run python presets/alignment/qwen38_flash_next_nvfp4_b200_tp1/make_pack.py
"""
import json
import math
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- Qwen3.8-Flash-Next limits
L = 131072
CHUNK = 2048
DRAFT = 3
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
GPU_MEMORY_UTILIZATION = 0.90


def capture_sizes(chunk: int) -> list[int]:
    """vLLM's own capture list up to 2048, then one size every 128 up to the chunk (skills/operate-run-alignment)."""
    sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, min(chunk, 2048) + 1, 16))
    return sizes + list(range(2048 + 128, chunk + 1, 128))


# ---------------------------------------------------------------- KV pool
# Measured at chunk 2048 (smoke job 6600, logs/20261013_0_qf_smoke/workload_{cold,warm}): `Available KV cache memory:
# 78.57 GiB`, vLLM metric cache_config_info num_gpu_blocks 1920, block_size 3200 ("Setting attention block size to 3200
# tokens": the GDN state page with 3 draft rows of conv state, 3,268,608 B, padded 0.25% to 3200 x 1024 B FP8 K/V per
# token), mamba_cache_mode align, and `Maximum concurrency for 131,072 tokens per request: 30.97x` = 62 blocks per
# max-length request (1920 / 62 = 30.97).
# blocks() splits those 62 as ceil(T/3200) attention blocks (41) + ceil(T/12800) QSA compressed-key blocks (compress
# ratio 4: 11) + 10 constant state blocks (GDN groups and PLE conv state in align mode). That split is INFERRED to fit
# the one logged figure; check it against case 01's running-request count before relying on short-request demand.
ATTN_BLOCK_TOKENS = 3200
COMPRESS_RATIO = 4
STATE_BLOCKS_PER_REQUEST = 10
POOL_BLOCKS_2048 = 1920
GIB = 1 << 30
BLOCK_BYTES = round(78.57 * GIB / POOL_BLOCKS_2048)  # ~43.94 MB; the logged GiB is rounded to 0.01

# Per-GPU KV pool (`Available KV cache memory`, GiB) logged by each chunk size's warm workload passes:
# chunk -> (GiB, Slurm jobs, evidence dir). None until measured; the chunk-2048 smoke pool stands in.
# The pool is not a pure function of the arguments: with identical argv, smoke 6600 logged 78.57 GiB (77.60 GiB
# weights + non-torch, 3.39 GiB graph pool) and diverse_100 6613/6614 logged 79.80 GiB (76.37 + 2.78). Each case's
# own workload pass is the authority for its pool; these values only size the case designs.
SMOKE_POOL = (78.57, "6600 (smoke, cold and warm start alike)", "logs/20261013_0_qf_smoke/workload_warm")
MEASURED_POOL: dict[int, tuple[float, str, str] | None] = {
    2048: (79.80, "6613, 6614 (diverse_100 nsys and workload)", "logs/20261013_1_qf_diverse100/profile_workload"),
    4096: (77.84, "6623 (case 16 workload)", "logs/20261013_2_qf_cases/16_prefill_spectrum_chunk4k_c32/profile_workload"),
    8192: (73.76, "6624 (case 17 workload)", "logs/20261013_2_qf_cases/17_prefill_spectrum_chunk8k_c32/profile_workload"),
}

# MTP acceptance per draft position (conditional: position 0 / drafts, then i / i-1) from the smoke's warm workload
# pass; every case carries it as `provisional` until its own workload pass measures it.
PROVISIONAL_ACCEPTANCE = (
    [0.7322, 0.6677, 0.6862],
    "conditional accepted-token ratios of the smoke's warm workload pass (24 enwik9 requests, c8; per position "
    "2639/3604, 1762/2639, 1209/1762); replace with this case's workload pass",
    "logs/20261013_0_qf_smoke/workload_warm/spec_decode_metrics.json",
)

# Conditional MTP acceptance per draft position (position 0 / drafts, then i / i-1) measured by each case's own
# workload pass: case -> (ratios, Slurm job and counts, evidence).
MEASURED_ACCEPTANCE: dict[int, tuple[list[float], str, str]] = {
    1: ([0.6655, 0.5804, 0.5649], "job 6637: 13185/19811, 7653/13185, 4323/7653",
        "logs/20261013_2_qf_cases/01_micro_throughput_c256/profile_workload/spec_decode_metrics.json"),
    2: ([0.6569, 0.5743, 0.5628], "job 6638: 7635/11622, 4385/7635, 2468/4385",
        "logs/20261013_2_qf_cases/02_graph_boundaries_c128/profile_workload/spec_decode_metrics.json"),
    3: ([0.7266, 0.6605, 0.6581], "job 6639: 7081/9745, 4677/7081, 3078/4677",
        "logs/20261013_2_qf_cases/03_balanced_anchor_c32/profile_workload/spec_decode_metrics.json"),
    4: ([0.707, 0.6278, 0.6061], "job 6640: 2394/3386, 1503/2394, 911/1503",
        "logs/20261013_2_qf_cases/04_prefill_spectrum_c32/profile_workload/spec_decode_metrics.json"),
    5: ([0.8114, 0.7858, 0.7961], "job 6641: 61337/75598, 48197/61337, 38370/48197",
        "logs/20261013_2_qf_cases/05_decode_spectrum_c64/profile_workload/spec_decode_metrics.json"),
    6: ([0.8974, 0.885, 0.8923], "job 6685: 29566/32948, 26165/29566, 23346/26165",
        "logs/20261013_2_qf_cases/06_long_long_c8/profile_workload/spec_decode_metrics.json"),
    7: ([0.8569, 0.8363, 0.8457], "job 6643: 48023/56043, 40162/48023, 33965/40162",
        "logs/20261013_2_qf_cases/07_quadrant_interference_c48/profile_workload/spec_decode_metrics.json"),
    8: ([0.8294, 0.8113, 0.8208], "job 6644: 47409/57158, 38465/47409, 31573/38465",
        "logs/20261013_2_qf_cases/08_completion_churn_c96/profile_workload/spec_decode_metrics.json"),
    9: ([0.8549, 0.8363, 0.8457], "job 6695: 229456/268416, 191905/229456, 162293/191905",
        "logs/20261013_2_qf_cases/09_rate_knee/profile_workload/spec_decode_metrics.json"),
    10: ([0.8573, 0.8418, 0.8514], "job 6646: 228742/266805, 192561/228742, 163953/192561",
        "logs/20261013_2_qf_cases/10_full_saturation_rate/profile_workload/spec_decode_metrics.json"),
    11: ([0.7704, 0.6286, 0.5649], "job 6647: 245/318, 154/245, 87/154",
        "logs/20261013_2_qf_cases/11_long_prefill_ladder_c1/profile_workload/spec_decode_metrics.json"),
    12: ([0.8762, 0.8686, 0.8843], "job 6648: 7115/8120, 6180/7115, 5465/6180",
        "logs/20261013_2_qf_cases/12_ultra_context_decode_c1/profile_workload/spec_decode_metrics.json"),
    13: ([0.937, 0.9385, 0.948], "job 6649: 94639/100999, 88815/94639, 84197/88815",
        "logs/20261013_2_qf_cases/13_generation_length_matrix_c8/profile_workload/spec_decode_metrics.json"),
    14: ([0.9037, 0.8908, 0.8951], "job 6651: 293614/324913, 261538/293614, 234113/261538",
        "logs/20261013_2_qf_cases/14_near_full_long_decode_c64/profile_workload/spec_decode_metrics.json"),
    15: ([0.9573, 0.9459, 0.9397], "job 6652: 221737/231632, 209730/221737, 197083/209730",
        "logs/20261013_2_qf_cases/15_kv_overcommit_mixed_churn_c32/profile_workload/spec_decode_metrics.json"),
    16: ([0.7111, 0.6481, 0.6159], "job 6623: 2370/3333, 1536/2370, 946/1536",
        "logs/20261013_2_qf_cases/16_prefill_spectrum_chunk4k_c32/profile_workload/spec_decode_metrics.json"),
    17: ([0.7303, 0.6443, 0.6217], "job 6624: 2404/3292, 1549/2404, 963/1549",
        "logs/20261013_2_qf_cases/17_prefill_spectrum_chunk8k_c32/profile_workload/spec_decode_metrics.json"),
    18: ([0.8429, 0.8226, 0.829], "job 6653: 48280/57277, 39715/48280, 32924/39715",
        "logs/20261013_2_qf_cases/18_quadrant_interference_chunk4k_c48/profile_workload/spec_decode_metrics.json"),
    19: ([0.8725, 0.8495, 0.8583], "job 6654: 47842/54833, 40642/47842, 34881/40642",
        "logs/20261013_2_qf_cases/19_quadrant_interference_chunk8k_c48/profile_workload/spec_decode_metrics.json"),
    20: ([0.8511, 0.8344, 0.8417], "job 6655: 229565/269730, 191544/229565, 161232/191544",
        "logs/20261013_2_qf_cases/20_full_saturation_chunk8k/profile_workload/spec_decode_metrics.json"),
}


def pool_gib(chunk: int = CHUNK) -> float:
    measured = MEASURED_POOL.get(chunk)
    return measured[0] if measured else SMOKE_POOL[0]


def pool_blocks(chunk: int = CHUNK) -> int:
    return round(pool_gib(chunk) * GIB / BLOCK_BYTES)


def blocks(terminal: int) -> int:
    """Pool blocks one running request of `terminal` tokens holds (inferred split; see above)."""
    return (math.ceil(terminal / ATTN_BLOCK_TOKENS) + math.ceil(terminal / (ATTN_BLOCK_TOKENS * COMPRESS_RATIO))
            + STATE_BLOCKS_PER_REQUEST)


assert blocks(L) == 62  # the logged 30.97x at max_model_len 131072


def mean_blocks(shapes) -> float:
    return sum(blocks(i + o) for i, o in shapes) / len(shapes)


def demand(shapes, concurrency: int, chunk: int = CHUNK) -> float:
    """Terminal KV demand: `concurrency` requests of the mean terminal length over the pool."""
    return concurrency * mean_blocks(shapes) / pool_blocks(chunk)


def kv_pinned_bytes(shapes, concurrency: int, ratio: float) -> int:
    """`--kv-cache-memory-bytes` that puts `concurrency` requests of the mean terminal length at `ratio` of the
    pool, the GLM-5.2 definition of a case's terminal KV demand."""
    return round(concurrency * mean_blocks(shapes) / ratio) * BLOCK_BYTES


# ---------------------------------------------------------------- the variants
SERVER = {
    "tp_size": 1,
    "dp_size": 1,
    "enforce_eager": False,
    "enable_server_load_tracking": True,
    "api_server_count": 4,
    "served_model_name": "qwen38-flash-next-nvfp4",
    "extra_args": [
        "--revision", REVISION,
        "--tokenizer-revision", REVISION,
        "--language-model-only",
        "--quantization", "modelopt_mixed",
        "--kv-cache-dtype", "fp8",
        "--attention-config", json.dumps({"indexer_kv_dtype": "fp8"}),
        "--mamba-ssm-cache-dtype", "float32",
        "--enable-prefix-caching",
        "--engram-config", json.dumps({"cpu_offload": True}),
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": DRAFT}),
        "--max-num-seqs", "256",
    ],
}
PYTHON_RUNTIME = {
    "packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
    "environment": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # A cold torch.compile / Inductor cache pushes engine startup past vLLM's 600 s default.
        "VLLM_ENGINE_READY_TIMEOUT_S": "2400",
        # Persist FA4 CuTeDSL compiles across passes; the directory is the job wrapper's.
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    },
    "lock_timeout_seconds": 30,
    "install_timeout_seconds": 30,
}
# Placeholder: the simulator arch for qwen4_exp is not named or built yet.
ARCH_TYPE = "qwen38_flash_next_TODO"
ARCH = {
    "type": ARCH_TYPE,
    "model_config": "model/config/qwen3_8_flash_next_nvfp4.json",
    "draft_tokens": DRAFT,
}
WORKER = {
    "type": "speculative",
    "max_batch_tokens": CHUNK,
    "batch_policy": "mix",
    "draft_tokens": DRAFT,
    "acceptance_seed": 73,
    # vLLM's admission: a request is admitted once its prompt fits, each running request's next window (the verify
    # width, 1 + draft) is allocated first, and a shortfall preempts the most recently admitted one
    # (skills/operate-run-alignment).
    "kv_admission_policy": "bounded-future",
    "kv_page_size": 1,
    "kv_max_future_tokens": DRAFT + 1,
    "kv_initial_new_token_ratio": 0.0,
    "kv_minimum_new_token_ratio": 0.0,
    "kv_new_token_ratio_decay_steps": 1,
    "kv_retract_decode_steps": 1,
    "decode_retraction_policy": "fcfs",
}


def variant(chunk: int = CHUNK, extra_args=()) -> dict:
    """One serving config at token budget `chunk`; the CUDA graph ceiling is the chunk (module docstring)."""
    compilation = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": capture_sizes(chunk)}
    server = {**SERVER, "chunk_size": chunk, "max_cudagraph_capture_size": chunk,
              "extra_args": SERVER["extra_args"] + ["--compilation-config", json.dumps(compilation,
                                                                                      separators=(",", ":"))]
              + list(extra_args)}
    return {
        "deployment": "unified",
        "gpu": "NVIDIA B200",
        "replicas": 1,
        "checkpoint": "qwen38_flash_next_nvfp4",
        "engine": "vllm",
        "backend": "vllm_tokens",
        "run_name_prefix": "qwen38_flash_next_nvfp4_b200_tp1",
        "python_runtime": PYTHON_RUNTIME,
        "server": server,
        "arch": {**ARCH, "cudagraph_capture_sizes": capture_sizes(chunk)},
        "worker": {**WORKER, "max_batch_tokens": chunk},
        "profile_passes": [
            {"kind": "nsys", "name": "profile_nsys", "trace": "kernel", "warmup": True},
            {"kind": "workload_metrics", "name": "profile_workload", "trace": "workload", "warmup": True},
        ],
        "input_builder": {"type": "speculative_engine_text", "draft_tokens": DRAFT, "measured_phase": "forward",
                          "group_assignment": "single"},
        "label_rules": "label_rules/manifest.json",
    }


# ---------------------------------------------------------------- the cases
def cycle(shapes, order, count, output):
    """`count` kernel-trace rows at terminal length - `output`, cycling `order` over `shapes` (1-based)."""
    return [[shapes[order[k % len(order)] - 1][0] + shapes[order[k % len(order)] - 1][1] - output, output]
            for k in range(count)]


GLM14 = [[8192, 8192], [12288, 8192], [16384, 8192], [24576, 4096], [32768, 4096], [49152, 2048]]
GLM15 = [[65536, 4096], [16384, 32768], [98304, 2048], [32768, 16384], [8192, 8192], [49152, 8192]]
MIX = [[128, 128], [128, 4096], [4096, 128], [3072, 3072], [512, 256], [2048, 1024], [6144, 512], [1024, 4096]]
MICRO = [[16, 16], [32, 32], [64, 64], [128, 64]]
C14, C15 = 64, 32
C_SAT = 256
KV14 = kv_pinned_bytes(GLM14, C14, 0.914)
KV15 = kv_pinned_bytes(GLM15, C15, 1.226)
# Case 09's open-loop rate: 0.8 x case 10's saturated request rate, once case 10's workload pass has run.
RATE_KNEE: tuple[float, str, str] | None = (
    2.70,
    "queueing knee = 0.8 x the case-10 saturated request rate on the same shapes: 512 requests over a 151.75 s "
    "client run = 3.374 req/s (req-frontend summary request_throughput_per_s, Slurm job 6646, this pack's serving "
    "config)",
    "logs/20261013_2_qf_cases/10_full_saturation_rate/profile_workload",
)
# A pinned pool's block count as logged by its workload pass (`Maximum concurrency` x 62 blocks).
PINNED_EVIDENCE: dict[str, tuple[str, str] | None] = {
    "kv14": ("`Maximum concurrency ... 25.23x` x 62 = 1564 blocks (Slurm job 6651)",
             "logs/20261013_2_qf_cases/14_near_full_long_decode_c64/profile_workload"),
    "kv15": ("`Maximum concurrency ... 13.97x` x 62 = 866 blocks (Slurm job 6652)",
             "logs/20261013_2_qf_cases/15_kv_overcommit_mixed_churn_c32/profile_workload"),
}


def pool_note(shapes, c: int, chunk: int = CHUNK) -> str:
    return (f"{c} x {mean_blocks(shapes):.2f} mean terminal blocks = {demand(shapes, c, chunk):.0%} of the "
            f"{pool_blocks(chunk)}-block pool")


CASES = [
    dict(index=1, name="micro_throughput_c256", purpose="Maximum request batching and scheduler pressure.",
         coverage=["batching", "scheduler_pressure"], max_concurrency=256, capture_seconds=15.0,
         workload_trace=dict(shapes=MICRO, repeats=256),
         derivation=f"GLM's c256 = max_num_seqs. Every request is <= 192 tokens, so each holds {blocks(192)} "
                    f"blocks: {pool_note(MICRO, 256)}. The per-request state blocks, not max_num_seqs, bind: at "
                    f"most {pool_blocks() // blocks(192)} requests run at once and the rest queue. That is this "
                    f"deployment's real batching ceiling, so the case keeps c256 (4 requests per slot)."),
    dict(index=2, name="graph_boundaries_c128", purpose="CUDA graph and small-batch kernel boundaries.",
         coverage=["cuda_graph_boundaries", "small_batch"], max_concurrency=128, capture_seconds=15.0,
         workload_trace=dict(shapes=[[1, 32], [15, 32], [16, 32], [31, 32], [32, 32], [63, 64], [64, 64],
                                     [127, 64], [128, 64], [255, 64], [256, 64]], repeats=48),
         derivation=f"Unchanged from GLM. Prompts straddle the capture edges 16/32/64/128/256 of the explicit list; "
                    f"decode verify steps of 4 tokens per request replay FULL graphs at multiples of 4. "
                    f"{pool_note([[256, 64]], 128)} (fits)."),
    dict(index=3, name="balanced_anchor_c32", purpose="Comparable extension of the existing c32 anchor.",
         coverage=["balanced_anchor"], max_concurrency=32, capture_seconds=15.0,
         workload_trace=dict(shapes=[[256, 128], [256, 256], [1024, 128], [1024, 256], [2048, 128],
                                     [2048, 256], [4096, 128], [4096, 256]], repeats=16),
         derivation=f"The cross-model anchor: unchanged. {pool_note([[4096, 256]], 32)} at its largest shape."),
    dict(index=4, name="prefill_spectrum_c32", purpose="Prefill scaling and the 2048-token chunk boundary.",
         coverage=["prefill_scaling", "chunk_boundary"], max_concurrency=32, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 64], [512, 64], [1984, 64], [2048, 64], [2112, 64], [4096, 64],
                                     [6144, 64], [7168, 64]], repeats=16),
         derivation="Unchanged: the chunk is 2048, so 1984/2048/2112 straddle it; 4096/6144/7168 are 2/3/3.5 "
                    "chunks. c32 <= max_num_seqs."),
    dict(index=5, name="decode_spectrum_c64", purpose="Decode residence, KV growth, and batch contraction.",
         coverage=["decode_residence", "kv_growth"], max_concurrency=64, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 16], [128, 128], [128, 512], [128, 1024], [128, 2048], [128, 4096],
                                     [512, 2048], [1024, 4096]], repeats=16),
         derivation=f"Unchanged: {pool_note([[1024, 4096]], 64)} at its largest shape. QSA attention context grows "
                    f"to 5120 while GDN state stays constant per request."),
    dict(index=6, name="long_long_c8",
         purpose="Long-prefill plus long-decode interaction near context capacity.",
         coverage=["long_prefill", "long_decode"], max_concurrency=8, capture_seconds=15.0,
         workload_trace=dict(shapes=[[32768, 4096], [49152, 3072], [65536, 3968], [81920, 2944], [98304, 1920]],
                             repeats=7),
         derivation=f"GLM's prompts are 25-75% of its 8192 context limit; this deployment serves 131072, so "
                    f"prompts x16 (32K-96K), as MiMo and Qwen3.8-27B. Outputs keep GLM's absolute 1920-4096: they "
                    f"set the decode step count. {pool_note([[98304, 1920]], 8)} at its largest shape."),
    dict(index=7, name="quadrant_interference_c48", purpose="All four IO quadrants sharing continuous batches.",
         coverage=["io_quadrants", "continuous_batching"], max_concurrency=48, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 128], [128, 4096], [4096, 128], [3072, 3072]], repeats=24),
         derivation="Unchanged: the quadrants are defined against the 2048 chunk (4096 = two chunks), not the "
                    "context limit. c48 <= max_num_seqs."),
    dict(index=8, name="completion_churn_c96",
         purpose="Rapid completions, shrinking batches, and request replacement.",
         coverage=["completion_churn", "admission"], max_concurrency=96, capture_seconds=15.0,
         workload_trace=dict(shapes=[[512, 1], [512, 16], [512, 64], [512, 256], [512, 1024], [512, 4096]],
                             repeats=32),
         derivation=f"Unchanged: {pool_note([[512, 4096]], 96)} at its largest shape, so replacements are admitted "
                    f"at once."),
    dict(index=9, name="rate_knee", purpose="Maximum sustainable open-loop request rate.",
         coverage=["arrival_rate_knee"], max_concurrency=C_SAT, capture_seconds=20.0, arrival="trace",
         workload_trace=dict(shapes=MIX, repeats=64),
         rate=(dict(value=RATE_KNEE[0], status="measured", derived_from=RATE_KNEE[1], evidence=RATE_KNEE[2])
               if RATE_KNEE else
               dict(value=1.0, status="provisional",
                    derived_from="placeholder until case 10's workload pass gives the saturated request rate; the "
                                 "knee is 0.8 x that rate", evidence="")),
         derivation="GLM's knee rule (0.8 x the case-10 saturated rate) on this model's throughput; same 8-shape "
                    "mix and 512 requests. max_concurrency = case 10's c256 caps in-flight requests on both sides."),
    dict(index=10, name="full_saturation_rate", purpose="Peak-throughput open-loop load beyond the sustainable knee.",
         coverage=["saturation", "peak_throughput"], max_concurrency=C_SAT, capture_seconds=20.0,
         workload_trace=dict(shapes=MIX, repeats=64),
         derivation=f"GLM's c256. {pool_note(MIX, C_SAT)}, so admission and preemption bind: saturation in KV as "
                    f"well as compute. Same trace as case 9 so the knee is a fraction of this rate."),
    dict(index=11, name="long_prefill_ladder_c1", purpose="4K-120K prompts: long-prefill ladder at concurrency 1.",
         coverage=["long_prefill", "bounded_kernel_trace"], max_concurrency=1, capture_seconds=120.0,
         workload_trace=dict(shapes=[[4096, 16], [8192, 64], [16384, 16], [24576, 64], [32768, 16], [49152, 64],
                                     [65536, 16], [98304, 64], [114688, 16], [122880, 64]], repeats=2,
                             id_suffix="full"),
         kernel_trace=dict(shapes=[[4096, 16], [8192, 64], [16384, 16], [24576, 64], [32768, 16], [49152, 64],
                                   [65536, 16]], repeats=1, id_suffix="nsys"),
         derivation="GLM's rungs are 1/32 .. 15/16 of its 524288 limit; the same fractions of 131072 (each "
                    "rung / 4), as MiMo. The NSYS trace is GLM's choice too: the rungs up to half the limit, once."),
    dict(index=12, name="ultra_context_decode_c1",
         purpose="Long context plus long output: ultra-context decode at concurrency 1.",
         coverage=["ultra_context", "long_decode"], max_concurrency=1, capture_seconds=120.0,
         workload_trace=dict(shapes=[[p, o] for p in (114688, 65536, 32768, 8192, 2048) for o in (256, 1024, 4096)],
                             repeats=1, id_suffix="full"),
         kernel_trace=dict(shapes=[[114688, 64], [2048, 256], [3072, 64]], repeats=1, id_suffix="nsys"),
         derivation="Prompts are GLM's / 4 (same fractions of the context limit; the top, 114688 + 4096, is 88% "
                    "of 131072). Outputs 256/1024/4096 unchanged: at c1 they are decode step counts."),
    dict(index=13, name="generation_length_matrix_c8",
         purpose="Mixed prompt and output lengths across a generation-length matrix.",
         coverage=["generation_length_matrix"], max_concurrency=8, capture_seconds=60.0,
         workload_trace=dict(shapes=[[p, o] for p in (128, 512, 2048, 8192, 16384, 32768)
                                     for o in (4096, 8192, 16384, 32768)], repeats=1, id_suffix="full"),
         kernel_trace=dict(shapes=[[128, 256], [512, 256], [2048, 256], [8192, 256], [16384, 256], [32768, 256],
                                   [4096, 256], [24576, 256]], repeats=1, id_suffix="nsys"),
         derivation=f"Unchanged: GLM already ran it at max_model_len 131072, and its longest row (65536) fits. "
                    f"{pool_note([[32768, 32768]], 8)} at its largest shape."),
    dict(index=14, name=f"near_full_long_decode_c{C14}", variant="kv14",
         purpose="About 91.4% terminal KV demand: near-full long decode.",
         coverage=["kv_near_full"], max_concurrency=C14, capture_seconds=120.0,
         workload_trace=dict(shapes=GLM14, repeats=32, id_suffix="full"),
         kernel_trace=dict(shapes=cycle(GLM14, [6, 1, 2, 3, 4, 5], C14, 256), repeats=1, id_suffix="nsys"),
         derivation=f"GLM's 91.4% = c x mean terminal blocks / pool, on GLM's shapes. Their mean holds "
                    f"{mean_blocks(GLM14):.2f} blocks (blocks() above). c{C14}: {C14} x {mean_blocks(GLM14):.2f} / "
                    f"0.914 = {KV14 // BLOCK_BYTES} blocks x {BLOCK_BYTES} B = {KV14} B ({KV14 / GIB:.2f} GiB), "
                    f"pinned with --kv-cache-memory-bytes, under the default {pool_blocks()}-block pool. repeats 32 "
                    f"keeps GLM's 3 requests per slot. NSYS trace: c requests at terminal - 256, so the window opens "
                    f"at the terminal KV state."),
    dict(index=15, name=f"kv_overcommit_mixed_churn_c{C15}", variant="kv15",
         purpose="About 122.6% average terminal KV demand, admission-controlled churn.",
         coverage=["kv_overcommit", "admission"], max_concurrency=C15, capture_seconds=180.0,
         workload_trace=dict(shapes=GLM15, repeats=12, id_suffix="full"),
         kernel_trace=dict(shapes=cycle(GLM15, [1, 2, 3, 4, 5, 6], C15, 256), repeats=1, id_suffix="nsys"),
         derivation=f"As case 14 at GLM's 122.6%: {C15} x {mean_blocks(GLM15):.2f} blocks / 1.226 = "
                    f"{KV15 // BLOCK_BYTES} blocks = {KV15} B ({KV15 / GIB:.2f} GiB). repeats 12 unchanged "
                    f"(72 requests, 2.25 per slot at c{C15})."),
]


def prefill_spectrum_at(chunk: int) -> list[list[int]]:
    """Case 04's prompt ladder with its chunk-relative rungs moved to `chunk`: 128 and 512 below one chunk, the
    boundary straddle chunk - 64 / chunk / chunk + 64, then 2, 3 and 3.5 chunks (case 04: 4096, 6144, 7168)."""
    return [[p, 64] for p in (128, 512, chunk - 64, chunk, chunk + 64, 2 * chunk, 3 * chunk, 7 * chunk // 2)]


def quadrants_at(chunk: int) -> list[list[int]]:
    """Case 07's four IO quadrants with the long prompts kept at 2 and 1.5 chunks (case 07: 4096 and 3072 at chunk
    2048). Outputs stay case 07's 128 / 4096 / 3072: they set the decode step count, not the chunk shape."""
    return [[128, 128], [128, 4096], [2 * chunk, 128], [3 * chunk // 2, 3072]]


for index, chunk in ((16, 4096), (17, 8192)):
    CASES.append(dict(
        index=index, name=f"prefill_spectrum_chunk{chunk // 1024}k_c32", chunk_size=chunk, base=4,
        purpose=f"Prefill scaling and the {chunk}-token chunk boundary.",
        coverage=["prefill_scaling", "chunk_boundary", "chunk_size"], max_concurrency=32, capture_seconds=15.0,
        workload_trace=dict(shapes=prefill_spectrum_at(chunk), repeats=16),
        derivation=f"Case 04 at chunk {chunk}: the same c32 and 16 repeats, rungs moved to the new chunk boundary "
                   f"({chunk - 64}/{chunk}/{chunk + 64}) and to 2/3/3.5 chunks. Steps reach {chunk} tokens; the "
                   f"variant's graph ceiling is the chunk, so they replay PIECEWISE graphs."))
for index, chunk in ((18, 4096), (19, 8192)):
    CASES.append(dict(
        index=index, name=f"quadrant_interference_chunk{chunk // 1024}k_c48", chunk_size=chunk, base=7,
        purpose=f"All four IO quadrants sharing continuous batches at a {chunk}-token chunk.",
        coverage=["io_quadrants", "continuous_batching", "chunk_size"], max_concurrency=48, capture_seconds=15.0,
        workload_trace=dict(shapes=quadrants_at(chunk), repeats=24),
        derivation=f"Case 07 at chunk {chunk}: same c48 and 24 repeats; the long prompts stay 2 and 1.5 chunks "
                   f"({2 * chunk} and {3 * chunk // 2}), so mixed steps pair one full chunk with up to 47 decodes."))
CASES.append(dict(
    index=20, name="full_saturation_chunk8k", chunk_size=8192, base=10,
    purpose="Peak-throughput load at an 8192-token chunk: the largest decode batches inside the largest steps.",
    coverage=["saturation", "peak_throughput", "chunk_size"], max_concurrency=C_SAT, capture_seconds=20.0,
    workload_trace=dict(shapes=MIX, repeats=64),
    derivation="Case 10 at chunk 8192: same c256 and trace. Mixed steps carry up to ~250 decodes plus an "
               "8192-token prefill budget, the top corner of the shape envelope (M 8192, B 256)."))


def trace_spec(case: dict, key: str) -> dict:
    spec = dict(case[key])
    stem = f"{case['index']:02d}_{case['name']}"
    spec = {"file": f"traces/{stem}{'.nsys' if key == 'kernel_trace' else ''}.csv", **spec}
    return spec


def kv_calibration(case: dict) -> dict:
    pinned = {"kv14": KV14, "kv15": KV15}.get(case.get("variant"))
    if pinned is not None:
        logged = PINNED_EVIDENCE[case["variant"]]
        return {
            "value": round(pinned / 1e9, 4),
            "status": "measured" if logged else "provisional",
            "derived_from": f"--kv-cache-memory-bytes {pinned} of this case's variant pins the KV pool to "
                            f"{pinned // BLOCK_BYTES} blocks" + (f"; the workload pass logs {logged[0]}" if logged
                                                                else "; confirm the block count in its workload pass"),
            "evidence": logged[1] if logged else "",
        }
    chunk = case.get("chunk_size", CHUNK)
    measured = MEASURED_POOL[chunk]
    gib, jobs, evidence = measured or SMOKE_POOL
    return {
        "value": round(gib * GIB / 1e9, 4),
        "status": "measured" if measured or chunk == CHUNK else "provisional",
        "derived_from": f"`Available KV cache memory: {gib} GiB` (x 1.073741824) at gpu_memory_utilization "
                        f"{GPU_MEMORY_UTILIZATION}, max_num_seqs 256, chunk {CHUNK if measured is None else chunk} "
                        f"(Slurm jobs {jobs})" + ("" if measured or chunk == CHUNK else
                                                  f"; replace with this deployment's chunk-{chunk} workload pass"),
        "evidence": evidence,
    }


def acceptance(case: dict) -> dict:
    measured = MEASURED_ACCEPTANCE.get(case["index"])
    if measured:
        value, counts, evidence = measured
        return {"value": value, "status": "measured",
                "derived_from": f"conditional accepted-token ratios of this case's workload pass ({counts})",
                "evidence": evidence}
    value, derived_from, evidence = PROVISIONAL_ACCEPTANCE
    return {"value": value, "status": "provisional", "derived_from": derived_from, "evidence": evidence}


def variant_name(case: dict) -> str:
    if "variant" in case:
        return f"tp1_{case['variant']}"
    chunk = case.get("chunk_size", CHUNK)
    return "tp1" if chunk == CHUNK else f"tp1_chunk{chunk // 1024}k"


cases = []
for case in CASES:
    out = {
        "index": case["index"],
        "name": case["name"],
        "variant": variant_name(case),
        "purpose": case["purpose"],
        "coverage": case["coverage"],
        "max_model_len": L,
        "max_concurrency": case["max_concurrency"],
        "profile_arrival_mode": "trace-timed" if case.get("arrival") == "trace" else "saturated",
        "simulation_arrival_mode": "trace_timed" if case.get("arrival") == "trace" else "saturated",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "capture_seconds": case["capture_seconds"],
        "device_role": "primary",
        "workload_trace": trace_spec(case, "workload_trace"),
    }
    if "chunk_size" in case:
        out["chunk_size"] = case["chunk_size"]
    out["speculative_acceptance"] = acceptance(case)
    if "kernel_trace" in case:
        out["kernel_trace"] = trace_spec(case, "kernel_trace")
    if "rate" in case:
        out["rate"] = case["rate"]
    out["attn_gpu_memory_gb"] = kv_calibration(case)
    out["provenance"] = {
        "derived_from_pack": f"qwen38_27b_nvfp4_b200_tp1 case {case.get('base', case['index']):02d}",
        "derivation": case["derivation"],
    }
    for spec in (out["workload_trace"], out.get("kernel_trace")):
        for i, o in (spec or {}).get("shapes", []):
            assert i + o < L, (case["name"], i, o)
    cases.append(out)


document = {
    "schema_version": 1,
    "pack": "qwen38_flash_next_nvfp4_b200_tp1",
    "description": "Qwen3.8-Flash-Next-NVFP4 (nvidia/Qwen3.8-Flash-Next-NVFP4 fc694b54, qwen4_exp) on one B200, TP1, "
                   "native MTP with 3 draft tokens, under the vLLM fork 892da082: FP8 KV and QSA indexer cache, FP32 "
                   "GDN state, PLE n-gram table on the host, prefix caching, max_num_seqs 256. The twenty operating "
                   "points of qwen38_27b_nvfp4_b200_tp1 re-derived for this model's limits; see make_pack.py.",
    "recorded_on": {
        "host": "b200_raid",
        "gpu": "NVIDIA B200",
        "checkpoint_snapshots": {
            "qwen38_flash_next_nvfp4": f"models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/{REVISION}",
        },
        "nsys_version": "2025.5.2",
        "device_roles": {"primary": "0"},
    },
    "variants": {
        "tp1": variant(),
        "tp1_kv14": variant(extra_args=["--kv-cache-memory-bytes", str(KV14)]),
        "tp1_kv15": variant(extra_args=["--kv-cache-memory-bytes", str(KV15)]),
        "tp1_chunk4k": variant(4096),
        "tp1_chunk8k": variant(8192),
    },
    "cases": cases,
}
(HERE / "campaign.yaml").write_text(
    "# Generated by make_pack.py; edit that, not this.\n" + yaml.safe_dump(document, sort_keys=False, width=110)
)
print(len(cases), "cases; pool blocks", pool_blocks(), "block bytes", BLOCK_BYTES, "kv14", KV14, "kv15", KV15)
for case in CASES:
    print(f"{case['index']:2d} {case['name']:40s} c{case['max_concurrency']:<4d} chunk {case.get('chunk_size', CHUNK)}"
          f" demand {demand(case['workload_trace']['shapes'], case['max_concurrency']):.2f}")

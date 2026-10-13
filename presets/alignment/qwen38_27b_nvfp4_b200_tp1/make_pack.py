"""Write campaign.yaml for Qwen3.8-27B-NVFP4 on one B200 (TP1) under the vLLM fork 892da082.

The fifteen operating points of `glm52_nvfp4_b200_tp8` keep their index, purpose and coverage: what each case
stresses. Their concurrency and lengths are re-derived from this model's limits below, the way
`mimo_v26_flash_b200_tp4` re-derived them for MiMo (`c x mean terminal blocks / KV pool` for the KV cases, the
knee as 0.8 x the case-10 saturated rate, lengths as fractions of the context limit). Cases 16-20 add the
scheduler token budget as an axis: case 04 (prefill spectrum) and case 07 (quadrant mix) at chunk 4096 and 8192,
as MiMo's cases 16-19, plus case 10 (full saturation, decode batches to 256) at chunk 8192, so the largest decode
batches also meet the largest mixed steps. Every case states its derivation in `provenance.derivation`; `CASES`
below is the one place it is computed.

Serving config: the draft's diverse_100 capture (wt-wt-qwen-3.8-27b logs/20261008_13_qwen38_enwik9_reference:
vLLM fork 892da082, checkpoint revision 482ca0f3, FP8 KV, FP32 SSM state, prefix caching, language-model-only,
max_cudagraph_capture_size 2048, gpu_memory_utilization 0.85) with one change by user decision: no
`--max-num-seqs 8`. Nothing is passed, so vLLM's B200 default applies, 1024
(vllm/engine/arg_utils.py:2789-2798, device memory >= 160 GiB). No case offers more than 256 concurrent
requests, so it never binds. Both passes run req-frontend's warmup (`warmup: true`), and the host's job wrapper
points every JIT cache (Inductor, Triton, vLLM, FlashInfer, CuTeDSL) at one persistent directory, so no
compilation lands inside a measured window.

Limits (TP1, gpu_memory_utilization 0.85, chunk 2048):

- S = 1024 (vLLM default `max_num_seqs`). The simulated chunked-prefill worker has no running-request cap.
- CUDA graphs, FULL_AND_PIECEWISE: `capture_sizes()` is vLLM 892da082's list for an explicit
  max_cudagraph_capture_size 2048 (vllm/config/vllm.py:2499-2553): [1, 2, 4], 8..248 step 8, 256..2048 step 16
  = 147 sizes. A chunk above 2048 does not extend it, so mixed steps above 2048 tokens run piecewise-eager.
- L = 131072 (`--max-model-len`); rendering rejects a trace row with input + output >= L.
- KV pool. vLLM's hybrid allocator puts the 64 layers ([GDN, GDN, GDN, GQA] x 16) into 4 groups of 16 layers:
  3 GDN (mamba) groups and 1 full-attention group, one block id spanning 16 per-layer tensors. The attention
  block size is raised until its page reaches the mamba page ("Setting attention block size to 1568 tokens",
  draft logs):
    mamba page = SSM 48 V heads x 128 x 128 x 4 B (FP32) + conv (2 x 16 x 128 + 48 x 128) x 3 x 2 B
               = 3,145,728 + 61,440 = 3,207,168 B, padded 0.13% to the attention page;
    attention page = 1568 tokens x 4 KV heads x 256 x 2 (K, V) x 1 B (FP8) = 3,211,264 B;
    block = 16 x 3,211,264 = 51,380,224 B.
  A running request of terminal length T holds ceil(T / 1568) attention blocks plus its GDN state. vLLM charges
  the state at 2 blocks per mamba group in prefix-caching "align" mode (its 131072-token admission figure is
  ceil(131072 / 1568) + 3 x 2 = 90 blocks: the draft's 2704-block pool / 90 = 30.04x against the logged 30.03x),
  so `blocks()` below charges 6. GDN state therefore dominates short requests: a 256-token request costs 7
  blocks, 360 MB, and the pool holds only ~380 of them at once.
  The pool is read from each chunk size's workload passes (`MEASURED_POOL`): 125.52 / 125.34 / 124.60 GiB at
  chunk 2048 / 4096 / 8192 = 2623 / 2619 / 2603 blocks, since the profile-run activation peak moves with
  --max-num-batched-tokens (the draft's 129.39 GiB was at max_num_seqs 8). A pass whose JIT and FlashInfer
  autotune caches were cold logged a pool ~2 GiB smaller and untuned fp8_gemm shapes, so such passes were re-run
  warm and only warm passes are cited.

Cases 14 and 15 pin their pool with --kv-cache-memory-bytes (their own variants), as MiMo and
`glm53_flash_fp8_b200_tp4_ep4` do, so their terminal KV demand is exact rather than a function of a measured
free-memory figure.

Run from the repo root: uv run python presets/alignment/qwen38_27b_nvfp4_b200_tp1/make_pack.py
"""
import math
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- Qwen3.8-27B limits
S = 1024
L = 131072
MAX_CUDAGRAPH_CAPTURE_SIZE = 2048
CHUNK = 2048
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"


def capture_sizes(max_size: int, max_num_tokens: int) -> list[int]:
    """vLLM 892da082 `VllmConfig._set_cudagraph_sizes` for an explicit max_cudagraph_capture_size, balanced
    performance mode, no sequence parallelism (vllm/config/vllm.py:2499-2553)."""
    max_size = min(max_num_tokens, max_size)
    sizes = [i for i in (1, 2, 4) if i <= max_size]
    if max_size >= 8:
        sizes += list(range(8, min(max_size + 1, 256), 8))
    if max_size >= 256:
        sizes += list(range(256, max_size + 1, 16))
    if max_num_tokens <= max_size and max_num_tokens not in sizes:
        sizes.append(max_num_tokens)
    return sorted(set(sizes))


CAPTURE_SIZES = capture_sizes(MAX_CUDAGRAPH_CAPTURE_SIZE, CHUNK)
ATTN_BLOCK_TOKENS = 1568
PAGE_BYTES = ATTN_BLOCK_TOKENS * 4 * 256 * 2  # = the padded mamba page
LAYERS_PER_GROUP = 16
BLOCK_BYTES = LAYERS_PER_GROUP * PAGE_BYTES
MAMBA_BLOCKS_PER_REQUEST = 3 * 2
GIB = 1 << 30

# Per-GPU KV pool (`Available KV cache memory`, GiB) logged by each chunk size's workload passes:
# chunk -> (GiB, Slurm jobs, evidence case dir). None until measured.
MEASURED_POOL: dict[int, tuple[float, str, str] | None] = {
    2048: (125.52, "6479 (case 01; every warm chunk-2048 pass logs the same pool)",
           "logs/20261012_1_q38_align_campaign/01_micro_throughput_c256/profile_workload"),
    4096: (125.34, "6461, 6525", "logs/20261012_1_q38_align_campaign/16_prefill_spectrum_chunk4k_c32/profile_workload"),
    8192: (124.60, "6534, 6536, 6538", "logs/20261012_1_q38_align_campaign/17_prefill_spectrum_chunk8k_c32/profile_workload"),
}
# A pinned pool's block count as logged (`Maximum concurrency for 131,072 tokens per request` x 90 blocks).
PINNED_EVIDENCE = {
    "kv14": ("19.84x = 1786 blocks (job 6459)", "logs/20261012_1_q38_align_campaign/14_near_full_long_decode_c64/profile_workload"),
    "kv15": ("12.42x = 1118 blocks (job 6473)", "logs/20261012_1_q38_align_campaign/15_kv_overcommit_mixed_churn_c32/profile_workload"),
}
# The draft's pool at max_num_seqs 8, chunk 2048: the estimate a case is derived against until measured.
DRAFT_POOL_GIB = 129.39


def pool_blocks(chunk: int = CHUNK) -> int:
    measured = MEASURED_POOL.get(chunk)
    gib = measured[0] if measured else DRAFT_POOL_GIB
    return int(gib * GIB) // BLOCK_BYTES


def blocks(terminal: int) -> int:
    """Pool blocks one running request of `terminal` tokens holds."""
    return math.ceil(terminal / ATTN_BLOCK_TOKENS) + MAMBA_BLOCKS_PER_REQUEST


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
    "chunk_size": CHUNK,
    "max_cudagraph_capture_size": MAX_CUDAGRAPH_CAPTURE_SIZE,
    "enforce_eager": False,
    "enable_server_load_tracking": True,
    "api_server_count": 1,
    "served_model_name": "qwen38-27b-nvfp4",
    "extra_args": [
        "--revision", REVISION,
        "--tokenizer-revision", REVISION,
        "--language-model-only",
        "--kv-cache-dtype", "fp8_e4m3",
        "--mamba-ssm-cache-dtype", "float32",
        "--enable-prefix-caching",
    ],
}
PYTHON_RUNTIME = {
    "packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
    "environment": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # A cold torch.compile / Inductor cache pushes engine startup past vLLM's 600 s default.
        "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
        # Persist FA4 CuTeDSL compiles across passes; the directory is the job wrapper's
        # FLASH_ATTENTION_CUTE_DSL_CACHE_DIR (a host path, so not in the pack).
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    },
    "lock_timeout_seconds": 30,
    "install_timeout_seconds": 30,
}
# The linears' precisions come from the checkpoint config's quantization_config (ModelOpt MIXED_PRECISION), so
# the arch takes no precision field. The capture list is the server's (explicit 2048 cap; chunk 4096/8192 steps
# above it run piecewise-eager), so every chunk size shares it.
ARCH_TYPE = "qwen35_vllm_nvfp4_gdn_dense"
ARCH = {
    "type": ARCH_TYPE,
    "model_config": "model/config/qwen3_8_27b_nvfp4.json",
    "cudagraph_capture_sizes": CAPTURE_SIZES,
}
WORKER = {
    "type": "chunked_prefill",
    "max_batch_tokens": CHUNK,
    "batch_policy": "mix",
    # vLLM's admission: a request is admitted once its prompt fits, each running request's next window is
    # allocated first, and a shortfall preempts the most recently admitted one (skills/operate-run-alignment).
    "kv_admission_policy": "bounded-future",
    "kv_page_size": 1,
    "kv_max_future_tokens": 1,
    "kv_initial_new_token_ratio": 0.0,
    "kv_minimum_new_token_ratio": 0.0,
    "kv_new_token_ratio_decay_steps": 1,
    "kv_retract_decode_steps": 1,
    "decode_retraction_policy": "fcfs",
}


def variant(extra_args=()) -> dict:
    server = dict(SERVER)
    server["extra_args"] = SERVER["extra_args"] + list(extra_args)
    return {
        "deployment": "unified",
        "gpu": "NVIDIA B200",
        "replicas": 1,
        "checkpoint": "qwen38_27b_nvfp4",
        "engine": "vllm",
        # The draft's validated backend: token-id prompts, vLLM started with skip_tokenizer_init.
        "backend": "vllm_tokens",
        "run_name_prefix": "qwen38_27b_nvfp4_b200_tp1",
        "python_runtime": PYTHON_RUNTIME,
        "server": server,
        "arch": dict(ARCH),
        "worker": dict(WORKER),
        "profile_passes": [
            {"kind": "nsys", "name": "profile_nsys", "trace": "kernel", "warmup": True},
            {"kind": "workload_metrics", "name": "profile_workload", "trace": "workload", "warmup": True},
        ],
        "input_builder": {"type": "engine_text", "measured_phase": "forward", "group_assignment": "single"},
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
KV14 = kv_pinned_bytes(GLM14, C14, 0.914)
KV15 = kv_pinned_bytes(GLM15, C15, 1.226)
C_SAT = 256

CASES = [
    dict(index=1, name="micro_throughput_c256", purpose="Maximum request batching and scheduler pressure.",
         coverage=["batching", "scheduler_pressure"], max_concurrency=256, capture_seconds=15.0,
         workload_trace=dict(shapes=MICRO, repeats=256),
         derivation=f"GLM's c256, the brief's decode-batch ceiling. Every request is <= 192 tokens, so each holds "
                    f"1 attention block + 6 GDN-state blocks: 256 x 7 = 1792 blocks = "
                    f"{demand(MICRO, 256):.0%} of the pool. Above ~380 short requests the GDN state, not "
                    f"max_num_seqs (1024), would bind. Shapes and repeats unchanged (4 requests per slot)."),
    dict(index=2, name="graph_boundaries_c128", purpose="CUDA graph and small-batch kernel boundaries.",
         coverage=["cuda_graph_boundaries", "small_batch"], max_concurrency=128, capture_seconds=15.0,
         workload_trace=dict(shapes=[[1, 32], [15, 32], [16, 32], [31, 32], [32, 32], [63, 64], [64, 64],
                                     [127, 64], [128, 64], [255, 64], [256, 64]], repeats=48),
         derivation="Unchanged from GLM: c128 <= S, and as the batch drains decode replays every captured size "
                    "1..128. Prompts straddle the capture edges 16/32/64/128/256 of the explicit 2048 list "
                    "(8-step below 256, 16-step above). KV 128 x 7 blocks = 33% of the pool."),
    dict(index=3, name="balanced_anchor_c32", purpose="Comparable extension of the existing c32 anchor.",
         coverage=["balanced_anchor"], max_concurrency=32, capture_seconds=15.0,
         workload_trace=dict(shapes=[[256, 128], [256, 256], [1024, 128], [1024, 256], [2048, 128],
                                     [2048, 256], [4096, 128], [4096, 256]], repeats=16),
         derivation="The cross-model anchor: unchanged. c32 x <= 9 blocks = 11% of the pool."),
    dict(index=4, name="prefill_spectrum_c32", purpose="Prefill scaling and the 2048-token chunk boundary.",
         coverage=["prefill_scaling", "chunk_boundary"], max_concurrency=32, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 64], [512, 64], [1984, 64], [2048, 64], [2112, 64], [4096, 64],
                                     [6144, 64], [7168, 64]], repeats=16),
         derivation="Unchanged: the chunk is 2048, so 1984/2048/2112 straddle it; 4096/6144/7168 are 2/3/3.5 "
                    "chunks. c32 <= S."),
    dict(index=5, name="decode_spectrum_c64", purpose="Decode residence, KV growth, and batch contraction.",
         coverage=["decode_residence", "kv_growth"], max_concurrency=64, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 16], [128, 128], [128, 512], [128, 1024], [128, 2048], [128, 4096],
                                     [512, 2048], [1024, 4096]], repeats=16),
         derivation="Unchanged: c64 <= S; at most 64 x 10 blocks = 24% of the pool. GQA decode context grows "
                    "to 5120 while GDN state stays constant per request."),
    dict(index=6, name="long_long_c8",
         purpose="Long-prefill plus long-decode interaction near context capacity.",
         coverage=["long_prefill", "long_decode"], max_concurrency=8, capture_seconds=15.0,
         workload_trace=dict(shapes=[[32768, 4096], [49152, 3072], [65536, 3968], [81920, 2944], [98304, 1920]],
                             repeats=7),
         derivation="GLM's prompts are 25-75% of its 8192 context limit; this deployment serves 131072, so "
                    "prompts x16 (32K-96K), as MiMo. Outputs keep GLM's absolute 1920-4096: they set the decode "
                    "step count. KV 8 x <= 72 blocks = 21% of the pool."),
    dict(index=7, name="quadrant_interference_c48", purpose="All four IO quadrants sharing continuous batches.",
         coverage=["io_quadrants", "continuous_batching"], max_concurrency=48, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 128], [128, 4096], [4096, 128], [3072, 3072]], repeats=24),
         derivation="Unchanged: the quadrants are defined against the 2048 chunk (4096 = two chunks), not the "
                    "context limit. c48 <= S."),
    dict(index=8, name="completion_churn_c96",
         purpose="Rapid completions, shrinking batches, and request replacement.",
         coverage=["completion_churn", "admission"], max_concurrency=96, capture_seconds=15.0,
         workload_trace=dict(shapes=[[512, 1], [512, 16], [512, 64], [512, 256], [512, 1024], [512, 4096]],
                             repeats=32),
         derivation="Unchanged: c96 <= S, at most 96 x 9 blocks = 32% of the pool, so replacements are admitted "
                    "at once."),
    dict(index=9, name="rate_knee", purpose="Maximum sustainable open-loop request rate.",
         coverage=["arrival_rate_knee"], max_concurrency=C_SAT, capture_seconds=20.0, arrival="trace",
         workload_trace=dict(shapes=MIX, repeats=64),
         rate=dict(value=2.97, status="measured",
                   derived_from="queueing knee = 0.8 x the case-10 saturated request rate on the same shapes: "
                                "512 requests over a 137.87 s client run = 3.714 req/s (req-frontend "
                                "summary request_throughput_per_s, Slurm job 6451, this pack's serving config)",
                   evidence="logs/20261012_1_q38_align_campaign/10_full_saturation_rate/profile_workload"),
         derivation="GLM's knee rule (0.8 x the case-10 saturated rate) on this model's throughput; same 8-shape "
                    "mix and 512 requests. max_concurrency = case 10's c256 caps in-flight requests on both sides."),
    dict(index=10, name="full_saturation_rate", purpose="Peak-throughput open-loop load beyond the sustainable knee.",
         coverage=["saturation", "peak_throughput"], max_concurrency=C_SAT, capture_seconds=20.0,
         workload_trace=dict(shapes=MIX, repeats=64),
         derivation=f"GLM's c256. Mean terminal demand 256 x {mean_blocks(MIX):.2f} blocks = "
                    f"{demand(MIX, C_SAT):.0%} of the pool, so the long-output requests can transiently preempt: "
                    f"saturation in KV as well as compute. Same trace as case 9 so the knee is a fraction of "
                    f"this rate."),
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
         derivation="Unchanged: GLM already ran it at max_model_len 131072, and its longest row (65536) fits. "
                    "c8 x <= 48 blocks = 14% of the pool."),
    dict(index=14, name=f"near_full_long_decode_c{C14}", variant="kv14",
         purpose="About 91.4% terminal KV demand: near-full long decode.",
         coverage=["kv_near_full"], max_concurrency=C14, capture_seconds=120.0,
         workload_trace=dict(shapes=GLM14, repeats=32, id_suffix="full"),
         kernel_trace=dict(shapes=cycle(GLM14, [6, 1, 2, 3, 4, 5], C14, 256), repeats=1, id_suffix="nsys"),
         derivation=f"GLM's 91.4% = c x mean terminal blocks / pool, on GLM's shapes. Their mean holds "
                    f"{mean_blocks(GLM14):.2f} blocks (ceil(T/1568) + 6 GDN-state). c{C14} keeps the pinned pool "
                    f"well under the default one: {C14} x {mean_blocks(GLM14):.2f} / 0.914 = "
                    f"{KV14 // BLOCK_BYTES} blocks x 51,380,224 B = {KV14} B ({KV14 / GIB:.2f} GiB). repeats 10 -> "
                    f"32 keeps GLM's 3 requests per slot. NSYS trace: c requests at terminal - 256, so the window "
                    f"opens at the terminal KV state."),
    dict(index=15, name=f"kv_overcommit_mixed_churn_c{C15}", variant="kv15",
         purpose="About 122.6% average terminal KV demand, admission-controlled churn.",
         coverage=["kv_overcommit", "admission"], max_concurrency=C15, capture_seconds=180.0,
         workload_trace=dict(shapes=GLM15, repeats=12, id_suffix="full"),
         kernel_trace=dict(shapes=cycle(GLM15, [1, 2, 3, 4, 5, 6], C15, 256), repeats=1, id_suffix="nsys"),
         derivation=f"As case 14 at GLM's 122.6%: {C15} x {mean_blocks(GLM15):.2f} blocks / 1.226 = "
                    f"{KV15 // BLOCK_BYTES} blocks = {KV15} B ({KV15 / GIB:.2f} GiB). repeats 12 unchanged "
                    f"(72 requests, 2.25 per slot at c{C15}; ~0.86M decode tokens)."),
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
                   f"({chunk - 64}/{chunk}/{chunk + 64}) and to 2/3/3.5 chunks. Steps reach {chunk} tokens, above "
                   f"the 2048 graph ceiling, so they run piecewise-eager."))
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
        logged, evidence = PINNED_EVIDENCE[case["variant"]]
        return {
            "value": round(pinned / 1e9, 4),
            "status": "measured",
            "derived_from": f"--kv-cache-memory-bytes {pinned} of this case's variant pins the KV pool to "
                            f"{pinned // BLOCK_BYTES} blocks; the workload pass logs `Maximum concurrency` {logged}",
            "evidence": evidence,
        }
    chunk = case.get("chunk_size", CHUNK)
    measured = MEASURED_POOL[chunk]
    if measured is None:
        return {
            "value": round(DRAFT_POOL_GIB * GIB / 1e9, 4),
            "status": "provisional",
            "derived_from": f"the draft's `Available KV cache memory: {DRAFT_POOL_GIB} GiB` at max_num_seqs 8 / "
                            f"chunk 2048; replace with this deployment's chunk-{chunk} workload pass",
            "evidence": "",
        }
    gib, jobs, evidence = measured
    return {
        "value": round(gib * GIB / 1e9, 4),
        "status": "measured",
        "derived_from": f"`Available KV cache memory: {gib} GiB` (x 1.073741824) in the chunk-{chunk} workload "
                        f"passes (Slurm jobs {jobs}) at gpu_memory_utilization 0.85",
        "evidence": evidence,
    }


cases = []
for case in CASES:
    out = {
        "index": case["index"],
        "name": case["name"],
        "variant": {"kv14": "tp1_kv14", "kv15": "tp1_kv15"}.get(case.get("variant"), "tp1"),
        "purpose": case["purpose"],
        "coverage": case["coverage"],
        "max_model_len": L,
        "max_concurrency": case["max_concurrency"],
        "profile_arrival_mode": "trace-timed" if case.get("arrival") == "trace" else "saturated",
        "simulation_arrival_mode": "trace_timed" if case.get("arrival") == "trace" else "saturated",
        "gpu_memory_utilization": 0.85,
        "capture_seconds": case["capture_seconds"],
        "device_role": "primary",
        "workload_trace": trace_spec(case, "workload_trace"),
    }
    if "chunk_size" in case:
        out["chunk_size"] = case["chunk_size"]
    if "kernel_trace" in case:
        out["kernel_trace"] = trace_spec(case, "kernel_trace")
    if "rate" in case:
        out["rate"] = case["rate"]
    out["attn_gpu_memory_gb"] = kv_calibration(case)
    out["provenance"] = {
        "derived_from_pack": f"glm52_nvfp4_b200_tp8 case {case.get('base', case['index']):02d}",
        "derivation": case["derivation"],
    }
    for spec in (out["workload_trace"], out.get("kernel_trace")):
        for i, o in (spec or {}).get("shapes", []):
            assert i + o < L, (case["name"], i, o)
    cases.append(out)

document = {
    "schema_version": 1,
    "pack": "qwen38_27b_nvfp4_b200_tp1",
    "description": "Qwen3.8-27B-NVFP4 (nvidia/Qwen3.8-27B-NVFP4 482ca0f3) on one B200, TP1, under the vLLM fork "
                   "892da082 with FP8 KV, FP32 GDN state, prefix caching and vLLM's default max_num_seqs. The "
                   "fifteen operating points of glm52_nvfp4_b200_tp8 re-derived for this model's limits, plus "
                   "cases 04 and 07 at chunk 4096 and 8192 and case 10 at chunk 8192; see make_pack.py.",
    "recorded_on": {
        "host": "b200_raid",
        "gpu": "NVIDIA B200",
        "checkpoint_snapshots": {
            "qwen38_27b_nvfp4": f"models--nvidia--Qwen3.8-27B-NVFP4/snapshots/{REVISION}",
        },
        "nsys_version": "2025.5.2",
        "device_roles": {"primary": "0"},
    },
    # One duty-cycle multiplier for every case, measured outside the matrix: the diverse_100 capture's kernel-align
    # `recommended_gpu_time_multiplier` (sum of measured GPU cycle / sum of measured critical path, 3408 iterations).
    "gpu_time_multiplier": {
        "value": 1.0483,
        "status": "measured",
        "derived_from": "recommended_gpu_time_multiplier 1.04834 of the diverse_100 kernel-align (nsys job 6443, "
                        "3408 iterations, labels from this pack's rules); not fitted to any case",
        "evidence": "logs/20261013_2_q38_diverse100/analysis_kernel/reports/alignment_iteration_report.json",
    },
    "variants": {
        "tp1": variant(),
        "tp1_kv14": variant(["--kv-cache-memory-bytes", str(KV14)]),
        "tp1_kv15": variant(["--kv-cache-memory-bytes", str(KV15)]),
    },
    "cases": cases,
}
(HERE / "campaign.yaml").write_text(
    "# Generated by make_pack.py; edit that, not this.\n" + yaml.safe_dump(document, sort_keys=False, width=110)
)
print(len(cases), "cases; pool blocks", pool_blocks(), "kv14", KV14, "kv15", KV15)
for case in CASES:
    print(f"{case['index']:2d} {case['name']:40s} c{case['max_concurrency']:<4d} chunk {case.get('chunk_size', CHUNK)}"
          f" demand {demand(case['workload_trace']['shapes'], case['max_concurrency']):.2f}")

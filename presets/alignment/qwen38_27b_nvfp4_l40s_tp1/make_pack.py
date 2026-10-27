"""Write campaign.yaml for Qwen3.8-27B-NVFP4 on one L40S (TP1) under the pinned vLLM alignment fork.

Serving config: the vLLM Recipes best-practice deployment of nvidia/Qwen3.8-27B-NVFP4
(https://recipes.vllm.ai/nvidia/Qwen3.8-27B-NVFP4.json, fetched 2026-10-23). The recipe publishes NVFP4 renderings
for rtx_pro_6000 and dgx_spark_gb10 only; L40S has none, so the single-GPU rtx_pro_6000 rendering is the baseline:

    --kv-cache-dtype fp8 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192
    --enable-chunked-prefill --async-scheduling --enable-prefix-caching --tensor-parallel-size 1
    (+ --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)

Deviations, each because it does not reach a GPU kernel or because the alignment runner owns the flag:

- The reasoning / tool-call parsers and `VLLM_USE_RUST_FRONTEND=1` are API-frontend features. The `vllm_tokens`
  load generator sends token ids, so they are dropped; no engine-core launch changes.
- `--api-server-count 4` (skills/operate-run-alignment): one API process queues requests at the frontend.
- `--gpu-memory-utilization 0.92`: the recipe passes none and vLLM's logged default on this host is 0.92.
- CUDA graphs stay at the recipe's (vLLM's) default for max_num_seqs 8: FULL_AND_PIECEWISE, capture sizes
  [1, 2, 4, 8, 16] (logged by tmp/recipe-serve.log). This is deliberately NOT the skill's "ceiling = chunk" rule:
  the recipe is the deployment being studied, so every step above 16 tokens runs eagerly, as it does in production
  with this recipe. The simulated arch carries the same list.

What vLLM selects on sm_89 for this checkpoint (recipe server log): CutlassFP8ScaledMMLinearKernel for the FP8
per-tensor projections, MarlinNvFp4LinearKernel (W4A16) for the NVFP4 MLP and lm_head, Triton/FLA GDN prefill,
the CUDA fused GDN decode, and the FLASHINFER attention backend (FA2 path) over the FP8 KV cache.

Limits (TP1, chunk 8192): S = 8, L = 262144. KV pool (warm chunk-8192 passes): `Available KV cache memory:
19.94 GiB`, `GPU KV cache size: 626,735 tokens`; the cold recipe server logged 18.44 GiB / 580,031 tokens. vLLM's hybrid allocator raises the
attention block to 1568 tokens so its page reaches the FP32 mamba page, as on B200 (same page arithmetic as the
B200 pack): block = 16 x 1568 x 4 x 256 x 2 x 1 B = 51,380,224 B, and a request holds ceil(T / 1568) + 6 blocks.

Cases. The B200 matrix's operating points, re-derived for S = 8: every concurrency is capped at max_num_seqs, and
the chunk-boundary cases are rendered at the recipe's 8192 plus 2048 and 4096 (skills/operate-run-alignment).

Run from the repo root: uv run python presets/alignment/qwen38_27b_nvfp4_l40s_tp1/make_pack.py
"""
import math
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

S = 8
L = 262144
CHUNK = 8192
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"
# vLLM's default list for max_num_seqs 8 under FULL_AND_PIECEWISE (logged; max_cudagraph_capture_size 16).
CAPTURE_SIZES = [1, 2, 4, 8, 16]
# The graph-enabled variant (`tp1_graphs8k`): skills/operate-run-alignment's real-deployment rule, the CUDA graph
# ceiling at the chunk. vLLM's own list up to 2048 (vllm/config/vllm.py `_set_cudagraph_sizes`: [1, 2, 4], 8..248
# step 8, 256..2048 step 16), then one size every 128 up to 8192, passed explicitly so vLLM does not fill 2048..8192
# at its default stride of 16 (531 graphs). The simulated arch pads to the same list.
GRAPH_CAP = 8192
GRAPH_CAPTURE_SIZES = ([1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 2049, 16))
                       + list(range(2048 + 128, GRAPH_CAP + 1, 128)))
GPU_MEMORY_UTILIZATION = 0.92

ATTN_BLOCK_TOKENS = 1568
BLOCK_BYTES = 16 * ATTN_BLOCK_TOKENS * 4 * 256 * 2
MAMBA_BLOCKS_PER_REQUEST = 3 * 2
GIB = 1 << 30

# `Available KV cache memory` (GiB) logged at each chunk size: chunk -> (GiB, evidence). None until measured.
# Only warm passes are cited: a pass whose torch.compile cache was cold (compile ~40-75 s instead of ~0.5 s) logs a
# pool 1.4-1.5 GiB smaller (the recipe server and the first chunk-8192/2048/4096 captures), and is re-run warm.
MEASURED_POOL: dict[int, tuple[float, str] | None] = {
    8192: (19.94, "logs/20261023_1_q38_l40s_campaign/03_balanced_anchor_c8/profile_nsys (warm; 626,735 tokens)"),
    4096: (20.51, "logs/20261023_1_q38_l40s_campaign/17_prefill_spectrum_chunk4k_c8/profile_nsys (warm re-run)"),
    2048: (20.8, "logs/20261023_1_q38_l40s_campaign/18_quadrant_interference_chunk2k_c8/profile_nsys (warm; "
                 "653,853 tokens)"),
}


# `Available KV cache memory` (GiB) of the tp1_graphs8k variant; None until its workload pass is read.
GRAPH_POOL: tuple[float, str] | None = (
    15.64, "logs/20261023_1_q38_l40s_campaign/22_quadrant_interference_graphs8k_c8/profile_nsys and 21's "
           "profile_workload (warm; 491,143 tokens; graph capture takes 4.46 GiB at profiling)")


def pool_blocks(chunk: int = CHUNK) -> int:
    measured = MEASURED_POOL.get(chunk) or MEASURED_POOL[CHUNK]
    return int(measured[0] * GIB) // BLOCK_BYTES


def blocks(terminal: int) -> int:
    return math.ceil(terminal / ATTN_BLOCK_TOKENS) + MAMBA_BLOCKS_PER_REQUEST


def demand(shapes, concurrency: int, chunk: int = CHUNK) -> float:
    return concurrency * sum(blocks(i + o) for i, o in shapes) / len(shapes) / pool_blocks(chunk)


SERVER = {
    "tp_size": 1,
    "dp_size": 1,
    "enforce_eager": False,
    "enable_server_load_tracking": True,
    "api_server_count": 4,
    "served_model_name": "qwen38-27b-nvfp4",
    "extra_args": [
        "--revision", REVISION,
        "--tokenizer-revision", REVISION,
        "--language-model-only",
        "--kv-cache-dtype", "fp8",
        "--max-num-seqs", str(S),
        "--async-scheduling",
        "--enable-prefix-caching",
    ],
}
PYTHON_RUNTIME = {
    "packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
    "environment": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "2400",
    },
    "lock_timeout_seconds": 30,
    "install_timeout_seconds": 30,
}
ARCH = {
    "type": "qwen35_vllm_nvfp4_gdn_dense",
    "model_config": "model/config/qwen3_8_27b_nvfp4.json",
    "cudagraph_capture_sizes": CAPTURE_SIZES,
}
WORKER = {
    "type": "chunked_prefill",
    "max_batch_tokens": CHUNK,
    "batch_policy": "mix",
    "kv_admission_policy": "bounded-future",
    "kv_page_size": 1,
    "kv_max_future_tokens": 1,
    "kv_initial_new_token_ratio": 0.0,
    "kv_minimum_new_token_ratio": 0.0,
    "kv_new_token_ratio_decay_steps": 1,
    "kv_retract_decode_steps": 1,
    "decode_retraction_policy": "fcfs",
}


def variant(chunk: int = CHUNK, graphs: bool = False) -> dict:
    server = {**SERVER, "chunk_size": chunk}
    arch = ARCH
    if graphs:
        server = {**server, "max_cudagraph_capture_size": GRAPH_CAP,
                  "extra_args": SERVER["extra_args"] + ["--cudagraph-capture-sizes",
                                                        *map(str, GRAPH_CAPTURE_SIZES)]}
        arch = {**ARCH, "cudagraph_capture_sizes": GRAPH_CAPTURE_SIZES}
    return {
        "deployment": "unified",
        "gpu": "NVIDIA L40S",
        "replicas": 1,
        "checkpoint": "qwen38_27b_nvfp4",
        "engine": "vllm",
        "backend": "vllm_tokens",
        "run_name_prefix": "qwen38_27b_nvfp4_l40s_tp1",
        "python_runtime": PYTHON_RUNTIME,
        "server": server,
        "arch": arch,
        "worker": {**WORKER, "max_batch_tokens": chunk},
        "profile_passes": [
            {"kind": "nsys", "name": "profile_nsys", "trace": "kernel", "warmup": True},
            {"kind": "workload_metrics", "name": "profile_workload", "trace": "workload", "warmup": True},
        ],
        "input_builder": {"type": "engine_text", "measured_phase": "forward", "group_assignment": "single"},
        "label_rules": "label_rules/manifest.json",
    }


def prefill_spectrum_at(chunk: int) -> list[list[int]]:
    return [[p, 64] for p in (128, 512, chunk - 64, chunk, chunk + 64, 2 * chunk, 3 * chunk, 7 * chunk // 2)]


def quadrants_at(chunk: int) -> list[list[int]]:
    return [[128, 128], [128, 4096], [2 * chunk, 128], [3 * chunk // 2, 3072]]


MIX = [[128, 128], [128, 4096], [4096, 128], [3072, 3072], [512, 256], [2048, 1024], [6144, 512], [1024, 4096]]

CASES = [
    dict(index=3, name="balanced_anchor_c8", purpose="Comparable extension of the cross-model c32 anchor at S = 8.",
         coverage=["balanced_anchor"], max_concurrency=S, capture_seconds=20.0,
         workload_trace=dict(shapes=[[256, 128], [256, 256], [1024, 128], [1024, 256], [2048, 128],
                                     [2048, 256], [4096, 128], [4096, 256]], repeats=4),
         derivation="B200 case 03's eight shapes; concurrency capped at max_num_seqs 8, repeats 16 -> 4 (4 per slot)."),
    dict(index=4, name="prefill_spectrum_c8", purpose="Prefill scaling and the 8192-token chunk boundary.",
         coverage=["prefill_scaling", "chunk_boundary"], max_concurrency=S, capture_seconds=20.0,
         workload_trace=dict(shapes=prefill_spectrum_at(CHUNK), repeats=4),
         derivation="B200 case 04 rungs moved to the recipe's 8192 chunk (8128/8192/8256, 2/3/3.5 chunks); c8."),
    dict(index=5, name="decode_spectrum_c8", purpose="Decode residence, KV growth, and batch contraction.",
         coverage=["decode_residence", "kv_growth"], max_concurrency=S, capture_seconds=20.0,
         workload_trace=dict(shapes=[[128, 16], [128, 128], [128, 512], [128, 1024], [128, 2048], [128, 4096],
                                     [512, 2048], [1024, 4096]], repeats=2),
         derivation="B200 case 05's shapes at c8; every decode batch is <= 8, inside the FULL graphs [1..16]."),
    dict(index=7, name="quadrant_interference_c8", purpose="All four IO quadrants sharing continuous batches.",
         coverage=["io_quadrants", "continuous_batching"], max_concurrency=S, capture_seconds=20.0,
         workload_trace=dict(shapes=quadrants_at(CHUNK), repeats=4),
         derivation="B200 case 07's quadrants with the long prompts at 2 and 1.5 recipe chunks; c8."),
    dict(index=10, name="full_saturation_c8", purpose="Peak-throughput load at the recipe's max_num_seqs.",
         coverage=["saturation", "peak_throughput"], max_concurrency=S, capture_seconds=20.0,
         workload_trace=dict(shapes=MIX, repeats=4),
         derivation="B200 case 10's 8-shape mix at c8 = max_num_seqs, so the scheduler, not the client, binds."),
    dict(index=11, name="long_prefill_ladder_c1", purpose="Long-prefill ladder at concurrency 1.",
         coverage=["long_prefill", "bounded_kernel_trace"], max_concurrency=1, capture_seconds=120.0,
         workload_trace=dict(shapes=[[8192, 16], [16384, 64], [32768, 16], [65536, 64], [131072, 16],
                                     [229376, 64]], repeats=1, id_suffix="full"),
         kernel_trace=dict(shapes=[[8192, 16], [16384, 64], [32768, 16], [65536, 64]], repeats=1,
                           id_suffix="nsys"),
         derivation="B200 case 11's rungs as fractions of the recipe's 262144 limit (1/32 .. 7/8); the NSYS trace "
                    "stops at 1/4 of the limit to bound the capture."),
]
for index, chunk in ((16, 2048), (17, 4096)):
    CASES.append(dict(
        index=index, name=f"prefill_spectrum_chunk{chunk // 1024}k_c8", chunk_size=chunk, base=4,
        purpose=f"Prefill scaling and the {chunk}-token chunk boundary.",
        coverage=["prefill_scaling", "chunk_boundary", "chunk_size"], max_concurrency=S, capture_seconds=20.0,
        workload_trace=dict(shapes=prefill_spectrum_at(chunk), repeats=4),
        derivation=f"Case 04 at chunk {chunk}: rungs at {chunk - 64}/{chunk}/{chunk + 64} and 2/3/3.5 chunks."))
CASES.append(dict(
    index=18, name="quadrant_interference_chunk2k_c8", chunk_size=2048, base=7,
    purpose="All four IO quadrants sharing continuous batches at a 2048-token chunk.",
    coverage=["io_quadrants", "continuous_batching", "chunk_size"], max_concurrency=S, capture_seconds=20.0,
    workload_trace=dict(shapes=quadrants_at(2048), repeats=4),
    derivation="Case 07 at chunk 2048, B200 case 07's own shapes."))
# The recipe's cases 03 and 07 again on the graph-enabled variant: the same traces, so the pair isolates what the
# eager prefill and mixed steps cost.
for index, base in ((21, 3), (22, 7)):
    src = next(case for case in CASES if case["index"] == base)
    CASES.append(dict(
        src, index=index, name=src["name"].replace("_c8", "_graphs8k_c8"), base=base, graphs=True,
        coverage=src["coverage"] + ["cuda_graph_ceiling"],
        purpose=src["purpose"] + " CUDA graphs up to 8192 tokens.",
        derivation=f"Case {base:02d} on tp1_graphs8k: same trace, graph ceiling at the 8192 chunk."))


def trace_spec(case: dict, key: str) -> dict:
    stem = f"{case['index']:02d}_{case['name']}"
    return {"file": f"traces/{stem}{'.nsys' if key == 'kernel_trace' else ''}.csv", **case[key]}


def kv_calibration(case: dict) -> dict:
    chunk = case.get("chunk_size", CHUNK)
    # Capturing up to 8192-token graphs takes graph memory from the pool; until its own pass is read, provisional.
    measured = GRAPH_POOL if case.get("graphs") else MEASURED_POOL[chunk]
    if measured is None:
        gib, evidence = MEASURED_POOL[CHUNK]
        return {
            "value": round(gib * GIB / 1e9, 4),
            "status": "provisional",
            "derived_from": f"the chunk-{CHUNK} pool ({gib} GiB); replace with this chunk's workload pass",
            "evidence": evidence,
        }
    gib, evidence = measured
    return {
        "value": round(gib * GIB / 1e9, 4),
        "status": "measured",
        "derived_from": f"`Available KV cache memory: {gib} GiB` (x 1.073741824) at chunk {chunk}, "
                        f"gpu_memory_utilization {GPU_MEMORY_UTILIZATION}",
        "evidence": evidence,
    }


def variant_name(case: dict) -> str:
    if case.get("graphs"):
        return "tp1_graphs8k"
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
        "profile_arrival_mode": "saturated",
        "simulation_arrival_mode": "saturated",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "capture_seconds": case["capture_seconds"],
        "device_role": "primary",
        "workload_trace": trace_spec(case, "workload_trace"),
    }
    if "chunk_size" in case:
        out["chunk_size"] = case["chunk_size"]
    if "kernel_trace" in case:
        out["kernel_trace"] = trace_spec(case, "kernel_trace")
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
    "pack": "qwen38_27b_nvfp4_l40s_tp1",
    "description": "Qwen3.8-27B-NVFP4 (nvidia/Qwen3.8-27B-NVFP4 482ca0f3) on one L40S, TP1, as the vLLM Recipes "
                   "best-practice deployment (rtx_pro_6000 NVFP4 rendering: FP8 KV, max_num_seqs 8, chunk 8192, "
                   "max_model_len 262144, prefix caching, async scheduling) under the pinned vLLM alignment fork; "
                   "see make_pack.py.",
    "recorded_on": {
        "host": "l40s_local",
        "gpu": "NVIDIA L40S",
        "checkpoint_snapshots": {
            "qwen38_27b_nvfp4": f"models--nvidia--Qwen3.8-27B-NVFP4/snapshots/{REVISION}",
        },
        "nsys_version": "2026.1.0",
        "device_roles": {"primary": "0"},
    },
    # One duty-cycle multiplier for every case, measured outside the matrix: the kernel-align
    # `recommended_gpu_time_multiplier` of the recipe-benchmark reference capture (random 1024 in / 512 out, c8).
    "gpu_time_multiplier": {
        "value": 1.0515,
        "status": "measured",
        "derived_from": "recommended_gpu_time_multiplier 1.05152 of the recipe-benchmark reference kernel-align "
                        "(64 x [1024, 512] at c8, 514 iterations, labels from this pack's rules, 99.7% mapped "
                        "critical path); not fitted to any case",
        "evidence": "logs/20261023_2_q38_l40s_reference_recipe_bench/analysis_kernel/reports/"
                    "alignment_iteration_report.json",
    },
    "variants": {
        "tp1": variant(),
        "tp1_chunk4k": variant(4096),
        "tp1_chunk2k": variant(2048),
        "tp1_graphs8k": variant(graphs=True),
    },
    "cases": cases,
}
(HERE / "campaign.yaml").write_text(
    "# Generated by make_pack.py; edit that, not this.\n" + yaml.safe_dump(document, sort_keys=False, width=110)
)
print(len(cases), "cases; pool blocks", pool_blocks())
for case in CASES:
    print(f"{case['index']:2d} {case['name']:40s} c{case['max_concurrency']:<3d} chunk {case.get('chunk_size', CHUNK)}"
          f" demand {demand(case['workload_trace']['shapes'], case['max_concurrency']):.2f}")

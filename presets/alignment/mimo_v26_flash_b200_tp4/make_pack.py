"""Write campaign.yaml for MiMo-V2.6-Flash-RL on B200, TP4 attention / EP4 experts, from the GLM-5.2 matrix.

The fifteen operating points of `glm52_nvfp4_b200_tp8` keep their index, purpose and coverage: what each case
stresses. Their concurrency, rate and lengths are re-derived from MiMo's own limits below, the way the GLM pack
derived them from GLM's (`c x mean terminal length / KV pool` for the KV cases, the knee as 0.8 x the case-10
saturated rate, lengths as fractions of the context limit). Every case states its derivation in
`provenance.derivation`; `CASES` below is the one place it is computed.

Serving config. The server block is the validated capture's (Slurm job 6264,
`logs/20261012_1_mimo_fused_ar_nsys/profile-6264.yaml` in wt-mimo-v2-clean: vLLM fork 892da082, checkpoint
revision 5711b268, FlashInfer 0.7.0, FlashInfer all-reduce + RMSNorm fusion on `backend=mnnvl`) with two capture
settings changed, both from the first campaign run (logs/20261012_4_mimo_align_campaign, progress.md Step 6):

- `max_cudagraph_capture_size: 2048`, as the Inkling pack sets. Job 6264 left it to vLLM's default,
  min(2 x max_num_seqs, 1024) = 200 (vllm/config/vllm.py:2408-2420), so every mixed step above 200 tokens ran
  eager at a ~47 ms host floor that the simulator does not model.
- `warmup: true` on both passes, as `glm53_flash_fp8_b200_tp4_ep4` sets, plus a persistent FA4 CuTeDSL disk cache
  (`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`; the host's job wrapper exports `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`
  next to its other JIT caches, since the pack holds no host paths). Without them FA4 JIT-compiled each new kernel
  variant inside the measured window: 3.6-6 s stalls at iteration 0 and on later variants. Req-frontend's warmup
  (req-frontend/src/runner.rs:101-140) replays one request alone, then the pass's whole trace at the same
  max_concurrency with saturated arrivals and outputs capped at 32 tokens, so every prompt shape is prefilled
  (chunks with a nonzero start for prompts over 2048, contexts to the case's longest) and decoded at the case's
  batch sizes before the measurement boundary.

Only `--max-model-len` moves out of `extra_args`, because rendering appends it from each case (always 131072 here).
Cases 14 and 15 add `--kv-cache-memory-bytes` (their own variants), as `glm53_flash_fp8_b200_tp4_ep4` does;
see `kv_pinned_bytes`.

MiMo's limits (TP4/EP4, gpu_memory_utilization 0.85, chunk 2048):

- S = 100 (`--max-num-seqs`). The simulated chunked-prefill worker has no running-request cap, so no case offers
  more than S concurrent requests: above S, vLLM queues and the simulator would batch them all.
- CUDA graphs, FULL_AND_PIECEWISE: `capture_sizes()` below is vLLM 892da082's list for an explicit
  max_cudagraph_capture_size 2048 with max_num_batched_tokens 2048 (vllm/config/vllm.py:2499-2553): [1, 2, 4],
  8..248 step 8, 256..2048 step 16 = 147 sizes. An explicit ceiling skips the default branch that would insert
  max_num_seqs (vllm.py:2408-2498), so 100 is not captured: 97-100 running requests pad to 104
  (v1/cudagraph_dispatcher.py:139). Every size is a PIECEWISE graph; uniform decode also gets FULL graphs at
  the sizes <= S, i.e. up to 96 (cudagraph_dispatcher.py:216), so a decode step of 97-100 requests replays the
  104 PIECEWISE graph.
- L = 131072 (`--max-model-len`); rendering rejects a trace row with input + output >= L.
- KV pool. 110.52 GiB per GPU in the draft captures (jobs 4198/4220/4223: `GPU KV cache size: 15,566,522 tokens,
  Maximum concurrency for 131,072 tokens per request: 118.76x`). The exact serving config (job 6264, whose
  FlashInfer MNNVL workspace takes 0.45 GiB more) reports 110.07 GiB / 15,503,779 tokens / 118.28x, and with
  this pack's 2048-token graph ceiling 108.44 GiB / 15,273,913 tokens / 116.53x (case 01 workload pass, job 6360).
  vLLM's hybrid allocator holds 6 KV groups of 9 layers, one block id spanning 9 per-position tensors:
    page   = 20,480 B per layer per block: GA 1 KV head/rank x (192 K + 128 V) x 2 B x 32 tokens
             = SWA 2 KV heads/rank x 320 x 2 B x 16 tokens;
    groups = 1 GA (9 layers, 32-token blocks) + 5 SWA (39 layers + 6 padding layers, 16-token blocks);
    block  = 9 x 20,480 = 184,320 B, so 110.52 GiB = 643,826 blocks = 20.6M GA tokens.
  A request of terminal length T holds ceil(T/32) GA blocks plus ~5 x 9 = 45 SWA blocks (window 128 = 8 blocks of
  16, + 1 partial). vLLM's own 118.76x charges the SWA admission cap instead, 5 x cdiv(128 + 2 x 2048, 16) = 1,320:
  643,826 / (4,096 + 1,320) = 118.87x, within 0.1% of the logged 118.76x.
  So at S x (L - 1) every slot full at the context limit holds 100 x (4,096 + 45) = 414,100 blocks = 64% of the
  pool: no workload the server admits can fill it. That is why cases 14 and 15 pin a smaller pool.

Cases 16-19 repeat the two chunk-sensitive operating points (case 04's prefill spectrum and case 07's quadrant
mix) at chunk 4096 and 8192, so the matrix spans the scheduler token budget rather than one template value. The
per-case `chunk_size` sets both vLLM's --max-num-batched-tokens and the simulated worker's max_batch_tokens. The
capture list is unchanged: `capture_sizes(2048, chunk)` caps at the explicit 2048 ceiling and appends no chunk
above it, so mixed steps above 2048 tokens run eagerly in every chunk case. A larger token budget enlarges vLLM's
profile-run activation peak, so their KV pool is read from their own workload passes (`CHUNK_KV_POOL`).

Run from the repo root: uv run python presets/alignment/mimo_v26_flash_b200_tp4/make_pack.py
"""
import math
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- MiMo limits
S = 100
L = 131072
MAX_CUDAGRAPH_CAPTURE_SIZE = 2048
CHUNK = 2048


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
PAGE_BYTES = 20_480
BLOCK_BYTES = 9 * PAGE_BYTES
GA_TOKENS_PER_BLOCK = 32
SWA_BLOCKS_PER_REQUEST = 5 * (math.ceil(128 / 16) + 1)
GIB = 1 << 30

# The arch block matches presets/public/MiMo-V2.6-Flash-RL/mimo_v2_vllm_fp8_swa_moe.yaml (its diverse_100 member)
# except cudagraph_capture_sizes: the public preset keeps the diverse_100 capture's default list (ceiling 200).
ARCH_TYPE = "mimo_v2_vllm_fp8_swa_moe"


def blocks(terminal: int) -> int:
    """Pool blocks one running request of `terminal` tokens holds."""
    return math.ceil(terminal / GA_TOKENS_PER_BLOCK) + SWA_BLOCKS_PER_REQUEST


def kv_pinned_bytes(shapes, concurrency: int, ratio: float) -> int:
    """`--kv-cache-memory-bytes` that puts `concurrency` requests of the mean terminal length at `ratio` of the
    pool, the GLM-5.2 definition of a case's terminal KV demand."""
    mean = sum(blocks(i + o) for i, o in shapes) / len(shapes)
    return round(concurrency * mean / ratio) * BLOCK_BYTES


# ---------------------------------------------------------------- the variants
SERVER = {
    "tp_size": 4,
    "dp_size": 1,
    "expert_parallel_size": 4,
    "chunk_size": CHUNK,
    "max_cudagraph_capture_size": MAX_CUDAGRAPH_CAPTURE_SIZE,
    "enforce_eager": False,
    "enable_server_load_tracking": True,
    "api_server_count": 4,
    "served_model_name": "mimo-v26-flash-rl",
    "extra_args": [
        "--trust-remote-code",
        "--revision", "5711b268169967567844e1e560e8a3966da959b1",
        "--tokenizer-revision", "5711b268169967567844e1e560e8a3966da959b1",
        "--dtype", "bfloat16",
        "--kv-cache-dtype", "bfloat16",
        "--block-size", "16",
        "--language-model-only",
        "--enable-expert-parallel",
        "--expert-placement-strategy", "linear",
        "--no-enable-eplb",
        "--all2all-backend", "allgather_reducescatter",
        "--moe-backend", "flashinfer_trtllm",
        "--attention-backend", "FLASH_ATTN_DIFFKV",
        "--no-disable-hybrid-kv-cache-manager",
        "--generation-config", "vllm",
        "--max-num-seqs", str(S),
        "--quantization-config", '{"moe":{"activation":"mxfp8"}}',
    ],
}
PYTHON_RUNTIME = {
    # Job 6264's runtime block, plus the engine-ready timeout that block exported from its sbatch (a cold
    # torch.compile / inductor cache pushes engine startup past vLLM's 600 s default) and the FA4 disk cache.
    "packages": [{
        "name": "flashinfer-python",
        "version": "0.7.0",
        "index_url": "https://pypi.org/simple",
        "required_files": ["flashinfer/__init__.py"],
    }],
    "environment": {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
        # Persist FA4 CuTeDSL compiles across passes; the directory is the job wrapper's
        # FLASH_ATTENTION_CUTE_DSL_CACHE_DIR (a host path, so not in the pack).
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    },
    "lock_timeout_seconds": 30,
    "install_timeout_seconds": 30,
}
ARCH = {
    "type": ARCH_TYPE,
    "model_config": "model/config/mimo_v2_6_flash_rl.json",
    # FP8 block-quantized QKV / dense FFN; the routed experts are expert-parallel over the TP ranks by construction.
    "fp8": True,
    "tp_size": 4,
    # Recorded routes of the diverse_100 capture (job 4220, EPLB off, linear placement).
    "routing": "corpus",
    "token_corpus_file": "hf://datasets/UW-SyFI/servingstudio-workload@18506fd7afbaf925da2d31821287d079fffba5a6/"
                         "mimo_v26_flash_rl/vllm/diverse_100/capture/20261007/manifest.json",
    "cudagraph_capture_sizes": CAPTURE_SIZES,
}


def variant(extra_args=()) -> dict:
    server = dict(SERVER)
    server["extra_args"] = SERVER["extra_args"] + list(extra_args)
    return {
        "deployment": "unified",
        "gpu": "NVIDIA B200",
        "replicas": 1,
        "checkpoint": "mimo_v26_flash_rl",
        "engine": "vllm",
        # The validated capture's backend. vllm_tokens starts vLLM with skip_tokenizer_init, which this
        # config was never run with.
        "backend": "openai",
        "run_name_prefix": "mimo_v26_flash_b200_tp4_ep4",
        "python_runtime": PYTHON_RUNTIME,
        "server": server,
        "arch": dict(ARCH),
        "worker": {"type": "chunked_prefill", "max_batch_tokens": CHUNK, "batch_policy": "mix"},
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
KV14 = kv_pinned_bytes(GLM14, S, 0.914)
KV15 = kv_pinned_bytes(GLM15, S, 1.226)

CASES = [
    dict(index=1, name="micro_throughput_c100", purpose="Maximum request batching and scheduler pressure.",
         coverage=["batching", "scheduler_pressure"], max_concurrency=S, capture_seconds=15.0,
         workload_trace=dict(shapes=[[16, 16], [32, 32], [64, 64], [128, 64]], repeats=100),
         derivation="GLM c256 is the matrix's largest offered batch; MiMo's is S = 100 (--max-num-seqs), and the "
                    "simulated worker has no running cap above it. Shapes unchanged (KV-free); repeats 256 -> 100 "
                    "keeps GLM's 4 requests per slot."),
    dict(index=2, name="graph_boundaries_c100", purpose="CUDA graph and small-batch kernel boundaries.",
         coverage=["cuda_graph_boundaries", "small_batch"], max_concurrency=S, capture_seconds=15.0,
         workload_trace=dict(shapes=[[1, 32], [15, 32], [16, 32], [31, 32], [32, 32], [63, 64], [64, 64],
                                     [99, 64], [100, 64], [199, 64], [200, 64], [201, 64]], repeats=34),
         derivation="c = S: as the batch drains, decode replays every captured size 1..104. Prompts straddle "
                    "capture edges: 15/16, 31/32, 63/64 kept from GLM; 127/128 -> 99/100 and 255/256 -> "
                    "199/200/201, chosen against the first run's default list (100 inserted, top 200). Under "
                    "max_cudagraph_capture_size 2048 they still straddle 96/104 and 200/208. repeats 48 -> 34 keeps "
                    "~4.1 requests per slot. No analyze_iterations: the whole capture is the subject."),
    dict(index=3, name="balanced_anchor_c32", purpose="Comparable extension of the existing c32 anchor.",
         coverage=["balanced_anchor"], max_concurrency=32, capture_seconds=15.0,
         workload_trace=dict(shapes=[[256, 128], [256, 256], [1024, 128], [1024, 256], [2048, 128],
                                     [2048, 256], [4096, 128], [4096, 256]], repeats=16),
         derivation="The cross-model anchor: unchanged. c32 <= S; KV demand 32 x 4352 tokens = 0.7% of the pool."),
    dict(index=4, name="prefill_spectrum_c32", purpose="Prefill scaling and the 2048-token chunk boundary.",
         coverage=["prefill_scaling", "chunk_boundary"], max_concurrency=32, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 64], [512, 64], [1984, 64], [2048, 64], [2112, 64], [4096, 64],
                                     [6144, 64], [7168, 64]], repeats=16),
         derivation="Unchanged: MiMo's chunk is also 2048, so 1984/2048/2112 are still the boundary. c32 <= S."),
    dict(index=5, name="decode_spectrum_c64", purpose="Decode residence, KV growth, and batch contraction.",
         coverage=["decode_residence", "kv_growth"], max_concurrency=64, capture_seconds=15.0,
         workload_trace=dict(shapes=[[128, 16], [128, 128], [128, 512], [128, 1024], [128, 2048], [128, 4096],
                                     [512, 2048], [1024, 4096]], repeats=16),
         derivation="Unchanged: c64 <= S, and every decode context already starts at the 128-token sliding "
                    "window, so SWA attention is at its full window throughout while GA grows."),
    dict(index=6, name="long_long_c8",
         purpose="Long-prefill plus long-decode interaction near context capacity.",
         coverage=["long_prefill", "long_decode"], max_concurrency=8, capture_seconds=15.0,
         workload_trace=dict(shapes=[[32768, 4096], [49152, 3072], [65536, 3968], [81920, 2944], [98304, 1920]],
                             repeats=7),
         derivation="GLM's prompts are 25-75% of its 8192 context limit; MiMo serves 131072, so prompts x16 "
                    "(32K-96K). Outputs keep GLM's absolute 1920-4096: they set the decode step count, i.e. the "
                    "case's GPU time, and x16 would make it 16x longer. c8 <= S."),
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
         derivation="Unchanged: c96 <= S keeps 4 slots of headroom, so replacements are admitted at once."),
    dict(index=9, name="rate_knee", purpose="Maximum sustainable open-loop request rate.",
         coverage=["arrival_rate_knee"], max_concurrency=S, capture_seconds=20.0, arrival="trace",
         workload_trace=dict(shapes=MIX, repeats=64),
         rate=dict(value=3.34, status="measured",
                   derived_from="queueing knee = 0.8 x the case-10 saturated request rate on the same shapes: "
                                "512 requests over a 122.67 s measured client completion span = 4.17 req/s "
                                "(Slurm job 6364, this pack's serving config)",
                   evidence="logs/20261012_5_mimo_align_campaign_v2/10_full_saturation_rate/profile_workload"),
         derivation="GLM's knee rule (0.8 x the case-10 saturated rate) on MiMo's throughput; same 8-shape mix and "
                    "512 requests. max_concurrency S caps in-flight requests on both sides, so an arrival burst "
                    "queues at the client instead of exceeding vLLM's max_num_seqs."),
    dict(index=10, name="full_saturation_rate", purpose="Peak-throughput open-loop load beyond the sustainable knee.",
         coverage=["saturation", "peak_throughput"], max_concurrency=S, capture_seconds=20.0,
         workload_trace=dict(shapes=MIX, repeats=64),
         derivation="GLM c256 -> S = 100, the engine's own batch ceiling (above it vLLM queues and the simulator "
                    "would not). Same trace as case 9 so the knee is a fraction of this rate."),
    dict(index=11, name="long_prefill_ladder_c1", purpose="4K-120K prompts: long-prefill ladder at concurrency 1.",
         coverage=["long_prefill", "bounded_kernel_trace"], max_concurrency=1, capture_seconds=120.0,
         workload_trace=dict(shapes=[[4096, 16], [8192, 64], [16384, 16], [24576, 64], [32768, 16], [49152, 64],
                                     [65536, 16], [98304, 64], [114688, 16], [122880, 64]], repeats=2,
                             id_suffix="full"),
         kernel_trace=dict(shapes=[[4096, 16], [8192, 64], [16384, 16], [24576, 64], [32768, 16], [49152, 64],
                                   [65536, 16]], repeats=1, id_suffix="nsys"),
         derivation="GLM's rungs are 1/32 .. 15/16 of its 524288 limit; the same fractions of 131072 (each "
                    "rung / 4). The NSYS trace is GLM's choice too: the rungs up to half the limit, once."),
    dict(index=12, name="ultra_context_decode_c1",
         purpose="Long context plus long output: ultra-context decode at concurrency 1.",
         coverage=["ultra_context", "long_decode"], max_concurrency=1, capture_seconds=120.0,
         workload_trace=dict(shapes=[[p, o] for p in (114688, 65536, 32768, 8192, 2048) for o in (256, 1024, 4096)],
                             repeats=1, id_suffix="full"),
         kernel_trace=dict(shapes=[[114688, 64], [2048, 256], [3072, 64]], repeats=1, id_suffix="nsys"),
         derivation="Prompts are GLM's / 4 (same fractions of the context limit; the top, 114688 + 4096, is 88% "
                    "of 131072 as GLM's 462848 is of 524288). Outputs 256/1024/4096 unchanged: at c1 they are "
                    "decode step counts, not capacity."),
    dict(index=13, name="generation_length_matrix_c8",
         purpose="Mixed prompt and output lengths across a generation-length matrix.",
         coverage=["generation_length_matrix"], max_concurrency=8, capture_seconds=60.0,
         workload_trace=dict(shapes=[[p, o] for p in (128, 512, 2048, 8192, 16384, 32768)
                                     for o in (4096, 8192, 16384, 32768)], repeats=1, id_suffix="full"),
         kernel_trace=dict(shapes=[[128, 256], [512, 256], [2048, 256], [8192, 256], [16384, 256], [32768, 256],
                                   [4096, 256], [24576, 256]], repeats=1, id_suffix="nsys"),
         derivation="Unchanged: GLM already ran it at max_model_len 131072, and its longest row (65536) fits. "
                    "c8 <= S."),
    dict(index=14, name="near_full_long_decode_c100", variant="kv14",
         purpose="About 91.4% terminal KV demand: near-full long decode.",
         coverage=["kv_near_full"], max_concurrency=S, capture_seconds=120.0,
         workload_trace=dict(shapes=GLM14, repeats=48, id_suffix="full"),
         kernel_trace=dict(shapes=cycle(GLM14, [6, 1, 2, 3, 4, 5], S, 256), repeats=1, id_suffix="nsys"),
         derivation=f"GLM's 91.4% = c x mean terminal / pool. MiMo's pool at 0.85 cannot be filled (S x L is 64% "
                    f"of it), so the shapes stay GLM's, c rises to S, and the pool is pinned: 100 x 973 blocks "
                    f"(mean ceil(T/32) 928 + 45 SWA) / 0.914 = {KV14 // BLOCK_BYTES} blocks x 184,320 B = "
                    f"{KV14} B ({KV14 / GIB:.2f} GiB). repeats 10 -> 48 keeps GLM's 2.9 requests per slot. NSYS "
                    f"trace: S requests at terminal - 256, so the window opens at the terminal KV state."),
    dict(index=15, name="kv_overcommit_mixed_churn_c100", variant="kv15",
         purpose="About 122.6% average terminal KV demand, admission-controlled churn.",
         coverage=["kv_overcommit", "admission"], max_concurrency=S, capture_seconds=180.0,
         workload_trace=dict(shapes=GLM15, repeats=34, id_suffix="full"),
         kernel_trace=dict(shapes=cycle(GLM15, [1, 2, 3, 4, 5, 6], S, 256), repeats=1, id_suffix="nsys"),
         derivation=f"As case 14 at GLM's 122.6%: 100 x 1826 blocks (mean ceil(T/32) 1781 + 45 SWA) / 1.226 = "
                    f"{KV15 // BLOCK_BYTES} blocks = {KV15} B ({KV15 / GIB:.2f} GiB). repeats 12 -> 34 is ~2 "
                    f"requests per slot, not GLM's 5: at c100 the 5-slot trace is ~6.1M decode tokens (~45 min per "
                    f"pass). The NSYS trace prefills ~5.7M tokens (~130 s) before its decode steps, hence 180 s."),
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
                   f"({chunk - 64}/{chunk}/{chunk + 64}) and to 2/3/3.5 chunks. Steps reach {chunk} tokens, "
                   f"{'inside' if chunk <= 4096 else 'above'} the TP4 fused all-reduce + RMSNorm cap of 4096."))
for index, chunk in ((18, 4096), (19, 8192)):
    CASES.append(dict(
        index=index, name=f"quadrant_interference_chunk{chunk // 1024}k_c48", chunk_size=chunk, base=7,
        purpose=f"All four IO quadrants sharing continuous batches at a {chunk}-token chunk.",
        coverage=["io_quadrants", "continuous_batching", "chunk_size"], max_concurrency=48, capture_seconds=15.0,
        workload_trace=dict(shapes=quadrants_at(chunk), repeats=24),
        derivation=f"Case 07 at chunk {chunk}: same c48 and 24 repeats; the long prompts stay 2 and 1.5 chunks "
                   f"({2 * chunk} and {3 * chunk // 2}), so mixed steps pair one full chunk with up to 47 decodes."))


def trace_spec(case: dict, key: str) -> dict:
    spec = dict(case[key])
    stem = f"{case['index']:02d}_{case['name']}"
    spec = {"file": f"traces/{stem}{'.nsys' if key == 'kernel_trace' else ''}.csv", **spec}
    return spec


# Per-GPU KV pool logged by the chunk 4096 / 8192 workload passes (both cases of a chunk agree).
CHUNK_KV_POOL = {4096: (108.29, "6429, 6431"), 8192: (107.72, "6425, 6427")}


def kv_calibration(case: dict) -> dict:
    if case.get("variant") == "kv14":
        pinned = KV14
    elif case.get("variant") == "kv15":
        pinned = KV15
    elif case.get("chunk_size", CHUNK) != CHUNK:
        gib, jobs = CHUNK_KV_POOL[case["chunk_size"]]
        return {
            "value": round(gib * GIB / 1e9, 4),
            "status": "measured",
            "derived_from": f"`Available KV cache memory: {gib} GiB` (x 1.073741824) in the chunk-{case['chunk_size']} "
                            f"workload passes (Slurm jobs {jobs}); a larger --max-num-batched-tokens raises vLLM's "
                            "profile-run peak, so the pool is below chunk 2048's 108.44 GiB",
            "evidence": f"logs/20261012_7_mimo_chunk_cases/{case['index']:02d}_{case['name']}/profile_workload",
        }
    else:
        return {
            "value": round(108.44 * GIB / 1e9, 4),
            "status": "measured",
            "derived_from": "`Available KV cache memory: 108.44 GiB` (x 1.073741824) in the case-01 workload pass "
                            "(Slurm job 6360) of this serving config at gpu_memory_utilization 0.85, 1.63 GiB "
                            "below job 6264's 110.07 GiB at the default 200-token graph ceiling",
            "evidence": "logs/20261012_5_mimo_align_campaign_v2/01_micro_throughput_c100/profile_workload",
        }
    return {
        "value": round(pinned / 1e9, 4),
        "status": "provisional",
        "derived_from": f"--kv-cache-memory-bytes {pinned} of this case's variant pins the per-GPU KV pool; confirm "
                        "against `Available KV cache memory` in this case's workload pass",
        "evidence": "",
    }


cases = []
for case in CASES:
    out = {
        "index": case["index"],
        "name": case["name"],
        "variant": {"kv14": "tp4_ep4_kv14", "kv15": "tp4_ep4_kv15"}.get(case.get("variant"), "tp4_ep4"),
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
    "pack": "mimo_v26_flash_b200_tp4",
    "description": "MiMo-V2.6-Flash-RL on B200, TP4 attention / EP4 experts / DP1 under the vLLM fork 892da082, "
                   "served exactly as the validated fused all-reduce capture (job 6264). The fifteen operating "
                   "points of glm52_nvfp4_b200_tp8 re-derived for MiMo's limits, plus cases 04 and 07 at chunk "
                   "4096 and 8192; see make_pack.py.",
    "recorded_on": {
        "host": "b200_raid",
        "gpu": "NVIDIA B200",
        "checkpoint_snapshots": {
            "mimo_v26_flash_rl": "models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/5711b268169967567844e1e560e8a3966da959b1",
        },
        "nsys_version": "2025.5.2",
        "device_roles": {"primary": "0,1,2,3"},
    },
    "variants": {
        "tp4_ep4": variant(),
        "tp4_ep4_kv14": variant(["--kv-cache-memory-bytes", str(KV14)]),
        "tp4_ep4_kv15": variant(["--kv-cache-memory-bytes", str(KV15)]),
    },
    "cases": cases,
}
(HERE / "campaign.yaml").write_text(
    "# Generated by make_pack.py; edit that, not this.\n" + yaml.safe_dump(document, sort_keys=False, width=110)
)
print(len(cases), "cases; kv14", KV14, "kv15", KV15)

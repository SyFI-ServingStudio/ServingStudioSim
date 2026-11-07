"""Write campaign.yaml for DeepSeek-V4.1-Flash-1017 + DSpark on 4x B200 under the vLLM fork (MRV2).

Two topologies over the same four GPUs, both with expert parallelism over all four ranks:

- TP4/EP4: tensor-parallel attention, one scheduler (`tp_size 4, dp_size 1`).
- DP4/EP4: data-parallel attention, four schedulers, experts all-gathered over the DP ranks
  (`tp_size 1, dp_size 4`).

Each topology is crossed with three scheduler token budgets (chunk 2048 / 4096 / 8192), and each budget
pairs with its own `--max-num-seqs` (64 / 32 / 128) so the matrix also spans the request cap. A variant is
one (topology, chunk, max_num_seqs) server: the CUDA-graph capture list is a server setting, and
`max_cudagraph_capture_size` must equal the chunk, so chunk cannot be a per-case value here.

Cases per topology (same shapes on both, so TP4 and DP4 compare case by case):

- prefill-heavy at every chunk (long prompts, 32-128 output tokens): full-budget prefill chunks, the
  ragged last chunk, and chunk starts deep into a long context.
- mixed at every chunk: the four IO quadrants sharing continuous batches (chunk 2048 and 4096), and a
  short-request churn at concurrency 128 that fills the 128-request cap (chunk 8192).
- long-context, decode-heavy, low concurrency: the target workload, long agent sessions. Prompts of
  16K-123K tokens with 512-4096 output tokens at concurrency 1, 4 and 8, so decode runs against a long
  KV and DP4 runs with idle ranks.

Serving config. The server is the one the manual captures validated
(logs/20261106_0_dsv41_dspark_capture/make_configs.py, jobs 17900-17989): MRV2
(`VLLM_USE_V2_MODEL_RUNNER=1`), block 128, fp8 KV, engram CPU offload, tokenizer mode `deepseek_v41`,
DSpark `num_speculative_tokens 5` (verify width 6), gpu_memory_utilization 0.9, --max-model-len 131072
on every case (rendered from the case). Changed per the alignment skill:

- `api_server_count: 4`.
- `max_cudagraph_capture_size` = chunk and an explicit capture list (`capture_sizes`): vLLM's legacy list
  up to 2048 ([1, 2, 4], 8..248 step 8, 256..2048 step 16), then every 128 up to the chunk. Under MRV2 the
  two legacy fields still set the graphs: the list becomes the PIECEWISE (prefill/mixed) token sizes, and
  the FULL decode request counts are derived from it as `min(ceil(s / 6), max_num_seqs)` for every `s`
  with `ceil(s / 6) * 6 <= chunk` (vllm/config/vllm.py `_split_cudagraph_capture_sizes`). The arch block
  carries the same token list.
- `warmup: true` on the timed passes (req-frontend's bounded warmup + prefix-cache reset), so no
  DeepGEMM/Triton JIT lands inside the measured window. The trace warmup does not reach every M bucket of the
  DSpark draft's eager MXFP8 context projections, whose CuTe DSL tactics JIT-compile on first use per process
  (logs/20261107_0_dsv41_arch_design/dp4_host_stalls.md); the job wrapper therefore also sends one prompt per
  FlashInfer M bucket up to the chunk, plus one past 32K, to every DP rank before measurement
  (logs/20261107_1_dsv41_dspark_align_pack/warm_draft_buckets.py), and `--jit-monitor-verbose` logs every
  in-inference compile so a pass with one inside the measured window is found and rerun.

Calibrated case values start `provisional` and are replaced from each case's own workload pass:
`speculative_acceptance` from `spec_decode_metrics.json`, `attn_gpu_memory_gb` from the server log's
KV block count x block bytes.

Run from the repo root: uv run python presets/alignment/dsv41_flash_b200_dspark/make_pack.py
"""
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
GIB = 1 << 30
L = 131072  # --max-model-len on every case
VERIFY = 6  # DSpark num_speculative_tokens 5 + 1
DRAFT = 5
EVIDENCE_DIR = "logs/20261107_1_dsv41_dspark_align_pack/runs"


def capture_sizes(chunk: int) -> list[int]:
    """vLLM's legacy explicit list up to 2048, then one size every 128 up to the chunk."""
    sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, min(chunk, 2048) + 1, 16))
    sizes += list(range(2048 + 128, chunk + 1, 128))
    assert sizes[-1] == chunk
    return sizes


def decode_request_sizes(chunk: int, max_num_seqs: int) -> list[int]:
    """The FULL decode request counts MRV2 derives from the legacy list (for the record, not rendered)."""
    return sorted({min(-(-s // VERIFY), max_num_seqs) for s in capture_sizes(chunk)
                   if -(-s // VERIFY) * VERIFY <= chunk})


TOPOLOGIES = {
    "tp4": {"tp_size": 4, "dp_size": 1},
    "dp4": {"tp_size": 1, "dp_size": 4},
}
# chunk -> max_num_seqs
CHUNKS = {2048: 64, 4096: 32, 8192: 128}

PYTHON_RUNTIME = {
    "packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
    "environment": {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        # DP4 loads four engine cores' weights and engram tables; the 600 s default expired mid-load
        # (job 17904). A cold torch.compile + FlashInfer autotune also runs ~17 min.
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    },
}

# Provisional chains: conditional per-position acceptance of the manual diverse_100 captures (jobs 17986 TP4,
# 17987 DP4), replaced per case from its own workload pass.
PROVISIONAL_ACCEPTANCE = {
    "tp4": ([0.7957, 0.802, 0.8402, 0.8661, 0.8805],
            "logs/20261106_0_dsv41_dspark_capture/profile_tp4ep4_diverse_workload/spec_decode_metrics.json"),
    "dp4": ([0.7732, 0.783, 0.8116, 0.8405, 0.8605],
            "logs/20261106_0_dsv41_dspark_capture/profile_dp4ep4_diverse_workload/spec_decode_metrics.json"),
}
# Provisional per-rank KV pool: the same captures' "Extended KV cache to N blocks (X GiB)".
PROVISIONAL_KV_GIB = {"tp4": 80.26, "dp4": 69.64}

MEASURED_PATH = HERE / "measured.json"
MEASURED = json.loads(MEASURED_PATH.read_text()) if MEASURED_PATH.exists() else {}


def variant_name(topo: str, chunk: int) -> str:
    return f"{topo}_c{chunk}_s{CHUNKS[chunk]}"


def variant(topo: str, chunk: int) -> dict:
    seqs = CHUNKS[chunk]
    sizes = capture_sizes(chunk)
    server = {
        **TOPOLOGIES[topo],
        "expert_parallel_size": 4,
        "expert_count_reduction_group_size": 4,
        "chunk_size": chunk,
        "max_cudagraph_capture_size": chunk,
        "enable_server_load_tracking": True,
        "api_server_count": 4,
        "served_model_name": "dsv41-flash",
        "extra_args": [
            "--enable-expert-parallel", "--trust-remote-code",
            "--tokenizer-mode", "deepseek_v41",
            "--block-size", "128",
            "--max-num-seqs", str(seqs),
            "--kv-cache-dtype", "fp8",
            "--seed", "0",
            "--generation-config", "vllm",
            # Log every in-inference JIT compile (not just the first), to audit the measured window.
            "--jit-monitor-verbose",
            "--engram-config", '{"cpu_offload": true}',
            "--cudagraph-capture-sizes", *[str(s) for s in sizes],
            "--speculative-config", json.dumps({"method": "dspark", "num_speculative_tokens": DRAFT},
                                               separators=(",", ":")),
        ],
    }
    arch = {
        # Lane C's DSpark wrapper (design D4) over the TP4 or the DP4 target; the tag and its DP field
        # follow that lane's merge.
        "type": "deepseek_v41_vllm_dspark",
        "model_config": "model/config/deepseek_v41_flash.json",
        "fp8": True,
        "num_attn_dp_groups": TOPOLOGIES[topo]["dp_size"],
        "routing": "corpus",
        # Old-vLLM diverse_100 routes; replaced by this pack's token_corpus capture once uploaded.
        "token_corpus_file": "hf://datasets/UW-SyFI/servingstudio-workload@c3f5ecaab0bbff757c64864bf2ec24f5e1f76b5c/"
                             "deepseek_v41_flash/vllm/diverse_100/capture/20260924/manifest.json",
        "draft_tokens": DRAFT,
        "cudagraph_capture_sizes": sizes,
    }
    worker = {
        "type": "speculative",
        "max_batch_tokens": chunk,
        "batch_policy": "mix",
        "draft_tokens": DRAFT,
        "acceptance_seed": 73,
        # vLLM's KV discipline (see glm53_nvfp4_b200_dflash2): admit on prompt + first verify window,
        # claim running decodes' next window first, preempt the most recent admission on a shortfall.
        "kv_admission_policy": "bounded-future",
        "kv_page_size": 1,
        "kv_max_future_tokens": VERIFY,
        "kv_initial_new_token_ratio": 0.0,
        "kv_minimum_new_token_ratio": 0.0,
        "kv_new_token_ratio_decay_steps": 1,
        "kv_retract_decode_steps": 1,
        "decode_retraction_policy": "fcfs",
    }
    return {
        "deployment": "unified",
        "gpu": "NVIDIA B200",
        "replicas": 1,
        "checkpoint": "dsv41_flash_1017",
        "engine": "vllm",
        # The validated captures' backend (vllm_tokens would start vLLM with skip_tokenizer_init).
        "backend": "openai",
        "run_name_prefix": f"dsv41_dspark_b200_{topo}",
        "python_runtime": PYTHON_RUNTIME,
        "server": server,
        "arch": arch,
        "worker": worker,
        "profile_passes": [
            {"kind": "workload_metrics", "name": "profile_workload", "trace": "workload", "warmup": True},
            {"kind": "nsys", "name": "profile_nsys", "trace": "workload", "warmup": True},
            {"kind": "token_corpus", "name": "profile_corpus", "trace": "workload"},
        ],
        "input_builder": {
            "type": "speculative_engine_text",
            "draft_tokens": DRAFT,
            "measured_phase": "forward",
            "group_assignment": "per_dp_rank",
        },
        "label_rules": "label_rules/manifest.json",
    }


# ---------------------------------------------------------------- workloads
PREFILL_HEAVY = [(1024, 32), (2047, 32), (2048, 64), (3000, 32), (4096, 64), (6000, 32), (8192, 64),
                 (12288, 32), (16384, 128), (32768, 64), (65536, 32)]
QUADRANT = [(128, 128), (128, 4096), (4096, 128), (3072, 3072)]
CHURN = [(16, 64), (64, 128), (256, 64), (512, 256), (1024, 32), (128, 512), (2048, 128), (32, 1024)]
AGENT_C1 = [(65536, 4096), (122880, 2048), (16384, 4096), (98304, 3072)]
AGENT_C4 = [(32768, 2048), (65536, 2048), (98304, 1024), (122880, 1024), (49152, 4096), (81920, 512)]
AGENT_C8 = [(16384, 1024), (49152, 512), (81920, 2048), (110592, 1024), (24576, 4096), (65536, 1024),
            (8192, 2048), (122880, 512)]

# (index within topology, name, chunk, purpose, coverage, shapes, repeats, concurrency, capture_seconds)
CASES = [
    (1, "prefill_heavy_c16", 2048, "Prefill chunks at the 2048 budget, ragged last chunks, deep chunk starts.",
     ["prefill_scaling", "chunk_boundary"], PREFILL_HEAVY, 3, 16, 20.0),
    (2, "prefill_heavy_c16", 4096, "Prefill chunks at the 4096 budget; PIECEWISE graphs above 2048 tokens.",
     ["prefill_scaling", "chunk_boundary"], PREFILL_HEAVY, 3, 16, 20.0),
    (3, "prefill_heavy_c16", 8192, "Prefill chunks at the 8192 budget; PIECEWISE graphs up to 8192 tokens.",
     ["prefill_scaling", "chunk_boundary"], PREFILL_HEAVY, 3, 16, 20.0),
    (4, "quadrant_c48", 2048, "Four IO quadrants sharing continuous batches at the 2048 budget.",
     ["io_quadrants", "continuous_batching"], QUADRANT, 12, 48, 30.0),
    (5, "quadrant_c32", 4096, "Four IO quadrants at the 4096 budget, at the 32-request cap.",
     ["io_quadrants", "continuous_batching", "request_cap"], QUADRANT, 8, 32, 30.0),
    (6, "churn_c128", 8192, "Short-request churn filling the 128-request cap; wide decode graphs.",
     ["completion_churn", "request_cap", "batching"], CHURN, 32, 128, 20.0),
    (7, "agent_long_c1", 8192, "One long agent session at a time: 16K-123K prompts, long decode.",
     ["long_context", "long_decode", "low_concurrency"], AGENT_C1, 1, 1, 30.0),
    (8, "agent_long_c4", 2048, "Four concurrent long sessions: 33K-123K prompts, 512-4096 outputs.",
     ["long_context", "long_decode", "low_concurrency"], AGENT_C4, 2, 4, 30.0),
    (9, "agent_long_c8", 4096, "Eight concurrent long sessions: 8K-123K prompts, 512-4096 outputs.",
     ["long_context", "long_decode", "mixed"], AGENT_C8, 2, 8, 30.0),
]


def case(topo: str, offset: int, spec) -> dict:
    index, name, chunk, purpose, coverage, shapes, repeats, conc, capture = spec
    number = offset + index
    name = f"{topo}_{name}_k{chunk}"
    slug = f"{number:02d}_{name}"
    measured = MEASURED.get(slug, {})
    if "acceptance" in measured:
        acceptance = {
            "value": measured["acceptance"],
            "status": "measured",
            "derived_from": "Conditional accepted-token count ratios from this case's warm workload pass: "
                            "position 0 / num_drafts, then position i / position i-1.",
            "evidence": f"{EVIDENCE_DIR}/{slug}/profile_workload/spec_decode_metrics.json",
        }
    else:
        chain, source = PROVISIONAL_ACCEPTANCE[topo]
        acceptance = {"value": chain, "status": "provisional",
                      "derived_from": "diverse_100 manual capture of the same server at chunk 2048; "
                                      "replace from this case's workload pass", "evidence": source}
    if "kv_blocks" in measured:
        kv = {
            "value": round(measured["kv_bytes"] / 1e9, 9),
            "status": "measured",
            "derived_from": f"Smallest rank's KV pool: {measured['kv_blocks']} blocks "
                            f"({measured['kv_gib']} GiB) per the server log; GPU KV cache size "
                            f"{measured['kv_tokens']:,} tokens.",
            "evidence": f"{EVIDENCE_DIR}/{slug}/profile_workload/vllm/",
        }
    else:
        kv = {"value": round(PROVISIONAL_KV_GIB[topo] * GIB / 1e9, 6), "status": "provisional",
              "derived_from": "chunk-2048 manual capture's per-rank KV pool; replace from this case's "
                              "workload server log"}
    return {
        "index": number,
        "name": name,
        "variant": variant_name(topo, chunk),
        "purpose": purpose,
        "coverage": coverage,
        "max_model_len": L,
        "max_concurrency": conc,
        "chunk_size": chunk,
        "speculative_acceptance": acceptance,
        "profile_arrival_mode": "saturated",
        "simulation_arrival_mode": "saturated",
        "gpu_memory_utilization": 0.9,
        "capture_seconds": capture,
        "device_role": "primary",
        "workload_trace": {
            "file": f"traces/{slug}.csv",
            "shapes": [list(s) for s in shapes],
            "repeats": repeats,
        },
        "attn_gpu_memory_gb": kv,
    }


def main() -> None:
    variants = {variant_name(t, c): variant(t, c) for t in TOPOLOGIES for c in CHUNKS}
    cases = [case("tp4", 0, spec) for spec in CASES] + [case("dp4", len(CASES), spec) for spec in CASES]
    pack = {
        "schema_version": 1,
        "pack": "dsv41_flash_b200_dspark",
        "description": "DeepSeek-V4.1-Flash-1017 with the DSpark proposer (5 drafted tokens, verify width 6) "
                       "under the vLLM fork's model runner V2 on 4x B200, TP4/EP4 and DP4/EP4. Chunk 2048/4096/"
                       "8192 x prefill-heavy, mixed, and long-context low-concurrency agent sessions; "
                       "max_num_seqs 64/32/128. See make_pack.py.",
        "recorded_on": {
            "host": "b200_cayenne",
            "gpu": "NVIDIA B200",
            "checkpoint_snapshots": {
                "dsv41_flash_1017": "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
                                    "1bae0e1088993b9b82de08f59de5be5a34e42170",
            },
            "nsys_version": "2025.5.2",
            "device_roles": {"primary": "0,1,2,3"},
        },
        "variants": variants,
        "cases": cases,
    }
    text = "# Generated by make_pack.py; edit that, not this.\n" + yaml.safe_dump(
        pack, sort_keys=False, width=120)
    (HERE / "campaign.yaml").write_text(text)
    for chunk, seqs in CHUNKS.items():
        print(f"chunk {chunk}: {len(capture_sizes(chunk))} token sizes, "
              f"{len(decode_request_sizes(chunk, seqs))} decode request sizes")


if __name__ == "__main__":
    main()

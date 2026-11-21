"""Write campaign.yaml for DeepSeek-V4.1-Flash-1017 without speculative decoding, vLLM DP4/EP4 on 4x B200.

The non-speculative sibling of dsv41_flash_b200_dspark's DP4 topology: data-parallel attention (four engine
cores, TP1 each) and the routed experts over the same four ranks (`tp_size 1, dp_size 4`,
`--enable-expert-parallel`). Every decode step is one row per request, so decode-only steps replay the
FULL graphs of the plain request count, not DSpark's verify width 6. Nothing here is read from the DSpark
captures: the routing corpus, the KV pool and the duty cycle come from this pack's own passes.

Serving config: the DSpark pack's chunk-2048 DP4 server (same reference image, same patches, MRV2
`VLLM_USE_V2_MODEL_RUNNER=1`, block 128, fp8 KV, engram CPU offload, tokenizer mode `deepseek_v41`,
`--max-num-seqs 64`, gpu_memory_utilization 0.9, --max-model-len 131072) without `--speculative-config`.
Per the alignment skill: `api_server_count: 4`, `max_cudagraph_capture_size` = chunk 2048 with an explicit
capture list (vLLM's list up to 2048: [1, 2, 4], 8..248 step 8, 256..2048 step 16). That list is the
arch's default `cudagraph_capture_sizes` (`vllm_default_cudagraph_capture_sizes(2048)`), so the public
preset's arch, which sets no list, prices the same graphs. Under MRV2 the FULL decode request counts are
`min(s, max_num_seqs)` for every listed `s`, the arch's `DeepseekV41CudaGraphs` with query length 1.
`warmup: true` and `--jit-monitor-verbose` so a JIT compile inside the measured window is logged.

Worker: chunked prefill at 2048 tokens per DP rank, requests placed on the least-loaded rank as vLLM's DP
coordinator does. Default (full-footprint) KV admission, as the DSpark DP4 case: bounded-future admission
is validated for one attention partition only (worker/admission/chunked_prefill_admission.rs).

Routing: the token_corpus pass of this case (non-speculative, so every routed token is a generated or
prompt token of the target). Until that pass has run, CORPUS is None and the arch prices balanced routing;
only the capture passes run in that state.

Calibrated case values are `from_capture`: `attn_gpu_memory_gb` from the workload pass's server log.

alignment Phase 1 only: one `phase: 1` case replaying trace/diverse_100.csv at its recorded arrival times.
No Phase 2 matrix (chunk 4096/8192, other max_num_seqs) yet.

Run from the repo root: uv run python presets/alignment/dsv41_flash_b200_dp4_ep4/make_pack.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for `launcher`
from launcher.alignment_campaign.packfile import write_campaign  # noqa: E402

L = 131072  # --max-model-len
CHUNK = 2048  # --max-num-batched-tokens per DP rank
MAX_NUM_SEQS = 64  # the arch's MAX_NUM_SEQS (simulator/src/arch/deepseek_v41_vllm.rs)
# This pack's own token_corpus pass (job 24586), published to the HF dataset.
CORPUS = ("hf://datasets/UW-SyFI/servingstudio-workload@fb58b1490e01341e0c42f5eea41823435285afdf/"
          "deepseek_v41_flash/vllm_dp4_ep4/diverse_100/capture/20261120/manifest.json")


def capture_sizes(chunk: int) -> list[int]:
    """vLLM's explicit list up to 2048: [1, 2, 4], every 8 to 248, every 16 from 256."""
    sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, chunk + 1, 16))
    assert sizes[-1] == chunk
    return sizes


PYTHON_RUNTIME = {
    "packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
    "environment": {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        # DP4 loads four engine cores' weights and engram tables; the 600 s default expired mid-load
        # in the DSpark DP4 captures (job 17904).
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    },
}

FROM_CAPTURE = {"status": "from_capture"}


def variant() -> dict:
    sizes = capture_sizes(CHUNK)
    server = {
        "tp_size": 1,
        "dp_size": 4,
        "expert_parallel_size": 4,
        "expert_count_reduction_group_size": 4,
        "chunk_size": CHUNK,
        "max_cudagraph_capture_size": CHUNK,
        "enable_server_load_tracking": True,
        "api_server_count": 4,
        "served_model_name": "dsv41-flash",
        "extra_args": [
            "--enable-expert-parallel", "--trust-remote-code",
            "--tokenizer-mode", "deepseek_v41",
            "--block-size", "128",
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--kv-cache-dtype", "fp8",
            "--seed", "0",
            "--generation-config", "vllm",
            "--jit-monitor-verbose",
            "--engram-config", '{"cpu_offload": true}',
            "--cudagraph-capture-sizes", *[str(s) for s in sizes],
        ],
    }
    arch = {
        "type": "deepseek_v41_vllm_dp_attn_ep_moe",
        "model_config": "model/config/deepseek_v41_flash.json",
        "fp8": True,
        "cudagraph_capture_sizes": sizes,
    }
    if CORPUS is None:
        arch["routing"] = "uniform"
    else:
        arch["routing"] = "corpus"
        arch["token_corpus_file"] = CORPUS
    worker = {
        "type": "chunked_prefill",
        "max_batch_tokens": CHUNK,
        "batch_policy": "mix",
        "dp_placement": "vllm-least-loaded",
    }
    return {
        "deployment": "unified",
        "gpu": "NVIDIA B200",
        "replicas": 1,
        "checkpoint": "dsv41_flash_1017",
        "engine": "vllm",
        # The DSpark captures' backend; token_corpus runs on openai anyway.
        "backend": "openai",
        "run_name_prefix": "dsv41_b200_dp4",
        "python_runtime": PYTHON_RUNTIME,
        "server": server,
        "arch": arch,
        "worker": worker,
        "profile_passes": [
            {"kind": "workload_metrics", "name": "profile_workload", "trace": "workload", "warmup": True},
            {"kind": "nsys", "name": "profile_nsys", "trace": "workload", "warmup": True},
            {"kind": "token_corpus", "name": "profile_corpus", "trace": "workload"},
        ],
        "input_builder": {"type": "engine_text", "measured_phase": "forward", "group_assignment": "per_dp_rank"},
        "label_rules": "label_rules/manifest.json",
    }


def phase1_case() -> dict:
    return {
        "index": 1,
        "name": "dp4_diverse_100",
        "variant": "dp4_c2048_s64",
        "phase": 1,
        "purpose": "alignment Phase 1: the tracked diverse_100 trace at its recorded arrival times.",
        "coverage": ["diverse_100"],
        "max_model_len": L,
        "max_concurrency": 64,
        "chunk_size": CHUNK,
        "profile_arrival_mode": "trace-timed",
        "simulation_arrival_mode": "trace_timed",
        "gpu_memory_utilization": 0.9,
        # The DSpark case's 60 s plus 30 s: the replay runs past its last arrival (115 s) on decode alone,
        # and Check 1 needs decode-only steps.
        "capture_seconds": 90.0,
        "device_role": "primary",
        "workload_trace": {"path": "trace/diverse_100.csv"},
        "attn_gpu_memory_gb": FROM_CAPTURE,
    }


def main() -> None:
    pack = {
        "schema_version": 1,
        "pack": "dsv41_flash_b200_dp4_ep4",
        "description": "DeepSeek-V4.1-Flash-1017 without speculative decoding under the vLLM fork's model runner "
                       "V2 on 4x B200, DP4 attention / EP4 experts, chunk 2048, max_num_seqs 64. alignment "
                       "Phase 1 (diverse_100) only. See make_pack.py.",
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
        "variants": {"dp4_c2048_s64": variant()},
        "cases": [phase1_case()],
    }
    write_campaign(HERE / "campaign.yaml", pack)


if __name__ == "__main__":
    main()

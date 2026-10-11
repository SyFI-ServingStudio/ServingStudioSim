"""Write campaign.yaml for Inkling-NVFP4 TP8/EP8 from the GLM-5.2 TP8 matrix's operating points.

All 15 operating points (Inkling's model_max_length is 1048576, so every context fits). The KV-pressure figures in
cases 14 and 15 are GLM-5.2's capacity; Inkling's KV per token is smaller, so the same traces press it less. Indices
keep the GLM numbering so a case compares across packs. Run from the repo root: uv run python presets/alignment/inkling_nvfp4_b200_tp8/make_pack.py
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "glm52_nvfp4_b200_tp8" / "campaign.yaml"
KEEP = set(range(1, 16))
# The public arch preset holds the capture's --cudagraph-capture-sizes.
arch0 = yaml.safe_load(open(HERE.parents[1] / "public" / "Inkling-NVFP4" / "inkling_vllm_nvfp4_rel_attn_moe.yaml"))["arch"]

src = yaml.safe_load(open(SRC))
variant = {
    "deployment": "unified",
    "gpu": "NVIDIA B200",
    "replicas": 1,
    "checkpoint": "inkling_nvfp4",
    "engine": "vllm",
    # Not vllm_tokens: that starts vLLM with skip_tokenizer_init, and the fork's Inkling renderer
    # (vllm/renderers/inkling.py) needs the tokenizer at API-server init (job 5221).
    "backend": "openai",
    "run_name_prefix": "inkling_b200_tp8_ep8",
    "server": {
        "tp_size": 8,
        "dp_size": 1,
        "expert_parallel_size": 8,
        "expert_count_reduction_group_size": 8,
        "chunk_size": 2048,
        "max_cudagraph_capture_size": 2048,
        "enable_server_load_tracking": True,
        "extra_args": ["--enable-expert-parallel", "--no-enable-eplb",
                       "--limit-mm-per-prompt", '{"image":0,"audio":0}'],
    },
    "arch": {
        "type": "inkling_vllm_nvfp4_rel_attn_moe",
        "model_config": "model/config/inkling_nvfp4.json",
        "fp8": False,
        "tp_size": 8,
        "enable_expert_parallel": True,
        # Recorded routes of the diverse_100 capture (EPLB off).
        "routing": "corpus",
        "token_corpus_file": "hf://datasets/UW-SyFI/servingstudio-workload@a6959f415133a183260ecba335fbdb0fb12c3db8/"
                             "inkling_nvfp4/vllm/diverse_100/capture/20261009/manifest.json",
        "cudagraph_capture_sizes": arch0["cudagraph_capture_sizes"],
    },
    "worker": {"type": "chunked_prefill", "max_batch_tokens": 2048, "batch_policy": "mix"},
    "profile_passes": [
        {"kind": "nsys", "name": "profile_nsys", "trace": "kernel"},
        {"kind": "workload_metrics", "name": "profile_workload", "trace": "workload"},
    ],
    "input_builder": {"type": "engine_text", "measured_phase": "forward", "group_assignment": "single"},
    "label_rules": "label_rules/manifest.json",
}
KV = {
    # vLLM measured 73.85 GiB of KV per GPU at gpu_memory_utilization 0.85, chunk 2048 (KV allocation does not
    # depend on max_model_len). Cases here run at 0.85, so it carries over; re-read each case's workload log.
    "value": 79.2967,
    "status": "provisional",
    "derived_from": "`Available KV cache memory: 73.85 GiB` (x 1.073741824) in the diverse_100 workload server log, "
                    "same server config except max_model_len",
    "evidence": "logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/profile_workload_metrics/vllm/",
}
# Per-case KV read from that case's own workload pass: index -> (GiB from `Available KV cache memory`, log dir).
CAMPAIGN = "logs/20261009_1_inkling_nvfp4_tp8_campaign"
MEASURED = {
    1: (73.85, f"{CAMPAIGN}/01_micro_throughput_c256/profile_workload/vllm/"),
    2: (73.85, f"{CAMPAIGN}/02_graph_boundaries_c128/profile_workload/vllm/"),
    3: (73.85, f"{CAMPAIGN}/03_balanced_anchor_c32/profile_workload/vllm/"),
    4: (73.85, f"{CAMPAIGN}/04_prefill_spectrum_c32/profile_workload/vllm/"),
    5: (73.85, f"{CAMPAIGN}/05_decode_spectrum_c64/profile_workload/vllm/"),
    6: (73.85, f"{CAMPAIGN}/06_long_long_c8/profile_workload/vllm/"),
    7: (73.85, f"{CAMPAIGN}/07_quadrant_interference_c48/profile_workload/vllm/"),
    8: (73.85, f"{CAMPAIGN}/08_completion_churn_c96/profile_workload/vllm/"),
    9: (73.85, f"{CAMPAIGN}/09_rate_knee/profile_workload/vllm/"),
    10: (73.85, f"{CAMPAIGN}/10_full_saturation_rate/profile_workload/vllm/"),
    11: (73.85, f"{CAMPAIGN}/11_long_prefill_ladder_c1/profile_workload/vllm/"),
    12: (73.85, f"{CAMPAIGN}/12_ultra_context_decode_c1/profile_workload/vllm/"),
    13: (73.85, f"{CAMPAIGN}/13_generation_length_matrix_c8/profile_workload/vllm/"),
    14: (73.85, f"{CAMPAIGN}/14_near_full_long_decode_c21/profile_workload/vllm/"),
    15: (73.85, f"{CAMPAIGN}/15_kv_overcommit_mixed_churn_c14/profile_workload/vllm/"),
}
cases = []
for c in src["cases"]:
    if c["index"] not in KEEP:
        continue
    c = copy.deepcopy(c)
    c["variant"] = "tp8_ep8"
    c["gpu_memory_utilization"] = 0.85
    c["attn_gpu_memory_gb"] = dict(KV)
    if c["index"] in MEASURED:
        gib, evidence = MEASURED[c["index"]]
        c["attn_gpu_memory_gb"] = {
            "value": round(gib * 1.073741824, 4),
            "status": "measured",
            "derived_from": f"`Available KV cache memory: {gib} GiB` (x 1.073741824) in this case's workload server log",
            "evidence": evidence,
        }
    c.pop("provenance", None)
    if c["index"] in (14, 15):
        c["purpose"] = c["purpose"] + " (terminal-KV figure is GLM-5.2's capacity; Inkling's is larger)"
    c["workload_trace"]["file"] = c["workload_trace"]["file"]  # same name; regenerated by render
    cases.append(c)
out = {
    "schema_version": 1,
    "pack": "inkling_nvfp4_b200_tp8",
    "description": "Inkling-NVFP4 on B200, TP8 attention / EP8 experts / DP1 under the vLLM alignment fork. "
                   "All 15 operating points of glm52_nvfp4_b200_tp8; see make_pack.py.",
    "recorded_on": {
        "host": "b200_raid",
        "gpu": "NVIDIA B200",
        "checkpoint_snapshots": {
            "inkling_nvfp4": "models--thinkingmachines--Inkling-NVFP4/snapshots/42a75a99a40eb2ba1e0717db6357a0bf15205044"},
        "nsys_version": "2025.5.2",
        "device_roles": {"primary": "0,1,2,3,4,5,6,7"},
    },
    "variants": {"tp8_ep8": variant},
    "cases": cases,
}
(HERE / "campaign.yaml").write_text("# Generated by make_pack.py; edit that, not this.\n" + yaml.safe_dump(out, sort_keys=False, width=110))
print(len(cases), "cases")

"""Render search configs as launcher presets (`launcher/README.md`).

Configs that share one family and one knob set become one preset: a
`compound` group `config` (one row per config, every knob that differs between
rows a member) crossed with a `compound` group `workload` (one row per trace).
Knobs equal across every row are written literally. `io.log_dir` is
`<root>/{config}/{workload}`, so each run lands at
`<root>/<config label>/<workload label>/`.
"""

from __future__ import annotations

from collections import defaultdict

from .space import BOOKKEEPING, GROUP_KNOBS, POOLS, WORKER_KNOBS

GPU = "AWS Trainium2 LNC2"
MODEL_CONFIG = "model/config/qwen3_8_flash_next_bf16.json"
CORE_HBM_GB = 25.769803776

_ARCH_TYPE = {
    "unified": "qwen4_exp_neuron_bf16",
    "mtp": "qwen4_exp_neuron_bf16_speculative",
    "pd": "qwen4_exp_neuron_bf16",
}
_WORKER_TYPE = {"unified": "chunked_prefill", "mtp": "speculative"}


def member_name(knob: str) -> str:
    """A compound member identifier for a knob (`prefill.attn_dp` -> `prefill__attn_dp`)."""
    return "k_" + knob.replace(".", "__")


def _group(family: str, pool: str | None, knobs: dict) -> dict:
    """One pool group from `knobs` (unprefixed), values literal or `${...}`."""
    arch = {"type": _ARCH_TYPE[family], "model_config": MODEL_CONFIG, "fp8": False}
    if pool is None:
        worker = {"type": _WORKER_TYPE[family], "attn_gpu_memory_gb": CORE_HBM_GB}
    else:
        worker = {"type": f"pd_{pool}", "attn_gpu_memory_gb": CORE_HBM_GB}
    group = {"gpu": GPU, "replicas": 1, "arch": arch, "worker": worker}
    for k, v in knobs.items():
        if k in GROUP_KNOBS:
            group[k] = v
        elif k in WORKER_KNOBS:
            worker[k] = v
        elif k == "draft_tokens":
            arch[k] = v
            worker[k] = v
        else:
            arch[k] = v
    if pool is None:
        # Required by chunked_prefill / speculative; the step-4 presets' chunk.
        worker.setdefault("max_batch_tokens", 2048)
    if family == "mtp":
        arch.setdefault("draft_tokens", 3)
        worker.setdefault("draft_tokens", arch["draft_tokens"])
    return group


def render(
    family: str,
    configs: list[dict],
    workloads: list[tuple[str, str]],
    log_root: str,
) -> dict:
    """One preset over `configs` (same family and knob set) x `workloads`
    (`(label, trace path)`)."""
    if not configs:
        raise ValueError("render needs at least one config")
    keysets = {tuple(sorted(k for k in c if k not in BOOKKEEPING)) for c in configs}
    if len(keysets) != 1:
        raise ValueError("render needs configs that share one knob set (see split_by_keyset)")
    keys = list(next(iter(keysets)))
    if len(configs) == 1:
        # Keep the config axis (one row) so every run sits at {config}/{workload}.
        varying = keys
    else:
        varying = [k for k in keys if len({repr(c[k]) for c in configs}) > 1]
        if not varying:
            raise ValueError("configs with one knob set must differ in some knob")

    def value(k, cfg):
        return f"${{{member_name(k)}}}" if k in varying else cfg[k]

    first = configs[0]
    if family == "pd":
        pools = {}
        for pool in POOLS:
            pre = pool + "."
            pk = {k[len(pre) :]: value(k, first) for k in keys if k.startswith(pre)}
            pools[pool] = {"groups": [_group(family, pool, pk)]}
        deployment = "pd"
    else:
        pools = {"main": {"groups": [_group(family, None, {k: value(k, first) for k in keys})]}}
        deployment = "unified"
    workload = {
        "trace_files": "${trace_files}",
        "input_file_format": "text-generation-independent",
        "arrival_mode": "trace_timed",
        # trace_timed divides the trace's arrival times by request_rate (default
        # 10); 1.0 replays them as written.
        "request_rate": 1.0,
        "run_to_end": True,
    }
    if family == "mtp":
        workload["input_file_tags"] = ["speculative"]
    preset = {
        "deployment": deployment,
        "workload": workload,
        "io": {"log_dir": f"{log_root}/{{config}}/{{workload}}", "quiet": True},
        "pools": pools,
        "compound": {
            "config": {c["label"]: {member_name(k): c[k] for k in varying} for c in configs},
            "workload": {label: {"trace_files": [path]} for label, path in workloads},
        },
    }
    return preset


def split_by_keyset(configs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group configs by (family, knob set): one preset each."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in configs:
        keys = tuple(sorted(k for k in c if k not in BOOKKEEPING))
        groups[(c["family"], keys)].append(c)
    return [
        (fam, cs) for (fam, _), cs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]

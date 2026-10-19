"""Bounded instruction/DMA capture using the validated stock LLM configuration."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path


def capture_config(config: dict, root: Path, request: dict, case: dict, phase: str) -> dict:
    prefill = phase == "prefill"
    if prefill and request["forwards"] != 1:
        raise ValueError("prefill capture requires one selected forward and one decode guard")
    config = copy.deepcopy(config)
    config["additional_config"]["neuron_profiler"] = {
        "output_dir": str(root / "device-profiles"),
        "activities": ["device_profile", "system_profile"],
        "neuron_cores": request["ranks"],
        "sys_trace_max_events_per_nc": 4000000,
    }
    config["profiler_config"] = {
        "profiler": "cuda",
        # Prefill uses actual scheduler geometry and completed output boundaries.
        # Decode retains the empirically checked public worker-step window.
        "delay_iterations": 0 if prefill else len(case["prompts"]) + 3,
        # In this pinned public scheduler the window includes one worker step
        # without a forward (observed: max=2 -> one prefill, max=3 -> two decode).
        # The exporter must still prove the exact requested native forward count.
        "max_iterations": 0 if prefill else request["forwards"] + 1,
    }
    if prefill:
        from profiling.runners.neuron.vllm_ops_control import WORKER_CLASS

        config["worker_cls"] = WORKER_CLASS
    return config


def main(root: Path, stage: str = "capture") -> None:
    from vllm import LLM, SamplingParams

    from profiling.runners.neuron.vllm_forward_engine import save

    plan = json.loads((root / "ops-plan.json").read_text())
    validated = Path(plan["validated_run"])
    case = plan["case"]
    if stage == "labels":
        config = copy.deepcopy(plan["profile_config"])
        config.pop("profiler_config", None)
        config["additional_config"].pop("neuron_profiler", None)
    elif stage == "capture":
        config = capture_config(
            plan["profile_config"], root, plan["request"], case, plan["shape"]["phase"]
        )
    else:
        raise ValueError(f"unknown per-op stage: {stage}")
    save(root, "capture-config.json", config)
    expected = {
        row["case_id"]: row["token_ids"]
        for row in json.loads((validated / "accuracy-outputs.json").read_text())
    }
    llm = LLM(**config)

    def generate():
        start = time.time_ns()
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in case["prompts"]],
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
        stop = time.time_ns()
        if len(outputs) != len(case["prompts"]):
            raise RuntimeError("capture request count differs from validation")
        for slot, output in enumerate(outputs):
            if (
                output.prompt_token_ids != case["prompts"][slot]
                or output.outputs[0].token_ids != expected[f"{case['id']}-slot{slot}"]
            ):
                raise RuntimeError("captured sampling differs from numerical validation")
        return {
            "case_id": case["id"],
            "shape_id": case["shape_id"],
            "bucket": case["bucket"],
            "batch": len(outputs),
            "start_epoch_ns": start,
            "stop_epoch_ns": stop,
        }

    save(root, "capture-warmup.json", generate())
    if stage == "labels":
        from profiling.runners.neuron.vllm_identity import (
            compile_cache_snapshot,
            stock_compile_cache_dir,
        )

        cache = stock_compile_cache_dir()
        save(
            root,
            "label-compile-cache.json",
            {
                "compile_cache_dir": str(cache),
                "graphs": compile_cache_snapshot(cache),
            },
        )
        print("OPS_LABEL_COMPILE_COMPLETE", flush=True)
        return
    llm.start_profile()
    try:
        save(root, "captured-outputs.json", [generate()])
    finally:
        llm.stop_profile()
    print("OPS_CAPTURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "capture")

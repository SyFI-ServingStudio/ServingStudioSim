"""Device instruction capture of the unchanged stock serving engine (capture stage).

Runs after the stock ``profile`` stage in the same run directory and reuses its
exact ``LLM`` configuration (``profile-config.json``), changing only the Neuron
profiler from system tracing to native device tracing of the selected ranks.
Compiled graphs therefore resolve to the same compile-cache entries. Sampling
must reproduce the numerical run's tokens. One profiler window covers a few
canonical public calls so every rank's instruction buffer holds all of them.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path


def capture_cases(plan: dict, cases: list[dict]) -> list[dict]:
    """The canonical cases traced in one window, from ``plan["capture_calls"]``.

    ``capture_calls`` maps decode bucket -> number of public calls of that
    bucket's first canonical case. Every call also runs its prompts' prefills.
    """
    selected = []
    for bucket, calls in sorted(plan["capture_calls"].items(), key=lambda item: int(item[0])):
        matching = [case for case in cases if case["bucket"] == int(bucket)]
        if not matching:
            raise ValueError(f"no canonical case for decode bucket {bucket}")
        selected.extend([matching[0]] * calls)
    return selected


def capture_config(profile_config: dict, output_dir: Path, ranks: list[int]) -> dict:
    """The profile stage's engine config with only the Neuron profiler replaced."""
    config = copy.deepcopy(profile_config)
    config["additional_config"]["neuron_profiler"] = {
        "output_dir": str(output_dir),
        "activities": ["device_profile"],
        "neuron_cores": ranks,
    }
    unchanged = {k: v for k, v in config.items() if k != "additional_config"}
    if unchanged != {k: v for k, v in profile_config.items() if k != "additional_config"} or (
        config["additional_config"]["neuron_config"]
        != profile_config["additional_config"]["neuron_config"]
    ):
        raise RuntimeError("capture must not change the serving configuration")
    return config


def main(root: Path) -> None:
    from vllm import LLM, SamplingParams

    from profiling.runners.neuron.vllm_forward_engine import save

    plan = json.loads((root / "plan.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    precision = json.loads((root / "precision.json").read_text())
    selected = capture_cases(plan, cases)
    if not all(precision["by_shape"][case["shape_id"]]["passed"] for case in selected):
        raise ValueError("capture cases must be numerically accepted")
    profile_config = json.loads((root / "profile-config.json").read_text())
    config = capture_config(profile_config, root / "device-profiles", plan["capture_ranks"])
    save(root, "capture-config.json", config)
    expected = {
        row["case_id"]: row["token_ids"]
        for row in json.loads((root / "accuracy-outputs.json").read_text())
    }
    llm = LLM(**config)

    def generate(case):
        start = time.time_ns()
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in case["prompts"]],
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
        stop = time.time_ns()
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

    save(root, "capture-warmup.json", [generate(case) for case in selected])
    captures = []
    # generate() returns after the device finished every traced forward, so the
    # window closes after the last execution rather than mid-forward.
    llm.start_profile()
    try:
        for case in selected:
            captures.append(generate(case))
    finally:
        llm.stop_profile()
    save(root, "captured-outputs.json", captures)
    print("SEGMENT_CAPTURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))  # argv[2] is the stage name passed by run_stage.

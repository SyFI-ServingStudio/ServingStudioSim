"""Tagged timing-predict input builder dispatch and artifact writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vllm_text import build_cases

INPUT_MANIFEST_NAME = "timing_predict_input_manifest.json"


@dataclass(frozen=True)
class VllmTextInputSpec:
    """One concrete measured-vLLM-text → iter-wise predictor conversion.

    Future multimodal or sharded variants must be sibling spec/builder types;
    they must not add optional placeholder fields to this text-only contract.
    """

    measured_phase: str = "forward"
    group_assignment: str = "single"

    def validate(self) -> None:
        if not self.measured_phase.strip():
            raise ValueError("input_builder.measured_phase must be non-empty")
        if self.group_assignment != "single":
            raise ValueError("input_builder.group_assignment must be 'single' in v1")

    def to_mapping(self) -> dict[str, str]:
        return {
            "type": "vllm_text",
            "measured_phase": self.measured_phase,
            "group_assignment": self.group_assignment,
        }


@dataclass(frozen=True)
class BuildRequest:
    """Resolved cross-stage artifacts supplied by the launcher.

    `simulation_preset` is the sim preset the gpu / arch / backends were read
    from (recorded for provenance only); timing-predict never consumes a
    completed simulation run.
    """

    simulation_preset: Path
    profile_log_dir: Path
    parsed_nsys: Path
    output_dir: Path
    gpu: str
    arch: dict[str, Any]
    # Backend policy is part of the simulated CostTree identity. Preserve the
    # normalized run's complete pool→role map so offline prediction cannot
    # silently fall back to an arch's best-of-N defaults.
    backends: dict[str, dict[str, list[str]]]
    input_spec: VllmTextInputSpec


@dataclass(frozen=True)
class BuildResult:
    predict_config: Path
    cases: Path
    case_map: Path
    input_manifest: Path


def build_inputs(request: BuildRequest) -> BuildResult:
    """Build canonical cases and the generic timing-predict config.

    All output stays under ``request.output_dir``. In particular, this function
    has no analysis directory and cannot create an analyzer manifest.
    """
    request.input_spec.validate()
    parsed = json.loads(request.parsed_nsys.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(f"parsed NSYS root must be a JSON object: {request.parsed_nsys}")
    cases, case_map, excluded = build_cases(parsed, request.input_spec.measured_phase)

    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "timing_predict_cases.json"
    case_map_path = output_dir / "timing_predict_case_map.json"
    predict_config_path = output_dir / "timing_predict_config.json"
    input_manifest_path = output_dir / INPUT_MANIFEST_NAME

    cases_path.write_text(json.dumps(cases, indent=2))
    case_map_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_builder": request.input_spec.to_mapping(),
                "measured_phase": request.input_spec.measured_phase,
                "cases": case_map,
                "excluded_iterations": excluded,
            },
            indent=2,
        )
    )
    predict_config_path.write_text(
        json.dumps(
            {
                "arch": {"iter": request.arch},
                "gpu": request.gpu,
                "backends": request.backends,
                "log_dir": str(output_dir),
                "cases_file": str(cases_path),
            },
            indent=2,
        )
    )
    input_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simulation_preset": str(request.simulation_preset.resolve()),
                "profile_log_dir": str(request.profile_log_dir.resolve()),
                "parsed_nsys": str(request.parsed_nsys.resolve()),
                "predict_log_dir": str(output_dir),
                "timing_predict_cases": str(cases_path),
                "timing_predict_case_map": str(case_map_path),
                "timing_predict_config": str(predict_config_path),
                "input_builder": request.input_spec.to_mapping(),
            },
            indent=2,
        )
    )
    return BuildResult(
        predict_config=predict_config_path,
        cases=cases_path,
        case_map=case_map_path,
        input_manifest=input_manifest_path,
    )

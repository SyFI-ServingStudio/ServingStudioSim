"""`python -m launcher timing-predict CONFIG.{yaml,yml,json} [...]` — offline
per-building-block timing prediction.

A predict config is NOT a deployment `RunConfig` (it has no workload / pools /
sweep — just one arch selector `{iter|attn|ffn}` + gpu + a cases file), so this
entry bypasses the schema expansion the normal run flow does. It performs the
shared `cargo_build`, then for each config runs the predictor subcommand (which
writes the standard `raw/cost_log` + `cost_manifest` artifacts, one row per
case/section) followed by the post-run analyzer — `run_analysis` emits the
Perfetto trace plus the cost reports from exactly those artifacts.

The `arch` selector picks one of `{iter|attn|ffn}` — an iter-wise whole-iteration
arch, or one half of an AFD split (attn / ffn), each costed independently.

Unlike a real `run` (whose launcher snapshots the expanded config into `log_dir`),
the predict path previously left no provenance behind; we now copy the config + its
cases file into `log_dir` so a predict result is self-describing. It also snapshots
the selected model/GPU as an internal `raw/params.json` compatibility input for
the model-aware necessary-work labeler; this does not create a public deployment
or pool resource for the prediction.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

from .artifact_kind import ArtifactKind, write_artifact_kind
from .exec import (
    REPO_ROOT,
    _build_subprocess_env,
    binary_path,
    cargo_build,
    run_analysis,
    run_iter_breakdown,
    run_logged_process,
)
from .managed_job import prepare_managed_job
from .process.artifacts import ArtifactValidationError, validate_timing_prediction_artifacts
from .process.journal import RunJournal, StageState
from .process.leases import LauncherLeases

_LAUNCHER_LEASES = LauncherLeases(REPO_ROOT)

# The simulator's offline writer uses this physical stream tag. The prediction
# API keeps it private, but the semantic labeler needs one stable key to match
# the CostTree worker while it reads the launcher-owned compatibility snapshot.
PREDICTION_POOL_TAG = "predict"
PREDICTION_PROVENANCE_FILE = "prediction_provenance.json"


def _load_config(path: Path) -> dict:
    """Parse the minimal predict config (JSON or YAML) — only `log_dir` /
    `cases_file` are needed launcher-side; the binary re-parses the whole thing."""
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def _prediction_labeler_params(cfg: dict) -> dict | None:
    """Project a predict config into the labeler's narrow params contract.

    Timing-predict intentionally has no deployment topology. The R6/R7 model
    labeler nevertheless needs the selected architecture, dtype, GPU, and the
    physical CostTree stream key. Keep this projection private to
    ``raw/params.json``; Analyzer discovery continues to use ``prediction.meta``
    and never publishes this synthetic stream as a pool/worker hierarchy.
    """
    architecture = cfg.get("arch")
    if not isinstance(architecture, dict) or len(architecture) != 1:
        return None
    _selector, architecture_config = next(iter(architecture.items()))
    if not isinstance(architecture_config, dict):
        return None
    model_config = architecture_config.get("model_config")
    gpu_name = cfg.get("gpu")
    if not isinstance(model_config, str) or not model_config:
        return None
    if not isinstance(gpu_name, str) or not gpu_name:
        return None
    return {
        "pools": {
            PREDICTION_POOL_TAG: {
                "groups": [{"arch": dict(architecture_config), "gpu": gpu_name}]
            }
        }
    }


def _snapshot_inputs(config_path: Path, cfg: dict, log_dir: Path) -> None:
    """Copy the predict config + its cases file into `log_dir` for provenance, so a
    predict result is self-describing (mirrors a real run's config snapshot). The
    cases file is resolved relative to the config's directory, the same way the
    binary resolves it. Also write the private model/GPU projection consumed by
    ``model.work.floors``. Best-effort: a snapshot failure warns but does not fail
    the prediction itself; the analyzer will report the unavailable labeler stage.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Alignment generates its config directly in the predict root. Treat
        # that as an already-complete snapshot instead of asking shutil to copy
        # a file onto itself and obscuring the rest of the provenance step.
        config_snapshot = log_dir / config_path.name
        if config_path.resolve() != config_snapshot.resolve():
            shutil.copy2(config_path, config_snapshot)
        cases_file = cfg.get("cases_file")
        if cases_file:
            cases_path = Path(cases_file)
            if not cases_path.is_absolute():
                cases_path = config_path.parent / cases_path
            if cases_path.is_file():
                cases_snapshot = log_dir / cases_path.name
                if cases_path.resolve() != cases_snapshot.resolve():
                    shutil.copy2(cases_path, cases_snapshot)
            else:
                print(
                    f"[warn] cases_file {cases_path} not found; not snapshotted",
                    file=sys.stderr,
                )
    except OSError as e:
        print(f"[warn] failed to snapshot predict inputs into {log_dir}: {e}", file=sys.stderr)

    labeler_params = _prediction_labeler_params(cfg)
    if labeler_params is None:
        return
    try:
        raw_dir = log_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = raw_dir / ".params.json.tmp"
        temporary_path.write_text(
            json.dumps(labeler_params, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(raw_dir / "params.json")
    except OSError as error:
        print(
            f"[warn] failed to snapshot predict labeler params into {log_dir}: {error}",
            file=sys.stderr,
        )


def _prediction_id(log_dir: Path) -> str:
    """Preserve an existing prediction identity across a same-directory rerun."""
    metadata_path = log_dir / "prediction.meta.json"
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = None
    if isinstance(metadata, dict):
        existing_id = metadata.get("prediction_id")
        if _valid_prediction_id(existing_id):
            return existing_id
    return f"p_{uuid.uuid4().hex}"


def _valid_prediction_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("p_"):
        return False
    suffix = value.removeprefix("p_")
    return (
        1 <= len(suffix) <= 64
        and suffix.isascii()
        and all(
            character.islower() or character.isdigit() or character == "_"
            for character in suffix
        )
    )


def _resolve_cases(config_path: Path, cfg: dict) -> list:
    cases_file = cfg.get("cases_file")
    if not isinstance(cases_file, str) or not cases_file:
        raise ValueError("timing-predict config requires cases_file")
    cases_path = Path(cases_file)
    if not cases_path.is_absolute():
        cases_path = config_path.parent / cases_path
    cases = _load_config(cases_path)
    if not isinstance(cases, list):
        raise ValueError("timing-predict cases_file must contain a list")
    return cases


def _write_prediction_metadata(
    config_path: Path,
    cfg: dict,
    log_dir: Path,
    prediction_id: str,
) -> None:
    """Publish the first-class Analyzer discovery marker atomically."""
    arch = cfg.get("arch")
    selector = next(iter(arch)) if isinstance(arch, dict) and len(arch) == 1 else None
    arch_config = arch.get(selector) if isinstance(selector, str) else None
    arch_type = arch_config.get("type") if isinstance(arch_config, dict) else None
    gpu_name, gpu_count = _read_prediction_provenance(log_dir)
    configured_gpu_name = cfg.get("gpu")
    if gpu_name != configured_gpu_name:
        raise ValueError(
            "timing-predict provenance GPU does not match the requested GPU: "
            f"{gpu_name!r} != {configured_gpu_name!r}"
        )
    cases = _resolve_cases(config_path, cfg)
    cases_name = "prediction.cases.json"
    metadata = {
        "schema_version": 1,
        "prediction_id": prediction_id,
        "selector": selector,
        "arch_type": arch_type,
        "gpu": gpu_name,
        "gpu_count": gpu_count,
        "config_file": config_path.name,
        "cases_file": cases_name,
        "case_count": len(cases),
    }
    if not isinstance(selector, str) or not isinstance(arch_type, str):
        raise ValueError("timing-predict metadata requires one typed arch selector")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError("timing-predict metadata requires gpu")
    cases_temporary_path = log_dir / ".prediction.cases.json.tmp"
    cases_temporary_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cases_temporary_path.replace(log_dir / cases_name)
    temporary_path = log_dir / ".prediction.meta.json.tmp"
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(log_dir / "prediction.meta.json")


def _read_prediction_provenance(log_dir: Path) -> tuple[str, int]:
    """Read the Rust model's physical GPU extent without deriving parallelism.

    The predictor constructs the concrete L4 model and is therefore the only
    timing-predict component that can authoritatively call ``gpus_per_replica``.
    """
    provenance_path = log_dir / "raw" / PREDICTION_PROVENANCE_FILE
    provenance = json.loads(provenance_path.read_text("utf-8"))
    gpu_name = provenance.get("gpu_name")
    gpu_count = provenance.get("gpu_count")
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported timing-predict provenance schema")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError("timing-predict provenance requires gpu_name")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
        raise ValueError("timing-predict provenance requires positive gpu_count")
    return gpu_name, gpu_count


def _predict_descriptor(config_path: Path, cfg: dict) -> dict:
    """Build the small catalog descriptor without interpreting case semantics."""
    arch = cfg.get("arch")
    selector = next(iter(arch)) if isinstance(arch, dict) and len(arch) == 1 else None
    descriptor = {
        "selector": selector,
        "configName": config_path.name,
        "caseCount": _predict_case_count(config_path, cfg),
    }
    return {key: value for key, value in descriptor.items() if value is not None}


def _predict_case_count(config_path: Path, cfg: dict) -> int | None:
    cases_file = cfg.get("cases_file")
    if not isinstance(cases_file, str) or not cases_file:
        return None
    cases_path = Path(cases_file)
    if not cases_path.is_absolute():
        cases_path = config_path.parent / cases_path
    try:
        payload = _load_config(cases_path)
    except (OSError, json.JSONDecodeError):
        return None
    return len(payload) if isinstance(payload, list) else None


async def run_one(config_path: Path, build_type: str, analyze: bool) -> bool:
    """Run one already-built predictor config.

    Public so the alignment timing-predict stage can reuse the exact predictor
    execution/snapshot path.
    """
    cfg = _load_config(config_path)
    log_dir = Path(cfg["log_dir"])
    if not log_dir.is_absolute():
        # The binary writes `raw/` relative to its cwd (REPO_ROOT); anchor the
        # launcher-side log_dir to the same root so stdout.log + analysis line up.
        log_dir = REPO_ROOT / log_dir

    prediction_id = _prediction_id(log_dir)
    descriptor = _predict_descriptor(config_path, cfg)
    managed_job = prepare_managed_job(
        "timing_predict",
        log_dir,
        descriptor=descriptor,
        analyzer_resource_id=prediction_id,
    )

    run_directory_lease = None
    try:
        run_directory_lease = _LAUNCHER_LEASES.run_directory(log_dir)
        await run_directory_lease.acquire()
        if managed_job is not None:
            managed_job.report("running")
        write_artifact_kind(log_dir, ArtifactKind.TIMING_PREDICTION)
        binary = binary_path(build_type)
        argv = [str(binary), "timing-predict", str(config_path)]
        journal = RunJournal(log_dir)
        async with _LAUNCHER_LEASES.profile_database(write=True):
            journal.update(
                "timing_predict",
                StageState.RUNNING,
                resources=["profile-db:exclusive"],
            )
            try:
                result = await run_logged_process(
                    argv,
                    log_dir,
                    env=_build_subprocess_env(),
                    name="timing_predict",
                )
            except asyncio.CancelledError:
                journal.update("timing_predict", StageState.CANCELLED)
                raise
        if not result.succeeded:
            journal.update(
                "timing_predict",
                StageState.FAILED,
                result=result,
                error="predictor failed or left process-group descendants",
            )
            if managed_job is not None:
                managed_job.report("failed")
            return False
        journal.update("timing_predict", StageState.VALIDATING, result=result)
        try:
            validation = validate_timing_prediction_artifacts(log_dir)
        except ArtifactValidationError as error:
            journal.update(
                "timing_predict",
                StageState.FAILED,
                result=result,
                error=str(error),
            )
            if managed_job is not None:
                managed_job.report("failed")
            return False
        # Publish private labeler inputs and the first-class resource only after
        # Rust has emitted authoritative L4 GPU provenance, and only once that
        # output has passed validation. The explicit type marker exists from
        # launch, but the catalog requires prediction.meta.json too, so an
        # in-flight prediction is not published and can never masquerade as a run.
        _snapshot_inputs(config_path, cfg, log_dir)
        _write_prediction_metadata(config_path, cfg, log_dir, prediction_id)
        journal.update(
            "timing_predict",
            StageState.SUCCEEDED,
            result=result,
            artifacts=validation.paths,
        )
        if analyze:
            if managed_job is not None:
                managed_job.report("analysis_running")
            # Best-effort: `analyze trace` (the Perfetto tree) + `analyze run`. Cost
            # subjects apply; request/throughput subjects self-skip on a predict dir.
            await run_analysis(log_dir, build_type)
            # Predict-only: the human-readable cost tree (reports/iter_breakdown.ans).
            # Not in the shared run_analysis — a real run's thousands of iters would
            # make that file enormous; a predict dir has only a few cases.
            await run_iter_breakdown(log_dir, build_type)
        if managed_job is not None:
            managed_job.report("ready", summary=descriptor)
        return True
    except (asyncio.CancelledError, KeyboardInterrupt):
        if managed_job is not None:
            managed_job.report("interrupted")
        raise
    except Exception:
        if managed_job is not None:
            managed_job.report("failed")
        raise
    finally:
        if run_directory_lease is not None:
            run_directory_lease.release()


def main(argv: list[str]) -> int:
    build_type = "release"
    analyze = True
    configs: list[str] = []
    it = iter(argv)
    for tok in it:
        if tok == "--build-type":
            build_type = next(it, "")
            if not build_type:
                sys.exit("timing-predict: --build-type needs a value")
        elif tok == "--no-analyze":
            analyze = False
        elif tok.startswith("-"):
            sys.exit(f"timing-predict: unknown flag {tok}")
        else:
            configs.append(tok)

    if not configs:
        sys.exit(
            "timing-predict: no config given; usage: "
            "python -m launcher timing-predict CONFIG.yaml|json [...]"
        )

    # INV-8: the single shared build (also produces the analyzer binary).
    if not cargo_build(build_type, build_analyzer=analyze):
        sys.exit("build failed (see errors above)")

    rc = 0
    for c in configs:
        config_path = Path(c).resolve()
        if not config_path.is_file():
            print(f"[invalid] {c}: not a file", file=sys.stderr)
            rc = 2
            continue
        if not asyncio.run(run_one(config_path, build_type, analyze)):
            rc = rc or 1
    return rc

"""Cache-free per-operation profiling facade and bounded offline analysis process.

Only capture enters the device executor. Saved captures are analyzed in a child
with an address-space limit; neither path reads or writes the timing database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from profiling.db.batch import args_to_spec, coerce_args
from profiling.db.registry import find_kernel_profiler_spec


def profile_ops(
    kernel_kind: str,
    spec: dict,
    *,
    backend: str,
    output_dir: Path,
    from_capture: Path | None = None,
    validated_run: Path | None = None,
    forwards: int | None = None,
    ranks: tuple[int, ...] = (0,),
    memory_limit_gb: int = 60,
    debug_labels: bool = False,
    labels_map: Path | None = None,
    dma_buffer_mb: int = 256,
    instruction_buffer_mb: int = 512,
) -> dict:
    registry = find_kernel_profiler_spec(kernel_kind, backend)
    if registry.ops_runner_ref is None or registry.supports.device_family != "neuron":
        raise ValueError(f"{kernel_kind}/{backend} does not support per-op capture")
    shape = args_to_spec(coerce_args(registry.args_schema, spec))
    if forwards is None:
        forwards = 1 if shape["phase"] == "prefill" else 2
    if forwards not in (1, 2):
        raise ValueError("ops captures one or two forwards per selected shape")
    if from_capture is None and shape["phase"] == "prefill" and forwards != 1:
        raise ValueError(
            "prefill capture supports one selected forward plus one audited decode guard"
        )
    if not ranks or len(set(ranks)) != len(ranks) or any(r not in range(4) for r in ranks):
        raise ValueError("ops ranks must be distinct values from 0 through 3")
    if not 1 <= memory_limit_gb <= 60:
        raise ValueError("ops memory limit must be between 1 and 60 GiB")
    buffer_sizes = (64, 128, 256, 512, 1024)
    if dma_buffer_mb not in buffer_sizes or instruction_buffer_mb not in buffer_sizes:
        raise ValueError("ops trace buffers require a power of two from 64 through 1024 MiB")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    if from_capture is None:
        from profiling.exec import get_default_pool
        from profiling.gpu_policy import require_gpu

        require_gpu("capturing Neuron per-op profiling evidence")
        request = {
            "forwards": forwards,
            "ranks": list(ranks),
            "memory_limit_gb": memory_limit_gb,
            "validated_run": str(Path(validated_run).resolve()) if validated_run else None,
            "debug_labels": debug_labels,
            "dma_buffer_mb": dma_buffer_mb,
            "instruction_buffer_mb": instruction_buffer_mb,
        }
        pool = get_default_pool("neuron")
        chunks = pool.acquire_chunks(1, max_concurrent=1)
        chunk = next(chunks)
        try:
            provenance = chunk.run_ops(kernel_kind, {**shape, "backend": backend}, request)
        finally:
            chunks.close()
        from_capture = Path(provenance["manifest_path"])
    else:
        from_capture = Path(from_capture).resolve()
        if validated_run is not None or debug_labels:
            raise ValueError("--validated-run and --debug-labels apply only to a new capture")
    manifest = json.loads(from_capture.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported per-op capture manifest schema")
    if manifest.get("kernel") != {"kind": kernel_kind, "backend": backend}:
        raise ValueError("capture kernel identity differs from the requested profiler")
    if manifest.get("shape") != shape:
        raise ValueError("capture shape differs from the requested shape")
    manifest = normalize_manifest_paths(manifest, from_capture.parent)
    validate_execution_receipts(manifest)
    if labels_map is not None:
        apply_labels_map(manifest, Path(labels_map).resolve())
    manifest["provenance"] = {**manifest.get("provenance", {}), **provenance}
    copied_manifest = output_dir / "capture-manifest.json"
    copied_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    result_file = output_dir / "analysis-result.json"
    with (output_dir / "analysis.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "profiling.ops",
                str(copied_manifest),
                str(output_dir),
                "--memory-limit-gb",
                str(memory_limit_gb),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
    if result.returncode:
        raise RuntimeError(f"per-op analysis failed; see {output_dir / 'analysis.log'}")
    result = json.loads(result_file.read_text())
    result.update(
        kernel_kind=kernel_kind,
        backend=backend,
        shape=shape,
        output_dir=str(output_dir),
        persisted=False,
    )
    result["artifacts"] = sorted(set(result.get("artifacts", [])) | {str(copied_manifest)})
    return result


_PATH_FIELDS = frozenset(
    {
        "session_dir",
        "path",
        "static_instructions",
        "capture_root",
        "validated_run",
        "accuracy_receipt",
        "binary_receipt",
        "identity_gate",
        "capture_control_receipt",
        "manifest_path",
        "semantic_source_identity",
    }
)


def normalize_manifest_paths(value, base: Path):
    """Resolve declared paths before copying a manifest to a different directory."""
    if isinstance(value, list):
        return [normalize_manifest_paths(item, base) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in _PATH_FIELDS and isinstance(item, str):
            path = Path(item)
            result[key] = str((path if path.is_absolute() else base / path).resolve())
        else:
            result[key] = normalize_manifest_paths(item, base)
    return result


def validate_execution_receipts(manifest: dict) -> dict:
    """Recheck declared numerical evidence; old trace-only manifests remain diagnostic."""
    execution = manifest.get("execution", {})
    names = {
        name.removesuffix("_sha256")
        for name in execution
        if name.removesuffix("_sha256").endswith("_receipt")
        or name.removesuffix("_sha256") == "identity_gate"
    }
    receipts, hashes = {}, {}
    for name in sorted(names):
        expected = execution.get(f"{name}_sha256", "")
        path = execution.get(name)
        if not path or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"declared execution receipt {name} needs a path and SHA256")
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"declared execution receipt is missing: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"declared execution receipt changed: {path}")
        hashes[name] = actual
        receipts[name] = json.loads(path.read_text())
    from profiling.runners.neuron.vllm_ops_control import METHOD, validate_control_receipt

    controlled = manifest.get("sampling", {}).get("window") == METHOD
    if controlled and "capture_control_receipt" not in receipts:
        raise ValueError("geometry-controlled capture is missing its control receipt")
    if "capture_control_receipt" in receipts:
        ranks = sorted(session["rank"] for session in manifest["sessions"])
        validate_control_receipt(receipts["capture_control_receipt"], manifest["shape"], ranks)
        if not controlled:
            raise ValueError("capture control receipt requires matching sampling metadata")
        if any(session.get("execution_selection", {}).get("policy")
               != "prefill_with_trailing_decode_guard_v1" for session in manifest["sessions"]):
            raise ValueError("geometry-controlled capture requires the exact P,D selection policy")
    required = {"accuracy_receipt", "binary_receipt", "identity_gate"}
    if not required <= receipts.keys():
        return {
            "status": "absent" if not receipts else "incomplete",
            "validated": False,
            "receipt_sha256": hashes,
            "scope": "diagnostic trace analysis; numerical validation is not established",
        }
    accuracy, binary, gate = (receipts[name] for name in
                              ("accuracy_receipt", "binary_receipt", "identity_gate"))
    if (
        accuracy.get("passed") is not True
        or binary.get("precision_passed") is not True
        or binary.get("binary_binding_available") is not True
        or binary.get("scope") != "warm_cache_neff_bytes_stable_across_public_accuracy_calls"
        or binary.get("files", {}).get("precision.json") != hashes["accuracy_receipt"]
        or gate.get("passed") is not True
        or gate.get("binary_receipt_sha256") != hashes["binary_receipt"]
    ):
        raise ValueError("execution receipts do not establish passed numerical/binary validation")
    for session in manifest["sessions"]:
        for neff in session["neffs"]:
            graph = binary.get("graphs", {}).get(neff["graph_id"], {})
            if (
                graph.get("binary", {}).get("neff_sha256") != neff["sha256"]
                or graph.get("phase") != neff["phase"]
                or graph.get("token_bucket") != neff["token_bucket"]
                or gate.get("graphs", {}).get(neff["graph_id"], {}).get("neff_sha256")
                != neff["sha256"]
            ):
                raise ValueError("captured NEFF is not bound to the declared numerical receipts")
    return {
        "status": "passed",
        "validated": True,
        "receipt_sha256": hashes,
        "scope": "retained numerical and stock-NEFF receipts rechecked; no new reference run",
    }


def apply_labels_map(manifest: dict, path: Path) -> None:
    """Attach an explicitly selected, hash-bound label map without changing timings."""
    from profiling.runners.neuron.vllm_identity import sha256

    receipt = normalize_manifest_paths(json.loads(path.read_text()), path.parent)
    if receipt.get("schema_version") != 1 or not receipt.get("graphs"):
        raise ValueError("labels map requires schema_version 1 and a nonempty graphs dictionary")
    matched = set()
    for session in manifest["sessions"]:
        for neff in session["neffs"]:
            graph_id = neff["graph_id"]
            label = receipt["graphs"].get(graph_id)
            if label is None:
                continue
            if label["neff_sha256"] != neff["sha256"]:
                raise ValueError("labels map targets a different stock NEFF")
            static_path = Path(label["static_instructions"])
            if sha256(static_path) != label["static_instructions_sha256"]:
                raise ValueError("labels map static instructions changed")
            data = json.loads(static_path.read_text())
            if data.get("neff_sha256") != neff["sha256"]:
                raise ValueError("static label data belongs to another NEFF")
            neff.update(
                static_instructions=str(static_path),
                static_instructions_sha256=label["static_instructions_sha256"],
            )
            matched.add(graph_id)
    if not matched or matched != set(receipt["graphs"]):
        raise ValueError("labels receipt includes graphs not present in the capture")
    if receipt.get("semantic_source_identity") is not None:
        manifest["semantic_source_identity"] = receipt["semantic_source_identity"]
    manifest["labels_receipt"] = {"path": str(path), "sha256": sha256(path)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Internal bounded per-op analyzer")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory-limit-gb", type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.memory_limit_gb <= 60:
        raise ValueError("memory limit must be between 1 and 60 GiB")
    import resource

    limit = args.memory_limit_gb * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    from profiling.profilers.neuron_ops import analyze_capture
    from profiling.profilers.neuron_ops_plot import render_coverage, render_ops

    manifest = normalize_manifest_paths(json.loads(args.manifest.read_text()), args.manifest.parent)
    numerical = validate_execution_receipts(manifest)
    result = analyze_capture(args.manifest, args.output)
    quality = result["summary"]["quality"]
    quality["numerical_provenance"] = numerical
    if "sampling" in manifest:
        result["summary"]["sampling"] = manifest["sampling"]
    (args.output / "summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    (args.output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    for renderer, name in (
        (render_ops, "neuron_ops.png"),
        (render_coverage, "neuron_ops_coverage.png"),
    ):
        plot = renderer(args.output / "summary.json", args.output / name)
        if plot is not None:
            result.setdefault("artifacts", []).append(str(plot))
            result.setdefault("plots", []).append(str(plot))
    (args.output / "analysis-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

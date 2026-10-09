"""Stock BF16 rank-local MLP; numerical validation precedes native timing.

Source/flags/input distribution are the validated trainium-composition/mlp
experiment, stock commit f8abae64. A fresh child isolates each lite runtime and
retains failures. Allocation/reservation and offline cache policy belong to exec.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from profiling.db.args import DType
from profiling.profilers.neuron_lite_timer import measure_lite, verify_no_event_drops
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.neuron.vllm_forward import MODEL_HASHES, VERSIONS, sha256

SOURCE_SHA256 = "86db8b306f5dec83ab8bceb24f8e725f0b176dcd675ea8cfdd0f60cdd344d104"
COMPILER_ARGS = (
    "--auto-cast=none",
    "--verbose=35",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
)
RELATIVE_L2_LIMIT = 0.02
NORMALIZED_PEAK_LIMIT = 0.05
_PROVENANCE = {}


def validate_shape(m, hidden, intermediate, dtype):
    if DType.from_value(dtype) != DType.BF16:
        raise ProfilerNotImplemented("stock Neuron MLP supports BF16 only")
    if (hidden, intermediate) != (4096, 3584) or m not in (1, 16, 512):
        raise ProfilerNotImplemented("stock Neuron MLP is validated at H4096/I3584 and m=1,16,512")


def mlp_reference(torch, x, gate, up, down):
    """Independent full FP32 mathematical oracle, with no NKI rounding emulation."""
    gated = torch.nn.functional.silu(x.float() @ gate.float())
    return (gated * (x.float() @ up.float())) @ down.float()


def check_output(torch, actual, expected):
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise ValueError("stock MLP returned an unexpected shape or dtype")
    error = actual.float() - expected
    relative_l2 = float(
        torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-30)
    )
    normalized_peak = float(error.abs().max() / expected.abs().max().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "finite": finite,
        "relative_l2": relative_l2,
        "normalized_peak": normalized_peak,
        "max_absolute_error": float(error.abs().max()),
        "passed": finite
        and relative_l2 <= RELATIVE_L2_LIMIT
        and normalized_peak <= NORMALIZED_PEAK_LIMIT,
        "relative_l2_limit": RELATIVE_L2_LIMIT,
        "normalized_peak_limit": NORMALIZED_PEAK_LIMIT,
    }


def compiled_identity(trace, cache, root, m):
    """Bind the isolated runtime's unique graph to four BF16 inputs and one NKI call."""
    names = [name for group in trace["device_names"] for name in group.values()]
    hashes = {re.search(r"/compile_cache/([0-9a-f]+)/", name)[1] for name in names}
    if len(hashes) != 1:
        raise ValueError("rank-local MLP trace has no unique compiled graph identity")
    graph_hash = hashes.pop()
    directory = cache / graph_hash
    metadata = (directory / "example_inputs.txt").read_text()
    inputs = re.findall(r"Input (\d+):\s+Shape: \(([^)]*)\)\s+Dtype: (\w+)", metadata)
    expected = [(m, 4096), (4096, 3584), (4096, 3584), (3584, 4096)]
    if len(inputs) != 4 or any(
        int(index) != ordinal
        or dtype != "bfloat16"
        or tuple(int(n.strip()) for n in dims.split(",") if n.strip()) != shape
        for ordinal, ((index, dims, dtype), shape) in enumerate(zip(inputs, expected, strict=True))
    ):
        raise ValueError("compiled MLP input geometry or ordered weight ABI changed")
    fx = (directory / "fxgraph.txt").read_text()
    functions = re.findall(r"call_function\[target=([^]]+)\]", fx)
    if functions != ["torch.ops.higher_order.nki_kernel_wrapper"] or "grid: (2,)" not in fx:
        raise ValueError("stock MLP did not compile as one LNC2 NKI invocation")
    args = re.search(r"args: \((.*?)\), arg_names: \[(.*?)\]", fx)
    if args is None:
        raise ValueError("compiled NKI argument metadata is unavailable")
    values, keys = (part.split(",") for part in args.groups())
    options = dict(
        zip((key.strip() for key in keys), (value.strip() for value in values), strict=True)
    )
    required = {
        "normalization_weights_tensor": "None",
        "fused_add_tensor": "None",
        "gate_proj_bias_tensor": "None",
        "up_proj_bias_tensor": "None",
        "down_proj_bias_tensor": "None",
        "normalization_type": "NormType.NO_NORM",
        "activation_fn": "ActFnType.SiLU",
        "quantization_type": "QuantizationType.NONE",
        "use_tkg_gate_up_proj_column_tiling": "True",
        "use_tkg_down_proj_column_tiling": "True",
    }
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError("compiled stock MLP fusion/column-tiling options changed")
    command = shlex.split((directory / "command.txt").read_text())
    if command[command.index("--auto-cast=none") :] != list(COMPILER_ARGS):
        raise ValueError("compiled stock MLP flags differ from validated component flags")
    files = [
        directory / name
        for name in (
            "example_inputs.txt",
            "fxgraph.txt",
            "graph.hlo",
            "command.txt",
            f"graph_{graph_hash}.neff",
        )
    ]
    copied = root / "compiled"
    copied.mkdir()
    for path in files:
        shutil.copyfile(path, copied / path.name)
    return {
        "graph_hash": graph_hash,
        "source_sha256": {str(path): sha256(path) for path in files},
        "copied_sha256": {str(path): sha256(path) for path in copied.iterdir()},
        "nki_wrapper_count": 1,
        "grid": [2],
        "fusion_options": required,
        "identity_binding": (
            "unique graph in isolated runtime trace device_names and compiled ordered inputs"
        ),
        "runtime_event_model_name_missing": all(
            not event.get("model_name")
            for event in trace["trace_event"]
            if event["name"] == "nc_exec_running"
        ),
    }


def profile_dense_mlp(m, hidden, intermediate, dtype):
    validate_shape(m, hidden, intermediate, dtype)
    cache = Path(os.environ["SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR"])
    root = cache / "mlp-runs" / uuid.uuid4().hex
    root.mkdir(parents=True)
    (root / "plan.json").write_text(
        json.dumps(
            {
                "m": m,
                "hidden": hidden,
                "intermediate": intermediate,
                "dtype": str(dtype),
                "model": os.environ["SERVINGSTUDIO_VLLM_NEURON_MODEL_DIR"],
            },
            indent=2,
        )
    )
    with (root / "worker.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", __name__, str(root)], stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"stock MLP failed ({result.returncode}); artifacts={root}")
    verify_no_event_drops((root / "worker.log").read_text())
    receipt = json.loads((root / "result.json").read_text())
    _PROVENANCE[m] = (
        f"stock_rank_local_mlp; artifacts={root}; graph={receipt['compiled']['graph_hash']}; "
        f"mlp_source={SOURCE_SHA256}; samples=20; accuracy=FP32_relL2<=.02_peak<=.05; "
        "tp4_rank0_weights; no_collectives_norm_residual_bias; additivity_validated=false"
    )
    time_ms = receipt["time_ms"]
    return ComputeMetrics(
        time_ms,
        6 * m * hidden * intermediate / (time_ms * 1e9),
        2 * (2 * m * hidden + 3 * hidden * intermediate) / (time_ms * 1e6),
        energy_j=0.0,
    )


def row_provenance(m, hidden, intermediate, dtype):
    return _PROVENANCE.get(m)


def _run(root):
    import libtorch_neuronx_lite.envs as lite_envs
    import torch
    import vllm_neuron  # noqa: F401 — registers the production lite backend/device
    from safetensors import safe_open
    from vllm_neuron.envs import get_compile_backend_name
    from vllm_neuron.functional.mlp import _can_use_kernel, mlp

    plan = json.loads((root / "plan.json").read_text())
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS or sha256(mlp.__code__.co_filename) != SOURCE_SHA256:
        raise RuntimeError("stock MLP package/source identity differs from validated environment")
    model = Path(plan["model"])
    index_path = model / "model.safetensors.index.json"
    if sha256(index_path) != MODEL_HASHES[index_path.name]:
        raise RuntimeError("checkpoint index differs from pinned Llama3.1-8B")
    index = json.loads(index_path.read_text())["weight_map"]
    shard_hashes = {}
    weights = {}
    for name in ("gate", "up", "down"):
        key = f"model.layers.0.mlp.{name}_proj.weight"
        path = model / index[key]
        if path.name not in shard_hashes:
            shard_hashes[path.name] = sha256(path)
            if shard_hashes[path.name] != MODEL_HASHES[path.name]:
                raise RuntimeError("MLP checkpoint shard differs from original checkpoint")
        with safe_open(path, framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            shard = tensor[:, :3584] if name == "down" else tensor[:3584, :]
            weights[name] = shard.T.contiguous().to(torch.bfloat16)
    generator = torch.Generator().manual_seed(20261009)
    x = torch.randn((512, 4096), generator=generator).to(torch.bfloat16)
    x = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).to(
        torch.bfloat16
    )
    x = x[: plan["m"]].clone()
    reference = mlp_reference(torch, x, *[weights[name] for name in ("gate", "up", "down")])
    torch.save({"input": x, "weights": weights, "reference_fp32": reference}, root / "inputs.pt")

    def forward(hidden, gate, up, down):
        return mlp(hidden, gate, up, down)

    device_weights = [weights[name].to("neuron:0") for name in ("gate", "up", "down")]
    device_x = x.to("neuron:0")
    if not _can_use_kernel(device_x, device_weights[0]):
        raise RuntimeError("stock MLP would fall back to Torch")
    compiled = torch.compile(
        forward,
        backend=get_compile_backend_name(),
        fullgraph=True,
        options={
            "alias_meta_to_neuron": True,
            "compiler_args": list(COMPILER_ARGS),
            "debug_hlo": True,
        },
    )

    def invoke():
        return compiled(device_x, *device_weights).to("cpu")

    actual = invoke()
    torch.save({"actual_bf16": actual, "reference_fp32": reference}, root / "outputs.pt")
    numerical = check_output(torch, actual, reference)
    (root / "numerical.json").write_text(json.dumps(numerical, indent=2))
    if not numerical["passed"]:
        raise RuntimeError("stock MLP failed the fixed independent FP32 numerical check")
    compile_cache = Path(lite_envs.get_neuron_compile_cache_dir())
    time_ms, executions, trace = measure_lite(
        invoke, torch.classes.neuron.Runtime(), root, compile_cache
    )
    identity = compiled_identity(trace, compile_cache, root, plan["m"])
    receipt = {
        "time_ms": time_ms,
        "executions": executions,
        "compiled": identity,
        "versions": versions,
        "mlp_source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": shard_hashes,
        "numerical": numerical,
        "compiler_args": COMPILER_ARGS,
        "timing_boundary": "one rank-local LNC2 native interval union, no collectives",
        "input_provenance": (
            "seed20261009 RMS-normalized synthetic rows; actual layer0 TP4 rank0 weights"
        ),
        "additivity_validated": False,
        "files_sha256": {
            str(root / name): sha256(root / name)
            for name in (
                "inputs.pt",
                "outputs.pt",
                "numerical.json",
                "invocations.json",
                "system-trace.json",
            )
        },
    }
    (root / "result.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    _run(Path(sys.argv[1]))

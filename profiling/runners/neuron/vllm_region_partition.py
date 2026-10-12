"""Split the stock vLLM Neuron Llama FX graph at the public ``LlamaModel`` return.

Worker-only. The stock Dynamo graph is partitioned, never rewritten: region 0
(``model``) owns the embedding, all decoder layers, the final norm, the SP
gather and every KV-cache mutation; region 1 (``head``) owns row selection, the
lm_head projection, logit collectives and greedy sampling. Each operation keeps
its original op/target/args/kwargs. The stock ``libtorch_neuronx_lite`` backend
compiles each region with the original compiler flags.

``install()`` replaces ``torch.compile`` for the stock Neuron backends only; the
region engine entry calls it before vLLM starts so every spawned rank compiles
through ``compile_regions``. Structural checks fail the compilation (and so the
engine) instead of emitting a partial graph.
"""

from __future__ import annotations

import collections
import copy
import functools
import hashlib
import json
import os
import re
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from profiling.runners.neuron.vllm_region_trace import ORIGINAL_COMPILER_ARGS, REGIONS

PARTITION_DIR_ENV = "SERVINGSTUDIO_REGION_PARTITION_DIR"
STOCK_BACKENDS = ("neuron_libtorch_graph_capture", "neuron_libtorch", "vllm_neuron")
MODEL_LABEL = "L['self'].model"
FINAL_NORM_LABEL = "L['self'].model.norm"
_LAYER_INDEX = re.compile(r"(?:layers\.|layers\[|layers\._modules\[\D*)(\d+)")
_KV_SUFFIXES = ("_k_cache", "_v_cache")
_STRUCTURAL = ("placeholder", "get_attr", "output")


class RegionPartitionError(ValueError):
    """The stock graph cannot be split without changing operations or ownership."""


def _require(condition, message):
    if not condition:
        raise RegionPartitionError(message)


def original_name(node) -> str:
    return node.meta.get("stock_original_name", node.name)


def signature(node) -> dict:
    """op/target/args/kwargs with node references resolved to original names."""

    def normalize(value):
        if isinstance(value, torch.fx.Node):
            return {"node": original_name(value)}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        return {"type": type(value).__name__, "repr": repr(value)}

    return {
        "op": node.op,
        "target": str(node.target),
        "args": normalize(node.args),
        "kwargs": normalize(node.kwargs),
    }


def _module_labels(node) -> list[str]:
    stack = node.meta.get("nn_module_stack", {})
    return [str(value[0]) for value in stack.values() if isinstance(value, tuple)]


def _graph_geometry(gm) -> tuple[str, int]:
    """(phase, token_bucket) from the original placeholders, before splitting."""
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    tokens = [node for node in placeholders if node.name.endswith("input_ids_")]
    _require(len(tokens) == 1, "stock graph must have exactly one input_ids placeholder")
    value = tokens[0].meta.get("example_value", tokens[0].meta.get("val"))
    _require(value is not None and len(value.shape) == 1, "input_ids must be one-dimensional")
    decode = any(node.name.endswith("block_table_tensor_") for node in placeholders)
    return ("decode" if decode else "prefill"), int(value.shape[0])


def partition_model_and_head(gm, *, num_layers: int = 32):
    """Split ``gm`` into submod_0 (model) and submod_1 (head); return (split, report).

    Ownership is assigned by original order: every compute node up to and
    including the last one under the public ``LlamaModel`` module goes to the
    model region. Grad-mode context nodes that ``split_module`` duplicates into
    the other region are removed; any other duplicate fails.
    """
    from torch.fx.passes.split_module import split_module

    nodes = list(gm.graph.nodes)
    compute = [node for node in nodes if node.op not in _STRUCTURAL]
    labels = {node: _module_labels(node) for node in compute}
    layers = {
        node: {int(index) for label in labels[node] for index in _LAYER_INDEX.findall(label)}
        for node in compute
    }
    observed = set().union(*layers.values()) if compute else set()
    _require(observed == set(range(num_layers)), f"expected {num_layers} layers, got {observed}")
    model_nodes = [i for i, node in enumerate(compute) if MODEL_LABEL in labels[node]]
    _require(model_nodes, "missing public LlamaModel module metadata")
    last = max(model_nodes)
    _require(0 < last < len(compute) - 1, "LlamaModel boundary leaves an empty region")
    _require(
        any(FINAL_NORM_LABEL in labels[node] for node in compute[: last + 1]),
        "final norm is not inside the model region",
    )
    assignment = {node: 0 if i <= last else 1 for i, node in enumerate(compute)}
    _require(
        all(assignment[node] == 0 for node in compute if layers[node]),
        "decoder layer operation escaped the model region",
    )
    for node in compute:
        for parent in node.all_input_nodes:
            if parent in assignment:
                _require(assignment[parent] <= assignment[node], "head feeds back into model")
    for node in compute:
        node.meta["stock_original_name"] = node.name
    by_name = {node.name: node for node in compute}
    split = split_module(
        gm,
        gm,
        lambda node: assignment[node],
        keep_original_order=True,
        keep_original_node_name=True,
    )
    regions = {}
    owned_names = []
    for index in range(len(REGIONS)):
        sub = split.get_submodule(f"submod_{index}")
        removed = _drop_duplicated_grad_context(sub, by_name, assignment, index)
        names = []
        for node in sub.graph.nodes:
            if node.op in _STRUCTURAL:
                continue
            original = by_name.get(original_name(node))
            _require(original is not None, f"region created a new operation: {node.name}")
            _require(
                signature(node) == signature(original),
                f"changed op/target/args/kwargs: {original.name}",
            )
            names.append(original.name)
        owned_names += names
        placeholders = [node.name for node in sub.graph.nodes if node.op == "placeholder"]
        regions[REGIONS[index]] = {
            "operations": len(names),
            "placeholders": placeholders,
            "cache_inputs": [name for name in placeholders if name.endswith(_KV_SUFFIXES)],
            "removed_grad_context": removed,
        }
    _require(
        collections.Counter(owned_names) == collections.Counter(node.name for node in compute),
        "compute-node ownership changed",
    )
    _require(
        len(regions["model"]["cache_inputs"]) == 2 * num_layers,
        "model region does not own every KV cache input",
    )
    _require(not regions["head"]["cache_inputs"], "head region owns a KV cache input")
    phase, bucket = _graph_geometry(gm)
    report = {
        "passed": True,
        "phase": phase,
        "token_bucket": bucket,
        "original_operations": len(compute),
        "layers": sorted(observed),
        "regions": regions,
        "boundary": "public LlamaModel.forward return, including final norm and SP gather",
    }
    return split, report


def _drop_duplicated_grad_context(sub, by_name, assignment, index) -> list[str]:
    removed = []
    for node in list(sub.graph.nodes):
        if node.op in _STRUCTURAL:
            continue
        original = by_name.get(original_name(node))
        if original is not None and assignment[original] != index:
            _require(
                node.op == "call_function"
                and node.target == torch._C._set_grad_enabled
                and not node.users,
                f"operation duplicated across regions: {node.name}",
            )
            removed.append(original.name)
            sub.graph.erase_node(node)
    sub.graph.lint()
    sub.recompile()
    return removed


def check_kv_aliases(report: dict, aliases: dict[str, dict[str, str]]) -> None:
    """Every model-region KV input must be an aliased output; head has none."""
    model, head = report["regions"]["model"], report["regions"]["head"]
    _require(
        set(aliases["model"].values()) == set(model["cache_inputs"]),
        "lowered model region lost or changed KV alias ownership",
    )
    _require(not aliases["head"], "lowered head region aliases an input")
    model["alias_outputs"], head["alias_outputs"] = aliases["model"], aliases["head"]


def _lowered_aliases(sub, options, workdir) -> dict[str, str]:
    from libtorch_neuronx_lite.compile.capture_backend import _run_fx_passes

    placeholders = [node.name for node in sub.graph.nodes if node.op == "placeholder"]
    _, io_map, _ = _run_fx_passes(copy.deepcopy(sub), options, str(workdir))
    return {str(output): placeholders[index] for output, index in (io_map or {}).items()}


def _audit_dir(gm) -> Path:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    key = hashlib.sha256(str(gm.graph).encode()).hexdigest()[:16]
    root = Path(os.environ[PARTITION_DIR_ENV])
    out = root / f"rank{rank}-{key}"
    out.mkdir(parents=True, exist_ok=True)
    return out


class _Region(torch.nn.Module):
    """Run one compiled region; check the output contract and KV identity early."""

    checked_calls = 2

    def __init__(self, execute, spec, expected, cache_indices):
        super().__init__()
        self.execute, self.spec = execute, spec
        self.expected, self.cache_indices = expected, cache_indices
        self.calls, self.cache_layout = 0, None

    def forward(self, *args):
        if self.calls >= self.checked_calls:
            output = self.execute(*args)
            if len(output) != len(self.expected):
                raise RuntimeError("region output arity changed")
            return tree_unflatten(output, self.spec)
        layout = [_descriptor(args[i]) for i in self.cache_indices]
        if self.cache_layout is not None and layout != self.cache_layout:
            raise RuntimeError("KV buffer storage or layout changed across calls")
        self.cache_layout = layout
        output = self.execute(*args)
        for tensor, (shape, dtype) in zip(output, self.expected, strict=True):
            if list(tensor.shape) != shape or str(tensor.dtype) != dtype:
                raise RuntimeError("region output shape or dtype changed")
            if not str(tensor.device).startswith("neuron"):
                raise RuntimeError("region output left the Neuron device")
        if [_descriptor(args[i]) for i in self.cache_indices] != layout:
            raise RuntimeError("region replaced a KV cache buffer")
        self.calls += 1
        return tree_unflatten(output, self.spec)


def _descriptor(tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), tensor.data_ptr())


def _example(node):
    value = node.meta.get("example_value", node.meta.get("val"))
    if not isinstance(value, torch.Tensor):
        raise RegionPartitionError(f"missing example tensor for {node.name}")
    return value


def compile_regions(gm, example_inputs, *, capture_only=False, options=None, **_):
    """Torch Dynamo backend: partition, audit, then stock-compile both regions."""
    from libtorch_neuronx_lite.compile.backend import compile as stock_compile
    from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
    from libtorch_neuronx_lite.compile.capture_backend import capture as stock_capture

    options = dict(options or {})
    incoming = list(options.get("compiler_args", []))
    options["compiler_args"] = list(ORIGINAL_COMPILER_ARGS)
    out = _audit_dir(gm)
    (out / "original.fx.txt").write_text(str(gm.graph))
    split, report = partition_model_and_head(gm)
    subs = [split.get_submodule(f"submod_{i}") for i in range(len(REGIONS))]
    check_kv_aliases(
        report,
        {
            name: _lowered_aliases(sub, options, out / f"{name}-passes")
            for name, sub in zip(REGIONS, subs)
        },
    )
    for name, sub in zip(REGIONS, subs):
        (out / f"{name}.fx.txt").write_text(str(sub.graph))
    report.update(
        capture_only=capture_only,
        incoming_compiler_args=incoming,
        compiler_args=options["compiler_args"],
    )
    (out / "receipt.json").write_text(json.dumps(report, indent=2))
    for index, sub in enumerate(subs):
        placeholders = [node for node in sub.graph.nodes if node.op == "placeholder"]
        examples = [_example(node) for node in placeholders]
        if capture_only:
            stock_capture(copy.deepcopy(sub), examples, options=options)
            continue
        leaves, spec = tree_flatten(list(sub.graph.nodes)[-1].args[0])
        expected = [(list(_example(leaf).shape), str(_example(leaf).dtype)) for leaf in leaves]
        cache_indices = [
            i for i, node in enumerate(placeholders) if node.name.endswith(_KV_SUFFIXES)
        ]
        executable = stock_compile(copy.deepcopy(sub), examples, options=options)
        split.set_submodule(f"submod_{index}", _Region(executable, spec, expected, cache_indices))
    if capture_only:

        def captured(*args, **kwargs):
            raise CaptureComplete()

        return captured
    return split.forward


def install() -> None:
    """Route the stock Neuron torch.compile backends through ``compile_regions``."""
    original = torch.compile

    def adapter(model=None, **kwargs):
        backend = kwargs.get("backend")
        if backend in STOCK_BACKENDS:
            kwargs["backend"] = functools.partial(
                compile_regions, capture_only=backend == "neuron_libtorch_graph_capture"
            )
        return original(model, **kwargs)

    torch.compile = adapter

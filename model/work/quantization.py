"""Checkpoint quantization, read straight off the model config.

A checkpoint declares its own precision. HF FP8 repos keep ``dtype:
"bfloat16"`` at the top level (that is the *master* dtype the weights were
trained in) and add a single ``quantization_config`` key describing what was
actually stored on disk. Everything this module needs comes from that key, so
``load_model(path)`` needs no extra argument: pointing at ``glm52.json`` gives
a BF16 accountant, ``glm52_fp8.json`` gives an FP8 one, and the ModelOpt
``glm52_nvfp4.json`` config gives an NVFP4 routed-expert accountant. A ModelOpt
config that instead lists its unquantized modules (``ignore``/``exclude_modules``)
converts every other Linear. A ModelOpt ``MIXED_PRECISION`` config lists the opposite:
``quantized_layers`` names every converted Linear with its own algorithm (static
per-tensor FP8, or NVFP4), and every Linear it does not name stays at the master
dtype (vLLM's ``ModelOptMixedPrecisionConfig``).

Two things follow from a scheme and both matter to the necessary-work floor:

- **weight bytes.** FP8 uses one byte per element plus an FP32 block scale;
  NVFP4 uses half a byte per element plus an FP8 scale per 16 weights. These
  scales are genuinely read from HBM alongside the weight.
- **compute dtype.** A converted matmul runs on the corresponding low-precision
  tensor cores, so its compute floor divides by a different peak. The model
  is mixed: ``mlp.gate`` and the GLM indexer's ``indexers_proj`` stay BF16 even
  in an FP8 repo, and so does every norm, the embedding, and ``lm_head``.

DeepSeek-V4.1 extends the FP8 block scheme in two ways its config states
directly: ``scale_fmt: "ue8m0"`` stores each block scale as one E8M0 byte rather
than an FP32 word, and ``expert_dtype: "fp4"`` stores routed experts as packed
E2M1 with one E8M0 scale per 32 weights along K (MXFP4). Those experts multiply
FP8 activations, which Blackwell's mixed FP8xFP4 MMA executes at the FP8 rate,
so their compute dtype stays ``fp8``.

vLLM's ``Fp8Config`` also reads two keys MiMo-V2 uses: ``ignored_layers`` (preferred
over ``modules_to_not_convert``; exact checkpoint paths) and ``store_dtype: "mxfp4"``,
which stores routed experts as packed E2M1 with one E8M0 scale per
``mxfp4_block_size`` weights along K. Those experts take unquantized activations,
so they compute at the master dtype.

For FP8, ``modules_to_not_convert`` (or ``ignored_layers``) is the authority for which
is which. It is stated as fully-qualified checkpoint paths (``model.layers.7.mlp.gate``); a
:class:`~model.work.core.MatmulGroup` names itself with the layer-relative path
(``mlp.gate``), so the list is normalized to layer-relative form once here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

# Strips the two wrappers HF puts in front of a layer-relative module path.
_LAYER_PREFIX = re.compile(r"^model\.(?:language_model\.)?layers\.\d+\.")
_MODEL_PREFIX = re.compile(r"^model\.")
_LAYER_INDEX = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")

_SUPPORTED_QUANT_METHODS = ("fp8", "modelopt")


def _layer_relative(module: str) -> str:
    return _MODEL_PREFIX.sub("", _LAYER_PREFIX.sub("", module))


@dataclass(frozen=True)
class QuantScheme:
    """How a checkpoint stored its weights, in the terms the accountant needs."""

    bytes_per_weight: float
    compute_dtype: str
    #: (block_n, block_k) of a shared scale, or None for per-tensor scales.
    block_shape: tuple[int, int] | None
    scale_dtype_bytes: float
    #: Layer-relative module paths that were left at the master dtype.
    not_converted: frozenset[str]
    #: If set, only these module subtrees are converted. ModelOpt's GLM-5.2
    #: NVFP4 checkpoint uses this to quantize routed experts and nothing else.
    converted_prefixes: frozenset[str] | None = None
    #: module -> is_converted. The answer depends only on the module path, but
    #: `Model.label` asks twice per matmul group per call; on a FP8 checkpoint
    #: with ~1,000 excluded paths the scan was ~90% of every label.
    _converted_memo: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )
    #: Converted module subtrees stored under a different scheme, as
    #: ``(prefix, scheme)`` pairs; DeepSeek-V4.1's MXFP4 routed experts.
    module_schemes: tuple[tuple[str, QuantScheme], ...] = ()

    def scheme_for(self, module: str | None) -> QuantScheme:
        """The scheme a converted ``module`` is stored under."""
        if module is not None:
            for prefix, scheme in self.module_schemes:
                if module == prefix or module.startswith(f"{prefix}."):
                    return scheme
        return self

    def is_converted(self, module: str) -> bool:
        """Was ``module`` (a layer-relative path) actually quantized?

        Matching is by path component, not raw string prefix: ``mlp.gate`` must
        not swallow the dense FFN's ``mlp.gate_proj``.
        """
        converted = self._converted_memo.get(module)
        if converted is None:
            converted = self._converted_memo[module] = self._match_converted(module)
        return converted

    def _match_converted(self, module: str) -> bool:
        if self.converted_prefixes is not None and not any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in self.converted_prefixes
        ):
            return False
        for excluded in self.not_converted:
            if module == excluded or module.startswith(f"{excluded}."):
                return False
        return True

    def scale_bytes(self, n: int, k: int) -> float:
        """Block-scale bytes read alongside an ``n x k`` converted matrix."""
        if self.block_shape is None:
            return self.scale_dtype_bytes
        block_n, block_k = self.block_shape
        return math.ceil(n / block_n) * math.ceil(k / block_k) * self.scale_dtype_bytes


def _modelopt_group_size(quant: dict) -> int:
    """The weight group size of a ModelOpt NVFP4 config, in either layout.

    The flat layout states ``weight_group_size``; the compressed-tensors layout
    (``config_groups``, as ModelOpt 0.47 writes) states it per group, and
    every group must agree.
    """
    groups = quant.get("config_groups")
    if not groups:
        return int(quant.get("weight_group_size", 16))
    sizes = {int(group["weights"]["group_size"]) for group in groups.values()}
    if len(sizes) != 1:
        raise ValueError(f"ModelOpt config_groups disagree on weight group size: {sorted(sizes)}")
    return sizes.pop()


def modelopt_excluded(quant: dict) -> list[str] | None:
    """A ModelOpt config's unquantized modules, from ``ignore`` or ``exclude_modules``.

    Entries are checkpoint paths with at most one ``*``, trailing. A leading or
    inner wildcard (``*.mlp.gate``) cannot be resolved to module paths here and is
    refused.
    """
    excluded = quant.get("ignore", quant.get("exclude_modules"))
    if excluded is None:
        return None
    for entry in excluded:
        if "*" in entry.rstrip("*") or entry.count("*") > 1:
            raise ValueError(f"ModelOpt exclusion {entry!r}: only one trailing `*` is modeled")
    return list(excluded)


def uniform_layer_exclusions(entries: list[str], num_layers: int) -> frozenset[str]:
    """Layer-relative exclusions from per-layer checkpoint paths that every body
    layer lists identically.

    ``model.layers.<i>.<module>`` entries must name the same ``<module>`` set for
    every ``i < num_layers``; otherwise one layer-relative set would convert or
    exclude a matrix on the wrong layers, and the builder must resolve the layers
    itself. Entries for layers past the body (a checkpoint's MTP layers) are not
    part of the labelled model and are dropped; any other entry is returned in
    layer-relative form.
    """
    per_layer: list[set[str]] = [set() for _ in range(num_layers)]
    other: set[str] = set()
    for entry in entries:
        match = _LAYER_INDEX.match(entry)
        if match is None:
            other.add(_layer_relative(entry))
            continue
        layer = int(match.group(1))
        if layer < num_layers:
            per_layer[layer].add(match.group(2))
    if any(modules != per_layer[0] for modules in per_layer):
        differing = next(i for i, modules in enumerate(per_layer) if modules != per_layer[0])
        raise ValueError(
            f"layer {differing} excludes {sorted(per_layer[differing])} but layer 0 excludes "
            f"{sorted(per_layer[0])}; a per-layer quantization schedule needs its own stacks"
        )
    return frozenset(per_layer[0] if per_layer else set()) | frozenset(other)


def _nvfp4_scheme(group_size: int) -> QuantScheme:
    """Packed E2M1 weights with one FP8 E4M3 scale per ``group_size`` weights along K,
    multiplied by NVFP4 activations on the FP4 tensor cores."""
    return QuantScheme(
        bytes_per_weight=0.5,
        compute_dtype="fp4",
        block_shape=(1, group_size),
        scale_dtype_bytes=1.0,
        not_converted=frozenset(),
    )


#: ModelOpt FP8: E4M3 weights with one static FP32 scale per tensor, multiplied by
#: statically scaled FP8 activations.
_FP8_PER_TENSOR = QuantScheme(
    bytes_per_weight=1.0,
    compute_dtype="fp8",
    block_shape=None,
    scale_dtype_bytes=4.0,
    not_converted=frozenset(),
)


def _mixed_algo_scheme(module: str, entry: dict) -> QuantScheme:
    algo = str(entry.get("quant_algo", "")).upper()
    if algo == "FP8":
        return _FP8_PER_TENSOR
    if algo == "NVFP4":
        return _nvfp4_scheme(int(entry.get("group_size", 16)))
    raise ValueError(f"MIXED_PRECISION {module!r}: model.work knows FP8 and NVFP4, not {algo!r}")


def _layer_type_algos(entries: dict[str, dict], layer_types: list[str]) -> dict[str, dict]:
    """Layer-relative ``module -> entry`` from per-layer ``quantized_layers`` keys.

    Every layer of one type must quantize the same modules the same way, since one
    layer-relative entry stands for all of them; two types that share a module (the
    MLP under every attention kind) must agree on it. Keys for layers past the body
    (a checkpoint's MTP layers) are not part of the labelled model and are dropped.
    """
    per_layer: list[dict[str, dict]] = [{} for _ in layer_types]
    merged: dict[str, dict] = {}
    for path, entry in entries.items():
        match = _LAYER_INDEX.match(path)
        if match is None:
            merged[_layer_relative(path)] = entry
        elif int(match.group(1)) < len(layer_types):
            per_layer[int(match.group(1))][match.group(2)] = entry
    first: dict[str, int] = {}
    for layer, (kind, modules) in enumerate(zip(layer_types, per_layer, strict=True)):
        reference = first.setdefault(kind, layer)
        if modules != per_layer[reference]:
            raise ValueError(
                f"MIXED_PRECISION quantizes {kind} layer {layer} differently from layer "
                f"{reference}; a per-layer schedule needs its own stacks"
            )
    for reference in first.values():
        for module, entry in per_layer[reference].items():
            if merged.setdefault(module, entry) != entry:
                raise ValueError(f"MIXED_PRECISION quantizes {module!r} differently across layers")
    return merged


def _pack_shards(
    algos: dict[str, dict], packed_modules: dict[str, tuple[str, ...]]
) -> dict[str, dict]:
    """Name fused matrices the way the accountant does (``self_attn.qkv_proj``).

    The checkpoint lists the shards (``q_proj``, ``k_proj``, ``v_proj``); a fused
    matrix is stored under its shards' algorithm, which must be shared by all of
    them — or by none, leaving it at the master dtype.
    """
    packed = dict(algos)
    for fused, shards in packed_modules.items():
        parents = {
            module.rsplit(".", 1)[0]
            for module in algos
            if "." in module and module.rsplit(".", 1)[1] in shards
        }
        for parent in parents:
            entries = [packed.pop(f"{parent}.{shard}", None) for shard in shards]
            if any(entry != entries[0] for entry in entries):
                raise ValueError(
                    f"MIXED_PRECISION lists the shards of {parent}.{fused} as {entries}; "
                    "a fused matrix needs one algorithm for all of them"
                )
            packed[f"{parent}.{fused}"] = entries[0]
    return packed


def _mixed_precision_scheme(
    quant: dict, layer_types: list[str] | None, packed_modules: dict[str, tuple[str, ...]]
) -> QuantScheme:
    """ModelOpt ``MIXED_PRECISION``: only the listed Linears are converted, each
    under its own algorithm. The scheme's own fields are the lowest-precision
    algorithm's (its headline precision); every other algorithm's modules carry
    theirs in ``module_schemes``."""
    if layer_types is None:
        raise ValueError(
            "ModelOpt MIXED_PRECISION names single layers; the model builder must pass "
            "its layer schedule"
        )
    # An excluded module is left unquantized even if listed (vLLM checks it first).
    excluded = [
        entry[:-1] if entry.endswith("*") else f"{entry}."
        for entry in modelopt_excluded(quant) or ()
    ]
    entries = {
        path: entry
        for path, entry in quant["quantized_layers"].items()
        if not any(f"{path}.".startswith(prefix) for prefix in excluded)
    }
    algos = _pack_shards(_layer_type_algos(entries, layer_types), packed_modules)
    schemes = {module: _mixed_algo_scheme(module, entry) for module, entry in algos.items()}
    if not schemes:
        raise ValueError("ModelOpt MIXED_PRECISION config lists no quantized_layers")
    base = min(schemes.values(), key=lambda scheme: scheme.bytes_per_weight)
    return replace(
        base,
        converted_prefixes=frozenset(schemes),
        module_schemes=tuple(
            (module, scheme) for module, scheme in sorted(schemes.items()) if scheme != base
        ),
    )


def parse_quantization_config(
    raw_config: dict,
    *,
    layer_qualified_exclusions: bool = False,
    layer_types: list[str] | None = None,
    packed_modules: dict[str, tuple[str, ...]] | None = None,
) -> QuantScheme | None:
    """Build a :class:`QuantScheme` from a raw HF config, or None if unquantized.

    A ModelOpt NVFP4 exclusion list that names single layers
    (``model.layers.7.self_attn*``) differs layer by layer, and one layer-relative
    set cannot say which; it raises unless the caller passes
    ``layer_qualified_exclusions=True``, which a model builder that knows the layer
    schedule does before replacing ``not_converted`` with its own per-layer reading.

    A ModelOpt ``MIXED_PRECISION`` config names every quantized Linear per layer, so
    it needs the builder's ``layer_types`` (one entry per decoder layer) and the
    ``packed_modules`` its fused matrices are built from (``{"qkv_proj": ("q_proj",
    "k_proj", "v_proj")}``, vLLM's ``packed_modules_mapping``).

    An unrecognized ``quant_method`` raises rather than silently falling back to
    the master dtype: a wrong precision here shows up as a "minimum" larger than
    the measured traffic, which is the one failure mode this accountant must
    never have.
    """
    quant = raw_config.get("quantization_config")
    if quant is None:
        return None
    method = quant.get("quant_method")
    if method not in _SUPPORTED_QUANT_METHODS:
        raise ValueError(
            f"unsupported quant_method {method!r}; model.work knows "
            f"{_SUPPORTED_QUANT_METHODS} — extend quantization.py before using this config"
        )
    if method == "modelopt":
        if quant.get("quant_algo") == "MIXED_PRECISION":
            return _mixed_precision_scheme(quant, layer_types, packed_modules or {})
        if quant.get("quant_algo") != "NVFP4":
            raise ValueError(
                "model.work only supports ModelOpt checkpoints with quant_algo='NVFP4' "
                "or 'MIXED_PRECISION'"
            )
        excluded = modelopt_excluded(quant)
        if not quant.get("routed_experts_only") and excluded is None:
            raise ValueError(
                "ModelOpt NVFP4 configs must declare routed_experts_only=true or list "
                "the unquantized modules under `ignore`"
            )
        group_size = _modelopt_group_size(quant)
        if group_size != 16:
            raise ValueError(f"unsupported NVFP4 weight_group_size {group_size}; expected 16")
        if quant.get("routed_experts_only"):
            not_converted = frozenset()
            converted_prefixes = frozenset({"mlp.experts"})
        else:
            # Every Linear is converted except the listed modules.
            qualified = [entry for entry in excluded if _LAYER_PREFIX.match(entry)]
            if qualified and not layer_qualified_exclusions:
                raise ValueError(
                    f"ModelOpt exclusions name single layers ({qualified[0]!r}, ...); "
                    "the model builder must resolve them per layer"
                )
            not_converted = frozenset(
                _layer_relative(module.rstrip("*").rstrip(".")) for module in excluded
            )
            converted_prefixes = None
        return replace(
            _nvfp4_scheme(group_size),
            not_converted=not_converted,
            converted_prefixes=converted_prefixes,
        )

    fmt = quant.get("fmt")
    if fmt not in (None, "e4m3", "e5m2"):
        raise ValueError(f"unsupported fp8 fmt {fmt!r}")
    scale_fmt = quant.get("scale_fmt")
    if scale_fmt not in (None, "ue8m0"):
        raise ValueError(f"unsupported fp8 scale_fmt {scale_fmt!r}")
    # A UE8M0 scale is a single exponent byte; the historical default is FP32.
    scale_dtype_bytes = 1.0 if scale_fmt == "ue8m0" else 4.0
    block = quant.get("weight_block_size")
    block_shape = (int(block[0]), int(block[1])) if block else None
    expert_dtype = quant.get("expert_dtype")
    store_dtype = quant.get("store_dtype")
    module_schemes: tuple[tuple[str, QuantScheme], ...] = ()
    if store_dtype is not None:
        master = raw_config.get("dtype") or raw_config.get("torch_dtype")
        if store_dtype != "mxfp4" or expert_dtype is not None or master != "bfloat16":
            raise ValueError(
                f"unsupported store_dtype {store_dtype!r} (expert_dtype {expert_dtype!r}, "
                f"master {master!r}); model.work knows only MXFP4 experts over BF16"
            )
        module_schemes = (
            (
                "mlp.experts",
                QuantScheme(
                    bytes_per_weight=0.5,
                    # vLLM's Mxfp4MoEMethod multiplies unquantized BF16 activations.
                    compute_dtype="bf16",
                    block_shape=(1, int(quant.get("mxfp4_block_size", 32))),
                    scale_dtype_bytes=1.0,
                    not_converted=frozenset(),
                ),
            ),
        )
    if expert_dtype is not None:
        if expert_dtype != "fp4" or scale_fmt != "ue8m0":
            raise ValueError(
                f"unsupported expert_dtype {expert_dtype!r} with scale_fmt {scale_fmt!r}; "
                "model.work knows only MXFP4 experts (fp4 + ue8m0)"
            )
        mxfp4_group = 32
        module_schemes = (
            (
                "mlp.experts",
                QuantScheme(
                    bytes_per_weight=0.5,
                    compute_dtype="fp8",
                    block_shape=(1, mxfp4_group),
                    scale_dtype_bytes=1.0,
                    not_converted=frozenset(),
                ),
            ),
        )
    # vLLM reads `ignored_layers` first and falls back to `modules_to_not_convert`.
    excluded = quant.get("ignored_layers") or quant.get("modules_to_not_convert") or ()
    qualified = [entry for entry in excluded if _LAYER_PREFIX.match(entry)]
    if "ignored_layers" in quant and qualified and not layer_qualified_exclusions:
        raise ValueError(
            f"FP8 ignored_layers name single layers ({qualified[0]!r}, ...); "
            "the model builder must resolve them per layer"
        )
    return QuantScheme(
        bytes_per_weight=1.0,
        compute_dtype="fp8",
        block_shape=block_shape,
        scale_dtype_bytes=scale_dtype_bytes,
        not_converted=frozenset(_layer_relative(module) for module in excluded),
        module_schemes=module_schemes,
    )

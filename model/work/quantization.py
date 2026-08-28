"""Checkpoint quantization, read straight off the model config.

A checkpoint declares its own precision. HF FP8 repos keep ``dtype:
"bfloat16"`` at the top level (that is the *master* dtype the weights were
trained in) and add a single ``quantization_config`` key describing what was
actually stored on disk. Everything this module needs comes from that key, so
``load_model(path)`` needs no extra argument: pointing at ``glm52.json`` gives
a BF16 accountant, ``glm52_fp8.json`` gives an FP8 one, and the ModelOpt
``glm52_nvfp4.json`` config gives an NVFP4 routed-expert accountant.

Two things follow from a scheme and both matter to the necessary-work floor:

- **weight bytes.** FP8 uses one byte per element plus an FP32 block scale;
  NVFP4 uses half a byte per element plus an FP8 scale per 16 weights. These
  scales are genuinely read from HBM alongside the weight.
- **compute dtype.** A converted matmul runs on the corresponding low-precision
  tensor cores, so its compute floor divides by a different peak. The model
  is mixed: ``mlp.gate`` and the GLM indexer's ``indexers_proj`` stay BF16 even
  in an FP8 repo, and so does every norm, the embedding, and ``lm_head``.

For FP8, ``modules_to_not_convert`` is the authority for which is which. It is stated
as fully-qualified checkpoint paths (``model.layers.7.mlp.gate``); a
:class:`~model.work.core.MatmulGroup` names itself with the layer-relative path
(``mlp.gate``), so the list is normalized to layer-relative form once here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Strips the two wrappers HF puts in front of a layer-relative module path.
_LAYER_PREFIX = re.compile(r"^model\.(?:language_model\.)?layers\.\d+\.")
_MODEL_PREFIX = re.compile(r"^model\.")

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

    def is_converted(self, module: str) -> bool:
        """Was ``module`` (a layer-relative path) actually quantized?

        Matching is by path component, not raw string prefix: ``mlp.gate`` must
        not swallow the dense FFN's ``mlp.gate_proj``.
        """
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


def parse_quantization_config(raw_config: dict) -> QuantScheme | None:
    """Build a :class:`QuantScheme` from a raw HF config, or None if unquantized.

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
        if quant.get("quant_algo") != "NVFP4":
            raise ValueError(
                "model.work only supports ModelOpt checkpoints with quant_algo='NVFP4'"
            )
        if not quant.get("routed_experts_only"):
            raise ValueError(
                "ModelOpt NVFP4 configs must declare routed_experts_only=true; "
                "the accountant cannot safely infer layer-relative exclusions from wildcards"
            )
        group_size = int(quant.get("weight_group_size", 16))
        if group_size != 16:
            raise ValueError(f"unsupported NVFP4 weight_group_size {group_size}; expected 16")
        return QuantScheme(
            bytes_per_weight=0.5,
            compute_dtype="fp4",
            # The checkpoint stores one FP8 E4M3 scale per 16 weights.
            block_shape=(1, group_size),
            scale_dtype_bytes=1.0,
            not_converted=frozenset(),
            converted_prefixes=frozenset({"mlp.experts"}),
        )

    fmt = quant.get("fmt")
    if fmt not in (None, "e4m3", "e5m2"):
        raise ValueError(f"unsupported fp8 fmt {fmt!r}")
    block = quant.get("weight_block_size")
    block_shape = (int(block[0]), int(block[1])) if block else None
    return QuantScheme(
        bytes_per_weight=1.0,
        compute_dtype="fp8",
        block_shape=block_shape,
        scale_dtype_bytes=4.0,
        not_converted=frozenset(
            _layer_relative(module) for module in quant.get("modules_to_not_convert", ())
        ),
    )

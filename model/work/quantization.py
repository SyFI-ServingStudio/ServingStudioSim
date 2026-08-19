"""Checkpoint quantization, read straight off the model config.

A checkpoint declares its own precision. HF FP8 repos keep ``dtype:
"bfloat16"`` at the top level (that is the *master* dtype the weights were
trained in) and add a single ``quantization_config`` key describing what was
actually stored on disk. Everything this module needs comes from that key, so
``load_model(path)`` needs no extra argument: pointing at ``glm52.json`` gives
a BF16 accountant and pointing at ``glm52_fp8.json`` gives an FP8 one.

Two things follow from a scheme and both matter to the necessary-work floor:

- **weight bytes.** A converted matrix is one byte per element plus its
  per-block FP32 scale, which is genuinely read from HBM alongside the weight.
- **compute dtype.** A converted matmul runs on the FP8 tensor cores, so its
  compute floor divides by a different peak than an unconverted one. The model
  is mixed: ``mlp.gate`` and the GLM indexer's ``indexers_proj`` stay BF16 even
  in an FP8 repo, and so does every norm, the embedding, and ``lm_head``.

``modules_to_not_convert`` is the authority for which is which. It is stated
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

_SUPPORTED_QUANT_METHODS = ("fp8",)


def _layer_relative(module: str) -> str:
    return _MODEL_PREFIX.sub("", _LAYER_PREFIX.sub("", module))


@dataclass(frozen=True)
class QuantScheme:
    """How a checkpoint stored its weights, in the terms the accountant needs."""

    bytes_per_weight: float
    compute_dtype: str
    #: (block_n, block_k) of the shared FP32 scale, or None for per-tensor scales.
    block_shape: tuple[int, int] | None
    scale_dtype_bytes: float
    #: Layer-relative module paths that were left at the master dtype.
    not_converted: frozenset[str]

    def is_converted(self, module: str) -> bool:
        """Was ``module`` (a layer-relative path) actually quantized?

        Matching is by path component, not raw string prefix: ``mlp.gate`` must
        not swallow the dense FFN's ``mlp.gate_proj``.
        """
        for excluded in self.not_converted:
            if module == excluded or module.startswith(f"{excluded}."):
                return False
        return True

    def scale_bytes(self, n: int, k: int) -> float:
        """FP32 block-scale bytes read alongside an ``n x k`` converted matrix."""
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

"""The gated activation a TRT-LLM NVFP4 fused-MoE GEMM1 epilogue applies.

The ``gated_activation`` profiler arg of ``nvfp4_fused_moe`` names it. Absent
means ``swiglu``: FlashInfer ``ActivationType.Swiglu`` with no alpha, beta or
clamp, the call every row measured before the arg existed. ``situ`` is Kimi's
SiTU-GLU (``ActivationType.Situ``), written with its two per-model constants:

    situ:beta=<activation_situ_beta>,linear_beta=<activation_situ_linear_beta>

vLLM's ``TrtLlmNvFp4ExpertsBase`` passes them as per-expert float32 tensors:
``gemm1_alpha = full(local_experts, situ_beta)`` and ``gemm1_beta =
full(local_experts, situ_linear_beta)``, unfolded by ``g1_alphas``, with no
clamp. The kernel computes ``beta*tanh(gate/beta)*sigmoid(gate) *
linear_beta*tanh(up/linear_beta)`` and FlashInfer selects the ``siTuGlu``
GEMM1 cubins, so the activation changes the kernels and the tactic space.

The numbers are written in their shortest round-trip form without a trailing
``.0`` (``4``, ``25``, ``0.5``), the form Rust's ``{}`` gives an ``f64``, so the
simulator and the profiler spell one call one way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SWIGLU = "swiglu"
SITU = "situ"

# flashinfer.tllm_enums.ActivationType values (stable for the TRT-LLM backend).
ACTIVATION_TYPE = {SWIGLU: 3, SITU: 10}

_SITU = re.compile(r"situ:beta=([0-9.eE+-]+),linear_beta=([0-9.eE+-]+)")


def _num(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


@dataclass(frozen=True)
class GatedActivation:
    name: str
    beta: float | None = None
    linear_beta: float | None = None

    @property
    def activation_type(self) -> int:
        return ACTIVATION_TYPE[self.name]

    @property
    def is_default(self) -> bool:
        return self.name == SWIGLU

    def canonical(self) -> str:
        if self.name == SWIGLU:
            return SWIGLU
        return f"situ:beta={_num(self.beta)},linear_beta={_num(self.linear_beta)}"

    def label_suffix(self) -> str:
        """Tactic-file suffix: empty for the default, so existing files keep their names."""
        if self.is_default:
            return ""
        return f".situ-b{_num(self.beta)}-lb{_num(self.linear_beta)}"


def situ(beta: float, linear_beta: float) -> str:
    return GatedActivation(SITU, float(beta), float(linear_beta)).canonical()


def parse(value: str | None) -> GatedActivation:
    """Parse a ``gated_activation`` arg; ``None`` is the default SwiGLU."""
    if value is None or value == SWIGLU:
        return GatedActivation(SWIGLU)
    match = _SITU.fullmatch(value)
    if match is None:
        raise ValueError(
            f"unsupported gated_activation {value!r}: expected 'swiglu' or "
            "'situ:beta=<b>,linear_beta=<lb>'"
        )
    beta, linear_beta = float(match.group(1)), float(match.group(2))
    # vLLM asserts both > 0 (the TRT-LLM SiTuGlu cubin has no up-passthrough).
    if not (beta > 0 and linear_beta > 0):
        raise ValueError("situ needs beta > 0 and linear_beta > 0, as vLLM asserts")
    parsed = GatedActivation(SITU, beta, linear_beta)
    if parsed.canonical() != value:
        raise ValueError(
            f"gated_activation {value!r} is not canonical; write {parsed.canonical()!r}"
        )
    return parsed


def gemm1_kwargs(torch, activation: GatedActivation, local_experts: int) -> dict:
    """The activation's ``trtllm_fp4_block_scale_*moe`` kwargs, as vLLM passes them."""
    if activation.is_default:
        return {
            "activation_type": activation.activation_type,
            "gemm1_alpha": None,
            "gemm1_beta": None,
            "gemm1_clamp_limit": None,
        }

    def per_expert(value: float):
        return torch.full((local_experts,), value, dtype=torch.float32, device="cuda")

    return {
        "activation_type": activation.activation_type,
        "gemm1_alpha": per_expert(activation.beta),
        "gemm1_beta": per_expert(activation.linear_beta),
        "gemm1_clamp_limit": None,
    }

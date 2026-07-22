"""Optimal necessary-work labeler.

The independent ground-truth accountant for VibeSim's redundancy analysis: from a
model ``config.json`` + a :class:`Workload`, compute the theoretical MINIMUM compute
(FLOPs) and memory traffic (bytes), plus total/activated parameter counts. See
``README.md`` for the contract and the four pinned conventions.

    from model.work import load_model, Workload
    label = load_model("model/config/llama3_8b.json").label(
        Workload.causal_lm(decode=[4096] * 256, sampled=256)
    )
"""

from .core import AttnInteraction, LayerStack, MatmulGroup, Model, WorkLabel, Workload
from .registry import UnknownArchitecture, build_model, load_model

__all__ = [
    "AttnInteraction",
    "LayerStack",
    "MatmulGroup",
    "Model",
    "WorkLabel",
    "Workload",
    "UnknownArchitecture",
    "build_model",
    "load_model",
]

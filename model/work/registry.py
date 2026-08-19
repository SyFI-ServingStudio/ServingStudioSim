"""Dispatch a raw HF ``config.json`` to its min-work builder by ``architectures[0]``.

This is the same key HF/vLLM use to pick a model implementation. The architecture
string is the model's *structure*; the config numbers are *data*, so one builder
covers every checkpoint of a family (e.g. all ``Qwen3MoeForCausalLM`` sizes). An
unregistered architecture is a hard error — that doubles as the "we have not labeled
this model yet" guard, so the analyzer refuses rather than silently mislabels.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .core import Model
from .models import glm52, llama3, qwen3_6, qwen3_6_moe, qwen3_moe

# architecture string -> builder. Add a row when you add a model file.
REGISTRY: dict[str, Callable[[dict], Model]] = {
    "LlamaForCausalLM": llama3.build,
    "MistralForCausalLM": llama3.build,  # also (GQA, dense)
    "Qwen3MoeForCausalLM": qwen3_moe.build,
    "Qwen3_5ForConditionalGeneration": qwen3_6.build,  # Qwen3.5/3.6 hybrid (linear+full)
    "Qwen3_5MoeForConditionalGeneration": qwen3_6_moe.build,
    "GlmMoeDsaForCausalLM": glm52.build,
}


class UnknownArchitecture(KeyError):
    """Raised when a config's architecture has no registered min-work builder."""


def build_model(raw_config: dict) -> Model:
    architectures = raw_config.get("architectures") or []
    if len(architectures) != 1:
        raise ValueError(f"expected exactly one architecture, got {architectures!r}")
    architecture = architectures[0]
    builder = REGISTRY.get(architecture)
    if builder is None:
        raise UnknownArchitecture(
            f"{architecture} has no min-work builder; add one under work/models/ and "
            f"register it in work/registry.py (see skill impl-add-model-work-label)"
        )
    return builder(raw_config)


def load_model(config_path: str | Path) -> Model:
    """Load a HF ``config.json`` and dispatch to its builder."""
    with open(config_path) as handle:
        raw_config = json.load(handle)
    return build_model(raw_config)

"""Model-independent physics for the optimal necessary-work labeler.

This module holds the *ground-truth* accountant for VibeSim's redundancy analysis:
given a model and a workload, what is the THEORETICAL MINIMUM compute (FLOPs) and
memory traffic (bytes) that the forward MUST do? A real run's *achieved* work
(the sim's logged ``slot_flops`` / ``slot_bytes``, summed with the physical GPU
multiplicity) is then measured against this minimum; ``achieved / minimum`` is the
redundancy factor.

The minimum is computed **only** from the model config + the workload — never from
the simulator's kernel tree. That independence is the whole point: an accountant
derived from the sim's own decomposition could never detect the sim doing redundant
work.

Everything model-specific lives in the ``attention/`` and ``ffn/`` specs; this file
holds only the shared arithmetic. The single common currency between a spec and this
module is :class:`MatmulGroup`. See ``README.md`` for the four pinned conventions
that define what does (and does not) count toward the minimum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .attention.base import AttentionSpec
    from .ffn.base import FFNSpec

# ---------------------------------------------------------------------------
# dtype -> bytes/element
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    "float32": 4, "fp32": 4, "f32": 4,
    "float16": 2, "fp16": 2, "f16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float8": 1, "fp8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1, "f8": 1,
    "int8": 1,
}


def dtype_bytes(dtype: str) -> float:
    """Bytes per element for a torch/HF dtype string (e.g. ``"bfloat16"`` -> 2)."""
    key = dtype.lower().removeprefix("torch.")
    if key not in _DTYPE_BYTES:
        raise KeyError(f"unknown dtype {dtype!r}; extend _DTYPE_BYTES in core.py")
    return _DTYPE_BYTES[key]


# ---------------------------------------------------------------------------
# Workload — modality-agnostic description of one forward pass
# ---------------------------------------------------------------------------


@dataclass
class AttnInteraction:
    """One attention interaction: a set of new queries attending to a set of keys.

    Modality-agnostic. ``mask`` selects the query/key pair count (causal triangle vs
    full rectangle); ``num_cached_key`` is how many of the attended keys already live
    in a cache/state and are therefore read from HBM (a cache-less full-attention
    interaction, e.g. a ViT layer, has ``num_cached_key == 0`` and moves no KV bytes).
    """

    num_query: int
    num_key: int
    num_cached_key: int
    mask: str  # "causal" | "full" | "cross"

    def pairs(self) -> float:
        """(query, key) pairs actually computed. Causal uses the exact triangle."""
        q, k = self.num_query, self.num_key
        if self.mask == "causal":
            # queries sit at the end of the sequence: query i attends to
            # (num_cached_key + i + 1) keys, summed over i in [0, q) -> q*(k - (q-1)/2).
            return q * (k - (q - 1) / 2.0)
        if self.mask in ("full", "cross"):
            return float(q * k)
        raise ValueError(f"unknown attention mask {self.mask!r}")


@dataclass
class Workload:
    """What a single forward pass processes. NOT a fixed causal-LM struct.

    The weight/FFN/param side depends only on ``matmul_tokens`` and
    ``head_positions`` (universal to every modality); all attention-specific geometry
    is confined to ``attn`` and is consumed only by an :class:`AttentionSpec`. New
    modalities are new constructors (see :meth:`causal_lm`), never a schema change.
    """

    matmul_tokens: int  # T: positions through every weight matmul + FFN
    head_positions: int  # T_out: positions through the output (lm_head) projection
    attn: list[AttnInteraction] = field(default_factory=list)

    @classmethod
    def causal_lm(
        cls,
        prefill: list[tuple[int, int]] | None = None,
        decode: list[int] | None = None,
        sampled: int | None = None,
    ) -> Workload:
        """Autoregressive-decoder workload.

        ``prefill`` = list of ``(append_len, prefix_len)`` chunks (new tokens, already
        cached tokens). ``decode`` = list of per-token ``kv_len``. ``sampled`` =
        positions through lm_head (default: one per sequence — the minimal, since only
        the last prefill token and each decode step are actually sampled).
        """
        prefill = prefill or []
        decode = decode or []
        attn: list[AttnInteraction] = []
        matmul_tokens = 0
        for append_len, prefix_len in prefill:
            attn.append(
                AttnInteraction(
                    num_query=append_len,
                    num_key=prefix_len + append_len,
                    num_cached_key=prefix_len,
                    mask="causal",
                )
            )
            matmul_tokens += append_len
        for kv_len in decode:
            attn.append(
                AttnInteraction(
                    num_query=1,
                    num_key=kv_len,
                    num_cached_key=kv_len - 1,
                    mask="causal",
                )
            )
            matmul_tokens += 1
        if sampled is None:
            sampled = len(prefill) + len(decode)
        return cls(matmul_tokens=matmul_tokens, head_positions=sampled, attn=attn)


# ---------------------------------------------------------------------------
# MatmulGroup — the common currency between a spec and this module
# ---------------------------------------------------------------------------

# Which FLOP bucket / param-breakdown key a group's ``bucket`` maps to.
_FLOPS_BUCKET = {
    "attn_proj": "attn_proj",
    "dense_ffn": "ffn",
    "expert": "ffn",
    "shared_expert": "ffn",
    "router": "router",
}
_PARAM_BUCKET = {
    "attn_proj": "attn",
    "dense_ffn": "ffn",
    "expert": "experts",
    "shared_expert": "shared",
    "router": "router",
}


@dataclass
class MatmulGroup:
    """One weight matmul (per layer): output dim ``n``, input dim ``k``.

    ``activated_mult`` = instances a single token flows through (``top_k`` for routed
    experts, 1 otherwise) — drives FLOPs and activated params. ``total_count`` = all
    instances that exist (``num_experts`` for experts, 1 otherwise) — drives total
    params. ``bucket`` names the FLOP/param category (see ``_FLOPS_BUCKET``).
    """

    name: str
    n: int
    k: int
    activated_mult: int = 1
    total_count: int = 1
    bucket: str = "dense_ffn"
    routed: bool = False  # weights are routed experts -> only the hit subset is loaded

    @property
    def activated_params(self) -> int:
        return self.activated_mult * self.n * self.k

    @property
    def total_params(self) -> int:
        return self.total_count * self.n * self.k

    def loaded_instances(self, matmul_tokens: int) -> float:
        """Distinct instances whose weights are read from HBM for this batch.

        A routed-expert group loads only the experts actually hit; under uniform
        routing the expected distinct count is balls-in-bins over
        ``matmul_tokens · activated_mult`` draws into ``total_count`` experts:
        ``E·(1 − (1 − 1/E)^draws)``. Non-routed groups load every instance.
        """
        if not self.routed or self.total_count <= 1:
            return float(self.total_count)
        draws = matmul_tokens * self.activated_mult
        experts = self.total_count
        return experts * (1.0 - (1.0 - 1.0 / experts) ** draws)


# ---------------------------------------------------------------------------
# WorkLabel — the result: minimum FLOPs, bytes, and parameter counts
# ---------------------------------------------------------------------------

_FLOPS_KEYS = ("attn_proj", "attn_internal", "ffn", "router", "lm_head")
_BYTES_KEYS = ("weights", "kv")
_PARAM_BREAKDOWN_KEYS = ("embedding", "attn", "ffn", "experts", "shared", "router", "lm_head")


@dataclass
class Segment:
    """One kernel-like unit of the forward with its own arithmetic intensity.

    ``flops`` / ``bytes`` are for ONE instance (one layer); ``count`` is how many
    identical instances run (``num_layers`` for a per-layer op, 1 for embedding /
    lm_head). ``bytes`` is the minimal HBM traffic the unit must move — weights read
    once, or the KV/state read for attention; intra-unit activations are fused away.
    A per-segment roofline (max of its own compute and memory time) summed over all
    segments is a tighter, more realistic lower bound than the fully-fused global one,
    because separate kernels run sequentially and cannot overlap each other's work.
    """

    name: str
    bucket: str  # a _FLOPS_KEYS entry, or "embedding" (zero-FLOP gather)
    byte_kind: str  # "weights" | "kv"
    flops: float
    bytes: float
    count: int

    @property
    def flops_total(self) -> float:
        return self.flops * self.count

    @property
    def bytes_total(self) -> float:
        return self.bytes * self.count

    def _instance_seconds(self, peak_tflops: float, bandwidth_gbps: float) -> tuple[float, float]:
        return self.flops / (peak_tflops * 1e12), self.bytes / (bandwidth_gbps * 1e9)

    def instance_bound(self, peak_tflops: float, bandwidth_gbps: float) -> str:
        compute_s, memory_s = self._instance_seconds(peak_tflops, bandwidth_gbps)
        return "compute" if compute_s >= memory_s else "memory"

    def time_ms(self, peak_tflops: float, bandwidth_gbps: float) -> float:
        """Minimal time for all ``count`` instances = count · max(compute, memory)."""
        compute_s, memory_s = self._instance_seconds(peak_tflops, bandwidth_gbps)
        return max(compute_s, memory_s) * self.count * 1e3


@dataclass
class WorkLabel:
    """Minimum necessary work for one forward pass over a :class:`Workload`."""

    flops: dict[str, float]  # keys in _FLOPS_KEYS
    bytes: dict[str, float]  # keys in _BYTES_KEYS
    # {"total": int, "activated": {"layers", "with_embed_head"}, "breakdown": {...}}
    params: dict[str, object]
    segments: list[Segment] = field(default_factory=list)

    @property
    def flops_total(self) -> float:
        return sum(self.flops.values())

    @property
    def bytes_total(self) -> float:
        return sum(self.bytes.values())

    def tflops(self, seconds: float) -> float:
        """Achieved TFLOP/s if this minimum work ran in ``seconds``."""
        return self.flops_total / seconds / 1e12

    def gbps(self, seconds: float) -> float:
        """Achieved GB/s if this minimum work ran in ``seconds``."""
        return self.bytes_total / seconds / 1e9

    def roofline_ms(
        self, gpu: str, dtype: str = "bf16", num_gpus: int = 1, spec_path: str | Path | None = None
    ) -> tuple[float, float, str]:
        """Compute-bound and memory-bound time floors (ms) and which one binds."""
        peak_tflops = gpu_peak_tflops(gpu, dtype, spec_path)
        bandwidth_gbps = gpu_mem_bandwidth_gbps(gpu, spec_path)
        compute_ms = self.flops_total / (peak_tflops * 1e12 * num_gpus) * 1e3
        memory_ms = self.bytes_total / (bandwidth_gbps * 1e9 * num_gpus) * 1e3
        bound = "compute" if compute_ms >= memory_ms else "memory"
        return compute_ms, memory_ms, bound

    def segmented_lower_bound_ms(
        self, gpu: str, dtype: str = "bf16", num_gpus: int = 1, spec_path: str | Path | None = None
    ) -> float:
        """Realistic lower bound: Σ per-segment max(compute, memory).

        Segments run sequentially and cannot overlap each other, so this is tighter
        (larger) than the fully-fused :meth:`roofline_ms` global floor — but still a
        valid lower bound on real time (it counts only necessary work per segment).
        """
        peak = gpu_peak_tflops(gpu, dtype, spec_path) * num_gpus
        bandwidth = gpu_mem_bandwidth_gbps(gpu, spec_path) * num_gpus
        return sum(seg.time_ms(peak, bandwidth) for seg in self.segments)

    def segment_rows(
        self, gpu: str, dtype: str = "bf16", num_gpus: int = 1, spec_path: str | Path | None = None
    ) -> list[dict]:
        """Per-segment {name, count, flops, bytes, bound, ms} for display/inspection."""
        peak = gpu_peak_tflops(gpu, dtype, spec_path) * num_gpus
        bandwidth = gpu_mem_bandwidth_gbps(gpu, spec_path) * num_gpus
        return [
            {
                "name": seg.name,
                "count": seg.count,
                "flops": seg.flops_total,
                "bytes": seg.bytes_total,
                "bound": seg.instance_bound(peak, bandwidth),
                "ms": seg.time_ms(peak, bandwidth),
            }
            for seg in self.segments
        ]

    def work_efficiency(self, achieved_flops: float) -> float:
        """F_min / achieved ∈ (0, 1]. Below 1 means the run did redundant compute."""
        return self.flops_total / achieved_flops


def empty_flops() -> dict[str, float]:
    return {key: 0.0 for key in _FLOPS_KEYS}


def empty_bytes() -> dict[str, float]:
    return {key: 0.0 for key in _BYTES_KEYS}


def empty_breakdown() -> dict[str, int]:
    return {key: 0 for key in _PARAM_BREAKDOWN_KEYS}


# ---------------------------------------------------------------------------
# Model — generic composition of one AttentionSpec + one FFNSpec
# ---------------------------------------------------------------------------


@dataclass
class LayerStack:
    """A run of identical decoder layers: one (attention, ffn) archetype repeated
    ``count`` times.

    A uniform model is a single stack; a hybrid model (e.g. Qwen3.5/3.6's interleaved
    linear + full attention) has one stack per archetype. ``tag`` prefixes this stack's
    segment names so the archetypes stay distinct in the per-op table — it is empty for
    uniform models, keeping their segment names bare (``qkv``, ``attn``, …).
    """

    attn: AttentionSpec
    ffn: FFNSpec
    count: int
    tag: str = ""


@dataclass
class Model:
    """A whole model: a list of layer archetypes + shared embedding/head.

    Per-model files under ``models/`` build one of these from a raw ``config.json``;
    the fold in :meth:`label` is model-independent. Use :meth:`uniform` for the common
    single-archetype case (every layer the same); pass ``layers`` directly for a hybrid
    stack.
    """

    name: str  # the config's architectures[0]
    hidden: int
    vocab: int
    weight_dtype_bytes: float
    tie_word_embeddings: bool
    layers: list[LayerStack]

    @property
    def num_layers(self) -> int:
        return sum(stack.count for stack in self.layers)

    @classmethod
    def uniform(
        cls,
        name: str,
        num_layers: int,
        hidden: int,
        vocab: int,
        attn: AttentionSpec,
        ffn: FFNSpec,
        weight_dtype_bytes: float,
        tie_word_embeddings: bool,
    ) -> Model:
        """A model whose every layer is the same (attn, ffn) archetype."""
        return cls(
            name=name,
            hidden=hidden,
            vocab=vocab,
            weight_dtype_bytes=weight_dtype_bytes,
            tie_word_embeddings=tie_word_embeddings,
            layers=[LayerStack(attn=attn, ffn=ffn, count=num_layers)],
        )

    def label(self, wl: Workload) -> WorkLabel:
        tokens = wl.matmul_tokens
        segments: list[Segment] = []
        breakdown = empty_breakdown()
        activated_params = 0
        total_params = 0

        # --- per-archetype per-layer weight matmuls + fused attention ---
        # Each layer stack contributes its own segments, repeated ``stack.count`` times.
        # A weight matmul computes 2·T·N·K FLOPs and reads its matrix from HBM once (only
        # the hit expert subset for a routed group). The stack ``tag`` prefixes segment
        # names so a hybrid model's archetypes stay distinct in the per-op table.
        for stack in self.layers:
            prefix = f"{stack.tag}." if stack.tag else ""
            for group in (*stack.attn.matmul_groups(), *stack.ffn.matmul_groups()):
                loaded = group.loaded_instances(tokens)
                segments.append(
                    Segment(
                        name=f"{prefix}{group.name}",
                        bucket=_FLOPS_BUCKET[group.bucket],
                        byte_kind="weights",
                        flops=2.0 * tokens * group.activated_mult * group.n * group.k,
                        bytes=loaded * group.n * group.k * self.weight_dtype_bytes,
                        count=stack.count,
                    )
                )
                activated_params += group.activated_params * stack.count
                total_params += group.total_params * stack.count
                breakdown[_PARAM_BUCKET[group.bucket]] += group.total_params * stack.count

            # attention: one fused kernel per layer — reads the KV cache / recurrent
            # state and computes scores+context (no weights of its own).
            segments.append(
                Segment(
                    name=f"{prefix}attn",
                    bucket="attn_internal",
                    byte_kind="kv",
                    flops=stack.attn.internal_flops(wl),
                    bytes=stack.attn.kv_bytes(wl),
                    count=stack.count,
                )
            )

        # --- embedding gather (whole iteration): reads the needed rows of the table. ---
        # The embedding matrix is a weight, so its read is capped at ONE pass of the
        # table (F_min's read-once convention): a batch with more tokens than vocab
        # entries cannot force more than one full table pass, and a smaller batch
        # touches at most ``tokens`` distinct rows. Without the cap a giant aggregate
        # batch (T ≫ vocab) would over-count the embedding read ~T/vocab times.
        embedding_params = self.vocab * self.hidden
        breakdown["embedding"] += embedding_params
        total_params += embedding_params
        segments.append(
            Segment(
                name="embedding",
                bucket="embedding",
                byte_kind="weights",
                flops=0.0,
                bytes=min(tokens, self.vocab) * self.hidden * self.weight_dtype_bytes,
                count=1,
            )
        )

        # --- lm_head projection (whole iteration): reads its weight matrix once. ---
        lm_head_params = 0 if self.tie_word_embeddings else self.vocab * self.hidden
        breakdown["lm_head"] += lm_head_params
        total_params += lm_head_params
        segments.append(
            Segment(
                name="lm_head",
                bucket="lm_head",
                byte_kind="weights",
                flops=2.0 * wl.head_positions * self.hidden * self.vocab,
                bytes=self.hidden * self.vocab * self.weight_dtype_bytes,
                count=1,
            )
        )

        # --- roll segments up into the aggregate FLOP / byte buckets ---
        flops = empty_flops()
        byte_sum = empty_bytes()
        for segment in segments:
            if segment.bucket in flops:
                flops[segment.bucket] += segment.flops_total
            byte_sum[segment.byte_kind] += segment.bytes_total

        # "activated" has more than one defensible convention; expose all of them
        # rather than picking one. "layers" is the per-token transformer compute
        # (attn + ffn/experts + router); "with_embed_head" additionally counts the
        # embedding + lm_head matrices — the "A-XXB" figure most model cards quote.
        activated = {
            "layers": activated_params,
            "with_embed_head": activated_params + embedding_params + lm_head_params,
        }
        params: dict[str, object] = {
            "total": total_params,
            "activated": activated,
            "breakdown": breakdown,
        }
        return WorkLabel(flops=flops, bytes=byte_sum, params=params, segments=segments)


# ---------------------------------------------------------------------------
# GPU roofline peaks — read from gpu/spec.json (name ∪ aliases match)
# ---------------------------------------------------------------------------


def _spec_path(spec_path: str | Path | None) -> Path:
    if spec_path is not None:
        return Path(spec_path)
    # core.py is model/work/core.py; repo root is parents[2].
    return Path(__file__).resolve().parents[2] / "gpu" / "spec.json"


@lru_cache(maxsize=8)
def _load_gpus(spec_path_str: str) -> tuple[dict, ...]:
    with open(spec_path_str) as handle:
        return tuple(json.load(handle)["gpus"])


def _find_gpu(name: str, spec_path: str | Path | None) -> dict:
    gpus = _load_gpus(str(_spec_path(spec_path)))
    wanted = name.lower()
    for entry in gpus:
        names = [entry["name"], *entry.get("aliases", [])]
        if any(wanted == candidate.lower() for candidate in names):
            return entry
    raise KeyError(
        f"GPU {name!r} not found in gpu/spec.json (matched against name ∪ aliases)"
    )


_TFLOPS_FIELD = {
    "bf16": "bf16_tflops",
    "fp16": "fp16_tflops",
    "fp8": "fp8_tflops",
    "fp32": "fp32_tflops",
}


def gpu_peak_tflops(name: str, dtype: str, spec_path: str | Path | None = None) -> float:
    """Dense peak TFLOP/s for ``dtype`` on ``name`` from gpu/spec.json (DENSE, not sparse)."""
    key = dtype.lower().removeprefix("torch.")
    key = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(key, key)
    field_name = _TFLOPS_FIELD.get(key)
    if field_name is None:
        raise KeyError(f"no peak-TFLOPS field for dtype {dtype!r}")
    entry = _find_gpu(name, spec_path)
    value = entry.get(field_name)
    if value is None:
        raise KeyError(f"{entry['name']} has no {field_name} in gpu/spec.json")
    return float(value)


def gpu_mem_bandwidth_gbps(name: str, spec_path: str | Path | None = None) -> float:
    """HBM bandwidth (GB/s) for ``name`` from gpu/spec.json."""
    return float(_find_gpu(name, spec_path)["mem_bandwidth_gbps"])

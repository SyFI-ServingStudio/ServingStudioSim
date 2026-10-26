"""Stock vLLM Neuron capture contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from alignment.load_generator.config import LoadGeneratorConfig
from profiling.exec.neuron import TRAINIUM2_LNC2, NeuronContainer
from profiling.gpu_catalog import same_canonical_sku
from profiling.runners.neuron.stock_contract import (
    BLOCK_SIZE,
    CONTEXTS,
    KV_BLOCKS,
    TP_SIZE,
    largest_decode_bucket,
    resident_capacity,
)

# Contexts whose prompts (C-8 tokens) fit one 9973-token museum pool segment and
# whose stock forward graphs the profiler measures with the same fixed KV pool.
STOCK_CONTEXTS = tuple(c for c in CONTEXTS if c >= 512)


@dataclass(frozen=True)
class VllmNeuronServerConfig:
    model_path: str
    cache_path: str
    image: str
    docker_host: str
    req_frontend_binary: str
    accepted_forward_path: str | None = None
    docker_command: tuple[str, ...] = ("docker",)
    python_executable: str = "/opt/conda/bin/python"
    host: str = "127.0.0.1"
    port: int = 63030
    startup_timeout: float = 1800
    neuron_device: int = 0
    tp_size: int = 4
    dp_size: int = 1
    logical_nc_config: int = 2
    max_model_len: int = 512
    max_num_seqs: int = 16
    token_buckets: tuple[int, ...] = (1, 16)
    kv_blocks: int = KV_BLOCKS
    block_size: int = BLOCK_SIZE
    dtype: str = "bfloat16"
    #: Opt-in API-server hand-off stamp (public request log + logging config);
    #: off keeps the server argv/env byte-identical to unstamped captures.
    handoff_stamp: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.docker_command, str) or isinstance(self.token_buckets, str):
            raise ValueError("Docker argv and token buckets require sequences, not strings")
        object.__setattr__(self, "docker_command", tuple(self.docker_command))
        object.__setattr__(self, "token_buckets", tuple(self.token_buckets))

    def container(self) -> NeuronContainer:
        """The server's Docker transport and image, validated as the profiler's."""
        return NeuronContainer(
            self.docker_command, self.docker_host, self.image, self.python_executable
        )


@dataclass(frozen=True)
class VllmNeuronProfileConfig:
    name: str
    log_dir: str
    gpu: str
    server: VllmNeuronServerConfig
    workload: LoadGeneratorConfig
    profile_kind: str = "neuron"
    engine: str = field(default="vllm_neuron", init=False)

    def validate(self) -> None:
        s = self.server
        if self.profile_kind not in {"neuron", "workload_metrics"}:
            raise ValueError("vllm_neuron requires neuron or workload_metrics")
        if not same_canonical_sku(self.gpu, TRAINIUM2_LNC2):
            raise ValueError("vllm_neuron requires Trainium2 LNC2")
        expected = (TP_SIZE, 1, 2, KV_BLOCKS, BLOCK_SIZE)
        actual = (s.tp_size, s.dp_size, s.logical_nc_config, s.kv_blocks, s.block_size)
        if (
            any(type(v) is not int for v in (*actual, s.max_model_len, s.max_num_seqs))
            or actual != expected
            or s.max_model_len not in STOCK_CONTEXTS
        ):
            raise ValueError(
                f"stock capture requires TP{TP_SIZE}/DP1/LNC2/pool{KV_BLOCKS}/page{BLOCK_SIZE} "
                f"and C in {STOCK_CONTEXTS}"
            )
        buckets = tuple(s.token_buckets)
        if (
            not buckets
            or any(type(v) is not int or v < 1 or v & (v - 1) for v in buckets)
            or list(buckets) != sorted(set(buckets))
            or buckets[-1] > largest_decode_bucket(s.max_model_len)
        ):
            raise ValueError(
                "stock token_buckets must be ascending unique powers of two at most "
                f"{largest_decode_bucket(s.max_model_len)} for C{s.max_model_len}"
            )
        if s.max_num_seqs != buckets[-1]:
            raise ValueError("stock max_num_seqs must equal the largest decode bucket")
        if s.dtype != "bfloat16" or s.host != "127.0.0.1":
            raise ValueError("stock capture requires BF16 and loopback HTTP")
        if type(s.port) is not int or not 1 <= s.port <= 65535:
            raise ValueError("invalid stock HTTP port")
        if type(s.neuron_device) is not int or s.neuron_device < 0:
            raise ValueError("invalid Neuron device")
        if type(s.handoff_stamp) is not bool:
            raise ValueError("server.handoff_stamp must be a boolean")
        if (
            isinstance(s.startup_timeout, bool)
            or not isinstance(s.startup_timeout, (int, float))
            or not math.isfinite(s.startup_timeout)
            or s.startup_timeout <= 0
        ):
            raise ValueError("invalid stock startup timeout")
        s.container()  # Immutable image, local socket URI and image-owned interpreter.
        for name in ("model_path", "cache_path", "req_frontend_binary"):
            if not Path(getattr(s, name)).is_absolute():
                raise ValueError(f"server.{name} must be absolute")
        if s.accepted_forward_path is None or not Path(s.accepted_forward_path).is_absolute():
            raise ValueError(
                "server.accepted_forward_path must declare an absolute accuracy artifact"
            )
        w = self.workload
        if w.frontend.type != "independent" or w.backend.type != "openai":
            raise ValueError("stock alignment requires independent OpenAI requests")
        resident = min(buckets[-1], resident_capacity(s.max_model_len))
        if (
            type(w.max_concurrency) is not int
            or type(w.max_items) is not int
            or w.max_items != w.max_concurrency
            or not 1 <= w.max_concurrency <= resident
            or type(w.max_model_len) is not int
            or w.max_model_len != s.max_model_len
        ):
            raise ValueError(
                f"stock replay requires 1..{resident} simultaneous requests "
                f"(max_items == max_concurrency) and context{s.max_model_len}"
            )
        if w.extra_args or w.warmup or w.context_limit_skip_enabled:
            raise ValueError("initial stock replay forbids extra args, warmup and skipped requests")
        if (
            type(w.token_pool_limit) is not int
            or w.token_pool_limit != 8 * 9973
            or Path(w.tokenizer) != Path(s.model_path)
        ):
            raise ValueError(
                "stock replay requires the local model tokenizer and exact museum pool"
            )

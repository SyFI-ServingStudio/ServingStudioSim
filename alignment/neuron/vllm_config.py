"""Stock vLLM Neuron capture contract, separate from the NxDI TP1 adapter."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from alignment.load_generator.config import LoadGeneratorConfig
from profiling.exec.env import ContainerProfileEnv


@dataclass(frozen=True)
class VllmNeuronServerConfig:
    model_path: str
    cache_path: str
    image: str
    docker_host: str
    req_frontend_binary: str
    accepted_forward_path: str | None = None
    docker_command: tuple[str, ...] = ("sudo", "-n", "docker")
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
    kv_blocks: int = 6782
    block_size: int = 32
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if isinstance(self.docker_command, str) or isinstance(self.token_buckets, str):
            raise ValueError("Docker argv and token buckets require sequences, not strings")
        object.__setattr__(self, "docker_command", tuple(self.docker_command))
        object.__setattr__(self, "token_buckets", tuple(self.token_buckets))

    def container_env(self) -> ContainerProfileEnv:
        if isinstance(self.docker_command, str):
            raise ValueError("docker_command must be an argv list, not a shell string")
        return ContainerProfileEnv(
            "vllm_neuron_alignment",
            self.image,
            tuple(self.docker_command),
            self.docker_host,
            Path(self.python_executable),
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
        if self.gpu not in {"AWS Trainium2 LNC2", "Trainium2-LNC2"}:
            raise ValueError("vllm_neuron requires Trainium2 LNC2")
        expected = (4, 1, 2, 512, 16, 6782, 32)
        actual = (
            s.tp_size,
            s.dp_size,
            s.logical_nc_config,
            s.max_model_len,
            s.max_num_seqs,
            s.kv_blocks,
            s.block_size,
        )
        if any(type(v) is not int for v in actual) or actual != expected:
            raise ValueError("stock capture requires TP4/DP1/LNC2/C512/maxseq16/pool6782/page32")
        if tuple(s.token_buckets) != (1, 16) or any(type(v) is not int for v in s.token_buckets):
            raise ValueError("stock capture requires token_buckets [1,16]")
        if s.dtype != "bfloat16" or s.host != "127.0.0.1":
            raise ValueError("stock capture requires BF16 and loopback HTTP")
        if type(s.port) is not int or not 1 <= s.port <= 65535:
            raise ValueError("invalid stock HTTP port")
        if type(s.neuron_device) is not int or s.neuron_device < 0:
            raise ValueError("invalid Neuron device")
        if (
            isinstance(s.startup_timeout, bool)
            or not isinstance(s.startup_timeout, (int, float))
            or not math.isfinite(s.startup_timeout)
            or s.startup_timeout <= 0
        ):
            raise ValueError("invalid stock startup timeout")
        if not re.fullmatch(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}", s.image):
            raise ValueError("stock alignment requires an immutable image digest")
        s.container_env()  # Validate local socket URI and image-owned interpreter syntax.
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
        if (
            type(w.max_concurrency) is not int
            or w.max_concurrency != 16
            or type(w.max_items) is not int
            or w.max_items != 16
            or type(w.max_model_len) is not int
            or w.max_model_len != 512
        ):
            raise ValueError("initial stock replay requires 16 requests/concurrency16/context512")
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

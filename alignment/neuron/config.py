"""The explicit batch-one NxDI runtime contract, separate from CUDA servers."""

import math
from dataclasses import dataclass, field

from alignment.load_generator.config import LoadGeneratorConfig


@dataclass
class NxdiServerConfig:
    model_path: str
    compiled_path: str
    host: str = "127.0.0.1"
    port: int = 63030
    startup_timeout: float = 900.0
    tp_size: int = 1
    dp_size: int = 1
    batch_size: int = 1
    context_bucket: int = 128
    kv_bucket: int = 512
    logical_nc_config: int = 2
    neuron_device: int = 0


@dataclass
class NxdiProfileConfig:
    name: str
    log_dir: str
    gpu: str
    server: NxdiServerConfig
    workload: LoadGeneratorConfig
    fork_python: str
    profile_kind: str = "neuron"
    engine: str = field(default="nxdi", init=False)

    def validate(self) -> None:
        if self.profile_kind not in {"neuron", "workload_metrics"}:
            raise ValueError("NxDI profile_kind must be neuron or workload_metrics")
        if self.gpu not in {"AWS Trainium2 LNC2", "Trainium2-LNC2"}:
            raise ValueError("NxDI alignment requires AWS Trainium2 LNC2")
        geometry = (
            self.server.tp_size,
            self.server.dp_size,
            self.server.batch_size,
            self.server.logical_nc_config,
            self.server.context_bucket,
            self.server.kv_bucket,
        )
        if any(type(value) is not int for value in geometry) or geometry != (1, 1, 1, 2, 128, 512):
            raise ValueError("NxDI alignment requires TP1/DP1/batch1/LNC2/CTE128/TKG512")
        if (
            self.server.host != "127.0.0.1"
            or type(self.server.port) is not int
            or not 1 <= self.server.port <= 65535
        ):
            raise ValueError("NxDI alignment requires a valid loopback port")
        if self.workload.frontend.type != "independent" or self.workload.backend.type != "openai":
            raise ValueError("NxDI alignment requires independent OpenAI token requests")
        if type(self.workload.max_concurrency) is not int or self.workload.max_concurrency != 1:
            raise ValueError("NxDI alignment requires explicit workload.max_concurrency: 1")
        timeout = self.server.startup_timeout
        if (
            type(self.server.neuron_device) is not int
            or self.server.neuron_device < 0
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("invalid NxDI device or startup timeout")

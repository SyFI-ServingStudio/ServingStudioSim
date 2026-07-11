"""Typed TraceLab configuration embedded in a YAML/JSON profiling config.

Trace frontends are a tagged union because session and independent-request
traces already have distinct runtime behavior. Request construction currently
has one concrete synthetic-text path, so its inputs stay direct workload fields
until a second implementation introduces real runtime dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class SessionFrontendConfig:
    type: ClassVar[str] = "session"
    path: str


@dataclass
class VibeSimFrontendConfig:
    type: ClassVar[str] = "vibesim"
    path: str


FrontendConfig = SessionFrontendConfig | VibeSimFrontendConfig
_FRONTEND_CONFIGS = {
    SessionFrontendConfig.type: SessionFrontendConfig,
    VibeSimFrontendConfig.type: VibeSimFrontendConfig,
}


def _load_tagged_config(value: dict, registry: dict, field_name: str):
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    raw = dict(value)
    tag = raw.pop("type", None)
    config_type = registry.get(tag)
    if config_type is None:
        raise ValueError(
            f"unsupported {field_name}.type {tag!r}; available: {sorted(registry)}"
        )
    try:
        return config_type(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid {field_name} config: {exc}") from exc


@dataclass
class LoadGeneratorConfig:
    """TraceLab frontend, synthetic-text inputs, and replay policy."""

    frontend: FrontendConfig
    text_file: str
    tokenizer: str
    max_items: int | None = None
    rate: float | None = None
    token_pool_limit: int | None = None
    stream_idle_timeout_secs: int = 600
    max_concurrency: int | None = None
    max_model_len: int | None = None
    fail_on_context_overflow: bool = False
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: dict) -> LoadGeneratorConfig:
        if not isinstance(value, dict):
            raise ValueError("profiling workload must be a mapping")
        raw = dict(value)
        try:
            frontend_raw = raw.pop("frontend")
        except KeyError as exc:
            raise ValueError(f"profiling workload missing {exc.args[0]!r}") from exc
        frontend = _load_tagged_config(frontend_raw, _FRONTEND_CONFIGS, "frontend")
        try:
            return cls(frontend=frontend, **raw)
        except TypeError as exc:
            raise ValueError(f"invalid profiling workload: {exc}") from exc

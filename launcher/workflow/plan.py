"""Typed workflow topology; stage implementations live at their call sites."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StageKind(StrEnum):
    BUILD_SIMULATOR = "build_simulator"
    BUILD_ANALYZER = "build_analyzer"
    ENSURE_CACHE = "ensure_cache"
    SIMULATE = "simulate"
    VALIDATE_RAW_ARTIFACTS = "validate_raw_artifacts"
    ANALYZE_COMPUTE = "analyze_compute"
    RENDER = "render"
    TRACE = "trace"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class StageNode:
    kind: StageKind
    dependencies: tuple[StageKind, ...] = ()
    required: bool = True


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    nodes: tuple[StageNode, ...]

    def __post_init__(self) -> None:
        seen: set[StageKind] = set()
        for node in self.nodes:
            missing = set(node.dependencies) - seen
            if missing:
                names = ", ".join(sorted(stage.value for stage in missing))
                raise ValueError(f"stage {node.kind.value} has unresolved dependencies: {names}")
            if node.kind in seen:
                raise ValueError(f"duplicate workflow stage: {node.kind.value}")
            seen.add(node.kind)

    def contains(self, kind: StageKind) -> bool:
        return any(node.kind == kind for node in self.nodes)


def simulation_workflow(*, analyze: bool) -> WorkflowPlan:
    """Compile the per-run graph; ``--no-analyze`` removes optional nodes."""

    nodes = [
        StageNode(StageKind.ENSURE_CACHE),
        StageNode(StageKind.SIMULATE, (StageKind.ENSURE_CACHE,)),
        StageNode(
            StageKind.VALIDATE_RAW_ARTIFACTS,
            (StageKind.SIMULATE,),
        ),
    ]
    final_dependencies = (StageKind.VALIDATE_RAW_ARTIFACTS,)
    if analyze:
        nodes.extend(
            [
                StageNode(
                    StageKind.ANALYZE_COMPUTE,
                    (StageKind.VALIDATE_RAW_ARTIFACTS,),
                    required=False,
                ),
                StageNode(
                    StageKind.RENDER,
                    (StageKind.ANALYZE_COMPUTE,),
                    required=False,
                ),
                StageNode(
                    StageKind.TRACE,
                    (StageKind.ANALYZE_COMPUTE,),
                    required=False,
                ),
            ]
        )
        final_dependencies = (StageKind.RENDER, StageKind.TRACE)
    nodes.append(StageNode(StageKind.FINALIZE, final_dependencies))
    return WorkflowPlan(tuple(nodes))

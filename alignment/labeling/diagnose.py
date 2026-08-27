"""Read-only views that turn "coverage is 70%" into "these kernels, this much".

`coverage` answers where the unmapped time is, on BOTH sides at once. That
pairing is the point: the analyzer already reports unmapped measured kernels and
unmapped simulated slots, but it reports measured kernels per folded row, so one
kernel name arrives split across dozens of rows and never surfaces in a
by-magnitude read. Aggregating by name and printing the two lists side by side
is what makes an unlabeled subsystem obvious — its measured kernels and its
simulated slots are both sitting at the top, describing the same work.

`check` is the mechanical half of review. Each rule here corresponds to a defect
that reached a published number at least once:

  * one kernel name carrying two operations — a CUTLASS tile reused by two
    different layers (`q_absorb` in a decoder layer, `lm_head` in postprocess)
    was labeled by name alone, which charged the lm_head's time to q_absorb and
    produced a "the model under-predicts 2x" finding that did not exist;
  * one operation carrying two different label bodies — the launcher rejects
    this, but only says which operation, not which two labels disagree;
  * a label with no `cross_rank` — the analyzer defaults it to independent and
    says nothing, so the omission survives all the way to a schema error in the
    UI contract.
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .inventory import KernelPosition, walk_kernels

TEMPLATE_ARGUMENTS = re.compile(r"<.*")


def short_name(name: str) -> str:
    """Kernel name without template arguments, argument list, or namespaces."""
    stripped = TEMPLATE_ARGUMENTS.sub("", name).replace("void ", "")
    for namespace in (
        "at::native::",
        "tensorrt_llm::kernels::cutlass_kernels::",
        "tensorrt_llm::kernels::",
        "flashinfer::trtllm_alltoall::",
        "deep_gemm::",
        "flashinfer::",
        "vllm::",
    ):
        stripped = stripped.replace(namespace, "")
    return stripped.split("(")[0]


@dataclass(frozen=True)
class UnmappedKernel:
    name: str
    total_ms: float
    rows: int
    calls: int


@dataclass(frozen=True)
class Coverage:
    iteration_count: int
    measured_fraction: float
    simulated_fraction: float
    measured_total_ms: float
    measured_mapped_ms: float
    kernels: list[UnmappedKernel]
    slots: list[tuple[str, float]]

    @property
    def measured_unmapped_ms(self) -> float:
        return self.measured_total_ms - self.measured_mapped_ms


def read_coverage(report_path: Path) -> Coverage:
    """Aggregate the analyzer's per-row unmapped lists into a by-name view."""
    report = json.loads(Path(report_path).read_text())
    mapping = report["mapping"]
    totals = mapping["coverage"]

    by_name: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0, 0])
    for row in mapping["unmapped_measured_kernels"]:
        entry = by_name[row["name"]]
        entry[0] += row["total_ms"]
        entry[1] += 1
        entry[2] += row.get("calls", 0)

    kernels = [
        UnmappedKernel(name=name, total_ms=total, rows=int(rows), calls=int(calls))
        for name, (total, rows, calls) in by_name.items()
    ]
    kernels.sort(key=lambda kernel: -kernel.total_ms)

    slots = sorted(
        ((slot["slot"], slot["total_ms"]) for slot in mapping["unmapped_simulated_slots"]),
        key=lambda pair: -pair[1],
    )
    return Coverage(
        iteration_count=int(report["meta"]["iterations"]),
        measured_fraction=totals["measured_duration_fraction"],
        simulated_fraction=totals["simulated_workload_fraction"],
        measured_total_ms=totals["measured_total_kernel_ms"],
        measured_mapped_ms=totals["measured_mapped_ms"],
        kernels=kernels,
        slots=slots,
    )


def format_coverage(coverage: Coverage, limit: int = 20) -> str:
    measured_unmapped_average_ms = coverage.measured_unmapped_ms / coverage.iteration_count
    measured_total_average_ms = coverage.measured_total_ms / coverage.iteration_count
    lines = [
        f"measured coverage {coverage.measured_fraction * 100:6.2f}%"
        f"   simulated coverage {coverage.simulated_fraction * 100:6.2f}%",
        f"unmapped measured {coverage.measured_unmapped_ms:.3f} /"
        f" {coverage.measured_total_ms:.3f} ms total across"
        f" {coverage.iteration_count} iterations",
        f"per-iteration average {measured_unmapped_average_ms:.3f} /"
        f" {measured_total_average_ms:.3f} ms",
        "",
        f"unmapped measured kernels ({len(coverage.kernels)} names)",
        f"{'total ms':>9} {'rows':>5} {'calls':>8}  name",
    ]
    for kernel in coverage.kernels[:limit]:
        lines.append(
            f"{kernel.total_ms:9.1f} {kernel.rows:5d} {kernel.calls:8d}"
            f"  {short_name(kernel.name)[:70]}"
        )
    if len(coverage.kernels) > limit:
        lines.append(f"{'':9} … {len(coverage.kernels) - limit} more names")

    lines += [
        "",
        f"unmapped simulated slots ({len(coverage.slots)})",
        f"{'total ms':>9}  slot",
    ]
    for slot, total_ms in coverage.slots[:limit]:
        lines.append(f"{total_ms:9.1f}  {slot}")
    if len(coverage.slots) > limit:
        lines.append(f"{'':9} … {len(coverage.slots) - limit} more slots")
    return "\n".join(lines)


@dataclass(frozen=True)
class Finding:
    severity: str
    """`error` blocks a run (the launcher would reject it); `warning` needs a human."""
    summary: str
    detail: str


def _label_body(label: dict) -> tuple:
    """The parts of a label that must agree across every kernel of an operation."""
    return (
        label.get("type"),
        label.get("role"),
        tuple(label.get("simulated_slots") or ()),
        label.get("cross_rank"),
    )


def _where(
    positions: list[KernelPosition],
    phase: str,
    stream_role: str,
    after: str | None,
    after_name: str | None,
    before: str | None,
    before_name: str | None,
) -> str:
    """A coordinate carrying one evidence key, plus how many share it."""
    matching = [
        position
        for position in positions
        if position.phase == phase
        and position.stream_role == stream_role
        and position.previous_operation == after
        and (None if position.previous_name is None else short_name(position.previous_name))
        == after_name
        and position.next_operation == before
        and (None if position.next_name is None else short_name(position.next_name)) == before_name
    ]
    if not matching:
        return "?"
    suffix = f" (+{len(matching) - 1} more)" if len(matching) > 1 else ""
    return f"{matching[0].coordinate}{suffix}"


def check(document: dict) -> list[Finding]:
    findings: list[Finding] = []

    operations_by_name: dict[str, dict[str, list[KernelPosition]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    bodies_by_operation: dict[str, dict[tuple, list[KernelPosition]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    missing_cross_rank: list[KernelPosition] = []

    for position in walk_kernels(document):
        label = position.label
        # A kernel with no label at all is simply not labeled yet — that is what
        # a fresh inventory out of `alignment parse` looks like, and coverage is
        # where it shows up. A label that states a status but no cross_rank is
        # the defect: the analyzer defaults it and says nothing.
        if label and "cross_rank" not in label:
            missing_cross_rank.append(position)
        operation = position.operation
        if operation is None:
            continue
        operations_by_name[position.name][operation].append(position)
        bodies_by_operation[operation][_label_body(label)].append(position)

    for name, operations in sorted(operations_by_name.items()):
        # A name serving several operations is normal and not by itself a
        # defect — a quant kernel runs before every projection. What matters is
        # whether the recorded evidence tells the roles apart. Two operations
        # sharing one (phase, predecessor) key means the file states a
        # distinction it does not support, so one of the two is charged time
        # that belongs to the other.
        evidence_of: dict[
            str, set[tuple[str, str, str | None, str | None, str | None, str | None]]
        ] = {
            operation: {
                (
                    position.phase,
                    position.stream_role,
                    position.previous_operation,
                    None if position.previous_name is None else short_name(position.previous_name),
                    position.next_operation,
                    None if position.next_name is None else short_name(position.next_name),
                )
                for position in positions
            }
            for operation, positions in operations.items()
        }
        collisions = [
            (left, right)
            for index, left in enumerate(sorted(evidence_of))
            for right in sorted(evidence_of)[index + 1 :]
            if evidence_of[left] & evidence_of[right]
        ]
        if collisions:
            lines = []
            for left, right in collisions:
                shared = sorted(
                    evidence_of[left] & evidence_of[right],
                    key=lambda key: tuple(part or "" for part in key),
                )
                for phase, stream_role, after, after_name, before, before_name in shared:
                    left_where = _where(
                        operations[left],
                        phase,
                        stream_role,
                        after,
                        after_name,
                        before,
                        before_name,
                    )
                    right_where = _where(
                        operations[right],
                        phase,
                        stream_role,
                        after,
                        after_name,
                        before,
                        before_name,
                    )
                    lines.append(
                        f"  (phase={phase}, stream_role={stream_role}, after={after}, "
                        f"after_name={after_name}, before={before}, before_name={before_name})"
                        f"\n    {left} at {left_where}"
                        f"\n    {right} at {right_where}"
                    )
            findings.append(
                Finding(
                    severity="warning",
                    summary=(
                        f"kernel name has {len(collisions)} role pair(s) separated only by "
                        f"position: {short_name(name)}"
                    ),
                    detail=(
                        "Neither the phase, stream role, nor either neighbour separates these, "
                        "so the only "
                        "thing behind the distinction is which folded segment the kernel sits "
                        "in — legitimate evidence (a full-index layer runs an indexer chain "
                        "that an index-share layer does not) but not something a rule can "
                        "state. Read both positions and confirm.\n" + "\n".join(lines)
                    ),
                )
            )

        # The opposite failure, and the one that actually reached a number: a
        # name mapped to a single operation while running in more than one
        # phase. `nvjet_tst_256x8_64x6_2x1_v_bz_TNT` is q_absorb in a decoder
        # layer and the lm_head in postprocess; labeled by name alone it charged
        # the lm_head's time to q_absorb and manufactured a 2x under-prediction.
        if len(operations) == 1:
            ((operation, positions),) = operations.items()
            phases = sorted({position.phase for position in positions})
            if len(phases) > 1:
                findings.append(
                    Finding(
                        severity="warning",
                        summary=(
                            f"one operation across {len(phases)} phases: "
                            f"{operation} <- {short_name(name)}"
                        ),
                        detail=(
                            "The same kernel doing the same job in two phases is possible but "
                            "uncommon; a tile reused by a different layer is the usual "
                            "explanation. Check each phase's position.\n"
                            f"  phases: {', '.join(phases)}"
                        ),
                    )
                )

    for operation, bodies in bodies_by_operation.items():
        if len(bodies) < 2:
            continue
        detail = "\n".join(
            f"  {positions[0].coordinate} (x{len(positions)}): "
            f"type={body[0]!r} role={body[1]!r} cross_rank={body[3]!r} slots={len(body[2])}"
            for body, positions in sorted(bodies.items(), key=lambda item: str(item[0]))
        )
        findings.append(
            Finding(
                severity="error",
                summary=f"operation {operation!r} has {len(bodies)} different labels",
                detail="The labeled inventory requires one identical label per operation.\n"
                + detail,
            )
        )

    if missing_cross_rank:
        sample = "; ".join(position.coordinate for position in missing_cross_rank[:4])
        findings.append(
            Finding(
                severity="error",
                summary=f"{len(missing_cross_rank)} labels have no cross_rank",
                detail=(
                    "The analyzer silently defaults an absent cross_rank to independent, so "
                    "the omission only surfaces downstream. Write it explicitly.\n"
                    f"  {sample}"
                ),
            )
        )

    return findings


def format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "no findings"
    return "\n\n".join(
        f"[{finding.severity}] {finding.summary}\n{finding.detail}" for finding in findings
    )

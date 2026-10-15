"""Per-op occupied time, rendered from the analyzed JSON only."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def render_ops(summary_path: Path, output: Path) -> Path | None:
    summary = json.loads(summary_path.read_text())
    rows = summary.get("per_op", [])
    if not rows:
        return None
    # Analyzer plotting modules deliberately share one common style namespace.
    render_root = Path(__file__).resolve().parents[2] / "analyzer" / "python"
    sys.path.insert(0, str(render_root))
    try:
        from common.figure import add_legend, finalize, new_axes
        from common.style import ACCENT, CURVE

        rows = sorted(rows, key=lambda row: row["active_us"], reverse=True)
        figure, axis = new_axes((11, max(5, len(rows) * 0.34)))
        positions = list(range(len(rows)))
        labels = [
            f"{r['phase']} B{r['token_bucket']} · {r['operation']}"
            + (" [compiler primitive]" if r.get("attribution_category") == "compiler_primitive"
               else "")
            for r in rows
        ]
        axis.barh(
            positions,
            [r["active_us"] for r in rows],
            color=CURVE,
            label="Op busy union (nonadditive)",
        )
        axis.barh(
            positions,
            [r["overlap_us"] for r in rows],
            color=ACCENT,
            label="Of which overlaps another op",
        )
        axis.set_yticks(positions, labels=labels)
        axis.invert_yaxis()
        add_legend(axis, loc="lower right")
        quality = summary.get("quality", {})
        label = summary.get("label", "Trainium2")
        status = quality.get("status", "see report")
        finalize(
            figure,
            axis,
            output,
            title="Measured op activity; overlap retained",
            xlabel="Median per selected, attributed execution / rank / core (µs)",
            ylabel="Operation",
            run_label=f"{label} · quality: {status}",
        )
    finally:
        sys.path.remove(str(render_root))
    return output


def render_coverage(summary_path: Path, output: Path) -> Path | None:
    """Show engine label coverage as independent fractions, never a partition."""
    summary = json.loads(summary_path.read_text())
    coverage = summary.get("quality", {}).get("per_engine_active_coverage", {})
    rows = [
        (engine, values)
        for engine, values in sorted(coverage.items())
        if values.get("active_ns", 0) > 0
    ]
    if not rows:
        return None
    render_root = Path(__file__).resolve().parents[2] / "analyzer" / "python"
    sys.path.insert(0, str(render_root))
    try:
        from common.figure import add_legend, finalize, new_axes
        from common.style import ACCENT, CURVE, MARKER

        figure, axis = new_axes((11, max(4.5, len(rows) * 0.85)))
        positions = list(range(len(rows)))
        known = [
            100 * values.get("model_labeled_active_ns", values.get("labeled_active_ns", 0))
            / values["active_ns"] for _, values in rows
        ]
        unresolved = [
            100 * values["unresolved_active_ns"] / values["active_ns"] for _, values in rows
        ]
        series = [(known, CURVE, "Model operation label (includes inference)")]
        if any("compiler_primitive_active_ns" in values for _, values in rows):
            primitive = [
                100 * values.get("compiler_primitive_active_ns", 0) / values["active_ns"]
                for _, values in rows
            ]
            series.append((primitive, MARKER, "Compiler primitive; model owner unknown"))
        series.append((unresolved, ACCENT, "Unresolved instruction"))
        height = 0.72 / len(series)
        for index, (values, color, label) in enumerate(series):
            offset = (index - (len(series) - 1) / 2) * height
            bars = axis.barh([value + offset for value in positions], values,
                             height=height * 0.9, color=color, label=label)
            axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
        axis.set_yticks(positions, labels=[engine for engine, _ in rows])
        axis.invert_yaxis()
        axis.set_xlim(0, 112)
        axis.set_xticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
        add_legend(axis, loc="upper left")
        axis.get_legend().set_bbox_to_anchor((0, -0.19))
        figure.text(
            0.12,
            -0.28,
            "Each fraction uses the engine's active instruction union. "
            "Activity categories\nmay overlap on the same engine; "
            "these bars do not partition time. Zero-activity tracks omitted.",
            fontsize=10,
        )
        finalize(
            figure,
            axis,
            output,
            title="Operation label coverage by engine",
            xlabel="Fraction of active instruction time across selected executions / ranks / cores",
            ylabel="Engine track",
            run_label=summary.get("label", "Trainium2"),
        )
    finally:
        sys.path.remove(str(render_root))
    return output

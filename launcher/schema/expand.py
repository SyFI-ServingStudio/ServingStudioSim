"""Turn a preset into the concrete param dicts a run actually uses.

Three stages a preset flows through here, all schema-driven (the ParamDef type
decides, not syntax — design §1.2.1.1):

- `_classify_dimension`: is a value concrete, a list/dict sweep dim, or a list
  *value*? (Shared with `validate`, which checks the same classification.)
- `expand_sweep_params`: the five-step pipeline (identify dims → cartesian
  product → apply `derived` → apply `constraints` → stash labels).
- `normalize_params` / `_format_log_dir`: coerce to schema types + fill
  defaults, then expand `{model}_{tp}`-style `log_dir` templates.

Expression evaluation lives in `expr.py`; this module only orchestrates it.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

from .expr import _eval, _make_evaluator
from .loader import CONTROL_KEYS, Schema

_SWEEP_LABELS_KEY = "_sweep_labels"
_UNKNOWN_LOG_DIR_PLACEHOLDERS_KEY = "_unknown_log_dir_placeholders"

_SCALAR_TYPES = frozenset({"int", "float", "bool", "string", "path"})
_LIST_TYPES = frozenset({"int_list", "float_list", "string_list", "path_list"})

# Scalar element type for each collection type (used to coerce list elements).
_LIST_ELEM = {
    "int_list": "int",
    "float_list": "float",
    "string_list": "string",
    "path_list": "path",
}


# ── dimension classification ────────────────────────────────────────────────


def _classify_dimension(key: str, value: Any, dep_schema) -> str:
    """One of: concrete | list_sweep | dict_sweep | list_value. Driven by the
    schema ParamDef type per design §1.2.1.1 (schema decides, not syntax)."""
    pdef = dep_schema.params.get(key)
    ptype = pdef["type"] if pdef else None
    if ptype in _LIST_TYPES:
        return "list_value"  # the list IS the value; never a sweep
    if ptype in _SCALAR_TYPES:
        if isinstance(value, list):
            return "list_sweep"
        if isinstance(value, dict):
            return "dict_sweep"
    return "concrete"


# ── normalization (type coercion + default fill) ────────────────────────────


def _coerce_scalar(value: Any, ptype: str) -> Any:
    if ptype == "int":
        return int(value)
    if ptype == "float":
        return float(value)
    if ptype == "bool":
        return bool(value)
    return str(value)  # string / path


def normalize_params(params: dict, schema: Schema) -> dict:
    """Coerce values to their schema type and fill defaults. Operates on a
    POST-expansion dict (all scalars hold scalar values)."""
    dep_schema = schema.deployment_schemas[params["deployment"]]
    out = dict(params)
    for key, pdef in dep_schema.params.items():
        ptype = pdef["type"]
        if key in out and out[key] is not None:
            value = out[key]
            if ptype in _LIST_TYPES:
                out[key] = [_coerce_scalar(v, _LIST_ELEM[ptype]) for v in value]
            else:
                out[key] = _coerce_scalar(value, ptype)
        elif "default" in pdef:
            out[key] = pdef["default"]
    return out


# ── sweep expansion (five-step pipeline, design §1.2.1.1) ───────────────────


def expand_sweep_params(preset: dict, schema: Schema) -> list[dict]:
    """Expand a preset into one or more concrete param dicts."""
    dep_schema = schema.deployment_schemas[preset["deployment"]]

    # Seed with schema defaults so `derived` / `constraints` can reference
    # defaulted params (e.g. ep_size) that the preset never mentions. Explicit
    # preset values and sweep picks overlay these below.
    base: dict[str, Any] = dict(dep_schema.defaults)
    # Each dim is a list of (assignments_dict, labels_dict) options.
    sweep_dims: list[list[tuple[dict[str, Any], dict[str, str]]]] = []

    for key, value in preset.items():
        if key in CONTROL_KEYS or key.startswith("_"):
            continue
        if key == "deployment" or key not in dep_schema.params:
            base[key] = value
            continue
        kind = _classify_dimension(key, value, dep_schema)
        if kind == "list_sweep":
            sweep_dims.append([({key: v}, {}) for v in value])
        elif kind == "dict_sweep":
            sweep_dims.append([({key: v}, {key: label}) for label, v in value.items()])
        else:  # concrete | list_value
            base[key] = value

    # sweep_groups: each group is one zipped dim (all fields of an entry together).
    for group_key, entries in (preset.get("sweep_groups") or {}).items():
        sweep_dims.append(
            [(dict(partial), {group_key: str(idx)}) for idx, partial in enumerate(entries)]
        )

    derived = preset.get("derived", {}) or {}
    constraints = preset.get("constraints", []) or []

    candidates: list[dict] = []
    for combo in (itertools.product(*sweep_dims) if sweep_dims else [()]):
        candidate = dict(base)
        labels: dict[str, str] = {}
        for assignments, dim_labels in combo:
            candidate.update(assignments)
            labels.update(dim_labels)

        # Step 3: apply derived in declaration order.
        if derived:
            interp = _make_evaluator()
            interp.symtable.update(candidate)
            for lhs, rhs in derived.items():
                candidate[lhs] = _eval(interp, rhs)
                interp.symtable[lhs] = candidate[lhs]

        # Step 4: apply constraints (drop on any false; silent).
        if not _all_constraints_pass(candidate, constraints):
            continue

        if labels:
            candidate[_SWEEP_LABELS_KEY] = labels
        candidates.append(candidate)

    return candidates


def _all_constraints_pass(candidate: dict, constraints: list) -> bool:
    if not constraints:
        return True
    interp = _make_evaluator()
    interp.symtable.update(candidate)
    return all(bool(_eval(interp, expr)) for expr in constraints)


# ── log_dir templating (design §1.7 Q3: lives in normalization) ─────────────

_LOG_DIR_ALIASES = {
    "model": lambda p: Path(str(p.get("model_config", ""))).stem,
    "tp": lambda p: p.get("tp_size"),
    "ep": lambda p: p.get("ep_size"),
    "rate": lambda p: p.get("request_rate"),
}


def _format_log_dir(params: dict) -> dict:
    """Expand `{model}_{tp}_{rate}`-style templates in `log_dir`. Placeholders
    resolve from sweep labels first, then param values, then aliases. Unknown
    placeholders stay literal, but warn because they often mean a misspelled
    sweep dimension in the output path."""
    template = params.get("log_dir")
    if not isinstance(template, str) or "{" not in template:
        return params

    fields: dict[str, Any] = {k: v for k, v in params.items() if not k.startswith("_")}
    fields.update(params.get(_SWEEP_LABELS_KEY, {}))
    for alias, alias_fn in _LOG_DIR_ALIASES.items():
        fields.setdefault(alias, alias_fn(params))

    warned_unknown_placeholders = set(params.get(_UNKNOWN_LOG_DIR_PLACEHOLDERS_KEY, ()))
    unknown_placeholders = set(warned_unknown_placeholders)

    class _Safe(dict):
        def __missing__(self, key):  # leave unknown placeholders untouched
            if key not in warned_unknown_placeholders:
                print(
                    f"[warn] unknown log_dir placeholder {{{key}}}; leaving literal",
                    file=sys.stderr,
                )
            unknown_placeholders.add(key)
            return "{" + key + "}"

    out = dict(params)
    out["log_dir"] = template.format_map(_Safe(fields))
    if unknown_placeholders:
        out[_UNKNOWN_LOG_DIR_PLACEHOLDERS_KEY] = sorted(unknown_placeholders)
    return out


def validate_unique_log_dirs(param_sets: list[dict]) -> bool:
    """Reject sweep plans whose concrete runs would write into the same log_dir."""
    seen: dict[str, int] = {}
    for params in param_sets:
        key = str(Path(str(params.get("log_dir", "logs"))).resolve())
        seen[key] = seen.get(key, 0) + 1
    collisions = {log_dir: count for log_dir, count in seen.items() if count > 1}
    for log_dir, count in collisions.items():
        print(
            f"[invalid] {count} runs share log_dir {log_dir!r}; aborting sweep",
            file=sys.stderr,
        )
    return not collisions

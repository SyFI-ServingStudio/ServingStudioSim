"""Static preset validation (design §1.2.1 + the §1.2.1.1 V1–V6 sweep checks).

`validate_params` returns human-readable error strings (empty = valid) and
never mutates the preset. It leans on `expr` for the expression grammar gate
and on `expand._classify_dimension` so it checks the exact classification the
expander will later apply. Per-param choices are Rust-authoritative ParamDef
metadata and are validated generically here.
"""

from __future__ import annotations

from .expand import _classify_dimension, _group_entries
from .expr import _check_expr_grammar, _expr_names
from .loader import CONTROL_KEYS, Schema


def _preset_values(preset: dict, key: str) -> list:
    """All scalar values `key` could take across scalar/list/dict preset forms."""
    values = []
    if key in preset:
        value = preset[key]
        if isinstance(value, list):
            values.extend(value)
        elif isinstance(value, dict):
            values.extend(value.values())
        else:
            values.append(value)
    for entries in (preset.get("sweep_groups") or {}).values():
        for _label, partial in _group_entries(entries):
            if key in partial:
                values.append(partial[key])
    return values


def _defined_names(preset: dict, dep_schema) -> set[str]:
    """Names available as expression free variables BEFORE `derived` runs:
    defaulted schema params + base concretes + list-valued params + sweep dims
    + sweep_group fields. Defaulted params count as defined because they always
    hold a value at evaluation time (seeded in `expand_sweep_params`)."""
    names: set[str] = set(dep_schema.defaults)
    for key in preset:
        if key in CONTROL_KEYS or key.startswith("_") or key == "deployment":
            continue
        if key in dep_schema.params:
            names.add(key)  # concrete, list_value, list_sweep, or dict_sweep
    for entries in (preset.get("sweep_groups") or {}).values():
        for _label, partial in _group_entries(entries):
            names.update(partial.keys())
    return names


def validate_params(preset: dict, schema: Schema) -> list[str]:
    """Return a list of human-readable error strings (empty = valid)."""
    errors: list[str] = []

    name = preset.get("deployment")
    if not name:
        return ["missing required key 'deployment'"]
    if name not in schema.deployment_schemas:
        known = ", ".join(sorted(schema.deployment_schemas)) or "(none)"
        return [f"unknown deployment {name!r}; known: {known}"]
    dep_schema = schema.deployment_schemas[name]

    # Unknown top-level keys.
    allowed = set(dep_schema.params) | CONTROL_KEYS | {"deployment"}
    for key in preset:
        if key in allowed or key.startswith("_"):
            continue
        errors.append(f"unknown param {key!r} for deployment {name!r}")

    # Rust-authoritative closed value sets. Works before expansion so scalar
    # values, list sweeps, and dict-labeled sweeps are all checked.
    for param_name, pdef in dep_schema.params.items():
        choices = pdef.get("choices")
        if not choices:
            continue
        for value in _preset_values(preset, param_name):
            if value not in choices:
                errors.append(f"{param_name}={value!r} is not one of {choices}")

    derived = preset.get("derived", {}) or {}
    constraints = preset.get("constraints", []) or []
    sweep_groups = preset.get("sweep_groups", {}) or {}

    available = _defined_names(preset, dep_schema)
    group_fields = {
        f
        for entries in sweep_groups.values()
        for _label, p in _group_entries(entries)
        for f in p
    }

    # V1: derived LHS must be a declared schema param.
    for lhs in derived:
        if lhs not in dep_schema.params:
            errors.append(
                f"derived target {lhs!r} is not a schema param of {name!r} "
                "(auxiliary names are not allowed; inline into a constraint)"
            )

    # V2: a derived LHS must not also be a sweep dim or a sweep_groups field.
    for lhs in derived:
        kind = (
            _classify_dimension(lhs, preset.get(lhs), dep_schema)
            if lhs in preset
            else "concrete"
        )
        if kind in ("list_sweep", "dict_sweep"):
            errors.append(f"derived target {lhs!r} also appears as a sweep dim")
        if lhs in group_fields:
            errors.append(f"derived target {lhs!r} also appears in sweep_groups")

    # §1.2.1.1 step 2: a param must not be BOTH an independent sweep dim and a
    # sweep_groups field. Expansion would otherwise let the group entry silently
    # overwrite the independent dim (double definition).
    for key in preset:
        if key in CONTROL_KEYS or key.startswith("_") or key == "deployment":
            continue
        if key not in group_fields:
            continue
        if _classify_dimension(key, preset[key], dep_schema) in ("list_sweep", "dict_sweep"):
            errors.append(
                f"param {key!r} is both an independent sweep dim and a "
                "sweep_groups field (double definition)"
            )

    # V3 + V6: derived RHS grammar must be supported; vars resolve in topo order.
    resolved = set(available)
    for lhs, rhs in derived.items():
        grammar_errors = _check_expr_grammar(str(rhs))
        errors.extend(f"derived[{lhs!r}]: {e}" for e in grammar_errors)
        if grammar_errors:
            continue
        missing = _expr_names(str(rhs)) - resolved
        if missing:
            errors.append(
                f"derived[{lhs!r}] references undefined names {sorted(missing)} "
                "(must come from base / sweep / an earlier derived entry)"
            )
        resolved.add(lhs)

    # V5 + V6: constraint grammar must be supported; free vars must be defined.
    for expr in constraints:
        grammar_errors = _check_expr_grammar(str(expr))
        errors.extend(f"constraint {e}" for e in grammar_errors)
        if grammar_errors:
            continue
        missing = _expr_names(str(expr)) - resolved
        if missing:
            errors.append(
                f"constraint {expr!r} references undefined names {sorted(missing)}"
            )

    # V4: after derived, every required param must have a value.
    for required_name in dep_schema.required_names:
        if required_name in preset or required_name in derived:
            continue
        errors.append(f"missing required param {required_name!r} for deployment {name!r}")

    return errors

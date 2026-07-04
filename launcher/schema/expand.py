"""Turn a structured preset into the concrete config trees a run actually uses.

A preset is the config tree (`deployment` / `workload` / `io` / `pools`) plus
three optional launcher control blocks (new-interface-design §11):

    sweep:        {name: [values] | {label: value}}   # independent dims
    compound:     {group: {label: {member: value}}}    # correlated dims (zip)
    derived:      {name: "<expr over sweep/earlier-derived names>"}
    constraints:  ["<expr>", ...]                      # drop a combo if false

A **`compound` group** zips several members into one cartesian factor: each labeled
row binds *all* its members together (so `tp`/`rate` move as a pair, not a 2×2
grid). The group *name* is one aggregation axis (its tick = the row label); the
members fill params. Sweep / compound-member / derived names are referenced inside
the tree as whole-leaf placeholders ``${name}`` (typed substitution) and inside
`io.log_dir` as ``{name}`` (string templating); a group name resolves to its row
label. The pipeline:

- `expand_sweep_params`: cartesian product over `sweep` dims → apply `derived`
  → drop on `constraints` → deep-copy the tree and substitute every ``${name}``
  with its resolved (typed) value;
- `normalize_params`: walk the tree against the schema, coercing present leaves
  to their ParamDef type and filling defaults (the Rust structs carry no serde
  defaults — the launcher writes complete configs);
- `_format_log_dir`: expand ``{name}`` templates in `io.log_dir` from the
  resolved sweep/derived env + dict-sweep labels.

Expression evaluation lives in `expr.py`; this module only orchestrates it.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

from .expr import _eval, _make_evaluator
from .loader import CONTROL_KEYS, Registry, iter_slots, log_dir_of

_SWEEP_LABELS_KEY = "_sweep_labels"
_ENV_KEY = "_env"
_COMPOUND_MEMBERS_KEY = "_compound_members"
_UNKNOWN_LOG_DIR_PLACEHOLDERS_KEY = "_unknown_log_dir_placeholders"

# A whole-leaf typed placeholder: the entire string is ``${name}``.
_PLACEHOLDER_RE = re.compile(r"^\$\{(\w+)\}$")

_SCALAR_TYPES = frozenset({"int", "float", "bool", "string", "path"})
_LIST_TYPES = frozenset({"int_list", "float_list", "string_list", "path_list"})
_LIST_ELEM = {
    "int_list": "int",
    "float_list": "float",
    "string_list": "string",
    "path_list": "path",
}


# ── placeholder collection / substitution ────────────────────────────────────


def _tree_only(preset: dict) -> dict:
    """The config tree with launcher control / internal keys removed."""
    return {
        k: v
        for k, v in preset.items()
        if k not in CONTROL_KEYS and not k.startswith("_")
    }


def collect_placeholders(node: Any) -> set[str]:
    """All ``${name}`` whole-leaf placeholder names anywhere in the tree."""
    found: set[str] = set()
    if isinstance(node, dict):
        for v in node.values():
            found |= collect_placeholders(v)
    elif isinstance(node, list):
        for v in node:
            found |= collect_placeholders(v)
    elif isinstance(node, str):
        m = _PLACEHOLDER_RE.match(node)
        if m:
            found.add(m.group(1))
    return found


def _substitute(node: Any, env: dict[str, Any]) -> Any:
    """Deep-copy `node`, replacing every whole-leaf ``${name}`` with `env[name]`
    (the typed value). Strings that merely *contain* `${...}` are left untouched
    (only `io.log_dir`'s `{name}` templating rewrites those, later)."""
    if isinstance(node, dict):
        return {k: _substitute(v, env) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, env) for v in node]
    if isinstance(node, str):
        m = _PLACEHOLDER_RE.match(node)
        if m:
            return env[m.group(1)]
    return node


# ── sweep dim parsing ────────────────────────────────────────────────────────


def _sweep_dim_options(value: Any) -> list[tuple[Any, str | None]]:
    """Normalize one `sweep` entry to `[(value, label_or_None), ...]`.

    - list  → index values, no explicit label (log_dir `{name}` resolves to the
      value via the env);
    - dict  → keys are labels, for readable per-run dirs
      (`{"lo": 1, "hi": 8}` → labels `lo`/`hi`)."""
    if isinstance(value, dict):
        return [(v, str(label)) for label, v in value.items()]
    if isinstance(value, list):
        return [(v, None) for v in value]
    return [(value, None)]  # scalar sweep = a one-point dim (rare, but valid)


# ── sweep expansion ──────────────────────────────────────────────────────────


def expand_sweep_params(preset: dict, registry: Registry) -> list[dict]:
    """Expand a structured preset into one or more concrete config trees."""
    tree = _tree_only(preset)
    sweep = preset.get("sweep", {}) or {}
    compound = preset.get("compound", {}) or {}
    derived = preset.get("derived", {}) or {}
    constraints = preset.get("constraints", []) or []
    # The per-kernel backend override map is control (schema-exempt) but is still
    # a sweep participant: its `${name}` values resolve from the same env, and the
    # substituted map is written back onto each candidate as a plain `backends`
    # block (Rust's `RunConfig.backends`). Nested `pool -> role -> value` by now
    # (`__main__._merge_backends_file` un-flattens the file's `pool/role` keys).
    backends = preset.get("backends", {}) or {}

    # All members across compound groups (each fills a param, like a sweep dim).
    compound_members = {m for rows in compound.values() for row in rows.values() for m in row}

    placeholders = collect_placeholders(tree) | collect_placeholders(backends)
    resolvable = set(sweep) | set(derived) | set(compound) | compound_members
    missing = placeholders - resolvable
    if missing:
        shown = sorted("${" + m + "}" for m in missing)
        raise ValueError(
            f"config references undefined placeholders {shown}; every ${{name}} "
            "must be declared in `sweep`, `compound`, or `derived`"
        )

    # One cartesian factor per sweep dim and per compound group. A sweep factor's
    # option is (value, label); a compound factor's is (row_members_dict, label).
    factors: list[tuple[str, str, list]] = [
        ("sweep", name, _sweep_dim_options(sweep[name])) for name in sweep
    ]
    factors += [
        ("compound", group, [(dict(row), str(label)) for label, row in rows.items()])
        for group, rows in compound.items()
    ]
    dims = [opts for _kind, _name, opts in factors]

    candidates: list[dict] = []
    for combo in itertools.product(*dims) if dims else [()]:
        env: dict[str, Any] = {}
        labels: dict[str, str] = {}
        for (kind, name, _opts), (payload, label) in zip(factors, combo):
            if kind == "sweep":
                env[name] = payload
                if label is not None:
                    labels[name] = label
            else:  # compound: bind every member of the chosen row together
                env.update(payload)
                env[name] = label  # the group name is one (labeled) aggregation axis
                labels[name] = label

        if derived:
            interp = _make_evaluator()
            interp.symtable.update(env)
            for lhs, rhs in derived.items():
                env[lhs] = _eval(interp, rhs)
                interp.symtable[lhs] = env[lhs]

        if not _all_constraints_pass(env, constraints):
            continue

        candidate = _substitute(tree, env)
        if backends:
            # Same env, same typed substitution as the tree — so `${attn_be}`
            # becomes this combo's candidate list. Attached as a plain (non-`_`)
            # key so `write_config` passes it straight to Rust.
            candidate["backends"] = _substitute(backends, env)
        if labels:
            candidate[_SWEEP_LABELS_KEY] = labels
        candidate[_ENV_KEY] = env
        if compound_members:
            candidate[_COMPOUND_MEMBERS_KEY] = sorted(compound_members)
        candidates.append(candidate)

    return candidates


def _all_constraints_pass(env: dict, constraints: list) -> bool:
    if not constraints:
        return True
    interp = _make_evaluator()
    interp.symtable.update(env)
    return all(bool(_eval(interp, expr)) for expr in constraints)


# ── normalization (type coercion + default fill, schema-walked) ──────────────


def _coerce_scalar(value: Any, ptype: str) -> Any:
    if ptype == "int":
        return int(value)
    if ptype == "float":
        return float(value)
    if ptype == "bool":
        return bool(value)
    return str(value)  # string / path


def _coerce(value: Any, ptype: str) -> Any:
    if ptype in _LIST_TYPES:
        elem = _LIST_ELEM[ptype]
        return [_coerce_scalar(v, elem) for v in value]
    return _coerce_scalar(value, ptype)


def normalize_params(candidate: dict, registry: Registry) -> dict:
    """Coerce every present leaf to its schema type and fill defaults, walking
    the config tree. Operates on a copy; preserves launcher-internal keys."""
    out = copy.deepcopy(candidate)
    for slot in iter_slots(registry, out, create=True):
        ptype = slot.pdef["type"]
        if slot.present:
            slot.container[slot.key] = _coerce(slot.value, ptype)
        elif "default" in slot.pdef:
            slot.container[slot.key] = slot.pdef["default"]
    return out


# ── log_dir templating (`{name}` over the resolved sweep/derived env) ────────


def _format_log_dir(candidate: dict) -> dict:
    """Expand `{name}` templates in `io.log_dir` from the resolved sweep/derived
    env, with dict-sweep labels taking precedence. Unknown placeholders stay
    literal but warn (usually a misspelled sweep name in the output path)."""
    template = log_dir_of(candidate, default="")
    if "{" not in template:
        return candidate

    fields: dict[str, Any] = dict(candidate.get(_ENV_KEY, {}))
    fields.update(candidate.get(_SWEEP_LABELS_KEY, {}))

    warned = set(candidate.get(_UNKNOWN_LOG_DIR_PLACEHOLDERS_KEY, ()))
    unknown = set(warned)

    class _Safe(dict):
        def __missing__(self, key):
            if key not in warned:
                print(
                    f"[warn] unknown log_dir placeholder {{{key}}}; leaving literal",
                    file=sys.stderr,
                )
            unknown.add(key)
            return "{" + key + "}"

    out = copy.deepcopy(candidate)
    out.setdefault("io", {})["log_dir"] = template.format_map(_Safe(fields))
    if unknown:
        out[_UNKNOWN_LOG_DIR_PLACEHOLDERS_KEY] = sorted(unknown)
    return out


def validate_unique_log_dirs(param_sets: list[dict]) -> bool:
    """Reject sweep plans whose concrete runs would write into the same log_dir."""
    seen: dict[str, int] = {}
    for params in param_sets:
        key = str(Path(log_dir_of(params)).resolve())
        seen[key] = seen.get(key, 0) + 1
    collisions = {log_dir: count for log_dir, count in seen.items() if count > 1}
    for log_dir, count in collisions.items():
        print(
            f"[invalid] {count} runs share log_dir {log_dir!r}; aborting sweep",
            file=sys.stderr,
        )
    return not collisions


def validate_distinct_configs(param_sets: list[dict]) -> bool:
    """Reject a plan where two runs share an identical config tree (ignoring
    `io.log_dir` + launcher-internal `_` keys). Such a pair is a duplicate
    simulation — the post-expansion dual of `validate_unique_log_dirs`. It is the
    catch-all for a swept/derived dim that moves only the log_dir (or nothing),
    not any config value: the runs differ on paper but compute the same thing."""
    seen: dict[str, list[str]] = {}
    for params in param_sets:
        tree = {k: v for k, v in params.items() if not k.startswith("_")}
        io = tree.get("io")
        if isinstance(io, dict):
            tree["io"] = {k: v for k, v in io.items() if k != "log_dir"}
        key = json.dumps(tree, sort_keys=True, default=str)
        seen.setdefault(key, []).append(log_dir_of(params))
    duplicates = [dirs for dirs in seen.values() if len(dirs) > 1]
    for dirs in duplicates:
        print(
            f"[invalid] {len(dirs)} runs share an identical config, differing only "
            f"in log_dir: {sorted(dirs)}. A swept dimension changes no config value "
            "— reference it on the param you mean to sweep (e.g. `tp_size: ${name}`).",
            file=sys.stderr,
        )
    return not duplicates

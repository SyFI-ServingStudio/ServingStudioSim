"""Static preset validation for the structured config interface.

`validate_params` returns human-readable error strings (empty = valid) and never
mutates the preset. Think of it as the preset *compiler's* front-end: every error
it can prove statically must be caught here, with a clear message, rather than
surfacing as a traceback in `normalize`/`expand` or as a silently-wrong run.

Layers of checks, against the Rust-authoritative `Registry`:

- **lexical / tags**: known `deployment`; no unknown keys anywhere (the
  payload-typo guard — a Rust tagged enum silently ignores `deny_unknown_fields`,
  so arch/worker typos must be caught here, plan G1); every `arch.type` /
  `worker.type` is a tag advertised for that pool's contract class;
- **structural**: the deployment's required pool roles are present, each pool has
  a non-empty `groups` list, each group is a mapping with arch + worker;
- **type / value**: every present, non-placeholder leaf has a value of its
  ParamDef type (TE) and, for closed sets, an allowed value; required leaves are
  present;
- **control-flow (sweep / derived / constraints)**: expression grammar +
  free-variable resolution (V1–V6); no partial `${...}` interpolation (R1); no
  dead sweep dim (R2) or dead derived name (R3); no empty sweep dim (R4); every
  `${name}` / `io.log_dir` `{name}` resolves to a declared name (P).

The validator is two phase-bound judgments over one shared schema walk
(`_walk_schema`): deployment tag → unknown keys → structure → per-leaf
type/choices/required. The phases differ on exactly one axis, `defer_placeholders`:

- **`validate_params` (raw / pre-expansion)** — `_walk_schema(defer=True)` plus the
  control-block checks (sweep / derived / constraints, R1–R7). A whole-leaf
  `${name}` is "supplied later", so its type / choice / provider-tag check is
  deferred; only required-presence is meaningful now.
- **`validate_expanded` (concrete / post-expansion)** — `_walk_schema(defer=False)`
  plus a placeholder-residue scan. After substitution every leaf is concrete, so
  this is the *complete* schema judgment and the single gate that says "this
  config is safe to hand to the Rust binary": no placeholder residue (whole or
  partial), no unknown key, every provider tag resolved and advertised, every
  required leaf present, every value the right type / an allowed choice. It runs
  BEFORE `normalize` (which would coerce — `bool("false") == True` — and hide an
  error). Control-block checks are NOT re-run: sweep / derived / constraints are
  raw-only and already consumed by expansion.

The remaining duals are cross-candidate plan guards, owned by the plan layer
(`__main__`), not by either per-config judgment: `validate_distinct_configs`
(every run is a *distinct* simulation, R5), `validate_unique_log_dirs`, and the
empty-plan guard (R6).
"""

from __future__ import annotations

import copy
import re
import string
from typing import Any

from .expand import _sweep_dim_options, collect_placeholders
from .expr import _check_expr_grammar, _expr_names
from .loader import CONTROL_KEYS, Registry, iter_slots, unknown_keys

_PLACEHOLDER_RE = re.compile(r"^\$\{(\w+)\}$")


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.match(value))


def _is_identifier(name: Any) -> bool:
    """A symbolic name usable as `${name}` and an `_env` key: a string identifier.
    Non-identifier keys (e.g. an int) never resolve and would crash the axis sort."""
    return isinstance(name, str) and name.isidentifier()


def _is_safe_label(label: Any) -> bool:
    """A label that may become a log_dir path segment: a non-empty string with no
    path separator or `.`/`..` traversal."""
    return isinstance(label, str) and bool(label) and "/" not in label and label not in (".", "..")


def _sweep_names(preset: dict) -> set[str]:
    return set(preset.get("sweep", {}) or {})


def _tree_of(preset: dict) -> dict:
    """The config tree alone — control blocks and internal keys removed."""
    return {k: v for k, v in preset.items() if k not in CONTROL_KEYS and not k.startswith("_")}


def _type_ok(value: Any, ptype: str) -> bool:
    """Whether `value` is a legal instance of ParamDef `ptype`. `bool` is NOT an
    int/float here (Python's `bool ⊂ int` would otherwise let `true` pass as an
    int param). List params accept whole-leaf `${name}` placeholders per element
    (substituted later)."""
    if ptype == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if ptype == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ptype == "bool":
        return isinstance(value, bool)
    if ptype in ("string", "path"):
        return isinstance(value, str)
    if ptype in ("int_list", "float_list", "string_list", "path_list"):
        if not isinstance(value, list):
            return False
        elem = ptype[: -len("_list")]
        return all(_is_placeholder(v) or _type_ok(v, elem) for v in value)
    return True  # unknown type → don't block (forward-compatible)


def _parse_log_dir_template(preset: dict) -> tuple[set[str], list[str]]:
    """Parse `io.log_dir` as a `{name}`-only path template. Returns
    `(names, errors)`: `names` = the bare-identifier fields it references;
    `errors` = every field that is *not* a bare `{identifier}`.

    `io.log_dir` is a filesystem path, not an arbitrary Python format expression,
    so a field may only be a bare sweep/derived name — no attribute access
    (`{ptp.__class__}`), indexing (`{ptp[0]}`), conversion (`{ptp!r}`), or format
    spec (`{ptp:04d}`). Those parse to a valid base name today and then either
    inject junk (`<class 'int'>`) or crash `str.format_map`; reject them here. A
    malformed template (unbalanced braces) becomes one error, not a traceback."""
    io = preset.get("io")
    template = io.get("log_dir") if isinstance(io, dict) else None
    if not isinstance(template, str):
        return set(), []
    try:
        fields = list(string.Formatter().parse(template))
    except ValueError as exc:
        return set(), [f"io.log_dir template is malformed: {exc}"]
    names: set[str] = set()
    errors: list[str] = []
    for _literal, field, spec, conv in fields:
        if field is None:
            continue
        if field.isidentifier() and conv is None and not spec:
            names.add(field)
            continue
        detail = []
        if not field.isidentifier():
            detail.append("not a bare identifier")
        if conv is not None:
            detail.append(f"conversion !{conv}")
        if spec:
            detail.append(f"format spec :{spec}")
        errors.append(
            f"io.log_dir field {field!r} is illegal ({', '.join(detail)}); only a "
            "bare `{name}` (a sweep/derived name) is allowed in the log_dir path"
        )
    return names, errors


def _partial_placeholder_paths(node: Any, path: str = "") -> list[str]:
    """Tree paths whose string value *contains* `${...}` but is not a whole-leaf
    `${name}`. `_substitute` only replaces whole-leaf placeholders, so a partial
    one is silently dropped — almost always a bug, so it is rejected (R1)."""
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _partial_placeholder_paths(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _partial_placeholder_paths(v, f"{path}[{i}]")
    elif isinstance(node, str):
        if "${" in node and not _PLACEHOLDER_RE.match(node):
            bad.append(path)
    return bad


def _walk_schema(config: dict, registry: Registry, *, defer_placeholders: bool) -> list[str]:
    """The schema-guided walk shared by both phase judgments: deployment tag →
    unknown keys (payload-typo guard) → structure (roles / groups / provider tags)
    → per-leaf type / choices / required. `defer_placeholders` is the ONLY axis on
    which the raw and concrete phases differ — see the module docstring."""
    deployment = config.get("deployment")
    if not deployment:
        return ["missing required key 'deployment'"]
    if _is_placeholder(deployment):
        # params-only: `deployment` selects the whole config-tree shape — structure,
        # not a value, so it must be a literal tag (never a `${...}` placeholder).
        return [
            f"deployment {deployment!r} is a placeholder; `deployment` selects the "
            "config-tree shape and must be a literal tag — split structural variants "
            "into separate presets (a `variants` manifest)"
        ]
    if deployment not in registry.deployments:
        known = ", ".join(sorted(registry.deployments)) or "(none)"
        return [f"unknown deployment {deployment!r}; known: {known}"]

    errors: list[str] = []
    for path in unknown_keys(registry, config):
        errors.append(f"unknown key {path!r} for deployment {deployment!r}")
    _check_structure(errors, config, registry, deployment)
    errors.extend(_check_leaves(config, registry, defer_placeholders=defer_placeholders))
    return errors


def validate_params(preset: dict, registry: Registry) -> list[str]:
    """Raw / pre-expansion judgment: the shared schema walk (placeholders deferred)
    plus the control-block checks. Returns human-readable error strings (empty =
    valid); never mutates the preset."""
    errors = _walk_schema(preset, registry, defer_placeholders=True)
    # Control blocks need a parsed tree; skip them if the deployment itself is bad
    # (the walk already returned a single deployment error in that case).
    if preset.get("deployment") in registry.deployments:
        errors.extend(_validate_control_blocks(preset))
    return errors


def _check_leaves(candidate: dict, registry: Registry, *, defer_placeholders: bool) -> list[str]:
    """Per-leaf type / choices / required checks over every expected slot.

    Walks a deep copy with `create=True` so a wholly-absent skeleton container
    (e.g. a missing `workload` block) is still materialized into slots — without
    it, a missing required leaf inside that block would yield no slot and slip
    past unflagged (the "drop the whole block to dodge `trace_files`" bypass).

    `defer_placeholders` distinguishes the two callers:
    - pre-expansion (`True`): a whole-leaf `${name}` has an unknown value/tag, so
      its type/choice is deferred; only required-presence is meaningful.
    - post-expansion (`False`): every leaf is concrete, so a lingering placeholder
      is itself an error and all type/choice/required checks run for real."""
    errors: list[str] = []
    for slot in iter_slots(registry, copy.deepcopy(candidate), create=True):
        path = ".".join(slot.path)
        if slot.present:
            if _is_placeholder(slot.value):
                if defer_placeholders:
                    continue
                errors.append(
                    f"{path} still holds an unresolved placeholder {slot.value!r} after expansion"
                )
                continue
            ptype = slot.pdef["type"]
            if not _type_ok(slot.value, ptype):
                errors.append(f"{path}={slot.value!r} is not a valid {ptype}")
                continue
            choices = slot.pdef.get("choices")
            if choices:
                values = slot.value if ptype.endswith("_list") else [slot.value]
                for index, value in enumerate(values):
                    if _is_placeholder(value) and defer_placeholders:
                        continue
                    if value not in choices:
                        value_path = f"{path}[{index}]" if ptype.endswith("_list") else path
                        errors.append(f"{value_path}={value!r} is not one of {choices}")
        else:
            required = slot.pdef.get("required", False)
            has_default = "default" in slot.pdef
            if required and not has_default:
                errors.append(f"missing required param {path!r}")
    return errors


def _check_structure(errors: list[str], preset: dict, registry: Registry, deployment: str) -> None:
    """Required roles present, each pool a non-empty `groups` list of mappings,
    each group's arch/worker a tag advertised for the pool's contract."""
    # `workload` / `io` must be mappings when present — a non-dict (`io: []` /
    # `io: null` / `io: "x"`) would otherwise be silently rebuilt as the default
    # block by the create=True leaf walk, masking the malformed config.
    for block in ("workload", "io"):
        node = preset.get(block)
        if node is not None and not isinstance(node, dict):
            errors.append(f"{block!r} must be a mapping")
    pools = preset.get("pools")
    if not isinstance(pools, dict):
        errors.append(f"deployment {deployment!r} requires a 'pools' mapping")
        return
    for role, contract in registry.roles(deployment).items():
        pool = pools.get(role)
        if pool is None:
            errors.append(f"deployment {deployment!r} requires pool role {role!r}")
            continue
        if not isinstance(pool, dict):
            errors.append(f"pools.{role} must be a mapping")
            continue
        groups = pool.get("groups")
        if not isinstance(groups, list) or not groups:
            errors.append(f"pools.{role} must have a non-empty 'groups' list")
            continue
        for gi, group in enumerate(groups):
            where = f"pools.{role}.groups[{gi}]"
            if not isinstance(group, dict):
                errors.append(f"{where} must be a mapping")
                continue
            _check_provider_tag(
                errors, group.get("arch"), "arch", registry.arch_tags(contract), where
            )
            _check_provider_tag(
                errors, group.get("worker"), "worker", registry.worker_tags(contract), where
            )


def _check_provider_tag(
    errors: list[str], node: Any, kind: str, tags: list[str], where: str
) -> None:
    # params-only: a provider block / its `type` tag is *structure* (it selects
    # which schema applies), not a value — so it must be literal, never a `${...}`
    # placeholder. (You sweep a tag's params, not the tag or the whole block.)
    if _is_placeholder(node):
        errors.append(
            f"{where}.{kind} is the placeholder {node!r}; a provider block is "
            "structure and must be literal — a whole arch/worker cannot be injected "
            "via `${...}`. Put the block inline and sweep its params, or split "
            "structural variants into separate presets (a `variants` manifest)"
        )
        return
    if not isinstance(node, dict):
        errors.append(f"{where}.{kind} must be a mapping with a 'type' tag")
        return
    tag = node.get("type")
    if _is_placeholder(tag):
        errors.append(
            f"{where}.{kind}.type={tag!r} is a placeholder; a provider `type` tag "
            "selects which schema applies — it is structure, not a value, and must "
            "be a literal tag. Sweep the tag's params instead, or use a `variants` "
            "manifest to compare different tags"
        )
        return
    if tag not in tags:
        known = ", ".join(tags) or "(none)"
        errors.append(f"{where}.{kind}.type={tag!r} is not one of: {known}")


def _check_compound_semantics(
    compound: dict, sweep_names: set[str], derived_names: set[str]
) -> list[str]:
    """Cross-row / cross-name checks for compound groups (shape already verified):
    every row of a group declares the *same* member set, and group + member names
    are globally unique (no collision with sweep dims, derived names, other groups,
    or members of another group). A consistent member set is what makes the group a
    well-formed zip; unique names keep the resolved `_env` unambiguous."""
    errors: list[str] = []
    member_owner: dict[str, str] = {}
    for group, rows in compound.items():
        member_sets = [frozenset(row) for row in rows.values()]
        union = set().union(*member_sets)
        if any(ms != member_sets[0] for ms in member_sets):
            errors.append(
                f"compound group {group!r}: every row must declare the same members; "
                f"members seen across rows: {sorted(union)}"
            )
        if group in sweep_names or group in derived_names:
            errors.append(f"compound group {group!r} collides with a sweep/derived name")
        for m in sorted(union):
            if m == group or m in compound:
                errors.append(f"compound member {m!r} collides with a group name")
            if m in sweep_names or m in derived_names:
                errors.append(f"compound member {m!r} collides with a sweep/derived name")
            if m in member_owner and member_owner[m] != group:
                errors.append(
                    f"compound member {m!r} appears in both groups "
                    f"{member_owner[m]!r} and {group!r}; member names must be unique"
                )
            member_owner[m] = group
    return errors


def _validate_control_blocks(preset: dict) -> list[str]:
    """V1–V6 + R1–R7 + placeholder-resolution over the sweep / derived /
    constraints free-variable universe."""
    errors: list[str] = []

    # Shape gate (compiler parse phase): the control blocks must have their exact
    # JSON shape before anything interprets them — otherwise a wrong shape either
    # silently degrades to a no-op (`sweep: []` is falsy → treated as no sweep) or
    # crashes downstream (`sweep: [..]` has no `.items()`). Reject here, precisely.
    shape_errors: list[str] = []
    sweep_raw = preset.get("sweep")
    if sweep_raw is not None and not isinstance(sweep_raw, dict):
        shape_errors.append(
            f"`sweep` must be a mapping of {{name: values}}, got {type(sweep_raw).__name__}"
        )
    derived_raw = preset.get("derived")
    if derived_raw is not None and not isinstance(derived_raw, dict):
        shape_errors.append(
            f"`derived` must be a mapping of {{name: expression}}, got {type(derived_raw).__name__}"
        )
    constraints_raw = preset.get("constraints")
    if constraints_raw is not None and not isinstance(constraints_raw, list):
        shape_errors.append(
            f"`constraints` must be a list of expression strings, got "
            f"{type(constraints_raw).__name__}"
        )
    elif isinstance(constraints_raw, list):
        for i, expr in enumerate(constraints_raw):
            if not isinstance(expr, str):
                shape_errors.append(
                    f"constraints[{i}] must be a string expression, got {type(expr).__name__}"
                )
    compound_raw = preset.get("compound")
    if compound_raw is not None and not isinstance(compound_raw, dict):
        shape_errors.append(
            f"`compound` must be a mapping of {{group: {{label: {{member: value}}}}}}, "
            f"got {type(compound_raw).__name__}"
        )
    elif isinstance(compound_raw, dict):
        for group, rows in compound_raw.items():
            if not isinstance(rows, dict) or not rows:
                shape_errors.append(
                    f"compound group {group!r} must be a non-empty mapping of "
                    "{label: {member: value}}"
                )
                continue
            for label, row in rows.items():
                if not isinstance(row, dict) or not row:
                    shape_errors.append(
                        f"compound[{group!r}][{label!r}] must be a non-empty mapping "
                        "of {member: value}"
                    )

    # Symbolic names must be string identifiers (they become ${name} / `_env` keys,
    # matched by the `${\w+}` regex and sorted across runs in aggregation; a
    # non-identifier key — e.g. an int — never resolves and crashes the axis sort).
    if isinstance(sweep_raw, dict):
        for name, value in sweep_raw.items():
            if not _is_identifier(name):
                shape_errors.append(f"sweep dim name {name!r} must be a string identifier")
            if isinstance(value, dict):  # dict-sweep labels reach io.log_dir as a path
                for label in value:
                    if not _is_safe_label(label):
                        shape_errors.append(
                            f"sweep dim {name!r} label {label!r} must be a non-empty "
                            "path-safe string"
                        )
    if isinstance(derived_raw, dict):
        for name in derived_raw:
            if not _is_identifier(name):
                shape_errors.append(f"derived name {name!r} must be a string identifier")
    if isinstance(compound_raw, dict):
        for group, rows in compound_raw.items():
            if not _is_identifier(group):
                shape_errors.append(f"compound group name {group!r} must be a string identifier")
            if isinstance(rows, dict):
                for label, row in rows.items():
                    if not _is_safe_label(label):
                        shape_errors.append(
                            f"compound[{group!r}] label {label!r} must be a "
                            "non-empty path-safe string"
                        )
                    if isinstance(row, dict):
                        for member in row:
                            if not _is_identifier(member):
                                shape_errors.append(
                                    f"compound member name {member!r} must be a string identifier"
                                )
    if shape_errors:
        return shape_errors  # don't interpret blocks of the wrong shape

    sweep = sweep_raw or {}
    derived = derived_raw or {}
    constraints = constraints_raw or []
    compound = compound_raw or {}

    sweep_names = set(sweep)
    # compound: members fill params (like sweep dims); the group name is an axis.
    compound_groups = set(compound)
    compound_members: set[str] = set()
    errors.extend(_check_compound_semantics(compound, sweep_names, set(derived)))
    for rows in compound.values():
        for row in rows.values():
            compound_members |= set(row)

    declared = sweep_names | set(derived) | compound_groups | compound_members

    # R4: an empty sweep dim expands to zero/garbage; reject it up front.
    for name, value in sweep.items():
        if isinstance(value, (list, dict)) and len(value) == 0:
            errors.append(f"sweep dim {name!r} is empty; give it at least one value")
        elif not isinstance(value, (list, dict)):
            # a scalar sweep is degenerate but allowed; lists/dicts are the norm.
            _sweep_dim_options(value)

    # V1/V2: a derived name must not collide with a sweep dim.
    for lhs in derived:
        if lhs in sweep_names:
            errors.append(f"derived target {lhs!r} also appears as a sweep dim")

    # V3/V6: derived RHS grammar + free vars resolve in topological order.
    # Compound members + group names are available to derived/constraints too.
    resolved = sweep_names | compound_members | compound_groups
    for lhs, rhs in derived.items():
        grammar_errors = _check_expr_grammar(str(rhs))
        errors.extend(f"derived[{lhs!r}]: {e}" for e in grammar_errors)
        if grammar_errors:
            continue
        rhs_names = _expr_names(str(rhs))
        missing = rhs_names - resolved
        if missing:
            errors.append(
                f"derived[{lhs!r}] references undefined names {sorted(missing)} "
                "(must come from sweep or an earlier derived entry)"
            )
        elif not rhs_names:
            # R7: a derived value referencing no sweep / earlier-derived name is a
            # constant — it varies with nothing. New-interface analogue of the old
            # INV-10 ("derived LHS must be a schema param"): with `${name}`
            # injection a derived name is an intermediate, but it must still
            # actually derive from the sweep, else it is dead/garbage.
            errors.append(
                f"derived[{lhs!r}] = {rhs!r} is constant (derives from no sweep "
                "dimension); put the literal on the param directly, or sweep it"
            )
        resolved.add(lhs)

    # V5/V6: constraint grammar + free vars must be defined.
    for expr in constraints:
        grammar_errors = _check_expr_grammar(str(expr))
        errors.extend(f"constraint {e}" for e in grammar_errors)
        if grammar_errors:
            continue
        missing = _expr_names(str(expr)) - resolved
        if missing:
            errors.append(f"constraint {expr!r} references undefined names {sorted(missing)}")

    # The reference universe: everywhere a sweep/derived name can legitimately be
    # used — a tree placeholder, a log_dir template field, or an expression input.
    tree = _tree_of(preset)
    # The per-kernel `backends` block is a control key (stripped from `tree`), but
    # its `${name}` values are ordinary sweep participants that DO reach a run's
    # config (Rust's `RunConfig.backends`). Count them as tree placeholders so a
    # backend-only sweep dim is config-effective (R2) and its `${...}` is P-checked.
    tree_placeholders = collect_placeholders(tree) | collect_placeholders(
        preset.get("backends") or {}
    )
    log_dir_names, log_dir_errors = _parse_log_dir_template(preset)
    errors.extend(log_dir_errors)
    referenced = tree_placeholders | log_dir_names
    for rhs in derived.values():
        try:
            referenced |= _expr_names(str(rhs))
        except SyntaxError:
            pass  # grammar error already reported above
    for expr in constraints:
        try:
            referenced |= _expr_names(str(expr))
        except SyntaxError:
            pass

    # P: every ${name} (tree) and {name} (log_dir) must resolve to a declared name.
    for name in sorted(tree_placeholders - declared):
        errors.append(f"placeholder ${{{name}}} is not declared in `sweep` or `derived`")
    for name in sorted(log_dir_names - declared):
        errors.append(
            f"io.log_dir references {{{name}}}, which is not a declared sweep/derived name"
        )

    # R1: a string leaf containing `${...}` that is not a whole-leaf placeholder
    # is silently ignored by substitution — reject it (partial interpolation).
    for path in _partial_placeholder_paths(tree):
        errors.append(
            f"{path}: value contains a `${{...}}` that is not the whole value; "
            "only a whole-leaf `${name}` is substituted (no partial interpolation)"
        )

    # Config-effective names: a name that reaches an actual run config value —
    # directly as a whole-leaf `${name}` in the tree, or transitively as an input
    # to a `derived` that is itself config-effective. A backward reachability
    # closure from the tree placeholders through the derived graph. NB: `io.log_dir`
    # `{name}` and `constraints` do NOT make a name config-effective — they change
    # only the output path / plan membership, never a simulation's config.
    effective = set(tree_placeholders)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in derived.items():
            if lhs not in effective:
                continue
            try:
                inputs = _expr_names(str(rhs))
            except SyntaxError:
                continue  # grammar error already reported
            for name in inputs - effective:
                effective.add(name)
                changed = True

    # R2 (strong): every sweep dim must be config-effective. A dim referenced only
    # in a constraint, only in `io.log_dir`, or nowhere never changes any run's
    # config — it is a phantom axis (a no-op sweep, or filtering a constraint could
    # state with a literal). This also subsumes the "constraint filters to a single
    # surviving combination" bypass that `validate_distinct_configs` cannot see (it
    # needs >=2 candidates to spot a duplicate).
    for name in sorted(sweep_names - effective):
        errors.append(
            f"sweep dim {name!r} never reaches a run config value; reference it as "
            f"`${{{name}}}` on the param you mean to sweep (or via a `derived` that "
            "lands in the config). Appearing only in `io.log_dir` or a constraint is "
            "a no-op — every run would compute the same thing"
        )

    # R2 for compound members: each member is a swept column, so it too must land
    # in a config leaf (directly or via a config-effective derived) — a member that
    # never reaches config is a phantom column.
    for name in sorted(compound_members - effective):
        errors.append(
            f"compound member {name!r} never reaches a run config value; reference "
            f"it as `${{{name}}}` on a param (or via a `derived` that lands in the "
            "config), else drop it from the group"
        )

    # R3: a derived name that is never used is dead computation.
    for name in sorted(set(derived) - referenced):
        errors.append(
            f"derived {name!r} is computed but never used; reference it as "
            f"`${{{name}}}`, in `io.log_dir`, or in a constraint (or remove it)"
        )

    return errors


def validate_expanded(candidate: dict, registry: Registry) -> list[str]:
    """Concrete / post-expansion judgment — the single gate before `normalize`.
    It is the same `_walk_schema` as the raw phase but with placeholders no longer
    deferred, so the checks `validate_params` could only postpone now run for real
    against concrete values:
    - **provider tag** — an `arch.type` / `worker.type` injected by a sweep is now
      a concrete tag, checked against the contract's advertised tags (the
      `${arch_kind}` → `not_a_real_arch` bypass);
    - **tag-specific required** — with the tag concrete, `iter_slots` yields that
      tag's own params, so a missing required one (e.g. `chunked_prefill` without
      `max_batch_tokens`, tag arrived via `${worker_kind}`) is flagged;
    - **type / choices / required common blocks** — every swept/derived value is
      re-typed, and `create=True` materializes any block a preset dropped to dodge
      a required leaf;
    - **placeholder residue** — a whole-leaf `${name}` left unresolved is flagged
      by the leaf walk; a *partial* `${...}` inside a larger string is flagged
      below. Post-condition: a passing candidate carries zero `${` anywhere."""
    errors = _walk_schema(candidate, registry, defer_placeholders=False)
    if candidate.get("deployment") in registry.deployments:
        for path in _partial_placeholder_paths(_tree_of(candidate)):
            errors.append(
                f"{path}: value still contains a `${{...}}` after expansion "
                "(unresolved placeholder residue)"
            )
    return errors

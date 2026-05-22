"""Sandboxed expression evaluation for `derived` / `constraints`.

Self-contained (no schema dependency): the validator and the sweep expander
both lean on this module. Two halves that must stay in lock-step:

- a *static* gate (`_check_expr_grammar`, `_expr_names`) built on the stdlib
  `ast` module — it rejects any construct the evaluator can't run, so nothing
  passes validation then silently evaluates to None;
- the *evaluator* (`_make_evaluator`, `_eval`) built on `asteval` (sandboxed
  AST, never `eval`), lazily imported so the rest of the launcher works without
  the `asteval` dependency.
"""

from __future__ import annotations

import ast

# Functions a `derived` / `constraint` expression may call. asteval ships these
# as safe builtins; the static validator additionally gates the *call* node so
# the grammar the validator accepts matches what the evaluator can run (no
# silent divergence — the bug that motivated `_check_expr_grammar`).
EXPR_FUNCS = frozenset({"min", "max", "abs"})

# AST node types permitted in an expression. Deliberately small: arithmetic,
# comparison, boolean, unary, ternary, and allow-listed calls. Everything else
# (subscript, attribute, comprehension, lambda, list/dict literals, ...) is
# rejected so a preset can never reach a construct the evaluator drops to None.
_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.IfExp,  # ternary `a if c else b`
    ast.Call,  # validated separately against EXPR_FUNCS
)


def _expr_names(expr: str) -> set[str]:
    """Free `Name` identifiers referenced by an expression, excluding the names
    used as call targets (those are functions, checked against EXPR_FUNCS)."""
    tree = ast.parse(expr, mode="eval")
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - called


def _check_expr_grammar(expr: str) -> list[str]:
    """Reject any AST construct the evaluator can't run. Keeps the validator's
    accepted grammar identical to the evaluator's so nothing passes validation
    then silently evaluates to None (e.g. ternary under asteval `minimal`)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return [f"{expr!r} is not a valid expression: {exc}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in EXPR_FUNCS):
                target = getattr(node.func, "id", type(node.func).__name__)
                errors.append(
                    f"{expr!r}: call to {target!r} not allowed "
                    f"(only {sorted(EXPR_FUNCS)} may be called)"
                )
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                errors.append(f"{expr!r}: keyword / *args calls are not allowed")
        elif not isinstance(node, _ALLOWED_EXPR_NODES):
            errors.append(
                f"{expr!r}: unsupported expression construct "
                f"{type(node).__name__} (allowed: arithmetic, comparison, "
                f"boolean, ternary, and {sorted(EXPR_FUNCS)})"
            )
    return errors


def _make_evaluator():
    """A fresh sandboxed asteval interpreter. Lazily imported so the rest of the
    launcher (loader / metadata) works without the `asteval` dependency. NOT
    `minimal=True`: that mode drops `IfExp` (ternary) and the math builtins we
    allow; the static validator (`_check_expr_grammar`) is what restricts the
    surface, so the evaluator only needs to be a superset of the allowed grammar.
    """
    from asteval import Interpreter

    return Interpreter(use_numpy=False, no_print=True)


def _eval(interp, expr: str):
    """Evaluate one expression and surface asteval errors instead of letting a
    failed eval silently return None (asteval records errors on `interp.error`
    rather than raising)."""
    interp.error = []
    value = interp(str(expr))
    if interp.error:
        msgs = "; ".join(e.get_error()[1] for e in interp.error)
        raise ValueError(f"failed to evaluate {expr!r}: {msgs}")
    return value

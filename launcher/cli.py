"""The launcher's command surface: global options and subcommand dispatch.

A launcher command line is ``[global options] [subcommand] args``, where the
global options may appear anywhere on the line and hold for every mode:

    --no-gpu              guarantee no GPU is used (see profiling/gpu_policy.py)
    --remote NAME         run this command on the remote NAME from remote.yaml
    --remote-set K=V      override one remote.yaml field of NAME (repeatable)

They are taken off the line before dispatch, so a subcommand's own parser never
sees them, and `--remote` hands the remaining, unchanged command to
``launcher.remote``. The first token selects a subcommand; anything else is a
simulation preset (the default mode, ``python -m launcher PRESET.yaml``).

Each subcommand keeps its own parser and owner module: `kernel-profile` stays in
``profiling.cli`` because ``python -m profiling`` is the same command. The table
below only names them, which keeps help text and remote support in one place.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass


class UsageError(ValueError):
    """A malformed global option; the message is shown to the user as is."""


@dataclass(frozen=True)
class GlobalOptions:
    no_gpu: bool = False
    remote: str | None = None
    remote_set: tuple[str, ...] = ()


GLOBAL_OPTIONS_HELP = (
    "Global options (anywhere on the line, every mode):\n"
    "  --no-gpu              guarantee no GPU is used (also SERVINGSTUDIO_NO_GPU=1); a\n"
    "                        missing profile.db row fails instead of being profiled\n"
    "  --remote NAME         run the command on remote NAME from remote.yaml and copy\n"
    "                        the results back (simulation, timing-predict,\n"
    "                        kernel-profile run|measure)\n"
    "  --remote-set K=V      override one field of that remote for this call\n"
    "                        (repeatable; the value is parsed as JSON when possible)"
)


def take_global_options(argv: list[str]) -> tuple[GlobalOptions, list[str]]:
    """Split the global options out of ``argv``; the rest keeps its order.

    Both ``--remote NAME`` and ``--remote=NAME`` spellings are accepted, the
    same for ``--remote-set``.
    """

    no_gpu = False
    remote: str | None = None
    remote_set: list[str] = []
    rest: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        if token == "--no-gpu":
            no_gpu = True
            continue
        name, has_inline_value, inline_value = token.partition("=")
        if name not in ("--remote", "--remote-set"):
            rest.append(token)
            continue
        value = inline_value if has_inline_value else next(tokens, None)
        if not value or value.startswith("--"):
            raise UsageError(f"{name} needs a value")
        if name == "--remote":
            if remote is not None and remote != value:
                raise UsageError(f"--remote given twice ({remote!r} and {value!r})")
            remote = value
        else:
            if "=" not in value:
                raise UsageError(f"bad --remote-set {value!r}; expected key=value")
            remote_set.append(value)
    if remote_set and remote is None:
        raise UsageError("--remote-set needs --remote NAME")
    return GlobalOptions(no_gpu, remote, tuple(remote_set)), rest


def _set_path(tree: dict, dotted: str, value) -> None:
    """Set `value` at a dotted path into the config tree, e.g.
    `io.log_dir`, `pools.main.groups.0.arch.tp_size`. Integer segments index
    into lists; everything else into dicts (auto-vivifying missing dicts)."""
    parts = dotted.split(".")
    node = tree
    for depth, part in enumerate(parts[:-1]):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node.setdefault(part, {})
        if not isinstance(node, dict | list):
            prefix = ".".join(parts[: depth + 1])
            raise UsageError(f"{dotted}: {prefix} is {node!r}, not a mapping or a list")
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def apply_dotted(tree: dict, assignments: list[str]) -> dict:
    """A copy of ``tree`` with each ``path=value`` assignment applied.

    The path is dotted (`io.log_dir`, `pools.main.groups.0.arch.tp_size`);
    values are parsed as JSON when possible (`tp_size=4`→int, `fp8=true`→bool,
    `[1,2]`→list) and otherwise kept as a bare string (`model_config=model/x.json`).
    `--override` (presets) and `--remote-set` (remote.yaml) share this form.
    """
    out = copy.deepcopy(tree)
    for item in assignments:
        if "=" not in item:
            raise UsageError(f"{item!r}; expected path=value")
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw  # bare string
        _set_path(out, key.strip(), value)
    return out


@dataclass(frozen=True)
class Subcommand:
    usage: str
    run: Callable[[list[str]], int]
    # Whether `--remote` can run this subcommand (see launcher/remote).
    remote: bool = False


def _alignment(argv: list[str]) -> int:
    from .alignment import main

    return main(argv)


def _timing_predict(argv: list[str]) -> int:
    from . import timing_predict

    return timing_predict.main(argv)


def _kernel_profile(argv: list[str]) -> int:
    from profiling.cli import main

    return main(argv, prog="python -m launcher kernel-profile")


def _alignment_campaign(argv: list[str]) -> int:
    from .alignment_campaign import main

    return main(argv)


def _migrate_artifact_kinds(argv: list[str]) -> int:
    from .migrate_artifact_kinds import main

    return main(argv)


def _list_params(argv: list[str]) -> int:
    from .__main__ import list_params

    return list_params(argv)


def _mstar_experiment(argv: list[str]) -> int:
    from .mstar_experiment.cli import main

    return main(argv)


SUBCOMMANDS: dict[str, Subcommand] = {
    # The offline per-building-block timing predictor. Its config is a minimal
    # arch+cases file, not a deployment RunConfig, so it bypasses sweeps.
    "timing-predict": Subcommand(
        "timing-predict CONFIG.yaml|json [...]", _timing_predict, remote=True
    ),
    # L1 profiling stays owned by profiling.cli; this is the same command.
    "kernel-profile": Subcommand(
        "kernel-profile {list,query,count-missing,run,measure,merge-db,audit-provenance} ...",
        _kernel_profile,
        remote=True,
    ),
    # Explicit workflow stages. Its `sim` stage re-enters the launcher's main.
    "alignment": Subcommand("alignment {sim,profile,timing-predict,analyze} ...", _alignment),
    # Drives `alignment` stages across case directories; top-level rather than
    # nested because `alignment`'s parser owns "one explicit stage".
    "alignment-campaign": Subcommand(
        "alignment-campaign {check,render,run,label,extract,compare} ...", _alignment_campaign
    ),
    "migrate-artifact-kinds": Subcommand(
        "migrate-artifact-kinds (--check|--apply) [--registry PATH]", _migrate_artifact_kinds
    ),
    # POC: shared M*-Sim experiment manifests via external lab (MSTAR_EXPERIMENT_LAB).
    "mstar-experiment": Subcommand(
        "mstar-experiment {run-experiment,run-campaign,evaluate} ...", _mstar_experiment
    ),
    "list-params": Subcommand("list-params [--human] [--build-type PROFILE]", _list_params),
}


def modes_help() -> str:
    lines = ["Other modes:"]
    lines += [f"  python -m launcher {command.usage}" for command in SUBCOMMANDS.values()]
    return "\n".join(lines) + "\n\n" + GLOBAL_OPTIONS_HELP


def dispatch(
    argv: list[str],
    *,
    simulation: Callable[[list[str]], int],
) -> int:
    """Apply the global options, then run the selected mode."""

    from profiling.gpu_policy import disable_gpus, gpu_disabled

    try:
        options, argv = take_global_options(argv)
    except UsageError as exc:
        print(f"python -m launcher: {exc}", file=sys.stderr)
        return 2

    # The environment carries the no-GPU guarantee to every child. The variable
    # alone gets the same hidden devices, so a child cannot see a GPU either way.
    if options.no_gpu or gpu_disabled():
        disable_gpus()

    name = argv[0] if argv and argv[0] in SUBCOMMANDS else None
    rest = argv[1:] if name is not None else argv

    if options.remote is not None:
        if name is not None and not SUBCOMMANDS[name].remote:
            print(f"python -m launcher: --remote does not support {name}", file=sys.stderr)
            return 2
        from .remote import run_remote

        return run_remote(name, rest, options)

    if name is not None:
        return SUBCOMMANDS[name].run(rest)
    return simulation(rest)

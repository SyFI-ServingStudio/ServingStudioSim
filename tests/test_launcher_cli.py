"""The launcher command surface: global options and subcommand dispatch."""

from __future__ import annotations

import pytest

from launcher import __main__ as launcher_main
from launcher import cli
from launcher.cli import GlobalOptions, UsageError, take_global_options
from profiling.gpu_policy import NO_GPU_ENV


@pytest.mark.parametrize(
    ("argv", "options", "rest"),
    [
        (["p.yaml"], GlobalOptions(), ["p.yaml"]),
        (["p.yaml", "--no-gpu"], GlobalOptions(no_gpu=True), ["p.yaml"]),
        (
            ["--remote", "big", "timing-predict", "c.json", "--no-gpu"],
            GlobalOptions(no_gpu=True, remote="big"),
            ["timing-predict", "c.json"],
        ),
        (
            ["p.yaml", "--remote=big", "--remote-set", "slurm.time=08:00:00", "--dry-run"],
            GlobalOptions(remote="big", remote_set=("slurm.time=08:00:00",)),
            ["p.yaml", "--dry-run"],
        ),
        (
            ["--remote-set=a=1", "--remote", "big", "--remote-set", "b.c=x"],
            GlobalOptions(remote="big", remote_set=("a=1", "b.c=x")),
            [],
        ),
    ],
)
def test_global_options_come_off_anywhere_on_the_line(argv, options, rest):
    assert take_global_options(argv) == (options, rest)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--remote"], "--remote needs a value"),
        (["--remote", "--no-gpu"], "--remote needs a value"),
        (["--remote", "a", "--remote", "b"], "--remote given twice"),
        (["--remote", "a", "--remote-set", "novalue"], "expected key=value"),
        (["--remote-set", "a=1"], "--remote-set needs --remote NAME"),
    ],
)
def test_malformed_global_options_are_usage_errors(argv, message):
    with pytest.raises(UsageError, match=message):
        take_global_options(argv)


def test_a_malformed_global_option_exits_2(capsys):
    assert launcher_main.main(["--remote"]) == 2
    assert "--remote needs a value" in capsys.readouterr().err


def _clear_no_gpu(monkeypatch):
    # disable_gpus writes os.environ directly; register both names with
    # monkeypatch first so they are restored after the test.
    monkeypatch.setenv(NO_GPU_ENV, "0")
    monkeypatch.delenv(NO_GPU_ENV)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def test_a_subcommand_gets_its_arguments_without_the_global_options(monkeypatch):
    _clear_no_gpu(monkeypatch)
    seen = []
    monkeypatch.setitem(
        cli.SUBCOMMANDS,
        "kernel-profile",
        cli.Subcommand("kernel-profile ...", lambda argv: seen.append(argv) or 0, remote=True),
    )

    assert launcher_main.main(["kernel-profile", "--no-gpu", "list", "--json"]) == 0

    assert seen == [["list", "--json"]]


def test_a_preset_path_selects_the_simulation_mode(monkeypatch):
    _clear_no_gpu(monkeypatch)
    seen = []
    monkeypatch.setattr(launcher_main, "run_simulation", lambda argv: seen.append(argv) or 0)

    assert launcher_main.main(["p.yaml", "--dry-run", "--no-gpu"]) == 0

    assert seen == [["p.yaml", "--dry-run"]]


def test_remote_hands_the_unchanged_command_to_the_remote_runner(monkeypatch):
    _clear_no_gpu(monkeypatch)
    import launcher.remote as remote

    seen = []
    monkeypatch.setattr(
        remote, "run_remote", lambda name, argv, options: seen.append((name, argv, options)) or 0
    )

    assert launcher_main.main(["timing-predict", "c.json", "--remote", "big"]) == 0
    assert launcher_main.main(["p.yaml", "--remote", "big", "--no-plot"]) == 0

    assert seen == [
        ("timing-predict", ["c.json"], GlobalOptions(remote="big")),
        (None, ["p.yaml", "--no-plot"], GlobalOptions(remote="big")),
    ]


def test_remote_refuses_a_subcommand_it_cannot_run(capsys):
    assert launcher_main.main(["alignment", "sim", "x.yaml", "--remote", "big"]) == 2
    assert "--remote does not support alignment" in capsys.readouterr().err


def test_timing_predict_options_parse_as_before():
    from launcher.timing_predict import _build_argparse

    args = _build_argparse().parse_intermixed_args(
        ["a.json", "--build-type", "dev", "--analyzer-essential-only", "b.json", "--no-plot"]
    )
    assert (args.configs, args.build_type, args.analysis, args.no_plot, args.dry_run) == (
        ["a.json", "b.json"],
        "dev",
        "essential",
        True,
        False,
    )
    assert _build_argparse().parse_intermixed_args([]).analysis == "full"
    with pytest.raises(SystemExit):
        _build_argparse().parse_intermixed_args(["--no-analyze", "--analyzer-essential-only"])
    with pytest.raises(SystemExit):
        _build_argparse().parse_intermixed_args(["--unknown"])

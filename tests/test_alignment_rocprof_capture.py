"""CPU test for the rocprofv3 capture driver (``alignment/profiler/rocprof_capture``).

Mirrors the nsys-capture contract tests. It proves, without rocprofv3 or a GPU:

1. executable resolution refuses a non-absolute path and records the version;
2. the assembled ``rocprofv3 ... -- <server argv>`` carries the mandatory
   ``--kernel-trace`` + ``--marker-trace`` + ``--output-format rocpd`` and the
   command after a literal ``--``;
3. ``validate_rocpd`` recognizes a complete capture and flags the two failure
   modes (no iteration roctx ranges / no dispatch rows);
4. the ``rocpd-capture`` CLI runs the whole capture -> rocpd -> parse path: with
   the rocprofv3 binary and the GPU subprocess stood in for (the fake runner
   writes a synthetic rocpd), the real ``alignment/rocpd`` producer emits
   ``parsed.json`` + ``parsed.kernels.parquet`` + ``kernel_sequences.json``.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from alignment.nsys.parsed_io import _KERNEL_SCHEMA, kernel_rows_path
from alignment.profiler import rocprof_capture
from alignment.profiler.config import RocprofConfig
from alignment.profiler.rocprof_capture import (
    OUTPUT_RANK_ENV_DEFAULT,
    VLLM_V1_MULTIPROCESSING_ENV,
    ResolvedRocprofExecutable,
    build_capture_argv,
    build_capture_server_env,
    locate_rocpd,
    locate_rocpd_per_rank,
    per_rank_output_name,
    rank_from_rocpd_path,
    resolve_rocprof_executable,
    validate_rocpd,
)
from alignment.profiler.roctx_shim import ROCTX_SCOPES_ENV

_GUID = "0000b1c7_c35b_735b_96a7_f0a02ff013cc"
_PID = 4242
_AGENT = 1


def _write_synthetic_rocpd(path, dispatches, regions):
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE rocpd_string_{_GUID} (id INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
        "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start BIGINT, end BIGINT, region_name_id INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_region_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, start BIGINT, end BIGINT, name_id INTEGER)"
    )
    strings: dict[str, int] = {}

    def intern(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            conn.execute(
                f"INSERT INTO rocpd_string_{_GUID} (id, string) VALUES (?, ?)",
                (strings[text], text),
            )
        return strings[text]

    symbols: dict[str, int] = {}
    for _s, _e, name, _stream in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name, stream) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, region_name_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, 55, _AGENT, symbols[name], i, 0, stream, start, end, 0),
        )
    for i, (start, end, text) in enumerate(regions, start=1):
        conn.execute(
            f"INSERT INTO rocpd_region_{_GUID} (id, pid, tid, start, end, name_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, _PID, 55, start, end, intern(text)),
        )
    conn.commit()
    conn.close()


def _complete_fixture(path):
    _write_synthetic_rocpd(
        path,
        dispatches=[
            (500, 900, "warmup_fill_kernel", 0),
            (1_100, 1_300, "rms_norm_kernel", 0),
            (1_400, 2_000, "hipblaslt_gemm_f16", 0),
            (2_100, 2_900, "flash_fwd_attn_kernel", 0),
            (3_000, 3_200, "silu_and_mul_kernel", 0),
            (3_300, 3_900, "hipblaslt_gemm_f16", 0),
            (4_000, 4_200, "vectorized_elementwise_kernel", 0),
        ],
        regions=[(1_000, 5_000, "vllm_iteration(0): forward")],
    )


def _fake_exe(tmp_path):
    return ResolvedRocprofExecutable(path=tmp_path / "rocprofv3", version="rocprofv3 1.0.0")


def test_resolve_requires_absolute_path(monkeypatch):
    monkeypatch.delenv("ROCPROF_BIN", raising=False)
    with pytest.raises(ValueError, match="not configured"):
        resolve_rocprof_executable(None)
    with pytest.raises(ValueError, match="must be an absolute path"):
        resolve_rocprof_executable("rocprofv3")


def test_resolve_records_version(tmp_path, monkeypatch):
    exe = tmp_path / "rocprofv3"
    exe.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then printf "rocprofv3 1.2.3\\n"; else exit 2; fi\n'
    )
    exe.chmod(0o755)
    monkeypatch.setenv("ROCPROF_BIN", str(exe))
    resolved = resolve_rocprof_executable(None)
    assert resolved.provenance() == {"executable": str(exe.resolve()), "version": "rocprofv3 1.2.3"}


def test_capture_argv_carries_mandatory_flags_and_command(tmp_path):
    argv = build_capture_argv(
        _fake_exe(tmp_path),
        RocprofConfig(),
        ["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"],
        tmp_path / "out",
        "rank0",
    )
    assert argv[0] == str(tmp_path / "rocprofv3")
    assert "--kernel-trace" in argv
    assert "--marker-trace" in argv  # roctx iteration ranges
    assert argv[argv.index("--output-format") + 1] == "rocpd"
    assert argv[argv.index("-d") + 1] == str(tmp_path / "out")
    assert argv[argv.index("-o") + 1] == "rank0"
    cut = argv.index("--")
    assert argv[cut + 1 :] == ["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"]


def test_config_requires_traces_and_rocpd_format():
    with pytest.raises(ValueError, match="kernel_trace is required"):
        RocprofConfig(kernel_trace=False).validate()
    with pytest.raises(ValueError, match="marker_trace is required"):
        RocprofConfig(marker_trace=False).validate()
    with pytest.raises(ValueError, match="output_format"):
        RocprofConfig(output_format="csv").validate()


def test_validate_rocpd_ok_and_failure_modes(tmp_path):
    good = tmp_path / "good.db"
    _complete_fixture(good)
    report = validate_rocpd(good)
    assert report["ok"] is True
    assert report["iteration_ranges"] == 1
    assert report["kernel_rows"] == 7

    no_markers = tmp_path / "no_markers.db"
    _write_synthetic_rocpd(no_markers, [(100, 200, "rms_norm_kernel", 0)], regions=[])
    assert validate_rocpd(no_markers)["ok"] is False

    no_kernels = tmp_path / "no_kernels.db"
    _write_synthetic_rocpd(no_kernels, [], regions=[(1_000, 5_000, "vllm_iteration(0): forward")])
    report = validate_rocpd(no_kernels)
    assert report["ok"] is False
    assert report["kernel_rows"] == 0


def test_locate_rocpd_accepts_results_and_plain_suffix(tmp_path):
    (tmp_path / "a_results.db").write_text("")
    assert locate_rocpd(tmp_path, "a").name == "a_results.db"
    (tmp_path / "b.db").write_text("")
    assert locate_rocpd(tmp_path, "b").name == "b.db"
    with pytest.raises(FileNotFoundError):
        locate_rocpd(tmp_path, "missing")


def test_cli_capture_then_parse_end_to_end(tmp_path):
    """The rocpd-capture CLI drives capture -> rocpd -> parse, GPU subprocess mocked."""
    out_dir = tmp_path / "cap"
    parsed_out = tmp_path / "parsed.json"
    sequences_out = tmp_path / "kernel_sequences.json"

    captured_argv = {}

    def fake_runner(argv, env=None, check=True):
        # Stand in for the real `rocprofv3 ... -- vllm serve ...` GPU subprocess:
        # write the synthetic rocpd where the driver told rocprofv3 to put it.
        captured_argv["argv"] = argv
        db_dir = argv[argv.index("-d") + 1]
        name = argv[argv.index("-o") + 1]
        from pathlib import Path

        Path(db_dir).mkdir(parents=True, exist_ok=True)
        _complete_fixture(Path(db_dir) / f"{name}_results.db")
        return subprocess.CompletedProcess(argv, 0)

    rc = rocprof_capture.main(
        [
            "--output-dir",
            str(out_dir),
            "--output-name",
            "rank0",
            "--parsed-output",
            str(parsed_out),
            "--sequences-output",
            str(sequences_out),
            "--",
            "python",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "model",
        ],
        resolver=lambda _p: _fake_exe(out_dir),
        runner=fake_runner,
    )
    assert rc == 0
    # The GPU subprocess really was invoked as a rocprofv3 capture of the server.
    assert "--kernel-trace" in captured_argv["argv"]
    assert captured_argv["argv"][captured_argv["argv"].index("--") + 1] == "python"

    # The real producer emitted the normalized Check-1 files.
    assert parsed_out.exists()
    rows_path = kernel_rows_path(parsed_out)
    assert rows_path.exists()
    table = pq.read_table(rows_path)
    assert table.schema.names == _KERNEL_SCHEMA.names
    for name in _KERNEL_SCHEMA.names:
        assert table.column(name).null_count == 0
    assert sequences_out.exists()


def test_capture_server_env_sets_roctx_flag():
    """The driver turns the roctx annotation on, preserving the rest of the env."""
    built = build_capture_server_env({"PATH": "/usr/bin", "HOME": "/home/x"})
    assert built[ROCTX_SCOPES_ENV] == "1"
    assert built["PATH"] == "/usr/bin" and built["HOME"] == "/home/x"
    # A pre-existing (stale) value is overridden to the timing-on setting.
    assert build_capture_server_env({ROCTX_SCOPES_ENV: "0"})[ROCTX_SCOPES_ENV] == "1"


def test_capture_server_env_defaults_v1_multiprocessing_off_but_overridable():
    """The driver puts the V1 engine in-process by default, overridably.

    vLLM runs EngineCore in a separate process by default, so a capture that wraps
    only the launcher sees no kernel dispatches. The driver defaults
    VLLM_ENABLE_V1_MULTIPROCESSING=0 to pull the engine in-process -- but as a
    DEFAULT, so an operator tracing the multiprocessing engine per-rank keeps their
    own value.
    """
    built = build_capture_server_env({"PATH": "/usr/bin"})
    assert built[VLLM_V1_MULTIPROCESSING_ENV] == "0"
    # An explicit operator value is preserved (overridable default).
    kept = build_capture_server_env({VLLM_V1_MULTIPROCESSING_ENV: "1"})
    assert kept[VLLM_V1_MULTIPROCESSING_ENV] == "1"


def test_per_rank_output_name_and_rank_parsing():
    """The per-rank -o template carries a rocprofv3 %q{ENV}% key; filenames round-trip."""
    templated = per_rank_output_name("trace")
    assert templated == f"trace_rank%q{{{OUTPUT_RANK_ENV_DEFAULT}}}%"
    assert per_rank_output_name("trace", "LOCAL_RANK") == "trace_rank%q{LOCAL_RANK}%"
    # Ranks are recovered from both the current and legacy rocpd spellings.
    from pathlib import Path

    assert rank_from_rocpd_path(Path("trace_rank0_results.db")) == 0
    assert rank_from_rocpd_path(Path("trace_rank3.db")) == 3
    assert rank_from_rocpd_path(Path("trace_results.db")) is None


def test_per_rank_argv_carries_templated_output_name(tmp_path):
    """build_capture_argv emits the templated -o so each worker writes its own db."""
    from alignment.profiler.config import RocprofConfig

    argv = build_capture_argv(
        _fake_exe(tmp_path),
        RocprofConfig(tp_size=4),
        ["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"],
        tmp_path / "out",
        per_rank_output_name("rank"),
    )
    assert argv[argv.index("-o") + 1] == f"rank_rank%q{{{OUTPUT_RANK_ENV_DEFAULT}}}%"


def test_locate_rocpd_per_rank_collects_in_rank_order(tmp_path):
    for r in (2, 0, 1):  # written out of order on purpose
        (tmp_path / f"trace_rank{r}_results.db").write_text("")
    found = locate_rocpd_per_rank(tmp_path, "trace")
    assert [p.name for p in found] == [
        "trace_rank0_results.db",
        "trace_rank1_results.db",
        "trace_rank2_results.db",
    ]
    with pytest.raises(FileNotFoundError):
        locate_rocpd_per_rank(tmp_path, "missing")


def test_config_rejects_bad_tp_size():
    with pytest.raises(ValueError, match="tp_size"):
        RocprofConfig(tp_size=0).validate()


def test_cli_per_rank_capture_parses_every_worker(tmp_path):
    """--tp-size>1 drives a per-rank capture: each worker db is parsed separately."""
    out_dir = tmp_path / "cap"
    parsed_out = tmp_path / "parsed.json"
    sequences_out = tmp_path / "kernel_sequences.json"
    seen = {}

    def fake_runner(argv, env=None, check=True):
        seen["argv"] = argv
        seen["env"] = env
        db_dir = Path(argv[argv.index("-d") + 1])
        name = argv[argv.index("-o") + 1]  # templated: "<base>_rank%q{RANK}%"
        base = name.split("_rank")[0]
        db_dir.mkdir(parents=True, exist_ok=True)
        # Stand in for rocprofv3 expanding %q{RANK}% in each of two worker procs.
        for rank in (0, 1):
            _complete_fixture(db_dir / f"{base}_rank{rank}_results.db")
        return subprocess.CompletedProcess(argv, 0)

    rc = rocprof_capture.main(
        [
            "--output-dir",
            str(out_dir),
            "--output-name",
            "trace",
            "--tp-size",
            "2",
            "--parsed-output",
            str(parsed_out),
            "--sequences-output",
            str(sequences_out),
            "--",
            "python",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "model",
        ],
        resolver=lambda _p: _fake_exe(out_dir),
        runner=fake_runner,
    )
    assert rc == 0
    # The -o carried the per-rank template.
    assert "_rank%q{" in seen["argv"][seen["argv"].index("-o") + 1]
    # Each worker's parse landed in its own rank-suffixed file.
    for rank in (0, 1):
        rank_parsed = tmp_path / f"parsed.rank{rank}.json"
        assert rank_parsed.exists()
        rows = kernel_rows_path(rank_parsed)
        assert rows.exists()
        table = pq.read_table(rows)
        for col in _KERNEL_SCHEMA.names:
            assert table.column(col).null_count == 0
        assert (tmp_path / f"kernel_sequences.rank{rank}.json").exists()


def test_cli_injects_roctx_flag_into_launched_server_env(tmp_path):
    """The rocpd-capture CLI passes the roctx-on env to the server it launches.

    Without this the capture would run the real `install_vllm_roctx_shim` plugin
    but leave it no-opping, recording kernels with no iteration ranges to own them.
    """
    seen = {}

    def fake_runner(argv, env=None, check=True):
        seen["env"] = env
        db_dir = argv[argv.index("-d") + 1]
        name = argv[argv.index("-o") + 1]
        from pathlib import Path

        Path(db_dir).mkdir(parents=True, exist_ok=True)
        _complete_fixture(Path(db_dir) / f"{name}_results.db")
        return subprocess.CompletedProcess(argv, 0)

    rc = rocprof_capture.main(
        [
            "--output-dir",
            str(tmp_path / "cap"),
            "--output-name",
            "rank0",
            "--no-parse",
            "--",
            "python",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "model",
        ],
        resolver=lambda _p: _fake_exe(tmp_path),
        runner=fake_runner,
    )
    assert rc == 0
    assert seen["env"] is not None
    assert seen["env"][ROCTX_SCOPES_ENV] == "1"


def test_cli_requires_double_dash():
    with pytest.raises(SystemExit):
        rocprof_capture.main(["--output-dir", "x", "--output-name", "n"])

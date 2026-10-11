"""`remote.yaml`: lookup, validation, and `--remote-set` overrides."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from launcher.remote import config
from launcher.remote.config import CONFIG_ENV, RemoteConfigError, Slurm, Ssh, load_remote

DOCUMENT = """
remotes:
  box:
    access: {ssh: me@box, port: 2222, identity_file: ~/.ssh/vibesim}
    workdir: /data/vibesim
    env: {HF_HOME: /data/hf, THREADS: 8}
  cluster:
    access: local
    launch: {slurm: {time: "04:00:00", partition: cpu, cpus_per_task: 32}}
    files: shared
    workdir: /shared/vibesim
"""


@pytest.fixture
def remote_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "remote.yaml"
    path.write_text(DOCUMENT)
    return path


def test_a_remote_parses_with_defaults(remote_yaml):
    box = load_remote("box", path=remote_yaml)
    assert box.ssh == Ssh("me@box", 2222, str(Path("~/.ssh/vibesim").expanduser()))
    assert (box.slurm, box.files, box.workdir) == (None, "rsync", PurePosixPath("/data/vibesim"))
    assert box.env == {"HF_HOME": "/data/hf", "THREADS": "8"}
    assert (box.keep_versions, box.profile_runtime, box.ephemeral) == (10, "docker", False)
    assert box.config_dir == remote_yaml.parent.resolve()

    cluster = load_remote("cluster", path=remote_yaml)
    assert cluster.ssh is None
    assert cluster.slurm == Slurm(time="04:00:00", partition="cpu", cpus_per_task=32)
    assert cluster.files == "shared"


def test_remote_set_overrides_one_field_for_this_call(remote_yaml):
    cluster = load_remote(
        "cluster",
        ("launch.slurm.time=08:00:00", "keep_versions=3", "env.TOKEN=x"),
        path=remote_yaml,
    )
    assert cluster.slurm.time == "08:00:00"
    assert (cluster.keep_versions, cluster.env) == (3, {"TOKEN": "x"})
    # The file itself is untouched.
    assert load_remote("cluster", path=remote_yaml).slurm.time == "04:00:00"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("workdir=relative/dir", "workdir must be an absolute path"),
        ("files=nfs", "files must be rsync or shared"),
        ("launch.slurm.time=", "launch.slurm.time is required"),
        ("launch.slurm.gpus=1", "unknown field.*launch.slurm.gpus"),
        ("lanch=direct", "unknown field.*lanch"),
        ("access=ssh", "access must be `local`"),
        ("profile_runtime=podman", "profile_runtime must be"),
    ],
)
def test_a_malformed_remote_names_the_field(remote_yaml, override, message):
    with pytest.raises(RemoteConfigError, match=message):
        load_remote("cluster", (override,), path=remote_yaml)


def test_an_unknown_remote_lists_the_known_ones(remote_yaml):
    with pytest.raises(RemoteConfigError, match=r"known: box, cluster"):
        load_remote("nope", path=remote_yaml)


def test_lookup_prefers_the_environment_then_this_checkout_then_the_main_one(
    tmp_path, monkeypatch, remote_yaml
):
    monkeypatch.setenv(CONFIG_ENV, str(remote_yaml))
    assert config.find_config(tmp_path / "elsewhere") == remote_yaml
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "missing.yaml"))
    with pytest.raises(RemoteConfigError, match="no such file"):
        config.find_config(tmp_path)

    monkeypatch.delenv(CONFIG_ENV)
    worktree, main = tmp_path / "wt", tmp_path / "main"
    worktree.mkdir()
    main.mkdir()
    monkeypatch.setattr(config, "_main_worktree_root", lambda root: main)
    with pytest.raises(RemoteConfigError, match="copy remote.example.yaml"):
        config.find_config(worktree)
    (main / "remote.yaml").write_text(DOCUMENT)
    assert config.find_config(worktree) == main / "remote.yaml"
    (worktree / "remote.yaml").write_text(DOCUMENT)
    assert config.find_config(worktree) == worktree / "remote.yaml"

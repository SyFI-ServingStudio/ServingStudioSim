"""The launcher's lock root must not be a directory one user can lock out.

``TMPDIR`` is shared by every account on the machine, so any directory the
launcher creates there is created under exactly one user's umask.  These tests
pin the property that keeps a second user working: the uid separates the two
roots at the *first* component below ``TMPDIR``, so neither user ever has to
mkdir inside a directory the other one owns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from launcher.process.leases import LauncherLeases, launcher_lock_root


def test_profile_database_override_has_its_own_lease(tmp_path, monkeypatch):
    leases = LauncherLeases(repository_root=tmp_path / "checkout")
    monkeypatch.delenv("VIBESIM_PROFILE_DB", raising=False)
    default = leases.profile_database(write=True)
    monkeypatch.setenv("VIBESIM_PROFILE_DB", str(tmp_path / "snapshot.db"))
    snapshot = leases.profile_database(write=True)
    assert snapshot.lock_path != default.lock_path
    assert snapshot.resource == f"profile-db:{tmp_path / 'snapshot.db'}"
    assert leases.profile_database(write=False).lock_path == snapshot.lock_path


def test_two_users_share_no_lock_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    checkout = tmp_path / "checkout"

    monkeypatch.setattr(os, "getuid", lambda: 1000)
    mine = launcher_lock_root(checkout)
    monkeypatch.setattr(os, "getuid", lambda: 2000)
    theirs = launcher_lock_root(checkout)

    assert mine != theirs
    # Distinct leaves are not enough. The uid has to split them at the topmost
    # component, because a shared parent is precisely what the first user's
    # umask makes unwritable for the second.
    assert mine.relative_to(tmp_path).parts[0] != theirs.relative_to(tmp_path).parts[0]


def test_one_user_still_gets_one_root_per_checkout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    first = launcher_lock_root(tmp_path / "a")
    again = launcher_lock_root(tmp_path / "a")
    other = launcher_lock_root(tmp_path / "b")

    assert first == again, "the root is the rendezvous point; it must be stable"
    assert first != other, "two checkouts must not contend for one lock"


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores directory permissions")
def test_a_lease_survives_an_unwritable_directory_owned_by_someone_else(
    tmp_path: Path, monkeypatch
) -> None:
    """Reproduce the multi-user failure with one uid and a mode bit.

    A directory named ``vibesim-launcher-locks`` that this process cannot write
    to stands in for the same directory owned by another account.  Acquiring a
    lease must not touch it.
    """

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    foreign = tmp_path / "vibesim-launcher-locks"
    foreign.mkdir()
    foreign.chmod(0o555)

    leases = LauncherLeases(repository_root=tmp_path / "checkout")
    with leases.build("release") as lease:
        assert lease.lock_path.is_file()
        assert foreign not in lease.lock_path.parents

    assert not any(foreign.iterdir())

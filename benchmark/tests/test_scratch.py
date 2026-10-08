import os
from pathlib import Path

import pytest

from robustness_lab.scratch import (
    SCRATCH_PREFIX,
    cleanup_stale_isaac_scratch,
    create_isaac_scratch,
    release_isaac_scratch,
)


def test_owned_scratch_is_removed_on_release(tmp_path):
    scratch = Path(create_isaac_scratch(str(tmp_path), "batch-7", owner_pid=os.getpid()))
    (scratch / "cache.bin").write_bytes(b"cache")
    result = release_isaac_scratch(str(scratch), str(tmp_path))
    assert result["status"] == "removed"
    assert result["removed_bytes"] >= 5
    assert not scratch.exists()


def test_stale_cleanup_removes_dead_owner_but_not_active_or_preserved(tmp_path):
    dead = Path(create_isaac_scratch(str(tmp_path), "dead", owner_pid=99_999_999))
    active = Path(create_isaac_scratch(str(tmp_path), "active", owner_pid=os.getpid()))
    preserved = Path(create_isaac_scratch(str(tmp_path), "debug", owner_pid=99_999_999, preserve=True))
    result = cleanup_stale_isaac_scratch(str(tmp_path), stale_after_s=0)
    assert str(dead) in result["removed"]
    assert str(active) in result["active"]
    assert str(preserved) in result["preserved"]
    assert not dead.exists() and active.exists() and preserved.exists()


def test_legacy_cleanup_requires_explicit_opt_in(tmp_path):
    legacy = tmp_path / (SCRATCH_PREFIX + "legacy")
    legacy.mkdir()
    (legacy / "cache.bin").write_bytes(b"old")
    first = cleanup_stale_isaac_scratch(str(tmp_path), stale_after_s=0)
    assert legacy.exists() and str(legacy) in first["legacy"]
    second = cleanup_stale_isaac_scratch(str(tmp_path), stale_after_s=0, include_legacy=True)
    assert not legacy.exists() and str(legacy) in second["removed"]


def test_release_refuses_broad_or_unowned_targets(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        release_isaac_scratch(str(tmp_path), str(tmp_path))
    unowned = tmp_path / (SCRATCH_PREFIX + "unowned")
    unowned.mkdir()
    with pytest.raises(ValueError, match="unowned"):
        release_isaac_scratch(str(unowned), str(tmp_path))

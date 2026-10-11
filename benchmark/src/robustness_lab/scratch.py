"""Ownership-aware lifecycle for disposable Isaac Sim portable roots."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


SCRATCH_PREFIX = "robustlab-isaacsim-"
MARKER_NAME = ".robustlab-scratch.json"
SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_candidate(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved.parent == resolved_root and resolved.name.startswith(SCRATCH_PREFIX)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _marker(path: Path) -> Optional[Dict[str, Any]]:
    marker = path / MARKER_NAME
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if value.get("schema_version") == SCHEMA_VERSION else None


def create_isaac_scratch(
    root: str = "/tmp", label: str = "run", owner_pid: Optional[int] = None, preserve: bool = False
) -> str:
    """Create a marked, uniquely named portable root owned by one launcher."""
    scratch_root = Path(root).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(character if character.isalnum() or character in "-_" else "-" for character in label)[:80] or "run"
    path = Path(tempfile.mkdtemp(prefix="%s%s-" % (SCRATCH_PREFIX, safe_label), dir=str(scratch_root)))
    marker = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "owner_pid": int(owner_pid if owner_pid is not None else os.getpid()),
        "preserve": bool(preserve),
        "label": safe_label,
    }
    (path / MARKER_NAME).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def release_isaac_scratch(path: str, root: str = "/tmp", force: bool = False) -> Dict[str, Any]:
    """Remove one marked portable root without accepting broad targets."""
    scratch_root = Path(root).resolve()
    candidate = Path(path)
    if not _safe_candidate(candidate, scratch_root):
        raise ValueError("refusing unsafe Isaac scratch target: %s" % candidate)
    marker = _marker(candidate)
    if marker is None:
        raise ValueError("refusing unowned Isaac scratch target: %s" % candidate)
    if marker.get("preserve") and not force:
        return {"status": "preserved", "path": str(candidate), "removed_bytes": 0}
    size = sum(item.stat().st_size for item in candidate.rglob("*") if item.is_file())
    shutil.rmtree(str(candidate))
    return {"status": "removed", "path": str(candidate), "removed_bytes": size}


def cleanup_stale_isaac_scratch(
    root: str = "/tmp", stale_after_s: float = 3600.0, include_legacy: bool = False
) -> Dict[str, Any]:
    """Remove dead owned roots; legacy removal is explicit and never automatic."""
    scratch_root = Path(root).resolve()
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    result: Dict[str, Any] = {"root": str(scratch_root), "removed": [], "removed_bytes": 0, "active": [], "preserved": [], "legacy": []}
    if not scratch_root.is_dir():
        return result
    for candidate in sorted(scratch_root.iterdir()):
        if not candidate.is_dir() or not _safe_candidate(candidate, scratch_root):
            continue
        marker = _marker(candidate)
        age_s = max(0.0, now - candidate.stat().st_mtime)
        if marker is None:
            result["legacy"].append(str(candidate))
            if not include_legacy or age_s < stale_after_s:
                continue
        else:
            if marker.get("preserve"):
                result["preserved"].append(str(candidate))
                continue
            if _pid_is_alive(int(marker.get("owner_pid", 0))):
                result["active"].append(str(candidate))
                continue
            if age_s < stale_after_s:
                continue
        size = sum(item.stat().st_size for item in candidate.rglob("*") if item.is_file())
        shutil.rmtree(str(candidate))
        result["removed"].append(str(candidate))
        result["removed_bytes"] += size
    result["removed_count"] = len(result["removed"])
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", default="/tmp")
    create.add_argument("--label", default="run")
    create.add_argument("--owner-pid", type=int)
    create.add_argument("--preserve", action="store_true")
    release = subparsers.add_parser("release")
    release.add_argument("path")
    release.add_argument("--root", default="/tmp")
    release.add_argument("--force", action="store_true")
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--root", default="/tmp")
    cleanup.add_argument("--stale-after", type=float, default=3600.0)
    cleanup.add_argument("--include-legacy", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "create":
        print(create_isaac_scratch(arguments.root, arguments.label, arguments.owner_pid, arguments.preserve))
    elif arguments.command == "release":
        print(json.dumps(release_isaac_scratch(arguments.path, arguments.root, arguments.force), sort_keys=True))
    else:
        print(json.dumps(cleanup_stale_isaac_scratch(arguments.root, arguments.stale_after, arguments.include_legacy), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

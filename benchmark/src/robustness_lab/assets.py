"""Pinned external asset acquisition and checksum verification."""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import load_data, sha256_file


def _entries(registry_path: str) -> List[Dict[str, Any]]:
    data = load_data(registry_path)
    entries = data.get("assets", [])
    if not isinstance(entries, list):
        raise ValueError("assets registry must contain an assets list")
    return entries


def verify_assets(registry_path: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    registry = Path(registry_path).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else registry.parent / "cache"
    registry_data = load_data(str(registry))
    results = []
    valid = True
    for entry in _entries(str(registry)):
        destination = cache / entry["path"]
        if entry.get("kind") == "git":
            if not (destination / ".git").exists():
                results.append({"name": entry["name"], "status": "missing", "path": str(destination)})
                valid = False
                continue
            result = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], capture_output=True, text=True)
            actual = result.stdout.strip() if result.returncode == 0 else ""
            expected = entry.get("revision", "")
            if not actual or (expected and actual != expected):
                results.append({"name": entry["name"], "status": "revision_mismatch", "expected": expected, "actual": actual})
                valid = False
            else:
                declared_licenses = entry.get("license_paths", [entry["license_path"]] if entry.get("license_path") else [])
                missing_licenses = [item for item in declared_licenses if not (destination / item).is_file()]
                if missing_licenses:
                    results.append({"name": entry["name"], "status": "license_missing", "revision": actual, "path": str(destination), "missing": missing_licenses})
                    valid = False
                else:
                    results.append({"name": entry["name"], "status": "verified", "revision": actual, "path": str(destination), "license": entry.get("license", ""), "license_files": [str((destination / item).resolve()) for item in declared_licenses]})
            continue
        if not destination.exists():
            results.append({"name": entry["name"], "status": "missing", "path": str(destination)})
            valid = False
            continue
        actual = sha256_file(str(destination))
        expected = entry.get("sha256", "")
        if expected and expected != actual:
            results.append({"name": entry["name"], "status": "checksum_mismatch", "expected": expected, "actual": actual})
            valid = False
        else:
            results.append({"name": entry["name"], "status": "verified", "sha256": actual, "path": str(destination)})
    verified_files = []
    for entry in registry_data.get("verified_files", []):
        path = cache / entry["path"]
        if not path.is_file():
            verified_files.append({"path": str(path), "status": "missing"})
            valid = False
            continue
        actual = sha256_file(str(path))
        matches = actual == entry["sha256"]
        verified_files.append({"path": str(path), "status": "verified" if matches else "checksum_mismatch", "expected": entry["sha256"], "actual": actual})
        valid = valid and matches
    return {"valid": valid, "registry": str(registry), "cache": str(cache), "assets": results, "verified_files": verified_files}


def fetch_assets(registry_path: str, cache_dir: Optional[str] = None, timeout_s: float = 30.0) -> Dict[str, Any]:
    registry = Path(registry_path).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else registry.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in _entries(str(registry)):
        destination = cache / entry["path"]
        if entry.get("kind") == "git":
            revision = entry["revision"]
            try:
                if not (destination / ".git").exists():
                    command = ["git", "clone", "--filter=blob:none", "--no-checkout", entry["url"], str(destination)]
                    subprocess.run(command, check=True, capture_output=True, text=True, timeout=max(timeout_s, 120.0))
                subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision], check=True, capture_output=True, text=True, timeout=max(timeout_s, 120.0))
                sparse_paths = entry.get("sparse_paths", [])
                if sparse_paths:
                    subprocess.run(["git", "-C", str(destination), "sparse-checkout", "set"] + sparse_paths, check=True, capture_output=True, text=True, timeout=max(timeout_s, 120.0))
                subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True, capture_output=True, text=True, timeout=max(timeout_s, 120.0))
                results.append({"name": entry["name"], "status": "downloaded", "path": str(destination), "revision": revision})
            except (OSError, subprocess.SubprocessError) as error:
                results.append({"name": entry["name"], "status": "error", "error": str(error), "path": str(destination)})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("sha256", "")
        if destination.exists() and (not expected or sha256_file(str(destination)) == expected):
            results.append({"name": entry["name"], "status": "cached", "path": str(destination)})
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        try:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "robustness-lab/0.1"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            actual = sha256_file(str(temporary))
            if expected and actual != expected:
                temporary.unlink(missing_ok=True)
                results.append({"name": entry["name"], "status": "checksum_mismatch", "expected": expected, "actual": actual})
                continue
            temporary.replace(destination)
            results.append({"name": entry["name"], "status": "downloaded", "path": str(destination), "sha256": actual})
        except (urllib.error.URLError, OSError) as error:
            if temporary.exists():
                temporary.unlink()
            results.append({"name": entry["name"], "status": "error", "error": str(error), "path": str(destination)})
    verification = verify_assets(str(registry), str(cache))
    return {"valid": all(item["status"] in {"cached", "downloaded"} for item in results) and verification["valid"], "assets": results, "verification": verification}

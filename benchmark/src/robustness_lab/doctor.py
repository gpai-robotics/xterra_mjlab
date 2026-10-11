"""Read-only environment and simulator capability diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .io import load_data


def expand_runtime_path(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _probe_python(executable: str, imports: list) -> Dict[str, Any]:
    if not Path(executable).exists() and not shutil.which(executable):
        return {"status": "missing", "executable": executable}
    code = """
import importlib.util
import json
import platform
import sys
from importlib import metadata
names = %r
versions = {}
for name in names:
    try:
        versions[name] = [metadata.version(name)]
    except metadata.PackageNotFoundError:
        versions[name] = []
print(json.dumps({
    'python': sys.version,
    'platform': platform.platform(),
    'imports': {name: bool(importlib.util.find_spec(name)) for name in names},
    'package_versions': versions,
}))
""" % imports
    try:
        result = subprocess.run([executable, "-c", code], capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return {"status": "error", "executable": executable, "stderr": result.stderr.strip()}
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        missing_imports = [name for name, available in payload.get("imports", {}).items() if not available]
        payload.update({
            "status": "imports_missing" if missing_imports else "available",
            "executable": executable,
            "missing_imports": missing_imports,
        })
        return payload
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        return {"status": "error", "executable": executable, "error": str(error)}


def _gpu_probe() -> Dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"status": "missing", "reason": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"status": "unavailable", "reason": result.stderr.strip() or result.stdout.strip()}
        return {"status": "available", "devices": [line.strip() for line in result.stdout.splitlines() if line.strip()]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": str(error)}


def run_doctor(config_path: str) -> Dict[str, Any]:
    config = load_data(config_path)
    runtimes = {}
    for name, settings in config.get("runtimes", {}).items():
        imports = settings.get("imports", [])
        if settings.get("launcher") and not settings.get("python"):
            executable = expand_runtime_path(settings["launcher"])
            status = "available" if Path(executable).exists() else "missing"
            runtimes[name] = {"status": status, "launcher": executable, "imports": imports}
        else:
            executable = expand_runtime_path(settings.get("python", sys.executable))
            runtimes[name] = _probe_python(executable, imports)
        if settings.get("version_file"):
            version_file = Path(settings["version_file"])
            runtimes[name]["installed_version"] = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else None
        if settings.get("expected_version"):
            runtimes[name]["expected_version"] = str(settings["expected_version"])
        if settings.get("expected_sim_version"):
            runtimes[name]["expected_sim_version"] = str(settings["expected_sim_version"])
        if settings.get("features"):
            runtimes[name]["features"] = settings["features"]
    disk = shutil.disk_usage(str(Path(config_path).resolve().parent))
    gpu = _gpu_probe()
    required_settings = {name: value for name, value in config.get("runtimes", {}).items() if value.get("required", True)}
    ready = all(runtimes.get(name, {}).get("status") == "available" for name in required_settings)
    if any(value.get("requires_gpu") for value in required_settings.values()) and gpu["status"] != "available":
        ready = False
    full_ready = all(value.get("status") == "available" for value in runtimes.values()) and (not any(value.get("requires_gpu") for value in config.get("runtimes", {}).values()) or gpu["status"] == "available")
    return {
        "ready": ready,
        "full_cross_sim_ready": full_ready,
        "gpu": gpu,
        "cuda_compatibility": {"status": "unverified" if gpu["status"] != "available" else "driver_visible", "reason": "CUDA compatibility requires a functioning GPU driver" if gpu["status"] != "available" else "run backend calibration to verify runtime compatibility"},
        "runtimes": runtimes,
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "environment": {"display": os.environ.get("DISPLAY", ""), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
    }

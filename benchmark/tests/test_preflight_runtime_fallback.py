"""Preflight must agree with doctor about whether a runtime exists.

`runtime._runtime_command` and `doctor._probe_python` both fall back to
sys.executable when a runtime declares no `python:` -- that is the whole point
of P0.1, so a clean clone works without a machine-specific runtimes.yaml.

`preflight._runtime_report` did not. It read `python` or `launcher` and called
anything else "configured executable was not found", so the shipped mujoco
runtime -- which deliberately declares no interpreter -- was reported missing on
every machine, and `simulator_runtime` was false even with MuJoCo installed and
working. doctor said available, preflight said missing, about the same runtime.
"""

import sys
import yaml


def _config(tmp_path, runtimes, skill_extra=None):
    """A minimal examiner pointing at a runtimes file we control."""
    runtimes_path = tmp_path / "runtimes.yaml"
    runtimes_path.write_text(yaml.safe_dump({
        "schema_version": "1.0", "runtimes": runtimes}), encoding="utf-8")
    return str(runtimes_path)


def _probe(tmp_path, settings):
    from robustness_lab.preflight import _runtime_static_status
    from robustness_lab.io import load_data
    path = _config(tmp_path, {"mujoco": settings})
    return _runtime_static_status(load_data(path), "mujoco")


def test_a_runtime_without_an_interpreter_falls_back_to_sys_executable(tmp_path):
    report = _probe(tmp_path, {"imports": ["numpy"], "required": True})
    assert report["status"] == "ready", report
    assert report["executable"] == sys.executable


def test_an_explicit_missing_interpreter_is_still_missing(tmp_path):
    report = _probe(tmp_path, {"python": "/nonexistent/python", "imports": []})
    assert report["status"] == "missing"


def test_an_explicit_present_interpreter_is_used_as_declared(tmp_path):
    report = _probe(tmp_path, {"python": sys.executable, "imports": []})
    assert report["status"] == "ready"
    assert report["executable"] == sys.executable


def test_an_undeclared_runtime_is_still_reported_missing(tmp_path):
    from robustness_lab.preflight import _runtime_static_status
    report = _runtime_static_status({"runtimes": {}}, "mujoco")
    assert report["status"] == "missing"
    assert "not declared" in report["reason"]

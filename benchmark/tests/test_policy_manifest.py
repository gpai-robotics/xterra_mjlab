"""Schema-v2 policy manifest: the file a researcher actually writes.

The manifest describes an existing policy. It never modifies it.
"""

import pytest
import yaml


REFERENCE_V2 = {
    "schema_version": "2.0",
    "policy_id": "reference-go2",
    "runtime": {"format": "torchscript", "path": "policy.pt", "control_frequency_hz": 50},
    "observation": {
        "terms": [
            {"source": "base_angular_velocity_b", "transform": [{"scale": 0.25}], "size": 3},
            {"source": "projected_gravity_b", "size": 3},
            {"source": "velocity_command", "transform": [{"scale": [2.0, 2.0, 0.25]}], "size": 3},
            {"source": "joint_position",
             "transform": [{"subtract": "default_joint_position"}], "size": 12},
            {"source": "joint_velocity", "transform": [{"scale": 0.05}], "size": 12},
            {"source": "previous_action", "size": 12},
        ],
        "history": {"frames": 1, "order": "newest_first"},
        "expected_size": 45,
    },
    "action": {"space": "joint_position_offset", "scale": 0.25,
               "offset": "default_joint_position", "clip": [-100.0, 100.0], "size": 12},
    # Naming default_joint_position obliges the manifest to say what it is:
    # unbound to a robot profile, nothing else can supply the Go2 stance, and
    # the loader refuses to invent zeros for it.
    "references": {"default_joint_position":
                   [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]},
}


def _write(tmp_path, payload, name="policy.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    (tmp_path / "policy.pt").write_bytes(b"placeholder")
    return str(path)


def test_reference_manifest_compiles_to_45_dimensions(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest
    manifest = load_policy_manifest(_write(tmp_path, REFERENCE_V2), check_files=False)
    assert manifest.policy_id == "reference-go2"
    assert manifest.observation_plan.size == 45


def test_expected_size_mismatch_is_rejected_at_load(tmp_path):
    """A wrong expected_size must fail loudly, not silently produce a bad vector."""
    from robustness_lab.policy_manifest import load_policy_manifest
    payload = yaml.safe_load(yaml.safe_dump(REFERENCE_V2))
    payload["observation"]["expected_size"] = 48
    with pytest.raises(ValueError, match="45"):
        load_policy_manifest(_write(tmp_path, payload), check_files=False)


def test_history_is_actually_applied_in_v2(tmp_path):
    """The v1 schema parsed `history` and discarded it. v2 must honour it."""
    from robustness_lab.policy_manifest import load_policy_manifest
    payload = yaml.safe_load(yaml.safe_dump(REFERENCE_V2))
    payload["observation"]["history"] = {"frames": 4, "order": "newest_first"}
    payload["observation"]["expected_size"] = 180
    manifest = load_policy_manifest(_write(tmp_path, payload), check_files=False)
    assert manifest.observation_plan.size == 180


def test_unknown_schema_version_is_rejected(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest
    payload = yaml.safe_load(yaml.safe_dump(REFERENCE_V2))
    payload["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="9.9"):
        load_policy_manifest(_write(tmp_path, payload), check_files=False)


def test_shipped_template_is_valid_and_compiles():
    """The template a researcher copies must load, not just look plausible."""
    from pathlib import Path
    from robustness_lab.policy_manifest import load_policy_manifest
    root = Path(__file__).resolve().parents[1]
    manifest = load_policy_manifest(
        str(root / "configs" / "templates" / "policy.yaml"), check_files=False)
    assert manifest.observation_plan.size == 45


def test_shipped_template_points_at_the_source_list():
    """A researcher must be able to find the vocabulary from the template."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "templates" / "policy.yaml").read_text(encoding="utf-8")
    assert "docs/observation_sources.md" in text
    assert (root / "docs" / "observation_sources.md").is_file()


def test_example_submission_loads_from_its_own_folder():
    """A submission must be self-contained: weights referenced relative to the
    manifest, so the folder can be moved or handed over whole."""
    from pathlib import Path
    from robustness_lab.policy_manifest import load_policy_manifest
    root = Path(__file__).resolve().parents[1]
    manifest = root / "submissions" / "example-go2-velocity-flat" / "policy.yaml"
    if not manifest.is_file():
        import pytest
        pytest.skip("example submission not present")
    loaded = load_policy_manifest(str(manifest), check_files=True)
    assert loaded.observation_plan.size == 45


def test_submissions_readme_and_template_exist():
    """A researcher must find the entry point without reading src/."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "submissions" / "README.md").is_file()
    assert (root / "submissions" / "_template" / "policy.yaml").is_file()

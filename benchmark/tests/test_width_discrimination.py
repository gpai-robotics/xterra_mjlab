"""Width acceptance is not proof when an artifact accepts a range.

check_dimension enforces an exact match between the compiled width, the declared
width and the artifact's width. But the artifact's width is established by
PROBING, and some exported policies slice or pad their input instead of
validating it. The real FoReSight exported teacher accepts 560, 600, 640, 680
and 720 -- every one passes individually.

So a researcher who declares the wrong width passes G1 and is silently fed a
misaligned observation. The gate must detect that it cannot discriminate and say
so, rather than returning a confident "ready".
"""

import pytest


def _fixed_width_artifact(tmp_path, width=45, out=12):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    path = tmp_path / "fixed.pt"
    torch.jit.script(nn.Sequential(nn.Linear(width, 16), nn.ELU(), nn.Linear(16, out))).save(str(path))
    return str(path)


def _permissive_artifact(tmp_path, out=12):
    """Mimics an exporter that slices its input: accepts anything wide enough."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Sliced(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(45, out)

        def forward(self, x):
            return self.net(x[:, :45])

    path = tmp_path / "permissive.pt"
    torch.jit.script(Sliced()).save(str(path))
    return str(path)


def test_fixed_width_artifact_is_reported_as_discriminating(tmp_path):
    from robustness_lab.observation.gates import check_artifact_accepts
    result = check_artifact_accepts(_fixed_width_artifact(tmp_path), 45)
    assert result["status"] == "ready"
    assert result["width_is_discriminating"] is True


def test_permissive_artifact_is_flagged(tmp_path):
    """It accepts 45 AND wider, so acceptance proves nothing."""
    from robustness_lab.observation.gates import check_artifact_accepts
    result = check_artifact_accepts(_permissive_artifact(tmp_path), 45)
    assert result["width_is_discriminating"] is False
    assert "accepted_widths" in result


def test_permissive_artifact_reports_the_range_it_accepts(tmp_path):
    from robustness_lab.observation.gates import check_artifact_accepts
    result = check_artifact_accepts(_permissive_artifact(tmp_path), 45)
    accepted = result["accepted_widths"]
    assert len(accepted) > 1, accepted
    assert 45 in accepted


def test_permissive_artifact_still_runs_but_warns(tmp_path):
    """Do not refuse a legitimate slicing policy -- warn that width alone is
    insufficient and the semantic gates must carry the verification."""
    from robustness_lab.observation.gates import check_artifact_accepts
    result = check_artifact_accepts(_permissive_artifact(tmp_path), 45)
    assert result["status"] == "ready"
    assert result.get("warning"), "a non-discriminating probe must warn"


def test_dimension_gate_surfaces_the_warning(tmp_path):
    from robustness_lab.observation.gates import check_dimension
    result = check_dimension(compiled_size=45, declared_size=45, artifact_size=45,
                             width_is_discriminating=False)
    assert result["status"] == "ready"
    assert result.get("warning")


def test_dimension_gate_is_silent_when_the_probe_discriminates(tmp_path):
    from robustness_lab.observation.gates import check_dimension
    result = check_dimension(compiled_size=45, declared_size=45, artifact_size=45,
                             width_is_discriminating=True)
    assert result["status"] == "ready"
    assert not result.get("warning")

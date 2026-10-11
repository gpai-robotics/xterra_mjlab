"""Evidence rendering was 76% of a parallel sweep's wall clock.

Profiling a 60-trial run at 24 workers:

    TOTAL 117.4s
      refresh    89.5s   76.2%
      dump        1.2s    1.0%
      prepare     0.1s    0.1%
      attach      0.0s    0.0%
      persist     0.0s    0.0%

So the trials themselves took about 28 s and the bookkeeping took 89 s. Not the
state file (1%), and not the quadratic-looking evidence matcher (0%) -- both of
which were the obvious suspects and both of which were wrong.

`refresh_evidence` renders per-trial snapshots and telemetry videos: PIL drawing
plus ffmpeg subprocesses, roughly 1.5 s per trial, in a serial loop in the
parent process while 56 cores sit idle. Each trial's media is independent of
every other trial's, and ffmpeg runs out of process, so this parallelises.

Correctness first: the rendered result must not depend on how many workers did
the rendering.
"""

import json
import tempfile
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
SKILL = ROOT / "submissions/go2_blind/skill.yaml"


def _requirements():
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("PIL")
    for path in (ASSET, PROFILE, SKILL):
        if not path.is_file():
            pytest.skip("missing %s" % path.name)


def _examination(tmp_path, name, trials=6):
    from robustness_lab.discrete_examiner import (
        initialize_discrete_examination, run_discrete_examination)
    body = {
        "schema_version": "2.0", "examiner_id": "render-%s" % name,
        "scheduler": "discrete_pruned_v2", "backend": "mujoco",
        "robot_profile": str(PROFILE), "robot_asset": str(ASSET),
        "robot_format": "mjcf", "skill": str(SKILL),
        "runtimes": str(ROOT / "configs/runtimes.yaml"),
        "output_dir": str(tmp_path / name),
        "terrain_codes": ["rs1"], "seed": 1701,
        "discrete": {"protocol_variant_id": "render-test",
                     "screening_repetitions": 2, "confirmation_repetitions": 3,
                     "confirmation_successes": 2, "parallel_trials": 4},
    }
    config = tmp_path / ("%s.yaml" % name)
    config.write_text(yaml.safe_dump(body), encoding="utf-8")
    initialize_discrete_examination(str(config))
    state_path = tmp_path / name / "examination_state.json"
    run_discrete_examination(str(state_path), max_trials=trials)
    return state_path


def _media_fingerprint(state_path):
    """Every media path recorded, in a stable order."""
    state = json.loads(Path(state_path).read_text())
    out = []
    for trial in sorted(state["trials"], key=lambda t: t["trial_id"]):
        media = trial.get("media", {})
        out.append((trial["trial_id"],
                    tuple(sorted(str(s) for s in media.get("snapshots", []))),
                    str(media.get("video"))))
    return out


# ---------------------------------------------------------------------------
# The rendering worker pool
# ---------------------------------------------------------------------------

def test_refresh_evidence_accepts_a_worker_count():
    from robustness_lab.evidence import refresh_evidence
    import inspect
    assert "workers" in inspect.signature(refresh_evidence).parameters


def test_worker_count_is_validated():
    from robustness_lab.evidence import refresh_evidence
    with pytest.raises(ValueError):
        refresh_evidence("/nonexistent/state.json", workers=0)


def test_rendering_is_identical_regardless_of_worker_count(tmp_path):
    """The criterion that matters: parallel rendering must change nothing."""
    _requirements()
    from robustness_lab.evidence import refresh_evidence

    state_path = _examination(tmp_path, "serial-render")
    refresh_evidence(str(state_path), workers=1)
    serial = _media_fingerprint(state_path)

    # Clear the recorded media so it must all be produced again.
    state = json.loads(Path(state_path).read_text())
    for trial in state["trials"]:
        trial.pop("media", None)
    Path(state_path).write_text(json.dumps(state), encoding="utf-8")

    refresh_evidence(str(state_path), workers=8)
    assert _media_fingerprint(state_path) == serial


def test_the_dashboard_is_still_produced(tmp_path):
    _requirements()
    from robustness_lab.evidence import refresh_evidence
    state_path = _examination(tmp_path, "dashboard")
    result = refresh_evidence(str(state_path), workers=8)
    assert Path(result["dashboard"]).is_file()


# ---------------------------------------------------------------------------
# The scheduler must use it
# ---------------------------------------------------------------------------

def test_the_scheduler_renders_with_its_configured_worker_count(tmp_path):
    """A pool nothing passes workers to is not a speedup."""
    _requirements()
    import robustness_lab.discrete_examiner as de

    seen = []
    original = de.refresh_evidence

    def spy(state_path, *args, **kwargs):
        seen.append(kwargs.get("workers", args[0] if args else None))
        return original(state_path, *args, **kwargs)

    de.refresh_evidence = spy
    try:
        _examination(tmp_path, "spy", trials=4)
    finally:
        de.refresh_evidence = original

    assert seen, "refresh_evidence was never called"
    assert any(value and int(value) > 1 for value in seen), (
        "scheduler always rendered with a single worker: %r" % seen)

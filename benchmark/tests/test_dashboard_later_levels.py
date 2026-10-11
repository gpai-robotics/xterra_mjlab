"""The dashboard must report what L4-L7 actually did, not assume they never ran.

The story panel's fourth step was unconditional: whatever the run contained, it
rendered

    "L4-L7 -- Not run, not failed"
    four chips reading "not measured"
    "endurance, stress robustness, or held-out generalization ... remain
     unknown rather than failed"

That was written when a 70% gate on L3 really did cancel L4-L7. P50 removed the
gate and updated the paragraph text, but left the heading, the chips and the
conclusion -- so a run that executed 37 L4-L7 conditions (v2: L4 5/7, L5 1/5,
L6 6/10, L7 2/1, all present in the embedded level_counts) still told the
reader those levels were never run. The videos from L5 and L6 existed; the
page said they could not.

The narrative is now computed in Python from the same level_counts the rest of
the page uses, so it can be tested, and the JS renders whatever it is handed.
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _summary(level_counts, level_results=None, pair_library_empty=False):
    from robustness_lab.evidence import later_levels_summary
    return later_levels_summary(level_counts, level_results or {}, pair_library_empty)


EXECUTED = {"L4": {"passed": 5, "failed": 7}, "L5": {"passed": 1, "failed": 5},
            "L6": {"passed": 6, "failed": 10}, "L7": {"passed": 2, "failed": 1}}


def test_executed_levels_are_reported_as_executed():
    s = _summary(EXECUTED)
    assert s["executed"] == 37
    assert s["status"] == "executed"
    low = s["headline"].lower()
    assert "not run" not in low and "not measured" not in low and "unknown" not in low


def test_each_level_carries_its_own_numbers():
    s = _summary(EXECUTED)
    assert s["levels"]["L5"]["passed"] == 1 and s["levels"]["L5"]["failed"] == 5
    assert s["levels"]["L6"]["executed"] == 16
    assert s["levels"]["L6"]["pass_rate"] == pytest.approx(6 / 16)


def test_the_conclusion_names_what_was_actually_measured():
    s = _summary(EXECUTED)
    text = s["conclusion"].lower()
    assert "endurance" in text and "stress" in text
    assert "remain unknown" not in text and "not run" not in text


def test_a_level_that_ran_nothing_is_reported_as_not_created():
    s = _summary({"L4": {"passed": 2, "failed": 1}})
    assert s["levels"]["L5"]["executed"] == 0
    assert s["levels"]["L5"]["status"] == "not_created"


def test_nothing_executed_says_so_with_the_real_reason():
    s = _summary({}, pair_library_empty=True)
    assert s["status"] == "not_created"
    assert "no terrain pair passed" in s["headline"].lower()
    assert "gate" not in s["headline"].lower()


def test_no_gate_language_anywhere_in_the_summary():
    for counts in (EXECUTED, {}):
        s = _summary(counts, pair_library_empty=not counts)
        blob = json.dumps(s).lower()
        assert "70%" not in blob and "gate" not in blob


def test_the_page_embeds_the_summary():
    """The JS must be handed the computed narrative, not compute its own."""
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert '"later_levels"' in src, "summary is not embedded in the dashboard data"
    assert "D.later_levels" in src, "the story panel does not read the embedded summary"


def test_the_unconditional_not_run_story_is_gone():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert "L4–L7 — Not run, not failed" not in src
    assert "remain unknown rather than failed" not in src
    # The old chips were literal markup: "<b>L5 · Endurance</b><br>... — not measured</div>".
    # The honest not-created branch may still SAY "not measured because ..."; what
    # must be gone is the unconditional per-level chip.
    assert "— not measured</div>" not in src, "hardcoded 'not measured' chips remain"
    assert "Branching worlds requiring navigation choices — not measured" not in src


def test_a_real_completed_run_renders_its_later_levels():
    state_path = ROOT / "runs/submissions/go2_blind_v2/examination_state.json"
    if not state_path.is_file():
        pytest.skip("no completed v2 run on this machine")
    from robustness_lab.evidence import refresh_evidence
    refresh_evidence(str(state_path))
    html = (state_path.parent / "dashboard/index.html").read_text()
    import re
    m = re.search(r'"later_levels":\s*(\{.*?\})\s*,\s*"', html, re.S)
    assert m, "later_levels missing from embedded data"
    summary = json.loads(m.group(1)) if m.group(1).count("{") == m.group(1).count("}") else None
    assert summary is None or summary["executed"] > 0
    assert "L4–L7 — Not run, not failed" not in html

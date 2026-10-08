"""A number on the page should explain itself on hover, without cluttering it.

'3/9' in the variant table says three of nine passed. It does not say WHICH
nine -- the three difficulties by three frictions -- nor which three passed,
nor how the six failed (fell? stuck?). The L4-L7 tiles likewise show a pass
rate and nothing about what happened. All of that is in the state: every
condition knows its difficulty and friction, every trial its terminal reason
and its event chain. It just was not surfaced.

Surfaced now as hover detail only: the page looks the same until the pointer
rests on a number. Detail is computed in Python from the state so it is
testable, embedded with the rest of the page data, and rendered by one small
tooltip in the page.
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _state():
    """Two terrains; rs1 has a left variant across three difficulty/friction
    combos with distinguishable outcomes and reasons."""
    conds, trials = {}, []

    def cond(cid, level, kind, terrain, status, geometry, friction, **extra):
        d = {"id": cid, "level": level, "kind": kind, "terrain": terrain, "status": status,
             "geometry_label": geometry, "friction_label": friction, "attempts": [],
             "summary": {}}
        d.update(extra)
        conds[cid] = d
        return d

    def trial(cid, n, success, reason, chain=None):
        tid = "%s-a%02d" % (cid, n)
        conds[cid]["attempts"].append(tid)
        trials.append({"trial_id": tid, "condition_id": cid, "attempt_index": n - 1,
                       "success": success, "terminal_reason": reason, "valid_behavior": True,
                       "status": "completed" if success else "failed",
                       "diagnosis": {"event_chain": chain or []}})

    p = cond("L1-rs1-g050-f050", "L1", "primitive", "rs1", "passed", "medium", "moderate")
    for n in (1, 2, 3):
        trial(p["id"], n, True, "goal")
    v1 = cond("L1-rs1-g050-f050-left-curve-left-low", "L1", "variant", "rs1", "passed",
              "medium", "moderate", parents=[p["id"]],
              modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0})
    for n in (1, 2, 3):
        trial(v1["id"], n, True, "goal")
    v2 = cond("L1-rs1-g025-f025-left-curve-left-low", "L1", "variant", "rs1", "failed",
              "easy", "low", parents=[p["id"]],
              modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0})
    trial(v2["id"], 1, False, "fall", ["forbidden body contact", "base instability / fall"])
    trial(v2["id"], 2, False, "fall", ["forbidden body contact", "base instability / fall"])
    trial(v2["id"], 3, False, "stuck", ["no forward progress"])
    v3 = cond("L1-rs1-g075-f075-left-curve-left-low", "L1", "variant", "rs1", "failed",
              "hard", "high", parents=[p["id"]],
              modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0})
    trial(v3["id"], 1, False, "stuck", ["no forward progress"])
    for c in conds.values():
        n = len(c["attempts"]); ok = sum(1 for t in trials if t["condition_id"] == c["id"] and t["success"])
        c["summary"] = {"valid": n, "successes": ok, "failures": n - ok}

    l4 = cond("L4-random-L-g050-b000", "L4", "progression", None, "failed", "medium", None,
              topology="random_connected", horizon="L", covered_pairs=[["rs2", "rs7"]])
    trial(l4["id"], 1, False, "fall", ["forbidden body contact", "base instability / fall"])
    trial(l4["id"], 2, True, "goal")
    trial(l4["id"], 3, False, "fall", ["forbidden body contact", "base instability / fall"])
    l4["summary"] = {"valid": 3, "successes": 1, "failures": 2}
    return {"conditions": conds, "trials": trials}


# ---------------------------------------------------------------------------
# Variant cells explain themselves
# ---------------------------------------------------------------------------

def _cell(state):
    from robustness_lab.evidence import variant_summary
    s = variant_summary(state["conditions"], state["trials"])
    return {r["code"]: r for r in s["terrains"]}["rs1"]["left-curve-left-low"]


def test_a_cell_lists_every_condition_behind_its_number():
    cell = _cell(_state())
    assert cell["executed"] == 3
    assert len(cell["conditions"]) == 3


def test_each_listed_condition_says_its_difficulty_friction_and_outcome():
    entries = {(c["difficulty"], c["friction"]): c for c in _cell(_state())["conditions"]}
    assert entries[("medium", "moderate")]["status"] == "passed"
    assert entries[("medium", "moderate")]["successes"] == 3
    assert entries[("easy", "low")]["status"] == "failed"
    assert entries[("easy", "low")]["valid"] == 3


def test_a_failed_condition_says_how_it_failed():
    entries = {(c["difficulty"], c["friction"]): c for c in _cell(_state())["conditions"]}
    reasons = entries[("easy", "low")]["reasons"]
    assert reasons.get("fall") == 2 and reasons.get("stuck") == 1


def test_the_dominant_event_chain_is_recorded():
    entries = {(c["difficulty"], c["friction"]): c for c in _cell(_state())["conditions"]}
    chain = entries[("easy", "low")]["event_chain"]
    assert "base instability / fall" in " ".join(chain)


def test_conditions_are_ordered_easy_to_hard():
    order = [(c["difficulty"], c["friction"]) for c in _cell(_state())["conditions"]]
    assert order.index(("easy", "low")) < order.index(("medium", "moderate")) < order.index(("hard", "high"))


def test_a_cell_carries_a_ready_to_show_hover_text():
    cell = _cell(_state())
    tip = cell["hover"]
    assert "medium" in tip and "moderate" in tip and "easy" in tip
    assert "fell" in tip.lower() or "fall" in tip.lower()
    assert "stuck" in tip.lower()


def test_variant_summary_still_works_without_trials():
    from robustness_lab.evidence import variant_summary
    s = variant_summary(_state()["conditions"])
    assert {r["code"] for r in s["terrains"]} == {"rs1"}


# ---------------------------------------------------------------------------
# Level tiles explain themselves
# ---------------------------------------------------------------------------

def test_level_details_summarise_each_level():
    from robustness_lab.evidence import level_details
    d = level_details(_state()["conditions"], _state()["trials"])
    assert d["L4"]["executed"] == 1
    assert d["L4"]["reasons"] == {"goal": 1, "fall": 2}


def test_level_details_list_conditions_with_a_readable_label_and_outcome():
    from robustness_lab.evidence import level_details
    d = level_details(_state()["conditions"], _state()["trials"])
    c = d["L4"]["conditions"][0]
    assert "random_connected" in c["label"] and "L" in c["label"]
    assert c["status"] == "failed" and c["successes"] == 1 and c["valid"] == 3


def test_level_details_carry_a_hover_text_that_names_the_failures():
    from robustness_lab.evidence import level_details
    d = level_details(_state()["conditions"], _state()["trials"])
    tip = d["L4"]["hover"].lower()
    assert "fall" in tip or "fell" in tip
    assert "1/3" in tip, "attempt-level outcome missing"      # the condition failed, but 1 of 3 attempts reached the goal
    assert "0 passed" in tip and "1 failed" in tip           # condition-level outcome


# ---------------------------------------------------------------------------
# The page uses it, on hover only
# ---------------------------------------------------------------------------

def test_the_page_embeds_level_details():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert '"level_details"' in src and "D.level_details" in src


def test_the_page_has_one_tooltip_and_attaches_it_by_data_attribute():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert 'id="tip"' in src or "id='tip'" in src, "no tooltip element"
    assert "data-tip" in src, "nothing carries hover text"
    assert "mouseenter" in src or "mouseover" in src


def test_variant_cells_and_level_tiles_carry_hover_text():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert "data-tip" in src[src.find("const vcell"):src.find("const vcell") + 600], "variant cells have no hover"
    i = src.find("$('levels').innerHTML")
    assert "data-tip" in src[i:i + 900], "level tiles have no hover"


def test_a_real_run_produces_hover_detail(tmp_path):
    state_path = ROOT / "runs/submissions/go2_blind_v2/examination_state.json"
    if not state_path.is_file():
        pytest.skip("no v2 run on this machine")
    from robustness_lab.evidence import variant_summary, level_details
    s = json.load(open(state_path))
    v = variant_summary(s["conditions"], s["trials"])
    rs1 = {r["code"]: r for r in v["terrains"]}["rs1"]["left-curve-left-low"]
    assert len(rs1["conditions"]) == 9 and rs1["hover"]
    assert any(c["reasons"] for c in rs1["conditions"] if c["status"] == "failed")
    assert level_details(s["conditions"], s["trials"])["L6"]["executed"] == 16


def test_l6_conditions_name_their_stressor():
    """L6 exists to apply stressors; a row reading 'chain · L' says nothing."""
    from robustness_lab.evidence import level_details
    conds = {"L6-x": {"id": "L6-x", "level": "L6", "kind": "progression", "status": "failed",
                      "topology": "chain", "horizon": "L", "geometry_label": "easy",
                      "stressors": [{"type": "action_latency_s", "value": 0.02}], "attempts": [], "summary": {}}}
    label = level_details(conds, [])["L6"]["conditions"][0]["label"]
    assert "action_latency_s" in label


def test_the_tooltip_css_is_in_the_stylesheet_that_is_actually_rendered():
    """The generator has a legacy template with its own <style>. The first
    version of this feature put the tooltip CSS there, so the live page rendered
    the tooltip position:static, in flow, at the bottom of the document -- reported
    'visible' by the DOM and invisible to a person."""
    import re
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    live = src.find("def _dashboard_html(")
    legacy = src.find("def _dashboard_html_legacy(")
    assert live != -1 and legacy != -1
    rule = src.find("#tip{position:fixed")
    assert rule != -1
    # the rule must sit inside _dashboard_html's body, not the legacy one
    owners = sorted([(legacy, "legacy"), (live, "live")])
    owner = [name for pos, name in owners if pos < rule][-1]
    assert owner == "live", "tooltip CSS is in the legacy template"

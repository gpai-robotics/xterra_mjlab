"""L3-L7 get the same treatment as L1 and L2: a table per level, after the pair map.

L1 had the banked/curved table and L2 the pair map, each with hover and click
detail. L3-L7 had only the pass-rate tiles: no list of the worlds that ran,
what they were built from, or how they failed. Each composed level is now a
section, in order after the pair map, with one row per world:

  world (topology . horizon . stressor / banked) . difficulty . outcome .
  built from (the pairs it covers) . retest

'Retest' is the counterfactual the scheduler runs when a composed world fails:
one covered pair on its own. If the pair alone still passes, the failure came
from the composition -- length, turns, the stressor; if the pair alone fails
too, the pair is dropped from later levels. That verdict is the most useful
sentence on the row and was not visible anywhere.
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    conds, trials = {}, []

    def cond(cid, level, kind, status, **extra):
        d = {"id": cid, "level": level, "kind": kind, "status": status, "attempts": [], "summary": {}}
        d.update(extra); conds[cid] = d; return d

    def trial(cid, n, ok, reason, chain=None):
        tid = "%s-a%02d" % (cid, n); conds[cid]["attempts"].append(tid)
        trials.append({"trial_id": tid, "condition_id": cid, "attempt_index": n - 1, "success": ok,
                       "terminal_reason": reason, "valid_behavior": True, "diagnosis": {"event_chain": chain or []}})

    w1 = cond("L3-chain-M-g025-b000", "L3", "progression", "passed", topology="chain", horizon="M",
              geometry_label="easy", covered_pairs=[["rs1", "rs2"]], stressors=[], modifier={})
    for n in (1, 2, 3): trial(w1["id"], n, True, "goal")
    w2 = cond("L3-grid-M-g025-b000", "L3", "progression", "failed", topology="grid", horizon="M",
              geometry_label="easy", covered_pairs=[["rs4", "rs1"], ["rs4", "rs2"]], stressors=[], modifier={})
    trial(w2["id"], 1, False, "fall", ["forbidden body contact", "base instability / fall"])
    trial(w2["id"], 2, False, "stuck", ["no forward progress"])
    trial(w2["id"], 3, True, "goal")
    cf = cond("L3-grid-M-g025-b000-cf-rs4-to-rs1", "L3", "counterfactual", "passed", pair=["rs4", "rs1"],
              parent_world=w2["id"], geometry_label="easy")
    for n in (1, 2, 3): trial(cf["id"], n, True, "goal")
    w3 = cond("L6-action-latency-L-g025-b000", "L6", "progression", "failed", topology="chain", horizon="L",
              geometry_label="easy", covered_pairs=[["rs4", "rs2"]],
              stressors=[{"type": "action_latency_s", "value": 0.02}], modifier={})
    trial(w3["id"], 1, False, "fall", ["forbidden body contact", "base instability / fall"])
    cf2 = cond("L6-action-latency-L-g025-b000-cf-rs4-to-rs2", "L6", "counterfactual", "failed",
               pair=["rs4", "rs2"], parent_world=w3["id"], geometry_label="easy")
    trial(cf2["id"], 1, False, "fall")
    w4 = cond("L7-locked-ood-L-g025-b000", "L7", "progression", "passed", topology="random_connected",
              horizon="L", geometry_label="easy", covered_pairs=[["rs1", "rs2"]], stressors=[],
              modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0})
    for n in (1, 2, 3): trial(w4["id"], n, True, "goal")
    for c in conds.values():
        n = len(c["attempts"]); ok = sum(1 for t in trials if t["condition_id"] == c["id"] and t["success"])
        c["summary"] = {"valid": n, "successes": ok, "failures": n - ok}
    return conds, trials


def _levels():
    from robustness_lab.evidence import composed_levels
    conds, trials = _fixture()
    return {lvl["level"]: lvl for lvl in composed_levels(conds, trials)}


# ---------------------------------------------------------------------------

def test_levels_come_out_in_order_l3_to_l7():
    from robustness_lab.evidence import composed_levels
    conds, trials = _fixture()
    assert [lvl["level"] for lvl in composed_levels(conds, trials)] == ["L3", "L4", "L5", "L6", "L7"]


def test_each_level_has_a_title_and_a_blurb():
    for lvl in _levels().values():
        assert lvl["title"] and lvl["blurb"]


def test_a_level_lists_its_worlds_not_its_retests():
    l3 = _levels()["L3"]
    assert [r["id"] for r in l3["rows"]] == ["L3-chain-M-g025-b000", "L3-grid-M-g025-b000"]


def test_a_world_row_says_what_it_is_and_how_it_went():
    row = _levels()["L3"]["rows"][1]
    assert "grid" in row["name"] and "M" in row["name"]
    assert row["difficulty"] == "easy"
    assert row["status"] == "failed" and row["successes"] == 1 and row["valid"] == 3
    assert row["reasons"] == {"fall": 1, "stuck": 1, "goal": 1}
    assert "base instability / fall" in " ".join(row["event_chain"])


def test_a_world_row_says_which_pairs_it_was_built_from():
    row = _levels()["L3"]["rows"][1]
    assert row["pairs"] == ["rs4→rs1", "rs4→rs2"]


def test_a_failed_world_carries_its_retest_and_a_verdict():
    row = _levels()["L3"]["rows"][1]
    assert row["retest"]["pair"] == "rs4→rs1"
    assert row["retest"]["status"] == "passed"
    v = row["retest"]["verdict"].lower()
    assert "alone" in v and ("composition" in v or "world" in v)


def test_a_retest_that_also_fails_says_the_pair_is_dropped():
    row = _levels()["L6"]["rows"][0]
    assert row["retest"]["status"] == "failed"
    assert "dropped" in row["retest"]["verdict"].lower() or "removed" in row["retest"]["verdict"].lower()


def test_a_passed_world_has_no_retest():
    assert _levels()["L3"]["rows"][0]["retest"] is None


def test_l6_rows_name_the_stressor_and_l7_rows_say_banked():
    assert "action_latency_s" in _levels()["L6"]["rows"][0]["name"]
    assert "banked" in _levels()["L7"]["rows"][0]["name"].lower() or "curved" in _levels()["L7"]["rows"][0]["name"].lower()


def test_an_empty_level_is_present_and_says_so():
    l4 = _levels()["L4"]
    assert l4["rows"] == [] and l4["executed"] == 0
    assert "no" in l4["empty_reason"].lower()


def test_rows_carry_hover_text():
    row = _levels()["L3"]["rows"][1]
    assert "fell" in row["hover"] and "rs4→rs1" in row["hover"].lower().replace("rs4 → rs1", "rs4→rs1")


# ---------------------------------------------------------------------------

def _live():
    s = (ROOT / "src/robustness_lab/evidence.py").read_text()
    i = s.find("def _dashboard_html("); j = s.find("\ndef ", i + 10)
    return s[i:j]


def test_the_page_embeds_and_renders_composed_levels():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert '"composed_levels"' in src and "D.composed_levels" in src
    t = _live()
    assert 'id="composed-panel"' in t


def test_the_composed_sections_come_after_the_pair_map():
    t = _live()
    assert t.find('id="pair-map"') < t.find('id="composed-panel"')


def test_a_real_run_renders_l3_to_l7_tables_in_order(tmp_path):
    state_path = ROOT / "runs/submissions/go2_blind_v2/examination_state.json"
    if not state_path.is_file():
        pytest.skip("no v2 run on this machine")
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright
    from robustness_lab.evidence import refresh_evidence
    dashboard = refresh_evidence(str(state_path))["dashboard"]
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + dashboard); pg.wait_for_timeout(800)
        heads = pg.locator("#composed-panel h3").all_inner_texts()
        assert [h[:2] for h in heads] == ["L3", "L4", "L5", "L6", "L7"], heads
        rows = pg.locator("#composed-panel table tbody tr").count()
        # one row per executed composed WORLD; retests are shown on their world's
        # row, not as rows (the tiles count them separately, which is why the
        # tiles say 37 and the tables say 34 on this run)
        state = json.load(open(state_path))
        worlds = sum(1 for c in state["conditions"].values()
                     if c["level"] in ("L3", "L4", "L5", "L6", "L7") and c["kind"] == "progression"
                     and c["status"] in ("passed", "failed"))
        assert rows == worlds, "expected %d world rows, got %d" % (worlds, rows)
        ys = [pg.locator("#pair-map").bounding_box()["y"], pg.locator("#composed-panel").bounding_box()["y"]]
        assert ys[0] < ys[1]
        # click a failed row: the popup opens with its pairs
        row = pg.locator("#composed-panel table tbody tr.fail").first
        row.scroll_into_view_if_needed(); row.click(); pg.wait_for_timeout(300)
        assert pg.locator("#pair-popup").is_visible()
        assert "built from" in pg.locator("#pair-popup").inner_text().lower()
        pg.keyboard.press("Escape")
        b.close()
    assert not errors, errors

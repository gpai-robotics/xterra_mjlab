"""A pair cell should explain the pair: what fed it, and how each difficulty went.

The 11x11 map showed three dots and 'P0 F1 X2' per pair. Clicking a cell
selected it and scrolled to a branch inspector -- which, after P59, is hidden,
so the click scrolled to nothing. What a reader actually wants from a cell:

  * which L1 results the pair was built from -- the two FLAT terrains at the
    qualification friction, one per difficulty. Banked/curved variants never
    feed a pair; the page should say so where the reader is looking;
  * for each difficulty: passed / failed / pruned, attempts, how the failures
    ended, and -- when pruned -- WHICH parent failed flat, by name;
  * so the pattern becomes visible: only terrains that passed flat form pairs.

Computed in Python from conditions + trials (pair conditions carry `parents`,
`pruned_by` and `prune_reason`; trials carry terminal reasons), embedded as
D.pair_details, hover-summarised via the existing tooltip, and opened in full
by a click popup.
"""

import json
import re
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    conds, trials = {}, []

    def cond(cid, level, kind, status, **extra):
        d = {"id": cid, "level": level, "kind": kind, "status": status, "attempts": [], "summary": {},
             "pruned_by": [], "prune_reason": ""}
        d.update(extra); conds[cid] = d; return d

    def trial(cid, n, ok, reason, chain=None):
        tid = "%s-a%02d" % (cid, n); conds[cid]["attempts"].append(tid)
        trials.append({"trial_id": tid, "condition_id": cid, "attempt_index": n - 1, "success": ok,
                       "terminal_reason": reason, "valid_behavior": True,
                       "diagnosis": {"event_chain": chain or []}})

    # L1 flat parents at the qualification friction, per difficulty
    for t, outcomes in (("rs1", {"easy": "passed", "medium": "passed", "hard": "failed"}),
                        ("rs2", {"easy": "passed", "medium": "failed", "hard": "failed"})):
        for g, st in outcomes.items():
            c = cond("L1-%s-g%s-f050" % (t, g), "L1", "primitive", st, terrain=t,
                     geometry_label=g, friction_label="moderate")
            for n in (1, 2, 3):
                trial(c["id"], n, st == "passed", "goal" if st == "passed" else "fall",
                      None if st == "passed" else ["forbidden body contact", "base instability / fall"])
    # a banked variant of rs1 that FAILED -- must not affect the pair
    v = cond("L1-rs1-geasy-f050-left-curve-left-low", "L1", "variant", "failed", terrain="rs1",
             geometry_label="easy", friction_label="moderate", parents=["L1-rs1-geasy-f050"],
             modifier={"camber_deg": -12.0})
    trial(v["id"], 1, False, "fall")
    # the pair rs1->rs2 at three difficulties: easy executed & failed, medium pruned (rs2 medium failed), hard pruned
    p_easy = cond("L2-rs1-to-rs2-g025", "L2", "pair", "failed", pair=["rs1", "rs2"], geometry_label="easy",
                  parents=["L1-rs1-geasy-f050", "L1-rs2-geasy-f050"])
    trial(p_easy["id"], 1, False, "fall", ["forbidden body contact", "base instability / fall"])
    trial(p_easy["id"], 2, True, "goal")
    trial(p_easy["id"], 3, False, "stuck", ["no forward progress"])
    cond("L2-rs1-to-rs2-g050", "L2", "pair", "pruned", pair=["rs1", "rs2"], geometry_label="medium",
         parents=["L1-rs1-gmedium-f050", "L1-rs2-gmedium-f050"], pruned_by=["L1-rs2-gmedium-f050"],
         prune_reason="one or both single-terrain geometry levels failed the configured qualification-friction gate")
    cond("L2-rs1-to-rs2-g075", "L2", "pair", "pruned", pair=["rs1", "rs2"], geometry_label="hard",
         parents=["L1-rs1-ghard-f050", "L1-rs2-ghard-f050"], pruned_by=["L1-rs1-ghard-f050", "L1-rs2-ghard-f050"],
         prune_reason="one or both single-terrain geometry levels failed the configured qualification-friction gate")
    for c in conds.values():
        n = len(c["attempts"]); ok = sum(1 for t in trials if t["condition_id"] == c["id"] and t["success"])
        c["summary"] = {"valid": n, "successes": ok, "failures": n - ok}
    return conds, trials


def _pair():
    from robustness_lab.evidence import pair_details
    conds, trials = _fixture()
    return pair_details(conds, trials)["rs1→rs2"]


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

def test_a_pair_lists_its_three_difficulties_in_order():
    d = _pair()
    assert [row["difficulty"] for row in d["levels"]] == ["easy", "medium", "hard"]


def test_an_executed_difficulty_says_how_it_went():
    easy = _pair()["levels"][0]
    assert easy["status"] == "failed" and easy["successes"] == 1 and easy["valid"] == 3
    assert easy["reasons"] == {"fall": 1, "goal": 1, "stuck": 1}


def test_each_difficulty_names_both_flat_parents_and_their_results():
    easy = _pair()["levels"][0]
    parents = {p["terrain"]: p for p in easy["parents"]}
    assert parents["rs1"]["status"] == "passed" and parents["rs1"]["successes"] == 3
    assert parents["rs2"]["status"] == "passed"
    assert all(p["kind"] == "primitive" for p in easy["parents"]), "a parent must be a FLAT terrain"


def test_a_pruned_difficulty_names_which_parent_failed():
    medium = _pair()["levels"][1]
    assert medium["status"] == "pruned"
    assert [p["terrain"] for p in medium["pruned_by"]] == ["rs2"]
    assert medium["pruned_by"][0]["status"] == "failed"
    hard = _pair()["levels"][2]
    assert sorted(p["terrain"] for p in hard["pruned_by"]) == ["rs1", "rs2"]


def test_variants_never_appear_as_parents():
    d = _pair()
    for row in d["levels"]:
        for p in row["parents"] + row["pruned_by"]:
            assert "curve" not in p["id"], "a banked variant leaked into a pair's parents"


def test_the_pair_carries_a_flat_only_note():
    d = _pair()
    assert "flat" in d["note"].lower() and ("banked" in d["note"].lower() or "curved" in d["note"].lower())


def test_the_pair_has_a_compact_hover_and_a_full_text():
    d = _pair()
    assert "easy" in d["hover"] and "pruned" in d["hover"]
    assert "RS2" in d["hover"] and "failed flat" in d["hover"].lower()


def test_a_pair_that_was_never_created_is_absent_not_erroneous():
    from robustness_lab.evidence import pair_details
    conds, trials = _fixture()
    assert "rs2→rs1" not in pair_details(conds, trials)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def _live():
    s = (ROOT / "src/robustness_lab/evidence.py").read_text()
    i = s.find("def _dashboard_html("); j = s.find("\ndef ", i + 10)
    return s[i:j]


def test_the_page_embeds_pair_details_and_uses_them():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert '"pair_details"' in src and "D.pair_details" in src


def test_pair_cells_have_hover_text():
    t = _live()
    k = t.find("data-map-pair=")
    assert "data-tip=" in t[k - 200:k + 400], "pair cells carry no hover text"


def test_clicking_a_cell_no_longer_scrolls_to_the_hidden_banner():
    t = _live()
    assert "$('selected-banner').scrollIntoView" not in t


def test_there_is_a_pair_popup():
    t = _live()
    assert 'id="pair-popup"' in t


def test_clicking_a_cell_opens_the_popup_with_the_parents(tmp_path):
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
        cell = pg.locator('[data-map-pair="rs1→rs2"]')
        cell.scroll_into_view_if_needed(); cell.click(); pg.wait_for_timeout(300)
        popup = pg.locator("#pair-popup")
        assert popup.is_visible(), "popup did not open"
        text = popup.inner_text()
        assert "RS1" in text and "RS2" in text
        assert "easy" in text.lower() and "hard" in text.lower()
        assert "flat" in text.lower()
        # The page has a global table{min-width:1050px}; the popup's table must
        # not inherit it, or the third column is clipped at the popup's edge.
        overflow = pg.evaluate("()=>{const t=document.querySelector('.pair-popup-table');"
                               "const b=document.getElementById('pair-popup');return t.scrollWidth-b.clientWidth}")
        assert overflow <= 0, "popup table overflows its box by %d px" % overflow
        pg.keyboard.press("Escape"); pg.wait_for_timeout(200)
        assert not popup.is_visible(), "popup did not close on Escape"
        b.close()
    assert not errors, errors

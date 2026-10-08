"""Dashboard layout: tiles first, then L1 detail, then the pair map -- always open.

Requested changes to the page under 'Whole-run accounting':

  * the L0-L7 pass-rate tiles move ABOVE the L1 banked/curved table, so the
    reader sees the whole progression before any one level's detail;
  * the 11x11 directional-pair map is no longer folded behind an 'Optional
    detail' disclosure -- it is the L2 result and is shown open;
  * the 'inspect one terrain-to-pair branch' section is removed from view.

The branch inspector's elements are kept in a hidden container rather than
deleted: the page's `$` is a bare getElementById, and the pair-map click
handlers write into those elements, so removing them would throw and stop
every script that runs afterwards. Hidden, they cost nothing and break nothing.
"""

import re
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _live_template():
    s = (ROOT / "src/robustness_lab/evidence.py").read_text()
    i = s.find("def _dashboard_html(")
    j = s.find("\ndef ", i + 10)
    assert i != -1
    return s[i:j]


def test_tiles_come_before_the_l1_variant_table():
    t = _live_template()
    assert t.find('id="levels"') < t.find('id="variant-panel"')


def test_the_pair_map_comes_after_the_l1_variant_table():
    t = _live_template()
    assert t.find('id="variant-panel"') < t.find('id="pair-map"')


def test_the_pair_map_is_not_folded_behind_a_disclosure():
    t = _live_template()
    assert "Optional detail: open the full 11" not in t
    # the map's container must not be inside a <details>
    k = t.find('id="pair-map"')
    opened = t.count("<details", 0, k)
    closed = t.count("</details>", 0, k)
    assert opened == closed, "pair map is nested inside an open <details>"


def test_the_branch_inspector_is_not_shown():
    t = _live_template()
    assert "Optional detail: inspect one terrain-to-pair branch" not in t
    assert "Inspect one branch in detail" not in t


def test_the_branch_inspector_elements_still_exist_for_the_scripts():
    """Deleting them would make $('branch-flow').innerHTML throw."""
    t = _live_template()
    for el in ("capability-tree", "selected-banner", "terrain-branches", "branch-flow", "tree-explanation"):
        assert 'id="%s"' % el in t, el
    k = t.find('id="capability-tree"')
    assert re.search(r"<div[^>]*\bhidden\b[^>]*>(?:(?!</div>).){0,400}id=\"capability-tree\"", t[max(0, k - 500):k + 40], re.S), (
        "branch inspector elements are not inside a hidden container")


def test_every_element_the_scripts_touch_exists_in_the_markup():
    """The first attempt removed the branch inspector's two toggle buttons along
    with it. `$('tree-compact').onclick=...` then threw, and because that runs
    before the pair map is drawn, the map rendered empty. A static check catches
    the whole class."""
    t = _live_template()
    ids = set(re.findall(r'id="([a-z0-9-]+)"', t))
    refs = set(re.findall(r"\$\('([a-z0-9-]+)'\)", t)) | set(re.findall(r"getElementById\('([a-z0-9-]+)'\)", t))
    assert not (refs - ids), "scripts reference elements that are not in the page: %s" % sorted(refs - ids)


def test_the_rendered_page_has_no_script_errors_and_shows_the_map_open(tmp_path):
    state_path = ROOT / "runs/submissions/go2_blind_v2/examination_state.json"
    if not state_path.is_file():
        pytest.skip("no v2 run on this machine")
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright
    from robustness_lab.evidence import refresh_evidence
    dashboard = refresh_evidence(str(state_path))["dashboard"]
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + dashboard)
        pg.wait_for_timeout(800)
        assert pg.locator("#pair-map").is_visible(), "pair map is not visible without clicking"
        assert pg.locator("#pair-map .pair-cell, #pair-map [class*=cell]").count() > 0, "pair map rendered empty"
        assert not pg.locator("#capability-tree").is_visible(), "branch inspector is still visible"
        levels_y = pg.locator("#levels").bounding_box()["y"]
        variant_y = pg.locator("#variant-panel").bounding_box()["y"]
        map_y = pg.locator("#pair-map").bounding_box()["y"]
        assert levels_y < variant_y < map_y
        b.close()
    assert not errors, "page threw: %s" % errors


def test_the_fourth_story_step_is_not_labelled_stop():
    """L4-L7 run now; the tab said '4 · Stop'."""
    assert "4 · Stop" not in _live_template()


def test_the_unsupported_sentence_is_omitted_when_nothing_was_unsupported():
    """'<b></b> was unsupported, so every pair containing it was left unmeasured'
    rendered with an empty subject whenever no terrain was unsupported."""
    t = _live_template()
    assert "</b> was unsupported, so every pair containing it" not in t
    assert "unsupportedTerrains.length?" in t

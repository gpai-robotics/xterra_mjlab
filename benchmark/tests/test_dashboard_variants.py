"""Banked/curved L1 variants ran; the dashboard must show them as such.

v2 executed 72 camber/curvature variants at L1 (28 passed, 44 failed). The page
had no concept of them: the L1 coverage tile counted by level, so they were
lumped into '171 cases' with no label, and the terrain matrix filtered to
kind == 'primitive', so they did not appear at all. The single most useful
comparison the variants exist to make -- does banking and curvature break a
terrain the policy walks flat? -- was not on the page anywhere.

Pairs stay at 330 by design: L2 is assembled from the plain terrains only, so
a banked failure cannot change what later levels are built from.
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cond(cid, level, kind, terrain, status, **extra):
    d = {"id": cid, "level": level, "kind": kind, "terrain": terrain, "status": status}
    d.update(extra)
    return d


def _conditions():
    c = {}
    for t, ok in (("rs1", "passed"), ("rs5", "passed"), ("rs8", "failed")):
        c["L1-%s-g050-f050" % t] = _cond("L1-%s-g050-f050" % t, "L1", "primitive", t, ok)
    # rs1: both variants pass; rs5: left passes, right fails; rs8 had no variants (failed flat)
    c["L1-rs1-g050-f050-left-curve-left-low"] = _cond(
        "L1-rs1-g050-f050-left-curve-left-low", "L1", "variant", "rs1", "passed",
        parents=["L1-rs1-g050-f050"], modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0})
    c["L1-rs1-g050-f050-right-curve-right-low"] = _cond(
        "L1-rs1-g050-f050-right-curve-right-low", "L1", "variant", "rs1", "passed",
        parents=["L1-rs1-g050-f050"], modifier={"camber_deg": 12.0, "curve_angle_deg": -65.0})
    c["L1-rs5-g050-f050-left-curve-left-low"] = _cond(
        "L1-rs5-g050-f050-left-curve-left-low", "L1", "variant", "rs5", "passed",
        parents=["L1-rs5-g050-f050"], modifier={"camber_deg": -12.0, "curve_angle_deg": 65.0})
    c["L1-rs5-g050-f050-right-curve-right-low"] = _cond(
        "L1-rs5-g050-f050-right-curve-right-low", "L1", "variant", "rs5", "failed",
        parents=["L1-rs5-g050-f050"], modifier={"camber_deg": 12.0, "curve_angle_deg": -65.0})
    return c


def _by_code(summary):
    return {row["code"]: row for row in summary["terrains"]}


def test_variants_are_counted_separately_from_plain_terrains():
    from robustness_lab.evidence import variant_summary
    s = variant_summary(_conditions())
    assert s["plain"]["executed"] == 3
    assert s["variants"]["executed"] == 4
    assert s["variants"]["passed"] == 3


def test_each_terrain_compares_flat_against_each_variant():
    from robustness_lab.evidence import variant_summary
    s = variant_summary(_conditions())
    rs5 = _by_code(s)["rs5"]
    assert rs5["plain"]["passed"] == 1
    assert rs5["left-curve-left-low"]["passed"] == 1
    assert rs5["right-curve-right-low"]["passed"] == 0
    assert rs5["right-curve-right-low"]["executed"] == 1


def test_a_terrain_that_failed_flat_has_no_variants_and_says_so():
    from robustness_lab.evidence import variant_summary
    s = variant_summary(_conditions())
    rs8 = _by_code(s)["rs8"]
    assert rs8["plain"]["passed"] == 0
    assert rs8["left-curve-left-low"]["executed"] == 0
    assert rs8["left-curve-left-low"]["status"] == "not_created"


def test_the_variant_name_is_derived_when_the_condition_lacks_it():
    """Runs materialised before a `variant` field existed must still be read."""
    from robustness_lab.evidence import variant_summary
    c = _conditions()
    for v in c.values():
        v.pop("variant", None)
    s = variant_summary(c)
    assert _by_code(s)["rs1"]["left-curve-left-low"]["executed"] == 1


def test_a_run_with_no_variants_reports_zero_not_error():
    from robustness_lab.evidence import variant_summary
    c = {k: v for k, v in _conditions().items() if v["kind"] == "primitive"}
    s = variant_summary(c)
    assert s["variants"]["executed"] == 0


def test_the_page_embeds_and_renders_the_variant_summary():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert '"variants"' in src, "variant summary is not embedded"
    assert "D.variants" in src, "the page does not render the variant summary"


def test_the_l1_tile_no_longer_lumps_variants_into_one_number():
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert "banked" in src.lower() or "curved" in src.lower(), (
        "the L1 tile does not label banked/curved variants")


def test_a_real_run_with_variants_embeds_their_numbers():
    state_path = ROOT / "runs/submissions/go2_blind_v2/examination_state.json"
    if not state_path.is_file():
        pytest.skip("no v2 run on this machine")
    from robustness_lab.evidence import variant_summary
    s = variant_summary(json.load(open(state_path))["conditions"])
    assert s["variants"]["executed"] == 72
    assert s["variants"]["passed"] == 28


# ---------------------------------------------------------------------------
# Readable, not just present
# ---------------------------------------------------------------------------
#
# Rendered and screenshotted, the first version of the table came out in
# alphabetical order (RS1, RS10, RS11, RS2, ...) because the page payload is
# json.dumps(sort_keys=True), which re-sorts a dict's keys; and it had no
# takeaway column, so the reader had to work out for themselves that RS1 is
# perfect flat and hopeless curved. The plain-text version with one note per
# terrain was clearer than the page. The page should be at least that clear.

def test_terrains_are_an_ordered_list_in_numeric_order():
    """A dict re-sorts under sort_keys=True; a list keeps RS1..RS11."""
    from robustness_lab.evidence import variant_summary
    c = _conditions()
    for t in ("rs10", "rs11"):
        c["L1-%s-g050-f050" % t] = _cond("L1-%s-g050-f050" % t, "L1", "primitive", t, "failed")
    s = variant_summary(c)
    assert isinstance(s["terrains"], list)
    assert [row["code"] for row in s["terrains"]] == ["rs1", "rs5", "rs8", "rs10", "rs11"]


def test_each_terrain_row_has_a_name_and_a_takeaway():
    from robustness_lab.evidence import variant_summary
    s = variant_summary(_conditions())
    for row in s["terrains"]:
        assert row["name"], row["code"]
        assert isinstance(row["takeaway"], str) and row["takeaway"], row["code"]


def _rows(**spec):
    """spec: code -> (flat_passed, flat_n, left_passed, left_n, right_passed, right_n)."""
    from robustness_lab.evidence import variant_summary
    c = {}
    for code, (fp, fn, lp, ln, rp, rn) in spec.items():
        for i in range(fn):
            c["L1-%s-%d" % (code, i)] = _cond("L1-%s-%d" % (code, i), "L1", "primitive", code,
                                              "passed" if i < fp else "failed")
        for name, (p, n) in (("left-curve-left-low", (lp, ln)), ("right-curve-right-low", (rp, rn))):
            for i in range(n):
                cid = "L1-%s-%d-%s" % (code, i, name)
                c[cid] = _cond(cid, "L1", "variant", code, "passed" if i < p else "failed",
                               parents=["L1-%s-%d" % (code, i)])
    return {row["code"]: row for row in variant_summary(c)["terrains"]}


def test_takeaway_calls_out_a_terrain_that_collapses_when_curved():
    row = _rows(rs1=(9, 9, 3, 9, 0, 9))["rs1"]
    t = row["takeaway"].lower()
    assert "flat" in t and ("collapse" in t or "curve" in t)


def test_takeaway_calls_out_one_sided_failure():
    row = _rows(rs5=(6, 9, 6, 6, 3, 6))["rs5"]
    t = row["takeaway"].lower()
    assert "left" in t and "right" in t


def test_takeaway_for_a_terrain_that_failed_flat():
    row = _rows(rs8=(0, 9, 0, 0, 0, 0))["rs8"]
    t = row["takeaway"].lower()
    assert "flat" in t and ("no variant" in t or "not re-run" in t or "not run" in t)


def test_takeaway_for_a_terrain_that_holds_up():
    row = _rows(rs2=(8, 9, 8, 8, 8, 8))["rs2"]
    assert "hold" in row["takeaway"].lower() or "unaffected" in row["takeaway"].lower()


def test_the_l1_story_splits_flat_from_banked():
    """'171 single-terrain conditions' lumped 72 banked results into the flat count
    while the chips beneath showed flat only. The story must say which is which."""
    src = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert "storyMetric(l1Coverage.total,'single-terrain conditions')" not in src
    assert "flat single-terrain" in src or "flat conditions" in src

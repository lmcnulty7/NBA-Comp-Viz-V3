"""tools.p8_placecause (docs/research/P8_placecause.md version 2): the conditions, the classes, the measures, the
720-order bounds and the reading rule, on synthetic inputs. No video, no model, no workspace."""

import itertools
import json

import numpy as np
import pytest

from tools import p8_placecause as C


def test_constants_are_the_registered_ones():
    assert (C.MISS_FT, C.NEAR_FT, C.EDGE_BOX_PX, C.EDGE_FOOT_PX, C.IOU_OVERLAP, C.CROWD_FT) == (2.5, 5.0, 2, 20, 0.3, 3.0)
    assert (C.HEIGHT_LO, C.HEIGHT_HI, C.FOOT_MOVE_PX, C.RESET_HORIZON, C.SINCE_CUT_S) == (0.75, 1.25, 8.0, 30, 1.0)
    assert (C.LINE_PCT, C.RESID_PCT, C.CARRY_RATIO, C.CARRY_SHARE, C.BOOT_N, C.SEED) == (20, 80, 1.5, 0.10, 2000, 0)
    assert C.SIX == ("C1", "C2", "C3", "C4", "C5", "C6") and C.MEASURED == ("gsw_cha_2016", "gsw_bkn_2015")
    assert C.GATE_ONLY == ("hou_cle_2016",)


def test_edges_overlap_recency_and_classes():
    assert C.box_at_edge([1, 100, 50, 300], 1280, 720) and C.box_at_edge([100, 100, 1278, 300], 1280, 720)
    assert C.box_at_edge([100, 100, 200, 718], 1280, 720) and not C.box_at_edge([3, 0, 1277, 717], 1280, 720)
    assert C.foot_near_edge([20, 300], 1280, 720) and C.foot_near_edge([640, 700], 1280, 720)
    assert not C.foot_near_edge([21, 699], 1280, 720)
    assert C.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and C.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert C.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert C.recent(130, [100], 30) and not C.recent(131, [100], 30) and not C.recent(99, [100], 30)
    assert C.classify(2.5, 9.0) == "hit" and C.classify(2.51, 5.0) == "placement" and C.classify(4.0, 5.01) == "detection"


def box(x1, y1, x2, y2, foot=None, raw=None):
    foot = foot or [(x1 + x2) / 2, y2]
    return {"bbox": [x1, y1, x2, y2], "foot_stab": foot, "foot_raw": raw or foot, "tid": 1}


def test_frame_conditions_on_a_synthetic_frame():
    # court coordinates are given directly (ct = projected feet); three players, three boxes
    sv = np.array([[10.0, 10.0], [13.5, 10.0], [40.0, 30.0], [80.0, 45.0]])
    inside = np.array([True, True, True, False])
    px = np.array([[100.0, 400.0], [200.0, 400.0], [1265.0, 500.0], [5.0, 5.0]])
    ct = np.array([[10.5, 10.0], [43.5, 30.0], [100.0, 0.0]])      # player 0 hit; player 2 at 3.5 ft: placement
    boxes = [box(80, 200, 120, 400), box(1240, 300, 1279, 500, foot=[1260, 500], raw=[1260, 490]),
             box(500, 100, 540, 200)]
    crowd = np.array([3.0, 3.0, 20.0, 30.0])
    heights = np.array([200.0, 200.0, 150.0, 100.0])
    third = ["near", "near", "far", "far"]
    o2o = {0: 0, 1: 2}
    res = {j: (cls, fl, k) for j, cls, fl, k in
           C.frame_conditions(sv, inside, px, ct, ct, boxes, crowd, heights, third, 1280, 720, o2o)}
    assert set(res) == {0, 1, 2}                                       # player 3 is not in frame
    assert res[0][0] == "hit" and res[0][1]["C1"] and res[0][2] == 0
    cls, fl, k = res[2]
    assert cls == "placement" and k == 1 and fl["C2_box"] and fl["C2_foot_truth"] and fl["C4_foot"] is True
    assert fl["C4_height_truth"] is True                               # 200 px box against 150 px expected
    assert not fl["C3b"] and not fl["nearest_is_other_1to1"]
    cls1, fl1, k1 = res[1]                                             # player 1: nearest box is player 0's box
    assert cls1 == "placement" and k1 == 0 and fl1["C3a_shared"] and fl1["nearest_is_other_1to1"] and not fl1["C3b"]
    assert fl1["C4_foot"] is False


def test_combine_and_stats_arithmetic():
    fl = {f: False for f in C.FLAGS} | {"C5a": False, "C5b": True, "C6": False, "D": True}
    c = C.combine({**fl, "C2_foot_truth": True})
    assert c["C2"] and c["C5"] and not c["C3"] and not c["C4"]
    tot = np.array([80.0, 15.0, 5.0, 100.0])                            # hit, placement, detection, in-frame
    wc = np.array([20.0, 9.0, 1.0, 30.0])
    s = C.stats(tot, wc)
    assert s["share_placement"] == pytest.approx(0.6) and s["share_hit"] == pytest.approx(0.25)
    assert s["enrichment"] == pytest.approx(2.4)
    assert s["rate_with"] == pytest.approx(0.3) and s["rate_without"] == pytest.approx(6 / 70)
    rr = 0.3 / (6 / 70)
    assert s["rate_ratio"] == pytest.approx(rr) and s["attributable_fraction"] == pytest.approx(0.6 * (1 - 1 / rr))


def rows_of(n_w=6, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_w):
        for _ in range(60):
            cls = rng.choice(["hit", "placement", "detection"], p=[0.7, 0.2, 0.1])
            fl = {c: bool(rng.random() < (0.5 if (c == "C2" and cls == "placement") else 0.15)) for c in C.SIX}
            rows.append((w, str(cls), fl))
    return rows


def test_exclusive_bounds_are_the_closed_form_over_720_orders():
    rows = rows_of()
    b = C.exclusive_bounds(rows)
    pl = [fl for _, cls, fl in rows if cls == "placement"]
    for c in C.SIX:
        hi = np.mean([fl[c] for fl in pl])
        lo = np.mean([fl[c] and not any(fl[o] for o in C.SIX if o != c) for fl in pl])
        assert b[c][0] == pytest.approx(lo) and b[c][1] == pytest.approx(hi)
    assert len(list(itertools.permutations(C.SIX))) == 720


def test_window_sums_bootstrap_and_reading():
    rows = rows_of()
    tot, wc = C.window_sums(rows, list(C.SIX))
    assert tot.shape == (6, 4) and tot[:, 3].sum() == len(rows)
    ci = C.bootstrap([tot], [wc["C2"]], n=200)
    pt = C.stats(tot.sum(0), wc["C2"].sum(0))
    assert ci["enrichment"][0] <= pt["enrichment"] <= ci["enrichment"][1]
    assert C.carries([1.6, 3.0], 0.10) and not C.carries([1.5, 3.0], 0.5) and not C.carries([2.0, 3.0], 0.09)


def test_summarize_reads_scattered_or_carried():
    def game(seed, strong):
        rng = np.random.default_rng(seed)
        rows = []
        for w in range(8):
            for _ in range(80):
                cls = str(rng.choice(["hit", "placement", "detection"], p=[0.7, 0.2, 0.1]))
                fl = {f: bool(rng.random() < 0.1) for f in C.FLAGS}
                if strong and cls == "placement":
                    fl["C2_box"] = bool(rng.random() < 0.8)
                rows.append([w, cls, fl, "near"])
        return {"rows": rows}
    s = C.summarize({"gsw_cha_2016": game(1, True), "gsw_bkn_2015": game(2, True)})
    assert s["reading"]["C2"]["carries"] and "C2" in s["verdict"]
    s = C.summarize({"gsw_cha_2016": game(3, False), "gsw_bkn_2015": game(4, False)})
    assert s["verdict"] == "SCATTERED"
    json.dumps(C.clean(s))


def test_refusals():
    with pytest.raises(SystemExit, match="not in this item"):
        C.collect("gsw_sac_2015")
    with pytest.raises(SystemExit, match="heavy-job lock"):
        C.collect("gsw_cha_2016")


def test_counts_only_matches_the_conditions_classes():
    sv = np.array([[10.0, 10.0], [13.5, 10.0], [40.0, 30.0]])
    inside = np.array([True, True, False])
    ct = np.array([[10.5, 10.0], [43.5, 30.0]])
    assert C.counts_only(sv, inside, ct) == [(0, "hit"), (1, "placement")]
    assert C.counts_only(sv, inside, np.zeros((0, 2))) == [(0, "detection"), (1, "detection")]


def test_d_is_reported_but_not_in_the_verdict():
    assert "D" not in C.RULED and set(C.RULED) == {"C1", "C2", "C3a", "C3b", "C4", "C5", "C6"}

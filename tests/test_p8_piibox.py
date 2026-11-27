"""tools.p8_piibox (docs/research/P8_piibox.md version 2): classes, track evidence, geometry, the D205 gate and the
report statistics, on synthetic data. No video, no model, no workspace."""

import numpy as np
import pytest

from tools import p8_piibox as M

W, H = 1280, 720
CAM = {"camera_C": [47.23, 137.06, -30.72], "camera_rvec": [-1.3447, -0.1647, -0.2669], "camera_f": 4300.0}


def test_classes_in_their_registered_order():
    c = M.classify
    assert c(3, 0, 0, False, 10) == "own"
    assert c(8, 2, 0, False, 10) == "own, with crossings"          # 2 of 10 is at most 0.2
    assert c(7, 3, 0, False, 10) == "switch"
    assert c(2, 8, 0, False, 10) == "another's, with crossings"
    assert c(0, 4, 5, True, 10) == "another's"                     # another's comes before own, misplaced
    assert c(0, 0, 1, True, 1) == "own, misplaced"                 # before every ghost
    assert c(0, 0, 0, True, 1) == "ghost, referee-flagged"         # referee before short
    assert c(0, 0, 0, False, 2) == "ghost, short"
    assert c(0, 0, 0, False, 3) == "ghost, persistent"
    assert set(M.CLASSES) == {c(*a) for a in [(3, 0, 0, 0, 9), (8, 2, 0, 0, 9), (7, 3, 0, 0, 9), (2, 8, 0, 0, 9),
                                               (0, 4, 0, 0, 9), (0, 0, 1, 0, 9), (0, 0, 0, 1, 9), (0, 0, 0, 0, 1),
                                               (0, 0, 0, 0, 9)]}


def frame(f, cids, A=None, Bm=None, ref=None, ft=None):
    n = len(cids)
    return {"f": f, "cid": cids, "A": A or {}, "B": Bm or {}, "ref": ref or [False] * n,
            "ft": np.asarray(ft if ft is not None else [[10.0 + i, 10.0] for i in range(n)], float)}


def test_evidence_excludes_the_frame_itself():
    frames = [frame(0, [7, 8], A={0: 101}),                          # track 7 with player 101
              frame(6, [7, 8], A={0: 101, 1: 102}),
              frame(12, [7], Bm={0: 101}),                           # track 7 is 101's stage B box
              frame(18, [7, 9], A={0: 103}, ref=[False, True])]      # track 7 with player 103
    t = M.track_table(frames)
    sv_of = {f: {101: np.array([10.0, 10.0])} for f in (0, 6, 12, 18)}
    ev = M.evidence(t[7], 101, 6, sv_of, fps=60.0)
    assert (ev["n_own"], ev["n_other"], ev["n_ownB"], ev["present"]) == (1, 1, 1, 4)
    assert ev["gap_own_s"] == pytest.approx(0.1) and ev["gap_other_s"] == pytest.approx(0.2)
    assert ev["top_other"] == 103 and ev["prox_share"] == pytest.approx(1.0) and not ev["referee_flagged"]
    ev9 = M.evidence(t[9], 101, 18, sv_of, fps=60.0)
    assert ev9["referee_flagged"] and ev9["present"] == 1 and ev9["prox_share"] is None
    assert M.classify(ev9["n_own"], ev9["n_other"], ev9["n_ownB"], ev9["referee_flagged"], ev9["present"]) == \
        "ghost, referee-flagged"


def test_proximity_share_counts_frames_where_the_track_is_near_the_player():
    frames = [frame(f, [5], ft=[[10.0 + (0 if f < 12 else 8), 10.0]]) for f in (0, 6, 12, 18)]
    t = M.track_table(frames)
    sv_of = {f: {101: np.array([10.0, 10.0])} for f in (0, 6, 12, 18)}
    assert M.evidence(t[5], 101, 0, sv_of, 60.0)["prox_share"] == pytest.approx(1 / 3)


def test_expected_height_grows_toward_the_camera_and_scales_with_height():
    from tools.p8_scorecard import project3d
    from sportvu.camera import h_px2ft
    Hc = np.asarray(h_px2ft(np.array(CAM["camera_C"]), np.array(CAM["camera_rvec"]), CAM["camera_f"], W, H)).reshape(3, 3)
    import cv2
    near = cv2.perspectiveTransform(np.array([[[640.0, 650.0]]]), Hc)[0, 0]
    far = cv2.perspectiveTransform(np.array([[[640.0, 300.0]]]), Hc)[0, 0]
    en, ef = M.expected_height(CAM, near, W, H), M.expected_height(CAM, far, W, H)
    assert en > ef > 0
    uv, _ = project3d(CAM["camera_C"], CAM["camera_rvec"], CAM["camera_f"], W, H, [[*near, 0.0], [*near, 3.25]])
    assert np.hypot(*(uv[1] - uv[0])) == pytest.approx(en / 2, rel=0.05)
    assert M.ratio_bin(0.7) == "short" and M.ratio_bin(1.0) == "normal" and M.ratio_bin(1.3) == "tall"
    assert M.ratio_bin(None) is None


def test_camera_check():
    from sportvu.camera import h_px2ft
    Hc = np.asarray(h_px2ft(np.array(CAM["camera_C"]), np.array(CAM["camera_rvec"]), CAM["camera_f"], W, H))
    row = {**CAM, "H_truth": (Hc.ravel() * 3.0).tolist()}
    assert M.camera_matches(row, W, H)
    bad = dict(row, H_truth=(Hc.ravel() * (1 + np.r_[1e-3, np.zeros(8)])).tolist())
    assert not M.camera_matches(bad, W, H)


def test_overlap_and_truncation():
    boxes = [[0, 0, 10, 10], [5, 0, 15, 10], [2, 2, 4, 4], [100, 100, 110, 110]]
    iou, ins = M.overlap(boxes, 0)
    assert iou == pytest.approx(50 / 150) and ins == pytest.approx(0.5)
    assert M.overlap(boxes, 2) == (pytest.approx(4 / 100), pytest.approx(1.0))
    assert M.overlap(boxes, 3) == (0.0, 0.0)
    assert M.truncation([100, 300, 140, 719], W, H) == (True, False)
    assert M.truncation([1, 300, 40, 500], W, H) == (False, True)
    assert M.truncation([100, 300, 140, 500], W, H) == (False, False)


def test_depth_is_positive_beyond_the_player():
    dep, lat = M.depth_lateral([0.0, 0.0, -30.0], [0.0, 12.0], [0.0, 10.0])
    assert dep == pytest.approx(2.0) and lat == pytest.approx(0.0)


def test_d205_gate():
    rows = [[0, {"class": "hit"}], [0, {"class": "P-ii", "stage_b_ft": 3.2}], [1, {"class": "P-ii", "stage_b_ft": 2.9}],
            [1, {"class": "P-ii", "stage_b_ft": 4.1}], [1, {"class": "D-i"}]]
    counts, sb = M.d205_window_counts(rows, 2)
    assert counts[1]["P-ii"] == 2 and counts[1]["D-i"] == 1 and sb[1] == [2.9, 4.1]
    assert M.gate_d205({"P-ii": 2, "D-i": 1}, [4.1, 2.9], counts[1], sb[1]) == []
    assert M.gate_d205({"P-ii": 2, "D-i": 1}, [4.1, 2.9 + 1e-6], counts[1], sb[1]) == ["stage B distances differ"]
    assert M.gate_d205({"P-ii": 2, "D-i": 1, "P-iii-free": 1}, [4.1, 2.9], counts[1], sb[1]) == ["class counts differ"]


def test_ratio_and_median_intervals():
    picks = M.picks_for([4], n=200)
    r = M.ratio_iv([np.array([1, 1, 1, 1])], [np.array([2, 2, 2, 2])], picks)
    assert r["value"] == 0.5 and r["interval"] == [0.5, 0.5]
    m = M.median_iv([[[1.0], [2.0], [3.0], []]], picks)
    assert m["value"] == 2.0 and 1.0 <= m["interval"][0] <= m["interval"][1] <= 3.0
    assert M.median_iv([[[], []]], M.picks_for([2], n=10)) == {"value": None, "interval": None}


def synthetic_game(rng, n_w=6):
    rows = []
    for wi in range(n_w):
        for kind, n in (("P-ii", 20), ("hit", 60)):
            for _ in range(n):
                cl = M.CLASSES[int(rng.integers(0, len(M.CLASSES)))]
                rows.append({"wi": wi, "kind": kind, "class": cl, "n_own": 1, "n_other": 0, "n_ownB": 0,
                             "prox_share": float(rng.random()), "gap_own_s": 0.1, "gap_other_s": None,
                             "partner_in_frame": None, "partner_ft": None, "h": 180.0,
                             "ratio": float(rng.uniform(0.5, 1.5)), "trunc_bottom": bool(rng.random() < 0.1),
                             "trunc_other": False, "iou_ge_0.3": bool(rng.random() < 0.3), "inside_ge_0.5": False,
                             "scorer_ghost": False, "depth": -1.0, "lateral": 0.5, "third": "middle"})
    for r in rows:
        r["bin"] = M.ratio_bin(r["ratio"])
    return {"windows": n_w, "rows": rows, "in_frame": 1000}


def test_summarize_shares_add_up_and_groups_hold():
    rng = np.random.default_rng(1)
    per = {M.CHA: synthetic_game(rng), M.BKN: synthetic_game(rng)}
    s = M.summarize(per)
    for name in (M.CHA, M.BKN, "pooled"):
        for kind in ("P-ii", "hit"):
            sh = s[name][kind]["shares"]
            assert sum(sh[c]["value"] for c in M.CLASSES) == pytest.approx(1.0)
            assert sum(sh[g]["value"] for g in M.GROUPS) == pytest.approx(1.0)
            assert all(sh[c]["interval"][0] <= sh[c]["value"] + 1e-12 for c in M.CLASSES)
    assert s["pooled"]["P-ii"]["n"] == 240 and s["pooled"]["hit"]["n"] == 720
    assert sum(s["pooled"]["P-ii"]["rates_over_in_frame"].values()) == pytest.approx(240 / 2000)
    g = s["pooled"]["P-ii"]["geometry"]["all"]
    assert g["n"] == 240 and 0.5 <= g["ratio_median"]["value"] <= 1.5
    assert g["ratio_short"]["value"] + g["ratio_normal"]["value"] + g["ratio_tall"]["value"] == pytest.approx(1.0)


def test_constants_match_the_design():
    from tools import p8_one2one as O
    assert (M.MISS_FT, M.NEAR_FT) == (O.MISS_FT, O.NEAR_FT) and M.BOOT_N == O.BOOT_N and M.SEED == O.SEED
    assert (M.CROSS_FRAC, M.SHORT_FRAMES, M.PROX_FT, M.PROX_SHARE) == (0.2, 3, 5.0, 0.5)
    assert (M.RATIO_BINS, M.IOU_BAR, M.INSIDE_BAR, M.PERSON_FT) == ((0.75, 1.25), 0.3, 0.5, 6.5)
    assert M.MEASURED == (M.CHA, M.BKN) and M.GATE_ONLY == (M.HOU,)

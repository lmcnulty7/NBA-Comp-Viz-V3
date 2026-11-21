"""tools.p8_teachers: the matching, the matched-precision sweep, the qualification rule, the teacher-only band and the
guard, on synthetic boxes. No model, no image."""

import numpy as np
import pytest

import config
from tools import p8_teachers as T

G = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]


def test_constants_follow_config_and_the_colab_qualification():
    assert T.DET_CONF == config.PLAYER_CONF and T.DET_NMS_IOU == config.PLAYER_IOU
    assert T.IOU == 0.5
    from colab_autolabel import AGREE_IOU, iou as colab_iou, pr_vs_gt
    assert T.IOU == AGREE_IOU
    a, b = [0, 0, 10, 10], [5, 0, 15, 10]
    assert T.iou(a, b) == pytest.approx(colab_iou(a, b))
    preds = [[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 31, 10], [60, 0, 70, 10]]
    flags, _ = T.greedy(preds, G)
    assert sum(flags) == pr_vs_gt(preds, G)["tp"]


def test_iou_and_the_half_threshold_counts_as_a_match():
    assert T.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert T.iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0
    half = [0, 0, 10, 10], [0, 0, 10, 5]          # IoU exactly 0.5
    assert T.iou(*half) == 0.5
    flags, used = T.greedy([half[1]], [half[0]])
    assert flags == [True] and used == {0}


def test_greedy_is_one_to_one_in_the_given_order():
    preds = [[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [21, 0, 31, 10, 0.7]]
    flags, used = T.greedy(preds, G)
    assert flags == [True, False, True] and used == {0, 1}
    # a truth index in skip is never matched
    flags, used = T.greedy(preds, G, skip={0})
    assert flags == [False, False, True] and used == {1}


def test_prefix_property_a_lower_threshold_never_changes_earlier_flags():
    rng = np.random.default_rng(0)
    gts = [[x, 0, x + 10, 20] for x in range(0, 200, 25)]
    preds = [[g[0] + rng.normal(0, 3), rng.normal(0, 3), g[2] + rng.normal(0, 3), 20, float(rng.random())]
             for g in gts for _ in range(2)]
    full_scores, full_flags = T.frame_flags(preds, gts)
    for t in (0.2, 0.5, 0.8):
        sub = [p for p in preds if p[4] >= t]
        s, f = T.frame_flags(sub, gts)
        assert f == full_flags[: len(f)] and s == full_scores[: len(s)]


def test_curve_pools_frames_and_keeps_tied_scores_together():
    frames = [([0.9, 0.5], [True, False]), ([0.9, 0.3], [True, True])]
    c = T.Curve(frames)
    thr, prec, rec = c.at(np.ones(2), 4.0)
    assert list(thr) == [0.9, 0.5, 0.3]                       # the two 0.9 boxes form one run
    assert list(prec) == pytest.approx([1.0, 2 / 3, 3 / 4])
    assert list(rec) == pytest.approx([0.5, 0.5, 0.75])
    thr, prec, rec = c.at(np.array([2.0, 0.0]), 4.0)          # frame 0 twice, frame 1 dropped
    assert list(prec) == pytest.approx([1.0, 0.5, 0.5]) and list(rec) == pytest.approx([0.5, 0.5, 0.5])


def test_recall_at_precision_takes_the_best_recall_that_holds_the_precision():
    thr = np.array([0.9, 0.7, 0.5, 0.3])
    prec = np.array([1.0, 0.96, 0.95, 0.80])
    rec = np.array([0.40, 0.80, 0.90, 0.99])
    assert T.recall_at_precision(thr, prec, rec, 0.95) == (0.90, 0.5)
    assert T.recall_at_precision(thr, prec, rec, 0.97) == (0.40, 0.9)
    assert T.recall_at_precision(thr, prec, rec, 1.01) == (0.0, None)
    # equal recall: the higher threshold
    assert T.recall_at_precision(thr, prec, np.array([0.5, 0.9, 0.9, 0.9]), 0.8) == (0.9, 0.7)


def test_qualification_needs_the_point_and_the_whole_interval_above_zero():
    assert T.qualifies(0.02, 0.001)
    assert not T.qualifies(0.02, 0.0)
    assert not T.qualifies(0.02, -0.01)
    assert not T.qualifies(0.0, 0.01)
    assert not T.qualifies(-0.01, -0.02)


def _synthetic(better: bool):
    """Our detector: 8 of 10 truth boxes per frame, 1 false box. The teacher: all 10 (better) or 6 (worse), with
    false boxes below its true ones."""
    rng = np.random.default_rng(1)
    gts, det, tea = [], [], []
    for _ in range(40):
        g = [[x, 0, x + 10, 30] for x in range(0, 300, 30)]
        gts.append(g)
        det.append([[*b, 0.9, "player"] for b in g[:8]] + [[500, 0, 510, 30, 0.95, "player"]])
        n = 10 if better else 6
        tea.append([[*b, float(0.5 + 0.5 * rng.random()), "person"] for b in g[:n]]
                   + [[600, 0, 610, 30, 0.2, "person"]])
    return gts, det, tea


@pytest.mark.parametrize("better", [True, False])
def test_the_rule_on_a_better_and_a_worse_synthetic_teacher(better):
    gts, det, tea = _synthetic(better)
    n_gt_f = np.array([len(g) for g in gts], float)
    counts = T.detector_counts(det, gts)
    curve = T.Curve([T.frame_flags(p, g) for p, g in zip(tea, gts)])
    fw, _ = T.resamples(len(gts), ["a"] * 20 + ["b"] * 20, b=200)
    pt = T.qualify_one(curve, counts, n_gt_f, np.ones((1, len(gts))))
    fb = T.qualify_one(curve, counts, n_gt_f, fw)
    lo, _ = T.percentile_interval(fb["diff"])
    assert pt["p_det"][0] == pytest.approx(8 / 9) and pt["r_det"][0] == pytest.approx(0.8)
    if better:
        assert pt["r_t"][0] == pytest.approx(1.0) and pt["thr"][0] >= 0.5
        assert T.qualifies(float(pt["diff"][0]), lo)
    else:
        assert pt["r_t"][0] == pytest.approx(0.6)
        assert not T.qualifies(float(pt["diff"][0]), lo)


def test_teacher_only_band_counts_only_truth_our_detector_missed():
    gts = [[0, 0, 10, 30], [30, 0, 40, 30], [60, 0, 70, 30]]
    det = [[0, 0, 10, 30, 0.9]]                                     # finds truth 0
    tea = [[0, 0, 10, 30, 0.8],                                     # agrees with ours: not in the band
           [30, 0, 40, 30, 0.7],                                    # a missed player: true
           [90, 0, 100, 30, 0.6],                                   # nothing there: false
           [60, 0, 70, 30, 0.1]]                                    # below the threshold
    only, flags = T.band(tea, det, gts, 0.5)
    assert [o[0] for o in only] == [30, 90] and flags == [True, False]
    other = [[31, 0, 41, 30, 0.9], [200, 0, 210, 30, 0.9]]
    agreed, f2 = T.pair_band(only, other, det, gts)
    assert len(agreed) == 1 and agreed[0][:4] == [30.5, 0, 40.5, 30] and f2 == [True]


def test_resamples_are_seeded_and_the_clip_bootstrap_moves_whole_clips():
    a = T.resamples(6, ["x", "x", "y", "y", "z", "z"], b=50)
    b = T.resamples(6, ["x", "x", "y", "y", "z", "z"], b=50)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert (a[0].sum(axis=1) == 6).all()
    cw = a[1]
    assert (cw[:, 0] == cw[:, 1]).all() and (cw[:, 2] == cw[:, 3]).all() and (cw.sum(axis=1) == 6).all()


def test_the_guard_drops_test_and_spent_items_and_keeps_the_box_truth_games():
    clip_game, games = T.clip_games()
    assert set(clip_game.values()) == {"201602270OKC", "201206070BOS"}
    ok = {"frame": "clip_10m00_18m00__f000315.jpg", "image_path": "x/clip_10m00_18m00__f000315.jpg"}
    assert T.check_record(ok, clip_game, games) is None
    spent = {"frame": "clip_10m00_18m00__f1.jpg", "image_path": "x/12.16.2015.PHX.at.GSW/f1.jpg"}
    assert T.check_record(spent, clip_game, games) is not None
    test_arena = {"frame": "clip_10m00_18m00__f1.jpg", "image_path": "x/01.01.2016.NOP.at.MIA/f1.jpg"}
    assert "test" in T.check_record(test_arena, clip_game, games)
    unknown = {"frame": "other_clip__f1.jpg", "image_path": "x/f1.jpg"}
    assert T.check_record(unknown, clip_game, games) == "no known game for its clip"
    # a test-arena home is refused through the homes argument too
    games2 = {**games, "201206070BOS": {**games["201206070BOS"], "home": "MIA"}}
    assert T.check_record(ok, clip_game, games2) is not None


def test_the_workspace_is_outside_the_repository(monkeypatch):
    monkeypatch.setenv("V3_P8_TEACHERS", str(T.REPO / "reports" / "x"))
    with pytest.raises(SystemExit):
        T.workspace()


def test_teachers_record_a_license_and_run_without_a_fit():
    for spec in T.TEACHERS.values():
        assert spec["license"] and spec["weights"] and 0 < spec["default_conf"] < 1
    src = (T.REPO / "tools" / "p8_teachers.py").read_text()
    for word in (".train(", ".fit(", "optimizer.step"):
        assert word not in src


# ── D131 source 2: misses on camera-truth frames ───────────────────────────────────────────────────────────────────
def test_miss_constants_follow_the_frozen_rule_and_the_qualified_threshold():
    from sportvu.autolabel import CONFIRM_FT
    assert T.CONFIRM_FT == CONFIRM_FT == 2.5
    assert T.MISS_TEACHER in T.TEACHERS and T.MISS_TEACHER_CONF == 0.1471


def test_stratified_pick_takes_one_per_stratum_in_time_order_and_is_seeded():
    p = T.stratified_pick(1000, 10)
    assert p == T.stratified_pick(1000, 10) and len(p) == 10
    assert all(100 * k <= x < 100 * (k + 1) for k, x in enumerate(p))
    assert T.stratified_pick(5, 10) == [0, 1, 2, 3, 4]


def test_feet_and_targets_through_a_homography():
    H = np.diag([0.1, 0.1, 1.0])                     # px -> ft: 10 px a foot
    assert T.feet_ft([[0, 0, 20, 50]], H).tolist() == [[1.0, 5.0]]
    sv = np.array([[1.0, 5.0], [200.0, 5.0], [-1.0, 5.0]])
    assert T.targets_in_frame(sv, H, 1280, 720).tolist() == [[1.0, 5.0]]


def test_found_targets_is_one_to_one_within_the_gate():
    targets = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    feet = np.array([[0.5, 0.0], [0.6, 0.0], [12.6, 0.0]])
    # two feet near target 0: only one can take it; the foot 2.6 ft from target 1 is outside the gate
    assert T.found_targets(feet, targets) == {0}
    assert T.found_targets(np.array([[0.5, 0.0], [10.0, 2.4]]), targets) == {0, 1}
    assert T.found_targets(np.zeros((0, 2)), targets) == set()

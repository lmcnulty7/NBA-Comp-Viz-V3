"""Scorecard v2 of P8 Phase A (docs/research/P8_FEASIBILITY.md section 7 (4)): the pin guard and the pure parts."""

import json

import numpy as np
import pytest

from tools import p8_pin
from tools import p8_scorecard as S


def test_the_scorer_its_constants_and_the_frozen_scorer_match_the_committed_pin():
    """Fails on a branch that changes tools/p8_scorecard.py, its CONSTANTS, a frozen file it scores with or a frozen
    constant it reads. Changing them is allowed only with a new pin (python3 -m tools.p8_pin --write) in the same
    reviewed commit, so the change is visible to the lead and the judge."""
    pinned = json.loads(p8_pin.PIN_FILE.read_text())
    now = p8_pin.current()
    assert now["scorer"] == pinned["scorer"], "tools/p8_scorecard.py changed since the pin"
    assert now["constants_sha256"] == pinned["constants_sha256"], "scorecard v2 CONSTANTS changed since the pin"
    assert now["frozen_files"] == pinned["frozen_files"], "a frozen file of the scorer changed"
    assert now["frozen_constants"] == pinned["frozen_constants"], "a frozen scorer constant changed"


def test_the_section_6_2_constants_are_the_documented_ones():
    C = S.CONSTANTS
    assert (C["holder_max_ft"], C["holder_ball_max_z_ft"], C["event_tol_s"]) == (3.0, 6.0, 1.0)
    assert (C["ball_f1_target"], C["carrier_target"], C["identity_target"]) == (0.75, 0.85, 0.80)
    assert (C["shots_found_target"], C["shots_false_max"], C["rebounds_found_target"]) == (0.95, 0.05, 0.90)
    assert (C["shooter_p50_ft"], C["shooter_p90_ft"], C["rebounder_p50_ft"]) == (3.0, 6.0, 4.0)
    assert C["jersey_min_crops"] == 8 and C["switches_goal_per_possession"] == 5.0


def test_bins_and_zones():
    e = [3.0, 6.0, 10.0]
    assert [S.bin_of(v, e) for v in (0.0, 3.0, 5.9, 10.0, 40.0)] == ["<3", "3-6", "3-6", ">=10", ">=10"]
    assert S.bin_of(None, e) is None and S.bin_of(float("nan"), e) is None
    assert S.zone_of(5.25, 25.0) == "<4" and S.zone_of(88.75, 35.0) == "4-16"
    assert S.zone_of(5.25 + 20, 25.0) == "16-23.75" and S.zone_of(47.0, 25.0) == ">=23.75"


def test_crowding_and_holder():
    sv = np.array([[10, 10], [12, 10], [40, 40], [80, 25]], np.float32)
    assert np.allclose(S.crowding(sv)[:2], [2, 2])
    assert S.holder(np.array([40.5, 40.0, 4.0]), sv) == 2
    assert S.holder(np.array([40.5, 40.0, 7.0]), sv) is None          # ball too high
    assert S.holder(np.array([11.0, 10.0, 4.0]), sv) is None          # two players within 3 ft
    assert S.holder(np.array([60.0, 10.0, 4.0]), sv) is None          # nobody near
    assert S.holder(np.array([np.nan, 1, 1]), sv) is None


def test_cuts_and_time_since_cut():
    frames = [0, 3, 6, 9, 12]
    corr = [None, 0.95, 0.2, 0.9, 0.59]
    cuts = S.cuts_from_corr(frames, corr)
    assert cuts == [6, 12]
    assert S.since_cut(9, cuts, 0, 30.0) == (0.1, False)
    assert S.since_cut(3, cuts, 0, 30.0) == (0.1, True)               # before the first cut: censored


def test_f1_and_event_matching_are_one_to_one():
    assert S.f1(0, 0, 5) is None and S.f1(8, 2, 2) == 0.8
    truth = [(1, 700.0), (1, 650.0), (2, 600.0)]
    pred = [(1, 699.5), (1, 699.0), (2, 598.5), (2, 300.0)]
    found, false, pairs = S.match_events(truth, pred, 1.0)
    assert found == 1 and false == 3 and pairs[0][:2] == (0, 0)
    found, false, _ = S.match_events(truth, [(1, 650.4), (2, 600.9), (1, 701.0)], 1.0)
    assert (found, false) == (3, 0)


def test_cluster_bootstrap_is_seeded_and_needs_two_groups():
    groups = {"a": [1.0, 2.0], "b": [3.0], "c": [10.0, 11.0]}
    stat = lambda p: float(np.median(p)) if p else None
    one = S.boot_windows(groups, stat, n=200)
    assert one == S.boot_windows(groups, stat, n=200) and one[0] <= one[1]
    assert S.boot_windows({"a": [1.0]}, stat) is None


# A camera-truth H of a real R1.3 label (gsw_bkn_2015, s00_w00, frame 120; px -> ft), 1280 x 720.
H_LABEL = [0.1715445705296883, 0.2643684746182778, -170.914688644882, -0.015034947590733308, 0.7122753653274643,
           -212.01520180800415, 0.0002780259742405138, 0.005364810186631225, 1.0]


def test_ball_projection_agrees_with_the_ground_plane_and_puts_height_up():
    cv2 = pytest.importorskip("cv2")
    from sportvu.camera import pose_from_h
    W, H = 1280, 720
    rvec, f, C = pose_from_h(H_LABEL, W, H)
    assert C[2] < 0                                                   # the camera sits at negative z: height is -z
    ground = np.array([[30.0, 20.0], [47.0, 25.0], [70.0, 30.0]])
    P = np.linalg.inv(np.array(H_LABEL).reshape(3, 3))
    want = cv2.perspectiveTransform(ground.reshape(-1, 1, 2), P).reshape(-1, 2)
    uv0, depth = S.project3d(C, rvec, f, W, H, np.column_stack([ground, np.zeros(3)]))
    assert (depth > 0).all() and np.allclose(uv0, want, atol=1.0)
    uv10, _ = S.project3d(C, rvec, f, W, H, np.column_stack([ground, np.full(3, 10.0)]))
    assert (uv10[:, 1] < uv0[:, 1]).all()                             # 10 ft up draws higher in the image


def test_third_of_clamps_points_just_off_the_court():
    assert S.third_of(-2.0, True) == "far" and S.third_of(52.0, True) == "near"
    assert S.third_of(-2.0, False) == "near" and S.third_of(25.0, False) == "middle"

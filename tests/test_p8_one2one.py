"""tools.p8_one2one (docs/research/P8_one2one.md version 2): the copies' equivalence with the scorer's one-to-one
functions at both gates, the two-stage assignment, every class, the impossibility counters and the measures, on
synthetic frames. No video, no model, no workspace."""

import numpy as np
import pytest

from tools import p8_one2one as O
from tools.p8_resetfix import one_to_one_missed, unmatched_1to1


def random_frame(rng):
    sv = rng.uniform([0, 0], [94, 50], (10, 2))
    nb = int(rng.integers(0, 12))
    feet = sv[rng.integers(0, 10, nb)] + rng.normal(0, 2.5, (nb, 2)) if nb else np.zeros((0, 2))
    is_ref = rng.random(nb) < 0.15
    inside = rng.random(10) < 0.8
    return feet, sv, inside, is_ref


def test_the_copies_equal_the_scorers_functions_at_both_gates():
    rng = np.random.default_rng(0)
    for _ in range(400):
        feet, sv, inside, is_ref = random_frame(rng)
        for gate in (2.5, 5.0):
            assert O.unmatched_count(O.assign_1to1(feet, sv, is_ref, gate), inside, is_ref) == \
                unmatched_1to1(feet, sv, inside, is_ref, gate)
        assert int(inside.sum()) - len(O.assign_inframe(feet, sv, inside, 2.5)) == one_to_one_missed(feet, sv, inside, 2.5)


def test_two_stage_gives_a_claimed_box_to_its_player_and_an_unclaimed_one_at_5ft():
    # player i (index 0) and player j (index 1); box k is 1 ft from i and 2.83 ft from j; box m is 4 ft from j
    sv = np.array([[10.0, 10.0], [12.0, 13.0]])
    feet = np.array([[10.0, 11.0], [12.0, 17.0]])           # k, m
    inside = np.array([True, True])
    is_ref = np.array([False, False])
    out = O.frame_classes(feet, sv, inside, is_ref)
    assert out[0]["class"] == "hit"
    assert out[1]["class"] == "P-ii" and out[1]["stage_b_ft"] == pytest.approx(4.0)
    # without box m, j has no box left: its nearest box k belongs to i
    out = O.frame_classes(feet[:1], sv, inside, is_ref[:1])
    assert out[1]["class"] == "P-i" and out[1]["claimer_in_frame"] and out[1]["claimer_ft"] == pytest.approx(np.hypot(2, 3))
    assert out[1]["unclaimed_5_10ft"] is False


def test_single_stage_can_differ_from_two_stage():
    # i at 0 with box k 2.4 ft away (stage A gives k to i) and box m 2.6 ft away on the other side; j 2.9 ft from k
    sv = np.array([[0.0, 0.0], [5.3, 0.0]])
    feet = np.array([[2.4, 0.0], [-2.6, 0.0]])               # k, m
    inside = np.array([True, True])
    out = O.frame_classes(feet, sv, inside, np.array([False, False]))
    assert out[0]["class"] == "hit"
    assert out[1]["class"] == "P-i"                              # two stages: k went to i at 2.4 ft; m is 7.9 ft from j
    assert out[1]["single"] == "P-ii"                            # one 5 ft stage: i-m 2.6 and j-k 2.9 cost less in total


def test_referee_and_empty_and_far_classes():
    sv = np.array([[10.0, 10.0], [40.0, 40.0]])
    inside = np.array([True, True])
    out = O.frame_classes(np.array([[13.0, 10.0]]), sv, inside, np.array([True]))
    assert out[0]["class"] == "P-iii-ref" and out[1]["class"] == "D-iii-ref"
    assert out[0]["inframe"] == "P-ii"                             # the in-frame convention keeps referee boxes
    out = O.frame_classes(np.zeros((0, 2)), sv, inside, np.zeros(0, bool))
    assert out[0]["class"] == "D-iii-none" and out[1]["class"] == "D-iii-none"
    out = O.frame_classes(np.array([[17.0, 10.0]]), sv[:1], inside[:1], np.array([False]))
    assert out[0]["class"] == "D-iii-free"                         # an unclaimed box 7 ft away
    two = O.frame_classes(np.array([[10.0, 11.0]]), np.array([[10.0, 10.0], [10.0, 18.0]]), inside, np.array([False]))
    assert two[1]["class"] == "D-i" and two[1]["claimer_in_frame"]


def test_classify_counters():
    is_ref = np.array([False, False])
    assert O.classify(3.0, 0, is_ref, (0, 2.0, "A"), {0: 1}, 1, 2) == "placement_matched_in_stage_A"
    assert O.classify(3.0, 0, is_ref, None, {}, 1, 2) == "P-iii-free"
    assert O.classify(7.0, 0, is_ref, (0, 4.0, "B"), {0: 1}, 1, 2) == "detection_matched"
    assert set(O.COUNTERS) == {"placement_matched_in_stage_A", "P-iii-free", "detection_matched"}


def test_shares_add_up_and_boot_brackets():
    recs = [(0, {"class": "hit"})] * 70 + [(0, {"class": "P-i"})] * 10 + [(1, {"class": "P-ii"})] * 6 + \
           [(1, {"class": "P-iii-ref"})] * 4 + [(1, {"class": "D-i"})] * 5 + [(0, {"class": "D-iii-none"})] * 5
    names, M = O.window_matrix(recs)
    tot = M.sum(0)
    s = O.shares(names, tot)
    assert sum(s[c]["share"] for c in O.P_CLASSES) == pytest.approx(1.0)
    assert sum(s[c]["share"] for c in O.D_CLASSES) == pytest.approx(1.0)
    assert s["P-i"]["rate"] == pytest.approx(0.1) and s["D-i"]["share"] == pytest.approx(0.5)
    ci = O.boot([M], names, n=200)
    assert ci["P-i"]["share"][0] <= s["P-i"]["share"] <= ci["P-i"]["share"][1]


def test_constants_git_and_refusals():
    assert (O.MISS_FT, O.NEAR_FT, O.FAR_REPORT_FT, O.BOOT_N, O.SEED) == (2.5, 5.0, 10.0, 2000, 0)
    assert O.MEASURED == ("gsw_cha_2016", "gsw_bkn_2015") and O.GATE_ONLY == ("hou_cle_2016",)
    g = O.git_head()
    assert len(g["head"]) == 40
    with pytest.raises(SystemExit, match="not in this item"):
        O.collect("gsw_sac_2015")
    with pytest.raises(SystemExit, match="heavy-job lock"):
        O.collect("gsw_cha_2016")


def test_stage_a_pairs_equal_assign_1to1_on_random_frames():
    rng = np.random.default_rng(3)
    for _ in range(200):
        feet, sv, inside, is_ref = random_frame(rng)
        O.frame_classes(feet, sv, inside, is_ref)                   # raises on a stage A mismatch (M1)

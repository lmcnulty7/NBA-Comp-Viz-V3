"""tools.p8_truth_v2: the v2 offset rule, the stability measures and the registered bar, on synthetic data. No video,
no SportVU file."""

import numpy as np
import pytest

from tools import p8_truth_v2 as V


def test_trimmed_match_is_one_to_one_and_drops_the_worst_fifth():
    sv = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]])
    feet = np.array([[1.0, 0.0], [1.5, 0.0], [21.0, 0.0], [31.0, 0.0], [49.0, 0.0]])
    # one to one: the second foot cannot take player 0 too; the worst pair (9 ft) is dropped (5 pairs, floor(1.0) = 1)
    d = V.trimmed_match_mean(feet, sv)
    assert d == pytest.approx(np.mean(sorted([1.0, 8.5, 1.0, 1.0, 9.0])[:4]))


def test_estimate_is_the_centre_of_a_flat_bottom_not_its_argmin():
    offs = np.round(np.arange(-1.5, 1.5001, 0.04), 2)
    J = 2.0 + 0.5 * (offs - 0.4) ** 2
    J[np.abs(offs - 0.4) <= 0.3] = 2.0                         # a flat bottom from 0.1 to 0.7 s
    J[offs == 0.66] = 1.999                                     # its argmin sits at one end
    e = V.v2_estimate(offs, J)
    assert abs(e["offset_s"] - 0.4) < 0.05 and not e["at_edge"]
    edge = V.v2_estimate(offs, 2.0 + 0.5 * (offs + 1.6) ** 2)
    assert edge["at_edge"]
    assert V.v2_estimate(offs, np.full(len(offs), np.nan))["offset_s"] is None


def test_stability_and_the_paired_share_interval():
    fz = [(0.0, 0.5), (0.1, 0.6), (0.2, 0.2), (0.0, 0.9), (0.3, 0.3)] * 10
    v2 = [(0.0, 0.05), (0.1, 0.12), (0.2, 0.2), (0.0, 0.3), (0.3, 0.3)] * 10
    s = V.stability(fz)
    assert s["n"] == 50 and s["share_over_0_2_s"] == pytest.approx(0.6) and s["max_abs_s"] == 0.9
    assert V.stability(v2)["share_over_0_2_s"] == pytest.approx(0.2)
    lo, hi = V.share_reduction_interval(fz, v2)
    assert 0 < lo <= 0.4 <= hi
    assert V.stability([(None, 0.1)])["n"] == 0


def _win(f3, f23, v3, v23, cf, cv, stored=None):
    return {"frozen": {"v3": f3, "r23": f23}, "stored_frozen_v3": f3 if stored is None else stored,
            "v2": {"v3": {"offset_s": v3, "at_edge": False}, "r23": {"offset_s": v23, "at_edge": False}},
            "candidates_v3": {"frozen": cf, "v2": cv}}


def test_the_bar_passes_a_stable_rule_and_fails_one_that_loses_candidates():
    good = {t: {f"w{i}": _win(0.0, 0.6 if i % 2 else 0.1, 0.2, 0.25, 50, 50) for i in range(20)} for t in V.POOLED}
    r = V.report(good)
    assert r["bar"]["pass"] and r["pooled"]["v2"]["median_abs_s"] < r["pooled"]["frozen"]["median_abs_s"]
    lossy = {t: {f"w{i}": _win(0.0, 0.6 if i % 2 else 0.1, 0.2, 0.25, 50, 40) for i in range(20)} for t in V.POOLED}
    assert not V.report(lossy)["bar"]["checks"]["candidates_at_least_95_percent"]
    # a window whose recomputed frozen offset differs from the stored one is left out of the bar
    good["gsw_bkn_2015"]["w0"]["stored_frozen_v3"] = 0.9
    assert V.report(good)["per_game"]["gsw_bkn_2015"]["frozen_recompute_mismatch"] == ["w0"]
    assert V.report({"gsw_bkn_2015": good["gsw_bkn_2015"]})["bar"] is None


def test_registered_constants_and_guards():
    assert V.DELTA_FT == 0.10 and V.TRIM == 0.20 and V.MIN_FRAMES == 20 and V.OVER_S == 0.2
    assert V.APART == ("gsw_ind_2016",) and "hou_cle_2016" not in V.POOLED + V.APART
    src = open(V.__file__).read()
    assert "refuse_cv_test([tag], WHO)" in src and "guard([tag], WHO)" in src

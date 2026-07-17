"""Matchup-eval tool: sampling determinism, Wilson CI, report math.
The interactive loop is human-driven; everything feeding and consuming it is
tested here so the accuracy number's plumbing can't silently rot."""
from conftest import N_FRAMES  # noqa: F401  (repo-root sys.path side effect)
from label_matchups import build_sample, make_report, wilson95


def test_wilson_ci_basics():
    lo, hi = wilson95(90, 100)
    assert 0.82 < lo < 0.90 < hi < 0.95
    assert wilson95(0, 0) == [0.0, 1.0]
    lo50, hi50 = wilson95(45, 50)
    lo500, hi500 = wilson95(450, 500)
    assert (hi500 - lo500) < (hi50 - lo50)     # more n, tighter interval


def sample_rec(key, dist, share, tag="gsw_x"):
    return {"key": key, "clip": f"{tag}_s00",
            "pred": {"dist_median_ft": dist, "time_share": share}}


def test_report_math_and_buckets():
    sample = [sample_rec("a", 4.0, 0.8), sample_rec("b", 8.0, 0.4),
              sample_rec("c", 15.0, 0.2), sample_rec("d", 5.0, 0.6)]
    labels = {"a": {"verdict": "y"}, "b": {"verdict": "n"},
              "c": {"verdict": "u"}, "d": {"verdict": "y"}}
    rep = make_report(sample, labels)
    assert rep["labeled"] == 4 and rep["unsure"] == 1
    assert rep["overall"]["n_judged"] == 3 and rep["overall"]["correct"] == 2
    assert rep["overall"]["accuracy"] == 0.667
    assert rep["by_distance"]["tight(<6ft)"]["n_judged"] == 2      # a + d
    assert rep["by_time_share"]["dominant(>0.5)"]["accuracy"] == 1.0


def test_report_ignores_unlabeled():
    sample = [sample_rec("a", 4.0, 0.8), sample_rec("zzz", 4.0, 0.8)]
    rep = make_report(sample, {"a": {"verdict": "y"}})
    assert rep["sampled"] == 2 and rep["labeled"] == 1


def test_build_sample_deterministic_and_stratified():
    s1 = build_sample(40, seed=42)
    s2 = build_sample(40, seed=42)
    assert [x["key"] for x in s1] == [x["key"] for x in s2]
    assert len(s1) <= 40 and len(s1) >= 35
    games = {x["game"] for x in s1}
    assert len(games) >= 8            # stratification spreads across games
    for x in s1:                      # frozen predictions are self-contained
        assert {"defender", "primary_man", "time_share", "dist_median_ft"} <= set(x["pred"])

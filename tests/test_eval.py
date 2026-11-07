"""analytics.eval: grouped folds, scoring rules, calibration, game-level bootstrap, gates and the held-out guard."""
import math

import numpy as np
import pandas as pd
import pytest

from analytics import eval as ev
from sportvu import splits

GAMES12 = ["g%02d" % i for i in range(12)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_problem(n_games=12, rows=80, seed=1, shrink=1.0):
    """Two-feature logistic problem: y ~ Bernoulli(sigmoid(1.5 x1 - x2)), rows per game, a true p column."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_games * rows)
    x2 = rng.normal(size=n_games * rows)
    p = sigmoid(1.5 * x1 - x2)
    y = (rng.uniform(size=len(p)) < p).astype(int)
    return pd.DataFrame({"game": np.repeat(["g%02d" % i for i in range(n_games)], rows),
                         "x1": x1, "x2": x2, "p_true": sigmoid(shrink * (1.5 * x1 - x2)), "y": y})


# ----------------------------------------------------------------------------- folds

def test_grouped_folds_partition_games_and_are_deterministic():
    folds = ev.grouped_folds(GAMES12, n_folds=5, seed=0)
    assert len(folds) == 5
    tests = [g for _, t in folds for g in t]
    assert sorted(tests) == sorted(GAMES12)                     # every game in exactly one test fold
    for train, test in folds:
        assert not set(train) & set(test)
        assert sorted(train + test) == sorted(GAMES12)
    assert folds == ev.grouped_folds(list(reversed(GAMES12)), n_folds=5, seed=0)   # input order irrelevant
    assert folds != ev.grouped_folds(GAMES12, n_folds=5, seed=1)                   # the seed moves games
    with pytest.raises(ValueError):
        ev.grouped_folds(GAMES12[:4], n_folds=5)


def test_grouped_cv_keeps_each_game_in_one_fold_and_aligns_predictions():
    from sklearn.linear_model import LogisticRegression

    df = logistic_problem().sample(frac=1, random_state=3)     # shuffled rows, non-trivial index
    fit = lambda d: LogisticRegression().fit(d[["x1", "x2"]], d["y"])
    pred = lambda m, d: m.predict_proba(d[["x1", "x2"]])[:, 1]
    out = ev.grouped_cv(df, "game", fit, pred, n_folds=5, seed=0)
    assert out.index.equals(df.index)
    assert list(out.columns) == ["fold", "p"]
    assert out["p"].notna().all() and ((out["p"] >= 0) & (out["p"] <= 1)).all()
    per_game = out.groupby(df["game"])["fold"].nunique()
    assert (per_game == 1).all() and out["fold"].nunique() == 5
    base = np.full(len(df), df["y"].mean())
    assert ev.log_loss(df["y"], out["p"]) < ev.log_loss(df["y"], base)   # aligned predictions beat the base rate
    assert np.abs(out["p"].to_numpy() - df["p_true"].to_numpy()).mean() < 0.1
    assert out.equals(ev.grouped_cv(df, "game", fit, pred, n_folds=5, seed=0))


# ----------------------------------------------------------------------------- scoring rules

def test_log_loss_and_brier_match_hand_values():
    y, p = [1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4]
    assert ev.brier(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.16) / 4)
    assert ev.log_loss(y, p) == pytest.approx(-(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.6)) / 4)
    assert ev.log_loss([1], [0.0]) == pytest.approx(-math.log(1e-6))   # clipped, finite


def test_calibrated_sample_has_small_ece_and_unit_slope_and_shrunk_sample_slope_above_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, size=20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert ev.ece(y, p) < 0.02
    assert ev.calibration_slope(y, p) == pytest.approx(1.0, abs=0.1)
    shrunk = sigmoid(0.5 * np.log(p / (1 - p)))                 # true slope 2
    assert ev.calibration_slope(y, shrunk) > 1.5
    assert ev.ece(y, shrunk) > ev.ece(y, p)
    dec = ev.brier_decomposition(y, p)
    assert dec["uncertainty"] == pytest.approx(y.mean() * (1 - y.mean()))
    assert dec["reliability"] < 0.001 < dec["resolution"]
    assert ev.brier_decomposition(y, shrunk)["reliability"] > dec["reliability"]
    assert math.isnan(ev.calibration_slope([1, 1, 1], [0.2, 0.5, 0.9]))


def test_calibration_table_shape_and_bins():
    t = ev.calibration_table([1, 0, 1, 0, 1], [0.05, 0.15, 0.95, 1.0, 0.5], bins=10)
    assert list(t.columns) == ["bin", "lo", "hi", "n", "mean_p", "mean_y"]
    assert len(t) == 10 and t["n"].sum() == 5
    assert t.loc[9, "n"] == 2 and t.loc[0, "n"] == 1 and t.loc[5, "n"] == 1
    assert math.isnan(t.loc[3, "mean_p"])


# ----------------------------------------------------------------------------- bootstrap

def noisy_games(n_games, rows=30, seed=0):
    rng = np.random.default_rng(seed)
    eff = rng.normal(0, 1, size=n_games)
    return pd.DataFrame({"game": np.repeat(np.arange(n_games), rows),
                         "v": np.repeat(eff, rows) + rng.normal(0, 0.1, size=n_games * rows)})


def test_bootstrap_interval_contains_point_and_narrows_with_more_games():
    mean_v = lambda d: d["v"].mean()
    small = ev.bootstrap_by_game(noisy_games(5), "game", mean_v, n=300, seed=0)
    large = ev.bootstrap_by_game(noisy_games(60), "game", mean_v, n=300, seed=0)
    for r in (small, large):
        assert r["lo"] <= r["point"] <= r["hi"]
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])
    assert small == ev.bootstrap_by_game(noisy_games(5), "game", mean_v, n=300, seed=0)


def test_paired_bootstrap_excludes_zero_for_a_clearly_better_model():
    df = logistic_problem(n_games=12, rows=60, seed=5)
    df["p_bad"] = 0.5
    la = lambda d: ev.log_loss(d["y"], d["p_true"])
    lb = lambda d: ev.log_loss(d["y"], d["p_bad"])
    r = ev.paired_bootstrap_by_game(df, "game", la, lb, n=300, seed=0)
    assert r["diff"] < 0 and r["hi"] < 0 and r["excludes_zero"]
    same = ev.paired_bootstrap_by_game(df, "game", la, la, n=50, seed=0)
    assert same["diff"] == 0 and not same["excludes_zero"]


# ----------------------------------------------------------------------------- gates

def test_min_sample_gate_and_gated():
    counts = pd.Series({"a": 299, "b": 300, "c": 1000})
    assert ev.min_sample_gate(counts).tolist() == [False, True, True]
    assert ev.min_sample_gate(counts, n_min=10).all()
    df = pd.DataFrame({"player": ["x"] * 3 + ["y"] * 1, "v": range(4)})
    g = ev.gated(df, "player", n_min=2)
    assert "insufficient" not in df.columns
    assert g["insufficient"].tolist() == [False, False, False, True]


# ----------------------------------------------------------------------------- swap test

def test_swap_test_passes_and_fails_on_constructed_inputs():
    widths = {"rim_rate": 0.05, "speed": 0.4}
    a = {"rim_rate": 0.30, "speed": 4.2}
    rank_a = ["p1", "p2", "p3", "p4", "p5", "p6"]
    ok = ev.swap_test(a, {"rim_rate": 0.32, "speed": 4.1}, widths, rank_a, ["p1", "p2", "p4", "p3", "p5", "p6"])
    assert ok["passed"] and ok["spearman"] >= 0.8
    assert all(v["ok"] for v in ok["per_quantity"].values())
    assert ok["per_quantity"]["rim_rate"]["diff"] == pytest.approx(0.02)
    off = ev.swap_test(a, {"rim_rate": 0.40, "speed": 4.1}, widths, rank_a, rank_a)
    assert not off["passed"] and not off["per_quantity"]["rim_rate"]["ok"] and off["per_quantity"]["speed"]["ok"]
    scrambled = ev.swap_test(a, {"rim_rate": 0.31, "speed": 4.1}, widths, rank_a, list(reversed(rank_a)))
    assert not scrambled["passed"] and scrambled["spearman"] == pytest.approx(-1.0)
    missing = ev.swap_test(a, {"rim_rate": 0.31}, widths, rank_a, rank_a)
    assert not missing["passed"] and not missing["per_quantity"]["speed"]["ok"]
    assert not ev.swap_test(a, a, widths, rank_a, ["q1"])["passed"]       # no shared entities to rank


# ----------------------------------------------------------------------------- held-out guard

def test_refuse_heldout_raises_on_a_held_out_id_and_split_games_separates_them():
    sp = splits.load()
    assert sp["heldout_game"]["games"]["gsw_phx_2016"]["sportvu"] == "12.16.2015.PHX.at.GSW"
    assert ev.refuse_heldout(["data/sportvu/11.14.2015.BKN.at.GSW_moments.json", "gsw_cha_2016"], "test") == 2
    with pytest.raises(SystemExit):
        ev.refuse_heldout(["data/sportvu/11.14.2015.BKN.at.GSW_moments.json",
                           "data/sportvu/12.16.2015.PHX.at.GSW_moments.json"], "test")
    s = ev.split_games(["gsw_bkn_2015", "12.16.2015.PHX.at.GSW", "gsw_phx_2016", "cle_nyk_2015", "gsw_sac_2015"])
    assert s["train"] == ["gsw_bkn_2015", "gsw_sac_2015"]
    assert s["heldout"] == ["12.16.2015.PHX.at.GSW", "gsw_phx_2016", "cle_nyk_2015"]


# ----------------------------------------------------------------------------- report

def test_report_block_lists_every_metric_with_an_interval_and_the_table():
    df = logistic_problem(n_games=6, rows=40, seed=2)
    md = ev.report_block("shot model", df["y"], df["p_true"], df, "game", n=40, seed=0)
    assert md.startswith("### shot model")
    for label in ("| log loss |", "| Brier |", "| ECE |", "| calibration slope |", "| bin | lo | hi | n | mean_p | mean_y |"):
        assert label in md
    assert "240 rows, 6 games" in md
    assert chr(0x2014) not in md

"""winprob-E2 machinery (docs/research/winprob.md section 6): the logit temperature, the vectorised ECE against the
frozen definition, the pathwise peak-on-loss check, the verdict rule, and where the test season may be read.
Synthetic data only; no DATA_ROOT, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_winprob_e2 as W2

REPO = Path(__file__).resolve().parents[1]


def test_temperature_one_is_the_identity_and_the_fit_recovers_a_known_multiplier():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.5, 200000)
    true = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(len(z)) < true).astype(float)
    assert np.allclose(W2.apply_temperature(true, 1.0), true)
    over = W2.apply_temperature(true, 1.25)                  # an over-confident forecast of the same outcomes
    fit = W2.fit_temperature(y, over)
    assert abs(fit["a"] - 0.8) < 0.02 and fit["log_loss_after"] <= fit["log_loss_before"]
    assert W2.fit_temperature(y, true)["near_one"]


def rows_frame(n_games=60, per=60, seed=0, shift=0.0):
    from analytics.winprob import features as F
    rng = np.random.default_rng(seed)
    buckets = [b[0] for b in F.BUCKETS]
    g = np.repeat([f"g{i:03d}" for i in range(n_games)], per)
    y = np.repeat(rng.integers(0, 2, n_games), per)
    b = np.tile(np.repeat(buckets, per // len(buckets)), n_games)
    p = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.15, len(g)), 0.01, 0.99)
    return pd.DataFrame({"game_id": g, "time_bucket": b, "home_win": y, "p_e1": p,
                         "p_e2": np.clip(p + shift, 0.01, 0.99), "p_e2_grouped": p})


def test_the_vectorised_ece_equals_the_frozen_definition_and_identical_arms_pair_to_zero():
    from analytics.eval import ece
    r = rows_frame()
    out = W2.summarize(r, n_boot=100)
    for b, v in out["arms"]["e1"]["bucket_ece"].items():
        m = r["time_bucket"] == b
        if m.sum():
            assert v["point"] == pytest.approx(ece(r.loc[m, "home_win"], r.loc[m, "p_e1"], 10), abs=1e-12)
    d = out["paired"]["e2_grouped_minus_e1"]["mean_bucket_ece"]
    assert d == {"point": 0.0, "lo": 0.0, "hi": 0.0}


def test_verdict_readings():
    def paired(e, b):
        ci = lambda v: {"point": v[0], "lo": v[1], "hi": v[2]}
        return {"e2_minus_e1": {"mean_bucket_ece": ci(e), "mean_bucket_brier": ci(b)},
                "e2_minus_e2_grouped": {"mean_bucket_ece": ci((0.0, -0.001, 0.001))}}
    assert W2.verdict(paired((-0.003, -0.005, -0.001), (0.0, -0.0002, 0.0004)))["reading"] == "pass"
    assert "guard" in W2.verdict(paired((-0.003, -0.005, -0.001), (0.0003, 0.0001, 0.0006)))["reading"]
    assert "includes zero" in W2.verdict(paired((-0.001, -0.003, 0.001), (0.0, -0.0002, 0.0002)))["reading"]
    assert "rises" in W2.verdict(paired((0.003, 0.001, 0.005), (0.0, -0.0002, 0.0002)))["reading"]


def test_peak_on_loss_counts_the_first_crossing_and_its_expected_loss():
    # game a: home reaches 0.92 then loses; game b: away reaches 0.95 (home 0.05) and wins
    r = pd.DataFrame({"game_id": ["a"] * 3 + ["b"] * 3, "home_win": [0] * 3 + [0] * 3,
                      "p_e1": [0.6, 0.92, 0.97, 0.4, 0.05, 0.02]})
    r["p_e2"] = r["p_e1"]
    r["p_e2_grouped"] = r["p_e1"]
    codes, _ = pd.factorize(r["game_id"], sort=True)
    out = W2.peak_on_loss(r, codes, 2, np.ones((5, 2)))
    x = out["e1"]["0.9"]
    assert x["games_reaching"] == 2 and x["observed_losses"] == 1.0
    assert x["expected_losses"] == pytest.approx((1 - 0.92) + (1 - 0.95))


def test_a_non_finite_prediction_refuses_the_verdict():
    r = rows_frame()
    r.loc[5, "p_e2"] = np.nan
    out = W2.summarize(r, n_boot=20)
    assert out["verdict"]["refused"]


def _tree():
    return ast.parse((REPO / "tools" / "research_winprob_e2.py").read_text())


def test_scoring_data_root_is_called_only_inside_the_scorer_given_to_score_once():
    tree = _tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "score_once"]
    assert len(calls) == 1 and "scorer" in {a.id for a in calls[0].args if isinstance(a, ast.Name)}
    inside = {id(c) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "scorer"
              for c in ast.walk(fn)}
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "scoring_data_root"]
    assert reads and all(id(n) in inside for n in reads)


def test_final_checks_a_clean_tree_first_and_fits_temperatures_on_development_only():
    tree = _tree()
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final")
    assert "clean_head()" in ast.unparse(final.body[0])
    scorer = next(n for n in ast.walk(final) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    assert "fit_temperature" not in ast.unparse(scorer) and "fit_e1" not in ast.unparse(scorer)
    text = (REPO / "tools" / "research_winprob_e2.py").read_text()
    for bad in ("elo_state_games_2024_25", "DEFAULT_PBP_ROOT", "PBP_FILES", "_research-data", "testset.json"):
        assert bad not in text, bad

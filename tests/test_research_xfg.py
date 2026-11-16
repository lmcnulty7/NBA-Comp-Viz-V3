"""xfg-E1 machinery (docs/research/xfg.md): the Venn-Abers predictor against a brute-force isotonic fit, and the
CVAP merge. No data, no held-out game."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from tools.research_xfg import VennAbers, merge_cvap, paired_by_game


def brute(scores, labels, s, label):
    x = np.append(scores, s)
    y = np.append(labels, label)
    return float(IsotonicRegression(out_of_bounds="clip").fit(x, y).predict([s])[0])


def test_venn_abers_matches_a_brute_force_isotonic_fit_including_ties_and_the_ends():
    rng = np.random.default_rng(0)
    scores = np.round(rng.random(300), 2)               # many ties
    labels = (rng.random(300) < scores).astype(float)
    tests = np.concatenate([np.round(rng.random(40), 2), rng.random(20), [-1.0, 2.0, scores.min(), scores.max()]])
    p0, p1 = VennAbers(scores, labels).p0p1(tests)
    for t, a, b in zip(tests, p0, p1):
        assert abs(a - brute(scores, labels, t, 0.0)) < 1e-12
        assert abs(b - brute(scores, labels, t, 1.0)) < 1e-12
        assert a <= b


def test_cvap_merge_is_the_log_loss_minimax_formula_and_stays_inside_zero_one():
    p0 = np.array([[0.0, 0.2, 0.5], [0.1, 0.3, 0.5]])
    p1 = np.array([[0.4, 0.3, 1.0], [0.2, 0.35, 1.0]])
    p = merge_cvap(p0, p1)
    g1 = np.sqrt(p1[0] * p1[1])
    g0 = np.sqrt((1 - p0[0]) * (1 - p0[1]))
    assert np.allclose(p, g1 / (g0 + g1))
    assert ((p > 0) & (p < 1)).all()


def test_paired_by_game_resamples_games_not_shots():
    games = np.array(["a"] * 100 + ["b"] * 100)
    a = np.concatenate([np.ones(100), np.zeros(100)])
    out = paired_by_game(games, a, np.zeros(200), n=500)
    assert out["games"] == 2 and out["lo"] == 0.0 and out["hi"] == 1.0 and abs(out["diff"] - 0.5) < 1e-12

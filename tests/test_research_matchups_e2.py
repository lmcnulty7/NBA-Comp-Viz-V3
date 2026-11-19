"""matchups-E2 machinery (docs/research/matchups.md, the D112 amendment): the HMM's forward-backward and Viterbi
against brute force, the simplex step, EM recovering known parameters on synthetic chains, shares and switches, and
guards on the module's source. No real data, no held-out game, no test game."""

import ast
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_matchups_e2 as H

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_matchups_e2.py"


def brute(logE, rho):
    """Every state path of short chains: log likelihood, state posteriors, the best path."""
    T = logE.shape[0]
    A = H._log_trans(rho)
    paths, scores = [], []
    for path in itertools.product(range(H.K), repeat=T):
        s = math.log(1 / H.K) + logE[0, path[0]] + sum(A[path[t - 1], path[t]] + logE[t, path[t]] for t in range(1, T))
        paths.append(path)
        scores.append(s)
    scores = np.array(scores)
    ll = np.log(np.exp(scores - scores.max()).sum()) + scores.max()
    post = np.zeros((T, H.K))
    for path, s in zip(paths, scores):
        for t, k in enumerate(path):
            post[t, k] += math.exp(s - ll)
    return ll, post, np.array(paths[int(np.argmax(scores))])


def test_forward_backward_and_viterbi_match_brute_force():
    rng = np.random.default_rng(0)
    logE = rng.normal(size=(3, 4, H.K))
    ll, post, _st = H.forward_backward(logE, 0.8)
    path = H.viterbi(logE, 0.8)
    for i in range(3):
        bl, bp, bpath = brute(logE[i], 0.8)
        assert abs(ll[i] - bl) < 1e-9 and np.allclose(post[i], bp, atol=1e-9)
        assert (path[i] == bpath).all()


def test_simplex_weights_stay_on_the_simplex_and_solve_the_free_case():
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    g = np.array([0.4, 0.2])
    assert np.allclose(H.simplex_weights(S, S @ g), g)
    for v in (np.array([-1.0, 3.0]), np.array([5.0, 5.0]), np.array([-1.0, -1.0])):
        w = H.simplex_weights(S, v)
        assert w.min() >= -1e-12 and w.sum() <= 1 + 1e-12


def synthetic_rows(n_poss=400, T=12, p=None, seed=1):
    """Cache-shaped rows drawn from the model with known parameters."""
    p = p or {"gO": 0.6, "gB": 0.15, "rho": 0.9, "s2": 2.0}
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_poss):
        offs = sorted(rng.choice(np.arange(100, 200), size=5, replace=False))
        O = rng.normal(0, 12, size=(T, 5, 2))
        B = O[np.arange(T), rng.integers(0, 5, size=T)] + rng.normal(0, 1, size=(T, 2))
        for dfn in range(5):
            k = int(rng.integers(0, 5))
            for t in range(T):
                if t and rng.random() > p["rho"]:
                    k = int(rng.choice([j for j in range(5) if j != k]))
                mu = p["gO"] * O[t, k] + p["gB"] * B[t]
                x = mu + rng.normal(0, math.sqrt(p["s2"]), size=2)
                rows.append(("G", q, t, t, 1000 + dfn, x[0], x[1], B[t, 0], B[t, 1], *offs, *O[t, :, 0], *O[t, :, 1]))
    return pd.DataFrame(rows, columns=H.CACHE_COLUMNS), p


def test_em_recovers_known_parameters():
    rows, p = synthetic_rows()
    fit = H.em(H.Chains(rows), log=lambda *a, **k: None)["params"]
    assert abs(fit["gO"] - p["gO"]) < 0.02 and abs(fit["gB"] - p["gB"]) < 0.02
    assert abs(fit["rho"] - p["rho"]) < 0.02 and abs(fit["s2"] - p["s2"]) / p["s2"] < 0.05


def test_decoding_beats_chance_and_the_hmm_switches_less_than_emission_only():
    rows, p = synthetic_rows(n_poss=150)
    ch = H.Chains(rows)
    hmm, emi = H.decode(ch, p, "hmm"), H.decode(ch, p, "emission")
    assert H.switches(hmm)["mean_switches"] < H.switches(emi)["mean_switches"]
    sh = H.shares_of(hmm, rows)
    tot = sh.groupby(["possession", "def_player_id"])["partial_possession"].sum()
    assert np.allclose(tot, 1.0)                              # every defender's shares sum to one


def test_a_chain_breaks_when_the_attackers_change():
    rows, _p = synthetic_rows(n_poss=2, T=4)
    rows.loc[(rows["possession"] == 0) & (rows["k"] >= 2), "o0"] = 999
    ch = H.Chains(rows)
    assert ch.n_chains == 2 * 5 + 5                            # possession 0 splits each defender's chain in two


def test_the_module_reads_only_development_games_and_never_runs_a_lane_command_line():
    text = SOURCE.read_text()
    for banned in ("T.run(", "train.main", "features.main", "load_games(", "heldout_once", "score_heldout("):
        assert banned not in text, banned
    assert "development_games(" in text and "refuse_test(" in text and "clean_head()" in text

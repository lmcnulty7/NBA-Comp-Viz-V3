"""rebounds-E2 machinery (docs/research/rebounds.md, "rebounds-E2"; PLAN.md D124), on synthetic candidate frames only:
the within-miss softmax objective against a brute-force softmax, the contiguity and sum guards, the softmax and binary
arms with the fixed XGBoost settings, and the development statistic. No DATA_ROOT, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.rebounds import features as F
from tools import research_rebounds_e2 as E

REPO = Path(__file__).resolve().parents[1]


def synthetic(n_games=12, misses=25, seed=0):
    """Candidate frames with a known within-miss softmax truth: 8 to 11 rows a miss, one of them the none row;
    players recur across games so the frozen prior has something to learn."""
    rng = np.random.default_rng(seed)
    w = {"distance_to_rim_ft": -0.25, "distance_to_ball_landing_ft": -0.35, "is_offense": -0.4,
         "nearest_opponent_dist_ft": 0.1, "n_opponents_closer_to_rim": -0.3, "is_none": 0.5}
    rows = []
    for gi in range(n_games):
        game = f"11.{gi + 1:02d}.2015.AAA.at.BBB"
        for ev in range(misses):
            n = int(rng.integers(8, 12))
            feats = {c: rng.normal(0, 1, n) for c in F.STATIC_FEATURES}
            feats["distance_to_rim_ft"] = rng.uniform(1, 25, n)
            feats["distance_to_ball_landing_ft"] = rng.uniform(0, 20, n)
            feats["is_offense"] = (np.arange(n) % 2).astype(float)
            feats["n_opponents_closer_to_rim"] = rng.integers(0, 5, n).astype(float)
            feats["is_none"] = np.zeros(n)
            feats["is_none"][-1] = 1.0
            score = sum(w.get(c, 0.0) * feats[c] for c in F.STATIC_FEATURES)
            p = np.exp(score - score.max())
            p /= p.sum()
            winner = rng.choice(n, p=p)
            for i in range(n):
                r = {c: float(feats[c][i]) for c in F.STATIC_FEATURES}
                r.update(sportvu_game=game, event=ev, heldout=False, is_rebounder=int(i == winner),
                         candidate_player_id=-1 if i == n - 1 else int(rng.integers(1, 60)))
                rows.append(r)
    return pd.DataFrame(rows)


def brute_softmax(scores, groups):
    out = np.empty(len(scores))
    for g in np.unique(groups):
        m = groups == g
        e = np.exp(scores[m] - scores[m].max())
        out[m] = e / e.sum()
    return out


def test_gradient_and_hessian_match_a_brute_force_softmax_and_finite_differences():
    rng = np.random.default_rng(1)
    sizes = [3, 5, 2, 7]
    groups = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    s = rng.normal(0, 2, len(groups))
    y = np.zeros(len(groups))
    for st in starts[:-1]:
        y[st + int(rng.integers(0, 2))] = 1.0
    p = brute_softmax(s, groups)
    assert np.allclose(E.softmax_blocks(s, starts), p)
    grad, hess = E.softmax_grad_hess(s, y, starts)
    assert np.allclose(grad, p - y) and np.allclose(hess, p * (1 - p))

    def total(v):                                       # the summed loss: its gradient is the objective's gradient
        return E.softmax_loss(v, y, starts) * len(sizes)
    h = 1e-6
    for i in range(len(s)):
        e = np.zeros(len(s))
        e[i] = h
        num_g = (total(s + e) - total(s - e)) / (2 * h)
        assert abs(num_g - grad[i]) < 1e-6
        g_plus, _ = E.softmax_grad_hess(s + e, y, starts)
        g_minus, _ = E.softmax_grad_hess(s - e, y, starts)
        assert abs((g_plus[i] - g_minus[i]) / (2 * h) - hess[i]) < 1e-6      # the diagonal of the Hessian


def test_the_guard_refuses_split_misses_and_misses_without_one_rebounder():
    f = synthetic(n_games=2, misses=3)
    E.check_blocks(f)
    swapped = pd.concat([f.iloc[[0]], f.iloc[20:21], f.iloc[1:20], f.iloc[21:]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="not contiguous"):
        E.check_blocks(swapped)
    two = f.copy()
    two.loc[two.index[:3], "is_rebounder"] = 1
    with pytest.raises(ValueError, match="exactly one rebounder"):
        E.check_blocks(two)
    starts = E.check_blocks(f)
    with pytest.raises(ValueError, match="sum to 1"):
        E.check_sums(np.full(len(f), 0.5), starts)


def test_the_settings_are_the_pre_registered_ones():
    assert (E.XGB_PARAMS["max_depth"], E.XGB_PARAMS["eta"], E.ROUNDS) == (3, 0.1, 80)
    assert E.THRESHOLD == -0.010 and E.FOLDS == 5 and E.BOOT == 2000


def test_the_softmax_arm_learns_the_within_miss_competition():
    train, test = synthetic(n_games=10, seed=0), synthetic(n_games=4, seed=1)
    p = E.fit_softmax(train)(test)
    starts = E.check_blocks(test)
    E.check_sums(p, starts)
    y = test["is_rebounder"].astype(bool).to_numpy()
    model_loss = -np.log(p[y]).mean()
    uniform_loss = np.log(np.diff(starts)).mean()
    assert model_loss < uniform_loss - 0.1


def test_the_binary_ablation_uses_the_frozen_tau_procedure():
    from analytics.rebounds import train as T
    train, test = synthetic(n_games=8, seed=2), synthetic(n_games=3, seed=3)
    model = E.fit_binary(train)
    assert T.TAU_BOUNDS[0] <= model.tau <= T.TAU_BOUNDS[1]
    E.check_sums(model(test), E.check_blocks(test))


def test_the_development_statistic_runs_and_pairs_the_arms_on_the_same_misses():
    cands = synthetic(n_games=10, misses=15, seed=4).sample(frac=1.0, random_state=0)      # shuffled rows
    cands = cands.sort_values(["sportvu_game", "event"], kind="stable")                     # misses rejoined
    pred, info = E.out_of_fold(cands)
    assert info["folds"] == 5 and len(info["binary_tau_by_fold"]) == 5
    s = E.summarize(pred, n_boot=200)
    assert s["misses"] == 150 and set(s["paired"]) == {"softmax_minus_baseline", "binary_minus_baseline",
                                                       "softmax_minus_binary"}
    a, b = s["arms"]["softmax"]["per_miss_log_loss"]["value"], s["arms"]["baseline"]["per_miss_log_loss"]["value"]
    assert s["paired"]["softmax_minus_baseline"]["value"] == pytest.approx(a - b)
    assert set(s["verdict"]) >= {"met", "reading", "difference", "lo", "hi"}


def test_out_of_fold_refuses_a_frame_where_a_rebounder_row_landed_in_another_miss():
    cands = synthetic(n_games=3, misses=4, seed=5)
    bad = cands.copy()
    first = bad[(bad["sportvu_game"] == bad["sportvu_game"].iloc[0]) & (bad["event"] == 0)]
    bad.loc[first.index[first["is_rebounder"] == 1][0], "event"] = 3     # miss 0 loses its rebounder, miss 3 has two
    with pytest.raises(ValueError, match="exactly one rebounder"):
        E.out_of_fold(bad)


def test_verdict_readings():
    assert E.verdict({"value": -0.02, "lo": -0.03, "hi": -0.01})["met"] is True
    assert "smaller than" in E.verdict({"value": -0.005, "lo": -0.008, "hi": -0.002})["reading"]
    assert "includes zero" in E.verdict({"value": -0.02, "lo": -0.05, "hi": 0.01})["reading"]
    assert "worse" in E.verdict({"value": 0.02, "lo": 0.01, "hi": 0.03})["reading"]


def test_the_module_reads_data_only_in_the_feasibility_check_and_never_test_data():
    tree = ast.parse((REPO / "tools" / "research_rebounds_e2.py").read_text())
    calls = {getattr(n.func, "attr", getattr(n.func, "id", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"score_once", "scoring_data_root", "research_data_root"}
    feas = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "feasibility")
    inside = {id(c) for c in ast.walk(feas)}
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "read_parquet"]
    assert reads and all(id(n) in inside for n in reads)
    assert "refuse_test(" in ast.unparse(feas) and "fold" in ast.unparse(feas)

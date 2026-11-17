"""winprob-E1 machinery (docs/research/winprob.md section 6, PLAN.md D89): the Elo ratings, the leak checks over the new
columns, the paired bootstrap and the verdict rule, and where the test season may be read. Synthetic data only; no
DATA_ROOT, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_winprob as W
from tools import research_winprob_elo as E

REPO = Path(__file__).resolve().parents[1]


def schedule(n_days=60, seed=0, season="2021-22"):
    """A round robin of 6 teams with fixed strengths, one season, two games a day."""
    rng = np.random.default_rng(seed)
    teams = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    strength = dict(zip(teams, [8, 4, 0, 0, -4, -8]))
    rows = []
    for day in range(n_days):
        order = rng.permutation(teams)
        for j in range(0, 4, 2):
            h, a = order[j], order[j + 1]
            m = strength[h] - strength[a] + 2 + rng.normal(0, 10)
            m = m if abs(m) >= 1 else 1.0
            hs = 100 + max(m, 0)
            rows.append({"game_id": f"g{day:03d}{j}{h}", "date": f"2021-{11 + day // 30:02d}-{day % 30 + 1:02d}",
                         "home": h, "away": a, "home_score": round(hs), "away_score": round(hs - m),
                         "season": season})
    return E.schedule_frame(pd.DataFrame(rows))


def test_elo_update_is_zero_sum_and_moves_toward_the_result():
    s = schedule(n_days=1)
    elo = E.Elo(20.0, 50.0, 0.25)
    pre = E.advance(elo, s)
    assert pre["elo_diff"].iloc[0] == pytest.approx(50.0)                 # both at MEAN plus the home advantage
    assert sum(elo.ratings.values()) == pytest.approx(E.MEAN * len(elo.ratings))
    g = s.iloc[0]
    won = g["home_score"] > g["away_score"]
    assert (elo.ratings[g["home"]] > E.MEAN) == bool(won)
    assert pre["mu_hat"].iloc[0] == pytest.approx(50.0 * E.POINTS_PER_ELO)


def test_between_season_regression_pulls_ratings_to_the_mean():
    elo = E.Elo(20.0, 0.0, 0.5, ratings={"AAA": 1600.0, "BBB": 1400.0}, season="2020-21")
    nxt = pd.DataFrame([{"game_id": "x", "date": "2021-10-20", "home": "CCC", "away": "DDD", "home_score": 100,
                         "away_score": 90, "season": "2021-22"}])
    E.advance(elo, E.schedule_frame(nxt), record=False)
    assert elo.ratings["AAA"] == pytest.approx(1550.0) and elo.ratings["BBB"] == pytest.approx(1450.0)


def test_record_false_keeps_no_pregame_value():
    assert E.advance(E.Elo(20.0, 50.0, 0.25), schedule(n_days=3), record=False) is None


def test_a_games_own_and_later_results_never_reach_its_pregame_value():
    s = schedule()
    assert E.prefix_violations(E.Elo(20.0, 50.0, 0.25), s, n=30) == []


def test_the_elo_prefix_check_catches_a_leak(monkeypatch):
    s = schedule()
    real = E.advance

    def leaky(elo, sched, record=True):
        out = real(elo, sched, record)
        if out is not None:                              # a pregame value that reads the game's own final margin
            out["mu_hat"] = out["mu_hat"] + (sched["home_score"].to_numpy() - sched["away_score"].to_numpy())
        return out
    monkeypatch.setattr(E, "advance", leaky)
    assert E.prefix_violations(E.Elo(20.0, 50.0, 0.25), s, n=10)


def test_choose_params_refuses_a_season_outside_development_and_returns_a_grid_point():
    with pytest.raises(SystemExit):
        E.choose_params(schedule(season="2024-25"))
    s = pd.concat([schedule(seed=i, season=x) for i, x in enumerate(("2020-21", "2021-22", "2022-23", "2023-24"))],
                  ignore_index=True)
    s["date"] = [f"{x[:4]}-{d[5:]}" for x, d in zip(s["season"], s["date"])]
    s = s.sort_values(["date", "game_id"]).reset_index(drop=True)
    out = E.choose_params(s)
    assert len(out["table"]) == len(E.GRID_K) * len(E.GRID_HOME) * len(E.GRID_REGRESS)
    best = min(r["mean_brier"] for r in out["table"])
    assert out["chosen_mean_brier"] == best
    assert out["chosen"]["k"] in E.GRID_K and out["chosen"]["home"] in E.GRID_HOME


def test_mu_hat_and_the_new_columns_pass_the_frozen_prefix_check_on_the_fixtures():
    from analytics.winprob import features as F
    d = F.load_fixtures(0, n_games=6, seasons=("2021-22",))
    games = d["games"].copy()
    games["game_id"] = games["game_id"].astype(str)
    sched = games.rename(columns={"game_date": "date"})
    sched["home"] = sched["home_team_id"].astype(str)
    sched["away"] = sched["away_team_id"].astype(str)
    sched = E.schedule_frame(sched[list(E.COLUMNS) + ["season"]])
    params = {"k": 20.0, "home": 50.0, "regress": 0.25}
    build = W.build_with_mu(sched, params)
    viol = W.frozen_prefix_check(d["playbyplay"], d["games"], build, n_games=3, cuts=3)
    assert viol == []
    full = build(d["playbyplay"], d["games"])
    assert {"mu_hat", "mu_drift"} <= set(full.columns)
    sec = np.maximum(full["seconds_remaining"].to_numpy(), 0)
    assert np.allclose(full["mu_drift"], full["mu_hat"] * np.sqrt(sec / 2880.0))

    def leaky(pbp, g, odds=None, **kw):                  # mu_hat that reads the game's own final score
        out = build(pbp, g, odds, **kw)
        fin = g.assign(game_id=g["game_id"].astype(str)).set_index("game_id")
        out["mu_hat"] = out["game_id"].map(fin["home_score"] - fin["away_score"]).astype(float).to_numpy()
        return out
    assert W.frozen_prefix_check(d["playbyplay"], d["games"], leaky, n_games=3, cuts=3)


def rows_frame(n_games=40, per=50, seed=0):
    from analytics.winprob import features as F
    rng = np.random.default_rng(seed)
    buckets = [b[0] for b in F.BUCKETS]
    g = np.repeat([f"g{i:03d}" for i in range(n_games)], per)
    y = np.repeat(rng.integers(0, 2, n_games), per)
    b = np.tile(np.repeat(buckets, per // len(buckets)), n_games)
    p = np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.1, len(g)), 0.01, 0.99)
    return pd.DataFrame({"game_id": g, "time_bucket": b, "home_win": y, "p_baseline": p, "p_e1": p, "p_ablation": p})


def test_identical_arms_give_a_zero_paired_difference_and_no_adoption():
    out = W.summarize(rows_frame(), n_boot=200)
    d = out["paired"]["e1_minus_baseline"]["mean_bucket_brier"]
    assert d == {"point": 0.0, "lo": 0.0, "hi": 0.0}
    assert out["verdict"]["adopt"] is False and "includes zero" in out["verdict"]["reading"]


def test_a_clearly_better_arm_is_adopted_and_the_point_matches_the_frozen_bucket_mean():
    from analytics.eval import brier
    from analytics.winprob import train as T
    r = rows_frame()
    r["p_e1"] = np.clip(r["p_baseline"] + (r["home_win"] - r["p_baseline"]) * 0.3, 0.01, 0.99)
    r["p_ablation"] = r["p_e1"]
    out = W.summarize(r, n_boot=200)
    scores = {b: {"brier": {"point": brier(s["home_win"], s["p_e1"])}} for b, s in r.groupby("time_bucket")}
    assert out["arms"]["e1"]["mean_bucket_brier"]["point"] == pytest.approx(T.selection_score(scores))
    v = out["verdict"]
    assert v["adopt"] and v["reading"] == "pass" and v["hi"] < 0 and v["difference"] <= -W.THRESHOLD


def test_paired_bootstrap_resamples_games_not_rows():
    r = rows_frame(n_games=2, per=50)
    r["p_e1"] = np.where(r["game_id"] == "g000", r["p_baseline"] + 0.0, r["p_baseline"])
    w = W._boot_weights(2, n=50)
    assert w.shape == (50, 2) and (w.sum(axis=1) == 2).all()


def test_verdict_readings():
    reg = ["48-36", "36-24", "24-12", "12-4", "4-0"]

    def paired(point, lo, hi, bucket_lo=-0.01):
        ci = {"point": point, "lo": lo, "hi": hi}
        bucket = {b: {"point": 0.0, "lo": bucket_lo, "hi": 0.01} for b in reg}
        return {"e1_minus_baseline": {"mean_bucket_brier": ci, "bucket_brier": bucket},
                "e1_minus_ablation": {"mean_bucket_brier": {"point": -0.001, "lo": -0.002, "hi": 0.0005}}}
    assert W.verdict(paired(-0.003, -0.004, -0.002), reg)["reading"] == "pass"
    assert "smaller than the threshold" in W.verdict(paired(-0.0015, -0.0025, -0.0005), reg)["reading"]
    assert W.verdict(paired(-0.0015, -0.0025, -0.0005), reg)["interval_straddles_threshold"]
    assert "includes zero" in W.verdict(paired(-0.003, -0.007, 0.001), reg)["reading"]
    assert "worse than" in W.verdict(paired(0.002, 0.001, 0.003), reg)["reading"]
    assert "bucket" in W.verdict(paired(-0.003, -0.004, -0.002, bucket_lo=0.0001), reg)["reading"]
    assert W.verdict(paired(-0.003, -0.004, -0.002), reg)["ablation"]["drift_form_adds_beyond_mu_alone"] is False


# ── where the test season may be read ──────────────────────────────────────────────────────────────────────────

def _tree():
    return ast.parse((REPO / "tools" / "research_winprob.py").read_text())


def test_scoring_data_root_is_called_only_inside_the_scorer_given_to_score_once():
    tree = _tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "score_once"]
    assert len(calls) == 1
    scorers = {a.id for a in calls[0].args if isinstance(a, ast.Name)}
    assert "scorer" in scorers
    inside = {id(c) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name in scorers
              for c in ast.walk(fn)}
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "scoring_data_root"]
    assert reads and all(id(n) in inside for n in reads)


def test_the_scorer_is_defined_in_final_after_the_fits_and_the_elo_state():
    tree = _tree()
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final")
    names = [getattr(s, "name", None) or ast.unparse(s)[:40] for s in final.body]
    i_scorer = next(i for i, s in enumerate(final.body) if isinstance(s, ast.FunctionDef) and s.name == "scorer")
    before = ast.unparse(ast.Module(body=final.body[:i_scorer], type_ignores=[]))
    assert "fit_arms(" in before and "state_through_2024_25(" in before and "require_gate(" in before, names


def test_only_the_elo_module_reads_2024_25_and_it_scores_nothing_there():
    import inspect
    src = inspect.getsource(E.state_through_2024_25)
    assert "record=False" in src and "brier" not in src.lower()
    assert "elo_state_games_2024_25" not in (REPO / "tools" / "research_winprob.py").read_text()


def test_dev_code_never_names_the_test_season_files():
    text = (REPO / "tools" / "research_winprob.py").read_text()
    for bad in ("DEFAULT_PBP_ROOT", "PBP_FILES", "PBP_ROOT_ENV", "_research-data", "testset.json"):
        assert bad not in text, bad


# ── data checks at scoring (the lead's pre-score review) ───────────────────────────────────────────────────────

def test_schedule_agreement_counts_games_whose_final_margins_differ():
    pbp = pd.DataFrame({"game_id": ["a", "b", "c"], "home_score": [100, 90, 80], "away_score": [90, 95, 70]})
    sched = pd.DataFrame({"game_id": ["a", "b"], "home_score": [100, 91], "away_score": [90, 95]})
    out = W.schedule_agreement(pbp, sched)
    assert out == {"games_compared": 2, "games_agree": 1, "games_disagree": 1, "games_without_a_schedule_row": 1,
                   "share_agree": 0.5}


def test_a_non_finite_prediction_refuses_the_verdict():
    r = rows_frame()
    r.loc[3, "p_e1"] = np.nan
    out = W.summarize(r, n_boot=50)
    assert out["non_finite_values"] == 1 and out["verdict"]["refused"] and "adopt" not in out["verdict"]


def test_final_checks_a_clean_tree_before_anything_else():
    tree = _tree()
    final = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final")
    first = final.body[0]
    assert "clean_head()" in ast.unparse(first)
    import inspect
    src = inspect.getsource(W.clean_head)
    assert '["git", "status", "--porcelain"]' in src          # the whole tree, no path filter

"""lineups-E2 machinery (docs/research/lineups.md): the defensive sign, the time-forward weekly window, the raw
baseline, the verdict, and a rehearsal of the scorer through score_once on synthetic play-by-play in a temporary
pinned root. No real data."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.test_research_lineups import dates, schedule_of, synth_games
from tools import research_lineups as E1
from tools import research_lineups_e2 as L
from tools import research_testset as R


def tables(start, n, seed):
    pbp, starters = synth_games(dates(start, n), seed=seed)
    feats, games, dropped, stints = E1.season_tables(pbp, starters)
    feats = L.with_points(feats, stints)
    return feats, L.observations(feats), pbp, starters


@pytest.fixture(scope="module")
def dev():
    return tables("2022-11-01", 70, seed=1)


def toy(rows):
    """Observations from (off, dfn, y, w, date) tuples."""
    out = pd.DataFrame(rows, columns=["off", "dfn", "y", "w", "game_date"])
    out["game_id"] = [f"g{i}" for i in range(len(out))]
    return out


def test_observations_split_each_stint_into_two_sides_with_their_own_points(dev):
    feats, obs = dev[0], dev[1]
    assert len(obs) == 2 * len(feats)
    home = obs[obs["off_home"]].reset_index(drop=True)
    assert np.allclose(home["y"] * home["w"], feats["points_home"])
    assert (home["off"].to_numpy() == feats["lineup_home"].to_numpy()).all()
    away = obs[~obs["off_home"]].reset_index(drop=True)
    assert np.allclose(away["y"] * away["w"], feats["points_away"])


def test_a_better_defense_lowers_the_prediction_and_its_rating_is_positive():
    rows = []
    for i in range(200):
        rows.append(("A", "B", 0.8, 10.0, "2025-11-01"))       # B allows little
        rows.append(("A", "C", 1.3, 10.0, "2025-11-01"))       # C allows a lot
        rows.append(("B", "A", 1.1, 10.0, "2025-11-01"))
        rows.append(("C", "A", 1.1, 10.0, "2025-11-01"))
    fit = L.fit_lineups(toy(rows), {}, {}, lam=1.0)
    assert fit["D"]["B"] > 0 > fit["D"]["C"]
    test = toy([("A", "B", 0.0, 1.0, "2025-11-08"), ("A", "C", 0.0, 1.0, "2025-11-08")])
    p = L.predict_lineups(fit, test, {}, {})
    assert p[0] < p[1]


def test_a_large_penalty_pulls_ratings_to_the_prior_and_an_unseen_lineup_gets_its_prior():
    rows = [("1|2|3|4|5", "6|7|8|9|10", 1.0, 10.0, "2025-11-01"), ("6|7|8|9|10", "1|2|3|4|5", 1.0, 10.0, "2025-11-01")]
    off = {p: 0.01 for p in range(1, 6)}
    dfn = {p: 0.02 for p in range(6, 11)}
    fit = L.fit_lineups(toy(rows), off, dfn, lam=1e9)
    assert np.isclose(fit["O"]["1|2|3|4|5"], 0.05) and np.isclose(fit["D"]["6|7|8|9|10"], 0.10)
    test = toy([("1|2|3|4|11", "6|7|8|9|10", 0.0, 1.0, "2025-11-08")])
    p = L.predict_lineups(fit, test, off, dfn)
    assert np.isclose(p[0], fit["b0"] + 0.04 - fit["D"]["6|7|8|9|10"])


def test_the_iterative_solve_matches_a_direct_solve(dev):
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    obs = dev[1]
    rng = np.random.default_rng(0)
    for lam in (0.5, 50.0):
        lineups = sorted(set(obs["off"]) | set(obs["dfn"]))
        fit = L.fit_lineups(obs, {}, {}, lam)
        n, Lc = len(obs), len(lineups)
        ix = {l: i for i, l in enumerate(lineups)}
        X = sp.csr_matrix((np.concatenate([np.ones(n), np.ones(n), -np.ones(n)]),
                           (np.tile(np.arange(n), 3), np.concatenate([np.zeros(n, dtype=int),
                                                                      1 + obs["off"].map(ix).to_numpy(),
                                                                      1 + Lc + obs["dfn"].map(ix).to_numpy()]))),
                          shape=(n, 1 + 2 * Lc))
        w, y = obs["w"].to_numpy(), obs["y"].to_numpy()
        pen = np.concatenate([[0.0], np.full(2 * Lc, lam)])
        direct = spsolve((X.T @ X.multiply(w[:, None]) + sp.diags(pen)).tocsc(), np.asarray(X.T @ (w * y)).ravel())
        assert abs(fit["b0"] - direct[0]) < 1e-8
        assert np.allclose([fit["O"][l] for l in lineups], direct[1:1 + Lc], atol=1e-8)
        assert np.allclose([fit["D"][l] for l in lineups], direct[1 + Lc:], atol=1e-8)
    del rng


def test_raw_baseline_adds_offense_and_allowed_and_counts_an_unseen_lineup_at_league_average():
    train = toy([("A", "B", 1.2, 10.0, "d"), ("B", "A", 0.8, 10.0, "d")])
    test = toy([("A", "B", 0, 1, "e"), ("A", "Z", 0, 1, "e"), ("Z", "Y", 0, 1, "e")])
    p = L.raw_predict(train, test)
    lg = 1.0
    assert np.isclose(p[0], 1.2 + 1.2 - lg) and np.isclose(p[1], 1.2) and np.isclose(p[2], lg)


def test_no_week_is_predicted_from_its_own_outcomes(dev):
    feats, obs = dev[0], dev[1]
    rows = L.weekly(obs, feats, {"shrink_zero": ({}, {}, 50.0)})
    weeks = L.week_of(obs["game_date"])
    assert rows["week"].min() == L.WARMUP_WEEKS
    # a week's prediction is unchanged when that week's outcomes are scrambled
    k = int(rows["week"].min())
    scrambled = obs.copy()
    m = weeks == k
    scrambled.loc[m, "y"] = np.random.default_rng(0).permutation(scrambled.loc[m, "y"].to_numpy())
    again = L.weekly(scrambled, feats, {"shrink_zero": ({}, {}, 50.0)})
    a, b = rows[rows["week"] == k], again[again["week"] == k]
    assert np.allclose(a["pred_shrink_zero"], b["pred_shrink_zero"]) and np.allclose(a["pred_raw"], b["pred_raw"])


def test_score_rows_reads_the_verdict_and_refuses_on_a_non_finite_value():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(1.1, 0.3, n)
    rows = pd.DataFrame({"game_id": [f"g{i % 400}" for i in range(n)], "week": (np.arange(n) % 400) // 50 + 4,
                         "w": 5.0, "y": y, "bin_poss": rng.choice([0.0, 50.0, 500.0], n)})
    rows["pred_raw"] = 1.1 + rng.normal(0, 0.3, n)
    for a in ("lrapm", "shrink_zero", "lrapm_net_prior"):
        rows[f"pred_{a}"] = 1.1
    rows["pred_rapm_synergy"] = 1.1 + rng.normal(0, 0.15, n)
    out = L.score_rows(rows, n=200)
    v = out["verdict"]
    assert out["relative_improvement_vs_raw"]["lrapm"]["point"] < -0.01 and v["passes"]
    assert v["against_raw"]["holds"] and v["against_rapm_synergy"]["holds"]
    assert out["paired_lrapm_minus_rapm_synergy"]["ci95"][1] < 0
    assert not v["toward_their_players_supported"]                   # lrapm equals shrink_zero here
    # equal to rapm_synergy: the rule fails against it, so the verdict fails
    rows2 = rows.assign(pred_rapm_synergy=1.1)
    v2 = L.score_rows(rows2, n=100)["verdict"]
    assert v2["against_raw"]["holds"] and not v2["against_rapm_synergy"]["holds"] and not v2["passes"]
    assert set(out["bins"]) == {"under_25", "25_to_100", "over_100"}
    rows.loc[5, "pred_lrapm"] = np.nan
    bad = L.score_rows(rows, n=50)
    assert bad["verdict"]["passes"] is None and "refused" in bad["verdict"]


def test_the_scorer_runs_through_score_once_on_a_pinned_root(dev, tmp_path):
    feats, obs = dev[0], dev[1]
    from analytics.lineups.train import fit_candidates
    od = L.fit_player_od(obs)
    fits = fit_candidates(feats, R.PBP_TEST_SEASON, E1.SPENT)
    lambdas = {name: 50.0 for name in L.PRIOR_ARMS}
    _f, _o, pbp, starters = tables("2025-11-01", 50, seed=2)
    root = tmp_path / "root"
    paths = dict(zip(("playbyplay", "starters", "schedule"), R.PBP_FILES))
    for key, frame in (("playbyplay", pbp), ("starters", starters), ("schedule", schedule_of(pbp))):
        p = root / paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(p, index=False)
    games = sorted(pbp["game_id"].unique())
    pins = {f: R.sha256_file(root / f) for f in R.PBP_FILES}
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.PBP: {"games": games, "sha256": R.list_hash(games),
                                                       "files_sha256": pins, "pins_sha256": R.map_hash(pins)}}}))
    ledger = tmp_path / "ledger.json"
    scorer = L.make_scorer(od, fits["rapm_synergy"], fits["rapm"]["player_coef"], lambdas, "abc")
    out = R.score_once("lineups-E2-rehearsal", R.PBP, scorer, testset_path=testset, ledger_path=ledger, data_root=root)
    assert out["test_games_listed"] == 50 and out["games_listed_without_playbyplay"] == 0
    assert out["schedule_agreement"]["share_equal"] == 1.0 and out["code_commit"] == "abc"
    assert out["results"]["non_finite_rows"] == 0 and out["results"]["verdict"]["passes"] in (True, False)
    assert set(out["results"]["mean_weekly_rmse"]) == set(L.ARMS)
    json.dumps(out, allow_nan=False)
    with pytest.raises(SystemExit):
        scorer(games)

"""tests/test_winprob.py: the winprob lane (analytics/winprob) on the synthetic fixtures of analytics.fixtures
(through the test adapter tests/_fx_winprob.py).

One trainer run is shared by the tests (module fixture) to keep the file fast: 8 games per season, 3 seasons
(one held out), 3 grouped folds, 5 bootstrap resamples. Every number here is fixture output, not a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import schema                                     # noqa: E402
from analytics.winprob import eda, features as F, train as T, validate as V   # noqa: E402
from tests._fx_winprob import synth_odds, synth_stints, team_names   # noqa: E402

SEASONS = ("2020-21", "2021-22", "2024-25")
HELD = "2024-25"


@pytest.fixture(scope="module")
def data():
    d = synth_stints(0, n_games=8, seasons=SEASONS)
    return {"playbyplay": d["playbyplay"], "games": d["games"], "odds": synth_odds(0, d["games"]), "files": ["<fixtures>"],
            "team_names": team_names(d["games"])}


@pytest.fixture(scope="module")
def states(data):
    return F.build_states(data["playbyplay"], data["games"], data["odds"], data["team_names"])


@pytest.fixture(scope="module")
def run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("winprob")
    T.FIT_LOG.clear()
    ctx = T.run(data, HELD, seed=0, n_boot=5, n_folds=3, out_root=out, fixtures=True, command="pytest",
                report=str(out / "winprob.md"))
    ctx["out_root"] = out
    ctx["fit_log"] = list(T.FIT_LOG)
    return ctx


# ── fixtures ───────────────────────────────────────────────────────────────────────────────────────
def test_fixture_is_deterministic_and_the_home_win_follows_the_final_margin(data):
    again = synth_stints(0, n_games=8, seasons=SEASONS)
    pd.testing.assert_frame_equal(again["playbyplay"], data["playbyplay"])
    g = data["games"]
    assert (g["home_win"] == (g["home_score"] > g["away_score"])).all()
    assert g["home_win"].nunique() == 2
    assert set(g["season"]) == set(SEASONS)
    derived = F.games_from_playbyplay(data["playbyplay"]).set_index("game_id")
    assert (derived.loc[g["game_id"], "home_score"].to_numpy() == g["home_score"].to_numpy()).all()


# ── validate ───────────────────────────────────────────────────────────────────────────────────────
def test_validate_passes_the_fixture_and_counts_injected_defects(data):
    res = V.validate(data["playbyplay"], data["games"], data["odds"], data["team_names"])
    assert res["ok"], res["problems"]
    assert res["counts"]["games"] == 24 and res["counts"]["duplicate_events"] == 0
    assert 0 < res["counts"]["spread_coverage"] <= 1
    bad = data["playbyplay"].copy()
    i = bad.index[bad["score"].notna()][5]
    bad.loc[i, "score"] = "12-x"
    j = bad.index[bad["score"].notna()][20]
    bad.loc[j, "score"] = "0-0"
    bad.loc[bad.index[3], "clock_s"] = 800.0
    k = bad.index[-1]
    bad.loc[k, ["period", "clock_s"]] = [5, 400.0]      # an overtime clock above 300 s
    bad = pd.concat([bad, bad.iloc[[7]]], ignore_index=True)
    res = V.validate(bad, data["games"], data["odds"])
    assert not res["ok"]
    c = res["counts"]
    assert c["score_unparsed"] == 1 and c["score_decreases"] >= 1 and c["clock_out_of_range"] == 2 and c["duplicate_events"] == 1
    assert any(p.startswith("monotone") for p in res["problems"])


# ── features ───────────────────────────────────────────────────────────────────────────────────────
def test_states_are_time_forward_and_well_formed(states, data):
    assert list(states.columns[:3]) == ["game_id", "season", "event"]
    assert len(states) == len(data["playbyplay"])
    for c in F.feature_columns():
        assert c in states.columns
    assert states["seconds_remaining"].between(0, 2880).all()
    reg = states[states["overtime"] == 0]
    assert ((4 - reg["period"]) * 720 + 0 <= reg["seconds_remaining"]).all()
    assert set(states["time_bucket"].unique()) <= set(F.BUCKET_NAMES)
    assert states["possession_home"].dropna().isin([0, 1]).all()
    assert 0 < states["possession_home"].notna().mean() <= 1.0
    sp = states.drop_duplicates("game_id")
    assert (sp["spread_missing"] == sp["spread_home"].isna().astype(int)).all()
    assert sp["spread_missing"].mean() < 1
    # the closing spread came from the odds table through the betting join (per-book pre-tip close, median over books)
    g = data["games"].set_index("game_id")["spread_home"].astype(float)
    have = sp[sp["spread_missing"] == 0]
    both = g.loc[have["game_id"]].notna().to_numpy()
    assert both.sum() >= 3 and np.nanmax(np.abs(have["spread_home"].to_numpy()[both] - g.loc[have["game_id"]].to_numpy()[both])) <= 1.5
    # with an odds table the games column is never read: a game absent from the odds is null with the flag
    gid = sp["game_id"].iloc[0]
    hk = F.TEAM_NAMES.get(int(data["games"].set_index("game_id").loc[gid, "home_team_id"]), None) or data["team_names"].get(
        int(data["games"].set_index("game_id").loc[gid, "home_team_id"]))
    o2 = data["odds"][~(data["odds"]["home_team"].astype(str) == str(hk))]
    s2 = F.build_states(data["playbyplay"][data["playbyplay"]["game_id"] == gid], data["games"][data["games"]["game_id"] == gid], o2,
                        data["team_names"])
    assert s2["spread_home"].isna().all() and (s2["spread_missing"] == 1).all()
    # without any odds table the games column is the fallback
    s3 = F.build_states(data["playbyplay"][data["playbyplay"]["game_id"] == gid], data["games"][data["games"]["game_id"] == gid], None)
    assert np.allclose(s3["spread_home"].to_numpy(), float(g.loc[gid]), equal_nan=True)
    X = F.design(states)
    assert "possession_missing" in X.columns and not X.isna().any().any()
    assert F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], n_games=2, cuts=2,
                                          team_names=data["team_names"]) == []


def test_overtime_seconds_and_buckets():
    assert F.seconds_remaining([1, 4, 5, 6], [720, 30, 300, 12]).tolist() == [2880, 30, 300, 12]
    ot = (np.array([1, 4, 5, 6]) > 4).astype(int)
    assert F.time_bucket(F.seconds_remaining([1, 4, 5, 6], [720, 30, 300, 12]), ot).tolist() == ["48-36", "4-0", "OT", "OT"]


def test_closing_spreads_land_on_the_right_games_with_real_ids_and_utc_tips():
    # two games on the same date (a date-only match cannot tell them apart) and a tip after midnight UTC
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "season": ["2023-24"] * 3,
                          "game_date": ["2024-01-05", "2024-01-05", "2024-01-06"],
                          "home_team_id": [1610612744, 1610612738, 1610612747], "away_team_id": [1610612747, 1610612748, 1610612744],
                          "home_win": [True, False, True]})
    def rows(key, commence, home, away, point, snap, close):
        return [{"game_key": key, "commence_time": commence, "home_team": home, "away_team": away, "book": "pinnacle",
                 "market": "spreads", "side": side, "point": pt, "price": -110, "snapshot_at": snap, "is_close": close,
                 "source": "the-odds-api.com", "endpoint": "historical/odds", "season": "2023-24", "fetched_at": "x"}
                for side, pt in ((home, point), (away, -point))]
    odds = pd.DataFrame(
        rows("2024-01-06_LosAngelesLakers_GoldenStateWarriors", "2024-01-06T03:00:00Z", "Golden State Warriors", "Los Angeles Lakers", -4.5, "2024-01-06T02:30:00Z", True)
        + rows("2024-01-06_LosAngelesLakers_GoldenStateWarriors", "2024-01-06T03:00:00Z", "Golden State Warriors", "Los Angeles Lakers", -3.0, "2024-01-05T12:00:00Z", False)
        + rows("2024-01-06_MiamiHeat_BostonCeltics", "2024-01-06T00:30:00Z", "Boston Celtics", "Miami Heat", -7.0, "2024-01-06T00:00:00Z", True)
        + rows("2024-01-07_GoldenStateWarriors_LosAngelesLakers", "2024-01-07T03:30:00Z", "Los Angeles Lakers", "Golden State Warriors", 2.5, "2024-01-07T03:00:00Z", True))
    sp = F.closing_spreads(odds, games)
    assert sp.loc["g1"] == -4.5 and sp.loc["g2"] == -7.0 and sp.loc["g3"] == 2.5
    # a snapshot after the tip is never the closing line
    late = odds.copy()
    late.loc[late["game_key"].str.endswith("BostonCeltics"), "snapshot_at"] = "2024-01-06T05:00:00Z"
    assert np.isnan(F.closing_spreads(late, games).loc["g2"])


def test_possession_indicator_reads_only_the_events_so_far():
    kind = ["period_start", "jump_ball", "shot_made", "shot_missed", "rebound", "turnover", "ft_made", "ft_made", "ft_missed", "rebound", "ft_made"]
    team = [None, "home", "home", "away", "away", "away", "home", "home", "away", "home", "away"]
    desc = ["", "", "", "MISS", "", "", "Free Throw 1 of 2", "Free Throw 2 of 2", "Free Throw 1 of 1 MISS", "", "Free Throw Technical"]
    out = F.possession_after(kind, team, desc, [1] * 11)
    exp = [np.nan, 1, 0, np.nan, 0, 1, 1, 0, np.nan, 1, 1]
    assert np.allclose(out, exp, equal_nan=True)


def test_a_feature_built_from_the_final_margin_is_caught(data):
    # by name: the outcome and the score never reach a design matrix
    with pytest.raises(ValueError, match="forbidden"):
        F.assert_no_leak(pd.DataFrame({"margin": [0.0], "home_win": [1]}))
    with pytest.raises(ValueError, match="forbidden"):
        F.assert_no_leak(pd.DataFrame({"final_margin": [3.0]}))
    with pytest.raises(ValueError):
        schema.assert_time_forward(pd.DataFrame({"margin": [0.0], "points": [2]}), "states_1hz")

    # by value: a builder that smuggles the final margin into a feature changes when the play-by-play is cut
    # (team_names go along because the betting join infers unknown fixture ids from date co-occurrence, which a
    # one-game cut cannot do; the real ids need none)
    tn = data["team_names"]

    def leaky(pbp, games, odds=None, **kw):
        s = F.build_states(pbp, games, odds, **kw)
        final = F.games_from_playbyplay(pbp).set_index("game_id")
        fm = s["game_id"].map(final["home_score"] - final["away_score"]).astype(float)
        s["margin_x_sqrt_time"] = fm * np.sqrt(s["seconds_remaining"])
        return s
    v = F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], build=leaky, n_games=2, cuts=2, team_names=tn)
    assert v and all(c == "margin_x_sqrt_time" for _, _, c in v)

    # games-table variant: a feature read from the games row's final score under a benign name is caught too,
    # because the probe scrambles the finals of the games row it passes
    def leaky_games(pbp, games, odds=None, **kw):
        s = F.build_states(pbp, games, odds, **kw)
        fm = games.set_index(games["game_id"].astype(str))
        s["seconds_remaining"] = s["seconds_remaining"] + s["game_id"].map(fm["home_score"] - fm["away_score"]).astype(float) * 0.001
        return s
    v = F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], build=leaky_games, n_games=2, cuts=2, team_names=tn)
    assert v and all(c in ("seconds_remaining", "margin_x_sqrt_time") for _, _, c in v)
    # and the states frame itself never carries the finals
    assert not ({"home_score", "away_score"} & set(states_cols(data)))
    assert F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], n_games=2, cuts=2,
                                          team_names=data["team_names"]) == []


def states_cols(data):
    return F.build_states(data["playbyplay"].head(60), data["games"], data["odds"], data["team_names"]).columns


# ── train ──────────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run_writes_report_and_tables(run):
    assert run["selected"] in T.MODELS
    rep = Path(run["out_root"]) / "winprob.md"
    text = rep.read_text()
    assert "fixture output, not a result" in text and "WinProbabilityPBP" in text and chr(0x2014) not in text
    assert run["prefix_violations"] == 0
    for m in T.MODELS:
        for b in ("48-36", "4-0"):
            assert 0 <= run["scores"][m][b]["brier"]["point"] <= 1
    # the margin and time carry signal on the fixture: the logistic beats the constant late in the game
    assert run["scores"]["logistic"]["4-0"]["brier"]["point"] < run["scores"]["constant"]["4-0"]["brier"]["point"]
    written = {Path(p).parent.name + "/" + Path(p).name for p in run["outputs"]}
    assert written == {"%s/%s" % (s, t) for s in SEASONS for t in ("winprob_states.parquet", "winprob_calibration.parquet")}
    st = pd.read_parquet(Path(run["out_root"]) / "analytics" / "winprob" / HELD / "winprob_states.parquet")
    assert list(st.columns) == ["game_id", "season", "event", "period", "seconds_remaining", "margin", "possession", "spread",
                                "p_home_win", "home_win", "fold", "heldout"]
    assert st["heldout"].all() and (st["fold"] == -1).all() and st["p_home_win"].between(0, 1).all()
    tr = pd.read_parquet(Path(run["out_root"]) / "analytics" / "winprob" / "2020-21" / "winprob_states.parquet")
    assert not tr["heldout"].any() and tr["fold"].between(0, 2).all()
    cal = pd.read_parquet(Path(run["out_root"]) / "analytics" / "winprob" / "2020-21" / "winprob_calibration.parquet")
    assert list(cal.columns) == ["season", "time_bucket", "n", "mean_p", "mean_y", "brier", "slope"]
    assert set(cal["time_bucket"]) <= set(F.BUCKET_NAMES) and cal["n"].sum() == len(tr)


def test_season_split_holds_a_whole_season_out_and_scores_it_once(run, states):
    held_ids = set(states.loc[states["season"] == HELD, "game_id"])
    assert held_ids and run["heldout_games"] == len(held_ids)
    for models, ids in run["fit_log"]:
        assert not (set(ids) & held_ids), "a held-out season game reached a fit (%s)" % (models,)
    outer = [e for e in run["fit_log"] if e[0] != ("hgb_inner",)]
    inner = [e for e in run["fit_log"] if e[0] == ("hgb_inner",)]
    # 3 outer folds (each with 3 inner isotonic fits), then exactly one refit on all training rows for the single
    # held-out scoring (with its inner fits only when the boosted candidate was selected)
    assert len(outer) == 4 and len(inner) == 3 * 3 + (3 if run["selected"] == "hgb_iso" else 0)
    last_models, last_ids = outer[-1]
    assert last_models == (run["selected"],) and set(last_ids) == set(states.loc[states["season"] != HELD, "game_id"])
    train_ids = set(states.loc[states["season"] != HELD, "game_id"])
    assert all(set(ids) < train_ids for _, ids in inner)
    assert set(run["train_seasons"]) == set(SEASONS) - {HELD}
    assert run["heldout_rows"] == int((states["season"] == HELD).sum())
    assert run["heldout_scores"]["4-0"]["n"] > 0


def test_split_refuses_2015_16_and_drops_heldout_flagged_rows(states):
    with pytest.raises(SystemExit):
        T.split_seasons(states, "2015-16")
    s = states.copy()
    flagged = s["game_id"].iloc[0]
    s.loc[s["game_id"] == flagged, "heldout"] = True
    train, heldout, ignored = T.split_seasons(s, HELD)
    assert ignored == [] and flagged not in set(train["game_id"]) and flagged not in set(heldout["game_id"])
    with pytest.raises(ValueError, match="held-out"):
        T.fit_models(s[s["game_id"] == flagged], models=("constant",))
    bad = states.copy()
    bad.loc[bad["game_id"] == flagged, "season"] = "2015-16"
    with pytest.raises(ValueError, match="2015-16"):
        T.fit_models(bad, models=("constant",))


# ── eda and clis ───────────────────────────────────────────────────────────────────────────────────
def test_eda_tables_never_read_the_held_out_season(states):
    t = eda.all_tables(states, seed=0, heldout_season=HELD)
    assert set(t) == {"margin_distribution", "time_distribution", "win_rate_by_margin_bucket", "spread_coverage", "shuffled_label_check"}
    assert set(t["spread_coverage"]["season"]) == set(SEASONS) - {HELD}
    assert int(t["shuffled_label_check"]["rows"].iloc[0]) == int((states["season"] != HELD).sum())
    wr = t["win_rate_by_margin_bucket"]
    assert int(wr[[c for c in wr.columns if c.startswith("n_")]].sum().sum()) == int((states["season"] != HELD).sum())
    sh = t["shuffled_label_check"].set_index("labels")
    assert sh.loc["shuffled labels (across games)", "brier_oof_logistic"] > sh.loc["real labels", "brier_oof_logistic"]
    assert sh.loc["shuffled labels (across games)", "brier_oof_logistic"] > 0.2


def test_cli_mains_on_fixtures(tmp_path, capsys):
    res = V.main(["--fixtures", "--out", str(tmp_path)])
    assert res["ok"] and (tmp_path / "analytics" / "winprob" / "validate.json").exists()
    summ = F.main(["--fixtures", "--out", str(tmp_path)])
    assert summ["rows"] > 0 and len(summ["written"]) == 2
    assert all(Path(w).name == "winprob_features.parquet" for w in summ["written"])
    assert "home_score" not in pd.read_parquet(summ["written"][0]).columns
    capsys.readouterr()


def test_bounded_slope_agrees_with_eval_calibration_slope():
    from analytics.eval import calibration_slope, calibration_slope_fast
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.5, 2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-0.7 * x))).astype(int)
    p = 1 / (1 + np.exp(-x))
    assert abs(calibration_slope_fast(y, p) - calibration_slope(y, p)) < 1e-2
    assert np.isnan(calibration_slope_fast(np.ones(10), np.full(10, 0.5)))
    sep = np.clip(np.where(y == 1, 0.999, 0.001), 1e-6, 1 - 1e-6)
    assert np.isfinite(calibration_slope_fast(y, sep))
    assert "calibration_slope_fast" in (Path(__file__).resolve().parents[1] / "analytics" / "winprob" / "train.py").read_text()


def test_four_game_fixture_root_runs_end_to_end(tmp_path):
    """run_all's smallest root (write_fixture_root n_games=4, heldout_every=4): no held-out season present, fixture
    team ids unknown to the odds feed, so the spread join relies on inference over the whole games table; the probe
    must see the same spread after a cut."""
    from analytics.fixtures import write_fixture_root
    root = tmp_path / "root"
    write_fixture_root(root, seed=0, n_games=4, heldout_every=4)
    d = F.load_data(root)
    assert F.prefix_invariance_violations(d["playbyplay"], d["games"], d["odds"], n_games=4, cuts=2, team_names=d["team_names"]) == []
    ctx = T.run(d, "2024-25", seed=0, n_boot=0, n_folds=2, out_root=tmp_path / "out", fixtures=True, command="pytest",
                report=str(tmp_path / "r.md"))
    assert ctx["heldout_rows"] == 0 and ctx["selected"] in T.MODELS and (tmp_path / "r.md").exists()


def test_seasons_later_than_the_held_out_season_are_ignored(data, states):
    # relabel the 2021-22 games as a season after the held-out one: they must not fit, score or appear in any fold
    s = states.copy()
    later = s["season"] == "2021-22"
    s.loc[later, "season"] = "2025-26"
    train, heldout, ignored = T.split_seasons(s, HELD)
    assert ignored == ["2025-26"] and set(train["season"]) == {"2020-21"} and set(heldout["season"]) == {HELD}
    pbp, games = data["playbyplay"].copy(), data["games"].copy()
    pbp.loc[pbp["season"] == "2021-22", "season"] = "2025-26"
    games.loc[games["season"] == "2021-22", "season"] = "2025-26"
    d2 = {**data, "playbyplay": pbp, "games": games}
    T.FIT_LOG.clear()
    ctx = T.run(d2, HELD, seed=0, n_boot=0, n_folds=2, fixtures=True, command="pytest")
    late_ids = set(games.loc[games["season"] == "2025-26", "game_id"].astype(str))
    assert ctx["ignored_seasons"] == ["2025-26"] and ctx["train_seasons"] == ["2020-21"]
    assert all(not (set(ids) & late_ids) for _, ids in T.FIT_LOG)
    text = T.write_report(Path(__import__("tempfile").mkdtemp()) / "r.md", ctx)
    assert "seasons later than the held-out season, ignored" in text and "2025-26 (" in text

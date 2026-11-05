"""tests/test_winprob.py: the winprob lane (analytics/winprob) on the synthetic fixtures of tests/_fx_winprob.py.

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
from tests._fx_winprob import synth_odds, synth_stints          # noqa: E402

SEASONS = ("2020-21", "2021-22", "2024-25")
HELD = "2024-25"


@pytest.fixture(scope="module")
def data():
    d = synth_stints(0, n_games=8, seasons=SEASONS)
    return {"playbyplay": d["playbyplay"], "games": d["games"], "odds": synth_odds(0, d["games"]), "files": ["<fixtures>"]}


@pytest.fixture(scope="module")
def states(data):
    return F.build_states(data["playbyplay"], data["games"], data["odds"])


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
    res = V.validate(data["playbyplay"], data["games"], data["odds"])
    assert res["ok"], res["problems"]
    assert res["counts"]["games"] == 24 and res["counts"]["duplicate_events"] == 0
    assert 0 < res["counts"]["spread_coverage"] <= 1
    bad = data["playbyplay"].copy()
    i = bad.index[bad["score"].notna()][5]
    bad.loc[i, "score"] = "12-x"
    j = bad.index[bad["score"].notna()][20]
    bad.loc[j, "score"] = "0-0"
    bad.loc[bad.index[3], "clock_s"] = 800.0
    bad = pd.concat([bad, bad.iloc[[7]]], ignore_index=True)
    res = V.validate(bad, data["games"], data["odds"])
    assert not res["ok"]
    c = res["counts"]
    assert c["score_unparsed"] == 1 and c["score_decreases"] >= 1 and c["clock_out_of_range"] == 1 and c["duplicate_events"] == 1
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
    assert 0.5 < states["possession_home"].notna().mean() < 1.0
    sp = states.drop_duplicates("game_id")
    assert (sp["spread_missing"] == sp["spread_home"].isna().astype(int)).all()
    assert 0 < sp["spread_missing"].mean() < 1
    # the spread came from the odds table, not the games column: the games column agrees where both exist
    g = data["games"].set_index("game_id")["spread_home"]
    have = sp[sp["spread_missing"] == 0]
    assert np.allclose(have["spread_home"].to_numpy(), g.loc[have["game_id"]].to_numpy())
    X = F.design(states)
    assert "possession_missing" in X.columns and not X.isna().any().any()
    assert F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], n_games=2, cuts=2) == []


def test_possession_indicator_reads_only_the_events_so_far():
    kind = ["period_start", "jump_ball", "shot_made", "shot_missed", "rebound", "turnover", "ft_made", "ft_made", "ft_missed", "rebound"]
    team = [None, "home", "home", "away", "away", "away", "home", "home", "away", "home"]
    desc = ["", "", "", "MISS", "", "", "Free Throw 1 of 2", "Free Throw 2 of 2", "Free Throw 1 of 1 MISS", ""]
    out = F.possession_after(kind, team, desc, [1] * 10)
    exp = [np.nan, 1, 0, np.nan, 0, 1, 1, 0, np.nan, 1]
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
    def leaky(pbp, games, odds=None):
        s = F.build_states(pbp, games, odds)
        final = F.games_from_playbyplay(pbp).set_index("game_id")
        fm = s["game_id"].map(final["home_score"] - final["away_score"]).astype(float)
        s["margin_x_sqrt_time"] = fm * np.sqrt(s["seconds_remaining"])
        return s
    v = F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], build=leaky, n_games=2, cuts=2)
    assert v and all(c == "margin_x_sqrt_time" for _, _, c in v)
    assert F.prefix_invariance_violations(data["playbyplay"], data["games"], data["odds"], n_games=2, cuts=2) == []


# ── train ──────────────────────────────────────────────────────────────────────────────────────────
def test_end_to_end_fixture_run_writes_report_and_tables(run):
    assert run["selected"] in T.MODELS
    rep = Path(run["out_root"]) / "winprob.md"
    text = rep.read_text()
    assert "fixture output, not a result" in text and "WinProbabilityPBP" in text and "—" not in text
    assert run["prefix_violations"] == 0
    for m in T.MODELS:
        for b in ("48-36", "4-0"):
            assert 0 <= run["scores"][m][b]["brier"]["point"] <= 1
    # the margin and time carry signal on the fixture: the logistic beats the constant late in the game
    assert run["scores"]["logistic"]["4-0"]["brier"]["point"] < run["scores"]["constant"]["4-0"]["brier"]["point"]
    written = {Path(p).parent.name + "/" + Path(p).name for p in run["written"]}
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
        assert not (set(ids) & held_ids), "a held-out season game reached a fit"
    # 3 outer folds, then exactly one refit on all training rows for the single held-out scoring
    assert len(run["fit_log"]) == 4
    last_models, last_ids = run["fit_log"][-1]
    assert last_models == (run["selected"],) and set(last_ids) == set(states.loc[states["season"] != HELD, "game_id"])
    assert set(run["train_seasons"]) == set(SEASONS) - {HELD}
    assert run["heldout_rows"] == int((states["season"] == HELD).sum())
    assert run["heldout_scores"]["4-0"]["n"] > 0


def test_split_refuses_2015_16_and_drops_heldout_flagged_rows(states):
    with pytest.raises(SystemExit):
        T.split_seasons(states, "2015-16")
    s = states.copy()
    flagged = s["game_id"].iloc[0]
    s.loc[s["game_id"] == flagged, "heldout"] = True
    train, heldout = T.split_seasons(s, HELD)
    assert flagged not in set(train["game_id"]) and flagged not in set(heldout["game_id"])
    with pytest.raises(ValueError, match="held-out"):
        T.fit_models(s[s["game_id"] == flagged], models=("constant",))
    bad = states.copy()
    bad.loc[bad["game_id"] == flagged, "season"] = "2015-16"
    with pytest.raises(ValueError, match="2015-16"):
        T.fit_models(bad, models=("constant",))


# ── eda and clis ───────────────────────────────────────────────────────────────────────────────────
def test_eda_tables(states):
    t = eda.all_tables(states, seed=0)
    assert set(t) == {"margin_distribution", "time_distribution", "win_rate_by_margin_bucket", "spread_coverage", "shuffled_label_check"}
    assert len(t["spread_coverage"]) == len(SEASONS)
    sh = t["shuffled_label_check"].set_index("labels")
    assert sh.loc["shuffled labels (across games)", "brier_oof_logistic"] > sh.loc["real labels", "brier_oof_logistic"]
    assert sh.loc["shuffled labels (across games)", "brier_oof_logistic"] > 0.2


def test_cli_mains_on_fixtures(tmp_path, capsys):
    res = V.main(["--fixtures", "--out", str(tmp_path)])
    assert res["ok"] and (tmp_path / "analytics" / "winprob" / "validate.json").exists()
    summ = F.main(["--fixtures", "--out", str(tmp_path)])
    assert summ["rows"] > 0 and len(summ["written"]) == 2
    capsys.readouterr()

"""analytics.lineups: stints from substitutions, validation, features, the season-split guard, end-to-end training.

Runs on the local fallback generator of tests/_fx_lineups.py (fixed rosters, repeating lineups, hidden ratings and
a lineup interaction); the shared generator, when importable, is exercised by the builder and validate tests only.
Fixture numbers are never results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import schema                                   # noqa: E402
from analytics.lineups import SECONDS_PER_POSS, lineup_id      # noqa: E402
from analytics.lineups import eda, features as F, stints as S, train as T, validate as V   # noqa: E402
from tests import _fx_lineups as fx                            # noqa: E402

SEASONS3 = ("2020-21", "2021-22", "2022-23")


def local_fixture(n_games=30, seasons=SEASONS3, seed=0):
    from analytics.lineups import fixture_tables
    return fixture_tables(seed, n_games=n_games, seasons=seasons, generator="local")


# ----------------------------------------------------------------------------- hand-built substitution sequence

def _ev(event, period, clock, team, kind, desc="", points=0, p1=None, p2=None, team_id=None):
    return {"game_id": "g1", "event": event, "period": period, "clock_s": float(clock), "team": team, "desc": desc, "points": points,
            "kind": kind, "player1_id": p1, "player2_id": p2, "player3_id": None, "team_id": team_id, "season": "2020-21"}


def hand_pbp_ids() -> pd.DataFrame:
    """Period 1 (720 s): home starters 1..5 appear by events, away starters 11..15. At 600 s remaining home 5 out, 6 in;
    at 300 s away 15 out, 16 in and home 6 out, 5 in (same cut). Points: home 2 at 700, away 3 at 500, home 2 at 100.
    Period 2: both sides keep their last lineups (every player appears), one home sub at 360 s."""
    rows = [
        _ev(0, 1, 720, None, "period_start", "Start of 1st period"),
        _ev(1, 1, 710, "home", "shot_missed", p1=1), _ev(2, 1, 705, "away", "rebound", p1=11),
        _ev(3, 1, 700, "home", "shot_made", points=2, p1=2), _ev(4, 1, 690, "away", "turnover", p1=12),
        _ev(5, 1, 680, "home", "foul", p1=3), _ev(6, 1, 670, "away", "shot_missed", p1=13), _ev(7, 1, 660, "home", "rebound", p1=4),
        _ev(8, 1, 650, "away", "foul", p1=14),
        _ev(9, 1, 600, "home", "substitution", "P6 enters the game for P5", p1=5, p2=6),
        _ev(10, 1, 500, "away", "shot_made", points=3, p1=15),
        _ev(11, 1, 300, "away", "substitution", "P16 enters the game for P15", p1=15, p2=16),
        _ev(12, 1, 300, "home", "substitution", "P5 enters the game for P6", p1=6, p2=5),
        _ev(13, 1, 100, "home", "shot_made", points=2, p1=5),
        _ev(14, 1, 0, None, "period_end", "End of 1st period"),
        _ev(15, 2, 720, None, "period_start", "Start of 2nd period"),
    ]
    k = 16
    for i, p in enumerate([1, 2, 3, 4, 5]):
        rows.append(_ev(k, 2, 700 - 5 * i, "home", "shot_missed", p1=p)); k += 1
    for i, p in enumerate([11, 12, 13, 14, 16]):
        rows.append(_ev(k, 2, 650 - 5 * i, "away", "rebound", p1=p)); k += 1
    rows.append(_ev(k, 2, 360, "home", "substitution", "P7 enters the game for P1", p1=1, p2=7)); k += 1
    rows.append(_ev(k, 2, 200, "home", "shot_made", points=3, p1=7)); k += 1
    rows.append(_ev(k, 2, 0, None, "period_end", "End of 2nd period"))
    return pd.DataFrame(rows)


def test_builder_on_a_hand_built_substitution_sequence():
    st = S.build_game_stints(hand_pbp_ids())
    assert st["start_s"].tolist() == [0.0, 120.0, 420.0, 720.0, 1080.0]
    assert st["end_s"].tolist() == [120.0, 420.0, 720.0, 1080.0, 1440.0]
    assert st["home_ids"].tolist() == ["1|2|3|4|5", "1|2|3|4|6", "1|2|3|4|5", "1|2|3|4|5", "2|3|4|5|7"]
    assert st["away_ids"].tolist() == ["11|12|13|14|15", "11|12|13|14|15", "11|12|13|14|16", "11|12|13|14|16", "11|12|13|14|16"]
    assert st["lineup_ok"].all() and (st["n_home"] == 5).all()
    assert st["points_home"].tolist() == [2, 0, 2, 0, 3] and st["points_away"].tolist() == [0, 3, 0, 0, 0]
    assert st["score_home"].tolist() == [2, 2, 4, 4, 7] and st["score_away"].tolist() == [0, 3, 3, 3, 3]
    assert st["period_start"].tolist() == [1, 1, 1, 2, 2] and st["period_end"].tolist() == [1, 1, 1, 2, 2]
    assert st["stint"].tolist() == list(range(5)) and (st["endpoint"] == S.ENDPOINT).all()
    assert st["duration_s"].sum() == pytest.approx(1440.0)


def test_builder_marks_a_missing_player_and_uses_the_starters_table():
    pbp = hand_pbp_ids()
    pbp = pbp[~((pbp["period"] == 1) & (pbp["player1_id"] == 4))]        # home 4 never appears in period 1
    st = S.build_game_stints(pbp)
    p1 = st[st["period_start"] == 1]
    assert (p1["n_home"] == 4).all() and (~p1["lineup_ok"]).all() and p1["home_5"].isna().all()
    assert (st[st["period_start"] == 2]["n_home"] == 5).all()
    st2 = S.build_game_stints(pbp, starters={"home": [1, 2, 3, 4, 5], "away": [11, 12, 13, 14, 15]})
    assert st2["lineup_ok"].all() and st2["home_ids"].iloc[0] == "1|2|3|4|5"


def test_starters_table_with_slugs_only_keys_on_side_and_name():
    """The basketball-reference box score driver writes starters without NBA.com ids (team_id and player_id null):
    the rows key on side plus player (the play-by-play name) or player_slug through the same name key the builder
    gives basketball-reference play-by-play names."""
    names_h = ["S. Curry", "D. Green", "K. Thompson", "A. Iguodala", "Z. Pachulia"]
    names_a = ["K. Durant", "R. Westbrook", "S. Adams", "A. Roberson", "V. Oladipo"]
    rows = []
    for side, names, team in (("home", names_h, "GSW"), ("away", names_a, "OKC")):
        for n in names:
            rows.append({"game_id": "201602270OKC", "game": "02.27.2016.GSW.at.OKC", "game_date": "2016-02-27", "season": "2015-16",
                         "side": side, "team": team, "team_id": pd.NA, "player_slug": n.lower().replace(". ", "_"), "player": n, "player_id": pd.NA})
    starters = pd.DataFrame(rows).astype({"team_id": "Int64", "player_id": "Int64"})
    st = S.starters_by_side(starters, "201602270OKC", None, None)
    assert st["home"] == [S.name_key(n) for n in names_h] and st["away"] == [S.name_key(n) for n in names_a]
    assert S.starters_by_side(starters, "02.27.2016.GSW.at.OKC", None, None) == st        # the game column matches too
    slug_only = starters.drop(columns=["player"])
    assert S.starters_by_side(slug_only, "201602270OKC", None, None)["home"] == [S.name_key(r) for r in starters.loc[starters["side"] == "home", "player_slug"]]
    assert S.starters_by_side(starters, "201602270OKC", None, None, roster={"S. Curry": 201939})["home"][0] == 201939
    # the id path still wins when ids are present
    ids = starters.copy()
    ids["team_id"] = [1 if r == "home" else 2 for r in ids["side"]]
    ids["player_id"] = range(100, 110)
    assert S.starters_by_side(ids, "201602270OKC", 1, 2) == {"home": list(range(100, 105)), "away": list(range(105, 110))}
    # a bref play-by-play game: the slug starters seed period 1 so a quiet starter is still found
    pbp = pd.DataFrame([
        _ev(0, 1, 700, "home", "shot_made", "D. Green makes 2-pt layup (assist by S. Curry)", points=2),
        _ev(1, 1, 650, "away", "rebound", "Defensive rebound by K. Durant"),
        _ev(2, 1, 600, "home", "other", "J. Bell enters the game for Z. Pachulia"),
        _ev(3, 1, 100, "away", "shot_made", "K. Durant makes 2-pt dunk", points=2),
    ])
    out = S.build_game_stints(pbp, starters=st)
    assert out["n_home"].tolist() == [5, 5] and out["n_away"].tolist() == [5, 5] and out["lineup_ok"].all()
    assert S.name_key("J. Bell") in [int(x) for x in out["home_ids"].iloc[1].split("|")]


def test_builder_reads_basketball_reference_descriptions_without_ids():
    rows = [
        _ev(0, 1, 710, "home", "shot_missed", "S. Curry misses 3-pt jump shot from 26 ft"),
        _ev(1, 1, 705, "away", "rebound", "Defensive rebound by K. Durant"),
        _ev(2, 1, 700, "home", "shot_made", "D. Green makes 2-pt layup (assist by S. Curry)", points=2),
        _ev(3, 1, 690, "home", "foul", "Shooting foul by K. Thompson (drawn by K. Durant)"),
        _ev(4, 1, 680, "home", "turnover", "Turnover by A. Iguodala (bad pass)"),
        _ev(5, 1, 670, "home", "rebound", "Offensive rebound by Z. Pachulia"),
        _ev(6, 1, 600, "home", "other", "J. Bell enters the game for Z. Pachulia"),
        _ev(7, 1, 100, "home", "shot_made", "J. Bell makes 2-pt dunk", points=2),
    ]
    st = S.build_game_stints(pd.DataFrame(rows))
    names = ["S. Curry", "D. Green", "K. Thompson", "A. Iguodala", "Z. Pachulia"]
    assert st["home_ids"].iloc[0] == "|".join(map(str, sorted(S.name_key(n) for n in names)))
    assert st["n_home"].tolist() == [5, 5] and st["n_away"].tolist() == [1, 1]
    assert S.name_key("Z. Pachulia") not in [int(x) for x in st["home_ids"].iloc[1].split("|")]
    assert S.name_key("J. Bell") in [int(x) for x in st["home_ids"].iloc[1].split("|")]
    assert S.name_key("S. Curry", {"S. Curry": 201939}) == 201939
    assert S.bref_substitution("A. One enters the game for C. Three") == ("A. One", "C. Three")
    assert S.bref_actor("Turnover by J. Jackson Jr. (bad pass)") == "J. Jackson Jr."
    assert st["points_home"].tolist() == [2, 2]


def test_builder_reproduces_the_fixture_stints():
    d = local_fixture(n_games=4, seasons=("2020-21",))
    st = S.build_stints(d["playbyplay"], d["games"], d["starters"])
    truth = d["stints"]
    m = st.merge(truth, on=["game_id", "start_s", "end_s"], suffixes=("", "_t"))
    assert len(m) == len(truth) == len(st)
    ok = m[m["lineup_ok"]]
    assert len(ok) >= 0.8 * len(m)
    assert (ok["home_ids"] == ok["home_ids_t"]).all() and (ok["away_ids"] == ok["away_ids_t"]).all()
    assert (m["points_home"] == m["points_home_t"]).all() and (m["points_away"] == m["points_away_t"]).all()
    assert (m["score_home"] == m["score_home_t"]).all()


def test_builder_runs_on_the_shared_generator_when_present():
    d = fx.synth_stints(1, n_games=2, seasons=("2020-21",))
    st = S.build_stints(d["playbyplay"], d["games"])
    assert len(st) and st["lineup_ok"].mean() > 0.5
    assert set(V.REQUIRED) <= set(st.columns)
    clean, counts = V.validate(st)
    assert counts["rows_out"] == int(st["lineup_ok"].sum()) - counts["dropped_duration"] - counts["dropped_duplicate"] - counts["dropped_points"]


def test_stints_cli_writes_per_season(tmp_path):
    s = S.main(["--fixtures", "--fixture-generator", "local", "--n-games", "12", "--out", str(tmp_path)])
    assert s["fixture"] and s["stints"] > 0 and s["lineup_ok_share"] > 0.8
    files = {Path(w).parent.name for w in s["written"]}
    assert files == set(SEASONS3)
    df = pd.read_parquet(tmp_path / "analytics" / "lineups" / "2020-21" / "stints.parquet")
    assert set(V.REQUIRED) <= set(df.columns)


# ----------------------------------------------------------------------------- validate

def test_validate_drops_and_counts():
    d = local_fixture(n_games=6, seasons=("2020-21",))
    st = d["stints"].copy()
    n = len(st)
    bad = st.iloc[:3].copy()
    bad.loc[bad.index[0], ["n_home", "lineup_ok"]] = [4, False]
    bad.loc[bad.index[1], "duration_s"] = 0.0
    bad.loc[bad.index[2], "points_away"] = 99
    bad["stint"] = [9001, 9002, 9003]
    dup = st.iloc[[5]].copy()
    held = st.iloc[[6]].copy()
    held["heldout"] = True
    held["stint"] = 9004
    df = pd.concat([st, bad, dup, held], ignore_index=True)
    clean, c = V.validate(df)
    assert c["rows_in"] == n + 5 and c["rows_out"] == n
    assert c["dropped_not_5v5"] == 1 and c["dropped_duration"] == 1 and c["dropped_points"] == 1 and c["dropped_duplicate"] == 1 and c["dropped_heldout"] == 1
    assert 0 < c["lineup_ok_share"] < 1
    assert not clean.duplicated(["game_id", "stint"]).any()
    with pytest.raises(ValueError, match="missing column"):
        V.validate(st.drop(columns=["home_3"]))


def test_validate_cli_writes_counts(tmp_path):
    c = V.main(["--fixtures", "--fixture-generator", "local", "--out", str(tmp_path)])
    j = json.loads((tmp_path / "analytics" / "lineups" / "validate.json").read_text())
    assert j["rows_out"] == c["rows_out"] and j["fixture"] is True and j["dropped_not_5v5"] == 0


# ----------------------------------------------------------------------------- features

def test_features_frame_target_and_guards():
    d = local_fixture(n_games=6, seasons=("2020-21",))
    f = F.features(d["stints"], d["games"])
    assert F.feature_columns() == list(F.FEATURES) and len(f) == len(d["stints"])
    assert f["possessions"].to_numpy() == pytest.approx(d["stints"]["duration_s"].to_numpy() / SECONDS_PER_POSS)
    exp = (d["stints"]["points_home"] - d["stints"]["points_away"]).to_numpy() / f["possessions"].to_numpy() * 100
    assert f[F.TARGET].to_numpy() == pytest.approx(exp)
    assert f["lineup_home"].iloc[0] == lineup_id(d["stints"].iloc[0][[f"home_{k}" for k in range(1, 6)]])
    assert f["home_team_id"].notna().all() and set(f["home_team_id"].unique()) <= set(d["games"]["home_team_id"])
    schema.assert_time_forward(f, "possessions")
    with pytest.raises(ValueError, match="forbidden"):
        F.assert_no_leak(f[F.feature_columns() + ["points_diff"]])
    with pytest.raises(ValueError):
        schema.assert_time_forward(f.assign(points=1), "possessions")
    st = d["stints"].assign(possessions=7.0)
    assert (F.features(st, d["games"])["possessions"] == 7.0).all()


def test_features_cli(tmp_path):
    s = F.main(["--fixtures", "--fixture-generator", "local", "--out", str(tmp_path)])
    assert s["rows"] > 0 and s["target"] == "diff_per100" and len(s["written"]) == 3


# ----------------------------------------------------------------------------- season split guard

def test_season_split_guard_refuses_held_out_and_later_seasons():
    d = local_fixture(n_games=6, seasons=SEASONS3)
    f = F.features(d["stints"], d["games"])
    T.assert_season_split(f[f["season"] == "2020-21"], "2021-22", "2022-23")
    with pytest.raises(ValueError, match="held-out season"):
        T.assert_season_split(f, "2023-24", "2022-23")
    with pytest.raises(ValueError, match="not before the target"):
        T.assert_season_split(f[f["season"] != "2022-23"], "2021-22", "2022-23")
    with pytest.raises(ValueError, match="held-out season"):
        T.fit_candidates(f, "2023-24", "2022-23", n_folds=2)
    with pytest.raises(ValueError):
        T.fit_candidates(f, "2021-22", "2099-00", n_folds=2)


def test_a_held_out_season_never_reaches_a_fit(monkeypatch):
    d = local_fixture()
    f = F.features(d["stints"], d["games"])
    seen = []
    real = T.fit_candidates

    def spy(train, target_season, heldout_season, *a, **k):
        seen.append((sorted(train["season"].unique()), target_season, heldout_season))
        return real(train, target_season, heldout_season, *a, **k)

    monkeypatch.setattr(T, "fit_candidates", spy)
    ctx = T.run(f, "2022-23", seed=0, n_boot=10, n_folds=3)
    assert seen, "no fit happened"
    for seasons, target, held in seen:
        assert "2022-23" not in seasons and held == "2022-23"
        if target is not None:
            assert all(T.season_key(s) < T.season_key(target) for s in seasons)
    assert ctx["heldout"]["n_lineups"] > 0
    assert sum(1 for s, t, h in seen if t == "2022-23") == 1, "the held-out season is scored once"
    with pytest.raises(SystemExit):
        T.run(f, "2020-21", n_boot=5)      # nothing before the held-out season


def test_tripwire_in_the_trainer_source():
    src = (ROOT / "analytics" / "lineups" / "train.py").read_text()
    assert "refuse_heldout(" in src and "assert_time_forward(" in src and "assert_season_split(" in src


# ----------------------------------------------------------------------------- end to end

def test_train_end_to_end_on_the_local_fixture(tmp_path):
    report = tmp_path / "lineups.md"
    ctx = T.main(["--fixtures", "--fixture-generator", "local", "--report", str(report), "--out", str(tmp_path), "--bootstrap", "20", "--folds", "3"])
    assert ctx["fixture"] and ctx["heldout_season"] == "2022-23" and ctx["train_seasons"] == ["2020-21", "2021-22"]
    assert ctx["selected"] in T.CANDIDATES and len(ctx["pairs"]) == 1
    text = report.read_text()
    assert "fixture output, not a result" in text and chr(0x2014) not in text
    for c in T.CANDIDATES:
        assert "| %s" % c in text
    assert "python -m analytics.lineups.train --data" in text
    for season in SEASONS3:
        lr = pd.read_parquet(tmp_path / "analytics" / "lineups" / season / "lineup_ratings.parquet")
        assert list(lr.columns[:10]) == ["season", "lineup_id", "team_id", "n_possessions", "plus_minus_per100", "rapm_sum_per100",
                                         "synergy_per100", "lo", "hi", "insufficient"]
        assert (lr["season"] == season).all() and lr["lineup_id"].is_unique and lr["team_id"].notna().all()
        assert (lr["insufficient"] == (lr["n_possessions"] < 300)).all()
        assert (lr["synergy_per100"][lr["insufficient"]] == 0).all()
        pr = pd.read_parquet(tmp_path / "analytics" / "lineups" / season / "player_rapm.parquet")
        assert list(pr.columns) == ["season", "player_id", "rapm", "lo", "hi", "team_id", "as_of", "in_sample", "fit_seasons"] and pr["player_id"].is_unique
        assert pr["team_id"].notna().all() and pr["as_of"].str.match(r"\d{4}-\d{2}-\d{2}").all()
        multi = lr["n_games"] >= 2
        assert (lr.loc[multi, "lo"] <= lr.loc[multi, "plus_minus_per100"] + 1e-9).all()
    held = pd.read_parquet(tmp_path / "analytics" / "lineups" / "2022-23" / "lineup_ratings.parquet")
    assert (held["fit_seasons"] == "2020-21,2021-22").all() and (~held["in_sample"]).all()
    pr_held = pd.read_parquet(tmp_path / "analytics" / "lineups" / "2022-23" / "player_rapm.parquet")
    d0 = local_fixture()
    assert (~pr_held["in_sample"]).all() and (pr_held["as_of"] == d0["games"].loc[d0["games"]["season"] == "2021-22", "game_date"].max()).all()
    assert (pr_held["as_of"] < d0["games"].loc[d0["games"]["season"] == "2022-23", "game_date"].min()).all()
    # RAPM recovers the hidden ratings (fixture: a sanity check of the fit, not a result)
    d = local_fixture()
    truth = d["truth"]["ratings"]
    pr = pd.read_parquet(tmp_path / "analytics" / "lineups" / "2021-22" / "player_rapm.parquet")
    r = np.corrcoef(pr["rapm"], [truth[int(p)] for p in pr["player_id"]])[0, 1]
    assert r > 0.2, r
    m = ctx["heldout"]["metrics"]
    assert np.isfinite(m["rapm"]["spearman"]["point"]) and np.isfinite(m["plus_minus"]["mse"]["point"])
    assert m["rapm"]["spearman_vs_plus_minus"] is not None and m["plus_minus"]["spearman_vs_plus_minus"] is None


def test_train_runs_on_the_shared_generator_when_present(tmp_path):
    """The shared generator (analytics.fixtures.synth_stints, when importable) draws a fresh five per stint, so no
    lineup repeats across seasons: the trainer must still run, score what it can and fall back to plus_minus."""
    report = tmp_path / "shared.md"
    ctx = T.main(["--fixtures", "--fixture-generator", "auto", "--n-games", "8", "--report", str(report), "--out", str(tmp_path),
                  "--bootstrap", "10", "--folds", "3"])
    assert ctx["fixture"] and ctx["selected"] in T.CANDIDATES and "fixture output, not a result" in report.read_text()
    assert ctx["inputs"]["source"].startswith("fixture generator ")
    for season in SEASONS3:
        lr = pd.read_parquet(tmp_path / "analytics" / "lineups" / season / "lineup_ratings.parquet")
        pr = pd.read_parquet(tmp_path / "analytics" / "lineups" / season / "player_rapm.parquet")
        assert len(lr) and len(pr) and lr["lineup_id"].is_unique and pr["rapm"].notna().all()


def test_ridge_recovers_a_planted_player_effect():
    rng = np.random.default_rng(3)
    n = 600
    rows = []
    for i in range(n):
        home = rng.choice(np.arange(1, 10), 5, replace=False)
        away = rng.choice(np.arange(11, 20), 5, replace=False)
        y = 10.0 * (1 in home) - 10.0 * (11 in away) + rng.normal(0, 5)
        rows.append({"game_id": "g%d" % (i // 30), "season": "2020-21", "stint": i, "possessions": 20.0, "lineup_home": lineup_id(home),
                     "lineup_away": lineup_id(away), "home_team_id": 1, "away_team_id": 2, "points_diff": y * 0.2, F.TARGET: y,
                     **{f"home_{k + 1}": int(v) for k, v in enumerate(sorted(home))}, **{f"away_{k + 1}": int(v) for k, v in enumerate(sorted(away))}})
    f = pd.DataFrame(rows)
    fit = T.fit_ridge(f, list(range(1, 20)), [], None, 3, 0)
    coef = fit["player_coef"]
    assert coef[1] > 5 and coef[11] > 5 and max(abs(coef[p]) for p in range(2, 10)) < 4
    assert fit["alpha"] in T.ALPHAS and fit["cv_mse"]


# ----------------------------------------------------------------------------- eda, fixture, hygiene

def test_eda_tables():
    d = local_fixture(n_games=6, seasons=("2020-21", "2021-22"))
    f = F.features(d["stints"], d["games"])
    assert eda.duration_summary(d["stints"])["season"].tolist() == ["2020-21", "2021-22"]
    assert eda.differential_summary(f)["stints"].sum() == len(f)
    lp = eda.lineup_possessions(f, gate=50)
    assert (lp["pass_gate_50"] <= lp["lineups"]).all()
    pm = eda.player_minutes(d["stints"], top=3)
    assert (pm.groupby("season").size() == 3).all()
    cs = eda.cross_season_correlation(f, min_poss=1)
    assert cs.columns.tolist() == ["season_from", "season_to", "n_lineups", "pearson", "spearman"] and len(cs) == 1


def test_fallback_fixture_is_deterministic_and_well_formed():
    a = fx._fallback_synth_stints(5, n_games=3, seasons=("2021-22",))
    b = fx._fallback_synth_stints(5, n_games=3, seasons=("2021-22",))
    pd.testing.assert_frame_equal(a["stints"], b["stints"])
    pd.testing.assert_frame_equal(a["playbyplay"], b["playbyplay"])
    st = a["stints"]
    assert set(V.REQUIRED) <= set(st.columns) and st["lineup_ok"].all()
    assert (st.groupby("game_id")["duration_s"].sum().round(1) == 2880.0).all()
    g = a["games"].set_index("game_id")
    last = st.sort_values("stint").groupby("game_id").last()
    assert (last["score_home"] == g.loc[last.index, "home_score"]).all()
    sub = a["playbyplay"][a["playbyplay"]["kind"] == "substitution"]
    assert len(sub) and sub["player1_id"].notna().all() and sub["player2_id"].notna().all()
    assert set(a["games"].columns) >= {"game_id", "season", "game_date", "home_team_id", "away_team_id", "home_score", "away_score", "home_win", "spread_home"}


def test_no_em_dash_in_lane_files():
    files = list((ROOT / "analytics" / "lineups").glob("*.py")) + [ROOT / "tests" / "test_lineups.py", ROOT / "tests" / "_fx_lineups.py",
                                                                     ROOT / "docs" / "contract" / "lineups.md", ROOT / "reports" / "lineups.md",
                                                                     ROOT / "notebooks" / "eda_lineups.ipynb"]
    bad = [str(p) for p in files if p.exists() and chr(0x2014) in p.read_text()]
    assert not bad, bad

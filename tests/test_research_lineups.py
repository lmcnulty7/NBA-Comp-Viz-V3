"""lineups-E1 machinery (docs/research/lineups.md): the replacement design, the game sums, the paired bootstrap, and a
rehearsal of the scorer through score_once on synthetic play-by-play in a temporary pinned root. No real data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import research_lineups as L
from tools import research_testset as R

TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
SURNAMES = ("Hone", "Htwo", "Hthree", "Hfour", "Hfive", "Hsix")


def roster(team):
    return [f"{chr(65 + k)}{team.lower()}x {SURNAMES[k]}{team.lower()}" for k in range(6)]


def abbrev(name):
    first, last = name.split(" ", 1)
    return f"{first[0]}. {last}"


def synth_games(dates, seed):
    """Basketball-reference-like play-by-play and starters: every on-court player acts in every period, one
    substitution in the fourth quarter, points drawn from a team strength."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(TEAMS, np.linspace(-0.25, 0.25, len(TEAMS)))}
    pbp, starters = [], []
    for i, date in enumerate(dates):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        gid = f"{date.replace('-', '')}0{home}"
        season = R.pbp_season(gid)
        event = 0
        for side, team in (("home", home), ("away", away)):
            for name in roster(team)[:5]:
                starters.append({"game_id": gid, "game": f"x.{gid}", "game_date": date, "season": season,
                                 "side": side, "team": team, "team_id": pd.NA, "player_slug": None, "player": name,
                                 "player_id": pd.NA})
        for period in (1, 2, 3, 4):
            clock = 700.0
            for k in range(5):
                for side, team in (("home", home), ("away", away)):
                    name = abbrev(roster(team)[k])
                    made = rng.random() < 0.5 + strength[team] + (0.04 if side == "home" else 0.0)
                    pbp.append({"game_id": gid, "event": event, "period": period, "clock_s": clock, "team": side,
                                "desc": f"{name} {'makes' if made else 'misses'} 2-pt jump shot from 10 ft",
                                "points": 2 if made else 0, "kind": "shot", "season": season, "game_date": date,
                                "season_type": "Regular Season"})
                    event += 1
                    clock -= 20.0
            if period == 4:
                for side, team in (("home", home), ("away", away)):
                    r = roster(team)
                    pbp.append({"game_id": gid, "event": event, "period": 4, "clock_s": 300.0, "team": side,
                                "desc": f"{abbrev(r[5])} enters the game for {abbrev(r[0])}", "points": 0,
                                "kind": "substitution", "season": season, "game_date": date,
                                "season_type": "Regular Season"})
                    event += 1
                for side, team in (("home", home), ("away", away)):
                    made = rng.random() < 0.5 + strength[team]
                    pbp.append({"game_id": gid, "event": event, "period": 4, "clock_s": 200.0, "team": side,
                                "desc": f"{abbrev(roster(team)[5])} {'makes' if made else 'misses'} 2-pt layup",
                                "points": 2 if made else 0, "kind": "shot", "season": season, "game_date": date,
                                "season_type": "Regular Season"})
                    event += 1
    return pd.DataFrame(pbp), pd.DataFrame(starters)


def dates(start, n):
    return [str(d.date()) for d in pd.date_range(start, periods=n, freq="D")]


@pytest.fixture(scope="module")
def dev_tables():
    pbp, starters = synth_games(dates("2022-11-01", 60), seed=1)
    feats, games, dropped, stints = L.season_tables(pbp, starters)
    return feats, games, dropped, stints


def test_the_frozen_builder_makes_full_lineups_and_whole_game_margins(dev_tables):
    feats, games, dropped, stints = dev_tables
    assert dropped == [] and len(games) == 60
    assert stints["lineup_ok"].all() and len(feats) == len(stints)
    assert (games["actual_all"] == games["actual_valid"]).all()
    assert np.allclose(games["poss_all"], games["poss_valid"])


def test_replacement_design_counts_low_minute_and_unseen_players_per_side():
    feats = pd.DataFrame({f"{s}_{k}": [k + (0 if s == "home" else 10)] for s in ("home", "away") for k in range(1, 6)})
    feats["possessions"] = 10.0
    X = L.replacement_design(feats, retained=[1, 2, 11]).toarray()[0]
    # [1 | players 1, 2, 11 | home count | away count]
    assert list(X) == [1.0, 1.0, 1.0, -1.0, 3.0, 4.0]


def test_every_arm_rates_every_stint_and_the_floor_is_the_weighted_mean(dev_tables):
    feats = dev_tables[0]
    model = L.fit_arms(feats, None)
    rep = model["rapm_replacement"]
    assert rep["rho"] in L.RHO_GRID and rep["players_retained"] <= rep["players_seen"]
    rat = L.stint_ratings(model, feats)
    assert set(rat) == set(L.ARMS) and all(len(v) == len(feats) for v in rat.values())
    assert np.allclose(rat["floor"], feats["points_diff"].sum() / feats["possessions"].sum() * 100)


def test_whole_game_prediction_adds_the_intercept_for_stints_the_validation_drops(dev_tables):
    feats, games = dev_tables[0], dev_tables[1].copy()
    model = L.fit_arms(feats, None)
    games.loc[0, "poss_all"] += 5.0            # five possessions of a dropped stint
    pred = L.game_predictions(model, feats, games)
    icp = L.intercepts(model)
    for arm in L.ARMS:
        assert np.isclose(pred.loc[0, f"all_{arm}"] - pred.loc[0, f"valid_{arm}"], icp[arm] * 5.0 / 100.0)
        assert np.allclose(pred.loc[1:, f"all_{arm}"], pred.loc[1:, f"valid_{arm}"])


def test_score_games_pairs_every_arm_on_the_same_resampled_games():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 12, 400)
    pred = pd.DataFrame({"actual_all": y, "actual_valid": y})
    for prefix in ("all", "valid"):
        pred[f"{prefix}_floor"] = 0.0
        pred[f"{prefix}_plus_minus"] = 0.0
        pred[f"{prefix}_rapm"] = y * 0.5
        pred[f"{prefix}_rapm_replacement"] = y * 0.5
    out = L.score_games(pred, n=300)
    w = out["whole_game"]
    assert w["paired"]["replacement_minus_rapm"]["ci95"] == [0.0, 0.0]
    assert w["paired"]["plus_minus_minus_floor"]["ci95"] == [0.0, 0.0]
    assert w["paired"]["rapm_minus_plus_minus"]["ci95"][1] < 0
    assert np.isclose(w["r2_vs_floor"]["rapm"]["point"], 0.75)
    assert out["verdict"]["passes"]


def test_the_scorer_runs_through_score_once_on_a_pinned_root(dev_tables, tmp_path):
    """A rehearsal of e1-final's scorer: the synthetic 2025-26 season in a temporary pinned root, a temporary test set
    and ledger. A fault here would otherwise spend the one score."""
    feats = dev_tables[0]
    model = L.fit_arms(feats, R.PBP_TEST_SEASON)
    pbp, starters = synth_games(dates("2025-11-01", 12), seed=2)
    root = tmp_path / "root"
    paths = dict(zip(("playbyplay", "starters", "schedule"), R.PBP_FILES))
    for key, frame in (("playbyplay", pbp), ("starters", starters),
                       ("schedule", pd.DataFrame({"game_id": sorted(pbp["game_id"].unique())}))):
        p = root / paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(p, index=False)
    games = sorted(pbp["game_id"].unique())
    pins = {f: R.sha256_file(root / f) for f in R.PBP_FILES}
    testset = tmp_path / "set.json"
    testset.write_text(json.dumps({"sources": {R.PBP: {"games": games, "sha256": R.list_hash(games),
                                                       "files_sha256": pins, "pins_sha256": R.map_hash(pins)}}}))
    ledger = tmp_path / "ledger.json"
    out = R.score_once("lineups-E1-rehearsal", R.PBP, L.make_scorer(model), testset_path=testset, ledger_path=ledger,
                       data_root=root)
    assert out["games"] == 12 and out["test_games_listed"] == 12 and out["games_dropped_shifted_periods"] == 0
    assert set(out["whole_game"]["rmse"]) == set(L.ARMS)
    json.dumps(out)                                   # the summary must be JSON-able for the ledger
    assert json.loads(ledger.read_text())[0]["status"] == "scored"
    with pytest.raises(SystemExit):
        L.make_scorer(model)(games)                   # outside score_once the test data is refused


def test_the_tool_reads_development_seasons_only():
    with pytest.raises(SystemExit):
        L.read_dev_season(Path("/nonexistent"), R.PBP_SPENT_SEASONS[0])
    with pytest.raises(SystemExit):
        L.read_dev_season(Path("/nonexistent"), R.PBP_TEST_SEASON)

"""analytics.fixtures: the synthetic fixtures every analytics lane tests against (docs/contract/fixtures.md).

Checks the contract shapes (analytics.schema), the physics the localisers rely on, consistency between the
frames, the events and the derived tables, determinism, the held-out id rule, the DATA_ROOT layout and speed.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import fixtures, gaps, schema  # noqa: E402

CONTRACT_TABLES = ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")


@pytest.fixture(scope="module")
def game():
    return fixtures.synth_game(0)


@pytest.fixture(scope="module")
def game_small():
    return fixtures.synth_game(7, n_possessions=16)


@pytest.fixture(scope="module")
def per_frame(game):
    """Per-frame arrays of the game: clock, period, ball, and a (frame, player_id) -> (x, y) lookup."""
    fr = game["frames"]
    first = fr.drop_duplicates("frame").set_index("frame")
    pos = fr.set_index(["frame", "player_id"])[["x", "y"]]
    return first, pos


def _xy(pos, frame, pid):
    return pos.loc[(int(frame), int(pid))].to_numpy(dtype=float)


# ── shapes ────────────────────────────────────────────────────────────────────────────────────────
def test_contract_tables_validate(game):
    for t in CONTRACT_TABLES:
        assert schema.validate(game[t], t) == [], t
        assert list(game[t].columns) == schema.columns(t)
        expected = schema.pandas_dtypes(t)
        for c, d in expected.items():
            actual = game[t][c].dtype
            assert (isinstance(actual, pd.StringDtype) if isinstance(d, pd.StringDtype) else actual == d), (t, c, actual)
        assert len(game[t]) > 0


def test_frames_follow_section_4(game):
    fr = game["frames"]
    cols = ["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
            "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]
    assert list(fr.columns) == cols
    assert fr["frame"].value_counts().eq(10).all()
    frames = np.sort(fr["frame"].unique())
    assert frames[0] == 0 and frames[-1] == len(frames) - 1
    assert sorted(fr["period"].unique().tolist()) == [1, 2, 3, 4]
    first = fr.drop_duplicates("frame")
    for p, part in first.groupby("period"):
        d = np.diff(part["game_clock_s"].to_numpy(dtype=float))
        assert (d < 0).all(), "clock must count down within period %d" % p
        assert part["game_clock_s"].iloc[0] <= 720.0
    assert fr["home"].sum() == len(fr) // 2
    assert fr.loc[fr["home"], "team_id"].nunique() == 1 and fr["team_id"].nunique() == 2
    assert fr["team_id"].min() >= 1610612700
    assert fr["player_id"].nunique() == 14 and fr["player"].str.contains(" ").all()
    assert (fr["x"].between(0, 94) & fr["y"].between(0, 50)).all()
    assert fr["ball_z"].min() >= 0 and fr["ball_z"].max() < 20
    assert fr["shot_clock_s"].between(0, 24).all()
    assert fr["player"].iloc[0] == game["rosters"]["player"].iloc[0]
    assert fr["game"].iloc[0] == game["meta"]["sportvu_game"]


def test_events_follow_section_5(game):
    ev = game["events"]
    for c in ("game", "event", "period", "clock_s", "team", "desc", "points", "score", "kind", "frame", "frame_clock_diff_s",
              "source", "endpoint", "season", "fetched_at", "event_id", "event_type", "action_type", "player1_id",
              "player2_id", "player3_id", "team_id", "clock_str"):
        assert c in ev.columns, c
    assert (ev["event"].to_numpy() == np.arange(len(ev))).all()
    assert set(ev["kind"]) <= {"shot_made", "shot_missed", "rebound", "turnover", "substitution"}
    assert {"shot_made", "shot_missed", "rebound"} <= set(ev["kind"])
    assert set(ev["team"].dropna()) == {"home", "away"}
    assert (ev["frame"] >= 0).all() and (ev["source"] == "fixture").all()
    made = ev[ev["kind"] == "shot_made"]
    assert made["desc"].str.match(r"^[A-Z]\. \w+ makes [23]-pt (jump shot|layup) from \d+ ft").all()
    assert ev[ev["kind"] == "rebound"]["desc"].str.match(r"^(Offensive|Defensive) rebound by [A-Z]\. \w+$").all()
    assert ev[ev["kind"] == "substitution"]["desc"].str.contains("enters the game for").all()
    assert (made["points"].isin([2, 3])).all() and (ev.loc[ev["kind"] != "shot_made", "points"] == 0).all()
    # running score away-home ends at the final score
    away, home = 0, 0
    for r in ev.itertuples(index=False):
        if r.points:
            if r.team == "home":
                home += int(r.points)
            else:
                away += int(r.points)
        assert r.score == "%d-%d" % (away, home)
    assert game["meta"]["final_score"] == "%d-%d" % (away, home)
    # the event frame carries the event clock
    first = game["frames"].drop_duplicates("frame").set_index("frame")
    assert np.allclose(first.loc[ev["frame"], "game_clock_s"].to_numpy(), ev["clock_s"].to_numpy(), atol=1e-3)
    assert (first.loc[ev["frame"], "period"].to_numpy() == ev["period"].to_numpy()).all()


def test_shots_table(game):
    sh = game["shots"]
    for c in ("sportvu_game", "game_id", "event", "period", "clock_s", "made", "loc_x", "loc_y", "shot_distance_ft",
              "shot_type", "action_type", "zone_basic", "player_id", "player", "team_id", "team", "game_date",
              "home_team", "away_team", "split", "heldout", "season", "season_type", "source", "endpoint", "fetched_at"):
        assert c in sh.columns, c
    ev = game["events"].set_index("event")
    assert (ev.loc[sh["event"], "kind"].isin(["shot_made", "shot_missed"])).all()
    assert (sh["made"].to_numpy() == (ev.loc[sh["event"], "kind"] == "shot_made").to_numpy()).all()
    assert ((sh["shot_type"] == "3PT") == (sh["shot_distance_ft"] >= 23)).all()
    dist = np.hypot(sh["loc_x"] / 10.0, sh["loc_y"] / 10.0)
    assert np.allclose(dist, sh["shot_distance_ft"], atol=0.6)
    assert (sh["season"] == "2015-16").all() and not sh["heldout"].any()
    assert len(sh) == len(game["shot_frames"])


# ── physics ───────────────────────────────────────────────────────────────────────────────────────
def test_shot_physics(game, per_frame):
    first, pos = per_frame
    sf = game["shot_frames"]
    assert (sf["release_frame"] >= 0).all() and (sf["confidence_flag"] == "high").all()
    for r in sf.itertuples(index=False):
        rel, apex, out = int(r.release_frame), int(r.apex_frame), int(r.outcome_frame)
        assert rel < apex < out
        ball = first.loc[rel, ["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)
        shooter = _xy(pos, rel, r.shooter_id)
        assert np.hypot(*(ball[:2] - shooter)) <= 2.5 and ball[2] < 8.0
        # the ball rises for at least 0.2 s after the release and clears 8 ft at the apex
        z = first.loc[rel:rel + 5, "ball_z"].to_numpy(dtype=float)
        assert (np.diff(z) >= 0).all() and z[-1] > z[0]
        assert first.loc[apex, "ball_z"] > 8.0
        assert first.loc[apex, "ball_z"] == first.loc[rel + 1:out, "ball_z"].max()
        bo = first.loc[out, ["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)
        assert bo[2] < 10.0 and min(np.hypot(bo[0] - bx, bo[1] - by) for bx, by in fixtures.BASKETS) <= 6.0
        assert first.loc[apex + 1:out - 1, "ball_z"].min() >= 10.0 if out - apex > 1 else True
        # the shooter is the nearest player to the ball at release
        d = np.hypot(pos.loc[rel]["x"] - ball[0], pos.loc[rel]["y"] - ball[1])
        assert int(d.idxmin()) == int(r.shooter_id)
        assert first.loc[out, "period"] == r.period and abs(float(first.loc[out, "game_clock_s"]) - float(r.clock_s)) <= 0.05


def test_rebound_physics(game, per_frame):
    first, pos = per_frame
    rb = game["rebound_frames"]
    sf = game["shot_frames"].set_index("event")
    ev = game["events"].set_index("event")
    assert (rb["rebound_frame"] >= 0).all() and rb["identity_match"].all()
    assert set(rb["event"]) == set(sf.index[~sf["made"]])
    for r in rb.itertuples(index=False):
        assert int(r.rim_frame) == int(sf.loc[r.event, "outcome_frame"])
        reb = int(r.rebound_frame)
        assert reb > int(r.rim_frame)
        ball = first.loc[reb, ["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)
        assert ball[2] < 7.0
        assert np.hypot(*(ball[:2] - _xy(pos, reb, r.rebounder_id))) <= 2.0
        # no earlier frame after the rim has a player holding the ball
        for f in range(int(r.rim_frame) + 1, reb):
            if first.loc[f, "ball_z"] < 7.0:
                d = np.hypot(pos.loc[f]["x"] - first.loc[f, "ball_x"], pos.loc[f]["y"] - first.loc[f, "ball_y"])
                assert d.min() > 2.0
        assert ev.loc[r.rebound_event, "kind"] == "rebound" and ev.loc[r.rebound_event, "player1_id"] == r.rebounder_id
        assert r.pbp_rebounder_name in ev.loc[r.rebound_event, "desc"]


def test_ball_is_carried_and_passed(game, per_frame):
    first, pos = per_frame
    fr = game["frames"]
    # most frames: the ball is within 2.5 ft of some player and under 6 ft (carried)
    d = np.hypot(fr["x"] - fr["ball_x"], fr["y"] - fr["ball_y"])
    near = d.groupby(fr["frame"]).min()
    carried = (near <= 2.5) & (first["ball_z"] < 6.0)
    assert carried.mean() > 0.5
    # the ball changes hands by travelling, never by jumping more than 3 ft between frames
    step = np.hypot(np.diff(first["ball_x"]), np.diff(first["ball_y"]))
    keep = (np.diff(first["period"].to_numpy()) == 0) & (np.abs(np.diff(first["game_clock_s"].to_numpy())) < 2)
    assert np.nanmax(step[keep]) < 3.0                # one frame of a pass or a flight, never a teleport
    # players never move faster than 32 ft/s between consecutive frames of one period (the gap excluded)
    piv = fr.pivot(index="frame", columns="player_id", values=["x", "y"])
    sp = np.hypot(piv["x"].diff().to_numpy(), piv["y"].diff().to_numpy())[1:]
    assert np.nanmax(sp[keep]) <= 32.0 * fixtures.DT + 1e-6


def test_gravity_signal(game):
    fr = game["frames"]
    sub = fr[fr["frame"] % 25 == 0]
    piv = sub.pivot(index="frame", columns="player_id", values=["x", "y"])
    home = fr.drop_duplicates("player_id").set_index("player_id")["home"]
    X, Y, pids = piv["x"].to_numpy(), piv["y"].to_numpy(), piv["x"].columns.to_numpy()
    mean_near = {}
    for i, pid in enumerate(pids):
        opp = [j for j, q in enumerate(pids) if home[q] != home[pid]]
        d = np.hypot(X[:, [i]] - X[:, opp], Y[:, [i]] - Y[:, opp])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_near[int(pid)] = float(np.nanmean(np.nanmin(d, axis=1)))
    stars = game["truth"]["star_player_ids"]
    others = [v for k, v in mean_near.items() if k not in stars]
    for s in stars:
        assert mean_near[s] < np.mean(others) - 1.5


# ── consistency ───────────────────────────────────────────────────────────────────────────────────
def test_possessions_consistent_with_events(game):
    ps, ev = game["possessions"], game["events"]
    assert len(ps) == game["meta"]["n_possessions"]
    assert (ps["possession"].to_numpy() == np.arange(len(ps))).all()
    assert int(ps["points"].sum()) == int(ev["points"].sum())
    assert set(ps["end_kind"]) <= {"shot_made", "shot_missed", "turnover"}
    evi = ev.set_index("event")
    for r in ps.itertuples(index=False):
        end = evi.loc[r.end_event]
        assert end["period"] == r.period and int(end["frame"]) == int(r.end_frame)
        assert r.start_event <= r.end_event
        expect = {"shot_made": "shot_made", "rebound": "shot_missed", "turnover": "turnover"}[end["kind"]]
        assert r.end_kind == expect
        span = evi.loc[r.start_event:r.end_event]
        side = "home" if r.offense_team_id == game["meta"]["home_team_id"] else "away"
        assert int(span.loc[span["team"] == side, "points"].sum()) == r.points
        assert r.n_frames == r.end_frame - r.start_frame + 1 and r.start_clock_s >= r.end_clock_s
    # possessions chain: each starts at the frame of the previous ending event of its period
    for p, part in ps.groupby("period"):
        s, e = part["start_frame"].to_numpy(), part["end_frame"].to_numpy()
        assert (s[1:] == e[:-1]).all()
    # offence alternates within a period (each period opens with a fixed side, like a possession arrow)
    for _p, part in ps.groupby("period"):
        off = part["offense_team_id"].to_numpy()
        assert (off[1:] != off[:-1]).all()


def test_states_1hz_consistent(game, per_frame):
    first, pos = per_frame
    st, ps = game["states_1hz"], game["possessions"].set_index("possession")
    assert set(st["possession"]) == set(ps.index)
    for i in range(1, 5):
        assert (st["off%d_id" % i] < st["off%d_id" % (i + 1)]).all() and (st["def%d_id" % i] < st["def%d_id" % (i + 1)]).all()
    off_ids = st[["off%d_id" % i for i in range(1, 6)]].to_numpy()
    assert (st["handler_id"].to_numpy()[:, None] == off_ids).any(axis=1).all()
    for pno, part in st.groupby("possession"):
        p = ps.loc[pno]
        assert (part["frame"].between(p.start_frame, p.end_frame)).all()
        assert (part["points_to_end"] == p.points).all() and (part["possession_end_kind"] == p.end_kind).all()
        assert (part["offense_team_id"] == p.offense_team_id).all()
        el = part["elapsed_in_possession_s"].to_numpy()
        assert el[0] == 0 and np.abs(el - np.round(el)).max() <= 0.5 and (np.diff(el) > 0).all()
        if not p.gap:
            assert np.allclose(np.round(el), np.arange(len(el)))
        else:
            assert len(el) < np.round(el[-1]) + 1          # the 3 s gap holds no states
    row = st.iloc[5]
    assert np.allclose([row.ball_x, row.ball_y], first.loc[row.frame, ["ball_x", "ball_y"]].to_numpy(dtype=float))
    assert np.allclose(_xy(pos, row.frame, row.off1_id), [row.off1_x, row.off1_y])


def test_rosters(game):
    ro = game["rosters"]
    assert len(ro) == 14 and ro["player_id"].is_unique and ro["home"].sum() == 7
    assert set(ro["player_id"]) == set(game["frames"]["player_id"])
    assert (ro["match_method"] == "exact").all()
    assert (ro["pbp_name"].str.match(r"^[A-Z]\. \w+$")).all()
    assert set(ro["team"]) == {game["meta"]["home"], game["meta"]["away"]}


def test_gap_in_exactly_one_possession(game):
    ps = game["possessions"]
    assert int(ps["gap"].sum()) == 1
    assert int(ps.loc[ps["gap"], "possession"].iloc[0]) == fixtures.GAP_POSSESSION(game["meta"]["n_possessions"])
    first = game["frames"].drop_duplicates("frame")
    jumps = []
    for _p, part in first.groupby("period"):
        d = -np.diff(part["game_clock_s"].to_numpy(dtype=float))
        jumps += d[d > 2.0].tolist()
    assert len(jumps) == 1 and abs(jumps[0] - 3.0) < 0.02
    g = game["manifest_entry"]["gaps_over_2s"]
    assert g["count"] == 1 and abs(g["seconds"] - 3.0) < 0.1 and sorted(g["by_quarter"]) == ["1", "2", "3", "4"]
    spans = [s for q in g["by_quarter"].values() for s in q]
    assert len(spans) == 1 and spans[0][0] > spans[0][1]
    p = ps[ps["gap"]].iloc[0]
    assert p.start_clock_s > spans[0][0] and p.end_clock_s < spans[0][1]


# ── determinism, ids, held-out ────────────────────────────────────────────────────────────────────
def test_deterministic_by_seed(game_small):
    again = fixtures.synth_game(7, n_possessions=16)
    for k in ("frames", "events", "shots") + CONTRACT_TABLES:
        pd.testing.assert_frame_equal(game_small[k], again[k])
    other = fixtures.synth_game(8, n_possessions=16)
    assert not game_small["frames"]["x"].equals(other["frames"]["x"])
    assert game_small["meta"]["sportvu_game"] != other["meta"]["sportvu_game"]


def test_game_ids_and_heldout(game_small):
    gid = game_small["meta"]["sportvu_game"]
    mm, dd, yyyy, away, at, home = gid.split(".")
    assert at == "at" and away in fixtures.TEAM_CODES and home in fixtures.TEAM_CODES and away != home
    assert gid != fixtures.HELDOUT_GAME and not game_small["meta"]["heldout"]
    assert game_small["meta"]["season"] == "2015-16" and game_small["manifest_entry"]["split"] == "train"
    held = fixtures.synth_game(7, n_possessions=8, heldout=True)
    assert held["meta"]["sportvu_game"] == fixtures.HELDOUT_GAME and held["meta"]["heldout"]
    assert held["shots"]["heldout"].all() and (held["shots"]["split"] == "heldout_game").all()
    assert set(held["frames"]["team"]) == {"PHX", "GSW"} and held["manifest_entry"]["heldout"]
    custom = fixtures.synth_game(7, n_possessions=8, sportvu_game="01.02.2016.KNX.at.TUL")
    assert custom["meta"]["sportvu_game"] == "01.02.2016.KNX.at.TUL" and not custom["meta"]["heldout"]
    arena = fixtures.synth_game(7, n_possessions=8, heldout=True, sportvu_game="01.02.2016.KNX.at.CLE")
    assert arena["meta"]["split"] == "heldout_arena" and arena["manifest_entry"]["heldout"]
    with pytest.raises(ValueError):
        fixtures.synth_game(7, n_possessions=8, heldout=True, sportvu_game="01.02.2016.KNX.at.TUL")
    with pytest.raises(ValueError):
        fixtures.synth_game(7, n_possessions=8, heldout=False, sportvu_game="01.02.2016.KNX.at.CLE")
    with pytest.raises(ValueError):
        fixtures.synth_game(0, n_possessions=2)


def test_split_matches_real_routing():
    """Every held-out fixture game is routed held out by sportvu.colab_contract.split_of with sportvu/splits.json."""
    from sportvu import colab_contract as cc, splits
    sp = splits.load()
    ids = [fixtures.HELDOUT_GAME, "01.02.2016.KNX.at.CLE", "01.02.2016.KNX.at.TUL"]
    for gid in ids:
        real = cc.split_of({"game": gid, "home": gid.rsplit(".", 1)[-1]}, sp)
        assert cc.is_heldout(real) == (fixtures.split_for(gid) != "train"), gid
        if real is not None:
            assert real == fixtures.split_for(gid)


def test_season_heldout_rule():
    plain = fixtures.synth_season(3, n_games=3, heldout_every=0, n_possessions=4)
    ids = [g["meta"]["sportvu_game"] for g in plain]
    assert len(set(ids)) == 3 and fixtures.HELDOUT_GAME not in ids and not any(g["meta"]["heldout"] for g in plain)
    real = set(["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
                "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
                "UTA", "WAS", "BRK", "CHO", "PHO"])
    assert not (set(fixtures.TEAM_CODES) & real)
    for gid in ids:
        assert gid.split(".")[2] in ("2015", "2016") and _date_of(gid) >= "2015-10-27" and _date_of(gid) <= "2016-04-13"
    mixed = fixtures.synth_season(3, n_games=4, heldout_every=2, n_possessions=4)
    flags = [g["meta"]["heldout"] for g in mixed]
    assert flags == [False, True, False, True]
    assert mixed[1]["meta"]["sportvu_game"] == fixtures.HELDOUT_GAME
    assert mixed[3]["meta"]["sportvu_game"] != fixtures.HELDOUT_GAME        # ids never repeat
    assert mixed[3]["meta"]["sportvu_game"].endswith(".at.CLE") and mixed[3]["meta"]["split"] == "heldout_arena"
    assert len({g["meta"]["sportvu_game"] for g in mixed}) == 4
    pd.testing.assert_frame_equal(mixed[0]["frames"], plain[0]["frames"])  # per-game seeds do not depend on the rule


def _date_of(gid):
    mm, dd, yyyy = gid.split(".")[:3]
    return "%s-%s-%s" % (yyyy, mm, dd)


# ── the other generators ──────────────────────────────────────────────────────────────────────────
def test_aggregates_and_positions():
    agg, pos = fixtures.synth_aggregates(1, n_players=30, seasons=("2015-16",))
    assert set(agg["endpoint"]) == {"leaguedashptstats", "leaguedashplayerptshot", "synergyplaytypes", "leaguehustlestatsplayer"}
    for c in ("player_id", "player_name", "team_id", "season", "season_type", "endpoint", "split_name", "split_value",
              "pt_measure_type", "play_type", "metric", "value", "source", "fetched_at"):
        assert c in agg.columns, c
    assert agg["player_id"].nunique() == 30 and agg["value"].dtype == "float64"
    assert set(pos["position"]) == {"G", "F", "C"} and len(pos) == 30
    arche = agg.attrs["archetype"]
    assert all(fixtures.POSITION_OF[arche[int(p)]] == q for p, q in zip(pos["player_id"], pos["position"]))
    drives = agg[(agg["metric"] == "drives")].set_index("player_id")["value"]
    g_mean = drives[[p for p in drives.index if arche[int(p)] == "guard"]].mean()
    c_mean = drives[[p for p in drives.index if arche[int(p)] == "big"]].mean()
    assert g_mean > 2 * c_mean
    pt = agg[agg["endpoint"] == "leaguedashplayerptshot"]
    assert set(pt["split_value"]) == set(fixtures.DEF_RANGES) and (pt["split_name"] == "closest_defender").all()
    ppp = agg[(agg["endpoint"] == "synergyplaytypes") & (agg["metric"] == "ppp")]
    assert set(ppp["play_type"]) == set(fixtures.PLAY_TYPES) and ppp["value"].between(0.3, 1.8).all()
    agg2, _ = fixtures.synth_aggregates(1, n_players=30, seasons=("2015-16",))
    pd.testing.assert_frame_equal(agg, agg2)


def test_stints_playbyplay_games():
    st = fixtures.synth_stints(2, n_games=4)
    stints, pbp, games = st["stints"], st["playbyplay"], st["games"]
    assert len(games) == 4 and games["game_id"].is_unique and games["game_id"].str.match(r"^002\d{7}$").all()
    for c in ("home_1", "away_5", "home_ids", "lineup_ok", "points_home", "score_away", "split", "heldout",
              "season_type", "game_date", "source", "endpoint", "fetched_at"):
        assert c in stints.columns, c
    assert stints["lineup_ok"].all() and (stints["n_home"] == 5).all()
    hl = stints[["home_%d" % i for i in range(1, 6)]].to_numpy()
    assert (np.diff(hl, axis=1) > 0).all()
    for gid, part in stints.groupby("game_id"):
        g = games.set_index("game_id").loc[gid]
        assert int(part["points_home"].sum()) == int(g["home_score"]) == int(part["score_home"].iloc[-1])
        assert int(part["points_away"].sum()) == int(g["away_score"]) == int(part["score_away"].iloc[-1])
        assert (part["stint"].to_numpy() == np.arange(len(part))).all()
        assert np.allclose(part["end_s"].to_numpy()[:-1], part["start_s"].to_numpy()[1:]) and part["end_s"].iloc[-1] == 2880
        ev = pbp[pbp["game_id"] == gid]
        assert int(ev.loc[ev["team"] == "home", "points"].sum()) == int(g["home_score"])
        assert ev["score"].iloc[-1] == "%d-%d" % (g["away_score"], g["home_score"])
        assert (ev["frame"] == -1).all() and ev["kind"].iloc[0] == "period_start" and ev["kind"].iloc[-1] == "period_end"
        subs = ev[ev["kind"] == "substitution"]
        assert len(subs) >= 2 and subs["player1_id"].notna().all() and subs["player2_id"].notna().all()
        assert (subs["desc"].str.contains("enters the game for")).all()
    assert (games["home_win"] == (games["home_score"] > games["away_score"])).all()
    assert "spread_home" in games.columns and games["spread_home"].dtype == "Float32"
    pbp_cols = {"game", "event", "period", "clock_s", "team", "desc", "points", "score", "kind", "frame", "game_id",
                "split", "heldout", "season_type", "game_date", "event_id", "event_type", "player1_id", "player2_id",
                "player3_id", "team_id", "clock_str", "source", "endpoint", "season", "fetched_at"}
    assert pbp_cols <= set(pbp.columns)


def test_odds():
    st = fixtures.synth_stints(2, n_games=4)
    odds = fixtures.synth_odds(2, st["games"])
    assert list(odds.columns) == ["game_key", "commence_time", "home_team", "away_team", "book", "market", "side",
                                  "point", "price", "snapshot_at", "is_close", "source", "endpoint", "fetched_at", "season"]
    assert set(odds["market"]) == {"h2h", "spreads", "totals"} and set(odds["book"]) == set(fixtures.BOOKS)
    assert odds["game_key"].nunique() == 4 and odds["is_close"].sum() == len(odds) // 2
    assert odds.loc[odds["market"] == "h2h", "point"].isna().all() and odds.loc[odds["market"] != "h2h", "point"].notna().all()
    assert set(odds.loc[odds["market"] == "totals", "side"]) == {"Over", "Under"}
    sp = odds[(odds["market"] == "spreads") & odds["is_close"]]
    for g in st["games"].itertuples(index=False):
        if pd.isna(g.spread_home):
            continue
        key = sp[sp["game_key"].str.startswith(g.game_date)]
        home_line = key[key["side"] == key["home_team"]]["point"].median()
        assert abs(float(home_line) - float(g.spread_home)) <= 1.5
    assert odds["price"].dtype == "int32" and (odds["price"].abs() >= 100).all()
    assert odds["game_key"].str.match(r"^\d{4}-\d{2}-\d{2}_\w+_\w+$").all()


def test_labels_screens_present_and_separable(game, per_frame):
    first, pos = per_frame
    lab = fixtures.synth_labels(0, game)
    acts, scr = lab["actions"], lab["screens"]
    assert list(acts.columns) == fixtures.ACTION_COLUMNS and list(scr.columns) == fixtures.SCREEN_COLUMNS
    assert acts["action_id"].is_unique and scr["screen_id"].is_unique
    assert set(scr["action_id"]) <= set(acts.loc[acts["action_type"] == "ball_screen", "action_id"])
    assert acts["labeller_pass"].dtype == "int16" and acts["labelled_at"].str.endswith("Z").all()
    assert set(acts["action_type"]) <= set(fixtures.ACTION_TYPES) and len(acts) == len(game["possessions"])
    assert (acts["label_source"] == "synthetic").all() and (acts["labeller_pass"] == 1).all()
    assert (acts["start_frame"] < acts["end_frame"]).all()
    ps = game["possessions"].set_index("possession")
    for a in acts.itertuples(index=False):
        p = ps.loc[a.possession]
        assert p.start_frame <= a.start_frame and a.end_frame <= p.end_frame
    assert set(scr["coverage"]) <= set(fixtures.COVERAGES) and (scr["confidence"] == "sure").all()
    bs = acts[acts["action_type"] == "ball_screen"].set_index("possession")
    assert len(scr) == len(bs) >= 5 and set(scr["possession"]) == set(bs.index)
    assert bs["screener_id"].notna().all()
    feats = []
    for s in scr.itertuples(index=False):
        a = bs.loc[s.possession]
        H, S, D1, D2 = int(a.ball_handler_id), int(a.screener_id), int(s.on_ball_defender_id), int(s.screener_defender_id)
        # the screen is set: the screener within 4 ft of the on-ball defender for 25 consecutive frames
        frames = np.arange(int(a.start_frame), int(a.end_frame) + 1)
        near = np.array([np.hypot(*(_xy(pos, f, S) - _xy(pos, f, D1))) <= 4.0 for f in frames])
        run = max(len(x) for x in "".join("1" if v else "0" for v in near).split("0"))
        assert run >= 25, (s.possession, s.coverage, run)
        e = int(a.end_frame) - 20                           # end of the drive, before the recovery
        period = int(first.loc[e, "period"])
        off_home = bool(game["rosters"].set_index("player_id").loc[H, "home"])
        basket = np.array(fixtures.BASKETS[1] if off_home == (period <= 2) else fixtures.BASKETS[0])
        h, s_, d1, d2 = _xy(pos, e, H), _xy(pos, e, S), _xy(pos, e, D1), _xy(pos, e, D2)
        hm, d2m = _xy(pos, e - 15, H), _xy(pos, e - 15, D2)
        lat = (h - _xy(pos, int(a.end_frame) - 50, H))[1] * (1 if off_home == (period <= 2) else 1)
        feats.append({"coverage": s.coverage, "d2h": np.linalg.norm(d2 - h), "d2r": np.linalg.norm(d2 - basket),
                      "d1h": np.linalg.norm(d1 - h), "d1s": np.linalg.norm(d1 - s_), "d2hm": np.linalg.norm(d2m - hm),
                      "d2s": np.linalg.norm(d2 - s_), "lat": lat})
    df = pd.DataFrame(feats)

    def rule(r):
        if r.lat > 0:
            return "ice"
        if r.d1s < r.d1h:
            return "switch"
        if r.d2h < 4 and r.d1h < 7:
            return "blitz_trap"
        if r.d2hm < 4:
            return "hedge_show"
        if r.d2r < 10 and r.d2s > 3.5:
            return "drop"
        return "under" if r.d1h < 4 else "over"

    pred = df.apply(rule, axis=1)
    acc = float((pred == df["coverage"]).mean())
    majority = float(df["coverage"].value_counts(normalize=True).iloc[0])
    assert acc > majority and acc >= 0.6, (acc, majority, df.assign(pred=pred).to_string())


# ── the DATA_ROOT layout ──────────────────────────────────────────────────────────────────────────
def test_write_fixture_root(tmp_path):
    man = fixtures.write_fixture_root(tmp_path, seed=0, n_games=3, heldout_every=3, n_possessions=8,
                                      n_stint_games=4, n_agg_players=12)
    sv = tmp_path / "sportvu_contract"
    assert json.loads((sv / "manifest.json").read_text()) == man
    assert man["paths"] == fixtures.fixture_paths(tmp_path) and Path(man["paths"]["manifest"]) == sv / "manifest.json"
    for gid, e in man["games"].items():
        assert e["heldout"] == (fixtures.split_for(gid) != "train") and e["split"] == fixtures.split_for(gid)
    assert man["counts"] == {"games": 3, "ok": 3, "failed": 0, "source_broken": 0, "pbp_joined": 3, "heldout": 1}
    assert man["heldout"] == [fixtures.HELDOUT_GAME] and man["contract_version"] == 1
    for gid, e in man["games"].items():
        assert e["status"] == "ok" and e["gaps_over_2s"]["count"] == 1 and e["pbp_joined"]
        assert (sv / e["parquet"]).exists() and (sv / e["pbp_parquet"]).exists()
        season = "2015-16"
        for table in CONTRACT_TABLES:
            p = schema.table_path(tmp_path / "analytics", table, season, gid, e["heldout"])
            assert p.exists(), p
            assert schema.validate(pd.read_parquet(p), table) == []
        assert (tmp_path / "labels" / "actions" / (gid + ".jsonl")).exists()
        lines = (tmp_path / "labels" / "screens" / (gid + ".jsonl")).read_text().splitlines()
        assert all(json.loads(l)["sportvu_game"] == gid for l in lines)
    assert (sv / "heldout" / (fixtures.HELDOUT_GAME + ".parquet")).exists()
    assert man["games"][fixtures.HELDOUT_GAME]["parquet"].startswith("heldout/")
    fr = pd.read_parquet(sv / man["games"][fixtures.HELDOUT_GAME]["parquet"])
    assert fr["frame"].nunique() == man["games"][fixtures.HELDOUT_GAME]["moments"]
    train_shots = pd.read_parquet(tmp_path / "collect" / "bref_shots" / "shotchart" / "2015-16.parquet")
    held_shots = pd.read_parquet(tmp_path / "collect" / "bref_shots" / "heldout" / "shotchart" / "2015-16.parquet")
    assert not train_shots["heldout"].any() and held_shots["heldout"].all()
    assert set(held_shots["sportvu_game"]) == {fixtures.HELDOUT_GAME}
    for rel in ("collect/tracking/aggregates/2015-16.parquet", "collect/tracking/aggregates/2016-17.parquet",
                "collect/tracking/positions/2015-16.parquet", "collect/stints/stints/2020-21.parquet",
                "collect/stints/playbyplay/2020-21.parquet", "collect/stints/games/2020-21.parquet",
                "collect/extras/odds/2020-21.parquet"):
        assert (tmp_path / rel).exists(), rel
    # analytics.gaps reads the manifest
    g = gaps.game_gaps(man)
    assert len(g) == 3 and np.allclose(g["gap_s"], 3.0, atol=0.02) and set(g["sportvu_game"]) == set(man["games"])
    assert gaps.summary(man)["games_with_gap_over_2s"] == 3
    flagged = gaps.flag_possessions(pd.read_parquet(schema.table_path(tmp_path / "analytics", "possessions", "2015-16",
                                                                      fixtures.HELDOUT_GAME, True)), g)
    assert int(flagged.sum()) == 1


def test_cli(tmp_path):
    assert fixtures.main(["--out", str(tmp_path / "root"), "--games", "1", "--heldout-every", "0", "--possessions", "4",
                          "--stint-games", "2", "--agg-players", "6"]) == 0
    man = json.loads((tmp_path / "root" / "sportvu_contract" / "manifest.json").read_text())
    assert man["counts"]["games"] == 1 and man["heldout"] == []


# ── speed ─────────────────────────────────────────────────────────────────────────────────────────
def test_speed_40_possessions_under_2s():
    t = time.time()
    g = fixtures.synth_game(12, n_possessions=40)
    assert time.time() - t < 2.0
    assert len(g["possessions"]) == 40

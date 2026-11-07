"""tests/_fx_load.py: the test-only local generator of the load and fatigue lane (analytics/load).

Production code reads the shared season of analytics.fixtures through analytics.load.validate.load_fixtures and
never imports this file. The tests also need a season with a planted signal they can assert on exactly, which the
shared one does not carry: this local generator, under its own name (local_season_inputs), never shadowing the
shared generator names.

The local frames are section 4 shaped at 25 Hz: two periods, possessions of 24 s, five players a side,
a lineup that rotates one player every possession (so every player has on-court stints of several minutes), one
absent player per team in about half the games (so the absence target has both classes). Every player
has a base speed and a per-stint speed decline that is larger on a back-to-back, so the within-game
decline target has signal; sprint bursts give the high-intensity share something to count. Positions
carry a small jitter (sd 0.02 ft). Deterministic given the seed.

local_season_inputs(...) returns the inputs dict of analytics.load.validate (packaged by
validate.inputs_from_games, the function the shared loader uses too). Fixture numbers are never results.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from analytics.load import validate as V

FPS = 25
POSSESSION_S = 24.0
PERIOD_S = 720.0
TEAMS = {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
PLAYERS_PER_TEAM = 10
SUB_EVERY = 1                      # possessions between lineup rotations (one player in, one out)
FRAME_COLUMNS = ["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
                 "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]
ROSTER_COLUMNS = ["sportvu_game", "player_id", "player", "team_id", "team", "home", "jersey", "bref_slug", "bref_name",
                  "pbp_name", "match_method"]


# ── player truth ───────────────────────────────────────────────────────────────────────────────────
def _player_truth(player_id: int) -> dict:
    """Base speed (ft per s), per-stint decline rate and sprint propensity of a player, fixed by the id."""
    r = np.random.default_rng(1000 + int(player_id))
    return {"base": float(r.uniform(4.0, 7.5)), "rate": float(r.uniform(0.03, 0.09)), "sprint": float(r.uniform(0.05, 0.2))}


def team_players(team: str) -> list:
    """The player ids of a local team."""
    return [TEAMS[team] * 100 + k for k in range(1, PLAYERS_PER_TEAM + 1)]


# ── one game ───────────────────────────────────────────────────────────────────────────────────────
def _fold(v, lo, hi):
    """Reflect a trajectory into [lo, hi] (a bounce off the court edge)."""
    w = hi - lo
    r = np.mod(v - lo, 2 * w)
    return lo + np.where(r > w, 2 * w - r, r)


def _local_synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None, back_to_back=False, absent=()):
    """Section 4 frames plus the other synth_game tables (empty but contract shaped) for one game."""
    from analytics import schema
    rng = np.random.default_rng(seed)
    game = sportvu_game or "11.%02d.2015.AAA.at.BBB" % (1 + seed % 28)
    mm, dd, yyyy, away, _at, home = game.split(".")
    date = "%s-%s-%s" % (yyyy, mm, dd)
    absent = set(int(a) for a in absent)
    sides = [(home, True), (away, False)]
    avail = {t: [p for p in team_players(t) if p not in absent] for t, _h in sides}
    n_frames_poss = int(POSSESSION_S * FPS)
    half = max(1, n_possessions // 2)
    state = {}                     # player_id -> dict(x, y, heading, stint_index, on)
    for t, _h in sides:
        for p in avail[t]:
            state[p] = {"x": rng.uniform(10, 84), "y": rng.uniform(5, 45), "heading": rng.uniform(0, 2 * np.pi),
                        "stint": -1, "on": False, "speed": 0.0}
    cols = {c: [] for c in ("period", "frame", "ts_ms", "game_clock_s", "team_id", "team", "home", "player_id", "x", "y")}
    frame = 0
    ts = 1_446_000_000_000 + int(seed) * 7_200_000
    b2b_mult = 1.5 if back_to_back else 1.0
    for p_ix in range(n_possessions):
        period = 1 if p_ix < half else 2
        window = p_ix // SUB_EVERY
        lineup = {}
        for t, h in sides:
            a = avail[t]
            lineup[t] = [a[(window + j) % len(a)] for j in range(5)]
        on_now = set(sum(lineup.values(), []))
        for pid, s in state.items():
            if pid in on_now and not s["on"]:
                s["on"], s["stint"] = True, s["stint"] + 1
                tr = _player_truth(pid)
                s["speed"] = max(1.0, tr["base"] * (1.0 - tr["rate"] * b2b_mult * s["stint"]))
            elif pid not in on_now:
                s["on"] = False
        within = (p_ix - (0 if period == 1 else half)) * POSSESSION_S
        clocks = PERIOD_S - within - np.arange(n_frames_poss) / FPS
        frames_ix = frame + np.arange(n_frames_poss)
        ts_ix = ts + np.arange(n_frames_poss) * (1000 // FPS)
        for t, h in sides:
            for pid in lineup[t]:
                s = state[pid]
                tr = _player_truth(pid)
                v = np.full(n_frames_poss, s["speed"])
                if rng.uniform() < tr["sprint"]:                       # one sprint burst of 1.5 s in the possession
                    b0 = int(rng.integers(0, n_frames_poss - int(1.5 * FPS)))
                    v[b0:b0 + int(1.5 * FPS)] += 9.0
                heading = s["heading"] + np.cumsum(rng.normal(0, 0.15, n_frames_poss))
                x = s["x"] + np.cumsum(np.cos(heading) * v / FPS)
                y = s["y"] + np.cumsum(np.sin(heading) * v / FPS)
                x = _fold(x, 0.5, 93.5)
                y = _fold(y, 0.5, 49.5)
                s["x"], s["y"], s["heading"] = float(x[-1]), float(y[-1]), float(heading[-1])
                cols["period"].append(np.full(n_frames_poss, period, dtype=np.int8))
                cols["frame"].append(frames_ix)
                cols["ts_ms"].append(ts_ix)
                cols["game_clock_s"].append(clocks)
                cols["team_id"].append(np.full(n_frames_poss, TEAMS[t], dtype=np.int32))
                cols["team"].append(np.full(n_frames_poss, t, dtype=object))
                cols["home"].append(np.full(n_frames_poss, h, dtype=bool))
                cols["player_id"].append(np.full(n_frames_poss, pid, dtype=np.int32))
                cols["x"].append(x + rng.normal(0, 0.02, n_frames_poss))
                cols["y"].append(y + rng.normal(0, 0.02, n_frames_poss))
        frame += n_frames_poss
        ts += n_frames_poss * (1000 // FPS)
    frames = pd.DataFrame({c: np.concatenate(v) for c, v in cols.items()}).sort_values(["frame", "home", "player_id"],
                                                                                        ascending=[True, False, True])
    frames = frames.reset_index(drop=True)
    frames["game"] = game
    frames["gameid"] = "00215%05d" % (seed % 100000)
    frames["date"] = date
    frames["shot_clock_s"] = np.float32(np.nan)
    frames["n_players"] = np.int8(10)
    frames["ball_x"] = np.float32(np.nan)
    frames["ball_y"] = np.float32(np.nan)
    frames["ball_z"] = np.float32(np.nan)
    frames["player"] = frames["player_id"].map(lambda p: "Player %d" % p)
    frames["jersey"] = frames["player_id"].map(lambda p: str(p % 100))
    frames = frames[FRAME_COLUMNS]
    frames = frames.astype({"period": "int8", "frame": "int32", "ts_ms": "int64", "game_clock_s": "float32",
                            "team_id": "int32", "player_id": "int32", "x": "float32", "y": "float32"})
    ros = frames[["player_id", "player", "team_id", "team", "home", "jersey"]].drop_duplicates("player_id").copy()
    ros.insert(0, "sportvu_game", game)
    ros["bref_slug"] = None
    ros["bref_name"] = None
    ros["pbp_name"] = None
    ros["match_method"] = "none"
    ros = ros.sort_values(["home", "team", "player_id"], ascending=[False, True, True]).reset_index(drop=True)
    rosters = ros[ROSTER_COLUMNS]
    events = pd.DataFrame(columns=["game", "event", "period", "clock_s", "team", "desc", "points", "kind", "frame"])
    shots = pd.DataFrame(columns=["sportvu_game", "event", "period", "clock_s", "made", "loc_x", "loc_y", "heldout"])
    return {"sportvu_game": game, "date": date, "away": away, "home": home, "heldout": bool(heldout),
            "frames": frames, "events": events, "shots": shots, "rosters": rosters,
            "shot_frames": schema.empty("shot_frames"), "rebound_frames": schema.empty("rebound_frames"),
            "possessions": schema.empty("possessions"), "states_1hz": schema.empty("states_1hz")}


# ── a season ───────────────────────────────────────────────────────────────────────────────────────
PAIRS = [("AAA", "BBB"), ("CCC", "DDD"), ("CCC", "AAA"), ("BBB", "DDD"), ("DDD", "AAA"), ("BBB", "CCC")]


def _local_synth_season(seed, n_games=8, heldout_every=0, n_possessions=40):
    """n_games local games over four teams on a schedule with 1 to 3 days between games (team days of rest
    vary, some back-to-backs), one absent player per team in about half the games, every heldout_every-th
    game flagged held out."""
    rng = np.random.default_rng(seed)
    date = dt.date(2015, 11, 1)
    last = {}
    games = []
    for g in range(n_games):
        if g:
            date = date + dt.timedelta(days=int(rng.integers(1, 4)))
        away, home = PAIRS[g % len(PAIRS)]
        sid = "%02d.%02d.%d.%s.at.%s" % (date.month, date.day, date.year, away, home)
        b2b = any(last.get(t) == date - dt.timedelta(days=1) for t in (away, home))
        absent = []
        for t in (away, home):
            if rng.uniform() < 0.5:
                absent.append(int(rng.choice(team_players(t))))
            last[t] = date
        held = bool(heldout_every) and (g + 1) % heldout_every == 0
        games.append(_local_synth_game(seed * 100 + g, n_possessions=n_possessions, heldout=held, sportvu_game=sid,
                                       back_to_back=b2b, absent=absent))
    return games


def _local_synth_aggregates(seed, n_players=40, seasons=("2015-16",)):
    """Long aggregates rows (section 11) of leaguedashptstats SpeedDistance per player and season, from the
    same player truth as the frames: dist_miles per game, avg_speed in mph, min and gp."""
    rng = np.random.default_rng(seed)
    pool = sorted(sum((team_players(t) for t in TEAMS), []))[:n_players]
    rows = []
    for season in seasons:
        for pid in pool:
            tr = _player_truth(pid)
            team = [t for t in TEAMS if TEAMS[t] == pid // 100][0]
            minutes = float(rng.uniform(18, 36))
            speed_mph = tr["base"] * 3600 / 5280 * (1 - 0.1 * tr["rate"] * 10) + rng.normal(0, 0.05)
            dist_miles = speed_mph * minutes / 60
            for metric, value in (("dist_miles", dist_miles), ("avg_speed", speed_mph), ("min", minutes), ("gp", 40.0)):
                rows.append({"player_id": pid, "player_name": "Player %d" % pid, "team_id": TEAMS[team],
                             "team_abbreviation": team, "season": season, "season_type": "Regular Season",
                             "endpoint": "leaguedashptstats", "pt_measure_type": "SpeedDistance", "player_or_team": "Player",
                             "metric": metric, "value": float(value), "heldout": None, "source": "fixture",
                             "fetched_at": "2026-01-01T00:00:00Z"})
    return pd.DataFrame(rows)


# ── the inputs dict ────────────────────────────────────────────────────────────────────────────────
def local_season_inputs(seed=0, n_games=8, heldout_every=4, n_possessions=20) -> dict:
    """The inputs dict of analytics.load.validate from the local generator: games (frames, rosters, schedule ids),
    a section 6 manifest, aggregates (section 11) from the same player truth, injuries None."""
    games = _local_synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    n_players = len({int(p) for g in games for p in g["rosters"]["player_id"]})
    agg = _local_synth_aggregates(seed, max(n_players, 1), ["2015-16"])
    return V.inputs_from_games(games, agg, source="local generator of tests/_fx_load.py")


__all__ = ["local_season_inputs", "team_players"]

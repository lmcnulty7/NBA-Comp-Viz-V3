"""tests/_fx_embeddings.py: fixtures for the player style embeddings lane (analytics/embeddings).

Prefers the shared generators of ``analytics.fixtures`` (lane I1) when they import and return the tables this
lane needs; otherwise falls back to a minimal local generator with the same signatures and column names:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None)
        -> dict(frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of synth_game dicts
    synth_aggregates(seed, n_players=60, seasons=("2015-16", "2016-17")) -> (aggregates long frame, positions frame)
    write_fixture_root(root, seed, n_games, heldout_every) -> dict summary of what was written

Fallback league: six teams (GSW, BKN, SAC, CHA, PHX, CLE) of ten players each, player ids 1001 to 1060, every
player with a hidden archetype in {G, F, C} (4 G, 3 F, 3 C per team) that drives the tracking frames (speed,
court zones, ball handling), the shots (distance mix) and the NBA.com style aggregates (drives, touches, play
types, hustle). A held-out game is ``12.16.2015.PHX.at.GSW`` (the sportvu/splits.json held-out game id) or a
game at CLE (the held-out arena); a training game is never at CLE. Frames run at 5 Hz, not 25 Hz, to keep the
fixtures small; speeds are computed from ``ts_ms`` so the rate does not matter to the features.

Fixture numbers are fixture output, not a result.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

FPS = 5
DT_MS = 1000 // FPS
PERIOD_S = 720.0
COURT_X, COURT_Y = 94.0, 50.0
RIM_X, RIM_Y = 5.25, 25.0
SEASON = "2015-16"
HELDOUT_GAME_ID = "12.16.2015.PHX.at.GSW"
HELDOUT_ARENA = "CLE"
TEAMS = {"GSW": 1610612744, "BKN": 1610612751, "SAC": 1610612758, "CHA": 1610612766, "PHX": 1610612756,
         "CLE": 1610612739}
TRAIN_TEAMS = [t for t in TEAMS if t != HELDOUT_ARENA]
ARCHETYPE_PATTERN = ("G", "G", "G", "G", "F", "F", "F", "C", "C", "C")
PLAY_TYPES = ("isolation", "transition", "prballhandler", "prrollman", "postup", "spotup", "handoff", "cut",
              "offscreen", "offrebound", "misc")

# archetype parameters
_STEP_SD = {"G": 1.1, "F": 0.8, "C": 0.5}                  # ft per frame of AR(1) noise
_HANDLER_W = {"G": 0.7, "F": 0.25, "C": 0.05}
_SHOOTER_W = {"G": 0.45, "F": 0.35, "C": 0.2}
_SHOT_DIST = {"G": (14.0, 27.0), "F": (4.0, 20.0), "C": (0.0, 8.0)}
_OFF_X = {"G": (18.0, 30.0), "F": (10.0, 22.0), "C": (2.0, 10.0)}   # feet beyond the rim, rim frame
_OFF_Y = {"G": (5.0, 45.0), "F": (8.0, 42.0), "C": (17.0, 33.0)}
_AGG_MEAN = {   # (endpoint, metric) -> per archetype mean
    ("leaguedashptstats", "drives"): {"G": 8.0, "F": 4.0, "C": 1.5},
    ("leaguedashptstats", "touches"): {"G": 70.0, "F": 50.0, "C": 40.0},
    ("leaguedashptstats", "passes_made"): {"G": 50.0, "F": 30.0, "C": 20.0},
    ("leaguedashplayerptshot", "catch_shoot_fga"): {"G": 4.0, "F": 3.0, "C": 0.5},
    ("leaguedashplayerptshot", "pull_up_fga"): {"G": 5.0, "F": 2.0, "C": 0.3},
    ("leaguehustlestatsplayer", "deflections"): {"G": 2.5, "F": 2.0, "C": 1.2},
    ("leaguehustlestatsplayer", "contested_shots"): {"G": 5.0, "F": 8.0, "C": 12.0},
    ("leaguehustlestatsplayer", "screen_assists"): {"G": 0.5, "F": 2.0, "C": 5.0},
}
_PLAY_MEAN = {  # play-type possession shares per archetype
    "G": (0.12, 0.18, 0.35, 0.02, 0.01, 0.15, 0.08, 0.03, 0.04, 0.01, 0.01),
    "F": (0.08, 0.15, 0.10, 0.08, 0.10, 0.22, 0.05, 0.10, 0.06, 0.04, 0.02),
    "C": (0.02, 0.08, 0.01, 0.30, 0.25, 0.08, 0.02, 0.12, 0.01, 0.10, 0.01),
}


# ----------------------------------------------------------------------------- league

def league(n_players: int = 60) -> pd.DataFrame:
    """One row per player: player_id, player, team, team_id, archetype, slug. Ten players per team in TEAMS order."""
    rows = []
    teams = list(TEAMS)
    for i in range(n_players):
        team = teams[(i // 10) % len(teams)]
        arch = ARCHETYPE_PATTERN[i % 10]
        pid = 1001 + i
        rows.append({"player_id": pid, "player": "Player %d" % pid, "team": team, "team_id": TEAMS[team],
                     "archetype": arch, "slug": "plyr%04d" % pid, "jersey": str(i % 10)})
    return pd.DataFrame(rows)


def _season_of(date: dt.date) -> str:
    y = date.year if date.month >= 10 else date.year - 1
    return "%d-%02d" % (y, (y + 1) % 100)


def _game_id(date: dt.date, away: str, home: str) -> str:
    return "%s.%s.at.%s" % (date.strftime("%m.%d.%Y"), away, home)


def _parse_game(sportvu_game: str):
    m, d, y, away, _, home = sportvu_game.split(".")
    return dt.date(int(y), int(m), int(d)), away, home


# ----------------------------------------------------------------------------- one game

def _lineup(rng, roster: pd.DataFrame) -> pd.DataFrame:
    """Five on the floor: two G, two F, one C of the ten."""
    parts = []
    for arch, k in (("G", 2), ("F", 2), ("C", 1)):
        pool = roster[roster["archetype"] == arch]
        parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
    return pd.concat(parts).sort_values("player_id").reset_index(drop=True)


def _mirror(x, y, attack_left: bool):
    if attack_left:
        return x, y
    return COURT_X - x, COURT_Y - y


def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
    """One synthetic game: section 4 frames, section 5 events, section 10a shots (player_id filled), section 16
    rosters, section 19 possessions, and empty section 17, 18, 20 tables."""
    from analytics import schema

    rng = np.random.default_rng(seed)
    lg = league()
    if sportvu_game is None:
        date = dt.date(2015, 11, 1) + dt.timedelta(days=int(seed % 150))
        if heldout:
            home, away = HELDOUT_ARENA, TRAIN_TEAMS[int(seed) % len(TRAIN_TEAMS)]
        else:
            home, away = [TRAIN_TEAMS[i] for i in rng.choice(len(TRAIN_TEAMS), 2, replace=False)]
        sportvu_game = _game_id(date, away, home)
    else:
        date, away, home = _parse_game(sportvu_game)
    if sportvu_game == HELDOUT_GAME_ID or home == HELDOUT_ARENA:
        heldout = True
    season = _season_of(date)
    rosters = {side: lg[lg["team"] == t].reset_index(drop=True) for side, t in (("home", home), ("away", away))}

    frame_blocks, poss_rows, shot_rows, event_rows = [], [], [], []
    frame0, ts0, clock_left = 0, 0, {p: PERIOD_S for p in range(1, 5)}
    lineups = {}
    for p in range(n_possessions):
        period = 1 + (p * 4) // n_possessions
        if p % 4 == 0:
            lineups = {s: _lineup(rng, rosters[s]) for s in ("home", "away")}
        dur = float(rng.uniform(8.0, 20.0))
        nf = max(int(round(dur * FPS)), 5)
        clocks = np.maximum(clock_left[period] - np.arange(nf) / FPS, 0.0)
        clock_left[period] = float(clocks[-1] - 1.0 / FPS)
        off_side = "home" if p % 2 == 0 else "away"
        def_side = "away" if off_side == "home" else "home"
        attack_left = (off_side == "home") == (period <= 2)
        off, dfn = lineups[off_side], lineups[def_side]
        # anchors in the rim frame (rim at RIM_X, RIM_Y, attacked baseline at x = 0)
        ax_off = np.array([RIM_X + rng.uniform(*_OFF_X[a]) for a in off["archetype"]])
        ay_off = np.array([rng.uniform(*_OFF_Y[a]) for a in off["archetype"]])
        ax_def = np.maximum(ax_off - 3.0 + rng.normal(0, 1.5, 5), 1.0)
        ay_def = ay_off + rng.normal(0, 2.0, 5)
        ax_all = np.concatenate([ax_off, ax_def])
        ay_all = np.concatenate([ay_off, ay_def])
        arch_all = list(off["archetype"]) + list(dfn["archetype"])
        noise = rng.normal(0, 1, (nf, 10, 2)) * np.array([_STEP_SD[a] for a in arch_all])[None, :, None]
        walk = lfilter([1.0], [1.0, -0.85], noise, axis=0)
        x = np.clip(ax_all[None, :] + walk[:, :, 0], 0.5, COURT_X - 0.5)
        y = np.clip(ay_all[None, :] + walk[:, :, 1], 0.5, COURT_Y - 0.5)
        x, y = _mirror(x, y, attack_left)
        # handler: three segments, offensive players by archetype weight
        w = np.array([_HANDLER_W[a] for a in off["archetype"]])
        w = w / w.sum()
        seg = np.minimum(np.arange(nf) * 3 // nf, 2)
        handlers = rng.choice(5, 3, p=w)
        h = handlers[seg]
        bx = x[np.arange(nf), h] + rng.normal(0, 0.8, nf)
        by = y[np.arange(nf), h] + rng.normal(0, 0.8, nf)
        bz = np.abs(3.0 + rng.normal(0, 0.5, nf))
        kind = rng.choice(["shot_made", "shot_missed", "turnover", "free_throws"], p=[0.45, 0.40, 0.10, 0.05])
        points = 0
        if kind in ("shot_made", "shot_missed"):
            bz[-FPS:] = np.linspace(4.0, 12.0, FPS)
            sw = np.array([_SHOOTER_W[a] for a in off["archetype"]])
            si = int(rng.choice(5, p=sw / sw.sum()))
            shooter = off.iloc[si]
            d = float(rng.uniform(*_SHOT_DIST[shooter["archetype"]]))
            ang = float(rng.uniform(-1.4, 1.4))
            cx, cy = RIM_X + d * np.cos(ang), RIM_Y + d * np.sin(ang)
            three = d >= 22.0
            made = kind == "shot_made"
            points = (3 if three else 2) if made else 0
            shot_rows.append({
                "game": sportvu_game, "sportvu_game": sportvu_game, "br_code": date.strftime("%Y%m%d") + "0" + home,
                "player": shooter["player"], "player_slug": shooter["slug"], "team": shooter["team"],
                "team_br": shooter["team"], "team_side": off_side, "period": period, "clock_s": float(clocks[-1]),
                "clock_str": "%d:%04.1f" % (int(clocks[-1] // 60), clocks[-1] % 60), "made": bool(made),
                "shot_type": "3PT" if three else "2PT", "shot_distance_ft": float(round(d)),
                "px_left": float(250 + (cy - RIM_Y) * 10 - 10.5), "px_top": float(52.5 + (cx - RIM_X) * 10 - 17),
                "court_x": float(cx), "court_y": float(cy), "basket": "left",
                "loc_x": float((cy - RIM_Y) * 10), "loc_y": float((cx - RIM_X) * 10), "dist_resid_ft": 0.0,
                "desc": "%s %s" % (shooter["player"], "makes" if made else "misses"), "game_date": date.isoformat(),
                "home_team": home, "away_team": away, "split": "heldout_game" if heldout else "train",
                "heldout": bool(heldout), "season_type": "Regular Season", "game_id": None, "event_id": None,
                "player_id": int(shooter["player_id"]), "team_id": int(shooter["team_id"]), "release_frame": None,
                "closest_defender_id": None, "closest_defender_dist_ft": None, "defender_closing_speed_ftps": None,
                "source": "fixture", "endpoint": "fixture", "season": season, "fetched_at": "2026-01-01T00:00:00Z"})
        elif kind == "free_throws":
            points = int(rng.choice([0, 1, 2]))
        block = pd.DataFrame({
            "game": sportvu_game, "gameid": "002150%04d" % (seed % 10000), "date": date.isoformat(),
            "period": np.repeat(period, nf * 10).astype("int8"),
            "frame": np.repeat(frame0 + np.arange(nf), 10).astype("int32"),
            "ts_ms": np.repeat(ts0 + np.arange(nf) * DT_MS, 10).astype("int64"),
            "game_clock_s": np.repeat(clocks, 10).astype("float32"),
            "shot_clock_s": np.repeat(np.maximum(24.0 - np.arange(nf) / FPS, 0.0), 10).astype("float32"),
            "n_players": np.int8(10),
            "ball_x": np.repeat(bx, 10).astype("float32"), "ball_y": np.repeat(by, 10).astype("float32"),
            "ball_z": np.repeat(bz, 10).astype("float32"),
        })
        # rows per frame: home players first, then away; columns x and y follow the lineup order used above
        home_first = (off_side == "home")
        sides = [(off, range(0, 5), home_first), (dfn, range(5, 10), not home_first)]
        sides.sort(key=lambda t: not t[2])
        per_frame = {"team_id": [], "team": [], "home": [], "player_id": [], "player": [], "jersey": []}
        xcols, ycols = [], []
        for tbl, idx, is_home in sides:
            for c in ("team_id", "team", "player_id", "player", "jersey"):
                per_frame[c].extend(tbl[c].tolist())
            per_frame["home"].extend([is_home] * 5)
            xcols.append(x[:, list(idx)])
            ycols.append(y[:, list(idx)])
        xx = np.concatenate(xcols, axis=1).reshape(-1)
        yy = np.concatenate(ycols, axis=1).reshape(-1)
        for c in per_frame:
            block[c] = np.tile(np.array(per_frame[c], dtype=object), nf)
        block["home"] = block["home"].astype(bool)
        block["x"] = xx.astype("float32")
        block["y"] = yy.astype("float32")
        frame_blocks.append(block)
        ev = len(event_rows)
        event_rows.append({"game": sportvu_game, "event": ev, "period": period, "clock_s": float(clocks[-1]),
                           "team": off_side, "desc": kind, "points": points,
                           "score": None, "kind": kind if kind != "free_throws" else "ft_made",
                           "frame": frame0 + nf - 1, "frame_clock_diff_s": 0.0})
        poss_rows.append({"sportvu_game": sportvu_game, "possession": p, "period": period, "start_frame": frame0,
                          "end_frame": frame0 + nf - 1, "start_clock_s": float(clocks[0]), "end_clock_s": float(clocks[-1]),
                          "start_event": ev, "end_event": ev, "offense_team_id": int(TEAMS[home if off_side == "home" else away]),
                          "end_kind": kind, "points": points, "gap": False, "n_frames": nf})
        frame0 += nf
        ts0 += nf * DT_MS
    frames = pd.concat(frame_blocks, ignore_index=True)
    for c in ("team_id", "player_id"):
        frames[c] = frames[c].astype("int32")
    for c in ("team", "player", "jersey", "game", "gameid", "date"):
        frames[c] = frames[c].astype(str)
    frames["n_players"] = frames["n_players"].astype("int8")
    roster_rows = []
    for side in ("home", "away"):
        for _, r in rosters[side].iterrows():
            roster_rows.append({"sportvu_game": sportvu_game, "player_id": int(r["player_id"]), "player": r["player"],
                                "team_id": int(r["team_id"]), "team": r["team"], "home": side == "home",
                                "jersey": r["jersey"], "bref_slug": r["slug"], "bref_name": r["player"],
                                "pbp_name": r["player"], "match_method": "exact"})
    shots = pd.DataFrame(shot_rows)
    return {"frames": frames, "events": pd.DataFrame(event_rows), "shots": shots,
            "rosters": schema.cast(pd.DataFrame(roster_rows), "rosters"),
            "shot_frames": schema.empty("shot_frames"), "rebound_frames": schema.empty("rebound_frames"),
            "possessions": schema.cast(pd.DataFrame(poss_rows), "possessions"),
            "states_1hz": schema.empty("states_1hz"),
            "sportvu_game": sportvu_game, "heldout": bool(heldout), "season": season}


def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0) -> list:
    """n_games games; every heldout_every-th game (1-based) is held out: the first such game is the real held-out
    id 12.16.2015.PHX.at.GSW, later ones are games at CLE."""
    games, n_held = [], 0
    for i in range(n_games):
        date = dt.date(2015, 11, 1) + dt.timedelta(days=3 * i)
        held = heldout_every > 0 and (i + 1) % heldout_every == 0
        if held:
            n_held += 1
            if n_held == 1:
                gid = HELDOUT_GAME_ID
            else:
                gid = _game_id(date, TRAIN_TEAMS[i % len(TRAIN_TEAMS)], HELDOUT_ARENA)
        else:
            home = TRAIN_TEAMS[i % len(TRAIN_TEAMS)]
            away = TRAIN_TEAMS[(i + 1 + i // len(TRAIN_TEAMS)) % len(TRAIN_TEAMS)]
            if away == home:
                away = TRAIN_TEAMS[(i + 2) % len(TRAIN_TEAMS)]
            gid = _game_id(date, away, home)
        games.append(synth_game(seed * 100 + i, heldout=held, sportvu_game=gid))
    return games


# ----------------------------------------------------------------------------- aggregates and positions

def synth_aggregates(seed: int, n_players: int = 60, seasons=("2015-16", "2016-17")):
    """(long aggregates frame per docs/DATA_CONTRACT.md section 11, positions frame). Each player's values follow
    the archetype mean plus a persistent player offset (so a 2016-17 stat is predictable from 2015-16) plus
    season noise. Play-type rows carry metric poss_pct and type_grouping offensive."""
    rng = np.random.default_rng(seed + 7919)
    lg = league(n_players)
    rows = []
    offsets = {(pid, key): float(rng.normal(0, 0.12)) for pid in lg["player_id"] for key in _AGG_MEAN}
    play_off = {pid: rng.normal(0, 0.02, len(PLAY_TYPES)) for pid in lg["player_id"]}
    base = {"season_type": "Regular Season", "split_name": None, "split_value": None, "pt_measure_type": None,
            "player_or_team": "Player", "play_type": None, "type_grouping": None, "measure_type": None,
            "heldout": None, "source": "fixture", "fetched_at": "2026-01-01T00:00:00Z"}
    for _, r in lg.iterrows():
        pid, arch = int(r["player_id"]), r["archetype"]
        for season in seasons:
            ent = {"player_id": pid, "player_name": r["player"], "team_id": int(r["team_id"]),
                   "team_abbreviation": r["team"], "season": season}
            for (endpoint, metric), means in _AGG_MEAN.items():
                mu = means[arch]
                val = mu * (1.0 + offsets[(pid, (endpoint, metric))]) * (1.0 + rng.normal(0, 0.06))
                rows.append({**base, **ent, "endpoint": endpoint, "metric": metric, "value": float(max(val, 0.0))})
            share = np.maximum(np.array(_PLAY_MEAN[arch]) + play_off[pid] + rng.normal(0, 0.01, len(PLAY_TYPES)), 0.0)
            share = share / share.sum()
            for pt, v in zip(PLAY_TYPES, share):
                rows.append({**base, **ent, "endpoint": "synergyplaytypes", "play_type": pt,
                             "type_grouping": "offensive", "metric": "poss_pct", "value": float(v)})
    agg = pd.DataFrame(rows)
    positions = pd.DataFrame([{"player_id": int(r["player_id"]), "season": s, "position": r["archetype"]}
                              for _, r in lg.iterrows() for s in seasons])
    return agg, positions


# ----------------------------------------------------------------------------- DATA_ROOT layout on disk

def write_fixture_root(root, seed: int, n_games: int = 8, heldout_every: int = 0) -> dict:
    """Write the DATA_ROOT layout the lane's loaders read (see analytics/embeddings/train.py):
    sportvu_contract/<game>.parquet (+ heldout/), collect/bref_shots/shotchart/2015-16.parquet (+ heldout/shotchart/),
    analytics/possessions/2015-16/<game>.parquet (+ heldout/2015-16/), analytics/rosters/2015-16/<game>.parquet,
    collect/tracking/aggregates/<season>.parquet, collect/positions.parquet."""
    root = Path(root)
    games = synth_season(seed, n_games, heldout_every)
    shots_train, shots_held, summary = [], [], {"train_games": [], "heldout_games": []}
    for g in games:
        gid, held, season = g["sportvu_game"], g["heldout"], g["season"]
        sub = "heldout" if held else ""
        fdir = root / "sportvu_contract" / sub if held else root / "sportvu_contract"
        fdir.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(fdir / (gid + ".parquet"), index=False)
        for table in ("possessions", "rosters"):
            tdir = root / "analytics" / table / ("heldout/" + season if held else season)
            tdir.mkdir(parents=True, exist_ok=True)
            g[table].to_parquet(tdir / (gid + ".parquet"), index=False)
        (shots_held if held else shots_train).append(g["shots"])
        summary["heldout_games" if held else "train_games"].append(gid)
    sdir = root / "collect" / "bref_shots" / "shotchart"
    sdir.mkdir(parents=True, exist_ok=True)
    pd.concat(shots_train, ignore_index=True).to_parquet(sdir / "2015-16.parquet", index=False)
    if shots_held:
        hdir = root / "collect" / "bref_shots" / "heldout" / "shotchart"
        hdir.mkdir(parents=True, exist_ok=True)
        pd.concat(shots_held, ignore_index=True).to_parquet(hdir / "2015-16.parquet", index=False)
    agg, positions = synth_aggregates(seed)
    adir = root / "collect" / "tracking" / "aggregates"
    adir.mkdir(parents=True, exist_ok=True)
    for season, part in agg.groupby("season"):
        part.to_parquet(adir / (season + ".parquet"), index=False)
    positions.to_parquet(root / "collect" / "positions.parquet", index=False)
    summary["root"] = str(root)
    return summary


# ----------------------------------------------------------------------------- prefer the shared generators

def _shared_usable() -> bool:
    """True when analytics.fixtures imports and its synth_game carries the columns this lane reads."""
    try:
        from analytics import fixtures as fx  # noqa: F401
    except ImportError:
        return False
    try:
        g = fx.synth_game(0, n_possessions=4)
        need = {"frames": {"game", "period", "frame", "ts_ms", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id",
                           "player_id", "x", "y", "date"},
                "shots": {"sportvu_game", "player_id", "shot_type", "made"},
                "possessions": {"sportvu_game", "possession", "start_frame", "end_frame", "offense_team_id"},
                "rosters": {"sportvu_game", "player_id", "team_id", "bref_slug"}}
        for k, cols in need.items():
            if k not in g or not cols <= set(g[k].columns):
                return False
        agg, pos = fx.synth_aggregates(0)
        return {"player_id", "season", "endpoint", "metric", "value"} <= set(agg.columns) and \
            {"player_id", "season", "position"} <= set(pos.columns)
    except Exception:  # noqa: BLE001  (a half-written shared module falls back to the local generator)
        return False


SHARED = False
try:
    from analytics import fixtures as _shared_fx  # noqa: E402
except ImportError:
    _shared_fx = None
if _shared_fx is not None and _shared_usable():
    SHARED = True
    synth_game = _shared_fx.synth_game            # noqa: F811
    synth_season = _shared_fx.synth_season        # noqa: F811
    synth_aggregates = _shared_fx.synth_aggregates  # noqa: F811
    write_fixture_root = _shared_fx.write_fixture_root  # noqa: F811

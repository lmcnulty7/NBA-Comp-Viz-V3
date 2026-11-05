"""tests/_fx_xfg.py: the synthetic season lane XFG tests and `--fixtures` runs read.

Tries the shared generator ``analytics.fixtures`` (lane I1) first and falls back to the local generator
below, which has the same signatures, dict keys and column names:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict(
        frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of those dicts
    write_fixture_root(root, seed, n_games, heldout_every) -> Path  (the DATA_ROOT layout)

The shared generator's output is checked for the keys and columns the xFG lane needs; if the check fails
(the shared module is unfinished or shaped differently) the local generator is used, so this lane's tests
never depend on another lane's state. ``USING_SHARED`` records which one is in use.

Local generator (deterministic given the seed): a game is ``n_possessions`` possessions of 6 s at 25 Hz
(150 frames, ten players on the floor, home and away alternate on offence). About 85 percent of possessions
end in a field-goal attempt released at frame 120 of the possession; the rest are turnovers. The shooter
holds the ball for a touch of 0.2 to 4.4 s before the release with 0 to 4 dribbles (ball z dips under 2 ft),
the closest defender closes at -2 to 8 ft/s over the last 0.4 s, the shooter moves at 0 to 8 ft/s. The make
probability is a known logistic in distance, three-point status, closest-defender distance at release,
closing speed and a per-player skill term, so a contest model with a shooter prior has something to find.
About 10 percent of attempts carry no release frame (``confidence_flag`` ``unmatched``, frames at -1).
Held-out games take the ids of sportvu/splits.json (12.16.2015.PHX.at.GSW first); training games take
synthetic ids ``MM.DD.2016.AAA.at.BBB`` that carry no held-out token. Nothing here is real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics import schema  # noqa: E402

FPS = 25
POSS_FRAMES = 150
RELEASE_AT = 120
APEX_AT = 132
OUTCOME_AT = 145
SEASON = "2015-16"
HELDOUT_IDS = ["12.16.2015.PHX.at.GSW", "12.23.2015.NYK.at.CLE", "01.18.2016.GSW.at.CLE"]
HOME_TEAM_ID, AWAY_TEAM_ID = 100, 200
HOME_PLAYERS = [1001, 1002, 1003, 1004, 1005]
AWAY_PLAYERS = [2001, 2002, 2003, 2004, 2005]
LEFT_BASKET, RIGHT_BASKET = (5.25, 25.0), (88.75, 25.0)
TABLE_KEYS = ("frames", "events", "shots", "rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")
NEEDED = {
    "frames": {"game", "period", "frame", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z",
               "team_id", "player_id", "x", "y"},
    "shots": {"sportvu_game", "event", "player_id", "period", "clock_s", "made", "loc_x", "loc_y", "heldout"},
    "shot_frames": set(schema.columns("shot_frames")),
    "rosters": set(schema.columns("rosters")),
}


def _skill(player_id: int) -> float:
    """A fixed per-player shooting skill in logit units, the same in every game (so a shooter prior helps)."""
    return ((player_id * 7919) % 11) / 10.0 * 0.8 - 0.4


def _game_id(seed: int) -> str:
    month = 1 + (seed // 28) % 12
    day = seed % 28 + 1
    return "%02d.%02d.2016.AAA.at.BBB" % (month, day)


def _clip_court(x, y):
    return float(np.clip(x, 0.0, 94.0)), float(np.clip(y, 0.0, 50.0))


def _local_synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
    rng = np.random.default_rng(seed)
    game = sportvu_game or (HELDOUT_IDS[0] if heldout else _game_id(seed))
    away_abbr, home_abbr = game.split(".")[3], game.split(".")[5]
    date = "2016-01-%02d" % (seed % 28 + 1)
    per_period = max(1, int(np.ceil(n_possessions / 4)))
    frame_blocks, event_rows, shot_rows, sf_rows, rb_rows, poss_rows = [], [], [], [], [], []
    event = 0
    for p in range(n_possessions):
        period = 1 + p // per_period
        i_in_period = p % per_period
        start_clock = 720.0 - 6.0 * i_in_period
        f0 = p * POSS_FRAMES
        home_offence = p % 2 == 0
        off_ids = HOME_PLAYERS if home_offence else AWAY_PLAYERS
        def_ids = AWAY_PLAYERS if home_offence else HOME_PLAYERS
        off_team = HOME_TEAM_ID if home_offence else AWAY_TEAM_ID
        left = (period <= 2) == home_offence
        basket = LEFT_BASKET if left else RIGHT_BASKET
        has_shot = rng.uniform() < 0.85
        # shot geometry
        r = float(rng.uniform(1.0, 28.0))
        th = float(rng.uniform(-np.pi / 2, np.pi / 2))
        loc_x, loc_y = r * np.sin(th) * 10.0, r * np.cos(th) * 10.0
        court_x, court_y = 5.25 + r * np.cos(th), 25.0 + r * np.sin(th)
        if not left:
            court_x, court_y = 94.0 - court_x, 50.0 - court_y
        court_x, court_y = _clip_court(court_x, court_y)
        is_three = r >= 22.0
        shooter = int(rng.choice(off_ids))
        d_def = float(rng.uniform(0.5, 9.0))
        v_close = float(rng.uniform(-2.0, 8.0))
        v_shooter = float(rng.uniform(0.0, 8.0))
        dir_def = float(rng.uniform(0, 2 * np.pi))
        dir_sh = float(rng.uniform(0, 2 * np.pi))
        touch_frames = int(rng.uniform(0.2, 4.4) * FPS)
        n_dribbles = int(rng.integers(0, 5))
        logit = 0.5 - 0.07 * r - 0.25 * is_three + 0.15 * min(d_def, 8.0) - 0.03 * max(v_close, 0.0) + _skill(shooter)
        made = bool(rng.uniform() < 1.0 / (1.0 + np.exp(-logit)))
        matched = rng.uniform() >= 0.10
        # static positions of the other players for this possession
        pos = {}
        for pid in off_ids + def_ids:
            pos[pid] = _clip_court(basket[0] + rng.uniform(-2, 28) * (1 if left else -1), rng.uniform(3, 47))
        pos[shooter] = (court_x, court_y)
        closest = int(def_ids[0])
        others = [d for d in def_ids if d != closest]
        for pid in others:                       # keep the other defenders out of the 6 ft ring with some probability
            if rng.uniform() < 0.7:
                ang = rng.uniform(0, 2 * np.pi)
                pos[pid] = _clip_court(court_x + 9.0 * np.cos(ang), court_y + 9.0 * np.sin(ang))
        handler0 = int(off_ids[(off_ids.index(shooter) + 1) % 5])   # who had the ball before the touch
        k = np.arange(POSS_FRAMES)
        gc = start_clock - k / FPS
        sc = np.clip(24.0 - k / FPS, 0.0, 24.0)
        t_touch0 = RELEASE_AT - touch_frames
        lag = np.clip(RELEASE_AT - k, 0, 10) / FPS
        # shooter and closest defender move at constant speed over the 0.4 s before the release
        sx = np.clip(court_x - v_shooter * lag * np.cos(dir_sh), 0.0, 94.0)
        sy = np.clip(court_y - v_shooter * lag * np.sin(dir_sh), 0.0, 50.0)
        dd = np.maximum(d_def + v_close * lag, 0.3)
        dx = np.clip(sx + dd * np.cos(dir_def), 0.0, 94.0)
        dy = np.clip(sy + dd * np.sin(dir_def), 0.0, 50.0)
        # the ball: with the handler, then held by the shooter (dribbles dip to 0.3 ft), then in flight
        bx = np.full(POSS_FRAMES, pos[handler0][0])
        by = np.full(POSS_FRAMES, pos[handler0][1])
        bz = np.full(POSS_FRAMES, 4.0)
        if has_shot:
            touch = (k >= t_touch0) & (k < RELEASE_AT)
            bx[touch], by[touch] = sx[touch] + 0.8, sy[touch]
            if n_dribbles and touch_frames > 2:
                u = (k - t_touch0) / max(touch_frames, 1)
                phase = u * n_dribbles
                frac = phase - np.floor(phase)
                dip = touch & (k - t_touch0 > 0) & (k - t_touch0 < touch_frames - 1) & (np.abs(frac - 0.5) < 0.5 / max(touch_frames / n_dribbles, 3))
                bz[dip] = 0.3
            flight = k >= RELEASE_AT
            fl = (k - RELEASE_AT) / 25.0
            bx[flight] = sx[flight] + 0.8 + (basket[0] - sx[flight]) * fl[flight]
            by[flight] = sy[flight]
            up = flight & (k <= APEX_AT)
            down = (k > APEX_AT) & (k <= OUTCOME_AT)
            bz[up] = 6.0 + 9.0 * (k[up] - RELEASE_AT) / (APEX_AT - RELEASE_AT)
            bz[down] = 15.0 - 5.0 * (k[down] - APEX_AT) / (OUTCOME_AT - APEX_AT)
            after = k > OUTCOME_AT
            bx[after], by[after], bz[after] = basket[0], basket[1], 9.0
        bx, by = np.clip(bx, 0.0, 94.0), np.clip(by, 0.0, 50.0)
        xs, ys = [], []
        for pid in HOME_PLAYERS + AWAY_PLAYERS:
            if pid == shooter and has_shot:
                x, y = sx, sy
            elif pid == closest and has_shot:
                x, y = dx, dy
            else:
                x, y = np.full(POSS_FRAMES, pos[pid][0]), np.full(POSS_FRAMES, pos[pid][1])
            xs.append(x)
            ys.append(y)
        # one block per possession: frame-major order, home players first within a frame (section 4 ordering)
        pids = np.array(HOME_PLAYERS + AWAY_PLAYERS)
        n_pl = len(pids)
        blk = {"period": np.full(POSS_FRAMES * n_pl, period, dtype=np.int8), "frame": np.repeat(f0 + k, n_pl),
               "game_clock_s": np.repeat(gc, n_pl), "shot_clock_s": np.repeat(sc, n_pl),
               "ball_x": np.repeat(bx, n_pl), "ball_y": np.repeat(by, n_pl), "ball_z": np.repeat(bz, n_pl),
               "player_id": np.tile(pids, POSS_FRAMES), "x": np.stack(xs, axis=1).reshape(-1), "y": np.stack(ys, axis=1).reshape(-1)}
        frame_blocks.append(blk)
        outcome_clock = float(gc[OUTCOME_AT])
        if has_shot:
            kind = "shot_made" if made else "shot_missed"
            points = (3 if is_three else 2) if made else 0
            event_rows.append((game, event, period, outcome_clock, "home" if home_offence else "away",
                               "P%d %s" % (shooter, "makes" if made else "misses"), points, "0-0", kind, f0 + OUTCOME_AT, 0.0))
            shot_rows.append({"game": game, "sportvu_game": game, "event": event, "br_code": "2016010%dBBB" % (seed % 9),
                              "player": "P%d" % shooter, "player_slug": "p%d" % shooter,
                              "team": home_abbr if home_offence else away_abbr, "team_br": home_abbr if home_offence else away_abbr,
                              "team_side": "home" if home_offence else "away", "period": period, "clock_s": outcome_clock,
                              "clock_str": "%d:%04.1f" % (int(outcome_clock // 60), outcome_clock % 60), "made": made,
                              "shot_type": "3PT" if is_three else "2PT", "shot_distance_ft": float(round(r)),
                              "px_left": 239.5 + loc_x, "px_top": 35.5 + loc_y, "court_x": court_x, "court_y": court_y,
                              "basket": "left" if left else "right", "loc_x": float(round(loc_x, 1)), "loc_y": float(round(loc_y, 1)),
                              "dist_resid_ft": float(r - round(r)), "desc": "fixture", "game_date": date,
                              "home_team": home_abbr, "away_team": away_abbr,
                              "split": "heldout_game" if heldout else None, "heldout": bool(heldout), "season_type": "Regular Season",
                              "game_id": None, "event_id": None, "player_id": shooter,
                              "team_id": off_team, "release_frame": None, "closest_defender_id": None,
                              "closest_defender_dist_ft": None, "defender_closing_speed_ftps": None,
                              "source": "fixture", "endpoint": "synth", "season": SEASON, "fetched_at": "2016-01-01T00:00:00Z"})
            sf_rows.append({"sportvu_game": game, "event": event, "period": period, "clock_s": outcome_clock, "made": made,
                            "pbp_shooter_name": "P%d" % shooter, "shooter_id": shooter,
                            "release_frame": f0 + RELEASE_AT if matched else -1,
                            "apex_frame": f0 + APEX_AT if matched else -1,
                            "outcome_frame": f0 + OUTCOME_AT if matched else -1,
                            "shooter_at_release_id": shooter if matched else None,
                            "identity_match": True if matched else None,
                            "confidence": 0.9 if matched else 0.0, "confidence_flag": "high" if matched else "unmatched"})
            if not made:
                rb_rows.append({"sportvu_game": game, "event": event, "rebound_event": None, "period": period, "clock_s": outcome_clock,
                                "rim_frame": f0 + OUTCOME_AT if matched else -1, "rebound_frame": f0 + OUTCOME_AT + 4 if matched else -1,
                                "rebounder_id": closest if matched else None, "pbp_rebounder_name": None, "pbp_rebounder_id": None,
                                "identity_match": None, "confidence": 0.8 if matched else 0.0,
                                "confidence_flag": "high" if matched else "unmatched"})
            end_kind = kind
        else:
            event_rows.append((game, event, period, outcome_clock, "home" if home_offence else "away",
                               "P%d turnover" % handler0, 0, "0-0", "turnover", f0 + OUTCOME_AT, 0.0))
            end_kind, points = "turnover", 0
        poss_rows.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": f0, "end_frame": f0 + POSS_FRAMES - 1,
                          "start_clock_s": float(gc[0]), "end_clock_s": float(gc[-1]), "start_event": event, "end_event": event,
                          "offense_team_id": off_team, "end_kind": end_kind, "points": points, "gap": False, "n_frames": POSS_FRAMES})
        event += 1
    cat = {c: np.concatenate([b[c] for b in frame_blocks]) for c in frame_blocks[0]}
    n = len(cat["frame"])
    is_home = np.isin(cat["player_id"], HOME_PLAYERS)
    frames = pd.DataFrame({"game": game, "gameid": "0021500000", "date": date, "period": cat["period"], "frame": cat["frame"],
                           "ts_ms": 40 * cat["frame"].astype(np.int64), "game_clock_s": cat["game_clock_s"],
                           "shot_clock_s": cat["shot_clock_s"], "n_players": np.full(n, 10, dtype=np.int8),
                           "ball_x": cat["ball_x"], "ball_y": cat["ball_y"], "ball_z": cat["ball_z"],
                           "team_id": np.where(is_home, HOME_TEAM_ID, AWAY_TEAM_ID),
                           "team": np.where(is_home, home_abbr, away_abbr), "home": is_home, "player_id": cat["player_id"],
                           "player": np.char.add("P", cat["player_id"].astype(str)),
                           "jersey": (cat["player_id"] % 100).astype(str), "x": cat["x"], "y": cat["y"]})
    frames = frames.astype({"period": "int8", "frame": "int32", "ts_ms": "int64", "game_clock_s": "float32", "shot_clock_s": "float32",
                            "n_players": "int8", "ball_x": "float32", "ball_y": "float32", "ball_z": "float32", "team_id": "int32",
                            "player_id": "int32", "x": "float32", "y": "float32"})
    events = pd.DataFrame(event_rows, columns=["game", "event", "period", "clock_s", "team", "desc", "points", "score", "kind",
                                               "frame", "frame_clock_diff_s"])
    events["source"], events["endpoint"], events["season"], events["fetched_at"] = "fixture", "synth", SEASON, "2016-01-01T00:00:00Z"
    shots = pd.DataFrame(shot_rows)
    rosters = schema.cast(pd.DataFrame([{"sportvu_game": game, "player_id": pid, "player": "P%d" % pid,
                                         "team_id": HOME_TEAM_ID if pid in HOME_PLAYERS else AWAY_TEAM_ID,
                                         "team": home_abbr if pid in HOME_PLAYERS else away_abbr, "home": pid in HOME_PLAYERS,
                                         "jersey": str(pid % 100), "bref_slug": "p%d" % pid, "bref_name": "P%d" % pid,
                                         "pbp_name": "P%d" % pid, "match_method": "exact"}
                                        for pid in HOME_PLAYERS + AWAY_PLAYERS]), "rosters")
    shot_frames = schema.cast(pd.DataFrame(sf_rows, columns=schema.columns("shot_frames")), "shot_frames") if sf_rows else schema.empty("shot_frames")
    rebound_frames = schema.cast(pd.DataFrame(rb_rows, columns=schema.columns("rebound_frames")), "rebound_frames") if rb_rows else schema.empty("rebound_frames")
    possessions = schema.cast(pd.DataFrame(poss_rows), "possessions")
    return {"frames": frames, "events": events, "shots": shots, "rosters": rosters, "shot_frames": shot_frames,
            "rebound_frames": rebound_frames, "possessions": possessions, "states_1hz": schema.empty("states_1hz"),
            "sportvu_game": game, "heldout": bool(heldout)}


def _local_synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
    out, n_held = [], 0
    for i in range(n_games):
        held = bool(heldout_every) and (i + 1) % heldout_every == 0
        gid = None
        if held:
            gid = HELDOUT_IDS[n_held % len(HELDOUT_IDS)]
            n_held += 1
        out.append(_local_synth_game(seed + i, n_possessions=n_possessions, heldout=held, sportvu_game=gid))
    return out


def _check(g: dict) -> bool:
    try:
        if not all(k in g for k in TABLE_KEYS):
            return False
        for k, cols in NEEDED.items():
            if not cols <= set(g[k].columns):
                return False
        return len(g["shots"]) > 0 and len(g["shot_frames"]) > 0
    except Exception:  # noqa: BLE001  (any shape problem means: use the local generator)
        return False


USING_SHARED = False
try:
    from analytics.fixtures import synth_game as _shared_game, synth_season as _shared_season  # type: ignore

    def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
        try:
            g = _shared_game(seed, n_possessions=n_possessions, heldout=heldout, sportvu_game=sportvu_game)
        except Exception:  # noqa: BLE001
            g = None
        if g is None or not _check(g):
            return _local_synth_game(seed, n_possessions, heldout, sportvu_game)
        g.setdefault("heldout", bool(heldout))
        g.setdefault("sportvu_game", str(g["shots"]["sportvu_game"].iloc[0]))
        return g

    def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
        try:
            season = _shared_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
        except Exception:  # noqa: BLE001
            season = None
        if not season or not all(_check(g) for g in season):
            return _local_synth_season(seed, n_games, heldout_every, n_possessions)
        for g in season:
            g.setdefault("heldout", bool(g["shots"]["heldout"].astype(bool).any()))
            g.setdefault("sportvu_game", str(g["shots"]["sportvu_game"].iloc[0]))
        return season

    USING_SHARED = True
except Exception:  # noqa: BLE001  (ImportError, or a shared module that is not ready)
    synth_game = _local_synth_game
    synth_season = _local_synth_season


def write_fixture_root(root, seed: int = 0, n_games: int = 8, heldout_every: int = 4, season: str = SEASON,
                       n_possessions: int = 40) -> Path:
    """Write a synthetic season in the DATA_ROOT layout the xFG CLIs read:

        DATA_ROOT/sportvu_contract/<game>.parquet, <game>_pbp.parquet            (held-out under heldout/)
        DATA_ROOT/collect/bref_shots/shotchart/<season>.parquet                  (+ heldout/shotchart/)
        DATA_ROOT/analytics/<table>/<season>/<game>.parquet                       (+ heldout/<season>/)
            for shot_frames, rosters, possessions, rebound_frames
    """
    root = Path(root)
    shots_train, shots_held = [], []
    for g in synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions):
        held = bool(g["heldout"])
        game = g["sportvu_game"]
        sv = root / "sportvu_contract" / ("heldout" if held else "")
        sv.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(sv / ("%s.parquet" % game), index=False)
        g["events"].to_parquet(sv / ("%s_pbp.parquet" % game), index=False)
        for table in ("shot_frames", "rosters", "possessions", "rebound_frames"):
            p = schema.table_path(root / "analytics", table, season, game, heldout=held)
            p.parent.mkdir(parents=True, exist_ok=True)
            g[table].to_parquet(p, index=False)
        (shots_held if held else shots_train).append(g["shots"])
    for held, frames in ((False, shots_train), (True, shots_held)):
        if not frames:
            continue
        d = root / "collect" / "bref_shots" / ("heldout/shotchart" if held else "shotchart")
        d.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(d / ("%s.parquet" % season), index=False)
    return root


__all__ = ["synth_game", "synth_season", "write_fixture_root", "USING_SHARED", "HELDOUT_IDS", "FPS", "SEASON"]

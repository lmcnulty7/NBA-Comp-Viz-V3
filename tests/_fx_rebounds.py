"""tests/_fx_rebounds.py: synthetic games for the rebound lane (analytics.rebounds).

Prefers the shared generator analytics.fixtures (lane I1) when it is importable; otherwise a minimal local
generator with the same signatures and the contract column names (docs/DATA_CONTRACT.md sections 4, 5, 16 to 20).
Deterministic given the seed. No real tracking data: every number here is fixture output, never a result.

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict(frames, events, shots, rosters,
        shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of such dicts (held-out id 12.16.2015.PHX.at.GSW first,
        further held-out games at the CLE arena)
    write_fixture_root(root, seed, n_games, heldout_every) -> list of game ids, writing the DATA_ROOT layout

The scripted game: 25 Hz, ten players, one shot per possession of 100 frames. A miss rises to an apex, falls through
rim height at the basket at the rim frame, then travels to the rebounder over ten frames. The rebounder is drawn by a
softmax over a fixed per-player skill (so a rebound-rate prior carries signal across games), distance to the rim at
the rim frame, speed toward the rim and a defensive advantage, plus a constant "none" alternative (team rebound).
Every 13th miss has no rim frame (localiser failure, confidence_flag unmatched) and every 9th located miss carries a
frame-level rebounder that disagrees with the play-by-play (identity_match false).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # pragma: no cover - the shared generator wins when it exists
    from analytics.fixtures import synth_game, synth_season, write_fixture_root  # noqa: F401
    SOURCE = "analytics.fixtures"
except ImportError:
    SOURCE = "tests._fx_rebounds"

    from analytics import schema

    FPS = 25
    POSS_FRAMES = 100
    RELEASE, APEX, RIM, REB = 40, 52, 65, 75
    BASKETS = ((5.25, 25.0), (88.75, 25.0))
    HELDOUT_ID = "12.16.2015.PHX.at.GSW"
    HOME_TEAM, AWAY_TEAM = (100, "FIX", True), (200, "SYN", False)
    PLAYERS = [  # (player_id, name, team_id, skill): skill is fixed across seeds so a prior can learn it
        (1001, "Hal Adams", 100, 1.1), (1002, "Ben Brook", 100, 0.2), (1003, "Cal Cole", 100, -0.4),
        (1004, "Dan Dent", 100, 0.6), (1005, "Eli Egan", 100, -0.9),
        (2001, "Fay Ford", 200, 0.9), (2002, "Gus Gill", 200, -0.2), (2003, "Hank Hale", 200, 0.4),
        (2004, "Ike Ivy", 200, -0.7), (2005, "Jon Judd", 200, 0.0),
    ]
    SKILL = {p[0]: p[3] for p in PLAYERS}
    NONE_UTILITY = -0.7

    def _pbp_name(full: str) -> str:
        first, last = full.split(" ", 1)
        return "%s. %s" % (first[0], last)

    def _game_id(k: int) -> str:
        return "11.%02d.2015.SYN.at.FIX" % (k + 1)

    def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
        rng = np.random.default_rng(seed)
        game = sportvu_game or (HELDOUT_ID if heldout else _game_id(seed))
        per_period = max(1, int(np.ceil(n_possessions / 4)))
        frame_rows, events, shot_frames, reb_frames, poss_rows, shots = [], [], [], [], [], []
        ev_idx = 0
        for p in range(n_possessions):
            period = p // per_period + 1
            f0 = p * POSS_FRAMES
            f_in_period = (p % per_period) * POSS_FRAMES
            clock0 = 720.0 - f_in_period / FPS
            home_off = (p % 2 == 0)
            basket = BASKETS[1] if home_off else BASKETS[0]
            bx, by = basket
            sign = 1.0 if bx < 47 else -1.0          # court centre is in this x direction from the basket
            off_ids = [pl[0] for pl in PLAYERS if (pl[2] == 100) == home_off]
            # player positions: around the basket at the rim frame, drifting with a velocity
            pos_rim, vel = {}, {}
            for pid, _, tid, _ in PLAYERS:
                is_off = pid in off_ids
                d = rng.uniform(5.0, 24.0) if is_off else rng.uniform(2.0, 14.0)
                th = rng.uniform(-np.pi / 2 + 0.15, np.pi / 2 - 0.15)
                x, y = bx + sign * d * np.cos(th), by + d * np.sin(th)
                toward = rng.uniform(-1.5, 8.0)       # ft/s toward the rim
                ux, uy = (bx - x) / max(d, 1e-6), (by - y) / max(d, 1e-6)
                pos_rim[pid] = (x, y)
                vel[pid] = (toward * ux + rng.normal(0, 0.5), toward * uy + rng.normal(0, 0.5))
            shooter = int(rng.choice(off_ids))
            sx, sy = pos_rim[shooter][0] - vel[shooter][0] * (RIM - RELEASE) / FPS, pos_rim[shooter][1] - vel[shooter][1] * (RIM - RELEASE) / FPS
            shot_dist = float(np.hypot(sx - bx, sy - by))
            made = bool(rng.uniform() < 0.42)
            apex_z = 13.0 + 0.15 * shot_dist + rng.normal(0, 0.8)
            # rebounder draw (misses only)
            rebounder = None
            if not made:
                util = {}
                for pid in SKILL:
                    x, y = pos_rim[pid]
                    d = np.hypot(x - bx, y - by)
                    ux, uy = (bx - x) / max(d, 1e-6), (by - y) / max(d, 1e-6)
                    spd = vel[pid][0] * ux + vel[pid][1] * uy
                    util[pid] = SKILL[pid] - 0.22 * d + 0.12 * spd + (0.0 if pid in off_ids else 0.5)
                keys = list(util) + [-1]
                u = np.array([util[k] for k in util] + [NONE_UTILITY])
                pr = np.exp(u - u.max()); pr /= pr.sum()
                rebounder = int(rng.choice(keys, p=pr))
            # ball keyframes
            mid = ((sx + bx) / 2, (sy + by) / 2)
            keys = [(0, (sx, sy, 5.0)), (RELEASE, (sx, sy, 6.5)), (APEX, (mid[0], mid[1], apex_z)),
                    (RIM - 1, (bx, by, 10.4)), (RIM, (bx, by, 9.6))]
            if made:
                keys += [(RIM + 7, (bx, by, 0.5)), (POSS_FRAMES - 1, (bx + sign * 3, by, 0.5))]
            elif rebounder == -1:
                keys += [(REB, (bx + sign * 1.0, 49.0 if by < 25 else 1.0, 3.0)), (POSS_FRAMES - 1, (bx + sign * 1.0, 49.0 if by < 25 else 1.0, 0.5))]
            else:
                rx = pos_rim[rebounder][0] + vel[rebounder][0] * (REB - RIM) / FPS
                ry = pos_rim[rebounder][1] + vel[rebounder][1] * (REB - RIM) / FPS
                keys += [(REB, (rx, ry, 4.0)), (POSS_FRAMES - 1, (rx, ry, 4.0))]
            kf = np.array([k for k, _ in keys], float); kv = np.array([v for _, v in keys], float)
            ff = np.arange(POSS_FRAMES, dtype=float)
            ball = np.stack([np.interp(ff, kf, kv[:, i]) for i in range(3)], axis=1)
            ball[:, 0] = np.clip(ball[:, 0], 0.5, 93.5); ball[:, 1] = np.clip(ball[:, 1], 0.5, 49.5)
            clocks = clock0 - ff / FPS
            pids = np.array([pl[0] for pl in PLAYERS], dtype="int32")
            px = np.stack([np.clip(pos_rim[pid][0] + vel[pid][0] * (ff - RIM) / FPS, 0.5, 93.5) for pid in pids])
            py = np.stack([np.clip(pos_rim[pid][1] + vel[pid][1] * (ff - RIM) / FPS, 0.5, 49.5) for pid in pids])
            n = len(pids)
            frame_rows.append({
                "period": np.full(n * POSS_FRAMES, period, dtype="int8"),
                "frame": np.tile(f0 + ff, n).astype("int32"),
                "game_clock_s": np.tile(clocks, n).astype("float32"),
                "ball_x": np.tile(ball[:, 0], n).astype("float32"), "ball_y": np.tile(ball[:, 1], n).astype("float32"),
                "ball_z": np.tile(ball[:, 2], n).astype("float32"),
                "team_id": np.repeat(np.array([pl[2] for pl in PLAYERS], dtype="int32"), POSS_FRAMES),
                "player_id": np.repeat(pids, POSS_FRAMES),
                "x": px.reshape(-1).astype("float32"), "y": py.reshape(-1).astype("float32")})
            side = "home" if home_off else "away"
            shooter_name = _pbp_name(dict((q[0], q[1]) for q in PLAYERS)[shooter])
            rim_clock = float(clock0 - RIM / FPS)
            shot_ev = ev_idx
            events.append({"game": game, "event": ev_idx, "period": period, "clock_s": rim_clock, "team": side,
                           "desc": "%s %s %d-pt jump shot from %d ft" % (shooter_name, "makes" if made else "misses", 3 if shot_dist >= 22 else 2, round(shot_dist)),
                           "points": (3 if shot_dist >= 22 else 2) if made else 0, "score": None,
                           "kind": "shot_made" if made else "shot_missed", "frame": f0 + RIM, "frame_clock_diff_s": 0.0})
            ev_idx += 1
            reb_ev = None
            if not made:
                reb_ev = ev_idx
                if rebounder == -1:
                    desc, rside = "%s rebound by Team" % ("Offensive" if rng.uniform() < 0.5 else "Defensive"), side
                else:
                    r_off = rebounder in off_ids
                    desc = "%s rebound by %s" % ("Offensive" if r_off else "Defensive", _pbp_name(dict((q[0], q[1]) for q in PLAYERS)[rebounder]))
                    rside = side if r_off else ("away" if side == "home" else "home")
                events.append({"game": game, "event": ev_idx, "period": period, "clock_s": float(clock0 - REB / FPS), "team": rside,
                               "desc": desc, "points": 0, "score": None, "kind": "rebound", "frame": f0 + REB, "frame_clock_diff_s": 0.0})
                ev_idx += 1
            shot_frames.append({"sportvu_game": game, "event": shot_ev, "period": period, "clock_s": rim_clock, "made": made,
                                "pbp_shooter_name": shooter_name, "shooter_id": shooter, "release_frame": f0 + RELEASE,
                                "apex_frame": f0 + APEX, "outcome_frame": f0 + RIM, "shooter_at_release_id": shooter,
                                "identity_match": True, "confidence": 0.8, "confidence_flag": "high"})
            shots.append({"sportvu_game": game, "event": shot_ev, "period": period, "clock_s": rim_clock, "made": made,
                          "shot_distance_ft": shot_dist, "heldout": heldout})
            if not made:
                n_miss = len(reb_frames)
                lost = (n_miss % 13 == 12)
                if lost:
                    row = {"rim_frame": -1, "rebound_frame": -1, "rebounder_id": None, "pbp_rebounder_name": None,
                           "pbp_rebounder_id": None, "identity_match": None, "confidence": 0.0, "confidence_flag": "unmatched"}
                elif rebounder == -1:
                    row = {"rim_frame": f0 + RIM, "rebound_frame": -1, "rebounder_id": None, "pbp_rebounder_name": None,
                           "pbp_rebounder_id": None, "identity_match": None, "confidence": 0.0, "confidence_flag": "unmatched"}
                else:
                    wrong = (n_miss % 9 == 8)
                    frame_id = rebounder if not wrong else int([q for q in SKILL if q != rebounder][n_miss % 9])
                    row = {"rim_frame": f0 + RIM, "rebound_frame": f0 + REB, "rebounder_id": frame_id,
                           "pbp_rebounder_name": _pbp_name(dict((q[0], q[1]) for q in PLAYERS)[rebounder]),
                           "pbp_rebounder_id": rebounder, "identity_match": not wrong,
                           "confidence": 0.87 if not wrong else 0.43, "confidence_flag": "high" if not wrong else "low"}
                reb_frames.append({"sportvu_game": game, "event": shot_ev, "rebound_event": reb_ev, "period": period,
                                   "clock_s": rim_clock, **row})
            poss_rows.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": f0,
                              "end_frame": f0 + POSS_FRAMES - 1, "start_clock_s": float(clock0),
                              "end_clock_s": float(clock0 - (POSS_FRAMES - 1) / FPS), "start_event": shot_ev,
                              "end_event": reb_ev if reb_ev is not None else shot_ev,
                              "offense_team_id": 100 if home_off else 200,
                              "end_kind": "shot_made" if made else "shot_missed",
                              "points": (3 if shot_dist >= 22 else 2) if made else 0, "gap": False, "n_frames": POSS_FRAMES})
        cols = {k: np.concatenate([r[k] for r in frame_rows]) for k in frame_rows[0]}
        names = {pl[0]: pl[1] for pl in PLAYERS}
        frames = pd.DataFrame(cols)
        frames["home"] = frames["team_id"].to_numpy() == 100
        frames["team"] = np.where(frames["home"], HOME_TEAM[1], AWAY_TEAM[1])
        frames["player"] = frames["player_id"].map(names)
        frames["jersey"] = (frames["player_id"] % 100).astype(str)
        frames["game"] = game
        frames["gameid"] = "0029900%03d" % (seed % 1000)
        frames["date"] = "2015-11-%02d" % (seed % 28 + 1)
        frames["ts_ms"] = (frames["frame"].astype("int64") * 40)
        frames["shot_clock_s"] = np.float32(np.nan)
        frames["n_players"] = np.int8(len(PLAYERS))
        frames = frames[["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
                         "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]]
        for c in ("game", "gameid", "date", "team", "player", "jersey"):
            frames[c] = frames[c].astype("string")
        frames = frames.sort_values(["frame", "home"], ascending=[True, False], kind="stable").reset_index(drop=True)
        ev = pd.DataFrame(events)
        for c, t in (("event", "int32"), ("period", "int8"), ("clock_s", "float32"), ("points", "int8"), ("frame", "int32"), ("frame_clock_diff_s", "float32")):
            ev[c] = ev[c].astype(t)
        ev["source"] = "fixture"; ev["endpoint"] = "synth"; ev["season"] = "2015-16"; ev["fetched_at"] = "2015-11-01T00:00:00Z"
        rosters = pd.DataFrame([{"sportvu_game": game, "player_id": pid, "player": name, "team_id": tid,
                                 "team": HOME_TEAM[1] if tid == 100 else AWAY_TEAM[1], "home": tid == 100, "jersey": str(pid % 100),
                                 "bref_slug": name.split()[1].lower() + "01", "bref_name": name, "pbp_name": _pbp_name(name),
                                 "match_method": "exact"} for pid, name, tid, _ in PLAYERS])
        return {"frames": frames, "events": ev, "shots": pd.DataFrame(shots),
                "rosters": schema.cast(rosters, "rosters"),
                "shot_frames": schema.cast(pd.DataFrame(shot_frames), "shot_frames"),
                "rebound_frames": schema.cast(pd.DataFrame(reb_frames, columns=schema.columns("rebound_frames")), "rebound_frames"),
                "possessions": schema.cast(pd.DataFrame(poss_rows), "possessions"),
                "states_1hz": schema.empty("states_1hz")}

    def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0) -> list:
        out, n_held = [], 0
        for k in range(n_games):
            held = heldout_every > 0 and (k + 1) % heldout_every == 0
            if held:
                gid = HELDOUT_ID if n_held == 0 else "12.%02d.2015.SYN.at.CLE" % (n_held + 1)
                n_held += 1
            else:
                gid = _game_id(k)
            out.append(synth_game(seed * 1000 + k, heldout=held, sportvu_game=gid))
        return out

    def write_fixture_root(root, seed: int, n_games: int = 8, heldout_every: int = 0) -> list:
        """DATA_ROOT layout: sportvu_contract/<game>.parquet and <game>_pbp.parquet (held-out under heldout/),
        analytics/<table>/2015-16/<game>.parquet (held-out under analytics/<table>/heldout/2015-16/)."""
        root = Path(root)
        ids = []
        for g in synth_season(seed, n_games, heldout_every):
            game = str(g["frames"]["game"].iat[0])
            held = bool(g["shots"]["heldout"].any()) if len(g["shots"]) else False
            sv = root / "sportvu_contract" / ("heldout" if held else "")
            sv.mkdir(parents=True, exist_ok=True)
            g["frames"].to_parquet(sv / (game + ".parquet"), index=False)
            g["events"].to_parquet(sv / (game + "_pbp.parquet"), index=False)
            for table in ("rebound_frames", "shot_frames", "rosters", "possessions"):
                p = schema.table_path(root / "analytics", table, "2015-16", game, heldout=held)
                p.parent.mkdir(parents=True, exist_ok=True)
                g[table].to_parquet(p, index=False)
            ids.append(game)
        return ids


__all__ = ["synth_game", "synth_season", "write_fixture_root", "SOURCE"]

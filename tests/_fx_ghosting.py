"""tests/_fx_ghosting.py: synthetic games for the ghosting lane.

Prefers the shared generator ``analytics.fixtures`` (lane I1) and falls back to the deterministic local
generator below with the same signatures and column names, so the ghosting tests pass with the fallback
alone. ``USING_SHARED`` says which one is active.

Fallback dynamics (25 Hz, 94 x 50 ft court, baskets at (5.25, 25) and (88.75, 25)):
  - ten players on the floor for the whole game: home team id 100 (players 1001..1005) and away team id 200
    (players 2001..2005); the home team attacks the right basket in periods 1 and 2, the left one in 3 and 4
  - possessions alternate offence, last 4 to 8 s and tile the game with no overlap; frames are numbered
    0..N-1 over the game, the game clock counts down 720 s per period, the shot clock is 24 minus elapsed
  - offensive players move with a smooth Ornstein-Uhlenbeck velocity inside the attacking half; the ball
    sits on a handler who changes every 2 s
  - defender i guards offensive player i (same slot): its target is the point ``gap_ft`` feet from the
    assignment toward the attacking basket, where ``gap_ft`` is drawn once per possession (1 to 8 ft, the
    possession's looseness); the defender moves 15 percent of the way to the target every frame plus
    0.15 ft of noise per axis, so its displacement is a noisy function of the relative positions
  - the chance of a score rises with the looseness (logistic in gap_ft), so a looser defence allows more
    points; made shots are 3 points with probability 0.35, misses end as shot_missed (80 percent) or
    turnover (20 percent)
Every table is cast to the contract dtypes where a contract exists (analytics.schema); the frames and
play-by-play tables follow docs/DATA_CONTRACT.md sections 4 and 5; the shots table carries the section
10a columns the xFG tests use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # pragma: no cover - exercised only once lane I1 lands
    from analytics.fixtures import synth_game, synth_season  # noqa: F401
    try:
        from analytics.fixtures import write_fixture_root  # noqa: F401
    except ImportError:  # the shared module may not expose it yet
        write_fixture_root = None  # type: ignore[assignment]
    USING_SHARED = True
except ImportError:
    USING_SHARED = False

    from analytics import schema

    HZ = 25
    COURT = (94.0, 50.0)
    BASKETS = ((5.25, 25.0), (88.75, 25.0))
    HOME_ID, AWAY_ID = 100, 200
    HOME_IDS = [1001, 1002, 1003, 1004, 1005]
    AWAY_IDS = [2001, 2002, 2003, 2004, 2005]
    HELDOUT_GAME = "12.16.2015.PHX.at.GSW"      # the held-out id of sportvu/splits.json

    def _game_id(seed: int, heldout: bool, sportvu_game) -> str:
        if sportvu_game:
            return str(sportvu_game)
        if heldout:
            return HELDOUT_GAME
        return "01.%02d.2016.AAA.at.BBB" % (1 + seed % 28)

    def _attack_right(home_offense: bool, period: int) -> bool:
        return home_offense if period <= 2 else not home_offense

    def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
        """One synthetic game -> dict(frames, events, shots, rosters, shot_frames, rebound_frames,
        possessions, states_1hz)."""
        rng = np.random.default_rng(1000 + seed)
        game = _game_id(seed, heldout, sportvu_game)
        n_possessions = max(int(n_possessions), 1)
        per_period = max(n_possessions // 4, 1)
        frame = 0
        clock = {p: 720.0 for p in range(1, 5)}
        frames_rows, poss_rows, ev_rows, shot_rows, sf_rows, rb_rows, st_rows = [], [], [], [], [], [], []
        event = 0
        for p in range(n_possessions):
            period = min(1 + p // per_period, 4)
            home_off = p % 2 == 0
            off_team, def_team = (HOME_ID, AWAY_ID) if home_off else (AWAY_ID, HOME_ID)
            off_ids, def_ids = (HOME_IDS, AWAY_IDS) if home_off else (AWAY_IDS, HOME_IDS)
            right = _attack_right(home_off, period)
            basket = np.array(BASKETS[1] if right else BASKETS[0])
            dur_s = float(rng.uniform(4.0, 8.0))
            n = int(round(dur_s * HZ))
            gap_ft = float(rng.uniform(1.0, 8.0))
            # offence: positions in the attacking half, smooth velocities
            xs = rng.uniform(55, 90, 5) if right else rng.uniform(4, 39, 5)
            O = np.stack([xs, rng.uniform(5, 45, 5)], axis=1)
            V = rng.normal(0, 1.5, (5, 2))
            to_b = basket - O
            D = O + gap_ft * to_b / np.maximum(np.linalg.norm(to_b, axis=1, keepdims=True), 1e-6)
            D += rng.normal(0, 1.0, (5, 2))
            handler = int(rng.integers(0, 5))
            start_frame, start_clock = frame, clock[period]
            for i in range(n):
                if i % (2 * HZ) == 0 and i:
                    handler = int(rng.integers(0, 5))
                V = 0.96 * V + rng.normal(0, 0.35, (5, 2))
                O = O + V / HZ
                lo_x, hi_x = (48.0, COURT[0] - 1) if right else (1.0, 46.0)
                O[:, 0] = np.clip(O[:, 0], lo_x, hi_x)
                O[:, 1] = np.clip(O[:, 1], 1.0, COURT[1] - 1)
                to_b = basket - O
                target = O + gap_ft * to_b / np.maximum(np.linalg.norm(to_b, axis=1, keepdims=True), 1e-6)
                D = D + 0.15 * (target - D) + rng.normal(0, 0.15, (5, 2))
                ball = np.array([O[handler, 0], O[handler, 1], 4.0 + 0.5 * np.sin(i / 5.0)])
                gc = start_clock - i / HZ
                sc = max(24.0 - i / HZ, 0.0)
                common = {"game": game, "gameid": "0021500000", "date": "2016-01-01", "period": period, "frame": frame,
                          "ts_ms": 1_450_000_000_000 + frame * 40, "game_clock_s": gc, "shot_clock_s": sc, "n_players": 10,
                          "ball_x": ball[0], "ball_y": ball[1], "ball_z": ball[2]}
                for k in range(5):      # home rows first, as the source orders them
                    hid = HOME_IDS[k]
                    hx, hy = (O[k] if home_off else D[k])
                    frames_rows.append({**common, "team_id": HOME_ID, "team": "BBB", "home": True, "player_id": hid,
                                        "player": "Home Player%d" % k, "jersey": str(k), "x": hx, "y": hy})
                for k in range(5):
                    aid = AWAY_IDS[k]
                    ax, ay = (O[k] if not home_off else D[k])
                    frames_rows.append({**common, "team_id": AWAY_ID, "team": "AAA", "home": False, "player_id": aid,
                                        "player": "Away Player%d" % k, "jersey": str(k), "x": ax, "y": ay})
                if i % HZ == 0:
                    oo = sorted(zip(off_ids, O.tolist()))
                    dd = sorted(zip(def_ids, D.tolist()))
                    hd = min(np.hypot(D[:, 0] - O[handler, 0], D[:, 1] - O[handler, 1]))
                    st_rows.append({"sportvu_game": game, "possession": p, "frame": frame, "period": period, "game_clock_s": gc,
                                    "shot_clock_s": sc, "elapsed_in_possession_s": i / HZ, "offense_team_id": off_team,
                                    "handler_id": off_ids[handler], "ball_x": ball[0], "ball_y": ball[1], "ball_z": ball[2],
                                    **{"off%d_id" % (j + 1): oo[j][0] for j in range(5)},
                                    **{"off%d_x" % (j + 1): oo[j][1][0] for j in range(5)},
                                    **{"off%d_y" % (j + 1): oo[j][1][1] for j in range(5)},
                                    **{"def%d_id" % (j + 1): dd[j][0] for j in range(5)},
                                    **{"def%d_x" % (j + 1): dd[j][1][0] for j in range(5)},
                                    **{"def%d_y" % (j + 1): dd[j][1][1] for j in range(5)},
                                    "handler_nearest_def_dist_ft": hd})
                frame += 1
            end_frame = frame - 1
            end_clock = start_clock - (n - 1) / HZ
            clock[period] = end_clock - 1.0 / HZ
            p_score = 1.0 / (1.0 + np.exp(-(-1.2 + 0.45 * (gap_ft - 4.5))))
            made = bool(rng.uniform() < p_score)
            three = bool(rng.uniform() < 0.35)
            if made:
                points, end_kind, kind = (3 if three else 2), "shot_made", "shot_made"
            elif rng.uniform() < 0.8:
                points, end_kind, kind = 0, "shot_missed", "shot_missed"
            else:
                points, end_kind, kind = 0, "turnover", "turnover"
            shooter = off_ids[handler]
            ev_rows.append({"game": game, "event": event, "period": period, "clock_s": end_clock,
                            "team": "home" if home_off else "away", "desc": "%s by %d" % (kind, shooter), "points": points,
                            "score": None, "kind": kind, "frame": end_frame, "frame_clock_diff_s": 0.0,
                            "source": "fixture", "endpoint": "synth", "season": "2015-16", "fetched_at": "2026-01-01T00:00:00Z",
                            "event_id": None, "event_type": None, "action_type": None, "player1_id": shooter,
                            "player2_id": None, "player3_id": None, "team_id": off_team, "clock_str": None})
            poss_rows.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": start_frame,
                              "end_frame": end_frame, "start_clock_s": start_clock, "end_clock_s": end_clock,
                              "start_event": event, "end_event": event, "offense_team_id": off_team, "end_kind": end_kind,
                              "points": points, "gap": False, "n_frames": n})
            if kind != "turnover":
                sx, sy = O[handler]
                dist = float(np.hypot(sx - basket[0], sy - basket[1]))
                shot_rows.append({"game": game, "sportvu_game": game, "period": period, "clock_s": end_clock, "made": made,
                                  "loc_x": (sy - 25.0) * 10, "loc_y": (dist * 10) if dist > 0 else 0.0,
                                  "shot_type": "3PT" if three else "2PT", "shot_distance_ft": float(round(dist)),
                                  "court_x": sx, "court_y": sy, "split": "heldout_game" if heldout else None,
                                  "heldout": heldout, "source": "bref", "release_frame": end_frame - 10,
                                  "closest_defender_dist_ft": None})
                sf_rows.append({"sportvu_game": game, "event": event, "period": period, "clock_s": end_clock, "made": made,
                                "pbp_shooter_name": "Player %d" % shooter, "shooter_id": shooter,
                                "release_frame": end_frame - 10, "apex_frame": end_frame - 5, "outcome_frame": end_frame,
                                "shooter_at_release_id": shooter, "identity_match": True, "confidence": 0.9,
                                "confidence_flag": "high"})
                if not made:
                    rb_rows.append({"sportvu_game": game, "event": event, "rebound_event": None, "period": period,
                                    "clock_s": end_clock, "rim_frame": end_frame, "rebound_frame": end_frame,
                                    "rebounder_id": def_ids[0], "pbp_rebounder_name": None, "pbp_rebounder_id": None,
                                    "identity_match": None, "confidence": 0.5, "confidence_flag": "low"})
            event += 1
        for r in st_rows:
            pts = next(q["points"] for q in poss_rows if q["possession"] == r["possession"])
            kind = next(q["end_kind"] for q in poss_rows if q["possession"] == r["possession"])
            r["points_to_end"], r["possession_end_kind"] = pts, kind
        rosters = []
        for pid, k in [(i, j) for j, i in enumerate(HOME_IDS)]:
            rosters.append({"sportvu_game": game, "player_id": pid, "player": "Home Player%d" % k, "team_id": HOME_ID,
                            "team": "BBB", "home": True, "jersey": str(k), "bref_slug": "homepl%02d" % k,
                            "bref_name": "Home Player%d" % k, "pbp_name": "H. Player%d" % k, "match_method": "exact"})
        for pid, k in [(i, j) for j, i in enumerate(AWAY_IDS)]:
            rosters.append({"sportvu_game": game, "player_id": pid, "player": "Away Player%d" % k, "team_id": AWAY_ID,
                            "team": "AAA", "home": False, "jersey": str(k), "bref_slug": "awaypl%02d" % k,
                            "bref_name": "Away Player%d" % k, "pbp_name": "A. Player%d" % k, "match_method": "exact"})
        frames = pd.DataFrame(frames_rows)
        for c, t in {"period": "int8", "frame": "int32", "ts_ms": "int64", "game_clock_s": "float32", "shot_clock_s": "float32",
                     "n_players": "int8", "ball_x": "float32", "ball_y": "float32", "ball_z": "float32", "team_id": "int32",
                     "player_id": "int32", "x": "float32", "y": "float32"}.items():
            frames[c] = frames[c].astype(t)
        events = pd.DataFrame(ev_rows)
        out = {"frames": frames, "events": events, "shots": pd.DataFrame(shot_rows),
               "rosters": schema.cast(pd.DataFrame(rosters), "rosters"),
               "possessions": schema.cast(pd.DataFrame(poss_rows), "possessions"),
               "states_1hz": schema.cast(pd.DataFrame(st_rows), "states_1hz")}
        out["shot_frames"] = schema.cast(pd.DataFrame(sf_rows), "shot_frames") if sf_rows else schema.empty("shot_frames")
        out["rebound_frames"] = schema.cast(pd.DataFrame(rb_rows), "rebound_frames") if rb_rows else schema.empty("rebound_frames")
        out["sportvu_game"], out["heldout"] = game, bool(heldout)
        return out

    def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
        """n_games synthetic games with distinct ids; every heldout_every-th game (1-based, 0 = none) is the
        held-out game of sportvu/splits.json. Returns the list of synth_game dicts."""
        games = []
        for g in range(n_games):
            held = bool(heldout_every) and (g + 1) % heldout_every == 0
            gid = None if held else "01.%02d.2016.AAA.at.BBB" % (1 + g)
            games.append(synth_game(seed * 100 + g, n_possessions=n_possessions, heldout=held, sportvu_game=gid))
        return games

    def write_fixture_root(root, seed: int, n_games: int, heldout_every: int = 0, n_possessions: int = 40,
                           season: str = "2015-16") -> list:
        """Write a DATA_ROOT with the Mac layout: root/sportvu_contract/<game>.parquet and <game>_pbp.parquet
        (held-out under heldout/), root/analytics/<table>/<season>/<game>.parquet (held-out under
        heldout/<season>/). Returns the game ids written."""
        root = Path(root)
        ids = []
        for g in synth_season(seed, n_games, heldout_every, n_possessions=n_possessions):
            gid, held = g["sportvu_game"], g["heldout"]
            sv = root / "sportvu_contract" / ("heldout" if held else "")
            sv.mkdir(parents=True, exist_ok=True)
            g["frames"].to_parquet(sv / (gid + ".parquet"), index=False)
            g["events"].to_parquet(sv / (gid + "_pbp.parquet"), index=False)
            for table in ("possessions", "rosters", "states_1hz", "shot_frames", "rebound_frames"):
                p = schema.table_path(root / "analytics", table, season, gid, heldout=held)
                p.parent.mkdir(parents=True, exist_ok=True)
                g[table].to_parquet(p, index=False)
            ids.append(gid)
        return ids


__all__ = ["synth_game", "synth_season", "write_fixture_root", "USING_SHARED"]

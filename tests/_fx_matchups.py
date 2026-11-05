"""tests/_fx_matchups.py: synthetic fixtures for the matchups lane.

Prefers the shared generator analytics.fixtures (lane I1) when it imports; otherwise a minimal local
generator with the same signatures and the same column names, deterministic given the seed:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict(
        frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of synth_game dicts
    write_fixture_root(root, seed, n_games, heldout_every) -> DATA_ROOT layout on disk

Fallback generator (nothing here is data: every number is drawn from a seeded generator):
  - 5 Hz frames (the contract fixes no frame rate; the 1 Hz sampler reads clocks, not row counts)
  - 7 players per team, 5 on the floor per possession; the offence alternates each possession
  - defenders track one offensive assignment each (a permutation drawn per possession) with a 1 ft
    position noise and a 1 to 6 ft gap toward the basket; with probability 0.3 two defenders swap
    assignments mid-possession, so the assignment procedure has switches to find
  - each possession ends with a shot (made with a probability that falls with distance and rises with
    the gap to the assigned defender, minus that defender's latent skill) or, with probability 0.12,
    a turnover
  - the held-out ids are the SportVU ids of sportvu/splits.json, so analytics.eval.refuse_heldout
    recognises them; training ids are 01.DD.2016.AAA.at.BBB and carry no held-out token
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - the shared generator is preferred when the lead has merged it
    from analytics.fixtures import synth_game, synth_season, write_fixture_root  # type: ignore  # noqa: F401
    SHARED = True
except ImportError:
    SHARED = False

HZ = 5
PERIOD_S = 720.0
BASKETS = ((5.25, 25.0), (88.75, 25.0))
HOME_ID, AWAY_ID = 1610612744, 1610612751
HELD_IDS = ("12.16.2015.PHX.at.GSW", "01.18.2016.GSW.at.CLE", "12.23.2015.NYK.at.CLE")
TRAIN_ID = "01.%02d.2016.AAA.at.BBB"
FIRST = ["Ann", "Bo", "Cy", "Di", "Ed", "Fay", "Gus"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _game_id(seed: int, heldout: bool) -> str:
    if heldout:
        return HELD_IDS[seed % len(HELD_IDS)]
    return TRAIN_ID % (seed % 28 + 1)


def _roster(game: str, rng) -> pd.DataFrame:
    away_abbr, home_abbr = game.split(".")[3], game.split(".")[5]
    rows = []
    for team_id, abbr, home, base in ((HOME_ID, home_abbr, True, 1000), (AWAY_ID, away_abbr, False, 2000)):
        for k in range(7):
            name = "%s %s%d" % (FIRST[k], abbr.title(), k + 1)
            rows.append({"sportvu_game": game, "player_id": base + k + 1, "player": name, "team_id": team_id,
                         "team": abbr, "home": home, "jersey": str(k + 1), "bref_slug": name.lower().replace(" ", "")[:9],
                         "bref_name": name, "pbp_name": name, "match_method": "exact"})
    return pd.DataFrame(rows)


def _fallback_synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
    rng = np.random.default_rng(seed)
    game = sportvu_game or _game_id(seed, heldout)
    roster = _roster(game, rng)
    skill = {int(p): float(v) for p, v in zip(roster["player_id"], rng.normal(0.0, 0.6, len(roster)))}
    home_ids = roster.loc[roster["home"], "player_id"].to_numpy()
    away_ids = roster.loc[~roster["home"], "player_id"].to_numpy()
    names = dict(zip(roster["player_id"].astype(int), roster["player"]))
    period_of = lambda p: 1 + (p * 4) // n_possessions
    clock = {q: PERIOD_S for q in range(1, 5)}
    frame_rows, events, shots, shot_frames, reb_frames, poss_rows, states = [], [], [], [], [], [], []
    frame0, ev, score = 0, 0, [0, 0]
    for p in range(n_possessions):
        q = period_of(p)
        off_home = p % 2 == 0
        off_team = HOME_ID if off_home else AWAY_ID
        basket = BASKETS[1] if (off_home == (q <= 2)) else BASKETS[0]
        bx, by = basket
        inward = 1.0 if bx < 47 else -1.0
        off_ids = np.sort(rng.choice(home_ids if off_home else away_ids, 5, replace=False))
        def_ids = np.sort(rng.choice(away_ids if off_home else home_ids, 5, replace=False))
        dur = int(rng.integers(6, 21))
        nf = dur * HZ
        if clock[q] - dur < 0:
            q = min(q + 1, 4)
        start_clock = clock[q]
        clocks = start_clock - np.arange(nf) / HZ
        # offensive paths: start around the basket, random walk
        r = rng.uniform(3, 28, 5)
        th = rng.uniform(-1.4, 1.4, 5)
        ox = bx + inward * r * np.cos(th)
        oy = by + r * np.sin(th)
        steps = rng.normal(0, 0.25, (nf, 5, 2)).cumsum(axis=0)
        off_xy = np.stack([ox, oy], axis=1)[None] + steps
        off_xy[..., 0] = np.clip(off_xy[..., 0], 0.5, 93.5)
        off_xy[..., 1] = np.clip(off_xy[..., 1], 0.5, 49.5)
        # defenders: assignment perm, gap toward the basket, noise; maybe one swap mid-possession
        perm = rng.permutation(5)
        assign = np.tile(perm, (nf, 1))
        if rng.uniform() < 0.3 and nf > 4 * HZ:
            at = int(rng.integers(2 * HZ, nf - 2 * HZ))
            a, b = rng.choice(5, 2, replace=False)
            assign[at:, a], assign[at:, b] = perm[b], perm[a]
        gap = rng.uniform(1.0, 6.0, 5)
        def_xy = np.empty((nf, 5, 2))
        for d in range(5):
            tgt = off_xy[np.arange(nf), assign[:, d]]
            v = np.stack([bx - tgt[:, 0], by - tgt[:, 1]], axis=1)
            norm = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-6)[:, None]
            def_xy[:, d] = tgt + v / norm * gap[d] + rng.normal(0, 1.0, (nf, 2))
        def_xy[..., 0] = np.clip(def_xy[..., 0], 0.5, 93.5)
        def_xy[..., 1] = np.clip(def_xy[..., 1], 0.5, 49.5)
        # outcome
        turnover = rng.uniform() < 0.12
        shooter_k = int(rng.integers(0, 5))
        release = nf - HZ
        sx, sy = off_xy[release, shooter_k]
        dist = float(np.hypot(sx - bx, sy - by))
        guard = int(np.flatnonzero(assign[release] == shooter_k)[0])
        def_dist = float(np.hypot(*(def_xy[release, guard] - off_xy[release, shooter_k])))
        three = dist >= 22.0
        pm = _sigmoid(1.1 - 0.06 * dist + 0.10 * def_dist - skill[int(def_ids[guard])])
        made = (not turnover) and bool(rng.uniform() < pm)
        points = (3 if three else 2) if made else 0
        ball = off_xy[:, shooter_k].copy()
        ball_z = np.full(nf, 4.0)
        ball_z[release:] = 4.0 + 8.0 * np.sin(np.linspace(0, np.pi, nf - release))
        frames = np.arange(frame0, frame0 + nf)
        for d in range(nf):
            for side, ids, xy in (("off", off_ids, off_xy[d]), ("def", def_ids, def_xy[d])):
                for k in range(5):
                    pid = int(ids[k])
                    frame_rows.append((frames[d], q, clocks[d], ball[d, 0], ball[d, 1], ball_z[d], pid, xy[k, 0], xy[k, 1]))
        end_clock = float(clocks[-1])
        shooter = int(off_ids[shooter_k])
        side = "home" if off_home else "away"
        start_event = ev
        if turnover:
            events.append((ev, q, end_clock, side, "Turnover by %s (bad pass)" % names[shooter], 0, "turnover", int(frames[-1])))
            end_event, end_kind = ev, "turnover"
            ev += 1
        else:
            verb = "makes" if made else "misses"
            desc = "%s %s %d-pt jump shot" % (names[shooter], verb, 3 if three else 2)
            if made:
                score[0 if side == "away" else 1] += points
            events.append((ev, q, end_clock, side, desc, points, "shot_made" if made else "shot_missed", int(frames[release])))
            shot_event = ev
            ev += 1
            loc_x = (sy - by) * 10.0 * inward
            loc_y = abs(sx - bx) * 10.0
            shots.append({"game": game, "sportvu_game": game, "period": q, "clock_s": end_clock, "made": made,
                          "loc_x": round(loc_x, 1), "loc_y": round(loc_y, 1), "shot_type": "3PT" if three else "2PT",
                          "shot_distance_ft": float(round(dist)), "court_x": sx, "court_y": sy,
                          "split": "heldout_game" if heldout else None, "heldout": heldout, "source": "bref",
                          "event_id": shot_event, "player_id": shooter})
            shot_frames.append({"sportvu_game": game, "event": shot_event, "period": q, "clock_s": end_clock, "made": made,
                                "pbp_shooter_name": names[shooter], "shooter_id": shooter, "release_frame": int(frames[release]),
                                "apex_frame": int(frames[release + HZ // 2]), "outcome_frame": int(frames[-1]),
                                "shooter_at_release_id": shooter, "identity_match": True, "confidence": 0.9,
                                "confidence_flag": "high"})
            if made:
                end_event, end_kind = shot_event, "shot_made"
            else:
                rebounder = int(def_ids[int(rng.integers(0, 5))])
                events.append((ev, q, end_clock, "away" if side == "home" else "home",
                               "Defensive rebound by %s" % names[rebounder], 0, "rebound", int(frames[-1])))
                reb_frames.append({"sportvu_game": game, "event": shot_event, "rebound_event": ev, "period": q,
                                   "clock_s": end_clock, "rim_frame": int(frames[-1]), "rebound_frame": int(frames[-1]),
                                   "rebounder_id": rebounder, "pbp_rebounder_name": names[rebounder],
                                   "pbp_rebounder_id": rebounder, "identity_match": True, "confidence": 0.8,
                                   "confidence_flag": "high"})
                end_event, end_kind = ev, "shot_missed"
                ev += 1
        poss_rows.append({"sportvu_game": game, "possession": p, "period": q, "start_frame": int(frames[0]),
                          "end_frame": int(frames[-1]), "start_clock_s": float(start_clock), "end_clock_s": end_clock,
                          "start_event": start_event, "end_event": end_event, "offense_team_id": off_team,
                          "end_kind": end_kind, "points": points, "gap": False, "n_frames": nf})
        # states at 1 Hz
        for s in range(dur):
            d = s * HZ
            off_sorted = np.argsort(off_ids)
            def_sorted = np.argsort(def_ids)
            handler = int(off_ids[shooter_k])
            hx, hy = off_xy[d, shooter_k]
            dd = float(np.min(np.hypot(def_xy[d, :, 0] - hx, def_xy[d, :, 1] - hy)))
            row = {"sportvu_game": game, "possession": p, "frame": int(frames[d]), "period": q,
                   "game_clock_s": float(clocks[d]), "shot_clock_s": float(max(24.0 - s, 0.0)),
                   "elapsed_in_possession_s": float(s), "offense_team_id": off_team, "handler_id": handler,
                   "ball_x": float(ball[d, 0]), "ball_y": float(ball[d, 1]), "ball_z": float(ball_z[d])}
            for i, k in enumerate(off_sorted):
                row["off%d_id" % (i + 1)] = int(off_ids[k])
                row["off%d_x" % (i + 1)], row["off%d_y" % (i + 1)] = float(off_xy[d, k, 0]), float(off_xy[d, k, 1])
            for i, k in enumerate(def_sorted):
                row["def%d_id" % (i + 1)] = int(def_ids[k])
                row["def%d_x" % (i + 1)], row["def%d_y" % (i + 1)] = float(def_xy[d, k, 0]), float(def_xy[d, k, 1])
            row["handler_nearest_def_dist_ft"] = dd
            row["points_to_end"] = points
            row["possession_end_kind"] = end_kind
            states.append(row)
        clock[q] = end_clock - 1.0 / HZ
        frame0 += nf
    fr = pd.DataFrame(frame_rows, columns=["frame", "period", "game_clock_s", "ball_x", "ball_y", "ball_z", "player_id", "x", "y"])
    fr = fr.merge(roster[["player_id", "team_id", "team", "home", "player", "jersey"]], on="player_id", how="left")
    fr.insert(0, "game", game)
    fr.insert(1, "gameid", "0021500%03d" % (seed % 1000))
    fr.insert(2, "date", "2016-01-%02d" % (seed % 28 + 1))
    fr["ts_ms"] = (fr["frame"] * (1000 // HZ)).astype("int64")
    fr["shot_clock_s"] = np.float32(np.nan)
    fr["n_players"] = np.int8(10)
    fr = fr[["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
             "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]]
    fr = fr.astype({"period": "int8", "frame": "int32", "game_clock_s": "float32", "ball_x": "float32", "ball_y": "float32",
                    "ball_z": "float32", "team_id": "int32", "player_id": "int32", "x": "float32", "y": "float32"})
    ev_df = pd.DataFrame(events, columns=["event", "period", "clock_s", "team", "desc", "points", "kind", "frame"])
    ev_df.insert(0, "game", game)
    ev_df["score"] = None
    ev_df["frame_clock_diff_s"] = 0.0
    ev_df["source"], ev_df["endpoint"], ev_df["season"], ev_df["fetched_at"] = "synthetic", "fixture", "2015-16", "2026-01-01T00:00:00Z"
    return {"frames": fr, "events": ev_df, "shots": pd.DataFrame(shots), "rosters": roster,
            "shot_frames": pd.DataFrame(shot_frames), "rebound_frames": pd.DataFrame(reb_frames),
            "possessions": pd.DataFrame(poss_rows), "states_1hz": pd.DataFrame(states)}


def _fallback_synth_season(seed: int, n_games: int = 8, heldout_every: int = 0) -> list:
    out = []
    for i in range(n_games):
        held = bool(heldout_every) and (i + 1) % heldout_every == 0
        out.append(_fallback_synth_game(seed * 100 + i, heldout=held))
    return out


def _fallback_write_fixture_root(root, seed: int, n_games: int = 8, heldout_every: int = 0, season: str = "2015-16") -> Path:
    """DATA_ROOT layout: sportvu_contract/<game>.parquet, <game>_pbp.parquet (held-out under heldout/),
    analytics/<table>/<season>/<game>.parquet (held-out under analytics/<table>/heldout/<season>/)."""
    root = Path(root)
    for g in _fallback_synth_season(seed, n_games, heldout_every):
        game = g["possessions"]["sportvu_game"].iloc[0]
        held = any(t in game for t in HELD_IDS)
        sv = root / "sportvu_contract" / ("heldout" if held else "")
        sv.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(sv / (game + ".parquet"), index=False)
        g["events"].to_parquet(sv / (game + "_pbp.parquet"), index=False)
        for table in ("possessions", "shot_frames", "rebound_frames", "rosters", "states_1hz"):
            d = root / "analytics" / table / ("heldout" if held else "") / season
            d.mkdir(parents=True, exist_ok=True)
            g[table].to_parquet(d / (game + ".parquet"), index=False)
    return root


if not SHARED:
    synth_game = _fallback_synth_game
    synth_season = _fallback_synth_season
    write_fixture_root = _fallback_write_fixture_root

__all__ = ["synth_game", "synth_season", "write_fixture_root", "SHARED", "HZ", "HELD_IDS"]

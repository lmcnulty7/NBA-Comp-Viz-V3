"""tests/_fx_phase0.py: the fixture DATA_ROOT for tests/test_run_phase0.py.

Prefers the shared writer of analytics/fixtures.py (lane I1) and falls back to a minimal local writer with
the same signature (``write_fixture_root_fallback``, always defined) when that module is absent or not
importable yet. ``fixture_data_root(root)`` gives the Step 1 folder either writer produced (the shared one
writes it under <root>/sportvu_contract). Both write the Step 1 layout of sportvu/colab_contract.py:

    <game>.parquet, <game>_pbp.parquet          training games (home GSW, split None or train)
    heldout/<game>.parquet, heldout/<game>_pbp.parquet   held-out games (home CLE: sportvu/splits.json heldout_arena)
    manifest.json                               per game: status, date, away, home, split, heldout,
                                                gaps_over_2s, pbp_joined (the Step 1 shape)

The fallback game is one 25 Hz period of 1500 frames with four static players (two per team) and six
scripted shots: the ball is carried at the shooter's feet, released at a scripted frame, flies on a parabola
to a basket (apex 25 frames after release, through rim height 22 frames after the apex) and, on a miss,
travels to a defensive rebounder who holds it. The script follows tests/test_shot_frames.py and
tests/test_rebound_frames.py, so every shot has a release frame and every miss a rebound frame. Nothing
here is real tracking data.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if True:                                         # the local fallback, always defined (write_fixture_root_fallback)
    HZ, N_FRAMES, PERIOD_S = 25, 1500, 720.0
    RELEASES = (150, 350, 550, 750, 950, 1150)   # release frame of each scripted shot
    FLIGHT, APEX_AFTER = 48, 25                  # frames in flight; apex this many frames after release
    OUTCOME_AFTER, REBOUND_AFTER = 50, 58        # the logged clock marks the outcome; the rebound event follows
    PARK = (47.0, 25.0, 5.0)                     # the ball between shots: mid court, nobody near it
    BASKET_HOME_END, BASKET_AWAY_END = (5.25, 25.0), (88.75, 25.0)
    AWAY_CODES = ("BKN", "SAC", "IND", "CHA", "PHX", "NYK", "DEN", "LAL", "MIA", "TOR")
    FRAME_COLUMNS = ["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s",
                     "n_players", "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player",
                     "jersey", "x", "y"]
    PBP_COLUMNS = ["game", "event", "period", "clock_s", "team", "desc", "points", "score", "kind", "frame",
                   "frame_clock_diff_s"]

    def _players(home_code: str, away_code: str, game_index: int) -> list[dict]:
        """Two players per side with distinct last names; positions far from the ball paths and from each other."""
        home_names = {"GSW": ("Stephen Curry", "Klay Thompson"), "CLE": ("LeBron James", "Kevin Love")}
        hn = home_names.get(home_code, ("Alan Smith", "Bob Jones"))
        an = ("Eric Bledsoe", "Brandon Knight") if game_index % 2 == 0 else ("Carl Brown", "Dan Walker")
        base = 300000 + game_index * 100
        return [
            {"player_id": 201939 if home_code == "GSW" else base + 1, "player": hn[0], "team": home_code,
             "team_id": 1610612744 if home_code == "GSW" else 1610612739, "home": True, "jersey": "30", "x": 20.0, "y": 12.0},
            {"player_id": 202691 if home_code == "GSW" else base + 2, "player": hn[1], "team": home_code,
             "team_id": 1610612744 if home_code == "GSW" else 1610612739, "home": True, "jersey": "11", "x": 24.0, "y": 38.0},
            {"player_id": base + 3, "player": an[0], "team": away_code, "team_id": 1610612700 + game_index,
             "home": False, "jersey": "2", "x": 74.0, "y": 12.0},
            {"player_id": base + 4, "player": an[1], "team": away_code, "team_id": 1610612700 + game_index,
             "home": False, "jersey": "3", "x": 70.0, "y": 38.0},
        ]

    def _initial_last(name: str) -> str:
        first, last = name.split(" ", 1)
        return "%s. %s" % (first[0], last)

    def _clock_fn(jump_frame: int | None, jump_s: float):
        def clock_of(f: int) -> float:
            c = PERIOD_S - f / HZ
            if jump_frame is not None and f >= jump_frame:
                c -= jump_s
            return c
        return clock_of

    def _script_game(game: str, players: list[dict], rng: np.random.Generator, jump_frame, jump_s):
        """(frames DataFrame, events DataFrame, final score (away, home))."""
        clock_of = _clock_fn(jump_frame, jump_s)
        bx = np.full(N_FRAMES, PARK[0]); by = np.full(N_FRAMES, PARK[1]); bz = np.full(N_FRAMES, PARK[2])
        events = []
        score = {"away": 0, "home": 0}
        events.append((clock_of(0), None, "Jump ball: tip to nobody", 0, "other", None))
        for k, rel in enumerate(RELEASES):
            side = "home" if k % 2 == 0 else "away"
            shooter = players[(k // 2) % 2] if side == "home" else players[2 + (k // 2) % 2]
            basket = BASKET_HOME_END if side == "home" else BASKET_AWAY_END
            made = bool(rng.random() < 0.55)
            three = k % 3 == 0
            px, py = shooter["x"], shooter["y"]
            for f in range(rel - 50, rel):                            # carried at the shooter's feet
                bx[f], by[f], bz[f] = px, py, 4.0
            apex = rel + APEX_AFTER
            for j in range(FLIGHT):                                   # the flight: 7.5 ft at release, 18 at the apex
                f, t = rel + j, j / (FLIGHT - 1)
                bx[f], by[f] = px + (basket[0] - px) * t, py + (basket[1] - py) * t
                bz[f] = 18.0 - 10.5 * ((f - apex) / 25.0) ** 2
            end = rel + FLIGHT
            if made:
                for f in range(end, end + 30):                        # through the net, then parked again
                    bx[f], by[f], bz[f] = basket[0], basket[1], 2.0
            else:
                reb = players[2 + k % 2] if side == "home" else players[k % 2]   # a defender takes it
                for j in range(10):                                   # to the rebounder in 10 frames
                    f, t = end + j, (j + 1) / 10.0
                    bx[f], by[f] = basket[0] + (reb["x"] - basket[0]) * t, basket[1] + (reb["y"] - basket[1]) * t
                    bz[f] = 2.0
                for f in range(end + 10, end + 30):
                    bx[f], by[f], bz[f] = reb["x"], reb["y"], 2.0
            pts = (3 if three else 2) if made else 0
            verb = "makes" if made else "misses"
            desc = "%s %s %d-pt jump shot from %d ft" % (_initial_last(shooter["player"]), verb, 3 if three else 2,
                                                         26 if three else 15)
            if made:
                score[side] += pts
            events.append((clock_of(rel + OUTCOME_AFTER), side, desc, pts, "shot_made" if made else "shot_missed",
                           "%d-%d" % (score["away"], score["home"]) if made else None))
            if not made:
                other = "away" if side == "home" else "home"
                events.append((clock_of(rel + REBOUND_AFTER), other, "Defensive rebound by %s" % _initial_last(reb["player"]),
                               0, "rebound", None))
            if k == 2:                                                # one turnover between shots 3 and 4
                events.append((clock_of(rel + 110), "away", "Turnover by %s (bad pass)" % _initial_last(players[2]["player"]),
                               0, "turnover", None))
        events.append((clock_of(N_FRAMES - 1), None, "End of 1st quarter", 0, "other", None))
        clocks = np.array([clock_of(f) for f in range(N_FRAMES)], dtype=float)
        rows = []
        for f in range(N_FRAMES):
            for p in players:
                rows.append({"game": game, "gameid": "0021500000", "date": None, "period": 1, "frame": f,
                             "ts_ms": 1_446_000_000_000 + f * 40, "game_clock_s": clocks[f], "shot_clock_s": None,
                             "n_players": len(players), "ball_x": bx[f], "ball_y": by[f], "ball_z": bz[f],
                             "team_id": p["team_id"], "team": p["team"], "home": p["home"], "player_id": p["player_id"],
                             "player": p["player"], "jersey": p["jersey"], "x": p["x"], "y": p["y"]})
        frames = pd.DataFrame(rows, columns=FRAME_COLUMNS)
        ev = pd.DataFrame(events, columns=["clock_s", "team", "desc", "points", "kind", "score"])
        ev.insert(0, "event", np.arange(len(ev)))
        ev.insert(0, "game", game)
        ev["period"] = 1
        near = np.array([int(np.argmin(np.abs(clocks - c))) for c in ev["clock_s"]])
        diff = np.abs(clocks[near] - ev["clock_s"].to_numpy(dtype=float))
        ev["frame"] = np.where(diff <= 2.0, near, -1)
        ev["frame_clock_diff_s"] = np.where(diff <= 2.0, np.round(diff, 3), np.nan)
        return frames, ev[PBP_COLUMNS], (score["away"], score["home"])

    def _frames_table(frames: pd.DataFrame, date_str: str) -> pa.Table:
        frames = frames.assign(date=date_str)
        schema = pa.schema([
            ("game", pa.string()), ("gameid", pa.string()), ("date", pa.string()), ("period", pa.int8()),
            ("frame", pa.int32()), ("ts_ms", pa.int64()), ("game_clock_s", pa.float32()), ("shot_clock_s", pa.float32()),
            ("n_players", pa.int8()), ("ball_x", pa.float32()), ("ball_y", pa.float32()), ("ball_z", pa.float32()),
            ("team_id", pa.int32()), ("team", pa.string()), ("home", pa.bool_()), ("player_id", pa.int32()),
            ("player", pa.string()), ("jersey", pa.string()), ("x", pa.float32()), ("y", pa.float32())])
        return pa.Table.from_pandas(frames, schema=schema, preserve_index=False)

    def _pbp_table(ev: pd.DataFrame) -> pa.Table:
        schema = pa.schema([
            ("game", pa.string()), ("event", pa.int32()), ("period", pa.int8()), ("clock_s", pa.float32()),
            ("team", pa.string()), ("desc", pa.string()), ("points", pa.int8()), ("score", pa.string()),
            ("kind", pa.string()), ("frame", pa.int32()), ("frame_clock_diff_s", pa.float32())])
        ev = ev.copy()
        ev["team"] = ev["team"].astype(object).where(ev["team"].notna(), None)
        ev["score"] = ev["score"].astype(object).where(ev["score"].notna(), None)
        return pa.Table.from_pandas(ev, schema=schema, preserve_index=False)

    def write_fixture_root_fallback(root, seed: int = 0, n_games: int = 6, heldout_every: int = 3) -> dict:
        """Write the Step 1 layout under ``root`` and return the manifest. Game i is held out when
        (i + 1) % heldout_every == 0 (home CLE, the held-out arena of sportvu/splits.json); the others are at GSW.
        Games with i % 3 == 1 carry a 12 s clock jump late in the period, i % 3 == 2 a 3 s jump between the
        second and third shots; both are recorded in gaps_over_2s."""
        root = Path(root)
        (root / "heldout").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        man = {"contract_version": 1, "contract": "docs/DATA_CONTRACT.md", "source": "tests/_fx_phase0.py synthetic",
               "coverage": ["2015-11-01", "2016-01-23"], "heldout_rule": "games under heldout/ are held out",
               "heldout": [], "games": {}}
        d0 = date(2015, 11, 1)
        for i in range(n_games):
            heldout = heldout_every > 0 and (i + 1) % heldout_every == 0
            d = d0 + timedelta(days=3 * i)
            away = AWAY_CODES[i % len(AWAY_CODES)]
            home = "CLE" if heldout else "GSW"
            game = "%02d.%02d.%d.%s.at.%s" % (d.month, d.day, d.year, away, home)
            jump_frame, jump_s = (1300, 12.0) if i % 3 == 1 else ((450, 3.0) if i % 3 == 2 else (None, 0.0))
            players = _players(home, away, i)
            frames, ev, final = _script_game(game, players, rng, jump_frame, jump_s)
            clock_of = _clock_fn(jump_frame, jump_s)
            gaps = {"count": 0, "seconds": 0.0, "by_quarter": {}}
            if jump_frame is not None:
                a, b = round(clock_of(jump_frame - 1), 3), round(clock_of(jump_frame), 3)
                gaps = {"count": 1, "seconds": round(a - b, 1), "by_quarter": {"1": [[a, b]]}}
            d_out = root / "heldout" if heldout else root
            fp, pp = d_out / (game + ".parquet"), d_out / (game + "_pbp.parquet")
            pq.write_table(_frames_table(frames, d.isoformat()), fp)
            pq.write_table(_pbp_table(ev), pp)
            man["games"][game] = {
                "game": game, "date": d.isoformat(), "away": away, "home": home,
                "split": "heldout_arena" if heldout else None, "heldout": heldout, "source_bytes": None,
                "status": "ok", "error": None, "moments": N_FRAMES, "frames_rows": int(len(frames)),
                "gaps_over_2s": gaps, "pbp_joined": True,
                "pbp": {"code": "synthetic", "events": int(len(ev)), "matched": int((ev["frame"] >= 0).sum()), "error": None},
                "parquet": str(fp.relative_to(root)), "parquet_bytes": fp.stat().st_size,
                "pbp_parquet": str(pp.relative_to(root)), "final_score": {"away": final[0], "home": final[1]},
                "seconds": 0.0, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        man["heldout"] = sorted(g for g, e in man["games"].items() if e["heldout"])
        man["counts"] = {"games": n_games, "ok": n_games, "failed": 0, "source_broken": 0,
                         "pbp_joined": n_games, "heldout": len(man["heldout"])}
        (root / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        return man


try:                                             # the shared writer of lane I1, when it is in place
    from analytics.fixtures import write_fixture_root   # noqa: F401
except Exception:                                # noqa: BLE001 (absent or half-written: use the fallback)
    write_fixture_root = write_fixture_root_fallback


def fixture_data_root(root) -> Path:
    """The DATA_ROOT inside what write_fixture_root wrote: the shared writer puts the Step 1 layout under
    <root>/sportvu_contract (with labels/ beside it), the fallback writes it at <root> itself."""
    root = Path(root)
    sub = root / "sportvu_contract"
    return sub if (sub / "manifest.json").exists() and not (root / "manifest.json").exists() else root


__all__ = ["write_fixture_root", "write_fixture_root_fallback", "fixture_data_root"]

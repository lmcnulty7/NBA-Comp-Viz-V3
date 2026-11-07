"""tests/_fx_labels.py: synthetic frames and possessions for the label tool tests.

Prefers lane I1's shared fixtures (``analytics.fixtures.synth_game``) when they import and return frames
with the section 4 columns; otherwise a minimal local fallback with the same call shape:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict(frames, events, possessions, ...)

The fallback is 25 Hz, two teams of five, one 12 s possession after another in period 1. Possession i is a
half-court possession (the ball crosses midcourt and dwells) unless i % 7 == 6 (the ball stays in the
front court: not sampled); every possession with i % 3 == 0 is a screen candidate (offensive player 2 within
2 ft of the handler's nearest defender for 2 s). ``plant_screen`` and ``plant_no_crossing`` rewrite a span
of any frames table so a test controls the geometry no matter which fixture produced it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HZ = 25
POSS_S = 12.0
POSS_FRAMES = int(POSS_S * HZ)
HOME_ID, AWAY_ID = 1610612744, 1610612751
HOME_PLAYERS = [(101, "Home One", "1"), (102, "Home Two", "2"), (103, "Home Three", "3"),
                (104, "Home Four", "4"), (105, "Home Five", "5")]
AWAY_PLAYERS = [(201, "Away One", "11"), (202, "Away Two", "12"), (203, "Away Three", "13"),
                (204, "Away Four", "14"), (205, "Away Five", "15")]
TRAIN_GAME = "01.01.2016.SYN.at.GSW"
HELDOUT_GAME = "02.02.2016.HLD.at.GSW"
HELDOUT_TOKENS = {HELDOUT_GAME, "HLD"}
REQUIRED = {"game", "frame", "period", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "team", "home",
            "player_id", "player", "jersey", "x", "y"}


def _possession_layout(i: int, seed: int):
    """Per-possession plan: offence side, whether the ball crosses, whether a screen is planted."""
    rng = np.random.RandomState(seed * 1000 + i)
    offense_home = (i % 2 == 0)
    crossing = (i % 7 != 6)
    screen = (i % 3 == 0)
    jitter = rng.uniform(-1.5, 1.5, size=(10, 2))
    return offense_home, crossing, screen, jitter


def _fallback_synth_game(seed: int = 0, n_possessions: int = 40, heldout: bool = False,
                         sportvu_game: str | None = None) -> dict:
    game = sportvu_game or (HELDOUT_GAME if heldout else TRAIN_GAME)
    n_frames = n_possessions * POSS_FRAMES
    period_s = n_frames / HZ + 1.0
    rows = []
    poss_rows = []
    events = []
    for i in range(n_possessions):
        offense_home, crossing, screen, jitter = _possession_layout(i, seed)
        f0 = i * POSS_FRAMES
        off_players = HOME_PLAYERS if offense_home else AWAY_PLAYERS
        def_players = AWAY_PLAYERS if offense_home else HOME_PLAYERS
        off_team = HOME_ID if offense_home else AWAY_ID
        def_team = AWAY_ID if offense_home else HOME_ID
        # home attacks the right basket (x near 94), away the left basket
        front = 1.0 if offense_home else -1.0
        for k in range(POSS_FRAMES):
            f = f0 + k
            t = k / HZ
            clock = np.float32(period_s - f / HZ)
            if crossing:
                # back court for 2 s, cross in 1 s, dwell in the front court
                if t < 2.0:
                    bx = 47 - front * 20
                elif t < 3.0:
                    bx = 47 - front * 20 + front * 40 * (t - 2.0)
                else:
                    bx = 47 + front * 20 + front * 2 * np.sin(t)
            else:
                bx = 47 + front * 20 + front * 2 * np.sin(t)
            by = 25 + 4 * np.sin(t * 0.7)
            # handler = off player 0 on the ball; the others spread in the half court the ball is in
            spread = [(0.0, 0.0), (-8.0, 10.0), (-8.0, -10.0), (-16.0, 6.0), (-16.0, -6.0)]
            offs = []
            for j, (pid, name, jersey) in enumerate(off_players):
                dx, dy = spread[j]
                px = bx + front * dx + jitter[j, 0]
                py = by + dy + jitter[j, 1]
                offs.append((pid, name, jersey, px, py))
            defs = []
            for j, (pid, name, jersey) in enumerate(def_players):
                ox, oy = offs[j][3], offs[j][4]
                # defender 3 ft toward the basket from the player he guards
                px = ox + front * 3.0 + jitter[5 + j, 0] * 0.2
                py = oy + jitter[5 + j, 1] * 0.2
                defs.append((pid, name, jersey, px, py))
            if screen and 4.0 <= t < 6.5:
                # off player 1 steps to 1.5 ft from the handler's defender for 2.5 s (candidate rule: 4 ft, 1 s)
                hx, hy = defs[0][3], defs[0][4]
                p1 = offs[1]
                offs[1] = (p1[0], p1[1], p1[2], hx + 1.0, hy + 1.0)
            home_first = (HOME_PLAYERS is off_players)
            ordered = (offs, defs) if home_first else (defs, offs)
            for group, tid, is_home in ((ordered[0], HOME_ID, True), (ordered[1], AWAY_ID, False)):
                tname = "GSW" if is_home else "SYN"
                for pid, name, jersey, px, py in group:
                    rows.append((game, "0021500000", "2016-01-01", 1, f, int(f * 40), clock, np.float32(24.0 - (t % 24.0)),
                                 10, np.float32(bx), np.float32(by), np.float32(4.0), tid, tname, is_home, pid, name,
                                 jersey, np.float32(px), np.float32(py)))
        poss_rows.append({"sportvu_game": game, "possession": i, "period": 1, "start_frame": f0,
                          "end_frame": f0 + POSS_FRAMES - 1, "start_clock_s": np.float32(period_s - f0 / HZ),
                          "end_clock_s": np.float32(period_s - (f0 + POSS_FRAMES - 1) / HZ), "start_event": i,
                          "end_event": i + 1, "offense_team_id": off_team, "end_kind": "shot_missed", "points": 0,
                          "gap": False, "n_frames": POSS_FRAMES})
        events.append({"event": i, "period": 1, "clock_s": float(period_s - f0 / HZ), "team": "home" if offense_home else "away",
                       "desc": "synthetic possession %d" % i, "points": 0, "kind": "shot_missed", "frame": f0})
    cols = ["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
            "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]
    frames = pd.DataFrame(rows, columns=cols)
    frames["period"] = frames["period"].astype("int8")
    frames["frame"] = frames["frame"].astype("int32")
    frames["team_id"] = frames["team_id"].astype("int32")
    frames["player_id"] = frames["player_id"].astype("int32")
    frames["n_players"] = frames["n_players"].astype("int8")
    possessions = pd.DataFrame(poss_rows)
    possessions["possession"] = possessions["possession"].astype("int32")
    possessions["period"] = possessions["period"].astype("int8")
    possessions["start_frame"] = possessions["start_frame"].astype("int32")
    possessions["end_frame"] = possessions["end_frame"].astype("int32")
    possessions["offense_team_id"] = possessions["offense_team_id"].astype("Int32")
    possessions["points"] = possessions["points"].astype("int16")
    possessions["n_frames"] = possessions["n_frames"].astype("int32")
    return {"frames": frames, "events": pd.DataFrame(events), "possessions": possessions, "sportvu_game": game,
            "heldout": heldout, "fps": HZ, "home_team_id": HOME_ID, "away_team_id": AWAY_ID, "source": "fallback"}


def _shared_ok(out) -> bool:
    try:
        fr, po = out["frames"], out["possessions"]
    except Exception:
        return False
    if not isinstance(fr, pd.DataFrame) or not isinstance(po, pd.DataFrame):
        return False
    if not REQUIRED <= set(fr.columns):
        return False
    need = {"sportvu_game", "possession", "start_frame", "end_frame", "offense_team_id"}
    return need <= set(po.columns) and len(po) > 0


def synth_game(seed: int = 0, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
    """Lane I1's shared synthetic game when it is importable and complete, else the local fallback
    (LABELS_FX_FALLBACK=1 in the environment forces the fallback)."""
    try:
        if os.environ.get("LABELS_FX_FALLBACK"):
            raise ImportError("fallback forced")
        from analytics import fixtures as _fx  # noqa: F401
        out = _fx.synth_game(seed, n_possessions=n_possessions, heldout=heldout, sportvu_game=sportvu_game)
        if _shared_ok(out):
            out.setdefault("source", "shared")
            return out
    except Exception:
        pass
    return _fallback_synth_game(seed, n_possessions=n_possessions, heldout=heldout, sportvu_game=sportvu_game)


def shared_truth_labels(game: dict):
    """Lane I1's truth labels of a shared synth_game, normalised to analytics.labels (action_id = the
    0-based action row the shared screens reference, screen_id = 1.., labelled_at filled); None when the
    shared fixture is not importable or ``game`` did not come from it."""
    if game.get("source") != "shared":
        return None
    try:
        from analytics import fixtures as _fx
        out = _fx.synth_labels(0, game)
    except Exception:
        return None
    acts = out["actions"].reset_index(drop=True).copy()
    scrs = out["screens"].reset_index(drop=True).copy()
    if "action_id" not in acts.columns:
        acts["action_id"] = np.arange(len(acts), dtype="int32")
    if "screen_id" not in scrs.columns:
        scrs["screen_id"] = np.arange(1, len(scrs) + 1, dtype="int32")
    for df in (acts, scrs):
        if "labelled_at" not in df.columns:
            df["labelled_at"] = "2026-01-01T00:00:00Z"
        if "labeller_pass" not in df.columns:
            df["labeller_pass"] = 1
    return {"actions": acts, "screens": scrs}


def synth_labels(seed: int, game: dict) -> dict:
    """A small valid label set for ``game`` in the exact analytics.labels shape: one ball_screen with a
    coverage on possession 0, one drive on possession 1. Always local: the shared fixture's truth labels
    have their own shape (see ``shared_truth_labels``)."""
    po = game["possessions"]
    fr = game["frames"]
    g = str(po["sportvu_game"].iloc[0])

    def players(p):
        r = po.iloc[p]
        span = fr.loc[(fr["frame"] >= int(r.start_frame)) & (fr["frame"] <= int(r.end_frame))]
        first = span.loc[span["frame"] == int(r.start_frame)]
        off = first.loc[first["team_id"] == int(r.offense_team_id), "player_id"].astype(int).tolist()
        de = first.loc[first["team_id"] != int(r.offense_team_id), "player_id"].astype(int).tolist()
        return int(r.start_frame), int(r.end_frame), off, de

    s0, e0, off0, def0 = players(0)
    s1, e1, off1, def1 = players(1)
    actions = pd.DataFrame([
        {"sportvu_game": g, "possession": 0, "action_id": 1, "action_type": "ball_screen", "start_frame": s0 + 100,
         "end_frame": s0 + 160, "ball_handler_id": off0[0], "screener_id": off0[1], "label_source": "synthetic",
         "labeller_pass": 1, "labelled_at": "2026-01-01T00:00:00Z"},
        {"sportvu_game": g, "possession": 1, "action_id": 2, "action_type": "drive", "start_frame": s1 + 120,
         "end_frame": s1 + 170, "ball_handler_id": off1[0], "screener_id": None, "label_source": "synthetic",
         "labeller_pass": 1, "labelled_at": "2026-01-01T00:00:00Z"},
    ])
    screens = pd.DataFrame([
        {"sportvu_game": g, "possession": 0, "screen_id": 1, "action_id": 1, "coverage": "drop",
         "on_ball_defender_id": def0[0], "screener_defender_id": def0[1], "label_source": "synthetic",
         "confidence": "sure", "labeller_pass": 1, "labelled_at": "2026-01-01T00:00:00Z"},
    ])
    return {"actions": actions, "screens": screens}


def plant_screen(frames: pd.DataFrame, start_frame: int, end_frame: int, offense_team_id: int, present: bool) -> None:
    """Rewrite the span in place: handler on the ball, defenders 3 ft off; when ``present`` the second
    offensive player sits 1 ft from the handler's defender for the whole span, else 12 ft away."""
    m = (frames["frame"] >= start_frame) & (frames["frame"] <= end_frame)
    idx = frames.index[m]
    sub = frames.loc[idx]
    bx = sub["ball_x"].to_numpy(dtype="float64")
    by = sub["ball_y"].to_numpy(dtype="float64")
    off = (sub["team_id"].to_numpy() == int(offense_team_id))
    # rank of each row among its frame's offensive (or defensive) rows
    rank = sub.groupby([sub["frame"].to_numpy(), off]).cumcount().to_numpy()
    r_off = np.where(off, rank, -1)
    r_def = np.where(~off, rank, -1)
    x = sub["x"].to_numpy(dtype="float64").copy()
    y = sub["y"].to_numpy(dtype="float64").copy()
    x[r_off == 0], y[r_off == 0] = bx[r_off == 0], by[r_off == 0]
    x[r_def == 0], y[r_def == 0] = bx[r_def == 0] + 3.0, by[r_def == 0]
    d = 1.0 if present else 12.0
    x[r_off == 1], y[r_off == 1] = bx[r_off == 1] + 3.0, by[r_off == 1] + d
    for k in (2, 3, 4):
        x[r_off == k], y[r_off == k] = bx[r_off == k] - 6.0 * k, by[r_off == k] + 15.0
        x[r_def == k], y[r_def == k] = bx[r_def == k] - 6.0 * k + 3.0, by[r_def == k] + 15.0
    x[r_def == 1], y[r_def == 1] = bx[r_def == 1] + 6.0, by[r_def == 1] + d + 3.0
    frames.loc[idx, "x"] = x.astype("float32")
    frames.loc[idx, "y"] = y.astype("float32")


def plant_no_crossing(frames: pd.DataFrame, start_frame: int, end_frame: int) -> None:
    """Keep the ball in one half for the whole span (a transition or back-court possession)."""
    m = (frames["frame"] >= start_frame) & (frames["frame"] <= end_frame)
    frames.loc[m, "ball_x"] = np.float32(70.0)

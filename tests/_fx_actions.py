"""tests/_fx_actions.py: fixtures for the action-recognition lane (analytics/actions).

Prefers the shared generator ``analytics.fixtures`` (lane I1) when it is importable and falls back to a
minimal local generator with the same signatures and the same column names:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None)
        -> dict(frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of game dicts (each also carries sportvu_game, heldout)
    synth_labels(seed, game) -> dict(actions, screens)   (game is the dict synth_game returned)
    write_fixture_root(root, seed, n_games, heldout_every) -> DATA_ROOT layout on disk

The fallback scripts every possession deterministically from the seed, so the labels it returns describe
motion that is really in the frames:
    ball_screen   a teammate walks to the handler's defender, stands within about 2 ft of him for over a
                  second while the handler dribbles past the screen and the defender trails (recovery)
    drive         the handler goes straight at the basket at about 11 ft per s for 1.2 s, defender chasing
    isolation     the handler holds the ball for the whole possession with every teammate over 10 ft away
    none          no labelled action; half of these carry a slow cutter (2 ft per s) who passes within 4 ft of
                  the handler's defender for about 3 s (a hard negative for the screen rule: he keeps moving
                  and the handler stays put), the other half a fast cutter (7 ft per s, no candidate)
Frames are 25 Hz with the section 4 columns, possessions follow section 19, rosters section 16; the shots,
shot_frames, rebound_frames and states_1hz tables are empty contract-shaped frames (this lane never reads
them). Every value is a float32 or an int32 as the contract says. Test adapters at the bottom
(``season_games``, ``labels_for``) normalise either generator to one shape for tests/test_actions.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # the shared generator of lane I1, when it exists
    from analytics.fixtures import synth_game, synth_season, synth_labels, write_fixture_root  # type: ignore  # noqa: F401
    SHARED = True
except ImportError:
    SHARED = False

from analytics import schema  # noqa: E402

HZ = 25
HOME_ID, AWAY_ID = 1610612744, 1610612751      # GSW, BKN ids
HOME, AWAY = "GSW", "BKN"
COURT_X, COURT_Y = 94.0, 50.0
BASKET_LEFT, BASKET_RIGHT = np.array([5.25, 25.0]), np.array([88.75, 25.0])
HELDOUT_GAME = "12.16.2015.PHX.at.GSW"
SEASON = "2015-16"
COVERAGES = ("over", "under", "switch")
ACTION_MIX = (("ball_screen", 0.35), ("drive", 0.20), ("isolation", 0.15), ("none", 0.30))


def _game_id(i: int, heldout: bool) -> str:
    if heldout:
        return HELDOUT_GAME if i == 0 else "12.%02d.2015.AAA.at.CLE" % (1 + i % 28)
    return "%02d.%02d.2016.AAA.at.BBB" % (1 + i // 28, 1 + i % 28)


# ── scripts ────────────────────────────────────────────────────────────────────────────────────────
def _keyframes_to_track(keys, t):
    """Piecewise-linear interpolation of [(t, x, y), ...] keyframes onto the sample times t."""
    kt = np.array([k[0] for k in keys], dtype=float)
    kx = np.array([k[1] for k in keys], dtype=float)
    ky = np.array([k[2] for k in keys], dtype=float)
    return np.stack([np.interp(t, kt, kx), np.interp(t, kt, ky)], axis=1)


def _script_possession(rng, kind: str) -> dict:
    """Keyframes in a basket-centred frame (out = away from the basket toward midcourt, lat = sideline
    direction) for 5 offensive players (index 0 is the handler) and 5 defenders, plus the label windows."""
    hold = 0.8 + rng.uniform(-0.2, 0.2)
    off0 = {0: (24.0, 0.0), 1: (17.0, 7.0), 2: (22.0, -18.0), 3: (6.0, 20.0), 4: (6.0, -20.0)}
    if kind == "isolation":
        off0[1] = (14.0, 14.0)
    off = {i: [(0.0, *p)] for i, p in off0.items()}
    labels = []
    extra = {}
    if kind == "ball_screen":
        side = rng.choice([-1.0, 1.0])
        spot = (21.5, side * 2.2)                                   # beside the handler's defender (21, 0)
        t_arrive = hold + 1.1
        t_use0, t_use1 = t_arrive + 0.7, t_arrive + 1.7
        T = t_use1 + 1.0
        off[1] = [(0.0, *off0[1]), (hold, *off0[1]), (t_arrive, *spot), (T, *spot)]
        off[0] = [(0.0, 24.0, 0.0), (t_use0, 24.0, 0.0), (t_use1, 18.0, side * 7.0), (T, 18.0, side * 7.0)]
        extra["def0"] = [(0.0, 21.0, 0.0), (t_use0 + 0.3, 21.0, 0.0), (t_use1 + 0.6, 17.0, side * 6.0),
                         (T, 17.0, side * 6.0)]
        labels.append(("ball_screen", t_arrive, t_use1, 0, 1))
    elif kind == "drive":
        t0 = hold + 0.4
        t1 = t0 + 1.2
        T = t1 + 1.2
        off[0] = [(0.0, 24.0, 0.0), (t0, 24.0, 0.0), (t1, 10.0, 0.0), (T, 10.0, 0.0)]
        extra["def0"] = [(0.0, 21.0, 0.0), (t0 + 0.1, 21.0, 0.0), (t1 + 0.1, 8.0, 0.0), (T, 8.0, 0.0)]
        labels.append(("drive", t0, t1, 0, None))
    elif kind == "isolation":
        T = 5.5
        off[0] = [(0.0, 24.0, 0.0), (T, 24.0, 0.0)]
        extra["iso_shuffle"] = True
        labels.append(("isolation", 0.4, T - 0.4, 0, None))
    else:
        T = 6.0
        slow = rng.uniform() < 0.5
        speed = 2.0 if slow else 7.0
        p0, p1 = (24.0, -8.0), (18.0, 6.0)
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        t0 = hold
        t1 = min(T, t0 + dist / speed)
        off[2] = [(0.0, *p0), (t0, *p0), (t1, *p1), (T, *p1)]
        extra["cutter"] = "slow" if slow else "fast"
    for i in off:
        if off[i][-1][0] < T:
            off[i].append((T, off[i][-1][1], off[i][-1][2]))
    # defenders: 3 ft from their man toward the basket, following him unless scripted
    de = {}
    for i in range(5):
        if i == 0 and "def0" in extra:
            de[i] = extra["def0"]
            continue
        keys = []
        for (t, x, y) in off[i]:
            r = max(np.hypot(x, y), 1e-6)
            keys.append((t, x - 3.0 * x / r, y - 3.0 * y / r))
        de[i] = keys
    return {"kind": kind, "T": T, "off": off, "def": de, "labels": labels, "extra": extra}


def _basket_for(home_offense: bool, period: int):
    left = home_offense == (period <= 2)
    return (BASKET_LEFT, 1.0) if left else (BASKET_RIGHT, -1.0)


def _to_court(local, basket, sign):
    """basket-centred (out, lat) -> court (x, y)."""
    out = np.stack([basket[0] + sign * local[:, 0], basket[1] + local[:, 1]], axis=1)
    return out


# ── synth_game ─────────────────────────────────────────────────────────────────────────────────────
def _fallback_synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
    rng = np.random.default_rng(seed)
    game = sportvu_game or _game_id(seed, heldout)
    mm, dd, yyyy, away, _at, home = game.split(".")
    date = "%s-%s-%s" % (yyyy, mm, dd)
    home_id, away_id = HOME_ID, AWAY_ID
    pids_home = [100 + i for i in range(5)]
    pids_away = [200 + i for i in range(5)]
    kinds, weights = zip(*ACTION_MIX)
    rows_frames, poss_rows, event_rows, scripts = [], [], [], []
    frame0 = 0
    ev = 0
    per_period = max(1, int(np.ceil(n_possessions / 4)))
    period_clock = {}
    for p in range(n_possessions):
        period = 1 + p // per_period
        clock_left = period_clock.setdefault(period, 720.0)
        home_off = p % 2 == 0
        kind = str(rng.choice(kinds, p=weights))
        sc = _script_possession(rng, kind)
        T = sc["T"]
        n = int(round(T * HZ)) + 1
        t = np.arange(n) / HZ
        basket, sign = _basket_for(home_off, period)
        off_ids = pids_home if home_off else pids_away
        def_ids = pids_away if home_off else pids_home
        off_tid, def_tid = (home_id, away_id) if home_off else (away_id, home_id)
        xy = {}
        for i in range(5):
            loc = _keyframes_to_track(sc["off"][i], t)
            if i == 0 and sc["extra"].get("iso_shuffle"):
                loc[:, 1] += 1.5 * np.sin(2 * np.pi * t / 3.0)
            xy[("off", i)] = _to_court(loc, basket, sign)
            loc = _keyframes_to_track(sc["def"][i], t)
            xy[("def", i)] = _to_court(loc, basket, sign)
        # smooth drift (under 1 ft per s) plus tiny jitter on everybody
        for k in xy:
            ph = rng.uniform(0, 2 * np.pi, size=2)
            amp = rng.uniform(0.2, 0.5, size=2)
            xy[k] = xy[k] + amp * np.sin(2 * np.pi * t[:, None] / 4.0 + ph) + rng.normal(0, 0.03, size=(n, 2))
        handler = xy[("off", 0)]
        ball = np.column_stack([handler[:, 0] + sign * 0.8, handler[:, 1] + 0.6,
                                2.0 + 1.5 * np.abs(np.sin(2 * np.pi * 2.0 * t))])
        frames = frame0 + np.arange(n)
        clocks = clock_left - t
        shot_clock = np.maximum(24.0 - t, 0.0)
        for side, ids, tid, team, is_home in (("off", off_ids, off_tid, HOME if off_tid == home_id else AWAY, off_tid == home_id),
                                              ("def", def_ids, def_tid, HOME if def_tid == home_id else AWAY, def_tid == home_id)):
            for i, pid in enumerate(ids):
                pos = xy[(side, i)]
                rows_frames.append(pd.DataFrame({
                    "frame": frames, "period": period, "game_clock_s": clocks, "shot_clock_s": shot_clock,
                    "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_z": ball[:, 2],
                    "team_id": tid, "team": team, "home": is_home, "player_id": pid,
                    "x": pos[:, 0], "y": pos[:, 1]}))
        end_kind = str(rng.choice(["shot_made", "shot_missed", "turnover"], p=[0.45, 0.4, 0.15]))
        points = 2 if end_kind == "shot_made" else 0
        side_txt = "home" if home_off else "away"
        event_rows.append({"event": ev, "period": period, "clock_s": float(clocks[-1]), "team": side_txt,
                           "desc": "%s %s" % (side_txt, end_kind), "points": points, "score": None, "kind": end_kind,
                           "frame": int(frames[-1]), "frame_clock_diff_s": 0.0})
        poss_rows.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": int(frames[0]),
                          "end_frame": int(frames[-1]), "start_clock_s": float(clocks[0]), "end_clock_s": float(clocks[-1]),
                          "start_event": ev, "end_event": ev, "offense_team_id": off_tid, "end_kind": end_kind,
                          "points": points, "gap": False, "n_frames": n})
        labels = []
        for (atype, ta, tb, hi, si) in sc["labels"]:
            labels.append({"action_type": atype, "start_frame": int(frame0 + round(ta * HZ)),
                           "end_frame": int(frame0 + round(tb * HZ)), "ball_handler_id": off_ids[hi],
                           "screener_id": off_ids[si] if si is not None else None,
                           "on_ball_defender_id": def_ids[0], "screener_defender_id": def_ids[si] if si is not None else None})
        scripts.append({"possession": p, "kind": kind, "labels": labels, "extra": sc["extra"]})
        frame0 += n
        ev += 1
        period_clock[period] = float(clocks[-1]) - 1.0 / HZ
    frames = pd.concat(rows_frames, ignore_index=True).sort_values(["frame", "home"], ascending=[True, False], kind="stable")
    frames = frames.reset_index(drop=True)
    frames.insert(0, "game", game)
    frames.insert(1, "gameid", "0021500%03d" % (seed % 1000))
    frames.insert(2, "date", date)
    frames["ts_ms"] = (frames["frame"].to_numpy(dtype=np.int64) * (1000 // HZ)).astype(np.int64)
    frames["n_players"] = np.int8(10)
    names = {pid: "Home Player%d" % i for i, pid in enumerate(pids_home)}
    names.update({pid: "Away Player%d" % i for i, pid in enumerate(pids_away)})
    frames["player"] = frames["player_id"].map(names)
    frames["jersey"] = frames["player_id"].map(lambda p: str(p % 100))
    for c in ("game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "x", "y"):
        frames[c] = frames[c].astype(np.float32)
    frames["period"] = frames["period"].astype(np.int8)
    frames["frame"] = frames["frame"].astype(np.int32)
    frames["team_id"] = frames["team_id"].astype(np.int32)
    frames["player_id"] = frames["player_id"].astype(np.int32)
    frames = frames[["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
                     "ball_x", "ball_y", "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]]
    events = pd.DataFrame(event_rows)
    events.insert(0, "game", game)
    possessions = schema.cast(pd.DataFrame(poss_rows), "possessions")
    roster_rows = [{"sportvu_game": game, "player_id": pid, "player": names[pid], "team_id": home_id, "team": HOME, "home": True,
                    "jersey": str(pid % 100), "bref_slug": "home%02d" % (pid % 100), "bref_name": names[pid],
                    "pbp_name": names[pid], "match_method": "exact"} for pid in pids_home]
    roster_rows += [{"sportvu_game": game, "player_id": pid, "player": names[pid], "team_id": away_id, "team": AWAY, "home": False,
                     "jersey": str(pid % 100), "bref_slug": "away%02d" % (pid % 100), "bref_name": names[pid],
                     "pbp_name": names[pid], "match_method": "exact"} for pid in pids_away]
    rosters = schema.cast(pd.DataFrame(roster_rows), "rosters")
    shots = pd.DataFrame({c: pd.Series([], dtype=d) for c, d in
                          (("sportvu_game", "string"), ("period", "int8"), ("clock_s", "float32"), ("made", "bool"),
                           ("loc_x", "float32"), ("loc_y", "float32"), ("heldout", "bool"))})
    return {"sportvu_game": game, "heldout": bool(heldout), "seed": int(seed), "frames": frames, "events": events,
            "shots": shots, "rosters": rosters, "shot_frames": schema.empty("shot_frames"),
            "rebound_frames": schema.empty("rebound_frames"), "possessions": possessions,
            "states_1hz": schema.empty("states_1hz"), "_script": scripts}


def _fallback_synth_labels(seed: int, game) -> dict:
    """The hand-label tables (actions, screens) of a fallback game dict, as the label tool would write them."""
    if not isinstance(game, dict) or "_script" not in game:
        raise TypeError("fallback synth_labels needs the dict synth_game returned")
    sg = game["sportvu_game"]
    rng = np.random.default_rng(seed + 7919)
    actions, screens = [], []
    aid = 0
    for sc in game["_script"]:
        for lab in sc["labels"]:
            actions.append({"sportvu_game": sg, "possession": sc["possession"], "action_id": aid, "action_type": lab["action_type"],
                            "start_frame": lab["start_frame"], "end_frame": lab["end_frame"],
                            "ball_handler_id": lab["ball_handler_id"], "screener_id": lab["screener_id"],
                            "label_source": "hand", "labeller_pass": "fixture"})
            if lab["action_type"] == "ball_screen":
                screens.append({"sportvu_game": sg, "possession": sc["possession"], "action_id": aid,
                                "coverage": str(rng.choice(COVERAGES)), "on_ball_defender_id": lab["on_ball_defender_id"],
                                "screener_defender_id": lab["screener_defender_id"], "label_source": "hand", "confidence": 1.0})
            aid += 1
    a_cols = ["sportvu_game", "possession", "action_id", "action_type", "start_frame", "end_frame", "ball_handler_id",
              "screener_id", "label_source", "labeller_pass"]
    s_cols = ["sportvu_game", "possession", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id",
              "label_source", "confidence"]
    return {"actions": pd.DataFrame(actions, columns=a_cols), "screens": pd.DataFrame(screens, columns=s_cols)}


def _fallback_synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
    games = []
    for i in range(n_games):
        held = bool(heldout_every) and (i + 1) % heldout_every == 0
        games.append(_fallback_synth_game(seed * 1000 + i, n_possessions=n_possessions, heldout=held,
                                          sportvu_game=_game_id(i, held)))
    return games


def _fallback_write_fixture_root(root, seed: int, n_games: int = 4, heldout_every: int = 0, n_possessions: int = 40) -> dict:
    """DATA_ROOT layout: sportvu_contract/<game>.parquet, <game>_pbp.parquet (held-out under heldout/),
    analytics/possessions/2015-16/<game>.parquet and analytics/rosters/2015-16/ (held-out under heldout/2015-16/),
    labels/actions/<game>.jsonl (one object per line; actions rows carry action_type, screens rows coverage)."""
    root = Path(root)
    games = _fallback_synth_season(seed, n_games, heldout_every, n_possessions)
    written = []
    for g in games:
        sg, held = g["sportvu_game"], g["heldout"]
        sv = root / "sportvu_contract" / ("heldout" if held else "")
        sv.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(sv / (sg + ".parquet"), index=False)
        g["events"].to_parquet(sv / (sg + "_pbp.parquet"), index=False)
        for table in ("possessions", "rosters"):
            d = root / "analytics" / table / ("heldout/" if held else "") / SEASON
            d.mkdir(parents=True, exist_ok=True)
            g[table].to_parquet(d / (sg + ".parquet"), index=False)
        lab = _fallback_synth_labels(seed, g)
        ld = root / "labels" / "actions"
        ld.mkdir(parents=True, exist_ok=True)
        with open(ld / (sg + ".jsonl"), "w") as fh:
            for table in ("actions", "screens"):
                for rec in lab[table].to_dict("records"):
                    rec = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
                           for k, v in rec.items()}
                    rec["table"] = table
                    fh.write(json.dumps(rec) + "\n")
        written.append({"sportvu_game": sg, "heldout": held, "actions": int(len(lab["actions"])), "screens": int(len(lab["screens"]))})
    return {"root": str(root), "games": written}


if not SHARED:
    synth_game = _fallback_synth_game
    synth_season = _fallback_synth_season
    synth_labels = _fallback_synth_labels
    write_fixture_root = _fallback_write_fixture_root


# ── adapters for tests (one shape whichever generator is in use) ───────────────────────────────────
def fixture_source() -> str:
    return "analytics.fixtures (shared, lane I1)" if SHARED else "tests/_fx_actions.py (fallback)"


def _normalise(raw, seed: int, label_fn) -> list:
    out = []
    for g in raw:
        g = dict(g)
        sg = g.get("sportvu_game") or str(g["possessions"]["sportvu_game"].iloc[0])
        g["sportvu_game"] = sg
        held = g.get("heldout")
        if held is None:
            held = bool((g.get("meta") or {}).get("heldout", False)) if isinstance(g.get("meta"), dict) else False
        g["heldout"] = bool(held)
        g["labels"] = label_fn(seed, g)
        out.append(g)
    return out


def fallback_season_games(seed: int = 0, n_games: int = 6, heldout_every: int = 0, n_possessions: int = 24) -> list:
    """The fallback generator whatever is importable: deterministic scripted games for the assertions of the tests."""
    raw = _fallback_synth_season(seed, n_games, heldout_every, n_possessions)
    return _normalise(raw, seed, lambda s, g: _fallback_synth_labels(s, g))


def season_games(seed: int = 0, n_games: int = 6, heldout_every: int = 0, n_possessions: int = 24) -> list:
    """List of game dicts with keys sportvu_game, heldout, frames, possessions, rosters, labels (actions, screens):
    the shared generator when importable (its own possession count), else the fallback."""
    if not SHARED:
        return fallback_season_games(seed, n_games, heldout_every, n_possessions)
    try:
        raw = synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    except TypeError:
        raw = synth_season(seed, n_games=n_games, heldout_every=heldout_every)
    if isinstance(raw, dict):
        raw = raw.get("games", list(raw.values()))
    return _normalise(raw, seed, labels_for)


def labels_for(seed: int, game: dict) -> dict:
    lab = synth_labels(seed, game)
    if isinstance(lab, dict) and "actions" in lab:
        return {"actions": pd.DataFrame(lab["actions"]), "screens": pd.DataFrame(lab.get("screens", []))}
    raise TypeError("synth_labels returned an unexpected shape: %r" % type(lab))

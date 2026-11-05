"""tests/_fx_coverage.py: fixture bundle for the coverage lane (analytics/coverage).

Prefers the shared generator of lane I1 (analytics/fixtures.py) when it imports; otherwise falls back to the
minimal local generator below with the same signatures and column names:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None)
        -> dict(frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of synth_game dicts
    synth_labels(seed, game) -> dict(actions, screens)
    write_fixture_root(root, seed, n_games, heldout_every) -> list of game ids written in the DATA_ROOT layout

The local generator writes scripted ball screens whose defenders move the way the coverage class says
(switch: the screener's defender picks up the handler and the on-ball defender stays with the screener;
drop: the screener's defender retreats toward the rim; hedge_show: he steps to the handler's path for
0.3 to 0.7 s then recovers; blitz_trap: both defenders converge on the handler and stay there; ice: the
on-ball defender sits on the middle side and the handler is pushed toward the sideline; under: the on-ball
defender passes between the screener and the basket; over: he passes on the far side of the screener).
Positions carry 0.1 ft Gaussian noise and per-screen jitter of the screen point and timing. Everything is
deterministic given the seed. The ``local_*`` names are always available so a test can pin itself to this
scripted geometry whatever the shared generator does.

Every number derived from these games is fixture output, not a result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HZ = 25
BASKETS = ((5.25, 25.0), (88.75, 25.0))
COVERAGES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over")
POSS_FRAMES = 150            # 6 s per possession
SCREEN_AT = 50               # local frame of the screen (start_frame)
SCREEN_LEN = 15              # end_frame = start_frame + 15 (0.6 s)
MOTION_END_S = 2.4           # scripted motion runs from t = -2.0 s to +2.4 s around the screen
TEAMS = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD"}
NOISE_FT = 0.1


# ── scripted keyframes in screen coordinates: (t_s, depth toward the basket, lateral toward the sideline) ──
def _keyframes(cov: str) -> dict:
    H_default = [(-2.0, -10, 4), (0.0, -3, 3), (0.4, -1, 1), (0.8, 1.5, -2.5), (1.4, 6, -5), (2.4, 12, -7)]
    H_ice = [(-2.0, -10, 4), (0.0, -3, 3), (0.5, -1, 5), (1.2, 3, 7), (2.4, 8, 8)]
    S = [(-2.0, -2, -3), (-0.3, 0, 0), (0.8, 0, 0), (2.4, 7, -1)]
    D1_start = [(-2.0, -8, 3.5), (0.0, -1.5, 2.5)]
    D2_start = [(-2.0, 6, -1), (0.0, 4, 0)]
    over = [(0.4, -2.0, 1.0), (0.8, -1.5, -1.5), (1.2, 1.5, -3.5), (1.8, 6, -5.5), (2.4, 11, -7)]
    under = [(0.4, 0.5, 1.5), (0.8, 2.5, -0.5), (1.2, 3.5, -3.5), (1.8, 7, -5.5), (2.4, 12, -7)]
    d1 = {"over": over, "drop": over, "hedge_show": over, "under": under,
          "switch": [(0.4, -0.5, 1.0), (0.8, 1.5, 0.5), (1.4, 2.0, 0.5), (2.4, 8.5, -0.5)],
          "blitz_trap": [(0.4, -1.5, 1.5), (0.8, 0.0, -1.0), (1.4, 4.5, -3.5), (2.4, 10.5, -5.5)],
          "ice": [(0.3, -1.0, 2.0), (0.6, -0.5, 2.5), (1.2, 3.0, 4.5), (2.4, 8, 5.5)]}
    d2 = {"switch": [(0.4, 3.5, -0.5), (0.8, 3.5, -2.5), (1.4, 7.5, -5.5), (2.4, 14, -7.5)],
          "drop": [(0.4, 6, -0.5), (0.8, 9, -1.5), (1.4, 12, -2.5), (2.4, 13, -3)],
          "hedge_show": [(0.2, 2.5, 0.5), (0.5, 0.5, -1.0), (0.7, 0.5, -1.5), (1.4, 2.5, 0.5), (2.4, 8, -1)],
          "blitz_trap": [(0.4, 1.0, 0.0), (0.8, 3.0, -4.0), (1.4, 7.5, -6.5), (2.4, 13.5, -8.5)],
          "ice": [(0.4, 5, 1), (0.8, 7, 3), (1.4, 8, 5), (2.4, 10, 6)],
          "under": [(0.4, 4.5, 0), (0.8, 5.5, -0.5), (1.4, 6, -1), (2.4, 9, -2)],
          "over": [(0.4, 3, -0.5), (0.8, 3, -1), (1.4, 4, -0.5), (2.4, 9, -1.5)]}
    return {"handler": H_ice if cov == "ice" else H_default, "screener": S,
            "d1": D1_start + d1[cov], "d2": D2_start + d2[cov]}


def _interp(keys, t):
    ks = np.array([k[0] for k in keys], dtype=float)
    d = np.interp(t, ks, [k[1] for k in keys])
    l = np.interp(t, ks, [k[2] for k in keys])
    return d, l


def _screen_axes(rng, basket):
    """Screen point 20 ft (+-2) from the basket within 50 degrees of its axis; (u toward the basket, v toward
    the nearer sideline)."""
    bx, by = basket
    inward = 1.0 if bx < 47 else -1.0
    ang = np.deg2rad(rng.uniform(-50, 50))
    r = rng.uniform(18, 22)
    P = np.array([bx + inward * r * np.cos(ang), by + r * np.sin(ang)])
    u = (np.array(basket) - P)
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    side = np.sign(P[1] - 25.0) or 1.0
    if v[1] * side < 0:
        v = -v
    return P, u, v


def _scripted_positions(rng, cov, basket):
    """-> dict(role -> (POSS_FRAMES, 2) array of court positions) for one possession with a screen at SCREEN_AT."""
    P, u, v = _screen_axes(rng, basket)
    keys = _keyframes(cov)
    t = (np.arange(POSS_FRAMES) - SCREEN_AT) / HZ
    speed_jit = rng.uniform(0.9, 1.1)
    out = {}
    for role, kf in keys.items():
        kf = [(k[0] * speed_jit, k[1], k[2]) for k in kf]
        tt = np.clip(t, kf[0][0], kf[-1][0])
        d, l = _interp(kf, tt)
        d = d + rng.normal(0, 0.3)
        l = l + rng.normal(0, 0.3)
        xy = P[None, :] + d[:, None] * u[None, :] + l[:, None] * v[None, :]
        out[role] = xy + rng.normal(0, NOISE_FT, size=xy.shape)
    out["screen_point"] = P
    return out


def _drift(rng, basket, n):
    """A bystander: slow random drift inside the attacking half."""
    bx = basket[0]
    x0 = rng.uniform(bx + (8 if bx < 47 else -40), bx + (40 if bx < 47 else -8))
    y0 = rng.uniform(3, 47)
    steps = rng.normal(0, 0.08, size=(n, 2)).cumsum(axis=0)
    xy = np.array([x0, y0])[None, :] + steps
    xy[:, 0] = np.clip(xy[:, 0], 0.5, 93.5)
    xy[:, 1] = np.clip(xy[:, 1], 0.5, 49.5)
    return xy


def _game_id(seed, home_abbr, away_abbr, heldout):
    return "01.%02d.2016.%s.at.%s" % (seed % 28 + 1, away_abbr, "CLE" if heldout else home_abbr)


def local_synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict:
    """One synthetic game: frames (section 4), rosters (16), possessions (19), the rest empty, plus the hidden
    key ``scripted`` (the screens written into the frames) that local_synth_labels reads."""
    rng = np.random.default_rng(seed)
    home_id = int(rng.integers(1, 5))
    away_id = int(rng.choice([t for t in TEAMS if t != home_id]))
    game = sportvu_game or _game_id(seed, TEAMS[home_id], TEAMS[away_id], heldout)
    players = {home_id: [home_id * 100 + i for i in range(1, 6)], away_id: [away_id * 100 + i for i in range(1, 6)]}
    rows, scripted, poss_rows = [], [], []
    per_period = max(1, int(np.ceil(n_possessions / 4)))
    for p in range(n_possessions):
        off = home_id if p % 2 == 0 else away_id
        de = away_id if off == home_id else home_id
        basket = BASKETS[0] if (p // 2) % 2 == 0 else BASKETS[1]
        f0 = p * POSS_FRAMES
        period = 1 + p // per_period
        has_screen = rng.uniform() > 0.2
        pos = {}
        if has_screen:
            cov = COVERAGES[int(rng.integers(0, len(COVERAGES)))]
            handler, screener = rng.choice(players[off], size=2, replace=False)
            d1, d2 = rng.choice(players[de], size=2, replace=False)
            sp = _scripted_positions(rng, cov, basket)
            pos[int(handler)], pos[int(screener)] = sp["handler"], sp["screener"]
            pos[int(d1)], pos[int(d2)] = sp["d1"], sp["d2"]
            scripted.append({"possession": p, "start_frame": f0 + SCREEN_AT, "end_frame": f0 + SCREEN_AT + SCREEN_LEN,
                             "ball_handler_id": int(handler), "screener_id": int(screener),
                             "on_ball_defender_id": int(d1), "screener_defender_id": int(d2), "coverage": cov,
                             "screen_point": [float(sp["screen_point"][0]), float(sp["screen_point"][1])],
                             "basket": list(basket)})
        ball_holder = None
        for tid in (home_id, away_id):
            for pid in players[tid]:
                if pid not in pos:
                    pos[pid] = _drift(rng, basket, POSS_FRAMES)
        if has_screen:
            ball_holder = scripted[-1]["ball_handler_id"]
        else:
            ball_holder = players[off][0]
        ball = pos[ball_holder] + np.array([0.5, 0.0])
        order = [(tid, pid) for tid in (home_id, away_id) for pid in players[tid]]   # home players first
        k = np.repeat(np.arange(POSS_FRAMES), 10)
        frame = f0 + k
        clock = 720.0 - ((p % per_period) * POSS_FRAMES + k) / HZ
        tids = np.tile([t for t, _ in order], POSS_FRAMES)
        pids = np.tile([q for _, q in order], POSS_FRAMES)
        xy = np.stack([pos[q] for _, q in order], axis=1).reshape(-1, 2)      # (POSS_FRAMES * 10, 2)
        rows.append(pd.DataFrame({
            "game": game, "gameid": "002150%04d" % (seed % 10000), "date": "2016-01-%02d" % (seed % 28 + 1),
            "period": np.int8(period), "frame": frame.astype(np.int32), "ts_ms": 1_450_000_000_000 + frame * 40,
            "game_clock_s": clock.astype(np.float32), "shot_clock_s": (24.0 - (k / HZ) % 24).astype(np.float32),
            "n_players": np.int8(10), "ball_x": ball[k, 0].astype(np.float32), "ball_y": ball[k, 1].astype(np.float32),
            "ball_z": np.float32(3.0), "team_id": tids.astype(np.int32), "team": [TEAMS[t] for t in tids],
            "home": tids == home_id, "player_id": pids.astype(np.int32), "player": ["Player %d" % q for q in pids],
            "jersey": [str(q % 100) for q in pids], "x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32)}))
        poss_rows.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": f0,
                          "end_frame": f0 + POSS_FRAMES - 1, "start_clock_s": 720.0 - (p % per_period) * POSS_FRAMES / HZ,
                          "end_clock_s": 720.0 - ((p % per_period) * POSS_FRAMES + POSS_FRAMES - 1) / HZ,
                          "start_event": p * 2, "end_event": p * 2 + 1, "offense_team_id": off,
                          "end_kind": ("shot_made", "shot_missed", "turnover")[p % 3], "points": (2, 0, 0)[p % 3],
                          "gap": False, "n_frames": POSS_FRAMES})
    frames = pd.concat(rows, ignore_index=True)
    rosters = pd.DataFrame([{"sportvu_game": game, "player_id": pid, "player": "Player %d" % pid, "team_id": tid,
                             "team": TEAMS[tid], "home": tid == home_id, "jersey": str(pid % 100), "bref_slug": None,
                             "bref_name": None, "pbp_name": None, "match_method": "none"}
                            for tid in (home_id, away_id) for pid in players[tid]])
    possessions = pd.DataFrame(poss_rows)
    empty = pd.DataFrame()
    return {"sportvu_game": game, "heldout": bool(heldout), "frames": frames, "events": empty, "shots": empty,
            "rosters": rosters, "shot_frames": empty, "rebound_frames": empty, "possessions": possessions,
            "states_1hz": empty, "scripted": scripted}


def local_synth_season(seed, n_games=8, heldout_every=0) -> list:
    """n_games synth_game dicts; every heldout_every-th game (1-based) is held out when heldout_every > 0."""
    out = []
    for i in range(n_games):
        held = bool(heldout_every) and ((i + 1) % heldout_every == 0)
        out.append(local_synth_game(seed * 1000 + i, n_possessions=24, heldout=held))
    return out


def local_synth_labels(seed, game) -> dict:
    """Hand-label tables (actions, screens) of one synth_game dict. Adds a few non-screen actions so the
    validator has something to ignore; one in ten screens is marked unsure."""
    rng = np.random.default_rng(seed + 7)
    g = game["sportvu_game"]
    actions, screens = [], []
    for s in game["scripted"]:
        aid = len(actions)
        actions.append({"sportvu_game": g, "possession": s["possession"], "action_id": aid, "action_type": "ball_screen",
                        "start_frame": s["start_frame"], "end_frame": s["end_frame"],
                        "ball_handler_id": s["ball_handler_id"], "screener_id": s["screener_id"],
                        "label_source": "human", "labeller_pass": 1})
        screens.append({"sportvu_game": g, "possession": s["possession"], "action_id": aid, "coverage": s["coverage"],
                        "on_ball_defender_id": s["on_ball_defender_id"], "screener_defender_id": s["screener_defender_id"],
                        "label_source": "human", "confidence": "unsure" if rng.uniform() < 0.1 else "sure"})
        if rng.uniform() < 0.15:
            actions.append({"sportvu_game": g, "possession": s["possession"], "action_id": len(actions),
                            "action_type": "isolation", "start_frame": s["end_frame"] + 20, "end_frame": s["end_frame"] + 60,
                            "ball_handler_id": s["ball_handler_id"], "screener_id": -1,
                            "label_source": "human", "labeller_pass": 1})
    cols_a = ["sportvu_game", "possession", "action_id", "action_type", "start_frame", "end_frame", "ball_handler_id",
              "screener_id", "label_source", "labeller_pass"]
    cols_s = ["sportvu_game", "possession", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id",
              "label_source", "confidence"]
    return {"actions": pd.DataFrame(actions, columns=cols_a), "screens": pd.DataFrame(screens, columns=cols_s)}


def _season(game_id: str) -> str:
    try:
        m, _d, y = game_id.split(".")[:3]
        y, m = int(y), int(m)
    except (ValueError, IndexError):
        return "2015-16"
    return "%d-%s" % (y, str(y + 1)[2:]) if m >= 9 else "%d-%s" % (y - 1, str(y)[2:])


def local_write_fixture_root(root, seed, n_games, heldout_every) -> list:
    """Write a season in the DATA_ROOT layout: sportvu_contract/<game>.parquet (held-out under heldout/),
    analytics/possessions/<season>/, analytics/rosters/<season>/ (held-out under heldout/<season>/),
    labels/actions/<game>.jsonl, labels/screens/<game>.jsonl."""
    root = Path(root)
    ids = []
    for i, game in enumerate(local_synth_season(seed, n_games, heldout_every)):
        g, held = game["sportvu_game"], game["heldout"]
        season = _season(g)
        fdir = root / "sportvu_contract" / ("heldout" if held else "")
        fdir.mkdir(parents=True, exist_ok=True)
        game["frames"].to_parquet(fdir / (g + ".parquet"), index=False)
        for table in ("possessions", "rosters"):
            tdir = root / "analytics" / table / ("heldout" if held else "") / season
            tdir.mkdir(parents=True, exist_ok=True)
            game[table].to_parquet(tdir / (g + ".parquet"), index=False)
        labels = local_synth_labels(seed * 1000 + i, game)
        for name in ("actions", "screens"):
            ldir = root / "labels" / name
            ldir.mkdir(parents=True, exist_ok=True)
            with open(ldir / (g + ".jsonl"), "w") as fh:
                for rec in labels[name].to_dict("records"):
                    fh.write(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}) + "\n")
        ids.append(g)
    return ids


try:
    from analytics.fixtures import synth_game, synth_season, synth_labels, write_fixture_root  # noqa: F401
    SHARED = True
except ImportError:
    synth_game, synth_season, synth_labels, write_fixture_root = (
        local_synth_game, local_synth_season, local_synth_labels, local_write_fixture_root)
    SHARED = False


def bundle(seed=0, n_games=8, heldout_every=0) -> dict:
    """dict(games=[game dicts with a 'labels' key], shared=bool): a season with labels for every game."""
    games = synth_season(seed, n_games=n_games, heldout_every=heldout_every)
    if isinstance(games, dict):
        games = list(games.values())
    out = []
    for i, game in enumerate(games):
        game = dict(game)
        game["labels"] = synth_labels(seed * 1000 + i, game)
        out.append(game)
    return {"games": out, "shared": SHARED}


__all__ = ["synth_game", "synth_season", "synth_labels", "write_fixture_root", "bundle", "SHARED",
           "local_synth_game", "local_synth_season", "local_synth_labels", "local_write_fixture_root",
           "COVERAGES", "HZ", "BASKETS", "SCREEN_AT", "SCREEN_LEN", "POSS_FRAMES"]

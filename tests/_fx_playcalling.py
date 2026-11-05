"""Fixture stand-ins for the play calling lane (tests/test_playcalling.py, notebooks/eda_playcalling.ipynb).

Shared generators come from analytics.fixtures (lane I1) when that module exists; otherwise the local
fallback below with the same signatures and column names is used. Both are deterministic given a seed.
The local generator makes points per possession depend on the first action's type and coverage (a fixed
expected-points table) plus a small lineup effect, so associations exist for the trainer to find. It is
also much faster (three frames per possession), so the tests run on it and check the shared module once.
Every number derived from these tables is fixture output, not a result.

Generators (shared signatures; the local versions carry the suffix _local)
    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None)
        -> dict(frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of game dicts (every heldout_every-th game held out)
    synth_labels(seed, game) -> dict(actions, screens)
    write_fixture_root(root, seed, n_games, heldout_every) -> root with the DATA_ROOT layout plus the upstream
        lane outputs (analytics/actions/<season>/actions_detected/<game>.parquet,
        analytics/coverage/<season>/coverage_predictions.parquet, analytics/epv/<season>/<game>.parquet)

Lane-local helpers (always local)
    synth_outputs(seed, game, labels) -> dict(actions_detected, coverage_predictions, epv_states)
    season_inputs(seed=0, n_games=8, n_possessions=40, heldout_every=0, with_epv=True, shared=True)
        -> the dict of concatenated inputs analytics.playcalling.features.build_action_table consumes

Conventions shared with analytics/fixtures and analytics/coverage: action_id is the row's position within its
game; screens carry it; only ball_screen actions carry a coverage; label_source "synthetic".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.playcalling import ACTION_TYPES, COVERAGES

try:  # shared fixtures of lane I1, when present
    from analytics.fixtures import synth_game as _shared_game, synth_season as _shared_season, synth_labels as _shared_labels
    HAVE_SHARED = True
except ImportError:
    HAVE_SHARED = False

HELDOUT_GAME = "12.16.2015.PHX.at.GSW"      # the SportVU held-out game of sportvu/splits.json
FRAMES_PER_POSSESSION = 3
N_ROSTER = 8
SEASON = "2015-16"

# expected points per possession by the first action's type, then a coverage shift (fixture truth)
_ACTION_EP = {"ball_screen": 1.06, "handoff": 1.00, "off_ball_screen": 1.02, "isolation": 0.84, "post_up": 0.90, "drive": 1.12, "cut": 1.20}
_COVERAGE_EP = {"switch": -0.08, "drop": 0.10, "hedge_show": 0.00, "blitz_trap": -0.12, "ice": 0.05, "under": 0.08, "over": -0.03, "none": 0.0}
_SCREEN_ACTIONS = ("ball_screen",)
_ACTION_P = np.array([0.40, 0.10, 0.10, 0.12, 0.08, 0.12, 0.08])
_CLASSES = [c for c in COVERAGES if c != "none"]
_COVERAGE_P = np.array([0.25, 0.30, 0.12, 0.08, 0.10, 0.08, 0.07])


def _game_id(seed: int, heldout: bool) -> str:
    if heldout:
        return HELDOUT_GAME if seed % 2 == 0 else "01.%02d.2016.AAA.at.CLE" % (seed % 28 + 1)
    return "01.%02d.2016.AAA.at.BBB" % (seed % 28 + 1)


def _team_ids(seed: int) -> tuple[int, int]:
    return 1000 + 2 * (seed % 50), 1001 + 2 * (seed % 50)


def _lineup(rng, team_id: int) -> list[int]:
    """Five of the eight roster players; three lineups dominate so lineup ids repeat."""
    roster = [team_id * 100 + k for k in range(N_ROSTER)]
    choice = rng.choice(3, p=[0.6, 0.3, 0.1])
    base = [roster[:5], roster[1:6], roster[3:8]][choice]
    return sorted(base)


def synth_game_local(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    game = sportvu_game or _game_id(seed, heldout)
    home_id, away_id = _team_ids(seed)
    teams = {home_id: ("HOM", True), away_id: ("AWY", False)}
    frames, poss = [], []
    ts = 1_450_000_000_000 + seed * 10_000_000
    for p in range(n_possessions):
        period = 1 + (p * 4) // n_possessions
        start_clock = 720.0 - 720.0 * ((p * 4) % n_possessions) / n_possessions
        off = home_id if p % 2 == 0 else away_id
        de = away_id if off == home_id else home_id
        off_lu, def_lu = _lineup(rng, off), _lineup(rng, de)
        f0 = p * FRAMES_PER_POSSESSION
        for k in range(FRAMES_PER_POSSESSION):
            clock = max(start_clock - 6.0 * k, 0.0)
            shot_clock = 24.0 - 7.0 * k if rng.uniform() > 0.05 else np.nan
            for team_id, pid in [(off, i) for i in off_lu] + [(de, i) for i in def_lu]:
                frames.append({"game": game, "gameid": "0021500%03d" % (seed % 1000), "date": "2016-01-%02d" % (seed % 28 + 1),
                               "period": period, "frame": f0 + k, "ts_ms": ts + 40 * (f0 + k), "game_clock_s": clock,
                               "shot_clock_s": shot_clock, "n_players": 10, "ball_x": float(rng.uniform(0, 94)),
                               "ball_y": float(rng.uniform(0, 50)), "ball_z": float(rng.uniform(0, 10)),
                               "team_id": team_id, "team": teams[team_id][0], "home": teams[team_id][1],
                               "player_id": pid, "player": "P %d" % pid, "jersey": str(pid % 100),
                               "x": float(rng.uniform(0, 94)), "y": float(rng.uniform(0, 50))})
        poss.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": f0,
                     "end_frame": f0 + FRAMES_PER_POSSESSION - 1, "start_clock_s": start_clock,
                     "end_clock_s": max(start_clock - 6.0 * (FRAMES_PER_POSSESSION - 1), 0.0),
                     "start_event": 2 * p, "end_event": 2 * p + 1, "offense_team_id": off,
                     "end_kind": "shot_missed", "points": 0, "gap": False, "n_frames": FRAMES_PER_POSSESSION,
                     "fx_off_lineup": off_lu, "fx_def_lineup": def_lu})
    frames = pd.DataFrame(frames)
    possessions = pd.DataFrame(poss)
    roster_rows = [{"sportvu_game": game, "player_id": t * 100 + k, "player": "P %d" % (t * 100 + k), "team_id": t,
                    "team": teams[t][0], "home": teams[t][1], "jersey": str(k), "bref_slug": None, "bref_name": None,
                    "pbp_name": None, "match_method": "none"} for t in (home_id, away_id) for k in range(N_ROSTER)]
    return {"frames": frames, "events": pd.DataFrame(columns=["game", "event", "period", "clock_s", "team", "desc", "points",
                                                             "score", "kind", "frame"]),
            "shots": pd.DataFrame(), "rosters": schema.cast(pd.DataFrame(roster_rows), "rosters"),
            "shot_frames": schema.empty("shot_frames"), "rebound_frames": schema.empty("rebound_frames"),
            "possessions": possessions, "states_1hz": schema.empty("states_1hz"),
            "meta": {"sportvu_game": game, "heldout": heldout, "season": SEASON, "seed": seed}}


def synth_labels_local(seed: int, game: dict) -> dict:
    """Hand labels: one or two actions per possession (action_id = position within the game); the points of the
    possession follow its first action's type and coverage plus a small lineup effect."""
    rng = np.random.default_rng(seed + 7919)
    poss = game["possessions"]
    gid = str(poss["sportvu_game"].iloc[0])
    actions, screens = [], []
    points = np.zeros(len(poss), dtype=int)
    kinds = []
    action_id = 0
    for i, row in enumerate(poss.itertuples(index=False)):
        n_actions = 1 if rng.uniform() < 0.7 else 2
        first = None
        off_lu = list(row.fx_off_lineup)
        for a in range(n_actions):
            at = ACTION_TYPES[rng.choice(len(ACTION_TYPES), p=_ACTION_P)]
            cov = _CLASSES[rng.choice(len(_CLASSES), p=_COVERAGE_P)] if at in _SCREEN_ACTIONS else "none"
            start = int(row.start_frame) + a
            end = min(start + 1, int(row.end_frame))
            actions.append({"sportvu_game": gid, "possession": int(row.possession), "action_type": at,
                            "start_frame": start, "end_frame": end, "ball_handler_id": int(off_lu[rng.integers(5)]),
                            "screener_id": int(off_lu[rng.integers(5)]) if at in _SCREEN_ACTIONS else None,
                            "label_source": "synthetic", "labeller_pass": 1})
            if at in _SCREEN_ACTIONS:
                screens.append({"sportvu_game": gid, "possession": int(row.possession), "action_id": action_id, "coverage": cov,
                                "on_ball_defender_id": int(row.fx_def_lineup[0]), "screener_defender_id": int(row.fx_def_lineup[1]),
                                "label_source": "synthetic", "confidence": "sure"})
            action_id += 1
            if first is None:
                first = (at, cov)
        at, cov = first
        lineup_effect = 0.06 * ((sum(off_lu) // 100) % 3 - 1)
        ep = _ACTION_EP[at] + _COVERAGE_EP[cov] + lineup_effect
        p3 = 0.12
        p2 = float(np.clip((ep - 3 * p3) / 2.0, 0.05, 0.8))
        u = rng.uniform()
        if u < p3:
            points[i] = 3
        elif u < p3 + p2:
            points[i] = 2
        kinds.append("shot_made" if points[i] > 0 else ("turnover" if rng.uniform() < 0.3 else "shot_missed"))
    poss["points"] = points
    poss["end_kind"] = kinds
    screen_cols = ["sportvu_game", "possession", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id", "label_source", "confidence"]
    return {"actions": pd.DataFrame(actions), "screens": pd.DataFrame(screens, columns=screen_cols)}


def synth_season_local(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
    out = []
    for k in range(n_games):
        held = bool(heldout_every) and (k + 1) % heldout_every == 0
        out.append(synth_game_local(seed * 100 + k, n_possessions=n_possessions, heldout=held))
    return out


def synth_outputs(seed: int, game: dict, labels: dict) -> dict:
    """Upstream lane stand-ins derived from a game and its labels: detected actions (the hand actions with
    label_source "model" and one in ten dropped, action_id kept), coverage predictions (the hand coverage flipped
    one time in five, with p_<class> columns; a coverage outside the classes gets no row) and epv_states
    (expected points per frame of each possession, drifting towards the points scored)."""
    from analytics.playcalling.features import assign_action_ids
    rng = np.random.default_rng(seed + 104729)
    actions = assign_action_ids(labels["actions"])
    keep = rng.uniform(size=len(actions)) > 0.1
    detected = actions[keep].copy()
    detected["label_source"] = "model"
    sc = labels["screens"]
    pred_rows = []
    for r in sc.itertuples(index=False):
        if r.coverage not in _CLASSES:
            continue
        cov = r.coverage if rng.uniform() > 0.2 else _CLASSES[rng.integers(len(_CLASSES))]
        probs = rng.dirichlet(np.ones(len(_CLASSES)) * 0.5)
        j = _CLASSES.index(cov)
        probs[j] = max(probs[j], 0.5)
        probs = probs / probs.sum()
        row = {"sportvu_game": r.sportvu_game, "possession": int(r.possession), "action_id": int(r.action_id), "coverage_label": r.coverage}
        row.update({"p_" + c: float(p) for c, p in zip(_CLASSES, probs)})
        row.update({"predicted": cov, "fold": 0, "heldout": False})
        pred_rows.append(row)
    cols = ["sportvu_game", "possession", "action_id", "coverage_label"] + ["p_" + c for c in _CLASSES] + ["predicted", "fold", "heldout"]
    coverage_predictions = pd.DataFrame(pred_rows, columns=cols)
    poss = game["possessions"]
    epv = []
    for r in poss.itertuples(index=False):
        base = 1.0 + 0.1 * rng.standard_normal()
        span = int(r.end_frame) - int(r.start_frame)
        step = max(1, span // 4)
        for k in range(int(r.start_frame), int(r.end_frame) + 1, step):
            frac = (k - r.start_frame) / max(span, 1)
            epv.append({"sportvu_game": r.sportvu_game, "possession": int(r.possession), "frame": int(k),
                        "epv_points": float(base + frac * (float(r.points) - base) * 0.6 + 0.05 * rng.standard_normal())})
    return {"actions_detected": detected, "coverage_predictions": coverage_predictions, "epv_states": pd.DataFrame(epv)}


# ── dispatch to the shared module when present and shaped as assumed ───────────────────────────────
def _shared_ok(games) -> bool:
    try:
        return isinstance(games, list) and all(isinstance(g, dict) and "frames" in g and "possessions" in g for g in games)
    except Exception:  # noqa: BLE001
        return False


def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
    if HAVE_SHARED:
        try:
            g = _shared_game(seed, n_possessions=n_possessions, heldout=heldout, sportvu_game=sportvu_game)
            if _shared_ok([g]):
                return g
        except Exception:  # noqa: BLE001
            pass
    return synth_game_local(seed, n_possessions=n_possessions, heldout=heldout, sportvu_game=sportvu_game)


def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
    if HAVE_SHARED:
        try:
            s = _shared_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
            if _shared_ok(s):
                return s
        except Exception:  # noqa: BLE001
            pass
    return synth_season_local(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)


def synth_labels(seed: int, game: dict) -> dict:
    local = "fx_off_lineup" in game["possessions"].columns
    if HAVE_SHARED and not local:
        try:
            lab = _shared_labels(seed, game)
            if isinstance(lab, dict) and "actions" in lab and len(lab["actions"]):
                return lab
        except Exception:  # noqa: BLE001
            pass
    if not local:   # a shared game without usable labels: derive lineups from its frames and label it locally
        _attach_lineups(game)
    return synth_labels_local(seed, game)


def _attach_lineups(game: dict) -> None:
    fr, poss = game["frames"], game["possessions"]
    offs, defs = [], []
    for r in poss.itertuples(index=False):
        rows = fr[fr["frame"] == r.start_frame]
        off = sorted(rows.loc[rows["team_id"] == r.offense_team_id, "player_id"].astype(int).tolist())
        de = sorted(rows.loc[rows["team_id"] != r.offense_team_id, "player_id"].astype(int).tolist())
        offs.append(off or [0] * 5)
        defs.append(de or [0] * 5)
    poss["fx_off_lineup"], poss["fx_def_lineup"] = offs, defs


def _strip(poss: pd.DataFrame) -> pd.DataFrame:
    return poss.drop(columns=[c for c in poss.columns if c.startswith("fx_")])


def _is_heldout(g: dict, k: int, heldout_every: int) -> bool:
    meta = g.get("meta") or {}
    return bool(meta.get("heldout", False)) or (bool(heldout_every) and (k + 1) % heldout_every == 0)


def season_inputs(seed: int = 0, n_games: int = 8, n_possessions: int = 40, heldout_every: int = 0,
                  with_epv: bool = True, use_detected: bool = False, use_predicted: bool = False, shared: bool = True) -> dict:
    """The concatenated inputs of a season: frames, possessions, actions, screens, coverage_predictions,
    epv_states, plus heldout_games (ids) and season. Held-out games are included with their flag so the
    trainer can set them aside and score them once. shared=False forces the local generator."""
    if shared and HAVE_SHARED:
        games = synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    else:
        games = synth_season_local(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    parts = {"frames": [], "possessions": [], "actions": [], "screens": [], "coverage_predictions": [], "epv_states": []}
    heldout_games = []
    for k, g in enumerate(games):
        gseed = seed * 100 + k
        lab = synth_labels(gseed, g)
        outs = synth_outputs(gseed, g, lab)
        gid = str(g["possessions"]["sportvu_game"].iloc[0])
        if _is_heldout(g, k, heldout_every):
            heldout_games.append(gid)
        parts["frames"].append(g["frames"])
        parts["possessions"].append(_strip(g["possessions"]))
        parts["actions"].append(outs["actions_detected"] if use_detected else lab["actions"])
        parts["screens"].append(pd.DataFrame(columns=lab["screens"].columns) if use_predicted else lab["screens"])
        parts["coverage_predictions"].append(outs["coverage_predictions"])
        parts["epv_states"].append(outs["epv_states"])
    out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    if not with_epv:
        out["epv_states"] = None
    out["heldout_games"] = heldout_games
    out["season"] = SEASON
    return out


def _jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in df.to_dict("records"):
            fh.write(json.dumps({k: (None if (v is pd.NA or (isinstance(v, float) and np.isnan(v))) else (int(v) if isinstance(v, np.integer) else v))
                                 for k, v in rec.items()}) + "\n")


def write_fixture_root(root, seed: int = 0, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40, shared: bool = False) -> Path:
    """Write the DATA_ROOT layout: sportvu_contract/<game>.parquet (held-out under heldout/),
    analytics/possessions/<season>/<game>.parquet (+ heldout/<season>/), labels/actions/<game>.jsonl,
    labels/screens/<game>.jsonl, and the upstream outputs in their lanes' layouts under analytics/actions,
    analytics/coverage and analytics/epv (held-out games under heldout/<season>/)."""
    root = Path(root)
    if shared and HAVE_SHARED:
        games = synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    else:
        games = synth_season_local(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    preds = {False: [], True: []}
    for k, g in enumerate(games):
        gseed = seed * 100 + k
        lab = synth_labels(gseed, g)
        outs = synth_outputs(gseed, g, lab)
        gid = str(g["possessions"]["sportvu_game"].iloc[0])
        held = _is_heldout(g, k, heldout_every)
        sub = "heldout/" if held else ""
        fp = root / "sportvu_contract" / sub / (gid + ".parquet")
        fp.parent.mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(fp, index=False)
        pp = root / "analytics" / "possessions" / sub / SEASON / (gid + ".parquet")
        pp.parent.mkdir(parents=True, exist_ok=True)
        schema.cast(_strip(g["possessions"]), "possessions").to_parquet(pp, index=False)
        _jsonl(root / "labels" / "actions" / (gid + ".jsonl"), lab["actions"])
        _jsonl(root / "labels" / "screens" / (gid + ".jsonl"), lab["screens"])
        ap = root / "analytics" / "actions" / sub / SEASON / "actions_detected" / (gid + ".parquet")
        ap.parent.mkdir(parents=True, exist_ok=True)
        outs["actions_detected"].to_parquet(ap, index=False)
        cp = root / "analytics" / "actions" / sub / SEASON / "action_candidates" / (gid + ".parquet")
        cp.parent.mkdir(parents=True, exist_ok=True)
        outs["actions_detected"].drop(columns=["action_type"]).to_parquet(cp, index=False)      # a decoy without the needed columns
        ep = root / "analytics" / "epv" / sub / SEASON / (gid + ".parquet")
        ep.parent.mkdir(parents=True, exist_ok=True)
        outs["epv_states"].to_parquet(ep, index=False)
        preds[held].append(outs["coverage_predictions"].assign(heldout=held))
    for held, parts in preds.items():
        if parts:
            d = root / "analytics" / "coverage" / ("heldout/" if held else "") / SEASON
            d.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(d / "coverage_predictions.parquet", index=False)
            pd.DataFrame({"team_id": [1], "n": [1]}).to_parquet(d / "coverage_team_rates.parquet", index=False)   # a decoy table
    return root

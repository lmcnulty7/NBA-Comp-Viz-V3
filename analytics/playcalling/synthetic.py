"""analytics/playcalling/synthetic.py: SYNTHETIC inputs of the play calling lane. Fixture output, never a result.

Production code (validate.fixture_inputs, the --fixtures runs of the CLIs, the notebook) builds its synthetic
inputs here, never from tests/. Two game generators:

    shared  analytics.fixtures.synth_season and synth_labels (the physics fixture every lane shares);
    local   synth_season_local / synth_game_local / synth_labels_local below: three frames per possession, much
            faster, and points per possession planted on the first action's type and coverage (a fixed
            expected-points table) plus a small lineup effect, so the trainer has an association to recover.
            The shared fixture plants no such association, which is why the local generator stays
            (tests/test_playcalling.py recovers it; the CLIs expose it as --fixture-generator local).

synth_outputs stands in for the upstream lanes (actions_detected, coverage_predictions, epv_states), which the
shared fixture does not produce; season_inputs concatenates a season into the dict build_action_table consumes.

Conventions shared with analytics/fixtures and analytics/coverage: action_id is the row's position within its
game; screens carry it; only ball_screen actions carry a coverage; label_source "synthetic".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import fixtures, schema
from analytics.playcalling import ACTION_TYPES, COVERAGES

HELDOUT_GAME = fixtures.HELDOUT_GAME          # the SportVU held-out game of sportvu/splits.json
FRAMES_PER_POSSESSION = 3
N_ROSTER = 8
SEASON = "2015-16"
LABELLED_AT = "2026-01-01T00:00:00Z"

# expected points per possession by the first action's type, then a coverage shift (fixture truth)
_ACTION_EP = {"ball_screen": 1.10, "handoff": 1.00, "off_ball_screen": 1.02, "isolation": 0.70, "post_up": 0.80, "drive": 1.25, "cut": 1.40}
_COVERAGE_EP = {"switch": -0.15, "drop": 0.20, "hedge_show": 0.00, "blitz_trap": -0.25, "ice": 0.10, "under": 0.15, "over": -0.05, "none": 0.0}
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
    """One local synthetic game (three frames per possession) with the shared generator's table names; its
    possessions carry fx_off_lineup and fx_def_lineup for synth_labels_local (stripped before the trainer sees them)."""
    rng = np.random.default_rng(seed)
    game = sportvu_game or _game_id(seed, heldout)
    home_id, away_id = _team_ids(seed)
    teams = {home_id: ("HOM", True), away_id: ("AWY", False)}
    cols = {k: [] for k in ("period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "player_id", "x", "y")}
    poss = []
    ts = 1_450_000_000_000 + seed * 10_000_000
    for p in range(n_possessions):
        period = 1 + (p * 4) // n_possessions
        start_clock = 720.0 - 720.0 * ((p * 4) % n_possessions) / n_possessions
        off = home_id if p % 2 == 0 else away_id
        de = away_id if off == home_id else home_id
        off_lu, def_lu = _lineup(rng, off), _lineup(rng, de)
        f0 = p * FRAMES_PER_POSSESSION
        ks = np.repeat(np.arange(FRAMES_PER_POSSESSION), 10)
        n = len(ks)
        shot = np.where(rng.uniform(size=FRAMES_PER_POSSESSION) > 0.05, 24.0 - 7.0 * np.arange(FRAMES_PER_POSSESSION), np.nan)
        cols["period"].append(np.full(n, period))
        cols["frame"].append(f0 + ks)
        cols["ts_ms"].append(ts + 40 * (f0 + ks))
        cols["game_clock_s"].append(np.maximum(start_clock - 6.0 * ks, 0.0))
        cols["shot_clock_s"].append(shot[ks])
        for c in ("ball_x", "ball_y", "ball_z"):
            cols[c].append(np.repeat(rng.uniform(0, 94 if c == "ball_x" else (50 if c == "ball_y" else 10), size=FRAMES_PER_POSSESSION), 10))
        cols["team_id"].append(np.tile(np.array([off] * 5 + [de] * 5), FRAMES_PER_POSSESSION))
        cols["player_id"].append(np.tile(np.array(off_lu + def_lu), FRAMES_PER_POSSESSION))
        cols["x"].append(rng.uniform(0, 94, size=n))
        cols["y"].append(rng.uniform(0, 50, size=n))
        poss.append({"sportvu_game": game, "possession": p, "period": period, "start_frame": f0,
                     "end_frame": f0 + FRAMES_PER_POSSESSION - 1, "start_clock_s": start_clock,
                     "end_clock_s": max(start_clock - 6.0 * (FRAMES_PER_POSSESSION - 1), 0.0),
                     "start_event": 2 * p, "end_event": 2 * p + 1, "offense_team_id": off,
                     "end_kind": "shot_missed", "points": 0, "gap": False, "n_frames": FRAMES_PER_POSSESSION,
                     "fx_off_lineup": off_lu, "fx_def_lineup": def_lu})
    frames = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    frames["game"] = game
    frames["gameid"] = "0021500%03d" % (seed % 1000)
    frames["date"] = "2016-01-%02d" % (seed % 28 + 1)
    frames["n_players"] = 10
    frames["team"] = np.where(frames["team_id"] == home_id, "HOM", "AWY")
    frames["home"] = frames["team_id"] == home_id
    frames["player"] = "P " + frames["player_id"].astype(str)
    frames["jersey"] = (frames["player_id"] % 100).astype(str)
    frames = frames[["game", "gameid", "date", "period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players", "ball_x", "ball_y",
                     "ball_z", "team_id", "team", "home", "player_id", "player", "jersey", "x", "y"]]
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
    action_id = 5          # explicit, non-contiguous ids (5, 8, 11, ...): the joins must not assume positions
    for i, row in enumerate(poss.itertuples(index=False)):
        n_actions = 1 if rng.uniform() < 0.85 else 2
        first = None
        off_lu = list(row.fx_off_lineup)
        for a in range(n_actions):
            at = ACTION_TYPES[rng.choice(len(ACTION_TYPES), p=_ACTION_P)]
            cov = _CLASSES[rng.choice(len(_CLASSES), p=_COVERAGE_P)] if at in _SCREEN_ACTIONS else "none"
            start = int(row.start_frame) + a
            end = min(start + 1, int(row.end_frame))
            actions.append({"sportvu_game": gid, "possession": int(row.possession), "action_id": action_id, "action_type": at,
                            "start_frame": start, "end_frame": end, "ball_handler_id": int(off_lu[rng.integers(5)]),
                            "screener_id": int(off_lu[rng.integers(5)]) if at in _SCREEN_ACTIONS else None,
                            "label_source": "synthetic", "labeller_pass": 1, "labelled_at": LABELLED_AT})
            if at in _SCREEN_ACTIONS:
                screens.append({"sportvu_game": gid, "possession": int(row.possession), "screen_id": len(screens), "action_id": action_id,
                                "coverage": cov, "on_ball_defender_id": int(row.fx_def_lineup[0]),
                                "screener_defender_id": int(row.fx_def_lineup[1]), "label_source": "synthetic", "confidence": "sure",
                                "labeller_pass": 1, "labelled_at": LABELLED_AT})
            action_id += 3
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
    screen_cols = ["sportvu_game", "possession", "screen_id", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id",
                   "label_source", "confidence", "labeller_pass", "labelled_at"]
    return {"actions": pd.DataFrame(actions), "screens": pd.DataFrame(screens, columns=screen_cols)}


def synth_season_local(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
    """n_games local games; every heldout_every-th game (1-based) is held out (0: none)."""
    out = []
    for k in range(n_games):
        held = bool(heldout_every) and (k + 1) % heldout_every == 0
        out.append(synth_game_local(seed * 100 + k, n_possessions=n_possessions, heldout=held))
    return out


def synth_outputs(seed: int, game: dict, labels: dict, with_action_id: bool = False) -> dict:
    """Upstream lane stand-ins derived from a game and its labels: detected actions (the hand actions with
    label_source "model", labeller_pass the string "model" as the actions lane writes it, one in ten dropped and
    NO action_id column unless with_action_id is set, as the actions lane may or may not write one), coverage predictions (the hand coverage flipped
    one time in five, with p_<class> columns; a coverage outside the classes gets no row) and epv_states
    (expected points per frame of each possession, drifting towards the points scored)."""
    from analytics.playcalling.features import assign_action_ids
    rng = np.random.default_rng(seed + 104729)
    actions = assign_action_ids(labels["actions"])
    keep = rng.uniform(size=len(actions)) > 0.1
    detected = actions[keep].copy()
    detected["label_source"] = "model"
    detected["labeller_pass"] = "model"
    detected = detected.drop(columns=[c for c in ("labelled_at",) if c in detected.columns])
    if not with_action_id:
        detected = detected.drop(columns=["action_id"])
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


def season_games(seed: int, n_games: int, heldout_every: int, n_possessions: int, shared: bool = True) -> list:
    """The games of a synthetic season: analytics.fixtures.synth_season (shared=True) or synth_season_local."""
    if shared:
        return fixtures.synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    return synth_season_local(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)


def labels_for(seed: int, game: dict) -> dict:
    """Hand-label stand-ins of a game: synth_labels_local for a local game (its possessions carry fx_ lineups),
    analytics.fixtures.synth_labels for a shared one."""
    if "fx_off_lineup" in game["possessions"].columns:
        return synth_labels_local(seed, game)
    return fixtures.synth_labels(seed, game)


def strip_fx(poss: pd.DataFrame) -> pd.DataFrame:
    """The possessions without the generator's fx_ helper columns."""
    return poss.drop(columns=[c for c in poss.columns if c.startswith("fx_")])


def is_heldout_fixture(g: dict, k: int, heldout_every: int) -> bool:
    """True when the k-th (0-based) game of a synthetic season is held out: its meta flag or the heldout_every rule."""
    meta = g.get("meta") or {}
    return bool(meta.get("heldout", False)) or (bool(heldout_every) and (k + 1) % heldout_every == 0)


def season_inputs(seed: int = 0, n_games: int = 8, n_possessions: int = 40, heldout_every: int = 0,
                  with_epv: bool = True, use_detected: bool = False, use_predicted: bool = False, shared: bool = True) -> dict:
    """The concatenated inputs of a season: frames, possessions, actions, screens, coverage_predictions,
    epv_states, plus heldout_games (ids) and season. Held-out games are included with their flag so the
    trainer can set them aside and score them once. shared=True draws the games from analytics.fixtures,
    shared=False from the local generator above. use_detected feeds the detected actions instead of the labels;
    use_predicted drops the hand screens so coverage comes from the predictions."""
    games = season_games(seed, n_games, heldout_every, n_possessions, shared)
    parts = {"frames": [], "possessions": [], "actions": [], "screens": [], "coverage_predictions": [], "epv_states": []}
    heldout_games = []
    for k, g in enumerate(games):
        gseed = seed * 100 + k
        lab = labels_for(gseed, g)
        outs = synth_outputs(gseed, g, lab)
        gid = str(g["possessions"]["sportvu_game"].iloc[0])
        if is_heldout_fixture(g, k, heldout_every):
            heldout_games.append(gid)
        parts["frames"].append(g["frames"])
        parts["possessions"].append(strip_fx(g["possessions"]))
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

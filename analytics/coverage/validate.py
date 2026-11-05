"""analytics/coverage/validate.py: inputs of the coverage lane and their checks.

    python -m analytics.coverage.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]

Inputs (the upstream interface of lane ACTIONS, consumed as given):
    DATA_ROOT/labels/actions/<game>.jsonl   actions: sportvu_game, possession, action_type, start_frame, end_frame,
                                            ball_handler_id, screener_id, label_source, labeller_pass
    DATA_ROOT/labels/screens/<game>.jsonl   screens: sportvu_game, possession, action_id, coverage, on_ball_defender_id,
                                            screener_defender_id, label_source, confidence
    OUT_ROOT/analytics/actions/<season>/<game>.parquet   the detector's actions (same columns, label_source model)
    DATA_ROOT/sportvu_contract/<game>.parquet (held-out under heldout/), DATA_ROOT/analytics/possessions/<season>/,
    DATA_ROOT/analytics/rosters/<season>/ (held-out under heldout/<season>/)

Checks (every count is logged and returned):
    required columns present; coverage in the 7 classes; confidence in {sure, unsure}; label_source in {human, model}
    (unknown values are logged, not dropped); every screen references an existing action of the same game whose
    action_type is ball_screen (otherwise dropped); a screen whose possession differs from its action's possession
    keeps the action's possession (logged); frames are non-negative with start_frame <= end_frame; player ids are
    integers, handler != screener, on-ball defender != screener's defender; duplicate (sportvu_game, action_id)
    screens keep the first row; rows with a missing required value are dropped; when a possessions table is
    given, a screen whose start_frame lies outside its possession span is logged (kept).

Assumptions taken without asking (also in reports/coverage.md):
    - the actions table carries no action_id column in the interface as given; when the column is absent the
      action_id is the 0-based position of the row within its game's file (jsonl line order, parquet row order)
    - a held-out game is one whose id carries a token of sportvu/splits.json or ends in `.at.CLE` (the held-out
      arena rule of docs/DATA_CONTRACT.md section 7), or whose frames sit under heldout/
    - the season directory is derived from the game id date (MM.DD.YYYY): September or later is the start year
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.coverage import CLASSES, CONFIDENCES, LABEL_SOURCES, SCREEN_ACTION

log = logging.getLogger("analytics.coverage")

ACTION_COLUMNS = ["sportvu_game", "possession", "action_type", "start_frame", "end_frame", "ball_handler_id",
                  "screener_id", "label_source", "labeller_pass"]
SCREEN_COLUMNS = ["sportvu_game", "possession", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id",
                  "label_source", "confidence"]
# The per-screen candidate table every later step consumes (one row per ball screen, labelled or not).
CANDIDATE_COLUMNS = ["sportvu_game", "possession", "action_id", "start_frame", "end_frame", "ball_handler_id",
                     "screener_id", "on_ball_defender_id", "screener_defender_id", "coverage", "label_source",
                     "confidence", "heldout"]


# ── game ids, seasons, held-out ───────────────────────────────────────────────────────────────────
def season_of(game_id: str) -> str:
    """`MM.DD.YYYY.AWAY.at.HOME` -> `YYYY-YY` (September or later starts the season); `unknown` otherwise."""
    try:
        m, _d, y = str(game_id).split(".")[:3]
        y, m = int(y), int(m)
    except (ValueError, IndexError):
        return "unknown"
    return "%d-%s" % (y, str(y + 1)[2:]) if m >= 9 else "%d-%s" % (y - 1, str(y)[2:])


def heldout_mask(game_ids) -> pd.Series:
    """True for ids that carry a held-out token of sportvu/splits.json or end in `.at.CLE`."""
    from sportvu import splits
    toks = splits.heldout_tokens()
    g = pd.Series(game_ids).astype(str)
    return g.map(lambda s: any(t in s for t in toks) or s.endswith(".at." + splits.HELDOUT_ARENA)).astype(bool)


# ── readers ───────────────────────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _with_action_id(df: pd.DataFrame) -> pd.DataFrame:
    """action_id = the row's position within its game when the column is absent."""
    df = df.copy()
    if "action_id" not in df.columns:
        df["action_id"] = df.groupby("sportvu_game", sort=False).cumcount()
    return df


def load_labels(data_root) -> dict:
    """dict(actions, screens) from DATA_ROOT/labels/{actions,screens}/<game>.jsonl (empty frames when none)."""
    root = Path(data_root) / "labels"
    out = {}
    for name, cols in (("actions", ACTION_COLUMNS), ("screens", SCREEN_COLUMNS)):
        files = sorted((root / name).glob("*.jsonl")) if (root / name).exists() else []
        frames = [_read_jsonl(p) for p in files]
        frames = [f for f in frames if len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        out[name] = _with_action_id(df) if name == "actions" else df
        log.info("labels/%s: %d files, %d rows", name, len(files), len(df))
    return out


def load_detected(out_root) -> pd.DataFrame:
    """The detector's actions from OUT_ROOT/analytics/actions/<season>/<game>.parquet (every season; held-out
    games are whatever sits under heldout/). Empty frame when the directory does not exist."""
    root = Path(out_root) / "analytics" / "actions"
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    frames = [pd.read_parquet(p) for p in files]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ACTION_COLUMNS)
    log.info("actions_detected: %d files, %d rows", len(files), len(df))
    return _with_action_id(df)


FRAME_COLUMNS = ["frame", "period", "game_clock_s", "team_id", "player_id", "x", "y"]


def _find(root: Path, sub: str, game: str, season: str) -> tuple:
    """-> (path, heldout) of the first existing layout variant, or (None, False)."""
    cands = [(root / sub / season / (game + ".parquet"), False), (root / sub / "heldout" / season / (game + ".parquet"), True),
             (root / sub / (game + ".parquet"), False), (root / sub / "heldout" / (game + ".parquet"), True)]
    for p, held in cands:
        if p.exists():
            return p, held
    return None, False


def load_game(data_root, game: str) -> dict | None:
    """dict(frames, rosters, possessions, heldout) of one game from the DATA_ROOT layout; None without frames."""
    root, season = Path(data_root), season_of(game)
    fpath, held = _find(root, "sportvu_contract", game, season)
    if fpath is None:
        return None
    frames = pd.read_parquet(fpath, columns=FRAME_COLUMNS)
    out = {"frames": frames, "heldout": bool(held or heldout_mask([game]).iloc[0])}
    for table in ("rosters", "possessions"):
        p, _ = _find(root / "analytics", table, game, season)
        out[table] = pd.read_parquet(p) if p is not None else None
    return out


def load_data_root(data_root, out_root=None) -> dict:
    """Everything the lane reads: dict(games={id: load_game(...)}, actions, screens, detected, missing_frames)."""
    labels = load_labels(data_root)
    detected = load_detected(out_root) if out_root else pd.DataFrame(columns=ACTION_COLUMNS + ["action_id"])
    ids = sorted(set(labels["screens"].get("sportvu_game", pd.Series(dtype=str)).astype(str))
                 | set(labels["actions"].get("sportvu_game", pd.Series(dtype=str)).astype(str))
                 | set(detected.get("sportvu_game", pd.Series(dtype=str)).astype(str)))
    games, missing = {}, []
    for g in ids:
        game = load_game(data_root, g)
        if game is None:
            missing.append(g)
        else:
            games[g] = game
    if missing:
        log.warning("%d game(s) with labels but no frames under sportvu_contract/: %s", len(missing), missing[:5])
    return {"games": games, "actions": labels["actions"], "screens": labels["screens"], "detected": detected,
            "missing_frames": missing}


# ── fixtures ──────────────────────────────────────────────────────────────────────────────────────
def _fixture_module():
    """tests/_fx_coverage.py (which itself prefers analytics.fixtures when that imports)."""
    path = Path(__file__).resolve().parents[2] / "tests" / "_fx_coverage.py"
    spec = importlib.util.spec_from_file_location("_fx_coverage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_bundle(seed: int = 0, n_games: int = 8, heldout_every: int = 0, local: bool = False) -> dict:
    """The load_data_root structure built in memory from the fixture generator (shared when it imports, else the
    local scripted one; local=True pins the local one)."""
    fx = _fixture_module()
    gen_season = fx.local_synth_season if local else fx.synth_season
    gen_labels = fx.local_synth_labels if local else fx.synth_labels
    season = gen_season(seed, n_games=n_games, heldout_every=heldout_every)
    if isinstance(season, dict):
        season = list(season.values())
    games, actions, screens = {}, [], []
    for i, game in enumerate(season):
        frames = game["frames"]
        gid = str(game.get("sportvu_game") or frames["game"].iloc[0])
        held = bool(game.get("heldout", False)) or bool(heldout_mask([gid]).iloc[0])
        labels = gen_labels(seed * 1000 + i, game)
        games[gid] = {"frames": frames[[c for c in FRAME_COLUMNS if c in frames.columns]], "rosters": game.get("rosters"),
                      "possessions": game.get("possessions"), "heldout": held}
        actions.append(labels["actions"])
        screens.append(labels["screens"])
    return {"games": games, "actions": _with_action_id(pd.concat(actions, ignore_index=True)),
            "screens": pd.concat(screens, ignore_index=True), "detected": pd.DataFrame(columns=ACTION_COLUMNS + ["action_id"]),
            "missing_frames": [], "fixture": "local" if (local or not fx.SHARED) else "shared"}


# ── checks ────────────────────────────────────────────────────────────────────────────────────────
def _int_ok(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.notna() & (v == np.floor(v.fillna(0)))


def check_actions(actions: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """Drop and count action rows that fail the schema or range checks; keep every action_type."""
    counts["actions_in"] = int(len(actions))
    missing = [c for c in ACTION_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError("actions table lacks column(s) %s" % missing)
    a = _with_action_id(actions)
    ok = a[["sportvu_game", "possession", "action_type", "start_frame", "end_frame", "ball_handler_id"]].notna().all(axis=1)
    counts["actions_missing_value"] = int((~ok).sum())
    a = a[ok]
    ints = _int_ok(a["start_frame"]) & _int_ok(a["end_frame"]) & _int_ok(a["ball_handler_id"]) & _int_ok(a["possession"])
    counts["actions_non_integer"] = int((~ints).sum())
    a = a[ints]
    for c in ("start_frame", "end_frame", "ball_handler_id", "possession", "action_id"):
        a[c] = pd.to_numeric(a[c]).astype(int)
    a["screener_id"] = pd.to_numeric(a["screener_id"], errors="coerce")
    rng_ok = (a["start_frame"] >= 0) & (a["end_frame"] >= a["start_frame"])
    counts["actions_bad_frame_range"] = int((~rng_ok).sum())
    a = a[rng_ok]
    screen = a["action_type"].astype(str) == SCREEN_ACTION
    scr_ok = (~screen) | (a["screener_id"].notna() & (a["screener_id"] != a["ball_handler_id"]) & (a["screener_id"] >= 0))
    counts["ball_screen_without_valid_screener"] = int((~scr_ok).sum())
    a = a[scr_ok]
    dup = a.duplicated(["sportvu_game", "action_id"], keep="first")
    counts["actions_duplicate_id"] = int(dup.sum())
    a = a[~dup]
    bad_src = ~a["label_source"].astype(str).isin(LABEL_SOURCES)
    counts["actions_unknown_label_source"] = int(bad_src.sum())
    counts["actions_kept"] = int(len(a))
    counts["ball_screen_actions"] = int((a["action_type"].astype(str) == SCREEN_ACTION).sum())
    return a.reset_index(drop=True)


def check_screens(screens: pd.DataFrame, actions: pd.DataFrame, counts: dict, possessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Drop and count screen rows that fail the schema, enum, reference and range checks; return the screens joined
    to their action (start_frame, end_frame, ball_handler_id, screener_id from the action)."""
    counts["screens_in"] = int(len(screens))
    missing = [c for c in SCREEN_COLUMNS if c not in screens.columns]
    if missing:
        raise ValueError("screens table lacks column(s) %s" % missing)
    s = screens.copy()
    ok = s[["sportvu_game", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id"]].notna().all(axis=1)
    counts["screens_missing_value"] = int((~ok).sum())
    s = s[ok]
    ints = _int_ok(s["action_id"]) & _int_ok(s["on_ball_defender_id"]) & _int_ok(s["screener_defender_id"])
    counts["screens_non_integer"] = int((~ints).sum())
    s = s[ints]
    for c in ("action_id", "on_ball_defender_id", "screener_defender_id"):
        s[c] = pd.to_numeric(s[c]).astype(int)
    enum_ok = s["coverage"].astype(str).isin(CLASSES)
    counts["screens_unknown_coverage"] = int((~enum_ok).sum())
    s = s[enum_ok]
    conf_ok = s["confidence"].astype(str).isin(CONFIDENCES)
    counts["screens_unknown_confidence"] = int((~conf_ok).sum())
    s = s[conf_ok]
    counts["screens_unknown_label_source"] = int((~s["label_source"].astype(str).isin(LABEL_SOURCES)).sum())
    dup = s.duplicated(["sportvu_game", "action_id"], keep="first")
    counts["screens_duplicate_id"] = int(dup.sum())
    s = s[~dup]
    acts = actions[actions["action_type"].astype(str) == SCREEN_ACTION]
    acts = acts[["sportvu_game", "action_id", "possession", "start_frame", "end_frame", "ball_handler_id", "screener_id"]]
    acts = acts.rename(columns={"possession": "action_possession"})
    j = s.merge(acts, on=["sportvu_game", "action_id"], how="left")
    ref_ok = j["start_frame"].notna()
    counts["screens_without_ball_screen_action"] = int((~ref_ok).sum())
    j = j[ref_ok]
    poss_mismatch = pd.to_numeric(j["possession"], errors="coerce") != j["action_possession"]
    counts["screens_possession_mismatch"] = int(poss_mismatch.sum())
    j["possession"] = j["action_possession"].astype(int)
    j = j.drop(columns=["action_possession"])
    ids_ok = ((j["on_ball_defender_id"] != j["screener_defender_id"])
              & ~j["on_ball_defender_id"].isin([-1]) & ~j["screener_defender_id"].isin([-1])
              & (j["on_ball_defender_id"] != j["ball_handler_id"]) & (j["screener_defender_id"] != j["screener_id"]))
    counts["screens_bad_defender_ids"] = int((~ids_ok).sum())
    j = j[ids_ok]
    if possessions is not None and len(possessions):
        p = possessions[["sportvu_game", "possession", "start_frame", "end_frame"]].rename(
            columns={"start_frame": "p_start", "end_frame": "p_end"})
        jj = j.merge(p, on=["sportvu_game", "possession"], how="left")
        inside = jj["p_start"].isna() | ((jj["start_frame"] >= jj["p_start"]) & (jj["start_frame"] <= jj["p_end"]))
        counts["screens_outside_possession_span"] = int((~inside).sum())
    for c in ("start_frame", "end_frame", "ball_handler_id", "screener_id"):
        j[c] = j[c].astype(int)
    counts["screens_kept"] = int(len(j))
    for k in CLASSES:
        counts["label_" + k] = int((j["coverage"] == k).sum())
    return j.reset_index(drop=True)


def candidates(actions: pd.DataFrame, screens: pd.DataFrame, detected: pd.DataFrame | None, heldout_by_game: dict,
               counts: dict) -> pd.DataFrame:
    """One row per ball screen with CANDIDATE_COLUMNS: hand-labelled screens (coverage set), hand-labelled
    ball_screen actions without a screen row (coverage null, defenders null) and the detector's screens (coverage
    null, defenders null, label_source model). A detected screen that duplicates a hand-labelled action (same
    game, same handler, start frames within 1 s) is dropped and counted."""
    rows = [screens.assign(label_source=screens["label_source"].astype(str))]
    acts = actions[actions["action_type"].astype(str) == SCREEN_ACTION]
    unl = acts.merge(screens[["sportvu_game", "action_id"]], on=["sportvu_game", "action_id"], how="left", indicator=True)
    unl = unl[unl["_merge"] == "left_only"].drop(columns="_merge")
    counts["ball_screen_actions_without_label"] = int(len(unl))
    if len(unl):
        rows.append(unl.assign(on_ball_defender_id=pd.NA, screener_defender_id=pd.NA, coverage=pd.NA, confidence=pd.NA))
    if detected is not None and len(detected):
        d = check_actions(detected, {})
        d = d[d["action_type"].astype(str) == SCREEN_ACTION].copy()
        d["label_source"] = "model"
        if len(acts):
            m = d.merge(acts[["sportvu_game", "ball_handler_id", "start_frame"]].rename(columns={"start_frame": "h_start"}),
                        on=["sportvu_game", "ball_handler_id"], how="left")
            near = (m["h_start"].notna()) & ((m["start_frame"] - m["h_start"]).abs() <= 25)
            dup_ids = set(zip(m.loc[near, "sportvu_game"], m.loc[near, "action_id"]))
            keep = ~pd.Series(list(zip(d["sportvu_game"], d["action_id"]))).isin(dup_ids).to_numpy()
            counts["detected_duplicating_hand_label"] = int((~keep).sum())
            d = d[keep]
        counts["detected_screens"] = int(len(d))
        d["action_id"] = d["action_id"].astype(int)
        rows.append(d.assign(on_ball_defender_id=pd.NA, screener_defender_id=pd.NA, coverage=pd.NA, confidence=pd.NA))
    out = pd.concat([r.reindex(columns=CANDIDATE_COLUMNS[:-1]) for r in rows], ignore_index=True)
    out["heldout"] = out["sportvu_game"].astype(str).map(lambda g: bool(heldout_by_game.get(g, False))).to_numpy()
    out["heldout"] = out["heldout"] | heldout_mask(out["sportvu_game"]).to_numpy()
    for c in ("possession", "action_id", "start_frame", "end_frame", "ball_handler_id", "screener_id"):
        out[c] = out[c].astype(int)
    for c in ("on_ball_defender_id", "screener_defender_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    counts["candidates"] = int(len(out))
    counts["candidates_heldout"] = int(out["heldout"].sum())
    return out


def validate_bundle(bundle: dict) -> tuple:
    """Run every check on a load_data_root or fixture_bundle structure -> (candidate table, counts)."""
    counts: dict = {"games": len(bundle["games"]), "games_heldout": int(sum(g["heldout"] for g in bundle["games"].values())),
                    "games_without_frames": len(bundle.get("missing_frames", []))}
    actions = check_actions(bundle["actions"], counts)
    poss = [g["possessions"] for g in bundle["games"].values() if g.get("possessions") is not None]
    poss = pd.concat(poss, ignore_index=True) if poss else None
    screens = check_screens(bundle["screens"], actions, counts, poss)
    held = {g: v["heldout"] for g, v in bundle["games"].items()}
    cand = candidates(actions, screens, bundle.get("detected"), held, counts)
    known = cand["sportvu_game"].isin(list(bundle["games"]))
    counts["candidates_without_frames"] = int((~known).sum())
    cand = cand[known].reset_index(drop=True)
    for k, v in counts.items():
        log.info("validate: %s = %s", k, v)
    return cand, counts


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────
def add_input_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="run on the in-memory fixture season (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT (sportvu_contract/, analytics/, labels/)")
    ap.add_argument("--out", default=None, help="OUT_ROOT (analytics/actions/ detected screens are read from here; outputs are written here)")
    ap.add_argument("--fixture-games", type=int, default=8)
    ap.add_argument("--fixture-heldout-every", type=int, default=0)


def load_inputs(a) -> dict:
    if a.fixtures:
        return fixture_bundle(seed=getattr(a, "seed", 0), n_games=a.fixture_games, heldout_every=a.fixture_heldout_every)
    return load_data_root(a.data, a.out)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_input_args(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stderr)
    bundle = load_inputs(a)
    _cand, counts = validate_bundle(bundle)
    counts["source"] = "fixture (%s generator): fixture output, not a result" % bundle.get("fixture", "local") if a.fixtures else str(a.data)
    print(json.dumps(counts, indent=1))
    return counts


if __name__ == "__main__":
    main()

"""analytics/actions/validate.py: input and label checks for the action-recognition lane.

    python -m analytics.actions.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]

Inputs per game (DATA_ROOT layout on the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet            frames, section 4 (held-out games under heldout/)
    DATA_ROOT/analytics/possessions/2015-16/<game>.parquet   section 19 (held-out under heldout/2015-16/)
    DATA_ROOT/analytics/rosters/2015-16/<game>.parquet       section 16 (optional for this lane)
    DATA_ROOT/labels/actions/<game>.jsonl, labels/screens/<game>.jsonl   hand labels, read only through
                                                         analytics.labels.read_labels(labeller_pass=1)

Checks, each logged with counts and returned in the report dict:
    frames        required columns, dtype casts, null ball counts, duplicate (frame, player_id) rows (dropped),
                  positions outside the court plus a 10 ft margin, frames with other than 10 players
    possessions   schema.validate (section 19), start_frame <= end_frame, spans inside the frame range
    rosters       schema.validate (section 16)
    labels        required columns, action_type / label_source / coverage enums, start_frame <= end_frame,
                  labels outside their possession span, duplicate labels, overlapping labels of the same
                  action_type in one possession (flagged, kept), screens whose action_id has no actions row
                  or points at a non ball_screen action, handler and screener ids missing from the roster
Problems are strings; counts are integers. A game whose frames or possessions fail a schema check is
reported and skipped by the trainer (validate.ok(report) is false). Nothing here fits anything.

Loaders (used by features.py, train.py and eda.py): load_fixture_games(seed, n_games, heldout_every) returns
the shared fixtures of analytics.fixtures when importable, else the fallback generator of tests/_fx_actions.py;
load_data_root(DATA_ROOT) discovers every game with frames and a possessions table. Held-out games are the
ones under heldout/, the ones whose id carries a held-out token of sportvu/splits.json, and the ones at the
held-out arena (CLE).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.actions import ACTION_TYPES, COVERAGES, LABEL_SOURCES, SEASON

log = logging.getLogger("analytics.actions.validate")

FRAME_COLUMNS = ("frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "home",
                 "player_id", "x", "y")
ACTION_COLUMNS = ("sportvu_game", "possession", "action_type", "start_frame", "end_frame", "ball_handler_id",
                  "screener_id", "label_source", "labeller_pass")
SCREEN_COLUMNS = ("sportvu_game", "possession", "action_id", "coverage", "on_ball_defender_id", "screener_defender_id",
                  "label_source", "confidence")
COURT_MARGIN_FT = 10.0
COURT_X, COURT_Y = 94.0, 50.0


# ── held-out detection ─────────────────────────────────────────────────────────────────────────────
def is_heldout_id(sportvu_game: str) -> bool:
    """True when sportvu/splits.json marks the game held out (by id token or by the CLE arena)."""
    from sportvu import splits
    g = str(sportvu_game)
    return any(t in g for t in splits.heldout_tokens()) or g.endswith(".at." + splits.HELDOUT_ARENA)


# ── labels ─────────────────────────────────────────────────────────────────────────────────────────
def read_labels_jsonl(path) -> dict:
    """A JSONL this lane wrote (actions_detected) -> dict(actions, screens) frames. NOT the hand-label reader:
    hand labels go through load_hand_labels (analytics.labels.read_labels), which applies tombstones and the
    labeller pass. A line's table is its "table" key when present, else actions when it carries action_type,
    screens when it carries coverage; other lines are counted and dropped."""
    actions, screens, unknown = [], [], 0
    path = Path(path)
    if path.exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                table = rec.pop("table", None)
                if table is None:
                    table = "actions" if "action_type" in rec else ("screens" if "coverage" in rec else None)
                if table == "actions":
                    actions.append(rec)
                elif table == "screens":
                    screens.append(rec)
                else:
                    unknown += 1
    a = pd.DataFrame(actions, columns=list(dict.fromkeys(list(ACTION_COLUMNS) + [c for r in actions for c in r])))
    s = pd.DataFrame(screens, columns=list(dict.fromkeys(list(SCREEN_COLUMNS) + [c for r in screens for c in r])))
    if "action_id" not in a.columns or a["action_id"].isna().all():
        a["action_id"] = np.arange(len(a))
    out = {"actions": a, "screens": s}
    if unknown:
        log.warning("%s: %d line(s) that are neither actions nor screens rows dropped", path.name, unknown)
        out["unknown_lines"] = unknown
    return out


TRAINING_PASS = 1


def load_hand_labels(root, game: str, labeller_pass: int = TRAINING_PASS) -> dict:
    """The hand labels of ``game`` through the sanctioned reader analytics.labels.read_labels: the last line per
    id wins, tombstoned (deleted) ids are dropped, and only ``labeller_pass`` (1, the first pass) is kept, so
    the pass-2 relabels of the consistency check never reach a fit. Never reads the raw lines."""
    from analytics.labels import read_labels
    actions, screens = read_labels(root, game, labeller_pass=labeller_pass)
    return {"actions": actions, "screens": screens, "labeller_pass": labeller_pass}


def write_labels_jsonl(path, actions: pd.DataFrame, screens: pd.DataFrame) -> int:
    """The inverse of read_labels_jsonl (used to write actions_detected in the hand-label layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for table, df in (("actions", actions), ("screens", screens)):
            for rec in df.to_dict("records"):
                clean = {}
                for k, v in rec.items():
                    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                        clean[k] = None
                    elif hasattr(v, "item"):
                        clean[k] = v.item()
                    else:
                        clean[k] = v
                clean["table"] = table
                fh.write(json.dumps(clean) + "\n")
                n += 1
    return n


def validate_labels(actions: pd.DataFrame, screens: pd.DataFrame, possessions: pd.DataFrame | None = None,
                    rosters: pd.DataFrame | None = None) -> dict:
    """Problems and counts of one game's hand labels (see the module docstring)."""
    problems, warnings, counts = [], [], {"actions": int(len(actions)), "screens": int(len(screens))}
    for c in ACTION_COLUMNS:
        if c not in actions.columns:
            problems.append("actions: missing column %r" % c)
    for c in SCREEN_COLUMNS:
        if c not in screens.columns:
            problems.append("screens: missing column %r" % c)
    if problems:
        return {"problems": problems, "warnings": warnings, "counts": counts}
    a = actions.copy()
    s = screens.copy()
    if "action_id" not in a.columns or a["action_id"].isna().all():
        a["action_id"] = np.arange(len(a))          # the row's position among the actions rows (module docstring)
    bad = sorted(set(a["action_type"].dropna().astype(str)) - set(ACTION_TYPES))
    if bad:
        problems.append("actions: unknown action_type value(s) %s" % bad)
    for name, df in (("actions", a), ("screens", s)):
        bad = sorted(set(df["label_source"].dropna().astype(str)) - set(LABEL_SOURCES))
        if bad:
            problems.append("%s: unknown label_source value(s) %s" % (name, bad))
    bad = sorted(set(s["coverage"].dropna().astype(str)) - set(COVERAGES))
    if bad:
        warnings.append("screens: coverage value(s) outside the assumed vocabulary %s (kept)" % bad)
    for c in ("start_frame", "end_frame", "possession", "ball_handler_id"):
        if a[c].isna().any():
            problems.append("actions: %d null(s) in %s" % (int(a[c].isna().sum()), c))
    if problems:
        return {"problems": problems, "warnings": warnings, "counts": counts}
    rev = a["end_frame"].astype(int) < a["start_frame"].astype(int)
    counts["reversed_windows"] = int(rev.sum())
    if rev.any():
        problems.append("actions: %d label(s) with end_frame before start_frame" % int(rev.sum()))
    counts["by_action_type"] = {k: int(v) for k, v in a["action_type"].value_counts().items()}
    counts["by_label_source"] = {k: int(v) for k, v in a["label_source"].value_counts().items()}
    key = ["sportvu_game", "possession", "action_type", "start_frame", "end_frame"]
    dup = a.duplicated(key, keep="first")
    counts["duplicate_labels"] = int(dup.sum())
    if dup.any():
        warnings.append("actions: %d duplicate label(s) (same possession, type and window)" % int(dup.sum()))
    # overlapping labels of one type in one possession
    n_overlap = 0
    for (_g, _p, _t), grp in a.sort_values("start_frame").groupby(["sportvu_game", "possession", "action_type"]):
        ends = grp["end_frame"].astype(int).to_numpy()
        starts = grp["start_frame"].astype(int).to_numpy()
        n_overlap += int((starts[1:] <= np.maximum.accumulate(ends)[:-1]).sum())
    counts["overlapping_labels"] = n_overlap
    if n_overlap:
        warnings.append("actions: %d label(s) overlapping an earlier label of the same type in the same possession" % n_overlap)
    if possessions is not None and len(possessions):
        span = possessions.set_index("possession")[["start_frame", "end_frame"]]
        pk = a["possession"].astype(int)
        known = pk.isin(span.index)
        counts["labels_unknown_possession"] = int((~known).sum())
        if (~known).any():
            problems.append("actions: %d label(s) name a possession not in the possessions table" % int((~known).sum()))
        ps = span.reindex(pk[known]).to_numpy()
        inside = (a.loc[known, "start_frame"].astype(int).to_numpy() >= ps[:, 0]) & (a.loc[known, "end_frame"].astype(int).to_numpy() <= ps[:, 1])
        counts["labels_outside_possession"] = int((~inside).sum())
        if (~inside).any():
            warnings.append("actions: %d label(s) extend outside their possession span" % int((~inside).sum()))
    if rosters is not None and len(rosters):
        ids = set(rosters["player_id"].astype(int))
        miss = ~a["ball_handler_id"].astype(int).isin(ids)
        scr = a["screener_id"].dropna().astype(int)
        miss_s = ~scr.isin(ids)
        counts["handler_not_on_roster"] = int(miss.sum())
        counts["screener_not_on_roster"] = int(miss_s.sum())
        if miss.any() or miss_s.any():
            problems.append("actions: %d handler and %d screener id(s) missing from the roster" % (int(miss.sum()), int(miss_s.sum())))
    if len(s):
        aid = a.set_index("action_id")["action_type"]
        ref = s["action_id"].isin(aid.index)
        counts["screens_unknown_action"] = int((~ref).sum())
        if (~ref).any():
            problems.append("screens: %d row(s) reference an action_id with no actions row" % int((~ref).sum()))
        not_screen = ref & (s["action_id"].map(aid) != "ball_screen")
        counts["screens_on_non_ball_screen"] = int(not_screen.sum())
        if not_screen.any():
            problems.append("screens: %d row(s) point at an action that is not a ball_screen" % int(not_screen.sum()))
        conf = pd.to_numeric(s["confidence"], errors="coerce")
        out_of_range = conf.notna() & ((conf < 0) | (conf > 1))
        if out_of_range.any():
            problems.append("screens: %d numeric confidence value(s) outside [0, 1]" % int(out_of_range.sum()))
        text = conf.isna() & s["confidence"].notna()
        try:
            from analytics.labels import CONFIDENCES
        except ImportError:   # pragma: no cover
            CONFIDENCES = ("sure", "unsure")
        odd = text & ~s["confidence"].astype(str).isin(CONFIDENCES)
        counts["screens_text_confidence"] = int(text.sum())
        if odd.any():      # a grade outside the label contract's vocabulary is kept and counted
            warnings.append("screens: %d confidence value(s) are neither a grade in %s nor a number in [0, 1] (kept)"
                            % (int(odd.sum()), list(CONFIDENCES)))
        if s["confidence"].isna().any():
            problems.append("screens: %d null confidence value(s)" % int(s["confidence"].isna().sum()))
    return {"problems": problems, "warnings": warnings, "counts": counts}


# ── frames, possessions, rosters ───────────────────────────────────────────────────────────────────
def validate_frames(frames: pd.DataFrame) -> dict:
    """Problems and counts of a section 4 frames table; returns the de-duplicated frame too (key frame, player_id)."""
    problems, warnings, counts = [], [], {"rows": int(len(frames))}
    missing = [c for c in FRAME_COLUMNS if c not in frames.columns]
    if missing:
        problems.append("frames: missing column(s) %s" % missing)
        return {"problems": problems, "warnings": warnings, "counts": counts, "frames": frames}
    f = frames
    for c in ("frame", "period", "team_id", "player_id"):
        try:
            pd.to_numeric(f[c])
        except (ValueError, TypeError):
            problems.append("frames: column %r is not numeric" % c)
    if problems:
        return {"problems": problems, "warnings": warnings, "counts": counts, "frames": frames}
    dup = f.duplicated(["frame", "player_id"], keep="first")
    counts["duplicate_rows"] = int(dup.sum())
    if dup.any():
        warnings.append("frames: %d duplicate (frame, player_id) row(s) dropped" % int(dup.sum()))
        f = f.loc[~dup]
    counts["frames"] = int(f["frame"].nunique())
    per = f.groupby("frame").size()
    counts["frames_not_10_players"] = int((per != 10).sum())
    if counts["frames_not_10_players"]:
        warnings.append("frames: %d frame(s) with other than 10 players" % counts["frames_not_10_players"])
    first = f.drop_duplicates("frame")
    counts["frames_null_ball"] = int(first["ball_x"].isna().sum())
    counts["frames_null_shot_clock"] = int(first["shot_clock_s"].isna().sum())
    x, y = f["x"].astype(float), f["y"].astype(float)
    off = (x < -COURT_MARGIN_FT) | (x > COURT_X + COURT_MARGIN_FT) | (y < -COURT_MARGIN_FT) | (y > COURT_Y + COURT_MARGIN_FT)
    counts["rows_off_court"] = int(off.sum())
    if off.any():
        warnings.append("frames: %d row(s) more than %g ft outside the court" % (int(off.sum()), COURT_MARGIN_FT))
    nan_xy = x.isna() | y.isna()
    counts["rows_null_position"] = int(nan_xy.sum())
    if nan_xy.any():
        warnings.append("frames: %d row(s) with a null position" % int(nan_xy.sum()))
    clock = first.sort_values("frame")["game_clock_s"].astype(float)
    counts["clock_jumps_over_2s"] = int(((clock.diff().abs() > 2.0) & (first.sort_values("frame")["period"].diff() == 0)).sum())
    return {"problems": problems, "warnings": warnings, "counts": counts, "frames": f}


def validate_possessions(possessions: pd.DataFrame, frames: pd.DataFrame | None = None) -> dict:
    problems = ["possessions: " + p for p in schema.validate(possessions, "possessions")]
    warnings, counts = [], {"rows": int(len(possessions))}
    if not problems and len(possessions):
        rev = possessions["end_frame"].astype(int) < possessions["start_frame"].astype(int)
        if rev.any():
            problems.append("possessions: %d row(s) with end_frame before start_frame" % int(rev.sum()))
        counts["gap_possessions"] = int(possessions["gap"].astype(bool).sum())
        if frames is not None and len(frames):
            lo, hi = int(frames["frame"].min()), int(frames["frame"].max())
            out = (possessions["start_frame"].astype(int) < lo) | (possessions["end_frame"].astype(int) > hi)
            counts["possessions_outside_frames"] = int(out.sum())
            if out.any():
                warnings.append("possessions: %d span(s) reach outside the frame range" % int(out.sum()))
    return {"problems": problems, "warnings": warnings, "counts": counts}


def validate_rosters(rosters: pd.DataFrame | None) -> dict:
    if rosters is None:
        return {"problems": [], "warnings": ["rosters: table absent (optional)"], "counts": {"rows": 0}}
    return {"problems": ["rosters: " + p for p in schema.validate(rosters, "rosters")], "warnings": [],
            "counts": {"rows": int(len(rosters))}}


def validate_game(game: dict) -> dict:
    """All checks on one game dict (keys sportvu_game, heldout, frames, possessions, rosters, labels)."""
    rep = {"sportvu_game": game["sportvu_game"], "heldout": bool(game.get("heldout", False))}
    rep["frames"] = validate_frames(game["frames"])
    game["frames"] = rep["frames"].pop("frames")
    rep["possessions"] = validate_possessions(game["possessions"], game["frames"])
    rep["rosters"] = validate_rosters(game.get("rosters"))
    lab = game.get("labels") or {"actions": pd.DataFrame(columns=ACTION_COLUMNS), "screens": pd.DataFrame(columns=SCREEN_COLUMNS)}
    rep["labels"] = validate_labels(lab["actions"], lab["screens"], game["possessions"], game.get("rosters"))
    rep["ok"] = not (rep["frames"]["problems"] or rep["possessions"]["problems"] or rep["labels"]["problems"])
    for part in ("frames", "possessions", "rosters", "labels"):
        for p in rep[part]["problems"]:
            log.error("%s: %s", rep["sportvu_game"], p)
        for w in rep[part]["warnings"]:
            log.warning("%s: %s", rep["sportvu_game"], w)
    return rep


def ok(report: dict) -> bool:
    return bool(report.get("ok", False))


# ── loaders ────────────────────────────────────────────────────────────────────────────────────────
def load_fixture_games(seed: int = 0, n_games: int = 6, heldout_every: int = 3, n_possessions: int = 24) -> list:
    """Synthetic games with labels: analytics.fixtures when importable, else the fallback of tests/_fx_actions.py."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from tests import _fx_actions as fx       # tests/ is an implicit namespace package from the repo root
    except ImportError:
        sys.path.insert(0, str(root / "tests"))
        import _fx_actions as fx                   # type: ignore
    return fx.season_games(seed=seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)


def discover_games(root) -> list:
    """[(sportvu_game, heldout)] for every game with frames and a possessions table under DATA_ROOT."""
    root = Path(root)
    out = []
    for held in (False, True):
        sv = root / "sportvu_contract" / ("heldout" if held else "")
        if not sv.exists():
            continue
        for p in sorted(sv.glob("*.parquet")):
            if p.name.endswith("_pbp.parquet"):
                continue
            game = p.stem
            poss = root / "analytics" / "possessions" / ("heldout/" if held else "") / SEASON / (game + ".parquet")
            if poss.exists():
                out.append((game, held or is_heldout_id(game)))
            else:
                log.warning("%s: frames present but no possessions table at %s; skipped", game, poss)
    return out


def load_game(root, game: str, heldout: bool) -> dict:
    root = Path(root)
    sub = "heldout/" if heldout else ""
    frames = pd.read_parquet(root / "sportvu_contract" / sub / (game + ".parquet"))
    poss = pd.read_parquet(root / "analytics" / "possessions" / sub / SEASON / (game + ".parquet"))
    rpath = root / "analytics" / "rosters" / sub / SEASON / (game + ".parquet")
    rosters = pd.read_parquet(rpath) if rpath.exists() else None
    labels = load_hand_labels(root, game)
    return {"sportvu_game": game, "heldout": bool(heldout), "frames": frames, "possessions": poss, "rosters": rosters,
            "labels": labels}


def load_data_root(root) -> list:
    games = discover_games(root)
    if not games:
        raise SystemExit("validate: no game with frames and possessions under %s" % root)
    return [load_game(root, g, h) for g, h in games]


def load_inputs(args) -> list:
    """The list of game dicts named by the CLI flags (--fixtures or --data)."""
    if args.fixtures:
        return load_fixture_games(seed=args.seed, n_games=args.fixture_games, heldout_every=args.fixture_heldout_every,
                                  n_possessions=args.fixture_possessions)
    return load_data_root(args.data)


def add_input_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic games (fixture output, never a result)")
    g.add_argument("--data", help="DATA_ROOT with sportvu_contract/, analytics/possessions/, labels/actions/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixture-games", type=int, default=6)
    ap.add_argument("--fixture-heldout-every", type=int, default=3)
    ap.add_argument("--fixture-possessions", type=int, default=24)


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def summarize(reports: list) -> dict:
    tot = {"games": len(reports), "ok": sum(1 for r in reports if r["ok"]), "heldout_games": sum(1 for r in reports if r["heldout"]),
           "actions": sum(r["labels"]["counts"].get("actions", 0) for r in reports),
           "screens": sum(r["labels"]["counts"].get("screens", 0) for r in reports),
           "problems": sum(len(r[p]["problems"]) for r in reports for p in ("frames", "possessions", "rosters", "labels")),
           "warnings": sum(len(r[p]["warnings"]) for r in reports for p in ("frames", "possessions", "rosters", "labels"))}
    by_type = {}
    for r in reports:
        for k, v in r["labels"]["counts"].get("by_action_type", {}).items():
            by_type[k] = by_type.get(k, 0) + v
    tot["by_action_type"] = by_type
    return tot


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m analytics.actions.validate", description=__doc__.split("\n")[0])
    add_input_args(ap)
    ap.add_argument("--out", help="OUT_ROOT: writes analytics/actions/validation.json under it")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    games = load_inputs(a)
    reports = [validate_game(g) for g in games]
    out = {"summary": summarize(reports), "games": reports, "fixtures": bool(a.fixtures)}
    if a.out:
        dest = Path(a.out) / "analytics" / "actions" / "validation.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1, default=str))
        out["path"] = str(dest)
    print(json.dumps(out["summary"], indent=1))
    return out


if __name__ == "__main__":
    main()

"""analytics/playcalling/validate.py: checks on every input of the play calling lane, and the data root loader.

    python -m analytics.playcalling.validate --fixtures [--fixture-games 8] | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

Inputs and what is checked (every check logs a count; nothing is silently dropped):
  frames (docs/DATA_CONTRACT.md section 4): required columns; duplicates on (game, frame, player_id) dropped;
      shot_clock_s outside [0, 24] counted; period outside 1..8 counted.
  possessions (section 19): analytics.schema.validate (dtypes, nulls, the end_kind enum); duplicates on
      (sportvu_game, possession) dropped; points outside 0..8 counted; start_frame > end_frame counted.
  actions: required columns; action_type against analytics.playcalling.ACTION_TYPES (unknown values counted
      and kept as their own category); label_source enum; start_frame > end_frame counted; duplicates on
      (sportvu_game, possession, action_id) dropped; actions whose possession is missing counted and dropped
      at build time; actions whose span leaves the possession span counted.
  screens and coverage_predictions: required columns; coverage (or predicted) against COVERAGES, unknown
      counted and kept; duplicates on the action key dropped; p_<class> columns in [0, 1] and summing near 1.
  epv_states: required columns; epv_points finite and within [-1, 4] counted; duplicates on
      (sportvu_game, possession, frame) dropped.

Hard failures (the CLI exits 1 and the trainer stops): a required column missing, an empty actions or
possessions table. Everything else is a logged count.

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet, held-out games under sportvu_contract/heldout/
    DATA_ROOT/analytics/possessions/<season>/<game>.parquet and analytics/possessions/heldout/<season>/
    DATA_ROOT/labels/actions/<game>.jsonl, DATA_ROOT/labels/screens/<game>.jsonl (hand labels, read through
    analytics.labels.read_labels(labeller_pass=1): tombstones applied, pass-2 relabels never reach training; the
    raw reader read_labels_raw is the explicit fallback with the same rules; detected rows get no pass filter)
    OUT_ROOT/analytics/actions/**.parquet (actions_detected), OUT_ROOT/analytics/coverage/**.parquet
    (coverage_predictions), OUT_ROOT/analytics/epv/**.parquet (epv_states), each optional; the upstream
    lanes' exact file layout was not fixed when this was written, so every parquet below those folders is
    read and a path holding "heldout" marks the game held out.
A game is held out when analytics.eval.is_heldout_game says so (a held-out token of sportvu/splits.json, the
held-out arena included) or when it sits under a heldout/ folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import is_heldout_game
from analytics.playcalling import ACTION_TYPES, COVERAGES, LABEL_SOURCES, SEASON, WHO

REQUIRED = {
    "frames": ("game", "frame", "period", "team_id", "player_id", "shot_clock_s"),
    "possessions": ("sportvu_game", "possession", "period", "start_frame", "end_frame", "offense_team_id", "end_kind", "points"),
    "actions": ("sportvu_game", "possession", "action_type", "start_frame", "end_frame"),
    "screens": ("sportvu_game", "possession", "action_id"),
    "coverage_predictions": ("sportvu_game", "possession", "action_id"),
    "epv_states": ("sportvu_game", "possession", "frame", "epv_points"),
}
KEYS = {
    "frames": ("game", "frame", "player_id"),
    "possessions": ("sportvu_game", "possession"),
    "actions": ("sportvu_game", "possession", "action_id"),
    "screens": ("sportvu_game", "possession", "action_id"),
    "coverage_predictions": ("sportvu_game", "possession", "action_id"),
    "epv_states": ("sportvu_game", "possession", "frame"),
}


class ValidationError(ValueError):
    """A hard failure: a required column is missing or a required table is empty."""


# ── per-table checks ────────────────────────────────────────────────────────────────────────────────
def _require(df: pd.DataFrame | None, name: str) -> None:
    if df is None:
        return
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValidationError("%s: %s is missing required column(s) %s" % (WHO, name, missing))


def _dedupe(df: pd.DataFrame, name: str, log: dict) -> pd.DataFrame:
    keys = [k for k in KEYS[name] if k in df.columns]
    if len(keys) < len(KEYS[name]):
        log["duplicates_dropped"] = 0
        return df
    n = int(df.duplicated(keys).sum())
    log["duplicates_dropped"] = n
    return df.drop_duplicates(keys, keep="first") if n else df


def _enum(df: pd.DataFrame, col: str, allowed, log: dict) -> None:
    if col not in df.columns:
        return
    vals = df[col].dropna().astype(str)
    bad = vals[~vals.isin(allowed)]
    log["unknown_" + col] = {k: int(v) for k, v in bad.value_counts().items()}
    log["null_" + col] = int(df[col].isna().sum())


def _count(mask, log: dict, key: str) -> None:
    log[key] = int(np.asarray(mask, dtype=bool).sum())


def validate_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Frames checks -> (deduplicated frames, log)."""
    _require(df, "frames")
    log = {"rows": int(len(df)), "games": int(df["game"].nunique())}
    df = _dedupe(df, "frames", log)
    sc = pd.to_numeric(df["shot_clock_s"], errors="coerce")
    _count(sc.notna() & ((sc < 0) | (sc > 24)), log, "shot_clock_out_of_range")
    _count(sc.isna(), log, "shot_clock_null")
    _count((df["period"] < 1) | (df["period"] > 8), log, "period_out_of_range")
    return df, log


def validate_possessions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Possessions checks (schema.validate included) -> (deduplicated possessions, log)."""
    _require(df, "possessions")
    if not len(df):
        raise ValidationError("%s: the possessions table is empty" % WHO)
    log = {"rows": int(len(df)), "games": int(df["sportvu_game"].nunique()), "schema": schema.validate(df, "possessions")}
    df = _dedupe(df, "possessions", log)
    _count((df["points"] < 0) | (df["points"] > 8), log, "points_out_of_range")
    _count(df["start_frame"] > df["end_frame"], log, "start_after_end")
    _count(df["offense_team_id"].isna(), log, "null_offense_team")
    _enum(df, "end_kind", schema.END_KINDS, log)
    return df, log


def validate_actions(df: pd.DataFrame, possessions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Actions checks, action ids assigned where missing -> (deduplicated actions, log)."""
    from analytics.playcalling.features import assign_action_ids
    _require(df, "actions")
    if not len(df):
        raise ValidationError("%s: the actions table is empty" % WHO)
    log = {"rows": int(len(df)), "games": int(df["sportvu_game"].nunique())}
    df = assign_action_ids(df)
    df = _dedupe(df, "actions", log)
    _enum(df, "action_type", ACTION_TYPES, log)
    _enum(df, "label_source", LABEL_SOURCES, log)
    _count(df["start_frame"] > df["end_frame"], log, "start_after_end")
    for c in ("ball_handler_id", "screener_id"):
        log["null_" + c] = int(df[c].isna().sum()) if c in df.columns else int(len(df))
    if possessions is not None and len(possessions):
        pk = ["sportvu_game", "possession"]
        m = df[pk + ["start_frame", "end_frame"]].merge(possessions[pk + ["start_frame", "end_frame"]].rename(
            columns={"start_frame": "_ps", "end_frame": "_pe"}), on=pk, how="left")
        _count(m["_ps"].isna(), log, "possession_missing")
        _count(m["_ps"].notna() & ((m["start_frame"] < m["_ps"]) | (m["end_frame"] > m["_pe"])), log, "outside_possession_span")
    return df, log


def _validate_coverage_like(df: pd.DataFrame, name: str, col: str) -> tuple[pd.DataFrame, dict]:
    _require(df, name)
    log = {"rows": int(len(df)), "games": int(df["sportvu_game"].nunique()) if len(df) else 0}
    df = _dedupe(df, name, log)
    if col in df.columns:
        _enum(df, col, COVERAGES, log)
    else:
        log["missing_" + col] = True
    pcols = [c for c in df.columns if c.startswith("p_")]
    if pcols and len(df):
        P = df[pcols].astype(float)
        _count(((P < 0) | (P > 1)).any(axis=1), log, "p_out_of_range")
        _count((P.sum(axis=1) - 1).abs() > 0.02, log, "p_not_summing_to_one")
    return df, log


def validate_screens(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Hand screen label checks -> (deduplicated screens, log)."""
    return _validate_coverage_like(df, "screens", "coverage")


def validate_coverage_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Coverage lane prediction checks -> (deduplicated predictions, log)."""
    col = "predicted" if "predicted" in df.columns else "coverage"
    return _validate_coverage_like(df, "coverage_predictions", col)


def validate_epv_states(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """EPV lane state checks -> (deduplicated states, log)."""
    _require(df, "epv_states")
    log = {"rows": int(len(df)), "games": int(df["sportvu_game"].nunique()) if len(df) else 0}
    df = _dedupe(df, "epv_states", log)
    e = pd.to_numeric(df["epv_points"], errors="coerce")
    _count(~np.isfinite(e.to_numpy(dtype=float)), log, "epv_not_finite")
    _count(e.notna() & ((e < -1) | (e > 4)), log, "epv_out_of_range")
    return df, log


VALIDATORS = {"frames": validate_frames, "possessions": validate_possessions, "screens": validate_screens,
              "coverage_predictions": validate_coverage_predictions, "epv_states": validate_epv_states}


def validate_inputs(inputs: dict) -> tuple[dict, dict]:
    """Run every validator -> (cleaned inputs, log by table). Missing optional tables are logged as absent."""
    out, log = dict(inputs), {}
    for name, fn in VALIDATORS.items():
        df = inputs.get(name)
        if df is None:
            log[name] = {"absent": True}
            out[name] = None
            continue
        out[name], log[name] = fn(df)
    out["actions"], log["actions"] = validate_actions(inputs["actions"], out.get("possessions"))
    log["heldout_games"] = sorted(map(str, inputs.get("heldout_games", ())))
    if inputs.get("files"):
        log["files"] = inputs["files"]
    return out, log


def format_report(log: dict) -> str:
    """The validation log as markdown, one section per table."""
    L = ["# playcalling validate", ""]
    for name, entry in log.items():
        L.append("## %s" % name)
        if isinstance(entry, dict):
            for k, v in entry.items():
                L.append("- %s: %s" % (k, json.dumps(v, default=str)))
        else:
            L.append("- %s" % json.dumps(entry, default=str))
        L.append("")
    return "\n".join(L)


# ── loaders ─────────────────────────────────────────────────────────────────────────────────────────
def read_labels_raw(path: Path, kind: str, labeller_pass: int | None = 1) -> pd.DataFrame:
    """Explicit fallback for a labels JSONL when analytics.labels.read_labels cannot read it: drops every line
    carrying ``deleted``, keeps the last line per id (action_id for actions, screen_id for screens; a line without
    an id takes the running position of the non-tombstone lines), then keeps ``labeller_pass == labeller_pass``
    (a line without labeller_pass counts as pass 1)."""
    idf = "action_id" if kind == "actions" else "screen_id"
    live, order, n = {}, [], 0
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        key = rec.get(idf)
        if rec.get("deleted"):
            if key is not None:
                live.pop(int(key), None)
            continue
        if key is None:
            key = n
            rec[idf] = key
        key = int(key)
        n += 1
        if key not in live:
            order.append(key)
        live[key] = rec
    df = pd.DataFrame([live[k] for k in order if k in live])
    if labeller_pass is not None and len(df):
        lp = pd.to_numeric(df.get("labeller_pass", pd.Series(1, index=df.index)), errors="coerce").fillna(1)
        df = df[lp == int(labeller_pass)].reset_index(drop=True)
    return df


def load_hand_labels(data_root: Path, games) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """(actions, screens, log) of the hand labels of ``games`` through analytics.labels.read_labels(labeller_pass=1):
    tombstones applied, pass 2 relabels excluded. A game whose files that reader cannot parse falls back to
    read_labels_raw with the same rules; the log counts both paths."""
    acts, scrs, log = [], [], {"games": 0, "read_labels": 0, "raw_fallback": 0, "actions_rows": 0, "screens_rows": 0}
    from analytics import labels as L
    for game in sorted(games):
        log["games"] += 1
        a = sc = None
        try:
            a, sc = L.read_labels(data_root, game, labeller_pass=1)
            log["read_labels"] += 1
        except (ValueError, TypeError, KeyError):      # a line the sanctioned reader cannot parse or cast: fall back, counted
            a = sc = None
        if a is None:
            log["raw_fallback"] += 1
            ap, sp = data_root / "labels" / "actions" / (game + ".jsonl"), data_root / "labels" / "screens" / (game + ".jsonl")
            a = read_labels_raw(ap, "actions") if ap.exists() else pd.DataFrame()
            sc = read_labels_raw(sp, "screens") if sp.exists() else pd.DataFrame()
        if len(a):
            acts.append(a)
        if len(sc):
            scrs.append(sc)
    actions = pd.concat(acts, ignore_index=True) if acts else None
    screens = pd.concat(scrs, ignore_index=True) if scrs else None
    log["actions_rows"] = 0 if actions is None else int(len(actions))
    log["screens_rows"] = 0 if screens is None else int(len(screens))
    return actions, screens, log


def _read_many(paths, reader=pd.read_parquet, required=(), prefer: str | None = None) -> tuple[pd.DataFrame | None, list[str], int]:
    """Concatenate the files -> (frame or None, game ids found under a heldout/ folder, files skipped). A file without
    every ``required`` column is skipped (the upstream folders hold several tables); when ``prefer`` names a file stem
    prefix and some files match it, only those are read."""
    paths = sorted(Path(p) for p in paths)
    if prefer:
        hit = [p for p in paths if p.stem.startswith(prefer) or prefer in p.parts]
        paths = hit or paths
    frames, held, skipped = [], [], 0
    for p in paths:
        df = reader(p)
        if any(c not in df.columns for c in required):
            skipped += 1
            continue
        if "heldout" in p.parts:
            col = "sportvu_game" if "sportvu_game" in df.columns else ("game" if "game" in df.columns else None)
            held += df[col].astype(str).unique().tolist() if col else [p.stem]
        frames.append(df)
    if not frames:
        return None, held, skipped
    return pd.concat(frames, ignore_index=True), held, skipped


def load_data_root(data_root, out_root=None, season: str = SEASON) -> dict:
    """Read the DATA_ROOT and OUT_ROOT layout of the module docstring into the inputs dict (plus a ``files`` log)."""
    data_root = Path(data_root)
    out_root = Path(out_root) if out_root else data_root
    inputs: dict = {"season": season, "files": {}}
    held: set = set()

    def take(name, paths, reader=pd.read_parquet, prefer=None):
        df, h, skipped = _read_many(paths, reader, REQUIRED[name], prefer)
        held.update(h)
        inputs["files"][name] = {"rows": 0 if df is None else int(len(df)), "skipped_files": skipped}
        return df

    inputs["frames"] = take("frames", data_root.glob("sportvu_contract/**/*.parquet"))
    inputs["possessions"] = take("possessions", (data_root / "analytics" / "possessions").rglob("*.parquet"))
    stems = {p.stem for p in (data_root / "labels" / "actions").glob("*.jsonl")} | {p.stem for p in (data_root / "labels" / "screens").glob("*.jsonl")}
    hand, screens, hlog = load_hand_labels(data_root, stems)
    inputs["files"]["labels_hand"] = hlog
    det = take("actions", (out_root / "analytics" / "actions").rglob("*.parquet"), prefer="actions_detected")
    inputs["files"]["actions_detected"] = inputs["files"].pop("actions")
    if det is not None and len(det) and "label_source" not in det.columns:
        det["label_source"] = "model"
    if hand is not None and len(hand):
        if "label_source" not in hand.columns:
            hand["label_source"] = "hand"
        if det is not None and len(det):          # detected rows only for games without hand labels; no pass filter on them
            det = det[~det["sportvu_game"].isin(hand["sportvu_game"].unique())]
            hand = pd.concat([hand, det], ignore_index=True)
        inputs["actions"] = hand
    elif det is not None:
        inputs["actions"] = det
    else:
        inputs["actions"] = pd.DataFrame(columns=list(REQUIRED["actions"]))
    inputs["screens"] = screens
    inputs["coverage_predictions"] = take("coverage_predictions", (out_root / "analytics" / "coverage").rglob("*.parquet"),
                                          prefer="coverage_predictions")
    inputs["epv_states"] = take("epv_states", (out_root / "analytics" / "epv").rglob("*.parquet"))
    games = set()
    for name in ("frames", "possessions", "actions"):
        df = inputs.get(name)
        if df is not None and len(df):
            col = "game" if name == "frames" else "sportvu_game"
            games |= set(df[col].astype(str).unique())
    inputs["heldout_games"] = sorted(g for g in games if g in held or is_heldout_game(g))
    return inputs


GENERATORS = ("auto", "shared", "local")


def fixture_inputs(seed: int = 0, n_games: int = 8, n_possessions: int = 40, heldout_every: int = 4, with_epv: bool = True,
                   generator: str = "auto") -> dict:
    """Synthetic inputs from analytics/playcalling/synthetic.py: the games of analytics.fixtures for generator "auto"
    or "shared", the lane's local generator (planted PPP associations) for "local". Fixture output, never a result."""
    from analytics.playcalling import synthetic
    if generator not in GENERATORS:
        raise ValueError("%s: unknown fixture generator %r (one of %s)" % (WHO, generator, GENERATORS))
    return synthetic.season_inputs(seed, n_games=n_games, n_possessions=n_possessions, heldout_every=heldout_every,
                                   with_epv=with_epv, shared=(generator != "local"))


def add_fixture_args(ap: argparse.ArgumentParser) -> None:
    """The fixture knobs every CLI of the lane takes (canonical --fixture-* spellings)."""
    ap.add_argument("--fixture-games", type=int, default=8, help="fixture: number of synthetic games (default %(default)s)")
    ap.add_argument("--fixture-possessions", type=int, default=40, help="fixture: possessions per synthetic game (default %(default)s)")
    ap.add_argument("--fixture-heldout-every", type=int, default=4,
                    help="fixture: every k-th synthetic game is held out, 0 for none (default %(default)s)")
    ap.add_argument("--fixture-generator", choices=GENERATORS, default="auto",
                    help="auto or shared: analytics.fixtures; local: the lane's generator with planted PPP associations")


def fixture_inputs_from_args(a) -> dict:
    """fixture_inputs with the knobs of add_fixture_args."""
    return fixture_inputs(a.seed, n_games=a.fixture_games, n_possessions=a.fixture_possessions,
                          heldout_every=a.fixture_heldout_every, generator=a.fixture_generator)


def main(argv=None) -> dict:
    """Validate the fixture or DATA_ROOT inputs and print the log; exits 1 on a hard failure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic inputs (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", default=None, help="OUT_ROOT holding the upstream lanes' outputs (default DATA_ROOT)")
    ap.add_argument("--season", default=SEASON, help="season directory (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    add_fixture_args(ap)
    a = ap.parse_args(argv)
    inputs = fixture_inputs_from_args(a) if a.fixtures else load_data_root(a.data, a.out, a.season)
    try:
        _, log = validate_inputs(inputs)
    except ValidationError as exc:
        print(str(exc))
        sys.exit(1)
    print(format_report(log))
    return log


if __name__ == "__main__":
    main()

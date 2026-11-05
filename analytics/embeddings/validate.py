"""analytics/embeddings/validate.py: input checks for the player style embeddings.

    python -m analytics.embeddings.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]

Per input table (the DATA_ROOT layout of analytics/embeddings/train.py), after the loader has set every held-out game
aside: the contract tables (possessions, rosters) go through analytics.schema.validate; the frames (section 4), shots
(sections 10 and 10a) and aggregates (section 11) are checked against the columns this lane reads; every table gets
range checks, a null count per column and a duplicate-key count. Duplicates are COUNTED here and DROPPED by the feature
builder (frames on (frame, player_id)); nulls are counted and left to the feature builder's rules (a null shooter id is
resolved through the rosters or dropped; a null ball position gives no handler on that frame). Games are streamed one
at a time. The summary is printed as JSON and, with --out, written to OUT_ROOT/analytics/embeddings/validate.json.
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
from analytics.embeddings import features as F

log = logging.getLogger("analytics.embeddings.validate")

FRAME_COLUMNS = ["game", "period", "frame", "ts_ms", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id",
                 "player_id", "x", "y"]
SHOT_COLUMNS = ["sportvu_game", "period", "clock_s", "made", "shot_type"]
AGG_COLUMNS = ["player_id", "season", "endpoint", "metric", "value"]
RANGES = {  # column -> (lo, hi) inclusive; a value outside is counted, never dropped here
    "x": (-5.0, 99.0), "y": (-5.0, 55.0), "ball_x": (-5.0, 99.0), "ball_y": (-5.0, 55.0), "ball_z": (0.0, 40.0),
    "game_clock_s": (0.0, 720.0), "period": (1, 8), "court_x": (-5.0, 99.0), "court_y": (-5.0, 55.0),
    "clock_s": (0.0, 720.0), "shot_distance_ft": (0.0, 94.0),
}


def _nulls(df: pd.DataFrame, cols) -> dict:
    return {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}


def _ranges(df: pd.DataFrame) -> dict:
    out = {}
    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            n = int(((v < lo) | (v > hi)).sum())
            if n:
                out[c] = n
    return out


def check_frames(frames: pd.DataFrame) -> dict:
    """One game's section 4 frames -> dict(rows, missing_columns, nulls, out_of_range, duplicate_rows, players,
    frames, clock_jumps_over_2s, problems)."""
    missing = [c for c in FRAME_COLUMNS if c not in frames.columns]
    out = {"rows": int(len(frames)), "missing_columns": missing, "problems": []}
    if missing:
        out["problems"].append("frames lack column(s) %s" % missing)
        return out
    out["nulls"] = _nulls(frames, FRAME_COLUMNS)
    out["out_of_range"] = _ranges(frames)
    out["duplicate_rows"] = int(frames.duplicated(["frame", "player_id"]).sum())
    out["players"] = int(frames["player_id"].nunique())
    out["frames"] = int(frames["frame"].nunique())
    per = frames.drop_duplicates("frame").sort_values("frame")
    jumps = per.groupby("period")["game_clock_s"].diff().abs()
    out["clock_jumps_over_2s"] = int((jumps > 2.0).sum())
    if not frames["frame"].is_monotonic_increasing:
        out["problems"].append("frames are not ordered by frame")
    if out["nulls"].get("x") or out["nulls"].get("y"):
        out["problems"].append("null player positions")
    return out


def check_contract_table(df: pd.DataFrame, table: str) -> dict:
    problems = schema.validate(df, table)
    keys = {"possessions": ["sportvu_game", "possession"], "rosters": ["sportvu_game", "player_id"]}[table]
    out = {"rows": int(len(df)), "problems": problems, "nulls": _nulls(df, schema.columns(table)),
           "duplicate_rows": int(df.duplicated([k for k in keys if k in df.columns]).sum()) if all(k in df.columns for k in keys) else None}
    if table == "possessions" and not problems:
        bad = int((df["end_frame"] < df["start_frame"]).sum())
        if bad:
            out["problems"].append("%d possession(s) end before they start" % bad)
    return out


def check_shots(shots: pd.DataFrame, rosters: pd.DataFrame | None) -> dict:
    missing = [c for c in SHOT_COLUMNS if c not in shots.columns]
    out = {"rows": int(len(shots)), "missing_columns": missing, "problems": []}
    if missing:
        out["problems"].append("shots lack column(s) %s" % missing)
        return out
    out["nulls"] = _nulls(shots, list(shots.columns))
    out["out_of_range"] = _ranges(shots)
    keys = [c for c in ("sportvu_game", "period", "clock_s", "player_id", "player_slug") if c in shots.columns]
    out["duplicate_rows"] = int(shots.duplicated(keys).sum())
    out["games"] = int(shots["sportvu_game"].nunique())
    pid = F.resolve_shooter_ids(shots, rosters)
    out["unresolved_shooter_ids"] = int(pid.isna().sum())
    if "heldout" in shots.columns and shots["heldout"].fillna(False).astype(bool).any():
        out["problems"].append("held-out rows present after the loader")
    try:
        zones = F.derive_zone_basic(shots)
        out["zone_counts"] = {str(k): int(v) for k, v in zones.value_counts().items()}
    except ValueError as exc:
        out["problems"].append(str(exc))
    return out


def check_aggregates(agg: pd.DataFrame) -> dict:
    missing = [c for c in AGG_COLUMNS if c not in agg.columns]
    out = {"rows": int(len(agg)), "missing_columns": missing, "problems": []}
    if missing:
        out["problems"].append("aggregates lack column(s) %s" % missing)
        return out
    out["nulls"] = _nulls(agg, AGG_COLUMNS)
    out["seasons"] = sorted(agg["season"].astype(str).unique().tolist())
    out["endpoints"] = sorted(agg["endpoint"].astype(str).unique().tolist())
    keys = [c for c in ("player_id", "season", "season_type", "endpoint", "metric", "split_name", "split_value",
                        "pt_measure_type", "play_type", "type_grouping") if c in agg.columns]
    out["duplicate_rows"] = int(agg.duplicated(keys).sum())
    found = {}
    for metric, col, endpoints in F.AGG_METRICS:
        found[col] = int((agg["endpoint"].astype(str).str.lower().isin(endpoints) & (agg["metric"].astype(str).str.lower() == metric)).sum())
    out["metric_rows"] = found
    out["non_finite_values"] = int((~np.isfinite(pd.to_numeric(agg["value"], errors="coerce"))).sum())
    return out


def check_positions(pos: pd.DataFrame) -> dict:
    missing = [c for c in ("player_id", "season", "position") if c not in pos.columns]
    out = {"rows": int(len(pos)), "missing_columns": missing, "problems": []}
    if missing:   # the loader has already joined a player_slug table to the rosters; a table still lacking the keys is unusable
        out["problems"].append("positions lack column(s) %s" % missing)
        return out
    from analytics.embeddings.train import collapse_position
    collapsed = pos["position"].map(collapse_position)
    bad = sorted(set(pos.loc[collapsed.isna(), "position"].dropna().astype(str).unique()))
    if bad:
        out["problems"].append("positions that do not collapse to G, F or C (part before the first hyphen): %s" % bad)
    out["duplicate_rows"] = int(pos.duplicated(["player_id", "season"]).sum())
    out["counts"] = {str(k): int(v) for k, v in collapsed.value_counts().items()}
    return out


def validate_inputs(inputs: dict) -> dict:
    """The loader's dict -> summary per table, with per-game frame checks and the totals."""
    summary = {"source": inputs.get("source", ""), "dropped_heldout": inputs["dropped"], "games": {}, "totals": {}}
    tot = {"frames_rows": 0, "frames_duplicates": 0, "frames_null_ball": 0, "clock_jumps_over_2s": 0, "problems": 0}
    rosters = []
    for g in inputs["games"]:
        gid = g["sportvu_game"]
        frames = g["frames"]()
        fr = check_frames(frames)
        del frames
        po = check_contract_table(g["possessions"], "possessions")
        entry = {"frames": fr, "possessions": po}
        if g.get("rosters") is not None:
            entry["rosters"] = check_contract_table(g["rosters"], "rosters")
            rosters.append(g["rosters"])
        summary["games"][gid] = entry
        tot["frames_rows"] += fr["rows"]
        tot["frames_duplicates"] += fr.get("duplicate_rows", 0)
        tot["frames_null_ball"] += fr.get("nulls", {}).get("ball_x", 0)
        tot["clock_jumps_over_2s"] += fr.get("clock_jumps_over_2s", 0)
        tot["problems"] += sum(len(v["problems"]) for v in entry.values())
        for k, v in entry.items():
            for pr in v["problems"]:
                log.warning("%s %s: %s", gid, k, pr)
    ros = pd.concat(rosters, ignore_index=True) if rosters else None
    if inputs.get("shots") is not None:
        summary["shots"] = check_shots(inputs["shots"], ros)
        tot["problems"] += len(summary["shots"]["problems"])
    else:
        summary["shots"] = {"rows": 0, "problems": ["no shots table: shot features will be NaN"]}
    if inputs.get("aggregates") is not None:
        summary["aggregates"] = check_aggregates(inputs["aggregates"])
        tot["problems"] += len(summary["aggregates"]["problems"])
    else:
        summary["aggregates"] = {"rows": 0, "problems": ["no aggregates: tracking and shot features only"]}
    if inputs.get("positions") is not None:
        summary["positions"] = check_positions(inputs["positions"])
        tot["problems"] += len(summary["positions"]["problems"])
    else:
        summary["positions"] = {"rows": 0, "problems": ["no positions table: 5-NN position accuracy not computable"]}
    tot["games"] = len(inputs["games"])
    summary["totals"] = tot
    log.info("validate: %d game(s), %d frame rows, %d duplicate frame rows, %d problem(s)",
             tot["games"], tot["frames_rows"], tot["frames_duplicates"], tot["problems"])
    return summary


def main(argv=None) -> dict:
    from analytics.embeddings import train as _train
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data", help="DATA_ROOT (layout in analytics/embeddings/train.py)")
    ap.add_argument("--out", help="OUT_ROOT: writes analytics/embeddings/validate.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.fixtures and not a.data:
        ap.error("one of --fixtures or --data is required")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    inputs = _train.fixture_inputs(a.seed) if a.fixtures else _train.load_inputs(a.data)
    summary = validate_inputs(inputs)
    summary["fixture"] = bool(a.fixtures)
    text = json.dumps(summary, indent=1, default=str)
    if a.out:
        p = Path(a.out) / "analytics" / "embeddings" / "validate.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        summary["out"] = str(p)
    print(text)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

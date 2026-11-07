"""analytics/lineups/validate.py: checks of a stints table (docs/DATA_CONTRACT.md section 12) before any fit.

    python -m analytics.lineups.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seasons ...]

validate(stints) -> (clean, counts)
    schema      every section 12 column present (REQUIRED); a missing one is a problem, not a drop
    lineup_ok   share of rows with lineup_ok true; rows with n_home or n_away not 5 are DROPPED and counted
    duration    rows with duration_s outside (0, 720] are dropped and counted (a stint never crosses a period
                boundary, so 720 s is the longest possible; an overtime period is shorter)
    points      rows with points_home or points_away outside [0, MAX_POINTS] are dropped and counted
                (MAX_POINTS = 70: more than a point every ten seconds of the longest stint is a data error)
    duplicates  rows duplicated on (game_id, stint) or on (game_id, start_s, end_s) beyond the first are dropped
                and counted
    heldout     rows with heldout true, or a split that starts with heldout, are dropped and counted: they never
                fit anything (the collectors write them under heldout/, which nothing here reads)
    season      rows without a season are dropped and counted
Nothing is modified in place; counts is a flat dict of integers and shares written to
OUT_ROOT/analytics/lineups/validate.json by the CLI and printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analytics.lineups import PLAYER_COLUMNS, SEASONS, add_fixture_args, load_inputs

REQUIRED = ("game_id", "stint", "start_s", "end_s", "duration_s", "period_start", "period_end") + PLAYER_COLUMNS + (
    "home_ids", "away_ids", "n_home", "n_away", "lineup_ok", "points_home", "points_away", "score_home", "score_away")
MAX_DURATION_S = 720.0
MAX_POINTS = 70


def schema_problems(stints: pd.DataFrame) -> list[str]:
    """One message per section 12 column missing from the stints table."""
    return [f"missing column {c!r}" for c in REQUIRED if c not in stints.columns]


def validate(stints: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """-> (clean stints, counts). Raises ValueError when a required column is missing."""
    problems = schema_problems(stints)
    if problems:
        raise ValueError("stints table: " + "; ".join(problems))
    df = stints.copy()
    counts = {"rows_in": int(len(df)), "games_in": int(df["game_id"].nunique()), "schema_problems": 0,
              "lineup_ok_share": float(df["lineup_ok"].astype(bool).mean()) if len(df) else float("nan")}
    keep = pd.Series(True, index=df.index)

    def drop(name, bad):
        nonlocal keep
        bad = bad.astype(bool) & keep
        counts[name] = int(bad.sum())
        keep &= ~bad

    if "heldout" in df.columns or "split" in df.columns:
        h = df["heldout"].fillna(False).astype(bool) if "heldout" in df.columns else pd.Series(False, index=df.index)
        if "split" in df.columns:
            h |= df["split"].astype("string").fillna("").str.startswith("heldout").astype(bool)
        drop("dropped_heldout", h)
    else:
        counts["dropped_heldout"] = 0
    drop("dropped_no_season", df["season"].isna() if "season" in df.columns else pd.Series(True, index=df.index))
    drop("dropped_not_5v5", (df["n_home"].astype(float) != 5) | (df["n_away"].astype(float) != 5) | ~df["lineup_ok"].astype(bool))
    dur = df["duration_s"].astype(float)
    drop("dropped_duration", ~((dur > 0) & (dur <= MAX_DURATION_S)) | dur.isna())
    ph, pa = df["points_home"].astype(float), df["points_away"].astype(float)
    drop("dropped_points", ~((ph >= 0) & (ph <= MAX_POINTS) & (pa >= 0) & (pa <= MAX_POINTS)))
    dup = df.duplicated(["game_id", "stint"]) | df.duplicated(["game_id", "start_s", "end_s"])
    drop("dropped_duplicate", dup)
    clean = df[keep].reset_index(drop=True)
    counts.update({"rows_out": int(len(clean)), "games_out": int(clean["game_id"].nunique()),
                   "duration_s_total": float(clean["duration_s"].astype(float).sum()) if len(clean) else 0.0,
                   "seasons": sorted(map(str, clean["season"].unique())) if "season" in clean.columns and len(clean) else []})
    if len(clean):
        counts["per_season_rows"] = {str(k): int(v) for k, v in clean.groupby("season").size().items()}
    return clean, counts


def parse_args(argv=None):
    """The CLI's arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic stints (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT (default DATA_ROOT)")
    ap.add_argument("--seasons", nargs="*", default=list(SEASONS), help="--data only: seasons to read (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Validate the fixture or DATA_ROOT stints, write validate.json under OUT_ROOT and print the counts."""
    a = parse_args(argv)
    inputs = load_inputs(a.fixtures, a.data, a.seasons, a.seed, a.fixture_generator, a.n_games)
    if len(inputs["stints"]) == 0:
        raise SystemExit("validate: no stints rows found")
    _clean, counts = validate(inputs["stints"])
    counts["fixture"] = inputs["fixture"]
    out_root = a.out or (a.data if not a.fixtures else None)
    if out_root:
        p = Path(out_root) / "analytics" / "lineups" / "validate.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(counts, indent=1, default=float))
        counts["written"] = str(p)
    print("validate%s: %s" % (" (fixture output, not a result)" if inputs["fixture"] else "", json.dumps(counts, default=float)))
    return counts


if __name__ == "__main__":
    main()

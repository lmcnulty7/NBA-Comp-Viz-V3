"""analytics/lineups/features.py: the stint feature frame of the lineup models.

    python -m analytics.lineups.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seasons ...]

features(stints, games=None) -> one row per validated stint (analytics.lineups.validate) with
    keys        game_id, season, stint, game_date (when present)
    FEATURES    home_1 .. home_5, away_1 .. away_5 (player ids, the categorical inputs of the indicator
                design), possessions (the weight: a `possessions` column of the table when it carries one,
                else duration_s / 28.8, one possession per team per 28.8 s, 100 per 48 minutes)
    lineups     lineup_home, lineup_away (sorted ids joined with |), home_team_id, away_team_id (from the games
                table; null without it)
    target      diff_per100 = (points_home - points_away) / possessions * 100 (TARGET), points_diff
The returned frame passes analytics.schema.assert_time_forward (against the possessions table: the union of
every target and post-outcome column of the contract) and assert_no_leak, which refuses a design holding
FORBIDDEN columns (the points, scores and the target). feature_columns() lists FEATURES.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analytics import schema
from analytics.lineups import NO_STINTS_HINT, PLAYER_COLUMNS, SEASONS, SECONDS_PER_POSS, add_fixture_args, lineup_id, load_inputs, out_dir
from analytics.lineups.validate import validate

FEATURES = PLAYER_COLUMNS + ("possessions",)
TARGET = "diff_per100"
FORBIDDEN = frozenset({"points_home", "points_away", "score_home", "score_away", "points_diff", TARGET, "points", "made"})
KEYS = ("game_id", "season", "stint", "game_date")


def feature_columns() -> list[str]:
    """The design columns (FEATURES)."""
    return list(FEATURES)


def assert_no_leak(X) -> None:
    """ValueError when a design holds a FORBIDDEN column (the points, scores or the target)."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("lineups.features: forbidden columns in the design: %s" % bad)


def possessions(stints: pd.DataFrame) -> pd.Series:
    """The stint's possessions per team: the table's own column when present, else duration_s / 28.8."""
    if "possessions" in stints.columns and stints["possessions"].notna().all():
        return stints["possessions"].astype(float)
    return stints["duration_s"].astype(float) / SECONDS_PER_POSS


def features(stints: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """The stint feature frame of the module docstring, from the validated stints and the optional games table."""
    clean, _counts = validate(stints)
    out = pd.DataFrame(index=clean.index)
    for c in KEYS:
        if c in clean.columns:
            out[c] = clean[c]
    out["game_id"] = out["game_id"].astype(str)
    for c in PLAYER_COLUMNS:
        out[c] = clean[c].astype("int64")
    out["possessions"] = possessions(clean).to_numpy()
    out["lineup_home"] = [lineup_id(r) for r in clean[list(PLAYER_COLUMNS[:5])].to_numpy()]
    out["lineup_away"] = [lineup_id(r) for r in clean[list(PLAYER_COLUMNS[5:])].to_numpy()]
    out["home_team_id"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["away_team_id"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if games is not None and len(games) and "game_id" in games.columns:
        g = games.drop_duplicates("game_id").set_index(games["game_id"].astype(str))
        out["home_team_id"] = out["game_id"].map(g["home_team_id"]).astype("Int64")
        out["away_team_id"] = out["game_id"].map(g["away_team_id"]).astype("Int64")
    out["points_diff"] = (clean["points_home"].astype(float) - clean["points_away"].astype(float)).to_numpy()
    out[TARGET] = out["points_diff"] / out["possessions"] * 100.0
    out = out.reset_index(drop=True)
    schema.assert_time_forward(out, "possessions")
    assert_no_leak(out[feature_columns()])
    return out


def parse_args(argv=None):
    """The CLI's arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic stints (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT (default DATA_ROOT; nothing is written for --fixtures without it)")
    ap.add_argument("--seasons", nargs="*", default=list(SEASONS), help="--data only: seasons to read (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Build the feature frame from the fixture or DATA_ROOT stints, write it per season under OUT_ROOT and print a summary."""
    a = parse_args(argv)
    inputs = load_inputs(a.fixtures, a.data, a.seasons, a.seed, a.fixture_generator, a.n_games, out_root=a.out)
    if len(inputs["stints"]) == 0:
        raise SystemExit("features: no stints rows found " + NO_STINTS_HINT)
    f = features(inputs["stints"], inputs["games"])
    out_root = a.out or (a.data if not a.fixtures else None)
    written = []
    if out_root:
        for season, grp in f.groupby("season"):
            p = out_dir(out_root, str(season)) / "features.parquet"
            grp.reset_index(drop=True).to_parquet(p, index=False)
            written.append(str(p))
    s = {"rows": int(len(f)), "games": int(f["game_id"].nunique()), "features": feature_columns(), "target": TARGET,
         "possessions_total": float(f["possessions"].sum()), "diff_per100_mean": float(np.average(f[TARGET], weights=f["possessions"])),
         "written": written, "fixture": inputs["fixture"]}
    print("features%s: %s" % (" (fixture output, not a result)" if inputs["fixture"] else "", s))
    return s


if __name__ == "__main__":
    main()

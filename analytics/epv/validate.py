"""analytics/epv/validate.py: input checks for the EPV lane, and the loaders every EPV CLI shares.

    python -m analytics.epv.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]

Per input table (states_1hz, possessions, rosters, shot_frames: analytics.schema.validate; shots: the section 10
and 10a columns the priors need) the report lists schema problems, null counts per column, out-of-range counts
(RANGES), duplicate keys and what was dropped. Handling, with every count logged in the report:
  - duplicate keys (states: sportvu_game, possession, frame; possessions: sportvu_game, possession; rosters:
    sportvu_game, player_id; shots: sportvu_game, period, clock_s, shooter) keep the first row;
  - a states row whose target is outside 0 to 4, whose elapsed time is negative or whose non-nullable column is
    null is dropped; coordinates outside the court by more than COURT_SLACK feet are logged, not dropped
    (SportVU positions do run a few feet outside the lines);
  - nothing else is altered: no imputation here (the models handle missing values themselves).

DATA_ROOT layout (the owner's Mac): DATA_ROOT/sportvu_contract/<game>.parquet and <game>_pbp.parquet (held-out
under heldout/), DATA_ROOT/analytics/<table>/<season>/<game>.parquet (held-out under heldout/<season>/) for
possessions, states_1hz, shot_frames, rosters, and DATA_ROOT/collect/bref_shots/shotchart/<season>.parquet
(held-out under collect/bref_shots/heldout/shotchart/); collect/shots/shotchartdetail/ is unioned when present.
Every loaded table carries a bool `heldout` column from the directory it came from, or-ed with the ids of
sportvu/splits.json, so a held-out game can never be mistaken for a training one by its path alone.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema

COURT_SLACK = 5.0
RANGES = {"ball_x": (-COURT_SLACK, 94 + COURT_SLACK), "ball_y": (-COURT_SLACK, 50 + COURT_SLACK), "ball_z": (-1.0, 40.0),
          "shot_clock_s": (0.0, 24.5), "game_clock_s": (0.0, 720.0 + 1e-3), "elapsed_in_possession_s": (0.0, 120.0),
          "handler_nearest_def_dist_ft": (0.0, 110.0), "points_to_end": (0, 4)}
for _side in ("off", "def"):
    for _i in range(1, 6):
        RANGES["%s%d_x" % (_side, _i)] = (-COURT_SLACK, 94 + COURT_SLACK)
        RANGES["%s%d_y" % (_side, _i)] = (-COURT_SLACK, 50 + COURT_SLACK)
KEYS = {"states_1hz": ["sportvu_game", "possession", "frame"], "possessions": ["sportvu_game", "possession"],
        "rosters": ["sportvu_game", "player_id"], "shot_frames": ["sportvu_game", "event"]}
SHOT_COLUMNS = ("sportvu_game", "period", "clock_s", "made", "loc_x", "loc_y")
TABLE_DIRS = ("possessions", "states_1hz", "shot_frames", "rosters")


# ── held-out ids ───────────────────────────────────────────────────────────────────────────────────
def heldout_by_id(games: pd.Series) -> pd.Series:
    """True where the game id carries a held-out token of sportvu/splits.json or ends at the CLE arena."""
    from sportvu import splits
    toks = splits.heldout_tokens()
    g = games.astype(str)
    return g.map(lambda s: any(t in s for t in toks) or s.endswith(".at." + splits.HELDOUT_ARENA)).astype(bool)


def with_heldout(df: pd.DataFrame, flag) -> pd.DataFrame:
    out = df.copy()
    base = pd.Series(np.asarray(flag, dtype=bool) if not np.isscalar(flag) else bool(flag), index=out.index)
    if "heldout" in out.columns:
        base = base | out["heldout"].fillna(False).astype(bool)
    if "sportvu_game" in out.columns and len(out):
        base = base | heldout_by_id(out["sportvu_game"]).to_numpy()
    out["heldout"] = base.astype(bool).to_numpy()
    return out


# ── checks ─────────────────────────────────────────────────────────────────────────────────────────
def _range_counts(df: pd.DataFrame) -> dict:
    out = {}
    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            n = int(((v < lo) | (v > hi)).sum())
            if n:
                out[c] = n
    return out


def _null_counts(df: pd.DataFrame) -> dict:
    n = df.isna().sum()
    return {c: int(v) for c, v in n.items() if v}


def check_table(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict]:
    """schema.validate plus duplicate-key handling for a contract table -> (cleaned frame, report)."""
    rep = {"table": table, "rows_in": int(len(df)), "schema_problems": schema.validate(df, table) if len(df) else [],
           "nulls": _null_counts(df), "out_of_range": _range_counts(df), "duplicates_dropped": 0, "rows_dropped": 0}
    keys = [k for k in KEYS.get(table, []) if k in df.columns]
    out = df
    if keys and len(df):
        dup = df.duplicated(keys, keep="first")
        rep["duplicates_dropped"] = int(dup.sum())
        out = df.loc[~dup]
    rep["rows_out"] = int(len(out))
    return out.reset_index(drop=True), rep


def check_states(states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """states_1hz: check_table, then drop rows with a target outside 0 to 4, a negative elapsed time or a null in a
    non-nullable column (counts in the report)."""
    out, rep = check_table(states, "states_1hz")
    if len(out) == 0:
        return out, rep
    bad = pd.Series(False, index=out.index)
    if "points_to_end" in out.columns:
        v = pd.to_numeric(out["points_to_end"], errors="coerce")
        bad |= v.isna() | (v < 0) | (v > 4)
    if "elapsed_in_possession_s" in out.columns:
        bad |= pd.to_numeric(out["elapsed_in_possession_s"], errors="coerce").fillna(-1) < 0
    non_null = [c for c in schema.columns("states_1hz") if c in out.columns and not schema.is_nullable("states_1hz", c)]
    bad |= out[non_null].isna().any(axis=1)
    rep["rows_dropped"] = int(bad.sum())
    out = out.loc[~bad].reset_index(drop=True)
    rep["rows_out"] = int(len(out))
    rep["games"] = int(out["sportvu_game"].nunique()) if "sportvu_game" in out.columns else 0
    rep["possessions"] = int(out.groupby(["sportvu_game", "possession"]).ngroups) if len(out) else 0
    return out, rep


def check_shots(shots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The shots table (section 10 or 10a): required columns, nulls, duplicates on (game, period, clock, shooter)."""
    rep = {"table": "shots", "rows_in": int(len(shots)), "schema_problems": [], "nulls": _null_counts(shots),
           "out_of_range": {}, "duplicates_dropped": 0, "rows_dropped": 0}
    for c in SHOT_COLUMNS:
        if c not in shots.columns:
            rep["schema_problems"].append("missing column %r" % c)
    out = shots
    if len(shots) and not rep["schema_problems"]:
        shooter = "player_id" if "player_id" in shots.columns and shots["player_id"].notna().any() else (
            "player_slug" if "player_slug" in shots.columns else "player")
        keys = [k for k in ("sportvu_game", "period", "clock_s", shooter) if k in shots.columns]
        dup = shots.duplicated(keys, keep="first")
        rep["duplicates_dropped"] = int(dup.sum())
        out = shots.loc[~dup]
        made = out["made"]
        bad = made.isna()
        rep["rows_dropped"] = int(bad.sum())
        out = out.loc[~bad]
    rep["rows_out"] = int(len(out))
    return out.reset_index(drop=True), rep


def validate_all(tables: dict) -> tuple[dict, dict]:
    """{name: frame or None} -> ({name: cleaned frame}, {name: report})."""
    clean, reports = {}, {}
    for name, df in tables.items():
        if df is None or name == "games":
            clean[name] = df
            continue
        if name == "states_1hz":
            clean[name], reports[name] = check_states(df)
        elif name == "shots":
            clean[name], reports[name] = check_shots(df)
        elif name in schema.TABLES:
            clean[name], reports[name] = check_table(df, name)
        else:
            clean[name] = df
    return clean, reports


def format_report(reports: dict, title: str = "EPV input validation") -> str:
    L = ["# %s" % title, "", "| table | rows in | schema problems | nulls | out of range | duplicates dropped | rows dropped | rows out |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in reports.items():
        nulls = ", ".join("%s %d" % kv for kv in sorted(r["nulls"].items())) or "none"
        rng = ", ".join("%s %d" % kv for kv in sorted(r["out_of_range"].items())) or "none"
        L.append("| %s | %d | %d | %s | %s | %d | %d | %d |" % (name, r["rows_in"], len(r["schema_problems"]), nulls, rng,
                                                               r["duplicates_dropped"], r["rows_dropped"], r["rows_out"]))
    for name, r in reports.items():
        if r["schema_problems"]:
            L += ["", "Schema problems in %s:" % name] + ["- %s" % p for p in r["schema_problems"]]
    return "\n".join(L) + "\n"


# ── loaders ────────────────────────────────────────────────────────────────────────────────────────
def _read_dir(base: Path, held: bool, season: str | None) -> list:
    out = []
    if not base.exists():
        return out
    for sd in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "heldout"):
        if season and sd.name != season:
            continue
        for f in sorted(sd.glob("*.parquet")):
            df = pd.read_parquet(f)
            if "sportvu_game" not in df.columns:
                df["sportvu_game"] = f.stem
            out.append(with_heldout(df, held))
    return out


def load_root(data_root, season: str | None = None) -> dict:
    """The EPV input tables under DATA_ROOT -> {states_1hz, possessions, rosters, shot_frames, shots, games};
    a table with no file is None. games is [(sportvu_game, heldout)] from the states files."""
    root = Path(data_root)
    out = {}
    for table in TABLE_DIRS:
        parts = _read_dir(root / "analytics" / table, False, season) + _read_dir(root / "analytics" / table / "heldout", True, season)
        out[table] = pd.concat(parts, ignore_index=True) if parts else None
    shots = []
    for fam in ("bref_shots/shotchart", "shots/shotchartdetail"):
        d = root / "collect" / fam
        hd = root / "collect" / fam.split("/")[0] / "heldout" / fam.split("/")[1]
        for base, held in ((d, False), (hd, True)):
            if base.exists():
                for f in sorted(base.glob("*.parquet")):
                    if season and f.stem != season:
                        continue
                    shots.append(with_heldout(pd.read_parquet(f), held))
    out["shots"] = pd.concat(shots, ignore_index=True) if shots else None
    st = out["states_1hz"]
    out["games"] = sorted(set(zip(st["sportvu_game"].astype(str), st["heldout"].astype(bool)))) if st is not None else []
    return out


def load_fixtures(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int | None = None) -> dict:
    """The synthetic season of tests/_fx_epv.py in the load_root shape (heldout flags from the generator)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests import _fx_epv as fx
    season = fx.synth_season(seed, n_games=n_games, heldout_every=heldout_every)
    held = {g: h for g, h in season["games"]}
    out = {}
    for table in TABLE_DIRS + ("shots",):
        df = season[table]
        flag = df["sportvu_game"].map(held).fillna(False).to_numpy() if len(df) else np.zeros(0, dtype=bool)
        out[table] = with_heldout(df, flag)
    out["games"] = list(season["games"])
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true")
    g.add_argument("--data")
    ap.add_argument("--out", help="OUT_ROOT: the report goes to OUT_ROOT/analytics/epv_validate.md")
    ap.add_argument("--season", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    tables = load_fixtures(a.seed, n_games=a.games) if a.fixtures else load_root(a.data, a.season)
    _clean, reports = validate_all(tables)
    title = "EPV input validation (%s)" % ("fixture output, not a result" if a.fixtures else a.data)
    text = format_report(reports, title)
    print(text)
    out_root = Path(a.out) if a.out else (Path(tempfile.gettempdir()) / "epv_fixture_root" if a.fixtures else Path(a.data))
    path = out_root / "analytics" / "epv_validate.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("written %s" % path)
    return reports


if __name__ == "__main__":
    main()

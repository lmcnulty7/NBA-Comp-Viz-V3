"""analytics/epv/validate.py: input checks for the EPV lane, and the loaders every EPV CLI shares.

    python -m analytics.epv.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--fixture-games 4]
                                     [--fixture-possessions 40] [--fixture-heldout-every 4]

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
Every loaded table carries a bool `heldout` column from the directory it came from, or-ed with
analytics.eval.heldout_mask (the ids of sportvu/splits.json and the held-out arena), so a held-out game can never be
mistaken for a training one by its path alone.

--fixtures builds the synthetic season of analytics.fixtures (one league, so teams and players recur across games)
in memory: fixture output, never a result.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import heldout_mask

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


# ── held-out flag ──────────────────────────────────────────────────────────────────────────────────
def with_heldout(df: pd.DataFrame, flag) -> pd.DataFrame:
    """df with a bool heldout column: the given flag (scalar or per row) or-ed with any existing column and with
    analytics.eval.heldout_mask of the game ids."""
    out = df.copy()
    base = pd.Series(np.asarray(flag, dtype=bool) if not np.isscalar(flag) else bool(flag), index=out.index)
    if "heldout" in out.columns:
        base = base | out["heldout"].fillna(False).astype(bool)
    if "sportvu_game" in out.columns and len(out):
        base = base | heldout_mask(out["sportvu_game"])
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
        if df is None or name in ("games", "sources"):
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
    """The validation reports as a markdown table (one row per table) plus the schema problems."""
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


def _has_parquet(base: Path) -> bool:
    return base.exists() and any(base.rglob("*.parquet"))


def load_root(data_root, season: str | None = None, out_root=None) -> dict:
    """The EPV input tables -> {states_1hz, possessions, rosters, shot_frames, shots, games, sources}; a table with
    no file is None. games is [(sportvu_game, heldout)] from the states files.

    Each Phase 0 table (analytics/<table>/<season>/ and analytics/<table>/heldout/<season>/) is read from OUT_ROOT
    when OUT_ROOT holds any parquet for it, else from DATA_ROOT: `states --data X --out Y` writes states_1hz under Y,
    and `features`/`train --data X --out Y` then find them there while possessions, rosters and shot_frames still
    come from X. sources records the root used per table. The shots tables (collect/) come from DATA_ROOT."""
    root = Path(data_root)
    out, sources = {}, {}
    for table in TABLE_DIRS:
        base = root / "analytics" / table
        if out_root is not None and _has_parquet(Path(out_root) / "analytics" / table):
            base = Path(out_root) / "analytics" / table
        parts = _read_dir(base, False, season) + _read_dir(base / "heldout", True, season)
        out[table] = pd.concat(parts, ignore_index=True) if parts else None
        sources[table] = str(base.parent.parent) if parts else None
    out["sources"] = sources
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


FIXTURE_TABLES = ("frames", "events", "shots", "rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")


def fixture_season(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40) -> dict:
    """The synthetic season of analytics.fixtures.synth_season (one league) as concatenated tables: keys
    FIXTURE_TABLES (one DataFrame each) and games [(sportvu_game, heldout)] in generation order."""
    from analytics import fixtures
    games = fixtures.synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    out = {k: pd.concat([g[k] for g in games], ignore_index=True) for k in FIXTURE_TABLES}
    out["games"] = [(str(g["meta"]["sportvu_game"]), bool(g["meta"]["heldout"])) for g in games]
    return out


def load_fixtures(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40) -> dict:
    """fixture_season in the load_root shape: states_1hz, possessions, rosters, shot_frames, shots (each with the
    heldout flag of the generator or-ed with the ids), games, and sources (None: nothing read from disk)."""
    season = fixture_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    held = {g: h for g, h in season["games"]}
    out = {}
    for table in TABLE_DIRS + ("shots",):
        df = season[table]
        flag = df["sportvu_game"].map(held).fillna(False).to_numpy(dtype=bool) if len(df) else np.zeros(0, dtype=bool)
        out[table] = with_heldout(df, flag)
    out["games"] = list(season["games"])
    out["sources"] = {t: None for t in TABLE_DIRS}
    return out


def add_fixture_args(ap: argparse.ArgumentParser, games: int = 4) -> None:
    """The fixture-size flags every EPV CLI takes (canonical spelling first; --games is the older alias)."""
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default 0)")
    ap.add_argument("--fixture-games", "--games", dest="fixture_games", type=int, default=games,
                    help="fixture games with --fixtures (default %d)" % games)
    ap.add_argument("--fixture-possessions", dest="fixture_possessions", type=int, default=40,
                    help="possessions per fixture game (default 40)")
    ap.add_argument("--fixture-heldout-every", dest="fixture_heldout_every", type=int, default=4,
                    help="hold out every k-th fixture game, the first as 12.16.2015.PHX.at.GSW (default 4)")


def load_inputs(a) -> dict:
    """The input tables named by the parsed CLI flags: the fixture season or DATA_ROOT (OUT_ROOT first)."""
    if a.fixtures:
        return load_fixtures(a.seed, n_games=a.fixture_games, heldout_every=a.fixture_heldout_every,
                             n_possessions=a.fixture_possessions)
    return load_root(a.data, a.season, out_root=a.out)


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The validate CLI flags."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="the synthetic season of analytics.fixtures (fixture output)")
    g.add_argument("--data", help="DATA_ROOT (layout in the module docstring)")
    ap.add_argument("--out", help="OUT_ROOT: the report goes to OUT_ROOT/analytics/epv_validate.md")
    ap.add_argument("--season", default=None, help="only this season directory, e.g. 2015-16 (default: every season)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Validate the inputs, print the report and write it under OUT_ROOT (a scratch directory with --fixtures)."""
    a = parse_args(argv)
    tables = load_inputs(a)
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

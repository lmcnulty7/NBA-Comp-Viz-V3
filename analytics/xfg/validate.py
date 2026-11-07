"""analytics/xfg/validate.py: input validation and cleaning for the xFG upgrade, plus the input loaders.

    python -m analytics.xfg.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

Inputs (docs/DATA_CONTRACT.md): the shots table (sections 10 and 10a), shot_frames (17), rosters (16),
possessions (19) and the frames (4) and play-by-play (5) tables. For the contract tables of analytics/schema.py
(rosters, shot_frames, possessions) ``schema.validate`` runs per table; the collector tables (frames, pbp, shots)
have no schema.py entry, so they are checked against the column lists and ranges below.

Checks and cleaning (every dropped row is counted; counts of rows in and out are logged and returned):
    frames        ball_x, ball_y inside the court (0 to 94 by 0 to 50 ft, may be null); x, y within a 5 ft margin
                  of it (players step out of bounds),
                  game_clock_s in 0 to 720, shot_clock_s in 0 to 24 or null; duplicates by (game, frame, player_id)
    pbp (events)  clock_s in 0 to 720; duplicates by (game, event)
    shots         loc_x in -250 to 250 and loc_y in -55 to 420 tenths of a foot (the half court around the basket:
                  5.25 ft behind it, 41.75 ft toward half court), clock_s in 0 to 720, period >= 1, made not null;
                  duplicates by (sportvu_game, event) when the table has events, else by
                  (sportvu_game, period, clock_s, player)
    shot_frames   schema.validate, release_frame >= -1, clock_s in 0 to 720; duplicates by (sportvu_game, event)
    rosters       schema.validate; duplicates by (sportvu_game, player_id)
    possessions   schema.validate; duplicates by (sportvu_game, possession)
Rows that fail a range check or a non-nullable column are dropped; a schema problem is reported, not fixed
(the loader casts what it can). Nothing is imputed here: null contest inputs stay null and become
has_contest 0 downstream.

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet and <game>_pbp.parquet         held-out games under heldout/
    DATA_ROOT/collect/bref_shots/shotchart/<season>.parquet                  + heldout/shotchart/<season>.parquet
    DATA_ROOT/analytics/<table>/<season>/<game>.parquet                       + heldout/<season>/<game>.parquet
Frames are large (about 700k rows a game), so ``load_data_root`` returns them as a loader callable
(sportvu_game -> cleaned DataFrame) that reads one game at a time; the fixture tables return a dict.
The ``heldout`` flag of a shots row is the one the collector wrote OR the file location (heldout/) OR a held-out
token of sportvu/splits.json in its game id; the three are reconciled in ``load_data_root``.

The fixture season (``--fixtures``) comes from analytics.fixtures.synth_season; production code never imports
tests/. Held-out detection by game id is analytics.eval.heldout_mask (splits tokens plus the held-out arena).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import fixtures as _fixtures
from analytics import schema
from analytics.eval import heldout_mask

log = logging.getLogger("analytics.xfg.validate")

COURT_X, COURT_Y = (0.0, 94.0), (0.0, 50.0)
PLAYER_MARGIN_FT = 5.0       # players step out of bounds; the ball is kept to the court
CLOCK = (0.0, 720.0)
SHOT_CLOCK = (0.0, 24.0)
LOC_X, LOC_Y = (-250.0, 250.0), (-55.0, 420.0)
FRAME_COLUMNS = ["game", "period", "frame", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "team_id",
                 "player_id", "x", "y"]
PBP_COLUMNS = ["game", "event", "period", "clock_s", "kind"]
SHOT_COLUMNS = ["sportvu_game", "period", "clock_s", "made", "loc_x", "loc_y"]
CONTRACT_TABLES = ("shot_frames", "rosters", "possessions")
SEASON = "2015-16"


# ----------------------------------------------------------------------------- fixtures

def fixture_season(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40) -> list:
    """The synthetic season (analytics.fixtures.synth_season, list of per-game dicts) the --fixtures runs use."""
    return _fixtures.synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)


def _game_id(g: dict) -> str:
    """The sportvu_game of a per-game fixture dict (meta of analytics.fixtures, or a top-level key)."""
    return str(g["meta"]["sportvu_game"]) if "meta" in g else str(g["sportvu_game"])


def tables_from_season(season: list) -> dict:
    """Per-game dicts -> one dict of concatenated tables; frames as a dict sportvu_game -> DataFrame."""
    out = {"frames": {}, "fixture": True}
    for k in ("events", "shots", "rosters", "shot_frames", "possessions"):
        out[k] = pd.concat([g[k] for g in season], ignore_index=True)
    for g in season:
        out["frames"][_game_id(g)] = g["frames"]
    out["shots"]["heldout"] = out["shots"]["heldout"].astype(bool)
    return out


def fixture_tables(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40) -> dict:
    """fixture_season as one dict of concatenated tables (tables_from_season)."""
    return tables_from_season(fixture_season(seed, n_games, heldout_every, n_possessions))


# ----------------------------------------------------------------------------- DATA_ROOT

def _read_many(paths, cols=None) -> pd.DataFrame:
    """The parquet files at ``paths`` concatenated (an empty frame with ``cols`` when there are none)."""
    frames = [pd.read_parquet(p, columns=cols) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols or [])


def _contract_files(root: Path, table: str, season: str) -> list:
    base = root / "analytics" / table
    out = []
    for held, d in ((False, base / season), (True, base / "heldout" / season)):
        for p in sorted(d.glob("*.parquet")) if d.exists() else []:
            out.append((p, held))
    return out


def load_data_root(root, season: str = SEASON) -> dict:
    """The DATA_ROOT layout -> dict(shots, shot_frames, rosters, possessions, events, frames=loader, fixture=False).

    Game ids found under heldout/ or carrying a held-out token of sportvu/splits.json (analytics.eval.heldout_mask)
    are marked heldout on the shots table whatever the collector wrote."""
    root = Path(root)
    held_ids: set = set()
    out = {"fixture": False}
    for table in CONTRACT_TABLES:
        parts = []
        for p, held in _contract_files(root, table, season):
            df = pd.read_parquet(p)
            parts.append(df)
            if held:
                held_ids |= set(df["sportvu_game"].astype(str).unique())
        out[table] = pd.concat(parts, ignore_index=True) if parts else schema.empty(table)
    shots_dir = root / "collect" / "bref_shots"
    shots_parts = []
    for p, held in ((shots_dir / "shotchart" / ("%s.parquet" % season), False),
                    (shots_dir / "heldout" / "shotchart" / ("%s.parquet" % season), True)):
        if p.exists():
            df = pd.read_parquet(p)
            if held:
                held_ids |= set(df["sportvu_game"].astype(str).unique())
            shots_parts.append(df)
    if not shots_parts:
        raise SystemExit("validate: no shots table under %s" % shots_dir)
    shots = pd.concat(shots_parts, ignore_index=True)
    g = shots["sportvu_game"].astype(str)
    by_id = heldout_mask(g)
    flag = shots["heldout"].fillna(False).astype(bool) if "heldout" in shots.columns else pd.Series(False, index=shots.index)
    shots["heldout"] = (flag | g.isin(held_ids).to_numpy() | by_id).astype(bool).to_numpy()
    out["shots"] = shots
    sv = root / "sportvu_contract"
    pbp = [p for d in (sv, sv / "heldout") if d.exists() for p in sorted(d.glob("*_pbp.parquet"))]
    out["events"] = _read_many(pbp) if pbp else pd.DataFrame(columns=PBP_COLUMNS)

    def frames_loader(game: str):
        for d in (sv, sv / "heldout"):
            p = d / ("%s.parquet" % game)
            if p.exists():
                df = pd.read_parquet(p, columns=FRAME_COLUMNS)
                df, s = clean_frames(df)
                log.info("frames %s: %d rows in, %d out", game, s["rows_in"], s["rows_out"])
                return df
        log.warning("frames: no file for %s under %s", game, sv)
        return None

    out["frames"] = frames_loader
    return out


# ----------------------------------------------------------------------------- checks

def _summary(table: str, n_in: int, n_out: int, **extra) -> dict:
    d = {"table": table, "rows_in": int(n_in), "rows_out": int(n_out), "dropped_missing": 0, "dropped_range": 0,
         "dropped_duplicate": 0, "schema_problems": []}
    d.update(extra)
    return d


def _in_range(s: pd.Series, lo: float, hi: float, nullable: bool = False) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce")
    ok = (v >= lo) & (v <= hi)
    return (ok | v.isna()).to_numpy() if nullable else ok.fillna(False).to_numpy()


def _apply(df: pd.DataFrame, table: str, required: list, range_masks: list, dup_keys: list, problems=None) -> tuple:
    s = _summary(table, len(df), len(df))
    s["schema_problems"] = list(problems or [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        s["schema_problems"].append("missing column(s) %s" % missing_cols)
        s["rows_out"] = 0
        return df.iloc[0:0], s
    keep = df[required].notna().all(axis=1).to_numpy()
    s["dropped_missing"] = int((~keep).sum())
    df = df[keep]
    rng = np.ones(len(df), dtype=bool)
    for col, lo, hi, nullable in range_masks:
        if col in df.columns:
            rng &= _in_range(df[col], lo, hi, nullable)
    s["dropped_range"] = int((~rng).sum())
    df = df[rng]
    keys = [k for k in dup_keys if k in df.columns]
    if keys:
        dup = df.duplicated(keys, keep="first").to_numpy()
        s["dropped_duplicate"] = int(dup.sum())
        df = df[~dup]
    s["rows_out"] = int(len(df))
    log.info("%s: %d rows in, %d out (missing %d, range %d, duplicate %d)", table, s["rows_in"], s["rows_out"],
             s["dropped_missing"], s["dropped_range"], s["dropped_duplicate"])
    return df.reset_index(drop=True), s


def clean_frames(df: pd.DataFrame) -> tuple:
    """Frames (section 4) -> (cleaned frame, summary): court and clock ranges, duplicates by (game, frame, player_id)."""
    required = ["game", "period", "frame", "game_clock_s", "player_id", "x", "y"]
    ranges = [("x", COURT_X[0] - PLAYER_MARGIN_FT, COURT_X[1] + PLAYER_MARGIN_FT, False),
              ("y", COURT_Y[0] - PLAYER_MARGIN_FT, COURT_Y[1] + PLAYER_MARGIN_FT, False),
              ("ball_x", *COURT_X, True), ("ball_y", *COURT_Y, True),
              ("game_clock_s", *CLOCK, False), ("shot_clock_s", *SHOT_CLOCK, True)]
    return _apply(df, "frames", required, ranges, ["game", "frame", "player_id"])


def clean_events(df: pd.DataFrame) -> tuple:
    """Play-by-play (section 5) -> (cleaned frame, summary): clock and period ranges, duplicates by (game, event)."""
    return _apply(df, "events", PBP_COLUMNS, [("clock_s", *CLOCK, False), ("period", 1, 20, False)], ["game", "event"])


def clean_shots(df: pd.DataFrame) -> tuple:
    """Shots (sections 10 and 10a) -> (cleaned frame, summary): location, clock and period ranges, duplicates."""
    keys = ["sportvu_game", "event"] if "event" in df.columns and df["event"].notna().all() else ["sportvu_game", "period", "clock_s", "player"]
    ranges = [("loc_x", *LOC_X, False), ("loc_y", *LOC_Y, False), ("clock_s", *CLOCK, False), ("period", 1, 20, False)]
    return _apply(df, "shots", SHOT_COLUMNS, ranges, keys)


def clean_contract(df: pd.DataFrame, table: str) -> tuple:
    """A schema.py contract table -> (cleaned and cast frame, summary); schema problems are reported, not fixed."""
    problems = schema.validate(df, table)
    keys = {"shot_frames": ["sportvu_game", "event"], "rosters": ["sportvu_game", "player_id"],
            "possessions": ["sportvu_game", "possession"]}[table]
    required = [c for c in schema.columns(table) if c not in schema.NULLABLE[table]]
    ranges = {"shot_frames": [("clock_s", *CLOCK, False), ("release_frame", -1, 1e12, False)],
              "rosters": [], "possessions": [("start_clock_s", *CLOCK, False), ("start_frame", 0, 1e12, False)]}[table]
    df2, s = _apply(df, table, required, ranges, keys, problems)
    try:
        df2 = schema.cast(df2, table)
    except (KeyError, ValueError) as exc:
        s["schema_problems"].append("cast failed: %s" % exc)
    return df2, s


def clean(tables: dict) -> tuple:
    """Clean every table in place of the originals -> (tables, list of per-table summaries).
    A frames loader (callable) is left as is: it cleans each game as it loads it."""
    out = dict(tables)
    summaries = []
    for table in CONTRACT_TABLES:
        if table in out and isinstance(out[table], pd.DataFrame):
            out[table], s = clean_contract(out[table], table)
            summaries.append(s)
    if isinstance(out.get("shots"), pd.DataFrame):
        out["shots"], s = clean_shots(out["shots"])
        summaries.append(s)
    if isinstance(out.get("events"), pd.DataFrame):
        out["events"], s = clean_events(out["events"])
        summaries.append(s)
    fr = out.get("frames")
    if isinstance(fr, dict):
        cleaned, tot_in, tot_out = {}, 0, 0
        agg = _summary("frames", 0, 0, games=len(fr))
        for game, df in fr.items():
            cleaned[game], s = clean_frames(df)
            tot_in += s["rows_in"]
            tot_out += s["rows_out"]
            for k in ("dropped_missing", "dropped_range", "dropped_duplicate"):
                agg[k] += s[k]
        agg["rows_in"], agg["rows_out"] = tot_in, tot_out
        out["frames"] = cleaned
        summaries.append(agg)
    elif isinstance(fr, pd.DataFrame):
        out["frames"], s = clean_frames(fr)
        summaries.append(s)
    return out, summaries


def summary_table(summaries: list) -> pd.DataFrame:
    """The per-table summaries of clean() as one DataFrame (schema problems as a count)."""
    rows = [{k: (len(v) if k == "schema_problems" else v) for k, v in s.items() if k != "games"} for s in summaries]
    return pd.DataFrame(rows).rename(columns={"schema_problems": "schema_problems_n"})


# ----------------------------------------------------------------------------- CLI

def add_source_args(ap: argparse.ArgumentParser, games: int = 8, possessions: int = 40, heldout_every: int = 4):
    """The --fixtures | --data group, --season, --seed and the fixture knobs every xFG CLI shares."""
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic season (fixture output, not a result)")
    src.add_argument("--data", metavar="DATA_ROOT", help="DATA_ROOT with sportvu_contract/, collect/bref_shots/ and analytics/")
    ap.add_argument("--season", default=SEASON, help="season directory to read and write (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    ap.add_argument("--fixture-games", type=int, default=games, dest="fixture_games",
                    help="fixtures only: games in the synthetic season (default %(default)s)")
    ap.add_argument("--fixture-possessions", type=int, default=possessions, dest="fixture_possessions",
                    help="fixtures only: possessions per game (default %(default)s)")
    ap.add_argument("--fixture-heldout-every", type=int, default=heldout_every, dest="fixture_heldout_every",
                    help="fixtures only: every k-th game is held out (default %(default)s)")
    return ap


def source_tables(a) -> dict:
    """The tables the parsed --fixtures or --data arguments name (not yet cleaned)."""
    if a.fixtures:
        return fixture_tables(a.seed, n_games=a.fixture_games, heldout_every=a.fixture_heldout_every,
                              n_possessions=a.fixture_possessions)
    return load_data_root(a.data, a.season)


def main(argv=None) -> list:
    """CLI: validate and clean the inputs, print the per-table summary, write validate.json under --out."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_args(ap)
    ap.add_argument("--out", metavar="OUT_ROOT", default=None,
                    help="writes analytics/xfg/<season>/validate.json under OUT_ROOT (default: nothing written)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stderr)
    tables = source_tables(a)
    _tables, summaries = clean(tables)
    text = summary_table(summaries).to_string(index=False)
    for s in summaries:
        for p in s["schema_problems"]:
            text += "\n%s: %s" % (s["table"], p)
    if not a.fixtures:
        text += "\nframes: validated game by game as the features loader reads them (python -m analytics.xfg.features --data)"
    if a.fixtures:
        text += "\nfixture output, not a result"
    print(text)
    if a.out:
        d = Path(a.out) / "analytics" / "xfg" / a.season
        d.mkdir(parents=True, exist_ok=True)
        (d / "validate.json").write_text(json.dumps({"fixture": bool(a.fixtures), "tables": summaries}, indent=1) + "\n")
    return summaries


if __name__ == "__main__":
    main()

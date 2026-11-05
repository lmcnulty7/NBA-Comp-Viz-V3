"""analytics/load/validate.py: inputs of the load lane and the checks on them.

    python -m analytics.load.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16] [--limit N]

Inputs (DATA_ROOT layout on the owner's Mac, docs/DATA_CONTRACT.md):
    sportvu_contract/manifest.json                       dates and teams of every game (section 6): the schedule
    sportvu_contract/<game>.parquet, heldout/<game>.parquet   section 4 frames (read lazily, game by game)
    analytics/rosters/<season>/<game>.parquet (+ heldout/<season>/)   section 16; built from the frames when absent
    collect/tracking/aggregates/<season>.parquet          section 11, when present: leaguedashptstats speed and
                                                          distance for a cross-check of the tracking totals
    collect/extras/injuries/<season>.parquet              section 15, when present: 2021-22 onward only, never
                                                          overlapping the 2015-16 tracking era; it is listed and
                                                          nothing joins to it
With --fixtures the synthetic season of tests/_fx_load.py (shared analytics.fixtures when it imports) stands in;
fixture output is never a result.

Checks on each frames table (validate_frames), all counted in a log dict:
    required columns; rows with a missing key or position dropped; duplicate (frame, player_id) rows dropped;
    positions outside the court margin (x in [-5, 99], y in [-5, 55] ft) dropped; a frame-to-frame speed above
    30 ft per s (per player, consecutive frames of the same period within 1 s of wall clock and 5 s of game
    clock) flags the arriving row, which is dropped (a one-frame glitch flags two rows: the jump out and the
    jump back); the per player-game distance is checked against a 35,000 ft ceiling (about 6.6 miles) and only
    flagged. Nothing is interpolated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FRAME_COLUMNS = ("game", "period", "frame", "game_clock_s", "team_id", "team", "home", "player_id", "x", "y")
READ_COLUMNS = ["game", "date", "period", "frame", "ts_ms", "game_clock_s", "team_id", "team", "home", "player_id", "x", "y"]
SPEED_MAX_FTPS = 30.0
MAX_STEP_DT_S = 1.0
CONTINUOUS_S = 5.0        # a game clock jump above this (or a period change) ends a continuous span: no step across it
COURT_X_FT = (-5.0, 99.0)
COURT_Y_FT = (-5.0, 55.0)
DISTANCE_GAME_MAX_FT = 35000.0
FPS = 25.0
WHO = "analytics.load"


# ── frames ─────────────────────────────────────────────────────────────────────────────────────────
def continuous_step(frames: pd.DataFrame) -> np.ndarray:
    """True where a row continues the previous row's span: same player, same period, game clock within
    CONTINUOUS_S. frames must be sorted by player, period, frame."""
    pid, per = frames["player_id"].to_numpy(), frames["period"].to_numpy()
    clock = frames["game_clock_s"].to_numpy(dtype="float64")
    same = (pid == np.roll(pid, 1)) & (per == np.roll(per, 1)) & (np.abs(np.roll(clock, 1) - clock) <= CONTINUOUS_S)
    same[0] = False
    return same


def step_dt_s(frames: pd.DataFrame) -> np.ndarray:
    """Seconds between a row and the previous row of the same continuous span (continuous_step; NaN elsewhere):
    from ts_ms when present, else from the frame index at 25 Hz. frames must be sorted by player, period, frame."""
    same = continuous_step(frames)
    if "ts_ms" in frames.columns and frames["ts_ms"].notna().all():
        t = frames["ts_ms"].to_numpy(dtype="float64") / 1000.0
    else:
        t = frames["frame"].to_numpy(dtype="float64") / FPS
    dt = t - np.roll(t, 1)
    return np.where(same, dt, np.nan)


def validate_frames(frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """-> (clean frames sorted by player_id, period, frame; log of counts). Raises ValueError on a missing column."""
    missing = [c for c in REQUIRED_FRAME_COLUMNS if c not in frames.columns]
    if missing:
        raise ValueError("%s: frames table lacks column(s) %s" % (WHO, missing))
    log = {"rows_in": int(len(frames))}
    df = frames
    keys = ["period", "frame", "player_id", "x", "y", "game_clock_s"]
    m_missing = df[keys].isna().any(axis=1)
    log["rows_missing_key_or_position"] = int(m_missing.sum())
    df = df[~m_missing]
    before = len(df)
    df = df.drop_duplicates(subset=["frame", "player_id"], keep="first")
    log["rows_duplicate_frame_player"] = int(before - len(df))
    x, y = df["x"].to_numpy(dtype="float64"), df["y"].to_numpy(dtype="float64")
    m_court = (x >= COURT_X_FT[0]) & (x <= COURT_X_FT[1]) & (y >= COURT_Y_FT[0]) & (y <= COURT_Y_FT[1])
    log["rows_outside_court"] = int((~m_court).sum())
    df = df[m_court]
    df = df.sort_values(["player_id", "period", "frame"], kind="mergesort").reset_index(drop=True)
    dt = step_dt_s(df)
    dx = df["x"].to_numpy(dtype="float64") - np.roll(df["x"].to_numpy(dtype="float64"), 1)
    dy = df["y"].to_numpy(dtype="float64") - np.roll(df["y"].to_numpy(dtype="float64"), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.hypot(dx, dy) / dt
    checked = np.isfinite(dt) & (dt > 0) & (dt <= MAX_STEP_DT_S)
    too_fast = checked & (speed > SPEED_MAX_FTPS)
    log["steps_checked"] = int(checked.sum())
    log["rows_speed_over_%d_ftps" % int(SPEED_MAX_FTPS)] = int(too_fast.sum())
    df = df[~too_fast].reset_index(drop=True)
    log["rows_out"] = int(len(df))
    log["players"] = int(df["player_id"].nunique())
    log["frames"] = int(df["frame"].nunique())
    # distance per player-game: flagged, never dropped
    dt = step_dt_s(df)
    dx = df["x"].to_numpy(dtype="float64") - np.roll(df["x"].to_numpy(dtype="float64"), 1)
    dy = df["y"].to_numpy(dtype="float64") - np.roll(df["y"].to_numpy(dtype="float64"), 1)
    ok = np.isfinite(dt) & (dt > 0) & (dt <= MAX_STEP_DT_S)
    dist = pd.Series(np.where(ok, np.hypot(dx, dy), 0.0)).groupby(df["player_id"].to_numpy()).sum()
    log["player_distance_ft_max"] = float(dist.max()) if len(dist) else 0.0
    log["players_distance_over_ceiling"] = int((dist > DISTANCE_GAME_MAX_FT).sum())
    return df, log


def validate_rosters(rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rosters need sportvu_game, player_id, team_id, team; duplicates by player_id are dropped and counted."""
    missing = [c for c in ("sportvu_game", "player_id", "team_id", "team") if c not in rosters.columns]
    if missing:
        raise ValueError("%s: rosters table lacks column(s) %s" % (WHO, missing))
    before = len(rosters)
    out = rosters.drop_duplicates(subset=["sportvu_game", "player_id"]).reset_index(drop=True)
    return out, {"rows_in": int(before), "rows_duplicate_player": int(before - len(out)), "rows_out": int(len(out))}


def roster_from_frames(frames: pd.DataFrame, sportvu_game: str) -> pd.DataFrame:
    ros = frames[["player_id", "team_id", "team", "home"]].drop_duplicates("player_id").copy()
    ros.insert(0, "sportvu_game", sportvu_game)
    return ros.reset_index(drop=True)


# ── game ids and the schedule ──────────────────────────────────────────────────────────────────────
def game_meta(sportvu_game: str) -> dict:
    """'12.16.2015.PHX.at.GSW' -> dict(sportvu_game, date, away, home, season)."""
    mm, dd, yyyy, away, _at, home = str(sportvu_game).split(".")
    y = int(yyyy)
    season = "%d-%02d" % (y, (y + 1) % 100) if int(mm) >= 9 else "%d-%02d" % (y - 1, y % 100)
    return {"sportvu_game": sportvu_game, "date": "%s-%s-%s" % (yyyy, mm, dd), "away": away, "home": home, "season": season}


def heldout_by_id(sportvu_game: str) -> bool:
    """True when sportvu/splits.json marks the game held out (by id or by the held-out arena)."""
    from sportvu import splits
    toks = splits.heldout_tokens()
    s = str(sportvu_game)
    return any(t in s for t in toks) or s.endswith(".at." + splits.HELDOUT_ARENA)


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


# ── inputs ─────────────────────────────────────────────────────────────────────────────────────────
def game_frames(game: dict) -> pd.DataFrame:
    """The frames of a game dict: in memory under 'frames', else read from 'frames_path' (needed columns only)."""
    if isinstance(game.get("frames"), pd.DataFrame):
        return game["frames"]
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(game["frames_path"]))
    cols = [c for c in READ_COLUMNS if c in pf.schema_arrow.names]
    return pf.read(columns=cols).to_pandas()


def game_rosters(game: dict, frames: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(game.get("rosters"), pd.DataFrame):
        return game["rosters"]
    p = game.get("rosters_path")
    if p and Path(p).exists():
        return pd.read_parquet(p)
    if frames is None:
        frames = game_frames(game)
    return roster_from_frames(frames, game["sportvu_game"])


def load_data_root(data_root, season: str = "2015-16", limit: int | None = None) -> dict:
    """The inputs dict from DATA_ROOT: games (lazy frames), manifest, aggregates, injuries."""
    root = Path(data_root)
    manifest = load_manifest(root / "sportvu_contract" / "manifest.json")
    games = []
    for sid, g in sorted(manifest.get("games", {}).items(), key=lambda kv: (kv[1].get("date", ""), kv[0])):
        if g.get("status") != "ok":
            continue
        held = bool(g.get("heldout")) or heldout_by_id(sid)
        fp = root / "sportvu_contract" / ("heldout/" if held else "") / ("%s.parquet" % sid)
        if not fp.exists():
            alt = root / "sportvu_contract" / ("" if held else "heldout/") / ("%s.parquet" % sid)
            if not alt.exists():
                continue
            fp = alt
            held = held or "heldout" in alt.parts        # a file under heldout/ is held out whatever the manifest says
        rp = root / "analytics" / "rosters" / ("heldout/" if held else "") / season / ("%s.parquet" % sid)
        meta = game_meta(sid)
        games.append({"sportvu_game": sid, "date": g.get("date") or meta["date"], "away": g.get("away") or meta["away"],
                      "home": g.get("home") or meta["home"], "heldout": held, "frames": None, "frames_path": str(fp),
                      "rosters": None, "rosters_path": str(rp) if rp.exists() else None})
        if limit and len(games) >= limit:
            break
    agg_p = root / "collect" / "tracking" / "aggregates" / ("%s.parquet" % season)
    aggregates = pd.read_parquet(agg_p) if agg_p.exists() else None
    inj_dir = root / "collect" / "extras" / "injuries"
    injuries = None
    if inj_dir.exists():
        parts = sorted(inj_dir.glob("*.parquet"))
        injuries = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None
    return {"games": games, "manifest": manifest, "aggregates": aggregates, "injuries": injuries, "season": season,
            "source": str(root)}


def load_fixtures(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 20,
                  prefer_shared: bool = True) -> dict:
    """The synthetic season: analytics.fixtures when it imports (through tests/_fx_load.py), else that file's
    local generator. Lazy import so the package never depends on tests/ outside --fixtures."""
    try:
        from tests import _fx_load as fx
    except ImportError:
        import importlib.util
        p = Path(__file__).resolve().parents[2] / "tests" / "_fx_load.py"
        spec = importlib.util.spec_from_file_location("_fx_load", p)
        fx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fx)
    return fx.season_inputs(seed=seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions,
                            prefer_shared=prefer_shared)


def injuries_note(injuries: pd.DataFrame | None) -> str:
    """One line about the injuries table: seasons present and the statement that it never overlaps 2015-16."""
    if injuries is None or not len(injuries):
        return ("injuries table (section 15): not present; it covers 2021-22 onward only and never overlaps the "
                "2015-16 tracking era, so absence here is read from the rosters, not from injury reports")
    seasons = sorted(str(s) for s in injuries.get("season", pd.Series(dtype=object)).dropna().unique())
    return ("injuries table (section 15): %d rows over seasons %s; 2021-22 onward only, no overlap with the 2015-16 "
            "tracking era, nothing joins to it; absence here is read from the rosters" % (len(injuries), seasons))


# ── run every check ────────────────────────────────────────────────────────────────────────────────
def validate_inputs(inputs: dict) -> dict:
    """Validate every game's frames and rosters -> dict(games={id: log}, totals, injuries, aggregates)."""
    out = {"games": {}, "totals": {}, "injuries": injuries_note(inputs.get("injuries")),
           "aggregates_rows": int(len(inputs["aggregates"])) if inputs.get("aggregates") is not None else 0}
    totals: dict = {}
    for g in inputs["games"]:
        frames = game_frames(g)
        _clean, log = validate_frames(frames)
        _ros, rlog = validate_rosters(game_rosters(g, frames))
        log["rosters"] = rlog
        log["heldout"] = bool(g["heldout"])
        out["games"][g["sportvu_game"]] = log
        for k, v in log.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                totals[k] = totals.get(k, 0) + v
    out["totals"] = totals
    out["n_games"] = len(inputs["games"])
    out["n_heldout"] = sum(1 for g in inputs["games"] if g["heldout"])
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--local-fixtures", action="store_true",
                    help="with --fixtures: the local generator of tests/_fx_load.py even when analytics.fixtures imports")
    ap.add_argument("--out", help="OUT_ROOT: writes OUT_ROOT/analytics/load/<season>/validation.json (counts only)")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--limit", type=int, default=None, help="first N games of the manifest (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    inputs = load_fixtures(seed=a.seed, prefer_shared=not a.local_fixtures) if a.fixtures else \
        load_data_root(a.data, a.season, a.limit)
    rep = validate_inputs(inputs)
    rep["source"] = inputs["source"]
    rep["note"] = "fixture output, not a result" if a.fixtures else "validation counts"
    text = json.dumps(rep, indent=1)
    if a.out:
        p = Path(a.out) / "analytics" / "load" / inputs["season"] / "validation.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print("wrote %s" % p, file=sys.stderr)
    print(text)
    return rep


if __name__ == "__main__":
    main()

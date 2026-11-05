"""analytics/matchups/validate.py: inputs of the matchups lane and their validation.

    python -m analytics.matchups.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

Loads one game dict per game (frames, events, possessions, shot_frames, rosters, heldout flag) from the
DATA_ROOT layout

    DATA_ROOT/sportvu_contract/<game>.parquet, <game>_pbp.parquet       (held-out games under heldout/)
    DATA_ROOT/analytics/<table>/<season>/<game>.parquet                 (held-out under analytics/<table>/heldout/<season>/)

or from the synthetic season (--fixtures: analytics.fixtures when present, else the fallback generator in
tests/_fx_matchups.py). A game is held out when it lives under a heldout/ directory or when its id carries a
held-out token of sportvu/splits.json.

Validation per game: schema.validate on possessions, shot_frames and rosters (contract tables); the frames
table checked for its section 4 columns, positions inside the court with a 5 ft margin (0 to 94 by 0 to 50),
game_clock_s in [0, 720] (overtime periods allow 300), ten or fewer players per frame; the play-by-play
checked for its section 5 columns. Missing positions (null x or y) and duplicate (frame, player_id) rows are
dropped with logged counts; out-of-range positions are logged and kept (the assignment is a nearest-neighbour
problem, a player 2 ft off the court still has a nearest opponent).

The xFG lane's predictions are read from OUT_ROOT/analytics/xfg/<season>/*.parquet when a file there has the
columns sportvu_game, event, player_id, p_selected and fold (plus heldout when the lane writes it: the held-out
rows are those flagged, else those whose id sportvu/splits.json marks); the fold column must be present and
non-negative on every training row (convention: >= 0 out-of-fold, -1 held-out, -2 in-sample; a training row at -1,
-2 or null is refused), otherwise the file is refused (logged) and the trainer uses the location-only fallback.
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

log = logging.getLogger("analytics.matchups.validate")

FRAME_COLUMNS = ["period", "frame", "game_clock_s", "team_id", "player_id", "x", "y"]
PBP_COLUMNS = ["event", "period", "clock_s", "team", "desc", "points", "kind", "frame"]
XFG_COLUMNS = ["sportvu_game", "event", "player_id", "p_selected", "fold"]
COURT_X, COURT_Y, MARGIN = 94.0, 50.0, 5.0
CLOCK_MAX, OT_CLOCK_MAX = 720.0, 300.0
TABLES = ("possessions", "shot_frames", "rosters")
FIXTURE_HELDOUT_EVERY = 4            # the fixture default of every CLI of the package: every 4th game held out


# ── held-out ──────────────────────────────────────────────────────────────────────────────────────
def is_heldout(sportvu_game: str) -> bool:
    """True when the id carries a held-out token of sportvu/splits.json or ends in the held-out arena."""
    from sportvu import splits
    s = str(sportvu_game)
    return any(t in s for t in splits.heldout_tokens()) or s.endswith(".at." + splits.HELDOUT_ARENA)


def xfg_heldout_mask(x: pd.DataFrame) -> pd.Series:
    """Held-out rows of an xFG predictions table: its heldout column when present (the xFG lane sets it for games
    held out by flag, directory or token), else sportvu/splits.json by game id."""
    if "heldout" in x.columns:
        return x["heldout"].fillna(False).astype(bool)
    return x["sportvu_game"].astype(str).map(is_heldout).astype(bool)


# ── loaders ───────────────────────────────────────────────────────────────────────────────────────
def _fixture_module():
    try:
        from analytics import fixtures as fx  # type: ignore
        return fx
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tests import _fx_matchups as fx  # type: ignore
        return fx


def _as_game_list(season) -> list:
    """Normalise a synth_season result (a list of game dicts, or one dict of season-wide tables) to a list of game dicts."""
    if isinstance(season, dict) and "frames" in season and isinstance(season["frames"], pd.DataFrame):
        tables = {k: v for k, v in season.items() if isinstance(v, pd.DataFrame)}
        key = "sportvu_game" if "sportvu_game" in tables["possessions"].columns else "game"
        out = []
        for gid in tables["possessions"][key].unique():
            g = {}
            for k, v in tables.items():
                col = "sportvu_game" if "sportvu_game" in v.columns else ("game" if "game" in v.columns else None)
                g[k] = v[v[col] == gid].reset_index(drop=True) if col else v
            out.append(g)
        return out
    return list(season)


def fixture_games(seed: int = 0, n_games: int = 8, heldout_every: int = FIXTURE_HELDOUT_EVERY, n_possessions: int | None = None) -> list:
    """The synthetic season as game dicts with a heldout flag (fixture output, not a result). With n_possessions
    the games come from synth_game one by one (smaller games for tests), else from synth_season."""
    fx = _fixture_module()
    if n_possessions is None:
        games = _as_game_list(fx.synth_season(seed, n_games=n_games, heldout_every=heldout_every))
    else:
        games = [fx.synth_game(seed * 100 + i, n_possessions=n_possessions,
                               heldout=bool(heldout_every) and (i + 1) % heldout_every == 0) for i in range(n_games)]
    for g in games:
        gid = str(g["possessions"]["sportvu_game"].iloc[0])
        g["sportvu_game"] = gid
        g["heldout"] = bool(g.get("heldout", False)) or is_heldout(gid)
    return games


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def load_games(data_root, season: str = "2015-16") -> list:
    """Game dicts from the DATA_ROOT layout; games are discovered from analytics/possessions/[heldout/]<season>/."""
    root = Path(data_root)
    out = []
    for held in (False, True):
        d = root / "analytics" / "possessions" / ("heldout" if held else "") / season
        for pq in sorted(d.glob("*.parquet")):
            gid = pq.stem
            sv = root / "sportvu_contract" / ("heldout" if held else "")
            frames = _read(sv / (gid + ".parquet"))
            if frames is None:                          # the other directory: the play-by-play is read next to the frames found
                sv = root / "sportvu_contract" / ("" if held else "heldout")
                frames = _read(sv / (gid + ".parquet"))
            if frames is None:
                log.warning("%s: no frames under sportvu_contract, skipped", gid)
                continue
            events = _read(sv / (gid + "_pbp.parquet"))
            g = {"sportvu_game": gid, "heldout": held or is_heldout(gid), "frames": frames, "events": events,
                 "possessions": pd.read_parquet(pq)}
            for table in ("shot_frames", "rosters", "rebound_frames"):
                t = _read(root / "analytics" / table / ("heldout" if held else "") / season / (gid + ".parquet"))
                g[table] = t if t is not None else schema.empty(table)
            out.append(g)
    return out


def load_xfg(out_root, season: str = "2015-16") -> pd.DataFrame | None:
    """The xFG lane's out-of-fold predictions, or None (with a logged reason) when absent or not verifiably out-of-fold."""
    if out_root is None:
        return None
    d = Path(out_root) / "analytics" / "xfg" / season
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    parts = []
    for f in files:
        t = pd.read_parquet(f)
        missing = [c for c in XFG_COLUMNS if c not in t.columns]
        if missing:
            log.info("xfg: %s lacks %s, ignored", f.name, missing)
            continue
        parts.append(t[XFG_COLUMNS + (["heldout"] if "heldout" in t.columns else [])])
    if not parts:
        log.info("xfg: no predictions under %s, using the location-only fallback", d)
        return None
    x = pd.concat(parts, ignore_index=True)
    train = x[~xfg_heldout_mask(x)]
    bad = train["fold"].isna() | (pd.to_numeric(train["fold"], errors="coerce").fillna(-1) < 0)     # -1, -2 and null alike
    if bad.any():
        log.warning("xfg: %d of %d training rows have no out-of-fold fold index; refused, using the fallback", int(bad.sum()), len(train))
        return None
    log.info("xfg: %d predictions from %d files, fold column verified on %d training rows", len(x), len(parts), len(train))
    return x


# ── validation ────────────────────────────────────────────────────────────────────────────────────
def clean_frames(frames: pd.DataFrame) -> tuple:
    """Drop rows with a null position and duplicate (frame, player_id) rows -> (frames, counts)."""
    n0 = len(frames)
    m = frames["x"].isna() | frames["y"].isna()
    f = frames[~m]
    f2 = f.drop_duplicates(["frame", "player_id"], keep="first")
    counts = {"rows": n0, "missing_position": int(m.sum()), "duplicate_rows": int(len(f) - len(f2))}
    return f2.reset_index(drop=True), counts


def validate_game(game: dict) -> dict:
    """Problems and counts of one game dict; cleans the frames in place (missing positions, duplicates)."""
    gid = str(game.get("sportvu_game") or game["possessions"]["sportvu_game"].iloc[0])
    problems: list = []
    counts: dict = {"sportvu_game": gid, "heldout": bool(game.get("heldout", False))}
    for table in TABLES:
        t = game.get(table)
        if t is None:
            problems.append("%s: missing" % table)
            continue
        problems += ["%s: %s" % (table, p) for p in schema.validate(t, table)]
        counts["n_" + table] = int(len(t))
    fr = game["frames"]
    missing = [c for c in FRAME_COLUMNS if c not in fr.columns]
    if missing:
        problems.append("frames: missing column(s) %s" % missing)
        counts["n_frames_rows"] = int(len(fr))
        return {"problems": problems, "counts": counts}
    fr, c = clean_frames(fr)
    game["frames"] = fr
    counts.update({"n_frames_rows": c["rows"], "missing_position": c["missing_position"], "duplicate_rows": c["duplicate_rows"]})
    if c["missing_position"] or c["duplicate_rows"]:
        log.info("%s: dropped %d rows with a missing position and %d duplicate (frame, player_id) rows",
                 gid, c["missing_position"], c["duplicate_rows"])
    x, y = fr["x"].to_numpy(dtype=float), fr["y"].to_numpy(dtype=float)
    off = ((x < -MARGIN) | (x > COURT_X + MARGIN) | (y < -MARGIN) | (y > COURT_Y + MARGIN)).sum()
    counts["off_court_rows"] = int(off)
    if off:
        log.info("%s: %d rows more than %.0f ft outside the court (kept)", gid, off, MARGIN)
    clk = fr["game_clock_s"].to_numpy(dtype=float)
    per = fr["period"].to_numpy(dtype=int)
    limit = np.where(per <= 4, CLOCK_MAX, OT_CLOCK_MAX)
    bad_clock = int(((clk < 0) | (clk > limit)).sum())
    counts["bad_clock_rows"] = bad_clock
    if bad_clock:
        problems.append("frames: %d rows with game_clock_s outside [0, period length]" % bad_clock)
    per_frame = fr.groupby("frame")["player_id"].size()
    counts["frames_over_10_players"] = int((per_frame > 10).sum())
    if counts["frames_over_10_players"]:
        problems.append("frames: %d frames with more than ten players" % counts["frames_over_10_players"])
    counts["n_frames"] = int(fr["frame"].nunique())
    ev = game.get("events")
    if ev is None:
        counts["n_events"] = 0
        log.info("%s: no play-by-play table (shot values fall back to distance)", gid)
    else:
        counts["n_events"] = int(len(ev))
        missing = [c for c in PBP_COLUMNS if c not in ev.columns]
        if missing:
            problems.append("events: missing column(s) %s" % missing)
    pos = game["possessions"]
    if len(pos):
        if {"end_frame", "start_frame"} <= set(pos.columns) and (pos["end_frame"] < pos["start_frame"]).any():
            problems.append("possessions: end_frame before start_frame")
        if "offense_team_id" in pos.columns:
            counts["possessions_no_offense"] = int(pos["offense_team_id"].isna().sum())
        if "gap" in pos.columns:
            counts["possessions_gap"] = int(pos["gap"].astype(bool).sum())
    return {"problems": problems, "counts": counts}


def validate_games(games: list) -> pd.DataFrame:
    """validate_game over games -> one row per game with counts and the problem list."""
    rows = []
    for g in games:
        r = validate_game(g)
        rows.append({**r["counts"], "n_problems": len(r["problems"]), "problems": "; ".join(r["problems"])})
        for p in r["problems"]:
            log.warning("%s: %s", r["counts"]["sportvu_game"], p)
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="validate the synthetic season (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes analytics/matchups/validate_<season>.json")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-games", type=int, default=8, help="fixtures only")
    ap.add_argument("--heldout-every", type=int, default=FIXTURE_HELDOUT_EVERY, help="fixtures only: every k-th game is held out")
    ap.add_argument("--n-possessions", type=int, default=None, help="fixtures only: possessions per game (default: the generator's season)")
    return ap.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    a = parse_args(argv)
    games = fixture_games(a.seed, a.n_games, a.heldout_every, a.n_possessions) if a.fixtures else load_games(a.data, a.season)
    if not games:
        raise SystemExit("validate: no games found")
    table = validate_games(games)
    print(table.drop(columns=["problems"]).to_string(index=False))
    bad = table[table["n_problems"] > 0]
    print("%d game(s), %d with problems" % (len(table), len(bad)))
    for r in bad.itertuples(index=False):
        print("  %s: %s" % (r.sportvu_game, r.problems))
    if a.out:
        p = Path(a.out) / "analytics" / "matchups"
        p.mkdir(parents=True, exist_ok=True)
        (p / ("validate_%s.json" % a.season)).write_text(json.dumps(table.to_dict("records"), indent=1, default=str) + "\n")
        print("wrote %s" % (p / ("validate_%s.json" % a.season)))
    return table


if __name__ == "__main__":
    main()

"""analytics/rebounds/validate.py: inputs of the rebound lane, their checks and their cleaning.

    python -m analytics.rebounds.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0] [--games 8]

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet and <game>_pbp.parquet        held-out games under heldout/
    DATA_ROOT/analytics/<table>/<season>/<game>.parquet                       table in rebound_frames, shot_frames,
    DATA_ROOT/analytics/<table>/heldout/<season>/<game>.parquet               rosters, possessions
A game is a bundle: dict(sportvu_game, heldout, frames, events, rebound_frames, shot_frames, rosters, possessions).
``bundles_from_root`` reads one, lazily per game (frames are large); ``bundles_from_fixtures`` builds synthetic
ones from analytics.fixtures when it exists, else tests._fx_rebounds. A bundle is held out when its files sit
under heldout/ or its id carries a held-out token of sportvu/splits.json (analytics.eval.split_games).

Checks (``check_bundle``): analytics.schema.validate on rebound_frames, shot_frames, rosters and possessions;
frames and ball positions inside the court (0 to 94 by 0 to 50 ft), game clocks 0 to 720 s, periods at least 1,
event kinds in the section 5 vocabulary; nulls in required columns; duplicate keys. ``clean_bundle`` drops the
duplicate keys and the out-of-range rows and returns the counts in and out of every table, which the report
prints. Nothing here fits anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import split_games

WHO = "rebounds.validate"
COURT_X, COURT_Y, CLOCK_MAX = (0.0, 94.0), (0.0, 50.0), 720.0
FRAME_COLUMNS = ["game", "period", "frame", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "home",
                 "player_id", "player", "x", "y"]
EVENT_COLUMNS = ["game", "event", "period", "clock_s", "team", "desc", "kind", "frame"]
EVENT_KINDS = {"shot_made", "shot_missed", "ft_made", "ft_missed", "turnover", "rebound", "foul", "other",
               "substitution", "timeout", "jump_ball", "period_start", "period_end", "violation", "ejection",
               "instant_replay"}
TABLES = ("rebound_frames", "shot_frames", "rosters", "possessions")
TABLE_KEYS = {"rebound_frames": ["sportvu_game", "event"], "shot_frames": ["sportvu_game", "event"],
              "rosters": ["sportvu_game", "player_id"], "possessions": ["sportvu_game", "possession"]}


# ── sources ───────────────────────────────────────────────────────────────────────────────────────
def _fixture_module():
    try:
        from analytics import fixtures as fx     # the shared generator of lane I1, when it exists
        return fx
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if str(root / "tests") not in sys.path:
            sys.path.insert(0, str(root / "tests"))
        import _fx_rebounds as fx
        return fx


def bundles_from_fixtures(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40) -> list:
    """Synthetic bundles. Fixture output, never a result."""
    fx = _fixture_module()
    try:
        games = fx.synth_season(seed, n_games, heldout_every, n_possessions=n_possessions)
    except TypeError:                               # a generator without the n_possessions argument
        games = fx.synth_season(seed, n_games, heldout_every)
    if isinstance(games, dict):
        games = list(games.values())
    out = []
    for g in games:
        gid = str(g["frames"]["game"].iat[0]) if "game" in g["frames"].columns else str(g["rebound_frames"]["sportvu_game"].iat[0])
        held = bool(split_games([gid])["heldout"]) or gid.endswith(".at.CLE")
        if "shots" in g and len(g["shots"]) and "heldout" in g["shots"].columns:
            held = held or bool(g["shots"]["heldout"].astype(bool).any())
        out.append({"sportvu_game": gid, "heldout": held, "frames": g["frames"], "events": g.get("events"),
                    "rebound_frames": g["rebound_frames"], "shot_frames": g.get("shot_frames"),
                    "rosters": g.get("rosters"), "possessions": g.get("possessions")})
    return out


def _is_heldout_path(p: Path) -> bool:
    return "heldout" in p.parts


def list_root(root) -> list:
    """Games of DATA_ROOT that have frames and a rebound_frames table -> [(game, heldout, paths dict)]."""
    root = Path(root)
    out = []
    for rf_path in sorted((root / "analytics" / "rebound_frames").rglob("*.parquet")):
        game = rf_path.stem
        held = _is_heldout_path(rf_path.relative_to(root)) or bool(split_games([game])["heldout"]) or game.endswith(".at.CLE")
        sv = root / "sportvu_contract" / ("heldout" if held else "")
        frames = sv / (game + ".parquet")
        if not frames.exists():
            frames = root / "sportvu_contract" / (game + ".parquet")
            if not frames.exists():
                frames = root / "sportvu_contract" / "heldout" / (game + ".parquet")
                held = held or frames.exists()
        if not frames.exists():
            continue
        paths = {"frames": frames, "events": frames.with_name(game + "_pbp.parquet"), "rebound_frames": rf_path}
        for t in ("shot_frames", "rosters", "possessions"):
            cand = list((root / "analytics" / t).rglob(game + ".parquet"))
            paths[t] = cand[0] if cand else None
        out.append((game, held, paths))
    return out


def read_frames(path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = [c for c in FRAME_COLUMNS if c in t.column_names]
    return t.select(cols).to_pandas()


def bundles_from_root(root):
    """Yield one bundle per game of DATA_ROOT, frames read on demand."""
    for game, held, paths in list_root(root):
        b = {"sportvu_game": game, "heldout": held, "frames": read_frames(paths["frames"]),
             "events": pd.read_parquet(paths["events"]) if paths["events"].exists() else None}
        for t in TABLES:
            b[t] = pd.read_parquet(paths[t]) if paths.get(t) is not None and Path(paths[t]).exists() else None
        yield b


# ── checks and cleaning ───────────────────────────────────────────────────────────────────────────
def _range_mask(df: pd.DataFrame, col: str, lo: float, hi: float, nullable: bool = False) -> np.ndarray:
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    ok = (v >= lo) & (v <= hi)
    return ok | (nullable & np.isnan(v))


def check_frames(frames: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """-> (keep mask, problem counts) for the section 4 frames."""
    missing = [c for c in FRAME_COLUMNS if c not in frames.columns and c not in ("player", "game")]
    if missing:
        raise ValueError("%s: frames lack column(s) %s" % (WHO, missing))
    n = len(frames)
    ok_x = _range_mask(frames, "x", *COURT_X); ok_y = _range_mask(frames, "y", *COURT_Y)
    ok_bx = _range_mask(frames, "ball_x", *COURT_X, nullable=True); ok_by = _range_mask(frames, "ball_y", *COURT_Y, nullable=True)
    ok_clock = _range_mask(frames, "game_clock_s", 0.0, CLOCK_MAX)
    ok_period = pd.to_numeric(frames["period"], errors="coerce").to_numpy(float) >= 1
    dup = frames.duplicated(["frame", "player_id"], keep="first").to_numpy()
    null_req = frames[["frame", "period", "game_clock_s", "player_id", "x", "y"]].isna().any(axis=1).to_numpy()
    keep = ok_x & ok_y & ok_bx & ok_by & ok_clock & ok_period & ~dup & ~null_req
    counts = {"rows_in": n, "player_out_of_court": int((~(ok_x & ok_y)).sum()), "ball_out_of_court": int((~(ok_bx & ok_by)).sum()),
              "clock_out_of_range": int((~ok_clock).sum()), "bad_period": int((~ok_period).sum()),
              "duplicate_frame_player": int(dup.sum()), "null_required": int(null_req.sum()),
              "ball_null": int(frames["ball_x"].isna().sum()), "rows_out": int(keep.sum()),
              "frames": int(frames["frame"].nunique())}
    return keep, counts


def check_events(events: pd.DataFrame | None) -> tuple[np.ndarray | None, dict]:
    if events is None:
        return None, {"rows_in": 0, "rows_out": 0, "absent": True}
    missing = [c for c in ("event", "period", "clock_s", "kind") if c not in events.columns]
    if missing:
        raise ValueError("%s: events lack column(s) %s" % (WHO, missing))
    ok_clock = _range_mask(events, "clock_s", 0.0, CLOCK_MAX)
    kind_ok = events["kind"].astype("string").isin(sorted(EVENT_KINDS)).fillna(False).to_numpy()
    dup = events.duplicated(["event"], keep="first").to_numpy()
    keep = ok_clock & ~dup
    return keep, {"rows_in": int(len(events)), "clock_out_of_range": int((~ok_clock).sum()), "unknown_kind": int((~kind_ok).sum()),
                  "duplicate_event": int(dup.sum()), "rows_out": int(keep.sum()), "absent": False}


def check_table(df: pd.DataFrame | None, table: str) -> tuple[np.ndarray | None, dict]:
    """schema.validate problems, duplicate keys, clock range (rebound_frames, shot_frames) -> (keep mask, counts)."""
    if df is None:
        return None, {"rows_in": 0, "rows_out": 0, "absent": True, "schema_problems": []}
    problems = schema.validate(df, table)
    keys = [k for k in TABLE_KEYS[table] if k in df.columns]
    dup = df.duplicated(keys, keep="first").to_numpy() if keys else np.zeros(len(df), bool)
    keep = ~dup
    counts = {"rows_in": int(len(df)), "schema_problems": problems, "duplicate_key": int(dup.sum()), "absent": False}
    if "clock_s" in df.columns:
        ok = _range_mask(df, "clock_s", 0.0, CLOCK_MAX)
        counts["clock_out_of_range"] = int((~ok).sum()); keep &= ok
    if table == "rebound_frames":
        rim = pd.to_numeric(df["rim_frame"], errors="coerce").to_numpy(float)
        counts["rim_frame_not_found"] = int((rim < 0).sum())
        counts["unmatched"] = int((df["confidence_flag"].astype("string") == "unmatched").sum())
    counts["rows_out"] = int(keep.sum())
    return keep, counts


def check_bundle(b: dict) -> dict:
    """Counts and problems of every table of one bundle (no cleaning)."""
    out = {"sportvu_game": b["sportvu_game"], "heldout": bool(b["heldout"])}
    out["frames"] = check_frames(b["frames"])[1]
    out["events"] = check_events(b.get("events"))[1]
    for t in TABLES:
        out[t] = check_table(b.get(t), t)[1]
    if b.get("rebound_frames") is not None and len(b["rebound_frames"]):
        rim = b["rebound_frames"]["rim_frame"].to_numpy()
        present = np.isin(rim[rim >= 0], b["frames"]["frame"].unique())
        out["rebound_frames"]["rim_frame_missing_in_frames"] = int((~present).sum())
    return out


def clean_bundle(b: dict) -> tuple[dict, dict]:
    """The bundle with duplicate keys and out-of-range rows dropped, plus the counts of check_bundle."""
    counts = {"sportvu_game": b["sportvu_game"], "heldout": bool(b["heldout"])}
    out = dict(b)
    keep, counts["frames"] = check_frames(b["frames"])
    out["frames"] = b["frames"][keep].reset_index(drop=True)
    keep, counts["events"] = check_events(b.get("events"))
    out["events"] = None if keep is None else b["events"][keep].reset_index(drop=True)
    for t in TABLES:
        keep, counts[t] = check_table(b.get(t), t)
        out[t] = None if keep is None else b[t][keep].reset_index(drop=True)
    return out, counts


def summarise(per_game: list) -> dict:
    tot = {}
    for t in ("frames", "events") + TABLES:
        tot[t] = {"rows_in": int(sum(g[t]["rows_in"] for g in per_game)), "rows_out": int(sum(g[t]["rows_out"] for g in per_game))}
        for k in ("player_out_of_court", "ball_out_of_court", "clock_out_of_range", "duplicate_frame_player", "duplicate_event",
                  "duplicate_key", "null_required", "unknown_kind", "rim_frame_not_found", "unmatched", "rim_frame_missing_in_frames"):
            if any(k in g[t] for g in per_game):
                tot[t][k] = int(sum(g[t].get(k, 0) for g in per_game))
        if t in TABLES:
            tot[t]["games_with_schema_problems"] = int(sum(1 for g in per_game if g[t].get("schema_problems")))
    return {"games": len(per_game), "heldout_games": int(sum(1 for g in per_game if g["heldout"])), "totals": tot, "per_game": per_game}


def markdown(summary: dict, fixtures: bool) -> str:
    L = ["# Rebound lane inputs (analytics.rebounds.validate)", ""]
    if fixtures:
        L.append("Fixture output, not a result: synthetic games, no real tracking data.")
        L.append("")
    L += ["%d game(s), %d held out." % (summary["games"], summary["heldout_games"]), "",
          "| table | rows in | rows out | problems |", "| --- | --- | --- | --- |"]
    for t, c in summary["totals"].items():
        # rim_frame_not_found and unmatched are localiser coverage counts (per-game table below), not problems
        probs = ", ".join("%s %d" % (k, v) for k, v in c.items()
                          if k not in ("rows_in", "rows_out", "rim_frame_not_found", "unmatched") and v)
        L.append("| %s | %d | %d | %s |" % (t, c["rows_in"], c["rows_out"], probs or "none"))
    L += ["", "| game | held out | frames in | frames out | misses | rim frame not found | unmatched |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for g in summary["per_game"]:
        r = g["rebound_frames"]
        L.append("| %s | %s | %d | %d | %d | %d | %d |" % (g["sportvu_game"], g["heldout"], g["frames"]["rows_in"], g["frames"]["rows_out"],
                                                         r["rows_in"], r.get("rim_frame_not_found", 0), r.get("unmatched", 0)))
        for t in TABLES:
            for p in g[t].get("schema_problems", []):
                L.append("| %s | | | | %s: %s | | |" % (g["sportvu_game"], t, p))
    return "\n".join(L) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="synthetic games (fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with sportvu_contract/ and analytics/ (see module docstring)")
    ap.add_argument("--out", help="OUT_ROOT: writes analytics/rebounds/validate.json under it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=8, help="fixture games")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    bundles = bundles_from_fixtures(a.seed, a.games) if a.fixtures else bundles_from_root(a.data)
    per_game = [check_bundle(b) for b in bundles]
    summary = summarise(per_game)
    summary["source"] = "fixtures (fixture output, not a result)" if a.fixtures else str(a.data)
    text = markdown(summary, a.fixtures)
    print(text)
    if a.out:
        p = Path(a.out) / "analytics" / "rebounds" / "validate.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1, default=str))
        summary["written"] = str(p)
    return summary


if __name__ == "__main__":
    main()

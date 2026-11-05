"""analytics/ghosting/validate.py: input checks and the loaders every ghosting module shares.

    python -m analytics.ghosting.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0] [--games 4]

Inputs per game (DATA_ROOT layout on the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet            frames, docs/DATA_CONTRACT.md section 4
    DATA_ROOT/sportvu_contract/<game>_pbp.parquet        play-by-play, section 5 (read only to count rows)
    DATA_ROOT/analytics/possessions/2015-16/<game>.parquet   section 19
    DATA_ROOT/analytics/rosters/2015-16/<game>.parquet       section 16 (optional for ghosting)
Held-out games live under sportvu_contract/heldout/ and analytics/<table>/heldout/2015-16/; a game is
also held out when its id carries a held-out token of sportvu/splits.json. A game needs frames and
possessions to be used; rosters are checked when present.

Checks (counts are logged with the ``analytics.ghosting`` logger and returned):
  frames       section 4 columns present; x in [-10, 104], y in [-10, 60] (SportVU strays a little past the
               lines), game_clock_s in [0, 720], shot_clock_s in [0, 24] or null, ball_z >= 0 or null;
               duplicate (frame, player_id) rows dropped, keeping the first; frames with n_players != 10
               counted; frames without a ball counted
  possessions  schema.validate against section 19; start_frame <= end_frame; spans inside the frames;
               duplicate possession numbers dropped; null offense_team_id counted; gap == true counted
  rosters      schema.validate against section 16; duplicate player_id dropped

Nothing here fits anything. ``load_games`` returns the cleaned tables; the trainer calls refuse_heldout on
the training ids itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema

log = logging.getLogger("analytics.ghosting")

REPO_ROOT = Path(__file__).resolve().parents[2]
SEASON = "2015-16"
FRAME_COLUMNS = ["game", "period", "frame", "game_clock_s", "shot_clock_s", "n_players", "ball_x", "ball_y", "ball_z",
                 "team_id", "player_id", "x", "y"]
RANGES = {"x": (-10.0, 104.0), "y": (-10.0, 60.0), "game_clock_s": (0.0, 720.0), "shot_clock_s": (0.0, 24.0),
          "ball_x": (-10.0, 104.0), "ball_y": (-10.0, 60.0), "ball_z": (0.0, 60.0)}


# ----------------------------------------------------------------------------- held-out ids

def is_heldout_id(sportvu_game: str) -> bool:
    """True when the id carries a held-out token of sportvu/splits.json (or sits in the CLE arena)."""
    from sportvu import splits
    g = str(sportvu_game)
    return any(t in g for t in splits.heldout_tokens()) or g.endswith(".at." + splits.HELDOUT_ARENA)


# ----------------------------------------------------------------------------- checks

def check_frames(frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """-> (cleaned frames, counts). Raises ValueError when a section 4 column is missing."""
    missing = [c for c in FRAME_COLUMNS if c not in frames.columns]
    if missing:
        raise ValueError("frames: missing section 4 column(s) %s" % missing)
    counts: dict = {"rows": int(len(frames))}
    dup = frames.duplicated(["frame", "player_id"], keep="first")
    counts["duplicate_rows_dropped"] = int(dup.sum())
    f = frames.loc[~dup].sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)
    for col, (lo, hi) in RANGES.items():
        v = pd.to_numeric(f[col], errors="coerce")
        counts["out_of_range_" + col] = int(((v < lo) | (v > hi)).sum())
    per = f.groupby("frame", sort=True).agg(n=("player_id", "size"), ball=("ball_x", "first"), gc=("game_clock_s", "first"))
    counts["frames"] = int(len(per))
    counts["frames_not_ten_players"] = int((per["n"] != 10).sum())
    counts["frames_without_ball"] = int(per["ball"].isna().sum())
    counts["null_shot_clock_rows"] = int(f["shot_clock_s"].isna().sum())
    return f, counts


def check_possessions(poss: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts: dict = {"rows": int(len(poss)), "schema_problems": schema.validate(poss, "possessions")}
    dup = poss.duplicated(["possession"], keep="first")
    counts["duplicate_rows_dropped"] = int(dup.sum())
    p = poss.loc[~dup].sort_values("possession", kind="stable").reset_index(drop=True)
    bad_span = p["start_frame"] > p["end_frame"]
    counts["start_after_end_dropped"] = int(bad_span.sum())
    p = p.loc[~bad_span].reset_index(drop=True)
    lo, hi = (int(frames["frame"].min()), int(frames["frame"].max())) if len(frames) else (0, -1)
    outside = (p["start_frame"] < lo) | (p["end_frame"] > hi)
    counts["span_outside_frames_dropped"] = int(outside.sum())
    p = p.loc[~outside].reset_index(drop=True)
    counts["null_offense_team"] = int(p["offense_team_id"].isna().sum())
    counts["gap"] = int(p["gap"].astype(bool).sum())
    counts["kept"] = int(len(p))
    return p, counts


def check_rosters(rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts: dict = {"rows": int(len(rosters)), "schema_problems": schema.validate(rosters, "rosters")}
    dup = rosters.duplicated(["player_id"], keep="first")
    counts["duplicate_rows_dropped"] = int(dup.sum())
    return rosters.loc[~dup].reset_index(drop=True), counts


def validate_game(game: dict) -> dict:
    """Clean the tables of one game dict in place and return its counts."""
    frames, fc = check_frames(game["frames"])
    poss, pc = check_possessions(game["possessions"], frames)
    game["frames"], game["possessions"] = frames, poss
    rep = {"sportvu_game": game["sportvu_game"], "heldout": bool(game["heldout"]), "frames": fc, "possessions": pc}
    if game.get("rosters") is not None:
        game["rosters"], rep["rosters"] = check_rosters(game["rosters"])
    for table, c in rep.items():
        if isinstance(c, dict):
            log.info("%s %s: %s", game["sportvu_game"], table, json.dumps(c))
    return rep


# ----------------------------------------------------------------------------- loaders

def _fixture_module():
    """The synthetic generator: analytics.fixtures when it exists, else tests/_fx_ghosting.py by path."""
    try:
        import analytics.fixtures as fx  # type: ignore[import-not-found]
        return fx
    except ImportError:
        pass
    path = REPO_ROOT / "tests" / "_fx_ghosting.py"
    spec = importlib.util.spec_from_file_location("_fx_ghosting", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_games(seed: int = 0, n_games: int = 4, n_possessions: int = 10, heldout_every: int = 0) -> list[dict]:
    """Synthetic games as game dicts (sportvu_game, heldout, frames, possessions, rosters)."""
    fx = _fixture_module()
    try:
        season = fx.synth_season(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    except TypeError:                       # the shared generator may not take n_possessions
        season = fx.synth_season(seed, n_games=n_games, heldout_every=heldout_every)
    out = []
    for i, g in enumerate(season):
        meta = g.get("meta") or {}
        gid = g.get("sportvu_game") or meta.get("sportvu_game") or str(g["possessions"]["sportvu_game"].iloc[0])
        held = bool(g.get("heldout", meta.get("heldout", is_heldout_id(gid))))
        out.append({"sportvu_game": gid, "heldout": held or is_heldout_id(gid), "frames": g["frames"],
                    "possessions": g["possessions"], "rosters": g.get("rosters"), "source": "fixture %d" % i})
    return out


def iter_data_root(data_root, season: str = SEASON) -> list[dict]:
    """Game dicts from the DATA_ROOT layout; a game is kept when it has frames and possessions."""
    root = Path(data_root)
    out = []
    for sub, held in (("", False), ("heldout", True)):
        d = root / "sportvu_contract" / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.parquet")):
            if p.name.endswith("_pbp.parquet"):
                continue
            gid = p.stem
            heldout = held or is_heldout_id(gid)
            pp = schema.table_path(root / "analytics", "possessions", season, gid, heldout=heldout)
            if not pp.exists():
                pp2 = schema.table_path(root / "analytics", "possessions", season, gid, heldout=not heldout)
                if pp2.exists():
                    pp, heldout = pp2, heldout or is_heldout_id(gid)
                else:
                    log.warning("%s: no possessions table at %s, skipped", gid, pp)
                    continue
            rp = schema.table_path(root / "analytics", "rosters", season, gid, heldout=heldout)
            out.append({"sportvu_game": gid, "heldout": heldout, "frames": pd.read_parquet(p),
                        "possessions": pd.read_parquet(pp), "rosters": pd.read_parquet(rp) if rp.exists() else None,
                        "source": str(p)})
    return out


def load_games(args) -> list[dict]:
    """The games a CLI run works on, cleaned by validate_game. ``args`` has fixtures, data, seed, games."""
    if getattr(args, "fixtures", False):
        games = fixture_games(seed=args.seed, n_games=args.games, n_possessions=getattr(args, "possessions", 10),
                              heldout_every=getattr(args, "heldout_every", 0))
    elif getattr(args, "data", None):
        games = iter_data_root(args.data)
    else:
        raise SystemExit("give --fixtures or --data DATA_ROOT")
    for g in games:
        g["validation"] = validate_game(g)
    return games


# ----------------------------------------------------------------------------- CLI

def add_input_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="synthetic games (fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with sportvu_contract/ and analytics/ (the Mac layout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=4, help="fixture games")
    ap.add_argument("--possessions", type=int, default=10, help="fixture possessions per game")
    ap.add_argument("--heldout-every", type=int, default=0, help="fixtures: every n-th game is the held-out game")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_input_args(ap)
    ap.add_argument("--out", help="OUT_ROOT: write ghosting_validate.json there")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = parse_args(argv)
    games = load_games(a)
    report = {"games": [g["validation"] for g in games], "n_games": len(games),
              "n_heldout": int(sum(g["heldout"] for g in games))}
    lines = ["| game | held out | frame rows | dup rows | not 10 players | no ball | possessions kept | gap | null offence | schema problems |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["games"]:
        f, p = r["frames"], r["possessions"]
        lines.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
            r["sportvu_game"], r["heldout"], f["rows"], f["duplicate_rows_dropped"], f["frames_not_ten_players"],
            f["frames_without_ball"], p["kept"], p["gap"], p["null_offense_team"], len(p["schema_problems"])))
    text = "\n".join(lines)
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ghosting_validate.json").write_text(json.dumps(report, indent=1))
        print("wrote", out / "ghosting_validate.json")
    return report


if __name__ == "__main__":
    main(sys.argv[1:])

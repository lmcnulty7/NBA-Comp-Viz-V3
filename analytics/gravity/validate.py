"""analytics/gravity/validate.py: input checks for the gravity lane, and the loaders every CLI of the lane shares.

    python -m analytics.gravity.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]
        [--fixture-games 8] [--fixture-possessions 40] [--fixture-heldout-every K] [--fixture-generator shared|pull]

Per game dict (keys frames, events, shots, rosters, shot_frames, possessions; as analytics.fixtures gives them):
  - schema.validate on every contract table present (rosters, shot_frames, rebound_frames, possessions,
    states_1hz; sections 16 to 20), problems listed per table
  - frames (section 4): required columns; ranges x in [X_RANGE], y in [Y_RANGE], game_clock_s in
    [0, CLOCK_MAX], shot_clock_s in [0, 24] or null, period >= 1; frames with other than ten rows counted
  - events (section 5): required columns, kind values outside the contract vocabulary counted
  - shots (section 10a): required columns, loc_x/loc_y finite, made boolean
  - missing: rows of frames with a null x or y are dropped and counted; frames without a ball are counted
    (features drops them); shots without a matchable shooter are counted
  - duplicates: (frame, player_id) in frames, (possession) in possessions, (player_id) in rosters are dropped
    keeping the first and counted
The cleaned dict is returned with the counts; nothing raises on a range problem (the counts say what was
wrong), only a missing required column raises ValueError.

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet, <game>_pbp.parquet         (held-out games under heldout/)
    DATA_ROOT/analytics/possessions/<season>/<game>.parquet               (held-out under heldout/<season>/)
    DATA_ROOT/analytics/rosters/<season>/<game>.parquet, shot_frames/...  (same rule)
    DATA_ROOT/collect/bref_shots/shotchart/<season>.parquet and collect/bref_shots/heldout/shotchart/<season>.parquet
A game is loaded when its possessions file and its frames file exist; the rest is optional and logged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema

X_RANGE, Y_RANGE, CLOCK_MAX, SHOT_CLOCK_MAX = (-5.0, 99.0), (-5.0, 55.0), 720.0, 24.0
FRAME_COLUMNS = ["game", "period", "frame", "game_clock_s", "ball_x", "ball_y", "team_id", "player_id", "x", "y"]
EVENT_COLUMNS = ["event", "period", "clock_s", "kind", "frame"]
SHOT_COLUMNS = ["sportvu_game", "player", "team", "period", "clock_s", "made", "loc_x", "loc_y"]
EVENT_KINDS = {"shot_made", "shot_missed", "ft_made", "ft_missed", "turnover", "rebound", "foul", "other", "substitution",
               "timeout", "jump_ball", "period_start", "period_end", "violation", "ejection", "instant_replay"}
CONTRACT_TABLES = ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("gravity.validate: %s lacks column(s) %s" % (what, missing))


def _count_out_of_range(s: pd.Series, lo: float, hi: float) -> int:
    v = pd.to_numeric(s, errors="coerce")
    return int(((v < lo) | (v > hi)).sum())


def validate_game(d: dict) -> tuple[dict, dict]:
    """(cleaned game dict, counts). See the module docstring for what is checked. counts carries heldout (the dict's
    flag, or analytics.eval.is_heldout_game on the id), so a summary never names a held-out game unflagged."""
    from analytics.gravity.features import game_id, is_heldout, shot_player_ids
    g = game_id(d)
    out = dict(d)
    counts = {"sportvu_game": g, "heldout": bool(is_heldout(d)), "schema": {}, "frames": {}, "events": {}, "shots": {}, "dropped": {}}
    for t in CONTRACT_TABLES:
        if d.get(t) is not None and len(d[t]):
            counts["schema"][t] = schema.validate(d[t], t)
    fr = d["frames"]
    _require(fr, FRAME_COLUMNS, "frames")
    n0 = len(fr)
    fr = fr[fr["x"].notna() & fr["y"].notna()]
    counts["dropped"]["frames_null_xy"] = int(n0 - len(fr))
    n1 = len(fr)
    fr = fr.drop_duplicates(["frame", "player_id"], keep="first")
    counts["dropped"]["frames_duplicate_frame_player"] = int(n1 - len(fr))
    rows_per_frame = fr.groupby("frame").size()
    counts["frames"] = {"rows": int(len(fr)), "frames": int(rows_per_frame.size),
                        "frames_not_ten_rows": int((rows_per_frame != 10).sum()),
                        "x_out_of_range": _count_out_of_range(fr["x"], *X_RANGE),
                        "y_out_of_range": _count_out_of_range(fr["y"], *Y_RANGE),
                        "clock_out_of_range": _count_out_of_range(fr["game_clock_s"], 0.0, CLOCK_MAX),
                        "shot_clock_out_of_range": _count_out_of_range(fr["shot_clock_s"].dropna(), 0.0, SHOT_CLOCK_MAX)
                        if "shot_clock_s" in fr.columns else 0,
                        "shot_clock_null_frames": int(fr.drop_duplicates("frame")["shot_clock_s"].isna().sum())
                        if "shot_clock_s" in fr.columns else int(rows_per_frame.size),
                        "ball_null_frames": int((fr.drop_duplicates("frame")["ball_x"].isna()).sum()),
                        "period_under_one": int((pd.to_numeric(fr["period"], errors="coerce") < 1).sum())}
    out["frames"] = fr.reset_index(drop=True)
    ev = d.get("events")
    if ev is not None and len(ev):
        _require(ev, EVENT_COLUMNS, "events")
        counts["events"] = {"rows": int(len(ev)), "unknown_kind": int((~ev["kind"].astype(str).isin(EVENT_KINDS)).sum()),
                            "frame_not_found": int((pd.to_numeric(ev["frame"], errors="coerce") < 0).sum())}
    poss = d.get("possessions")
    if poss is not None and len(poss):
        n2 = len(poss)
        poss = poss.drop_duplicates(["possession"], keep="first")
        counts["dropped"]["possessions_duplicate"] = int(n2 - len(poss))
        counts["possessions"] = {"rows": int(len(poss)), "gap": int(poss["gap"].astype(bool).sum()),
                                 "offense_null": int(poss["offense_team_id"].isna().sum())}
        out["possessions"] = poss.reset_index(drop=True)
    ros = d.get("rosters")
    if ros is not None and len(ros):
        n3 = len(ros)
        ros = ros.drop_duplicates(["player_id"], keep="first")
        counts["dropped"]["rosters_duplicate_player"] = int(n3 - len(ros))
        out["rosters"] = ros.reset_index(drop=True)
    sh = d.get("shots")
    if sh is not None and len(sh):
        _require(sh, SHOT_COLUMNS, "shots")
        lx, ly = pd.to_numeric(sh["loc_x"], errors="coerce"), pd.to_numeric(sh["loc_y"], errors="coerce")
        pid = shot_player_ids(sh, out.get("rosters"))
        counts["shots"] = {"rows": int(len(sh)), "loc_not_finite": int((~np.isfinite(lx) | ~np.isfinite(ly)).sum()),
                           "made_not_bool": int(sh["made"].map(lambda v: not isinstance(v, (bool, np.bool_))).sum()),
                           "shooter_unmatched": int(pid.isna().sum())}
    return out, counts


def validate_games(games) -> tuple[list, list]:
    """validate_game over a list of game dicts -> (cleaned list, counts list)."""
    cleaned, counts = [], []
    for d in games:
        c, n = validate_game(d)
        cleaned.append(c)
        counts.append(n)
    return cleaned, counts


# ── loaders ────────────────────────────────────────────────────────────────────────────────────────
def _read_if(p: Path):
    """The parquet at p, or None when it does not exist (the optional tables)."""
    return pd.read_parquet(p) if p.exists() else None


def load_data_root(root, season: str = "2015-16", log=print) -> list:
    """Game dicts from the owner's DATA_ROOT layout (module docstring). Held-out games come from the heldout/
    folders and carry heldout True; the shots table is split per game from the season file(s)."""
    root = Path(root)
    shots_all = []
    for p in (root / "collect" / "bref_shots" / "shotchart" / ("%s.parquet" % season),
              root / "collect" / "bref_shots" / "heldout" / "shotchart" / ("%s.parquet" % season)):
        s = _read_if(p)
        if s is not None:
            shots_all.append(s)
        else:
            log("gravity.validate: no shots file %s" % p)
    shots_all = pd.concat(shots_all, ignore_index=True) if shots_all else None
    games = []
    for held, pdir in ((False, root / "analytics" / "possessions" / season),
                       (True, root / "analytics" / "possessions" / "heldout" / season)):
        if not pdir.exists():
            continue
        for pp in sorted(pdir.glob("*.parquet")):
            g = pp.stem
            fdir = root / "sportvu_contract" / ("heldout" if held else "")
            frames = _read_if(fdir / ("%s.parquet" % g))
            if frames is None:
                log("gravity.validate: %s has possessions but no frames file, skipped" % g)
                continue
            sub = "heldout/%s" % season if held else season
            d = {"sportvu_game": g, "heldout": held, "frames": frames, "possessions": pd.read_parquet(pp),
                 "events": _read_if(fdir / ("%s_pbp.parquet" % g)),
                 "rosters": _read_if(root / "analytics" / "rosters" / sub / ("%s.parquet" % g)),
                 "shot_frames": _read_if(root / "analytics" / "shot_frames" / sub / ("%s.parquet" % g)),
                 "rebound_frames": None, "states_1hz": None,
                 "shots": shots_all[shots_all["sportvu_game"].astype(str) == g].reset_index(drop=True)
                 if shots_all is not None else None}
            games.append(d)
    log("gravity.validate: loaded %d game(s) from %s" % (len(games), root))
    return games


GENERATORS = {"shared": "analytics.fixtures", "pull": "analytics.gravity.fixture_pull"}
GENERATOR_ALIASES = {"auto": "shared", "fallback": "pull"}     # the spellings of earlier runs


def generator_name(name: str = "shared") -> str:
    """The canonical generator name ("shared" or "pull") of a --fixture-generator value (aliases auto, fallback)."""
    name = GENERATOR_ALIASES.get(name, name)
    if name not in GENERATORS:
        raise SystemExit("gravity: unknown fixture generator %r (shared or pull)" % name)
    return name


def fixture_generator(name: str = "shared"):
    """The season generator for --fixtures: "shared" is analytics.fixtures.synth_season, "pull" the lane's synthetic
    planted-pull generator analytics.gravity.fixture_pull.synth_pull_season."""
    if generator_name(name) == "pull":
        from analytics.gravity.fixture_pull import synth_pull_season
        return synth_pull_season
    from analytics.fixtures import synth_season
    return synth_season


def fixture_source(name: str = "shared") -> str:
    """The module name of the generator fixture_generator(name) resolves to."""
    return GENERATORS[generator_name(name)]


def fixture_heldout_every(a) -> int:
    """--fixture-heldout-every, defaulting to every fourth game held out when the season has at least four games."""
    k = getattr(a, "fixture_heldout_every", None)
    if k is not None:
        return int(k)
    return 4 if int(getattr(a, "fixture_games", 8)) >= 4 else 0


def load_games(a, who: str = "gravity") -> list:
    """The game dicts a CLI of the lane asked for: --fixtures (a season of fixture games from the generator
    --fixture-generator names) or --data DATA_ROOT."""
    if getattr(a, "fixtures", False):
        gen = fixture_generator(getattr(a, "fixture_generator", "shared"))
        return gen(int(getattr(a, "seed", 0)), n_games=int(getattr(a, "fixture_games", 8)), heldout_every=fixture_heldout_every(a),
                   n_possessions=int(getattr(a, "fixture_possessions", 40)))
    if getattr(a, "data", None):
        return load_data_root(a.data, getattr(a, "season", "2015-16"))
    raise SystemExit("%s: pass --fixtures or --data DATA_ROOT" % who)


def add_input_args(ap: argparse.ArgumentParser) -> None:
    """The input flags every CLI of the lane shares: the source (--fixtures or --data) and the fixture knobs."""
    ap.add_argument("--fixtures", action="store_true", help="run on a synthetic season (fixture output, not a result)")
    ap.add_argument("--data", default=None, help="DATA_ROOT in the owner's Mac layout (module docstring)")
    ap.add_argument("--season", default="2015-16", help="season folder under DATA_ROOT (default 2015-16)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed, also the seed of folds and bootstrap (default 0)")
    ap.add_argument("--fixture-games", "--n-games", dest="fixture_games", type=int, default=8,
                    help="--fixtures: games in the synthetic season (default 8)")
    ap.add_argument("--fixture-possessions", "--n-possessions", dest="fixture_possessions", type=int, default=40,
                    help="--fixtures: possessions per synthetic game (default 40)")
    ap.add_argument("--fixture-heldout-every", dest="fixture_heldout_every", type=int, default=None,
                    help="--fixtures: hold out every k-th game, 0 for none (default 4 with at least four games, else 0)")
    ap.add_argument("--fixture-generator", default="shared", choices=("shared", "pull", "auto", "fallback"),
                    help="--fixtures: shared is analytics.fixtures, pull the lane's planted-pull generator "
                         "(analytics/gravity/fixture_pull.py); auto and fallback are the older spellings of the two")


def main(argv=None) -> dict:
    """CLI: validate the inputs, print the summary, write validate.json under --out."""
    ap = argparse.ArgumentParser(description="validate the gravity lane inputs")
    add_input_args(ap)
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes OUT_ROOT/analytics/gravity/validate.json")
    a = ap.parse_args(argv)
    games = load_games(a, who="gravity.validate")
    _, counts = validate_games(games)
    summary = {"games": len(games), "fixture": bool(a.fixtures), "per_game": counts,
               "schema_problems": int(sum(len(v) for c in counts for v in c["schema"].values())),
               "dropped": {k: int(sum(c["dropped"].get(k, 0) for c in counts))
                           for k in sorted({k for c in counts for k in c["dropped"]})}}
    if a.out:
        p = Path(a.out) / "analytics" / "gravity" / "validate.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1, default=str))
        summary["written"] = str(p)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_game"}, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()

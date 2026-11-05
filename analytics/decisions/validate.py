"""analytics/decisions/validate.py: inputs of the decision-quality lane, validated.

    python -m analytics.decisions.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

A game is carried as a bundle: dict(sportvu_game, heldout, frames, events, shot_frames, rebound_frames,
possessions, states_1hz, rosters, epv, xfg). frames and events are the section 4 and 5 tables; the five
Phase 0 tables follow analytics.schema; epv and xfg are the upstream lanes' outputs when present (None otherwise).

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet and <game>_pbp.parquet     (held-out games under heldout/)
    DATA_ROOT/analytics/<table>/<season>/<game>.parquet                 (held-out under heldout/<season>/)
    OUT_ROOT/analytics/epv/<season>/<game>.parquet                       (lane EPV, optional)
    OUT_ROOT/analytics/xfg/<season>/*.parquet                            (lane XFG, optional)
The games are the possessions files; a game without frames is skipped and logged.

Checks per bundle (validate_bundle): schema.validate on each Phase 0 table present; range checks (player x in
[-5, 99] ft, y in [-5, 55] ft, ball z in [-1, 40] ft, possession spans inside the frames, release frames inside
the game); duplicates ((frame, player_id) in frames, possession ids, shot events) dropped with logged counts;
missing values counted (frames without a ball, possessions without an offence). Nothing is modified on disk.

Assumption: a held-out game is one written under heldout/, or whose id carries a held-out token of
sportvu/splits.json; the flag follows the bundle so train.py never fits on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema

PHASE0_TABLES = ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")
FRAME_COLUMNS = ["frame", "period", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "home", "player_id", "x", "y"]
FRAME_OPTIONAL = ["ts_ms", "shot_clock_s"]
EVENT_COLUMNS = ["event", "period", "clock_s", "kind", "frame", "points"]
X_RANGE, Y_RANGE, Z_RANGE = (-5.0, 99.0), (-5.0, 55.0), (-1.0, 40.0)
DEFAULT_SEASON = "2015-16"


# ── held-out ───────────────────────────────────────────────────────────────────────────────────────
def is_heldout_id(sportvu_game: str) -> bool:
    """True when the game id carries a held-out token of sportvu/splits.json (or the held-out arena suffix)."""
    from sportvu import splits
    toks = splits.heldout_tokens()
    g = str(sportvu_game)
    return any(t in g for t in toks) or g.endswith(".at." + splits.HELDOUT_ARENA)


# ── loading ────────────────────────────────────────────────────────────────────────────────────────
def _fixture_module():
    try:
        from analytics import fixtures as fx  # type: ignore
        return fx
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tests import _fx_decisions as fx  # type: ignore
        return fx


def _as_bundle(game: dict, name=None, heldout=None) -> dict:
    meta = game.get("meta") or {}
    if heldout is None:
        heldout = game.get("heldout", meta.get("heldout", False))
    b = {"sportvu_game": name or game.get("sportvu_game") or meta.get("sportvu_game") or str(game["possessions"]["sportvu_game"].iloc[0]),
         "heldout": bool(heldout),
         "frames": game["frames"], "events": game.get("events"), "epv": game.get("epv"), "xfg": game.get("xfg")}
    for t in PHASE0_TABLES:
        b[t] = game.get(t)
    return b


def load_fixture_bundles(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40,
                         season_fn=None) -> list:
    """Synthetic games from analytics.fixtures when present, else tests/_fx_decisions (same signatures).
    season_fn overrides the generator: any synth_season(seed, n_games, heldout_every[, n_possessions])."""
    fn = season_fn or _fixture_module().synth_season
    try:
        season = fn(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    except TypeError:
        season = fn(seed, n_games=n_games, heldout_every=heldout_every)
    if isinstance(season, dict):
        season = [dict(g, sportvu_game=g.get("sportvu_game", k)) for k, g in season.items()]
    out = []
    for i, g in enumerate(season):
        b = _as_bundle(g)
        if n_possessions and len(b["possessions"]) > n_possessions:   # the shared generator may ignore the count
            keep = b["possessions"]["possession"] < n_possessions
            b["possessions"] = b["possessions"][keep].reset_index(drop=True)
        out.append(b)
    return out


def _read(path: Path, columns=None):
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:  # noqa: BLE001  (a column subset the file lacks: read everything)
        return pd.read_parquet(path)


def load_root_bundles(data_root, season: str = DEFAULT_SEASON, out_root=None, games=None, log=None) -> list:
    """Bundles for every game with a possessions file under DATA_ROOT (held-out under heldout/)."""
    data_root = Path(data_root)
    out_root = Path(out_root) if out_root else data_root
    log = log if log is not None else []
    found = []
    for held in (False, True):
        d = data_root / "analytics" / "possessions" / ("heldout/" + season if held else season)
        for p in sorted(d.glob("*.parquet")):
            found.append((p.stem, held))
    if games:
        found = [(g, h) for g, h in found if g in set(games)]
    bundles = []
    for g, held in found:
        held = held or is_heldout_id(g)
        sv = data_root / "sportvu_contract" / ("heldout" if held else "")
        frames = _read(sv / (g + ".parquet"), FRAME_COLUMNS + FRAME_OPTIONAL)
        if frames is None:
            frames = _read(data_root / "sportvu_contract" / (g + ".parquet"), FRAME_COLUMNS + FRAME_OPTIONAL)
        if frames is None:
            log.append("%s: no frames file under sportvu_contract, skipped" % g)
            continue
        events = _read(sv / (g + "_pbp.parquet"))
        game = {"sportvu_game": g, "heldout": held, "frames": frames, "events": events}
        for t in PHASE0_TABLES:
            game[t] = _read(schema.table_path(data_root / "analytics", t, season, g, heldout=held))
        if game["possessions"] is None or game["shot_frames"] is None:
            log.append("%s: possessions or shot_frames missing, skipped" % g)
            continue
        epv_dir = out_root / "analytics" / "epv" / ("heldout/" + season if held else season)
        game["epv"] = _read(epv_dir / (g + ".parquet"))
        xfg_dir = out_root / "analytics" / "xfg" / ("heldout/" + season if held else season)
        parts = [x for x in (_read(p) for p in sorted(xfg_dir.glob("*.parquet"))) if x is not None]
        if parts:
            x = pd.concat(parts, ignore_index=True)
            x = x[x["sportvu_game"].astype(str) == g] if "sportvu_game" in x.columns else x
            game["xfg"] = x.reset_index(drop=True) if len(x) else None
        else:
            game["xfg"] = None
        bundles.append(_as_bundle(game))
    return bundles


# ── checks ─────────────────────────────────────────────────────────────────────────────────────────
def _range_count(s: pd.Series, lo: float, hi: float) -> int:
    v = pd.to_numeric(s, errors="coerce")
    return int(((v < lo) | (v > hi)).sum())


def validate_bundle(bundle: dict) -> tuple[dict, dict]:
    """-> (cleaned bundle, report). The report holds schema problems per table, range violations, duplicate and
    missing counts. Duplicates are dropped in the cleaned copy; everything else is only counted."""
    rep = {"sportvu_game": bundle["sportvu_game"], "heldout": bool(bundle["heldout"]), "schema": {}, "range": {},
           "duplicates": {}, "missing": {}, "rows": {}}
    out = dict(bundle)
    fr = bundle["frames"]
    missing_cols = [c for c in FRAME_COLUMNS if c not in fr.columns]
    rep["schema"]["frames"] = ["missing column %r" % c for c in missing_cols]
    if missing_cols:
        return out, rep
    rep["rows"]["frames"] = int(len(fr))
    n0 = len(fr)
    fr = fr.drop_duplicates(["frame", "player_id"], keep="first")
    rep["duplicates"]["frames"] = int(n0 - len(fr))
    rep["range"]["frames.x"] = _range_count(fr["x"], *X_RANGE)
    rep["range"]["frames.y"] = _range_count(fr["y"], *Y_RANGE)
    rep["range"]["frames.ball_z"] = _range_count(fr["ball_z"].dropna(), *Z_RANGE)
    rep["missing"]["frames.ball"] = int(fr.drop_duplicates("frame")["ball_x"].isna().sum())
    if not fr["frame"].is_monotonic_increasing:
        fr = fr.sort_values("frame", kind="stable")
        rep["range"]["frames.unsorted"] = 1
    out["frames"] = fr.reset_index(drop=True)
    fmin, fmax = int(fr["frame"].min()), int(fr["frame"].max())
    ev = bundle.get("events")
    if ev is not None:
        rep["schema"]["events"] = ["missing column %r" % c for c in EVENT_COLUMNS if c not in ev.columns]
        rep["rows"]["events"] = int(len(ev))
    for t in PHASE0_TABLES:
        df = bundle.get(t)
        if df is None:
            rep["missing"][t] = "absent"
            continue
        rep["rows"][t] = int(len(df))
        rep["schema"][t] = schema.validate(df, t)
        key = {"rosters": ["player_id"], "shot_frames": ["event"], "rebound_frames": ["event"],
               "possessions": ["possession"], "states_1hz": ["possession", "frame"]}[t]
        if all(k in df.columns for k in key):
            n0 = len(df)
            df = df.drop_duplicates(key, keep="first").reset_index(drop=True)
            rep["duplicates"][t] = int(n0 - len(df))
        out[t] = df
    poss = out.get("possessions")
    if poss is not None and {"start_frame", "end_frame"} <= set(poss.columns):
        s, e = pd.to_numeric(poss["start_frame"]), pd.to_numeric(poss["end_frame"])
        rep["range"]["possessions.span"] = int(((e < s) | (s < fmin) | (e > fmax)).sum())
        if "offense_team_id" in poss.columns:
            rep["missing"]["possessions.offense_team_id"] = int(poss["offense_team_id"].isna().sum())
    sf = out.get("shot_frames")
    if sf is not None and "release_frame" in sf.columns:
        r = pd.to_numeric(sf["release_frame"])
        rep["missing"]["shot_frames.release_frame"] = int((r == schema.NOT_FOUND).sum())
        rep["range"]["shot_frames.release_frame"] = int(((r != schema.NOT_FOUND) & ((r < fmin) | (r > fmax))).sum())
        if "shooter_id" in sf.columns:
            rep["missing"]["shot_frames.shooter_id"] = int(sf["shooter_id"].isna().sum())
    for k in ("epv", "xfg"):
        df = bundle.get(k)
        rep["rows"][k] = int(len(df)) if df is not None else 0
    return out, rep


def validate_bundles(bundles: list) -> tuple[list, list]:
    cleaned, reports = [], []
    for b in bundles:
        c, r = validate_bundle(b)
        cleaned.append(c)
        reports.append(r)
    return cleaned, reports


def summary_lines(reports: list) -> list:
    L = []
    for r in reports:
        probs = sum(len(v) for v in r["schema"].values())
        dups = sum(r["duplicates"].values())
        rng = sum(v for v in r["range"].values() if isinstance(v, int))
        L.append("%s%s: frames %s, possessions %s, shots %s, schema problems %d, duplicates dropped %d, range violations %d, "
                 "epv rows %d, xfg rows %d"
                 % (r["sportvu_game"], " (held out)" if r["heldout"] else "", r["rows"].get("frames", 0),
                    r["rows"].get("possessions", 0), r["rows"].get("shot_frames", 0), probs, dups, rng,
                    r["rows"].get("epv", 0), r["rows"].get("xfg", 0)))
        for t, p in r["schema"].items():
            for msg in p:
                L.append("    %s: %s" % (t, msg))
    return L


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT (upstream lanes' outputs; default DATA_ROOT)")
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-games", type=int, default=8)
    ap.add_argument("--json", help="write the per-game report here")
    return ap.parse_args(argv)


def main(argv=None) -> list:
    a = parse_args(argv)
    if not a.fixtures and not a.data:
        raise SystemExit("validate: pass --fixtures or --data DATA_ROOT")
    log = []
    bundles = load_fixture_bundles(a.seed, a.n_games) if a.fixtures else load_root_bundles(a.data, a.season, a.out, log=log)
    _, reports = validate_bundles(bundles)
    for line in log + summary_lines(reports):
        print(line)
    if a.fixtures:
        print("fixture output, not a result")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"log": log, "games": reports}, indent=1, default=str))
    return reports


if __name__ == "__main__":
    main()

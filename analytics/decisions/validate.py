"""analytics/decisions/validate.py: inputs of the decision-quality lane, validated.

    python -m analytics.decisions.validate --fixtures [--fixture-games 8] | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

A game is carried as a bundle: dict(sportvu_game, heldout, frames, events, shot_frames, rebound_frames,
possessions, states_1hz, rosters, epv, xfg). frames and events are the section 4 and 5 tables; the five
Phase 0 tables follow analytics.schema; epv and xfg are the upstream lanes' outputs when present (None otherwise).

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet and <game>_pbp.parquet     (held-out games under heldout/)
    DATA_ROOT/analytics/<table>/<season>/<game>.parquet                 (held-out under heldout/<season>/)
    OUT_ROOT/analytics/epv/<season>/<game>.parquet                       (lane EPV, optional)
    OUT_ROOT/analytics/xfg/<season>/*.parquet                            (lane XFG, optional)
The games are the possessions files; a game without frames is skipped and logged. OUT_ROOT defaults to DATA_ROOT,
and both are searched, so the xFG and EPV outputs under DATA_ROOT/analytics are found without --out.
Frames are read one game at a time: on DATA_ROOT a bundle's "frames" is a zero-argument loader (frames_of reads
it), and stream_bundles reads, checks and uses one game's frames, then drops them before the next game is read.

Checks per bundle (validate_bundle): schema.validate on each Phase 0 table present; range checks (player x in
[-5, 99] ft, y in [-5, 55] ft, ball z in [-1, 40] ft, possession spans inside the frames, release frames inside
the game); duplicates ((frame, player_id) in frames, possession ids, shot events) dropped with logged counts;
missing values counted (frames without a ball, possessions without an offence). Nothing is modified on disk.

Assumption: a held-out game is one written under heldout/, or whose id carries a held-out token of
sportvu/splits.json (analytics.eval.is_heldout_game); the flag follows the bundle so train.py never fits on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analytics import fixtures, schema
from analytics.eval import is_heldout_game

PHASE0_TABLES = ("rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")
FRAME_COLUMNS = ["frame", "period", "game_clock_s", "ball_x", "ball_y", "ball_z", "team_id", "home", "player_id", "x", "y"]
FRAME_OPTIONAL = ["ts_ms", "shot_clock_s"]
EVENT_COLUMNS = ["event", "period", "clock_s", "kind", "frame", "points"]
X_RANGE, Y_RANGE, Z_RANGE = (-5.0, 99.0), (-5.0, 55.0), (-1.0, 40.0)
DEFAULT_SEASON = "2015-16"
XFG_SKIP_STEMS = ("xfg_features",)        # the xFG lane's feature table sits beside its predictions: not a prediction table


def _analytics_roots(out_root: Path, data_root: Path) -> list:
    """Where the Phase 0 tables and the upstream lanes' outputs are looked for, in order: OUT_ROOT/analytics
    first (the Mac may write there), then DATA_ROOT/analytics; the same root once when OUT_ROOT is DATA_ROOT."""
    roots = [Path(out_root) / "analytics", Path(data_root) / "analytics"]
    return roots if roots[0].resolve() != roots[1].resolve() else roots[:1]


# ── loading ────────────────────────────────────────────────────────────────────────────────────────
def _as_bundle(game: dict, name=None, heldout=None) -> dict:
    """A generator or DATA_ROOT game dict as a lane bundle (the keys listed in the module docstring)."""
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
    """Synthetic games from analytics.fixtures.synth_season (fixture output, never a result). season_fn overrides
    the generator: any synth_season(seed, n_games=, heldout_every=, n_possessions=) returning a list of game dicts."""
    fn = season_fn or fixtures.synth_season
    season = fn(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    return [_as_bundle(g) for g in season]


def _read(path: Path, columns=None):
    """The parquet file at path (None when absent). With columns, a file that lacks one of the optional columns is
    read whole: only that schema mismatch is retried, any other fault (a corrupt file, a missing engine) raises."""
    if not path.exists():
        return None
    if columns is None:
        return pd.read_parquet(path)
    import pyarrow.parquet as pq
    present = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in columns if c in present])


def _frames_loader(path: Path):
    """A zero-argument loader of one game's frames parquet (read on each call, nothing kept)."""
    return lambda: _read(path, FRAME_COLUMNS + FRAME_OPTIONAL)


def frames_of(bundle: dict) -> pd.DataFrame:
    """The frames of a bundle: called when bundle["frames"] is a loader, else the DataFrame itself."""
    f = bundle["frames"]
    return f() if callable(f) else f


def load_root_bundles(data_root, season: str = DEFAULT_SEASON, out_root=None, games=None, log=None) -> list:
    """Bundles for every game with a possessions file under DATA_ROOT (held-out under heldout/)."""
    data_root = Path(data_root)
    out_root = Path(out_root) if out_root else data_root
    log = log if log is not None else []
    found, seen = [], set()
    for held in (False, True):
        for root in _analytics_roots(out_root, data_root):
            d = root / "possessions" / ("heldout/" + season if held else season)
            for p in sorted(d.glob("*.parquet")):
                if (p.stem, held) not in seen:
                    seen.add((p.stem, held))
                    found.append((p.stem, held))
    if games:
        found = [(g, h) for g, h in found if g in set(games)]
    bundles, xfg_cache = [], {}
    for g, held in found:
        held = held or is_heldout_game(g)
        sv = data_root / "sportvu_contract" / ("heldout" if held else "")
        # frames as a loader: no game's frames are read here
        fpath = sv / (g + ".parquet")
        if not fpath.exists():
            fpath = data_root / "sportvu_contract" / (g + ".parquet")
        if not fpath.exists():
            log.append("%s: no frames file under sportvu_contract, skipped" % g)
            continue
        events = _read(sv / (g + "_pbp.parquet"))
        game = {"sportvu_game": g, "heldout": held, "frames": _frames_loader(fpath), "events": events}
        # Phase 0 tables: OUT_ROOT/analytics first (the Mac may write them there), then DATA_ROOT/analytics, per table
        for t in PHASE0_TABLES:
            game[t] = None
            for root in _analytics_roots(out_root, data_root):
                game[t] = _read(schema.table_path(root, t, season, g, heldout=held))
                if game[t] is not None:
                    break
        if game["possessions"] is None or game["shot_frames"] is None:
            log.append("%s: possessions or shot_frames missing, skipped" % g)
            continue
        game["epv"] = None
        for root in _analytics_roots(out_root, data_root):
            epv_dir = root / "epv" / ("heldout/" + season if held else season)
            game["epv"] = _read(epv_dir / (g + ".parquet"))
            if game["epv"] is not None:
                break
        # the xFG lane writes OUT_ROOT/analytics/xfg/<season>/xfg_predictions.parquet with held-out rows flagged
        # heldout=True, fold=-1 in the same file; a heldout/<season>/ directory is read too when it exists; the
        # feature table (xfg_features*.parquet) beside it is not a prediction table and is skipped
        xfg_dirs = []
        for root in _analytics_roots(out_root, data_root):
            xfg_dirs.append(root / "xfg" / season)
            if held:
                xfg_dirs.append(root / "xfg" / "heldout" / season)
        # the season tables are read once per set of directories, then filtered per game
        key = tuple(xfg_dirs)
        if key not in xfg_cache:
            parts = [x for d in xfg_dirs for x in (_read(p) for p in sorted(d.glob("*.parquet"))
                                                   if not p.stem.startswith(XFG_SKIP_STEMS)) if x is not None]
            xfg_cache[key] = pd.concat(parts, ignore_index=True) if parts else None
        x = xfg_cache[key]
        if x is not None:
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
    missing counts. Duplicates are dropped in the cleaned copy; everything else is only counted. A frames loader is
    called here and the cleaned bundle carries the frames as a DataFrame (stream_bundles then drops them)."""
    rep = {"sportvu_game": bundle["sportvu_game"], "heldout": bool(bundle["heldout"]), "schema": {}, "range": {},
           "duplicates": {}, "missing": {}, "rows": {}}
    out = dict(bundle)
    fr = frames_of(bundle)
    out["frames"] = fr
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


def clean_frames(fr: pd.DataFrame) -> pd.DataFrame:
    """The frames as validate_bundle cleans them: duplicate (frame, player_id) rows dropped, sorted by frame."""
    fr = fr.drop_duplicates(["frame", "player_id"], keep="first")
    if not fr["frame"].is_monotonic_increasing:
        fr = fr.sort_values("frame", kind="stable")
    return fr.reset_index(drop=True)


def _cleaned_loader(load):
    """A zero-argument loader of one game's cleaned frames, from the loader of its raw frames (not wrapped twice)."""
    if getattr(load, "cleaned", False):
        return load

    def cleaned():
        return clean_frames(load())
    cleaned.cleaned = True
    return cleaned


def stream_bundles(bundles: list, per_game=None) -> tuple[list, list, list]:
    """validate_bundle on each bundle, one game at a time -> (cleaned bundles, reports, per_game results), in input
    order. per_game(cleaned bundle) runs while that game's frames are in memory and its result is kept. A bundle
    whose frames came from a loader keeps only a loader of its cleaned frames afterwards, so at most one game's
    frames are alive at a time. A bundle whose frames came as a DataFrame (fixtures) keeps the cleaned DataFrame."""
    cleaned, reports, results = [], [], []
    for b in bundles:
        c, r = validate_bundle(b)
        results.append(per_game(c) if per_game is not None else None)
        if callable(b.get("frames")) and not r["schema"]["frames"]:
            c["frames"] = _cleaned_loader(b["frames"])     # drop this game's frames before the next is read
        elif callable(b.get("frames")):
            c["frames"] = b["frames"]                       # frames missing a column: left as read, not cleaned
        cleaned.append(c)
        reports.append(r)
        del c
    return cleaned, reports, results


def validate_bundles(bundles: list) -> tuple[list, list]:
    """validate_bundle on each bundle -> (cleaned bundles, reports), in input order, one game's frames at a time."""
    cleaned, reports, _ = stream_bundles(bundles)
    return cleaned, reports


def summary_lines(reports: list) -> list:
    """One printable line per game report, then one indented line per schema problem."""
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
def add_fixture_args(ap: argparse.ArgumentParser, n_games: int = 8, n_possessions: int = 40, heldout_every: int = 4) -> None:
    """The fixture knobs every CLI of the lane takes: canonical --fixture-* spellings, the older ones kept as aliases."""
    ap.add_argument("--fixture-games", "--n-games", dest="n_games", type=int, default=n_games,
                    help="fixture: number of synthetic games (default %(default)s)")
    ap.add_argument("--fixture-possessions", "--n-possessions", dest="n_possessions", type=int, default=n_possessions,
                    help="fixture: possessions per synthetic game (default %(default)s)")
    ap.add_argument("--fixture-heldout-every", "--heldout-every", dest="heldout_every", type=int, default=heldout_every,
                    help="fixture: every k-th synthetic game is held out, 0 for none (default %(default)s)")


def parse_args(argv=None):
    """The validate CLI's arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true", help="validate synthetic games from analytics.fixtures")
    ap.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT (upstream lanes' outputs; default DATA_ROOT)")
    ap.add_argument("--season", default=DEFAULT_SEASON, help="season directory under DATA_ROOT/analytics (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    add_fixture_args(ap)
    ap.add_argument("--json", help="write the per-game report here")
    return ap.parse_args(argv)


def main(argv=None) -> list:
    """Load, validate and print the per-game reports; returns them."""
    a = parse_args(argv)
    if not a.fixtures and not a.data:
        raise SystemExit("validate: pass --fixtures or --data DATA_ROOT")
    log = []
    bundles = load_fixture_bundles(a.seed, a.n_games, a.heldout_every, a.n_possessions) if a.fixtures else load_root_bundles(a.data, a.season, a.out, log=log)
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

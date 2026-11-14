"""analytics/epv/states.py: the section 20 states_1hz table from frames (section 4) and possessions (section 19).

    python -m analytics.epv.states --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--tol 0.5] [--fixture-games 4]
                                   [--fixture-possessions 40] [--fixture-heldout-every 4]

writes OUT_ROOT/analytics/states_1hz/<season>/<game>.parquet (held-out games under heldout/<season>/) for every game
that has a possessions file and a frames file. With --data, OUT_ROOT defaults to DATA_ROOT; with --fixtures the
synthetic season of analytics.fixtures is built in memory and written under a scratch directory (never into the
repo). <season> is analytics.schema.season_of_game.

Rules (docs/DATA_CONTRACT.md section 20) and the assumptions taken where it is silent:
  - one row per possession per whole second of elapsed play: elapsed is measured on the game clock
    (start_clock_s - game_clock_s), so a stalled clock inside a possession (a dead ball) adds no states and a
    frame gap is skipped rather than interpolated; the state is the span frame whose clock is nearest the whole
    second and it is kept only when that clock is within --tol seconds (0.5) of it;
  - the ten players of the frame are split by team_id against offense_team_id; a frame without exactly five
    offensive and five defensive rows is skipped (counted in attrs["n_skipped_split"]); a possession whose
    offense_team_id is null is skipped whole;
  - handler_id is the offensive player nearest the ball in the floor plane; null when the frame has no ball;
  - points_to_end is the possession's points less the offence's points logged (events with clock_s strictly
    greater than the state's game clock inside the possession's event span) before the state; without an events
    table it is the possession's points for every state (points come at the end of a possession except for the
    and-one, which the events refine);
  - possession_end_kind is the possession's end_kind. Both are TARGET columns and never features.
The returned frame carries the contract dtypes (analytics.schema.cast) and ``attrs`` with the counts logged.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema

STATE_COLUMNS = schema.columns("states_1hz")
FRAME_LEVEL = ["frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z"]


def frame_level(frames: pd.DataFrame) -> pd.DataFrame:
    """First row of each frame with the per-frame columns, sorted by frame, plus n_rows per frame."""
    cols = [c for c in FRAME_LEVEL if c in frames.columns]
    fl = frames.drop_duplicates("frame", keep="first")[cols].copy()
    for c in ("shot_clock_s", "ball_x", "ball_y", "ball_z"):
        if c not in fl.columns:
            fl[c] = np.nan
    fl["n_rows"] = frames.groupby("frame").size().reindex(fl["frame"]).to_numpy()
    return fl.sort_values("frame", kind="stable").reset_index(drop=True)


def select_state_frames(fl: pd.DataFrame, possessions: pd.DataFrame, tol_s: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """(possession, frame, elapsed_in_possession_s) for every whole second of every possession, with the log."""
    log = {"n_possessions": int(len(possessions)), "n_skipped_no_offense": 0, "n_skipped_empty_span": 0,
           "n_seconds_no_frame_within_tol": 0, "n_seconds_duplicate_frame": 0}
    fr = fl["frame"].to_numpy(dtype=np.int64)
    clk = fl["game_clock_s"].to_numpy(dtype=np.float64)
    rows = []
    for p in possessions.sort_values("possession").itertuples(index=False):
        if pd.isna(p.offense_team_id):
            log["n_skipped_no_offense"] += 1
            continue
        lo, hi = np.searchsorted(fr, int(p.start_frame), "left"), np.searchsorted(fr, int(p.end_frame), "right")
        if hi <= lo:
            log["n_skipped_empty_span"] += 1
            continue
        span_fr, span_clk = fr[lo:hi], clk[lo:hi]
        start_clock = float(p.start_clock_s)
        n_sec = int(np.floor(start_clock - np.nanmin(span_clk) + 1e-6)) + 1
        targets = start_clock - np.arange(n_sec, dtype=np.float64)
        diff = np.abs(span_clk[None, :] - targets[:, None])
        j = np.nanargmin(diff, axis=1)
        ok = diff[np.arange(n_sec), j] <= tol_s
        log["n_seconds_no_frame_within_tol"] += int((~ok).sum())
        seen = set()
        for k in np.flatnonzero(ok):
            f = int(span_fr[j[k]])
            if f in seen:
                log["n_seconds_duplicate_frame"] += 1
                continue
            seen.add(f)
            rows.append((int(p.possession), f, float(k)))
    sel = pd.DataFrame(rows, columns=["possession", "frame", "elapsed_in_possession_s"])
    return sel, log


def _points_before(possessions: pd.DataFrame, events: pd.DataFrame | None, sel: pd.DataFrame, game_clock: np.ndarray,
                   side_of_team: dict) -> np.ndarray:
    """Offence points logged before each state: events of the possession's event span whose clock_s is strictly
    greater than the state's game clock, credited to the offence's side (`team` is home or away in section 5;
    side_of_team maps a team id to that side). Zero everywhere without an events table."""
    out = np.zeros(len(sel), dtype=np.int64)
    if events is None or len(events) == 0 or "points" not in events.columns:
        return out
    ev = events.sort_values("event")
    pts = pd.to_numeric(ev["points"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    ev_idx, ev_clock = ev["event"].to_numpy(dtype=np.int64), ev["clock_s"].to_numpy(dtype=np.float64)
    ev_team = ev["team"].astype(object).to_numpy() if "team" in ev.columns else np.array([None] * len(ev), dtype=object)
    by_poss = {int(p.possession): p for p in possessions.itertuples(index=False)}
    poss_arr = sel["possession"].to_numpy()
    scoring = pts > 0
    if not scoring.any():
        return out
    for i in range(len(sel)):
        p = by_poss[int(poss_arr[i])]
        m = scoring & (ev_idx >= int(p.start_event)) & (ev_idx <= int(p.end_event)) & (ev_clock > game_clock[i] + 1e-6)
        if not m.any():
            continue
        side = side_of_team.get(int(p.offense_team_id))
        if side is not None:
            m &= ev_team == side
        out[i] = int(pts[m].sum())
    return out


def build_states(frames: pd.DataFrame, possessions: pd.DataFrame, sportvu_game: str,
                 events: pd.DataFrame | None = None, tol_s: float = 0.5) -> pd.DataFrame:
    """The section 20 table for one game. See the module docstring for the rules."""
    for c in ("frame", "period", "game_clock_s", "team_id", "player_id", "x", "y"):
        if c not in frames.columns:
            raise ValueError("states: frames lack column %r" % c)
    poss = possessions[possessions["sportvu_game"].astype(str) == str(sportvu_game)] if "sportvu_game" in possessions.columns else possessions
    fl = frame_level(frames)
    sel, log = select_state_frames(fl, poss, tol_s)
    log["n_states_selected"] = int(len(sel))
    log["n_skipped_split"] = 0
    if len(sel) == 0:
        out = schema.empty("states_1hz")
        out.attrs.update(log)
        return out
    pinfo = poss.set_index("possession")
    sel["offense_team_id"] = pinfo.loc[sel["possession"], "offense_team_id"].to_numpy().astype(np.int64)
    sub = frames.loc[frames["frame"].isin(sel["frame"].unique()), ["frame", "team_id", "player_id", "x", "y"]]
    m = sel.merge(sub, on="frame", how="inner")
    m["is_off"] = m["team_id"].to_numpy() == m["offense_team_id"].to_numpy()
    g = m.groupby(["possession", "frame"])
    cnt = g.size().rename("n")
    noff = g["is_off"].sum().rename("n_off")
    good = pd.concat([cnt, noff], axis=1)
    good = good[(good["n"] == 10) & (good["n_off"] == 5)]
    log["n_skipped_split"] = int(len(sel) - len(good))
    m = m.set_index(["possession", "frame"]).loc[good.index].reset_index()
    m = m.sort_values(["possession", "frame", "is_off", "player_id"], ascending=[True, True, False, True], kind="stable")
    n = len(m) // 10
    ids = m["player_id"].to_numpy(dtype=np.int64).reshape(n, 10)
    xs = m["x"].to_numpy(dtype=np.float64).reshape(n, 10)
    ys = m["y"].to_numpy(dtype=np.float64).reshape(n, 10)
    keys = m[["possession", "frame", "elapsed_in_possession_s", "offense_team_id"]].iloc[::10].reset_index(drop=True)
    fli = fl.set_index("frame").loc[keys["frame"]]
    bx, by, bz = (fli[c].to_numpy(dtype=np.float64) for c in ("ball_x", "ball_y", "ball_z"))
    d_ball = np.hypot(xs[:, :5] - bx[:, None], ys[:, :5] - by[:, None])
    has_ball = np.isfinite(bx) & np.isfinite(by)
    h_idx = np.where(has_ball, np.nanargmin(np.where(np.isfinite(d_ball), d_ball, np.inf), axis=1), 0)
    hx, hy = xs[np.arange(n), h_idx], ys[np.arange(n), h_idx]
    d_def = np.hypot(xs[:, 5:] - hx[:, None], ys[:, 5:] - hy[:, None]).min(axis=1)
    handler = pd.array(np.where(has_ball, ids[np.arange(n), h_idx], 0), dtype="Int32")
    handler[~has_ball] = pd.NA
    out = pd.DataFrame({"sportvu_game": str(sportvu_game), "possession": keys["possession"].to_numpy(),
                        "frame": keys["frame"].to_numpy(), "period": fli["period"].to_numpy(),
                        "game_clock_s": fli["game_clock_s"].to_numpy(), "shot_clock_s": fli["shot_clock_s"].to_numpy(),
                        "elapsed_in_possession_s": keys["elapsed_in_possession_s"].to_numpy(),
                        "offense_team_id": keys["offense_team_id"].to_numpy(), "handler_id": handler,
                        "ball_x": bx, "ball_y": by, "ball_z": bz})
    for i in range(5):
        out["off%d_id" % (i + 1)] = ids[:, i]
    for i in range(5):
        out["off%d_x" % (i + 1)], out["off%d_y" % (i + 1)] = xs[:, i], ys[:, i]
    for i in range(5):
        out["def%d_id" % (i + 1)] = ids[:, 5 + i]
    for i in range(5):
        out["def%d_x" % (i + 1)], out["def%d_y" % (i + 1)] = xs[:, 5 + i], ys[:, 5 + i]
    out["handler_nearest_def_dist_ft"] = np.where(has_ball, d_def, np.nan)
    pts = pinfo.loc[out["possession"], "points"].to_numpy(dtype=np.int64)
    side_of_team = {}
    if "home" in frames.columns:
        for tid, is_home in frames.drop_duplicates("team_id")[["team_id", "home"]].itertuples(index=False):
            side_of_team[int(tid)] = "home" if bool(is_home) else "away"
    before = _points_before(poss, events, out[["possession"]], out["game_clock_s"].to_numpy(dtype=np.float64), side_of_team)
    out["points_to_end"] = np.clip(pts - before, 0, None)
    out["possession_end_kind"] = pinfo.loc[out["possession"], "end_kind"].astype(str).to_numpy()
    # a frame with no ball still has positions: ball columns are non-nullable in the contract, so drop those states
    no_ball = ~has_ball | ~np.isfinite(bz)
    log["n_skipped_no_ball"] = int(no_ball.sum())
    out = out.loc[~no_ball].reset_index(drop=True)
    out = schema.cast(out, "states_1hz")
    out = out.sort_values(["possession", "elapsed_in_possession_s"], kind="stable").reset_index(drop=True)
    log["n_states"] = int(len(out))
    out.attrs.update(log)
    return out


# ── IO ─────────────────────────────────────────────────────────────────────────────────────────────
def game_files(data_root) -> list:
    """[(sportvu_game, season, heldout, frames_path, pbp_path_or_None, possessions_path)] under DATA_ROOT."""
    root = Path(data_root)
    out = []
    for held, sub in ((False, ""), (True, "heldout")):
        base = root / "analytics" / "possessions" / sub if sub else root / "analytics" / "possessions"
        if not base.exists():
            continue
        for season_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "heldout"):
            for pp in sorted(season_dir.glob("*.parquet")):
                game = pp.stem
                fdir = root / "sportvu_contract" / ("heldout" if held else "")
                fp = fdir / ("%s.parquet" % game)
                if not fp.exists():
                    fp = root / "sportvu_contract" / ("%s.parquet" % game)
                pbp = fp.with_name("%s_pbp.parquet" % game)
                if fp.exists():
                    out.append((game, season_dir.name, held, fp, pbp if pbp.exists() else None, pp))
    return out


def write_states(states: pd.DataFrame, out_root, season: str, game: str, heldout: bool) -> Path:
    """Write one game's states table at its contract path under OUT_ROOT/analytics and return the path."""
    path = schema.table_path(Path(out_root) / "analytics", "states_1hz", season, game, heldout)
    path.parent.mkdir(parents=True, exist_ok=True)
    states.to_parquet(path, index=False)
    return path


def _log_line(game: str, states: pd.DataFrame) -> str:
    a = states.attrs
    return ("%s: %d states from %d possessions (skipped: %d no offence, %d empty span, %d split, %d no ball; "
            "%d seconds without a frame in tolerance, %d duplicate-frame seconds)"
            % (game, len(states), a.get("n_possessions", 0), a.get("n_skipped_no_offense", 0), a.get("n_skipped_empty_span", 0),
               a.get("n_skipped_split", 0), a.get("n_skipped_no_ball", 0), a.get("n_seconds_no_frame_within_tol", 0),
               a.get("n_seconds_duplicate_frame", 0)))


def parse_args(argv=None):
    """The states CLI flags."""
    from analytics.epv.validate import add_fixture_args
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="build the synthetic season of analytics.fixtures (fixture output)")
    g.add_argument("--data", help="DATA_ROOT with sportvu_contract/ and analytics/possessions/")
    ap.add_argument("--out", help="OUT_ROOT (default: DATA_ROOT, or a scratch directory with --fixtures)")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="seconds between a whole second of the clock and its nearest frame for a state to be kept (default 0.5)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> list:
    """Build and write states_1hz for every game (the fixture season or DATA_ROOT); returns the written paths."""
    a = parse_args(argv)
    written = []
    if a.fixtures:
        from analytics.epv.validate import fixture_season
        out_root = Path(a.out) if a.out else Path(tempfile.gettempdir()) / "epv_fixture_root"
        season = fixture_season(a.seed, n_games=a.fixture_games, heldout_every=a.fixture_heldout_every,
                                n_possessions=a.fixture_possessions)
        for game, held in season["games"]:
            frames = season["frames"][season["frames"]["game"] == game]
            poss = season["possessions"][season["possessions"]["sportvu_game"] == game]
            ev = season["events"][season["events"]["game"] == game]
            st = build_states(frames, poss, game, events=ev, tol_s=a.tol)
            written.append(write_states(st, out_root, schema.season_of_game(game), game, held))
            print(_log_line(game, st))
        print("fixture output, not a result: %d game(s) written under %s" % (len(written), out_root))
        return written
    out_root = Path(a.out) if a.out else Path(a.data)
    files = game_files(a.data)
    if not files:
        raise SystemExit("states: no game with both a possessions file and a frames file under %s" % a.data)
    for game, season, held, fp, pbp, pp in files:
        frames = pd.read_parquet(fp)
        poss = pd.read_parquet(pp)
        ev = pd.read_parquet(pbp) if pbp else None
        st = build_states(frames, poss, game, events=ev, tol_s=a.tol)
        written.append(write_states(st, out_root, season, game, held))
        print(_log_line(game, st))
    print("%d game(s) written under %s" % (len(written), out_root))
    return written


if __name__ == "__main__":
    main()

"""analytics/load/features.py: player-stint and player-game load tables, the schedule features and the guards.

    python -m analytics.load.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16] [--limit N]

Units and definitions (docs/contract/load.md):

player_stints  one row per continuous on-court span of a player. Frames of a player are sorted by period and
    frame; a new stint starts when the period changes or the game clock jumps by more than STINT_BREAK_S (5 s)
    between consecutive rows of that player (a substitution leaves a hole in the clock; a stopped clock does
    not). Positions are smoothed by a centred rolling mean over SMOOTH_FRAMES (5) rows before differencing.
    A step is the move between consecutive rows of a stint; it counts when its wall-clock dt (ts_ms, else the
    frame index at 25 Hz) is in (0, 1] s. minutes = game clock elapsed over the stint; distance_ft = sum of
    counted step lengths; avg_speed_ftps = distance / counted seconds; high_intensity_share = counted seconds
    above HIGH_INTENSITY_FTPS (12) over counted seconds; accelerations = episodes where the smoothed speed
    rises faster than ACCEL_FTPS2 (6 ft per s^2). Dead-ball movement counts (the wall clock runs), a limit.

player_games  one row per player with frames in a game. Totals over the stints; the within-game trend from
    the stints of at least MIN_TREND_STINT_MIN (1.0) minutes in stint order: decline_ftps = avg speed of the
    last such stint minus the first (NaN with fewer than two), slope_ftps_per_stint = least-squares slope of
    avg speed on the stint index. Schedule features from the manifest dates of the player's team: days_rest
    (capped at REST_CAP = 7, the team's first listed game takes the cap), back_to_back (days_rest == 1),
    games_last_7 (team games in the previous 7 days), and the player's load over the team's previous games:
    minutes_last_3 and distance_last_3_ft (sum over the team's last three games, 0 when the player had no
    frames in one, NaN when one of them is held out or not loaded), prev_decline_ftps and prev_avg_speed_ftps
    (the previous team game, NaN when absent, held out or not loaded). absent_next = the team's next manifest
    game has a loaded, non-held-out roster and the player is not on it; NaN otherwise. Absence is not injury:
    rest, a coach's decision, a suspension, a trade or a tracking miss all read as absent.

FEATURES are the model inputs; they come from the schedule and from previous games only. The same-game
totals and trend columns (SAME_GAME) and the targets (TARGETS) are FORBIDDEN in a design matrix:
assert_no_leak refuses them and schema.assert_time_forward runs on every feature frame. The stints table
holds end_frame (a span bound of a descriptive unit) and is never fed to a model.

load_summary  per player over non-held-out games: n_games, n_stints, mean_decline with a bootstrap
    interval over the player's games, insufficient = fewer than MIN_STINTS_RANK (10) stints (gated: never
    ranked).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.load import validate as V

STINT_BREAK_S = 5.0
SMOOTH_FRAMES = 5
MAX_STEP_DT_S = V.MAX_STEP_DT_S
HIGH_INTENSITY_FTPS = 12.0
ACCEL_FTPS2 = 6.0
MIN_TREND_STINT_MIN = 1.0
REST_CAP = 7
MIN_STINTS_RANK = 10
FPS = V.FPS

FEATURES = ["days_rest", "back_to_back", "games_last_7", "minutes_last_3", "distance_last_3_ft",
            "prev_decline_ftps", "prev_avg_speed_ftps"]
TARGETS = ["decline_ftps", "slope_ftps_per_stint", "absent_next"]
SAME_GAME = ["minutes", "distance_ft", "avg_speed_ftps", "high_intensity_share", "accelerations"]
FORBIDDEN = frozenset(TARGETS) | frozenset(SAME_GAME) | frozenset({"absent", "next_game", "end_frame", "start_frame"})

STINT_COLUMNS = ["sportvu_game", "player_id", "stint", "start_frame", "end_frame", "minutes", "distance_ft",
                 "avg_speed_ftps", "high_intensity_share", "accelerations", "heldout"]
GAME_COLUMNS = ["sportvu_game", "game_date", "player_id", "team", "minutes", "distance_ft", "avg_speed_ftps",
                "decline_ftps", "slope_ftps_per_stint", "days_rest", "back_to_back", "games_last_7", "minutes_last_3",
                "distance_last_3_ft", "prev_decline_ftps", "prev_avg_speed_ftps", "absent_next", "heldout"]
SUMMARY_COLUMNS = ["player_id", "n_games", "n_stints", "mean_decline", "lo", "hi", "insufficient"]
GUARD_TABLE = "rosters"      # the schema table name assert_time_forward is called with (the check is the union of all tables)


def feature_columns() -> list:
    return list(FEATURES)


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a target, a same-game total or any other forbidden column."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("analytics.load: forbidden columns in the design matrix: %s" % bad)


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """The design matrix (FEATURES as floats) with both guards run on it."""
    X = frame[FEATURES].astype(float)
    assert_no_leak(X)
    schema.assert_time_forward(X, GUARD_TABLE)
    return X


# ── stints ─────────────────────────────────────────────────────────────────────────────────────────
def _roll_prev(a: np.ndarray):
    return np.roll(a, 1)


def player_stints(frames: pd.DataFrame, sportvu_game: str | None = None, heldout: bool = False,
                  validated: bool = False) -> pd.DataFrame:
    """Section 4 frames of one game -> the player_stints table (STINT_COLUMNS). validate_frames runs first
    unless validated is true (then frames must be its output: sorted by player_id, period, frame)."""
    df = frames if validated else V.validate_frames(frames)[0]
    if sportvu_game is None:
        sportvu_game = str(df["game"].iloc[0]) if len(df) else ""
    if not len(df):
        return pd.DataFrame(columns=STINT_COLUMNS)
    pid = df["player_id"].to_numpy()
    per = df["period"].to_numpy()
    clock = df["game_clock_s"].to_numpy(dtype="float64")
    same_player = pid == _roll_prev(pid)
    same_player[0] = False
    jump = np.abs(_roll_prev(clock) - clock) > STINT_BREAK_S
    new = ~same_player | (per != _roll_prev(per)) | jump
    key = np.cumsum(new) - 1                                            # stint key over the whole game
    kser = pd.Series(key, index=df.index)
    sm = df[["x", "y"]].astype("float64").groupby(kser).rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean()
    sm = sm.reset_index(level=0, drop=True).sort_index()
    xs, ys = sm["x"].to_numpy(), sm["y"].to_numpy()
    dt = V.step_dt_s(df)
    same_stint = key == _roll_prev(key)
    same_stint[0] = False
    dt = np.where(same_stint, dt, np.nan)
    dist = np.hypot(xs - _roll_prev(xs), ys - _roll_prev(ys))
    counted = np.isfinite(dt) & (dt > 0) & (dt <= MAX_STEP_DT_S)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(counted, dist / dt, np.nan)
    dt_c = np.where(counted, dt, 0.0)
    dist_c = np.where(counted, dist, 0.0)
    hi_s = np.where(counted & (speed > HIGH_INTENSITY_FTPS), dt, 0.0)
    sp = pd.Series(speed, index=df.index).groupby(kser).rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean()
    sp = sp.reset_index(level=0, drop=True).sort_index().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.where(counted, (sp - _roll_prev(sp)) / dt, np.nan)
    above = np.isfinite(accel) & (accel > ACCEL_FTPS2)
    rising = above & ~(_roll_prev(above) & same_stint)
    g = pd.DataFrame({"key": key, "player_id": pid, "frame": df["frame"].to_numpy(), "clock": clock,
                      "dt": dt_c, "dist": dist_c, "hi": hi_s, "rise": rising.astype(int)}).groupby("key", sort=True)
    out = g.agg(player_id=("player_id", "first"), start_frame=("frame", "min"), end_frame=("frame", "max"),
                start_clock=("clock", "first"), end_clock=("clock", "last"), run_s=("dt", "sum"),
                distance_ft=("dist", "sum"), hi_s=("hi", "sum"), accelerations=("rise", "sum")).reset_index(drop=True)
    out["minutes"] = np.clip(out["start_clock"] - out["end_clock"], 0, None) / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out["avg_speed_ftps"] = np.where(out["run_s"] > 0, out["distance_ft"] / out["run_s"], np.nan)
        out["high_intensity_share"] = np.where(out["run_s"] > 0, out["hi_s"] / out["run_s"], np.nan)
    out = out.sort_values(["player_id", "start_frame"], kind="mergesort").reset_index(drop=True)
    out["stint"] = out.groupby("player_id").cumcount()
    out["sportvu_game"] = sportvu_game
    out["heldout"] = bool(heldout)
    out["run_s"] = out["run_s"].astype(float)                           # kept for the game totals, dropped on write
    return out[STINT_COLUMNS + ["run_s"]]


# ── schedule ───────────────────────────────────────────────────────────────────────────────────────
def schedule_table(manifest: dict) -> pd.DataFrame:
    """(team, sportvu_game) rows from the manifest dates: game_date, days_rest, back_to_back, games_last_7,
    prev_game, next_game. Every listed game counts as played whatever its status (the schedule is public)."""
    rows = []
    for sid, g in manifest.get("games", {}).items():
        meta = V.game_meta(sid)
        date = g.get("date") or meta["date"]
        for side in ("away", "home"):
            rows.append({"team": g.get(side) or meta[side], "sportvu_game": sid, "game_date": date})
    if not rows:
        return pd.DataFrame(columns=["team", "sportvu_game", "game_date", "days_rest", "back_to_back", "games_last_7",
                                     "prev_game", "next_game"])
    sch = pd.DataFrame(rows).drop_duplicates(["team", "sportvu_game"])
    sch["_d"] = pd.to_datetime(sch["game_date"])
    sch = sch.sort_values(["team", "_d", "sportvu_game"], kind="mergesort").reset_index(drop=True)
    out = []
    for team, part in sch.groupby("team", sort=False):
        d = part["_d"].to_numpy()
        ids = part["sportvu_game"].tolist()
        for i in range(len(part)):
            rest = (d[i] - d[i - 1]) / np.timedelta64(1, "D") if i else float(REST_CAP)
            rest = float(min(rest, REST_CAP))
            last7 = int(((d >= d[i] - np.timedelta64(7, "D")) & (d < d[i])).sum())
            out.append({"team": team, "sportvu_game": ids[i], "game_date": part["game_date"].iloc[i], "days_rest": rest,
                        "back_to_back": bool(rest <= 1), "games_last_7": last7,
                        "prev_game": ids[i - 1] if i else None, "next_game": ids[i + 1] if i + 1 < len(ids) else None})
    return pd.DataFrame(out)


# ── player-games ───────────────────────────────────────────────────────────────────────────────────
def _trend(part: pd.DataFrame) -> tuple[float, float]:
    q = part[part["minutes"] >= MIN_TREND_STINT_MIN].sort_values("stint")
    q = q[np.isfinite(q["avg_speed_ftps"])]
    if len(q) < 2:
        return np.nan, np.nan
    v = q["avg_speed_ftps"].to_numpy(dtype=float)
    k = np.arange(len(v), dtype=float)
    slope = float(np.polyfit(k, v, 1)[0])
    return float(v[-1] - v[0]), slope


def player_games(stints: pd.DataFrame, games: list, schedule: pd.DataFrame, rosters: dict) -> pd.DataFrame:
    """player_stints of every loaded game (with run_s) + game dicts + schedule + {sportvu_game: roster} ->
    the player_games table (GAME_COLUMNS). History features read previous games only."""
    meta = {g["sportvu_game"]: g for g in games}
    if not len(stints):
        return pd.DataFrame(columns=GAME_COLUMNS)
    grp = stints.groupby(["sportvu_game", "player_id"], sort=True)
    tot = grp.agg(minutes=("minutes", "sum"), distance_ft=("distance_ft", "sum"), run_s=("run_s", "sum")).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        tot["avg_speed_ftps"] = np.where(tot["run_s"] > 0, tot["distance_ft"] / tot["run_s"], np.nan)
    trend = grp.apply(lambda p: pd.Series(_trend(p), index=["decline_ftps", "slope_ftps_per_stint"]), include_groups=False)
    tot = tot.merge(trend.reset_index(), on=["sportvu_game", "player_id"], how="left")
    team_of = {}
    for sid, ros in rosters.items():
        for p, t in zip(ros["player_id"].to_numpy(), ros["team"].to_numpy()):
            team_of[(sid, int(p))] = str(t)
    tot["team"] = [team_of.get((s, int(p))) for s, p in zip(tot["sportvu_game"], tot["player_id"])]
    tot["heldout"] = tot["sportvu_game"].map(lambda s: bool(meta[s]["heldout"]))
    tot["game_date"] = tot["sportvu_game"].map(lambda s: meta[s]["date"])
    tot = tot.merge(schedule.drop(columns=["game_date"]), on=["team", "sportvu_game"], how="left")
    # previous games only: the chain of the team's earlier games from the schedule
    _id = lambda v: v if isinstance(v, str) else None                 # None becomes NaN inside a frame
    prev_of = {(r.team, r.sportvu_game): _id(r.prev_game) for r in schedule.itertuples()}
    next_of = {(r.team, r.sportvu_game): _id(r.next_game) for r in schedule.itertuples()}
    known = {(s, int(p)): (m, d, dec, sp) for s, p, m, d, dec, sp in zip(
        tot["sportvu_game"], tot["player_id"], tot["minutes"], tot["distance_ft"], tot["decline_ftps"], tot["avg_speed_ftps"])}
    loaded_ok = {s for s, g in meta.items() if not g["heldout"]}      # a held-out game is never read for features
    m3, d3, pdec, pspd, absent = [], [], [], [], []
    for s, p, t in zip(tot["sportvu_game"], tot["player_id"], tot["team"]):
        p = int(p)
        chain, g = [], s
        for _ in range(3):
            g = prev_of.get((t, g))
            if g is None:
                break
            chain.append(g)
        if any(c not in loaded_ok for c in chain):
            m3.append(np.nan)
            d3.append(np.nan)
        else:
            m3.append(float(sum(known.get((c, p), (0.0, 0.0, np.nan, np.nan))[0] for c in chain)))
            d3.append(float(sum(known.get((c, p), (0.0, 0.0, np.nan, np.nan))[1] for c in chain)))
        pg = chain[0] if chain else None
        if pg is not None and pg in loaded_ok and (pg, p) in known:
            pdec.append(known[(pg, p)][2])
            pspd.append(known[(pg, p)][3])
        else:
            pdec.append(np.nan)
            pspd.append(np.nan)
        ng = next_of.get((t, s))
        if ng is not None and ng in loaded_ok and ng in rosters:
            absent.append(p not in set(int(q) for q in rosters[ng]["player_id"]))
        else:
            absent.append(pd.NA)
    tot["minutes_last_3"], tot["distance_last_3_ft"] = m3, d3
    tot["prev_decline_ftps"], tot["prev_avg_speed_ftps"] = pdec, pspd
    tot["absent_next"] = pd.array(absent, dtype="boolean")
    tot["player_id"] = tot["player_id"].astype("int32")
    tot["games_last_7"] = tot["games_last_7"].astype(float)
    tot = tot.sort_values(["game_date", "sportvu_game", "player_id"], kind="mergesort").reset_index(drop=True)
    return tot[GAME_COLUMNS]


# ── summary ────────────────────────────────────────────────────────────────────────────────────────
def load_summary(games_df: pd.DataFrame, stints: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Per player over non-held-out games: n_games (with a decline), n_stints, mean_decline, lo, hi (2.5 and 97.5
    percentiles of n_boot resamples of the player's games), insufficient = n_stints < MIN_STINTS_RANK."""
    rng = np.random.default_rng(seed)
    tr = games_df[~games_df["heldout"].astype(bool)]
    st = stints[~stints["heldout"].astype(bool)]
    n_st = st.groupby("player_id").size()
    rows = []
    for p, part in tr.groupby("player_id"):
        y = part["decline_ftps"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        n = len(y)
        if n:
            draws = y[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if n > 1 else (np.nan, np.nan)
            mean = float(y.mean())
        else:
            mean, lo, hi = np.nan, np.nan, np.nan
        k = int(n_st.get(p, 0))
        rows.append({"player_id": int(p), "n_games": int(n), "n_stints": k, "mean_decline": mean, "lo": lo, "hi": hi,
                     "insufficient": bool(k < MIN_STINTS_RANK)})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


# ── build everything ───────────────────────────────────────────────────────────────────────────────
def build_tables(inputs: dict, n_boot: int = 200, seed: int = 0) -> dict:
    """inputs (validate.load_fixtures or load_data_root) -> dict(player_stints, player_games, load_summary,
    schedule, logs). Frames are read one game at a time and dropped after the stints are extracted."""
    stints, rosters, logs = [], {}, {}
    for g in inputs["games"]:
        frames = V.game_frames(g)
        clean, log = V.validate_frames(frames)
        ros, rlog = V.validate_rosters(V.game_rosters(g, frames))
        log["rosters"] = rlog
        logs[g["sportvu_game"]] = log
        rosters[g["sportvu_game"]] = ros
        stints.append(player_stints(clean, g["sportvu_game"], g["heldout"], validated=True))
        del frames, clean
    st = pd.concat(stints, ignore_index=True) if stints else pd.DataFrame(columns=STINT_COLUMNS + ["run_s"])
    schedule = schedule_table(inputs["manifest"])
    pg = player_games(st, inputs["games"], schedule, rosters)
    schema.assert_time_forward(pg, GUARD_TABLE)
    summary = load_summary(pg, st, n_boot=n_boot, seed=seed)
    return {"player_stints": st[STINT_COLUMNS], "player_games": pg, "load_summary": summary, "schedule": schedule,
            "logs": logs}


def write_tables(tables: dict, out_root, season: str) -> list:
    """OUT_ROOT/analytics/load/<season>/{player_stints,player_games,load_summary}.parquet; held-out rows of the
    first two go under OUT_ROOT/analytics/load/heldout/<season>/ instead."""
    base = Path(out_root) / "analytics" / "load"
    written = []
    for name in ("player_stints", "player_games"):
        df = tables[name]
        for held, part in ((False, df[~df["heldout"].astype(bool)]), (True, df[df["heldout"].astype(bool)])):
            if held and not len(part):
                continue
            p = base / ("heldout" if held else "") / season / ("%s.parquet" % name)
            p.parent.mkdir(parents=True, exist_ok=True)
            part.reset_index(drop=True).to_parquet(p, index=False)
            written.append(str(p))
    p = base / season / "load_summary.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    tables["load_summary"].to_parquet(p, index=False)
    written.append(str(p))
    return written


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--local-fixtures", action="store_true",
                    help="with --fixtures: the local generator of tests/_fx_load.py even when analytics.fixtures imports")
    ap.add_argument("--out", help="OUT_ROOT for the three tables")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args(argv)
    inputs = V.load_fixtures(seed=a.seed, prefer_shared=not a.local_fixtures) if a.fixtures else \
        V.load_data_root(a.data, a.season, a.limit)
    tables = build_tables(inputs, n_boot=a.bootstrap, seed=a.seed)
    if a.out:
        for p in write_tables(tables, a.out, inputs["season"]):
            print("wrote %s" % p, file=sys.stderr)
    print("%s: %d stints, %d player-games, %d players in the summary (%d insufficient)%s" % (
        inputs["source"], len(tables["player_stints"]), len(tables["player_games"]), len(tables["load_summary"]),
        int(tables["load_summary"]["insufficient"].sum()), "; fixture output, not a result" if a.fixtures else ""))
    return tables


if __name__ == "__main__":
    main()

"""analytics/actions/features.py: candidate windows and per-candidate features for action recognition.

    python -m analytics.actions.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]

Step 1, rule-based candidate generation from the frames at 5 Hz (every round(hz / 5)-th frame of the
possession; frame ids stay the contract's 25 Hz ids). The ball handler at a frame is the offensive player
nearest the ball, none when the ball is over 6 ft from every offensive player (in flight); the handler's
defender is the defender nearest the handler at the window start.
    ball_screen  any window of at least 0.8 s (a 1 s set lands on 4 or 5 intervals of the 5 Hz grid) where an
                 offensive player other than the handler is within 4 ft of the handler's defender on every
                 frame, moving under 3 ft per s, with the handler within 12 ft of him; the handler's defender
                 is either of the two defenders nearest the handler at the window start (a switch, ice or
                 under puts the screened man second), within 9 ft of the handler (guarding him) and not the
                 screener's own matchup (the defender nearest him on most frames before the window on which
                 he is not already in contact with that defender: a player standing beside his own man is not
                 screening, and a player who never approached is not either); overlapping windows of one (handler, screener)
                 merge into the set, and the
                 candidate's extent is the set padded by 1 s before (the approach) and 1 s after (the use),
                 clipped to the possession, because a labelled screen spans approach, set and use (set
                 SCREEN_PAD_PRE_S and SCREEN_PAD_POST_S to 0 if the hand labels mark the set alone)
    drive        any 0.6 s window where the handler, starting within 30 ft of the basket (a half-court
                 action, not the transition sprint), closes on the basket faster than 8 ft per s
    isolation    any 2 s window where the same handler, starting within 30 ft of the basket, keeps the ball
                 (ball within 4 ft) under 8 ft per s and no teammate comes within 10 ft. The plan said 3 s;
                 2 s because a 3 s minimum missed the fixture's 2.4 s holds and the trainer, not the window
                 length, decides the class
Step 2, features per candidate from the positions and velocities inside the window (FEATURES), plus one
declared post-window feature, def_recovery_ft: the handler's defender's distance to the handler averaged
over the 1 s after the window minus its average inside the window. A detector may look 1 s past the window
it is classifying; nothing else looks past the window, and nothing looks at the possession outcome.

Target: a candidate overlaps a labelled action at IoU over frames >= 0.5 (IOU_MIN); the class is that
action's type when it is ball_screen, drive or isolation, else "none". Label columns never enter the
feature frame: build() returns the candidate table (ids, windows, rule class, players) and a separate
feature frame (ids plus FEATURES) on which schema.assert_time_forward and assert_no_leak are run.

Assumptions: a possession whose offense_team_id is null is skipped and counted; the first five players of
each side by row count stand in for a lineup that changed mid-possession (the rest are ignored); a window
that would run past the possession end is not generated; the screener proxy of a drive or isolation
candidate (for the screener features) is the teammate nearest the handler's defender at the window start.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.actions import CLASSES, IOU_MIN

log = logging.getLogger("analytics.actions.features")

TARGET_HZ = 5.0
SCREEN_WINDOW_S, SCREEN_DIST_FT, SCREEN_SPEED_FTPS, SCREEN_HANDLER_DIST_FT = 0.8, 4.0, 3.0, 12.0
SCREEN_N_DEFENDERS = 2                               # the screened man is one of the two defenders nearest the handler
SCREEN_DEF_HANDLER_FT = 9.0                          # and is guarding him: within this of the handler at the window start
SCREEN_PAD_PRE_S, SCREEN_PAD_POST_S = 1.0, 1.0      # the approach before the set and the use after it
DRIVE_WINDOW_S, DRIVE_SPEED_FTPS, DRIVE_START_FT = 0.6, 8.0, 30.0
ISO_WINDOW_S, ISO_SPREAD_FT, ISO_BALL_FT, ISO_HANDLER_SPEED_FTPS, ISO_START_FT = 2.0, 10.0, 4.0, 8.0, 30.0
HANDLER_BALL_FT = 6.0                                # farther than this from every offensive player, the ball has no handler
RECOVERY_S = 1.0
NEAR_FT = 8.0
BASKETS = (np.array([5.25, 25.0]), np.array([88.75, 25.0]))
DETECT_CLASSES = CLASSES[:3]

FEATURES = [
    "scr_def_dist_min", "scr_def_dist_mean", "scr_speed_mean", "scr_speed_max",
    "handler_speed_mean", "handler_speed_max", "handler_dir_change_rad", "handler_radial_speed_mean",
    "handler_basket_dist_start", "handler_basket_dist_end",
    "def_handler_dist_mean", "def_recovery_ft",
    "ball_z_mean", "ball_z_max", "shot_clock_start", "n_within_8ft_mean", "teammate_min_dist_mean",
    "window_s", "rule_is_screen", "rule_is_drive", "rule_is_iso",
]
POST_WINDOW_FEATURES = ("def_recovery_ft",)
ID_COLUMNS = ["sportvu_game", "possession", "candidate"]
CANDIDATE_COLUMNS = ID_COLUMNS + ["start_frame", "end_frame", "rule_class", "ball_handler_id", "screener_id"]
LABEL_COLUMNS = frozenset({"label_class", "label_iou", "label_action_type", "action_type", "coverage", "label_source",
                           "labeller_pass", "confidence", "pred_class", "p_ball_screen", "p_selected_class",
                           "start_frame", "end_frame", "rule_class"})


def feature_columns() -> list:
    return list(FEATURES)


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a label, prediction, window or contract-excluded column."""
    cols = list(getattr(X, "columns", X))
    bad = sorted(set(cols) & (LABEL_COLUMNS | set(schema.ALL_EXCLUDED)))
    if bad:
        raise ValueError("actions.features: forbidden column(s) in the design matrix: %s" % bad)


# ── per-possession arrays ──────────────────────────────────────────────────────────────────────────
class PossessionArrays:
    """Frames of one possession at 5 Hz as arrays: t (n,), frames (n,), ball (n, 3), shot_clock (n,),
    off (n, k, 2), de (n, m, 2), off_ids (k,), def_ids (m,), basket (2,), plus velocities and speeds."""

    def __init__(self, frames, t, ball, shot_clock, off_ids, off, def_ids, de, basket):
        self.frames, self.t, self.ball, self.shot_clock = frames, t, ball, shot_clock
        self.off_ids, self.off, self.def_ids, self.de, self.basket = off_ids, off, def_ids, de, basket
        self.n = len(t)
        self.dt = np.diff(t) if self.n > 1 else np.zeros(0)
        self.v_off = velocities(off, t)
        self.v_def = velocities(de, t)
        self.speed_off = np.linalg.norm(self.v_off, axis=2)
        self.handler = handler_index(ball, off)
        self.near_def = nearest_defender_per_frame(off, de)
        self.r_handler = np.full(self.n, np.nan)
        for i in range(self.n):
            if self.handler[i] >= 0:
                self.r_handler[i] = float(np.linalg.norm(off[i, self.handler[i]] - basket))


def velocities(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Forward-difference velocity (n, k, 2) in ft per s; the last frame repeats the previous one."""
    n = len(t)
    v = np.zeros_like(xy, dtype=float)
    if n < 2:
        return v
    dt = np.diff(t)
    dt = np.where(dt <= 0, np.nan, dt)
    v[:-1] = (xy[1:] - xy[:-1]) / dt[:, None, None]
    v[-1] = v[-2]
    return v


def handler_index(ball: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Index of the offensive player nearest the ball per frame; -1 when nothing is finite or the ball is more
    than HANDLER_BALL_FT from every offensive player (a pass or shot in flight, a loose ball)."""
    d = np.linalg.norm(off - ball[:, None, :2], axis=2)
    out = np.full(len(ball), -1, dtype=int)
    ok = np.isfinite(d).any(axis=1)
    if ok.any():
        dd = np.where(np.isfinite(d[ok]), d[ok], np.inf)
        best = np.argmin(dd, axis=1)
        best[dd[np.arange(len(best)), best] > HANDLER_BALL_FT] = -1
        out[ok] = best
    return out


def nearest_defender_per_frame(off: np.ndarray, de: np.ndarray) -> np.ndarray:
    """(n, k) index of the defender nearest each offensive player on each frame, -1 without data."""
    d = np.linalg.norm(de[:, None, :, :] - off[:, :, None, :], axis=3)
    out = np.where(np.isfinite(d).any(axis=2), np.argmin(np.where(np.isfinite(d), d, np.inf), axis=2), -1)
    return out.astype(int)


def matchup_before(a: "PossessionArrays", s: int, d: int, i: int) -> int:
    """The defender nearest offensive player s on most frames before frame i on which s is not in contact with
    defender d (within SCREEN_DIST_FT): his matchup before the screen. Without such a frame (s has stood beside d
    since the possession began, no approach seen) d counts as his own man."""
    k = max(i, 1)
    near = a.near_def[:k, s]
    free = ~(np.linalg.norm(a.off[:k, s] - a.de[:k, d], axis=1) < SCREEN_DIST_FT)
    sel = near[free & (near >= 0)]
    return int(np.bincount(sel).argmax()) if len(sel) else d


def nearest_defenders(arrs: PossessionArrays, i: int, h: int, n: int = 1) -> list:
    """Indices of the n defenders nearest offensive player h at frame i, nearest first (finite positions only)."""
    d = np.linalg.norm(arrs.de[i] - arrs.off[i, h], axis=1)
    order = [int(k) for k in np.argsort(np.where(np.isfinite(d), d, np.inf)) if np.isfinite(d[k])]
    return order[:n]


def nearest_defender(arrs: PossessionArrays, i: int, h: int) -> int:
    near = nearest_defenders(arrs, i, h, 1)
    return near[0] if near else -1


def _lineup(pid: np.ndarray, mask: np.ndarray) -> list:
    """Up to five most frequent player ids among the rows of one side (ties by id)."""
    ids, counts = np.unique(pid[mask], return_counts=True)
    order = np.lexsort((ids, -counts))
    return [int(i) for i in ids[order][:5]]


def frame_columns(frames: pd.DataFrame) -> dict:
    """The section 4 columns this module reads, as numpy arrays sorted by frame (one conversion per game)."""
    fr = frames.sort_values("frame", kind="stable")
    cols = {"frame": fr["frame"].to_numpy(dtype=np.int64), "player_id": fr["player_id"].to_numpy(dtype=np.int64),
            "team_id": fr["team_id"].to_numpy(dtype=np.int64), "x": fr["x"].to_numpy(dtype=float), "y": fr["y"].to_numpy(dtype=float),
            "game_clock_s": fr["game_clock_s"].to_numpy(dtype=float)}
    for c in ("ball_x", "ball_y", "ball_z", "shot_clock_s"):
        cols[c] = fr[c].to_numpy(dtype=float) if c in fr.columns else np.full(len(fr), np.nan)
    return cols


def possession_arrays(sub, offense_team_id: int, target_hz: float = TARGET_HZ):
    """Rows of one possession (a frames DataFrame or a dict of numpy columns from frame_columns, sliced to the
    span) -> PossessionArrays at target_hz, or None when a side has no player."""
    if isinstance(sub, pd.DataFrame):
        sub = frame_columns(sub)
    fr = sub["frame"]
    if len(fr) == 0:
        return None
    uf, first_idx = np.unique(fr, return_index=True)
    clock = sub["game_clock_s"][first_idx]
    stride = 1
    if len(uf) > 1:
        step = float(np.nanmedian(np.abs(np.diff(clock))))
        stride = max(1, int(round(1.0 / (step * target_hz)))) if step > 0 else 1
    keep_idx = np.arange(0, len(uf), stride)
    keep = uf[keep_idx]
    pos = np.searchsorted(keep, fr)
    posc = np.minimum(pos, len(keep) - 1)
    mask = keep[posc] == fr
    fi, pid, tid = posc[mask], sub["player_id"][mask], sub["team_id"][mask]
    x, y = sub["x"][mask], sub["y"][mask]
    off_ids = _lineup(pid, tid == offense_team_id)
    def_ids = _lineup(pid, tid != offense_team_id)
    if not off_ids or not def_ids:
        return None
    n = len(keep)

    def stack(ids):
        out = np.full((n, len(ids), 2), np.nan)
        for j, i in enumerate(ids):
            sel = pid == i
            out[fi[sel], j, 0] = x[sel]
            out[fi[sel], j, 1] = y[sel]
        return out

    off, de = stack(off_ids), stack(def_ids)
    rows5 = first_idx[keep_idx]
    t = float(clock[0]) - clock[keep_idx]
    t = np.maximum.accumulate(np.where(np.isfinite(t), t, 0.0))
    ball = np.stack([sub["ball_x"][rows5], sub["ball_y"][rows5], sub["ball_z"][rows5]], axis=1)
    shot_clock = sub["shot_clock_s"][rows5]
    mean_off = np.nanmean(off.reshape(-1, 2), axis=0) if np.isfinite(off).any() else np.array([np.nan, np.nan])
    basket = min(BASKETS, key=lambda b: np.linalg.norm(mean_off - b)) if np.isfinite(mean_off).all() else BASKETS[0]
    return PossessionArrays(keep.astype(int), t, ball, shot_clock, np.array(off_ids), off, np.array(def_ids), de, basket)


# ── candidate generation ───────────────────────────────────────────────────────────────────────────
def window_end(t: np.ndarray, i: int, length_s: float, tol: float = 0.11):
    """Largest index j with t[j] <= t[i] + length_s + tol, when the window is at least length_s - tol long."""
    j = int(np.searchsorted(t, t[i] + length_s + tol, side="right")) - 1
    if j <= i or t[j] - t[i] < length_s - tol:
        return None
    return j


def _merge(windows: list) -> list:
    """[(key, i, j)] -> merged [(key, i, j)] where windows of one key that overlap or touch become one."""
    out = []
    for key, i, j in sorted(windows, key=lambda w: (str(w[0]), w[1], w[2])):
        if out and out[-1][0] == key and i <= out[-1][2]:
            out[-1] = (key, out[-1][1], max(out[-1][2], j))
        else:
            out.append((key, i, j))
    return out


def screen_windows(a: PossessionArrays) -> list:
    wins = []
    k = a.off.shape[1]
    for i in range(a.n):
        j = window_end(a.t, i, SCREEN_WINDOW_S)
        h = a.handler[i]
        if j is None or h < 0:
            continue
        hxy = a.off[i:j + 1, h]
        for d in nearest_defenders(a, i, h, SCREEN_N_DEFENDERS):
            if not np.linalg.norm(a.de[i, d] - a.off[i, h]) < SCREEN_DEF_HANDLER_FT:
                continue
            dxy = a.de[i:j + 1, d]
            for s in range(k):
                if s == h or matchup_before(a, s, d, i) == d:   # a player standing beside his own man is not screening
                    continue
                sxy = a.off[i:j + 1, s]
                near_def = np.linalg.norm(sxy - dxy, axis=1) < SCREEN_DIST_FT
                slow = a.speed_off[i:j, s] < SCREEN_SPEED_FTPS
                near_handler = np.linalg.norm(sxy - hxy, axis=1) < SCREEN_HANDLER_DIST_FT
                if near_def.all() and slow.all() and near_handler.all():
                    wins.append(((h, s), i, j, d))
                    break          # one screened defender per (handler, screener, window start)
    screened = {}
    for key, i, j, d in sorted(wins, key=lambda w: w[1]):
        screened.setdefault((key, i), d)
    out = []
    for key, i, j in _merge([(key, i, j) for key, i, j, _d in wins]):
        # the candidate spans the approach, the set and the use (module docstring); its defender is the one of the first window
        i0 = int(np.searchsorted(a.t, a.t[i] - SCREEN_PAD_PRE_S - 1e-6, side="left"))
        i1 = int(np.searchsorted(a.t, a.t[j] + SCREEN_PAD_POST_S + 1e-6, side="right")) - 1
        out.append(("ball_screen", key[0], key[1], screened[(key, i)], i0, max(i1, j)))
    return out


def drive_windows(a: PossessionArrays) -> list:
    wins = []
    for i in range(a.n):
        j = window_end(a.t, i, DRIVE_WINDOW_S)
        h = a.handler[i]
        if j is None or h < 0 or not (a.r_handler[i] <= DRIVE_START_FT):
            continue
        r = np.linalg.norm(a.off[i:j + 1, h] - a.basket, axis=1)
        radial = -np.diff(r) / a.dt[i:j]
        if np.isfinite(radial).all() and (radial > DRIVE_SPEED_FTPS).all():
            wins.append((h, i, j))
    return [("drive", key, None, nearest_defender(a, i, key), i, j) for key, i, j in _merge(wins)]


def isolation_windows(a: PossessionArrays) -> list:
    wins = []
    k = a.off.shape[1]
    for i in range(a.n):
        j = window_end(a.t, i, ISO_WINDOW_S)
        h = a.handler[i]
        if j is None or h < 0 or not (a.r_handler[i] <= ISO_START_FT):
            continue
        if not (a.handler[i:j + 1] == h).all() or not (a.speed_off[i:j, h] < ISO_HANDLER_SPEED_FTPS).all():
            continue
        hxy = a.off[i:j + 1, h]
        if not (np.linalg.norm(a.ball[i:j + 1, :2] - hxy, axis=1) < ISO_BALL_FT).all():
            continue
        mates = [s for s in range(k) if s != h]
        if mates:
            dm = np.linalg.norm(a.off[i:j + 1][:, mates] - hxy[:, None, :], axis=2)
            dm = np.where(np.isfinite(dm), dm, np.inf)
            if (dm.min(axis=1) < ISO_SPREAD_FT).any():
                continue
        wins.append((h, i, j))
    return [("isolation", key, None, nearest_defender(a, i, key), i, j) for key, i, j in _merge(wins)]


def candidate_windows(a: PossessionArrays) -> list:
    """[(rule_class, handler_idx, screener_idx or None, defender_idx, i0, i1)] ordered by start then class order.
    The defender is the screened man of a ball screen, else the handler's nearest defender at the window start."""
    wins = screen_windows(a) + drive_windows(a) + isolation_windows(a)
    return sorted(wins, key=lambda w: (w[4], CLASSES.index(w[0]), w[5]))


# ── features ───────────────────────────────────────────────────────────────────────────────────────
def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _nanmax(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


def _nanmin(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else float("nan")


def candidate_features(a: PossessionArrays, rule_class: str, h: int, s, d: int, i0: int, i1: int) -> dict:
    k = a.off.shape[1]
    if d is None or d < 0:
        d = nearest_defender(a, i0, h)
    mates = [m for m in range(k) if m != h]
    hxy = a.off[i0:i1 + 1, h]
    if s is None and mates and d >= 0:
        dd = np.linalg.norm(a.off[i0][mates] - a.de[i0, d], axis=1)
        s = mates[int(np.nanargmin(np.where(np.isfinite(dd), dd, np.inf)))] if np.isfinite(dd).any() else None
    sl = slice(i0, max(i0 + 1, i1))          # intervals inside the window (at least one)
    f = {}
    if s is not None and d >= 0:
        sd = np.linalg.norm(a.off[i0:i1 + 1, s] - a.de[i0:i1 + 1, d], axis=1)
        f["scr_def_dist_min"], f["scr_def_dist_mean"] = _nanmin(sd), _nanmean(sd)
    else:
        f["scr_def_dist_min"] = f["scr_def_dist_mean"] = float("nan")
    if s is not None:
        f["scr_speed_mean"], f["scr_speed_max"] = _nanmean(a.speed_off[sl, s]), _nanmax(a.speed_off[sl, s])
    else:
        f["scr_speed_mean"] = f["scr_speed_max"] = float("nan")
    sp = a.speed_off[sl, h]
    f["handler_speed_mean"], f["handler_speed_max"] = _nanmean(sp), _nanmax(sp)
    v0, v1 = a.v_off[i0, h], a.v_off[max(i0, i1 - 1), h]
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if np.isfinite(n0) and np.isfinite(n1) and n0 > 0.5 and n1 > 0.5:
        f["handler_dir_change_rad"] = float(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0)))
    else:
        f["handler_dir_change_rad"] = 0.0
    r = np.linalg.norm(hxy - a.basket, axis=1)
    if i1 > i0:
        f["handler_radial_speed_mean"] = _nanmean(-np.diff(r) / a.dt[i0:i1])
    else:
        f["handler_radial_speed_mean"] = 0.0
    f["handler_basket_dist_start"], f["handler_basket_dist_end"] = float(r[0]), float(r[-1])
    if d >= 0:
        dh = np.linalg.norm(a.de[i0:i1 + 1, d] - hxy, axis=1)
        f["def_handler_dist_mean"] = _nanmean(dh)
        post = [m for m in range(i1 + 1, a.n) if a.t[m] <= a.t[i1] + RECOVERY_S + 1e-6 and a.handler[m] == h]
        if post:
            dp = np.linalg.norm(a.de[post, d] - a.off[post, h], axis=1)
            f["def_recovery_ft"] = _nanmean(dp) - f["def_handler_dist_mean"]
        else:
            f["def_recovery_ft"] = float("nan")
    else:
        f["def_handler_dist_mean"] = f["def_recovery_ft"] = float("nan")
    bz = a.ball[i0:i1 + 1, 2]
    f["ball_z_mean"], f["ball_z_max"] = _nanmean(bz), _nanmax(bz)
    f["shot_clock_start"] = float(a.shot_clock[i0]) if np.isfinite(a.shot_clock[i0]) else float("nan")
    others = np.concatenate([a.off[i0:i1 + 1][:, mates], a.de[i0:i1 + 1]], axis=1) if mates else a.de[i0:i1 + 1]
    do = np.linalg.norm(others - hxy[:, None, :], axis=2)
    f["n_within_8ft_mean"] = float(np.mean((do < NEAR_FT).sum(axis=1)))
    if mates:
        dm = np.linalg.norm(a.off[i0:i1 + 1][:, mates] - hxy[:, None, :], axis=2)
        f["teammate_min_dist_mean"] = _nanmean(np.nanmin(np.where(np.isfinite(dm), dm, np.nan), axis=1)) if np.isfinite(dm).any() else float("nan")
    else:
        f["teammate_min_dist_mean"] = float("nan")
    f["window_s"] = float(a.t[i1] - a.t[i0])
    f["rule_is_screen"] = float(rule_class == "ball_screen")
    f["rule_is_drive"] = float(rule_class == "drive")
    f["rule_is_iso"] = float(rule_class == "isolation")
    return f


# ── build ──────────────────────────────────────────────────────────────────────────────────────────
def build(frames: pd.DataFrame, possessions: pd.DataFrame, sportvu_game: str | None = None) -> tuple:
    """-> (candidates, features): the candidate table (CANDIDATE_COLUMNS) and the feature frame
    (ID_COLUMNS + FEATURES) for every possession of one game. features.attrs['skipped'] counts the
    possessions without arrays (null offense, no rows)."""
    game = sportvu_game or str(possessions["sportvu_game"].iloc[0])
    cand_rows, feat_rows, skipped = [], [], 0
    cols = frame_columns(frames)
    fr_frame = cols["frame"]
    for p in possessions.sort_values("possession").itertuples(index=False):
        if pd.isna(p.offense_team_id):
            skipped += 1
            continue
        lo, hi = np.searchsorted(fr_frame, int(p.start_frame), "left"), np.searchsorted(fr_frame, int(p.end_frame), "right")
        a = possession_arrays({k: v[lo:hi] for k, v in cols.items()}, int(p.offense_team_id))
        if a is None:
            skipped += 1
            continue
        for c, (rule_class, h, s, d, i0, i1) in enumerate(candidate_windows(a)):
            cand_rows.append({"sportvu_game": game, "possession": int(p.possession), "candidate": c,
                              "start_frame": int(a.frames[i0]), "end_frame": int(a.frames[i1]), "rule_class": rule_class,
                              "ball_handler_id": int(a.off_ids[h]),
                              "screener_id": int(a.off_ids[s]) if s is not None else None})
            f = {"sportvu_game": game, "possession": int(p.possession), "candidate": c}
            f.update(candidate_features(a, rule_class, h, s, d, i0, i1))
            feat_rows.append(f)
    cands = pd.DataFrame(cand_rows, columns=CANDIDATE_COLUMNS)
    feats = pd.DataFrame(feat_rows, columns=ID_COLUMNS + FEATURES)
    cands["screener_id"] = cands["screener_id"].astype("Int32")
    for c in FEATURES:
        feats[c] = feats[c].astype(float)
    schema.assert_time_forward(feats, "possessions")
    assert_no_leak(feats.drop(columns=ID_COLUMNS))
    feats.attrs["skipped"] = skipped
    cands.attrs["skipped"] = skipped
    return cands, feats


# ── labels ─────────────────────────────────────────────────────────────────────────────────────────
def iou_frames(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0


def match_labels(cands: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Candidate table + label_class (ball_screen, drive, isolation, none), label_iou (best IoU with any label of
    the possession, 0 without labels) and label_action_type (the best-matched label's type at IoU >= IOU_MIN,
    whatever it is, else None)."""
    out = cands.copy()
    cls, ious, types = [], [], []
    by = {}
    if actions is not None and len(actions):
        for r in actions.itertuples(index=False):
            by.setdefault((str(r.sportvu_game), int(r.possession)), []).append((int(r.start_frame), int(r.end_frame), str(r.action_type)))
    for r in out.itertuples(index=False):
        best, best_t = 0.0, None
        for (s, e, t) in by.get((str(r.sportvu_game), int(r.possession)), []):
            v = iou_frames(int(r.start_frame), int(r.end_frame), s, e)
            if v > best:
                best, best_t = v, t
        hit = best >= IOU_MIN
        types.append(best_t if hit else None)
        cls.append(best_t if hit and best_t in DETECT_CLASSES else "none")
        ious.append(best)
    out["label_class"] = pd.Series(cls, index=out.index, dtype="object")
    out["label_iou"] = np.asarray(ious, dtype=float)
    out["label_action_type"] = pd.Series(types, index=out.index, dtype="object")
    return out


def candidate_recall(cands: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Per detected class: labels, labels hit by any candidate at IoU >= IOU_MIN, labels hit by a candidate of
    the same rule class, and the two recalls."""
    rows = []
    by = {}
    for r in cands.itertuples(index=False):
        by.setdefault((str(r.sportvu_game), int(r.possession)), []).append((int(r.start_frame), int(r.end_frame), str(r.rule_class)))
    for c in DETECT_CLASSES:
        lab = actions[actions["action_type"] == c] if actions is not None and len(actions) else actions
        n = int(len(lab)) if lab is not None else 0
        hit_any = hit_same = 0
        if n:
            for r in lab.itertuples(index=False):
                ws = by.get((str(r.sportvu_game), int(r.possession)), [])
                v = [(iou_frames(int(r.start_frame), int(r.end_frame), s, e), rc) for s, e, rc in ws]
                hit_any += any(x >= IOU_MIN for x, _ in v)
                hit_same += any(x >= IOU_MIN and rc == c for x, rc in v)
        rows.append({"class": c, "labels": n, "hit_any": hit_any, "hit_same_class": hit_same,
                     "recall_any": hit_any / n if n else float("nan"), "recall_same_class": hit_same / n if n else float("nan")})
    return pd.DataFrame(rows)


def build_season(games: list) -> tuple:
    """Over game dicts (sportvu_game, heldout, frames, possessions, labels) -> (candidates with labels and a
    heldout column, features, recall table, skipped possessions)."""
    cs, fs, skipped = [], [], 0
    for g in games:
        c, f = build(g["frames"], g["possessions"], g["sportvu_game"])
        c = match_labels(c, (g.get("labels") or {}).get("actions"))
        c["heldout"] = bool(g.get("heldout", False))
        cs.append(c)
        fs.append(f)
        skipped += f.attrs.get("skipped", 0)
    cands = pd.concat(cs, ignore_index=True) if cs else match_labels(pd.DataFrame(columns=CANDIDATE_COLUMNS), None)
    feats = pd.concat(fs, ignore_index=True) if fs else pd.DataFrame(columns=ID_COLUMNS + FEATURES)
    actions = [g["labels"]["actions"] for g in games if g.get("labels") is not None and len(g["labels"]["actions"])]
    actions = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(columns=["sportvu_game", "possession", "action_type", "start_frame", "end_frame"])
    return cands, feats, candidate_recall(cands, actions), skipped


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> dict:
    from analytics.actions import validate as V
    ap = argparse.ArgumentParser(prog="python -m analytics.actions.features", description=__doc__.split("\n")[0])
    V.add_input_args(ap)
    ap.add_argument("--out", help="OUT_ROOT: writes analytics/actions/<season>/features/<game>.parquet under it")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    games = V.load_inputs(a)
    cands, feats, recall, skipped = build_season(games)
    summary = {"games": len(games), "candidates": int(len(cands)), "skipped_possessions": int(skipped),
               "by_rule_class": {k: int(v) for k, v in cands["rule_class"].value_counts().items()},
               "by_label_class": {k: int(v) for k, v in cands["label_class"].value_counts().items()},
               "recall": recall.to_dict("records"), "fixtures": bool(a.fixtures)}
    if a.out:
        from analytics.possessions import game_meta
        for g in games:
            season = game_meta(g["sportvu_game"])["season"]
            d = Path(a.out) / "analytics" / "actions" / ("heldout/" if g["heldout"] else "") / season / "features"
            d.mkdir(parents=True, exist_ok=True)
            m = feats["sportvu_game"] == g["sportvu_game"]
            feats[m].to_parquet(d / (g["sportvu_game"] + ".parquet"), index=False)
        summary["out"] = str(Path(a.out) / "analytics" / "actions")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()

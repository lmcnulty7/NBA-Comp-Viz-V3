"""analytics/decisions/features.py: decision events from the ball path, their contexts and option values.

    python -m analytics.decisions.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]
    (prints the decision counts per kind and game; the contexts are not written, train.py writes the events)

Step 1, event detection (per possession, offence players only; fps from the frames, 25 when unreadable):
  holder   the offensive player within HOLD_FT = 2.5 ft (horizontal) of the ball while the ball is under
           HOLD_Z_FT = 8 ft; the nearest when several; a holder segment shorter than MIN_HOLD_FRAMES = 3 frames
           is ignored (the ball passing near a player is not a possession of it)
  pass     holder A's segment ends at frame f and teammate B's segment starts at frame g with
           (g - f) / fps <= PASS_MAX_S = 1.5 and no shot release in [f - 2, g]; the decision frame is f
  shot     every shot_frames row with release_frame in the possession span; the decision frame is the release;
           the player is shooter_at_release_id, else shooter_id; skipped (and counted) when not on the offence
  drive    the holder's speed toward the basket (from the previous frame) stays above DRIVE_FTPS = 8 for at least
           DRIVE_S = 0.6 s inside one holder segment; the decision frame is the first frame of that run, one per
           segment
Step 2, option values at the decision frame (positions at that frame and earlier only):
  shot value(x, y, d)   = points(dist) * max(p(x, y) - CONTEST_PER_FT * max(0, CONTEST_FT - d), P_FLOOR)
                          points = 3 at 22 ft or more, else 2; p from the value model (location-only xFG), the
                          contest term is the assumption that stands in for a contest model (minus 0.03 per foot
                          below 6 ft of nearest-defender distance). When the xFG lane's p_selected is joined for
                          the shot event, it replaces p and the contest term is not applied (p_selected has one).
  pass option(teammate) = teammate's shot value * openness * risk, openness = 0.5 + 0.5 * min(d, 6) / 6 of the
                          teammate's nearest-defender distance, risk = 0.95 - 0.01 * max(0, pass length - 15 ft)
  hold                  = the EPV state value (epv_points of the last state at or before the frame) when present
  drive (chosen only)   = shot value at the position 1 s ahead along the decision-frame velocity (never nearer
                          than 3 ft to the basket), with the decision-frame defender distance, times DRIVE_RISK 0.9
  decision quality      = chosen value minus the max over {shot, pass options, hold}; a drive is not in the option
                          set (best_option is one of shot, pass_to_<rank>, hold), so a drive's quality may exceed 0
The attacked basket of a possession is the end of the court where the offence's mean x over the span lies
(assumption: the frames do not say which basket a team attacks).

Leakage: contexts hold the decision frame's geometry only; assert_time_forward (schema) runs on every context frame
and the FEATURES subset; assert_decisions_before_outcome refuses a decision frame later than its possession's end
or, for a shot, later than its release frame.
"""
from __future__ import annotations

import argparse
from typing import Callable, Optional

import numpy as np
import pandas as pd

from analytics import schema

FPS_DEFAULT = 25.0
HOLD_FT, HOLD_Z_FT, MIN_HOLD_FRAMES = 2.5, 8.0, 3
PASS_MAX_S, RELEASE_TOL_FRAMES = 1.5, 2
DRIVE_FTPS, DRIVE_S = 8.0, 0.6
CONTEST_PER_FT, CONTEST_FT, P_FLOOR = 0.03, 6.0, 0.01
OPEN_FT, OPEN_MIN = 6.0, 0.5
PASS_RISK_BASE, PASS_RISK_PER_FT, PASS_RISK_FT = 0.95, 0.01, 15.0
DRIVE_LOOKAHEAD_S, DRIVE_MIN_FT, DRIVE_RISK = 1.0, 3.0, 0.9
THREE_FT = 22.0
RIM_LEFT, RIM_RIGHT, HALF_X = (5.25, 25.0), (88.75, 25.0), 47.0
MAX_TEAMMATES = 4
KINDS = ("shot", "pass", "drive")

FEATURES = ["dist_to_basket_ft", "angle_rad", "nearest_def_dist_ft", "n_teammates", "n_teammates_open",
            "elapsed_in_possession_s", "ball_z", "speed_toward_basket_ftps"]
TEAMMATE_FIELDS = ("id", "x", "y", "def_ft", "pass_ft")
CONTEXT_IDS = ["sportvu_game", "possession", "frame", "event", "player_id", "receiver_id", "decision_kind", "heldout"]


def feature_columns() -> list:
    return list(FEATURES)


def teammate_columns() -> list:
    return ["tm%d_%s" % (k, f) for k in range(1, MAX_TEAMMATES + 1) for f in TEAMMATE_FIELDS]


# ── geometry ───────────────────────────────────────────────────────────────────────────────────────
def fps_of(frames: pd.DataFrame) -> float:
    """Frames per second from ts_ms when present (median step of the first 2000 frames), else from the game
    clock, else 25."""
    fl = frames.drop_duplicates("frame").head(2000)
    if "ts_ms" in fl.columns and fl["ts_ms"].notna().all() and len(fl) > 1:
        step = np.median(np.diff(fl["ts_ms"].to_numpy(dtype=float)))
        if 10 <= step <= 1000:
            return float(1000.0 / step)
    if len(fl) > 1:
        dc = -np.diff(fl["game_clock_s"].to_numpy(dtype=float))
        dc = dc[(dc > 0) & (dc < 1)]
        if len(dc):
            return float(1.0 / np.median(dc))
    return FPS_DEFAULT


def points_of(dist_ft) -> np.ndarray:
    return np.where(np.asarray(dist_ft, dtype=float) >= THREE_FT, 3.0, 2.0)


def basket_of(mean_x: float) -> tuple:
    return RIM_RIGHT if mean_x > HALF_X else RIM_LEFT


def loc_tenths(x, y, bx, by) -> tuple:
    """Court feet -> the section 10 convention in tenths of a foot: loc_y toward half court, loc_x across."""
    x, y, bx, by = (np.asarray(v, dtype=float) for v in (x, y, bx, by))
    inward = np.where(bx > HALF_X, -1.0, 1.0)
    return (y - by) * inward * 10.0, (x - bx) * inward * 10.0


def contested(p, def_ft):
    """The assumption that stands in for a contest model: minus 0.03 per foot below 6 ft."""
    p = np.asarray(p, dtype=float)
    d = np.asarray(def_ft, dtype=float)
    pen = CONTEST_PER_FT * np.clip(CONTEST_FT - d, 0, None)
    return np.maximum(p - np.where(np.isfinite(d), pen, 0.0), P_FLOOR)


def openness(def_ft):
    d = np.asarray(def_ft, dtype=float)
    return OPEN_MIN + (1 - OPEN_MIN) * np.clip(d, 0, OPEN_FT) / OPEN_FT


def pass_risk(length_ft):
    return PASS_RISK_BASE - PASS_RISK_PER_FT * np.clip(np.asarray(length_ft, dtype=float) - PASS_RISK_FT, 0, None)


# ── per-possession arrays ──────────────────────────────────────────────────────────────────────────
class Possession:
    """Frames of one possession as arrays: ball (n, 3), offence xy (n, k, 2), defence xy (n, m, 2)."""

    def __init__(self, frames_sorted: pd.DataFrame, frame_values: np.ndarray, row, fps: float):
        self.possession = int(row.possession)
        self.start, self.end = int(row.start_frame), int(row.end_frame)
        self.fps = fps
        lo = int(np.searchsorted(frame_values, self.start, side="left"))
        hi = int(np.searchsorted(frame_values, self.end, side="right"))
        sub = frames_sorted.iloc[lo:hi]
        self.frames, inv = np.unique(sub["frame"].to_numpy(), return_inverse=True)
        n = len(self.frames)
        self.n = n
        off_team = row.offense_team_id
        is_off = (sub["team_id"].to_numpy() == off_team)
        self.ok = n > 0 and is_off.any() and (~is_off).any()
        ball = np.full((n, 3), np.nan)
        for j, c in enumerate(("ball_x", "ball_y", "ball_z")):
            ball[inv, j] = sub[c].to_numpy(dtype=float)
        self.ball = ball
        self.off_ids, self.off = self._side(sub, inv, is_off)
        self.def_ids, self.dfn = self._side(sub, inv, ~is_off)
        self.basket = basket_of(float(np.nanmean(self.off[:, :, 0]))) if self.ok else RIM_LEFT
        self.bx, self.by = self.basket

    def _side(self, sub, inv, mask):
        ids, pinv = np.unique(sub["player_id"].to_numpy()[mask], return_inverse=True)
        xy = np.full((self.n, len(ids), 2), np.nan)
        xy[inv[mask], pinv, 0] = sub["x"].to_numpy(dtype=float)[mask]
        xy[inv[mask], pinv, 1] = sub["y"].to_numpy(dtype=float)[mask]
        return ids, xy

    def holders(self) -> np.ndarray:
        """Index into off_ids of the holder per frame, -1 when nobody holds the ball."""
        d = np.hypot(self.off[:, :, 0] - self.ball[:, None, 0], self.off[:, :, 1] - self.ball[:, None, 1])
        d = np.where(np.isnan(d), np.inf, d)
        near = np.argmin(d, axis=1)
        dmin = d[np.arange(self.n), near]
        ok = (dmin <= HOLD_FT) & (self.ball[:, 2] < HOLD_Z_FT)
        return np.where(ok, near, -1)

    def segments(self, holder: np.ndarray) -> list:
        """(holder index, first frame index, last frame index) of the holder runs at least MIN_HOLD_FRAMES long."""
        out = []
        if self.n == 0:
            return out
        cut = np.flatnonzero(np.diff(holder) != 0) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut - 1, self.n - 1]
        for s, e in zip(starts, ends):
            if holder[s] >= 0 and e - s + 1 >= MIN_HOLD_FRAMES:
                out.append((int(holder[s]), int(s), int(e)))
        return out

    def def_dist(self, i: int) -> np.ndarray:
        """Nearest-defender distance of every offensive player at frame index i."""
        o, d = self.off[i], self.dfn[i]
        dist = np.hypot(o[:, None, 0] - d[None, :, 0], o[:, None, 1] - d[None, :, 1])
        dist = np.where(np.isnan(dist), np.inf, dist)
        out = dist.min(axis=1) if dist.shape[1] else np.full(len(o), np.inf)
        return np.where(np.isfinite(out), out, np.nan)


# ── detection ──────────────────────────────────────────────────────────────────────────────────────
def detect_passes(pos: Possession, releases: np.ndarray, segs=None) -> list:
    """[(frame index of the decision, passer index, receiver index)] from the holder runs."""
    segs = pos.segments(pos.holders()) if segs is None else segs
    out = []
    rel = np.asarray(releases)
    for (a, sa, ea), (b, sb, eb) in zip(segs, segs[1:]):
        if a == b:
            continue
        if (sb - ea) / pos.fps > PASS_MAX_S:
            continue
        fa, fb = pos.frames[ea], pos.frames[sb]
        if len(rel) and np.any((rel >= fa - RELEASE_TOL_FRAMES) & (rel <= fb)):
            continue
        out.append((ea, a, b))
    return out


def detect_drives(pos: Possession, segs=None) -> list:
    """[(frame index of the drive start, handler index)]: speed toward the basket above DRIVE_FTPS for DRIVE_S."""
    segs = pos.segments(pos.holders()) if segs is None else segs
    need = max(int(np.ceil(DRIVE_S * pos.fps)), 2)
    out = []
    for h, s, e in segs:
        if e - s + 1 < need + 1:
            continue
        xy = pos.off[s:e + 1, h]
        dist = np.hypot(xy[:, 0] - pos.bx, xy[:, 1] - pos.by)
        speed = -np.diff(dist) * pos.fps            # speed[k] is from frame s + k to s + k + 1
        fast = speed > DRIVE_FTPS
        run = 0
        for k, f in enumerate(fast):
            run = run + 1 if f else 0
            if run >= need:
                out.append((s + k + 1 - need + 1, h))
                break
    return out


def shot_rows(shot_frames: pd.DataFrame, pos: Possession) -> list:
    """[(frame index of the release, shooter index, event)] of the shots released inside the possession."""
    out = []
    sf = shot_frames
    rel = pd.to_numeric(sf["release_frame"]).to_numpy()
    m = (rel >= pos.start) & (rel <= pos.end) & (rel != schema.NOT_FOUND)
    for _, r in sf[m].iterrows():
        pid = r["shooter_at_release_id"] if pd.notna(r.get("shooter_at_release_id", None)) else r.get("shooter_id", None)
        if pd.isna(pid):
            out.append((None, None, int(r["event"])))
            continue
        hit = np.flatnonzero(pos.off_ids == int(pid))
        i = int(np.searchsorted(pos.frames, int(r["release_frame"])))
        if not len(hit) or i >= pos.n or pos.frames[i] != int(r["release_frame"]):
            out.append((None, None, int(r["event"])))
            continue
        out.append((i, int(hit[0]), int(r["event"])))
    return out


# ── contexts ───────────────────────────────────────────────────────────────────────────────────────
def _context(pos: Possession, i: int, h: int, kind: str, game: str, heldout: bool, event: int = -1,
             receiver: int = -1) -> Optional[dict]:
    o = pos.off[i]
    if np.isnan(o[h]).any():
        return None
    dd = pos.def_dist(i)
    hx, hy = o[h]
    dist = float(np.hypot(hx - pos.bx, hy - pos.by))
    inward = -1.0 if pos.bx > HALF_X else 1.0
    if i > 0 and not np.isnan(pos.off[i - 1, h]).any():
        v = (o[h] - pos.off[i - 1, h]) * pos.fps
        d_prev = float(np.hypot(*(pos.off[i - 1, h] - np.array(pos.basket))))
        speed = (d_prev - dist) * pos.fps
    else:
        v, speed = np.zeros(2), 0.0
    c = {"sportvu_game": game, "possession": pos.possession, "frame": int(pos.frames[i]), "event": int(event),
         "player_id": int(pos.off_ids[h]), "receiver_id": int(pos.off_ids[receiver]) if receiver >= 0 else -1,
         "decision_kind": kind, "heldout": bool(heldout), "basket_x": pos.bx, "basket_y": pos.by,
         "x": float(hx), "y": float(hy), "vx": float(v[0]), "vy": float(v[1]),
         "dist_to_basket_ft": dist, "angle_rad": float(np.arctan2((hy - pos.by) * inward, (hx - pos.bx) * inward)),
         "nearest_def_dist_ft": float(dd[h]), "elapsed_in_possession_s": float((pos.frames[i] - pos.start) / pos.fps),
         "ball_z": float(pos.ball[i, 2]), "speed_toward_basket_ftps": float(speed)}
    mates = [j for j in range(len(pos.off_ids)) if j != h and not np.isnan(o[j]).any()]
    mates = mates[:MAX_TEAMMATES]
    c["n_teammates"] = len(mates)
    c["n_teammates_open"] = int(sum(1 for j in mates if dd[j] >= OPEN_FT))
    for k in range(1, MAX_TEAMMATES + 1):
        if k <= len(mates):
            j = mates[k - 1]
            c["tm%d_id" % k], c["tm%d_x" % k], c["tm%d_y" % k] = int(pos.off_ids[j]), float(o[j, 0]), float(o[j, 1])
            c["tm%d_def_ft" % k] = float(dd[j])
            c["tm%d_pass_ft" % k] = float(np.hypot(o[j, 0] - hx, o[j, 1] - hy))
        else:
            c["tm%d_id" % k] = -1
            for f in ("x", "y", "def_ft", "pass_ft"):
                c["tm%d_%s" % (k, f)] = np.nan
    return c


def context_columns() -> list:
    return CONTEXT_IDS + ["basket_x", "basket_y", "x", "y", "vx", "vy"] + FEATURES + teammate_columns()


def decision_contexts(bundle: dict, counts: Optional[dict] = None) -> pd.DataFrame:
    """One row per decision event of the game with its decision-frame geometry (no values, no outcomes)."""
    counts = counts if counts is not None else {}
    game, heldout = bundle["sportvu_game"], bool(bundle["heldout"])
    frames = bundle["frames"]
    if not frames["frame"].is_monotonic_increasing:
        frames = frames.sort_values("frame", kind="stable")
    fv = frames["frame"].to_numpy()
    fps = fps_of(frames)
    poss = bundle["possessions"]
    sf = bundle["shot_frames"]
    rows = []
    for row in poss.itertuples(index=False):
        if pd.isna(row.offense_team_id):
            counts["possessions_no_offence"] = counts.get("possessions_no_offence", 0) + 1
            continue
        if bool(getattr(row, "gap", False)):
            counts["possessions_gap"] = counts.get("possessions_gap", 0) + 1
            continue
        pos = Possession(frames, fv, row, fps)
        if not pos.ok:
            counts["possessions_empty"] = counts.get("possessions_empty", 0) + 1
            continue
        shots = shot_rows(sf, pos)
        releases = np.array([pos.frames[i] for i, _, _ in shots if i is not None])
        segs = pos.segments(pos.holders())
        for i, h, ev in shots:
            if i is None:
                counts["shots_unlocated"] = counts.get("shots_unlocated", 0) + 1
                continue
            c = _context(pos, i, h, "shot", game, heldout, event=ev)
            if c:
                rows.append(c)
        for i, a, b in detect_passes(pos, releases, segs):
            c = _context(pos, i, a, "pass", game, heldout, receiver=b)
            if c:
                rows.append(c)
        for i, h in detect_drives(pos, segs):
            c = _context(pos, i, h, "drive", game, heldout)
            if c:
                rows.append(c)
        counts["possessions_used"] = counts.get("possessions_used", 0) + 1
    out = pd.DataFrame(rows, columns=context_columns())
    if len(out):
        out = out.sort_values(["possession", "frame", "decision_kind"], kind="stable").reset_index(drop=True)
    schema.assert_time_forward(out, "states_1hz")
    return out


def feature_frame(contexts: pd.DataFrame) -> pd.DataFrame:
    """The FEATURES subset, checked by schema.assert_time_forward."""
    f = contexts[feature_columns()].astype(float)
    schema.assert_time_forward(f, "states_1hz")
    return f


# ── leakage guard ──────────────────────────────────────────────────────────────────────────────────
def assert_decisions_before_outcome(contexts: pd.DataFrame, shot_frames: pd.DataFrame, possessions: pd.DataFrame) -> None:
    """ValueError when a decision frame lies after its possession's last frame, or a shot decision after its
    release frame (the outcome of a shot begins at the release)."""
    if not len(contexts):
        return
    p = possessions[["possession", "end_frame", "start_frame"]].drop_duplicates("possession")
    m = contexts.merge(p, on="possession", how="left")
    late = m["frame"] > m["end_frame"]
    early = m["frame"] < m["start_frame"]
    if late.any() or early.any():
        raise ValueError("decisions: %d decision frame(s) outside their possession span (%d after the end, %d before "
                         "the start): the decision frame must precede the outcome" % (int(late.sum() + early.sum()),
                                                                                    int(late.sum()), int(early.sum())))
    shots = m[m["decision_kind"] == "shot"]
    if len(shots):
        rel = shot_frames[["event", "release_frame"]].drop_duplicates("event")
        s = shots.merge(rel, on="event", how="left")
        bad = s["release_frame"].notna() & (s["frame"] > s["release_frame"])
        if bad.any():
            raise ValueError("decisions: %d shot decision frame(s) later than the release frame: option values would "
                             "see the outcome" % int(bad.sum()))


# ── option values ──────────────────────────────────────────────────────────────────────────────────
def _shot_value(p_fn: Callable, x, y, bx, by, def_ft, p_override=None):
    dist = np.hypot(np.asarray(x, float) - np.asarray(bx, float), np.asarray(y, float) - np.asarray(by, float))
    p = p_fn(x, y, bx, by)
    pc = contested(p, def_ft)
    if p_override is not None:
        ov = np.asarray(p_override, dtype=float)
        pc = np.where(np.isfinite(ov), np.maximum(ov, P_FLOOR), pc)
    return points_of(dist) * pc, dist


def option_values(contexts: pd.DataFrame, p_fn: Callable, hold: Optional[pd.Series] = None,
                  p_shot_override: Optional[pd.Series] = None, prior: str = "location") -> pd.DataFrame:
    """Decision events with chosen_value, best_value, best_option and decision_quality.

    p_fn(x, y, bx, by) -> make probability at a court position toward the basket (bx, by); hold is the EPV value per
    row (NaN where absent); p_shot_override replaces the handler's shot probability where finite (the xFG lane)."""
    c = contexts
    n = len(c)
    out = c[CONTEXT_IDS].copy()
    if n == 0:
        for col in ("chosen_value", "best_value", "decision_quality", "shot_value", "hold_value"):
            out[col] = pd.Series(dtype=float)
        out["best_option"] = pd.Series(dtype="string")
        out["prior"] = pd.Series(dtype="string")
        return out
    bx, by = c["basket_x"].to_numpy(float), c["basket_y"].to_numpy(float)
    ov = None if p_shot_override is None else np.asarray(p_shot_override, dtype=float)
    shot_v, _ = _shot_value(p_fn, c["x"].to_numpy(float), c["y"].to_numpy(float), bx, by,
                            c["nearest_def_dist_ft"].to_numpy(float), ov)
    passes = np.full((n, MAX_TEAMMATES), -np.inf)
    tm_ids = np.full((n, MAX_TEAMMATES), -1)
    for k in range(1, MAX_TEAMMATES + 1):
        ids = c["tm%d_id" % k].to_numpy()
        present = ids >= 0
        if not present.any():
            continue
        tx, ty = c["tm%d_x" % k].to_numpy(float), c["tm%d_y" % k].to_numpy(float)
        dft = c["tm%d_def_ft" % k].to_numpy(float)
        sv, _ = _shot_value(p_fn, np.where(present, tx, 0.0), np.where(present, ty, 0.0), bx, by, dft)
        v = sv * openness(dft) * pass_risk(c["tm%d_pass_ft" % k].to_numpy(float))
        passes[:, k - 1] = np.where(present, v, -np.inf)
        tm_ids[:, k - 1] = ids
    hold_v = np.full(n, np.nan) if hold is None else np.asarray(hold, dtype=float)
    # drive value: the position one second ahead at the decision-frame velocity, never nearer than 3 ft
    px = c["x"].to_numpy(float) + c["vx"].to_numpy(float) * DRIVE_LOOKAHEAD_S
    py = c["y"].to_numpy(float) + c["vy"].to_numpy(float) * DRIVE_LOOKAHEAD_S
    dx, dy = px - bx, py - by
    dd = np.hypot(dx, dy)
    scale = np.where(dd < DRIVE_MIN_FT, DRIVE_MIN_FT / np.maximum(dd, 1e-9), 1.0)
    px, py = bx + dx * scale, by + dy * scale
    drive_v, _ = _shot_value(p_fn, px, py, bx, by, c["nearest_def_dist_ft"].to_numpy(float))
    drive_v = drive_v * DRIVE_RISK
    kind = c["decision_kind"].to_numpy()
    recv = c["receiver_id"].to_numpy()
    chosen = np.full(n, np.nan)
    chosen[kind == "shot"] = shot_v[kind == "shot"]
    chosen[kind == "drive"] = drive_v[kind == "drive"]
    is_pass = kind == "pass"
    hit = (tm_ids == recv[:, None]) & is_pass[:, None]
    has = hit.any(axis=1)
    chosen[has] = passes[hit]
    # best over shot, pass options, hold
    hold_f = np.where(np.isfinite(hold_v), hold_v, -np.inf)
    cand = np.column_stack([shot_v, passes, hold_f])
    best_i = np.argmax(cand, axis=1)
    best_v = cand[np.arange(n), best_i]
    order = np.argsort(-passes, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(1, MAX_TEAMMATES + 1)
    names = np.where(best_i == 0, "shot", np.where(best_i == MAX_TEAMMATES + 1, "hold", ""))
    pass_best = (best_i >= 1) & (best_i <= MAX_TEAMMATES)
    names = names.astype(object)
    for r in np.flatnonzero(pass_best):
        names[r] = "pass_to_%d" % rank[r, best_i[r] - 1]
    out["chosen_value"] = chosen
    out["best_value"] = best_v
    out["best_option"] = pd.array(names, dtype="string")
    out["decision_quality"] = chosen - best_v
    out["shot_value"] = shot_v
    out["hold_value"] = hold_v
    out["prior"] = prior
    keep = np.isfinite(chosen)
    out = out[keep].reset_index(drop=True)
    return out


# ── EPV hold values and the xFG join ───────────────────────────────────────────────────────────────
def hold_values(contexts: pd.DataFrame, epv: Optional[pd.DataFrame]) -> pd.Series:
    """epv_points of the last EPV state at or before each decision frame in the same possession (NaN when absent)."""
    if epv is None or not len(contexts) or "epv_points" not in epv.columns:
        return pd.Series(np.nan, index=contexts.index)
    e = epv[["possession", "frame", "epv_points"]].dropna().copy()
    e["possession"] = e["possession"].astype(int)
    e["frame"] = e["frame"].astype(int)
    e = e.sort_values("frame")
    c = contexts[["possession", "frame"]].copy()
    c["possession"] = c["possession"].astype(int)
    c["frame"] = c["frame"].astype(int)
    c["_i"] = np.arange(len(c))
    m = pd.merge_asof(c.sort_values("frame"), e, on="frame", by="possession", direction="backward")
    return pd.Series(m.sort_values("_i")["epv_points"].to_numpy(dtype=float), index=contexts.index)


def xfg_override(contexts: pd.DataFrame, xfg: Optional[pd.DataFrame], column: str = "p_selected") -> pd.Series:
    """The xFG lane's probability for the shot decisions (joined by event), NaN elsewhere."""
    out = pd.Series(np.nan, index=contexts.index)
    if xfg is None or column not in xfg.columns or "event" not in xfg.columns or not len(contexts):
        return out
    x = xfg[["event", column]].dropna().drop_duplicates("event")
    x["event"] = x["event"].astype(int)
    s = contexts[["event", "decision_kind"]].copy()
    s["event"] = s["event"].astype(int)
    m = s.merge(x, on="event", how="left")
    vals = np.where(m["decision_kind"].to_numpy() == "shot", m[column].to_numpy(dtype=float), np.nan)
    return pd.Series(vals, index=contexts.index)


# ── on-court offensive rating ──────────────────────────────────────────────────────────────────────
def on_court_offensive_rating(frames: pd.DataFrame, possessions: pd.DataFrame, sportvu_game: str) -> pd.DataFrame:
    """Per player of the game: possessions on the floor for the offence, points on them, and points per 100.
    The players of a possession are the offence team's rows at the first tracked frame of its span."""
    p = possessions.dropna(subset=["offense_team_id"])
    if not len(p):
        return pd.DataFrame(columns=["sportvu_game", "player_id", "n_possessions", "points", "ortg"])
    fv = np.sort(frames["frame"].unique())
    idx = np.searchsorted(fv, p["start_frame"].to_numpy())
    idx = np.clip(idx, 0, len(fv) - 1)
    first = fv[idx]
    key = pd.DataFrame({"frame": first, "offense_team_id": p["offense_team_id"].astype(int).to_numpy(),
                        "points": p["points"].astype(int).to_numpy(), "possession": p["possession"].to_numpy()})
    fr = frames[frames["frame"].isin(set(first.tolist()))][["frame", "team_id", "player_id"]]
    m = key.merge(fr, left_on=["frame", "offense_team_id"], right_on=["frame", "team_id"], how="inner")
    g = m.groupby("player_id").agg(n_possessions=("possession", "nunique"), points=("points", "sum")).reset_index()
    g["ortg"] = 100.0 * g["points"] / g["n_possessions"]
    g.insert(0, "sportvu_game", sportvu_game)
    return g


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> pd.DataFrame:
    from analytics.decisions import validate as V
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--season", default=V.DEFAULT_SEASON)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-games", type=int, default=8)
    a = ap.parse_args(argv)
    if not a.fixtures and not a.data:
        raise SystemExit("features: pass --fixtures or --data DATA_ROOT")
    bundles = V.load_fixture_bundles(a.seed, a.n_games) if a.fixtures else V.load_root_bundles(a.data, a.season, a.out)
    bundles, _ = V.validate_bundles(bundles)
    parts = []
    for b in bundles:
        counts = {}
        c = decision_contexts(b, counts)
        assert_decisions_before_outcome(c, b["shot_frames"], b["possessions"])
        parts.append(c)
        print("%s%s: %s | %s" % (b["sportvu_game"], " (held out)" if b["heldout"] else "",
                                 c["decision_kind"].value_counts().to_dict(), counts))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=context_columns())
    print("features: %s" % feature_columns())
    if a.fixtures:
        print("fixture output, not a result")
    return out


if __name__ == "__main__":
    main()

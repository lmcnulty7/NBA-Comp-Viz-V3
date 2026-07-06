"""
court/trajectories.py — Component A: trajectory correction.

Turns noisy per-frame player court positions into clean per-player court paths:
robustly removes teleport-like outliers (the symptom of a bad single-frame
homography), interpolates the gaps, and Savitzky-Golay-smooths the result.

`clean_paths` is vendored from roboflow/sports (sports/common/path.py,
feat/basketball, Apache-2.0) — it's pure numpy + scipy, so we keep it in-house
rather than depend on the whole library. `clean_trajectories` adapts it to our
BoT-SORT output (variable track rosters) by processing each track over its own
presence span.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# ── vendored from roboflow/sports (Apache-2.0) ────────────────────────────────
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _runs(bool_arr: np.ndarray):
    idx = np.flatnonzero(bool_arr)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(g[0], g[-1]) for g in np.split(idx, splits)]


def _linear_interp_1d(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    isnan = np.isnan(out)
    if isnan.all():
        return out
    x = np.arange(out.shape[0])
    valid = np.flatnonzero(~isnan)
    out[: valid[0]] = out[valid[0]]
    out[valid[-1] + 1 :] = out[valid[-1]]
    isnan = np.isnan(out)
    if isnan.any():
        out[isnan] = np.interp(x[isnan], x[~isnan], out[~isnan])
    return out


def _savgol_safe(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    y = y.astype(float)
    n = y.shape[0]
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2:
        k = min(5, n)
        if k % 2 == 0:        # force odd kernel so 'valid' convolution preserves length
            k -= 1
        if k < 2:
            return y
        pad = k // 2
        return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def clean_paths(video_xy, jump_sigma=5.0, min_jump_dist=0.7, max_jump_run=6,
                pad_around_runs=1, smooth_window=9, smooth_poly=2):
    """(T, P, 2) court coords (NaN = missing) → (cleaned (T,P,2), edited_mask (T,P))."""
    T, P, _ = video_xy.shape
    cleaned = video_xy.astype(float).copy()
    edited = np.zeros((T, P), dtype=bool)
    for p in range(P):
        traj = cleaned[:, p, :]
        speed = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        if speed.size == 0 or np.all(~np.isfinite(speed)):
            continue
        fin = speed[np.isfinite(speed)]
        med = np.median(fin)
        scale = max(_mad(fin), 1e-6)
        jump_mask = (speed > med + jump_sigma * scale) & (speed > min_jump_dist)
        remove = np.zeros(T, dtype=bool)
        for s, e in _runs(jump_mask):
            if (e - s + 1) <= max_jump_run:
                remove[max(0, s - pad_around_runs): min(T - 1, e + 1 + pad_around_runs) + 1] = True
        if remove.any():
            edited[:, p] |= remove
            traj[remove, :] = np.nan
        for d in range(2):
            traj[:, d] = _savgol_safe(_linear_interp_1d(traj[:, d]), smooth_window, smooth_poly)
        cleaned[:, p, :] = traj
    return cleaned, edited


# ── adapter for our (variable-roster) BoT-SORT tracks ─────────────────────────
def clean_trajectories(per_track: dict, frame_order: list, **params) -> dict:
    """per_track: {track_id: {frame_idx: (x, y)}} → {track_id: [(frame, x, y, was_edited)]}.
    Each track is cleaned over its own presence span (first→last observed frame)."""
    fidx = {f: i for i, f in enumerate(frame_order)}
    out = {}
    for tid, positions in per_track.items():
        frames = sorted(positions)
        if len(frames) < 2:
            out[tid] = [(f, float(positions[f][0]), float(positions[f][1]), False) for f in frames]
            continue
        span = frame_order[fidx[frames[0]]: fidx[frames[-1]] + 1]
        arr = np.full((len(span), 1, 2), np.nan)
        for k, f in enumerate(span):
            if f in positions:
                arr[k, 0, :] = positions[f]
        cleaned, edited = clean_paths(arr, **params)
        out[tid] = [(span[k], float(cleaned[k, 0, 0]), float(cleaned[k, 0, 1]), bool(edited[k, 0]))
                    for k in range(len(span))]
    return out

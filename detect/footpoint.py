"""
detect/footpoint.py — FootPointStabilizer: stable per-track foot points.

Why: court positions come from projecting the bbox bottom-center through the
homography. The court solver is now good (DEVLOG 2026-07-05); the remaining
teleports come from the FOOT POINT moving when the BOX changes, not the player:
  • occlusion clips the legs → the box bottom jumps UP the image → the projection
    slides several court-feet toward the camera in one frame;
  • per-frame detector jitter on box edges (±2–4 px) is amplified by perspective
    (at the far court 1 px ≈ 0.5+ ft on the floor).

Two per-track corrections, both in PIXEL space (before the homography amplifies):
  1. Height-clip correction — keep a running median of the track's bbox height
     (≈ standing height at the current zoom). If this frame's box is much shorter
     than that median, the bottom is presumed clipped (occlusion, or legs off the
     bottom of the frame) and the foot is re-extended to top + median height.
     The raw height is always appended to the history, so the median stays robust
     to short occlusion runs (< half the window) yet still adapts to real zoom
     changes; long occlusions gracefully degrade to "no correction".
  2. EMA smoothing on the foot point to kill single-frame edge jitter, applied
     only within a continuous run — a gap > FOOT_MAX_GAP processed frames resets
     the state so a genuinely new position isn't dragged toward a stale one.

Known limits (accepted): a jumping player violates the ground-plane assumption
regardless of the foot point; a deep crouch (>15% height loss) triggers a small
over-extension. Both are transient and downstream trajectory cleaning absorbs them.

Streaming, mirrors the tracker:
    stab.stabilize(tracks) -> {track_id: (foot_uv, was_corrected)}
"""
from __future__ import annotations

from collections import deque

import numpy as np

import config


class _TrackState:
    __slots__ = ("heights", "ema", "last_step")

    def __init__(self, height_win: int):
        self.heights: deque[float] = deque(maxlen=height_win)
        self.ema: np.ndarray | None = None
        self.last_step = -1


class FootPointStabilizer:
    WARMUP = 5  # need this many height obs before trusting the median

    def __init__(self, height_win: int = None, occl_frac: float = None,
                 ema_alpha: float = None, max_gap: int = None):
        self.height_win = height_win or config.FOOT_HEIGHT_WIN
        self.occl_frac = occl_frac or config.FOOT_OCCLUSION_FRAC
        self.alpha = ema_alpha or config.FOOT_EMA_ALPHA
        self.max_gap = max_gap or config.FOOT_MAX_GAP
        self._tracks: dict[int, _TrackState] = {}
        self._step = 0
        self.n_clip_fixes = 0   # diagnostics: how often the height correction fired

    def stabilize(self, tracks) -> dict[int, tuple[np.ndarray, bool]]:
        """Feed one frame's tracks; return {track_id: (stabilized foot (2,) f32, corrected)}."""
        self._step += 1
        out: dict[int, tuple[np.ndarray, bool]] = {}
        for t in tracks:
            st = self._tracks.get(t.track_id)
            if st is None or self._step - st.last_step > self.max_gap:
                st = _TrackState(self.height_win)
                self._tracks[t.track_id] = st
            st.last_step = self._step

            h = t.height
            # median of PREVIOUS heights, so an anomalous current box can't defeat its own check
            med = float(np.median(st.heights)) if len(st.heights) >= self.WARMUP else None
            st.heights.append(h)

            foot = t.foot_point.astype(np.float32)
            corrected = False
            if med is not None and h < self.occl_frac * med:
                foot = np.array([foot[0], float(t.bbox[1]) + med], np.float32)
                corrected = True
                self.n_clip_fixes += 1

            st.ema = foot if st.ema is None else self.alpha * foot + (1 - self.alpha) * st.ema
            out[t.track_id] = (st.ema.astype(np.float32), corrected)

        if self._step % 200 == 0:   # GC state for long-gone tracks
            stale = [k for k, v in self._tracks.items() if self._step - v.last_step > self.max_gap]
            for k in stale:
                del self._tracks[k]
        return out

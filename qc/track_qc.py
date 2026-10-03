"""
qc/track_qc.py: label-free quality checks on one build (FIX_PLAN Phase A).

Inputs are what build_trajectories already writes: the <clip>_frames.json sidecar (A0) and the
source video. Every check is a pure function of (frame image, sidecar row) so the same code runs
on a triage snippet locally and on a harvest section in Colab. Nothing here changes the pipeline.

A1  line_overlay(frame, H, boxes): how far the detected line ridges sit from the projected
    court template. Ridge pixels (same ridge_field as the solver, peak >= PEAK_MIN) outside every
    player box are measured against a distance transform of the template drawn through H.
    Reported: dist_px = median distance of ridge pixels to the nearest template line; score =
    share of ridge pixels within NEAR_PX of a template line; support = share of on-frame template
    samples with a ridge within NEAR_PX along their normal (template side). A right H reads a few
    px; a 20 px offset reads ~20; a wrong H reads 60+ with score < 0.08.
    Calibration note (2026-10-03, 55 triage renders): two template-side variants (hit rate at 3 px,
    nearest-ridge distance at 40 px) saturated at ~0.02 and ~17 px for every clip, because most
    template samples have no visible line nearby on 720p footage and production H's are typically
    10..40 px off the paint. Measuring from the ridge side separates clips (27 px good, 70..90 px
    for the clips a human called off). Non-line ridges (logos, crowd remnants, player edges) inflate
    the absolute scale per arena; use it as a ranking and threshold per arena until B6 calibrates it.
"""
from __future__ import annotations

import numpy as np

from court.snap_track import CH_A, CH_B, CH_MID, match_lines, project, ridge_field

NEAR_PX = 6                   # within this of a template line counts as "on the line"
BOX_PAD = 4                   # px grown around player boxes before removing their ridge pixels
MIN_RIDGE_PX = 500            # fewer ridge pixels than this: score undefined (no line evidence in frame)
MIN_SAMPLES = 40              # fewer on-frame template samples than this: support undefined
LINE_SUPPORT_PX = 6           # template-side support: ridge peak within this along the normal


def _inside_boxes(pts: np.ndarray, boxes: list) -> np.ndarray:
    """Boolean mask: pixel point lies inside any xyxy box."""
    inside = np.zeros(len(pts), bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
    return inside


def line_overlay(frame_bgr: np.ndarray, H_px2ft, boxes: list, ridge=None) -> dict:
    """A1 for one frame: {score, dist_px, support, n_ridge, n_sampled}; None fields when undefined."""
    import cv2
    from court.snap_track import PEAK_MIN, _bilinear

    h, w = frame_bgr.shape[:2]
    H = np.asarray(H_px2ft, np.float64).reshape(3, 3)
    try:
        P = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}
    if ridge is None:
        ridge = ridge_field(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    if ridge is None:
        return {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}
    R = (ridge > PEAK_MIN).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        R[max(0, y1 - BOX_PAD):y2 + BOX_PAD, max(0, x1 - BOX_PAD):x2 + BOX_PAD] = 0
    n_ridge = int(R.sum())
    pa, pb = project(P, CH_A), project(P, CH_B)
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    ok &= (np.abs(pa) < 8 * max(w, h)).all(1) & (np.abs(pb) < 8 * max(w, h)).all(1)
    T = np.full((h, w), 255, np.uint8)
    for a, b in zip(pa[ok], pb[ok]):
        cv2.line(T, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 0, 1)
    if n_ridge < MIN_RIDGE_PX or (T == 0).sum() == 0:
        return {"score": None, "dist_px": None, "support": None, "n_ridge": n_ridge, "n_sampled": 0}
    dt = cv2.distanceTransform(T, cv2.DIST_L2, 3)
    d = dt[R > 0]
    # template-side support: on-frame samples outside boxes with a ridge peak within LINE_SUPPORT_PX
    mid = (pa + pb) / 2.0
    on = ok & (mid[:, 0] >= 1) & (mid[:, 0] < w - 1) & (mid[:, 1] >= 1) & (mid[:, 1] < h - 1)
    on &= ~_inside_boxes(mid, boxes)
    tang = pb - pa
    tlen = np.linalg.norm(tang, axis=1)
    on &= tlen > 1e-3
    idx = np.where(on)[0]
    support = None
    if len(idx) >= MIN_SAMPLES:
        tg = tang[idx] / tlen[idx, None]
        nrm = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
        Tn = np.arange(-LINE_SUPPORT_PX, LINE_SUPPORT_PX + 1, dtype=np.float32)
        resp = _bilinear(ridge, mid[idx][:, None, :] + Tn[None, :, None] * nrm[:, None, :])
        support = round(float((resp >= PEAK_MIN).any(axis=1).mean()), 4)
    return {"score": round(float((d <= NEAR_PX).mean()), 4),
            "dist_px": round(float(np.median(d)), 2),
            "support": support, "n_ridge": n_ridge, "n_sampled": int(len(idx))}


def score_build(video_path, sidecar: dict, checks=("overlay",)) -> list[dict]:
    """Run the per-frame checks over one build. Reads the video sequentially (never seeks)."""
    import cv2

    rows = []
    cap = cv2.VideoCapture(str(video_path))
    want = {r["frame"]: r for r in sidecar["frames"]}
    idx = 0
    while want:
        ret, frame = cap.read()
        if not ret:
            break
        r = want.pop(idx, None)
        if r is not None:
            out = {"frame": idx, "state": r["state"]}
            boxes = [b["bbox"] for b in r["boxes"]]
            if "overlay" in checks:
                out["overlay"] = line_overlay(frame, r["H"], boxes) if r["H"] is not None else {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}
            rows.append(out)
        idx += 1
    cap.release()
    for f in sorted(want):                      # frames the video ran out before reaching
        rows.append({"frame": f, "state": want[f]["state"], "overlay": {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}, "unread": True})
    return rows


def summarize(rows: list[dict], far: float = 40.0) -> dict:
    """Per-clip A1 summary over frames with an H: median dist_px, p90, median score and support,
    and share_far = share of frames whose lines sit more than `far` px from the template."""
    ov = [r["overlay"] for r in rows if r.get("overlay", {}).get("dist_px") is not None]
    dd = [o["dist_px"] for o in ov]; sc = [o["score"] for o in ov]; su = [o["support"] for o in ov if o["support"] is not None]
    has_h = sum(1 for r in rows if r["state"] != "LOST")
    return {"frames": len(rows), "frames_with_H": has_h, "frames_scored": len(ov),
            "dist_px_median": round(float(np.median(dd)), 2) if dd else None,
            "dist_px_p90": round(float(np.percentile(dd, 90)), 2) if dd else None,
            "score_median": round(float(np.median(sc)), 4) if sc else None,
            "support_median": round(float(np.median(su)), 4) if su else None,
            "share_far": round(sum(d > far for d in dd) / len(dd), 4) if dd else None}

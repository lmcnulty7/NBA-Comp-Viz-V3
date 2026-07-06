"""
court/refine.py — sharpen a homography by aligning court LINES to image edges.

The keypoint model gives a rough homography from sparse intersections (4–8 points,
often noisy). This refines it with the court's line structure, which is far richer
and more occlusion-tolerant:

  1. Isolate court-line pixels in the image (white top-hat — thin bright lines on the
     wood floor; suppresses large bright regions like jerseys).
  2. Using the rough H, project each known straight court line into the image — this
     tells us WHERE each line should be, which solves the line↔court correspondence
     for free.
  3. For points sampled along each projected line, snap to the nearest court-line
     pixel along the line's normal → a point correspondence (exact court coord ↔
     observed edge pixel).
  4. Re-solve H (RANSAC) from the original keypoints PLUS these line samples. Iterate
     a few times (ICP-style).

Confidence stays KEYPOINT-based (so masking trust is unchanged), and a guard reverts
to the rough H if refinement worsens keypoint agreement (line outliers, bad frame).
Refinement can sharpen a roughly-right H a lot; it cannot rescue a totally-wrong one
(no nearby lines to snap to) — those need a better keypoint model / more data.
"""
from __future__ import annotations

import numpy as np

import config
from .geometry import COURT_KEYPOINTS, HALF_LENGTH, HALF_WIDTH, court_line_segments
from .homography import CourtHomography


_SEGMENTER = None


def court_line_mask(frame) -> np.ndarray:
    """Binary mask of court-line pixels. Uses the learned segmentation model if its
    weights exist (much cleaner on low-contrast wood); otherwise falls back to a
    white top-hat filter."""
    global _SEGMENTER
    if config.LINE_SEG_WEIGHTS.exists():
        if _SEGMENTER is None:
            from .line_seg import CourtLineSegmenter
            _SEGMENTER = CourtLineSegmenter()
        return _SEGMENTER.mask(frame)
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (config.COURT_LINE_TOPHAT_K,) * 2)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    _, m = cv2.threshold(tophat, config.COURT_LINE_THRESH, 255, cv2.THRESH_BINARY)
    return m


def _boundary_px(hom, margin_ft):
    L, W = HALF_LENGTH + margin_ft, HALF_WIDTH + margin_ft
    px = hom.to_pixel_batch(np.array([[-L, -W], [L, -W], [L, W], [-L, W]], np.float32))
    return px if np.isfinite(px).all() else None


def _line_matches(mask, hom, search_px, step_ft):
    """For each court line: project, sample, and snap to nearest line pixel along the
    normal. Returns (image_pts, court_pts) correspondences."""
    Himg, Wimg = mask.shape
    img_pts, court_pts = [], []
    for P1, P2 in court_line_segments():
        e = hom.to_pixel_batch(np.array([P1, P2], np.float32))
        if not np.isfinite(e).all():
            continue
        d = e[1] - e[0]
        nrm = float(np.linalg.norm(d))
        if nrm < 2:
            continue
        normal = np.array([-d[1], d[0]]) / nrm
        nsamp = max(2, int(np.linalg.norm(P2 - P1) / step_ft))
        ts = np.linspace(0, 1, nsamp)[:, None]
        court_samples = P1[None] * (1 - ts) + P2[None] * ts
        proj = hom.to_pixel_batch(court_samples.astype(np.float32))
        for c, q in zip(court_samples, proj):
            if not np.isfinite(q).all():
                continue
            qx, qy = float(q[0]), float(q[1])
            if not (0 <= qx < Wimg and 0 <= qy < Himg):
                continue
            hit = None
            for r in range(0, search_px + 1):
                for s in ((0,) if r == 0 else (-r, r)):
                    sx = int(round(qx + normal[0] * s))
                    sy = int(round(qy + normal[1] * s))
                    if 0 <= sx < Wimg and 0 <= sy < Himg and mask[sy, sx] > 0:
                        hit = (sx, sy)
                        break
                if hit:
                    break
            if hit:
                img_pts.append([hit[0], hit[1]])
                court_pts.append([float(c[0]), float(c[1])])
    return np.array(img_pts, np.float32), np.array(court_pts, np.float32)


def _line_residual(hom, dt):
    """Mean distance (px) from projected court-line samples to the nearest line pixel."""
    Himg, Wimg = dt.shape
    ds = []
    for P1, P2 in court_line_segments():
        n = max(2, int(np.linalg.norm(P2 - P1) / 2))
        ts = np.linspace(0, 1, n)[:, None]
        for q in hom.to_pixel_batch((P1[None] * (1 - ts) + P2[None] * ts).astype(np.float32)):
            if np.isfinite(q).all() and 0 <= q[0] < Wimg and 0 <= q[1] < Himg:
                ds.append(dt[int(q[1]), int(q[0])])
    return float(np.mean(ds)) if ds else float("inf")


def _set_keypoint_confidence(hom, kp_img, kp_court):
    """Set n_inliers/quality from KEYPOINT agreement only (keeps masking trust
    keypoint-based; line samples otherwise trivially inflate it)."""
    proj = hom.to_court_batch(kp_img)
    err = np.linalg.norm(proj - kp_court, axis=1)
    inl = err < config.RANSAC_REPROJ_THRESHOLD
    hom.n_inliers = int(inl.sum())
    hom.quality = float(err[inl].mean()) if inl.any() else None


def refine_homography(frame, rough, detected):
    """Return a line-refined CourtHomography (or `rough` if refinement can't help)."""
    import cv2

    names = [k for k in detected if k in COURT_KEYPOINTS]
    if len(names) < config.MIN_KEYPOINTS_FOR_H:
        return rough
    kp_img = np.array([detected[n] for n in names], np.float32)
    kp_court = np.array([COURT_KEYPOINTS[n] for n in names], np.float32)
    base_q = rough.quality

    mask = court_line_mask(frame)
    poly = _boundary_px(rough, config.COURT_MARGIN_FT)        # ignore edges outside the rough court
    if poly is not None:
        region = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(region, [poly.astype(np.int32)], 255)
        mask = cv2.bitwise_and(mask, region)
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)    # px to nearest court-line pixel

    cur = rough
    for _ in range(config.COURT_REFINE_ITERS):
        li, lc = _line_matches(mask, cur, config.COURT_REFINE_SEARCH_PX, config.COURT_REFINE_STEP_FT)
        if len(li) < 8:                                       # not enough line support to refine
            break
        new = CourtHomography()
        if not new.compute(np.vstack([kp_img, li]), np.vstack([kp_court, lc])):
            break
        cur = new

    if cur is rough:
        return rough
    # MONOTONIC guard: accept refinement ONLY if it (a) clearly improves court-line
    # alignment AND (b) doesn't wreck keypoint agreement. Otherwise keep rough → never worse.
    _set_keypoint_confidence(cur, kp_img, kp_court)
    line_improved = _line_residual(cur, dt) < _line_residual(rough, dt) - 0.3
    kp_ok = cur.quality is not None and (base_q is None or cur.quality <= max(1.0, base_q * 2.0))
    return cur if (line_improved and kp_ok) else rough

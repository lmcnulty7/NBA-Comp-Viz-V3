#!/usr/bin/env python
"""
snap_projections.py — Step 3: line-snap refinement of the human-verified homographies.

Input: the frames the user marked `lineup` in projection_verdicts.tsv. Each already
has a globally-correct H (fit from the code-2 download points, post swap-fix). This
script polishes each H so the projected court sits on the painted lines to ~sub-pixel
accuracy, ICP-style:

  iterate: sample the template lines/arcs → search along each sample's NORMAL for the
  strongest ridge response (sub-pixel parabola peak) → robust RANSAC refit → repeat.

Occlusion handling (why crowds/players can't hurt it):
  • the ridge mask is density-gated — crowd texture is zeroed, so fully occluded
    lines (near sideline under the crowd) produce NO correspondences at all; their
    position comes from the global H, which the visible lines fully determine;
  • samples with no strong ridge peak within the search window are dropped;
  • players/logos ON a line give sparse wrong matches — RANSAC outliers.

Accept metric — matched-offset residual: median distance from projected template
samples to their matched line peaks (fixed search radius, before vs after). The
global all-samples DT median does NOT work here: it is dominated by samples on
occluded lines (players, crowd, score graphics) and cannot see the improvement.

A snap is ACCEPTED only if ALL guards pass:
  • matched-offset residual strictly improves AND the match count doesn't drop
    >20% (the H can't cheat by aligning fewer samples better);
  • ≥ MIN_MATCH correspondences whose bounding box covers a decent frame fraction
    (a clustered match set can't be allowed to steer a global H);
  • the code-2 anchors' reprojection didn't blow up (catches along-line sliding);
  • no frame corner moved more than MAX_CORNER_PX (the true correction is small —
    the input H is already human-verified as globally correct).
Otherwise the frame KEEPS its original verified H (status KEPT).

Outputs:
  data/court_review/snapped_H.npz   — per frame: P (court-ft → px), status, residuals
  data/court_review/snap_rank.tsv   — worst final residual first
  data/court_review/snap_review/    — overlay renders, worst first (spot-check the tail)
Usage:
  /opt/anaconda3/bin/python snap_projections.py
"""
from __future__ import annotations
import os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, court33_segments, court33_curves

cv2.setRNGSeed(42)
ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
VERDICTS = os.path.join(ROOT, "projection_verdicts.tsv")
OUT_DIR = os.path.join(ROOT, "snap_review")
TSV = os.path.join(ROOT, "snap_rank.tsv")
NPZ = os.path.join(ROOT, "snapped_H.npz")

STEP_FT = 1.0            # template sampling density
RADII = (12, 10, 8, 6)   # normal-search radius per iteration (px), coarse → fine
EVAL_R = 8               # fixed search radius for the before/after accept metric
PEAK_MIN = 12.0          # ridge response a peak must reach to count as a match
RANSAC_PX = 3.0          # refit inlier threshold (image px)
MIN_MATCH = 40           # accept snap only with at least this many correspondences
MIN_SPREAD = 0.08        # ...whose bbox covers at least this fraction of the frame
MAX_CORNER_PX = 40.0     # max allowed corner displacement between H0 and snapped H
ANCHOR_SLACK = 5.0       # anchors may get at most this much worse (px, vs max(before, 3))


# ── template chords: midpoints + endpoint pairs (for tangents), built once ──────
def build_chords():
    chords = []
    for P1, P2 in court33_segments():
        P1, P2 = np.asarray(P1, np.float64), np.asarray(P2, np.float64)
        n = max(1, int(np.linalg.norm(P2 - P1) / STEP_FT))
        ts = np.linspace(0, 1, n + 1)
        for a, b in zip(ts[:-1], ts[1:]):
            chords.append((P1 + a * (P2 - P1), P1 + b * (P2 - P1)))
    for poly in court33_curves(step_deg=3.0):
        for a, b in zip(poly[:-1], poly[1:]):
            chords.append((np.asarray(a, np.float64), np.asarray(b, np.float64)))
    A = np.array([c[0] for c in chords], np.float32)
    B = np.array([c[1] for c in chords], np.float32)
    return A, B, (A + B) / 2.0                       # endpoints + midpoints, court ft


CH_A, CH_B, CH_MID = build_chords()


def load_code2(lp):
    pts = {}
    for line in open(lp):
        t = line.split()
        if len(t) >= 4 and int(t[3]) == 2:
            pts[int(t[0])] = (float(t[1]), float(t[2]))
    return pts


def fit_P0(c2, w, h):
    """Initial court-ft → pixel homography from the code-2 anchors (same fit as the ranker)."""
    if len(c2) < 4:
        return None
    idxs = list(c2)
    src = np.array([c2[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    dst = np.array([COURT_VERTICES_33[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    method = 0 if len(idxs) == 4 else cv2.RANSAC
    H, _ = cv2.findHomography(src, dst, method, 3.0)   # norm img → ft
    if H is None:
        return None
    S = np.array([[1.0 / w, 0, 0], [0, 1.0 / h, 0], [0, 0, 1]], np.float64)
    try:
        return np.linalg.inv(H.astype(np.float64) @ S)
    except np.linalg.LinAlgError:
        return None


def ridge_field(gray):
    """Density-gated ridge response (float). Same evidence family as the ranker:
    top-hat pair (bright AND dark lines), median-blur denoise, crowd-texture gate."""
    g = cv2.medianBlur(gray, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    ridge = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                    cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)).astype(np.float32)
    mask = (ridge > 10).astype(np.uint8)
    density = cv2.boxFilter(mask.astype(np.float32), -1, (31, 31))
    ridge[density > 0.20] = 0.0
    return ridge if (ridge > PEAK_MIN).any() else None


def project(P, pts_ft):
    return cv2.perspectiveTransform(np.asarray(pts_ft, np.float32).reshape(-1, 1, 2),
                                    P.astype(np.float64)).reshape(-1, 2)


def bilinear(imgf, xy):
    """Sample imgf at float (x, y) positions; out-of-bounds → 0. xy shape (..., 2)."""
    h, w = imgf.shape
    x, y = xy[..., 0], xy[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    fx, fy = x - x0c, y - y0c
    v = (imgf[y0c, x0c] * (1 - fx) * (1 - fy) + imgf[y0c, x0c + 1] * fx * (1 - fy)
         + imgf[y0c + 1, x0c] * (1 - fx) * fy + imgf[y0c + 1, x0c + 1] * fx * fy)
    v[~ok] = 0.0
    return v


def match_lines(P, ridge, w, h, radius):
    """Correspondences (court-ft midpoint → sub-pixel line peak in px) via normal search."""
    pa, pb = project(P, CH_A), project(P, CH_B)
    mid = (pa + pb) / 2.0
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    ok &= (mid[:, 0] >= 1) & (mid[:, 0] < w - 1) & (mid[:, 1] >= 1) & (mid[:, 1] < h - 1)
    tang = pb - pa
    tlen = np.linalg.norm(tang, axis=1)
    ok &= tlen > 1e-3
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    tang = tang[idx] / tlen[idx, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)          # unit normal in image
    T = np.arange(-radius, radius + 1, dtype=np.float32)
    grid = mid[idx][:, None, :] + T[None, :, None] * nrm[:, None, :]   # (n, 2r+1, 2)
    resp = bilinear(ridge, grid)                                        # (n, 2r+1)
    k = resp.argmax(axis=1)
    peak = resp[np.arange(len(k)), k]
    inner = (k > 0) & (k < len(T) - 1) & (peak >= PEAK_MIN)             # peak inside window
    if not inner.any():
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    idx, k, nrm = idx[inner], k[inner], nrm[inner]
    r0 = resp[inner, :]
    left = r0[np.arange(len(k)), k - 1]
    cen = r0[np.arange(len(k)), k]
    right = r0[np.arange(len(k)), k + 1]
    den = left - 2 * cen + right
    delta = np.where(den < -1e-6, 0.5 * (left - right) / den, 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    t_star = T[k] + delta
    px = mid[idx] + t_star[:, None] * nrm
    return CH_MID[idx], px.astype(np.float32)


def eval_res(P, ridge, w, h):
    """Matched-offset residual: median distance from projected template samples to
    their matched line peaks at a FIXED radius (fair before/after comparison).
    Returns (median_px or None, n_matches)."""
    ft, px = match_lines(P, ridge, w, h, EVAL_R)
    if len(ft) == 0:
        return None, 0
    d = np.linalg.norm(px - project(P, ft), axis=1)
    return float(np.median(d)), len(ft)


def render(img, P, c2, hud):
    h, w = img.shape[:2]
    for P1, P2 in court33_segments():
        seg = project(P, np.stack([P1, P2]))
        if np.isfinite(seg).all():
            cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    for poly in court33_curves():
        px = project(P, poly)
        for a, b in zip(px[:-1], px[1:]):
            if (np.isfinite(a).all() and np.isfinite(b).all()
                    and max(abs(a[0]), abs(b[0])) < 4 * w and max(abs(a[1]), abs(b[1])) < 4 * h):
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    for i, (nx, ny) in c2.items():
        cv2.circle(img, (int(nx * w), int(ny * h)), 5, (0, 255, 0), -1)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import glob
    for old in glob.glob(os.path.join(OUT_DIR, "*.jpg")):
        os.remove(old)

    stems = []
    with open(VERDICTS) as f:
        next(f)
        for line in f:
            t = line.rstrip("\n").split("\t")
            if len(t) >= 2 and t[1] == "lineup":
                stems.append(t[0])
    stems.sort()
    print(f"snapping {len(stems)} human-verified lineup frames …")

    results = []          # (stem, P, status, res0, res1, n_match, spread)
    n_snap = n_keep = 0
    for n, stem in enumerate(stems):
        if n % 200 == 0:
            print(f"  {n}/{len(stems)} …")
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        c2 = load_code2(os.path.join(LBL_DIR, stem + ".txt"))
        P0 = fit_P0(c2, w, h)
        if P0 is None:
            continue
        ridge = ridge_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ridge is None:
            results.append((stem, P0, "KEPT-NOEVID", None, None, 0, 0.0))
            n_keep += 1
            continue
        res0, n0 = eval_res(P0, ridge, w, h)

        anchors_ft = np.array([COURT_VERTICES_33[i] for i in c2], np.float32)
        anchors_px = np.array([(x * w, y * h) for x, y in c2.values()], np.float32)

        P, spread = P0.copy(), 0.0
        for radius in RADII:
            ft, px = match_lines(P, ridge, w, h, radius)
            if len(ft) < MIN_MATCH:
                break
            src = np.concatenate([ft, anchors_ft]).reshape(-1, 1, 2)
            dst = np.concatenate([px, anchors_px]).reshape(-1, 1, 2)
            Pn, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
            if Pn is None:
                break
            P = Pn.astype(np.float64)
            lo, hi = px.min(0), px.max(0)
            spread = float((hi[0] - lo[0]) * (hi[1] - lo[1]) / (w * h))

        res1, n_match = eval_res(P, ridge, w, h)

        # ── guards ────────────────────────────────────────────────────────────
        ok = res0 is not None and res1 is not None and res1 < res0
        ok = ok and n_match >= max(MIN_MATCH, int(0.8 * n0)) and spread >= MIN_SPREAD
        if ok:
            a0 = np.median(np.linalg.norm(project(P0, anchors_ft) - anchors_px, axis=1))
            a1 = np.median(np.linalg.norm(project(P, anchors_ft) - anchors_px, axis=1))
            ok = a1 <= max(a0, 3.0) + ANCHOR_SLACK
        if ok:
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                             (P @ np.linalg.inv(P0))).reshape(-1, 2)
            ok = np.isfinite(moved).all() and float(np.linalg.norm(moved - corners, axis=1).max()) <= MAX_CORNER_PX

        if ok:
            results.append((stem, P, "SNAPPED", res0, res1, n_match, spread))
            n_snap += 1
        else:
            results.append((stem, P0, "KEPT", res0, res0, n_match, spread))
            n_keep += 1

    # ── outputs: worst final residual first (no-evidence frames at the top) ──
    results.sort(key=lambda r: -(r[4] if r[4] is not None else 99.0))
    with open(TSV, "w") as f:
        f.write("rank\tfile\tframe\tstatus\tres_before\tres_after\tn_match\tspread\n")
        for rank, (stem, P, status, r0, r1, nm, sp) in enumerate(results):
            fn = f"{rank:04d}_{stem}.jpg"
            img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
            if img is not None:
                c2 = load_code2(os.path.join(LBL_DIR, stem + ".txt"))
                b = f"{r0:.2f}" if r0 is not None else "-"
                a = f"{r1:.2f}" if r1 is not None else "-"
                hud = f"#{rank}  {status}  line-res {b} -> {a} px  matches={nm}  {stem[:30]}"
                cv2.imwrite(os.path.join(OUT_DIR, fn), render(img, P, c2, hud))
            f.write(f"{rank}\t{fn}\t{stem}\t{status}\t{r0 if r0 is not None else 'nan'}\t"
                    f"{r1 if r1 is not None else 'nan'}\t{nm}\t{sp:.3f}\n")

    np.savez(NPZ,
             stems=np.array([r[0] for r in results]),
             P=np.stack([r[1] for r in results]),
             status=np.array([r[2] for r in results]),
             res_before=np.array([r[3] if r[3] is not None else np.nan for r in results]),
             res_after=np.array([r[4] if r[4] is not None else np.nan for r in results]))

    snapped = [r for r in results if r[2] == "SNAPPED"]
    if snapped:
        b = np.array([r[3] for r in snapped]); a = np.array([r[4] for r in snapped])
        print(f"\nSNAPPED {n_snap}   KEPT (guards fired) {n_keep}")
        print(f"residual before: p50={np.median(b):.2f}px  p90={np.percentile(b, 90):.2f}px")
        print(f"residual after:  p50={np.median(a):.2f}px  p90={np.percentile(a, 90):.2f}px")
    print(f"H matrices -> {NPZ}\nrank TSV -> {TSV}\nrenders -> {OUT_DIR}/ (0000 = worst)")


if __name__ == "__main__":
    main()

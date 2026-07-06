#!/usr/bin/env python
"""
triage_court_labels.py — geometry-based triage of the densified court labels.

For every frame we fit a homography (normalized image coords → court feet) ROBUSTLY
with RANSAC over ALL its points, then reproject each point to where the court's rigid
geometry says it should land. A point whose reprojection is far from its labeled spot
is geometrically inconsistent — a likely bad label. Orange (code 3, unverified fill)
points are the ones we care about; green (2) outliers are surfaced too (bad seeds).

The court is rigid, so this catches placement errors for FREE — no model, no guessing.
Frames are ranked by how many suspect orange points they contain, so manual review
(and the Fable spot-check) starts where the fills are most likely wrong.

Outputs:
  • data/court_review/triage.tsv   — one row per frame, ranked worst-first
  • data/court_review/triage_viz/  — annotated images for the top-N worst frames
                                     (green=ok, RED=suspect, ring=orange fill)

Usage:
  python triage_court_labels.py                 # full run, default thresholds
  python triage_court_labels.py --thr-ft 4 --viz 60
"""
from __future__ import annotations
import argparse, glob, os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33

ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
VIZ_DIR = os.path.join(ROOT, "triage_viz")
TSV = os.path.join(ROOT, "triage.tsv")


def load_label(lp):
    pts = {}
    for line in open(lp):
        t = line.split()
        if len(t) >= 4:
            pts[int(t[0])] = (float(t[1]), float(t[2]), int(t[3]))
    return pts


def residuals(pts, ransac_ft):
    """Fit H (norm img -> court ft) with RANSAC over all points; return per-index
    reprojection error in feet + median inlier error. None if too few / degenerate."""
    idxs = list(pts)
    if len(idxs) < 6:
        return None, None
    src = np.array([[pts[i][0], pts[i][1]] for i in idxs], np.float32).reshape(-1, 1, 2)
    dst = np.array([COURT_VERTICES_33[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_ft)
    if H is None:
        return None, None
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = {i: float(np.linalg.norm(proj[k] - COURT_VERTICES_33[i])) for k, i in enumerate(idxs)}
    inl = mask.ravel().astype(bool)
    med = float(np.median([np.linalg.norm(proj[k] - dst[k, 0]) for k in range(len(idxs)) if inl[k]])) if inl.any() else None
    return err, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr-ft", type=float, default=4.0, help="reproj error (ft) above which a point is suspect")
    ap.add_argument("--ransac-ft", type=float, default=3.0, help="RANSAC inlier threshold (ft)")
    ap.add_argument("--viz", type=int, default=50, help="render annotated images for the top-N worst frames")
    args = ap.parse_args()

    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(LBL_DIR, "*.txt")))]
    rows = []
    n_bad_geom = 0
    for stem in stems:
        pts = load_label(os.path.join(LBL_DIR, stem + ".txt"))
        n_orange = sum(1 for v in pts.values() if v[2] == 3)
        err, med = residuals(pts, args.ransac_ft)
        if err is None or med is None or med > args.thr_ft:
            # can't trust the fit itself → can't triage this frame geometrically
            n_bad_geom += 1
            rows.append((stem, n_orange, -1, -1, med if med else float("nan"), []))
            continue
        suspect = [(i, err[i]) for i in err if err[i] > args.thr_ft]
        susp_orange = [(i, e) for i, e in suspect if pts[i][2] == 3]
        rows.append((stem, n_orange, len(susp_orange), len(suspect), med, sorted(suspect, key=lambda x: -x[1])))

    # rank: frames with a trustworthy fit first, worst (most suspect orange) on top
    rows.sort(key=lambda r: (r[2] < 0, -(r[2] if r[2] >= 0 else 0), -(r[3] if r[3] >= 0 else 0)))

    with open(TSV, "w") as f:
        f.write("frame\tn_orange\tsuspect_orange\tsuspect_total\tmedian_reproj_ft\tsuspect_idx:err\n")
        for stem, no, so, st, med, susp in rows:
            s = " ".join(f"{i}:{e:.1f}" for i, e in susp)
            f.write(f"{stem}\t{no}\t{so}\t{st}\t{med:.2f}\t{s}\n")

    trustworthy = [r for r in rows if r[2] >= 0]
    with_susp = [r for r in trustworthy if r[2] > 0]
    print(f"frames: {len(rows)}  |  geometrically-triageable: {len(trustworthy)}  |  un-triageable fit: {n_bad_geom}")
    print(f"frames with >=1 suspect orange point: {len(with_susp)}")
    print(f"total suspect orange points: {sum(r[2] for r in trustworthy)}")
    print(f"ranked TSV -> {TSV}")

    # annotate the worst frames for eyeballing (and the Fable spot-check)
    os.makedirs(VIZ_DIR, exist_ok=True)
    for rank, (stem, no, so, st, med, susp) in enumerate(with_susp[:args.viz]):
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = load_label(os.path.join(LBL_DIR, stem + ".txt"))
        bad = {i for i, _ in susp}
        for i, (nx, ny, code) in pts.items():
            cx, cy = int(nx * w), int(ny * h)
            col = (0, 0, 255) if i in bad else (0, 255, 0)          # red=suspect, green=consistent
            if code == 3:                                          # orange fill → hollow ring
                cv2.circle(img, (cx, cy), 6, col, 2)
            else:
                cv2.circle(img, (cx, cy), 5, col, -1)
            cv2.putText(img, str(i), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(i), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"{stem}  suspect_orange={so}  median_reproj={med:.1f}ft", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{stem}  suspect_orange={so}  median_reproj={med:.1f}ft", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(VIZ_DIR, f"{rank:04d}_susp{so:02d}_{stem}.jpg"), img)   # rank 0000 = worst
    print(f"annotated all {min(args.viz, len(with_susp))} flagged frames (worst-first) -> {VIZ_DIR}/")


if __name__ == "__main__":
    main()

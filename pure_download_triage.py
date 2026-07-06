#!/usr/bin/env python
"""
pure_download_triage.py — homography check + full-court projection using ONLY code-2
points (the downloaded NBA Court v4 human labels). Zero model fills, zero edits.

Frames with < 4 code-2 points are EXCLUDED (can't fit a homography) and counted.
For each used frame we fit H from code-2 points, measure per-point reprojection error,
and render the full projected court so the raw downloaded labels can be eyeballed.

Note: a frame with exactly 4 code-2 points fits H with zero redundancy -> reproj error
is trivially ~0 and CANNOT reveal a bad point. Only frames with >=5 code-2 points get a
meaningful internal-consistency check; all frames still get the projection overlay, which
is judged against the real painted lines.

Outputs:
  • data/court_review/pure_download.tsv    — per-used-frame stats, worst-first
  • data/court_review/pure_download_viz/   — projection overlays, worst-first
"""
from __future__ import annotations
import glob, os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, court33_segments

ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
VIZ_DIR = os.path.join(ROOT, "pure_download_viz")
TSV = os.path.join(ROOT, "pure_download.tsv")
THR_FT = 4.0
RANSAC_FT = 3.0
VIZ_N = 400


def load_code2(lp):
    pts = {}
    for line in open(lp):
        t = line.split()
        if len(t) >= 4 and int(t[3]) == 2:          # code 2 only = downloaded human labels
            pts[int(t[0])] = (float(t[1]), float(t[2]))
    return pts


def main():
    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(LBL_DIR, "*.txt")))]
    total = len(stems)
    excluded, used = [], []
    counts = []
    for stem in stems:
        pts = load_code2(os.path.join(LBL_DIR, stem + ".txt"))
        counts.append(len(pts))
        if len(pts) < 4:
            excluded.append(stem)
            continue
        idxs = list(pts)
        src = np.array([pts[i] for i in idxs], np.float32).reshape(-1, 1, 2)   # normalized img coords
        dst = np.array([COURT_VERTICES_33[i] for i in idxs], np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_FT)              # thr is in dst units = court ft
        if H is None:
            excluded.append(stem)          # degenerate (e.g. collinear) -> can't use
            continue
        proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        err = {idxs[k]: float(np.linalg.norm(proj[k] - COURT_VERTICES_33[idxs[k]])) for k in range(len(idxs))}
        med = float(np.median(list(err.values())))
        suspect = sorted([(i, e) for i, e in err.items() if e > THR_FT], key=lambda x: -x[1])
        used.append((stem, len(pts), len(suspect), med, H, suspect))

    # rank worst-first: most suspect code-2 points, then highest median error
    used.sort(key=lambda r: (-r[2], -r[3]))

    with open(TSV, "w") as f:
        f.write("frame\tn_code2\tsuspect_code2\tmedian_reproj_ft\tsuspect_idx:err\n")
        for stem, n2, ns, med, H, susp in used:
            f.write(f"{stem}\t{n2}\t{ns}\t{med:.2f}\t{' '.join(f'{i}:{e:.1f}' for i, e in susp)}\n")

    # ── report ────────────────────────────────────────────────────────────────
    counts = np.array(counts)
    n_excl, n_used = len(excluded), len(used)
    exactly4 = sum(1 for u in used if u[1] == 4)
    ge5 = n_used - exactly4
    print(f"TOTAL frames:            {total}")
    print(f"EXCLUDED (<4 code-2):    {n_excl}  ({100*n_excl/total:.1f}%)")
    print(f"USED (>=4 code-2):       {n_used}  ({100*n_used/total:.1f}%)")
    print(f"  of used, exactly 4 (no internal check possible): {exactly4}  ({100*exactly4/total:.1f}% of all)")
    print(f"  of used, >=5 (meaningful consistency check):     {ge5}  ({100*ge5/total:.1f}% of all)")
    print(f"code-2 count per frame: min={counts.min()} median={int(np.median(counts))} "
          f"mean={counts.mean():.1f} max={counts.max()}")
    hist = {c: int((counts == c).sum()) for c in range(0, 8)}
    print(f"frames by code-2 count 0..7: {hist}   (>=8: {int((counts>=8).sum())})")
    with_susp = [u for u in used if u[2] > 0]
    print(f"used frames with >=1 internally-suspect code-2 point (>{THR_FT}ft, needs >=5 pts): {len(with_susp)}")
    print(f"ranked TSV -> {TSV}")

    # ── projection overlays, worst-first ───────────────────────────────────────
    os.makedirs(VIZ_DIR, exist_ok=True)
    for rank, (stem, n2, ns, med, H, susp) in enumerate(used[:VIZ_N]):
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        # H was fit on normalized coords; convert to pixel-domain by pre-scaling input
        S = np.array([[1.0 / w, 0, 0], [0, 1.0 / h, 0], [0, 0, 1]], np.float32)
        Hpx = H @ S                       # pixel img -> court ft
        try:
            Hinv = np.linalg.inv(Hpx)     # court ft -> pixel img
        except np.linalg.LinAlgError:
            continue                       # degenerate H (collinear pts) -> skip overlay
        for P1, P2 in court33_segments():
            seg = cv2.perspectiveTransform(np.array([[P1], [P2]], np.float32), Hinv).reshape(-1, 2)
            if np.isfinite(seg).all():
                cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
        proj = cv2.perspectiveTransform(COURT_VERTICES_33.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 2)
        bad = {i for i, _ in susp}
        for i, (px, py) in enumerate(proj):
            if np.isfinite([px, py]).all():
                cv2.drawMarker(img, (int(px), int(py)), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        for i, (nx, ny) in load_code2(os.path.join(LBL_DIR, stem + ".txt")).items():
            col = (0, 0, 255) if i in bad else (0, 255, 0)      # red = internally-suspect code-2 dot
            cv2.circle(img, (int(nx * w), int(ny * h)), 4, col, -1)
        cv2.putText(img, f"{stem}  code2={n2}  suspect={ns}  med={med:.1f}ft   magenta=projected court (pure download)",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(VIZ_DIR, f"{rank:04d}_susp{ns:02d}_n{n2:02d}_{stem}.jpg"), img)
    print(f"projection overlays (worst-first, top {min(VIZ_N, n_used)}) -> {VIZ_DIR}/")


if __name__ == "__main__":
    main()

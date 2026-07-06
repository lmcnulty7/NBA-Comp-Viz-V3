#!/usr/bin/env python
"""
step2_sparse_frames.py — Step 2 of the homography-perfection pipeline: surface the
frames the pure-download fit can't handle, so the user can decide add-points vs drop.

Cohort A — <4 code-2 points (51 frames): no homography possible from downloaded data.
  We render each with everything available (green=download, orange ring=model fill) and,
  where the fills allow, a PROVISIONAL court projection (magenta) fit from all points —
  clearly model-assisted, for orientation only. User decides: add points / drop frame.

Cohort B — exactly 4 code-2 points (134 frames): H fits exactly (zero redundancy), so a
  bad point is undetectable from the downloads alone. Mechanical cross-check: the code-3
  fills came from the OLD MODEL (independent of the download), so we score each frame by
  how well its fills agree with the 4-point H. High disagreement = suspect H -> eyes first.
  Also flags ill-conditioned quads (4 points nearly collinear / tiny hull area).

Outputs:
  data/court_review/step2_sparse/   renders + step2_sparse.tsv   (cohort A)
  data/court_review/step2_fourpt/   renders + step2_fourpt.tsv   (cohort B, worst-first)
"""
from __future__ import annotations
import glob, os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, court33_segments

cv2.setRNGSeed(42)
ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
SPARSE_DIR = os.path.join(ROOT, "step2_sparse")
FOURPT_DIR = os.path.join(ROOT, "step2_fourpt")
FILL_THR = 4.0            # ft — fill counts as agreeing with H below this
HULL_MIN = 0.03           # normalized hull area below which a 4-pt quad is ill-conditioned


def load_label(lp):
    pts = {}
    for line in open(lp):
        t = line.split()
        if len(t) >= 4:
            pts[int(t[0])] = (float(t[1]), float(t[2]), int(t[3]))
    return pts


def fit_H(pts_xy_by_idx, ransac_ft=3.0):
    """pts_xy_by_idx: {idx: (nx, ny)} normalized. Returns H (norm img -> court ft) or None."""
    idxs = list(pts_xy_by_idx)
    if len(idxs) < 4:
        return None
    src = np.array([pts_xy_by_idx[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    dst = np.array([COURT_VERTICES_33[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    method = 0 if len(idxs) == 4 else cv2.RANSAC       # 4 pts = exact solve, RANSAC meaningless
    H, _ = cv2.findHomography(src, dst, method, ransac_ft)
    return H


def reproj_err(pts_xy_by_idx, H):
    """{idx: err_ft} for each point under H."""
    idxs = list(pts_xy_by_idx)
    if not idxs or H is None:
        return {}
    src = np.array([pts_xy_by_idx[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    return {i: float(np.linalg.norm(proj[k] - COURT_VERTICES_33[i])) for k, i in enumerate(idxs)}


def draw_overlay(stem, pts, H, hud, fill_err=None):
    img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
    if img is None:
        return None
    h, w = img.shape[:2]
    if H is not None:
        S = np.array([[1.0 / w, 0, 0], [0, 1.0 / h, 0], [0, 0, 1]], np.float32)
        try:
            Hinv = np.linalg.inv(H @ S)
            for P1, P2 in court33_segments():
                seg = cv2.perspectiveTransform(np.array([[P1], [P2]], np.float32), Hinv).reshape(-1, 2)
                if np.isfinite(seg).all():
                    cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (255, 0, 255), 1, cv2.LINE_AA)
        except np.linalg.LinAlgError:
            pass
    for i, (nx, ny, code) in pts.items():
        cx, cy = int(nx * w), int(ny * h)
        if code == 2:
            col = (0, 255, 0); cv2.circle(img, (cx, cy), 5, col, -1)                   # download: solid green
        elif code == 1:
            col = (0, 255, 255); cv2.circle(img, (cx, cy), 6, col, 2)                  # user-occluded: yellow ring
        else:
            bad = fill_err is not None and fill_err.get(i, 0.0) > FILL_THR
            col = (0, 0, 255) if bad else (0, 165, 255)                                # fill: ring, red if disagrees
            cv2.circle(img, (cx, cy), 6, col, 2)
        lbl = str(i)
        tw = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        tx = cx + 7 if cx + 7 + tw < w else cx - 7 - tw
        ty = min(max(cy, 15), h - 5)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(SPARSE_DIR, exist_ok=True)
    os.makedirs(FOURPT_DIR, exist_ok=True)
    cohort_a, cohort_b = [], []

    for lp in sorted(glob.glob(os.path.join(LBL_DIR, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        pts = load_label(lp)
        c2 = {i: (v[0], v[1]) for i, v in pts.items() if v[2] == 2}
        if len(c2) < 4:
            cohort_a.append((stem, pts, c2))
        elif len(c2) == 4:
            cohort_b.append((stem, pts, c2))

    # ── Cohort A: <4 downloaded points ────────────────────────────────────────
    with open(os.path.join(SPARSE_DIR, "step2_sparse.tsv"), "w") as f:
        f.write("frame\tn_code2\tn_fills\tprovisional_H\n")
        n_prov = 0
        for stem, pts, c2 in sorted(cohort_a, key=lambda r: len(r[2])):
            allpts = {i: (v[0], v[1]) for i, v in pts.items()}
            H = fit_H(allpts) if len(allpts) >= 4 else None
            if H is not None:
                n_prov += 1
            n_fill = sum(1 for v in pts.values() if v[2] == 3)
            hud = (f"{stem[:34]}  downloads={len(c2)}  fills={n_fill}  "
                   f"{'PROVISIONAL H (model-assisted)' if H is not None else 'NO H possible'}")
            img = draw_overlay(stem, pts, H, hud)
            if img is not None:
                cv2.imwrite(os.path.join(SPARSE_DIR, f"c2-{len(c2)}_{stem}.jpg"), img)
            f.write(f"{stem}\t{len(c2)}\t{n_fill}\t{'yes' if H is not None else 'no'}\n")
    print(f"Cohort A (<4 downloads): {len(cohort_a)} frames  |  provisional model-assisted H: {n_prov}  |  no H at all: {len(cohort_a) - n_prov}")

    # ── Cohort B: exactly 4 downloaded points ─────────────────────────────────
    # A disagreeing fill has two possible causes: bad H, or bad fill (the densify is known
    # to dump off-screen vertices onto the visible end). Discriminate: if vertex i belongs
    # OFF-frame under H, the fill for i is a dump artifact -> doesn't indict the H. Only
    # disagreement from fills whose vertex belongs ON-screen counts against the H.
    def vertex_onscreen(i, H, slack=0.05):
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return True                                   # conservative: count against H
        p = cv2.perspectiveTransform(COURT_VERTICES_33[i].reshape(1, 1, 2).astype(np.float32), Hinv).reshape(2)
        if not np.isfinite(p).all():
            return False
        return -slack <= p[0] <= 1 + slack and -slack <= p[1] <= 1 + slack

    rows = []
    for stem, pts, c2 in cohort_b:
        H = fit_H(c2)
        hull = cv2.convexHull(np.array(list(c2.values()), np.float32))
        area = float(cv2.contourArea(hull))
        fills = {i: (v[0], v[1]) for i, v in pts.items() if v[2] == 3}
        ferr = reproj_err(fills, H)
        bad_on = [i for i, e in ferr.items() if e > FILL_THR and H is not None and vertex_onscreen(i, H)]
        bad_off = [i for i, e in ferr.items() if e > FILL_THR and i not in bad_on]
        on_errs = [e for i, e in ferr.items() if H is not None and vertex_onscreen(i, H)]
        med_on = float(np.median(on_errs)) if on_errs else float("nan")
        rows.append((stem, pts, H, ferr, med_on, len(bad_on), len(bad_off), len(ferr), area))

    # worst-first by ON-SCREEN disagreement (the signal that indicts the H)
    rows.sort(key=lambda r: (-r[5], -(r[4] if np.isfinite(r[4]) else -1), r[8]))
    with open(os.path.join(FOURPT_DIR, "step2_fourpt.tsv"), "w") as f:
        f.write("frame\tn_fills\tbad_onscreen\tbad_offscreen_dumps\tmedian_onscreen_err_ft\thull_area\n")
        for rank, (stem, pts, H, ferr, med_on, nbo, nboff, n_f, area) in enumerate(rows):
            hud = (f"{stem[:28]}  4 downloads  onscreen fills bad={nbo}  offscreen dumps={nboff}  "
                   f"med_on={med_on:.1f}ft")
            img = draw_overlay(stem, pts, H, hud, fill_err=ferr)
            if img is not None:
                cv2.imwrite(os.path.join(FOURPT_DIR, f"{rank:03d}_onbad{nbo:02d}_{stem}.jpg"), img)
            f.write(f"{stem}\t{n_f}\t{nbo}\t{nboff}\t{med_on:.2f}\t{area:.4f}\n")

    ok = sum(1 for r in rows if r[5] == 0)
    some = sum(1 for r in rows if 1 <= r[5] <= 2)
    bad = sum(1 for r in rows if r[5] > 2)
    dumps = sum(r[6] for r in rows)
    print(f"Cohort B (exactly 4): {len(rows)} frames")
    print(f"  no on-screen fill disagrees (H trustworthy):     {ok}")
    print(f"  1-2 on-screen fills disagree (likely fill noise): {some}")
    print(f"  >2 on-screen fills disagree (suspect H):          {bad}")
    print(f"  off-screen dump fills excluded from the verdict:  {dumps} (fill artifacts, not H errors)")
    meds = [r[4] for r in rows if np.isfinite(r[4])]
    if meds:
        m = np.array(meds)
        print(f"  median ON-SCREEN fill-err: p50={np.median(m):.2f}ft  p90={np.percentile(m, 90):.2f}ft")


if __name__ == "__main__":
    main()

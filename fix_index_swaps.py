#!/usr/bin/env python
"""
fix_index_swaps.py — Step 1 of the homography-perfection pipeline: correct the
index-number swaps in the downloaded (code-2) court labels.

Detection (deterministic, seeded RANSAC): a code-2 point whose reprojection error is
>20 ft under the frame's homography but which sits within 5 ft of a DIFFERENT court
vertex j — i.e. right spot, wrong number.

Each candidate is applied ONLY if it survives a guard: after relabeling i→j the frame
is refit, and we require (a) the moved point is now consistent (<4 ft) and (b) the
frame's median code-2 residual did not get worse. Anything ambiguous (target index
already occupied by another human point that is NOT its mirror swap, guard failure)
is SKIPPED and logged for manual review.

Before/after renders go to data/court_review/swap_verify/ for human verification.
A full log goes to data/court_review/swap_log.tsv. Labels were backed up to
data/court_review/labels_backup_preswap/ before any edit.
"""
from __future__ import annotations
import glob, os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, court33_segments

cv2.setRNGSeed(42)
ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
VERIFY_DIR = os.path.join(ROOT, "swap_verify")
LOG = os.path.join(ROOT, "swap_log.tsv")
BIG, NEAR, OK = 20.0, 5.0, 4.0


def load_label(lp):
    pts = {}
    for line in open(lp):
        t = line.split()
        if len(t) >= 4:
            pts[int(t[0])] = [float(t[1]), float(t[2]), int(t[3])]
    return pts


def save_label(lp, pts):
    with open(lp, "w") as f:
        for i in sorted(pts):
            x, y, s = pts[i]
            f.write(f"{i} {x:.6f} {y:.6f} {int(s)}\n")


def fit_code2(pts):
    """Fit H on the code-2 points; return (err_by_idx, median_err, H) or (None,)*3."""
    c2 = {i: v for i, v in pts.items() if v[2] == 2}
    if len(c2) < 5:
        return None, None, None
    idxs = list(c2)
    src = np.array([[c2[i][0], c2[i][1]] for i in idxs], np.float32).reshape(-1, 1, 2)
    dst = np.array([COURT_VERTICES_33[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, None, None
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = {idxs[k]: float(np.linalg.norm(proj[k] - COURT_VERTICES_33[idxs[k]])) for k in range(len(idxs))}
    return err, float(np.median(list(err.values()))), H


def court_pos(pt_xy, H):
    return cv2.perspectiveTransform(np.array([[pt_xy]], np.float32), H).reshape(2)


def render(stem, pts, H, mark, tag):
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
        if code != 2:
            continue
        cx, cy = int(nx * w), int(ny * h)
        col = (0, 0, 255) if i in mark else (0, 255, 0)
        cv2.circle(img, (cx, cy), 10 if i in mark else 4, col, 3 if i in mark else -1)
        lbl = str(i)
        tw = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        tx = cx + 8 if cx + 8 + tw < w else cx - 8 - tw
        ty = min(max(cy, 16), h - 6)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(img, tag, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, tag, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(VERIFY_DIR, exist_ok=True)
    rows = []
    n_applied = n_skipped = 0
    for lp in sorted(glob.glob(os.path.join(LBL_DIR, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        pts = load_label(lp)
        err, med0, H = fit_code2(pts)
        if err is None:
            continue
        cands = []
        for i, e in err.items():
            if e <= BIG:
                continue
            cp = court_pos([pts[i][0], pts[i][1]], H)
            d = np.linalg.norm(COURT_VERTICES_33 - cp, axis=1)
            j = int(np.argmin(d))
            if j != i and d[j] < NEAR:
                cands.append((i, j, e))
        if not cands:
            continue

        mod = {k: v[:] for k, v in pts.items()}
        moved, skipped_here = [], []
        for i, j, e in cands:
            tgt = mod.get(j)
            if tgt is None or tgt[2] == 3:
                # free slot (or displaces an unverified fill): relabel i -> j
                mod[j] = [mod[i][0], mod[i][1], 2]
                del mod[i]
                moved.append((i, j, e, "relabel"))
            elif tgt[2] == 2:
                # mutual swap? point j must in turn sit on vertex i
                cpj = court_pos([tgt[0], tgt[1]], H)
                if np.linalg.norm(cpj - COURT_VERTICES_33[i]) < NEAR:
                    mod[i], mod[j] = [tgt[0], tgt[1], 2], [mod[i][0], mod[i][1], 2]
                    moved.append((i, j, e, "mutual-swap"))
                else:
                    skipped_here.append((i, j, e, "target j occupied by non-mirror code-2"))
            else:
                skipped_here.append((i, j, e, "target j is user-reviewed (code 1)"))

        verdict = "SKIP"
        err1 = med1 = H1 = None
        if moved:
            err1, med1, H1 = fit_code2(mod)
            guard_ok = (
                err1 is not None
                and med1 <= med0 + 0.1
                and all(err1.get(j, 99) < OK for _, j, _, _ in moved)
            )
            if guard_ok:
                save_label(lp, mod)
                verdict = "APPLIED"
                n_applied += len(moved)
            else:
                skipped_here.extend((i, j, e, "guard failed (fit not improved)") for i, j, e, _ in moved)
                moved = []
        n_skipped += len(skipped_here)

        for i, j, e, kind in moved:
            rows.append((stem, i, j, e, kind, "APPLIED", med0, med1))
        for i, j, e, why in skipped_here:
            rows.append((stem, i, j, e, why, "SKIPPED", med0, med1))

        # before/after render for human verification
        before = render(stem, pts, H, {i for i, j, e in cands}, f"BEFORE  med={med0:.2f}ft")
        after_pts = load_label(lp)
        errA, medA, HA = fit_code2(after_pts)
        mark_after = {j for _, j, _, _ in moved} if verdict == "APPLIED" else {i for i, j, e in cands}
        after = render(stem, after_pts, HA, mark_after,
                       f"AFTER ({verdict})  med={medA:.2f}ft" if medA else f"AFTER ({verdict})")
        if before is not None and after is not None:
            cv2.imwrite(os.path.join(VERIFY_DIR, f"{verdict}_{stem}.jpg"), np.vstack([before, after]))

    with open(LOG, "w") as f:
        f.write("frame\ti\tj\terr_ft\taction\tverdict\tmed_before\tmed_after\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    print(f"swap candidates: {n_applied + n_skipped}   APPLIED: {n_applied}   SKIPPED (needs eyes): {n_skipped}")
    print(f"log -> {LOG}")
    print(f"before/after renders -> {VERIFY_DIR}/")


if __name__ == "__main__":
    main()

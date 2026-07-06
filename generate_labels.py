#!/usr/bin/env python
"""
generate_labels.py — produce YOLO-pose training labels from the verified + line-snapped
homographies (snapped_H.npz). Two formats, same projection, two keypoint schemes:

  data/court_labels_33/     — the 33-vertex scheme (drop-in retrain benchmark)
  data/court_labels_grid/   — a 13×7 court grid (91 points, ~7.8 × 8.3 ft spacing)

Every keypoint is the projection of its court-feet position through the frame's H —
including the original download anchors' vertices: after the line snap the H is more
accurate (~0.5 px median) than the box-center anchors themselves (~few px), so pure
projection beats mixing sources. On-frame points get v=2; off-frame points 0 0 0.

Label line (ultralytics pose): class cx cy bw bh  x1 y1 v1 … xk yk vk   (normalized)
The court bbox = bounding box of the VISIBLE court region (projected court boundary
clipped to the frame, plus any frame corner that lies inside the court).

Also writes:
  <dir>/manifest.tsv                 — stem, status, snap residual, #visible keypoints
  data/court_labels_grid/grid_def.tsv — grid index → court-feet position (idx = iy*13+ix)
  data/court_review/label_preview/    — 33 | grid side-by-side renders on sample frames

Usage:  /opt/anaconda3/bin/python generate_labels.py
"""
from __future__ import annotations
import os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, COURT_LENGTH_FT, COURT_WIDTH_FT

ROOT = "data/court_review"
IMG_DIR = os.path.join(ROOT, "images")
NPZ = os.path.join(ROOT, "snapped_H.npz")
DIR33 = "data/court_labels_33"
DIRGR = "data/court_labels_grid"
PREVIEW = os.path.join(ROOT, "label_preview")

GRID_NX, GRID_NY = 13, 7
GRID_FT = np.array([(x, y)
                    for y in np.linspace(0.0, COURT_WIDTH_FT, GRID_NY)
                    for x in np.linspace(0.0, COURT_LENGTH_FT, GRID_NX)], np.float32)


def project(P, pts_ft):
    return cv2.perspectiveTransform(np.asarray(pts_ft, np.float32).reshape(-1, 1, 2),
                                    P.astype(np.float64)).reshape(-1, 2)


def visible_court_bbox(P, w, h):
    """Bbox (normalized cx, cy, bw, bh) of the visible court region."""
    per = []
    for t in np.arange(0.0, COURT_LENGTH_FT + 0.5, 1.0):
        per += [(min(t, COURT_LENGTH_FT), 0.0), (min(t, COURT_LENGTH_FT), COURT_WIDTH_FT)]
    for t in np.arange(0.0, COURT_WIDTH_FT + 0.5, 1.0):
        per += [(0.0, min(t, COURT_WIDTH_FT)), (COURT_LENGTH_FT, min(t, COURT_WIDTH_FT))]
    px = project(P, per)
    ok = np.isfinite(px).all(1) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    pts = [px[ok]]
    try:
        Pinv = np.linalg.inv(P)
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        cft = project(Pinv, corners)
        inside = (np.isfinite(cft).all(1) & (cft[:, 0] >= 0) & (cft[:, 0] <= COURT_LENGTH_FT)
                  & (cft[:, 1] >= 0) & (cft[:, 1] <= COURT_WIDTH_FT))
        pts.append(corners[inside])
    except np.linalg.LinAlgError:
        pass
    pts = np.concatenate(pts)
    if len(pts) < 2:
        return None
    x0, y0 = np.clip(pts.min(0), 0, [w, h])
    x1, y1 = np.clip(pts.max(0), 0, [w, h])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def keypoint_fields(P, pts_ft, w, h):
    """Per keypoint: 'x y 2' if on-frame else '0 0 0'. Returns (fields, n_visible)."""
    px = project(P, pts_ft)
    fields, n_vis = [], 0
    for x, y in px:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
            fields.append(f"{x / w:.6f} {y / h:.6f} 2")
            n_vis += 1
        else:
            fields.append("0 0 0")
    return fields, n_vis


def render_preview(img, P, bbox33, w, h):
    left, right = img.copy(), img.copy()
    if bbox33 is not None:
        cx, cy, bw, bh = bbox33
        p0 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
        p1 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        for panel in (left, right):
            cv2.rectangle(panel, p0, p1, (0, 255, 255), 2)
    for x, y in project(P, COURT_VERTICES_33):
        if np.isfinite(x) and 0 <= x < w and 0 <= y < h:
            cv2.circle(left, (int(x), int(y)), 5, (0, 255, 0), -1)
    for x, y in project(P, GRID_FT):
        if np.isfinite(x) and 0 <= x < w and 0 <= y < h:
            cv2.circle(right, (int(x), int(y)), 4, (255, 255, 0), -1)
    cv2.putText(left, "33-vertex", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(right, f"grid {GRID_NX}x{GRID_NY}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    return np.hstack([left, right])


def main():
    for d in (DIR33, DIRGR, PREVIEW):
        os.makedirs(d, exist_ok=True)

    z = np.load(NPZ, allow_pickle=False)
    stems, Ps, status = z["stems"], z["P"], z["status"]
    res = z["res_after"]

    with open(os.path.join(DIRGR, "grid_def.tsv"), "w") as f:
        f.write("idx\tx_ft\ty_ft\n")
        for i, (x, y) in enumerate(GRID_FT):
            f.write(f"{i}\t{x:.3f}\t{y:.3f}\n")

    man33 = open(os.path.join(DIR33, "manifest.tsv"), "w")
    mangr = open(os.path.join(DIRGR, "manifest.tsv"), "w")
    for m in (man33, mangr):
        m.write("stem\tstatus\tsnap_res_px\tn_visible_kp\n")

    n_ok = n_skip = 0
    vis33, visgr = [], []
    preview_every = max(1, len(stems) // 9)
    for k in range(len(stems)):
        stem, P = str(stems[k]), Ps[k]
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            n_skip += 1
            continue
        h, w = img.shape[:2]
        bbox = visible_court_bbox(P, w, h)
        if bbox is None:
            n_skip += 1
            continue
        f33, n33 = keypoint_fields(P, COURT_VERTICES_33, w, h)
        fgr, ngr = keypoint_fields(P, GRID_FT, w, h)
        if n33 < 4 or ngr < 8:
            n_skip += 1
            continue
        head = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        with open(os.path.join(DIR33, stem + ".txt"), "w") as f:
            f.write(head + " " + " ".join(f33) + "\n")
        with open(os.path.join(DIRGR, stem + ".txt"), "w") as f:
            f.write(head + " " + " ".join(fgr) + "\n")
        r = f"{res[k]:.2f}" if np.isfinite(res[k]) else "nan"
        man33.write(f"{stem}\t{status[k]}\t{r}\t{n33}\n")
        mangr.write(f"{stem}\t{status[k]}\t{r}\t{ngr}\n")
        vis33.append(n33)
        visgr.append(ngr)
        n_ok += 1
        if k % preview_every == 0:
            cv2.imwrite(os.path.join(PREVIEW, f"{stem}.jpg"), render_preview(img, P, bbox, w, h))

    man33.close(); mangr.close()
    v33, vgr = np.array(vis33), np.array(visgr)
    print(f"labeled {n_ok} frames   skipped {n_skip}")
    print(f"33-vertex : visible kp/frame  p10={np.percentile(v33,10):.0f}  p50={np.median(v33):.0f}  p90={np.percentile(v33,90):.0f}")
    print(f"grid 13x7 : visible kp/frame  p10={np.percentile(vgr,10):.0f}  p50={np.median(vgr):.0f}  p90={np.percentile(vgr,90):.0f}")
    print(f"labels -> {DIR33}/  and  {DIRGR}/   previews -> {PREVIEW}/")


if __name__ == "__main__":
    main()

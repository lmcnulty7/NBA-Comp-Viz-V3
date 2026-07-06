#!/usr/bin/env python
"""
benchmark_court_models.py — head-to-head court-detector benchmark on the 279 held-out
val frames (data/court_pose_split.json), scored on what actually matters: the quality
of the homography you get from each model's detections.

Ground truth = the human-verified, line-snapped homographies (snapped_H.npz,
median 0.52 px line residual — far above any model's precision).

Per frame, per model:
  1. predict keypoints; accept those with conf ≥ KP_CONF;
  2. fit H (px → court ft, RANSAC 3 ft) from accepted points;
  3. H error   — project the court template through the model's H and through the GT H;
                 median px discrepancy over the GT-visible court samples;
  4. KP error  — px distance of each accepted keypoint to its GT-projected position.

Reported per model: valid-H rate, median/p90 of per-frame H error, KP error, points used.
NOTE: the old models' TRAIN set overlaps these val frames (they trained on the original
court_det2, a source of this dataset) — any bias favors the OLD models.

Renders side-by-side GT vs model overlays for the frames where models disagree most:
  data/court_review/benchmark_viz/

Usage:  /opt/anaconda3/bin/python benchmark_court_models.py
"""
from __future__ import annotations
import json, os
import cv2, numpy as np
from ultralytics import YOLO
from court.court33 import COURT_VERTICES_33, court33_segments, court33_curves
from generate_labels import GRID_FT
from snap_projections import CH_MID, project

cv2.setRNGSeed(42)
ROOT = "data/court_review"
IMG_DIR = os.path.join(ROOT, "images")
NPZ = os.path.join(ROOT, "snapped_H.npz")
SPLIT = "data/court_pose_split.json"
VIZ = os.path.join(ROOT, "benchmark_viz")
KP_CONF = 0.5
MODELS = [("old-aug", "models/court_kp33_aug.pt", COURT_VERTICES_33),
          ("NEW-33", "models/court_kp33_snapped.pt", COURT_VERTICES_33),
          ("NEW-grid", "models/court_grid_snapped.pt", GRID_FT)]


def predict_kps(model, img):
    """(33,2) px + conf for the highest-confidence court detection, or None."""
    r = model.predict(img, verbose=False, device="mps")[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return None, None
    b = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[b].cpu().numpy()
    conf = (r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None
            else np.ones(len(xy)))
    return xy, conf


def fit_H_from_kps(xy, conf, w, h, coords):
    """H: px -> court ft from accepted keypoints (conf gate + on-frame)."""
    keep = [(i, xy[i]) for i in range(len(xy))
            if conf[i] >= KP_CONF and 0 < xy[i][0] < w and 0 < xy[i][1] < h]
    if len(keep) < 4:
        return None, keep
    src = np.array([p for _, p in keep], np.float32).reshape(-1, 1, 2)
    dst = np.array([coords[i] for i, _ in keep], np.float32).reshape(-1, 1, 2)
    method = 0 if len(keep) == 4 else cv2.RANSAC
    H, _ = cv2.findHomography(src, dst, method, 3.0)
    return H, keep


def draw_court(img, P, color, thick=2):
    h, w = img.shape[:2]
    for P1, P2 in court33_segments():
        seg = project(P, np.stack([P1, P2]))
        if np.isfinite(seg).all():
            cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), color, thick, cv2.LINE_AA)
    for poly in court33_curves():
        px = project(P, poly)
        for a, b in zip(px[:-1], px[1:]):
            if (np.isfinite(a).all() and np.isfinite(b).all()
                    and max(abs(a[0]), abs(b[0])) < 4 * w and max(abs(a[1]), abs(b[1])) < 4 * h):
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def main():
    os.makedirs(VIZ, exist_ok=True)
    z = np.load(NPZ)
    gtP = {str(s): z["P"][k] for k, s in enumerate(z["stems"])}
    val = [s for s in json.load(open(SPLIT))["val"] if s in gtP]
    print(f"benchmarking on {len(val)} val frames\n")

    stats = {name: {"herr": {}, "kperr": [], "npts": [], "noH": 0} for name, _, _ in MODELS}
    models = [(name, YOLO(path), coords) for name, path, coords in MODELS]

    for n, stem in enumerate(val):
        if n % 50 == 0:
            print(f"  {n}/{len(val)} …")
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        P_gt = gtP[stem]
        px_gt = project(P_gt, CH_MID)
        vis = (np.isfinite(px_gt).all(1) & (px_gt[:, 0] >= 0) & (px_gt[:, 0] < w)
               & (px_gt[:, 1] >= 0) & (px_gt[:, 1] < h))
        for name, model, coords in models:
            xy, conf = predict_kps(model, img)
            if xy is None:
                stats[name]["noH"] += 1
                continue
            H, keep = fit_H_from_kps(xy, conf, w, h, coords)
            stats[name]["npts"].append(len(keep))
            for i, p in keep:
                gt_pt = project(P_gt, coords[i].reshape(1, 2))[0]
                stats[name]["kperr"].append(float(np.linalg.norm(p - gt_pt)))
            if H is None:
                stats[name]["noH"] += 1
                continue
            try:
                P_pred = np.linalg.inv(H.astype(np.float64))
            except np.linalg.LinAlgError:
                stats[name]["noH"] += 1
                continue
            px_pred = project(P_pred, CH_MID)
            d = np.linalg.norm(px_pred - px_gt, axis=1)[vis]
            d = d[np.isfinite(d)]
            if len(d) < 30:
                stats[name]["noH"] += 1
                continue
            stats[name]["herr"][stem] = float(np.median(np.minimum(d, 200.0)))

    print(f"\n{'model':<14}{'valid H':>9}{'H-err p50':>11}{'H-err p90':>11}{'>10px':>7}"
          f"{'kp-err p50':>12}{'pts/frame':>11}")
    for name, _, _ in MODELS:
        s = stats[name]
        he = np.array(list(s["herr"].values()))
        ke, np_ = np.array(s["kperr"]), np.array(s["npts"])
        print(f"{name:<14}{len(he)}/{len(val):>4}"
              f"{np.median(he):>10.1f}px{np.percentile(he, 90):>9.1f}px"
              f"{(he > 10).sum():>7}"
              f"{np.median(ke):>10.1f}px{np.median(np_):>9.0f}")

    # render the frames where the two NEW models disagree most (either direction)
    new, old = stats["NEW-grid"]["herr"], stats["NEW-33"]["herr"]
    both = [(abs(new[s] - old[s]), s) for s in new if s in old]
    both.sort(reverse=True)
    bym = {name: (m, c) for name, m, c in models}
    for gap, stem in both[:10]:
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        h, w = img.shape[:2]
        panels = []
        for tag, err in (("NEW-33", old[stem]), ("NEW-grid", new[stem])):
            model, coords = bym[tag]
            panel = img.copy()
            draw_court(panel, gtP[stem], (0, 255, 0), 1)                 # GT thin green
            xy, conf = predict_kps(model, panel)
            H, _ = fit_H_from_kps(xy, conf, w, h, coords) if xy is not None else (None, [])
            if H is not None:
                try:
                    draw_court(panel, np.linalg.inv(H.astype(np.float64)), (255, 0, 255), 2)
                except np.linalg.LinAlgError:
                    pass
            cv2.rectangle(panel, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(panel, f"{tag}  H-err {err:.1f}px  (green=GT)", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        cv2.imwrite(os.path.join(VIZ, f"gap{gap:05.1f}_{stem}.jpg"), np.hstack(panels))
    print(f"\nside-by-sides (biggest disagreements) -> {VIZ}/")


if __name__ == "__main__":
    main()

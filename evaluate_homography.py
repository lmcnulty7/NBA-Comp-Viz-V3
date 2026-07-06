#!/usr/bin/env python
"""
evaluate_homography.py — Component 3 evaluation, on the held-out val split.

Three things, all on frames NOT used in training:
  1. Keypoint accuracy — predicted vs hand-labeled pixel error, per keypoint and
     overall, plus per-keypoint detection rate.
  2. Homography quality — fraction of frames yielding a valid H, and the RANSAC
     reprojection error (feet) of the fits.
  3. THE visual proof — project the full court template back onto each frame via
     the predicted homography. If the green lines land on the real court lines,
     the homography is good. Saved as a montage.

Outputs: reports/homography_metrics.{json,txt}, reports/viz/homography_overlay.png.

Examples
────────
  python evaluate_homography.py
  python evaluate_homography.py --conf 0.4
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from court.detector import CourtKeypointDetector
from court.geometry import KEYPOINT_NAMES, court_polylines

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("evaluate_homography")


def load_val_records():
    split_path = config.COURT_DIR / "kp_split.json"
    if not split_path.exists():
        raise SystemExit(f"No split at {split_path}. Run train_court_kp.py first.")
    val_stems = set(json.loads(split_path.read_text())["val"])
    recs = []
    for stem in sorted(val_stems):
        jp = config.COURT_KP_LABELS / f"{stem}.json"
        if jp.exists():
            recs.append(json.loads(jp.read_text()))
    if not recs:
        raise SystemExit("No val labels found — relabel/retrain.")
    return recs


def project_court_overlay(frame, hom):
    """Draw the court template projected via H_inv (court→pixel) onto the frame."""
    out = frame.copy()
    for poly in court_polylines():
        px = hom.to_pixel_batch(poly)
        if np.isnan(px).any():
            continue
        cv2.polylines(out, [px.astype(np.int32)], False, (60, 230, 60), 2, cv2.LINE_AA)
    return out


def make_overlay_sheet(samples, out_path, cols=2, max_tiles=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = samples[:max_tiles]
    if not samples:
        return
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, s in zip(axes, samples):
        ax.imshow(s["img"][:, :, ::-1])
        ax.set_title(s["title"], fontsize=9)
    fig.suptitle("Homography check — green = projected court template (should align with court lines)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate court keypoints + homography on the val split.")
    ap.add_argument("--conf", type=float, default=None, help="Keypoint confidence threshold.")
    ap.add_argument("--ransac-ft", type=float, default=None, help="Override RANSAC reproj threshold (ft).")
    args = ap.parse_args()
    if args.ransac_ft is not None:
        config.RANSAC_REPROJ_THRESHOLD = args.ransac_ft

    recs = load_val_records()
    log.info("Val frames: %d (held out from training).", len(recs))
    det = CourtKeypointDetector(conf=args.conf)

    n_valid_H = 0
    reproj_errs = []
    px_err_per_kp = defaultdict(list)     # name -> list of px errors (pred vs GT)
    detect_count = defaultdict(int)       # name -> times detected
    gt_count = defaultdict(int)           # name -> times labeled in GT
    samples = []

    for rec in recs:
        img = cv2.imread(rec["image_path"])
        if img is None:
            continue
        pred = det.detect(img)
        gt = {n: np.array(v, np.float32) for n, v in rec["keypoints"].items() if v is not None}
        for n in gt:
            gt_count[n] += 1
        for n, p in pred.items():
            detect_count[n] += 1
            if n in gt:
                px_err_per_kp[n].append(float(np.linalg.norm(p - gt[n])))

        hom = det.homography(img)
        if hom is not None and hom.is_valid:
            n_valid_H += 1
            if hom.quality is not None:
                reproj_errs.append(hom.quality)
            overlay = project_court_overlay(img, hom)
            samples.append({"img": overlay,
                            "title": f"{rec['frame']}  |  {len(pred)} kps, reproj {hom.quality:.2f} ft"})

    all_px = [e for errs in px_err_per_kp.values() for e in errs]
    out = {
        "val_frames": len(recs),
        "homography_success_rate": round(n_valid_H / len(recs), 3),
        "reproj_error_ft": {
            "mean": round(float(np.mean(reproj_errs)), 3) if reproj_errs else None,
            "median": round(float(np.median(reproj_errs)), 3) if reproj_errs else None,
            "p90": round(float(np.percentile(reproj_errs, 90)), 3) if reproj_errs else None,
        },
        "keypoint_pixel_error": {
            "mean": round(float(np.mean(all_px)), 2) if all_px else None,
            "median": round(float(np.median(all_px)), 2) if all_px else None,
        },
        "per_keypoint": {
            n: {
                "gt_count": gt_count[n],
                "detected_count": detect_count[n],
                "mean_px_err": round(float(np.mean(px_err_per_kp[n])), 2) if px_err_per_kp[n] else None,
            } for n in KEYPOINT_NAMES
        },
        "note": "reproj_error = RANSAC fit consistency (ft); keypoint_pixel_error = detector "
                "accuracy vs hand labels (px). Visual proof in reports/viz/homography_overlay.png.",
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "homography_metrics.json").write_text(json.dumps(out, indent=2))

    lines = ["COURT KEYPOINTS + HOMOGRAPHY — EVAL (held-out val)", "=" * 52,
             f"val frames: {out['val_frames']}",
             f"homography success rate: {out['homography_success_rate']:.1%}",
             f"reprojection error (ft): mean={out['reproj_error_ft']['mean']} "
             f"median={out['reproj_error_ft']['median']} p90={out['reproj_error_ft']['p90']}",
             f"keypoint pixel error: mean={out['keypoint_pixel_error']['mean']} "
             f"median={out['keypoint_pixel_error']['median']}", "",
             f"  {'keypoint':<16}{'GT':>4}{'det':>5}{'px_err':>8}"]
    for n in KEYPOINT_NAMES:
        pk = out["per_keypoint"][n]
        lines.append(f"  {n:<16}{pk['gt_count']:>4}{pk['detected_count']:>5}"
                     f"{(pk['mean_px_err'] if pk['mean_px_err'] is not None else '-'):>8}")
    lines += ["", "reproj < ~1.5 ft = usable court coords. Check the overlay PNG: green lines",
              "should sit on the real court lines."]
    (config.REPORTS_DIR / "homography_metrics.txt").write_text("\n".join(lines))

    make_overlay_sheet(samples, config.VIZ_DIR / "homography_overlay.png")
    print("\n".join(lines))
    log.info("Wrote reports/homography_metrics.{json,txt} + reports/viz/homography_overlay.png")


if __name__ == "__main__":
    main()

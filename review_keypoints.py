#!/usr/bin/env python
"""
review_keypoints.py — QA your hand-labeled keypoints BEFORE training.

Idea: a mislabeled point betrays itself geometrically. Fit a homography from each
frame's labels; a bad point lands far from where it should, producing a large
reprojection residual. So we rank frames by error and show the worst ones with the
court template projected from YOUR labels — if a frame's green lines don't sit on
the real court, that frame has a bad point (named for you).

No model needed — this checks the LABELS themselves. Run it, fix the flagged frames
with `label_keypoints.py --review`, then train.

Outputs: reports/viz/kp_review_worst.png + a ranked console list.

Usage
─────
  python review_keypoints.py
  python review_keypoints.py --flag-ft 0.5
"""
from __future__ import annotations

import argparse
import json
import logging

import cv2
import numpy as np

import config
from court.geometry import COURT_KEYPOINTS, court_polylines
from court.homography import CourtHomography

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("review_keypoints")


def per_point_residuals(hom, labels):
    """ft residual between each labeled point projected to court vs its true court coord."""
    res = {}
    for name, px in labels.items():
        court = hom.to_court_batch(np.array([px]))[0]
        res[name] = float(np.linalg.norm(court - COURT_KEYPOINTS[name]))
    return res


def draw_tile(img, labels, hom, residuals):
    out = img.copy()
    if hom is not None and hom.is_valid:  # projected court overlay from the labels' homography
        for poly in court_polylines():
            px = hom.to_pixel_batch(poly)
            if not np.isnan(px).any():
                cv2.polylines(out, [px.astype(np.int32)], False, (60, 230, 60), 2, cv2.LINE_AA)
    residuals = residuals or {}
    for name, p in labels.items():
        r = residuals.get(name, 0.0)
        col = (60, 220, 60) if r < 0.5 else ((40, 170, 255) if r < 1.5 else (40, 40, 255))
        cv2.circle(out, (int(p[0]), int(p[1])), 5, col, -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="QA hand-labeled court keypoints via homography residuals.")
    ap.add_argument("--flag-ft", type=float, default=0.5, help="Reproj error (ft) above which to flag a frame.")
    ap.add_argument("--max-tiles", type=int, default=12)
    args = ap.parse_args()

    files = sorted(config.COURT_KP_LABELS.glob("*.json"))
    if not files:
        raise SystemExit(f"No labels in {config.COURT_KP_LABELS}. Run label_keypoints.py first.")

    def fit(labels):
        hom = CourtHomography()
        ok = hom.compute(np.array(list(labels.values())),
                         np.array([COURT_KEYPOINTS[n] for n in labels]))
        return hom if ok else None

    clean, mislabel, degenerate, too_few = [], [], [], []
    for f in files:
        rec = json.loads(f.read_text())
        labels = {n: np.array(v, np.float32) for n, v in rec["keypoints"].items() if v is not None}
        n = len(labels)
        if n < config.MIN_KEYPOINTS_FOR_H:
            too_few.append((rec["frame"], n))
            continue
        hom = fit(labels)
        if hom is None:                       # findHomography failed → fully collinear
            degenerate.append({"rec": rec, "labels": labels, "hom": None, "n": n, "reason": "points collinear"})
            continue
        resid = per_point_residuals(hom, labels)
        worst = max(resid.items(), key=lambda kv: kv[1])
        err = hom.quality
        row = {"rec": rec, "labels": labels, "hom": hom, "resid": resid, "err": err, "n": n, "worst": worst}
        if err < args.flag_ft:
            clean.append(row)
        elif n >= 5:
            # leave-one-out: drop the worst point and refit. If the rest now fit, it's a real mislabel.
            sub = {k: v for k, v in labels.items() if k != worst[0]}
            hom2 = fit(sub)
            if hom2 is not None and hom2.quality is not None and hom2.quality < args.flag_ft:
                mislabel.append(row)          # worst[0] is the culprit
            else:
                row["reason"] = "near-collinear / too few off-line points"
                degenerate.append(row)
        else:                                  # n==4 high error → can't isolate; geometric
            row["reason"] = "only 4 points and ≥3 near-collinear"
            degenerate.append(row)

    clean.sort(key=lambda r: r["err"])
    mislabel.sort(key=lambda r: r["err"], reverse=True)

    # ── console report ──
    errs = [r["err"] for r in clean] + [r["err"] for r in mislabel]
    print(f"\nReviewed {len(files)} frames →  {len(clean)} clean | {len(mislabel)} likely-mislabeled | "
          f"{len(degenerate)} geometrically-degenerate | {len(too_few)} with <4 points")
    if errs:
        print(f"Reprojection error on fittable frames (ft): median={np.median(errs):.3f}  "
              f"mean={np.mean(errs):.3f}   (≈ inches; <0.3 ft is excellent)")

    print(f"\n>>> FIX THESE — likely a genuinely misplaced point (reproj ≥ {args.flag_ft} ft, isolated by refit):")
    if not mislabel:
        print("    none 🎉  no individual mislabeled points detected")
    for r in mislabel[:25]:
        print(f"    {r['rec']['frame']:<34} err={r['err']:.2f} ft  →  check '{r['worst'][0]}' ({r['worst'][1]:.2f} ft off)")

    if degenerate:
        print(f"\n(info) {len(degenerate)} frame(s) are GEOMETRICALLY DEGENERATE — points correct but too "
              "collinear to anchor a\n       homography alone (labels still fine for TRAINING). Add an off-line "
              "point if you want them usable:")
        for r in degenerate[:8]:
            print(f"    {r['rec']['frame']:<34} n={r['n']}  ({r.get('reason','')})")
    if too_few:
        print(f"\n(info) {len(too_few)} frame(s) have <4 points — excluded from training. Add points or ignore:")
        for name, n in too_few[:8]:
            print(f"    {name}  ({n} pts)")

    # ── visual sheet: genuine mislabels first, then degenerate examples ──
    tiles = (mislabel + degenerate)[:args.max_tiles]
    if tiles:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = 3
        nrows = int(np.ceil(len(tiles) / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(cols * 4.5, nrows * 2.8))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, r in zip(axes, tiles):
            img = cv2.imread(r["rec"]["image_path"])
            if img is None:
                continue
            tile = draw_tile(img, r["labels"], r.get("hom"), r.get("resid"))
            ax.imshow(tile[:, :, ::-1])
            if "err" in r:
                title = f"{r['rec']['frame']}\nerr={r['err']:.2f}ft worst={r['worst'][0]}"
            else:
                title = f"{r['rec']['frame']}\n{r.get('reason','degenerate')}"
            ax.set_title(title, fontsize=8)
        fig.suptitle("Keypoint review — green lines should sit on the court; red dots = high-residual.", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = config.VIZ_DIR / "kp_review_worst.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log.info("Wrote %s", out)
    print("\nFix flagged frames:  python label_keypoints.py --review   (page to them, correct, save)")
    print("When clean:           python train_court_kp.py")


if __name__ == "__main__":
    main()

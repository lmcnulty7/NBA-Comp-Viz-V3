#!/usr/bin/env python
"""
make_line_dataset.py — build a court-line segmentation dataset for FREE.

No hand-tracing of lines. Every keypoint-labeled frame already yields an accurate
homography, so we project the known court template (all lines + arcs + circle)
through it and rasterize it into a per-pixel line mask. That (image, line-mask)
pair is a segmentation training example.

Output: data/court/line_dataset/{images,masks}/{train,val}/  (mask = 0/255 PNG).
Reuses the keypoint train/val split if present so the two models share a split.

Usage
─────
  python make_line_dataset.py
  python make_line_dataset.py --thickness 3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import config
from court.geometry import COURT_KEYPOINTS, court_polylines
from court.homography import homography_from_named_keypoints


def render_line_mask(hom, w, h, thickness):
    """Project the full court template into a (h, w) 0/255 line mask."""
    mask = np.zeros((h, w), np.uint8)
    for poly in court_polylines():
        px = hom.to_pixel_batch(poly)
        if np.isfinite(px).all():
            cv2.polylines(mask, [px.astype(np.int32)], False, 255, thickness, cv2.LINE_AA)
    return mask


def load_split():
    """{stem: 'train'|'val'} from the keypoint split, if available."""
    sp = config.COURT_DIR / "kp_split.json"
    if not sp.exists():
        return None
    rec = json.loads(sp.read_text())
    out = {}
    for s in ("train", "val"):
        for stem in rec.get(s, []):
            out[stem] = s
    return out


def main():
    ap = argparse.ArgumentParser(description="Auto-generate court-line segmentation labels from keypoint homographies.")
    ap.add_argument("--thickness", type=int, default=config.LINE_MASK_THICKNESS)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    label_files = sorted(config.COURT_KP_LABELS.glob("*.json"))
    if not label_files:
        raise SystemExit(f"No keypoint labels in {config.COURT_KP_LABELS}. Label keypoints first.")

    root = config.LINE_DATASET
    for sub in ("images/train", "images/val", "masks/train", "masks/val"):
        d = root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    split = load_split()
    import random
    rng = random.Random(args.seed)

    n_ok = n_skip = 0
    counts = {"train": 0, "val": 0}
    for lf in label_files:
        rec = json.loads(lf.read_text())
        kps = {n: np.array(v, np.float32) for n, v in rec["keypoints"].items()
               if v is not None and n in COURT_KEYPOINTS}
        hom = homography_from_named_keypoints(kps)
        if hom is None:
            n_skip += 1
            continue
        img = cv2.imread(rec["image_path"])
        if img is None:
            n_skip += 1
            continue
        h, w = img.shape[:2]
        mask = render_line_mask(hom, w, h, args.thickness)
        stem = Path(rec["frame"]).stem
        sp = split.get(stem) if split else ("val" if rng.random() < args.val_frac else "train")
        sp = sp or "train"
        cv2.imwrite(str(root / f"images/{sp}/{stem}.png"), img)
        cv2.imwrite(str(root / f"masks/{sp}/{stem}.png"), mask)
        counts[sp] += 1
        n_ok += 1

    print(f"Line dataset built from {n_ok} frames ({n_skip} skipped: no homography).")
    print(f"  train: {counts['train']}   val: {counts['val']}")
    print(f"  → {root}/  (images/ + masks/)")
    print("Review a few image/mask pairs, then: python train_line_seg.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
train_court_kp.py — Component 3 training: keypoint labels → YOLOv8-pose model.

Converts the per-frame keypoint JSON from label_keypoints.py into a YOLO-pose
dataset, makes a deterministic train/val split (saved), and fine-tunes
yolov8n-pose to predict the 16 court landmarks. The trained weights feed the
homography (court.homography), which is already verified.

Only frames with ≥ MIN_KEYPOINTS_FOR_H visible points are used (fewer can't anchor
a homography and would just be label noise).

Usage
─────
  python train_court_kp.py
  python train_court_kp.py --epochs 120 --imgsz 960
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np

import config
from court.geometry import KEYPOINT_NAMES, FLIP_IDX

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_court_kp")

N_KP = len(KEYPOINT_NAMES)
PAD = 0.04  # bbox padding (fraction of frame) around visible keypoints


def load_labels() -> list[dict]:
    files = sorted(config.COURT_KP_LABELS.glob("*.json"))
    recs = []
    for f in files:
        r = json.loads(f.read_text())
        n_vis = sum(1 for v in r["keypoints"].values() if v is not None)
        if n_vis >= config.MIN_KEYPOINTS_FOR_H:
            recs.append(r)
    return recs


def to_yolo_line(rec: dict) -> str | None:
    W, H = rec["width"], rec["height"]
    kps = rec["keypoints"]
    n_vis = sum(1 for n in KEYPOINT_NAMES if kps.get(n) is not None)
    if n_vis < config.MIN_KEYPOINTS_FOR_H:
        return None
    # FULL-FRAME box for the single "court" instance. A consistent box makes detection
    # trivial so the model's capacity goes to keypoint regression. (Per-keypoint-hull
    # boxes were thin/degenerate on collinear frames and prevented the pose head from
    # ever training — flat pose loss, zero pose mAP.)
    parts = ["0", "0.500000", "0.500000", "1.000000", "1.000000"]
    for n in KEYPOINT_NAMES:
        p = kps.get(n)
        if p is None:
            parts += ["0", "0", "0"]            # v=0 not labeled
        else:
            parts += [f"{p[0]/W:.6f}", f"{p[1]/H:.6f}", "2"]  # v=2 visible
    return " ".join(parts)


def build_dataset(recs, val_frac, seed):
    root = config.COURT_KP_DATASET
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    recs = recs[:]
    random.shuffle(recs)
    n_val = max(1, int(len(recs) * val_frac))
    split = {"val": recs[:n_val], "train": recs[n_val:]}

    counts = {}
    for name, items in split.items():
        for r in items:
            line = to_yolo_line(r)
            if line is None:
                continue
            stem = Path(r["frame"]).stem
            shutil.copy2(r["image_path"], root / f"images/{name}/{Path(r['frame']).name}")
            (root / f"labels/{name}/{stem}.txt").write_text(line + "\n")
        counts[name] = len(list((root / f"images/{name}").glob("*")))

    # save split (frame stems) for evaluate_homography.py
    config.COURT_DIR.mkdir(parents=True, exist_ok=True)
    (config.COURT_DIR / "kp_split.json").write_text(json.dumps(
        {k: [Path(r["frame"]).stem for r in v] for k, v in split.items()}, indent=2))
    return counts


def write_yaml():
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(["court"]))
    yaml = (
        f"path: {config.COURT_KP_DATASET.resolve()}\n"
        f"train: images/train\nval: images/val\n\n"
        f"kpt_shape: [{N_KP}, 3]\n"
        f"flip_idx: {FLIP_IDX}\n\n"
        f"names:\n{names_block}\n"
    )
    config.COURT_KP_DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    config.COURT_KP_DATASET_YAML.write_text(yaml)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8-pose court-keypoint model.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)   # thin court lines like bigger, but CPU-bound
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (e.g. 'cpu'). YOLOv8-pose keypoint loss doesn't "
                         "train on Apple MPS, so court-kp training uses CPU.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    # YOLOv8-pose keypoint loss does NOT train on Apple MPS (gradients don't flow → flat
    # pose loss, zero pose mAP); detection trains but the pose head stays dead. So default
    # to CPU for THIS training only. Inference and every other component still use MPS.
    device = args.device or "cpu"

    config.set_seed(args.seed)
    recs = load_labels()
    n_all = len(list(config.COURT_KP_LABELS.glob("*.json")))
    log.info("Labeled frames: %d total, %d usable (≥%d visible kps).",
             n_all, len(recs), config.MIN_KEYPOINTS_FOR_H)
    if len(recs) < 30:
        log.warning("Only %d usable frames — pose training wants ≥~100 for a good model. "
                    "Label more with label_keypoints.py.", len(recs))
    if len(recs) < 10:
        raise SystemExit("Too few labeled frames to train. Run label_keypoints.py first.")

    counts = build_dataset(recs, args.val_frac, args.seed)
    write_yaml()
    log.info("Dataset built: %d train / %d val → %s", counts.get("train", 0), counts.get("val", 0),
             config.COURT_KP_DATASET)

    from ultralytics import YOLO
    model = YOLO(config.COURT_KP_BASE)   # auto-downloads yolov8n-pose.pt
    log.info("Fine-tuning %s for %d epochs (imgsz=%d, device=%s) …",
             config.COURT_KP_BASE, args.epochs, args.imgsz, device)
    results = model.train(
        data=str(config.COURT_KP_DATASET_YAML),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, patience=25,
        project=str(config.MODELS_DIR / "court_kp_runs"), name="train",
        fliplr=0.5, degrees=0.0, translate=0.05, scale=0.2, mosaic=0.0,  # geometry-preserving aug
        verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, config.COURT_KP_WEIGHTS)
        log.info("Saved court-keypoint weights → %s", config.COURT_KP_WEIGHTS)
        log.info("Next: python evaluate_homography.py")
    else:
        log.error("best.pt not found at %s", best)


if __name__ == "__main__":
    main()

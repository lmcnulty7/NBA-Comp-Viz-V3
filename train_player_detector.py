#!/usr/bin/env python
"""
train_player_detector.py — fine-tune a basketball YOLOv8 detector.

Trains on the remapped player-detection-3 set (player/referee/ball/rim/number) so
the tracker can detect PLAYERS specifically — distinct from referees, and without
firing on crowd/bench the way the generic COCO "person" detector does.

Run prepare_player_dataset.py first. Detection trains fine on MPS (the MPS issue
was pose-specific), but --device cpu is available if needed.

Usage
─────
  python prepare_player_dataset.py
  python train_player_detector.py
  python train_player_detector.py --model yolov8s.pt --epochs 80
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config


def main():
    ap = argparse.ArgumentParser(description="Train basketball player/ref/ball/rim detector.")
    ap.add_argument("--model", type=str, default=config.PLAYER_DETECTOR_BASE)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    if not config.PLAYER_DETECTOR_DATA.exists():
        raise SystemExit("Run prepare_player_dataset.py first (no data.yaml).")
    device = args.device or config.get_device()

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"Training {args.model} on {config.PLAYER_DETECTOR_DATA} | {args.epochs} ep, imgsz {args.imgsz}, device {device}")
    results = model.train(
        data=str(config.PLAYER_DETECTOR_DATA),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        patience=20, project=str(config.MODELS_DIR / "player_runs"), name="train",
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, translate=0.1, scale=0.5, mosaic=1.0,
        verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, config.PLAYER_DETECTOR_WEIGHTS)
        print(f"\nSaved → {config.PLAYER_DETECTOR_WEIGHTS}")
        print("The tracker auto-uses it (PlayerTracker tracks the 'player' class only).")
    else:
        print(f"WARNING: best.pt not found at {best}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
train_court_pose.py — fine-tune YOLOv8-pose on a projection-labeled court dataset.

Hyperparameters replicate the original court_kp33 Colab recipe exactly
(train_court_colab.ipynb: yolov8m-pose, 300 ep, 640 px, batch 16, fliplr 0.5,
mosaic 0, degrees 0, translate 0.05, scale 0.2, patience 30) so a comparison
against the old model isolates the effect of the new labels.

Usage:
  /opt/anaconda3/bin/python train_court_pose.py data/court_pose33/data.yaml pose33_snapped
  /opt/anaconda3/bin/python train_court_pose.py data/court_pose_grid/data.yaml grid_snapped
"""
from __future__ import annotations
import sys
from ultralytics import YOLO


def main():
    data_yaml, run_name = sys.argv[1], sys.argv[2]
    model = YOLO("yolov8m-pose.pt")
    model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=16,
        device="mps",
        patience=30,
        seed=42,
        deterministic=True,
        mosaic=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        project="court_kp_runs",
        name=run_name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

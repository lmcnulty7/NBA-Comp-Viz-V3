"""
detect/detector.py — single-frame player detection (no tracking).

The tracker (tracker.py) is the streaming workhorse; this is the stateless
per-frame detector used for *evaluation* (detection precision/recall against
hand-labeled boxes) and any one-off snapshot use. Same YOLO weights and
thresholds as the tracker, so detection metrics reflect what the tracker sees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config


class PlayerDetector:
    def __init__(self, weights=None, conf=None, iou=None, device=None):
        from ultralytics import YOLO

        default_w = (config.PLAYER_DETECTOR_WEIGHTS if config.PLAYER_DETECTOR_WEIGHTS.exists()
                     else config.YOLO_WEIGHTS)
        self.weights = str(weights or default_w)
        self.conf = config.PLAYER_CONF if conf is None else conf
        self.iou = config.PLAYER_IOU if iou is None else iou
        self.device = device or config.get_device()
        self.model = YOLO(self.weights)

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return (N, 5) array of [x1, y1, x2, y2, conf] for person detections,
        sorted by confidence descending."""
        results = self.model.predict(
            bgr, classes=[config.PERSON_CLASS], conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                out.append([b[0], b[1], b[2], b[3], c])
        arr = np.array(out, dtype=np.float32) if out else np.zeros((0, 5), np.float32)
        if len(arr):
            arr = arr[arr[:, 4].argsort()[::-1]]  # high conf first
        return arr

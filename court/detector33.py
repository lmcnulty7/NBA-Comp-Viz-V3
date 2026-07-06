"""
court/detector33.py — 33-keypoint court detector (external court-detection-2 model).

Predicts the 33 court landmarks (court/court33.py scheme) and returns them as a
{index: [u, v]} dict (index == vertex index in COURT_VERTICES_33), ready to feed
homography_from_indexed_keypoints. Trained on 850 images vs the legacy 20-pt model's
400 hand-labels, so it should give a stronger, more consistent homography.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from .court33 import COURT_VERTICES_33
from .homography import homography_from_indexed_keypoints


class CourtKeypointDetector33:
    def __init__(self, weights=None, conf=None, device=None):
        from ultralytics import YOLO

        self.weights = str(weights or config.COURT_KP33_WEIGHTS)
        self.conf = config.COURT_KP_CONF if conf is None else conf
        self.device = device or config.get_device()
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"No 33-pt court weights at {self.weights}.")
        self.model = YOLO(self.weights)

    def detect(self, bgr) -> dict[int, np.ndarray]:
        """Return {keypoint_index: [u, v]} for points above confidence."""
        res = self.model.predict(bgr, device=self.device, verbose=False)
        for r in res:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            bi = int(r.boxes.conf.argmax())
            xy = r.keypoints.xy[bi].cpu().numpy()
            kc = (r.keypoints.conf[bi].cpu().numpy()
                  if r.keypoints.conf is not None else np.ones(len(xy)))
            return {i: xy[i].astype(np.float32) for i in range(len(xy))
                    if kc[i] >= self.conf and not (xy[i][0] == 0 and xy[i][1] == 0)}
        return {}

    def homography(self, bgr):
        return homography_from_indexed_keypoints(self.detect(bgr), COURT_VERTICES_33)

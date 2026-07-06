"""
court/detector.py — YOLOv8-pose court-keypoint detector.

Predicts the 16 court landmarks in a frame and returns them as a
{name: [u, v]} dict (only points above the confidence threshold), ready to feed
court.homography.homography_from_named_keypoints. Falls back to an empty dict
if weights are missing, so the pipeline degrades gracefully.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from .geometry import KEYPOINT_NAMES
from .homography import homography_from_named_keypoints


class CourtKeypointDetector:
    def __init__(self, weights=None, conf=None, device=None):
        from ultralytics import YOLO

        self.weights = str(weights or config.COURT_KP_WEIGHTS)
        self.conf = config.COURT_KP_CONF if conf is None else conf
        self.device = device or config.get_device()
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"No court-keypoint weights at {self.weights}. Run train_court_kp.py.")
        self.model = YOLO(self.weights)

    def detect(self, bgr) -> dict[str, np.ndarray]:
        """Return {keypoint_name: [u, v]} for points above confidence."""
        res = self.model.predict(bgr, device=self.device, verbose=False)
        for r in res:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            bi = int(r.boxes.conf.argmax())             # the most confident court instance
            xy = r.keypoints.xy[bi].cpu().numpy()       # (N_KP, 2)
            kc = (r.keypoints.conf[bi].cpu().numpy()
                  if r.keypoints.conf is not None else np.ones(len(xy)))
            out = {}
            for i, name in enumerate(KEYPOINT_NAMES):
                if kc[i] >= self.conf and not (xy[i][0] == 0 and xy[i][1] == 0):
                    out[name] = xy[i].astype(np.float32)
            return out
        return {}

    def homography(self, bgr):
        """Detect keypoints and return a CourtHomography (or None)."""
        return homography_from_named_keypoints(self.detect(bgr))

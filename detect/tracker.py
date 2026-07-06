"""
detect/tracker.py — PlayerTracker: streaming player detection + tracking.

One ultralytics call does both: YOLO detects people and BoT-SORT assigns
persistent IDs in a single forward pass. `persist=True` keeps tracker state alive
between frames so IDs are consistent. We only track the person class; ref/player
separation is a downstream concern.

Streaming contract (the stable interface, mirroring the gate):
    tracker.update(bgr, frame_idx, timestamp_sec) -> list[Track]

Two robustness behaviors carried over from the old project:
  • Camera-cut reset — see camera_cut.py.
  • MPS Kalman guard — BoT-SORT's Kalman filter can become non-positive-definite on
    Apple MPS due to float precision (np.linalg.LinAlgError). We catch it, reset,
    and emit no tracks for that frame rather than crashing the run.

ID-collision fix (Component B prerequisite): ultralytics' BYTETracker.__init__
calls reset_id(), so every reset() restarted track IDs at 1 — a player in shot 3
would silently inherit the ID of a different player from shot 1, and anything
keyed by track_id downstream (trajectories!) merged different people. We now add
a monotonically-increasing offset across resets so every emitted track_id is
globally unique for the life of this PlayerTracker.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import config
from .camera_cut import CameraCutDetector
from .types import Track

logger = logging.getLogger(__name__)


class PlayerTracker:
    def __init__(
        self,
        weights: str | Path = None,
        conf: float = None,
        iou: float = None,
        device: str = None,
        tracker_config: str = None,
    ):
        from ultralytics import YOLO  # lazy import

        # Prefer the basketball-trained detector (player/ref/ball/rim) when present — its
        # "player" class (index 0) excludes refs and crowd, unlike generic COCO "person".
        default_w = (config.PLAYER_DETECTOR_WEIGHTS if config.PLAYER_DETECTOR_WEIGHTS.exists()
                     else config.YOLO_WEIGHTS)
        self.weights = str(weights or default_w)
        self.conf = config.PLAYER_CONF if conf is None else conf
        self.iou = config.PLAYER_IOU if iou is None else iou
        self.device = device or config.get_device()
        self.tracker_config = tracker_config or config.TRACKER_CONFIG

        self.model = YOLO(self.weights)
        self.cut = CameraCutDetector(config.CAMERA_CUT_VANISH_FRAC, config.CAMERA_CUT_MIN_TRACKS)
        self.n_resets = 0
        self._id_offset = 0   # added to raw BoT-SORT ids (which restart at 1 after each reset)
        self._max_id = 0      # highest globally-unique id emitted so far
        logger.info("PlayerTracker ready (weights=%s, device=%s, tracker=%s, conf=%.2f)",
                    Path(self.weights).name, self.device, self.tracker_config, self.conf)

    def update(self, bgr: np.ndarray, frame_idx: int = 0, timestamp_sec: float = 0.0) -> list[Track]:
        try:
            results = self.model.track(
                bgr,
                classes=[config.PERSON_CLASS],
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            tracks = self._parse(results, frame_idx, timestamp_sec)
        except np.linalg.LinAlgError:
            logger.warning("Kalman LinAlgError at frame %d (MPS precision) — resetting tracker", frame_idx)
            self.reset()
            return []

        # Camera-cut handling: if most tracks vanished, reset for the next frame.
        if self.cut.update({t.track_id for t in tracks}):
            logger.info("Camera cut at frame %d — resetting track IDs", frame_idx)
            self.reset()
        return tracks

    def reset(self) -> None:
        """Clear tracker state so IDs restart fresh (new camera shot).
        Raw BoT-SORT ids restart at 1, so bump the offset to keep emitted ids unique."""
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None
        self.cut.reset()
        self.n_resets += 1
        self._id_offset = self._max_id

    def _parse(self, results, frame_idx: int, timestamp_sec: float) -> list[Track]:
        tracks: list[Track] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue  # no tracked boxes this frame
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for bbox, tid, conf in zip(boxes, ids, confs):
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                uid = int(tid) + self._id_offset
                self._max_id = max(self._max_id, uid)
                tracks.append(Track(
                    track_id=uid,
                    bbox=bbox.astype(np.float32),
                    center=np.array([cx, cy], dtype=np.float32),
                    confidence=float(conf),
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                ))
        tracks.sort(key=lambda t: t.track_id)
        return tracks

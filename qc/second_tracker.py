"""
qc/second_tracker.py: an INDEPENDENT detector + tracker over the same processed frames (FIX_PLAN A4).

COCO yolov8m (generic "person", not the basketball-trained player_detector.pt) with ByteTrack
(not BoT-SORT). Different weights, different training data, different association: where the two
disagree, one of them is wrong. Output per frame: boxes with ids. Cached next to the sidecar as
<clip>_second.json so the detection runs once per build.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import config

COCO_WEIGHTS = config.YOLO_WEIGHTS          # generic COCO yolov8m, never the basketball model
SECOND_TRACKER = "bytetrack.yaml"           # ultralytics bundled config; the pipeline uses botsort.yaml
SECOND_CONF, SECOND_IOU = config.PLAYER_CONF, config.PLAYER_IOU   # same operating point as the pipeline
COCO_PERSON = 0


def run_second(video_path, sidecar: dict, out_path: Path, device=None) -> dict:
    """Detect+track on every sidecar frame (sequential read, same stride). Writes and returns the cache."""
    import cv2
    from ultralytics import YOLO

    if out_path.exists():
        return json.loads(out_path.read_text())
    model = YOLO(str(COCO_WEIGHTS))
    device = device or config.get_device()
    want = {r["frame"] for r in sidecar["frames"]}
    cap = cv2.VideoCapture(str(video_path))
    frames, idx = {}, 0
    while want:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in want:
            want.discard(idx)
            res = model.track(frame, classes=[COCO_PERSON], persist=True, tracker=SECOND_TRACKER,
                              conf=SECOND_CONF, iou=SECOND_IOU, device=device, verbose=False)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else np.full(len(xyxy), -1)
                cf = res.boxes.conf.cpu().numpy()
                boxes = [{"tid": int(i), "bbox": [int(v) for v in b], "conf": round(float(c), 3)} for b, i, c in zip(xyxy, ids, cf)]
            frames[idx] = boxes
        idx += 1
    cap.release()
    out = {"weights": str(COCO_WEIGHTS), "tracker": SECOND_TRACKER, "conf": SECOND_CONF,
           "frames": [{"frame": f, "boxes": frames[f]} for f in sorted(frames)]}
    out_path.write_text(json.dumps(out))
    return out


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def disagreement(boxes1: list, boxes2: list, thr: float = 0.5) -> dict:
    """Greedy mutual matching at IoU >= thr. Rate = matches / max(n1, n2) so a box missing on
    either side counts. Returns n1, n2, matches, only_pipeline, only_second, disagreement."""
    n1, n2 = len(boxes1), len(boxes2)
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(boxes1) for j, b in enumerate(boxes2)), reverse=True)
    used1, used2, m = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used1 or j in used2:
            continue
        used1.add(i); used2.add(j); m += 1
    denom = max(n1, n2)
    return {"n_pipeline": n1, "n_second": n2, "matches": m,
            "only_pipeline": [i for i in range(n1) if i not in used1],
            "only_second": [j for j in range(n2) if j not in used2],
            "disagreement": round(1 - m / denom, 4) if denom else 0.0}

"""sportvu.label_check: the key-edge distance is recovered on a synthetic GSW-coloured frame."""
import cv2
import numpy as np

from sportvu.label_check import EDGE_MAX_PX, key_edges, verdict

P = np.array([[14.0, 0.0, -300.0], [0.0, 12.0, 60.0], [0.0, 0.0, 1.0]])   # court ft -> px (right key in frame)
H = np.linalg.inv(P)                                                     # px -> court ft, as H_truth
WOOD = (18, 80, 200)                                                     # OpenCV HSV
GOLD = (22, 230, 210)


def frame_with_key(dx: float, dy: float, paint: bool = True) -> np.ndarray:
    hsv = np.zeros((720, 1280, 3), np.uint8); hsv[:] = WOOD
    if paint:
        key = cv2.perspectiveTransform(np.array([[[75, 17]], [[94, 17]], [[94, 33]], [[75, 33]]], np.float64), P).reshape(-1, 2)
        cv2.fillPoly(hsv, [np.round(key + [dx, dy]).astype(np.int32)], GOLD)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_aligned_label_is_kept():
    r = key_edges(frame_with_key(0, 0), H)
    assert r["status"] == "measured" and r["edge_px"] <= 1.5
    assert verdict(r) is None


def test_offset_label_reports_the_offset_and_is_rejected():
    r = key_edges(frame_with_key(0, 20), H)          # lane lines 20 px off, free-throw line on
    assert r["status"] == "measured" and abs(r["edge_px"] - 20) <= 1.5
    assert r["edge_px"] > EDGE_MAX_PX and verdict(r) == "paint_edge_high"


def test_key_in_frame_without_paint_is_unmatched():
    r = key_edges(frame_with_key(0, 0, paint=False), H)
    assert r["status"] == "unmatched" and verdict(r) == "paint_edge_unmatched"

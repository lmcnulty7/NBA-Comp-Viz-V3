"""
court/homography.py — image⇄court homography via matched keypoints.

A broadcast camera is a perspective projection; a 3×3 homography H inverts it.
Given ≥4 matched pairs (pixel ↔ court-feet), H maps any pixel to court coords
(and H_inv back). Math reused from the old project (it was correct); the only
thing that was ever missing upstream was reliable keypoints.

Convention:  court_pt = H · pixel_pt   (homogeneous);  H_inv: court → pixel.
"""
from __future__ import annotations

import logging

import numpy as np

import config
from .geometry import COURT_KEYPOINTS

logger = logging.getLogger(__name__)


class CourtHomography:
    def __init__(self, ransac_threshold: float = None):
        self.ransac_threshold = config.RANSAC_REPROJ_THRESHOLD if ransac_threshold is None else ransac_threshold
        self._H = None        # pixel → court
        self._H_inv = None    # court → pixel
        self.quality = None   # mean reprojection error on inliers (feet)
        self.n_inliers = 0

    @property
    def is_valid(self) -> bool:
        return self._H is not None

    def compute(self, pixel_pts: np.ndarray, court_pts: np.ndarray) -> bool:
        import cv2

        pixel_pts = np.asarray(pixel_pts, np.float32)
        court_pts = np.asarray(court_pts, np.float32)
        if len(pixel_pts) < config.MIN_KEYPOINTS_FOR_H:
            return False
        H, mask = cv2.findHomography(pixel_pts, court_pts, method=cv2.RANSAC,
                                     ransacReprojThreshold=self.ransac_threshold)
        if H is None or mask is None:
            return False
        m = mask.ravel().astype(bool)
        n_in = int(m.sum())
        if n_in < config.MIN_KEYPOINTS_FOR_H:   # RANSAC found H but too few inliers → unreliable
            return False
        self._H = H
        self._H_inv = np.linalg.inv(H)
        self.n_inliers = n_in
        reproj = self.to_court_batch(pixel_pts[m])
        errs = np.linalg.norm(reproj - court_pts[m], axis=1)
        self.quality = float(errs.mean()) if len(errs) else None
        return True

    @classmethod
    def from_matrix(cls, H: np.ndarray, quality: float | None = None,
                    n_inliers: int = 0) -> "CourtHomography":
        """Wrap an externally computed H (pixel → court ft), e.g. from the snap-tracker."""
        hom = cls()
        hom._H = np.asarray(H, np.float64)
        hom._H_inv = np.linalg.inv(hom._H)
        hom.quality = quality
        hom.n_inliers = n_inliers
        return hom

    def to_court_batch(self, pixel_pts: np.ndarray) -> np.ndarray:
        import cv2
        pts_in = np.asarray(pixel_pts, np.float32)
        if self._H is None or len(pts_in) == 0:
            return np.full((len(pts_in), 2), np.nan, np.float32)
        return cv2.perspectiveTransform(pts_in.reshape(-1, 1, 2), self._H).reshape(-1, 2)

    def to_pixel_batch(self, court_pts: np.ndarray) -> np.ndarray:
        import cv2
        pts_in = np.asarray(court_pts, np.float32)
        if self._H_inv is None or len(pts_in) == 0:
            return np.full((len(pts_in), 2), np.nan, np.float32)
        return cv2.perspectiveTransform(pts_in.reshape(-1, 1, 2), self._H_inv).reshape(-1, 2)

    def foot_points_to_court(self, bboxes: np.ndarray) -> np.ndarray:
        """(N,4) xyxy boxes → (N,2) court coords of each foot-point (bottom-center)."""
        bboxes = np.atleast_2d(bboxes)
        feet = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2.0, bboxes[:, 3]], axis=1)
        return self.to_court_batch(feet)

    def save(self, path) -> None:
        if self._H is None:
            raise RuntimeError("No H to save.")
        np.save(str(path), self._H)

    def load(self, path) -> None:
        H = np.load(str(path))
        self._H, self._H_inv = H, np.linalg.inv(H)


def homography_from_indexed_keypoints(detected: dict[int, np.ndarray], court_vertices) -> CourtHomography | None:
    """Build H from {keypoint_index: [u, v] pixel} using `court_vertices` (index→court ft)
    as reference. For the 33-point scheme where keypoint index == vertex index."""
    idxs = [i for i in detected if detected[i] is not None]
    if len(idxs) < config.MIN_KEYPOINTS_FOR_H:
        return None
    pixel = np.array([detected[i] for i in idxs], np.float32)
    court = np.array([court_vertices[i] for i in idxs], np.float32)
    hom = CourtHomography()
    return hom if hom.compute(pixel, court) else None


def homography_from_named_keypoints(detected: dict[str, np.ndarray]) -> CourtHomography | None:
    """
    Build H from {keypoint_name: [u, v] pixel} using COURT_KEYPOINTS as reference.
    Returns a valid CourtHomography, or None if too few shared points.
    """
    shared = [k for k in detected if k in COURT_KEYPOINTS and detected[k] is not None]
    if len(shared) < config.MIN_KEYPOINTS_FOR_H:
        # Normal on non-court frames (intro/replay/crowd) — debug, not a warning.
        logger.debug("Only %d usable keypoints (need %d) — no homography", len(shared), config.MIN_KEYPOINTS_FOR_H)
        return None
    pixel = np.array([detected[k] for k in shared], np.float32)
    court = np.array([COURT_KEYPOINTS[k] for k in shared], np.float32)
    hom = CourtHomography()
    return hom if hom.compute(pixel, court) else None

"""
court/mapper.py — CourtMapper: ties the trained court-keypoint model to the
homography and to court-masking.

Now uses the 33-point scheme (court/court33.py, external court-detection-2 model) —
corner-origin court coords (x: 0→94 ft, y: 0→50 ft). Provides per-frame homography,
player court coordinates, and image-space court-masking.

If no homography is found for a frame (too few court keypoints), it degrades
gracefully: tracks keep court_pos=None and nothing is masked.
"""
from __future__ import annotations

import numpy as np

import config
from .court33 import COURT_VERTICES_33, court33_boundary
from .homography import homography_from_indexed_keypoints


class CourtMapper:
    def __init__(self, kp_detector=None, margin_ft: float | None = None,
                 use_tracker: bool | None = None):
        self.use_tracker = (config.COURT_USE_GRID_TRACKER if use_tracker is None
                            else use_tracker)
        self.tracker = None
        self.kp = None
        if self.use_tracker and kp_detector is None:
            from .snap_track import CourtTracker
            self.tracker = CourtTracker()
        else:
            from .detector33 import CourtKeypointDetector33
            self.use_tracker = False
            self.kp = kp_detector or CourtKeypointDetector33()
        self.margin = config.COURT_MARGIN_FT if margin_ft is None else margin_ft
        self.last_hom = None
        self.last_kp = {}        # {index: [u, v]} detected this frame (legacy path only)
        self._hull = None        # pixel-space convex hull of the points supporting H

    def update(self, frame):
        """Compute this frame's homography (pixel → court ft).

        Tracker path (default): grid model → sanity → line-snap → temporal tracking
        (court/snap_track.py). The hull comes from whatever points support the H —
        model keypoints on TRACK frames, matched line points on LINE_TRACK frames —
        so is_extrapolated keeps meaning "outside the evidence region"."""
        import cv2

        if self.tracker is not None:
            self.last_hom = self.tracker.update(frame)
            pts = self.tracker.last_pts
            self._hull = cv2.convexHull(pts) if len(pts) >= 3 else None
            return self.last_hom

        detected = self.kp.detect(frame)
        self.last_kp = detected
        pts = np.array([detected[i] for i in detected], np.float32) if detected else np.empty((0, 2), np.float32)
        self._hull = cv2.convexHull(pts) if len(pts) >= 3 else None
        self.last_hom = homography_from_indexed_keypoints(detected, COURT_VERTICES_33)
        return self.last_hom

    @property
    def keypoint_hull(self):
        """Pixel-space convex hull of the detected court keypoints (None if <3)."""
        return self._hull

    def is_extrapolated(self, foot_pt, margin_px: float = 40.0) -> bool:
        """True if a foot-point lies outside (beyond margin of) the keypoint hull — i.e. its
        court position is an EXTRAPOLATION, not constrained by any nearby landmark, so it is
        unreliable. (100% of off-court projections in the diagnostic were outside the hull.)"""
        import cv2

        if self._hull is None:
            return True
        d = cv2.pointPolygonTest(self._hull.astype(np.float32),
                                 (float(foot_pt[0]), float(foot_pt[1])), True)
        return d < -margin_px      # negative distance = outside; beyond margin = extrapolated

    @property
    def has_homography(self) -> bool:
        return self.last_hom is not None and self.last_hom.is_valid

    @property
    def confident(self) -> bool:
        """High-confidence H, safe to trust for hard masking (enough inliers + low reproj)."""
        h = self.last_hom
        return (self.has_homography and h.n_inliers >= config.COURT_MASK_MIN_INLIERS
                and h.quality is not None and h.quality <= config.COURT_MASK_MAX_REPROJ_FT)

    def court_pos(self, foot_pt) -> np.ndarray:
        """Project a foot-point [u, v] → court [x_ft, y_ft] (corner-origin). For STATS;
        masking uses the image-space polygon (court_pos can blow up near the horizon)."""
        if not self.has_homography:
            return np.array([np.nan, np.nan], np.float32)
        return self.last_hom.to_court_batch(np.atleast_2d(foot_pt))[0]

    def court_polygon_px(self, margin_ft: float | None = None):
        """The court boundary (+margin) projected court→pixel — stable image-space polygon."""
        if not self.has_homography:
            return None
        m = self.margin if margin_ft is None else margin_ft
        px = self.last_hom.to_pixel_batch(court33_boundary(m))
        return px if np.isfinite(px).all() else None

    def is_inside_court(self, foot_pt, margin_ft: float | None = None) -> bool:
        """Is a foot-point inside the projected court polygon? Returns True (keep) unless
        we have a CONFIDENT homography AND the point is clearly outside."""
        import cv2

        if not self.confident:
            return True
        poly = self.court_polygon_px(margin_ft)
        if poly is None:
            return True
        return cv2.pointPolygonTest(poly.astype(np.float32),
                                    (float(foot_pt[0]), float(foot_pt[1])), False) >= 0

    def split_tracks(self, tracks):
        """Set track.court_pos for each track; return (on_court, off_court)."""
        if not self.has_homography:
            for t in tracks:
                t.court_pos = None
            return list(tracks), []
        on, off = [], []
        for t in tracks:
            t.court_pos = self.court_pos(t.foot_point)
            (on if self.is_inside_court(t.foot_point) else off).append(t)
        return on, off

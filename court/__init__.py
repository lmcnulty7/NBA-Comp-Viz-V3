"""
court — court keypoints + homography (Component 3).

  • geometry.py    — 14-keypoint scheme + NBA court geometry + court drawing
  • homography.py  — image⇄court homography (RANSAC) from matched keypoints
  • detector.py    — YOLOv8-pose court-keypoint detector (added after training)

The math is complete; the work is producing reliable keypoints (label → train).
"""
from .geometry import COURT_KEYPOINTS, KEYPOINT_NAMES, is_on_court, keypoint_court_array
from .homography import CourtHomography, homography_from_named_keypoints

__all__ = [
    "COURT_KEYPOINTS", "KEYPOINT_NAMES", "keypoint_court_array", "is_on_court",
    "CourtHomography", "homography_from_named_keypoints",
    "CourtKeypointDetector", "CourtMapper",
]


def __getattr__(name):  # lazy: only import ultralytics when these are used
    if name == "CourtKeypointDetector":
        from .detector import CourtKeypointDetector
        return CourtKeypointDetector
    if name == "CourtMapper":
        from .mapper import CourtMapper
        return CourtMapper
    raise AttributeError(name)

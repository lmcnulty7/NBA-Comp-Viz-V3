"""
detect — player detection + tracking (Component 2).

Streaming interface, mirroring the gate so it drops into a larger pipeline:
    tracker = PlayerTracker()
    for frame in video:                       # contiguous frames
        tracks = tracker.update(frame, idx, ts)   # -> list[Track]
"""
from .camera_cut import CameraCutDetector
from .tracker import PlayerTracker
from .types import Track

__all__ = ["PlayerTracker", "Track", "CameraCutDetector"]

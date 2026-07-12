"""videoseq.py — sequential-first frame access for sampled video scans.

Run-9 postmortem: the 720p60 game's builds still hit the 60-min wall AFTER the
fps-aware stride fix, because per-sample `cap.set(CAP_PROP_POS_FRAMES, i)` makes
FFmpeg seek to the previous keyframe and re-decode forward EVERY time. YouTube
streams carry 3–7 s GOPs — at 60fps that is ~100–400 re-decoded frames per
sample, so decode (not inference) dominated and doubled with fps. Reading
forward with grab() decodes each frame exactly once; a real seek is only worth
it for jumps longer than about one GOP.
"""
from __future__ import annotations

import cv2

# Beyond ~one YouTube GOP a keyframe seek beats grabbing forward frame-by-frame.
SEEK_MIN_GAP = 300


class SeqReader:
    """Monotonic-friendly frame reader: grab() forward for small gaps, true
    seek only for jumps > SEEK_MIN_GAP or backwards. Drop-in for the
    cap.set(POS_FRAMES, i) + read() pattern."""

    def __init__(self, cap: cv2.VideoCapture, pos: int = 0):
        self.cap = cap
        self.next = pos          # frame index the decoder will yield next

    def read(self, idx: int):
        """Return (ok, frame) for frame index idx."""
        if idx < self.next or idx - self.next > SEEK_MIN_GAP:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.next = idx
        while self.next < idx:
            if not self.cap.grab():
                return False, None
            self.next += 1
        ok, frame = self.cap.read()
        if ok:
            self.next = idx + 1
        return ok, frame

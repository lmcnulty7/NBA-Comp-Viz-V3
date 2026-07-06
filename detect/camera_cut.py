"""
detect/camera_cut.py — detect broadcast camera cuts from track turnover.

Why this exists: BoT-SORT keeps track IDs alive indefinitely (persist=True). When
the broadcast cuts to a different angle, almost all of the previous players vanish
and new ones appear — but the tracker would happily glue the old IDs onto the new
players. We watch for that mass-vanish event and tell the caller to reset the
tracker so IDs start fresh on the new shot.

Heuristic recovered from the old project: a cut is flagged when more than
`vanish_frac` of the previously-active track IDs (and at least `min_tracks` of
them) are absent in the current frame. There is an inherent one-frame lag (the cut
frame itself is tracked with stale state, then we reset for the next frame); that's
acceptable and matches the old behavior. A pixel-histogram scene-cut detector would
be crisper and is noted as a future upgrade.
"""
from __future__ import annotations


class CameraCutDetector:
    def __init__(self, vanish_frac: float = 0.80, min_tracks: int = 4):
        self.vanish_frac = vanish_frac
        self.min_tracks = min_tracks
        self._prev_ids: set[int] = set()

    def update(self, current_ids: set[int]) -> bool:
        """Feed this frame's active track IDs; return True if a cut is detected."""
        cut = False
        if len(self._prev_ids) >= self.min_tracks:
            vanished = self._prev_ids - current_ids
            if len(vanished) / len(self._prev_ids) >= self.vanish_frac:
                cut = True
        self._prev_ids = set(current_ids)
        return cut

    def reset(self) -> None:
        self._prev_ids = set()

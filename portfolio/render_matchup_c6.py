"""C6 wrapper: re-render curry_q1_clip span set_start_frame==0 with the EXCLUDED
banner relocated to the bottom edge via a cv2 proxy passed as mm.cv2.
Never calls mm.main(); writes nothing under the repo."""
from pathlib import Path
import sys, json
import cv2 as _cv2
import matchup_metrics as mm
import config

OUT = str(Path(__file__).resolve().parent / "build" / "c6_raw.mp4")


class Cv2Proxy:
    def __init__(self, real):
        self._real = real
        self.n_rect = 0
        self.n_text = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def rectangle(self, img, pt1, pt2, color, thickness=1, *a, **k):
        h, w = img.shape[:2]
        if tuple(pt1) == (0, 0) and tuple(pt2) == (w, 24) and tuple(color) == (0, 0, 120) and thickness == -1:
            self.n_rect += 1
            return self._real.rectangle(img, (0, h - 26), (w, h), color, thickness, *a, **k)
        return self._real.rectangle(img, pt1, pt2, color, thickness, *a, **k)

    def putText(self, img, text, org, font, scale, color, *a, **k):
        if isinstance(text, str) and text.startswith("frame") and "EXCLUDED" in text:
            h = img.shape[0]
            self.n_text += 1
            org = (org[0], h - 8)
        return self._real.putText(img, text, org, font, scale, color, *a, **k)


proxy = Cv2Proxy(_cv2)
mm.cv2 = proxy

frames, poss, cols = mm.load("curry_q1_clip")
assert not (config.PROJECT_ROOT / "data" / "pbp" / "curry_q1_clip_orientation.json").exists()
stride = poss.get("stride", 3)
dt = stride / 30.0
span = [s for s in poss["spans"] if s.get("set_start_frame") == 0][0]
assert span["kind"] == "halfcourt" and span.get("metrics_eligible"), span
rec, asg, invalid = mm.possession_metrics(frames, span, dt)
core = sorted(f for f in frames if rec["core_start_frame"] <= f <= rec["core_end_frame"])
print("core", rec["core_start_frame"], rec["core_end_frame"], "n_core_frames", len(core),
      "n_invalid", len(invalid), "json_excluded", rec.get("frames_excluded_team_gt5"))
mm.render_review(frames, [rec], {rec["core_start_frame"]: asg},
                 {rec["core_start_frame"]: invalid}, cols, OUT, 10.0)
print("proxy intercepts: rect", proxy.n_rect, "text", proxy.n_text)
json.dump({"core": core, "invalid": sorted(int(f) for f in invalid), "n_invalid": len(invalid)},
          open(str(Path(__file__).resolve().parent / "build" / "c6_invalid.json"), "w"))

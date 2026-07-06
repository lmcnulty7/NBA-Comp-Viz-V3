"""
gate — the court-visibility gate package.

Two interchangeable gates expose the SAME interface so either can drop into a
larger pipeline untouched:

    score(frame_bgr) -> P(live) in [0, 1]
    is_court_visible(frame_bgr, threshold=None) -> bool

  • ZeroShotGate   (Approach A) — CLIP zero-shot, no training.
  • TrainedHeadGate(Approach B) — logistic/MLP head on frozen CLIP embeddings.
  • HsvGate                     — the recovered shipped heuristic, as a baseline.
"""
from .hsv_baseline import HsvGate
from .trained_head import TrainedHeadGate, build_head
from .zero_shot import ZeroShotGate

__all__ = ["ZeroShotGate", "TrainedHeadGate", "HsvGate", "build_head"]

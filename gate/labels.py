"""
gate/labels.py — the ONE place that loads labels, and it loads them EXCLUSIVELY
from truth/. This is the single chokepoint that enforces labeling integrity:

  • truth/ is the only source of labels for train / val / test.
  • predicted/ (CLIP's guesses) is NEVER read for any labeling purpose. Grading a
    model on predicted/ would measure CLIP against its own guesses — circular,
    ~100% meaningless, target leakage.
  • If truth/ is empty (or far smaller than predicted/), callers ERROR OUT rather
    than silently falling back to predicted/.

Both train_gate.py and evaluate_gate.py import from here and nowhere else for
labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from .common import list_images


@dataclass
class TruthSet:
    paths: list[Path]      # absolute frame paths
    labels: list[int]      # 0=dead, 1=live  (parallel to paths)
    n_live: int
    n_dead: int


def _assert_not_predicted(paths: list[Path]) -> None:
    """Hard guard: no labeled path may come from predicted/."""
    pred_root = config.PREDICTED_DIR.resolve()
    for p in paths:
        rp = p.resolve()
        if pred_root in rp.parents:
            raise AssertionError(
                f"INTEGRITY VIOLATION: labeled frame came from predicted/ ({rp}). "
                "Labels must come exclusively from truth/."
            )


def load_truth(require: bool = True) -> TruthSet:
    """
    Load human-verified frames from truth/live and truth/dead.

    Parameters
    ──────────
    require : if True, raise when truth/ is empty or smaller than predicted/.
    """
    live = list_images(config.TRUTH_LIVE)
    dead = list_images(config.TRUTH_DEAD)
    paths = live + dead
    labels = [1] * len(live) + [0] * len(dead)

    _assert_not_predicted(paths)

    if require:
        if not paths:
            raise SystemExit(
                "truth/ is EMPTY. Verify frames into data/visibility/truth/{live,dead} "
                "before training/evaluating. predicted/ is NOT a fallback — grading on "
                "CLIP's own guesses is target leakage. (Run presort_frames.py, then move "
                "eyeballed frames into truth/.)"
            )
        n_pred = len(list_images(config.PREDICTED_LIVE)) + len(list_images(config.PREDICTED_DEAD))
        if n_pred > 0 and len(paths) < 0.2 * n_pred:
            raise SystemExit(
                f"truth/ has only {len(paths)} frames vs {n_pred} in predicted/. "
                "This looks like labeling isn't finished. Finish moving verified frames "
                "into truth/ — predicted/ will NOT be used as a fallback."
            )

    return TruthSet(paths=paths, labels=labels, n_live=len(live), n_dead=len(dead))

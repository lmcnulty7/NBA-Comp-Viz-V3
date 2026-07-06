#!/usr/bin/env python
"""
presort_frames.py — Phase 1, step 2.

Run the CLIP zero-shot gate (Approach A) over data/visibility/_unsorted/ and
COPY each frame into predicted/live or predicted/dead based on CLIP's guess.

⚠️  THIS IS A LABELING CONVENIENCE ONLY. predicted/ holds CLIP's GUESSES, not
ground truth. Nothing in predicted/ is ever read by train_gate.py or
evaluate_gate.py. Your job: eyeball predicted/, then MOVE each frame you have
personally verified into the matching truth/ folder (correct CLIP freely — a
frame CLIP called "live" may belong in truth/dead).

Every copied filename is prefixed with CLIP's P(live), e.g.
    plive0.082__clip_10m00_18m00__f000045.jpg
so you can sort by name and immediately spot the low-confidence guesses to
scrutinize first (P(live) near 0.5).

Examples
────────
  python presort_frames.py
  python presort_frames.py --threshold 0.5
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import config
from gate.backbones import get_backbone
from gate.common import get_image_embeddings, list_images
from gate.zero_shot import ZeroShotGate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("presort_frames")


def main() -> None:
    ap = argparse.ArgumentParser(description="Presort _unsorted/ into predicted/ using CLIP zero-shot (GUESSES only).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(live) >= threshold → predicted/live (default 0.5).")
    ap.add_argument("--unsorted", type=Path, default=config.UNSORTED_DIR)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    frames = list_images(args.unsorted)
    if not frames:
        log.error("No frames in %s — run extract_frames.py first.", args.unsorted)
        sys.exit(1)

    config.PREDICTED_LIVE.mkdir(parents=True, exist_ok=True)
    config.PREDICTED_DEAD.mkdir(parents=True, exist_ok=True)

    device = config.get_device()
    log.info("Loading CLIP backbone on %s …", device)
    backbone = get_backbone("clip", device)
    gate = ZeroShotGate(backbone, config.LIVE_PROMPTS, config.DEAD_PROMPTS, threshold=args.threshold)

    log.info("Embedding %d frames …", len(frames))
    emb = get_image_embeddings(frames, backbone, config.emb_cache_path("clip"))
    p_live = gate.score_embeddings(emb)

    n_live = 0
    for path, p in zip(frames, p_live):
        dest_dir = config.PREDICTED_LIVE if p >= args.threshold else config.PREDICTED_DEAD
        n_live += int(p >= args.threshold)
        out_name = f"plive{p:.3f}__{path.name}"
        shutil.copy2(path, dest_dir / out_name)

    log.info("Presorted %d frames → %d predicted-live / %d predicted-dead",
             len(frames), n_live, len(frames) - n_live)
    log.info("predicted/live: %s", config.PREDICTED_LIVE)
    log.info("predicted/dead: %s", config.PREDICTED_DEAD)
    log.info("")
    log.info("⚠️  These are CLIP GUESSES, not labels. Now VERIFY by eye and MOVE")
    log.info("    each confirmed frame into truth/live or truth/dead. Aim for")
    log.info("    >= %d per class. Then: python train_gate.py", config.MIN_PER_CLASS_WARN)


if __name__ == "__main__":
    main()

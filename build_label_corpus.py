#!/usr/bin/env python
"""
build_label_corpus.py — sample a stratified frame corpus for EXTERNAL auto-labeling.

Foundation Refresh stage 1. The corpus feeds outside teacher models (Grounding
DINO / SAM2 for boxes, Claude for adjudication + court landmarks) — the
in-house detector/court models are deliberately kept OUT of the loop, including
here: frames are sampled by TIME, not by any model's opinion, so the corpus
can't inherit the current models' blind spots (self-training inbreeds false
negatives; a model's systematic misses never even surface for verification).

Sampling: uniform-with-jitter across each game's full timeline (seeded).
Dead-ball/crowd/bench frames are kept — they're the negative examples the
detector currently confuses (referee/crowd FPs), and teachers can score
liveness later as metadata rather than a filter.

Output: data/label_corpus/<tag>/f<frame>.jpg + manifest.jsonl
        (one record per frame: tag, source file, frame index, sample seed).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2

import config
from videoseq import SeqReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("label_corpus")

# corpus lives on the Colab Drive (teachers run there); the local mount makes
# writes transparent. Not under data/ — nothing local should depend on it.
CORPUS_DIR = (Path.home() / "Library/CloudStorage"
              / "GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest/label_corpus")
HARVEST_VIDEO = config.PROJECT_ROOT / "data" / "harvest" / "video"


def sources() -> dict[str, Path]:
    """{tag: full-game video} — harvest games with a local full download,
    plus the old-project clips (extra arena/era diversity)."""
    out = {}
    reg = json.loads((HARVEST_VIDEO.parent / "games.json").read_text())
    for tag in reg:
        p = HARVEST_VIDEO / f"{tag}.mp4"
        if p.exists():
            out[tag] = p
    for p in sorted(config.VIDEO_DIR.glob("*.mp4")):
        out[p.stem] = p
    return out


def sample_game(tag: str, src: Path, k: int, rng: random.Random, manifest) -> int:
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < k:
        cap.release()
        return 0
    reader = SeqReader(cap)
    out_dir = CORPUS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # uniform strata with jitter: coverage of the whole broadcast, no clumping
    step = n / k
    idxs = sorted(min(n - 1, int(i * step + rng.uniform(0, step))) for i in range(k))
    saved = 0
    for f in idxs:
        dst = out_dir / f"f{f:07d}.jpg"
        if dst.exists():
            saved += 1
            continue
        ok, frame = reader.read(f)
        if not ok:
            continue
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.write(json.dumps({"tag": tag, "source": src.name, "frame": f}) + "\n")
        saved += 1
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified frame corpus for external auto-labeling.")
    ap.add_argument("--per-game", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if not CORPUS_DIR.parent.is_dir():
        raise SystemExit(f"Drive mount not found at {CORPUS_DIR.parent} — is Google "
                         "Drive for desktop running (lucienmmcnulty account)?")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(CORPUS_DIR / "manifest.jsonl", "a") as manifest:
        for tag, src in sources().items():
            got = sample_game(tag, src, args.per_game, rng, manifest)
            total += got
            log.info("%-24s %4d frames  (%s)", tag, got, src.name)
    log.info("corpus: %d frames → %s", total, CORPUS_DIR)


if __name__ == "__main__":
    main()

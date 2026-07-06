#!/usr/bin/env python
"""
prepare_player_dataset.py — remap basketball-player-detection-3 to a clean class set.

The Roboflow set has 10 classes including 5 player-action variants. For a robust
player/ref/ball/rim detector we collapse the action variants into one `player`
class (the subtle action distinctions are low-value here and would just split
training data / add confusion). Result: 5 classes.

  player  ← player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block
  referee ← referee
  ball    ← ball, ball-in-basket
  rim     ← rim
  number  ← number

Writes a new dataset (images symlinked, labels remapped) so we don't copy 171 MB.
Output: data/external/player_det3_remap/  +  data.yaml.

Usage:  python prepare_player_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import config

SRC = config.PROJECT_ROOT / "data" / "external" / "player_det3"
DST = config.PROJECT_ROOT / "data" / "external" / "player_det3_remap"

# old index → new index   (old order from the dataset's data.yaml)
REMAP = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0,   # player-* → player
         8: 1,                            # referee
         0: 2, 1: 2,                       # ball, ball-in-basket → ball
         9: 3,                             # rim
         2: 4}                             # number
NEW_NAMES = ["player", "referee", "ball", "rim", "number"]


def main():
    n_lbl = n_box = 0
    for split in ("train", "valid", "test"):
        img_src, lbl_src = SRC / split / "images", SRC / split / "labels"
        if not img_src.exists():
            continue
        img_dst, lbl_dst = DST / split / "images", DST / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img in img_src.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            link = img_dst / img.name
            if not link.exists():
                link.symlink_to(img.resolve())

        for lf in lbl_src.glob("*.txt"):
            out_lines = []
            for line in lf.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                old = int(parts[0])
                if old in REMAP:
                    parts[0] = str(REMAP[old])
                    out_lines.append(" ".join(parts))
                    n_box += 1
            (lbl_dst / lf.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            n_lbl += 1

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(NEW_NAMES))
    (DST / "data.yaml").write_text(
        f"path: {DST.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(NEW_NAMES)}\nnames:\n{names_block}\n")
    print(f"Remapped {n_lbl} label files, {n_box} boxes → {len(NEW_NAMES)} classes: {NEW_NAMES}")
    print(f"  → {DST}/  (data.yaml written)")
    print("Next: python train_player_detector.py")


if __name__ == "__main__":
    main()

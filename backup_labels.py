#!/usr/bin/env python
"""
backup_labels.py — zip the irreplaceable human-labeled data to Google Drive.

Git holds the code; these label directories are gitignored (images) and live
on ONE machine. They are the only truly unrecoverable assets in the project:
~1,500 hand-labeled frames across gate/keypoints/teams/clock/possession evals.

Writes nba_harvest/labels_backup/labels_<date>.zip on the mounted Colab Drive
(dated, never overwritten — 2 TB policy: history is cheaper than regret).
Run after any labeling session; idempotent per day.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import config

# every gitignored dir that contains human labels or the pairing indexes that
# make them meaningful (gitignore: "human labels + index pairing are precious")
LABEL_DIRS = [
    "data/visibility/truth",
    "data/court/kp_labels",
    "data/court_labels_33",
    "data/court_labels_grid",
    "data/teams_eval",
    "data/clock_eval",
    "data/poss_eval",
    "data/anchor_truth.json",
]
DRIVE = Path.home() / ("Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com"
                       "/My Drive/nba_harvest/labels_backup")


def main() -> None:
    if not DRIVE.parent.is_dir():
        sys.exit(f"Drive mount not found at {DRIVE.parent} — is Google Drive "
                 "for desktop running (lucienmmcnulty account)?")
    DRIVE.mkdir(parents=True, exist_ok=True)
    out = DRIVE / f"labels_{time.strftime('%Y-%m-%d')}.zip"
    if out.exists():
        sys.exit(f"{out.name} already exists — one backup per day is plenty.")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in LABEL_DIRS:
            p = config.PROJECT_ROOT / rel
            if not p.exists():
                print(f"  (missing, skipped: {rel})")
                continue
            files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
            for f in files:
                z.write(f, f.relative_to(config.PROJECT_ROOT))
                n += 1
            print(f"  {rel}: {len(files)} files")
    print(f"backup: {n} files, {out.stat().st_size / 1e6:.0f} MB → {out}")


if __name__ == "__main__":
    main()

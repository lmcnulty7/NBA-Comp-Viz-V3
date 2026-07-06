#!/usr/bin/env python
"""
colab_train_player.py — train the basketball player detector on a Colab GPU.

Self-contained (no project imports) so it runs on a bare Colab runtime. Trains
~15-30 min on a free GPU vs ~22 h on local MPS.

SETUP on Colab (Runtime → Change runtime type → GPU):
  1. Upload  basketball-player-detection-3.v18i.yolov8.zip  to the runtime
     (it's in your project at data/external/), OR pull it via the Roboflow API.
  2. Run this script:   %run colab_train_player.py
  3. Download  player_runs/train/weights/best.pt  and drop it into your project at
     models/player_detector.pt  — the tracker auto-uses it (tracks the 'player' class).

Same 10→5 class remap as prepare_player_dataset.py:
  player ← player + 4 action variants | referee | ball ← ball + ball-in-basket | rim | number
"""
import glob
import os
import subprocess
import sys
import zipfile

ZIP = "basketball-player-detection-3.v18i.yolov8.zip"   # path to the uploaded zip
ROOT = "player_det3"
EPOCHS, IMGSZ, BATCH = 100, 640, 16

REMAP = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 0: 2, 1: 2, 9: 3, 2: 4}
NEW_NAMES = ["player", "referee", "ball", "rim", "number"]


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

    if not os.path.isdir(ROOT):
        if not os.path.exists(ZIP):
            raise SystemExit(f"Upload {ZIP} to the runtime first (or set ZIP to its path).")
        zipfile.ZipFile(ZIP).extractall(ROOT)

    # remap class indices in-place
    n = 0
    for split in ("train", "valid", "test"):
        for f in glob.glob(f"{ROOT}/{split}/labels/*.txt"):
            out = []
            for line in open(f).read().splitlines():
                p = line.split()
                if p and int(p[0]) in REMAP:
                    p[0] = str(REMAP[int(p[0])])
                    out.append(" ".join(p))
            open(f, "w").write("\n".join(out) + ("\n" if out else ""))
            n += 1
    print(f"remapped {n} label files → {len(NEW_NAMES)} classes {NEW_NAMES}")

    open(f"{ROOT}/data.yaml", "w").write(
        f"path: {os.path.abspath(ROOT)}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NEW_NAMES)}\nnames: {NEW_NAMES}\n")

    from ultralytics import YOLO
    YOLO("yolov8m.pt").train(
        data=f"{ROOT}/data.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
        project="player_runs", name="train", patience=20,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, translate=0.1, scale=0.5, mosaic=1.0)
    print("\nDONE → download player_runs/train/weights/best.pt  →  project models/player_detector.pt")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
build_pose_datasets.py — assemble the two YOLO-pose datasets from the projection-
generated labels (generate_labels.py), with ONE shared train/val split so the
33-vertex and grid models are compared on identical frames.

  data/court_pose33/    images/{train,val}  labels/{train,val}  data.yaml
  data/court_pose_grid/ images/{train,val}  labels/{train,val}  data.yaml
  data/court_pose_split.json                — the shared split (seed 42, 15% val)

Images are hard-linked from data/court_review/images (no extra disk; they become
real copies if the folder is transferred). flip_idx:
  33-vertex — the left/right mirror mapping from the original court_det2 yaml;
  grid      — ix → (NX-1-ix) within each row (court is symmetric in length).

Usage:  /opt/anaconda3/bin/python build_pose_datasets.py
"""
from __future__ import annotations
import glob, json, os, random, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_SRC = os.path.join(ROOT, "data/court_review/images")
SPLIT_JSON = os.path.join(ROOT, "data/court_pose_split.json")
VAL_FRAC, SEED = 0.15, 42
GRID_NX, GRID_NY = 13, 7

FLIP_33 = [27, 28, 29, 30, 31, 32, 26, 24, 25, 21, 22, 23, 18, 19, 20,
           15, 16, 17, 12, 13, 14, 9, 10, 11, 7, 8, 6, 0, 1, 2, 3, 4, 5]
FLIP_GRID = [iy * GRID_NX + (GRID_NX - 1 - ix) for iy in range(GRID_NY) for ix in range(GRID_NX)]

SETS = [("data/court_pose33", "data/court_labels_33", 33, FLIP_33),
        ("data/court_pose_grid", "data/court_labels_grid", GRID_NX * GRID_NY, FLIP_GRID)]


def main():
    stems = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(ROOT, "data/court_labels_33/*.txt"))
                   if not p.endswith("manifest.tsv"))
    if os.path.exists(SPLIT_JSON):
        split = json.load(open(SPLIT_JSON))
        print(f"reusing existing split ({len(split['train'])} train / {len(split['val'])} val)")
    else:
        rng = random.Random(SEED)
        shuffled = stems[:]
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * VAL_FRAC))
        split = {"seed": SEED, "val": sorted(shuffled[:n_val]), "train": sorted(shuffled[n_val:])}
        json.dump(split, open(SPLIT_JSON, "w"), indent=1)
        print(f"new split: {len(split['train'])} train / {len(split['val'])} val -> {SPLIT_JSON}")

    for ds_rel, lbl_rel, n_kp, flip in SETS:
        ds = os.path.join(ROOT, ds_rel)
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            d = os.path.join(ds, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        n = {"train": 0, "val": 0}
        for part in ("train", "val"):
            for stem in split[part]:
                src_img = os.path.join(IMG_SRC, stem + ".jpg")
                src_lbl = os.path.join(ROOT, lbl_rel, stem + ".txt")
                if not (os.path.exists(src_img) and os.path.exists(src_lbl)):
                    continue
                dst_img = os.path.join(ds, "images", part, stem + ".jpg")
                try:
                    os.link(src_img, dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, os.path.join(ds, "labels", part, stem + ".txt"))
                n[part] += 1
        with open(os.path.join(ds, "data.yaml"), "w") as f:
            f.write(f"path: {ds}\ntrain: images/train\nval: images/val\n\n"
                    f"kpt_shape: [{n_kp}, 3]\nflip_idx: {flip}\n\nnc: 1\nnames: ['court']\n")
        print(f"{ds_rel}: {n['train']} train / {n['val']} val   (kpt_shape [{n_kp},3])")


if __name__ == "__main__":
    main()

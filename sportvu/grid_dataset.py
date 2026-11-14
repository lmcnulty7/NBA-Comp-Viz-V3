"""sportvu/grid_dataset.py: the YOLO-pose dataset for ROADMAP R2.3 (grid court keypoints).

V3's court solver is the 13x7 grid-pose model (models/court_grid_snapped.pt, court/grid.py) inside the
snap tracker. R2.3 fine-tunes it on the R1.3 SportVU labels, mixed with its own training set so the four
Oracle (GSW) training games do not crowd out the other arenas it already knows:

  old  data/court_pose_grid train (multi-arena, line-snapped projection labels, ~0.5 px), each image
       OLD_REPEAT times (hard links under different names)
  r13  R1.3 labels kept by the paint check (sportvu.splits.train_labels: train games only), every
       R13_EVERY-th frame of a window (consecutive labelled frames are 0.2 s apart); whole windows go to
       val (VAL_FRAC per game, seeded), so near-duplicate frames never straddle the split

Labels use the exact writer of the old set (generate_labels.visible_court_bbox / keypoint_fields) with
P = inv(H_truth), so the format, bbox rule and visibility flags match. Output: data/sportvu/r23_dataset/
{images,labels}/{train,val} + data.yaml (flip_idx of the grid) and reports/r23_dataset.{json,txt}.
Every image name passes sportvu.splits.refuse_heldout.

  python -m sportvu.grid_dataset            # then: tar -cf data/sportvu/r23_dataset.tar -C data/sportvu r23_dataset
"""
from __future__ import annotations
import json, os, random, shutil
from pathlib import Path
import numpy as np
import config

OUT = config.PROJECT_ROOT / "data" / "sportvu" / "r23_dataset"
OLD = config.PROJECT_ROOT / "data" / "court_pose_grid"
OLD_REPEAT = 3
R13_EVERY = 2
VAL_FRAC = 0.15
SEED = 42
MIN_VISIBLE = 4


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)


def r13_label(row: dict, w: int, h: int):
    """(label line, visible keypoints) for one R1.3 row, or (None, n) when the frame cannot be a label."""
    from court.grid import GRID_FT
    from generate_labels import visible_court_bbox, keypoint_fields
    P = np.linalg.inv(np.asarray(row["H_truth"], np.float64).reshape(3, 3))
    bbox = visible_court_bbox(P, w, h)
    fields, n_vis = keypoint_fields(P, GRID_FT, w, h)
    if bbox is None or n_vis < MIN_VISIBLE:
        return None, n_vis
    return "0 %.6f %.6f %.6f %.6f %s" % (*bbox, " ".join(fields)), n_vis


def build() -> dict:
    from PIL import Image
    from sportvu import splits
    from build_pose_datasets import FLIP_GRID
    if OUT.exists():
        shutil.rmtree(OUT)
    rep = {"item": "ROADMAP R2.3 dataset", "old_repeat": OLD_REPEAT, "r13_every": R13_EVERY, "val_frac": VAL_FRAC,
           "seed": SEED, "counts": {}, "r13": {}, "skipped": {}}
    names = []
    # old set: train repeated, val once
    for split in ("train", "val"):
        n = 0
        for img in sorted((OLD / "images" / split).glob("*")):
            lab = OLD / "labels" / split / (img.stem + ".txt")
            if not lab.exists():
                continue
            for k in range(OLD_REPEAT if split == "train" else 1):
                stem = "old_%s_r%d" % (img.stem, k) if split == "train" else "old_" + img.stem
                _link(img, OUT / "images" / split / (stem + img.suffix))
                _write(OUT / "labels" / split / (stem + ".txt"), lab.read_text())
                names.append(stem); n += 1
        rep["counts"]["old_" + split] = n
    # R1.3 kept labels, window-level val split per game
    rows = list(splits.train_labels())
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], {}).setdefault(r["window"], []).append(r)
    rng = random.Random(SEED)
    vis = []
    for game, wins in sorted(by_game.items()):
        ws = sorted(wins)
        val_w = set(rng.sample(ws, max(1, round(VAL_FRAC * len(ws)))))
        g = {"windows": len(ws), "val_windows": len(val_w), "train": 0, "val": 0}
        for wname in ws:
            split = "val" if wname in val_w else "train"
            for r in sorted(wins[wname], key=lambda r: r["frame"])[::R13_EVERY]:
                img = config.PROJECT_ROOT / r["image"]
                if not img.exists():
                    rep["skipped"]["no_image"] = rep["skipped"].get("no_image", 0) + 1
                    continue
                w, h = Image.open(img).size
                line, n_vis = r13_label(r, w, h)
                if line is None:
                    rep["skipped"]["few_visible_or_no_bbox"] = rep["skipped"].get("few_visible_or_no_bbox", 0) + 1
                    continue
                stem = "r13_%s_%06d" % (wname, r["frame"])
                _link(img, OUT / "images" / split / (stem + ".jpg"))
                _write(OUT / "labels" / split / (stem + ".txt"), line + "\n")
                names.append(stem); g[split] += 1; vis.append(n_vis)
        rep["r13"][game] = g
    rep["counts"]["r13_train"] = sum(g["train"] for g in rep["r13"].values())
    rep["counts"]["r13_val"] = sum(g["val"] for g in rep["r13"].values())
    rep["r13_visible_kp"] = {"p10": int(np.percentile(vis, 10)), "p50": int(np.median(vis)), "p90": int(np.percentile(vis, 90))}
    rep["heldout_checked"] = splits.refuse_heldout(names, "grid_dataset")
    (OUT / "data.yaml").write_text("path: %s\ntrain: images/train\nval: images/val\n\nkpt_shape: [91, 3]\nflip_idx: %s\n\nnc: 1\nnames: ['court']\n"
                                   % (OUT, json.dumps(FLIP_GRID)))
    return rep


def main() -> None:
    rep = build()
    (config.REPORTS_DIR / "r23_dataset.json").write_text(json.dumps(rep, indent=1))
    c = rep["counts"]
    L = ["R2.3 DATASET (data/sportvu/r23_dataset, gitignored): grid 13x7 YOLO-pose",
         "  train: old %d (%d images x %d) + R1.3 %d | val: old %d + R1.3 %d (whole windows)" % (
             c["old_train"], c["old_train"] // OLD_REPEAT, OLD_REPEAT, c["r13_train"], c["old_val"], c["r13_val"])]
    for g, v in rep["r13"].items():
        L.append("    %-14s windows %3d (val %2d) | frames train %4d, val %3d" % (g, v["windows"], v["val_windows"], v["train"], v["val"]))
    L += ["  R1.3 visible grid points per frame p10/p50/p90: %(p10)d/%(p50)d/%(p90)d" % rep["r13_visible_kp"],
          "  skipped: %s | held-out guard: %d names checked, none held out" % (rep["skipped"] or "none", rep["heldout_checked"])]
    (config.REPORTS_DIR / "r23_dataset.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

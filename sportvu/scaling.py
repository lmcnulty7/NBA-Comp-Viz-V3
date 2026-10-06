"""sportvu/scaling.py: data-scaling study for the grid court model (how many frames, games, arenas).

Every arm is a subset of data/sportvu/r23_dataset (sportvu.grid_dataset; the tar is on Drive as r23/), so Colab
builds the arms from the same images and nothing new is uploaded. Every arm gets the same compute:

  train list  OLD_LINES lines of the old Roboflow set (V3's grid set, data/court_pose_grid train) +
              R13_LINES lines of R1.3 SportVU labels, each part resampled with repetition from the arm's pool
              (hard links under distinct names, since Ultralytics keys its label cache by file name)
  val         the R1.3 val windows only (whole windows of the training games); the old val split is not used:
              81% of its images have an augmented twin in train (sportvu/old_set_groups.json)
  recipe      init V3's grid model, EPOCHS epochs of TRAIN_LINES images (about 8k iterations at batch 32),
              SGD + cosine LR to the end, no early stopping, V3's augmentation; the scored weights are last.pt

The primary metric is held-out position error from sportvu.court_replay (exact V3-pipeline court rows on the
held-out game and arena), compared pairwise against arm A_full with sportvu.compare --paired.

R1.3 frames are correlated (0.1 s apart); the unit of variety is the window (254 windows, 42 sections), so arms
vary windows, games and frames per window.

  python -m sportvu.scaling list                       arms and pool sizes (local check)
  python -m sportvu.scaling build <arm> <out_root>     materialise an arm (Colab runner calls this)
"""
from __future__ import annotations
import argparse, json, os, random, re, shutil
from pathlib import Path
import config

DS = config.PROJECT_ROOT / "data" / "sportvu" / "r23_dataset"
GROUPS = Path(__file__).resolve().parent / "old_set_groups.json"
OLD_LINES, R13_LINES = 4737, 4779          # r23's split: 1,579 old images x3 and 4,779 R1.3 frames
EPOCHS, IMGSZ, BATCH = 27, 640, 32
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015")
R13_RE = re.compile(r"^r13_(gsw_[a-z]+_\d{4})_s(\d+)_w(\d+)_(\d{6})\.jpg$")

# arm -> (pool rule, seed). Pool rules over the R1.3 train frames of r23_dataset (every 2nd kept label of a window).
ARMS = {
    "A_full":  {"r13": {"windows": "all", "per_window": "all"}, "old": "all", "seed": 1},
    "B_f1":    {"r13": {"windows": "all", "per_window": 1}, "old": "all", "seed": 1},
    "B_f4":    {"r13": {"windows": "all", "per_window": 4}, "old": "all", "seed": 1},
    "C_bkn":   {"r13": {"games": ["gsw_bkn_2015"], "windows": 48, "per_window": "all"}, "old": "all", "seed": 1},
    "C_cha":   {"r13": {"games": ["gsw_cha_2016"], "windows": 48, "per_window": "all"}, "old": "all", "seed": 1},
    "C_ind":   {"r13": {"games": ["gsw_ind_2016"], "windows": 48, "per_window": "all"}, "old": "all", "seed": 1},
    "C_sac":   {"r13": {"games": ["gsw_sac_2015"], "windows": 48, "per_window": "all"}, "old": "all", "seed": 1},
    "C_all4":  {"r13": {"games": list(GAMES), "windows": 12, "per_game": True, "per_window": "all"}, "old": "all", "seed": 1},
    "D_nocle": {"r13": {"windows": "all", "per_window": "all"}, "old": "no_cle", "seed": 1},
}
# second pass (2026-10-05): training-seed repeats (same data, train_seed differs) and an arena curve from the old
# set: k non-Cleveland arenas (nested random order, clip_arena tags in old_set_groups.json) at fixed compute, so the
# held-out Cleveland arena is never seen; k=0 means no old set (R1.3 fills the whole list).
for _arm, _ts in (("A_full", 2), ("A_full", 3), ("C_all4", 2), ("C_all4", 3)):
    ARMS["%s_s%d" % (_arm, _ts)] = {**ARMS[_arm], "train_seed": _ts}
for _k in (0, 3, 9, "all"):
    ARMS["E_k%s" % _k] = {"r13": {"windows": "all", "per_window": "all"}, "old": "arenas:%s" % _k, "seed": 1}
# third pass (2026-10-05): r23's recipe (up to 100 epochs, patience 20, auto optimizer, best.pt) on the diversity-
# balanced subset (C_all4: 48 windows over 4 games) vs the full data, 3 seeds each (r23 itself is the third F_full seed).
for _ts in (1, 2, 3):
    ARMS["F_c4_s%d" % _ts] = {**ARMS["C_all4"], "train_seed": _ts, "recipe": "r23"}
for _ts in (2, 3):
    ARMS["F_full_s%d" % _ts] = {**ARMS["A_full"], "train_seed": _ts, "recipe": "r23"}
THIRD_PASS = ["F_c4_s1", "F_full_s2", "F_c4_s2", "F_full_s3", "F_c4_s3"]
FIRST_PASS = ["A_full", "B_f1", "B_f4", "C_bkn", "C_cha", "C_ind", "C_sac", "C_all4", "D_nocle"]
SECOND_PASS = ["A_full_s2", "A_full_s3", "C_all4_s2", "C_all4_s3", "E_k0", "E_k3", "E_k9", "E_kall"]


def r13_train(ds: Path) -> dict:
    """{window: [image names sorted by frame]} for the R1.3 train images of the dataset."""
    out = {}
    for n in sorted(os.listdir(ds / "images" / "train")):
        m = R13_RE.match(n)
        if m:
            out.setdefault("%s_s%s_w%s" % m.groups()[:3], []).append(n)
    return out


def old_train(ds: Path) -> list:
    """Unique old-set source images (r23 stores each 3 times as old_<stem>_r<k>.jpg)."""
    return sorted({re.sub(r"_r\d\.jpg$", ".jpg", n) for n in os.listdir(ds / "images" / "train") if n.startswith("old_")})


def pools(arm: str, ds: Path = DS) -> tuple:
    spec, rng = ARMS[arm], random.Random(ARMS[arm]["seed"])
    wins = r13_train(ds)
    r = spec["r13"]
    games = r.get("games", list(GAMES))
    cand = {w: ns for w, ns in wins.items() if w.rsplit("_s", 1)[0] in games}
    if r["windows"] == "all":
        chosen = sorted(cand)
    elif r.get("per_game"):
        chosen = sorted(w for g in games for w in rng.sample(sorted(x for x in cand if x.startswith(g)), r["windows"]))
    else:
        chosen = sorted(rng.sample(sorted(cand), r["windows"]))
    r13 = []
    for w in chosen:
        ns = cand[w]
        if r["per_window"] == "all" or len(ns) <= r["per_window"]:
            r13 += ns
        else:   # evenly spaced frames across the window
            k = r["per_window"]
            r13 += [ns[round(i * (len(ns) - 1) / max(k - 1, 1))] for i in range(k)] if k > 1 else [ns[len(ns) // 2]]
    old = old_train(ds)
    if spec["old"] == "no_cle":
        cle = set(json.loads(GROUPS.read_text())["cle_images"])
        old = [n for n in old if n[len("old_"):] not in cle]
    elif spec["old"].startswith("arenas:"):
        k = spec["old"].split(":")[1]
        g = json.loads(GROUPS.read_text())
        arena_of = lambda n: g["clip_arena"][str(g["source_to_clip"][str(g["image_to_source"][n[len("old_"):]])])]
        arenas = sorted({arena_of(n) for n in old} - {"CLE"})
        order = random.Random(7).sample(arenas, len(arenas))          # nested: k arenas are the first k
        keep = set(order if k == "all" else order[:int(k)])
        old = [n for n in old if arena_of(n) in keep]
    return r13, old, chosen


def _resample(names: list, lines: int, rng: random.Random) -> list:
    """Every name at least floor(lines/len) times, the remainder drawn without replacement."""
    if not names:
        return []
    base, rem = divmod(lines, len(names))
    return names * base + rng.sample(names, rem)


def build(arm: str, out_root: Path, ds: Path = DS) -> Path:
    r13, old, chosen = pools(arm, ds)
    rng = random.Random(1000 + ARMS[arm]["seed"])
    out = out_root / arm
    if out.exists():
        shutil.rmtree(out)
    r13_lines = R13_LINES if old else OLD_LINES + R13_LINES          # no old set: R1.3 fills the list
    lines = [("train", n) for n in _resample(old, OLD_LINES, rng) + _resample(r13, r13_lines, rng)]
    lines += [("val", n) for n in sorted(os.listdir(ds / "images" / "val")) if n.startswith("r13_")]
    seen = {}
    for split, n in lines:
        k = seen.get((split, n), 0); seen[(split, n)] = k + 1
        stem = n[:-4]
        src_img = ds / "images" / split / (n if split == "val" or not n.startswith("old_") else stem + "_r0.jpg")
        src_lab = src_img.parent.parent.parent / "labels" / split / (src_img.stem + ".txt")
        dst = out / "images" / split / ("%s_k%d.jpg" % (stem, k))
        dst.parent.mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        os.link(src_img, dst)
        os.link(src_lab, out / "labels" / split / ("%s_k%d.txt" % (stem, k)))
    yaml = (ds / "data.yaml").read_text().splitlines()
    (out / "data.yaml").write_text("\n".join(["path: " + str(out)] + [l for l in yaml if not l.startswith("path:")]) + "\n")
    meta = {"arm": arm, "spec": ARMS[arm], "r13_frames": len(r13), "r13_windows": len(chosen), "old_images": len(old),
            "train_lines": sum(1 for s, _ in lines if s == "train"), "val_images": sum(1 for s, _ in lines if s == "val")}
    (out / "arm.json").write_text(json.dumps(meta, indent=1))
    return out


def train(arm_dir: Path, init: Path, project: Path, workers: int = 4) -> Path:
    from ultralytics import YOLO
    spec = json.loads((arm_dir / "arm.json").read_text())["spec"]
    seed = spec.get("train_seed", spec["seed"])
    aug = dict(mosaic=0.0, degrees=0.0, translate=0.05, scale=0.2, fliplr=0.5)
    if spec.get("recipe") == "r23":     # colab_train_court.py's R2.3 recipe; scored weights are best.pt
        YOLO(str(init)).train(data=str(arm_dir / "data.yaml"), epochs=100, imgsz=IMGSZ, batch=BATCH, device=0, patience=20,
                              seed=seed, deterministic=True, workers=workers, **aug,
                              project=str(project), name=arm_dir.name, exist_ok=True, plots=False)
        return project / arm_dir.name / "weights" / "best.pt"
    YOLO(str(init)).train(data=str(arm_dir / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
                          optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, cos_lr=True, warmup_epochs=3,
                          patience=1000, seed=seed, deterministic=True, workers=workers, **aug,
                          project=str(project), name=arm_dir.name, exist_ok=True, plots=False)
    return project / arm_dir.name / "weights" / "last.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "build", "train"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--ds", type=Path, default=DS)
    ap.add_argument("--init", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "list":
        for arm in ARMS:
            r13, old, chosen = pools(arm, a.ds)
            print("%-8s R1.3 windows %3d frames %4d (x%.1f) | old %4d (x%.1f)" % (
                arm, len(chosen), len(r13), R13_LINES / max(len(r13), 1), len(old), OLD_LINES / max(len(old), 1)))
    elif a.cmd == "build":
        print(build(a.arm, a.out, a.ds))
    else:
        print(train(a.out / a.arm, a.init, a.out.parent / "runs", a.workers))


if __name__ == "__main__":
    main()

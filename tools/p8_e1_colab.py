"""tools/p8_e1_colab.py: the E1 training runs, leave one game out, ready for Lucien's Colab session (PLAN.md D137,
D155, D156). Nothing runs here on the Mac.

    python -m tools.p8_e1_colab train --corpus <e1_corpus> --fold gsw_bkn_2015 --seed 0 --out <run folder>
    python -m tools.p8_e1_colab colab --fold gsw_bkn_2015 --seed 0        (the Colab bootstrap cell calls this)

Folds (D156): three, named by the held-out game. A fold trains on the other two games' corpus and is scored later, on
the Mac, by tools/p8_e1_score.py on its held-out game (and on hou_cle_2016, reported apart). The held-out game's frames
never enter its fold's training or validation.

Recipe: the production detector's (colab_train_player.py: yolov8m.pt, 100 epochs, image size 640, batch 16,
patience 20), one class, player. One seed first (seed 0); the other two only after the first is scored.

Validation split, for early stopping only: the last VAL_SHARE of each training game's windows in time order (whole
windows). It is not the adoption test (D5 with the pinned scorecard, D156).

Labels: the corpus labels players only (D131 sources 1 and 2); referees and unconfirmed people are background, as in
the human box truth (D156 leaves a referee class to a later, separately registered arm).

Every run writes run_record.json beside the weights: fold, held-out game, training games, the corpus meta.json
(teacher, weights, thresholds, code commit, dataset versions), the split, the seed, the recipe and the library versions
(D150 condition 4).

Checks before training: tools.p8_common.guard on every game, SportVU id and video id (D85 test set and the spent
held-out games); for each training game, images, labels and the corpus manifest's frame count agree; Ultralytics' own
label check (verify_image_label, the function its dataset loader uses) on the train and validation lists finds every
label and drops no image as corrupt. Its found, missing, empty and corrupt counts go into run_record.json.

Colab (`colab`): Drive mounted on the runtime (its folder named by P8_E1_DRIVE_DIR); the corpus tars written by tools/p8_e1_pack.py in DRIVE_CORPUS are
checked against their pack manifest (members and bytes) and extracted straight from the Drive mount to local disk (no
separate copy step); the fold trains on the GPU; complete.json (the sha256 of best.pt and last.pt) is written into the
run folder after training; the run folder goes to DRIVE_RUNS as one tar per run, with <run>.complete.json beside it. A
result tar of the same name already on Drive is renamed (never deleted) before training starts. The cell flushes Drive.

Colab API runtimes (D164): drive.mount fails there ("mount failed", smoke test of 2026-11-24), so `colab` also takes
--corpus-dir and --runs-dir. tools/p8_colab_xfer.py uploads the pack tars and pack_manifest.json over the runtime's
HTTP contents API into --corpus-dir, and downloads the result tar and its complete.json from --runs-dir afterwards.
The checks are the same either way: unpack_checked verifies members and bytes before extracting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
import time
from pathlib import Path

from tools.p8_common import guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_colab"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
BASE, EPOCHS, IMGSZ, BATCH, PATIENCE = "yolov8m.pt", 100, 640, 16, 20
VAL_SHARE = 0.1
# The mounted Drive folder for E1 comes from the runtime's environment, so no cloud mount path is written here.
_DRIVE_DIR = os.environ.get("P8_E1_DRIVE_DIR")
DRIVE_CORPUS = Path(_DRIVE_DIR) / "corpus" if _DRIVE_DIR else None
DRIVE_RUNS = Path(_DRIVE_DIR) / "runs" if _DRIVE_DIR else None
LOCAL = Path("/content/p8_e1")


def fold_games(fold: str) -> list:
    """The training games of a fold: every corpus game but the held-out one."""
    if fold not in GAMES:
        raise SystemExit(f"{WHO}: unknown fold {fold}; folds are named by their held-out game, one of {GAMES}")
    return [g for g in GAMES if g != fold]


def split_windows(stems: list, val_share: float = VAL_SHARE) -> tuple:
    """Image stems "<game>/<window>_f<frame>" -> (train, val): per game, the last val_share of its windows in time
    order (window names sort in time: <game>_sNN_wNN) go to val, whole."""
    by_game = {}
    for s in stems:
        game, rest = s.split("/", 1)
        by_game.setdefault(game, {}).setdefault(rest.rsplit("_f", 1)[0], []).append(s)
    train_s, val_s = [], []
    for game, wins in by_game.items():
        names = sorted(wins)
        n_val = max(1, round(val_share * len(names))) if len(names) > 1 else 0
        for i, w in enumerate(names):
            (val_s if i >= len(names) - n_val else train_s).extend(sorted(wins[w]))
    return train_s, val_s


def fold_stems(corpus: Path, fold: str) -> list:
    keep = set(fold_games(fold))
    stems = sorted(str(p.relative_to(corpus / "images"))[:-4] for p in (corpus / "images").rglob("*.jpg"))
    return [s for s in stems if s.split("/", 1)[0] in keep]


def corpus_ids(meta: dict) -> list:
    return (list(meta["games"]) + [d["sportvu"] for d in meta["datasets"].values()]
            + [d["video_id"] for d in meta["datasets"].values()])


def check_counts(corpus: Path, games: list) -> dict:
    """Per training game, images, labels and the corpus manifest's frame count must agree -> the counts."""
    counts = json.loads((corpus / "manifest.json").read_text())["counts"]
    out = {}
    for g in games:
        imgs = len(list((corpus / "images" / g).glob("*.jpg")))
        labs = len(list((corpus / "labels" / g).glob("*.txt")))
        want = counts[g]["frames"]
        if not imgs == labs == want:
            raise SystemExit(f"{WHO}: {g}: {imgs} images, {labs} labels, manifest {want} frames")
        out[g] = imgs
    return out


def label_check(im_files: list) -> dict:
    """Ultralytics' own image and label verification over a split -> found, missing, empty and corrupt counts."""
    from ultralytics.data.utils import img2label_paths, verify_image_label
    tot = {"images": len(im_files), "found": 0, "missing": 0, "empty": 0, "corrupt": 0}
    for im, lb in zip(im_files, img2label_paths(im_files)):
        r = verify_image_label((im, lb, "", False, 1, 0, 0, False))
        nm, nf, ne, nc = r[5:9]
        tot["missing"] += nm; tot["found"] += nf; tot["empty"] += ne; tot["corrupt"] += nc
    return tot


def prepare(corpus: Path, fold: str, seed: int, out: Path) -> dict:
    """The fold's split files, data.yaml and run_record.json (no training) -> the record."""
    meta = json.loads((corpus / "meta.json").read_text())
    refuse_cv_test(corpus_ids(meta), WHO)
    guard(corpus_ids(meta), WHO)
    games = fold_games(fold)
    image_counts = check_counts(corpus, games)
    train_s, val_s = split_windows(fold_stems(corpus, fold))
    if any(s.split("/", 1)[0] == fold for s in train_s + val_s):
        raise SystemExit(f"{WHO}: held-out game {fold} reached its own fold")
    out.mkdir(parents=True, exist_ok=True)
    for name, lst in (("train", train_s), ("val", val_s)):
        (out / f"{name}.txt").write_text("".join(f"{corpus / 'images' / s}.jpg\n" for s in lst))
    (out / "data.yaml").write_text(f"train: {out / 'train.txt'}\nval: {out / 'val.txt'}\nnames:\n  0: player\n")
    record = {"rule": "D155, D156 (leave one game out), D150 (4)", "fold": fold, "held_out_game": fold,
              "training_games": games, "corpus_meta": meta,
              "corpus_counts": {g: c for g, c in json.loads((corpus / "manifest.json").read_text())["counts"].items()
                                if g in games},
              "split": {"train": len(train_s), "val": len(val_s), "val_share": VAL_SHARE}, "image_counts": image_counts,
              "recipe": {"base": BASE, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "patience": PATIENCE},
              "seed": seed}
    (out / "run_record.json").write_text(json.dumps(record, indent=1))
    return record


def checked_labels(out: Path, split: dict) -> dict:
    """label_check on the split lists -> counts; SystemExit when an image is corrupt or a label is not found."""
    res = {}
    for name in ("train", "val"):
        files = [l for l in (out / f"{name}.txt").read_text().splitlines() if l]
        c = label_check(files)
        if c["corrupt"] or c["found"] != split[name]:
            raise SystemExit(f"{WHO}: {name}: Ultralytics found {c['found']} labels for {split[name]} images, "
                             f"{c['corrupt']} corrupt")
        res[name] = c
    return res


def weights_hashes(out: Path) -> dict:
    w = out / "train" / "weights"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(w.glob("*.pt"))}


def train(corpus: Path, fold: str, seed: int, out: Path, device: str = "0") -> Path:
    record = prepare(corpus, fold, seed, out)
    import ultralytics
    from ultralytics import YOLO
    record["ultralytics"] = ultralytics.__version__
    record["label_check"] = checked_labels(out, record["split"])
    (out / "run_record.json").write_text(json.dumps(record, indent=1))
    random.seed(seed)
    YOLO(BASE).train(data=str(out / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
                     seed=seed, deterministic=True, device=device, project=str(out), name="train", exist_ok=True)
    hashes = weights_hashes(out)
    if "best.pt" not in hashes:
        raise SystemExit(f"{WHO}: training wrote no best.pt")
    (out / "complete.json").write_text(json.dumps({"fold": fold, "seed": seed, "weights": hashes}, indent=1))
    return out


def run_name(fold: str, seed: int) -> str:
    return f"e1_{fold}_seed{seed}"


def set_aside(path: Path) -> Path | None:
    """Rename an existing file to <name>.prev-<UTC time> (never delete) -> the new path, or None."""
    if not path.exists():
        return None
    new = path.with_name(f"{path.name}.prev-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    path.rename(new)
    return new


def colab(fold: str, seed: int, corpus_dir: Path | None = None, runs_dir: Path | None = None) -> Path:
    """The whole Colab run for one fold: corpus from corpus_dir (Drive by default), checked, training, one result tar
    to runs_dir (Drive by default)."""
    from tools.p8_e1_pack import unpack_checked
    corpus_dir = corpus_dir or DRIVE_CORPUS
    runs_dir = runs_dir or DRIVE_RUNS
    if corpus_dir is None or runs_dir is None:
        raise SystemExit("colab: pass --corpus-dir and --runs-dir, or set P8_E1_DRIVE_DIR to the mounted Drive folder")
    games = fold_games(fold)
    guard(list(GAMES), WHO)
    name = run_name(fold, seed)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for p in (runs_dir / f"{name}.tar", runs_dir / f"{name}.complete.json"):
        moved = set_aside(p)
        if moved:
            print(f"{WHO}: an earlier {p.name} was renamed to {moved.name}", flush=True)
    corpus = LOCAL / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    unpack_checked(corpus_dir, corpus, ["common", *games])
    out = train(corpus, fold, seed, LOCAL / "runs" / name)
    tmp = LOCAL / f"{name}.tar"
    with tarfile.open(tmp, "w") as tf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(out.parent)))
    shutil.copy2(tmp, runs_dir / tmp.name)
    shutil.copy2(out / "complete.json", runs_dir / f"{name}.complete.json")
    print(f"{WHO}: {name} written to {runs_dir / tmp.name}", flush=True)
    return runs_dir / tmp.name


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.p8_e1_colab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", required=True)
    t.add_argument("--fold", required=True, choices=GAMES)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="0")
    c = sub.add_parser("colab")
    c.add_argument("--fold", required=True, choices=GAMES)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--corpus-dir", help="where the pack tars are (Drive by default)")
    c.add_argument("--runs-dir", help="where the result tar goes (Drive by default)")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(Path(a.corpus), a.fold, a.seed, Path(a.out), a.device)
    else:
        colab(a.fold, a.seed, a.corpus_dir and Path(a.corpus_dir), a.runs_dir and Path(a.runs_dir))


if __name__ == "__main__":
    main()

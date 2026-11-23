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

Colab (`colab`): Drive mounted at /content/drive; the corpus tars written by tools/p8_e1_pack.py in DRIVE_CORPUS are
copied to local disk, checked against their pack manifest by member count, unpacked; the fold trains on the GPU; the
run folder (weights, results, run_record.json) goes back to DRIVE_RUNS as one tar per run, and Drive is flushed by the
cell.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path

from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_colab"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
BASE, EPOCHS, IMGSZ, BATCH, PATIENCE = "yolov8m.pt", 100, 640, 16, 20
VAL_SHARE = 0.1
DRIVE_CORPUS = Path("/content/drive/MyDrive/nba_harvest/p8_e1/corpus")
DRIVE_RUNS = Path("/content/drive/MyDrive/nba_harvest/p8_e1/runs")
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


def prepare(corpus: Path, fold: str, seed: int, out: Path) -> dict:
    """The fold's split files, data.yaml and run_record.json (no training) -> the record."""
    meta = json.loads((corpus / "meta.json").read_text())
    refuse_cv_test(list(meta["games"]) + [d["sportvu"] for d in meta["datasets"].values()]
                   + [d["video_id"] for d in meta["datasets"].values()], WHO)
    games = fold_games(fold)
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
              "split": {"train": len(train_s), "val": len(val_s), "val_share": VAL_SHARE},
              "recipe": {"base": BASE, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "patience": PATIENCE},
              "seed": seed}
    (out / "run_record.json").write_text(json.dumps(record, indent=1))
    return record


def train(corpus: Path, fold: str, seed: int, out: Path, device: str = "0") -> Path:
    record = prepare(corpus, fold, seed, out)
    import ultralytics
    from ultralytics import YOLO
    record["ultralytics"] = ultralytics.__version__
    (out / "run_record.json").write_text(json.dumps(record, indent=1))
    random.seed(seed)
    YOLO(BASE).train(data=str(out / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
                     seed=seed, deterministic=True, device=device, project=str(out), name="train", exist_ok=True)
    return out


def run_name(fold: str, seed: int) -> str:
    return f"e1_{fold}_seed{seed}"


def colab(fold: str, seed: int) -> Path:
    """The whole Colab run for one fold: corpus from Drive, checked by count, training, one result tar to Drive."""
    from tools.p8_e1_pack import unpack_checked
    games = fold_games(fold)
    corpus = LOCAL / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    unpack_checked(DRIVE_CORPUS, corpus, ["common", *games])
    name = run_name(fold, seed)
    out = train(corpus, fold, seed, LOCAL / "runs" / name)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL / f"{name}.tar"
    with tarfile.open(tmp, "w") as tf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(out.parent)))
    shutil.copy2(tmp, DRIVE_RUNS / tmp.name)
    print(f"{WHO}: {name} written to {DRIVE_RUNS / tmp.name}", flush=True)
    return DRIVE_RUNS / tmp.name


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
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(Path(a.corpus), a.fold, a.seed, Path(a.out), a.device)
    else:
        colab(a.fold, a.seed)


if __name__ == "__main__":
    main()

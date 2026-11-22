"""tools/p8_e1_colab.py: the E1 training run, ready for Lucien's Colab session (PLAN.md D137, D155). Not run here.

    python -m tools.p8_e1_colab --corpus <e1_corpus folder> --seed 0 --out <run folder>        one seed first

Trains a player detector on the E1 corpus (tools/p8_e1data.py: D131 sources 1 and 2 on the 3 R1.3 games' Mac camera
truth), with the recipe of the production detector (colab_train_player.py: yolov8m.pt, 100 epochs, image size 640,
batch 16, patience 20), so that P8_step0_trackB.md section 6 (b1 scaled by images) prices it: 5,592 images, about
0.5 x 5,592 / 654 = 4.3 T4-hours a seed (an estimate). One seed first; the other two only after the first is scored.

Validation split, for early stopping only: the last VAL_SHARE of each game's windows in time order (whole windows, so
no frame has a neighbour on the other side). It is not the adoption test: adoption is the pinned scorecard on the
development games with a game-bootstrap interval (D5, D150 (3)), and the D85 set once at the end of P8.

Labels hold one class, player. Referees and every box the two sources did not confirm are unlabelled, so they are
background to the trainer, as they are in the human box truth (players only); a referee class would need its own
source, which D131 does not give.

Every run writes run_record.json beside the weights: the corpus meta.json (games, teacher, weights, thresholds, code
commit, dataset versions), the corpus counts, the split, the seed and the library versions (D150 condition 4).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_colab"
BASE, EPOCHS, IMGSZ, BATCH, PATIENCE = "yolov8m.pt", 100, 640, 16, 20
VAL_SHARE = 0.1


def split_windows(stems: list, val_share: float = VAL_SHARE) -> tuple:
    """Image stems "<game>/<window>_f<frame>" -> (train, val): per game, the last val_share of its windows in time
    order (window names sort in time: <game>_sNN_wNN) go to val, whole."""
    by_game = {}
    for s in stems:
        game, rest = s.split("/", 1)
        by_game.setdefault(game, {}).setdefault(rest.rsplit("_f", 1)[0], []).append(s)
    train, val = [], []
    for game, wins in by_game.items():
        names = sorted(wins)
        n_val = max(1, round(val_share * len(names))) if len(names) > 1 else 0
        for i, w in enumerate(names):
            (val if i >= len(names) - n_val else train).extend(sorted(wins[w]))
    return train, val


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.p8_e1_colab")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    corpus, out = Path(a.corpus), Path(a.out)
    meta = json.loads((corpus / "meta.json").read_text())
    refuse_cv_test(list(meta["games"]) + [d["sportvu"] for d in meta["datasets"].values()]
                   + [d["video_id"] for d in meta["datasets"].values()], WHO)
    stems = sorted(str(p.relative_to(corpus / "images"))[:-4] for p in (corpus / "images").rglob("*.jpg"))
    train, val = split_windows(stems)
    out.mkdir(parents=True, exist_ok=True)
    for name, lst in (("train", train), ("val", val)):
        (out / f"{name}.txt").write_text("".join(f"{corpus / 'images' / s}.jpg\n" for s in lst))
    (out / "data.yaml").write_text(f"train: {out / 'train.txt'}\nval: {out / 'val.txt'}\nnames:\n  0: player\n")
    import ultralytics
    from ultralytics import YOLO
    random.seed(a.seed)
    record = {"rule": "D155, D150 (4)", "corpus_meta": meta,
              "corpus_counts": json.loads((corpus / "manifest.json").read_text())["counts"],
              "split": {"train": len(train), "val": len(val), "val_share": VAL_SHARE},
              "recipe": {"base": BASE, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "patience": PATIENCE},
              "seed": a.seed, "ultralytics": ultralytics.__version__}
    (out / "run_record.json").write_text(json.dumps(record, indent=1))
    YOLO(BASE).train(data=str(out / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
                     seed=a.seed, deterministic=True, device=a.device, project=str(out), name=f"seed{a.seed}")


if __name__ == "__main__":
    main()

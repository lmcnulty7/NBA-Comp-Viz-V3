"""tools/p8_e1_m4.py: E1 post-mortem M4, threshold loss or learned loss (docs/research/P8_E1.md, "M4", definitions
committed before any number). Report only: no threshold is adopted from it. Nothing trains.

    python3 -m export.heavy --name ... -- <frozen CV python> -m tools.p8_e1_m4 run --game G     one held-out game
    python3 -m tools.p8_e1_m4 score                                                            -> e1/p8_e1_m4.json

run: the fold's E1 seed 0 best.pt (provenance by tools.p8_e1_score.check_provenance) and r23 (the production
detector, player class) on the corpus JPEGs of that fold's held-out game only, CPU, NMS 0.45, image size 640, once at
confidence 0.10 (the sweep floor). score: M3's rules (tools.p8_e1_postmortem.oracle_counts) at every threshold of the
sweep, E1 minus r23 at the same threshold and E1 at each threshold minus r23 at 0.40, window bootstrap.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_m4"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
FLOOR = 0.10
SWEEP = tuple(round(0.10 + 0.05 * k, 2) for k in range(13))         # 0.10 .. 0.70, fixed before any number
OPERATING = 0.40                                                    # config.PLAYER_CONF (tests assert they agree)
NMS_IOU = 0.45                                                      # config.PLAYER_IOU
IMGSZ = 640
MODELS = ("e1", "r23")


def check_games(games) -> list:
    games = list(games)
    bad = [g for g in games if g not in GAMES]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not a pool game")
    refuse_cv_test(games, WHO)
    guard(games, WHO)
    return games


def weights_of(fold: str) -> Path:
    run = p8_root().parent / "e1_runs" / f"e1_{fold}_seed0.unpacked" / f"e1_{fold}_seed0"
    return run / "train" / "weights" / "best.pt"


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def keep_at(boxes: list, t: float) -> list:
    """Boxes [x1, y1, x2, y2, score] scoring at or above t (a run at confidence t, NMS being greedy by score)."""
    return [b for b in boxes if b[4] >= t - 1e-9]


def key_of(t: float) -> str:
    return f"{t:.2f}"


def frame_counts(boxes_by_model: dict, H_truth, sv, inside, feet_fn, counts_fn) -> dict:
    """{model: boxes} of one frame -> {model: {threshold: counts}} under M3's rules."""
    out = {}
    for m, boxes in boxes_by_model.items():
        out[m] = {}
        for t in SWEEP:
            kept = keep_at(boxes, t)
            out[m][key_of(t)] = counts_fn(feet_fn(kept, H_truth), sv, inside)
    return out


def add_counts(acc: dict, c: dict) -> None:
    for k, v in c.items():
        acc[k] = acc.get(k, 0) + v


def operating_point(curve: dict, ref: dict) -> list:
    """Thresholds whose missed and ghost rates both sit at or below the reference's (point estimates)."""
    return [t for t, r in curve.items()
            if r["missed_rate"] is not None and r["missed_rate"] <= ref["missed_rate"]
            and r["ghost_rate"] is not None and r["ghost_rate"] <= ref["ghost_rate"]]


# ── run (frozen CV python, inside the lock) ─────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import cv2
    import ultralytics
    from ultralytics import YOLO
    from tools.p8_e1_score import check_provenance
    from tools.p8_e1data import corpus_root
    game = check_games([args.game])[0]
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    prov = check_provenance(game, game, 0, weights_of(game))         # fold = held-out game; never a training game
    r23_w = main_checkout() / "models" / "player_detector.pt"
    from tools.p8_e1_score import _sha
    r23_sha = _sha(r23_w)
    if r23_sha != meta["detector"]["weights_sha256"]:
        raise SystemExit(f"{WHO}: the production detector is not the one the corpus used")
    models = {"e1": YOLO(str(weights_of(game))), "r23": YOLO(str(r23_w))}
    frames = [fr for ch in plan["chunks"] for w in ch for fr in w["frames"] if fr["game"] == game]
    preds, secs = {}, {m: 0.0 for m in models}
    for fr in frames:
        stem = f"{fr['window']}_f{int(fr['frame']):06d}"
        img = cv2.imread(str(root / "images" / game / f"{stem}.jpg"))
        if img is None:
            raise SystemExit(f"{WHO}: {stem}.jpg unreadable")
        rec = {}
        for m, model in models.items():
            t0 = time.time()
            r = model.predict(img, conf=FLOOR, iou=NMS_IOU, imgsz=IMGSZ, device="cpu", verbose=False)[0]
            secs[m] += time.time() - t0
            rec[m] = [[round(float(v), 2) for v in b.xyxy[0]] + [round(float(b.conf), 5)]
                      for b in r.boxes if r.names[int(b.cls)] == "player"]
        preds[stem] = rec
    out = {"game": game, "fold": game, "seed": 0, "frames": len(preds), "device": "cpu", "conf_floor": FLOOR,
           "nms_iou": NMS_IOU, "imgsz": IMGSZ, "ultralytics": ultralytics.__version__,
           "weights_sha256": {"e1": prov["sha256"], "r23": r23_sha}, "run": prov["run"],
           "seconds_inference": {m: round(s, 1) for m, s in secs.items()}, "preds": preds}
    (p8_root() / "e1").mkdir(parents=True, exist_ok=True)
    atomic_write(p8_root() / "e1" / f"m4_preds__{game}.json", json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "preds"}, indent=1))
    return out


# ── score (system python) ───────────────────────────────────────────────────────────────────────────────────────────
def summarise(per_w: dict) -> dict:
    """per_w: {window: {model: {threshold: counts}}} -> curves, both comparisons and the operating point."""
    from tools.p8_e1_postmortem import ORACLE_KEYS, oracle_boot, oracle_rates
    tot = {m: {k: {f: sum(per_w[w][m][k][f] for w in per_w) for f in ORACLE_KEYS} for k in map(key_of, SWEEP)}
           for m in MODELS}
    curves = {m: {k: oracle_rates(tot[m][k]) for k in tot[m]} for m in MODELS}
    op = key_of(OPERATING)
    same, vs_op = {}, {}
    for k in map(key_of, SWEEP):
        flat = {w: {"e1": per_w[w]["e1"][k], "r23": per_w[w]["r23"][k], "r23_op": per_w[w]["r23"][op]} for w in per_w}
        same[k] = oracle_boot(flat, "r23", "e1")
        vs_op[k] = oracle_boot(flat, "r23_op", "e1")
    return {"windows": len(per_w), "curves": curves, "e1_minus_r23_same_threshold": same,
            "e1_minus_r23_at_0.40": vs_op,
            "e1_thresholds_at_or_below_r23_at_0.40": operating_point(curves["e1"], curves["r23"][op])}


def cmd_score(args) -> dict:
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    from tools.p8_e1_postmortem import oracle_counts, scorer_inside
    from tools.p8_e1data import corpus_root
    check_games(GAMES)
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    guard([v for g in plan["games"].values() for v in (g["sportvu"], g["video_id"])], WHO)
    corpus = {}
    for k in range(len(plan["chunks"])):
        corpus.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])
    m3 = json.loads((wr / "e1" / "postmortem_m3.json").read_text())
    games, allw, runs = {}, {}, {}
    for tag in GAMES:
        p = wr / "e1" / f"m4_preds__{tag}.json"
        if not p.exists():
            raise SystemExit(f"{WHO}: M4 has not run on {tag}")
        run = json.loads(p.read_text())
        runs[tag] = {k: v for k, v in run.items() if k != "preds"}
        g = plan["games"][tag]
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        per_w, check = {}, {"frames": 0, "boxes_here": 0, "boxes_corpus": 0, "frames_same_count": 0}
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    rec = corpus[f"{tag}/{stem}"]
                    wpx, hpx = rec["size"]
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    inside = scorer_inside(sv, fr["H_truth"], wpx, hpx)
                    fc = frame_counts(run["preds"][stem], fr["H_truth"], sv, inside, T.feet_ft, oracle_counts)
                    acc = per_w.setdefault(fr["window"], {m: {} for m in MODELS})
                    for m in MODELS:
                        for k, c in fc[m].items():
                            add_counts(acc[m].setdefault(k, {}), c)
                    here = len(keep_at(run["preds"][stem]["r23"], OPERATING))
                    there = len([b for b in rec[T.OURS] if b[5] == "player"])
                    check["frames"] += 1
                    check["boxes_here"] += here
                    check["boxes_corpus"] += there
                    check["frames_same_count"] += here == there
        if check["frames"] != run["frames"]:
            raise SystemExit(f"{WHO}: {tag} has {run['frames']} predicted frames, the plan {check['frames']}")
        games[tag] = {**summarise(per_w), "r23_at_0.40_against_corpus_preds": check,
                      "labels_m3": {k: m3["games"][tag]["labels"][k] for k in ("missed_rate", "ghost_rate",
                                                                               "boxes_per_in_frame_player",
                                                                               "duplicate_rate")},
                      "r23_corpus_preds_m3": {k: m3["games"][tag]["ours"][k] for k in ("missed_rate", "ghost_rate")}}
        allw.update({f"{tag}/{k}": v for k, v in per_w.items()})
    pooled = {**summarise(allw),
              "labels_m3": {k: m3["pooled"]["labels"][k] for k in ("missed_rate", "ghost_rate",
                                                                    "boxes_per_in_frame_player", "duplicate_rate")}}
    q1 = pooled["e1_minus_r23_same_threshold"][key_of(OPERATING)]["missed_rate"]
    out = {"what": "E1 post-mortem M4 (docs/research/P8_E1.md): raw E1 and r23 on each fold's held-out corpus frames, "
                   "confidence sweep, M3 rules; report only, no threshold adopted",
           "sweep": [key_of(t) for t in SWEEP], "runs": runs, "games": games, "pooled": pooled,
           "question_1": {"e1_minus_r23_missed_at_0.40": q1,
                          "reproduces_rise": bool(q1["diff"] > 0 and q1["window_interval"][0] > 0),
                          "registered_build_rise": 0.0278},
           "question_2": {"e1_thresholds_at_or_below_r23_at_0.40": pooled["e1_thresholds_at_or_below_r23_at_0.40"]}}
    atomic_write(wr / "e1" / "p8_e1_m4.json", json.dumps(out, indent=1))
    show = {t: {"r23@0.40": v["curves"]["r23"][key_of(OPERATING)]["missed_rate"],
                "e1@0.40": v["curves"]["e1"][key_of(OPERATING)]["missed_rate"],
                "same": v["e1_minus_r23_same_threshold"][key_of(OPERATING)],
                "op_ok": v["e1_thresholds_at_or_below_r23_at_0.40"]} for t, v in {**games, "pooled": pooled}.items()}
    print(json.dumps({"show": show, "q1": out["question_1"], "q2": out["question_2"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_e1_m4")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--game", required=True, choices=GAMES)
    sub.add_parser("score")
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()

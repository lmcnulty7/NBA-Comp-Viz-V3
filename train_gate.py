#!/usr/bin/env python
"""
train_gate.py — Phase 3.

Trains the gate from HUMAN-VERIFIED labels only (truth/). Steps:
  1. Load labels from truth/ (NEVER predicted/ — see gate/labels.py).
  2. Deterministic stratified split into train/val/test (70/15/15), SAVED to
     models/split.json so it is reproducible and re-runnable.
  3. Train Approach B (head on frozen embeddings) on the TRAIN split.
  4. Tune BOTH gates' decision thresholds on the VAL split (objective: minimize
     false negatives — live frames skipped — subject to an FP-rate cap). TEST is
     left untouched here; final reporting lives in evaluate_gate.py.
  5. Persist: trained head, both thresholds, the split, and a config record.

Examples
────────
  python train_gate.py
  python train_gate.py --head mlp --backbone clip --max-fp-rate 0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from gate.backbones import get_backbone
from gate.common import (
    choose_threshold_min_fn,
    evaluate_scores,
    get_image_embeddings,
)
from gate.labels import load_truth
from gate.trained_head import TrainedHeadGate, build_head
from gate.zero_shot import ZeroShotGate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_gate")


def stratified_split(paths, labels, seed):
    """Deterministic 70/15/15 stratified split → dict of index lists."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(paths))
    y = np.array(labels)
    # 15% test
    tr_val, te = train_test_split(idx, test_size=0.15, random_state=seed, stratify=y)
    # 15/85 of the remainder ≈ 15% overall for val
    tr, va = train_test_split(
        tr_val, test_size=0.15 / 0.85, random_state=seed, stratify=y[tr_val]
    )
    return {"train": sorted(tr.tolist()), "val": sorted(va.tolist()), "test": sorted(te.tolist())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the court-visibility gate on truth/ labels.")
    ap.add_argument("--backbone", choices=["clip", "dinov2"], default="clip")
    ap.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--max-fp-rate", type=float, default=0.10,
                    help="Cap on dead-class false-positive rate when tuning thresholds on VAL.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Labels (truth/ only) ───────────────────────────────────────────────
    truth = load_truth(require=True)
    log.info("truth/ counts → live=%d  dead=%d  total=%d", truth.n_live, truth.n_dead, len(truth.paths))
    for name, n in (("live", truth.n_live), ("dead", truth.n_dead)):
        if n < config.MIN_PER_CLASS_HARD:
            log.error("Only %d '%s' frames (< %d). Verify more frames into truth/%s.",
                      n, name, config.MIN_PER_CLASS_HARD, name)
            sys.exit(1)
        if n < config.MIN_PER_CLASS_WARN:
            log.warning("Only %d '%s' frames (< %d recommended). TEST metrics will be noisy.",
                        n, name, config.MIN_PER_CLASS_WARN)

    # ── 2. Deterministic split (saved) ────────────────────────────────────────
    split_idx = stratified_split(truth.paths, truth.labels, args.seed)
    split_record = {
        "seed": args.seed,
        "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "files": {
            sp: [str(truth.paths[i]) for i in idxs] for sp, idxs in split_idx.items()
        },
        "labels": {
            sp: [int(truth.labels[i]) for i in idxs] for sp, idxs in split_idx.items()
        },
    }
    config.SPLIT_PATH.write_text(json.dumps(split_record, indent=2))
    log.info("Saved split → %s (train=%d val=%d test=%d)", config.SPLIT_PATH,
             *(len(split_idx[s]) for s in ("train", "val", "test")))

    # ── 3. Embeddings + train Approach B ──────────────────────────────────────
    device = config.get_device()
    log.info("Loading %s backbone on %s …", args.backbone, device)
    backbone = get_backbone(args.backbone, device)

    emb_all = get_image_embeddings(truth.paths, backbone, config.emb_cache_path(args.backbone))
    y_all = np.array(truth.labels)

    tr, va = split_idx["train"], split_idx["val"]
    X_tr, y_tr = emb_all[tr], y_all[tr]
    X_va, y_va = emb_all[va], y_all[va]

    log.info("Training '%s' head on %d train embeddings (dim=%d) …", args.head, len(tr), emb_all.shape[1])
    clf = build_head(args.head, seed=args.seed)
    clf.fit(X_tr, y_tr)
    trained = TrainedHeadGate(clf, backbone=None, meta={
        "backbone": args.backbone, "head": args.head, "embed_dim": int(emb_all.shape[1])})

    # ── 4. Tune thresholds on VAL ─────────────────────────────────────────────
    pB_va = trained.score_embeddings(X_va)
    thrB = choose_threshold_min_fn(pB_va, y_va, max_fp_rate=args.max_fp_rate)
    trained.threshold = thrB

    thresholds = {"trained": float(thrB), "objective": f"min FN s.t. FP-rate<= {args.max_fp_rate}"}
    val_report = {"trained": evaluate_scores(pB_va, y_va, thrB)}

    # Approach A (zero-shot) — text backbone required; tune its threshold on VAL too.
    if getattr(backbone, "has_text", False):
        zgate = ZeroShotGate(backbone, config.LIVE_PROMPTS, config.DEAD_PROMPTS)
        pA_va = zgate.score_embeddings(X_va)
        thrA = choose_threshold_min_fn(pA_va, y_va, max_fp_rate=args.max_fp_rate)
        thresholds["zero_shot"] = float(thrA)
        val_report["zero_shot"] = evaluate_scores(pA_va, y_va, thrA)
    else:
        log.warning("Backbone '%s' has no text encoder — zero-shot threshold not tuned "
                    "(evaluate_gate.py will use CLIP for the zero-shot baseline).", args.backbone)

    # HSV band has no tunable threshold; record the recovered operating point.
    thresholds["hsv_band"] = {"min": config.HSV_MIN_FLOOR_FRACTION, "max": config.HSV_MAX_FLOOR_FRACTION}

    # ── 5. Persist artifacts ──────────────────────────────────────────────────
    trained.save(config.HEAD_PATH)
    config.THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))

    import sklearn

    config.CONFIG_RECORD_PATH.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "backbone": args.backbone,
        "clip_model": config.CLIP_MODEL_NAME,
        "dinov2_model": config.DINOV2_MODEL_NAME if args.backbone == "dinov2" else None,
        "head": args.head,
        "embed_dim": int(emb_all.shape[1]),
        "max_fp_rate": args.max_fp_rate,
        "live_prompts": config.LIVE_PROMPTS,
        "dead_prompts": config.DEAD_PROMPTS,
        "counts": {"live": truth.n_live, "dead": truth.n_dead},
        "split_sizes": {s: len(split_idx[s]) for s in ("train", "val", "test")},
        "device": device,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
    }, indent=2))

    # ── Console summary (VAL only; TEST stays untouched) ──────────────────────
    log.info("Saved head → %s", config.HEAD_PATH)
    log.info("Saved thresholds → %s", config.THRESHOLDS_PATH)
    log.info("Saved config record → %s", config.CONFIG_RECORD_PATH)
    log.info("── VAL summary (threshold tuned to minimize FN) ──")
    for name, m in val_report.items():
        log.info("  %-10s thr=%.3f  acc=%.3f  live_recall=%.3f  FN=%d  FP=%d",
                 name, m["threshold"], m["accuracy"], m["live"]["recall"],
                 m["false_negatives_live_skipped"], m["false_positives_deadball_processed"])
    log.info("Next: python evaluate_gate.py   (reports on the untouched TEST split)")


if __name__ == "__main__":
    main()

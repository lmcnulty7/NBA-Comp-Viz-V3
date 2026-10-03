#!/usr/bin/env python
"""
gate_ship_v2.py — train and export the v2 court-visibility gate head (LABEL_PLAN §5, decision 2026-10-02).

Head B from gate_retrain_experiment.py won by the adoption rule (leave-one-source-out: accuracy 0.970 vs 0.957,
FP-rate 0.068 vs 0.098 at recall 0.998, worst source 0.948 vs 0.920; prototype test 0.975 within v1's interval).
This trains it on everything it is allowed to see — all 6,159 human-verified harvest frames + the prototype
train/val split (never the prototype test) — and exports it beside v1, never over it:
  models/trained_head_v2.joblib, models/trained_head_v2_coefs.npz, thresholds.json["v2"]
Production harvesting keeps HEAD_PATH @ 0.35 for the published sample; v2 is the gate for future games.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import joblib
import config
from gate.common import get_image_embeddings
from gate.backbones import ClipBackbone
from gate.trained_head import build_head, TrainedHeadGate
from gate.labels import load_truth

DRIVE = Path.home() / "Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest"
SHEET = config.PROJECT_ROOT / "data" / "gate_sheet"
GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]

def metrics(prob, y, t):
    pred = prob >= t; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"threshold": float(t), "recall": round(tp / (tp + fn), 4), "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "fp_rate": round(fp / (fp + tn), 4) if fp + tn else None, "accuracy": round((tp + tn) / len(y), 4), "fn": fn, "fp": fp}

def pick_thr(prob, y, max_fpr=0.10):
    ok = [m for m in (metrics(prob, y, t) for t in GRID) if m["fp_rate"] is not None and m["fp_rate"] <= max_fpr]
    return min(ok, key=lambda m: (m["fn"], -m["threshold"]))["threshold"] if ok else 0.5

items = []
for ix, lb in (("index.json", "labels.json"), ("index_rest.json", "labels_rest.json")):
    I = json.loads((SHEET / ix).read_text()); L = json.loads((SHEET / lb).read_text()); items += [{**d, "wide": L[d["key"]]["wide"]} for d in I if d["key"] in L]
paths = [DRIVE / "label_corpus" / d["tag"] / ("f%07d.jpg" % d["frame"]) for d in items]; y = np.array([int(d["wide"]) for d in items])
backbone = ClipBackbone(config.CLIP_MODEL_NAME, config.get_device())
X = get_image_embeddings(paths, backbone, SHEET / "emb_cache_clip.pkl")
truth = load_truth(); Xp = get_image_embeddings(truth.paths, backbone, config.emb_cache_path("clip")); yp = np.array(truth.labels)
test_keys = {str(Path(p).resolve()) for p in json.loads(config.SPLIT_PATH.read_text())["files"]["test"]}
is_test = np.array([str(Path(p).resolve()) in test_keys for p in truth.paths])

Xb = np.vstack([X, Xp[~is_test]]); yb = np.concatenate([y, yp[~is_test]])
clf = build_head("logreg", seed=42).fit(Xb, yb)
thr = pick_thr(clf.predict_proba(Xb)[:, 1], yb)
meta = {"version": "v2", "trained": "2026-10-02", "recipe": "LogisticRegression C=1.0 class_weight=balanced on L2-normalised CLIP ViT-B/32 embeddings",
        "train": {"harvest_frames": int(len(y)), "harvest_wide": int(y.sum()), "prototype_train_val": int((~is_test).sum()), "sources": 21},
        "threshold_objective": "min FN s.t. FP-rate <= 0.10, chosen on the training set",
        "experiment": "reports/gate_retrain_experiment.json (leave-one-source-out: acc 0.970, FP-rate 0.068 @ recall 0.998; worst source 0.948; prototype test 0.975)",
        "status": "gate for future harvesting; published numbers were produced with trained_head.joblib @ 0.35"}
gate = TrainedHeadGate(clf=clf, threshold=thr, meta=meta)
out = config.HEAD_V2_PATH; out.parent.mkdir(exist_ok=True)
joblib.dump({"clf": clf, "threshold": thr, "meta": meta}, out)
np.savez(out.with_name(out.stem + "_coefs.npz"), coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)
th = json.loads(config.THRESHOLDS_PATH.read_text()); th["v2"] = thr; th["v2_objective"] = meta["threshold_objective"]; config.THRESHOLDS_PATH.write_text(json.dumps(th, indent=2))

# verify: reload through the standard loader (npz path) and reproduce the fit on both sets
g2 = TrainedHeadGate.load(config.HEAD_V2_PATH)
assert g2.meta.get("src") == "npz", g2.meta
p_h = g2.score_embeddings(X); p_t = g2.score_embeddings(Xp[is_test])
print("v2 saved ->", out.name, "+", out.stem + "_coefs.npz", "| threshold %.2f" % g2.threshold)
print("  training-set fit (harvest, in-sample, optimistic): %s" % metrics(p_h, y, g2.threshold))
print("  prototype test (158, never seen): %s" % metrics(p_t, yp[is_test], g2.threshold))
print("  v1 untouched:", TrainedHeadGate.load(config.HEAD_PATH).threshold)

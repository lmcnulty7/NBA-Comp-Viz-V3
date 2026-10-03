#!/usr/bin/env python
"""
gate_retrain_experiment.py — retrain-vs-recalibrate for the court-visibility gate (LABEL_PLAN §5).

Question: on harvest-domain footage, is a head RETRAINED on the 6,159 human-verified harvest
frames better than the CURRENT head (trained on 1,050 prototype-clip frames) with a recalibrated
threshold? Evaluated leave-one-source-out over the 21 harvest sources; every threshold is chosen
on training folds only (objective: min FN s.t. FP-rate <= 0.10, the gate's original objective).
Regression check: heads trained on harvest data must not collapse on the original 158-frame
prototype test split (current head: 0.987 there).

Nothing here changes production. Output: reports/gate_retrain_experiment.{json,txt}
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import config
from gate.common import get_image_embeddings
from gate.backbones import ClipBackbone
from gate.trained_head import build_head
from gate.labels import load_truth
from label_matchups import wilson95

DRIVE = Path.home() / "Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest"
SHEET = config.PROJECT_ROOT / "data" / "gate_sheet"
GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]


def metrics(prob, y, t):
    pred = prob >= t
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None, "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "fp_rate": round(fp / (fp + tn), 4) if fp + tn else None, "accuracy": round((tp + tn) / len(y), 4)}


def pick_thr(prob, y, max_fpr=0.10):
    ok = [m for m in (metrics(prob, y, t) for t in GRID) if m["fp_rate"] is not None and m["fp_rate"] <= max_fpr]
    return min(ok, key=lambda m: (m["fn"], -m["threshold"]))["threshold"] if ok else 0.5


def main():
    t0 = time.time()
    items = []
    for ix, lb in (("index.json", "labels.json"), ("index_rest.json", "labels_rest.json")):
        I = json.loads((SHEET / ix).read_text()); L = json.loads((SHEET / lb).read_text())
        items += [{**d, "wide": L[d["key"]]["wide"]} for d in I if d["key"] in L]
    paths = [DRIVE / "label_corpus" / d["tag"] / ("f%07d.jpg" % d["frame"]) for d in items]
    y = np.array([int(d["wide"]) for d in items]); groups = np.array([d["tag"] for d in items]); cur = np.array([d["score"] for d in items])
    print("harvest set: %d frames, %d wide, %d sources" % (len(y), y.sum(), len(set(groups))), flush=True)

    backbone = ClipBackbone(config.CLIP_MODEL_NAME, config.get_device())
    X = get_image_embeddings(paths, backbone, SHEET / "emb_cache_clip.pkl")
    print("embeddings: %s in %.0f s" % (X.shape, time.time() - t0), flush=True)

    truth = load_truth(); Xp = get_image_embeddings(truth.paths, backbone, config.emb_cache_path("clip")); yp = np.array(truth.labels)
    split = json.loads(config.SPLIT_PATH.read_text())["files"]
    test_keys = {str(Path(p).resolve()) for p in split["test"]}
    is_test = np.array([str(Path(p).resolve()) in test_keys for p in truth.paths])
    print("prototype set: %d frames, test split %d" % (len(yp), is_test.sum()), flush=True)

    rep = {"harvest_n": int(len(y)), "harvest_wide": int(y.sum()), "sources": int(len(set(groups))), "prototype_test_n": int(is_test.sum())}
    # ---- current head on harvest (its scores were computed by gate_sheet.py)
    rep["current_head_on_harvest"] = {"%.2f" % t: metrics(cur, y, t) for t in (0.35, 0.42, 0.47, 0.55, 0.70)}
    rep["current_head_on_harvest"]["objective_pick"] = metrics(cur, y, pick_thr(cur, y))   # NOTE: picked on the same data — optimistic by construction

    # ---- leave-one-source-out: A = harvest-only head, B = harvest + prototype train/val head
    tags = sorted(set(groups)); oof = {"A": np.zeros(len(y)), "B": np.zeros(len(y))}; thr_fold = {"A": np.zeros(len(y)), "B": np.zeros(len(y))}
    per_source = []
    for tag in tags:
        te = groups == tag; tr = ~te
        for name, Xtr, ytr in (("A", X[tr], y[tr]), ("B", np.vstack([X[tr], Xp[~is_test]]), np.concatenate([y[tr], yp[~is_test]]))):
            clf = build_head("logreg", seed=42).fit(Xtr, ytr)
            thr = pick_thr(clf.predict_proba(Xtr)[:, 1], ytr)
            oof[name][te] = clf.predict_proba(X[te])[:, 1]; thr_fold[name][te] = thr
        per_source.append({"source": tag, "n": int(te.sum()), "wide": int(y[te].sum()),
                           "current@0.42": metrics(cur[te], y[te], 0.42)["accuracy"],
                           "A@fold_thr": metrics(oof["A"][te], y[te], thr_fold["A"][te][0])["accuracy"],
                           "B@fold_thr": metrics(oof["B"][te], y[te], thr_fold["B"][te][0])["accuracy"]})
        print("  held out %-22s n=%4d  cur@0.42 %.3f  A %.3f  B %.3f" % (tag, te.sum(), per_source[-1]["current@0.42"], per_source[-1]["A@fold_thr"], per_source[-1]["B@fold_thr"]), flush=True)

    def pooled(name):
        pred = oof[name] >= thr_fold[name]
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        return {"recall": round(tp / (tp + fn), 4), "precision": round(tp / (tp + fp), 4), "fp_rate": round(fp / (fp + tn), 4),
                "accuracy": round((tp + tn) / len(y), 4), "fn": fn, "fp": fp,
                "fold_thresholds": sorted(set(float(t) for t in thr_fold[name])), "at_0.5": metrics(oof[name], y, 0.5)}
    rep["retrained_A_harvest_only_oof"] = pooled("A"); rep["retrained_B_harvest_plus_prototype_oof"] = pooled("B")
    rep["per_source"] = per_source
    rep["per_source_min_accuracy"] = {"current@0.42": min(s["current@0.42"] for s in per_source),
                                      "A": min(s["A@fold_thr"] for s in per_source), "B": min(s["B@fold_thr"] for s in per_source)}

    # ---- regression check on the prototype TEST split (heads trained on everything they are allowed to see)
    A_all = build_head("logreg", seed=42).fit(X, y); tA = pick_thr(A_all.predict_proba(X)[:, 1], y)
    Xb = np.vstack([X, Xp[~is_test]]); yb = np.concatenate([y, yp[~is_test]])
    B_all = build_head("logreg", seed=42).fit(Xb, yb); tB = pick_thr(B_all.predict_proba(Xb)[:, 1], yb)
    rep["prototype_test_check"] = {
        "current_head@0.70_reference": {"accuracy": 0.987, "note": "reports/metrics.json; Wilson 95%% of 156/158 = %s" % wilson95(156, 158)},
        "A_harvest_only": metrics(A_all.predict_proba(Xp[is_test])[:, 1], yp[is_test], tA),
        "B_harvest_plus_prototype": metrics(B_all.predict_proba(Xp[is_test])[:, 1], yp[is_test], tB),
    }
    rep["adoption_rule"] = ("adopt a retrained head only if its leave-one-source-out harvest metrics beat the current head at its best "
                            "recalibrated threshold AND the prototype test split does not fall outside the current head's interval; "
                            "then re-run the PBP canary and the bias audit, since the gate changes what is harvested")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "gate_retrain_experiment.json").write_text(json.dumps(rep, indent=1))
    L = ["GATE RETRAIN vs RECALIBRATE — harvest %d frames / %d sources, leave-one-source-out" % (rep["harvest_n"], rep["sources"])]
    L.append("current head (prototype-trained) on harvest:")
    for t in ("0.35", "0.42", "0.47", "0.55", "0.70"):
        m = rep["current_head_on_harvest"][t]; L.append("  @%s  acc %.3f  recall %.3f  precision %.3f  FP-rate %.3f  FN %d  FP %d" % (t, m["accuracy"], m["recall"], m["precision"], m["fp_rate"], m["fn"], m["fp"]))
    for name, key in (("A  retrained, harvest only", "retrained_A_harvest_only_oof"), ("B  retrained, harvest + prototype", "retrained_B_harvest_plus_prototype_oof")):
        m = rep[key]; L.append("%s (OOF, per-fold objective thresholds %s):" % (name, m["fold_thresholds"]))
        L.append("  acc %.3f  recall %.3f  precision %.3f  FP-rate %.3f  FN %d  FP %d   | at fixed 0.5: acc %.3f" % (m["accuracy"], m["recall"], m["precision"], m["fp_rate"], m["fn"], m["fp"], m["at_0.5"]["accuracy"]))
    L.append("per-source minimum accuracy: %s" % rep["per_source_min_accuracy"])
    pc = rep["prototype_test_check"]; L.append("prototype test split (158): current 0.987 %s | A %.3f @%.2f | B %.3f @%.2f" % (pc["current_head@0.70_reference"]["note"], pc["A_harvest_only"]["accuracy"], pc["A_harvest_only"]["threshold"], pc["B_harvest_plus_prototype"]["accuracy"], pc["B_harvest_plus_prototype"]["threshold"]))
    txt = "\n".join(L); (config.REPORTS_DIR / "gate_retrain_experiment.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()

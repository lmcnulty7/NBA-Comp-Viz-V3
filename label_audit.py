#!/usr/bin/env python
"""
label_audit.py — human audit of ACCEPTED auto-labels (LABEL_SCHEMA rule 3).

The number this produces — auto-label error rate per class, with CIs — is what
makes the 65k-box dataset citable and gates the retrain. Verification-style:
each sampled accepted label renders as a context crop with the claim in the
banner; you judge the claim, you never draw anything.

Strata (the dataset's trust boundaries differ, so they're audited separately):
  agreement_player   teacher∩pipeline boxes — accepted WITHOUT any judge
  adj_player         Claude-kept players (attributes shown: kit/on_court/number)
  scorebug/rim/backboard/referee/ball   Claude-kept object classes

Keys:  y = box, class AND shown attributes all correct
       b = box or class wrong (bad box, wrong object, not this class)
       a = box/class fine but a shown ATTRIBUTE is wrong (players only)
       u = can't tell   ·   q = quit (labels save incrementally)

Modes: --sample N (freeze + prefetch crops locally) · --label · --report
Sample/labels live in data/label_audit/ (add to backup_labels).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import config
from label_matchups import wilson95

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("label_audit")

DRIVE = (Path.home() / "Library/CloudStorage"
         / "GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest")
AUDIT_DIR = config.PROJECT_ROOT / "data" / "label_audit"
SAMPLE_PATH = AUDIT_DIR / "sample.json"
LABELS_PATH = AUDIT_DIR / "labels.json"
CROPS_DIR = AUDIT_DIR / "crops"

# per-stratum quota for a 300 sample; scaled proportionally for other N
QUOTA = {"agreement_player": 100, "adj_player": 80, "scorebug": 30,
         "rim": 30, "backboard": 30, "referee": 20, "ball": 10}


def collect_pool() -> dict[str, list[dict]]:
    """Every accepted label in the dataset, bucketed by stratum."""
    pool = defaultdict(list)
    for p in sorted((DRIVE / "autolabels" / "adjudicated").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            base = {"tag": r["tag"], "frame": r["frame"]}
            for c in r["pre_accepted"]:
                if c["cls"] == "player":
                    pool["agreement_player"].append(
                        {**base, "box": c["box"], "cls": "player", "attrs": {}})
            for i, v in r["verdicts"].items():
                if not v.get("keep"):
                    continue
                cls = v.get("cls")
                stratum = "adj_player" if cls == "player" else cls
                if stratum not in QUOTA:
                    continue
                pool[stratum].append(
                    {**base, "box": r["adjudicated"][int(i)]["box"], "cls": cls,
                     "attrs": {k: v.get(k) for k in
                               ("team_kit", "on_court", "number", "occlusion")
                               if v.get(k) is not None}})
    return pool


def build_sample(n_target: int, seed: int) -> list[dict]:
    import cv2
    rng = random.Random(seed)
    pool = collect_pool()
    scale = n_target / sum(QUOTA.values())
    sample = []
    for stratum, quota in QUOTA.items():
        items = pool.get(stratum, [])
        k = min(len(items), max(1, round(quota * scale)))
        sample += [dict(x, stratum=stratum) for x in rng.sample(items, k)]
    rng.shuffle(sample)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    kept = []
    for j, s in enumerate(sample):
        img = cv2.imread(str(DRIVE / "label_corpus" / s["tag"] / f"f{s['frame']:07d}.jpg"))
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in s["box"]]
        mx, my = max(60, (x2 - x1)), max(40, (y2 - y1) // 2)
        H, W = img.shape[:2]
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        crop = img[cy1:min(H, y2 + my), cx1:min(W, x2 + mx)].copy()
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
        s["key"] = f"{s['tag']}@{s['frame']}#{j}"
        s["crop"] = f"crops/{j:04d}.jpg"
        cv2.imwrite(str(AUDIT_DIR / s["crop"]), crop)
        kept.append(s)
    return kept


def label_loop(sample: list[dict], labels: dict) -> None:
    import cv2
    todo = [s for s in sample if s["key"] not in labels]
    log.info("%d to audit (%d done). y correct · b box/class wrong · "
             "a attribute wrong · u unsure · q quit", len(todo), len(labels))
    for i, s in enumerate(todo):
        crop = cv2.imread(str(AUDIT_DIR / s["crop"]))
        h = max(360, crop.shape[0])
        scale = min(3.0, max(1.0, 320 / crop.shape[0]))
        view = cv2.resize(crop, None, fx=scale, fy=scale)
        attrs = "  ".join(f"{k}={v}" for k, v in s["attrs"].items())
        banner = (f"[{i + 1}/{len(todo)}] {s['stratum']}: claim = {s['cls'].upper()}"
                  + (f"  |  {attrs}" if attrs else "") + "   y/b/a/u q")
        view = cv2.copyMakeBorder(view, 30, 0, 0, max(0, 900 - view.shape[1]),
                                  cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(view, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.imshow("label audit", view)
        key = None
        while key not in list("ybau") + ["q"]:
            key = chr(cv2.waitKey(0) & 0xFF)
        if key == "q":
            cv2.destroyAllWindows()
            log.info("stopped — %d audited", len(labels))
            return
        labels[s["key"]] = {"verdict": key, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        LABELS_PATH.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows()
    log.info("audit complete (%d).", len(labels))


def make_report(sample: list[dict], labels: dict) -> dict:
    def acc(items):
        n = {k: sum(1 for s in items if labels[s["key"]]["verdict"] == k)
             for k in "ybau"}
        judged = n["y"] + n["b"] + n["a"]
        return {"n_judged": judged, "correct": n["y"],
                "box_or_class_wrong": n["b"], "attribute_wrong": n["a"],
                "unsure": n["u"],
                "accuracy": round(n["y"] / judged, 3) if judged else None,
                "wilson95": wilson95(n["y"], judged) if judged else None}
    done = [s for s in sample if s["key"] in labels]
    by = defaultdict(list)
    for s in done:
        by[s["stratum"]].append(s)
    return {"sampled": len(sample), "audited": len(done),
            "overall": acc(done),
            "by_stratum": {k: acc(v) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Human audit of accepted auto-labels.")
    ap.add_argument("--sample", type=int, metavar="N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    if args.sample:
        if labels:
            raise SystemExit(f"{len(labels)} audit labels exist — move "
                             "data/label_audit/ aside for a fresh audit.")
        sample = build_sample(args.sample, args.seed)
        SAMPLE_PATH.write_text(json.dumps(sample, indent=1))
        from collections import Counter
        log.info("sample: %d items %s → %s", len(sample),
                 dict(Counter(s["stratum"] for s in sample)), SAMPLE_PATH)
    if args.label:
        label_loop(json.loads(SAMPLE_PATH.read_text()), labels)
    if args.report:
        rep = make_report(json.loads(SAMPLE_PATH.read_text()), labels)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / "label_audit.json").write_text(json.dumps(rep, indent=1))
        lines = [f"AUTO-LABEL AUDIT — {rep['audited']}/{rep['sampled']} audited",
                 f"overall: {rep['overall']}"]
        lines += [f"  {k:18s} {v}" for k, v in rep["by_stratum"].items()]
        txt = "\n".join(lines)
        (config.REPORTS_DIR / "label_audit.txt").write_text(txt + "\n")
        print(txt)


if __name__ == "__main__":
    main()

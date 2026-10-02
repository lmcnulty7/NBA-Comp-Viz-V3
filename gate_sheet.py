#!/usr/bin/env python
"""
gate_sheet.py — human contact-sheet pass over every frame the judge labeled wide_broadcast
(LABEL_PLAN §3.6, revised 2026-10-02 after the shot_type audit found those labels 68.8% right).

One binary question per frame: is this a wide game shot (the court full-frame, usable by the
tracker) or not?  Pages are ORDERED by the current gate's P(live) so most pages are homogeneous;
the model supplies ordering only, never a label, no score is displayed, and every frame is seen.

  --pool      wide (default): the 5,454 frames the judge called wide_broadcast
              rest: the 705 frames it called closeup / graphic / split_screen / replay — same
              question, so the whole 6,159-frame corpus ends up human-verified for the gate
  --prepare   Drive frames -> thumbnails + gate scores (resumable; checkpoint every 256 frames)
  --label     contact sheets, 30 per page:
                click = flag a frame as NOT wide (click again to clear)
                i = invert the page (for pages that are mostly not-wide)
                SPACE = save the page and go on · b = back one page · q = quit
  --report    counts; test-retest against the 80 wide-claimed frames of the shot_type audit;
              the gate's in-domain precision/recall at 0.35 / 0.50 / 0.70 and the threshold the
              original objective (min FN s.t. FP-rate <= 0.1) picks on these labels
Files: data/gate_sheet/{scores.json, index.json, labels.json, thumbs/} · reports/gate_sheet.{json,txt}
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import config

DRIVE = Path.home() / "Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest"
SHEET = config.PROJECT_ROOT / "data" / "gate_sheet"
SCORES, INDEX, LABELS, THUMBS = SHEET / "scores.json", SHEET / "index.json", SHEET / "labels.json", SHEET / "thumbs"
COLS, ROWS, TW, TH, BAN = 6, 5, 220, 124, 44
PER = COLS * ROWS


POOL = "wide"

def set_pool(kind):
    """Select which frames this run covers; file names carry the pool so the two never mix."""
    global POOL, SCORES, INDEX, LABELS
    POOL = kind; suf = "" if kind == "wide" else "_" + kind
    SCORES, INDEX, LABELS = SHEET / ("scores%s.json" % suf), SHEET / ("index%s.json" % suf), SHEET / ("labels%s.json" % suf)

def pool():
    out = []
    for p in sorted((DRIVE / "autolabels" / "adjudicated").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line); st = r.get("shot_type")
            if (st == "wide_broadcast") == (POOL == "wide"):
                out.append({"tag": r["tag"], "frame": r["frame"], "key": "%s@%d" % (r["tag"], r["frame"]), "claim": st})
    return out


def src_path(it): return DRIVE / "label_corpus" / it["tag"] / ("f%07d.jpg" % it["frame"])
def thumb_name(it): return "%s__%07d.jpg" % (it["tag"], it["frame"])


def prepare(chunk=256):
    import cv2
    from gate.backbones import ClipBackbone
    from gate.trained_head import TrainedHeadGate
    SHEET.mkdir(parents=True, exist_ok=True); THUMBS.mkdir(exist_ok=True)
    items = pool()
    scores = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    todo = [it for it in items if it["key"] not in scores]
    print("pool '%s': %d frames; %d already scored; %d to do" % (POOL, len(items), len(scores), len(todo)), flush=True)
    if todo:
        backbone = ClipBackbone(config.CLIP_MODEL_NAME, config.get_device())
        gate = TrainedHeadGate.load(config.HEAD_PATH)
        t0 = time.time()
        for i in range(0, len(todo), chunk):
            batch = todo[i:i + chunk]; paths = [src_path(it) for it in batch]
            emb = backbone.embed_image_paths(paths, batch_size=32, progress=False)
            sc = gate.score_embeddings(emb)
            for it, s, p in zip(batch, sc, paths):
                img = cv2.imread(str(p))
                cv2.imwrite(str(THUMBS / thumb_name(it)), cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 82])
                scores[it["key"]] = round(float(s), 5)
            SCORES.write_text(json.dumps(scores))
            done = i + len(batch)
            print("  %d/%d  (%.0f s, ~%.0f s left)" % (done, len(todo), time.time() - t0,
                                                      (time.time() - t0) / done * (len(todo) - done)), flush=True)
    index = sorted(({**it, "score": scores[it["key"]], "thumb": thumb_name(it)} for it in items), key=lambda d: -d["score"])
    INDEX.write_text(json.dumps(index))
    missing = [d["thumb"] for d in index if not (THUMBS / d["thumb"]).exists()]
    print("index: %d frames, %d pages of %d; thumbnails missing: %d" % (len(index), -(-len(index) // PER), PER, len(missing)), flush=True)


def compose_page(items, flagged, title):
    import cv2, numpy as np
    page = np.full((BAN + ROWS * TH, COLS * TW, 3), 20, np.uint8)
    cv2.putText(page, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    for k, it in enumerate(items):
        r, c = divmod(k, COLS); x, y = c * TW, BAN + r * TH
        page[y:y + TH, x:x + TW] = cv2.imread(str(THUMBS / it["thumb"]))
        cv2.rectangle(page, (x, y), (x + TW - 1, y + TH - 1), (60, 60, 60), 1)
        if k in flagged:
            cv2.rectangle(page, (x + 2, y + 2), (x + TW - 3, y + TH - 3), (40, 40, 230), 4)
            cv2.line(page, (x + 12, y + 12), (x + 46, y + 46), (40, 40, 230), 4, cv2.LINE_AA)
            cv2.line(page, (x + 46, y + 12), (x + 12, y + 46), (40, 40, 230), 4, cv2.LINE_AA)
    return page


def label_loop():
    import cv2
    index = json.loads(INDEX.read_text())
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    pages = [index[i:i + PER] for i in range(0, len(index), PER)]
    n = next((k for k, pg in enumerate(pages) if any(it["key"] not in labels for it in pg)), len(pages))
    print("%d frames, %d pages; resuming at page %d (%d frames done)" % (len(index), len(pages), n + 1, len(labels)))
    win = "gate sheet — click the frames that are NOT a wide game shot"
    cv2.namedWindow(win)
    while n < len(pages):
        items = pages[n]
        flagged = {k for k, it in enumerate(items) if it["key"] in labels and not labels[it["key"]]["wide"]}
        def on_mouse(ev, mx, my, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and my >= BAN:
                k = ((my - BAN) // TH) * COLS + (mx // TW)
                if k < len(items): flagged.symmetric_difference_update({k})
        cv2.setMouseCallback(win, on_mouse)
        title = "[page %d/%d]  click every frame that is NOT a wide game shot    i invert | SPACE save+next | b back | q quit" % (n + 1, len(pages))
        act = None
        while act is None:
            cv2.imshow(win, compose_page(items, flagged, title))
            key = cv2.waitKey(40) & 0xFF
            if key in (32, 13): act = "next"
            elif key == ord("i"): flagged.symmetric_difference_update(set(range(len(items))))
            elif key == ord("b"): act = "back"
            elif key == ord("q"): act = "quit"
        if act == "quit":
            cv2.destroyAllWindows(); print("stopped at page %d — %d frames saved" % (n + 1, len(labels))); return
        if act == "back":
            n = max(0, n - 1); continue
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for k, it in enumerate(items):
            labels[it["key"]] = {"wide": k not in flagged, "ts": ts, "page": n + 1}
        LABELS.write_text(json.dumps(labels))
        n += 1
    cv2.destroyAllWindows(); print("complete — %d frames labeled." % len(labels))


def report():
    from label_matchups import wilson95
    index = json.loads(INDEX.read_text()); labels = json.loads(LABELS.read_text())
    done = [d for d in index if d["key"] in labels]
    wide = [d for d in done if labels[d["key"]]["wide"]]
    rep = {"pool": len(index), "labeled": len(done), "wide": len(wide), "not_wide": len(done) - len(wide),
           "judge_wide_precision": round(len(wide) / len(done), 3) if done else None,
           "judge_wide_precision_wilson95": wilson95(len(wide), len(done)) if done else None}
    # test-retest against the shot_type audit (same frames judged unsorted, one claim per page)
    ap = config.PROJECT_ROOT / "data" / "shot_type_audit"
    if (ap / "labels.json").exists():
        AS = {s["key"]: s for s in json.loads((ap / "sample.json").read_text()) if s["claim"] == "wide_broadcast"}
        AL = json.loads((ap / "labels.json").read_text())
        both = [k for k in AS if k in labels and AL[k]["truth"] is not None]
        agree = sum(1 for k in both if (AL[k]["truth"] == "wide_broadcast") == labels[k]["wide"])
        rep["test_retest_vs_audit"] = {"n": len(both), "agree": agree, "rate": round(agree / len(both), 3) if both else None,
                                       "disagree_keys": [k for k in both if (AL[k]["truth"] == "wide_broadcast") != labels[k]["wide"]]}
    # the gate against these human labels (wide == the gate's positive class, on wide-CLAIMED frames only)
    def at(t):
        tp = sum(1 for d in done if d["score"] >= t and labels[d["key"]]["wide"]); fp = sum(1 for d in done if d["score"] >= t and not labels[d["key"]]["wide"])
        fn = sum(1 for d in done if d["score"] < t and labels[d["key"]]["wide"]); tn = len(done) - tp - fp - fn
        return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                "fp_rate": round(fp / (fp + tn), 3) if fp + tn else None, "accuracy": round((tp + tn) / len(done), 3) if done else None}
    rep["gate_at"] = [at(t) for t in (0.35, 0.50, 0.70)]
    cands = [at(round(t, 2)) for t in [x / 100 for x in range(5, 96)]]
    ok = [c for c in cands if c["fp_rate"] is not None and c["fp_rate"] <= 0.10]
    rep["gate_objective_pick"] = min(ok, key=lambda c: (c["fn"], -c["threshold"])) if ok else None
    rep["note"] = ("human labels cover the judge's wide-CLAIMED frames only, so gate metrics here are conditional on that claim; "
                   "page order used the gate score (ordering only) — the test-retest row measures any anchoring that introduced")
    rep["pool"] = POOL
    other = SHEET / ("labels_rest.json" if POOL == "wide" else "labels.json"); oidx = SHEET / ("index_rest.json" if POOL == "wide" else "index.json")
    if other.exists() and oidx.exists():
        OL = json.loads(other.read_text()); OI = [d for d in json.loads(oidx.read_text()) if d["key"] in OL]
        allrows = [(d["score"], labels[d["key"]]["wide"]) for d in done] + [(d["score"], OL[d["key"]]["wide"]) for d in OI]
        def at_all(t):
            tp = sum(1 for s, w in allrows if s >= t and w); fp = sum(1 for s, w in allrows if s >= t and not w)
            fn = sum(1 for s, w in allrows if s < t and w); tn = len(allrows) - tp - fp - fn
            return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None, "fp_rate": round(fp / (fp + tn), 3) if fp + tn else None, "accuracy": round((tp + tn) / len(allrows), 3)}
        candsA = [at_all(round(t, 2)) for t in [x / 100 for x in range(5, 96)]]; okA = [c for c in candsA if c["fp_rate"] is not None and c["fp_rate"] <= 0.10]
        rep["full_corpus"] = {"n": len(allrows), "wide": sum(1 for _, w in allrows if w), "gate_at": [at_all(t) for t in (0.35, 0.50, 0.70)],
                              "objective_pick": min(okA, key=lambda c: (c["fn"], -c["threshold"])) if okA else None,
                              "note": "unconditional: every frame of the 6,159-frame corpus, human-verified"}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / ("gate_sheet%s.json" % ("" if POOL == "wide" else "_" + POOL))).write_text(json.dumps(rep, indent=1))
    lines = ["GATE SHEET — %d/%d wide-claimed frames labeled" % (rep["labeled"], rep["pool"]),
             "  truly wide: %d   not wide: %d   judge's wide precision: %s %s" % (rep["wide"], rep["not_wide"], rep["judge_wide_precision"], rep["judge_wide_precision_wilson95"])]
    if "test_retest_vs_audit" in rep: lines.append("  test-retest vs audit: %s" % {k: v for k, v in rep["test_retest_vs_audit"].items() if k != "disagree_keys"})
    lines += ["  gate @%.2f: %s" % (g["threshold"], {k: g[k] for k in ("precision", "recall", "fp_rate", "accuracy", "fn", "fp")}) for g in rep["gate_at"]]
    lines.append("  objective (min FN s.t. FP-rate <= 0.10) picks: %s" % rep["gate_objective_pick"])
    if "full_corpus" in rep:
        fc = rep["full_corpus"]; lines.append("FULL CORPUS (both pools, n = %d, wide = %d):" % (fc["n"], fc["wide"]))
        lines += ["  gate @%.2f: %s" % (g["threshold"], {k: g[k] for k in ("precision", "recall", "fp_rate", "accuracy", "fn", "fp")}) for g in fc["gate_at"]]
        lines.append("  objective picks: %s" % fc["objective_pick"])
    txt = "\n".join(lines); (config.REPORTS_DIR / ("gate_sheet%s.txt" % ("" if POOL == "wide" else "_" + POOL))).write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Contact-sheet pass over the judge's wide-claimed frames.")
    ap.add_argument("--pool", choices=("wide", "rest"), default="wide")
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--label", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(); set_pool(a.pool)
    if a.prepare: prepare()
    if a.label: label_loop()
    if a.report: report()

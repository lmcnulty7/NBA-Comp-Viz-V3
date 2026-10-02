#!/usr/bin/env python
"""
audit_shot_type.py — human audit of the judge's frame-level shot_type labels (LABEL_PLAN §3.6 item 1).

The adjudication pass assigned every frame a shot_type (wide_broadcast / closeup / replay /
graphic / split_screen). Those labels were never audited; if they hold up they are ~6k free
gate-v2 training labels, if not they are discarded like the adjudicated boxes were.
Verification-style: the frame is shown with the CLAIM in the banner; you confirm or name the
true class. You never label from scratch.

Keys:  y = the claim is right
       w / c / r / g / s = the claim is wrong; the true class is wide / closeup / replay / graphic / split_screen
       u = can't tell   ·   q = quit (labels save after every key)

Modes: --sample N (freeze a class-stratified sample + prefetch frames) · --label (one frame at a time)
       --grid (contact sheets: 20 thumbnails sharing one claim per page; click the wrong ones,
               SPACE accepts the rest; each flagged frame is then shown full size for its true class)
       --report
Files: data/shot_type_audit/{sample.json, labels.json, frames/} · reports/shot_type_audit.{json,txt}
"""
from __future__ import annotations
import argparse, json, random, shutil, time
from collections import defaultdict, Counter
from pathlib import Path
import config
from label_matchups import wilson95

DRIVE = Path.home() / "Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest"
AUDIT = config.PROJECT_ROOT / "data" / "shot_type_audit"
SAMPLE, LABELS, FRAMES = AUDIT / "sample.json", AUDIT / "labels.json", AUDIT / "frames"
CLASSES = ["wide_broadcast", "closeup", "replay", "graphic", "split_screen"]
KEY2CLASS = {"w": "wide_broadcast", "c": "closeup", "r": "replay", "g": "graphic", "s": "split_screen"}
# per-class quota for a 200 sample: minority classes oversampled so each gets a usable interval
QUOTA = {"wide_broadcast": 80, "closeup": 50, "graphic": 40, "split_screen": 27, "replay": 3}


def pool():
    out = defaultdict(list)
    for p in sorted((DRIVE / "autolabels" / "adjudicated").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[r.get("shot_type")].append({"tag": r["tag"], "frame": r["frame"], "claim": r.get("shot_type")})
    return out


def build_sample(n, seed):
    rng = random.Random(seed); P = pool()
    scale = n / sum(QUOTA.values()); items = []
    for cls, q in QUOTA.items():
        cand = P.get(cls, []); rng.shuffle(cand)
        items += cand[:min(len(cand), max(1, round(q * scale)))]
    rng.shuffle(items)
    FRAMES.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        src = DRIVE / "label_corpus" / it["tag"] / ("f%07d.jpg" % it["frame"])
        dst = FRAMES / ("%04d.jpg" % i)
        if not dst.exists(): shutil.copyfile(src, dst)
        it["key"] = "%s@%d" % (it["tag"], it["frame"]); it["img"] = "frames/%04d.jpg" % i
    return items


def label_loop(sample, labels):
    import cv2
    todo = [s for s in sample if s["key"] not in labels]
    print("%d to audit (%d done). y right · w/c/r/g/s = wrong, true class · u unsure · q quit" % (len(todo), len(labels)))
    for i, s in enumerate(todo):
        img = cv2.imread(str(AUDIT / s["img"]))
        h, w = img.shape[:2]; sc = min(1.0, 1100 / w)
        view = cv2.resize(img, None, fx=sc, fy=sc) if sc < 1 else img.copy()
        view = cv2.copyMakeBorder(view, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(view, "[%d/%d] claim = %s     y right | w c r g s = true class | u | q" % (i + 1, len(todo), s["claim"].upper()),
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("shot_type audit", view)
        key = None
        while key not in list("ywcrgsu") + ["q"]:
            key = chr(cv2.waitKey(0) & 0xFF)
        if key == "q":
            cv2.destroyAllWindows(); print("stopped — %d audited" % len(labels)); return
        truth = s["claim"] if key == "y" else (None if key == "u" else KEY2CLASS[key])
        labels[s["key"]] = {"verdict": "y" if key == "y" else ("u" if key == "u" else "n"), "truth": truth,
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        LABELS.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows(); print("audit complete (%d)." % len(labels))


# ---- contact-sheet mode ---------------------------------------------------------------------
COLS, ROWS, TW, TH, BAN = 5, 4, 256, 144, 44

def compose_page(items, flagged, title):
    """Pure layout: COLS x ROWS thumbnails of one claimed class; flagged tiles get a red frame and an X."""
    import cv2, numpy as np
    page = np.full((BAN + ROWS * TH, COLS * TW, 3), 20, np.uint8)
    cv2.putText(page, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    for k, it in enumerate(items):
        r, c = divmod(k, COLS); x, y = c * TW, BAN + r * TH
        img = cv2.imread(str(AUDIT / it["img"]))
        page[y:y + TH, x:x + TW] = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
        cv2.rectangle(page, (x, y), (x + TW - 1, y + TH - 1), (60, 60, 60), 1)
        if k in flagged:
            cv2.rectangle(page, (x + 2, y + 2), (x + TW - 3, y + TH - 3), (40, 40, 230), 4)
            cv2.line(page, (x + 14, y + 14), (x + 54, y + 54), (40, 40, 230), 4, cv2.LINE_AA)
            cv2.line(page, (x + 54, y + 14), (x + 14, y + 54), (40, 40, 230), 4, cv2.LINE_AA)
    return page


def grid_loop(sample, labels):
    import cv2
    todo = [s for s in sample if s["key"] not in labels]
    by = defaultdict(list)
    for s in todo: by[s["claim"]].append(s)
    pages = [(cls, by[cls][i:i + COLS * ROWS]) for cls in CLASSES for i in range(0, len(by.get(cls, [])), COLS * ROWS)]
    print("%d frames in %d pages (%d already done). click = flag wrong · SPACE = accept the rest · q = quit" % (len(todo), len(pages), len(labels)))
    win = "shot_type audit — contact sheet"
    cv2.namedWindow(win)
    for n, (cls, items) in enumerate(pages):
        flagged = set()
        def on_mouse(ev, mx, my, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and my >= BAN:
                k = ((my - BAN) // TH) * COLS + (mx // TW)
                if k < len(items): flagged.symmetric_difference_update({k})
        cv2.setMouseCallback(win, on_mouse)
        title = "[page %d/%d] every frame here claims: %s      click the WRONG ones, then SPACE   (q quits)" % (n + 1, len(pages), cls.upper())
        while True:
            cv2.imshow(win, compose_page(items, flagged, title))
            key = cv2.waitKey(40) & 0xFF
            if key in (32, 13): break
            if key == ord("q"):
                cv2.destroyAllWindows(); print("stopped — %d audited (this page not saved)" % len(labels)); return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for k, it in enumerate(items):
            if k not in flagged:
                labels[it["key"]] = {"verdict": "y", "truth": it["claim"], "ts": ts, "mode": "grid"}
        for k in sorted(flagged):                      # full-size follow-up: name the true class
            it = items[k]; img = cv2.imread(str(AUDIT / it["img"]))
            view = cv2.resize(img, None, fx=min(1.0, 1100 / img.shape[1]), fy=min(1.0, 1100 / img.shape[1]))
            view = cv2.copyMakeBorder(view, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(view, "claimed %s — true class?   w wide | c closeup | r replay | g graphic | s split | u unsure" % it["claim"].upper(),
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, view)
            kk = None
            while kk not in list("wcrgsu"):
                kk = chr(cv2.waitKey(0) & 0xFF)
            labels[it["key"]] = {"verdict": "u" if kk == "u" else "n", "truth": None if kk == "u" else KEY2CLASS[kk], "ts": ts, "mode": "grid"}
        LABELS.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows(); print("audit complete (%d)." % len(labels))


def make_report(sample, labels):
    def acc(items):
        # correctness follows from truth == claim (a frame flagged and then given its own claim is correct)
        u = sum(1 for s in items if labels[s["key"]]["truth"] is None)
        y = sum(1 for s in items if labels[s["key"]]["truth"] == s["claim"])
        n = len(items) - y - u
        j = y + n
        return {"n_judged": j, "correct": y, "wrong": n, "unsure": u,
                "accuracy": round(y / j, 3) if j else None, "wilson95": wilson95(y, j) if j else None}
    done = [s for s in sample if s["key"] in labels]
    by = defaultdict(list)
    for s in done: by[s["claim"]].append(s)
    conf = Counter((s["claim"], labels[s["key"]]["truth"] or "unsure") for s in done)
    return {"sampled": len(sample), "audited": len(done), "overall": acc(done),
            "by_claimed_class": {k: acc(v) for k, v in sorted(by.items())},
            "confusion_claim_to_truth": {"%s -> %s" % k: v for k, v in sorted(conf.items())},
            "note": "stratified by the judge's claim; minority classes oversampled, so 'overall' is not a corpus rate — read by_claimed_class"}


def main():
    ap = argparse.ArgumentParser(description="Human audit of the judge's shot_type labels.")
    ap.add_argument("--sample", type=int, metavar="N"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", action="store_true"); ap.add_argument("--grid", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(); AUDIT.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    if a.sample:
        if labels: raise SystemExit("%d labels exist — move data/shot_type_audit/ aside for a fresh audit." % len(labels))
        sample = build_sample(a.sample, a.seed); SAMPLE.write_text(json.dumps(sample, indent=1))
        print("sample: %d frames %s -> %s" % (len(sample), dict(Counter(s["claim"] for s in sample)), SAMPLE))
    if a.label: label_loop(json.loads(SAMPLE.read_text()), labels)
    if a.grid: grid_loop(json.loads(SAMPLE.read_text()), labels)
    if a.report:
        rep = make_report(json.loads(SAMPLE.read_text()), labels)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / "shot_type_audit.json").write_text(json.dumps(rep, indent=1))
        lines = ["SHOT_TYPE AUDIT — %d/%d audited" % (rep["audited"], rep["sampled"]), "overall (stratified, not a corpus rate): %s" % rep["overall"]]
        lines += ["  %-16s %s" % (k, v) for k, v in rep["by_claimed_class"].items()]
        lines += ["  confusion: %s" % rep["confusion_claim_to_truth"]]
        txt = "\n".join(lines); (config.REPORTS_DIR / "shot_type_audit.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()

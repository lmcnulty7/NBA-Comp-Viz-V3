#!/usr/bin/env python
"""
clock_reader.py — Tier 2 step 1: scorebug game-clock reader (sparse anchors).

Reads (period, game clock) from the broadcast scorebug. This is the ONLY OCR
Tier 2's play-by-play alignment needs, and it is deliberately the easy version
of the OCR problem: one reliable read per possession core, retryable across
neighboring frames — never continuous tracking.

Per-layout calibration (measured on frames, see reports/viz/clockcrop_*.png):
the ESPN bug's period+clock box, in pixels, per broadcast package. Each clip is
registered to a layout in CLIP_LAYOUT. New broadcasts = one new entry each.

Engine: easyocr (already installed), digits+colon allowlist on a 4× upscaled
crop; period and clock are disambiguated by x-position (period is the leftmost
token; the clock is the M:SS regex match; the shot clock, when caught by the
crop, sits right of the game clock and is ignored). Sanity gates: period 1–5,
minutes ≤ 12, seconds ≤ 59. Anything else → None (caller retries elsewhere).

Eval (same human pattern as teams/possessions): --eval-export samples frames
across the four clips' analysis windows and saves crop+prediction;
--eval-label shows each crop with the predicted reading — y = correct,
n = wrong (then type the truth in the terminal), s = skip, q = quit;
--eval-report gives exact-read accuracy and the error taxonomy.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("clock_reader")

# (x1, y1, x2, y2) per broadcast layout: SEPARATE period and clock sub-boxes —
# a digits-only allowlist mangles "1ST" (S→5, T→7) if both share one crop, and a
# tight clock box also excludes the shot clock sitting to its right
LAYOUTS = {
    "espn_saturday_480": {"period_box": (546, 378, 586, 408),    # GSW@OKC, 854x480
                          "clock_box": (584, 378, 634, 408)},
    "espn_ecf_720":      {"period_box": (712, 575, 782, 615),    # MIA@BOS 2012, 1280x720
                          "clock_box": (783, 575, 860, 615)},
    # Phase B calibrations (reports/viz/clockcal_*.png)
    "abc_finals16_720":  {"period_box": (830, 575, 880, 610),    # 2016 Finals, ABC
                          "clock_box": (883, 575, 950, 610)},
    "tnt_playoffs16_720": {"period_box": (855, 606, 912, 640),   # 2016 WCF, TNT
                           "clock_box": (955, 606, 1010, 640)},
    "espn_strip16_720":  {"period_box": (562, 640, 606, 672),    # Xmas 2016, ESPN strip
                          "clock_box": (608, 640, 665, 672)},
    "csn_ba16_720":      {"period_box": (1002, 622, 1052, 652),  # CSN Bay Area 2016
                          "clock_box": (1066, 622, 1132, 652)},
    "espn17_720":        {"period_box": (542, 636, 585, 668),    # 2017 ESPN/ABC strip
                          "clock_box": (588, 636, 648, 668)},
    "csn_ba15_720":      {"period_box": (1092, 620, 1140, 652),  # CSN 2015: clock LEFT of period
                          "clock_box": (1022, 620, 1080, 652)},
    "espn13_720":        {"period_box": (770, 574, 826, 610),    # 2013 ESPN strip
                          "clock_box": (838, 574, 908, 610)},
}
CLIP_LAYOUT = {
    "curry_q1_clip": "espn_saturday_480",
    "curry_classic_clip": "espn_saturday_480",
    "clip_10m00_18m00": "espn_ecf_720",
    "clip_26m00_34m00": "espn_ecf_720",
    "clip_40m00_48m00": "espn_ecf_720",
    "clip_55m00_63m00": "espn_ecf_720",
    "clip_70m00_78m00": "espn_ecf_720",
}
PERIOD_RE = re.compile(r"([1-4])\s*(?:st|nd|rd|th)?", re.I)
CLOCK_EVAL_DIR = config.PROJECT_ROOT / "data" / "clock_eval"


def layout_for_clip(stem: str) -> str:
    """Layout for a clip stem: static registry first, then the harvest games
    registry (Phase B sections are named <tag>_sNN)."""
    if stem in CLIP_LAYOUT:
        return CLIP_LAYOUT[stem]
    tag = re.sub(r"_s\d+$", "", stem)
    reg = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())
    if tag in reg and "layout" in reg[tag]:
        return reg[tag]["layout"]
    raise KeyError(f"no layout registered for {stem!r} (tag {tag!r})")


class ClockReader:
    def __init__(self, layout: str):
        import easyocr
        self.period_box = LAYOUTS[layout]["period_box"]
        self.clock_box = LAYOUTS[layout]["clock_box"]
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def _ocr(self, frame_bgr, box, allowlist):
        x1, y1, x2, y2 = box
        big = cv2.resize(frame_bgr[y1:y2, x1:x2], None, fx=4, fy=4,
                         interpolation=cv2.INTER_CUBIC)
        return "".join(self.reader.readtext(big, allowlist=allowlist, detail=0))

    def read(self, frame_bgr: np.ndarray):
        """→ (period 1–5 | None, clock_seconds int | None, raw string)."""
        ptxt = self._ocr(frame_bgr, self.period_box, "1234stndrhSTNDRHoO")
        ctxt = self._ocr(frame_bgr, self.clock_box, "0123456789:.")
        raw = f"{ptxt}|{ctxt}"
        period = None
        if "ot" in ptxt.lower() or ptxt.lower().startswith("o"):
            period = 5
        else:
            m = PERIOD_RE.search(ptxt)
            if m:
                period = int(m.group(1))
        # clock formats (hand-verification 2026-07-09 caught two traps: the old
        # parser turned sub-minute "43.3" into 4:33, inventing fake splices; and
        # easyocr reads colons as dots, so the SEPARATOR GLYPH is unreliable —
        # the digit count after it is the discriminator):
        #   two digits after separator → M:SS  ("9:13", "11.46" ≡ 11:46)
        #   one digit after a dot      → SS.T  ("43.3" = 43.3 s, tenths shown <1:00)
        #   bare digit group           → M:SS fallback ("1101" → 11:01)
        clock = None
        mm = ss = None
        m_mss = re.search(r"(\d{1,2})[.:](\d{2})(?!\d)", ctxt)
        m_sst = re.search(r"(\d{1,2})\.(\d)(?!\d)", ctxt)
        if m_mss:
            mm, ss = int(m_mss.group(1)), int(m_mss.group(2))
        elif m_sst:
            sec = float(f"{m_sst.group(1)}.{m_sst.group(2)}")
            if sec < 60:
                return period, sec, raw
        else:
            d = re.sub(r"\D", "", ctxt)
            mm, ss = (int(d[:-2]), int(d[-2:])) if 3 <= len(d) <= 4 else \
                     ((0, int(d)) if len(d) == 2 else (None, None))
        if mm is not None and ss is not None and mm <= 12 and ss <= 59:
            clock = mm * 60 + ss
        return period, clock, raw

    def anchor(self, cap, frame_idx: int, stride: int = 3, tries: int = 15):
        """Robust read near frame_idx: walk outward until a sane reading.
        Failures measured in Phase A were ABSTENTIONS (eval: 0 confident-wrong),
        so wider retries recover them: 15 tries spans ±7 strides (~±0.7 s)."""
        for k in range(tries):
            off = (k + 1) // 2 * stride * (1 if k % 2 else -1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + off)
            ret, frame = cap.read()
            if not ret:
                continue
            period, clock, raw = self.read(frame)
            if period is not None and clock is not None:
                return {"frame": frame_idx + off, "period": period,
                        "clock_s": clock, "raw": raw}
        return None

    def anchor_multi(self, cap, frame_idx: int, fallbacks: list[int], stride: int = 3):
        """anchor() at frame_idx, else at each fallback frame (bug can be briefly
        hidden by broadcast graphics right at the preferred moment)."""
        for f in [frame_idx] + list(fallbacks):
            a = self.anchor(cap, f, stride=stride)
            if a:
                return a
        return None


# ── eval (the usual pattern) ──────────────────────────────────────────────────
WINDOWS = {  # frame ranges the pipeline analyzed (live footage, bug visible)
    "curry_q1_clip": (11520, 13320), "curry_classic_clip": (2000, 2900),
    "clip_10m00_18m00": (2000, 3600), "clip_26m00_34m00": (2000, 3340),
}


def do_export(args):
    rng = np.random.default_rng(config.SEED)
    CLOCK_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip, (f0, f1) in WINDOWS.items():
        reader = ClockReader(CLIP_LAYOUT[clip])
        src = config.VIDEO_DIR / f"{clip}.mp4"
        cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(src))
        picks = sorted(rng.choice(np.arange(f0, f1, 3), args.n_per_clip, replace=False))
        for i, f in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ret, frame = cap.read()
            if not ret:
                continue
            period, clock, raw = reader.read(frame)
            pred = (f"{period or '?'}|{clock // 60}:{clock % 60:02d}" if clock is not None
                    else f"{period or '?'}|?")
            pb, cb = reader.period_box, reader.clock_box
            x1, y1 = min(pb[0], cb[0]), min(pb[1], cb[1])
            x2, y2 = max(pb[2], cb[2]), max(pb[3], cb[3])
            fname = f"{clip}_{i:02d}.png"
            cv2.imwrite(str(CLOCK_EVAL_DIR / fname),
                        cv2.resize(frame[y1:y2, x1:x2], None, fx=4, fy=4,
                                   interpolation=cv2.INTER_NEAREST))
            rows.append({"sample_id": f"{clip}:{i}", "file": fname, "frame": int(f),
                         "pred": pred, "raw": raw})
        cap.release()
        log.info("%s: %d reads", clip, args.n_per_clip)
    with (CLOCK_EVAL_DIR / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "file", "frame", "pred", "raw"])
        w.writeheader()
        w.writerows(rows)
    log.info("%d samples → %s  (next: --eval-label)", len(rows), CLOCK_EVAL_DIR)


def do_label(args):
    index = list(csv.DictReader((CLOCK_EVAL_DIR / "index.csv").open()))
    tpath = CLOCK_EVAL_DIR / "truth.csv"
    truth = {r["sample_id"]: r for r in (csv.DictReader(tpath.open()) if tpath.exists() else [])}
    pending = [r for r in index if r["sample_id"] not in truth]
    log.info("%d samples, %d labeled, %d to go — y=pred correct, n=wrong (type truth "
             "in terminal as e.g. 2|7:28), s=skip, q=quit", len(index), len(truth), len(pending))
    win = "clock eval"
    cv2.namedWindow(win)
    for r in pending:
        img = cv2.imread(str(CLOCK_EVAL_DIR / r["file"]))
        canvas = cv2.copyMakeBorder(img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(canvas, f"pred: {r['pred']}   correct? y/n (s skip, q quit)", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            continue
        if key == ord("y"):
            truth[r["sample_id"]] = {"sample_id": r["sample_id"], "truth": r["pred"]}
        elif key == ord("n"):
            t = input(f"  true reading for {r['sample_id']} (e.g. 2|7:28, or blank=unreadable): ").strip()
            truth[r["sample_id"]] = {"sample_id": r["sample_id"], "truth": t or "unreadable"}
        with tpath.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sample_id", "truth"])
            w.writeheader()
            w.writerows(truth.values())
    cv2.destroyAllWindows()
    log.info("%d/%d labeled → %s", len(truth), len(index), tpath)


def do_report(args):
    index = {r["sample_id"]: r for r in csv.DictReader((CLOCK_EVAL_DIR / "index.csv").open())}
    truth = list(csv.DictReader((CLOCK_EVAL_DIR / "truth.csv").open()))
    n = ok = unread = miss = wrong = 0
    for t in truth:
        r = index.get(t["sample_id"])
        if not r:
            continue
        n += 1
        if t["truth"] == "unreadable":
            unread += 1
        elif r["pred"] == t["truth"]:
            ok += 1
        elif "?" in r["pred"]:
            miss += 1     # reader abstained on a readable clock (retryable, benign)
        else:
            wrong += 1    # confident misread — the dangerous case for alignment
    rep = {"n": n, "exact_correct": ok, "abstained_readable": miss,
           "confident_wrong": wrong, "human_unreadable": unread,
           "accuracy_on_readable": round(ok / max(n - unread, 1), 3)}
    (config.REPORTS_DIR / "clock_eval.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Scorebug game-clock reader + eval.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-export", action="store_true")
    mode.add_argument("--eval-label", action="store_true")
    mode.add_argument("--eval-report", action="store_true")
    ap.add_argument("--n-per-clip", type=int, default=14)
    args = ap.parse_args()
    if args.eval_export:
        do_export(args)
    elif args.eval_label:
        do_label(args)
    else:
        do_report(args)


if __name__ == "__main__":
    main()

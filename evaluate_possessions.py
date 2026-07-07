#!/usr/bin/env python
"""
evaluate_possessions.py — human-label eval for Component C2 (possession segmentation).

Same pattern as evaluate_teams.py: export a stratified sample → keypress-label
it BLIND to predictions → report accuracy. Two failure modes are scored
SEPARATELY (they are different mechanisms):
  • attacked basket (occupancy geometry)
  • offense/defense assignment (defenders-sit-closer heuristic)

Modes
  --export   For a clip that already has <clip>_possessions.json (+ _identity.json):
             sample frames stratified over situations —
               span_mid        middle of a halfcourt span (easy case)
               span_start      ~1 s after a span begins (the boundary call)
               pre_start       ~1 s BEFORE a span begins (should not be that span)
               transition_mid  middle of a transition span (should have no set)
             Saves the frames + index.csv (with hidden predictions) to
             data/poss_eval/<clip>/. Repeat per clip.
  --label    One frame at a time, TWO keypresses each, predictions never shown:
               attacked basket:  a = left   d = right   u = unclear
               offense (kit):    w = light/white   n = dark/navy   u = unclear
             (left/right = as seen on the broadcast frame; the sideline camera
             keeps court-x aligned with screen-x on these clips.)
             z = undo previous sample, s = skip, q = quit. Resumable.
             Labels all exported clips in one session (order shuffled).
  --report   Aggregates ALL labeled clips: attacked-basket accuracy + confusion,
             offense accuracy, both split by situation kind; plus how often a
             "transition" prediction hid a human-visible halfcourt set (missed
             coverage). → reports/poss_eval.{json,txt}

Examples
  python evaluate_possessions.py --export --source ".../curry_q1_clip.mp4"
  python evaluate_possessions.py --label
  python evaluate_possessions.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("evaluate_possessions")

POSS_EVAL_DIR = config.PROJECT_ROOT / "data" / "poss_eval"
INDEX_FIELDS = ["sample_id", "clip", "file", "frame", "kind",
                "pred_state", "pred_offense", "pred_confidence"]
BASKET_KEYS = {"a": "left", "d": "right", "u": "unclear"}
OFFENSE_KEYS = {"w": "light", "n": "dark", "u": "unclear"}


def read_rows(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open())) if p.exists() else []


def write_rows(p: Path, rows: list[dict], fields: list[str]) -> None:
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── export ────────────────────────────────────────────────────────────────────
def span_at(spans: list[dict], frame: int) -> dict | None:
    for s in spans:
        if s["start_frame"] <= frame <= s["end_frame"]:
            return s
    return None


def pred_for(spans: list[dict], frame: int, light_team) -> tuple[str, str, float]:
    s = span_at(spans, frame)
    if s is None or s["kind"] != "halfcourt":
        return "transition", "none", 0.0
    off = s.get("offense_team")
    if off is None or light_team is None:
        off_name = "none"
    else:
        off_name = "light" if off == light_team else "dark"
    return s["attacked_basket"], off_name, float(s.get("confidence", 0.0))


def do_export(args) -> None:
    stem = args.source.stem
    poss_path = config.TRACKING_DIR / f"{stem}_possessions.json"
    id_path = config.TRACKING_DIR / f"{stem}_identity.json"
    if not poss_path.exists():
        raise SystemExit(f"{poss_path} missing — run segment_possessions.py first.")
    poss = json.loads(poss_path.read_text())
    spans = poss["spans"]
    light_team = json.loads(id_path.read_text()).get("light_team") if id_path.exists() else None
    if light_team is None:
        log.warning("%s: no light_team in identity json — offense preds exported as 'none'", stem)
    fps, stride = poss.get("fps", 30.0), poss.get("stride", 3)
    sec = int(fps)   # frames per second in source-frame units

    # stratified sample frames; long spans are sampled every ~2.5 s so the
    # sample size isn't capped by the number of spans
    picks: list[tuple[int, str]] = []
    for s in spans:
        f0, f1 = s["start_frame"], s["end_frame"]
        if s["kind"] == "halfcourt":
            if f1 - f0 > 2 * sec:
                picks.append((f0 + sec, "span_start"))
            for f in range(f0 + 2 * sec, f1 - sec // 2, int(2.5 * sec)):
                picks.append((f, "span_mid"))
            if not any(k == "span_mid" and f0 <= f <= f1 for f, k in picks):
                picks.append(((f0 + f1) // 2, "span_mid"))
            picks.append((max(f0 - sec, 0), "pre_start"))
        elif s["duration_s"] >= 1.5:
            picks.append(((f0 + f1) // 2, "transition_mid"))
    # dedupe + cap per clip
    seen, uniq = set(), []
    for f, k in picks:
        if f not in seen:
            seen.add(f)
            uniq.append((f, k))
    rng = np.random.default_rng(config.SEED)
    if len(uniq) > args.per_clip:
        uniq = [uniq[i] for i in sorted(rng.choice(len(uniq), args.per_clip, replace=False))]

    out = POSS_EVAL_DIR / stem
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    rows = []
    for i, (f, kind) in enumerate(uniq):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"frame_{i:03d}.jpg"
        cv2.imwrite(str(out / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        state, off, conf = pred_for(spans, f, light_team)
        rows.append({"sample_id": f"{stem}:{i}", "clip": stem, "file": fname, "frame": f,
                     "kind": kind, "pred_state": state, "pred_offense": off,
                     "pred_confidence": round(conf, 2)})
    cap.release()
    write_rows(out / "index.csv", rows, INDEX_FIELDS)
    log.info("%s: exported %d samples (%s) → %s", stem, len(rows),
             dict(Counter(r["kind"] for r in rows)), out)


# ── repredict ─────────────────────────────────────────────────────────────────
def do_repredict(args) -> None:
    """After changing segment_possessions: refresh every exported clip's pred
    columns from the current <clip>_possessions.json (frames + your labels are
    untouched, exactly like evaluate_teams --repredict). Re-run
    segment_possessions per clip FIRST, then this, then --report."""
    for d in sorted(POSS_EVAL_DIR.iterdir()):
        if not d.is_dir():
            continue
        rows = read_rows(d / "index.csv")
        if not rows:
            continue
        stem = d.name
        poss = json.loads((config.TRACKING_DIR / f"{stem}_possessions.json").read_text())
        id_path = config.TRACKING_DIR / f"{stem}_identity.json"
        light = json.loads(id_path.read_text()).get("light_team") if id_path.exists() else None
        changed = 0
        for r in rows:
            state, off, conf = pred_for(poss["spans"], int(r["frame"]), light)
            if (state, off) != (r["pred_state"], r["pred_offense"]):
                changed += 1
            r.update({"pred_state": state, "pred_offense": off,
                      "pred_confidence": round(conf, 2)})
        write_rows(d / "index.csv", rows, INDEX_FIELDS)
        log.info("%s: repredicted %d samples (%d changed)", stem, len(rows), changed)
    log.info("next: python evaluate_possessions.py --report")


# ── label ─────────────────────────────────────────────────────────────────────
def do_label(args) -> None:
    index = []
    for d in sorted(POSS_EVAL_DIR.iterdir()) if POSS_EVAL_DIR.exists() else []:
        index += read_rows(d / "index.csv")
    if not index:
        raise SystemExit("Nothing exported — run --export per clip first.")
    rng = np.random.default_rng(config.SEED)
    index = [index[i] for i in rng.permutation(len(index))]   # shuffled label order

    truth_path = POSS_EVAL_DIR / "truth.csv"
    truth = {r["sample_id"]: r for r in read_rows(truth_path)}
    pending = [r for r in index if r["sample_id"] not in truth]
    log.info("%d samples, %d labeled, %d to go", len(index), len(truth), len(pending))

    win = "possession labeler"
    cv2.namedWindow(win)
    history: list[str] = []
    i = 0
    while 0 <= i < len(pending):
        r = pending[i]
        img = cv2.imread(str(POSS_EVAL_DIR / r["clip"] / r["file"]))
        scale = min(1.0, 640 / img.shape[0])
        show = cv2.resize(img, None, fx=scale, fy=scale)
        canvas = cv2.copyMakeBorder(show, 58, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        answers: dict[str, str] = {}
        aborted = False
        for q, keys, prompt in (("basket", BASKET_KEYS, "ATTACKED BASKET:  a=left  d=right  u=unclear"),
                                ("offense", OFFENSE_KEYS, "OFFENSE (kit):  w=light/white  n=dark/navy  u=unclear")):
            c = canvas.copy()
            cv2.putText(c, f"{len(truth)}/{len(index)} labeled   {r['sample_id']}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 255, 90), 1)
            cv2.putText(c, prompt + "   (z=undo prev, s=skip, q=quit)", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            cv2.imshow(win, c)
            while True:
                key = cv2.waitKey(0) & 0xFF
                ch = chr(key) if key < 128 else ""
                if key in (ord("q"), 27):
                    aborted = True
                elif ch == "z" and history:
                    truth.pop(history.pop(), None)
                    i = max(-1, i - 2)   # revisit previous
                    aborted = True
                elif ch == "s":
                    aborted = True
                elif ch in keys:
                    answers[q] = keys[ch]
                    break
                else:
                    continue
                break
            if aborted:
                break
        if aborted:
            if key in (ord("q"), 27):
                break
            i += 1
            continue
        truth[r["sample_id"]] = {"sample_id": r["sample_id"],
                                 "basket": answers["basket"], "offense": answers["offense"]}
        history.append(r["sample_id"])
        i += 1
        write_rows(truth_path, list(truth.values()), ["sample_id", "basket", "offense"])
    cv2.destroyAllWindows()
    write_rows(truth_path, list(truth.values()), ["sample_id", "basket", "offense"])
    log.info("%d/%d labeled → %s", len(truth), len(index), truth_path)


# ── report ────────────────────────────────────────────────────────────────────
def do_report(args) -> None:
    index = []
    for d in sorted(POSS_EVAL_DIR.iterdir()):
        if d.is_dir():
            index += read_rows(d / "index.csv")
    idx = {r["sample_id"]: r for r in index}
    truth = read_rows(POSS_EVAL_DIR / "truth.csv")
    joined = [(idx[t["sample_id"]], t) for t in truth if t["sample_id"] in idx]
    if len(joined) < 15:
        raise SystemExit(f"Only {len(joined)} labeled — label more first.")

    def acc_block(pairs):
        n = len(pairs)
        ok = sum(1 for p, h in pairs if p == h)
        return {"n": n, "accuracy": round(ok / n, 3) if n else None}

    # attacked basket: prediction says a basket AND human is definite
    bask = [(r["pred_state"], t["basket"], r["kind"]) for r, t in joined
            if r["pred_state"] in ("left", "right") and t["basket"] in ("left", "right")]
    off = [(r["pred_offense"], t["offense"], r["kind"]) for r, t in joined
           if r["pred_offense"] in ("light", "dark") and t["offense"] in ("light", "dark")]
    by_kind_b = defaultdict(list)
    for p, h, k in bask:
        by_kind_b[k].append((p, h))
    by_kind_o = defaultdict(list)
    for p, h, k in off:
        by_kind_o[k].append((p, h))

    conf_b = Counter((p, h) for p, h, _ in bask)
    # transition predictions where the human saw a definite halfcourt set
    trans = [(r, t) for r, t in joined if r["pred_state"] == "transition"]
    trans_missed = sum(1 for r, t in trans if t["basket"] in ("left", "right"))

    report = {
        "n_labeled": len(joined),
        "attacked_basket": {"overall": acc_block([(p, h) for p, h, _ in bask]),
                            "by_kind": {k: acc_block(v) for k, v in by_kind_b.items()},
                            "confusion": {f"{p}->{h}": n for (p, h), n in sorted(conf_b.items())}},
        "offense": {"overall": acc_block([(p, h) for p, h, _ in off]),
                    "by_kind": {k: acc_block(v) for k, v in by_kind_o.items()}},
        "transition_predictions": {"n": len(trans), "human_saw_halfcourt": trans_missed,
                                   "note": "missed coverage, scored separately from wrong-basket errors"},
        "unclear_rates": {"basket": round(sum(1 for _, t in joined if t["basket"] == "unclear") / len(joined), 3),
                          "offense": round(sum(1 for _, t in joined if t["offense"] == "unclear") / len(joined), 3)},
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "poss_eval.json").write_text(json.dumps(report, indent=2))

    lines = [f"POSSESSION EVAL — {len(joined)} labeled samples across "
             f"{len({r['clip'] for r, _ in joined})} clips",
             f"attacked basket: {report['attacked_basket']['overall']}   "
             f"by kind: { {k: v['accuracy'] for k, v in report['attacked_basket']['by_kind'].items()} }",
             f"  confusion: {report['attacked_basket']['confusion']}",
             f"offense/defense: {report['offense']['overall']}   "
             f"by kind: { {k: v['accuracy'] for k, v in report['offense']['by_kind'].items()} }",
             f"transition preds: {len(trans)}, of which human saw a set: {trans_missed}",
             f"unclear rates: {report['unclear_rates']}"]
    txt = "\n".join(lines)
    (config.REPORTS_DIR / "poss_eval.txt").write_text(txt + "\n")
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Human-label eval for possession segmentation.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true")
    mode.add_argument("--repredict", action="store_true")
    mode.add_argument("--label", action="store_true")
    mode.add_argument("--report", action="store_true")
    ap.add_argument("--source", type=Path, default=config.VIDEO_DIR / "curry_q1_clip.mp4")
    ap.add_argument("--per-clip", type=int, default=30, help="max samples per clip (--export)")
    args = ap.parse_args()
    if args.export:
        do_export(args)
    elif args.repredict:
        do_repredict(args)
    elif args.label:
        do_label(args)
    else:
        do_report(args)


if __name__ == "__main__":
    main()

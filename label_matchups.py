#!/usr/bin/env python
"""
label_matchups.py — human ground truth for the matchup engine (CLAIMS C1).

The matchup stage is the only pipeline stage with structural checks but NO
labeled accuracy number, and every defender-level claim inherits its error.
This tool closes that: a human verifies, possession by possession, the
engine's PRIMARY claim — "defender fragment D was primarily guarding
offensive fragment M" — and the report turns verdicts into an accuracy
number with a Wilson CI and an error breakdown.

Verification (y/n/u), not free labeling: judging a shown pairing is ~10×
faster than picking defenders from scratch, and it measures exactly the
quantity the credit chain consumes (the primary-defender attribution).

Modes
  --sample N   freeze a stratified sample (per game, seeded) of joined
               possessions WITH the engine's predictions to
               data/matchup_eval/sample.json. Refuses to resample once
               labels exist (the sample is part of the eval's identity).
  --label      cv2 review loop: broadcast frames + synced top-down diagram,
               claimed pair highlighted. Keys: y correct · n wrong ·
               u can't tell · r replay · q quit (labels saved incrementally).
  --report     accuracy (y / (y+n)) + Wilson 95% CI, breakdown by matchup
               distance and time-share → reports/matchup_eval.{json,txt}.

Truth separation: labels live in labels.json keyed by possession, never
inside sample.json; --report recomputes from the two files every time.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import config
from fetch_pbp import PBP_DIR, video_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("label_matchups")

EVAL_DIR = config.PROJECT_ROOT / "data" / "matchup_eval"
SAMPLE_PATH = EVAL_DIR / "sample.json"
LABELS_PATH = EVAL_DIR / "labels.json"
# games built on Colab may have no local video — the Colab Drive is mounted
# locally (Google Drive desktop, lucienmmcnulty account) and streams on demand
DRIVE_VIDEO = (Path.home() / "Library/CloudStorage"
               / "GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest/video")


def resolve_video(clip: str) -> Path:
    p = video_path(clip)
    if Path(p).exists():
        return Path(p)
    q = DRIVE_VIDEO / f"{clip}.mp4"
    if q.exists():
        return q
    raise FileNotFoundError(f"no video for {clip} locally or on the Drive mount")


def pkey(clip: str, set_start: int) -> str:
    return f"{clip}@{set_start}"


# ── sampling ──────────────────────────────────────────────────────────────────
def build_sample(n_target: int, seed: int = 42) -> list[dict]:
    """Stratified by clip-tag (game), proportional to joined counts, seeded.
    Predictions are FROZEN here: the engine's primary pairing at sample time
    is what gets judged, even if the engine changes later (--repredict-style
    re-evals would freeze a new sample, never mutate this one)."""
    join = json.loads((PBP_DIR / "tier2_join.json").read_text())
    by_tag = defaultdict(list)
    for r in join["joined"]:
        tag = r["clip"].rsplit("_s", 1)[0] if "_s" in r["clip"] else r["clip"]
        by_tag[tag].append(r)
    total = sum(len(v) for v in by_tag.values())
    rng = random.Random(seed)
    picked = []
    for tag in sorted(by_tag):
        recs = sorted(by_tag[tag], key=lambda r: (r["clip"], r["set_start_frame"]))
        k = max(1, round(n_target * len(recs) / total))
        picked += rng.sample(recs, min(k, len(recs)))
    rng.shuffle(picked)
    picked = picked[:n_target]

    sample = []
    mcache: dict[str, dict] = {}
    for r in picked:
        clip = r["clip"]
        if clip not in mcache:
            mcache[clip] = {p["set_start_frame"]: p for p in json.loads(
                (config.TRACKING_DIR / f"{clip}_matchups.json").read_text())["possessions"]}
        m = mcache[clip].get(r["set_start_frame"])
        if m is None:
            continue
        top = sorted(m["defenders"], key=lambda d: -d["time_assigned_s"])[0]
        t_total = sum(d["time_assigned_s"] for d in m["defenders"]) or 1.0
        sample.append({
            "key": pkey(clip, r["set_start_frame"]),
            "clip": clip, "game": r["game"],
            "set_start_frame": r["set_start_frame"],
            "core_start_frame": m["core_start_frame"],
            "core_end_frame": m["core_end_frame"],
            "offense_team": m["offense_team"], "defense_team": m["defense_team"],
            "offense_real": r["offense_real"], "defense_real": r["defense_real"],
            "pred": {"defender": top["defender"], "primary_man": top["primary_man"],
                     "time_assigned_s": top["time_assigned_s"],
                     "time_share": round(top["time_assigned_s"] / t_total, 2),
                     "dist_median_ft": top["matchup_dist_median_ft"]},
        })
    return sample


# ── report ────────────────────────────────────────────────────────────────────
def wilson95(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def make_report(sample: list[dict], labels: dict) -> dict:
    def acc(items):
        y = sum(1 for s in items if labels[s["key"]]["verdict"] == "y")
        n = sum(1 for s in items if labels[s["key"]]["verdict"] == "n")
        return {"n_judged": y + n, "correct": y,
                "accuracy": round(y / (y + n), 3) if y + n else None,
                "wilson95": wilson95(y, y + n) if y + n else None}

    done = [s for s in sample if s["key"] in labels]
    judged = [s for s in done if labels[s["key"]]["verdict"] in "yn"]
    unsure = sum(1 for s in done if labels[s["key"]]["verdict"] == "u")

    def dist_bucket(s):
        d = s["pred"]["dist_median_ft"]
        return "tight(<6ft)" if d < 6 else ("mid(6-12ft)" if d < 12 else "loose(>12ft)")

    def share_bucket(s):
        sh = s["pred"]["time_share"]
        return "dominant(>0.5)" if sh > 0.5 else ("split(0.25-0.5)" if sh > 0.25 else "thin(<0.25)")

    by = lambda fn: {k: acc(v) for k, v in sorted(
        _group(judged, fn).items(), key=lambda kv: -len(kv[1]))}
    return {"sampled": len(sample), "labeled": len(done), "unsure": unsure,
            "overall": acc(judged),
            "by_distance": by(dist_bucket), "by_time_share": by(share_bucket),
            "by_game_tag": by(lambda s: s["clip"].rsplit("_s", 1)[0])}


def _group(items, fn):
    g = defaultdict(list)
    for x in items:
        g[fn(x)].append(x)
    return g


# ── interactive labeling ──────────────────────────────────────────────────────
def label_loop(sample: list[dict], labels: dict) -> None:
    import cv2
    import numpy as np
    from court.court33 import court_ft_to_px, draw_court_topdown
    from matchup_metrics import load as load_frames
    from videoseq import SeqReader

    scale, margin = 7.0, 12
    fcache: dict[str, tuple] = {}
    todo = [s for s in sample if s["key"] not in labels]
    log.info("%d to label (%d already done). y correct · n wrong · u unsure · "
             "r replay · q quit", len(todo), len(labels))

    for i, s in enumerate(todo):
        clip = s["clip"]
        if clip not in fcache:
            frames, _, cols = load_frames(clip)
            fcache[clip] = (frames, cols)
        frames, cols = fcache[clip]
        src = resolve_video(clip)            # local first, Drive mount fallback
        cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(src))
        reader = SeqReader(cap)
        d_tid, o_tid = s["pred"]["defender"], s["pred"]["primary_man"]
        data_f = sorted(f for f in frames if s["core_start_frame"] <= f <= s["core_end_frame"])

        def draw(fidx, video_frame):
            img, _, _ = draw_court_topdown(scale, margin)
            near = max((f for f in data_f if f <= fidx), default=None)
            if near is not None:
                pos = {}
                for tid, team, x, y in frames[near]:
                    p = court_ft_to_px(np.array((x, y), np.float32), scale, margin)[0].astype(int)
                    pos[tid] = p
                    col = cols.get(team, (150, 150, 150))
                    cv2.circle(img, tuple(p), 6, col, -1, cv2.LINE_AA)
                if d_tid in pos and o_tid in pos:
                    cv2.line(img, tuple(pos[d_tid]), tuple(pos[o_tid]), (0, 255, 255), 2, cv2.LINE_AA)
                for tid, ring in ((d_tid, (0, 255, 255)), (o_tid, (255, 0, 255))):
                    if tid in pos:
                        cv2.circle(img, tuple(pos[tid]), 10, ring, 2, cv2.LINE_AA)
                        cv2.putText(img, str(tid), tuple(pos[tid] + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ring, 1)
            vh = 420
            vid = cv2.resize(video_frame, (int(video_frame.shape[1] * vh / video_frame.shape[0]), vh))
            ch = img.shape[0]
            if ch < vh:
                img = cv2.copyMakeBorder(img, 0, vh - ch, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            canvas = np.hstack([vid, img[:vh]])
            banner = (f"[{i + 1}/{len(todo)}] {s['key']}  CLAIM: defender {d_tid} (yellow) "
                      f"guards {o_tid} (magenta)  share {s['pred']['time_share']:.2f}  "
                      f"median {s['pred']['dist_median_ft']:.1f} ft   y/n/u r q")
            canvas = cv2.copyMakeBorder(canvas, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(canvas, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            return canvas

        verdict = None
        while verdict is None:
            key = None
            for fidx in range(s["core_start_frame"], s["core_end_frame"] + 1, 2):
                ok, vf = reader.read(fidx)
                if not ok:
                    break
                cv2.imshow("matchup eval", draw(fidx, vf))
                k = cv2.waitKey(30) & 0xFF
                if k != 255:
                    key = chr(k)
                    break
            while key not in list("ynur") + ["q"]:
                key = chr(cv2.waitKey(0) & 0xFF)
            if key == "r":
                reader = SeqReader(cap)          # replay from a fresh seek
                continue
            if key == "q":
                cap.release()
                cv2.destroyAllWindows()
                log.info("stopped — %d labeled this session", len(labels))
                return
            verdict = key
        labels[s["key"]] = {"verdict": verdict, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        LABELS_PATH.write_text(json.dumps(labels, indent=1))
        cap.release()
    cv2.destroyAllWindows()
    log.info("sample fully labeled (%d).", len(labels))


def main() -> None:
    ap = argparse.ArgumentParser(description="Human eval of matchup primary-defender claims.")
    ap.add_argument("--sample", type=int, metavar="N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}

    if args.sample:
        if labels:
            raise SystemExit(f"{len(labels)} labels exist — resampling would orphan them. "
                             "Move data/matchup_eval/ aside first if a fresh eval is intended.")
        sample = build_sample(args.sample, args.seed)
        SAMPLE_PATH.write_text(json.dumps(sample, indent=1))
        log.info("sample: %d possessions across %d games → %s", len(sample),
                 len({s['game'] for s in sample}), SAMPLE_PATH)
    if args.label:
        label_loop(json.loads(SAMPLE_PATH.read_text()), labels)
    if args.report:
        rep = make_report(json.loads(SAMPLE_PATH.read_text()), labels)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / "matchup_eval.json").write_text(json.dumps(rep, indent=1))
        lines = [f"MATCHUP PRIMARY-DEFENDER EVAL — {rep['labeled']}/{rep['sampled']} labeled, "
                 f"{rep['unsure']} unsure",
                 f"overall: {rep['overall']}"]
        for dim in ("by_distance", "by_time_share", "by_game_tag"):
            lines.append(f"-- {dim}")
            lines += [f"  {k:20s} {v}" for k, v in rep[dim].items()]
        txt = "\n".join(lines)
        (config.REPORTS_DIR / "matchup_eval.txt").write_text(txt + "\n")
        print(txt)


if __name__ == "__main__":
    main()

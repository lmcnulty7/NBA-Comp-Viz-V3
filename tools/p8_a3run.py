"""tools/p8_a3run.py: on the a3-heavy windows, does r23 in the build's runtime see the boxes the corpus saw, and does
the build's tracker output them (docs/research/P8_a3run.md)? Report only: nothing is built, trained or changed.

    python3 -m export.heavy --name "p8 a3run smoke" -- <frozen CV python> -m tools.p8_a3run run --window W --frames 20
    python3 -m export.heavy --name "p8 a3run W" -- <frozen CV python> -m tools.p8_a3run run --window W
    python3 -m tools.p8_a3run report        (system python, stored outputs only)

<frozen CV python> is the anaconda interpreter the builds use (ultralytics 8.4.48, torch 2.2.2, OpenCV 4.13.0). `run`
refuses outside the heavy-job lock.

Windows, fixed now: the a3-heavy windows of P8_trackreset.md (gsw_bkn_2015_s04_w00; gsw_cha_2016_s05_w00 and
s05_w03; gsw_sac_2015_s01_w00, s09_w01 and s07_w00) and one control per game: among the game's scored windows that
P8_a3align.md found aligned and that are not heavy, the one with the most corpus frames (gsw_bkn_2015_s07_w00,
gsw_cha_2016_s07_w00, gsw_sac_2015_s04_w00).

`run`, per window, in the build's runtime with the detector on CPU:
- Weights: models/player_detector.pt of the main checkout, sha256 4c886d5f... checked (r23); config.PLAYER_DETECTOR_WEIGHTS
  set to it, as the E1R build child does.
- Frames: the clip label_build/<window>.mp4 read with videoseq.SeqReader in the build's order: first the pre-gate pass
  (every config.pregate_params(fps) stride frame from 0 to the end), then the reference-platform r23 sidecar's
  processed frames in order. Gate-rejected frames the build read but did not process are not read; reads that skip
  them need no seek unless the gap exceeds videoseq.SEEK_MIN_GAP, as in the build.
- (1) r23 model.predict on each processed frame: player class, confidence 0.40, NMS 0.45, CPU, its own YOLO instance.
- (2) detect.tracker.PlayerTracker(device="cpu").update on the same frames in order, exactly as build_trajectories.py
  calls it (camera-cut resets and exception handling included). After each update, from the BoT-SORT tracker: the
  boxes of tracked tracks that are not yet activated (unconfirmed) and the count of lost tracks.
- Output: <P8 workspace>/placement/a3run/<window>.json with every frame's (1) boxes, (2) boxes and ids, unconfirmed
  boxes, lost count, and the run's versions, weights sha256, device and seconds.

`report`, the reading rule fixed now:
- Reproduction check: per window, the share of processed frames where (2) equals the sidecar (same box count; every
  box matched one to one with its largest coordinate difference at most 2 px). The tracker test is read only for
  windows at 0.95 or more; others are reported as not reproduced.
- Runtime test, on corpus frames: boxes per frame of (1) against the corpus "ours" boxes (r23 at 0.40 on the decoded
  clip frame in the corpus runtime), and the share of "ours" boxes matched one to one by a (1) box at IoU 0.7 or more.
  The runtime differs in a window set when (1) has fewer than 0.9 times the "ours" boxes or under 0.9 of "ours" boxes
  matched.
- Tracker test, on all processed frames: yield = (2) boxes over (1) boxes; a (1) box is dropped when no (2) box
  reaches IoU 0.5 with it; the share of dropped boxes that an unconfirmed track box covers at IoU 0.5 or more. The
  tracker explains the loss when the heavy windows' pooled yield is at most 0.80 and the controls' at least 0.95
  (both reproduced).
- Readings: tracker explains and runtime does not: a tracker arm follows (activation or confirmation; its own
  registration, a rebuild under the lock, scored by the registered scorer); runtime differs and tracker does not: a3
  is mainly a runtime difference between the corpus and the builds, so the corpus-based raw and teacher analyses
  carry it and no build arm follows; both: both reported with their shares; neither: unexplained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from tools.p8_common import REPO, atomic_write, main_checkout, p8_root
from tools.p8_placement import BUILDS, POOL, check_games, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_a3run"
HEAVY = ("gsw_bkn_2015_s04_w00", "gsw_cha_2016_s05_w00", "gsw_cha_2016_s05_w03", "gsw_sac_2015_s01_w00",
         "gsw_sac_2015_s09_w01", "gsw_sac_2015_s07_w00")
CONTROL = ("gsw_bkn_2015_s07_w00", "gsw_cha_2016_s07_w00", "gsw_sac_2015_s04_w00")
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
CONF, NMS_IOU = 0.40, 0.45
REPRO_PX, REPRO_SHARE = 2.0, 0.95
RUNTIME_RATIO, RUNTIME_MATCH, RUNTIME_IOU = 0.9, 0.9, 0.7
DROP_IOU, YIELD_HEAVY, YIELD_CONTROL = 0.5, 0.80, 0.95


def game_of_window(w: str) -> str:
    return w.rsplit("_s", 1)[0]


def need(path):
    from pathlib import Path
    if not Path(path).exists():
        raise SystemExit(f"{WHO}: required file {Path(path).name} is missing")
    return path


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def matched_count(a: list, b: list, thr: float) -> int:
    """One-to-one (Hungarian on IoU) matches of boxes a to boxes b at IoU >= thr."""
    if not a or not b:
        return 0
    from scipy.optimize import linear_sum_assignment
    M = np.array([[iou(x, y) for y in b] for x in a])
    ri, ci = linear_sum_assignment(-M)
    return int(sum(M[i, j] >= thr for i, j in zip(ri, ci)))


def same_boxes(a: list, b: list, px: float = REPRO_PX) -> bool:
    """Same count, and a one-to-one match with every coordinate within px."""
    if len(a) != len(b):
        return False
    if not a:
        return True
    from scipy.optimize import linear_sum_assignment
    A, B = np.asarray(a, float)[:, :4], np.asarray(b, float)[:, :4]
    D = np.abs(A[:, None, :] - B[None, :, :]).max(-1)
    ri, ci = linear_sum_assignment(D)
    return bool(D[ri, ci].max() <= px)


def dropped(pred: list, track: list, cover: list, thr: float = DROP_IOU) -> tuple:
    """(1) boxes with no (2) box at IoU >= thr, and how many of those an unconfirmed box covers at IoU >= thr."""
    d = [p for p in pred if not any(iou(p, t) >= thr for t in track)]
    c = sum(1 for p in d if any(iou(p, u) >= thr for u in cover))
    return len(d), c


def read_v(res: dict) -> str:
    t, r = res["tracker_explains"], res["runtime_differs"]
    if t and not r:
        return "tracker"
    if r and not t:
        return "runtime"
    return "both" if (t and r) else "neither"


# ── run (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import cv2
    import config
    from detect.tracker import PlayerTracker
    from tools.p8_frozen import layout
    from ultralytics import YOLO
    from videoseq import SeqReader
    w = args.window
    if w not in HEAVY + CONTROL:
        raise SystemExit(f"{WHO}: {w} is not a registered window")
    refuse_cv_test(check_games([game_of_window(w)]), WHO)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if sha != R23_SHA:
        raise SystemExit(f"{WHO}: weights sha256 {sha} is not r23's")
    config.PLAYER_DETECTOR_WEIGHTS = weights
    P = layout(p8_root())
    g = game_of_window(w)
    side = json.loads(need(P["builds"] / BUILDS[g] / f"{w}_frames.json").read_text())
    frames = sorted(int(r["frame"]) for r in side["frames"])
    if args.frames:
        frames = frames[:args.frames]
    clip = need(P["label_build"] / f"{w}.mp4")
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = SeqReader(cap)
    pg_stride, _ = config.pregate_params(fps)
    t0 = time.time()
    for i in range(0, total, pg_stride):                    # the build's pre-gate pass: decode state only
        ok, _ = reader.read(i)
        if not ok:
            break
    det = YOLO(str(weights))
    tracker = PlayerTracker(device="cpu")
    out = []
    for f in frames:
        ok, img = reader.read(f)
        if not ok:
            raise SystemExit(f"{WHO}: {w} frame {f} unreadable")
        r = det.predict(img, classes=[config.PLAYER_CLASS_ID], conf=CONF, iou=NMS_IOU, device="cpu", verbose=False)[0]
        pred = [[round(float(v), 2) for v in b] + [round(float(c), 5)]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        tracks = tracker.update(img, f, f / fps)
        st = getattr(getattr(tracker.model, "predictor", None), "trackers", None)
        unconf, n_lost = [], None
        if st:
            bt = st[0]
            unconf = [[round(float(v), 2) for v in s.xyxy] for s in bt.tracked_stracks if not s.is_activated]
            n_lost = len(bt.lost_stracks)
        out.append({"frame": f, "pred": pred,
                    "track": [[round(float(v), 2) for v in t.bbox] + [int(t.track_id)] for t in tracks],
                    "unconfirmed": unconf, "lost": n_lost})
    cap.release()
    import torch
    import ultralytics
    rec = {"window": w, "frames": len(out), "smoke": bool(args.frames), "weights_sha256": sha, "device": "cpu",
           "versions": {"opencv": cv2.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
           "tracker_config": str(config.TRACKER_CONFIG), "seconds": round(time.time() - t0, 1),
           "n_resets": tracker.n_resets, "per_frame": out}
    d = p8_root() / "placement" / "a3run"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{w}__smoke.json" if args.frames else f"{w}.json"
    atomic_write(d / name, json.dumps(rec))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_frame"}))
    return rec


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def window_report(rec: dict, side: dict, ours: dict) -> dict:
    srow = {int(r["frame"]): [b["bbox"] for b in r["boxes"]] for r in side["frames"]}
    rep = n = 0
    n_pred = n_track = n_drop = n_cover = 0
    ours_n = pred_n = matched = cf = 0
    for fr in rec["per_frame"]:
        f = fr["frame"]
        tb = [t[:4] for t in fr["track"]]
        n += 1
        rep += same_boxes(tb, srow.get(f, []))
        pb = [p[:4] for p in fr["pred"]]
        n_pred += len(pb)
        n_track += len(tb)
        d, c = dropped(pb, tb, fr["unconfirmed"])
        n_drop += d
        n_cover += c
        if f in ours:
            cf += 1
            ours_n += len(ours[f])
            pred_n += len(pb)
            matched += matched_count(ours[f], pb, RUNTIME_IOU)
    return {"frames": n, "reproduced_share": rep / n if n else None, "pred_boxes": n_pred, "track_boxes": n_track,
            "dropped": n_drop, "dropped_under_unconfirmed": n_cover, "corpus_frames": cf, "ours_boxes": ours_n,
            "pred_boxes_on_corpus_frames": pred_n, "ours_matched": matched}


def cmd_report(args) -> dict:
    from tools import p8_teachers as T
    from tools.p8_e1data import corpus_root
    from tools.p8_frozen import layout
    refuse_cv_test(check_games(POOL), WHO)
    wr = p8_root()
    P = layout(wr)
    root = corpus_root()
    plan = json.loads(need(root / "plan.json").read_text())
    ours = {}
    for k in range(len(plan["chunks"])):
        for key, rec in json.loads(need(root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
            stem = key.split("/", 1)[1]
            w, f = stem.rsplit("_f", 1)
            ours.setdefault(w, {})[int(f)] = [p[:4] for p in rec[T.OURS] if p[5] == "player" and p[4] >= CONF]
    per = {}
    for w in HEAVY + CONTROL:
        rec = json.loads(need(wr / "placement" / "a3run" / f"{w}.json").read_text())
        if rec.get("smoke"):
            raise SystemExit(f"{WHO}: {w} holds a smoke run only")
        side = json.loads(need(P["builds"] / BUILDS[game_of_window(w)] / f"{w}_frames.json").read_text())
        per[w] = window_report(rec, side, ours.get(w, {}))

    def pool(ws, need_repro):
        ws = [w for w in ws if (not need_repro) or (per[w]["reproduced_share"] or 0) >= REPRO_SHARE]
        s = {k: sum(per[w][k] for w in ws) for k in ("pred_boxes", "track_boxes", "dropped",
                                                    "dropped_under_unconfirmed", "ours_boxes",
                                                    "pred_boxes_on_corpus_frames", "ours_matched")}
        return {"windows": ws, **s,
                "yield": s["track_boxes"] / s["pred_boxes"] if s["pred_boxes"] else None,
                "unconfirmed_share_of_dropped": s["dropped_under_unconfirmed"] / s["dropped"] if s["dropped"] else None,
                "pred_over_ours": s["pred_boxes_on_corpus_frames"] / s["ours_boxes"] if s["ours_boxes"] else None,
                "ours_matched_share": s["ours_matched"] / s["ours_boxes"] if s["ours_boxes"] else None}
    heavy_t, ctrl_t = pool(HEAVY, True), pool(CONTROL, True)
    heavy_r = pool(HEAVY, False)
    res = {"tracker_explains": bool(heavy_t["yield"] is not None and ctrl_t["yield"] is not None and
                                    heavy_t["yield"] <= YIELD_HEAVY and ctrl_t["yield"] >= YIELD_CONTROL),
           "runtime_differs": bool(heavy_r["pred_over_ours"] is not None and
                                   (heavy_r["pred_over_ours"] < RUNTIME_RATIO or
                                    heavy_r["ours_matched_share"] < RUNTIME_MATCH))}
    out = rnd({"what": "r23 in the build runtime (predict and the build's tracker) on the a3-heavy and control windows "
                       "(docs/research/P8_a3run.md); report only", "windows": per,
               "heavy_tracker": heavy_t, "control_tracker": ctrl_t, "heavy_runtime": heavy_r,
               "control_runtime": pool(CONTROL, False), **res, "reading": read_v(res),
               "rule": {"repro_px": REPRO_PX, "repro_share": REPRO_SHARE, "runtime_ratio": RUNTIME_RATIO,
                        "runtime_match": RUNTIME_MATCH, "runtime_iou": RUNTIME_IOU, "drop_iou": DROP_IOU,
                        "yield_heavy": YIELD_HEAVY, "yield_control": YIELD_CONTROL}})
    atomic_write(wr / "placement" / "a3run_report.json", json.dumps(out, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_a3run.json", json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("heavy_tracker", "control_tracker", "heavy_runtime", "reading")}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_a3run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--window", required=True)
    r.add_argument("--frames", type=int, default=0)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

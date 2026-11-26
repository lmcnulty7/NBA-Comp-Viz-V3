"""tools/p8_a3trace.py: an instrumented replay of the build's tracker on one a3-heavy window, to find the step where
high-confidence detections vanish (docs/research/P8_a3trace.md). Report only: nothing is built or changed; frozen and
installed files are untouched (instrumentation is by wrapping methods in the running process only).

    python3 -m export.heavy --name "p8 a3trace" -- <frozen CV python> -m tools.p8_a3trace run
    python3 -m tools.p8_a3trace report          (system python, stored output only)

Filters between model.track and the sidecar, read before registering:
- ultralytics trackers/track.py on_predict_postprocess_end: the tracker gets result.boxes (class filter already applied
  by NMS, classes=[0]); its output rows carry the matched detection's index; results are re-indexed to those rows and
  their boxes replaced by the track boxes. No area, aspect, court or count filter.
- detect/tracker.py PlayerTracker.update: model.track(classes=[PERSON_CLASS], conf 0.40, iou 0.45, persist); _parse
  keeps every box that has an id; a LinAlgError or the re-identification AttributeError resets the tracker and returns
  no tracks; a camera cut (80 percent of 4 or more ids gone) resets after the frame. No other filter.
- build_trajectories.py: every track PlayerTracker returns goes to _frame_record; the foot stabiliser changes feet,
  not boxes. No box filter.
So no wrapper filter explains dropped boxes; the instrumented run checks that too.
(Amendment, 2026-11-26, after the a3input result: this reading missed the reset path. PlayerTracker.reset drops the
predictor, and Model.track then re-registers the tracker callbacks, which append; after a reset the tracker updates
twice per frame. See docs/research/P8_a3input.md. The report's reproduction check now uses tools.p8_a3run.same_boxes,
one to one.)

`run` (window gsw_cha_2016_s05_w00, the build's runtime, detector on CPU, the read order of P8_a3run.md with both
amendments) wraps, in this process only:
- BYTETracker.update: the detections it receives (xyxy, conf, index), the frame id, and after the call the tracked
  tracks (id, activated, state, start frame, detection index, box), the lost and removed counts;
- BOTSORT.get_dists and matching.linear_assignment: each association call's track ids and detection indices, its
  threshold and its matches;
- BYTETracker.remove_duplicate_stracks: the tracks it removes (id, detection index, which list);
- PlayerTracker.reset: every reset, with its cause (camera cut or exception).
It also runs r23 predict at 0.40 on each frame (as P8_a3run.md) and stores everything per frame in
<P8 workspace>/placement/a3trace/gsw_cha_2016_s05_w00.json.

`report`, reading classes fixed now, for every dropped predict box (no output box at IoU 0.5 or more):
- G not in the tracker's input: no input detection at IoU 0.9 or more with it;
- F matched in an association but not output (a matched detection index absent from the output);
- E new track removed by remove_duplicate_stracks;
- D new track created and not output (unconfirmed on its first frame);
- U unmatched and no new track recorded;
- H other.
The class holding at least 50 percent of the dropped boxes names the step: G, the input to the tracker differs from
predict (an input check comes next); F, a wrapper or callback loss (a code fix arm); E, a duplicate-removal arm; D, an
activation arm; U or H, the trace is incomplete (reported, nothing follows). Below 50 percent, every class is reported
with its share and nothing follows by itself. Any arm needs its own registration and a rebuild under the lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from tools.p8_common import REPO, atomic_write, main_checkout, p8_root
from tools.p8_placement import BUILDS, check_games, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_a3trace"
WINDOW = "gsw_cha_2016_s05_w00"
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
CONF, NMS_IOU, SAME_IOU, KEEP_IOU, STEP_SHARE = 0.40, 0.45, 0.9, 0.5, 0.50
CLASSES = ("G_not_in_input", "F_matched_not_output", "E_removed_as_duplicate", "D_new_unconfirmed",
           "U_unmatched_no_new_track", "H_other")


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


def classify_drop(box, frame: dict) -> str:
    """One dropped predict box against the frame's trace record."""
    inp = frame.get("input", [])
    best, di = 0.0, None
    for d in inp:
        v = iou(box[:4], d["xyxy"])
        if v > best:
            best, di = v, d["idx"]
    if di is None or best < SAME_IOU:
        return "G_not_in_input"
    out_idx = {t["idx"] for t in frame.get("output", [])}
    matched = {m[1] for call in frame.get("assoc", []) for m in call["matches"]}
    if di in matched and di not in out_idx:
        return "F_matched_not_output"
    if any(r["idx"] == di for r in frame.get("removed_dup", [])):
        return "E_removed_as_duplicate"
    new = [t for t in frame.get("tracked_after", []) if t["idx"] == di and t["start_frame"] == frame.get("frame_id")
           and not t["activated"]]
    if new:
        return "D_new_unconfirmed"
    if di not in matched:
        return "U_unmatched_no_new_track"
    return "H_other"


def step_of(shares: dict) -> str | None:
    top = max(shares, key=lambda k: shares[k] or 0.0) if shares else None
    return top if top is not None and (shares[top] or 0.0) >= STEP_SHARE else None


# ── run (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import cv2
    import config
    import detect.tracker as DT
    import ultralytics.trackers.bot_sort as BS
    import ultralytics.trackers.byte_tracker as BT
    from tools.p8_frozen import layout
    from ultralytics import YOLO
    from videoseq import SEEK_MIN_GAP, SeqReader
    refuse_cv_test(check_games(["gsw_cha_2016"]), WHO)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if sha != R23_SHA:
        raise SystemExit(f"{WHO}: weights sha256 {sha} is not r23's")
    config.PLAYER_DETECTOR_WEIGHTS = weights

    cur = {}                                             # the trace of the frame being processed

    def box_of(t):
        return [round(float(v), 2) for v in t.xyxy]

    orig_update = BT.BYTETracker.update

    def update(self, results, img=None, feats=None):
        cur["frame_id"] = self.frame_id + 1
        cur["input"] = [{"xyxy": [round(float(v), 2) for v in b], "conf": round(float(c), 5), "idx": i}
                        for i, (b, c) in enumerate(zip(results.xyxy, results.conf))]
        cur["assoc"], cur["removed_dup"], cur["pending"] = [], [], None
        out = orig_update(self, results, img, feats)
        cur["tracked_after"] = [{"id": int(t.track_id), "activated": bool(t.is_activated), "state": int(t.state),
                                 "start_frame": int(t.start_frame), "idx": int(t.idx), "box": box_of(t)}
                                for t in self.tracked_stracks]
        cur["lost_after"] = len(self.lost_stracks)
        cur["output"] = [{"id": int(r[4]), "idx": int(r[-1])} for r in out] if len(out) else []
        return out
    BT.BYTETracker.update = update

    orig_dists = BS.BOTSORT.get_dists

    def get_dists(self, tracks, detections):
        cur["pending"] = {"tracks": [int(t.track_id) for t in tracks], "dets": [int(d.idx) for d in detections]}
        return orig_dists(self, tracks, detections)
    BS.BOTSORT.get_dists = get_dists

    orig_la = BT.matching.linear_assignment

    def linear_assignment(cost, thresh, *a, **k):
        res = orig_la(cost, thresh, *a, **k)
        p = cur.get("pending")
        if p is not None and cost.shape == (len(p["tracks"]), len(p["dets"])):
            cur.setdefault("assoc", []).append({"thresh": float(thresh),
                                                "matches": [[p["tracks"][i], p["dets"][j]] for i, j in res[0]]})
        cur["pending"] = None
        return res
    BT.matching.linear_assignment = linear_assignment

    orig_dup = BT.BYTETracker.remove_duplicate_stracks

    def remove_dup(a, b):
        ra, rb = orig_dup(a, b)
        gone = [t for t in a if t not in ra] + [t for t in b if t not in rb]
        cur.setdefault("removed_dup", []).extend({"id": int(t.track_id), "idx": int(t.idx)} for t in gone)
        return ra, rb
    BT.BYTETracker.remove_duplicate_stracks = staticmethod(remove_dup)

    resets = []
    orig_reset = DT.PlayerTracker.reset

    def reset(self):
        resets.append(cur.get("frame"))
        return orig_reset(self)
    DT.PlayerTracker.reset = reset

    P = layout(p8_root())
    side = json.loads(need(P["builds"] / BUILDS["gsw_cha_2016"] / f"{WINDOW}_frames.json").read_text())
    frames = sorted(int(r["frame"]) for r in side["frames"])
    cap = cv2.VideoCapture(str(need(P["label_build"] / f"{WINDOW}.mp4")))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = SeqReader(cap)
    pg_stride, _ = config.pregate_params(fps)
    t0 = time.time()
    for i in range(0, total, pg_stride):
        ok, _ = reader.read(i)
        if not ok:
            break
    stride = int(side.get("stride") or 1)
    if frames and frames[0] > 0:
        reader.read(0)
        for i in range(stride, frames[0], stride):
            reader.read(i)
    det = YOLO(str(weights))
    tracker = DT.PlayerTracker(device="cpu")
    out, prev = [], None
    for f in frames:
        if prev is not None and f - reader.next > SEEK_MIN_GAP:
            for i in range(prev + stride, f, stride):
                ok, _ = reader.read(i)
                if not ok:
                    raise SystemExit(f"{WHO}: gap frame {i} unreadable")
        prev = f
        ok, img = reader.read(f)
        if not ok:
            raise SystemExit(f"{WHO}: frame {f} unreadable")
        r = det.predict(img, classes=[config.PLAYER_CLASS_ID], conf=CONF, iou=NMS_IOU, device="cpu", verbose=False)[0]
        pred = [[round(float(v), 2) for v in b] + [round(float(c), 5)]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        cur.clear()
        cur["frame"] = f
        tracks = tracker.update(img, f, f / fps)
        rec = {k: v for k, v in cur.items() if k != "pending"}
        rec.update({"frame": f, "pred": pred,
                    "track": [[round(float(v), 2) for v in t.bbox] + [int(t.track_id)] for t in tracks]})
        out.append(rec)
    cap.release()
    import torch
    import ultralytics
    res = {"window": WINDOW, "frames": len(out), "weights_sha256": sha, "device": "cpu",
           "versions": {"opencv": cv2.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
           "resets_at": resets, "seconds": round(time.time() - t0, 1), "per_frame": out}
    d = p8_root() / "placement" / "a3trace"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{WINDOW}.json", json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}))
    return res


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def cmd_report(args) -> dict:
    from tools.p8_a3run import same_boxes
    from tools.p8_frozen import layout
    refuse_cv_test(check_games(["gsw_cha_2016"]), WHO)
    wr = p8_root()
    rec = json.loads(need(wr / "placement" / "a3trace" / f"{WINDOW}.json").read_text())
    side = json.loads(need(layout(wr)["builds"] / BUILDS["gsw_cha_2016"] / f"{WINDOW}_frames.json").read_text())
    srow = {int(r["frame"]): [b["bbox"] for b in r["boxes"]] for r in side["frames"]}
    counts = {c: 0 for c in CLASSES}
    reproduced = n_drop = n_pred = n_in = 0
    for fr in rec["per_frame"]:
        tb = [t[:4] for t in fr["track"]]
        sb = srow.get(fr["frame"], [])
        reproduced += same_boxes(tb, sb)        # amendment (2026-11-26): one-to-one, tools.p8_a3run.same_boxes
        n_pred += len(fr["pred"])
        n_in += len(fr.get("input", []))
        for p in fr["pred"]:
            if any(iou(p[:4], t) >= KEEP_IOU for t in tb):
                continue
            n_drop += 1
            counts[classify_drop(p, fr)] += 1
    shares = {c: (counts[c] / n_drop if n_drop else None) for c in CLASSES}
    step = step_of(shares)
    out = rnd({"what": "Instrumented replay of the build's tracker on gsw_cha_2016_s05_w00 (docs/research/P8_a3trace.md); "
                       "report only", "frames": rec["frames"], "reproduced_share": reproduced / rec["frames"],
               "predict_boxes": n_pred, "tracker_input_boxes": n_in, "dropped": n_drop, "classes": counts,
               "shares": shares, "step": step, "resets_at": rec["resets_at"], "versions": rec["versions"],
               "status": "withdrawn (2026-11-26): the rule returned G, but the trace is incomplete, the situation its "
                         "U or H case covers; the reading is withdrawn outside the rule. It recorded only the last BYTETracker.update call of each frame, and after the reset at frame 312 every frame "
                         "ran two (docs/research/P8_a3input.md, cause confirmed). The G reading and the input-check "
                         "proposal are withdrawn.",
               "rule": {"same_iou": SAME_IOU, "keep_iou": KEEP_IOU, "step_share": STEP_SHARE}})
    atomic_write(wr / "placement" / "a3trace_report.json", json.dumps(out, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_a3trace.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_a3trace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

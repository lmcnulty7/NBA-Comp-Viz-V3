"""tools/p8_a3input.py: does the build's tracker run twice per frame after a reset, and is that why it drops most of
its detections on the a3-heavy windows (docs/research/P8_a3input.md)? Report only: nothing is built or changed; frozen
and installed files are untouched (the fix variant is patched in the running process only).

    python3 -m export.heavy --name "p8 a3input" -- <frozen CV python> -m tools.p8_a3input run
    python3 -m tools.p8_a3input report          (system python, stored outputs only)

The code path found by reading (the free step, P8_a3input.md):
- detect/tracker.py PlayerTracker.reset() sets self.model.predictor = None (camera cut or tracker exception).
- ultralytics 8.4.48 engine/model.py Model.track: `if not hasattr(self.predictor, "trackers"): register_tracker(self,
  persist)`. With the predictor gone, the next call registers the tracker callbacks again, and Model.add_callback
  appends (it never replaces). The new predictor takes the model's callback lists (_callbacks=self.callbacks).
- So after the first reset, trackers/track.py on_predict_postprocess_end runs twice per frame: the first call updates
  the tracker with every detection and replaces the frame's results with the tracked boxes; the second call updates the
  same tracker again with only those tracked boxes. New detections become unconfirmed tracks in the first call and are
  removed in the second (no match), and the tracker's frame_id advances by 2 per frame. Each further reset adds
  another call.
- Stored evidence already (P8_a3trace.md's record): after the window's reset, frame_id rises by 2 per processed frame
  on 1,038 of 1,052 steps. P8_a3trace.md captured only the last update call per frame, so its class G is that second
  call's input, not the detector's output.

`run`, window gsw_cha_2016_s05_w00, the build's runtime, detector on CPU, P8_a3run.md's read order (both amendments),
two tracker variants on the same decoded frames:
- A, as built: detect.tracker.PlayerTracker unchanged.
- B, reset fixed in process: after PlayerTracker.reset, the tracker callbacks (partials of
  ultralytics.trackers.track.on_predict_start and on_predict_postprocess_end) are removed from the model's callback
  lists, so Model.track registers them exactly once again.
Per frame and variant: every BYTETracker.update call (its input count and frame id) and the output boxes. Output:
<P8 workspace>/placement/a3input/gsw_cha_2016_s05_w00.json.

`report`, reading rule fixed now:
- A must reproduce the sidecar (same box count and every box within 2 px) on at least 95 percent of frames, or nothing
  is read.
- Doubling confirmed when, in A, frames after the first reset hold two or more update calls on at least 95 percent of
  them, and in B no frame holds more than one.
- Detector output per frame: the input of the first update call (B's, which sees every detection). Yield = output
  boxes over that input, per variant.
- Cause confirmed when doubling is confirmed and B's yield is at least 0.90 while A's is at most 0.50 (A's input count
  is checked against P8_a3trace.md's predict count). Then a build arm follows: the reset fix as a tools-side patch,
  registered on its own and rebuilt on the reference platform under the lock, scored by the registered scorer.
- Doubling confirmed but B's yield under 0.90: the doubling is part of the cause, not all of it; reported.
- Doubling not confirmed: the code reading is wrong; reported, nothing follows.

Amendment 1 (2026-11-26, registered before the run started, at the reviewer's suggestion): per frame and variant, the
number of entries in model.callbacks["on_predict_postprocess_end"] before the update call is recorded and reported
(report only; the reading rule is unchanged). The rule reads the detector's output from the FIRST update call's input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from tools.p8_common import REPO, atomic_write, main_checkout, p8_root
from tools.p8_placement import BUILDS, check_games, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_a3input"
WINDOW = "gsw_cha_2016_s05_w00"
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
REPRO_SHARE, DOUBLE_SHARE, FIXED_YIELD, BUG_YIELD = 0.95, 0.95, 0.90, 0.50


def need(path):
    from pathlib import Path
    if not Path(path).exists():
        raise SystemExit(f"{WHO}: required file {Path(path).name} is missing")
    return path


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def same_boxes(a: list, b: list, px: float = 2.0) -> bool:
    if len(a) != len(b):
        return False
    return all(any(max(abs(x - y) for x, y in zip(t, s)) <= px for s in b) for t in a)


def doubling(frames: list) -> dict:
    """frames: per-frame records with "calls" (list of update calls) and "reset_before" (a reset has happened)."""
    after = [f for f in frames if f["reset_before"]]
    multi = sum(1 for f in after if len(f["calls"]) >= 2)
    single = sum(1 for f in frames if len(f["calls"]) == 1)
    any_multi = sum(1 for f in frames if len(f["calls"]) >= 2)
    return {"frames_after_first_reset": len(after), "with_two_or_more_calls": multi,
            "share": multi / len(after) if after else None, "frames_with_one_call": single,
            "frames_with_two_or_more_calls": any_multi, "frames": len(frames)}


def yield_of(frames: list, inputs: list) -> float | None:
    out = sum(len(f["output"]) for f in frames)
    inp = sum(inputs)
    return out / inp if inp else None


def read(repro: float, dA: dict, dB: dict, yA: float | None, yB: float | None) -> str:
    if repro < REPRO_SHARE:
        return "not reproduced"
    doubled = (dA["share"] or 0) >= DOUBLE_SHARE and dB["frames_with_two_or_more_calls"] == 0
    if not doubled:
        return "doubling not confirmed"
    if yB is not None and yA is not None and yB >= FIXED_YIELD and yA <= BUG_YIELD:
        return "cause confirmed"
    return "doubling is part of the cause"


# ── run (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import cv2
    from functools import partial
    import config
    import detect.tracker as DT
    import ultralytics.trackers.byte_tracker as BT
    import ultralytics.trackers.track as TR
    from tools.p8_frozen import layout
    from videoseq import SEEK_MIN_GAP, SeqReader
    refuse_cv_test(check_games(["gsw_cha_2016"]), WHO)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if sha != R23_SHA:
        raise SystemExit(f"{WHO}: weights sha256 {sha} is not r23's")
    config.PLAYER_DETECTOR_WEIGHTS = weights

    calls = []
    orig_update = BT.BYTETracker.update

    def update(self, results, img=None, feats=None):
        calls.append({"n_input": int(len(results.conf)), "frame_id": int(self.frame_id + 1)})
        return orig_update(self, results, img, feats)
    BT.BYTETracker.update = update

    def is_tracker_cb(c):
        return isinstance(c, partial) and c.func in (TR.on_predict_start, TR.on_predict_postprocess_end)

    class Fixed(DT.PlayerTracker):
        def reset(self):
            super().reset()
            for ev in ("on_predict_start", "on_predict_postprocess_end"):
                self.model.callbacks[ev] = [c for c in self.model.callbacks.get(ev, []) if not is_tracker_cb(c)]

    P = layout(p8_root())
    side = json.loads(need(P["builds"] / BUILDS["gsw_cha_2016"] / f"{WINDOW}_frames.json").read_text())
    frames = sorted(int(r["frame"]) for r in side["frames"])
    stride = int(side.get("stride") or 1)
    res = {}
    t0 = time.time()
    for name, cls in (("A", DT.PlayerTracker), ("B", Fixed)):
        cap = cv2.VideoCapture(str(need(P["label_build"] / f"{WINDOW}.mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reader = SeqReader(cap)
        pg_stride, _ = config.pregate_params(fps)
        for i in range(0, total, pg_stride):
            ok, _ = reader.read(i)
            if not ok:
                break
        if frames and frames[0] > 0:
            reader.read(0)
            for i in range(stride, frames[0], stride):
                reader.read(i)
        tracker = cls(device="cpu")
        out, prev = [], None
        for f in frames:
            if prev is not None and f - reader.next > SEEK_MIN_GAP:
                for i in range(prev + stride, f, stride):
                    reader.read(i)
            prev = f
            ok, img = reader.read(f)
            if not ok:
                raise SystemExit(f"{WHO}: {name} frame {f} unreadable")
            calls.clear()
            resets_before = tracker.n_resets
            n_cb = len(tracker.model.callbacks.get("on_predict_postprocess_end", []))     # amendment 1
            tracks = tracker.update(img, f, f / fps)
            out.append({"frame": f, "reset_before": resets_before > 0, "calls": list(calls),
                        "postprocess_callbacks_before": n_cb,
                        "output": [[round(float(v), 2) for v in t.bbox] for t in tracks]})
        cap.release()
        res[name] = {"n_resets": tracker.n_resets, "per_frame": out}
    import torch
    import ultralytics
    rec = {"window": WINDOW, "weights_sha256": sha, "device": "cpu", "seconds": round(time.time() - t0, 1),
           "versions": {"opencv": cv2.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
           "variants": res}
    d = p8_root() / "placement" / "a3input"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{WINDOW}.json", json.dumps(rec))
    print(json.dumps({"seconds": rec["seconds"], "resets": {k: v["n_resets"] for k, v in res.items()}}))
    return rec


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def cmd_report(args) -> dict:
    from tools.p8_frozen import layout
    refuse_cv_test(check_games(["gsw_cha_2016"]), WHO)
    wr = p8_root()
    rec = json.loads(need(wr / "placement" / "a3input" / f"{WINDOW}.json").read_text())
    trace = json.loads(need(wr / "placement" / "a3trace" / f"{WINDOW}.json").read_text())
    side = json.loads(need(layout(wr)["builds"] / BUILDS["gsw_cha_2016"] / f"{WINDOW}_frames.json").read_text())
    srow = {int(r["frame"]): [b["bbox"] for b in r["boxes"]] for r in side["frames"]}
    A, B = rec["variants"]["A"]["per_frame"], rec["variants"]["B"]["per_frame"]
    repro = sum(same_boxes(f["output"], srow.get(f["frame"], [])) for f in A) / len(A)
    det_in = [f["calls"][0]["n_input"] if f["calls"] else 0 for f in B]       # the detector's output per frame
    pred = {f["frame"]: len(f["pred"]) for f in trace["per_frame"]}
    a_first = sum(f["calls"][0]["n_input"] for f in A if f["calls"])
    dA, dB = doubling(A), doubling(B)
    yA, yB = yield_of(A, det_in), yield_of(B, det_in)
    out = rnd({"what": "Doubled tracker callbacks after a reset, as built (A) and with the reset fixed in process (B), "
                       "on gsw_cha_2016_s05_w00 (docs/research/P8_a3input.md); report only",
               "reproduced_share_A": repro, "doubling_A": dA, "doubling_B": dB,
               "detector_boxes_first_call_B": sum(det_in), "detector_boxes_first_call_A": a_first,
               "predict_boxes_a3trace": sum(pred.get(f["frame"], 0) for f in A),
               "output_boxes": {"A": sum(len(f["output"]) for f in A), "B": sum(len(f["output"]) for f in B)},
               "yield": {"A": yA, "B": yB}, "resets": {k: v["n_resets"] for k, v in rec["variants"].items()},
               "postprocess_callbacks_count": {k: {str(n): sum(1 for f in v["per_frame"]
                                                               if f.get("postprocess_callbacks_before") == n)
                                                   for n in sorted({f.get("postprocess_callbacks_before")
                                                                    for f in v["per_frame"]}, key=str)}
                                               for k, v in rec["variants"].items()},
               "reading": read(repro, dA, dB, yA, yB), "versions": rec["versions"],
               "rule": {"repro_share": REPRO_SHARE, "double_share": DOUBLE_SHARE, "fixed_yield": FIXED_YIELD,
                        "bug_yield": BUG_YIELD}})
    atomic_write(wr / "placement" / "a3input_report.json", json.dumps(out, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_a3input.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_a3input")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

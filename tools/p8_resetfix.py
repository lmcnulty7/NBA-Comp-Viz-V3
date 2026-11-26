"""tools/p8_resetfix.py: the tracker reset fix (docs/research/P8_resetfix.md). This module holds the pure fix used by
tools/p8_device.py's build child and a stored-files estimate for the design. Nothing here builds or trains.

    python3 -m tools.p8_resetfix estimate   -> <P8 workspace>/placement/resetfix_estimate.json and
                                               reports/research/p8_resetfix_estimate.json   (stored files only, no lock)

The bug (docs/research/P8_a3input.md, cause confirmed): detect/tracker.py PlayerTracker.reset sets
self.model.predictor = None; ultralytics 8.4.48 Model.track then re-registers the tracker callbacks
(ultralytics.trackers.track.on_predict_start and on_predict_postprocess_end, as functools.partial objects) and
Model.add_callback appends, so after a reset the tracker updates twice per frame and new detections never confirm.

The fix: after PlayerTracker.reset, remove every tracker callback from the model's callback lists, so Model.track
registers them exactly once again. Applied only in the build child when P8_TRACKER_RESET_FIX=1 (tools/p8_device.py).

Estimate rules (stored files only; the r23 reference-platform sidecars, identity files, the scorer universe, the E1
corpus "ours" boxes on the decoded clip frames):
- Per window: identity["camera_cut_resets"] (cut and exception resets together) and the first reset-like frame of the
  sidecar (80 percent or more of 4 or more previous ids absent; tools.p8_trackreset.reset_like); processed frames from
  that frame on are "after a reset".
- Per pool game, on corpus frames of the scorer's universe: in-frame players; missed (no foot_stab within 2.5 ft);
  raw-reachable misses (missed, and a corpus "ours" box at 0.40 has its bottom-centre foot within 2.5 ft); the same
  split by before or after the window's first reset; far misses (no foot_stab within 5 ft) and those an "ours" box
  reaches within 5 ft; ghost rate of the sidecar's boxes and of the "ours" boxes (more than 3.0 ft from all 10
  players, all boxes) on the same frames.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

P8_NO_GAMES = False
TRACKER_EVENTS = ("on_predict_start", "on_predict_postprocess_end")
TRACKER_MODULE = "ultralytics.trackers.track"
WHO = "tools.p8_resetfix"


# ── the fix (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────
def is_tracker_callback(c) -> bool:
    """A functools.partial of ultralytics.trackers.track.on_predict_start or on_predict_postprocess_end."""
    f = getattr(c, "func", None)
    return f is not None and getattr(f, "__module__", "") == TRACKER_MODULE and \
        getattr(f, "__name__", "") in TRACKER_EVENTS


def strip_tracker_callbacks(callbacks: dict) -> int:
    """Remove every tracker callback from a model's callback lists, in place -> the number removed."""
    n = 0
    for ev in TRACKER_EVENTS:
        keep = [c for c in callbacks.get(ev, []) if not is_tracker_callback(c)]
        n += len(callbacks.get(ev, [])) - len(keep)
        if ev in callbacks:
            callbacks[ev] = keep
    return n


RESET_LOG = {"path": None, "resets": []}


def apply_reset_fix() -> str:
    """In the build child, before the build runs: PlayerTracker.reset also strips the tracker callbacks. Each reset
    appends {"stripped": n} to RESET_LOG (n = 2 when the tracker had registered once: one on_predict_start and one
    on_predict_postprocess_end); write_reset_log() saves it at exit when start_reset_log() named a file."""
    import detect.tracker as DT
    orig = DT.PlayerTracker.reset

    def reset(self):
        orig(self)
        cbs = getattr(getattr(self, "model", None), "callbacks", None)
        RESET_LOG["resets"].append({"stripped": strip_tracker_callbacks(cbs) if isinstance(cbs, dict) else None})
    DT.PlayerTracker.reset = reset
    return "detect.tracker.PlayerTracker.reset"


def write_reset_log() -> None:
    import os
    path = RESET_LOG["path"]
    if path is None:
        return
    part = f"{path}.part"
    with open(part, "w") as fh:
        json.dump({"reset_fix": True, "resets": RESET_LOG["resets"]}, fh)
    os.replace(part, path)


def start_reset_log(path) -> None:
    """<window>_resetfix.json in config.TRACKING_DIR (sportvu.rebuild.build moves every <window>_* file next to the
    window's outputs), written when the build child exits."""
    import atexit
    RESET_LOG["path"] = str(path)
    atexit.register(write_reset_log)


# ── estimate (stored files only) ───────────────────────────────────────────────────────────────────────────────────
def first_reset_index(frames: list) -> int | None:
    from tools.p8_trackreset import reset_like
    ids = [{b["tid"] for b in r["boxes"]} for r in frames]
    for i in range(1, len(ids)):
        if reset_like(ids[i - 1], ids[i]):
            return i
    return None


def cmd_estimate(args) -> dict:
    from tools import p8_teachers as T
    from tools.p8_common import REPO, atomic_write, p8_root
    from tools.p8_e1data import corpus_root
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_frozen import layout
    from tools.p8_placement import BUILDS, CHECK, POOL, SCORES, check_games, rnd
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(check_games(POOL + CHECK), WHO)
    wr = p8_root()
    P = layout(wr)
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    ours = {}
    for k in range(len(plan["chunks"])):
        for key, rec in json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
            g, stem = key.split("/", 1)
            ours[stem] = [p for p in rec[T.OURS] if p[5] == "player" and p[4] >= 0.40]
    out = {}
    for g in POOL + CHECK:
        sf = json.loads((wr / "e1" / SCORES[g]).read_text())
        wins, after = {}, {}
        for w in sf["scored_windows"]:
            fr = sorted(json.loads((P["builds"] / BUILDS[g] / f"{w}_frames.json").read_text())["frames"],
                        key=lambda r: int(r["frame"]))
            ident = json.loads((P["builds"] / BUILDS[g] / f"{w}_identity.json").read_text())["identity"]
            i0 = first_reset_index(fr)
            wins[w] = {"resets": ident.get("camera_cut_resets"), "frames": len(fr),
                       "frames_after_first_reset": 0 if i0 is None else len(fr) - i0}
            after[w] = None if i0 is None else int(fr[i0]["frame"])
        c = {k: 0 for k in ("in_frame", "missed", "raw_reach", "raw_reach_after", "missed_after", "in_frame_after",
                            "far", "far_raw_reach", "side_boxes", "side_ghosts", "ours_boxes", "ours_ghosts",
                            "corpus_frames")}
        if g in POOL:
            game = load_frames(g)
            for f in game["frames"]:
                stem = f"{f['w']}_f{int(f['f']):06d}"
                if stem not in ours:
                    continue
                c["corpus_frames"] += 1
                sv = f["sv"]
                feet = scorer_court(f["Ht"], f["px"])
                of = T.feet_ft(ours[stem], f["Ht"])
                post = after[f["w"]] is not None and int(f["f"]) >= after[f["w"]]
                for j in np.where(f["inside"])[0]:
                    db = np.hypot(*(feet - sv[j]).T).min() if len(feet) else np.inf
                    do = np.hypot(*(of - sv[j]).T).min() if len(of) else np.inf
                    c["in_frame"] += 1
                    c["in_frame_after"] += post
                    if db > 2.5:
                        c["missed"] += 1
                        c["missed_after"] += post
                        if do <= 2.5:
                            c["raw_reach"] += 1
                            c["raw_reach_after"] += post
                    if db > 5.0:
                        c["far"] += 1
                        c["far_raw_reach"] += do <= 5.0
                for arr, kb, kg in ((feet, "side_boxes", "side_ghosts"), (of, "ours_boxes", "ours_ghosts")):
                    if len(arr):
                        D = np.sqrt(((sv[:, None, :] - arr[None, :, :]) ** 2).sum(-1))
                        c[kb] += len(arr)
                        c[kg] += int((D.min(0) > 3.0).sum())
        n = c["in_frame"] or None
        out[g] = {"windows": len(wins), "windows_with_reset": sum(1 for v in wins.values() if (v["resets"] or 0) > 0),
                  "resets_total": sum(v["resets"] or 0 for v in wins.values()),
                  "processed_frames": sum(v["frames"] for v in wins.values()),
                  "frames_after_first_reset": sum(v["frames_after_first_reset"] for v in wins.values()),
                  "counts": c,
                  "rates": None if n is None else {
                      "missed": c["missed"] / n, "raw_reachable_missed": c["raw_reach"] / n,
                      "raw_reachable_missed_after_reset": c["raw_reach_after"] / n,
                      "far_missed": c["far"] / n, "far_raw_reachable": c["far_raw_reach"] / n,
                      "in_frame_after_reset_share": c["in_frame_after"] / n,
                      "sidecar_ghost_rate_all_boxes": c["side_ghosts"] / c["side_boxes"] if c["side_boxes"] else None,
                      "ours_ghost_rate_all_boxes": c["ours_ghosts"] / c["ours_boxes"] if c["ours_boxes"] else None,
                      "sidecar_boxes_per_in_frame": c["side_boxes"] / n, "ours_boxes_per_in_frame": c["ours_boxes"] / n}}
        print(g, json.dumps(rnd({k: out[g][k] for k in ("windows", "windows_with_reset", "resets_total",
                                                        "processed_frames", "frames_after_first_reset", "rates")})))
    res = rnd({"what": "Reset-fix design estimate from stored files (docs/research/P8_resetfix.md); not a result",
               "games": out})
    atomic_write(wr / "placement" / "resetfix_estimate.json", json.dumps(res, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_resetfix_estimate.json", json.dumps(res, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_resetfix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    a = ap.parse_args(argv)
    {"estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()

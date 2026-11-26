"""tools/p8_trackreset.py: tracker resets, empty frames and box yield on the windows that carry the build's
"never tracked nearby" loss (docs/research/P8_trackreset.md). Report only: no model runs, nothing is built or changed.

    python3 -m tools.p8_trackreset measure -> <P8 workspace>/placement/trackreset.json and reports/research/p8_trackreset.json

What an exception does (detect/tracker.py, PlayerTracker.update, read before registering):
- model.track raises numpy.linalg.LinAlgError (Kalman, "MPS precision"): a warning is logged, reset() runs and update
  returns [] for that frame (no boxes).
- model.track raises AttributeError whose text contains "cpu" (the ultralytics 8.4 native re-identification bug): a
  warning is logged, reset() runs, update returns [].
- Any other AttributeError or exception is raised (the build fails).
- After a normal frame, CameraCutDetector.update flags a cut when 80 percent or more of 4 or more previous track ids
  vanished; reset() then runs, after the frame's tracks were returned.
- reset() drops the predictor (a fresh BoT-SORT on the next frame, whose first frame activates every detection at
  once), clears the cut detector, adds one to n_resets and offsets later track ids. build_trajectories.py stores
  n_resets as identity["camera_cut_resets"] in <window>_identity.json: it counts cut resets and exception resets
  together. Log lines are not stored for successful builds (sportvu.rebuild.build keeps stderr only on failure).

Inputs, stored P8-workspace files only: the reference-platform r23 sidecars and identity files of the pool games, the
registered score files' scored windows, the E1 corpus preds ("ours": r23 at 0.40 on the decoded clip frame), and the
build loss version 2 per-player rows (placement/buildloss_v2.json; the tool stops if any is missing).

Rules, fixed before any number:
- Reachable a3: build loss version 2 rows of class a3_never_tracked_nearby with d_raw at most 2.5 ft.
- a3-heavy windows, per game: rank the game's windows by their reachable a3 count; the heavy set is the smallest
  leading set that holds at least 80 percent of the game's reachable a3. The comparison set is every other scored
  window of the pool games.
- Per processed frame (all sidecar rows of a window): the box count; empty when it is 0. On corpus frames: "ours" boxes
  (player, at 0.40); detector fired when at least one; box yield = sidecar boxes over "ours" boxes.
- Per window: processed frames; the share of empty frames; the share of empty frames among corpus frames where the
  detector fired; runs of consecutive empty processed frames (count, longest); track-id turnover (per consecutive
  frame pair with a previous frame of 1 or more ids, the share of this frame's ids not in the previous frame, mean);
  reset-like frames (80 percent or more of 4 or more previous ids absent); identity["camera_cut_resets"] and
  identity["median_players_per_frame"]; median box yield on corpus frames where the detector fired.
- Per reachable a3 player-frame: on an empty frame (its frame has no sidecar box), or on a reset frame (its frame or
  one of the two processed frames before it is reset-like), or neither.
- Reading rule: when at least 50 percent of the pool's reachable a3 lies on empty or reset frames, a tracker
  robustness arm follows (its own registration, then a rebuild of the affected windows on the reference platform under
  the lock, scored by the registered scorer). Otherwise the loss is in tracks the tracker does not output on frames
  where it outputs others, and resets and exceptions do not explain it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_placement import BUILDS, POOL, SCORES, check_games, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_trackreset"
REACH_FT, HEAVY_SHARE, ARM_SHARE = 2.5, 0.80, 0.50
DET_CONF, RESET_FRAC, RESET_MIN = 0.40, 0.80, 4


def need(path):
    from pathlib import Path
    if not Path(path).exists():
        raise SystemExit(f"{WHO}: required file {Path(path).name} is missing")
    return path


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def heavy_windows(counts: dict, share: float = HEAVY_SHARE) -> list:
    """The smallest leading set of windows (by count, ties by name) holding at least share of the total."""
    total = sum(counts.values())
    if not total:
        return []
    out, acc = [], 0
    for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if acc >= share * total:
            break
        out.append(w)
        acc += n
    return out


def runs(flags: list) -> list:
    """Lengths of consecutive True runs."""
    out, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def reset_like(prev_ids: set, ids: set) -> bool:
    return len(prev_ids) >= RESET_MIN and len(prev_ids - ids) / len(prev_ids) >= RESET_FRAC


def window_metrics(frames: list, ours: dict, identity: dict) -> tuple:
    """frames: sidecar rows sorted by frame; ours: {frame: ours box count} on corpus frames.
    -> (metrics, {frame: (empty, reset_near)})."""
    empty = [len(r["boxes"]) == 0 for r in frames]
    ids = [{b["tid"] for b in r["boxes"]} for r in frames]
    resets = [False] + [reset_like(ids[i - 1], ids[i]) for i in range(1, len(ids))]
    turnover = [len(ids[i] - ids[i - 1]) / len(ids[i]) for i in range(1, len(ids)) if ids[i - 1] and ids[i]]
    fired = [i for i, r in enumerate(frames) if ours.get(int(r["frame"]), 0) >= 1]
    yields = [len(frames[i]["boxes"]) / ours[int(frames[i]["frame"])] for i in fired]
    rl = runs(empty)
    status = {}
    for i, r in enumerate(frames):
        near = any(resets[k] for k in range(max(0, i - 2), i + 1))
        status[int(r["frame"])] = (empty[i], near)
    m = {"processed_frames": len(frames), "empty_share": float(np.mean(empty)) if frames else None,
         "corpus_frames_detector_fired": len(fired),
         "empty_share_where_detector_fired": float(np.mean([empty[i] for i in fired])) if fired else None,
         "empty_runs": len(rl), "longest_empty_run": max(rl) if rl else 0,
         "turnover_mean": float(np.mean(turnover)) if turnover else None,
         "reset_like_frames": int(sum(resets)),
         "camera_cut_resets": identity.get("camera_cut_resets"),
         "median_players_per_frame": identity.get("median_players_per_frame"),
         "median_box_yield_where_detector_fired": float(np.median(yields)) if yields else None}
    return m, status


# ── run ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_measure(args) -> dict:
    from tools import p8_teachers as T
    from tools.p8_e1data import corpus_root
    from tools.p8_frozen import layout
    refuse_cv_test(check_games(POOL), WHO)
    wr = p8_root()
    P = layout(wr)
    v2 = json.loads(need(wr / "placement" / "buildloss_v2.json").read_text())["per_player"]
    root = corpus_root()
    plan = json.loads(need(root / "plan.json").read_text())
    ours = {}
    for k in range(len(plan["chunks"])):
        for key, rec in json.loads(need(root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
            tag, stem = key.split("/", 1)
            w, f = stem.rsplit("_f", 1)
            ours.setdefault(w, {})[int(f)] = sum(1 for p in rec[T.OURS] if p[5] == "player" and p[4] >= DET_CONF)
    games, all_a3 = {}, []
    for g in POOL:
        sf = json.loads(need(wr / "e1" / SCORES[g]).read_text())
        a3 = [r for r in v2[g] if r["cls"] == "a3_never_tracked_nearby" and r["d_raw"] <= REACH_FT]
        counts = {}
        for r in a3:
            counts[r["w"]] = counts.get(r["w"], 0) + 1
        heavy = heavy_windows(counts)
        per_w, status = {}, {}
        for w in sf["scored_windows"]:
            fr = sorted(json.loads(need(P["builds"] / BUILDS[g] / f"{w}_frames.json").read_text())["frames"],
                        key=lambda r: int(r["frame"]))
            ident = json.loads(need(P["builds"] / BUILDS[g] / f"{w}_identity.json").read_text())["identity"]
            per_w[w], status[w] = window_metrics(fr, ours.get(w, {}), ident)
        on = {"empty": 0, "reset": 0, "neither": 0}
        for r in a3:
            e, rs = status[r["w"]][int(r["f"])]
            on["empty" if e else ("reset" if rs else "neither")] += 1
            all_a3.append(("empty" if e else ("reset" if rs else "neither")))

        def agg(ws):
            ms = [per_w[w] for w in ws]
            num = lambda k: [m[k] for m in ms if m[k] is not None]
            return {"windows": len(ws), "processed_frames": sum(m["processed_frames"] for m in ms),
                    "empty_share_mean": float(np.mean(num("empty_share"))) if num("empty_share") else None,
                    "empty_share_where_detector_fired_mean": float(np.mean(num("empty_share_where_detector_fired")))
                    if num("empty_share_where_detector_fired") else None,
                    "longest_empty_run_max": max(num("longest_empty_run")) if num("longest_empty_run") else 0,
                    "turnover_mean": float(np.mean(num("turnover_mean"))) if num("turnover_mean") else None,
                    "reset_like_frames": sum(num("reset_like_frames")),
                    "camera_cut_resets": sum(num("camera_cut_resets")),
                    "median_players_per_frame_median": float(np.median(num("median_players_per_frame")))
                    if num("median_players_per_frame") else None,
                    "box_yield_median": float(np.median(num("median_box_yield_where_detector_fired")))
                    if num("median_box_yield_where_detector_fired") else None}
        others = [w for w in sf["scored_windows"] if w not in heavy]
        games[g] = {"reachable_a3": len(a3), "a3_by_window": counts, "heavy_windows": heavy,
                    "heavy_share": sum(counts[w] for w in heavy) / len(a3) if a3 else None,
                    "a3_on": on, "heavy": {w: per_w[w] for w in heavy}, "heavy_summary": agg(heavy),
                    "others_summary": agg(others)}
        print(g, json.dumps(rnd({k: games[g][k] for k in ("reachable_a3", "heavy_windows", "a3_on", "heavy_summary",
                                                          "others_summary")})))
    n = len(all_a3)
    share = (sum(x in ("empty", "reset") for x in all_a3) / n) if n else None
    reading = {"share_on_empty_or_reset": share, "arm_share": ARM_SHARE,
               "tracker_robustness_arm": bool(share is not None and share >= ARM_SHARE)}
    out = rnd({"what": "Tracker resets, empty frames and box yield on the a3-heavy windows (docs/research/P8_trackreset.md); "
                       "report only; stored files; pool games", "games": games, "reading": reading,
               "rule": {"reach_ft": REACH_FT, "heavy_share": HEAVY_SHARE, "arm_share": ARM_SHARE,
                        "reset_frac": RESET_FRAC, "reset_min": RESET_MIN, "det_conf": DET_CONF}})
    d = wr / "placement"
    atomic_write(d / "trackreset.json", json.dumps(out, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_trackreset.json", json.dumps(out, indent=1))
    print(json.dumps(reading, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_trackreset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()

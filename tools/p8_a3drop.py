"""tools/p8_a3drop.py: what the build's tracker drops on the a3-heavy windows (docs/research/P8_a3drop.md). Report only:
stored files (the a3run replay records, truth rows, SportVU), no model, no lock.

    python3 -m tools.p8_a3drop measure  -> <P8 workspace>/placement/a3drop.json and reports/research/p8_a3drop.json

The tracker's real gates (ultralytics 8.4.48 trackers/byte_tracker.py and bot_sort.py, botsort.yaml; read before
registering):
- Detections reach the tracker at the build's confidence 0.40, so track_high_thresh 0.25 and new_track_thresh 0.25 never
  bind and the second, low-score association (0.1 to 0.25) never receives a box.
- First association: tracked and lost tracks against detections, cost 1 - IoU x score (fuse_score true; BoT-SORT's
  proximity mask only acts with re-identification, which is off); matched when the cost is at most match_thresh 0.8,
  that is IoU x score at least 0.2 with the Kalman-predicted, GMC-warped track box.
- Unconfirmed tracks (created on the previous frame) against the remaining detections: cost at most 0.7 (IoU x score at
  least 0.3); an unconfirmed track that finds no match is removed.
- A remaining detection starts a new track, which is not output on its first frame (is_activated is false except on the
  tracker's first frame) and is recorded in the replay as an unconfirmed box.
- remove_duplicate_stracks drops the younger of a tracked and a lost track whose boxes overlap at IoU above 0.85.
- Output: every activated tracked track, with its Kalman state box (track.result), not the detection box.

Rules, fixed before any number (the a3run yield result is already seen):
- Windows: the six a3-heavy windows of P8_a3run.md. Frames: the replay frames that are scored frames (truth rows,
  tools.p8_footbias.load_frames), since the SportVU test needs the truth camera.
- Dropped (1) box: a predict box with no kept (2) box at IoU 0.5 or more in its frame (P8_a3run.md's definition).
- Measures per dropped box: confidence; its best IoU with a kept box of the frame; whether that kept box itself matches
  another predict box at IoU 0.5 or more; the foot (bottom centre) through the truth camera, as the scorer maps a foot,
  and its distance to the nearest of the 10 SportVU players; whether a kept box lies within two processed frames before
  or after at IoU 0.3 or more; box height bin (under 150, 150 to 200, 200 to 250, 250 px and over); camera third.
- Classes, the first that holds:
  1 duplicate: a kept box at IoU 0.3 or more that matches another predict box at IoU 0.5 or more (the player is kept
    through another box);
  2 displaced track box: a kept box at IoU 0.3 or more that matches no predict box at IoU 0.5 or more (the track's
    output box has moved off its own detection);
  3 low confidence: confidence under 0.5 (with fuse_score, a detection below 0.5 needs IoU at least 0.4 with the
    predicted track to pass the first gate, and 0.6 to confirm an unconfirmed one);
  4 unmatched real player: foot within 2.5 ft of a SportVU player; split 4a gap (a kept box within two frames at IoU 0.3
    or more) and 4b never tracked nearby;
  5 near a player: foot 2.5 to 3.0 ft from the nearest player;
  6 false positive: foot more than 3.0 ft from every player (the scorer's ghost rule).
- Per class: count and share of dropped boxes, per window and pooled; and the same over the dropped boxes whose foot is
  within 2.5 ft of a far-miss player of the frame (an in-frame SportVU player with no build foot_stab within 5 ft under
  the truth camera), the players the a3 loss is made of.
- Reading rule: a class leads to a tracker arm when it holds at least 25 percent of the pooled dropped boxes. Class 2
  leads to a tracker-state arm (output the matched detection box, or change GMC); class 3 to a score-fusion arm
  (fuse_score off or a looser match_thresh); 4a to an association arm (match_thresh, track_buffer); 4b to an
  activation arm (output a new track on its first frame, or new_track_thresh). Classes 1, 5 and 6 lead to nothing (the
  drop loses no player, or none that can be scored). Any arm needs its own registration, then a rebuild on the
  reference platform under the lock, scored by the registered scorer.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_placement import POOL, check_games, hbin, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_a3drop"
HEAVY = ("gsw_bkn_2015_s04_w00", "gsw_cha_2016_s05_w00", "gsw_cha_2016_s05_w03", "gsw_sac_2015_s01_w00",
         "gsw_sac_2015_s09_w01", "gsw_sac_2015_s07_w00")
KEEP_IOU, NEAR_IOU, LOW_CONF, GAP_FRAMES = 0.5, 0.3, 0.5, 2
HIT_FT, GHOST_FT, REACH_FT, ARM_SHARE = 2.5, 3.0, 2.5, 0.25
CLASSES = ("1_duplicate", "2_displaced_track_box", "3_low_confidence", "4a_real_player_gap",
           "4b_real_player_never_tracked", "5_near_player", "6_false_positive")


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


def classify(near_kept_is_dup: bool | None, conf: float, d_sv: float, gap: bool) -> str:
    """near_kept_is_dup: None when no kept box reaches IoU 0.3; True when that kept box matches another predict box."""
    if near_kept_is_dup is True:
        return "1_duplicate"
    if near_kept_is_dup is False:
        return "2_displaced_track_box"
    if conf < LOW_CONF:
        return "3_low_confidence"
    if d_sv <= HIT_FT:
        return "4a_real_player_gap" if gap else "4b_real_player_never_tracked"
    if d_sv <= GHOST_FT:
        return "5_near_player"
    return "6_false_positive"


def frame_drops(pred: list, kept: list) -> list:
    """Per dropped predict box: (index, best kept IoU, that kept box matches another predict box or None)."""
    out = []
    for i, p in enumerate(pred):
        ious = [iou(p[:4], k[:4]) for k in kept]
        if ious and max(ious) >= KEEP_IOU:
            continue
        if ious and max(ious) >= NEAR_IOU:
            k = kept[int(np.argmax(ious))]
            other = any(iou(q[:4], k[:4]) >= KEEP_IOU for j, q in enumerate(pred) if j != i)
            out.append((i, max(ious), other))
        else:
            out.append((i, max(ious) if ious else 0.0, None))
    return out


def arms(shares: dict) -> list:
    lead = {"2_displaced_track_box": "tracker-state arm", "3_low_confidence": "score-fusion arm",
            "4a_real_player_gap": "association arm", "4b_real_player_never_tracked": "activation arm"}
    return [lead[c] for c, s in shares.items() if c in lead and s is not None and s >= ARM_SHARE]


# ── run ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_measure(args) -> dict:
    from sportvu.eval import near_side
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_scorecard import third_of
    refuse_cv_test(check_games(POOL), WHO)
    wr = p8_root()
    games = {g: load_frames(g) for g in POOL}
    per_w, allrows = {}, []
    for w in HEAVY:
        g = w.rsplit("_s", 1)[0]
        rec = json.loads(need(wr / "placement" / "a3run" / f"{w}.json").read_text())
        frames = rec["per_frame"]
        idx = {fr["frame"]: i for i, fr in enumerate(frames)}
        truth = {fr["f"]: fr for fr in games[g]["frames"] if fr["w"] == w}
        rows = []
        for f, tf in truth.items():
            if f not in idx:
                continue
            fr = frames[idx[f]]
            kept = [t[:4] for t in fr["track"]]
            pred = fr["pred"]
            drops = frame_drops(pred, kept)
            if not drops:
                continue
            feet = scorer_court(tf["Ht"], [[(p[0] + p[2]) / 2, p[3]] for p in pred])
            nh = near_side(tf["Ht"])
            sv = tf["sv"]
            # far-miss players of this frame: in-frame, no build foot_stab within 5 ft under the truth camera
            bfeet = scorer_court(tf["Ht"], tf["px"])
            far = [sv[j] for j in np.where(tf["inside"])[0]
                   if not len(bfeet) or np.hypot(*(bfeet - sv[j]).T).min() > 5.0]
            near_frames = [frames[k] for k in range(max(0, idx[f] - GAP_FRAMES), min(len(frames), idx[f] + GAP_FRAMES + 1))
                           if k != idx[f]]
            for i, best, dup in drops:
                p = pred[i]
                d = np.hypot(*(sv - feet[i]).T)
                gap = any(iou(p[:4], t[:4]) >= NEAR_IOU for nf in near_frames for t in nf["track"])
                cls = classify(dup, float(p[4]), float(d.min()), gap)
                near_a3 = any(np.hypot(*(np.asarray(s) - feet[i])) <= HIT_FT for s in far)
                rows.append({"w": w, "f": f, "cls": cls, "conf": float(p[4]), "best_kept_iou": float(best),
                             "d_sv": float(d.min()), "gap": gap, "height": hbin(p[3] - p[1]),
                             "third": third_of(float(feet[i][1]), nh), "near_a3": near_a3})
        per_w[w] = rows
        allrows += rows

    def summary(rows):
        n = len(rows)
        return {"dropped": n, "classes": {c: {"count": sum(r["cls"] == c for r in rows),
                                              "share": (sum(r["cls"] == c for r in rows) / n) if n else None}
                                          for c in CLASSES},
                "median_conf": float(np.median([r["conf"] for r in rows])) if rows else None,
                "by_third": {t: sum(r["third"] == t for r in rows) for t in ("near", "middle", "far")},
                "by_height": {h: sum(r["height"] == h for r in rows) for h in ("<150", "150-200", "200-250", ">=250")}}
    pooled = summary(allrows)
    near = summary([r for r in allrows if r["near_a3"]])
    out = rnd({"what": "What the build's tracker drops on the a3-heavy windows (docs/research/P8_a3drop.md); report "
                       "only; stored files (a3run replays, truth rows, SportVU)",
               "windows": {w: summary(r) for w, r in per_w.items()}, "pooled": pooled, "near_reachable_a3": near,
               "arms": arms({c: v["share"] for c, v in pooled["classes"].items()}),
               "rule": {"keep_iou": KEEP_IOU, "near_iou": NEAR_IOU, "low_conf": LOW_CONF, "gap_frames": GAP_FRAMES,
                        "hit_ft": HIT_FT, "ghost_ft": GHOST_FT, "arm_share": ARM_SHARE}})
    atomic_write(wr / "placement" / "a3drop.json", json.dumps(out, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_a3drop.json", json.dumps(out, indent=1))
    print(json.dumps({"pooled": out["pooled"], "near_reachable_a3": out["near_reachable_a3"]["classes"],
                      "arms": out["arms"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_a3drop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()

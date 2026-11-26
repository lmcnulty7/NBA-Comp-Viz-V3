"""tools/p8_buildloss.py: where the build loses r23's raw boxes for far-miss players (docs/research/P8_buildloss.md).
Report only: no model runs, nothing is built, trained or changed.

    python3 -m tools.p8_buildloss measure  -> <P8 workspace>/placement/buildloss.json and reports/research/p8_buildloss.json
    python3 -m tools.p8_buildloss measure2 -> (v2) <P8 workspace>/placement/buildloss_v2.json (with per-player rows) and
                                              reports/research/p8_buildloss_v2.json

Build path, read before registering (what is recorded where):
- Gate v2 (build_trajectories.py, --pregate and the per-frame gate): a rejected frame is skipped whole and gets no
  sidecar row. The scorer's universe needs a sidecar row, so a gate drop never produces a far miss in this population
  (it lowers coverage instead). No gate class is therefore possible here.
- Detector and tracker (detect/tracker.py): one ultralytics model.track call, confidence 0.40, BoT-SORT
  (botsort.yaml: track_high_thresh 0.25, new_track_thresh 0.25, track_buffer 30, match_thresh 0.8). Only tracks the
  tracker outputs reach the sidecar; raw detections are not stored. In ultralytics 8.4.48 (trackers/byte_tracker.py)
  a new track is activated at once only on the tracker's first frame; later it starts unconfirmed and is output after a
  second match. A camera-cut detector (detect/camera_cut.py) resets the tracker when 80 percent of 4 or more tracks
  vanish at once; resets are not recorded, but new track ids show them.
- Foot stabilisation (detect/footpoint.py): per track, an EMA of the foot (alpha 0.6) and an occlusion correction
  that moves the foot down to the median box height when a box is shorter than 0.85 of it; the sidecar records
  foot_raw, foot_stab and foot_corrected.
- Sidecar rows (build_trajectories._frame_record): per processed frame, boxes with tid, bbox, foot_raw, foot_stab,
  foot_corrected.

Inputs, stored P8-workspace files only: tools.p8_footbias.load_frames (scorer universe, truth, count check), the
reference-platform r23 sidecars (all processed frames, for neighbours), M4's raw CPU r23 boxes on the corpus JPEGs.
hou_cle_2016 has no corpus frames and no M4 boxes, so it cannot be measured this way.

Rules, fixed before any number:
- Population, pool games only: far misses (no foot_stab within 5 ft under the truth camera, the scorer's 10 px
  in-frame margin) for which an M4 r23 raw box at 0.40 has its bottom-centre foot, through the truth camera, within
  5 ft (order-free). The raw box is the one whose foot is nearest. Check: per game the count equals the order-free
  "raw had a box" count of reports/research/p8_detrow.json, or the tool stops.
- Classes (a partition, in this order):
  a1 activation lag: no sidecar box at IoU >= 0.5 with the raw box in this frame; no box at IoU >= 0.3 in the previous
     processed frame; a box at IoU >= 0.3 in the next processed frame whose track id appears in neither this frame nor
     the previous one (the tracker started the track later);
  a2 track gap: no sidecar box at IoU >= 0.5 in this frame; a box at IoU >= 0.3 in the previous processed frame, or in
     the next with a track id already present before (a track existed and was not output here: lost, cut or unmatched);
     this is where "a track dropped or cut there" shows, since a box that exists in the frame would be scored;
  a3 never tracked nearby: no sidecar box at IoU >= 0.5 here and none at IoU >= 0.3 in the previous or next processed
     frame (the tracker never output this player around here, or the build's own detector on the decoded video did not
     fire: the data cannot tell them apart);
  b  tracked box differs: a sidecar box at IoU >= 0.5 exists, but its foot_raw, through the truth camera, is also more
     than 5 ft away (the tracker's box state, not the stabiliser);
  c  foot stabilisation moved it: a sidecar box at IoU >= 0.5 exists, its foot_raw is within 5 ft and its foot_stab is
     not; split c_occlusion (foot_corrected true) and c_ema (false);
  e  other: anything not above (expected empty).
- (d) decode or frame mismatch between corpus JPEG and video is not separable per player. It is sized from M6's
  stand-in on gsw_bkn_2015 with both sides on CPU: raw boxes at 0.40 with no sidecar box at IoU 0.5 (1,356 of 19,511,
  6.9 percent), which bounds tracking plus decode together; it is mixed into a1 to a3.
- Report only: for a1 and a2, the frames where the track ids look reset (80 percent or more of 4 or more previous ids
  absent).
- Per class, per game: count, share of the population, rate among in-frame players, window intervals (2,000
  resamples, seed 0, windows within the game).
- Reading rule: a class matters at a rate of at least 0.010 among in-frame players on every pool game. a1 leads to a
  tracker activation arm; a2 to a tracker buffer or association arm; a3 to a check of the build's own detector on the
  decoded video (a model run, under the lock); b to a tracker box-state arm; c to a foot_stab arm. Any arm needs its
  own registration and a rebuild under the lock, scored on the reference platform.

Version 2 (2026-11-26, after the ml-review of P8_detrow.md and after version 1's numbers were seen; registered before
version 2's numbers):
- Per player, recorded: d_raw (nearest M4 raw bottom-centre at 0.40, through the truth camera), d_video (nearest
  r23 box at 0.40 in the E1 corpus preds "ours", run on the decoded clip frame before the JPEG was written, the same
  clip the build read), d_build (nearest foot_stab, above 5 ft by definition), and for the box matched at IoU >= 0.5
  its foot_raw and foot_stab distances.
- d, decode, is now measured per player: no sidecar box at IoU >= 0.5, and no r23 box at 0.40 within 5 ft on the
  decoded video frame (corpus "ours"), so the box exists only on the JPEG. Order: a matched box gives b or c (as
  version 1); otherwise d if the video-decoded detector has no box within 5 ft; otherwise a1, a2 or a3.
- Reachable share: only a recovered foot within 2.5 ft of the player can lower the missed row; a box recovered at 2.5
  to 5 ft becomes a placement miss. Each class's reachable rate counts only its players with d_raw at most 2.5 ft.
- Reading rule (replaces version 1's): a class matters when its reachable rate among in-frame players is at least
  0.010 on every pool game (point estimate; window intervals, 2,000 resamples, seed 0, reported beside it). Arms as
  version 1; d leads to no build-side arm (the detector input differs, not the build).
- Fails loudly if a record or corpus file it needs is missing.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_placement import BUILDS, POOL, check_games, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_buildloss"
NEAR_FT, DET_CONF = 5.0, 0.40
IOU_SAME, IOU_NEIGHBOUR = 0.5, 0.3
CLASSES = ("a1_activation_lag", "a2_track_gap", "a3_never_tracked_nearby", "b_tracked_box_differs",
           "c_occlusion", "c_ema", "e_other")
BAR = 0.010
BOOT_N, SEED = 2000, 0
RESET_FRAC, RESET_MIN = 0.80, 4


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def best_match(raw_box, boxes: list, thr: float):
    """The sidecar box with the largest IoU to raw_box if it reaches thr, else None."""
    best, bi = 0.0, None
    for i, b in enumerate(boxes):
        v = iou(raw_box, b["bbox"])
        if v > best:
            best, bi = v, i
    return boxes[bi] if bi is not None and best >= thr else None


def classify(here, prev_match, next_match, prev_ids: set, here_ids: set, foot_raw_ok: bool | None,
             foot_stab_ok: bool | None) -> str:
    """here: the sidecar box matched at IoU >= 0.5 (or None); prev_match/next_match: boxes matched at IoU >= 0.3 in the
    neighbouring processed frames (or None)."""
    if here is None:
        if prev_match is not None:
            return "a2_track_gap"
        if next_match is not None:
            return "a1_activation_lag" if next_match["tid"] not in prev_ids | here_ids else "a2_track_gap"
        return "a3_never_tracked_nearby"
    if foot_raw_ok is False:
        return "b_tracked_box_differs"
    if foot_raw_ok and foot_stab_ok is False:
        return "c_occlusion" if here.get("foot_corrected") else "c_ema"
    return "e_other"


def looks_reset(prev_ids: set, here_ids: set) -> bool:
    return len(prev_ids) >= RESET_MIN and len(prev_ids - here_ids) / len(prev_ids) >= RESET_FRAC


def matters(rates: dict) -> bool:
    return all(rates.get(g) is not None and rates[g] >= BAR for g in POOL)


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def game_rows(tag: str) -> tuple:
    from tools import p8_teachers as T
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_frozen import layout
    game = load_frames(tag)
    P = layout(p8_root())
    m4 = json.loads((p8_root() / "e1" / f"m4_preds__{tag}.json").read_text())
    if m4["game"] != tag or m4["device"] != "cpu":
        raise SystemExit(f"{WHO}: m4_preds__{tag}.json is not the CPU run on {tag}")
    m4 = m4["preds"]
    side, order = {}, {}
    rows, in_frame, resets = [], {}, {"a1_activation_lag": 0, "a2_track_gap": 0}
    for fr in game["frames"]:
        w, f = fr["w"], int(fr["f"])
        n_in = int(fr["inside"].sum())
        in_frame[w] = in_frame.get(w, 0) + n_in
        if w not in side:
            s = json.loads((P["builds"] / BUILDS[tag] / f"{w}_frames.json").read_text())["frames"]
            side = {w: {int(r["frame"]): r for r in s}}
            order = {w: sorted(side[w])}
        stem = f"{w}_f{f:06d}"
        rp = m4.get(stem)
        if not n_in or rp is None:
            continue
        feet = scorer_court(fr["Ht"], fr["px"])
        raw = [b for b in rp["r23"] if b[4] >= DET_CONF]
        rfeet = T.feet_ft(raw, fr["Ht"])
        k = order[w].index(f)
        prev_r = side[w][order[w][k - 1]] if k > 0 else None
        next_r = side[w][order[w][k + 1]] if k + 1 < len(order[w]) else None
        here_boxes = side[w][f]["boxes"]
        here_ids = {b["tid"] for b in here_boxes}
        prev_ids = {b["tid"] for b in prev_r["boxes"]} if prev_r else set()
        for j in np.where(fr["inside"])[0]:
            sv = fr["sv"][j]
            d = np.hypot(*(feet - sv).T).min() if len(feet) else np.inf
            if d <= NEAR_FT or not len(raw):
                continue
            dr = np.hypot(*(rfeet - sv).T)
            if dr.min() > NEAR_FT:
                continue
            rb = raw[int(dr.argmin())][:4]
            here = best_match(rb, here_boxes, IOU_SAME)
            pm = best_match(rb, prev_r["boxes"], IOU_NEIGHBOUR) if prev_r else None
            nm = best_match(rb, next_r["boxes"], IOU_NEIGHBOUR) if next_r else None
            fr_ok = fs_ok = None
            if here is not None:
                fr_ok = bool(np.hypot(*(scorer_court(fr["Ht"], [here["foot_raw"]])[0] - sv)) <= NEAR_FT)
                fs_ok = bool(np.hypot(*(scorer_court(fr["Ht"], [here["foot_stab"]])[0] - sv)) <= NEAR_FT)
            cls = classify(here, pm, nm, prev_ids, here_ids, fr_ok, fs_ok)
            if cls in resets and looks_reset(prev_ids, here_ids):
                resets[cls] += 1
            rows.append({"w": w, "cls": cls})
    return rows, in_frame, resets


def summarise(rows: list, in_frame: dict) -> dict:
    n_in, n = sum(in_frame.values()), len(rows)
    return {"in_frame": n_in, "population": n,
            "classes": {c: {"count": sum(r["cls"] == c for r in rows),
                            "share": (sum(r["cls"] == c for r in rows) / n) if n else None,
                            "rate_in_frame": (sum(r["cls"] == c for r in rows) / n_in) if n_in else None}
                        for c in CLASSES}}


def boot(rows: list, in_frame: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    keys = sorted(in_frame)
    cnt = {}
    for r in rows:
        cnt.setdefault(r["w"], {c: 0 for c in CLASSES})[r["cls"]] += 1
    rng = np.random.default_rng(seed)
    reps = {c: [] for c in CLASSES}
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        nin = sum(in_frame[w] for w in pick)
        for c in CLASSES:
            reps[c].append(sum(cnt.get(w, {}).get(c, 0) for w in pick) / nin if nin else 0.0)
    return {c: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for c, v in reps.items()}


def cmd_measure(args) -> dict:
    refuse_cv_test(check_games(POOL), WHO)
    det = json.loads((REPO / "reports" / "research" / "p8_detrow.json").read_text())
    games = {}
    for t in POOL:
        rows, in_frame, resets = game_rows(t)
        expect = det["games"][t]["report_only_prevalence"]["raw_had_box"]["count"]
        if len(rows) != expect:
            raise SystemExit(f"{WHO}: {t} population {len(rows)} differs from p8_detrow.json's {expect}")
        games[t] = {**summarise(rows, in_frame), "rate_intervals": boot(rows, in_frame),
                    "report_only_reset_like_frames": resets}
        print(t, json.dumps(rnd({c: games[t]["classes"][c] for c in CLASSES})))
    reading = {c: {"rates": {g: games[g]["classes"][c]["rate_in_frame"] for g in POOL},
                   "matters": matters({g: games[g]["classes"][c]["rate_in_frame"] for g in POOL})} for c in CLASSES}
    m6 = json.loads((REPO / "reports" / "research" / "p8_e1_m6.json").read_text())["m4_raw_vs_cpu_sidecar"]["r23"]
    out = rnd({"what": "Where the build loses r23's raw boxes for far-miss players (docs/research/P8_buildloss.md); "
                       "report only; stored files; pool games only (hou_cle_2016 has no M4 raw boxes)",
               "games": games, "reading": reading,
               "d_decode_bound": {"source": "reports/research/p8_e1_m6.json m4_raw_vs_cpu_sidecar r23 (gsw_bkn_2015, "
                                            "both sides on CPU)", "raw_only": m6["raw_only"], "raw": m6["raw"],
                                  "share": m6["raw_only"] / m6["raw"],
                                  "note": "tracking and decode together; not separable per player"},
               "rule": {"near_ft": NEAR_FT, "det_conf": DET_CONF, "iou_same": IOU_SAME, "iou_neighbour": IOU_NEIGHBOUR,
                        "bar": BAR, "boot": {"n": BOOT_N, "seed": SEED, "unit": "window"}}})
    d = p8_root() / "placement"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "buildloss.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_buildloss.json", text)
    print(json.dumps({"matters": {c: v["matters"] for c, v in reading.items()}}, indent=1))
    return out


# ── version 2 ──────────────────────────────────────────────────────────────────────────────────────────────────────
CLASSES_V2 = ("d_decode",) + CLASSES
REACH_FT = 2.5


def need(path):
    from pathlib import Path as _P
    if not _P(path).exists():
        raise SystemExit(f"{WHO}: required file {_P(path).name} is missing")
    return path


def classify_v2(here, video_ok: bool, prev_match, next_match, prev_ids: set, here_ids: set,
                foot_raw_ok: bool | None, foot_stab_ok: bool | None) -> str:
    if here is None and not video_ok:
        return "d_decode"
    return classify(here, prev_match, next_match, prev_ids, here_ids, foot_raw_ok, foot_stab_ok)


def game_rows_v2(tag: str) -> tuple:
    from tools import p8_teachers as T
    from tools.p8_e1data import corpus_root
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_frozen import layout
    game = load_frames(tag)
    P = layout(p8_root())
    m4 = json.loads(need(p8_root() / "e1" / f"m4_preds__{tag}.json").read_text())
    if m4["game"] != tag or m4["device"] != "cpu":
        raise SystemExit(f"{WHO}: m4_preds__{tag}.json is not the CPU run on {tag}")
    m4 = m4["preds"]
    root = corpus_root()
    plan = json.loads(need(root / "plan.json").read_text())
    video = {}
    for k in range(len(plan["chunks"])):
        for key, rec in json.loads(need(root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
            if key.startswith(tag + "/"):
                video[key.split("/", 1)[1]] = [p for p in rec[T.OURS] if p[5] == "player"]
    side, order = {}, {}
    rows, in_frame = [], {}
    for fr in game["frames"]:
        w, f = fr["w"], int(fr["f"])
        n_in = int(fr["inside"].sum())
        in_frame[w] = in_frame.get(w, 0) + n_in
        if w not in side:
            s = json.loads(need(P["builds"] / BUILDS[tag] / f"{w}_frames.json").read_text())["frames"]
            side = {w: {int(r["frame"]): r for r in s}}
            order = {w: sorted(side[w])}
        stem = f"{w}_f{f:06d}"
        rp = m4.get(stem)
        if not n_in or rp is None:
            continue
        if stem not in video:
            raise SystemExit(f"{WHO}: {stem} has M4 raw boxes but no corpus preds")
        feet = scorer_court(fr["Ht"], fr["px"])
        raw = [b for b in rp["r23"] if b[4] >= DET_CONF]
        rfeet = T.feet_ft(raw, fr["Ht"])
        vfeet = T.feet_ft([b for b in video[stem] if b[4] >= DET_CONF], fr["Ht"])
        k = order[w].index(f)
        prev_r = side[w][order[w][k - 1]] if k > 0 else None
        next_r = side[w][order[w][k + 1]] if k + 1 < len(order[w]) else None
        here_boxes = side[w][f]["boxes"]
        here_ids = {b["tid"] for b in here_boxes}
        prev_ids = {b["tid"] for b in prev_r["boxes"]} if prev_r else set()
        for j in np.where(fr["inside"])[0]:
            sv = fr["sv"][j]
            d_build = float(np.hypot(*(feet - sv).T).min()) if len(feet) else float("inf")
            if d_build <= NEAR_FT or not len(raw):
                continue
            dr = np.hypot(*(rfeet - sv).T)
            if dr.min() > NEAR_FT:
                continue
            d_video = float(np.hypot(*(vfeet - sv).T).min()) if len(vfeet) else float("inf")
            rb = raw[int(dr.argmin())][:4]
            here = best_match(rb, here_boxes, IOU_SAME)
            pm = best_match(rb, prev_r["boxes"], IOU_NEIGHBOUR) if prev_r else None
            nm = best_match(rb, next_r["boxes"], IOU_NEIGHBOUR) if next_r else None
            d_fr = d_fs = None
            if here is not None:
                d_fr = float(np.hypot(*(scorer_court(fr["Ht"], [here["foot_raw"]])[0] - sv)))
                d_fs = float(np.hypot(*(scorer_court(fr["Ht"], [here["foot_stab"]])[0] - sv)))
            cls = classify_v2(here, d_video <= NEAR_FT, pm, nm, prev_ids, here_ids,
                              None if d_fr is None else d_fr <= NEAR_FT, None if d_fs is None else d_fs <= NEAR_FT)
            rows.append({"w": w, "f": f, "cls": cls, "d_raw": float(dr.min()), "d_video": d_video,
                         "d_build": d_build, "d_match_foot_raw": d_fr, "d_match_foot_stab": d_fs,
                         "foot_corrected": None if here is None else bool(here.get("foot_corrected"))})
    return rows, in_frame


def summarise_v2(rows: list, in_frame: dict) -> dict:
    n_in, n = sum(in_frame.values()), len(rows)
    out = {"in_frame": n_in, "population": n, "reachable": sum(r["d_raw"] <= REACH_FT for r in rows), "classes": {}}
    for c in CLASSES_V2:
        rs = [r for r in rows if r["cls"] == c]
        reach = [r for r in rs if r["d_raw"] <= REACH_FT]
        out["classes"][c] = {"count": len(rs), "share": len(rs) / n if n else None,
                             "rate_in_frame": len(rs) / n_in if n_in else None, "reachable": len(reach),
                             "reachable_rate_in_frame": len(reach) / n_in if n_in else None}
    cs = [r for r in rows if r["cls"] in ("c_occlusion", "c_ema")]
    out["c_shift_ft"] = {"median_foot_raw": float(np.median([r["d_match_foot_raw"] for r in cs])) if cs else None,
                         "median_foot_stab": float(np.median([r["d_match_foot_stab"] for r in cs])) if cs else None}
    out["d_raw_quartiles"] = [float(np.percentile([r["d_raw"] for r in rows], q)) for q in (25, 50, 75)] if rows else None
    return out


def boot_v2(rows: list, in_frame: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    keys = sorted(in_frame)
    cnt = {}
    for r in rows:
        if r["d_raw"] <= REACH_FT:
            cnt.setdefault(r["w"], {c: 0 for c in CLASSES_V2})[r["cls"]] += 1
    rng = np.random.default_rng(seed)
    reps = {c: [] for c in CLASSES_V2}
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        nin = sum(in_frame[w] for w in pick)
        for c in CLASSES_V2:
            reps[c].append(sum(cnt.get(w, {}).get(c, 0) for w in pick) / nin if nin else 0.0)
    return {c: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for c, v in reps.items()}


def cmd_measure2(args) -> dict:
    refuse_cv_test(check_games(POOL), WHO)
    det = json.loads(need(REPO / "reports" / "research" / "p8_detrow.json").read_text())
    games, per_player = {}, {}
    for t in POOL:
        rows, in_frame = game_rows_v2(t)
        expect = det["games"][t]["report_only_prevalence"]["raw_had_box"]["count"]
        if len(rows) != expect:
            raise SystemExit(f"{WHO}: {t} population {len(rows)} differs from p8_detrow.json's {expect}")
        games[t] = {**summarise_v2(rows, in_frame), "reachable_rate_intervals": boot_v2(rows, in_frame)}
        per_player[t] = rows
        print(t, json.dumps(rnd({c: {k: games[t]["classes"][c][k] for k in ("count", "reachable",
                                                                           "reachable_rate_in_frame")}
                                 for c in CLASSES_V2})))
    reading = {c: {"reachable_rates": {g: games[g]["classes"][c]["reachable_rate_in_frame"] for g in POOL},
                   "matters": matters({g: games[g]["classes"][c]["reachable_rate_in_frame"] for g in POOL})}
               for c in CLASSES_V2}
    out = rnd({"what": "Build loss version 2 (docs/research/P8_buildloss.md): per-player distances, decode measured per "
                       "player against the corpus preds on the decoded clip frame, reading on the reachable share "
                       "(raw foot within 2.5 ft); report only; stored files; pool games only",
               "games": games, "reading": reading,
               "rule": {"near_ft": NEAR_FT, "reach_ft": REACH_FT, "det_conf": DET_CONF, "iou_same": IOU_SAME,
                        "iou_neighbour": IOU_NEIGHBOUR, "bar": BAR, "boot": {"n": BOOT_N, "seed": SEED,
                                                                              "unit": "window"}}})
    d = p8_root() / "placement"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "buildloss_v2.json", json.dumps({**out, "per_player": rnd(per_player)}))
    atomic_write(REPO / "reports" / "research" / "p8_buildloss_v2.json", json.dumps(out, indent=1))
    print(json.dumps({"matters": {c: v["matters"] for c, v in reading.items()}}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_buildloss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    sub.add_parser("measure2")
    a = ap.parse_args(argv)
    {"measure": cmd_measure, "measure2": cmd_measure2}[a.cmd](a)


if __name__ == "__main__":
    main()

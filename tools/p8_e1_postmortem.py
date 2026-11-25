"""tools/p8_e1_postmortem.py: E1 post-mortem, a measurement only (PLAN.md D155, D156, D171; docs/research/P8_E1.md,
"Post-mortem (measured)", whose definitions were written before this ran). Nothing trains, no model runs, no window is
rebuilt.

    python3 -m tools.p8_e1_postmortem labels                                       M1: label audit of the E1 corpus
    python3 -m export.heavy --name ... -- <frozen CV python> -m tools.p8_e1_postmortem misses --game G [--model F]
                                                                                   M2: r23 against E1, one game
    python3 -m tools.p8_e1_postmortem oracle                                       M3: labels scored as a detector
    python3 -m tools.p8_e1_postmortem report                                       all -> e1/p8_e1_postmortem.json

M1 reads the corpus (tools.p8_e1data.corpus_root(): plan.json, preds/, labels/) and the SportVU moments of the P8
workspace. Every rule is the corpus build's: in-image players by tools.p8_teachers.targets_in_frame, feet by
feet_ft, a find by found_targets (Hungarian, 2.5 ft). M2 runs the pinned scorer's tools.p8_scorecard.score_window
(truth camera) again on the existing r23 and E1 sidecars, read only; a window whose cached references are missing is
refused. Outputs go to the P8 workspace e1/ folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_postmortem"
POOL = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
APART = ("hou_cle_2016",)
SEED = 0
EDGE_PX = 40.0                       # fixed in the definitions before computing
NOMINAL_HEIGHT_FT = 6.5
HEIGHT_EDGES = (50.0, 75.0, 100.0, 150.0)
HEIGHT_EDGES_FINE = (150.0, 200.0, 250.0)     # report only, added after the first run (most players were >= 150 px)
NEAR_FT = 5.0                        # sportvu.bench.MATCH_FT (tests assert they agree)
BOOT_N = 2000
CLASSES = ("r23 only", "teacher only", "both, IoU < 0.5", "both, IoU >= 0.5", "neither")
THIRDS = ("near", "middle", "far")


def check_games(games) -> list:
    games = list(games)
    bad = [g for g in games if g not in POOL + APART]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not a pool game (or hou_cle_2016)")
    refuse_cv_test(games, WHO)
    guard(games, WHO)
    return games


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def yolo_to_boxes(text: str, w: int, h: int) -> list:
    """YOLO lines (class cx cy bw bh, normalised) -> [[x1, y1, x2, y2]] in pixels."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def assignment(feet, targets, gate: float) -> dict:
    """{target index: box index} of the Hungarian match of feet to targets within gate (the pairs
    tools.p8_teachers.found_targets counts)."""
    from scipy.optimize import linear_sum_assignment
    if len(feet) == 0 or len(targets) == 0:
        return {}
    D = np.sqrt(((np.asarray(feet)[:, None, :] - np.asarray(targets)[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(D)
    return {int(j): int(i) for i, j in zip(ri, ci) if D[i, j] <= gate}


def in_image_mask(sv_xy, H_truth, w: int, h: int) -> np.ndarray:
    """The mask of tools.p8_teachers.targets_in_frame (projection through inv(H_truth) inside the image)."""
    import cv2
    P = np.linalg.inv(np.asarray(H_truth, np.float64).reshape(3, 3))
    px = cv2.perspectiveTransform(np.asarray(sv_xy, np.float64).reshape(-1, 1, 2), P).reshape(-1, 2)
    return np.isfinite(px).all(1) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)


def box_class(ours_hit: bool, teacher_hit: bool, pair_iou) -> str:
    if ours_hit and teacher_hit:
        return "both, IoU < 0.5" if pair_iou < 0.5 else "both, IoU >= 0.5"
    if ours_hit:
        return "r23 only"
    if teacher_hit:
        return "teacher only"
    return "neither"


def is_edge(u: float, v: float, w: int, h: int, margin: float = EDGE_PX) -> bool:
    return bool(min(u, w - u, v, h - v) < margin)


def height_bin(hpx, edges=HEIGHT_EDGES) -> str:
    if hpx is None or not np.isfinite(hpx):
        return "no pose"
    for k, e in enumerate(edges):
        if hpx < e:
            return f"<{e:g}" if k == 0 else f"{edges[k - 1]:g}-{e:g}"
    return f">={edges[-1]:g}"


HEIGHT_BINS = tuple(height_bin(v) for v in (0.0, 60.0, 80.0, 120.0, 200.0)) + ("no pose",)
HEIGHT_BINS_FINE = tuple(height_bin(v, HEIGHT_EDGES_FINE) for v in (100.0, 180.0, 220.0, 300.0)) + ("no pose",)


def foot_and_height(sv_xy, H_truth, w: int, h: int) -> tuple:
    """(foot pixels (n, 2) through inv(H_truth), expected box heights (n,) or None when the camera has no pose)."""
    import cv2
    from sportvu.camera import pose_from_h
    from tools.p8_scorecard import project3d
    xy = np.asarray(sv_xy, np.float64).reshape(-1, 2)
    Ht = np.asarray(H_truth, np.float64).reshape(3, 3)
    foot = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
    pose = pose_from_h(Ht, w, h)
    if pose is None:
        return foot, None
    lo, _ = project3d(pose[2], pose[0], pose[1], w, h, np.column_stack([xy, np.zeros(len(xy))]))
    hi, _ = project3d(pose[2], pose[0], pose[1], w, h, np.column_stack([xy, np.full(len(xy), NOMINAL_HEIGHT_FT)]))
    return foot, np.abs(lo[:, 1] - hi[:, 1])


def tally(rows: list, field: str, levels) -> dict:
    out = {str(k): 0 for k in levels}
    for r in rows:
        out[str(r[field])] = out.get(str(r[field]), 0) + 1
    return out


def pair_classes(obs_r23: list, obs_new: list) -> dict:
    """Scorer observations of two builds on the same frames -> {(window, frame, pid): class}."""
    a = {(o["w"], int(o["f"]), int(o["pid"])): bool(o["missed"]) for o in obs_r23}
    b = {(o["w"], int(o["f"]), int(o["pid"])): bool(o["missed"]) for o in obs_new}
    if a.keys() != b.keys():
        raise SystemExit(f"{WHO}: r23 and E1 observe different player-frames ({len(a)} against {len(b)})")
    names = {(False, False): "both found", (False, True): "E1 extra miss", (True, False): "E1 extra find",
             (True, True): "both missed"}
    return {k: names[(a[k], b[k])] for k in a}


def share_diff(groups: dict, n: int = BOOT_N, seed: int = SEED) -> dict:
    """groups: {window: (extra misses, unlabelled extra misses, r23 found, unlabelled r23 found)} -> the pooled
    difference (unlabelled share among extra misses minus among r23-found) and its window bootstrap interval."""
    keys = sorted(groups)
    arr = np.array([groups[k] for k in keys], float).reshape(-1, 4)

    def diff(a):
        if a[:, 0].sum() == 0 or a[:, 2].sum() == 0:
            return None
        return a[:, 1].sum() / a[:, 0].sum() - a[:, 3].sum() / a[:, 2].sum()

    point = diff(arr)
    rng = np.random.default_rng(seed)
    vals = [v for v in (diff(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)) if v is not None]
    iv = [round(float(np.percentile(vals, 2.5)), 5), round(float(np.percentile(vals, 97.5)), 5)] if vals else None
    return {"diff": None if point is None else round(float(point), 5), "window_interval": iv, "windows": len(keys)}


def rate(num: int, den: int):
    return round(num / den, 4) if den else None


# ── M1: label audit (system python) ─────────────────────────────────────────────────────────────────────────────────
def audit_frame(rec: dict, label_text: str, sv_xy, pids, H_truth, T) -> tuple:
    """One corpus frame -> (per in-image player rows, label boxes no player claims)."""
    from tools.p8_scorecard import third_of
    from sportvu.eval import near_side
    w, h = rec["size"]
    d = [p for p in rec[T.OURS] if p[5] == "player"]
    t = [p for p in rec[T.MISS_TEACHER] if p[4] >= T.MISS_TEACHER_CONF]
    mask = in_image_mask(sv_xy, H_truth, w, h)
    tg = T.targets_in_frame(sv_xy, H_truth, w, h)
    if not np.array_equal(np.asarray(sv_xy, np.float64)[mask], tg):
        raise SystemExit(f"{WHO}: in-image mask differs from tools.p8_teachers.targets_in_frame")
    idx = np.where(mask)[0]
    labels = yolo_to_boxes(label_text, w, h)
    fl, fd, ft = T.feet_ft(labels, H_truth), T.feet_ft(d, H_truth), T.feet_ft(t, H_truth)
    lab, ad, at = assignment(fl, tg, T.CONFIRM_FT), assignment(fd, tg, T.CONFIRM_FT), assignment(ft, tg, T.CONFIRM_FT)
    if set(lab) != T.found_targets(fl, tg) or set(ad) != T.found_targets(fd, tg) or set(at) != T.found_targets(ft, tg):
        raise SystemExit(f"{WHO}: assignment differs from tools.p8_teachers.found_targets")
    Ht = np.asarray(H_truth, np.float64).reshape(3, 3)
    nh = near_side(Ht)
    foot, hts = foot_and_height(tg, H_truth, w, h) if len(tg) else (np.zeros((0, 2)), None)
    allfeet = np.vstack([fd.reshape(-1, 2), ft.reshape(-1, 2)])
    s1 = T.agree_boxes(d, t)
    own = own_label_within(fl, tg, lab)
    rows = []
    for j in range(len(tg)):
        oh, th = j in ad, j in at
        pi = T.iou(d[ad[j]], t[at[j]]) if oh and th else None
        near = bool(len(allfeet)) and bool(np.sqrt(((allfeet - tg[j]) ** 2).sum(1)).min() <= NEAR_FT)
        hj = None if hts is None else float(hts[j])
        row = {"pid": int(pids[idx[j]]), "labelled": j in lab,
               "cls": None if j in lab else box_class(oh, th, pi),
               "near_box_5ft": near, "third": third_of(tg[j][1], nh),
               "height": height_bin(hj), "height_fine": height_bin(hj, HEIGHT_EDGES_FINE),
               "edge": is_edge(float(foot[j][0]), float(foot[j][1]), w, h),
               # report only, added after the first run: a label box near the player, its foot off by 2.5 to 5 ft
               "label_foot_within_5ft": j not in lab and bool(len(fl)) and
                                        bool(np.sqrt(((fl.reshape(-1, 2) - tg[j]) ** 2).sum(1)).min() <= NEAR_FT),
               # report only, added after review: a label box no in-image player claims at 2.5 ft, matched one to
               # one to the unlabelled players within 5 ft
               "own_label_within_5ft": j in own}
        if row["cls"] == "teacher only":   # report only: the teacher box lost to a source 1 box it overlaps (merge rule)
            row["teacher_box_under_source1"] = any(T.iou(t[at[j]], a) >= 0.5 for a in s1)
        rows.append(row)
    return rows, len(labels) - len(lab), len(labels)


def own_label_within(feet_labels, targets, lab: dict, gate: float = NEAR_FT) -> set:
    """Unlabelled targets (not in lab) matched one to one, Hungarian, within gate to label boxes that no target
    claims in lab (the unclaimed label boxes): the player's own label box with its foot off, not a neighbour's."""
    from scipy.optimize import linear_sum_assignment
    claimed = set(lab.values())
    free = [i for i in range(len(feet_labels)) if i not in claimed]
    open_t = [j for j in range(len(targets)) if j not in lab]
    if not free or not open_t:
        return set()
    F = np.asarray(feet_labels, np.float64).reshape(-1, 2)[free]
    G = np.asarray(targets, np.float64).reshape(-1, 2)[open_t]
    D = np.sqrt(((F[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(D)
    return {open_t[c] for r, c in zip(ri, ci) if D[r, c] <= gate}


def breakdown(rows: list) -> dict:
    un = [r for r in rows if not r["labelled"]]
    out = {"in_image": len(rows), "labelled": len(rows) - len(un), "unlabelled": len(un),
           "label_recall": rate(len(rows) - len(un), len(rows)),
           "unlabelled_by_class": tally(un, "cls", CLASSES)}
    out["neither_with_a_box_foot_within_5ft"] = sum(1 for r in un if r["cls"] == "neither" and r["near_box_5ft"])
    out["report_only_added_after_first_run"] = {
        "unlabelled_with_a_label_foot_within_5ft": sum(1 for r in un if r["label_foot_within_5ft"]),
        "unlabelled_with_a_label_foot_within_5ft_by_class": tally([r for r in un if r["label_foot_within_5ft"]], "cls", CLASSES),
        "teacher_only_with_teacher_box_under_a_source1_box": sum(1 for r in un if r.get("teacher_box_under_source1")),
        "unlabelled_with_an_unclaimed_label_box_within_5ft_one_to_one": sum(1 for r in un if r.get("own_label_within_5ft")),
        "unlabelled_with_an_unclaimed_label_box_within_5ft_one_to_one_by_class":
            tally([r for r in un if r.get("own_label_within_5ft")], "cls", CLASSES)}
    for field, levels in (("third", THIRDS), ("height", HEIGHT_BINS), ("height_fine", HEIGHT_BINS_FINE),
                          ("edge", (True, False))):
        out[f"by_{field}"] = {"all": tally(rows, field, levels), "unlabelled": tally(un, field, levels),
                              "label_recall": {str(k): rate(sum(1 for r in rows if str(r[field]) == str(k) and r["labelled"]),
                                                            sum(1 for r in rows if str(r[field]) == str(k)))
                                               for k in levels}}
        out[f"class_by_{field}"] = {str(k): tally([r for r in un if str(r[field]) == str(k)], "cls", CLASSES)
                                    for k in levels}
    out["labels_e1b_could_add"] = out["unlabelled_by_class"]["r23 only"] + out["unlabelled_by_class"]["both, IoU < 0.5"]
    return out


def cmd_labels(args) -> dict:
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    from tools.p8_e1data import corpus_root
    check_games(POOL)
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    if sorted(plan["games"]) != sorted(POOL):
        raise SystemExit(f"{WHO}: the corpus plan names {sorted(plan['games'])}, not the pool games")
    guard([v for g in plan["games"].values() for v in (g["sportvu"], g["video_id"])], WHO)
    preds = {}
    for k in range(len(plan["chunks"])):
        preds.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])
    manifest = json.loads((root / "manifest.json").read_text())["frames"]
    players, per_game, orphans = {}, {}, {}
    for tag, g in plan["games"].items():
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        rows, n_lab, n_orph = [], 0, 0
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    text = (root / "labels" / tag / f"{stem}.txt").read_text()
                    d = index.q[fr["q"]]
                    sv = apply_mirror(d["xy"][fr["moment"]], fx, fy)
                    fr_rows, orph, nl = audit_frame(preds[f"{tag}/{stem}"], text, sv, d["pid"][fr["moment"]],
                                                    fr["H_truth"], T)
                    if nl != manifest[f"{tag}/{stem}"]["labels"]:
                        raise SystemExit(f"{WHO}: {stem} holds {nl} label lines, the manifest says "
                                         f"{manifest[f'{tag}/{stem}']['labels']}")
                    n_lab += nl
                    n_orph += orph
                    for r in fr_rows:
                        players[f"{fr['window']}:{int(fr['frame'])}:{r['pid']}"] = r
                    rows += fr_rows
        per_game[tag] = {**breakdown(rows), "frames": sum(1 for k in manifest if k.startswith(tag + "/")),
                         "label_boxes": n_lab, "label_boxes_no_player_within_2_5ft": n_orph}
        orphans[tag] = n_orph
    allrows = list(players.values())
    out = {"what": "M1 label audit of the E1 corpus (docs/research/P8_E1.md, Post-mortem)",
           "rules": {"in_image": "tools.p8_teachers.targets_in_frame", "find": "tools.p8_teachers.found_targets",
                     "confirm_ft": T.CONFIRM_FT, "teacher_conf": T.MISS_TEACHER_CONF, "edge_px": EDGE_PX,
                     "nominal_height_ft": NOMINAL_HEIGHT_FT, "height_edges_px": list(HEIGHT_EDGES),
                     "near_box_ft": NEAR_FT},
           "games": per_game,
           "pooled": {**breakdown(allrows), "frames": sum(v["frames"] for v in per_game.values()),
                      "label_boxes": sum(v["label_boxes"] for v in per_game.values()),
                      "label_boxes_no_player_within_2_5ft": sum(orphans.values())}}
    d = wr / "e1"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "postmortem_m1.json", json.dumps(out, indent=1))
    atomic_write(d / "postmortem_m1_players.json", json.dumps(players))
    print(json.dumps({t: {k: v[k] for k in ("frames", "in_image", "labelled", "label_recall", "unlabelled_by_class")}
                      for t, v in {**per_game, "pooled": out["pooled"]}.items()}, indent=1))
    return out


# ── M3: label oracle (system python) ─────────────────────────────────────────────────────────────────────────────
MISS_FT, GHOST_FT, FRAME_MARGIN_PX = 2.5, 3.0, 10       # sportvu.eval (tests assert they agree)
ORACLE_KEYS = ("in_frame", "missed", "boxes", "ghosts", "duplicates")


def scorer_inside(sv_xy, H_truth, w: int, h: int) -> np.ndarray:
    """sportvu.eval.eval_frame's in-frame rule: the truth projection strictly inside the image with a 10 px margin."""
    import cv2
    sv = np.asarray(sv_xy, np.float64).reshape(-1, 2)
    px = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32),
                                  np.linalg.inv(np.asarray(H_truth, np.float64).reshape(3, 3))).reshape(-1, 2)
    m = FRAME_MARGIN_PX
    return (px[:, 0] > m) & (px[:, 0] < w - m) & (px[:, 1] > m) & (px[:, 1] < h - m)


def oracle_counts(feet_ft, sv_xy, inside) -> dict:
    """sportvu.eval.eval_frame's missed and ghost counts for one frame's box feet (court ft, truth frame), plus boxes
    and duplicates (in-frame players with two or more feet within MISS_FT)."""
    sv = np.asarray(sv_xy, np.float64).reshape(-1, 2)
    ct = np.asarray(feet_ft, np.float64).reshape(-1, 2)
    inside = np.asarray(inside, bool)
    out = {"in_frame": int(inside.sum()), "boxes": int(len(ct)), "missed": int(inside.sum()), "ghosts": 0,
           "duplicates": 0}
    if len(ct):
        D = np.sqrt(((sv[:, None, :] - ct[None, :, :]) ** 2).sum(-1))
        out["missed"] = int((inside & (D.min(1) > MISS_FT)).sum())
        out["ghosts"] = int((D.min(0) > GHOST_FT).sum())
        out["duplicates"] = int((inside & ((D <= MISS_FT).sum(1) >= 2)).sum())
    return out


def oracle_rates(c: dict) -> dict:
    return {"missed_rate": rate(c["missed"], c["in_frame"]), "ghost_rate": rate(c["ghosts"], c["boxes"]),
            "boxes_per_in_frame_player": rate(c["boxes"], c["in_frame"]),
            "duplicate_rate": rate(c["duplicates"], c["in_frame"]), **c}


def oracle_boot(per_window: dict, a: str, b: str, n: int = BOOT_N, seed: int = SEED) -> dict:
    """per_window: {window: {source: counts}} -> b minus a for the missed and ghost rates, window bootstrap."""
    keys = sorted(per_window)
    arr = {s: np.array([[per_window[k][s][f] for f in ORACLE_KEYS] for k in keys], float) for s in (a, b)}
    cols = {f: i for i, f in enumerate(ORACLE_KEYS)}

    def diff(ix):
        out = {}
        for row, num, den in (("missed_rate", "missed", "in_frame"), ("ghost_rate", "ghosts", "boxes")):
            x, y = arr[a][ix], arr[b][ix]
            out[row] = (y[:, cols[num]].sum() / y[:, cols[den]].sum()
                        - x[:, cols[num]].sum() / x[:, cols[den]].sum())
        return out

    point = diff(np.arange(len(keys)))
    rng = np.random.default_rng(seed)
    reps = [diff(rng.integers(0, len(keys), len(keys))) for _ in range(n)]
    return {row: {"diff": round(float(point[row]), 5),
                  "window_interval": [round(float(np.percentile([r[row] for r in reps], 2.5)), 5),
                                      round(float(np.percentile([r[row] for r in reps], 97.5)), 5)]}
            for row in point}


def cmd_oracle(args) -> dict:
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    from tools.p8_e1data import corpus_root
    check_games(POOL)
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    if sorted(plan["games"]) != sorted(POOL):
        raise SystemExit(f"{WHO}: the corpus plan names {sorted(plan['games'])}, not the pool games")
    guard([v for g in plan["games"].values() for v in (g["sportvu"], g["video_id"])], WHO)
    preds = {}
    for k in range(len(plan["chunks"])):
        preds.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])
    sources = ("labels", "ours", "teacher")
    games, allw = {}, {}
    for tag, g in plan["games"].items():
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        per_w = {}
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    rec = preds[f"{tag}/{stem}"]
                    wpx, hpx = rec["size"]
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    inside = scorer_inside(sv, fr["H_truth"], wpx, hpx)
                    boxes = {"labels": yolo_to_boxes((root / "labels" / tag / f"{stem}.txt").read_text(), wpx, hpx),
                             "ours": [p for p in rec[T.OURS] if p[5] == "player"],
                             "teacher": [p for p in rec[T.MISS_TEACHER] if p[4] >= T.MISS_TEACHER_CONF]}
                    acc = per_w.setdefault(fr["window"], {s: {k: 0 for k in ORACLE_KEYS} for s in sources})
                    for s in sources:
                        for k, v in oracle_counts(T.feet_ft(boxes[s], fr["H_truth"]), sv, inside).items():
                            acc[s][k] += v
        tot = {s: {k: sum(v[s][k] for v in per_w.values()) for k in ORACLE_KEYS} for s in sources}
        games[tag] = {"windows": len(per_w), **{s: oracle_rates(tot[s]) for s in sources},
                      "labels_minus_ours": oracle_boot(per_w, "ours", "labels")}
        allw.update({f"{tag}/{k}": v for k, v in per_w.items()})
    tot = {s: {k: sum(v[s][k] for v in allw.values()) for k in ORACLE_KEYS} for s in sources}
    out = {"what": "M3 label oracle on the E1 corpus frames (docs/research/P8_E1.md, Post-mortem); sportvu.eval rules "
                   "on raw boxes (bottom-centre foot), not the registered build rows",
           "rules": {"miss_ft": MISS_FT, "ghost_ft": GHOST_FT, "frame_margin_px": FRAME_MARGIN_PX},
           "games": games,
           "pooled": {"windows": len(allw), **{s: oracle_rates(tot[s]) for s in sources},
                      "labels_minus_ours": oracle_boot(allw, "ours", "labels"),
                      "teacher_minus_ours": oracle_boot(allw, "ours", "teacher")}}
    d = wr / "e1"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "postmortem_m3.json", json.dumps(out, indent=1))
    show = {}
    for t, v in {**games, "pooled": out["pooled"]}.items():
        show[t] = {s: {k: v[s][k] for k in ("missed_rate", "ghost_rate", "boxes_per_in_frame_player", "duplicate_rate")}
                   for s in sources}
        show[t]["labels_minus_ours"] = v["labels_minus_ours"]
    print(json.dumps(show, indent=1))
    return out


# ── M2: where E1's extra misses are (frozen CV python, inside the lock) ─────────────────────────────────────────────
def cmd_misses(args) -> dict:
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    from tools.p8_e1_score import build_name, score_name
    game = check_games([args.game])[0]
    model = args.model or game
    if model not in POOL or (game in POOL and model != game):
        raise SystemExit(f"{WHO}: a pool game is scored by the fold that held it out (D156); {model} on {game} is not")
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    g = F.game_of(game)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, game)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    sf = json.loads((p8_root() / "e1" / score_name(model, 0, game)).read_text())
    if sf.get("frame_sets_differ"):
        raise SystemExit(f"{WHO}: frame sets differ in {score_name(model, 0, game)}; this tool scores whole windows")
    dirs = {"r23": P["builds"] / "r23", "e1": P["builds"] / build_name(model, 0)}
    wins = {w["window"]: w for w in F.label_windows(P, game)}
    rows, check = [], {"r23": {"in_frame": 0, "missed": 0}, "e1": {"in_frame": 0, "missed": 0}}
    for s in sf["scored_windows"]:
        if not (P["cache"] / "scorecard" / f"{s}.json").exists():
            raise SystemExit(f"{WHO}: {s} has no cached scorecard references; refusing to compute them")
        w = wins[s]
        obs = {}
        for name, bd in dirs.items():
            rec, ob = S.score_window(P, game, w, bd, "camera", mom, mirror, refs, events)[:2]
            if "skipped" in rec:
                raise SystemExit(f"{WHO}: {s} skipped for {name}: {rec['skipped']}")
            want = sf["windows"][s]["r23" if name == "r23" else "new"]
            if rec["missed"] != want["missed"] or rec["in_frame"] != want["in_frame"]:
                raise SystemExit(f"{WHO}: {s} {name} scores {rec['missed']}/{rec['in_frame']}, the score file "
                                 f"{want['missed']}/{want['in_frame']}")
            check[name]["in_frame"] += rec["in_frame"]
            check[name]["missed"] += rec["missed"]
            obs[name] = ob
        cls = pair_classes(obs["r23"], obs["e1"])
        truth, _ = S.truth_rows(P, game, w, "camera", mom.index, mirror)
        H_of = {int(r["frame"]): r["H_truth"] for r in truth}
        wd, hd = _size(P, s)
        by_f = {}
        for o in obs["r23"]:
            by_f.setdefault(int(o["f"]), []).append(o)
        for f, os_ in by_f.items():
            xy = np.array([[o["x"], o["y"]] for o in os_])
            foot, hts = foot_and_height(xy, H_of[f], wd, hd)
            for k, o in enumerate(os_):
                rows.append({"w": s, "f": f, "pid": int(o["pid"]), "cls": cls[(s, f, int(o["pid"]))],
                             "third": o["third"], "height": height_bin(None if hts is None else float(hts[k])),
                             "height_fine": height_bin(None if hts is None else float(hts[k]), HEIGHT_EDGES_FINE),
                             "edge": is_edge(float(foot[k][0]), float(foot[k][1]), wd, hd)})
    out = {"model": model, "seed": 0, "game": game, "truth": "camera", "windows": len(sf["scored_windows"]),
           "totals": check, "rows": rows}
    atomic_write(p8_root() / "e1" / f"postmortem_m2__{model}__{game}.json", json.dumps(out))
    print(json.dumps({"game": game, "model": model, "totals": check, "player_frames": len(rows),
                      "classes": tally(rows, "cls", ("both found", "E1 extra miss", "E1 extra find", "both missed"))},
                     indent=1))
    return out


def _size(P, stem: str) -> tuple:
    import cv2
    cap = cv2.VideoCapture(str(P["label_build"] / f"{stem}.mp4"))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    return w, h


# ── report (system python) ──────────────────────────────────────────────────────────────────────────────────────────
M2_CLASSES = ("both found", "E1 extra miss", "E1 extra find", "both missed")
STATUS = ("labelled",) + CLASSES + ("not a corpus frame",)


STATUS_NEAR = ("labelled", "unlabelled, a label foot within 5 ft", "unlabelled, no label foot within 5 ft",
               "not a corpus frame")
STATUS_OWN = ("labelled", "unlabelled, an unclaimed label box within 5 ft", "unlabelled, none", "not a corpus frame")
REPORT_ONLY = {
    "m1": ["report_only_added_after_first_run", "by_height_fine", "class_by_height_fine",
           "neither_with_a_box_foot_within_5ft (report only by the definitions; nearest box, not one to one)"],
    "m2": ["net_extra_misses_by_label", "e1_recovers_r23_misses", "e1_loses_r23_finds",
           "extra_miss_rate_given_status", "by_status_near", "by_status_own", "by_height_fine"],
    "note": "fields added after the definitions; none replaces a registered number. The 5 ft counts of "
            "by_status_near and label_foot_within_5ft use the nearest label box of any player (a neighbour's box "
            "counts); the own-box counts match unclaimed label boxes one to one."}


def m2_status(rows: list, players: dict) -> list:
    out = []
    for r in rows:
        p = players.get(f"{r['w']}:{r['f']}:{r['pid']}")
        st = "not a corpus frame" if p is None else ("labelled" if p["labelled"] else p["cls"])
        near = st if p is None or p["labelled"] else STATUS_NEAR[1 if p.get("label_foot_within_5ft") else 2]
        own = st if p is None or p["labelled"] else STATUS_OWN[1 if p.get("own_label_within_5ft") else 2]
        out.append({**r, "status": st, "status_near": near, "status_own": own})
    return out


def m2_breakdown(rows: list) -> dict:
    out = {"player_frames": len(rows), "classes": tally(rows, "cls", M2_CLASSES)}
    out["net_extra_misses"] = out["classes"]["E1 extra miss"] - out["classes"]["E1 extra find"]
    groups = {"all player-frames": rows, "r23 found": [r for r in rows if r["cls"] in ("both found", "E1 extra miss")],
              "E1 extra miss": [r for r in rows if r["cls"] == "E1 extra miss"],
              "E1 extra find": [r for r in rows if r["cls"] == "E1 extra find"]}
    for field, levels in (("status", STATUS), ("status_near", STATUS_NEAR), ("status_own", STATUS_OWN),
                          ("third", THIRDS), ("height", HEIGHT_BINS),
                          ("height_fine", HEIGHT_BINS_FINE), ("edge", (True, False))):
        out[f"by_{field}"] = {k: tally(v, field, levels) for k, v in groups.items()}
    un = lambda rs: sum(1 for r in rs if r["status"] not in ("labelled", "not a corpus frame"))
    out["unlabelled_share"] = {k: rate(un(v), len(v)) for k, v in groups.items()}
    # report only, added after the first report: where the net change sits, and E1's recovery of r23's misses
    lab = lambda rs: sum(1 for r in rs if r["status"] == "labelled")
    r23_missed = [r for r in rows if r["cls"] in ("E1 extra find", "both missed")]
    out["net_extra_misses_by_label"] = {
        "labelled": lab(groups["E1 extra miss"]) - lab(groups["E1 extra find"]),
        "unlabelled": un(groups["E1 extra miss"]) - un(groups["E1 extra find"])}
    out["e1_recovers_r23_misses"] = {
        "labelled": rate(lab(groups["E1 extra find"]), lab(r23_missed)),
        "unlabelled": rate(un(groups["E1 extra find"]), un(r23_missed))}
    out["e1_loses_r23_finds"] = {
        "labelled": rate(lab(groups["E1 extra miss"]), lab(groups["r23 found"])),
        "unlabelled": rate(un(groups["E1 extra miss"]), un(groups["r23 found"]))}
    out["extra_miss_rate_given_status"] = {
        st: rate(sum(1 for r in groups["E1 extra miss"] if r["status"] == st),
                 sum(1 for r in groups["r23 found"] if r["status"] == st)) for st in STATUS}
    return out


def window_groups(rows: list) -> dict:
    g = {}
    for r in rows:
        a = g.setdefault(r["w"], [0, 0, 0, 0])
        unl = r["status"] not in ("labelled", "not a corpus frame")
        if r["cls"] == "E1 extra miss":
            a[0] += 1
            a[1] += unl
        if r["cls"] in ("both found", "E1 extra miss"):
            a[2] += 1
            a[3] += unl
    return g


def cmd_report(args) -> dict:
    check_games(POOL)
    d = p8_root() / "e1"
    m1 = json.loads((d / "postmortem_m1.json").read_text())
    m3 = json.loads((d / "postmortem_m3.json").read_text())
    players = json.loads((d / "postmortem_m1_players.json").read_text())
    m2, allrows, apart = {}, [], {}
    for g in POOL:
        p = d / f"postmortem_m2__{g}__{g}.json"
        if not p.exists():
            raise SystemExit(f"{WHO}: M2 has not run on {g}")
        f = json.loads(p.read_text())
        rows = m2_status(f["rows"], players)
        m2[g] = {**m2_breakdown(rows), "totals": f["totals"], "windows": f["windows"],
                 "share_diff": share_diff(window_groups(rows))}
        allrows += rows
    for p in sorted(d.glob("postmortem_m2__*__hou_cle_2016.json")):
        f = json.loads(p.read_text())
        rows = m2_status(f["rows"], players)
        apart[f"{f['model']}__hou_cle_2016"] = {**m2_breakdown(rows), "totals": f["totals"], "windows": f["windows"]}
    pooled = {**m2_breakdown(allrows), "share_diff": share_diff(window_groups(allrows))}
    same_dir = all(m2[g]["share_diff"]["diff"] is not None and m2[g]["share_diff"]["diff"] > 0 for g in POOL)
    iv = pooled["share_diff"]["window_interval"]
    out = {"what": "E1 post-mortem (docs/research/P8_E1.md, Post-mortem (measured)); a measurement, nothing trained",
           "report_only_fields": REPORT_ONLY,
           "m1": m1, "m2": {"games": m2, "pooled": pooled, "apart": apart}, "m3": m3,
           "reading_test_b": {"pooled_diff": pooled["share_diff"]["diff"], "window_interval": iv,
                              "same_direction_all_games": same_dir,
                              "holds": bool(iv and iv[0] > 0 and same_dir)}}
    atomic_write(d / "p8_e1_postmortem.json", json.dumps(out, indent=1))
    print(json.dumps({"m1_pooled": {k: m1["pooled"][k] for k in ("label_recall", "unlabelled_by_class")},
                      "m2_pooled": {k: pooled[k] for k in ("classes", "unlabelled_share", "share_diff")},
                      "reading_test_b": out["reading_test_b"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_e1_postmortem")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    m = sub.add_parser("misses")
    m.add_argument("--game", required=True, choices=POOL + APART)
    m.add_argument("--model", default=None, choices=POOL)
    sub.add_parser("oracle")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"labels": cmd_labels, "misses": cmd_misses, "oracle": cmd_oracle, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

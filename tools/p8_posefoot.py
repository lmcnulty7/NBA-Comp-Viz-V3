"""tools/p8_posefoot.py: a per-box foot from pose keypoints (docs/research/P8_posefoot.md, version 1.1). Report only.

Stages 0a and A in this version (no download):

    <frozen CV python> -m tools.p8_posefoot stage0a GAME   inside the lock ("p8 posefoot 0a GAME"), V3_P8_ROOT = work_ref
    python3 -m tools.p8_posefoot report0a                   -> reports/research/p8_posefoot_0a.json
    <frozen CV python> -m tools.p8_posefoot align GAME      inside the lock ("p8 posefoot align GAME")
    python3 -m tools.p8_posefoot report_align               -> reports/research/p8_posefoot_align.json

align (section 14 with 14.2, v1.3): per window, 15 sidecar frames ranked by the sidecar's own box motion; the clip read
by the build's own read path (CAP_FFMPEG first, the pre-gate pass with the build's gate v2 on the build's gate device,
its intervals, the tracking loop's jumps through one SeqReader); r23's detector at f only (CPU, player class, confidence 0.40, NMS 0.45), matched one to one
(Hungarian on IoU) to the sidecar at f - stride, f and f + stride; M_k the mean IoU over sidecar boxes (unmatched and
pairs below 0.1 count 0), L_0 the median matched IoU at f of pairs at 0.3 or more; misaligned if M_0 < 0.82, L_0 < 0.85
or M_+-1 >= M_0 + 0.05, ties aligned; undetermined with fewer than 10 eligible frames or a read path not reproduced.

stage0a: D207's boxes recomputed with tools.p8_piibox's functions (every P-ii box and every classified hit box), gated
per window on D207's ordered (kind, class, height) rows and on D202's counts; on each box, the distance to its player
under T (the scorer's mapping) and under B (the sidecar's accepted H) of foot_stab, foot_raw (the per-frame box bottom)
and the same-run null (another box's foot_raw - foot_stab offset on the same frame, scaled by box height, added to this
box's foot_stab), with scorer-ghost flags and the player's SportVU speed. hou_cle_2016 is refused before any file is
read. Writes only work_ref/posefoot/0a/<game>.json and the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_posefoot"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
POOL = (CHA, BKN)
MISS_FT, GHOST_FT = 2.5, 3.0
PARTNER_IOU = 0.5                         # v1.2: a null partner overlapping the box at IoU 0.5 or more is excluded
OWN = ("own", "own, with crossings", "own, misplaced, with crossings", "own, misplaced")
SPEED_BINS = (5.0, 10.0)                  # ft/s: under 5, 5 to 10, 10 and over
BOOT_N, SEED = 2000, 0
# the registered rules (section 11.3, 11.4, 11.5)
NULL_MARGIN = 0.10                        # rule 1: R - R_null at least 0.10
MED_BAR_FT = -0.5                         # rule 2
HIT_MED_FT, HIT_LOST = 0.10, 0.05         # rule 4
EST_BAR = -0.010                          # rule 5
RECORD_0A = REPO / "reports" / "research" / "p8_posefoot_0a.json"
RECORD_ALIGN = REPO / "reports" / "research" / "p8_posefoot_align.json"
RECORD_MODEL = REPO / "reports" / "research" / "p8_posefoot_model.json"
# stage 0 (section 11.8): the one download, at a pinned revision
MODEL_REPO = "usyd-community/vitpose-base-simple"
MODEL_REV = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors", "README.md")
MODEL_LFS_SHA = {"model.safetensors": "85375373893ddd3641f3912821073e53f5435f9e966e1dca59d004454bfe4fdf"}
# stage A (section 14)
ALIGN_SEGMENTS, ALIGN_MIN_ELIGIBLE, ALIGN_MIN_BOXES = 15, 10, 4
ALIGN_MARGIN, ALIGN_LEVEL = 0.05, 0.85     # v1.3: L_0 under 0.85 reads misaligned
ALIGN_M0 = 0.82                            # v1.3: M_0 under 0.82 reads misaligned
PAIR_FLOOR, LEVEL_PAIR = 0.1, 0.3
STOP_SHARE = 0.25
ALIGN_CAP_S = 60 * 60
D207_RUN_FILES = {CHA: "gsw_cha_2016.json", BKN: "gsw_bkn_2015.json"}


def refuse_game(game: str) -> None:
    """Every stage refuses hou_cle_2016 (and any non-pool game) before any file is read."""
    if game not in POOL:
        raise SystemExit(f"{WHO}: {game} is refused; this item reads gsw_cha_2016 and gsw_bkn_2015 only")


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def map_px(H, px) -> np.ndarray:
    """Pixels to the court as the scorer maps foot_stab: float32 pixels through a float64 H."""
    import cv2
    p = np.asarray(px, np.float32).reshape(-1, 1, 2)
    if not len(p):
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2).astype(np.float64)


def null_partner(heights: list, b: int, ok: list) -> int | None:
    """The box on the frame, other than b, with ok[k] true, whose height is nearest b's (ties: lower index)."""
    best, bd = None, None
    for k, h in enumerate(heights):
        if k == b or not ok[k]:
            continue
        d = abs(h - heights[b])
        if bd is None or d < bd:
            best, bd = k, d
    return best


def null_foot(stab_b, off_k, h_b: float, h_k: float) -> list:
    """b's foot_stab plus k's offset scaled by h_b / h_k."""
    s = h_b / h_k if h_k > 0 else 1.0
    return [float(stab_b[0] + off_k[0] * s), float(stab_b[1] + off_k[1] * s)]


def cross_frame(frames_sorted: list, f: int, fps: float) -> int | None:
    """v1.2's cross-frame null: the first scored frame of the window at least 1 s after f, else the last at least 1 s
    before f, else None."""
    after = [g for g in frames_sorted if g - f >= fps]
    if after:
        return after[0]
    before = [g for g in frames_sorted if f - g >= fps]
    return before[-1] if before else None


def speed_bin(v) -> str | None:
    if v is None:
        return None
    return "under 5" if v < SPEED_BINS[0] else ("5 to 10" if v < SPEED_BINS[1] else "10 and over")


def rule_set(rows_pii_own: dict, rows_hit: dict, est: dict, cand: str) -> dict:
    """The five rules on one game. rows_*: {"R": .., "R_null": .., "med_T": {value, interval}, "med_B": {value}, ...}."""
    r1 = rows_pii_own["R"] is not None and rows_pii_own["R_null"] is not None and \
        rows_pii_own["R"] - rows_pii_own["R_null"] >= NULL_MARGIN
    mt = rows_pii_own["med_T"]
    r2 = mt["value"] is not None and mt["interval"] is not None and mt["value"] <= MED_BAR_FT and mt["interval"][1] < 0
    r3 = rows_pii_own["med_B"]["value"] is not None and rows_pii_own["med_B"]["value"] <= 0
    r4 = rows_hit["med_T"]["value"] is not None and rows_hit["med_T"]["value"] <= HIT_MED_FT and \
        rows_hit["lost"] is not None and rows_hit["lost"] <= HIT_LOST
    r5 = est["value"] is not None and est["value"] <= EST_BAR
    return {"candidate": cand, "rule1_null_margin": r1, "rule2_median_T": r2, "rule3_median_B": r3,
            "rule4_hits": r4, "rule5_est": r5, "all": bool(r1 and r2 and r3 and r4 and r5)}


def readings_0a(rules: dict) -> dict:
    """Section 11.3's readings (c) and (d) from the per-game rule sets of foot_raw."""
    stab = all(v["rule1_null_margin"] and v["rule2_median_T"] and v["rule3_median_B"] and v["rule5_est"]
               for v in rules.values())
    hybrid_only = any(not v["rule4_hits"] for v in rules.values())
    return {"c_stabiliser_finding": stab, "d_pose_reads_hybrid_only": hybrid_only}


# ── pure: stage A ──────────────────────────────────────────────────────────────────────────────────────────────────
def box_iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def motion(row_prev: dict, row_next: dict) -> float | None:
    """Median, over tids present in both rows, of the box centre's displacement in px."""
    cen = lambda b: ((b["bbox"][0] + b["bbox"][2]) / 2.0, (b["bbox"][1] + b["bbox"][3]) / 2.0)
    a = {b["tid"]: cen(b) for b in row_prev["boxes"]}
    d = [float(np.hypot(c[0] - a[t][0], c[1] - a[t][1])) for t, c in ((b["tid"], cen(b)) for b in row_next["boxes"])
         if t in a]
    return float(np.median(d)) if d else None


def pick_frames(side: dict) -> tuple:
    """Section 14's frames: eligible frames (both neighbours present, at least 4 boxes at f), cut into 15 equal time
    segments, the frame of largest motion in each (ties: earlier). -> (picked frames, eligible count)."""
    stride = int(side.get("stride") or 1)
    srow = {int(r["frame"]): r for r in side["frames"]}
    elig = []
    for f in sorted(srow):
        if f - stride in srow and f + stride in srow and len(srow[f]["boxes"]) >= ALIGN_MIN_BOXES and \
                srow[f - stride]["boxes"] and srow[f + stride]["boxes"]:             # v1.3: boxes at both neighbours
            m = motion(srow[f - stride], srow[f + stride])
            elig.append((f, -1.0 if m is None else m))
    if len(elig) < ALIGN_MIN_ELIGIBLE:
        return [], len(elig)
    picks = []
    for seg in np.array_split(np.arange(len(elig)), ALIGN_SEGMENTS):
        if len(seg):
            best = max(seg, key=lambda i: (elig[i][1], -elig[i][0]))
            picks.append(elig[int(best)][0])
    return picks, len(elig)


def frame_ious(side_boxes: list, pred: list) -> list:
    """Per sidecar box, the IoU of its Hungarian pair with a prediction (0 if unmatched or below PAIR_FLOOR)."""
    if not side_boxes:
        return []
    if not pred:
        return [0.0] * len(side_boxes)
    from scipy.optimize import linear_sum_assignment
    M = np.array([[box_iou(a, b) for b in pred] for a in side_boxes])
    ri, ci = linear_sum_assignment(-M)
    out = [0.0] * len(side_boxes)
    for i, j in zip(ri, ci):
        if M[i, j] >= PAIR_FLOOR:
            out[int(i)] = float(M[i, j])
    return out


def align_stats(per_frame: list) -> dict:
    """per_frame: [{"-1": ious, "0": ious, "+1": ious}] -> M_k pooled over frames and L_0."""
    M = {}
    for k in ("-1", "0", "+1"):
        v = [x for fr in per_frame for x in fr[k]]
        M[k] = float(np.mean(v)) if v else None
    lv = [x for fr in per_frame for x in fr["0"] if x >= LEVEL_PAIR]
    return {"M": M, "L0": float(np.median(lv)) if lv else None}


def align_verdict(st: dict) -> str:
    M, L0 = st["M"], st["L0"]
    if M["0"] is None or L0 is None:
        return "undetermined"
    for k in ("-1", "+1"):
        if M[k] is not None and M[k] >= M["0"] + ALIGN_MARGIN:
            return "misaligned"
    if L0 < ALIGN_LEVEL or M["0"] < ALIGN_M0:
        return "misaligned"
    return "aligned"


def game_alignment(verdicts: dict) -> dict:
    n = len(verdicts)
    c = {v: sum(x == v for x in verdicts.values()) for v in ("aligned", "misaligned", "undetermined")}
    bad = (c["misaligned"] + c["undetermined"]) / n if n else None
    return {"windows": n, **c, "share_not_aligned": bad, "stop": bool(bad is not None and bad > STOP_SHARE)}


def cross_table(verdicts: dict, a3: dict) -> dict:
    """Against a3align's 'misaligned' on the windows both cover (report only)."""
    out = {}
    for w in sorted(set(verdicts) & set(a3)):
        key = f"{verdicts[w]}|a3align_{'misaligned' if a3[w] else 'aligned'}"
        out[key] = out.get(key, 0) + 1
    return out


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def collect0a(game: str) -> dict:
    refuse_game(game)
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import MIRRORS, apply_mirror
    from tools import p8_frozen as F
    from tools import p8_one2one as O
    from tools import p8_piibox as PB
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_e1_score import window_stats
    from tools.p8_placecause import cache_gaps
    from tools.p8_resetfix import one_to_one_missed, unmatched_1to1
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    head = PB.git_head()
    if not head["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    pinned, now = RB.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if any(now[k] != pinned.get(k) for k in now):
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    score = RB.read_json(r["ref"] / "rebaseline" / f"score__{game}__current.json")
    d207 = RB.read_json(r["ref"] / "piibox" / D207_RUN_FILES[game])
    root = RB.truths(r)[game]["current"]
    watched = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    mirror = F.mirror_of(P, game)
    fx, fy = MIRRORS[mirror]
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    comp = P["builds"] / "r23"
    P_t = {**P, "labels": root / "labels"}
    wins = sorted((w for w in F.label_windows(P, game) if (comp / f"{w['window']}_frames.json").exists()),
                  key=lambda w: w["window"])
    stems = [w["window"] for w in wins]
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if sorted(score["windows"]) != stems or d207.get("windows") != len(stems):
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's or D207's records")
    want = {}
    for x in d207["rows"]:
        want.setdefault(x["wi"], []).append((x["kind"], x["class"], x["h"]))
    rows, counters = [], {"no_null_partner": 0, "no_cross_partner": 0, "no_B": 0, "no_speed": 0,
                          "partners_excluded_iou_0.5": 0}
    for wi, w in enumerate(wins):
        s = w["window"]
        for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                       ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
            wh = (score.get("hashes") or {}).get(s, {}).get(key)
            if wh is None and key in ("sidecar", "timemap", "labels"):
                raise SystemExit(f"{WHO}: {s} has no {key} hash in D202's record")
            if wh is not None and RB.sha256(p) != wh:
                raise SystemExit(f"{WHO}: {s} {key} differs from D202's record")
        why = cache_gaps(P_t, game, w, comp, mom, mirror)
        if why:
            raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {why}")
        full = S.score_window(P_t, game, w, comp, "camera", mom, mirror, refs, events)
        if "skipped" in full[0]:
            raise SystemExit(f"{WHO}: {s} skipped: {full[0]['skipped']}")
        st = window_stats(full[0])
        scored = {int(o["f"]) for o in full[1]}
        side = json.loads((comp / f"{s}_frames.json").read_text())
        srow = {int(x["frame"]): x for x in side["frames"]}
        sidecar_cls = any("cls" in b for x in srow.values() for b in x["boxes"])
        ident = json.loads((comp / f"{s}_identity.json").read_text()) if (comp / f"{s}_identity.json").exists() else {}
        idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
        cache = json.loads((P_t["cache"] / "scorecard" / f"{s}.json").read_text())
        refd = {int(k): v for k, v in cache.get("referees", {}).items()}
        tm = json.loads((P_t["label_build"] / f"{s}_timemap.json").read_text())
        fps = float(tm["fps"])
        cap = cv2.VideoCapture(str(P_t["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        truth, _ = S.truth_rows(P_t, game, w, "camera", mom.index, mirror)
        c = {k: 0 for k in ("in_frame", "missed", "far", "missed_1to1", "unmatched_1to1")}
        frames, sv_of, items, fboxes = [], {}, [], {}
        for row in truth:
            f = int(row["frame"])
            x = srow.get(f)
            if x is None or f not in scored:
                continue
            d = mom.index.q[row["q"]]
            sv = apply_mirror(d["xy"][row["moment"]], fx, fy)
            pids = [int(v) for v in d["pid"][row["moment"]]]
            Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
            px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
            m = FRAME_MARGIN_PX
            inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
            stab = [b["foot_stab"] for b in x["boxes"]]
            ft = map_px(Ht, stab) if stab else np.zeros((0, 2))
            is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                               any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
                               for b in x["boxes"]], bool)
            svf = np.asarray(sv, np.float64)
            c["unmatched_1to1"] += unmatched_1to1(ft, svf, inside, is_ref, 2.5)
            c["missed_1to1"] += one_to_one_missed(ft, svf, inside, 2.5)
            Dm = np.sqrt(((svf[:, None, :] - ft[None, :, :]) ** 2).sum(-1)).min(1) if len(ft) else np.full(len(svf), np.inf)
            c["in_frame"] += int(inside.sum())
            c["missed"] += int((inside & (Dm > MISS_FT)).sum())
            c["far"] += int((inside & (Dm > 5.0)).sum())
            cls = O.frame_classes(ft, svf, inside, is_ref)
            nonref = [int(b) for b in np.where(~is_ref)[0]]
            main, _ = O.two_stage(ft, svf, range(len(svf)), nonref, O._assign_all)
            cids = []
            for b in x["boxes"]:
                if b.get("tid") is None:
                    raise SystemExit(f"{WHO}: {s} frame {f}: a box has no track id")
                cids.append(idmap.get(int(b["tid"]), int(b["tid"])))
            frames.append({"f": f, "cid": cids, "A": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "A"},
                           "B": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "B"},
                           "ref": list(is_ref), "ft": ft})
            sv_of[f] = {pids[j]: svf[j] for j in range(len(svf))}
            picked, _why = PB.select_items(cls, main)
            Hb = np.asarray(x["H"], np.float64).reshape(3, 3) if B.accepted(x) and x.get("H") is not None else None
            hts = [float(b["bbox"][3] - b["bbox"][1]) for b in x["boxes"]]
            offs = [[b["foot_raw"][0] - b["foot_stab"][0], b["foot_raw"][1] - b["foot_stab"][1]] for b in x["boxes"]]
            fboxes[f] = [(hts[k], offs[k], not bool(is_ref[k])) for k in range(len(x["boxes"]))]
            for j, kind, bk in picked:
                bx = x["boxes"][bk]
                okn = []
                for kk in range(len(x["boxes"])):          # v1.2: partners overlapping b at IoU 0.5 or more excluded
                    near = kk != bk and B._iou(x["boxes"][kk]["bbox"], bx["bbox"]) >= PARTNER_IOU
                    counters["partners_excluded_iou_0.5"] += bool(near and not is_ref[kk])
                    okn.append(not bool(is_ref[kk]) and not near)
                k = null_partner(hts, bk, okn)
                nf = None
                if k is not None:
                    ok_ = x["boxes"][k]
                    off = [ok_["foot_raw"][0] - ok_["foot_stab"][0], ok_["foot_raw"][1] - ok_["foot_stab"][1]]
                    nf = null_foot(bx["foot_stab"], off, hts[bk], hts[k])
                feet = {"stab": bx["foot_stab"], "raw": bx["foot_raw"], "null": nf}
                dT, gT, dB = {}, {}, {}
                for name, p in feet.items():
                    if p is None:
                        dT[name] = gT[name] = dB[name] = None
                        continue
                    q = map_px(Ht, [p])[0]
                    dT[name] = float(np.hypot(*(q - svf[j])))
                    gT[name] = bool(np.sqrt(((svf - q) ** 2).sum(1)).min() > GHOST_FT)
                    dB[name] = float(np.hypot(*(map_px(Hb, [p])[0] - svf[j]))) if Hb is not None else None
                spd = mom.speed(int(row["q"]), int(row["moment"]), pids[j])
                counters["no_speed"] += spd is None
                items.append({"wi": wi, "kind": kind, "f": f, "pid": pids[j], "cid": cids[bk], "h": hts[bk],
                              "dT": dT, "gT": gT, "dB": dB, "speed": spd,
                              "_Ht": Ht, "_Hb": Hb, "_sv": svf, "_j": j, "_stab": bx["foot_stab"]})
                counters["no_null_partner"] += k is None
                counters["no_B"] += Hb is None
        wd = score["windows"][s]
        mine = {**c, "nr_boxes": st["nr_boxes"]}
        if any(mine[k] != wd.get(k) for k in mine) or st["in_frame"] != c["in_frame"] or st["missed"] != c["missed"]:
            raise SystemExit(f"{WHO}: D202 count gate failed on {s}")
        tracks = PB.track_table(frames)
        got = []
        for it in items:
            ev = PB.evidence(tracks[it["cid"]], it["pid"], it["f"], sv_of, fps)
            it["class"] = PB.classify(ev["n_own"], ev["n_other"], ev["n_ownB"], ev["referee_flagged"], ev["present"])
            got.append((it["kind"], it["class"], it["h"]))
        if got != want.get(wi, []):
            raise SystemExit(f"{WHO}: D207 gate failed on {s}: the recomputed (kind, class, height) rows differ")
        fr_sorted = sorted(fboxes)
        for it in items:                                   # v1.2: the cross-frame null, report only
            g_ = cross_frame(fr_sorted, it["f"], fps)
            xk = null_partner([b[0] for b in fboxes[g_]] + [it["h"]], len(fboxes[g_]),
                              [b[2] for b in fboxes[g_]] + [False]) if g_ is not None else None
            if xk is None:
                counters["no_cross_partner"] += 1
                for dct in (it["dT"], it["gT"], it["dB"]):
                    dct["xnull"] = None
                continue
            h_k, off_k, _ = fboxes[g_][xk]
            q = map_px(it["_Ht"], [null_foot(it["_stab"], off_k, it["h"], h_k)])[0]
            sv = it["_sv"]
            it["dT"]["xnull"] = float(np.hypot(*(q - sv[it["_j"]])))
            it["gT"]["xnull"] = bool(np.sqrt(((sv - q) ** 2).sum(1)).min() > GHOST_FT)
            it["dB"]["xnull"] = float(np.hypot(*(map_px(it["_Hb"], [null_foot(it["_stab"], off_k, it["h"], h_k)])[0]
                                                - sv[it["_j"]]))) if it["_Hb"] is not None else None
        for it in items:
            rows.append({k: it[k] for k in ("wi", "kind", "class", "h", "dT", "gT", "dB", "speed")})
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_posefoot.md version 1.1, stage 0a", "git": head,
           "windows": len(stems), "in_frame_by_window": [score["windows"][s]["in_frame"] for s in stems],
           "nr_boxes_by_window": [score["windows"][s]["nr_boxes"] for s in stems],
           "gate": {"d202": "six fields per window", "d207": "ordered (kind, class, height) rows per window",
                    "passes": True}, "counters": counters, "manifest_unchanged": after == watched,
           "seconds": round(time.time() - t0, 1), "rows": rows}
    dd = r["ref"] / "posefoot" / "0a"
    dd.mkdir(parents=True, exist_ok=True)
    atomic_write(dd / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "windows", "gate", "counters", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


class SeekCount:
    """videoseq.SeqReader's seek rule, counted: a read seeks when the index is behind the decoder or more than
    SEEK_MIN_GAP ahead (videoseq.py)."""
    def __init__(self, reader):
        from videoseq import SEEK_MIN_GAP
        self.reader, self.gap, self.seeks = reader, SEEK_MIN_GAP, []

    def read(self, idx: int):
        if idx < self.reader.next or idx - self.reader.next > self.gap:
            self.seeks.append(int(idx))
        return self.reader.read(idx)


def pregate_intervals(reader, total: int, pg_stride: int, pg_pad: int, is_live, start: int = 0) -> list:
    """build_trajectories.py's pre-gate pass and interval merge, copied (lines 180-197): every pg_stride frame from
    start through reader.read, a live hit when is_live(frame), each hit padded by pg_pad and merged."""
    hits, i = [], start
    while i < total:
        ret, frame = reader.read(i)
        if not ret:
            break
        if is_live(frame):
            hits.append(i)
        i += pg_stride
    intervals = []
    for h in hits:
        a, b = h - pg_pad, h + pg_pad
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = b
        else:
            intervals.append([a, b])
    return intervals


def main_reads(reader, total: int, intervals: list, stride: int, accepted: set, max_frames: int, need: set, predict,
               start: int = 0) -> tuple:
    """build_trajectories.py's tracking loop read path, copied (lines 210-224 and the idx steps): jump to each
    interval's start, read with the same reader, step by stride; a frame counts towards max_frames when the build
    accepted it (it has a sidecar row), which stands in for the gate's per-frame decision. predict(img) on frames in
    need. -> ({frame: prediction}, [frames read])."""
    idx, n, seg_i, out, reads = start, 0, 0, {}, []
    while idx < total and n < max_frames:
        if intervals is not None:
            while seg_i < len(intervals) and idx > intervals[seg_i][1]:
                seg_i += 1
            if seg_i >= len(intervals):
                break
            if idx < intervals[seg_i][0]:
                idx = intervals[seg_i][0]
        ret, frame = reader.read(idx)
        if not ret:
            break
        reads.append(int(idx))
        if idx in need:
            out[idx] = predict(frame)
        if idx in accepted:
            n += 1
        idx += stride
    return out, reads


def open_like_build(clip):
    """build_trajectories.py line 172: cv2.CAP_FFMPEG first, then the default backend."""
    import cv2
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(clip))
    return cap


def build_read(clip, side: dict, max_frames: int, is_live, need: set, predict) -> dict:
    """The build's own read path (v1.3, 14.2): open as the build opens, the pre-gate pass with the build's gate, the
    intervals, then the tracking loop's reads through one videoseq.SeqReader. Validation: every sidecar frame is read,
    and the frames the loop counts as accepted equal the sidecar's. -> preds, reads, seeks, backend, validation."""
    import config
    from videoseq import SeqReader
    cap = open_like_build(clip)
    try:
        import cv2
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reader = SeekCount(SeqReader(cap))
        pg_stride, pg_pad = config.pregate_params(fps)
        intervals = pregate_intervals(reader, total, pg_stride, pg_pad, is_live)
        pre_seeks = list(reader.seeks)
        stride = int(side.get("stride") or 1)
        accepted = {int(r["frame"]) for r in side["frames"]}
        preds, reads = main_reads(reader, total, intervals, stride, accepted, max_frames, need, predict)
        rs = set(reads)
        valid = accepted <= rs and len(rs & accepted) == len(accepted)
        return {"preds": preds, "n_reads": len(reads), "intervals": intervals, "backend": backend,
                "seeks_pregate": pre_seeks, "seeks_main": reader.seeks[len(pre_seeks):],
                "validation": {"sidecar_frames_read": len(rs & accepted), "sidecar_frames": len(accepted),
                               "passes": bool(valid)}}
    finally:
        cap.release()


SMOKE_WINDOWS = ("gsw_bkn_2015_s04_w00", "gsw_cha_2016_s05_w00", "gsw_cha_2016_s05_w03",     # good (a3run reproduced)
                 "gsw_bkn_2015_s07_w00", "gsw_cha_2016_s07_w00",                              # bad under the old read
                 "gsw_bkn_2015_s00_w07")       # first frame 54, no gap over 300 frames (the first such window by name)


ALIGN_ENV = {"HF_HUB_OFFLINE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}    # the build's MPS fallback; no hub fetch


def collect_align(game: str, only: tuple | None = None) -> dict:
    refuse_game(game)
    bad_env = {k: os.environ.get(k) for k, v in ALIGN_ENV.items() if os.environ.get(k) != v}
    if bad_env:
        raise SystemExit(f"{WHO}: set {ALIGN_ENV} before the run (got {bad_env})")
    import hashlib
    import config
    from tools import p8_frozen as F
    from tools import p8_piibox as PB
    from tools import p8_rebaseline as RB
    from tools.p8_a3run import CONF, NMS_IOU, R23_SHA
    from tools.p8_common import main_checkout
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    head = PB.git_head()
    if not head["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if sha != R23_SHA:
        raise SystemExit(f"{WHO}: weights sha256 {sha} is not r23's")
    config.PLAYER_DETECTOR_WEIGHTS = weights
    score = RB.read_json(r["ref"] / "rebaseline" / f"score__{game}__current.json")
    watched = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    t0 = time.time()
    P = F.install_paths()
    comp = P["builds"] / "r23"
    wins = sorted((w for w in F.label_windows(P, game) if (comp / f"{w['window']}_frames.json").exists()),
                  key=lambda w: w["window"])
    stems = [w["window"] for w in wins]
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if sorted(score["windows"]) != stems:
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's score record")
    from ultralytics import YOLO
    det = YOLO(str(weights))
    from gate.backbones import get_backbone
    from gate.trained_head import TrainedHeadGate
    from sportvu.rebuild import build_stride
    mc = main_checkout() / "models"
    head_p, thr = mc / config.HEAD_V2_PATH.name, json.loads((mc / config.THRESHOLDS_PATH.name).read_text())["v2"]
    gates = {}

    gate_rec = {"head": head_p.name, "head_sha256": RB.sha256(head_p),
                "coefs": head_p.with_name(head_p.stem + "_coefs.npz").name,
                "coefs_sha256": RB.sha256(head_p.with_name(head_p.stem + "_coefs.npz")),
                "threshold_v2": thr, "clip_backbone": config.CLIP_MODEL_NAME, "env": dict(ALIGN_ENV)}

    def gate_for(dev: str):
        if dev not in gates:
            gates[dev] = TrainedHeadGate.load(head_p, backbone=get_backbone("clip", dev), threshold=thr)
        return gates[dev]

    def predict(img):
        res = det.predict(img, classes=[config.PLAYER_CLASS_ID], conf=CONF, iou=NMS_IOU, device="cpu", verbose=False)[0]
        return [[float(v) for v in b] for b in res.boxes.xyxy.cpu().numpy()]

    out_w = {}
    for w in wins:
        s = w["window"]
        if only is not None and s not in only:
            continue
        if time.time() - t0 > ALIGN_CAP_S:
            raise SystemExit(f"{WHO}: the 60-minute cap was reached; nothing is reported")
        side_p = comp / f"{s}_frames.json"
        if RB.sha256(side_p) != score["hashes"][s]["sidecar"]:
            raise SystemExit(f"{WHO}: {s} sidecar differs from D202's record")
        side = json.loads(side_p.read_text())
        stride = int(side.get("stride") or 1)
        srow = {int(x["frame"]): x for x in side["frames"]}
        picks, n_elig = pick_frames(side)
        rec = {"eligible": n_elig, "picks": picks}
        if not picks:
            out_w[s] = {**rec, "verdict": "undetermined", "why": "fewer than 10 eligible frames"}
            continue
        tw = time.time()
        dev_rec = RB.read_json(comp / f"{s}_device.json") if (comp / f"{s}_device.json").exists() else {}
        gdev = (dev_rec.get("resolved") or {}).get("gate") or "cpu"
        max_frames = (w["f_end"] - w["f_start"]) // build_stride(w["fps"]) + 2
        if build_stride(w["fps"]) != stride:
            raise SystemExit(f"{WHO}: {s}: the sidecar stride is not the build's")
        br = build_read(P["label_build"] / f"{s}.mp4", side, max_frames, gate_for(gdev).is_court_visible, set(picks),
                        predict)
        rec.update({"gate_device": gdev, "backend": br["backend"], "n_reads": br["n_reads"],
                    "intervals": br["intervals"], "seeks_pregate": len(br["seeks_pregate"]),
                    "seeks_main": br["seeks_main"], "read_validation": br["validation"]})
        if not br["validation"]["passes"] or set(picks) - set(br["preds"]):
            out_w[s] = {**rec, "verdict": "undetermined", "why": "the build's read path is not reproduced"}
            continue
        pred = br["preds"]
        per = []
        for f in picks:
            pf = pred[f]
            fr = {k: frame_ious([b["bbox"] for b in srow[f + o * stride]["boxes"]], pf)
                  for k, o in (("-1", -1), ("0", 0), ("+1", 1))}
            fr["frame"] = f
            fr["unmatched_detections"] = max(0, len(pf) - sum(x > 0 for x in fr["0"]))
            fr["unmatched_sidecar_at_f"] = sum(x == 0 for x in fr["0"])
            per.append(fr)
        st = align_stats(per)
        out_w[s] = {**rec, **st, "verdict": align_verdict(st), "per_frame": per,
                    "seconds": round(time.time() - tw, 1)}
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    import cv2
    import torch
    import ultralytics
    out = {"game": game, "design": "docs/research/P8_posefoot.md section 14 (stage A)", "git": head,
           "weights_sha256": sha, "versions": {"opencv": cv2.__version__, "torch": torch.__version__,
                                              "ultralytics": ultralytics.__version__},
           "gate": gate_rec, "windows": out_w, "summary": game_alignment({k: v["verdict"] for k, v in out_w.items()}),
           "manifest_unchanged": after == watched, "seconds": round(time.time() - t0, 1)}
    dd = r["ref"] / "posefoot" / ("alignment_smoke" if only is not None else "alignment")
    dd.mkdir(parents=True, exist_ok=True)
    out["smoke"] = only is not None
    atomic_write(dd / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "summary", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


def by_cause(windows: dict) -> dict:
    """The windows not aligned, by cause: misaligned, or undetermined by its reason."""
    out = {}
    for w, x in sorted(windows.items()):
        if x["verdict"] == "aligned":
            continue
        key = "misaligned" if x["verdict"] == "misaligned" else f"undetermined: {x.get('why')}"
        out.setdefault(key, []).append(w)
    return {k: {"n": len(v), "windows": v} for k, v in out.items()}


def report_align(smoke: bool = False) -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    sub = "alignment_smoke" if smoke else "alignment"
    recs = {g: RB.read_json(r["ref"] / "posefoot" / sub / f"{g}.json") for g in POOL}
    a3 = RB.read_json(REPO / "reports" / "research" / "p8_a3align.json")["games"]
    out = {"what": "docs/research/P8_posefoot.md section 14 with 14.2 (v1.3), stage A: decode alignment of the pool "
                   "windows" + (" (the six-window smoke)" if smoke else "") + "; report only", "smoke": smoke,
           "git": {g: recs[g]["git"] for g in POOL}, "seconds": {g: recs[g]["seconds"] for g in POOL},
           "versions": {g: recs[g]["versions"] for g in POOL}, "per_game": {}}
    for g in POOL:
        v = {w: x["verdict"] for w, x in recs[g]["windows"].items()}
        a3g = {w: x["misaligned"] for w, x in a3.get(g, {}).get("windows", {}).items()}
        out["per_game"][g] = {"summary": game_alignment(v), "cross_table_a3align": cross_table(v, a3g),
                              "not_aligned": sorted(w for w, x in v.items() if x != "aligned"),
                              "not_aligned_by_cause": by_cause(recs[g]["windows"]),
                              "gate": recs[g].get("gate"),
                              "M_and_L0": {w: {"M": x.get("M"), "L0": x.get("L0"), "verdict": x["verdict"],
                                               "why": x.get("why"), "read_validation": x.get("read_validation"),
                                               "seeks_main": x.get("seeks_main"), "backend": x.get("backend")}
                                           for w, x in recs[g]["windows"].items()},
                              "manifest_unchanged": recs[g]["manifest_unchanged"]}
    out["stop"] = any(out["per_game"][g]["summary"]["stop"] for g in POOL)
    out = clean(out)
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD_ALIGN.with_name("p8_posefoot_align_smoke.json") if smoke else RECORD_ALIGN, text)
    print(json.dumps({g: out["per_game"][g]["summary"] for g in POOL} | {"stop": out["stop"]}, indent=1))
    return out


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def unit_rows(per_game: dict, gs: list, cand: str = "raw", hit_all: dict | None = None) -> dict:
    """The rows of section 11.3 for a unit (one game or both), with window intervals."""
    from tools.p8_piibox import median_iv, per_window, picks_for, ratio_iv, values_by_window
    n_w = [per_game[g]["windows"] for g in gs]
    picks = picks_for(n_w)
    pii = [[x for x in per_game[g]["rows"] if x["kind"] == "P-ii"] for g in gs]
    own = [[x for x in rs if x["class"] in OWN] for rs in pii]
    hit = [[x for x in per_game[g]["rows"] if x["kind"] == "hit"] for g in gs]

    def share(rs_g, fn, need):
        return ratio_iv([per_window([x for x in rs if need(x)], nw, fn) for rs, nw in zip(rs_g, n_w)],
                        [per_window([x for x in rs if need(x)], nw, lambda x: 1) for rs, nw in zip(rs_g, n_w)], picks)

    def delta(rs_g, frame):
        vals = [values_by_window([{"wi": x["wi"], "v": x[frame][cand] - x[frame]["stab"]} for x in rs
                                  if x[frame][cand] is not None and x[frame]["stab"] is not None], nw, "v")
                for rs, nw in zip(rs_g, n_w)]
        return median_iv(vals, picks)

    within = lambda key: (lambda x: x["dT"][key] is not None and x["dT"][key] <= MISS_FT)
    paired = lambda key: (lambda x: x["dT"][cand] is not None and x["dT"].get(key) is not None)
    R = share(own, within(cand), paired("null"))                     # v1.2 (A1): the same boxes as R_null
    Rn = share(own, within("null"), paired("null"))
    Rx = share(own, within(cand), paired("xnull"))
    Rxn = share(own, within("xnull"), paired("xnull"))
    lost = share(hit, lambda x: x["dT"][cand] > MISS_FT, lambda x: x["dT"][cand] is not None)
    out = {"P-ii own": {"n": sum(len(x) for x in own), "R": R["value"], "R_interval": R["interval"],
                        "R_null": Rn["value"], "R_null_interval": Rn["interval"],
                        "R_all_boxes": share(own, within(cand), lambda x: x["dT"][cand] is not None),
                        "cross_frame_null": {"R": Rx, "R_xnull": Rxn},
                        "med_T": delta(own, "dT"), "med_B": delta(own, "dB")},
           "P-ii all": {"n": sum(len(x) for x in pii), "R": share(pii, within(cand), lambda x: x["dT"][cand] is not None)},
           "hits": {"n": sum(len(x) for x in hit), "lost": lost["value"], "lost_interval": lost["interval"],
                    "med_T": delta(hit, "dT"), "med_B": delta(hit, "dB")}}
    # est and the ghost change (section 11.4), coverage 1 for foot_raw
    hits_all = sum(hit_all[g] for g in gs) if hit_all else None
    s_hit = hits_all / out["hits"]["n"] if hits_all and out["hits"]["n"] else 1.0
    num = [per_window(rs, nw, lambda x: -(x["dT"][cand] is not None and x["dT"][cand] <= MISS_FT))
           + s_hit * per_window(hs, nw, lambda x: x["dT"][cand] is not None and x["dT"][cand] > MISS_FT)
           for rs, hs, nw in zip(pii, hit, n_w)]
    den = [np.asarray(per_game[g]["in_frame_by_window"], float) for g in gs]
    out["est"] = ratio_iv(num, den, picks)
    gnum = [per_window(rs, nw, lambda x: int(bool(x["gT"][cand])) - int(bool(x["gT"]["stab"])))
            + s_hit * per_window(hs, nw, lambda x: int(bool(x["gT"][cand])) - int(bool(x["gT"]["stab"])))
            for rs, hs, nw in zip(pii, hit, n_w)]
    out["ghost_change_per_box"] = ratio_iv(gnum, [np.asarray(per_game[g]["nr_boxes_by_window"], float) for g in gs], picks)
    out["s_hit"] = s_hit
    sp = {}
    for b in ("under 5", "5 to 10", "10 and over"):
        sp[b] = share(own, within(cand), lambda x, b=b: speed_bin(x["speed"]) == b and x["dT"][cand] is not None)
        sp[b]["n"] = sum(speed_bin(x["speed"]) == b and x["dT"][cand] is not None for rs in own for x in rs)
    out["R_by_speed"] = sp
    out["no_speed_rows"] = sum(x["speed"] is None for rs in own for x in rs)
    return out


def offset_sizes(sidecar_paths: list) -> dict:
    """|foot_raw - foot_stab| in px over every box of the given sidecars (the risk note of section 13)."""
    v = []
    for p in sidecar_paths:
        for row in json.loads(Path(p).read_text())["frames"]:
            for b in row["boxes"]:
                v.append(float(np.hypot(b["foot_raw"][0] - b["foot_stab"][0], b["foot_raw"][1] - b["foot_stab"][1])))
    v = np.asarray(v)
    return {"boxes": int(len(v)), "sidecars": len(sidecar_paths),
            "median_px": float(np.median(v)) if len(v) else None,
            "quartiles_px": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if len(v) else None}


def report0a() -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    per = {g: RB.read_json(r["ref"] / "posefoot" / "0a" / f"{g}.json") for g in POOL}
    o2o = {g: RB.read_json(r["ref"] / "one2one" / f"{g}.json") for g in POOL}
    hit_all = {g: sum(1 for _, rec in o2o[g]["rows"] if rec["class"] == "hit") for g in POOL}
    res = {g: unit_rows(per, [g], "raw", hit_all) for g in POOL}
    res["pooled"] = unit_rows(per, list(POOL), "raw", hit_all)
    rules = {g: rule_set(res[g]["P-ii own"], res[g]["hits"], res[g]["est"], "foot_raw") for g in POOL}
    out = {"what": "docs/research/P8_posefoot.md version 1.1, stage 0a: foot_raw against foot_stab on D207's boxes; "
                   "report only", "git": {g: per[g]["git"] for g in POOL}, "gates": {g: per[g]["gate"] for g in POOL},
           "counters": {g: per[g]["counters"] for g in POOL}, "seconds": {g: per[g]["seconds"] for g in POOL},
           "N_hits_all (D205 hit class)": hit_all, "rows": res, "rules_foot_raw": rules,
           "readings": readings_0a(rules)}
    comp = r["ref"] / "builds" / "r23"
    paths = sorted(p for g in POOL for p in comp.glob(f"{g}_s*_frames.json"))
    out["foot_raw_minus_foot_stab_px"] = {
        "what": "|foot_raw - foot_stab| over every box of the two games' builds/r23 sidecars (section 13's risk note)",
        **offset_sizes(paths)}
    out = clean(out)
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD_0A, text)
    print(json.dumps({"rules": out["rules_foot_raw"], "readings": out["readings"]}, indent=1))
    return out


def fetch_model() -> dict:
    """Stage 0: the one download of the pinned revision into the Hugging Face cache (outside the repository and
    iCloud), SHA-256 per file, model.safetensors checked against the Hub's LFS oid. Later loads use HF_HUB_OFFLINE=1."""
    from huggingface_hub import hf_hub_download
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    files = {}
    for name in MODEL_FILES:
        p = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REV))
        files[name] = {"sha256": RB.sha256(p), "bytes": p.stat().st_size}
        if name in MODEL_LFS_SHA and files[name]["sha256"] != MODEL_LFS_SHA[name]:
            raise SystemExit(f"{WHO}: {name} does not match the Hub's LFS oid")
    out = clean({"what": "docs/research/P8_posefoot.md stage 0: the pose model's one download", "repo": MODEL_REPO,
                 "revision": MODEL_REV, "files": files, "licence": "docs/DATA_SOURCES.md, ViTPose-B row",
                 "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD_MODEL, text)
    print(text)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("stage0a")
    c.add_argument("game")
    sub.add_parser("report0a")
    c2 = sub.add_parser("align")
    c2.add_argument("game")
    c2.add_argument("--smoke", action="store_true", help="the six smoke windows of 14.2 only")
    sub.add_parser("fetch_model")
    ra = sub.add_parser("report_align")
    ra.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "stage0a":
        refuse_game(a.game)
        collect0a(a.game)
    elif a.cmd == "align":
        refuse_game(a.game)
        collect_align(a.game, SMOKE_WINDOWS if a.smoke else None)
    elif a.cmd == "report0a":
        report0a()
    elif a.cmd == "fetch_model":
        fetch_model()
    else:
        report_align(a.smoke)


if __name__ == "__main__":
    sys.exit(main())

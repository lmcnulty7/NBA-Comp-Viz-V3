"""tools/p8_piibox.py: what the P-ii boxes are (docs/research/P8_piibox.md, version 2). Report only.

    <frozen CV python> -m tools.p8_piibox collect GAME   inside the lock ("p8 piibox GAME"), V3_P8_ROOT = work_ref
    python3 -m tools.p8_piibox report                    -> reports/research/p8_piibox.json

collect: per window, the pinned score_window for the scored frames and D202's counts; per scored frame, D205's main
two-stage assignment (tools.p8_one2one.two_stage and frame_classes) on the truth camera's feet; then, per window, each
P-ii stage B box and each stage A hit box classified by its canonical track's pairings on the window's other scored
frames (own, own with crossings, another's with crossings, switch, another's, own misplaced, ghost referee / short /
persistent), with box geometry (height against the label row camera's expected height, truncation, overlap). Count
gates against D202's and D205's records per window. hou_cle_2016: D202's count gate only; no class, geometry or camera
is computed on it. Writes only work_ref/piibox/<game>.json and the report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_piibox"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
MEASURED, GATE_ONLY = (CHA, BKN), (HOU,)
MISS_FT, NEAR_FT, GHOST_FT = 2.5, 5.0, 3.0
CROSS_FRAC = 0.2                      # the minority rule (section 4)
SHORT_FRAMES = 3                      # ghost, short: fewer than 3 scored frames
PROX_FT, PROX_SHARE = 5.0, 0.5        # the 5 ft proximity share
PERSON_FT = 6.5                       # nominal height for the expected box height
RATIO_BINS = (0.75, 1.25)
EDGE_PX = 2
IOU_BAR, INSIDE_BAR = 0.3, 0.5
CAM_TOL = 1e-6
BOOT_N, SEED = 2000, 0
CLASSES = ("own", "own, with crossings", "another's, with crossings", "switch", "another's", "own, misplaced",
           "ghost, referee-flagged", "ghost, short", "ghost, persistent")
GROUPS = {"own (all three)": ("own", "own, with crossings", "own, misplaced"),
          "another's (both)": ("another's", "another's, with crossings"),
          "switch": ("switch",),
          "ghost (all three)": ("ghost, referee-flagged", "ghost, short", "ghost, persistent")}
PROX_CLASSES = ("another's", "another's, with crossings", "own, misplaced", "ghost, persistent")
D205_CLASSES = ("hit", "P-i", "P-ii", "P-iii-ref", "D-i", "D-iii-ref", "D-iii-free", "D-iii-none")
GATE_FIELDS = ("in_frame", "missed", "far", "nr_boxes", "missed_1to1", "unmatched_1to1")
RECORD = REPO / "reports" / "research" / "p8_piibox.json"


# ── pure: classes ──────────────────────────────────────────────────────────────────────────────────────────────────
def classify(n_own: int, n_other: int, n_ownB: int, referee_flagged: bool, present: int) -> str:
    """Section 4's classes, in their registered order."""
    if n_own >= 1 and n_other == 0:
        return "own"
    if n_own >= 1 and n_other >= 1:
        tot = n_own + n_other
        if n_other <= CROSS_FRAC * tot:
            return "own, with crossings"
        if n_own <= CROSS_FRAC * tot:
            return "another's, with crossings"
        return "switch"
    if n_other >= 1:
        return "another's"
    if n_ownB >= 1:
        return "own, misplaced"
    if referee_flagged:
        return "ghost, referee-flagged"
    if present < SHORT_FRAMES:
        return "ghost, short"
    return "ghost, persistent"


def track_table(frames: list) -> dict:
    """frames: [{"f", "cid": [per box], "A": {box: pid}, "B": {box: pid}, "ref": [per box], "ft": (n, 2) court feet}]
    -> {cid: {f: {"A": set of pids, "B": set of pids, "ref": bool, "ft": [court feet]}}}."""
    out = {}
    for fr in frames:
        for k, c in enumerate(fr["cid"]):
            e = out.setdefault(c, {}).setdefault(fr["f"], {"A": set(), "B": set(), "ref": False, "ft": []})
            if k in fr["A"]:
                e["A"].add(fr["A"][k])
            if k in fr["B"]:
                e["B"].add(fr["B"][k])
            e["ref"] = e["ref"] or bool(fr["ref"][k])
            e["ft"].append(np.asarray(fr["ft"][k], float))
    return out


def evidence(track: dict, pid: int, f: int, sv_of: dict, fps: float) -> dict:
    """Counts and report-only fields of one box (its track's frames, frame f excluded). sv_of: {frame: {pid: xy}}."""
    n_own = n_other = n_ownB = 0
    others, own_fr, other_fr, near, both = {}, [], [], 0, 0
    for g, e in track.items():
        if g == f:
            continue
        if pid in e["A"]:
            n_own += 1; own_fr.append(g)
        if e["A"] - {pid}:
            n_other += 1; other_fr.append(g)
            for q in e["A"] - {pid}:
                others[q] = others.get(q, 0) + 1
        if pid in e["B"]:
            n_ownB += 1
        xy = sv_of.get(g, {}).get(pid)
        if xy is not None:
            both += 1
            near += min(float(np.hypot(*(p - xy))) for p in e["ft"]) <= PROX_FT
    gap = lambda fr: (min(abs(g - f) for g in fr) / fps) if fr else None
    return {"n_own": n_own, "n_other": n_other, "n_ownB": n_ownB, "present": len(track),
            "referee_flagged": any(e["ref"] for e in track.values()),
            "gap_own_s": gap(own_fr), "gap_other_s": gap(other_fr),
            "top_other": max(others, key=lambda q: (others[q], -q)) if others else None,
            "prox_share": near / both if both else None}


# ── pure: geometry ─────────────────────────────────────────────────────────────────────────────────────────────────
def expected_height(cam: dict, foot_ft, W: int, H: int) -> float | None:
    """Pixel distance between a court point projected at height 0 and at PERSON_FT through the label row's camera."""
    from tools.p8_scorecard import project3d
    x, y = float(foot_ft[0]), float(foot_ft[1])
    uv, depth = project3d(cam["camera_C"], cam["camera_rvec"], cam["camera_f"], W, H, [[x, y, 0.0], [x, y, PERSON_FT]])
    if not (depth > 0).all():
        return None
    return float(np.hypot(*(uv[1] - uv[0])))


def ratio_bin(r: float | None) -> str | None:
    if r is None:
        return None
    return "short" if r < RATIO_BINS[0] else ("tall" if r > RATIO_BINS[1] else "normal")


def overlap(boxes: list, k: int) -> tuple:
    """(largest IoU with another box, largest share of box k's area inside another box)."""
    from sportvu import bench as Bn
    a = boxes[k]
    area = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    best_iou = best_in = 0.0
    for i, b in enumerate(boxes):
        if i == k:
            continue
        best_iou = max(best_iou, Bn._iou(a, b))
        iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        if area > 0:
            best_in = max(best_in, iw * ih / area)
    return best_iou, best_in


def truncation(bbox, W: int, H: int) -> tuple:
    x1, y1, x2, y2 = bbox[:4]
    return bool(y2 >= H - EDGE_PX), bool(x1 <= EDGE_PX or x2 >= W - EDGE_PX or y1 <= EDGE_PX)


def camera_matches(row: dict, W: int, H: int) -> bool:
    from sportvu.camera import h_px2ft
    Hc = np.asarray(h_px2ft(np.asarray(row["camera_C"], float), np.asarray(row["camera_rvec"], float),
                            float(row["camera_f"]), W, H), float).reshape(3, 3)
    Ht = np.asarray(row["H_truth"], float).reshape(3, 3)
    return bool(np.abs(Hc / Hc[2, 2] - Ht / Ht[2, 2]).max() <= CAM_TOL)


def depth_lateral(cam_C, foot_ft, sv_xy) -> tuple:
    from tools.p8_placement import depth_basis
    u, v = depth_basis((float(cam_C[0]), float(cam_C[1])), foot_ft)
    o = np.asarray(foot_ft, float) - np.asarray(sv_xy, float)
    return float(o @ u), float(o @ v)


# ── pure: the D205 gate ────────────────────────────────────────────────────────────────────────────────────────────
def d205_window_counts(d205_rows: list, n_windows: int) -> tuple:
    """D205's record rows ([window index, rec]) -> ({window index: {class: count}}, {window index: sorted stage B ft})."""
    counts = {i: {c: 0 for c in D205_CLASSES} for i in range(n_windows)}
    sb = {i: [] for i in range(n_windows)}
    for wi, rec in d205_rows:
        counts[wi][rec["class"]] = counts[wi].get(rec["class"], 0) + 1
        if rec["class"] == "P-ii":
            sb[wi].append(rec["stage_b_ft"])
    return counts, {i: sorted(v) for i, v in sb.items()}


def gate_d205(mine_counts: dict, mine_sb: list, want_counts: dict, want_sb: list) -> list:
    bad = []
    keys = set(D205_CLASSES) | set(mine_counts) | set(want_counts)
    if {c: mine_counts.get(c, 0) for c in keys} != {c: want_counts.get(c, 0) for c in keys}:
        bad.append("class counts differ")
    if len(mine_sb) != len(want_sb) or any(abs(a - b) > 1e-9 for a, b in zip(sorted(mine_sb), want_sb)):
        bad.append("stage B distances differ")
    return bad


# ── pure: report statistics ────────────────────────────────────────────────────────────────────────────────────────
def picks_for(n_windows: list, n: int = BOOT_N, seed: int = SEED) -> list:
    """Per game, (n, windows) resample indices: windows resampled whole within each game (D205's convention), drawn
    once per unit and shared by every statistic of that unit."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, nw, (n, nw)) if nw else np.zeros((n, 0), int) for nw in n_windows]


def ratio_iv(nums: list, dens: list, picks: list) -> dict:
    """nums, dens: per game, per-window arrays. Point sum(num) / sum(den) and its window interval."""
    N, D = sum(float(np.sum(x)) for x in nums), sum(float(np.sum(x)) for x in dens)
    sn = sum(np.asarray(x, float)[p].sum(1) for x, p in zip(nums, picks))
    sd = sum(np.asarray(x, float)[p].sum(1) for x, p in zip(dens, picks))
    keep = sd > 0
    return {"value": N / D if D else None,
            "interval": [float(np.percentile(sn[keep] / sd[keep], 2.5)), float(np.percentile(sn[keep] / sd[keep], 97.5))]
            if keep.any() else None}


def median_iv(vals: list, picks: list) -> dict:
    """vals: per game, a list (per window) of arrays. Point median and its window interval."""
    allv = np.concatenate([np.asarray(a, float) for g in vals for a in g]) if any(len(a) for g in vals for a in g) \
        else np.zeros(0)
    if not len(allv):
        return {"value": None, "interval": None}
    reps = []
    for k in range(len(picks[0]) if picks else 0):
        parts = [np.asarray(vals[gi][i], float) for gi, p in enumerate(picks) for i in p[k]]
        v = np.concatenate(parts) if parts else np.zeros(0)
        if len(v):
            reps.append(float(np.median(v)))
    return {"value": float(np.median(allv)),
            "interval": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))] if reps else None}


def per_window(rows: list, n_windows: int, value) -> np.ndarray:
    """Sum of value(row) per window (value returns a number, 0 for rows it does not count)."""
    out = np.zeros(n_windows)
    for r in rows:
        out[r["wi"]] += value(r)
    return out


def values_by_window(rows: list, n_windows: int, key: str) -> list:
    out = [[] for _ in range(n_windows)]
    for r in rows:
        if r.get(key) is not None:
            out[r["wi"]].append(r[key])
    return out


def class_shares(rows_g: list, n_w: list, picks: list) -> dict:
    """Each class's (and each group's) share of the rows, with its window interval."""
    dens = [per_window(rs, nw, lambda r: 1) for rs, nw in zip(rows_g, n_w)]
    out = {}
    for label, cls in [(c, (c,)) for c in CLASSES] + list(GROUPS.items()):
        nums = [per_window(rs, nw, lambda r, cls=cls: r["class"] in cls) for rs, nw in zip(rows_g, n_w)]
        out[label] = ratio_iv(nums, dens, picks)
    return out


def geometry(rows_g: list, n_w: list, picks: list) -> dict:
    """Section 5's geometry of a set of rows (per game lists), with window intervals."""
    out = {"n": sum(len(rs) for rs in rows_g), "no_ratio": sum(r.get("ratio") is None for rs in rows_g for r in rs),
           "ratio_median": median_iv([values_by_window(rs, nw, "ratio") for rs, nw in zip(rows_g, n_w)], picks)}
    v = [r["ratio"] for rs in rows_g for r in rs if r.get("ratio") is not None]
    out["ratio_quartiles"] = [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v else None
    dens_r = [per_window(rs, nw, lambda r: r.get("bin") is not None) for rs, nw in zip(rows_g, n_w)]
    for b in ("short", "normal", "tall"):
        out[f"ratio_{b}"] = ratio_iv([per_window(rs, nw, lambda r, b=b: r.get("bin") == b) for rs, nw in zip(rows_g, n_w)],
                                     dens_r, picks)
    dens = [per_window(rs, nw, lambda r: 1) for rs, nw in zip(rows_g, n_w)]
    for k in ("trunc_bottom", "trunc_other", "iou_ge_0.3", "inside_ge_0.5", "scorer_ghost"):
        out[k] = ratio_iv([per_window(rs, nw, lambda r, k=k: bool(r[k])) for rs, nw in zip(rows_g, n_w)], dens, picks)
    h = [r["h"] for rs in rows_g for r in rs]
    out["height_px_median"] = float(np.median(h)) if h else None
    out["third"] = {t: sum(r.get("third") == t for rs in rows_g for r in rs) for t in ("near", "middle", "far")}
    return out


def median_of(rows: list, key: str):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(v)) if v else None


def share_true(rows: list, key: str):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def git_head() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    return {"head": head, "tools_and_sportvu_clean": not dirty}


def collect(game: str) -> dict:
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX, near_side
    from sportvu.sync import MIRRORS, apply_mirror
    from tools import p8_frozen as F
    from tools import p8_one2one as O
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_e1_score import window_stats
    from tools.p8_placecause import cache_gaps
    from tools.p8_resetfix import one_to_one_missed, unmatched_1to1
    if game not in MEASURED + GATE_ONLY:
        raise SystemExit(f"{WHO}: {game} is not in this item")
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    head = git_head()
    if not head["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    pinned, now = RB.read_json(p8_pin.PIN_FILE), p8_pin.current()
    if any(now[k] != pinned.get(k) for k in now):
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    measured = game in MEASURED
    score = RB.read_json(r["ref"] / "rebaseline" / f"score__{game}__current.json")
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
    if sorted(score["windows"]) != stems:
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's score record")
    if measured:
        d205 = RB.read_json(r["ref"] / "one2one" / f"{game}.json")
        if d205.get("windows") != len(stems):
            raise SystemExit(f"{WHO}: {game}: D205's record covers {d205.get('windows')} windows, not {len(stems)}")
        want_counts, want_sb = d205_window_counts(d205["rows"], len(stems))
    rows, counters = [], {"no_track": 0, "camera_mismatch": 0, "stage_b_disagrees": 0}
    for wi, w in enumerate(wins):
        s = w["window"]
        for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                       ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
            want = (score.get("hashes") or {}).get(s, {}).get(key)
            if want is None and key in ("sidecar", "timemap", "labels"):
                raise SystemExit(f"{WHO}: {s} has no {key} hash in D202's record")
            if want is not None and RB.sha256(p) != want:
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
        c = {k: 0 for k in GATE_FIELDS if k != "nr_boxes"}
        frames, mine_counts, mine_sb, sv_of, items = [], {k: 0 for k in D205_CLASSES}, [], {}, []
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
            feet = np.array([b["foot_stab"] for b in x["boxes"]], np.float32).reshape(-1, 1, 2)
            ft = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2).astype(np.float64) if len(feet) else np.zeros((0, 2))
            is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                               any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
                               for b in x["boxes"]], bool)
            svf = np.asarray(sv, np.float64)
            c["unmatched_1to1"] += unmatched_1to1(ft, svf, inside, is_ref, 2.5)
            c["missed_1to1"] += one_to_one_missed(ft, svf, inside, 2.5)
            Dm = np.sqrt(((svf[:, None, :] - ft[None, :, :]) ** 2).sum(-1)).min(1) if len(ft) else np.full(len(svf), np.inf)
            c["in_frame"] += int(inside.sum())
            c["missed"] += int((inside & (Dm > MISS_FT)).sum())
            c["far"] += int((inside & (Dm > NEAR_FT)).sum())
            if not measured:
                continue
            if not camera_matches(row, W, H):
                counters["camera_mismatch"] += 1
                raise SystemExit(f"{WHO}: {s} frame {f}: the label row's camera does not reproduce its H_truth")
            cls = O.frame_classes(ft, svf, inside, is_ref)
            nonref = [int(b) for b in np.where(~is_ref)[0]]
            main, _ = O.two_stage(ft, svf, range(len(svf)), nonref, O._assign_all)
            cids = []
            for b in x["boxes"]:
                if b.get("tid") is None:
                    counters["no_track"] += 1
                    raise SystemExit(f"{WHO}: {s} frame {f}: a box has no track id")
                cids.append(idmap.get(int(b["tid"]), int(b["tid"])))
            A = {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "A"}
            Bm = {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "B"}
            frames.append({"f": f, "cid": cids, "A": A, "B": Bm, "ref": list(is_ref), "ft": ft})
            sv_of[f] = {pids[j]: svf[j] for j in range(len(svf))}
            bbs = [b["bbox"] for b in x["boxes"]]
            for j, rec in cls.items():
                mine_counts[rec["class"]] = mine_counts.get(rec["class"], 0) + 1
                if rec["class"] not in ("P-ii", "hit"):
                    continue
                bk, dd, stg = main[j]
                if rec["class"] == "P-ii":
                    mine_sb.append(rec["stage_b_ft"])
                    if stg != "B" or abs(dd - rec["stage_b_ft"]) > 1e-12:
                        counters["stage_b_disagrees"] += 1
                        raise SystemExit(f"{WHO}: {s} frame {f}: two_stage's stage B box disagrees with frame_classes")
                elif stg != "A":
                    raise SystemExit(f"{WHO}: {s} frame {f}: a hit's box is not a stage A match")
                bb = bbs[bk]
                h = float(bb[3] - bb[1])
                e = expected_height(row, ft[bk], W, H)
                iou, ins = overlap(bbs, bk)
                tb, to = truncation(bb, W, H)
                dep, lat = depth_lateral(row["camera_C"], ft[bk], svf[j])
                items.append({"wi": wi, "kind": rec["class"], "f": f, "pid": pids[j], "cid": cids[bk], "h": h,
                              "ratio": h / e if e else None, "trunc_bottom": tb, "trunc_other": to,
                              "iou": iou, "inside": ins, "depth": dep, "lateral": lat,
                              "scorer_ghost": bool(np.sqrt(((svf - ft[bk]) ** 2).sum(1)).min() > GHOST_FT),
                              "third": S.third_of(float(ft[bk][1]), near_side(Ht)),
                              "in_frame_of": {q: bool(inside[jj]) for jj, q in enumerate(pids)},
                              "dist_of": {q: float(np.hypot(*(svf[jj] - svf[j]))) for jj, q in enumerate(pids)}})
        want = score["windows"][s]
        mine = {**c, "nr_boxes": st["nr_boxes"]}
        if any(mine[k] != want.get(k) for k in GATE_FIELDS) or st["in_frame"] != c["in_frame"] or st["missed"] != c["missed"]:
            raise SystemExit(f"{WHO}: D202 count gate failed on {s}: mine {mine}, D202 {[want.get(k) for k in GATE_FIELDS]}")
        if not measured:
            continue
        bad = gate_d205(mine_counts, mine_sb, want_counts[wi], want_sb[wi])
        if bad:
            raise SystemExit(f"{WHO}: D205 count gate failed on {s}: {bad}")
        tracks = track_table(frames)
        for it in items:
            ev = evidence(tracks[it["cid"]], it["pid"], it["f"], sv_of, fps)
            cl = classify(ev["n_own"], ev["n_other"], ev["n_ownB"], ev["referee_flagged"], ev["present"])
            top = ev["top_other"]
            rows.append({"wi": wi, "kind": it["kind"], "class": cl, **{k: ev[k] for k in ("n_own", "n_other", "n_ownB",
                                                                                         "present", "gap_own_s",
                                                                                         "gap_other_s", "prox_share")},
                         "partner_in_frame": None if top is None else it["in_frame_of"].get(top),
                         "partner_ft": None if top is None else it["dist_of"].get(top),
                         "h": it["h"], "ratio": it["ratio"], "bin": ratio_bin(it["ratio"]),
                         "trunc_bottom": it["trunc_bottom"], "trunc_other": it["trunc_other"],
                         "iou_ge_0.3": it["iou"] >= IOU_BAR, "inside_ge_0.5": it["inside"] >= INSIDE_BAR,
                         "scorer_ghost": it["scorer_ghost"], "depth": it["depth"], "lateral": it["lateral"],
                         "third": it["third"]})
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_piibox.md version 2", "git": head, "windows": len(stems),
           "gate": {"passes": True, "d202_fields": list(GATE_FIELDS),
                    "d205": "class counts and sorted stage B distances per window" if measured else "not computed"},
           "counters": counters, "manifest_unchanged": after == watched, "seconds": round(time.time() - t0, 1)}
    if measured:
        out["rows"] = rows
    else:
        out["note"] = "count gate only; no class, geometry or camera computed, so this game stays unspent"
    d = r["ref"] / "piibox"
    d.mkdir(exist_ok=True)
    atomic_write(d / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "windows", "gate", "counters", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


# ── report (system python, no lock) ────────────────────────────────────────────────────────────────────────────────
def summarize(per_game: dict) -> dict:
    sets = {**{g: [g] for g in per_game}, "pooled": list(per_game)}
    out = {}
    for name, gs in sets.items():
        n_w = [per_game[g]["windows"] for g in gs]
        picks = picks_for(n_w)
        res = {}
        for kind in ("P-ii", "hit"):
            rows_g = [[r for r in per_game[g]["rows"] if r["kind"] == kind] for g in gs]
            allr = [r for rs in rows_g for r in rs]
            res[kind] = {"n": len(allr), "counts": {c: sum(r["class"] == c for r in allr) for c in CLASSES},
                         "shares": class_shares(rows_g, n_w, picks)}
            if kind == "P-ii":
                n_in = sum(per_game[g]["in_frame"] for g in gs)
                res[kind]["rates_over_in_frame"] = {c: res[kind]["counts"][c] / n_in for c in CLASSES}
            geo = {}
            for label, cls in [(c, (c,)) for c in CLASSES] + list(GROUPS.items()) + [("all", CLASSES)]:
                sub = [[r for r in rs if r["class"] in cls] for rs in rows_g]
                if any(sub):
                    geo[label] = geometry(sub, n_w, picks)
            res[kind]["geometry"] = geo
            prox = {}
            for c in PROX_CLASSES:
                sub = [[r for r in rs if r["class"] == c and r["prox_share"] is not None] for rs in rows_g]
                prox[c] = ratio_iv([per_window(rs, nw, lambda r: r["prox_share"] >= PROX_SHARE) for rs, nw in zip(sub, n_w)],
                                   [per_window(rs, nw, lambda r: 1) for rs, nw in zip(sub, n_w)], picks)
            res[kind]["proximity_5ft_share"] = prox
            dist = {}
            for r in allr:
                key = f"{r['class']}|{r['n_own']},{r['n_other']},{r['n_ownB']}"
                dist[key] = dist.get(key, 0) + 1
            res[kind]["n_own_n_other_n_ownB"] = dist
            res[kind]["report_only"] = {
                c: {"median_gap_own_s": median_of(rc, "gap_own_s"), "median_gap_other_s": median_of(rc, "gap_other_s"),
                    "partner_in_frame_share": share_true(rc, "partner_in_frame"),
                    "median_partner_ft": median_of(rc, "partner_ft"), "median_depth_ft": median_of(rc, "depth"),
                    "median_lateral_ft": median_of(rc, "lateral")}
                for c in CLASSES for rc in [[r for r in allr if r["class"] == c]] if rc}
        out[name] = res
    return out


def report() -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    recs = {g: RB.read_json(r["ref"] / "piibox" / f"{g}.json") for g in MEASURED + GATE_ONLY}
    for g in MEASURED:
        recs[g]["in_frame"] = sum(v["in_frame"] for v in
                                  RB.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json")["windows"].values())
    s = summarize({g: recs[g] for g in MEASURED})
    out = {"what": "docs/research/P8_piibox.md version 2; report only; nothing fitted",
           "git": {g: recs[g]["git"] for g in recs}, "gates": {g: recs[g]["gate"] for g in recs},
           "counters": {g: recs[g]["counters"] for g in recs}, "seconds": {g: recs[g]["seconds"] for g in recs},
           "classes": s, "hou_cle_2016": "D202 count gate only; no class, geometry or camera computed (unspent)",
           "limits": "tracks come from the build's identity step; own classes are a lower bound on own boxes; scored "
                     "frames only (every sixth video frame with an accepted truth); build and truth errors are not "
                     "separable and not independent"}
    res = clean(out)
    text = json.dumps(res, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD, text)
    print(json.dumps({g: {k: v["shares"]["own (all three)"] for k, v in res["classes"][g].items()}
                      for g in res["classes"]}, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("game")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    collect(a.game) if a.cmd == "collect" else report()


if __name__ == "__main__":
    sys.exit(main())

"""tools/p8_placecause.py: placement misses by cause on the decided truths (docs/research/P8_placecause.md, version 2).
Report only: nothing is fitted, nothing is built, no truth or score changes.

    <frozen CV python> -m tools.p8_placecause collect GAME   (under the heavy-job lock, "p8 placecause GAME";
        V3_P8_ROOT = work_ref) -> work_ref/placecause/GAME.json (the count gate; for gsw_cha_2016 and gsw_bkn_2015
        also the per-player conditions; hou_cle_2016 gets the gate only and stays unspent)
    python3 -m tools.p8_placecause report                   (system python, no lock) -> reports/research/p8_placecause.json

Per window the pinned scorer (tools.p8_scorecard.score_window) runs as D202 ran it; on its scored frames each in-frame
player's nearest box foot_stab through the truth H classes it hit, placement miss (2.5 to 5 ft) or detection miss
(none within 5 ft), with the scorer's own arithmetic (float32 feet for missed, as score_window; float64 for far, as
tools.p8_resetfix.frame_extra). The count gate compares in_frame, missed, far and nr_boxes with D202's score record
of the game, window by window, and stops on any difference. Conditions and thresholds are those of the design's
section 5, fixed before any number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_placecause"
CHA, BKN, HOU = "gsw_cha_2016", "gsw_bkn_2015", "hou_cle_2016"
MEASURED, GATE_ONLY = (CHA, BKN), (HOU,)
MISS_FT, NEAR_FT = 2.5, 5.0
EDGE_BOX_PX, EDGE_FOOT_PX = 2, 20
IOU_OVERLAP, CROWD_FT = 0.3, 3.0
HEIGHT_LO, HEIGHT_HI, FOOT_MOVE_PX = 0.75, 1.25, 8.0
RESET_HORIZON = 30                       # processed frames
SINCE_CUT_S = 1.0
LINE_PCT, RESID_PCT = 20, 80
CARRY_RATIO, CARRY_SHARE = 1.5, 0.10
BOOT_N, SEED = 2000, 0
SIX = ("C1", "C2", "C3", "C4", "C5", "C6")
RULED = ("C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "D")
CLAUSES = {"C2": ("C2_box", "C2_foot_truth"), "C3": ("C3a", "C3b"), "C3a": ("C3a_iou", "C3a_shared"),
           "C4": ("C4_height_truth", "C4_foot"), "C5": ("C5a", "C5b"), "C6": ("C6",), "D": ("D",), "C1": ("C1",)}
FLAGS = ("C1", "C2_box", "C2_foot_truth", "C3a_iou", "C3a_shared", "C3b", "C4_height_truth", "C4_foot", "C5a", "C5b",
         "C6", "D", "since_cut_lt_1s", "nearest_is_other_1to1")
CLS = {"hit": 0, "placement": 1, "detection": 2}


# ── pure: per-frame conditions ─────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def box_at_edge(bbox, W: int, H: int, px: int = EDGE_BOX_PX) -> bool:
    x1, _, x2, y2 = bbox[:4]
    return x1 <= px or x2 >= W - px or y2 >= H - px


def foot_near_edge(p, W: int, H: int, px: int = EDGE_FOOT_PX) -> bool:
    return p[0] <= px or p[0] >= W - px or p[1] >= H - px


def recent(frame: int, events: list, horizon_frames: int) -> bool:
    """An event at most horizon_frames source frames at or before frame."""
    return any(0 <= frame - e <= horizon_frames for e in events)


def classify(d_missed: float, d_far: float) -> str:
    """hit, placement or detection: missed by the scorer's float32 distance, far by frame_extra's float64 distance."""
    if d_missed <= MISS_FT:
        return "hit"
    return "detection" if d_far > NEAR_FT else "placement"


def frame_conditions(sv, inside, px, ct32, ct64, boxes, crowd, heights, third, W, H, one_to_one: dict) -> list:
    """Per in-frame player of one frame: (index, class, flags without the frame-level ones, nearest box index)."""
    out = []
    n_box = len(boxes)
    if n_box:
        D32 = np.sqrt(((sv[:, None, :] - ct32[None, :, :]) ** 2).sum(-1))
        D64 = np.sqrt(((sv[:, None, :] - ct64[None, :, :]) ** 2).sum(-1))
        nearest = D32.argmin(1)
    ins = np.where(inside)[0]
    nearest_of = {int(j): int(nearest[j]) for j in ins} if n_box else {}
    for j in ins:
        j = int(j)
        if not n_box:
            out.append((j, "detection", {f: False for f in FLAGS}, None))
            continue
        k = nearest_of[j]
        cls = classify(float(D32[j, k]), float(D64[j].min()))
        b = boxes[k]
        h_box = float(b["bbox"][3] - b["bbox"][1])
        fl = {"C1": third[j] == "near",
              "C2_box": box_at_edge(b["bbox"], W, H), "C2_foot_truth": foot_near_edge(px[j], W, H),
              "C3a_iou": any(iou(b["bbox"], boxes[m]["bbox"]) >= IOU_OVERLAP for m in range(n_box) if m != k),
              "C3a_shared": any(nearest_of[o] == k for o in nearest_of if o != j),
              "C3b": bool(crowd[j] < CROWD_FT),
              "C4_height_truth": heights is not None and heights[j] > 0 and
              not (HEIGHT_LO <= h_box / heights[j] <= HEIGHT_HI),
              "C4_foot": float(np.hypot(*(np.asarray(b["foot_stab"]) - np.asarray(b["foot_raw"])))) > FOOT_MOVE_PX,
              "nearest_is_other_1to1": one_to_one.get(k) is not None and one_to_one[k] != j}
        out.append((j, cls, fl, k))
    return out


def combine(fl: dict) -> dict:
    """Condition columns from clause columns."""
    c = dict(fl)
    c["C2"] = fl["C2_box"] or fl["C2_foot_truth"]
    c["C3a"] = fl["C3a_iou"] or fl["C3a_shared"]
    c["C3"] = c["C3a"] or fl["C3b"]
    c["C4"] = fl["C4_height_truth"] or fl["C4_foot"]
    c["C5"] = fl["C5a"] or fl["C5b"]
    return c


# ── pure: measures ─────────────────────────────────────────────────────────────────────────────────────────────────
def window_sums(rows: list, names: list) -> tuple:
    """rows: (window index, class, {name: bool}) -> (n_windows x 4) totals [hit, placement, detection, in-frame] and
    {name: (n_windows x 4) with-condition counts}."""
    nw = 1 + max((r[0] for r in rows), default=-1)
    tot = np.zeros((nw, 4))
    withc = {n: np.zeros((nw, 4)) for n in names}
    for w, cls, fl in rows:
        c = CLS[cls]
        tot[w, c] += 1; tot[w, 3] += 1
        for n in names:
            if fl[n]:
                withc[n][w, c] += 1; withc[n][w, 3] += 1
    return tot, withc


def stats(tot: np.ndarray, wc: np.ndarray) -> dict:
    """From summed counts (length 4: hit, placement, detection, in-frame)."""
    h, p, d, n = tot
    hw, pw, dw, nw = wc
    sp = pw / p if p else np.nan
    sh = hw / h if h else np.nan
    rate_w = pw / nw if nw else np.nan
    rate_wo = (p - pw) / (n - nw) if n - nw else np.nan
    rr = rate_w / rate_wo if rate_wo else np.nan
    return {"share_placement": sp, "share_hit": sh, "share_detection": dw / d if d else np.nan,
            "enrichment": sp / sh if sh else np.nan, "rate_with": rate_w, "rate_without": rate_wo, "rate_ratio": rr,
            "attributable_fraction": sp * (1 - 1 / rr) if rr and np.isfinite(rr) and rr != 0 else np.nan}


def bootstrap(tot_by_game: list, wc_by_game: list, n: int = BOOT_N, seed: int = SEED) -> dict:
    """Windows resampled whole within each game; summed over games -> per statistic [2.5, 97.5] percentiles."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        t, c = np.zeros(4), np.zeros(4)
        for tot, wc in zip(tot_by_game, wc_by_game):
            idx = rng.integers(0, len(tot), len(tot))
            t += tot[idx].sum(0); c += wc[idx].sum(0)
        draws.append(stats(t, c))
    return {k: [float(np.nanpercentile([d[k] for d in draws], 2.5)), float(np.nanpercentile([d[k] for d in draws], 97.5))]
            for k in draws[0]}


def exclusive_bounds(rows: list, conds=SIX) -> dict:
    """Over all orders of conds, each condition's smallest and largest share of placement misses counted under the
    first condition each carries (exhaustive over the len(conds)! orders)."""
    pl = [fl for _, cls, fl in rows if cls == "placement"]
    if not pl:
        return {c: [None, None] for c in conds}
    lo, hi = {c: 1.0 for c in conds}, {c: 0.0 for c in conds}
    for order in itertools.permutations(conds):
        counts = {c: 0 for c in conds}
        for fl in pl:
            for c in order:
                if fl[c]:
                    counts[c] += 1
                    break
        for c in conds:
            s = counts[c] / len(pl)
            lo[c], hi[c] = min(lo[c], s), max(hi[c], s)
    return {c: [lo[c], hi[c]] for c in conds}


def carries(ci_enrichment, share: float) -> bool:
    return ci_enrichment[0] > CARRY_RATIO and share >= CARRY_SHARE


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def collect(game: str) -> dict:
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX, near_side
    from sportvu.sync import MIRRORS, apply_mirror
    from sportvu.truth import _match
    from tools import p8_frozen as F
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_e1_postmortem import foot_and_height
    from tools.p8_e1_score import window_stats
    from tools.p8_placement import depth_basis
    from tools.p8_trackreset import reset_like
    if game not in MEASURED + GATE_ONLY:
        raise SystemExit(f"{WHO}: {game} is not in this item")
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    rec_p = r["ref"] / "rebaseline" / f"score__{game}__current.json"
    score = json.loads(rec_p.read_text())
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
    # percentiles over all of the game's truth rows (never split by outcome)
    lab = [json.loads(x) for s in stems for x in (P_t["labels"] / game / f"{s}.jsonl").read_text().splitlines() if x.strip()]
    line_p = float(np.percentile([x["line_matches"] for x in lab if x.get("line_matches") is not None], LINE_PCT))
    resid_p = float(np.percentile([x["feet_resid_ft"] for x in lab if x.get("feet_resid_ft") is not None], RESID_PCT))
    gate, rows, depth = {}, [], []
    for wi, w in enumerate(wins):
        s = w["window"]
        # the cache file is not compared: D202's score step completed missing cache entries after hashing it; the
        # count gate guards what the scorer reads from it
        for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                       ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
            want = (score.get("hashes") or {}).get(s, {}).get(key)
            if want is not None and RB.sha256(p) != want:
                raise SystemExit(f"{WHO}: {s} {key} differs from D202's record")
        full = S.score_window(P_t, game, w, comp, "camera", mom, mirror, refs, events)
        if "skipped" in full[0]:
            raise SystemExit(f"{WHO}: {s} skipped: {full[0]['skipped']}")
        st = window_stats(full[0])
        obs = {(int(o["f"]), int(o["pid"])): o for o in full[1]}
        scored = {f for f, _ in obs}
        side = json.loads((comp / f"{s}_frames.json").read_text())
        srow = {int(x["frame"]): x for x in side["frames"]}
        tm = json.loads((P_t["label_build"] / f"{s}_timemap.json").read_text())
        stride = int(tm["stride"])
        cap = cv2.VideoCapture(str(P_t["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        lp = comp / f"{s}_resetfix.json"
        logged = [int(e["frame"]) for e in (json.loads(lp.read_text()).get("resets") or [])] if lp.exists() else []
        frames_sorted = sorted(srow)
        reset_frames, prev = [], None
        for f in frames_sorted:
            ids = {int(b["tid"]) for b in srow[f]["boxes"]}
            if prev is not None and reset_like(prev, ids):
                reset_frames.append(f)
            prev = ids
        truth, _ = S.truth_rows(P_t, game, w, "camera", mom.index, mirror)
        c = {"in_frame": 0, "missed": 0, "far": 0}
        for row in truth:
            f = int(row["frame"])
            x = srow.get(f)
            if x is None or f not in scored:
                continue
            q, i = row["q"], row["moment"]
            dq = mom.index.q[q]
            sv = apply_mirror(dq["xy"][i], fx, fy).astype(np.float64)
            pid = dq["pid"][i]
            Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
            px = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Ht)).reshape(-1, 2)
            m = FRAME_MARGIN_PX
            inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
            boxes = x["boxes"]
            feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 1, 2)
            ct32 = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2) if len(boxes) else np.zeros((0, 2), np.float32)
            ct64 = ct32.astype(np.float64)
            nh = near_side(Ht)
            third = [S.third_of(sv[j][1], nh) for j in range(len(sv))]
            crowd = S.crowding(sv)
            _, heights = foot_and_height(sv, Ht, W, H)
            o2o = {bi: j for bi, j, _ in _match(ct32.astype(np.float64), sv, MISS_FT)} if len(boxes) else {}
            res = frame_conditions(sv, inside, px, ct32.astype(np.float64), ct64, boxes, crowd, heights, third, W, H, o2o)
            ground = np.asarray(row["camera_C"][:2], float) if row.get("camera_C") is not None else None
            for j, cls, fl, k in res:
                c["in_frame"] += 1; c["missed"] += cls != "hit"; c["far"] += cls == "detection"
                o = obs.get((f, int(pid[j])))
                fl = {**fl, "C5a": recent(f, logged, RESET_HORIZON * stride), "C5b": recent(f, reset_frames, RESET_HORIZON * stride),
                      "C6": row.get("line_matches") is not None and row["line_matches"] <= line_p,
                      "D": row.get("feet_resid_ft") is not None and row["feet_resid_ft"] >= resid_p,
                      "since_cut_lt_1s": o is not None and o["since_cut_s"] < SINCE_CUT_S}
                rows.append([wi, cls, {k2: bool(v) for k2, v in fl.items()}, third[j]])
                if cls == "placement" and ground is not None and k is not None:
                    u, _ = depth_basis(ground, sv[j])
                    depth.append([third[j], float((ct64[k] - sv[j]) @ u)])
        want = score["windows"][s]
        mine = {"in_frame": c["in_frame"], "missed": c["missed"], "far": c["far"], "nr_boxes": st["nr_boxes"]}
        gate[s] = {"mine": mine, "d202": {k: want.get(k) for k in mine},
                   "scorer_in_frame": st["in_frame"], "scorer_missed": st["missed"]}
        if any(mine[k] != want.get(k) for k in mine) or st["in_frame"] != mine["in_frame"] or st["missed"] != mine["missed"]:
            raise SystemExit(f"{WHO}: count gate failed on {s}: {gate[s]}")
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    out = {"game": game, "design": "docs/research/P8_placecause.md version 2", "score_record": str(rec_p.relative_to(r["ref"])),
           "run_id": score.get("run_id"), "windows": len(stems), "gate": {"passes": True, "windows": len(gate)},
           "percentiles": {"line_matches_p20": line_p, "feet_resid_ft_p80": resid_p},
           "manifest_unchanged": after == watched, "seconds": round(time.time() - t0, 1)}
    if game in MEASURED:
        out["windows_list"] = stems
        out["rows"] = rows
        out["placement_depth"] = depth
    else:
        out["note"] = "count gate only; no condition is computed, so this game stays unspent (design section 4)"
    d = r["ref"] / "placecause"
    d.mkdir(exist_ok=True)
    atomic_write(d / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "windows", "gate", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


# ── report (system python, no lock) ────────────────────────────────────────────────────────────────────────────────
def summarize(per_game: dict) -> dict:
    names = list(FLAGS) + ["C2", "C3a", "C3", "C4", "C5"]
    rows_g = {g: [(r[0], r[1], combine(r[2])) for r in d["rows"]] for g, d in per_game.items()}
    out = {}
    sets = {**{g: [g] for g in per_game}, "pooled": list(per_game)}
    for name, gs in sets.items():
        tots, wcs = {}, {}
        for g in gs:
            tots[g], wcs[g] = window_sums(rows_g[g], names)
        tot = sum(t.sum(0) for t in tots.values())
        res = {}
        for n in names:
            pt = stats(tot, sum(wcs[g][n].sum(0) for g in gs))
            ci = bootstrap([tots[g] for g in gs], [wcs[g][n] for g in gs])
            res[n] = {k: {"value": pt[k], "interval": ci[k]} for k in pt}
        allrows = [r for g in gs for r in rows_g[g]]
        union = lambda cls: np.mean([any(fl[c] for c in SIX) for _, k, fl in allrows if k == cls]) if allrows else None
        out[name] = {"counts": {"hit": int(tot[0]), "placement": int(tot[1]), "detection": int(tot[2]), "in_frame": int(tot[3])},
                     "conditions": res, "union_coverage": {"placement": union("placement"), "hit": union("hit")},
                     "exclusive_bounds_720": exclusive_bounds(allrows)}
    reading = {}
    for c in RULED:
        per = {g: carries(out[g]["conditions"][c]["enrichment"]["interval"], out[g]["conditions"][c]["share_placement"]["value"])
               for g in per_game}
        reading[c] = {"carries_on": [g for g, v in per.items() if v], "carries": all(per.values())}
    verdict = "SCATTERED" if not any(v["carries"] for v in reading.values()) else \
        "carried by " + ", ".join(c for c, v in reading.items() if v["carries"])
    return {"by_set": out, "reading": reading, "verdict": verdict}


def clean(x):
    """JSON-safe: numpy numbers to Python, NaN and infinities to None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def report() -> dict:
    from tools import p8_rebaseline as RB
    r = RB.roots()
    per_game = {g: json.loads((r["ref"] / "placecause" / f"{g}.json").read_text()) for g in MEASURED}
    gates = {g: json.loads((r["ref"] / "placecause" / f"{g}.json").read_text())["gate"] for g in MEASURED + GATE_ONLY}
    s = summarize(per_game)
    depth = {}
    for g, d in per_game.items():
        for third in ("near", "middle", "far"):
            v = [x for t, x in d["placement_depth"] if t == third]
            depth.setdefault(g, {})[third] = {"n": len(v), "median_ft": float(np.median(v)) if v else None}
    c6_third = {}
    for g, d in per_game.items():
        for third in ("near", "middle", "far"):
            pl = [r for r in d["rows"] if r[1] == "placement" and r[3] == third]
            hit = [r for r in d["rows"] if r[1] == "hit" and r[3] == third]
            c6_third.setdefault(g, {})[third] = {"share_placement": float(np.mean([r[2]["C6"] for r in pl])) if pl else None,
                                                 "share_hit": float(np.mean([r[2]["C6"] for r in hit])) if hit else None}
    out = {"what": "docs/research/P8_placecause.md version 2; report only; nothing fitted",
           "limits": "truth frames selected by fits on the same r23 label build's boxes (rates lean toward r23); "
                     "multipool v1 is a better fit, not a truth shown better; build and truth error not separable; "
                     "frames the truth rejected are not seen",
           "gates": gates, **s, "c1_depth_by_third_description": depth, "c6_by_third": c6_third,
           "hou_cle_2016": "count gate only; no condition computed (unspent)"}
    text = json.dumps(clean(out), indent=1)
    atomic_write(REPO / "reports" / "research" / "p8_placecause.json", text)
    print(json.dumps({"verdict": out["verdict"], "reading": out["reading"]}, indent=1))
    return out


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

"""tools/p8_ghostsrc.py: where the reset fix's ghosts come from (docs/research/P8_ghostsrc.md). Stored files only; no
model runs, no lock.

    python3 -m tools.p8_ghostsrc run      -> reports/research/p8_ghostsrc.json
    python3 -m tools.p8_ghostsrc pairs    -> reports/research/p8_ghostsrc_pairs.json   (the pairing check)

Population (the one the reset-fix forecasts used, docs/research/P8_resetfix.md sections 12 to 19): on the scored
frames of the registered score files (the MPS-truth universe; tools.p8_footbias.load_frames), the r23 corpus player
boxes at 0.40 ("ours") whose bottom centre, sent through H_truth, lies more than 3.0 ft from all 10 SportVU players,
on frames from the r23 build's first reset-like frame on (tools.p8_resetfix.first_reset_index). Two comparison
populations, report only: the same ghosts before the first reset, and the sidecar's non-referee ghosts after it.

Each ghost box gets every flag below; its class is the first that applies, in this order:
  A  not a standard live view: the frame's camera focal length (truth camera_f) is in the game's top decile of scored
     frames (a1, zoom); or its gate v2 wide score is in the game's bottom decile of scored frames (a2, marginal
     wide); or a cut (cache corr below the frozen cut_hist_corr 0.6) lies within the 1.0 s before it (a3)
  B  off court under the build's H: the box's bottom centre through the sidecar row's H (accepted rows; else
     H_truth, counted apart) lies outside the court by more than 2 ft (x outside -2 to 96, y outside -2 to 52); split
     near (within 8 ft of the court: sideline, bench) and far (crowd)
  C  referee-like: an r23 corpus referee box (score at least 0.25) or a scorecard cache referee detection overlaps it
     at IoU 0.3 or more
  D  on court, unmatched: split near (3 to 6 ft from the nearest player) and far (more than 6 ft)
Also per class: the share of the game's after-reset ghosts, and what a filter on that class would remove among the
r23 boxes that are not ghosts (the collateral), after the first reset.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_ghostsrc"
GAMES = ("gsw_cha_2016", "gsw_bkn_2015", "gsw_sac_2015")     # gsw_cha_2016 first, as the lead asked
GHOST_FT, NEAR_FT = 3.0, 6.0
COURT_X, COURT_Y, COURT_MARGIN, OFF_NEAR_FT = 94.0, 50.0, 2.0, 8.0
ZOOM_Q, WIDE_Q, CUT_S = 90, 10, 1.0
REF_SCORE, REF_IOU = 0.25, 0.3
CLASSES = ("A", "B", "C", "D")
LEADS_TO = {
    "A": "a frame filter arm (zoom, marginal wide score or time since a cut), its own registered arm",
    "B": "a court-region filter arm (drop boxes whose foot lies off court under the build's H), its own registered "
         "arm, possibly combined with the reset fix",
    "C": "a referee-handling arm (the referee class or the scorer's IoU rule), its own registered arm",
    "D": "no filter arm without a further check: near ones bear on foot placement, far ones on detector false "
         "positives on court or truth and sync error"}
LEAD_SHARE = 0.30          # a class holding this share of after-reset ghosts on gsw_cha_2016 or pooled is named next


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def off_court(pt) -> str | None:
    """None on court (within COURT_MARGIN), else "near" (within OFF_NEAR_FT of the court) or "far"."""
    x, y = float(pt[0]), float(pt[1])
    dx = max(-COURT_MARGIN - x, x - (COURT_X + COURT_MARGIN), 0.0)
    dy = max(-COURT_MARGIN - y, y - (COURT_Y + COURT_MARGIN), 0.0)
    if dx == 0.0 and dy == 0.0:
        return None
    return "near" if max(dx, dy) + COURT_MARGIN <= OFF_NEAR_FT else "far"


def cut_before(frame: int, cuts: list, fps: float, seconds: float = CUT_S) -> bool:
    return any(0 <= frame - c <= seconds * fps for c in cuts)


def classify(flags: dict) -> str:
    """flags: {"A": bool, "B": None or "near" or "far", "C": bool, "dist": ft to the nearest player} -> class label."""
    if flags["A"]:
        return "A"
    if flags["B"]:
        return f"B_{flags['B']}"
    if flags["C"]:
        return "C"
    return "D_near" if flags["dist"] <= NEAR_FT else "D_far"


def top(label: str) -> str:
    return label.split("_")[0]


def summarize(rows: list) -> dict:
    """rows: [{"cls", "a1", "a2", "a3", "b", "c", "after", "ghost", "src"}] -> counts and shares."""
    out = {}
    for pop, sel in (("ours_after", lambda r: r["src"] == "ours" and r["after"] and r["ghost"]),
                     ("ours_before", lambda r: r["src"] == "ours" and not r["after"] and r["ghost"]),
                     ("sidecar_after", lambda r: r["src"] == "sidecar" and r["after"] and r["ghost"])):
        g = [r for r in rows if sel(r)]
        n = len(g)
        labels = {}
        for r in g:
            labels[r["cls"]] = labels.get(r["cls"], 0) + 1
        tops = {c: sum(v for k, v in labels.items() if top(k) == c) for c in CLASSES}
        out[pop] = {"n": n, "labels": labels, "classes": tops,
                    "shares": {c: (tops[c] / n if n else None) for c in CLASSES},
                    "flags": {f: sum(1 for r in g if r[f]) for f in ("a1", "a2", "a3", "c")} |
                             {"b": sum(1 for r in g if r["b"])}}
    # collateral after the reset: r23 boxes that are not ghosts but carry a class's flag
    ok = [r for r in rows if r["src"] == "ours" and r["after"] and not r["ghost"]]
    out["ours_after_not_ghost"] = {"n": len(ok), "with_flag": {
        "A": sum(1 for r in ok if r["a1"] or r["a2"] or r["a3"]), "B": sum(1 for r in ok if r["b"]),
        "C": sum(1 for r in ok if r["c"])}}
    return out


def leads(summary_by_game: dict, pooled: dict) -> list:
    """Classes named as the next arm: at least LEAD_SHARE of after-reset ghosts on gsw_cha_2016 or pooled."""
    named = []
    for c in CLASSES:
        s_cha = (summary_by_game.get("gsw_cha_2016") or {}).get("ours_after", {}).get("shares", {}).get(c)
        s_pool = pooled.get("ours_after", {}).get("shares", {}).get(c)
        if (s_cha or 0) >= LEAD_SHARE or (s_pool or 0) >= LEAD_SHARE:
            named.append({"class": c, "gsw_cha_2016": s_cha, "pooled": s_pool, "leads_to": LEADS_TO[c]})
    return named


# ── run (stored files) ─────────────────────────────────────────────────────────────────────────────────────────────
def game_rows(game: str) -> tuple:
    import cv2
    from tools import p8_teachers as T
    from tools.p8_common import p8_root
    from tools.p8_e1data import corpus_root
    from tools.p8_footbias import load_frames
    from tools.p8_frozen import layout
    from tools.p8_placement import BUILDS, SCORES
    from tools.p8_resetfix import first_reset_index
    from tools.p8_scorecard import CONSTANTS
    wr = p8_root()
    P = layout(wr)
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    ours, refs = {}, {}
    for k in range(len(plan["chunks"])):
        for key, rec in json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
            g, stem = key.split("/", 1)
            if g == game:
                ours[stem] = [p for p in rec[T.OURS] if p[5] == "player" and p[4] >= 0.40]
                refs[stem] = [p for p in rec[T.OURS] if p[5] == "referee" and p[4] >= REF_SCORE]
    sf = json.loads((wr / "e1" / SCORES[game]).read_text())
    first, side_rows, cuts, fps, cache, label = {}, {}, {}, {}, {}, {}
    for w in sf["scored_windows"]:
        side = json.loads((P["builds"] / BUILDS[game] / f"{w}_frames.json").read_text())
        fr = sorted(side["frames"], key=lambda r: int(r["frame"]))
        i0 = first_reset_index(fr)
        first[w] = None if i0 is None else int(fr[i0]["frame"])
        side_rows[w] = {int(r["frame"]): r for r in fr}
        fps[w] = float(side["fps"])
        c = json.loads((P["cache"] / "scorecard" / f"{w}.json").read_text())
        cache[w] = c
        corr = sorted((int(k), v) for k, v in c.get("corr", {}).items())
        cuts[w] = [f for f, v in corr if v is not None and v < CONSTANTS["cut_hist_corr"]]
        lp = P["labels"] / game / f"{w}.jsonl"
        label[w] = {int(json.loads(x)["frame"]): json.loads(x) for x in lp.read_text().splitlines() if x.strip()} \
            if lp.exists() else {}
    frames = load_frames(game)["frames"]
    zoom = [label[f["w"]][int(f["f"])]["camera_f"] for f in frames]
    wide = [cache[f["w"]]["wide"].get(str(int(f["f"]))) for f in frames]
    z_hi = float(np.percentile(zoom, ZOOM_Q))
    w_lo = float(np.percentile([x for x in wide if x is not None], WIDE_Q))
    rows, truth_h_used = [], 0
    for f, zf, wd in zip(frames, zoom, wide):
        w, fi = f["w"], int(f["f"])
        stem = f"{w}_f{fi:06d}"
        if stem not in ours:
            raise SystemExit(f"{WHO}: scored frame {stem} has no corpus prediction")
        after = first[w] is not None and fi >= first[w]
        a1, a2, a3 = zf > z_hi, wd is not None and wd < w_lo, cut_before(fi, cuts[w], fps[w])
        srow = side_rows[w][fi]
        Hb = f["Hb"] if f["Hb"] is not None else f["Ht"]
        sv = f["sv"]
        cref = [d[:4] for d in cache[w].get("referees", {}).get(str(fi), [])]
        boxes = [("ours", b) for b in ours[stem]] + \
                [("sidecar", b["bbox"]) for b, r in zip(srow["boxes"], f["is_ref"]) if not r]
        for src, b in boxes:
            foot = np.array([[(b[0] + b[2]) / 2.0, b[3]]], np.float64).reshape(-1, 1, 2)
            ft = cv2.perspectiveTransform(foot, np.asarray(f["Ht"], np.float64).reshape(3, 3)).reshape(2)
            dist = float(np.hypot(*(sv - ft).T).min())
            fb = cv2.perspectiveTransform(foot, np.asarray(Hb, np.float64).reshape(3, 3)).reshape(2)
            truth_h_used += src == "ours" and f["Hb"] is None and dist > GHOST_FT and after
            c = any(iou(b, rb) >= REF_IOU for rb in [x[:4] for x in refs[stem]] + cref)
            flags = {"A": bool(a1 or a2 or a3), "B": off_court(fb), "C": c, "dist": dist}
            rows.append({"src": src, "after": after, "ghost": dist > GHOST_FT, "cls": classify(flags),
                         "a1": bool(a1), "a2": bool(a2), "a3": bool(a3), "b": flags["B"], "c": c})
    meta = {"zoom_threshold_camera_f": z_hi, "wide_threshold_bottom_decile": w_lo, "frames": len(frames),
            "after_reset_ghosts_on_truth_h": truth_h_used}
    return rows, meta


def cmd_run(args) -> dict:
    from tools.p8_common import REPO, atomic_write
    from tools.p8_placement import check_games, rnd
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(check_games(list(GAMES)), WHO)
    per, allrows = {}, []
    for g in GAMES:
        rows, meta = game_rows(g)
        per[g] = {**summarize(rows), "meta": meta}
        allrows += rows
        s = per[g]["ours_after"]
        print(g, json.dumps(rnd({"n": s["n"], "shares": s["shares"], "labels": s["labels"]})))
    pooled = summarize(allrows)
    res = rnd({"what": "Ghost source of r23 boxes after a reset (docs/research/P8_ghostsrc.md); report only",
               "games": per, "pooled": pooled, "named_next": leads(per, pooled), "leads_to": LEADS_TO})
    atomic_write(REPO / "reports" / "research" / "p8_ghostsrc.json", json.dumps(res, indent=1))
    print(json.dumps({"named_next": res["named_next"]}, indent=1))
    return res


# ── pairing check (docs/research/P8_ghostsrc.md, "Pairing check") ─────────────────────────────────────────────
PAIR_FT, PAIR_FT_TIGHT, MISS_FT = 6.0, 4.5, 2.5


def frame_pairs(feet, sv, inside) -> dict:
    """One frame, one box set: missed players, ghost boxes and one-to-one ghost-missed pairs within PAIR_FT and
    PAIR_FT_TIGHT."""
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    ins = np.asarray(inside, bool)
    out = {"in_frame": int(ins.sum()), "missed": int(ins.sum()), "ghosts": 0, "pairs": 0, "pairs_tight": 0}
    if not len(feet):
        return out
    D = np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1))
    miss = np.where(ins & (D.min(1) > MISS_FT))[0]
    ghost = np.where(D.min(0) > GHOST_FT)[0]
    out["missed"], out["ghosts"] = len(miss), len(ghost)
    if len(miss) and len(ghost):
        from scipy.optimize import linear_sum_assignment
        C = D[np.ix_(miss, ghost)]
        ri, ci = linear_sum_assignment(C)
        d = C[ri, ci]
        out["pairs"], out["pairs_tight"] = int((d <= PAIR_FT).sum()), int((d <= PAIR_FT_TIGHT).sum())
    return out


def stab_feet(ours: list, side_boxes: list, raw_ft, side_ft) -> np.ndarray:
    """r23 feet with each r23 box that matches a sidecar box one to one at IoU 0.5 or more taking that box's foot."""
    out = np.asarray(raw_ft, np.float64).reshape(-1, 2).copy()
    if not len(ours) or not len(side_boxes):
        return out
    from scipy.optimize import linear_sum_assignment
    M = np.array([[iou(a, b) for b in side_boxes] for a in ours])
    ri, ci = linear_sum_assignment(-M)
    sf = np.asarray(side_ft, np.float64).reshape(-1, 2)
    for i, j in zip(ri, ci):
        if M[i, j] >= 0.5:
            out[i] = sf[j]
    return out


def explained(r23: dict, side: dict) -> dict:
    """The full-fix change in net error and the part of it from added double-count pairs."""
    n = r23["in_frame"] or None
    if n is None:
        return {"dE": None, "dE_from_pairs": None, "explained_share": None}
    e = lambda c: (c["missed"] + c["ghosts"]) / c["in_frame"]
    dE = e(r23) - e(side)
    dp = 2 * (r23["pairs"] - side["pairs"]) / n
    return {"E_r23": e(r23), "E_sidecar": e(side), "dE": dE, "dE_from_pairs": dp,
            "explained_share": (dp / dE) if dE > 0 else None}


def cmd_pairs(args) -> dict:
    import cv2
    from tools import p8_teachers as T
    from tools.p8_common import REPO, atomic_write, p8_root
    from tools.p8_e1data import corpus_root
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_frozen import layout
    from tools.p8_placement import BUILDS, SCORES, check_games, rnd
    from tools.p8_resetfix import first_reset_index
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(check_games(list(GAMES)), WHO)
    wr = p8_root()
    P = layout(wr)
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    out = {}
    for game in GAMES:
        ours = {}
        for k in range(len(plan["chunks"])):
            for key, rec in json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
                g, stem = key.split("/", 1)
                if g == game:
                    ours[stem] = [p for p in rec[T.OURS] if p[5] == "player" and p[4] >= 0.40]
        sf = json.loads((wr / "e1" / SCORES[game]).read_text())
        first, srows = {}, {}
        for w in sf["scored_windows"]:
            fr = sorted(json.loads((P["builds"] / BUILDS[game] / f"{w}_frames.json").read_text())["frames"],
                        key=lambda r: int(r["frame"]))
            i0 = first_reset_index(fr)
            first[w] = None if i0 is None else int(fr[i0]["frame"])
            srows[w] = {int(r["frame"]): r for r in fr}
        tot = {(per, s): {"in_frame": 0, "missed": 0, "ghosts": 0, "pairs": 0, "pairs_tight": 0}
               for per in ("after", "before") for s in ("sidecar", "r23_raw", "r23_stab")}
        for f in load_frames(game)["frames"]:
            w, fi = f["w"], int(f["f"])
            per = "after" if first[w] is not None and fi >= first[w] else "before"
            keep = ~np.asarray(f["is_ref"], bool)
            side_ft = scorer_court(f["Ht"], f["px32"])[keep] if len(f["px32"]) else np.zeros((0, 2))
            side_boxes = [b["bbox"] for b, r in zip(srows[w][fi]["boxes"], f["is_ref"]) if not r]
            ob = ours[f"{w}_f{fi:06d}"]
            raw = T.feet_ft(ob, f["Ht"]) if ob else np.zeros((0, 2))
            stab = stab_feet([b[:4] for b in ob], side_boxes, raw, side_ft)
            for s, ft in (("sidecar", side_ft), ("r23_raw", raw), ("r23_stab", stab)):
                c = frame_pairs(ft, f["sv"], f["inside"])
                for k2, v in c.items():
                    tot[(per, s)][k2] += v
        res = {}
        for per in ("after", "before"):
            side = tot[(per, "sidecar")]
            res[per] = {"counts": {s: tot[(per, s)] for s in ("sidecar", "r23_raw", "r23_stab")},
                        "r23_raw": explained(tot[(per, "r23_raw")], side),
                        "r23_stab": explained(tot[(per, "r23_stab")], side)}
        out[game] = res
        print(game, json.dumps(rnd({s: res["after"][s] for s in ("r23_raw", "r23_stab")})))
    cha = out["gsw_cha_2016"]["after"]
    shares = [cha[s]["explained_share"] for s in ("r23_raw", "r23_stab")]
    if all(x is not None and x >= 0.5 for x in shares):
        reading = "mostly placement counted twice"
    elif all(x is not None and x < 0.5 for x in shares):
        reading = "mostly not placement counted twice"
    else:
        reading = "the two foot choices disagree"
    res = rnd({"what": "Double-count pairs (one player missed and its box a ghost) after a reset "
                       "(docs/research/P8_ghostsrc.md, pairing check); report only", "games": out,
               "reading_gsw_cha_2016": reading})
    atomic_write(REPO / "reports" / "research" / "p8_ghostsrc_pairs.json", json.dumps(res, indent=1))
    print(reading)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_ghostsrc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("pairs")
    a = ap.parse_args(argv)
    {"run": cmd_run, "pairs": cmd_pairs}[a.cmd](a)


if __name__ == "__main__":
    main()

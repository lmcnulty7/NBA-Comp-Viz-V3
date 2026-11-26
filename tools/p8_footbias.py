"""tools/p8_footbias.py: placement route, part 2 of the step 1b follow-up (docs/research/P8_footbias.md). Report only:
no model runs, nothing is built, trained or changed.

    python3 -m tools.p8_footbias measure  -> <P8 workspace>/placement/footbias.json and reports/research/p8_footbias.json

Question: is r23's foot offset, measured under the build's own court H (which needs no truth to apply), consistent
enough across games to correct, and would a correction fitted leave one game out lower the registered missed row on
every pool game without raising ghosts?

Inputs: the stored P8-workspace files of step 1 (tools/p8_placement.py): reference-platform r23 sidecars, their
registered score files, the camera-truth rows, the scorecard cache (wide scores, threshold and referee detections),
the time maps, the SportVU moments and the mirror. Nothing under DATA_ROOT, no model, no video decode.

Rules, fixed before any number:
- Games: pool gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 for every measure, fit and rule; hou_cle_2016 report only
  (corrections fitted on the whole pool applied to it; nothing fitted on it or chosen with it). Guards on every game,
  SportVU id and window.
- Universe and scoring: the registered scorer's (tools.p8_scorecard.score_window, truth mode camera). In-frame players
  with the 10 px margin; missed when no foot_stab, sent through the TRUTH camera, lies within 2.5 ft; a box is a ghost
  when its foot through the truth camera is more than 3.0 ft from all 10 SportVU players; referee boxes are those with
  IoU at least 0.5 to a cached referee detection of the frame (the scorer's rule when sidecars carry no class); the
  ghost row is non-referee ghosts over non-referee boxes. Check: per game, in-frame, missed, non-referee ghosts and
  non-referee boxes must equal the registered score file's sums for the CPU build, or the tool stops.
- Offsets under the build's H (the measurement): on frames whose sidecar row is accepted with an H, every box foot
  pixel p goes to the court through the build's per-frame H, q = B(p). The build's camera is
  sportvu.camera.pose_from_h(B); its ground point G_B. For each in-frame player, the nearest q (re-matched under B);
  if it lies within 5 ft (sportvu.bench.MATCH_FT) the player enters the offset population with offset o = q - SportVU,
  depth = o along the unit vector from G_B to q (positive: the foot lands beyond the player, away from the camera),
  lateral = o across it (90 degrees counter-clockwise).
- Cells: the box's own pixel height (y2 - y1: under 150, 150 to 200, 200 to 250, 250 and over) by the third of q
  (tools.p8_scorecard.third_of with sportvu.eval.near_side of B). Report only: distance from G_B to q in four bins at
  the quartiles of the pool's offset population.
- Per cell: the median depth and lateral offset per game and pooled, the game spread (largest minus smallest game
  median), the counts.
- Consistency (fixed now): a cell is consistent when it holds at least 200 offset rows on every pool game, the three
  game medians of depth share the sign of the pooled median, and the game spread is at most half the absolute pooled
  median. The bias is consistent when the consistent cells hold at least 60 percent of the pool's offset population.
- Corrections, each fitted leave one game out on the other two pool games' offset populations and applied to every box
  of the held-out game (frames without an accepted H or a pose keep their feet):
  K1 depth shift by cell: the fit games' pooled median depth of the cell (0 for a cell with fewer than 50 fit rows)
     is removed along the unit vector from G_B to q; the corrected court point q' goes back to the image through the
     inverse of B, p' = B^-1(q'), and p' is scored through the truth camera, exactly as the scorer maps foot_stab.
  K2 vertical pixel offset by box height: for fit rows, the target pixel is the SportVU position sent into the image
     through the inverse of B; the fit games' median of (target y - foot y) / box height per height bin; applied as
     p'_y = p_y + ratio x box height (x unchanged); p' is scored through the truth camera.
- Worth registering a correction arm (fixed now): the bias is consistent, and some correction K1 or K2 has, on every
  pool game held out: net change in missed (all in-frame players, scored through the truth camera) at most -0.010, and
  a non-referee ghost rate change of at most 0 (point estimates).
- Intervals: window bootstrap, 2,000 resamples, seed 0, windows resampled whole, for the net missed and ghost changes.
- Scoring path, explicit: every foot pixel (corrected or not) goes to the truth court frame as the scorer maps foot_stab
  (float32 pixels through the float64 truth H, scorer_court); no correction changes the truth.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, guard, p8_root
from tools.p8_placement import (BUILDS, CHECK, POOL, SCORES, SIDE, THIRDS, BOOT_N, SEED, MISS_FT, NEAR_FT, MARGIN,
                                W_PX, H_PX, check_games, depth_basis, hbin, rnd)

WHO = "tools.p8_footbias"
GHOST_FT = 3.0
HEIGHTS4 = ("<150", "150-200", "200-250", ">=250")
CELL_MIN, FIT_MIN = 200, 50
SPREAD_FRAC, CONSISTENT_SHARE = 0.5, 0.60
NET_BAR, GHOST_BAR = -0.010, 0.0


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def to_court(H, px) -> np.ndarray:
    import cv2
    p = np.asarray(px, np.float64).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)


def scorer_court(Ht, px) -> np.ndarray:
    """Foot pixels to the truth court frame exactly as the scorer maps foot_stab (float32 pixels, float64 H)."""
    import cv2
    p = np.asarray(px, np.float32).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(Ht, np.float64).reshape(3, 3)).reshape(-1, 2).astype(np.float64)


def to_image(H, q) -> np.ndarray:
    return to_court(np.linalg.inv(np.asarray(H, np.float64).reshape(3, 3)), q)


def score_frame(ct, sv10, inside, is_ref) -> dict:
    """The scorer's counts of one frame from truth-frame feet ct (ft): in-frame, missed, non-referee ghosts and boxes."""
    sv = np.asarray(sv10, np.float64).reshape(-1, 2)
    ct = np.asarray(ct, np.float64).reshape(-1, 2)
    inside = np.asarray(inside, bool)
    is_ref = np.asarray(is_ref, bool)
    out = {"in_frame": int(inside.sum()), "missed": int(inside.sum()), "nr_ghosts": 0, "nr_boxes": int((~is_ref).sum())}
    if len(ct):
        D = np.sqrt(((sv[:, None, :] - ct[None, :, :]) ** 2).sum(-1))
        out["missed"] = int((inside & (D.min(1) > MISS_FT)).sum())
        out["nr_ghosts"] = int(((D.min(0) > GHOST_FT) & ~is_ref).sum())
    return out


def cell_consistency(medians: dict, counts: dict, pooled: dict) -> dict:
    """medians/counts: {cell: {game: value}}, pooled: {cell: pooled median} -> {cell: consistent}."""
    out = {}
    for c, m in medians.items():
        n = counts[c]
        ok = all(n.get(g, 0) >= CELL_MIN for g in POOL) and pooled.get(c) is not None and pooled[c] != 0
        if ok:
            vals = [m[g] for g in POOL]
            same = all(np.sign(v) == np.sign(pooled[c]) for v in vals)
            spread = max(vals) - min(vals)
            ok = bool(same and spread <= SPREAD_FRAC * abs(pooled[c]))
        out[c] = bool(ok)
    return out


def worth_arm(consistent: bool, per_corr: dict) -> list:
    """per_corr: {name: {game: {net, ghost}}} -> the corrections that pass on every pool game (when consistent)."""
    if not consistent:
        return []
    return [k for k, gs in per_corr.items()
            if gs and all(v["net"] <= NET_BAR and v["ghost"] <= GHOST_BAR for v in gs.values())]


def median_table(rows: list, key, field: str, min_n: int) -> dict:
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r[field])
    return {k: (float(np.median(v)) if len(v) >= min_n else 0.0) for k, v in out.items()}


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_frames(tag: str) -> dict:
    import cv2
    from sportvu import bench as Bn
    from sportvu.camera import pose_from_h
    from sportvu.eval import near_side
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from sportvu.truth import CLOCK_TOL_S
    from tools.p8_frozen import game_of, layout, mirror_of
    from tools.p8_scorecard import universe
    wr = p8_root()
    P = layout(wr)
    g = game_of(tag)
    guard([tag, g["sportvu"]], WHO)
    fx, fy = MIRRORS[mirror_of(P, tag)]
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    sf = json.loads((wr / "e1" / SCORES[tag]).read_text())
    side = SIDE[tag]
    if BUILDS[tag] not in sf.get("sides", {}).get(side, BUILDS[tag]):
        raise SystemExit(f"{WHO}: {SCORES[tag]} side {side} is not {BUILDS[tag]}")
    wins = list(sf["scored_windows"])
    guard(wins, WHO)
    reg = {k: sum(sf["windows"][w][side][k] for w in wins) for k in ("in_frame", "missed", "nr_ghosts", "nr_boxes")}
    frames = []
    for stem in wins:
        cache = json.loads((P["cache"] / "scorecard" / f"{stem}.json").read_text())
        tm = json.loads((P["label_build"] / f"{stem}_timemap.json").read_text())
        wide = {int(k): v for k, v in cache["wide"].items()}
        refd = {int(k): v for k, v in cache.get("referees", {}).items()}
        lw = {int(f) for f, _, _ in tm["frames"] if wide.get(int(f), 0.0) >= float(cache["wide_threshold"])}
        srow = {int(r["frame"]): r for r in json.loads((P["builds"] / BUILDS[tag] / f"{stem}_frames.json")
                                                       .read_text())["frames"]}
        sidecar_cls = any("cls" in b for r in srow.values() for b in r["boxes"])
        lp = P["labels"] / tag / f"{stem}.jsonl"
        rows = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()] if lp.exists() else []
        for r in rows:
            if r.get("moment") is None:
                r["moment"] = index.at(r["q"], r["clock"], CLOCK_TOL_S)
        rows = [r for r in rows if r["moment"] is not None]
        rows, _ = universe(rows, lw, srow)
        for r in rows:
            f = int(r["frame"])
            d = index.q[int(r["q"])]
            sv = apply_mirror(d["xy"][int(r["moment"])], fx, fy)
            Ht = np.asarray(r["H_truth"], np.float64).reshape(3, 3)
            px32 = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Ht)).reshape(-1, 2)
            inside = (px32[:, 0] > MARGIN) & (px32[:, 0] < W_PX - MARGIN) & (px32[:, 1] > MARGIN) & \
                     (px32[:, 1] < H_PX - MARGIN)
            sr = srow[f]
            boxes = sr["boxes"]
            px = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
            is_ref = [(b.get("cls") in ("referee", Bn.REFEREE_CLASS)) if sidecar_cls else
                      any(Bn._iou(b["bbox"], dd[:4]) >= Bn.REF_IOU for dd in refd.get(f, [])) for b in boxes]
            bh = np.array([float(b["bbox"][3] - b["bbox"][1]) for b in boxes]).reshape(-1)
            Hb = np.asarray(sr["H"], np.float64).reshape(3, 3) if Bn.accepted(sr) and sr.get("H") is not None else None
            Gb, nhb = None, None
            if Hb is not None:
                pose = pose_from_h(Hb, W_PX, H_PX)
                if pose is not None:
                    Gb = np.asarray(pose[2][:2], float)
                    nhb = near_side(Hb)
            frames.append({"w": stem, "f": f, "Ht": Ht, "Hb": Hb, "Gb": Gb, "nhb": nhb,
                           "px": px.astype(np.float64), "px32": px, "bh": bh, "is_ref": np.asarray(is_ref, bool),
                           "sv": np.asarray(sv, np.float64), "inside": inside})
    got = {"in_frame": 0, "missed": 0, "nr_ghosts": 0, "nr_boxes": 0}
    for fr in frames:
        for k, v in score_frame(scorer_court(fr["Ht"], fr["px32"]), fr["sv"], fr["inside"], fr["is_ref"]).items():
            got[k] += v
    if got != reg:
        raise SystemExit(f"{WHO}: {tag} gives {got}, the registered score file {reg}")
    return {"tag": tag, "frames": frames, "registered": reg, "windows": len(wins)}


def offset_rows(game: dict) -> tuple:
    """The offset population under the build's H, and the count of player-frames left out (no accepted H or pose)."""
    from tools.p8_scorecard import third_of
    rows, left = [], 0
    for fr in game["frames"]:
        n_in = int(fr["inside"].sum())
        if fr["Hb"] is None or fr["Gb"] is None or not len(fr["px"]):
            left += n_in
            continue
        q = to_court(fr["Hb"], fr["px"])
        for j in np.where(fr["inside"])[0]:
            sv = fr["sv"][j]
            dist = np.hypot(*(q - sv).T)
            k = int(dist.argmin())
            if dist[k] > NEAR_FT:
                continue
            u, v = depth_basis(fr["Gb"], q[k])
            o = q[k] - sv
            target = to_image(fr["Hb"], [sv])[0]
            rows.append({"w": fr["w"], "hb": hbin(fr["bh"][k]), "third": third_of(q[k][1], fr["nhb"]),
                         "depth": float(o @ u), "lateral": float(o @ v),
                         "cam_ft": float(np.hypot(*(q[k] - fr["Gb"]))),
                         "dy_ratio": float((target[1] - fr["px"][k][1]) / fr["bh"][k]) if fr["bh"][k] > 0 else 0.0})
    return rows, left


def corrected_px(fr: dict, corr: str, table: dict) -> np.ndarray:
    """Foot pixels after a correction (feet of frames without an accepted H or pose are unchanged for K1)."""
    from tools.p8_scorecard import third_of
    px = fr["px"].copy()
    if not len(px):
        return px
    if corr == "K2":
        for i in range(len(px)):
            px[i][1] += table.get(hbin(fr["bh"][i]), 0.0) * fr["bh"][i]
        return px
    if fr["Hb"] is None or fr["Gb"] is None:
        return px
    q = to_court(fr["Hb"], px)
    for i in range(len(q)):
        cell = (hbin(fr["bh"][i]), third_of(q[i][1], fr["nhb"]))
        u, _ = depth_basis(fr["Gb"], q[i])
        q[i] = q[i] - table.get(cell, 0.0) * u
    return to_image(fr["Hb"], q)


def score_correction(game: dict, corr: str, table: dict) -> dict:
    """Baseline against corrected counts, scored through the truth camera, with window intervals of the changes."""
    per_w = {}
    for fr in game["frames"]:
        acc = per_w.setdefault(fr["w"], {k: 0 for k in ("in_frame", "m0", "m1", "g0", "g1", "b0", "b1")})
        base = score_frame(scorer_court(fr["Ht"], fr["px"]), fr["sv"], fr["inside"], fr["is_ref"])
        new = score_frame(scorer_court(fr["Ht"], corrected_px(fr, corr, table)), fr["sv"], fr["inside"], fr["is_ref"])
        acc["in_frame"] += base["in_frame"]
        acc["m0"] += base["missed"]; acc["m1"] += new["missed"]
        acc["g0"] += base["nr_ghosts"]; acc["g1"] += new["nr_ghosts"]
        acc["b0"] += base["nr_boxes"]; acc["b1"] += new["nr_boxes"]
    keys = sorted(per_w)
    A = np.array([[per_w[k][c] for c in ("in_frame", "m0", "m1", "g0", "g1", "b0", "b1")] for k in keys], float)

    def stats(a):
        s = a.sum(0)
        return (s[2] - s[1]) / s[0], s[4] / s[6] - s[3] / s[5]

    net, ghost = stats(A)
    rng = np.random.default_rng(SEED)
    reps = np.array([stats(A[rng.integers(0, len(A), len(A))]) for _ in range(BOOT_N)])
    tot = dict(zip(("in_frame", "m0", "m1", "g0", "g1", "b0", "b1"), A.sum(0).tolist()))
    return {"net": float(net), "ghost": float(ghost),
            "net_interval": [float(np.percentile(reps[:, 0], 2.5)), float(np.percentile(reps[:, 0], 97.5))],
            "ghost_interval": [float(np.percentile(reps[:, 1], 2.5)), float(np.percentile(reps[:, 1], 97.5))],
            "missed_rate_before": tot["m0"] / tot["in_frame"], "missed_rate_after": tot["m1"] / tot["in_frame"],
            "ghost_rate_before": tot["g0"] / tot["b0"], "ghost_rate_after": tot["g1"] / tot["b1"]}


def fit(rows: list) -> dict:
    return {"K1": median_table(rows, lambda r: (r["hb"], r["third"]), "depth", FIT_MIN),
            "K2": median_table(rows, lambda r: r["hb"], "dy_ratio", FIT_MIN)}


def cell_summary(rows_by_game: dict, games) -> dict:
    cells = sorted({(r["hb"], r["third"]) for g in games for r in rows_by_game[g]})
    med, cnt, lat, pooled = {}, {}, {}, {}
    for c in cells:
        key = f"{c[0]}|{c[1]}"
        med[key], cnt[key], lat[key] = {}, {}, {}
        allv = []
        for g in games:
            v = [r["depth"] for r in rows_by_game[g] if (r["hb"], r["third"]) == c]
            lv = [r["lateral"] for r in rows_by_game[g] if (r["hb"], r["third"]) == c]
            cnt[key][g] = len(v)
            med[key][g] = float(np.median(v)) if v else None
            lat[key][g] = float(np.median(lv)) if lv else None
            if g in POOL:
                allv += v
        pooled[key] = float(np.median(allv)) if allv else None
    return {"median_depth": med, "median_lateral": lat, "count": cnt, "pooled_median_depth": pooled}


def cmd_measure(args) -> dict:
    check_games(POOL + CHECK)
    games = {t: load_frames(t) for t in POOL + CHECK}
    rows, left = {}, {}
    for t in games:
        rows[t], left[t] = offset_rows(games[t])
    cs = cell_summary(rows, POOL + CHECK)
    med_pool = {c: {g: v for g, v in m.items() if g in POOL} for c, m in cs["median_depth"].items()}
    cnt_pool = {c: {g: v for g, v in m.items() if g in POOL} for c, m in cs["count"].items()}
    cons = cell_consistency({c: m for c, m in med_pool.items() if all(v is not None for v in m.values())},
                            cnt_pool, cs["pooled_median_depth"])
    pool_n = sum(len(rows[g]) for g in POOL)
    cons_n = sum(sum(cnt_pool[c].values()) for c, ok in cons.items() if ok)
    consistent = pool_n > 0 and cons_n / pool_n >= CONSISTENT_SHARE
    # distance from the build's camera, by the pool's quartiles (report only)
    cam = np.array([r["cam_ft"] for g in POOL for r in rows[g]])
    edges = [float(np.percentile(cam, q)) for q in (25, 50, 75)]

    def dbin(x):
        return int(np.searchsorted(edges, x))
    by_dist = {g: {str(b): {"n": sum(1 for r in rows[g] if dbin(r["cam_ft"]) == b),
                            "median_depth": (float(np.median([r["depth"] for r in rows[g] if dbin(r["cam_ft"]) == b]))
                                             if any(dbin(r["cam_ft"]) == b for r in rows[g]) else None)}
                   for b in range(4)} for g in POOL + CHECK}
    by_third = {g: {t: (float(np.median([r["depth"] for r in rows[g] if r["third"] == t]))
                        if any(r["third"] == t for r in rows[g]) else None) for t in THIRDS} for g in POOL + CHECK}
    by_height = {g: {h: (float(np.median([r["depth"] for r in rows[g] if r["hb"] == h]))
                         if any(r["hb"] == h for r in rows[g]) else None) for h in HEIGHTS4} for g in POOL + CHECK}
    logo, tables = {"K1": {}, "K2": {}}, {}
    for held in POOL:
        tab = fit([r for g in POOL if g != held for r in rows[g]])
        tables[held] = {"K1": {f"{a}|{b}": v for (a, b), v in tab["K1"].items()}, "K2": tab["K2"]}
        for k in ("K1", "K2"):
            logo[k][held] = score_correction(games[held], k, tab[k])
        print(held, json.dumps(rnd({k: {x: logo[k][held][x] for x in ("net", "net_interval", "ghost")}
                                    for k in logo})))
    passing = worth_arm(consistent, logo)
    tab_all = fit([r for g in POOL for r in rows[g]])
    check = {t: {k: score_correction(games[t], k, tab_all[k]) for k in ("K1", "K2")} for t in CHECK}
    out = rnd({"what": "Foot bias under the build's H (docs/research/P8_footbias.md); report only; stored files; "
                       "corrections scored through the truth camera as the registered scorer maps foot_stab",
               "registered_counts": {t: games[t]["registered"] for t in games},
               "offset_population": {t: len(rows[t]) for t in rows}, "left_out_no_H_or_pose": left,
               "median_depth": {"by_third": by_third, "by_height": by_height, "by_camera_distance": by_dist,
                                "distance_edges_ft": edges},
               "cells": cs, "cell_consistent": cons, "consistent_share": cons_n / pool_n if pool_n else None,
               "bias_consistent": consistent, "logo": logo, "fit_tables": tables, "passing": passing,
               "worth_registering_an_arm": bool(passing),
               "check_hou_cle_2016": check,
               "rule": {"cell_min": CELL_MIN, "spread_frac": SPREAD_FRAC, "consistent_share": CONSISTENT_SHARE,
                        "net_bar": NET_BAR, "ghost_bar": GHOST_BAR, "fit_min": FIT_MIN,
                        "boot": {"n": BOOT_N, "seed": SEED, "unit": "window"}}})
    d = p8_root() / "placement"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "footbias.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_footbias.json", text)
    print(json.dumps({"consistent_share": out["consistent_share"], "bias_consistent": consistent,
                      "passing": passing}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_footbias")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()

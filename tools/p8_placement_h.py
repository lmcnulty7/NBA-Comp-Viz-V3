"""tools/p8_placement_h.py: placement route, step 1b (docs/research/P8_placement.md, section "Step 1b"). Report only:
no model runs, nothing is built, trained or changed.

    python3 -m tools.p8_placement_h measure  -> <P8 workspace>/placement/step1b.json and
                                                reports/research/p8_placement_h.json

Question: how much of r23's build-level foot offset is a disagreement between the build's own court homography and
the truth camera, with the detector's foot pixel held fixed?

Inputs and universe: exactly step 1's (tools.p8_placement.load_game: the reference-platform r23 sidecars, the scorer's
universe, the per-game count check against the registered score files), plus each scored frame's build H from the
same sidecar. Report only: the registered builds/r23 sidecar H of the same frame, to check that the CPU build's court
H is the selecting build's. Stored P8-workspace files only; nothing under DATA_ROOT, no model.

Rules, fixed before any number:
- Player-frames: in-frame players (10 px margin) of the pool games gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015 on frames
  whose sidecar row is accepted with an H (sportvu.bench.accepted; the scorer's position rows use the same frames);
  hou_cle_2016 measured and reported apart. Classes as step 1 (nearest foot_stab through the truth camera: hit at most
  2.5 ft, placement miss 2.5 to 5 ft).
- p: the foot_stab pixel of the player's nearest box under the truth camera (the box the missed rule scores). T(p):
  p through the truth H. B(p): p through the build's per-frame H (as the scorer's position rows map it).
- e_T = T(p) - SportVU (step 1's offset, the missed rule). d_H = B(p) - T(p), the H disagreement at the foot. The
  contribution of the disagreement to the scored offset is -d_H (if the build's H were right, the truth would have
  moved the foot by -d_H); the residual after removing it is r = e_T + d_H = B(p) - SportVU, the same foot pixel's
  offset under the build's H.
- Depth: components along the court-plane unit vector from the truth camera's ground point to the SportVU position
  (positive away from the camera), as step 1.
- Per game and third, for placement misses and for hits: n; median depth of e_T, of -d_H and of r; the share with
  |r| at most 2.5 ft (placement misses "moved in"; hits "kept"); the Pearson correlation of the depth of e_T with the
  depth of -d_H and the least-squares slope of the first on the second (1 = the disagreement accounts for the depth
  offset one for one); median |d_H|.
- Null: d_H permuted among the game's placement misses of the same third (seed 0), the moved-in share recomputed.
  Excess = moved-in share minus the null share: what the pairing explains beyond adding any vector of that size.
- Report only: the share of placement misses with any box within 2.5 ft under the build's H (all boxes re-matched);
  the missed rate of all in-frame players under the build's H (same box, and re-matched).
- Interval: window bootstrap (2,000, seed 0, windows resampled whole) of the moved-in share and of the excess per game.
- Reading rule, fixed now, on the three pool games:
  (i) GEOMETRY if, on every pool game, the moved-in share is at least 0.50 and the excess at least 0.25;
  (ii) FOOT PIXEL if, on every pool game, the excess is at most 0.10;
  (iii) MIXED otherwise: some game with an excess between 0.10 and 0.25, or games that fall on different sides, or a
      moved-in share of 0.50 or more without an excess of 0.25. The doc then reports the split per game and third.
- It cannot separate build error from truth error: the truth camera is initialised from the selecting build's H
  (sportvu/camera.py lines 169 to 174, 206 to 208) and fitted to that build's foot_stab pixels paired with SportVU
  (lines 120 to 160), so d_H is how far the truth fit moved from its starting H, and an error both share is invisible.
- Correction (2026-11-26, ml-review): the truth started from the V3 label build's H (camera.py line 169, H_v3), not
  from B, the r23 court's H; so d_H = (r23 court minus V3 court) + (V3 start minus fitted truth). The point null is a
  single permutation (seed 0); each bootstrap resample draws its own.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_placement import (CHECK, POOL, THIRDS, BOOT_N, SEED, MISS_FT, NEAR_FT, check_games, depth_basis, klass,
                                load_game, nearest_foot, rnd)

WHO = "tools.p8_placement_h"
GEO_SHARE, GEO_EXCESS, FOOT_EXCESS = 0.50, 0.25, 0.10


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def to_court(H, px) -> np.ndarray:
    import cv2
    p = np.asarray(px, np.float64).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)


def corr_slope(x, y) -> tuple:
    """(Pearson correlation of y with x, least-squares slope of y on x with an intercept); None when undefined."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None, None
    c = float(np.corrcoef(x, y)[0, 1])
    s = float(np.polyfit(x, y, 1)[0])
    return c, s


def moved_in(e_t, d_h) -> np.ndarray:
    """|e_T + d_H| at most MISS_FT, per row."""
    r = np.asarray(e_t, float).reshape(-1, 2) + np.asarray(d_h, float).reshape(-1, 2)
    return np.hypot(r[:, 0], r[:, 1]) <= MISS_FT


def permuted_share(e_t, d_h, groups, seed: int = SEED) -> float:
    """Moved-in share with d_H permuted within each group (same game and third)."""
    e_t = np.asarray(e_t, float).reshape(-1, 2)
    d_h = np.asarray(d_h, float).reshape(-1, 2)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    perm = d_h.copy()
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        perm[ix] = d_h[rng.permutation(ix)]
    return float(moved_in(e_t, perm).mean()) if len(e_t) else 0.0


def read_rule(per_game: dict) -> str:
    """per_game: {game: {share, excess}} -> GEOMETRY, FOOT PIXEL or MIXED."""
    if all(v["share"] >= GEO_SHARE and v["excess"] >= GEO_EXCESS for v in per_game.values()):
        return "GEOMETRY"
    if all(v["excess"] <= FOOT_EXCESS for v in per_game.values()):
        return "FOOT PIXEL"
    return "MIXED"


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def rows_of(tag: str) -> tuple:
    from sportvu.bench import accepted
    from tools.p8_frozen import layout
    from tools.p8_placement import BUILDS
    game = load_game(tag)                     # step 1's universe and count check
    P = layout(p8_root())
    side, reg = {}, {}
    same_h = {"frames": 0, "equal": 0, "missing": 0}
    rows, skipped = [], 0
    for fr in game["frames"]:
        if fr["w"] not in side:
            side = {fr["w"]: {int(r["frame"]): r for r in json.loads(
                (P["builds"] / BUILDS[tag] / f"{fr['w']}_frames.json").read_text())["frames"]}}
            rp = P["builds"] / "r23" / f"{fr['w']}_frames.json"
            reg = {fr["w"]: {int(r["frame"]): r for r in json.loads(rp.read_text())["frames"]} if rp.exists() else {}}
        sr = side[fr["w"]][fr["f"]]
        rr = reg[fr["w"]].get(fr["f"])
        same_h["frames"] += 1
        if rr is None or rr.get("H") is None:
            same_h["missing"] += 1
        else:
            same_h["equal"] += bool(sr.get("H") is not None and np.allclose(rr["H"], sr["H"], rtol=0, atol=1e-9))
        if not accepted(sr) or sr.get("H") is None or fr["ground"] is None or not len(fr["feet"]):
            skipped += len(fr["players"])
            continue
        px = np.array([b["foot_stab"] for b in sr["boxes"]], np.float64).reshape(-1, 2)
        feet_b = to_court(sr["H"], px)
        for p in fr["players"]:
            k, d = nearest_foot([p["sv"]], fr["feet"])
            k, d = int(k[0]), float(d[0])
            e_t = fr["feet"][k] - p["sv"]
            d_h = feet_b[k] - fr["feet"][k]
            u, _ = depth_basis(fr["ground"], p["sv"])
            _, d_b_any = nearest_foot([p["sv"]], feet_b)
            rows.append({"w": fr["w"], "cls": klass(d), "third": p["third"], "e_t": e_t, "d_h": d_h,
                         "dep_e": float(e_t @ u), "dep_mdh": float(-d_h @ u), "dep_r": float((e_t + d_h) @ u),
                         "any_b": bool(d_b_any[0] <= MISS_FT)})
    return rows, {"skipped_player_frames_no_accepted_build_H": skipped, "cpu_H_equals_registered_r23_H": same_h,
                  "registered": game["registered"]}


def summarise(rows: list, tag: str) -> dict:
    out = {}
    for cls in ("placement", "hit"):
        rs = [r for r in rows if r["cls"] == cls]
        cell = {}
        for third in THIRDS + ("all",):
            sub = rs if third == "all" else [r for r in rs if r["third"] == third]
            if not sub:
                cell[third] = {"n": 0}
                continue
            e = np.array([r["e_t"] for r in sub]); d = np.array([r["d_h"] for r in sub])
            mi = moved_in(e, d)
            c, s = corr_slope([r["dep_mdh"] for r in sub], [r["dep_e"] for r in sub])
            cell[third] = {"n": len(sub), "median_depth_e_T": float(np.median([r["dep_e"] for r in sub])),
                           "median_depth_minus_d_H": float(np.median([r["dep_mdh"] for r in sub])),
                           "median_depth_r": float(np.median([r["dep_r"] for r in sub])),
                           "median_abs_d_H": float(np.median(np.hypot(d[:, 0], d[:, 1]))),
                           "share_within_2_5_after_removing_d_H": float(mi.mean()),
                           "corr_depth": c, "slope_depth": s,
                           "report_only_share_any_box_within_2_5_under_build_H": float(np.mean([r["any_b"] for r in sub]))}
        out[cls] = cell
    pl = [r for r in rows if r["cls"] == "placement"]
    e = np.array([r["e_t"] for r in pl]).reshape(-1, 2); d = np.array([r["d_h"] for r in pl]).reshape(-1, 2)
    grp = [r["third"] for r in pl]
    share = float(moved_in(e, d).mean()) if len(pl) else 0.0
    null = permuted_share(e, d, grp)
    # window bootstrap of the share and of the excess (null recomputed in each resample, one permutation each)
    byw = {}
    for i, r in enumerate(pl):
        byw.setdefault(r["w"], []).append(i)
    keys = sorted(byw)
    rng = np.random.default_rng(SEED)
    sh, ex = [], []
    for b in range(BOOT_N):
        ix = np.concatenate([byw[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        s_b = float(moved_in(e[ix], d[ix]).mean())
        n_b = permuted_share(e[ix], d[ix], [grp[i] for i in ix], seed=b + 1)
        sh.append(s_b); ex.append(s_b - n_b)
    out["rule"] = {"share": share, "null_share": null, "excess": share - null,
                   "share_interval": [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))],
                   "excess_interval": [float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))]}
    n = len(rows)
    out["report_only_missed_rate"] = {"truth_H": float(np.mean([r["cls"] != "hit" for r in rows])) if n else None,
                                      "build_H_same_box": float(np.mean([not moved_in([r["e_t"]], [r["d_h"]])[0]
                                                                         for r in rows])) if n else None,
                                      "build_H_rematched": float(np.mean([not r["any_b"] for r in rows])) if n else None,
                                      "n": n}
    return out


def cmd_measure(args) -> dict:
    check_games(POOL + CHECK)
    res, meta = {}, {}
    for t in POOL + CHECK:
        rows, m = rows_of(t)
        res[t] = summarise(rows, t)
        meta[t] = m
        print(t, json.dumps(rnd({"rule": res[t]["rule"], "placement_all": res[t]["placement"]["all"],
                                 "hit_all": res[t]["hit"]["all"], "meta": m})))
    reading = read_rule({g: res[g]["rule"] for g in POOL})
    out = rnd({"what": "Placement step 1b (docs/research/P8_placement.md): the build H against the truth camera at the "
                       "same foot pixel; report only; stored files",
               "games": res, "meta": meta, "reading": reading,
               "rule": {"geometry": f"share >= {GEO_SHARE} and excess >= {GEO_EXCESS} on every pool game",
                        "foot_pixel": f"excess <= {FOOT_EXCESS} on every pool game", "mixed": "otherwise",
                        "miss_ft": MISS_FT, "near_ft": NEAR_FT, "boot": {"n": BOOT_N, "seed": SEED, "unit": "window"}}})
    d = p8_root() / "placement"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "step1b.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_placement_h.json", text)
    print("reading", reading)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_placement_h")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()

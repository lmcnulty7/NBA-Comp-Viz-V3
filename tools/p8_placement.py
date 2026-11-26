"""tools/p8_placement.py: placement route, step 1 (docs/research/P8_placement.md). Report only: no model runs, nothing
is built or trained, no sidecar is changed.

    python3 -m tools.p8_placement measure    -> <P8 workspace>/placement/step1.json and reports/research/p8_placement.json

Question: r23's build-level misses whose nearest stabilised foot lies 2.5 to 5 ft from SportVU ("placement misses")
make up most of its misses. Is that offset systematic (a shift a correction could remove) or scattered (truth, timing
or foot noise at the 2.5 ft gate)?

Inputs, all stored files in the P8 workspace (nothing under DATA_ROOT, no model, no video decode): the reference-
platform r23 sidecars (builds/e1m6_r23_detcpu for gsw_bkn_2015, builds/m9_r23_detcpu__<game> otherwise), their
registered score files (e1/m6__e1__gsw_bkn_2015.json, e1/m9__r23_detcpu__<game>.json: the scored windows and the r23
counts), the camera-truth rows (labels/<game>/<window>.jsonl), the scorecard reference cache (cache/scorecard/<window>
.json: the gate v2 wide scores and threshold), the window time maps (label_build/<window>_timemap.json: live frames),
the SportVU moments (sportvu/) and the game's mirror (reports/sportvu_direction_<game>.json, as
tools.p8_frozen.mirror_of reads it).

Rules, fixed before any number:
- Games: the development pool gsw_bkn_2015, gsw_cha_2016 and gsw_sac_2015 for every measure, fit and threshold.
  hou_cle_2016 is a check: measured and reported apart, corrections fitted on the pool applied to it, never fitted on
  or used to choose anything. tools.p8_common.guard and refuse_cv_test run on every game, SportVU id and window.
- Universe: the scorer's (tools.p8_scorecard.score_window, truth mode camera): truth rows of the scored windows on live
  frames whose cached wide score is at or above the cached threshold and that the sidecar processed; in-frame players
  with the 10 px margin on a 1280 by 720 frame. Check: per game, in-frame players and missed players (nearest
  foot_stab through the camera truth more than 2.5 ft) must equal the registered r23 counts of the score file summed
  over its scored windows, or the tool stops.
- Classes of in-frame players by the nearest foot_stab distance d (court ft through the camera truth, nearest, not one
  to one, as the missed rule): hit d <= 2.5; placement miss 2.5 < d <= 5.0; far miss otherwise (no box within 5 ft).
- Offset of a player: nearest foot minus the SportVU position (ft). Camera-depth direction: the unit vector on the
  court plane from the camera's ground point (camera_C of the truth row, else the pose of sportvu.camera.pose_from_h)
  to the SportVU position; depth = offset along it (positive: the foot lands beyond the player, away from the
  camera); lateral = offset along the depth direction turned 90 degrees counter-clockwise.
- Strata: camera third (tools.p8_scorecard.third_of of the SportVU y with sportvu.eval.near_side); expected height of
  a 6.5 ft player (tools.p8_e1_postmortem.foot_and_height), bins under 150, 150 to 200, 200 to 250, 250 px and over.
- Matched comparison of hits: medians of hits and of placement misses in every third by height cell, and the hits'
  median reweighted to the placement misses' cell mix.
- Timing signature: the player's SportVU velocity by central difference over +-5 moments (+-0.2 s, game clock dt, as
  tools.p8_scorecard.Moments.speed), mirrored like the positions. For hits and placement misses moving at 3 ft/s or
  more: the offset's projection on the velocity direction; its least-squares slope on speed through the origin is a lag
  in seconds (positive: the foot shows where SportVU puts the player later).
- Lag scan: SportVU positions at moment i + k for k = -12 to +12 (0.04 s a step, +-0.48 s), the feet unchanged; the
  missed rate of the in-frame players whose id is present at all 25 moments.
- Corrections, each fitted leave one game out on the other two pool games (fit population: hits and placement
  misses, d <= 5 ft) and applied to every sidecar foot of the held-out game in court ft, then the missed rule
  recomputed with all boxes:
  C1 depth shift by third: subtract the fit population's median depth offset of the foot's third, along the depth
     direction from the camera's ground point to the foot (the third of the foot's own court y, so it needs no truth);
  C2 depth shift by box height: as C1 with bins of the foot's own box height in pixels (under 150, 150 to 200, 200 to
     250, 250 and over) in place of the third;
  C3 two-dimensional shift by third: subtract the median depth and the median lateral offset of the third.
  L  lag: the k minimising the training games' pooled lag-scan missed rate (ties to the smaller |k|), applied to the
     held-out game's SportVU positions.
- Reading rule (fixed now), on the three leave-one-game-out held-out games:
  SYSTEMATIC if some correction C1, C2 or C3 has, on every held-out game: (a) net missed change at most -0.010 (all
     in-frame players of the game); (b) at least 10 percent of the game's placement misses moved within 2.5 ft; and
     (c) the placement misses' median depth offset of the same sign on all three games, each game's window interval
     (2,000 resamples, seed 0, windows resampled whole) excluding zero.
  TIMING if the lag L is non-zero in every fold and lowers the held-out game's lag-scan missed rate by at least 0.010
     on every held-out game. If TIMING holds, the corrections C1 to C3 are fitted and tested again with each fold's
     lag applied to the SportVU positions, and SYSTEMATIC is read from that second pass.
  SCATTERED if neither holds.
- Intervals: window bootstrap, 2,000 resamples, seed 0, windows resampled whole within a game.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_placement"
POOL = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
CHECK = ("hou_cle_2016",)
BUILDS = {"gsw_bkn_2015": "e1m6_r23_detcpu", "gsw_cha_2016": "m9_r23_detcpu__gsw_cha_2016",
          "gsw_sac_2015": "m9_r23_detcpu__gsw_sac_2015", "hou_cle_2016": "m9_r23_detcpu__hou_cle_2016"}
SCORES = {"gsw_bkn_2015": "m6__e1__gsw_bkn_2015.json", "gsw_cha_2016": "m9__r23_detcpu__gsw_cha_2016.json",
          "gsw_sac_2015": "m9__r23_detcpu__gsw_sac_2015.json", "hou_cle_2016": "m9__r23_detcpu__hou_cle_2016.json"}
W_PX, H_PX, MARGIN = 1280, 720, 10
MISS_FT, NEAR_FT = 2.5, 5.0
VEL_K, MIN_SPEED = 5, 3.0
LAGS = tuple(range(-12, 13))
HEIGHT_EDGES = (150.0, 200.0, 250.0)
HEIGHTS = ("<150", "150-200", "200-250", ">=250", "no pose")
THIRDS = ("near", "middle", "far")
NET_BAR, GROSS_BAR, LAG_BAR = -0.010, 0.10, -0.010
BOOT_N, SEED = 2000, 0


def check_games(games) -> list:
    games = list(games)
    bad = [g for g in games if g not in POOL + CHECK]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not a pool or check game")
    refuse_cv_test(games, WHO)
    guard(games, WHO)
    return games


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def hbin(v) -> str:
    if v is None or not np.isfinite(v):
        return "no pose"
    for e, lab in zip(HEIGHT_EDGES, HEIGHTS):
        if v < e:
            return lab
    return HEIGHTS[3]


def depth_basis(ground, p) -> tuple:
    """Unit depth vector from the camera's ground point to p, and its 90 degree counter-clockwise turn."""
    d = np.asarray(p, float) - np.asarray(ground, float)
    n = np.hypot(*d)
    u = d / n if n > 0 else np.array([0.0, 1.0])
    return u, np.array([-u[1], u[0]])


def nearest_foot(sv, feet) -> tuple:
    """Per player: (index of the nearest foot or -1, distance or inf)."""
    sv = np.asarray(sv, float).reshape(-1, 2)
    ft = np.asarray(feet, float).reshape(-1, 2)
    if len(ft) == 0:
        return np.full(len(sv), -1), np.full(len(sv), np.inf)
    D = np.sqrt(((sv[:, None, :] - ft[None, :, :]) ** 2).sum(-1))
    return D.argmin(1), D.min(1)


def klass(d: float) -> str:
    return "hit" if d <= MISS_FT else ("placement" if d <= NEAR_FT else "far")


def shift_feet(feet, ground, keys, table: dict, lateral: dict | None = None) -> np.ndarray:
    """Feet (ft) moved by minus table[key] along each foot's own depth direction (and minus lateral[key] across it)."""
    out = np.asarray(feet, float).reshape(-1, 2).copy()
    for i, k in enumerate(keys):
        u, v = depth_basis(ground, out[i])
        out[i] = out[i] - table.get(k, 0.0) * u - ((lateral or {}).get(k, 0.0)) * v
    return out


def lag_slope(proj, speed) -> float | None:
    """Least-squares slope through the origin of the offset's velocity projection on speed (seconds)."""
    s = np.asarray(speed, float)
    p = np.asarray(proj, float)
    den = float((s * s).sum())
    return float((s * p).sum() / den) if den > 0 else None


def median_or_none(xs):
    return float(np.median(xs)) if len(xs) else None


def boot_median(groups: dict, n: int = BOOT_N, seed: int = SEED) -> list | None:
    """groups: {window: values} -> 95 percent window-bootstrap interval of the median."""
    keys = [k for k in sorted(groups) if len(groups[k])]
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(float(np.median(np.concatenate([np.asarray(groups[keys[i]], float) for i in pick]))))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def choose_lag(rates: dict) -> int:
    """{k: missed rate} -> the k with the lowest rate, ties to the smaller |k| (then the negative one)."""
    return min(rates, key=lambda k: (round(rates[k], 12), abs(k), k))


def read_systematic(per_corr: dict, sign_ok: bool) -> list:
    """per_corr: {name: {game: {net, gross_share}}} -> the corrections that pass (a) and (b) on every game, when (c)
    (sign_ok) holds; empty otherwise."""
    if not sign_ok:
        return []
    return [c for c, games in per_corr.items()
            if games and all(v["net"] <= NET_BAR and v["gross_share"] >= GROSS_BAR for v in games.values())]


def read_timing(per_fold: dict) -> bool:
    """per_fold: {game: {lag, change}} -> TIMING by the rule."""
    return bool(per_fold) and all(v["lag"] != 0 and v["change"] <= LAG_BAR for v in per_fold.values())


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_game(tag: str) -> dict:
    """Every scored frame of the game: truth, sidecar feet and SportVU players, plus the registered count check."""
    import cv2
    from sportvu.camera import pose_from_h
    from sportvu.eval import near_side
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from sportvu.truth import CLOCK_TOL_S
    from tools.p8_e1_postmortem import foot_and_height
    from tools.p8_frozen import game_of, layout, mirror_of
    from tools.p8_scorecard import third_of, universe
    wr = p8_root()
    P = layout(wr)
    g = game_of(tag)
    guard([tag, g["sportvu"]], WHO)
    mirror = mirror_of(P, tag)
    fx, fy = MIRRORS[mirror]
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    sf = json.loads((wr / "e1" / SCORES[tag]).read_text())
    wins = list(sf["scored_windows"])
    guard(wins, WHO)
    reg = {"in_frame": sum(sf["windows"][w]["r23"]["in_frame"] for w in wins),
           "missed": sum(sf["windows"][w]["r23"]["missed"] for w in wins)}
    bd = P["builds"] / BUILDS[tag]
    frames = []
    for stem in wins:
        cache = json.loads((P["cache"] / "scorecard" / f"{stem}.json").read_text())
        tm = json.loads((P["label_build"] / f"{stem}_timemap.json").read_text())
        wide = {int(k): v for k, v in cache["wide"].items()}
        live = {int(f) for f, _, _ in tm["frames"]}
        lw = {f for f in live if wide.get(f, 0.0) >= float(cache["wide_threshold"])}
        srow = {int(r["frame"]): r for r in json.loads((bd / f"{stem}_frames.json").read_text())["frames"]}
        lp = P["labels"] / tag / f"{stem}.jsonl"
        rows = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()] if lp.exists() else []
        for r in rows:
            if r.get("moment") is None:
                r["moment"] = index.at(r["q"], r["clock"], CLOCK_TOL_S)
        rows = [r for r in rows if r["moment"] is not None]
        rows, _ = universe(rows, lw, srow)
        for r in rows:
            q, i = int(r["q"]), int(r["moment"])
            d = index.q[q]
            Ht = np.asarray(r["H_truth"], np.float64).reshape(3, 3)
            sv = apply_mirror(d["xy"][i], fx, fy).astype(np.float64)
            pid = np.asarray(d["pid"][i])
            px = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Ht)).reshape(-1, 2)
            inside = (px[:, 0] > MARGIN) & (px[:, 0] < W_PX - MARGIN) & (px[:, 1] > MARGIN) & (px[:, 1] < H_PX - MARGIN)
            boxes = srow[int(r["frame"])]["boxes"]
            fp = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
            feet = cv2.perspectiveTransform(fp.reshape(-1, 1, 2), Ht).reshape(-1, 2).astype(np.float64) \
                if len(fp) else np.zeros((0, 2))
            bh = [float(b["bbox"][3] - b["bbox"][1]) for b in boxes]
            if r.get("camera_C") is not None:
                ground = np.asarray(r["camera_C"][:2], float)
            else:
                pose = pose_from_h(Ht, W_PX, H_PX)
                ground = None if pose is None else np.asarray(pose[2][:2], float)
            nh = near_side(Ht)
            _, hts = foot_and_height(sv, Ht, W_PX, H_PX)
            # SportVU positions at shifted moments (lag scan) and the velocity, per player id
            lagged = {}
            for k in LAGS:
                j = i + k
                if 0 <= j < len(d["clock"]):
                    lagged[k] = {int(p): apply_mirror(d["xy"][j][n:n + 1], fx, fy)[0].astype(np.float64)
                                 for n, p in enumerate(d["pid"][j])}
            a, b2 = max(0, i - VEL_K), min(len(d["clock"]) - 1, i + VEL_K)
            dt = float(d["clock"][a] - d["clock"][b2])
            pa = {int(p): apply_mirror(d["xy"][a][n:n + 1], fx, fy)[0] for n, p in enumerate(d["pid"][a])}
            pb = {int(p): apply_mirror(d["xy"][b2][n:n + 1], fx, fy)[0] for n, p in enumerate(d["pid"][b2])}
            players = []
            for j in np.where(inside)[0]:
                p_id = int(pid[j])
                vel = (pb[p_id] - pa[p_id]) / dt if dt > 0 and p_id in pa and p_id in pb else None
                lag_xy = [lagged[k][p_id] if k in lagged and p_id in lagged[k] else None for k in LAGS]
                players.append({"pid": p_id, "sv": sv[j], "third": third_of(sv[j][1], nh),
                                "height": hbin(None if hts is None else float(hts[j])),
                                "vel": None if vel is None else np.asarray(vel, float),
                                "lag_xy": lag_xy})
            frames.append({"w": stem, "f": int(r["frame"]), "feet": feet, "box_h": bh, "ground": ground, "nh": nh,
                           "players": players})
    got = {"in_frame": 0, "missed": 0}
    for fr in frames:
        _, dist = nearest_foot([p["sv"] for p in fr["players"]], fr["feet"])
        got["in_frame"] += len(fr["players"])
        got["missed"] += int((dist > MISS_FT).sum())
    if got != reg:
        raise SystemExit(f"{WHO}: {tag} gives {got}, the registered r23 score file {reg}")
    return {"tag": tag, "mirror": mirror, "windows": len(wins), "frames": frames, "registered": reg}


# ── measures ───────────────────────────────────────────────────────────────────────────────────────────────────────
def player_rows(game: dict, sv_key=None) -> list:
    """One row per in-frame player: class, offset components, strata, timing projection. sv_key(k) picks a lagged
    position (None: the scored moment)."""
    rows = []
    for fr in game["frames"]:
        if fr["ground"] is None:
            continue
        for p in fr["players"]:
            sv = p["sv"] if sv_key is None else p["lag_xy"][LAGS.index(sv_key)]
            if sv is None:
                continue
            idx, dist = nearest_foot([sv], fr["feet"])
            row = {"w": fr["w"], "cls": klass(float(dist[0])), "third": p["third"], "height": p["height"],
                   "d": float(dist[0])}
            if idx[0] >= 0:
                e = fr["feet"][idx[0]] - sv
                u, v = depth_basis(fr["ground"], sv)
                row.update(depth=float(e @ u), lateral=float(e @ v))
                if p["vel"] is not None and np.hypot(*p["vel"]) >= MIN_SPEED:
                    sp = float(np.hypot(*p["vel"]))
                    row.update(speed=sp, vproj=float(e @ (p["vel"] / sp)))
            rows.append(row)
    return rows


def offsets_summary(rows: list) -> dict:
    out = {}
    for cls in ("hit", "placement"):
        rs = [r for r in rows if r["cls"] == cls and "depth" in r]
        dep, lat = [r["depth"] for r in rs], [r["lateral"] for r in rs]
        groups = {}
        for r in rs:
            groups.setdefault(r["w"], []).append(r["depth"])
        out[cls] = {"n": len(rs), "depth_median": median_or_none(dep), "depth_median_interval": boot_median(groups),
                    "depth_iqr": [float(np.percentile(dep, 25)), float(np.percentile(dep, 75))] if dep else None,
                    "lateral_median": median_or_none(lat),
                    "abs_lateral_median": median_or_none(np.abs(lat)), "abs_depth_median": median_or_none(np.abs(dep)),
                    "share_depth_beyond": float(np.mean(np.asarray(dep) > 0)) if dep else None,
                    "share_depth_dominant": float(np.mean(np.abs(dep) > np.abs(lat))) if dep else None}
        for field, levels in (("third", THIRDS), ("height", HEIGHTS)):
            out[cls][f"by_{field}"] = {lv: {"n": sum(1 for r in rs if r[field] == lv),
                                            "depth_median": median_or_none([r["depth"] for r in rs if r[field] == lv]),
                                            "lateral_median": median_or_none([r["lateral"] for r in rs
                                                                              if r[field] == lv])}
                                       for lv in levels}
    # matched comparison: hits reweighted to the placement misses' third by height mix
    pl = [r for r in rows if r["cls"] == "placement" and "depth" in r]
    hi = [r for r in rows if r["cls"] == "hit" and "depth" in r]
    cells = {}
    for r in pl:
        cells[(r["third"], r["height"])] = cells.get((r["third"], r["height"]), 0) + 1
    vals, wts = [], []
    for c, n in cells.items():
        h = [r["depth"] for r in hi if (r["third"], r["height"]) == c]
        if h:
            vals.append(float(np.median(h)))
            wts.append(n)
    out["matched"] = {"cells": {f"{a}|{b}": {"placement_n": n,
                                             "placement_depth_median": median_or_none(
                                                 [r["depth"] for r in pl if (r["third"], r["height"]) == (a, b)]),
                                             "hit_n": sum(1 for r in hi if (r["third"], r["height"]) == (a, b)),
                                             "hit_depth_median": median_or_none(
                                                 [r["depth"] for r in hi if (r["third"], r["height"]) == (a, b)])}
                                for (a, b), n in sorted(cells.items())},
                      "hit_depth_median_reweighted": float(np.average(vals, weights=wts)) if vals else None}
    tm = [r for r in rows if r["cls"] in ("hit", "placement") and "speed" in r]
    groups = {}
    for r in tm:
        groups.setdefault(r["w"], []).append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(BOOT_N):
        pick = [r for i in rng.integers(0, len(keys), len(keys)) for r in groups[keys[i]]] if keys else []
        s = lag_slope([r["vproj"] for r in pick], [r["speed"] for r in pick])
        if s is not None:
            reps.append(s)
    out["timing"] = {"n": len(tm), "slope_s": lag_slope([r["vproj"] for r in tm], [r["speed"] for r in tm]),
                     "slope_interval": [round(float(np.percentile(reps, 2.5)), 4),
                                        round(float(np.percentile(reps, 97.5)), 4)] if reps else None,
                     "placement_share_ahead": float(np.mean([r["vproj"] > 0 for r in tm if r["cls"] == "placement"]))
                     if any(r["cls"] == "placement" for r in tm) else None}
    return out


def lag_scan(game: dict) -> dict:
    """{k: (missed, n)} over in-frame players present at every lag."""
    out = {k: [0, 0] for k in LAGS}
    for fr in game["frames"]:
        for p in fr["players"]:
            if any(x is None for x in p["lag_xy"]):
                continue
            for k, xy in zip(LAGS, p["lag_xy"]):
                _, dist = nearest_foot([xy], fr["feet"])
                out[k][0] += int(dist[0] > MISS_FT)
                out[k][1] += 1
    return out


def fit_tables(rows: list, key: str) -> tuple:
    """Median depth and lateral offset of the fit population (hits and placement misses) per key value."""
    pop = [r for r in rows if r["cls"] in ("hit", "placement") and "depth" in r]
    levels = sorted({r[key] for r in pop})
    dep = {lv: float(np.median([r["depth"] for r in pop if r[key] == lv])) for lv in levels}
    lat = {lv: float(np.median([r["lateral"] for r in pop if r[key] == lv])) for lv in levels}
    return dep, lat


def box_rows(game: dict) -> list:
    """Fit rows keyed by the nearest box's own pixel-height bin (for C2)."""
    rows = []
    for fr in game["frames"]:
        if fr["ground"] is None:
            continue
        for p in fr["players"]:
            idx, dist = nearest_foot([p["sv"]], fr["feet"])
            if idx[0] < 0 or dist[0] > NEAR_FT:
                continue
            e = fr["feet"][idx[0]] - p["sv"]
            u, v = depth_basis(fr["ground"], p["sv"])
            rows.append({"cls": klass(float(dist[0])), "boxh": hbin(fr["box_h"][idx[0]]), "depth": float(e @ u),
                         "lateral": float(e @ v)})
    return rows


def apply_correction(game: dict, corr: str, tables: dict, lag: int = 0) -> dict:
    """Missed counts of the game after a correction; per window for intervals."""
    from tools.p8_scorecard import third_of
    per_w = {}
    for fr in game["frames"]:
        acc = per_w.setdefault(fr["w"], {"n": 0, "missed0": 0, "missed1": 0, "placement0": 0, "moved_in": 0,
                                         "pushed_out": 0})
        feet = fr["feet"]
        if fr["ground"] is not None and len(feet) and corr != "none":
            if corr in ("C1", "C3"):
                keys = [third_of(y, fr["nh"]) for y in feet[:, 1]]
            else:
                keys = [hbin(h) for h in fr["box_h"]]
            new = shift_feet(feet, fr["ground"], keys, tables["depth"], tables["lateral"] if corr == "C3" else None)
        else:
            new = feet
        for p in fr["players"]:
            sv = p["sv"] if lag == 0 else p["lag_xy"][LAGS.index(lag)]
            if sv is None:
                sv = p["sv"]
            _, d0 = nearest_foot([sv], feet)
            _, d1 = nearest_foot([sv], new)
            m0, m1 = bool(d0[0] > MISS_FT), bool(d1[0] > MISS_FT)
            acc["n"] += 1
            acc["missed0"] += m0
            acc["missed1"] += m1
            pl = MISS_FT < d0[0] <= NEAR_FT
            acc["placement0"] += pl
            acc["moved_in"] += pl and not m1
            acc["pushed_out"] += (not m0) and m1
    tot = {k: sum(v[k] for v in per_w.values()) for k in ("n", "missed0", "missed1", "placement0", "moved_in",
                                                          "pushed_out")}
    keys = sorted(per_w)
    arr = np.array([[per_w[k]["n"], per_w[k]["missed1"] - per_w[k]["missed0"]] for k in keys], float)
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(BOOT_N):
        a = arr[rng.integers(0, len(arr), len(arr))]
        reps.append(a[:, 1].sum() / a[:, 0].sum())
    return {**tot, "net": (tot["missed1"] - tot["missed0"]) / tot["n"],
            "net_interval": [round(float(np.percentile(reps, 2.5)), 5), round(float(np.percentile(reps, 97.5)), 5)],
            "gross_share": tot["moved_in"] / tot["placement0"] if tot["placement0"] else 0.0}


def corrections_logo(games: dict, lag_of: dict | None = None) -> dict:
    """Leave one pool game out: fit on two, apply to the third. lag_of: {held-out game: lag} for the second pass."""
    out = {c: {} for c in ("C1", "C2", "C3")}
    tables_out = {}
    for held in POOL:
        train = [g for g in POOL if g != held]
        lag = (lag_of or {}).get(held, 0)
        trows = [r for g in train for r in player_rows(games[g], None if not lag else lag)]
        brows = [r for g in train for r in box_rows(games[g])] if not lag else \
            [r for g in train for r in box_rows_lag(games[g], lag)]
        t1 = fit_tables(trows, "third")
        t2 = fit_tables(brows, "boxh")
        tables_out[held] = {"C1": t1[0], "C2": t2[0], "C3": {"depth": t1[0], "lateral": t1[1]}}
        out["C1"][held] = apply_correction(games[held], "C1", {"depth": t1[0], "lateral": {}}, lag)
        out["C2"][held] = apply_correction(games[held], "C2", {"depth": t2[0], "lateral": {}}, lag)
        out["C3"][held] = apply_correction(games[held], "C3", {"depth": t1[0], "lateral": t1[1]}, lag)
    return {"results": out, "tables": tables_out}


def box_rows_lag(game: dict, lag: int) -> list:
    rows = []
    for fr in game["frames"]:
        if fr["ground"] is None:
            continue
        for p in fr["players"]:
            sv = p["lag_xy"][LAGS.index(lag)]
            if sv is None:
                continue
            idx, dist = nearest_foot([sv], fr["feet"])
            if idx[0] < 0 or dist[0] > NEAR_FT:
                continue
            e = fr["feet"][idx[0]] - sv
            u, v = depth_basis(fr["ground"], sv)
            rows.append({"cls": klass(float(dist[0])), "boxh": hbin(fr["box_h"][idx[0]]), "depth": float(e @ u),
                         "lateral": float(e @ v)})
    return rows


def sign_rule(offsets: dict) -> bool:
    """(c): the placement misses' median depth has the same sign on every pool game and each interval excludes 0."""
    meds = [offsets[g]["placement"]["depth_median"] for g in POOL]
    ivs = [offsets[g]["placement"]["depth_median_interval"] for g in POOL]
    if any(m is None for m in meds) or any(i is None for i in ivs):
        return False
    same = all(m > 0 for m in meds) or all(m < 0 for m in meds)
    excl = all(i[0] > 0 or i[1] < 0 for i in ivs)
    return bool(same and excl)


def rnd(x):
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rnd(v) for v in x]
    return x


def cmd_measure(args) -> dict:
    check_games(POOL + CHECK)
    games = {t: load_game(t) for t in POOL + CHECK}
    offsets = {t: offsets_summary(player_rows(games[t])) for t in POOL + CHECK}
    counts = {}
    for t in POOL + CHECK:
        rows = player_rows(games[t])
        counts[t] = {"in_frame": sum(len(f["players"]) for f in games[t]["frames"]),
                     "rows_with_camera": len(rows),
                     "hit": sum(r["cls"] == "hit" for r in rows), "placement": sum(r["cls"] == "placement" for r in rows),
                     "far": sum(r["cls"] == "far" for r in rows), "windows": games[t]["windows"],
                     "registered": games[t]["registered"]}
    scans = {t: lag_scan(games[t]) for t in POOL + CHECK}
    lag_rates = {t: {k: (v[0] / v[1] if v[1] else None) for k, v in s.items()} for t, s in scans.items()}
    lag_folds = {}
    for held in POOL:
        tr = [g for g in POOL if g != held]
        pooled = {k: sum(scans[g][k][0] for g in tr) / sum(scans[g][k][1] for g in tr) for k in LAGS}
        k = choose_lag(pooled)
        lag_folds[held] = {"lag": k, "lag_s": k * 0.04, "change": lag_rates[held][k] - lag_rates[held][0]}
    timing = read_timing(lag_folds)
    first = corrections_logo(games)
    sign_ok = sign_rule(offsets)
    passing = read_systematic({c: {g: v for g, v in r.items()} for c, r in first["results"].items()}, sign_ok)
    second = None
    if timing:
        second = corrections_logo(games, {g: v["lag"] for g, v in lag_folds.items()})
        passing = read_systematic(second["results"], sign_ok)
    reading = "SYSTEMATIC" if passing else ("TIMING" if timing else "SCATTERED")
    if passing and timing:
        reading = "TIMING and SYSTEMATIC"
    # check game: corrections fitted on the whole pool, applied to hou_cle_2016 (report only)
    prow = [r for g in POOL for r in player_rows(games[g])]
    brow = [r for g in POOL for r in box_rows(games[g])]
    t1, t2 = fit_tables(prow, "third"), fit_tables(brow, "boxh")
    pooled_scan = {k: sum(scans[g][k][0] for g in POOL) / sum(scans[g][k][1] for g in POOL) for k in LAGS}
    kp = choose_lag(pooled_scan)
    check = {t: {"C1": apply_correction(games[t], "C1", {"depth": t1[0], "lateral": {}}),
                 "C2": apply_correction(games[t], "C2", {"depth": t2[0], "lateral": {}}),
                 "C3": apply_correction(games[t], "C3", {"depth": t1[0], "lateral": t1[1]}),
                 "pool_lag": {"lag": kp, "change": lag_rates[t][kp] - lag_rates[t][0]}} for t in CHECK}
    out = {"what": "Placement route step 1 (docs/research/P8_placement.md); report only; stored files, r23 reference-"
                   "platform sidecars, scorer universe, truth camera",
           "builds": BUILDS, "counts": counts, "offsets": offsets,
           "lag_scan_missed_rate": {t: {str(k): v for k, v in r.items()} for t, r in lag_rates.items()},
           "lag_scan_players": {t: scans[t][0][1] for t in scans},
           "lag_folds": lag_folds, "timing_reading": timing,
           "corrections_logo": first, "corrections_logo_with_lag": second,
           "sign_rule_c": sign_ok, "passing_corrections": passing, "reading": reading,
           "check_hou_cle_2016": check, "pool_fit_tables": {"C1": t1[0], "C2": t2[0], "C3_lateral": t1[1]},
           "rule": {"net_bar": NET_BAR, "gross_bar": GROSS_BAR, "lag_bar": LAG_BAR, "miss_ft": MISS_FT,
                    "near_ft": NEAR_FT, "min_speed_ftps": MIN_SPEED, "lags": [LAGS[0], LAGS[-1]],
                    "boot": {"n": BOOT_N, "seed": SEED, "unit": "window"}}}
    out = rnd(out)
    d = p8_root() / "placement"
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    atomic_write(d / "step1.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_placement.json", text)
    print(json.dumps({"counts": counts, "reading": reading, "lag_folds": lag_folds, "sign_rule_c": sign_ok,
                      "logo": {c: {g: {k: v[k] for k in ("net", "net_interval", "gross_share", "moved_in",
                                                         "pushed_out")} for g, v in r.items()}
                               for c, r in out["corrections_logo"]["results"].items()}}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_placement")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()

"""sportvu/label_refine.py: R1.3 attempt 3, NOT ADOPTED. Refit each auto-label's camera to the painted key.

Idea: the R1.3 camera-truth labels (run r13_20261004_1846) sit a median ~7 px from the GSW key's gold
edges with a tail past 15 px (sportvu/label_check.py). The per-frame fit sees line ridges and noisy feet,
never the gold-to-wood edges. So, per label, camera position C fixed, start from the label's pose and refit
pan, tilt and zoom (rvec, log f) to
  stage 1  the key's free-throw and lane edges on the gold-paint edge field, radius PAINT_RADII
  stage 2  those edges plus every court line on the ridge field (court/snap_track), radius LINE_RADII
with a weak pull (PRIOR_W) toward the starting pose on the in-frame grid points.

Verification (--holdout): the y=17 lane edges are left out of the refit and scored before and after on the
paint-edge field; neither fit uses them, and the parallel y=33 lane gives the refit the same information.
Result 2026-10-04 on 300 gsw_bkn_2015 labels: the fitted edges move onto the paint, but the held-out lane
gets WORSE (median 6.04 -> 7.53 px, 159 frames worse vs 75 better by > 1 px), and the SportVU feet
distance of the confirmed boxes rises (1.15 -> 1.55 ft median; biased toward the old fit, which selected
them). A pinhole camera with C pinned cannot put both lanes on the paint, so pulling one edge in pushes the
other out. The labels keep the camera truth; sportvu/label_check.py rejects the ones that are off instead.
Kept so that result can be reproduced: `python -m sportvu.label_refine --games gsw_bkn_2015 --sample 300
--holdout` writes reports/sportvu_label_refine.{json,txt}. Training games only.
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np
import config
from sportvu.label_check import LANE17, LANE33, FT_EDGES, PAINT_CHORDS, chords as _chords, paint_edge_field

LABELS = config.PROJECT_ROOT / "data" / "sportvu" / "labels"
PAINT_RADII = (40, 16)
LINE_RADII = (12, 6)
F_SCALE_PX = 2.0
PRIOR_W = 0.05            # 20 px of drift on a grid point costs as much as 1 px of line residual
MIN_PAINT = 12            # paint-edge matches needed to run stage 1
MIN_LINES = 20            # ridge matches needed to run stage 2

FIT_CHORDS, TEST_CHORDS = _chords(LANE33 + FT_EDGES), _chords(LANE17)
HOLDOUT_RADIUS = 30       # the gold edge field has no clutter, so a wide search is safe


def holdout_edge(Hc, pe, w, h) -> tuple:
    """Hold-out score: (matched samples, median px offset) of the y=17 lane lines on the gold-paint edge
    field. Neither the R1.3 fit (feet + ridges) nor the hold-out refit (y=33 lane + free-throw edges +
    ridges) uses that edge; the y=33 lane is parallel to it, so the refit has the same information."""
    from court.snap_track import match_lines
    import cv2
    mid, peaks = match_lines(Hc, pe, w, h, HOLDOUT_RADIUS, chords=TEST_CHORDS)
    if len(peaks) < 4:
        return int(len(peaks)), None
    p = cv2.perspectiveTransform(mid.reshape(-1, 1, 2).astype(np.float64), Hc).reshape(-1, 2)
    return int(len(peaks)), round(float(np.median(np.linalg.norm(p - peaks, axis=1))), 2)


def refine(row: dict, img, holdout: bool = False) -> dict:
    """holdout=True leaves the y=17 lane edges out of the fit and scores them before and after."""
    import cv2
    from scipy.optimize import least_squares
    from court.grid import GRID_FT
    from court.snap_track import ridge_field, match_lines
    from sportvu import camera as cam
    h, w = img.shape[:2]
    C = np.asarray(row["camera_C"], np.float64)
    init = cam.pose_from_h(row["H_truth"], w, h)
    if init is None:
        return {"status": "no_init"}
    lo, hi = np.log(cam.F_BOUNDS_PX[0]), np.log(cam.F_BOUNDS_PX[1])
    G0 = cv2.perspectiveTransform(np.asarray(GRID_FT, np.float32).reshape(-1, 1, 2),
                                  np.linalg.inv(np.asarray(row["H_truth"], np.float64).reshape(3, 3))).reshape(-1, 2)
    on = np.isfinite(G0).all(1) & (G0[:, 0] >= 0) & (G0[:, 0] < w) & (G0[:, 1] >= 0) & (G0[:, 1] < h)
    Gft, G0 = np.asarray(GRID_FT, np.float64)[on], G0[on].astype(np.float64)
    if len(Gft) < 4:
        return {"status": "few_grid"}
    # start: the label's pose with C pinned (the decomposition re-estimates C, so refit rvec, f to the grid)
    x = np.array([*init[0], float(np.clip(np.log(init[1]), lo + 1e-3, hi - 1e-3))])
    bounds = ([-np.inf] * 3 + [lo], [np.inf] * 3 + [hi])
    proj = lambda z, pts: cam.project(C, z[:3], np.exp(z[3]), w, h, pts)
    x = least_squares(lambda z: (proj(z, Gft) - G0).ravel(), x, bounds=bounds, max_nfev=100).x
    start_px = float(np.median(np.linalg.norm(proj(x, Gft) - G0, axis=1)))
    ridge = ridge_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ridge is not None:
        for b in row.get("boxes", []):
            x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
            ridge[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
    pe = paint_edge_field(img)
    n_paint = n_lines = 0
    hold = {"hold_before": holdout_edge(cam.h_c2i(C, x[:3], np.exp(x[3]), w, h), pe, w, h)} if holdout else {}
    stages = [(r, False) for r in PAINT_RADII] + [(r, True) for r in LINE_RADII]
    for rad, with_lines in stages:
        Hc = cam.h_c2i(C, x[:3], np.exp(x[3]), w, h)
        pm, pp = match_lines(Hc, pe, w, h, rad, chords=FIT_CHORDS if holdout else PAINT_CHORDS)
        lm, lp = (match_lines(Hc, ridge, w, h, rad) if with_lines and ridge is not None
                  else (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)))
        if len(pp) < MIN_PAINT and (not with_lines or len(lp) < MIN_LINES):
            continue
        if len(pp) < MIN_PAINT:
            pm, pp = pm[:0], pp[:0]
        if with_lines and len(lp) < MIN_LINES:
            lm, lp = lm[:0], lp[:0]
        n_paint, n_lines = len(pp), len(lp)
        def fun(z):
            return np.concatenate([(proj(z, pm) - pp).ravel(), (proj(z, lm) - lp).ravel(),
                                   PRIOR_W * (proj(z, Gft) - G0).ravel()])
        x = least_squares(fun, x, loss="soft_l1", f_scale=F_SCALE_PX, max_nfev=100, bounds=bounds).x
    disp = np.linalg.norm(proj(x, Gft) - G0, axis=1)
    H = cam.h_px2ft(C, x[:3], float(np.exp(x[3])), w, h)
    if holdout:
        hold["hold_after"] = holdout_edge(cam.h_c2i(C, x[:3], np.exp(x[3]), w, h), pe, w, h)
    return {"status": "ok", **hold, "H_refined": H.ravel().tolist(), "f": float(np.exp(x[3])), "paint_matches": n_paint,
            "line_matches": n_lines, "start_px": round(start_px, 2),
            "move_px_p50": round(float(np.median(disp)), 1), "move_px_max": round(float(disp.max()), 1)}


_INDEX = {}


def _sportvu(game: str):
    """(SportVUIndex, mirror flips) for a training game, cached per worker."""
    if game not in _INDEX:
        from sportvu.sync import SportVUIndex, MIRRORS
        g = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())[game]
        mom = json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (g["sportvu"] + "_moments.json")).read_text())
        mirror = json.loads((config.REPORTS_DIR / ("sportvu_direction_%s.json" % game)).read_text())["resolution"]["mirror"]
        _INDEX[game] = (SportVUIndex(mom["moments"]), MIRRORS[mirror])
    return _INDEX[game]


def feet_d(row: dict, H) -> list:
    """Distance (ft) from each SportVU-confirmed box's foot, projected through H, to that player in SportVU."""
    import cv2
    from sportvu.sync import apply_mirror
    from sportvu.truth import CLOCK_TOL_S
    if not row.get("boxes"):
        return []
    index, (fx, fy) = _sportvu(row["game"])
    i = index.at(row["q"], row["clock"], CLOCK_TOL_S + 0.01)
    if i is None:
        return []
    d = index.q[row["q"]]
    sv = apply_mirror(np.asarray(d["xy"][i], np.float32), fx, fy)
    pid = [int(p) for p in d["pid"][i]]
    feet = np.array([b["foot"] for b in row["boxes"]], np.float32).reshape(-1, 1, 2)
    ct = cv2.perspectiveTransform(feet, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)
    out = []
    for b, c in zip(row["boxes"], ct):
        if b["pid"] in pid:
            out.append(float(np.linalg.norm(c - sv[pid.index(b["pid"])])))
    return out


def _do_row(arg) -> dict:
    row, holdout = arg
    import cv2
    from sportvu.label_check import key_edges
    out = {"window": row["window"], "frame": row["frame"]}
    img = cv2.imread(str(config.PROJECT_ROOT / row["image"])) if row.get("image") else None
    if img is None:
        return {**out, "status": "no_image"}
    r = refine(row, img, holdout)
    out.update(r)
    if r["status"] != "ok":
        return out
    before, after = key_edges(img, row["H_truth"]), key_edges(img, r["H_refined"])
    out["paint_before"], out["paint_after"] = before.get("edge_px"), after.get("edge_px")
    out["feet_before"] = feet_d(row, row["H_truth"])
    out["feet_after"] = feet_d(row, r["H_refined"])
    return out


def run_game(game: str, workers: int = 8, sample: int = 0, holdout: bool = False) -> list:
    from multiprocessing import Pool
    rows = [json.loads(l) for f in sorted(glob.glob(str(LABELS / game / "*.jsonl"))) for l in open(f)]
    if sample:
        rows = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, sample)]
    with Pool(workers) as pool:
        res = pool.map(_do_row, [(r, holdout) for r in rows], chunksize=16)
    if not sample and not holdout:
        (LABELS / (game + "_refined.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in res))
    return res


def summarise(game: str, res: list) -> dict:
    ok = [r for r in res if r["status"] == "ok"]
    st = {}
    for r in res:
        st[r["status"]] = st.get(r["status"], 0) + 1
    both = [r for r in ok if r.get("paint_before") is not None and r.get("paint_after") is not None]
    pb = np.array([r["paint_before"] for r in both]); pa = np.array([r["paint_after"] for r in both])
    fb = np.array([d for r in ok for d in r["feet_before"]]); fa = np.array([d for r in ok for d in r["feet_after"]])
    q = lambda a, p: round(float(np.percentile(a, p)), 2) if len(a) else None
    mv = np.array([r["move_px_p50"] for r in ok])
    cur = [r for r in ok if "hold_before" in r and r["hold_before"][1] is not None and r["hold_after"][1] is not None]
    hb = np.array([r["hold_before"][1] for r in cur]); ha = np.array([r["hold_after"][1] for r in cur])
    better, worse = int((ha < hb - 1).sum()), int((ha > hb + 1).sum())
    return {"game": game, "labels": len(res), "status": st, "paint_measured_both": len(both),
            "paint_px": {"before": {"p50": q(pb, 50), "p90": q(pb, 90), "le10": round(float((pb <= 10).mean()), 3) if len(pb) else None},
                         "after": {"p50": q(pa, 50), "p90": q(pa, 90), "le10": round(float((pa <= 10).mean()), 3) if len(pa) else None}},
            "feet_ft": {"n": int(len(fb)), "before": {"p50": q(fb, 50), "p90": q(fb, 90)}, "after": {"p50": q(fa, 50), "p90": q(fa, 90)}},
            "move_px_p50": {"p50": q(mv, 50), "p90": q(mv, 90)},
            **({"holdout_lane17": {"frames": len(cur), "offset_px_p50_before": q(hb, 50), "offset_px_p50_after": q(ha, 50),
                                    "offset_px_p90_before": q(hb, 90), "offset_px_p90_after": q(ha, 90),
                                    "frames_better_1px": better, "frames_worse_1px": worse}} if cur else {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=0, help="evenly spaced labels per game; no files written")
    ap.add_argument("--holdout", action="store_true", help="verification: fit without the y=17 lane edges, score them")
    a = ap.parse_args()
    games = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())
    train = [t for t, g in games.items() if g.get("split") == "train" and (LABELS / t).is_dir()]
    sel = a.games or train
    bad = [t for t in sel if t not in train]
    if bad:
        raise SystemExit("refusing %s: not a labelled training game" % bad)
    summ = [summarise(t, run_game(t, a.workers, a.sample, a.holdout)) for t in sel]
    print(json.dumps(summ, indent=1))
    if a.holdout:
        (config.REPORTS_DIR / "sportvu_label_refine.json").write_text(json.dumps(
            {"item": "ROADMAP R1.3 attempt 3 (not adopted): paint refit, held-out lane check", "sample": a.sample,
             "rules": {"paint_radii": PAINT_RADII, "line_radii": LINE_RADII, "prior_w": PRIOR_W, "holdout_radius": HOLDOUT_RADIUS},
             "games": summ}, indent=1))
        L = ["R1.3 ATTEMPT 3, PAINT REFIT (NOT ADOPTED): y=17 lane edges held out of the refit, px to the gold edge"]
        for g in summ:
            ho, pp, ft = g.get("holdout_lane17", {}), g["paint_px"], g["feet_ft"]
            L.append("  %-14s labels %d | held-out lane p50 %s -> %s, p90 %s -> %s | frames better %s, worse %s (> 1 px)" % (
                g["game"], g["labels"], ho.get("offset_px_p50_before"), ho.get("offset_px_p50_after"),
                ho.get("offset_px_p90_before"), ho.get("offset_px_p90_after"), ho.get("frames_better_1px"), ho.get("frames_worse_1px")))
            L.append("                 fitted key edges p50 %s -> %s px | SportVU feet p50 %s -> %s ft (biased to the old fit)" % (
                pp["before"]["p50"], pp["after"]["p50"], ft["before"]["p50"], ft["after"]["p50"]))
        L += ["verdict: the held-out lane gets worse, so the refit is not used; label_check rejects off labels instead"]
        (config.REPORTS_DIR / "sportvu_label_refine.txt").write_text("\n".join(L) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    main()

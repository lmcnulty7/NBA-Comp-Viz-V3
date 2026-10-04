"""
sportvu/camera.py: camera-model truth H per frame (ROADMAP R1.3, chosen by Lucien 2026-10-04).

Why: the B4 truth H is a free 8-parameter homography fitted to 6..10 feet bunched in one part of the
court; it fits those feet and nothing else (reports/sportvu_truth_line_check.txt). A broadcast main
camera stays in one place, so this module models it as a pinhole camera with ONE position per game
(C, ft, court frame) and, per frame, a rotation (Rodrigues rvec) and a focal length f (zoom); the
principal point sits at the image centre, pixels are square, distortion is ignored. A frame then has
4 free parameters instead of 8, and the shared position ties every frame to the same geometry, so the
court far from the fitted players is constrained too.

Fit (per game, training windows only):
  1. init      decompose each frame's V3 court H into (f, R, C) with the principal point fixed;
               C0 = median over frames (V3's H is 20..40 px off but globally court-shaped).
  2. pooled    bundle adjustment over POOL_MAX frames whose B4-style ICP truth has >= 6 inlier pairs:
               shared C, per-frame (rvec, log f); residual = reprojection of the matched SportVU
               positions onto the feet (px), soft-L1 with F_SCALE_PX.
  3. per frame C fixed, (rvec, log f) refit on the frame's pairs, f bounded to F_BOUNDS_PX.
  4. lines     the same 4 parameters refit jointly to the feet and to the painted-line ridges
               (court/snap_track ridge_field + match_lines along each template sample's normal, player
               boxes masked) at shrinking search radii LINE_RADII; feet residuals are weighted by
               FEET_WEIGHT, line residuals use a soft-L1 at LINE_F_SCALE_PX. A 4-parameter camera cannot
               bend to fit crowd or logo ridges the way a free homography can.
Validation (before any label is written): the A1 line indicator (median px from painted-line ridges to
the projected template) for the camera H vs V3's H vs the B4 truth H on the same frames, and a
contact sheet with the camera template drawn over the frame.

  python -m sportvu.camera gsw_bkn_2015      (uses the R1.2 direction windows in data/sportvu/build)
      -> reports/sportvu_camera_<game>.json, data/sportvu/labels/<game>_camera_sheet.jpg (gitignored)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import config

POOL_MAX = 400
F_SCALE_PX = 6.0
MAX_RESID_FT = 0.5
MIN_INLIERS = 6
MIN_PAIRS = 6
F_BOUNDS_PX = (500.0, 12000.0)
LINE_RADII = (20, 12, 6)
LINE_F_SCALE_PX = 2.0
FEET_WEIGHT = 0.5


def K_of(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def h_c2i(C, rvec, f, w, h) -> np.ndarray:
    """Court ground plane (ft) -> image (px) homography of the camera."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = -R @ np.asarray(C, np.float64).reshape(3)
    return K_of(f, w, h) @ np.column_stack([R[:, 0], R[:, 1], t])


def h_px2ft(C, rvec, f, w, h) -> np.ndarray:
    H = np.linalg.inv(h_c2i(C, rvec, f, w, h))
    return H / H[2, 2]


def project(C, rvec, f, w, h, pts_ft) -> np.ndarray:
    pts = np.asarray(pts_ft, np.float64).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts))]) @ h_c2i(C, rvec, f, w, h).T
    return hom[:, :2] / hom[:, 2:3]


def pose_from_h(H_px2ft, w: int, h: int):
    """(rvec, f, C) of the pinhole camera whose ground-plane homography best matches H, with the
    principal point at the image centre and square pixels; None if H is not a camera homography."""
    import cv2
    try:
        A = np.linalg.inv(np.asarray(H_px2ft, np.float64).reshape(3, 3))
    except np.linalg.LinAlgError:
        return None
    T = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1.0]])
    Hn = T @ A
    h1, h2, h3 = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    cands = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
    den = h2[2] ** 2 - h1[2] ** 2
    if abs(den) > 1e-12:
        cands.append((h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den)
    f2 = [c for c in cands if c > 0]
    if not f2:
        return None
    f = float(np.sqrt(np.median(f2)))
    Ki = np.linalg.inv(K_of(f, 0, 0))
    r1, r2, t = Ki @ h1, Ki @ h2, Ki @ h3
    lam = 2.0 / (np.linalg.norm(r1) + np.linalg.norm(r2))
    r1, r2, t = r1 * lam, r2 * lam, t * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), f, (-R.T @ t).ravel()


def collect(game: str, windows_dir) -> list:
    """Frames of the game's rebuilt windows with ICP truth pairs (B4 method): sidecar row, pairs,
    SportVU positions, image size."""
    import cv2
    from sportvu.sync import SportVUIndex, MIRRORS, apply_mirror
    from sportvu.truth import window_truth
    reg = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())[game]
    if reg.get("split") != "train":
        raise SystemExit("refusing %s: held out (camera fit and validation run on training games)" % game)
    mirror = json.loads((config.REPORTS_DIR / ("sportvu_direction_%s.json" % game)).read_text())["resolution"]["mirror"]
    index = SportVUIndex(json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (reg["sportvu"] + "_moments.json")).read_text())["moments"])
    windows = json.loads((windows_dir / (game + "_windows.json")).read_text())
    fx, fy = MIRRORS[mirror]
    out = []
    for w in windows:
        sp, tp = windows_dir / (w["window"] + "_frames.json"), windows_dir / (w["window"] + "_timemap.json")
        if not sp.exists() or not tp.exists():
            continue
        sc, tm = json.loads(sp.read_text()), json.loads(tp.read_text())
        cap = cv2.VideoCapture(str(windows_dir / (w["window"] + ".mp4"))); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        srow = {r["frame"]: r for r in sc["frames"]}
        _, _, rows = window_truth(w, sc, tm, index, mirror)
        for r in rows:
            if r["status"] != "ok":
                continue
            q, i = r["q"], r["moment"]
            sv = apply_mirror(index.q[q]["xy"][i], fx, fy); pid = index.q[q]["pid"][i]
            side = srow[r["frame"]]
            pairs = [(side["boxes"][p["box"]]["foot_stab"], sv[int(np.where(pid == p["pid"])[0][0])]) for p in r["pairs"] if p["inlier"]]
            if len(pairs) >= MIN_PAIRS:
                out.append({"window": w["window"], "frame": r["frame"], "w": W, "h": H, "H_v3": side["H"], "H_b4": r["H_truth"],
                            "px": np.array([a for a, _ in pairs], np.float64), "ft": np.array([b for _, b in pairs], np.float64),
                            "boxes": [b["bbox"] for b in side["boxes"]]})
    return out


def fit_game(frames: list, seed: int = 0) -> dict:
    """Pooled bundle adjustment: shared C, per-frame (rvec, log f)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    inits = []
    for fr in frames:
        p = pose_from_h(fr["H_v3"], fr["w"], fr["h"])
        if p is not None and p[2][2] != 0:
            inits.append((fr, p))
    if len(inits) < 20:
        return {"status": "too_few_inits", "n": len(inits)}
    C0 = np.median(np.array([p[2] for _, p in inits]), axis=0)
    rng = np.random.default_rng(seed)
    pool = [inits[i] for i in sorted(rng.choice(len(inits), size=min(POOL_MAX, len(inits)), replace=False))]
    x0 = [*C0]
    for _, (rv, f, _) in pool:
        x0 += [*rv, np.log(f)]
    x0 = np.array(x0)
    sizes = [len(fr["px"]) for fr, _ in pool]

    def resid(x):
        C = x[:3]; out = []
        for k, (fr, _) in enumerate(pool):
            rv, lf = x[3 + 4 * k: 6 + 4 * k], x[6 + 4 * k]
            out.append((project(C, rv, np.exp(lf), fr["w"], fr["h"], fr["ft"]) - fr["px"]).ravel())
        return np.concatenate(out)

    m = 2 * sum(sizes)
    S = lil_matrix((m, len(x0)), dtype=int)
    row = 0
    for k, n in enumerate(sizes):
        S[row:row + 2 * n, :3] = 1
        S[row:row + 2 * n, 3 + 4 * k: 7 + 4 * k] = 1
        row += 2 * n
    r0 = np.median(np.abs(resid(x0)))
    sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=F_SCALE_PX, x_scale="jac", max_nfev=200)
    r1 = np.abs(sol.fun)
    return {"status": "ok", "C": sol.x[:3].tolist(), "C0": C0.tolist(), "pooled_frames": len(pool), "inits": len(inits),
            "reproj_px_median_before": round(float(r0), 2), "reproj_px_median_after": round(float(np.median(r1)), 2),
            "cost": float(sol.cost)}


def fit_frame(C, fr: dict) -> dict:
    """C fixed; (rvec, log f) for one frame, initialised from V3's H (or the B4 H)."""
    from scipy.optimize import least_squares
    init = pose_from_h(fr["H_v3"], fr["w"], fr["h"]) or pose_from_h(fr["H_b4"], fr["w"], fr["h"])
    if init is None:
        return {"status": "no_init"}
    lo, hi = np.log(F_BOUNDS_PX[0]), np.log(F_BOUNDS_PX[1])
    x0 = np.array([*init[0], float(np.clip(np.log(init[1]), lo + 1e-3, hi - 1e-3))])
    fun = lambda x: (project(C, x[:3], np.exp(x[3]), fr["w"], fr["h"], fr["ft"]) - fr["px"]).ravel()
    sol = least_squares(fun, x0, loss="soft_l1", f_scale=F_SCALE_PX, max_nfev=200,
                        bounds=([-np.inf] * 3 + [lo], [np.inf] * 3 + [hi]))
    rv, f = sol.x[:3], float(np.exp(sol.x[3]))
    return _finish(C, fr, rv, f)


def _finish(C, fr: dict, rv, f) -> dict:
    Hp = h_px2ft(C, rv, f, fr["w"], fr["h"])
    import cv2
    ft_hat = cv2.perspectiveTransform(fr["px"].reshape(-1, 1, 2).astype(np.float32), Hp).reshape(-1, 2)
    d = np.linalg.norm(ft_hat - fr["ft"], axis=1)
    return {"status": "ok", "rvec": rv.tolist(), "f": f, "H": Hp.ravel().tolist(), "resid_ft": round(float(np.median(d)), 3),
            "inliers": int((d <= 1.5).sum())}


def refine_lines(C, fr: dict, fit: dict, img) -> dict:
    """Refit (rvec, log f) to the feet AND the painted-line ridges (C fixed)."""
    import cv2
    from scipy.optimize import least_squares
    from court.snap_track import ridge_field, match_lines
    if fit.get("status") != "ok":
        return fit
    ridge = ridge_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ridge is None:
        return {**fit, "lines": 0}
    for x1, y1, x2, y2 in fr["boxes"]:
        ridge[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
    w, h = fr["w"], fr["h"]
    lo, hi = np.log(F_BOUNDS_PX[0]), np.log(F_BOUNDS_PX[1])
    x = np.array([*fit["rvec"], float(np.clip(np.log(fit["f"]), lo + 1e-3, hi - 1e-3))])
    n_lines = 0
    for rad in LINE_RADII:
        court_mid, peaks = match_lines(h_c2i(C, x[:3], np.exp(x[3]), w, h), ridge, w, h, rad)
        if len(peaks) < 20:
            break
        n_lines = len(peaks)
        def fun(z):
            feet = FEET_WEIGHT * (project(C, z[:3], np.exp(z[3]), w, h, fr["ft"]) - fr["px"]).ravel()
            lines = (project(C, z[:3], np.exp(z[3]), w, h, court_mid) - peaks).ravel()
            return np.concatenate([feet, lines])
        x = least_squares(fun, x, loss="soft_l1", f_scale=LINE_F_SCALE_PX, max_nfev=100,
                          bounds=([-np.inf] * 3 + [lo], [np.inf] * 3 + [hi])).x
    out = _finish(C, fr, x[:3], float(np.exp(x[3])))
    out["lines"] = n_lines
    return out


def validate(game: str, windows_dir, frames: list, fits: list, every: int = 4, refine: bool = True) -> dict:
    """A1 line indicator for camera H vs V3 H vs B4 truth H on the same frames + a contact sheet."""
    import cv2
    from qc.track_qc import line_overlay
    from videoseq import SeqReader
    from court.snap_track import CH_A, CH_B
    rows, sheet, refined = [], [], []
    byw = {}
    for fr, ft in zip(frames, fits):
        if ft.get("status") == "ok":
            byw.setdefault(fr["window"], []).append((fr, ft))
    for wname, items in byw.items():
        sr = SeqReader(cv2.VideoCapture(str(windows_dir / (wname + ".mp4"))))
        for fr, ft in items[::every]:
            ok, img = sr.read(fr["frame"])
            if not ok:
                continue
            if refine:
                ft = refine_lines(np.array(fr["C"]), fr, ft, img)
                if ft.get("status") != "ok":
                    continue
                refined.append(ft)
            lo = [line_overlay(img, H, fr["boxes"]) for H in (ft["H"], fr["H_v3"], fr["H_b4"])]
            a = [o["dist_px"] for o in lo] + [o["support"] for o in lo]
            if None not in a:
                rows.append(a)
            if len(sheet) < 400:
                sheet.append((img, ft["H"]))
    a = np.array(rows) if rows else np.zeros((0, 6))
    tiles = []
    for img, Hl in [sheet[int(i)] for i in np.linspace(0, len(sheet) - 1, min(12, len(sheet)))] if sheet else []:
        im = img.copy(); P = np.linalg.inv(np.array(Hl).reshape(3, 3))
        pa = cv2.perspectiveTransform(CH_A.reshape(-1, 1, 2), P).reshape(-1, 2); pb = cv2.perspectiveTransform(CH_B.reshape(-1, 1, 2), P).reshape(-1, 2)
        for p0, p1 in zip(pa, pb):
            if np.isfinite(p0).all() and np.isfinite(p1).all() and (np.abs(p0) < 4000).all() and (np.abs(p1) < 4000).all():
                cv2.line(im, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (0, 255, 255), 2)
        tiles.append(cv2.resize(im, (640, 360)))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        out = config.PROJECT_ROOT / "data" / "sportvu" / "labels"; out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / (game + "_camera_sheet.jpg")), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    rres = [r["resid_ft"] for r in refined]
    return {"frames": int(len(a)), "refined": bool(refine),
            "refined_resid_ft_median": round(float(np.median(rres)), 3) if rres else None,
            "refined_line_matches_median": float(np.median([r.get("lines", 0) for r in refined])) if refined else None,
            "a1_dist_px_median": {"camera": round(float(np.median(a[:, 0])), 1) if len(a) else None,
                                  "v3": round(float(np.median(a[:, 1])), 1) if len(a) else None,
                                  "b4_truth": round(float(np.median(a[:, 2])), 1) if len(a) else None},
            "share_camera_better_than_v3": round(float((a[:, 0] < a[:, 1]).mean()), 3) if len(a) else None,
            "line_support_median": {"camera": round(float(np.median(a[:, 3])), 3), "v3": round(float(np.median(a[:, 4])), 3),
                                    "b4_truth": round(float(np.median(a[:, 5])), 3)} if len(a) else None,
            "share_camera_support_above_v3": round(float((a[:, 3] > a[:, 4]).mean()), 3) if len(a) else None,
            "share_over_60px": {k: round(float((a[:, i] > 60).mean()), 3) for i, k in enumerate(("camera", "v3", "b4_truth"))} if len(a) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game")
    a = ap.parse_args()
    from sportvu.rebuild import BUILD_DIR
    t0 = time.time()
    frames = collect(a.game, BUILD_DIR)
    print("%s: %d frames with >= %d ICP inlier pairs (%.0f s)" % (a.game, len(frames), MIN_PAIRS, time.time() - t0), flush=True)
    g = fit_game(frames)
    print("pooled fit:", {k: v for k, v in g.items() if k != "cost"}, flush=True)
    if g["status"] != "ok":
        return
    fits = [fit_frame(np.array(g["C"]), fr) for fr in frames]
    for fr in frames:
        fr["C"] = g["C"]
    ok = [f for f in fits if f.get("status") == "ok"]
    acc = [f for f in ok if f["resid_ft"] <= MAX_RESID_FT and f["inliers"] >= MIN_INLIERS]
    val = validate(a.game, BUILD_DIR, frames, fits)
    rep = {"game": a.game, "frames": len(frames), "camera": g, "per_frame": {"fitted": len(ok), "accepted": len(acc),
           "resid_ft_median": round(float(np.median([f["resid_ft"] for f in ok])), 3) if ok else None,
           "f_px_range": [round(min(f["f"] for f in ok)), round(max(f["f"] for f in ok))] if ok else None},
           "validation": val}
    (config.REPORTS_DIR / ("sportvu_camera_%s.json" % a.game)).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "camera"}, indent=1))


if __name__ == "__main__":
    main()

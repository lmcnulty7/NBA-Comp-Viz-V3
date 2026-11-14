"""sportvu/camera_track.py: ROADMAP R2.4, a camera-model court solver on top of CourtTracker.

The free homography of CourtTracker has 8 parameters per frame and is fitted to the lines in view, which on a
broadcast wide shot are mostly in the far half; the near third is extrapolated. A broadcast main camera stays in
one place for the whole game, so here, per game:

  1. trace     run CourtTracker (any grid weights) over a window's frames as the pipeline does, and keep per frame
               its state, its free H and the painted-line matches (court ft, px) of the final H (ridge field with
               the player boxes masked, normal search at TRACE_RADIUS).
  2. position  self-calibrate ONE camera position C from the first frames of the game in time order (POOL_N
               TRACK frames with >= POOL_MIN_MATCH line matches and snap residual <= POOL_MAX_RES px): pooled
               bundle adjustment, shared C, per-frame (rotation, log focal), soft-L1 on the line residuals.
               No SportVU, no labels: only the game's own detected lines.
  3. per frame C fixed, (rotation, log focal) fitted to the frame's line matches (>= MIN_MATCH), started from the
               previous frame's camera or from the free H; with fewer matches, the camera closest to the free H
               (fitted to the court grid projected through it). The written H is derived from the camera, never
               free. The tracker's state (TRACK / LINE_TRACK / HELD / LOST) is kept.

Constants are the training-game values of sportvu/camera.py (R1.3) or are set on the training games here
(`dev`); the held-out games are only scored (`heldout`).

  python -m sportvu.camera_track dev --weights models/court_grid_r23.pt --name r23   (training games, R1.3 labels)
  python -m sportvu.camera_track heldout --weights models/court_grid_r23.pt --name r23_cam [--card-dir DIR]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import config
from sportvu.camera import project as cam_project, h_px2ft, pose_from_h, F_BOUNDS_PX

TRACE_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "camera_trace"
LABEL_BUILD = config.PROJECT_ROOT / "data" / "sportvu" / "label_build"
LABELS = config.PROJECT_ROOT / "data" / "sportvu" / "labels"
REPORT_DIR = config.PROJECT_ROOT / "reports"
TRACE_RADIUS = 8            # px, normal search around the tracker's final H
TRACE_KEEP = 160            # line matches kept per frame (evenly subsampled)
POOL_N = 150                # frames pooled for the camera position
POOL_MIN_MATCH = 60
POOL_MAX_RES = 2.0
MIN_MATCH = 25              # per-frame camera fit on lines needs this many matches
LINE_F_SCALE_PX = 2.0       # sportvu/camera.py (set on gsw_bkn_2015 at R1.3)


# ---- 1. trace ------------------------------------------------------------------------------------------------
def trace_window(clip: Path, side: list, weights: Path, offs: list | None = None) -> list:
    """[{frame, state, H, ft, px, res}] for the sidecar's frames, tracker fed in sidecar order."""
    import cv2
    from court.snap_track import CourtTracker, ridge_field, match_lines
    from sportvu.court_replay import _read_sequential
    offs = offs or []

    def k_of(f):
        k = 0
        for s0, kk in offs:
            if f >= s0:
                k = kk
        return k
    src = {r["frame"] + k_of(r["frame"]): r for r in side}
    frames = _read_sequential(clip, set(src))
    trk = CourtTracker(weights=weights)
    out = []
    for seq_i in sorted(src):
        r, img = src[seq_i], frames.get(seq_i)
        hom = trk.update(img) if img is not None else None
        row = {"frame": r["frame"], "state": "LOST", "H": None, "ft": [], "px": [], "res": trk.last_res_px}
        if hom is not None and hom.is_valid:
            H = np.asarray(hom._H, np.float64)
            row.update(state=trk.state, H=[float(v) for v in H.ravel()])
            ridge = ridge_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ridge is not None:
                for b in r.get("boxes") or []:
                    x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"][:4]]
                    ridge[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
                ft, px = match_lines(np.linalg.inv(H), ridge, img.shape[1], img.shape[0], TRACE_RADIUS)
                if len(ft) > TRACE_KEEP:
                    sel = np.linspace(0, len(ft) - 1, TRACE_KEEP).round().astype(int)
                    ft, px = ft[sel], px[sel]
                row["ft"], row["px"] = np.round(ft, 2).tolist(), np.round(px, 2).tolist()
        out.append(row)
    return out


def trace_game(name: str, windows: list, build_dir: Path, weights: Path, offsets: dict | None = None) -> dict:
    """{window: {"w", "h", "rows"}}, cached per window under TRACE_DIR/<name>/."""
    import cv2
    d = TRACE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    out, t0 = {}, time.time()
    for i, w in enumerate(windows):
        p = d / (w + ".json")
        if not p.exists():
            side = json.loads((build_dir / (w + "_frames.json")).read_text())["frames"]
            cap = cv2.VideoCapture(str(build_dir / (w + ".mp4")))
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            offs = (offsets or {}).get(w, {}).get("offsets")
            p.write_text(json.dumps({"w": W, "h": H, "rows": trace_window(build_dir / (w + ".mp4"), side, weights, offs)}))
            print("  trace [%d/%d] %s (%.0f s)" % (i + 1, len(windows), w, time.time() - t0), flush=True)
        out[w] = json.loads(p.read_text())
    return out


# ---- 2. camera position ----------------------------------------------------------------------------------------
def _pool(trace: dict) -> list:
    frames = []
    for w in sorted(trace):                         # window names sort in game time (section, window)
        t = trace[w]
        for r in t["rows"]:
            if r["state"] == "TRACK" and len(r["ft"]) >= POOL_MIN_MATCH and r["res"] is not None and r["res"] <= POOL_MAX_RES:
                frames.append({"w": t["w"], "h": t["h"], "H": np.array(r["H"]).reshape(3, 3),
                               "ft": np.array(r["ft"]), "px": np.array(r["px"])})
    return frames[:POOL_N]


def fit_position(trace: dict) -> dict:
    """Shared camera position C (ft) from the game's first good frames; bundle adjustment on line matches."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    pool = []
    for fr in _pool(trace):
        p = pose_from_h(fr["H"], fr["w"], fr["h"])
        if p is not None:
            pool.append((fr, p))
    if len(pool) < 20:
        return {"status": "too_few_frames", "n": len(pool)}
    C0 = np.median(np.array([p[2] for _, p in pool]), axis=0)
    x0 = np.array([*C0] + [v for _, (rv, f, _) in pool for v in (*rv, np.log(f))])
    sizes = [len(fr["ft"]) for fr, _ in pool]

    def resid(x):
        C = x[:3]
        return np.concatenate([(cam_project(C, x[3 + 4 * k: 6 + 4 * k], np.exp(x[6 + 4 * k]), fr["w"], fr["h"], fr["ft"])
                                - fr["px"]).ravel() for k, (fr, _) in enumerate(pool)])
    S = lil_matrix((2 * sum(sizes), len(x0)), dtype=int)
    row = 0
    for k, n in enumerate(sizes):
        S[row:row + 2 * n, :3] = 1
        S[row:row + 2 * n, 3 + 4 * k: 7 + 4 * k] = 1
        row += 2 * n
    r0 = float(np.median(np.abs(resid(x0))))
    sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=LINE_F_SCALE_PX, x_scale="jac", max_nfev=300)
    return {"status": "ok", "C": sol.x[:3].tolist(), "C0": C0.tolist(), "frames": len(pool),
            "reproj_px_before": round(r0, 2), "reproj_px_after": round(float(np.median(np.abs(sol.fun))), 2)}


# ---- 3. per-frame camera -----------------------------------------------------------------------------------------
def _grid_pts(H_px2ft, w, h):
    """Court grid points (ft) on frame under the free H, with their pixels: the free H's own correspondences."""
    from court.grid import GRID_FT
    P = np.linalg.inv(H_px2ft)
    ft = np.asarray(GRID_FT, np.float64)
    hom = np.column_stack([ft, np.ones(len(ft))]) @ P.T
    px = hom[:, :2] / hom[:, 2:3]
    ok = (hom[:, 2] > 0) & (px[:, 0] >= -w * .25) & (px[:, 0] <= w * 1.25) & (px[:, 1] >= -h * .25) & (px[:, 1] <= h * 1.25)
    return ft[ok], px[ok]


def fit_frame(C, w, h, ft, px, x0):
    from scipy.optimize import least_squares
    lo, hi = np.log(F_BOUNDS_PX[0]), np.log(F_BOUNDS_PX[1])
    x0 = np.array([*x0[:3], float(np.clip(x0[3], lo + 1e-3, hi - 1e-3))])
    fun = lambda x: (cam_project(C, x[:3], np.exp(x[3]), w, h, ft) - px).ravel()
    sol = least_squares(fun, x0, loss="soft_l1", f_scale=LINE_F_SCALE_PX, max_nfev=100,
                        bounds=([-np.inf] * 3 + [lo], [np.inf] * 3 + [hi]))
    return sol.x, float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)))


def derive(trace: dict, C) -> dict:
    """{window: [{frame, state, H, how, res_px}]}: camera-derived H for every frame the tracker kept."""
    C = np.asarray(C, np.float64)
    out = {}
    for w in sorted(trace):
        t, prev, rows = trace[w], None, []
        W, Hh = t["w"], t["h"]
        for r in t["rows"]:
            if r["H"] is None:
                rows.append({"frame": r["frame"], "state": "LOST", "H": None, "how": None}); prev = None
                continue
            Hf = np.array(r["H"]).reshape(3, 3)
            init = None
            if prev is not None:
                init = prev
            else:
                p = pose_from_h(Hf, W, Hh)
                if p is not None:
                    init = np.array([*p[0], np.log(p[1])])
            if init is None:         # free H is not a camera homography: start from the camera looking at court centre
                init = np.array([0.0, 0.0, 0.0, np.log(2000.0)])
            if len(r["ft"]) >= MIN_MATCH:
                how, ft, px = "lines", np.array(r["ft"]), np.array(r["px"])
            else:
                how, (ft, px) = "free_h", _grid_pts(Hf, W, Hh)
            if len(ft) < 4:
                rows.append({"frame": r["frame"], "state": "LOST", "H": None, "how": "no_points"}); prev = None
                continue
            x, res = fit_frame(C, W, Hh, ft, px, init)
            prev = x
            H = h_px2ft(C, x[:3], float(np.exp(x[3])), W, Hh)
            rows.append({"frame": r["frame"], "state": r["state"], "H": [float(v) for v in H.ravel()], "how": how,
                         "res_px": round(res, 2), "f": round(float(np.exp(x[3])), 1)})
        out[w] = rows
    return out


# ---- training-game check (R1.3 camera-truth labels) -------------------------------------------------------------
def _feet(boxes):
    return np.array([b["foot_stab"] for b in boxes], np.float64).reshape(-1, 2)


def _map(H, px):
    hom = np.column_stack([px, np.ones(len(px))]) @ np.asarray(H, np.float64).reshape(3, 3).T
    return hom[:, :2] / hom[:, 2:3]


def dev_errors(game: str, trace: dict, derived: dict) -> dict:
    """Court-induced position error on the training game's kept R1.3 label frames: every box's foot point mapped
    by the pipeline H vs by the camera-truth H. Near / far third and the bias toward the camera follow the
    truth camera's side. Returns {"free": [...], "camera": [...]} of (err_ft, toward_camera_ft, third)."""
    keep = {(r["window"], r["frame"]) for r in map(json.loads, (LABELS / (game + "_paint.jsonl")).read_text().splitlines()) if r["keep"]}
    out = {"free": [], "camera": []}
    for p in sorted((LABELS / game).glob("*.jsonl")):
        w = p.stem
        if w not in trace:
            continue
        side = {r["frame"]: r for r in json.loads((LABEL_BUILD / (w + "_frames.json")).read_text())["frames"]}
        free = {r["frame"]: r["H"] for r in trace[w]["rows"]}
        cam = {r["frame"]: r["H"] for r in derived[w]}
        for lab in map(json.loads, p.read_text().splitlines()):
            f = lab["frame"]
            if (w, f) not in keep or f not in side or not side[f].get("boxes"):
                continue
            px = _feet(side[f]["boxes"])
            tru = _map(lab["H_truth"], px)
            on = (tru[:, 0] > -2) & (tru[:, 0] < 96) & (tru[:, 1] > -2) & (tru[:, 1] < 52)
            if not on.any():
                continue
            cy = lab["camera_C"][1]
            toward = 1.0 if cy > 25 else -1.0
            near_line = 50.0 if cy > 25 else 0.0
            third = np.where(np.abs(tru[on, 1] - near_line) < 50 / 3, "near", np.where(np.abs(tru[on, 1] - near_line) > 100 / 3, "far", "mid"))
            for k, Hs in (("free", free), ("camera", cam)):
                if Hs.get(f) is None:
                    continue
                est = _map(Hs[f], px[on])
                e = est - tru[on]
                out[k] += [(float(np.hypot(*v)), float(v[1] * toward), str(t)) for v, t in zip(e, third)]
    return out


def summarize(rows: list) -> dict:
    if not rows:
        return {"n": 0}
    e = np.array([r[0] for r in rows]); b = {t: [r[1] for r in rows if r[2] == t] for t in ("near", "far")}
    return {"n": len(rows), "p50": round(float(np.percentile(e, 50)), 2), "p90": round(float(np.percentile(e, 90)), 2),
            **{"bias_" + t: (round(float(np.median(v)), 2) if v else None) for t, v in b.items()},
            **{"p50_" + t: (round(float(np.median([r[0] for r in rows if r[2] == t])), 2) if b[t] else None) for t in ("near", "far")}}


def dev(weights: Path, name: str, games: list | None = None) -> dict:
    from sportvu import splits
    games = games or splits.train_games()
    rep = {"weights": str(weights), "games": {}}
    for g in games:
        wins = sorted(p.stem for p in (LABELS / g).glob("*.jsonl") if (LABEL_BUILD / (p.stem + ".mp4")).exists())
        if not wins:
            continue
        tr = trace_game(name + "__" + g, wins, LABEL_BUILD, weights)
        pos = fit_position(tr)
        if pos["status"] != "ok":
            rep["games"][g] = {"position": pos}
            continue
        der = derive(tr, pos["C"])
        errs = dev_errors(g, tr, der)
        truth_C = np.median([json.loads(l)["camera_C"] for p in (LABELS / g).glob("*.jsonl") for l in p.read_text().splitlines()[:5]], axis=0)
        hows = [r["how"] for rows in der.values() for r in rows if r["H"] is not None]
        rep["games"][g] = {"windows": len(wins), "position": pos, "truth_C": truth_C.round(1).tolist(),
                           "C_minus_truth_ft": (np.array(pos["C"]) - truth_C).round(1).tolist(),
                           "fit_from": {h: hows.count(h) for h in set(hows)},
                           "free": summarize(errs["free"]), "camera": summarize(errs["camera"])}
        print(g, json.dumps({k: rep["games"][g][k] for k in ("C_minus_truth_ft", "fit_from", "free", "camera")}), flush=True)
    return rep


def heldout(weights: Path, name: str, games: list | None = None, card_dir: Path | None = None) -> dict:
    """Score the camera model on the held-out games: trace with V3's decode offsets, self-calibrate C per game,
    write a stand-in build (V3's sidecars, H and state replaced) and run sportvu.bench on it."""
    import shutil
    from sportvu import bench
    from sportvu.court_replay import BUILD_DIR, REPLAY_DIR, heldout_games, windows_of, load_offsets
    games = games or heldout_games()
    out = REPLAY_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    rep = {"weights": str(weights), "games": {}}
    for g in games:
        wins = windows_of(g)
        offs = load_offsets(wins)
        tr = trace_game(name + "__" + g, wins, BUILD_DIR, weights, offs)
        pos = fit_position(tr)
        der = derive(tr, pos["C"]) if pos["status"] == "ok" else {}
        shutil.copy(BUILD_DIR / (g + "_windows.json"), out / (g + "_windows.json"))
        for w in wins:
            side = json.loads((BUILD_DIR / (w + "_frames.json")).read_text())
            d = {r["frame"]: r for r in der.get(w, [])}
            for r in side["frames"]:
                r["H"], r["state"] = (d[r["frame"]]["H"], d[r["frame"]]["state"]) if r["frame"] in d else (None, "LOST")
                r.pop("accepted", None)
            side["replay"] = {"weights": str(weights), "camera_C": pos.get("C"), "offsets": offs[w]}
            (out / (w + "_frames.json")).write_text(json.dumps(side))
            ident = BUILD_DIR / (w + "_identity.json")
            if ident.exists():
                shutil.copy(ident, out / ident.name)
        card = bench.score_build(out, g, name, card_dir or bench.SCORECARD_DIR, verbose=False)
        pe, nf, cv = (card["metrics"][k] for k in ("position_error", "near_field_bias", "coverage"))
        hows = [r["how"] for rows in der.values() for r in rows if r["H"] is not None]
        rep["games"][g] = {"position": pos, "fit_from": {h: hows.count(h) for h in set(hows)}}
        print("%s %s: C %s | position p50 %s p90 %s | near %s far %s | coverage %s" % (
            name, g, np.round(pos.get("C", []), 1).tolist(), pe["p50_ft"], pe["p90_ft"], nf["near_third_ft"], nf["far_third_ft"], cv["coverage"]), flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "heldout"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--card-dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "dev":
        rep = dev(a.weights.resolve(), a.name, a.games)
        (REPORT_DIR / "camera_track_dev.json").write_text(json.dumps(rep, indent=1))
    else:
        heldout(a.weights.resolve(), a.name, a.games, a.card_dir)


if __name__ == "__main__":
    main()

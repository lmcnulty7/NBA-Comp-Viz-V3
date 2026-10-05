"""
sportvu/truth.py: per-frame truth homography from pixel feet <-> SportVU positions (FIX_PLAN B4).

For each rebuilt window frame with a pipeline H and enough boxes: (1) project the stabilized feet
through the pipeline H (mirror from B3 applied), (2) Hungarian-match them to the SportVU players at
the frame's clock (gate MATCH_GATE_FT), (3) RANSAC findHomography from the matched feet (px) to the
SportVU positions (ft) at RANSAC_FT; accept H_truth with >= MIN_INLIERS inliers. Frames with fewer
are "untestable" and counted. Caveat on record: H_truth passes through the pipeline's own boxes, so
a frame where every box is wrong has no truth, and the matching itself leans on the pipeline H
(a grossly wrong H can fail to match and becomes untestable rather than measured).

  python -m sportvu.truth gsw_phx_2016 12.16.2015.PHX.at.GSW
      -> data/sportvu/truth/<window>_truth.json, reports/sportvu_truth_<game>.json
"""
from __future__ import annotations
import json, sys
import numpy as np
import config
from sportvu.sync import SportVUIndex, SYNC_DIR, apply_mirror, MIRRORS
from sportvu.rebuild import BUILD_DIR

TRUTH_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "truth"
GATES_FT = (20.0, 12.0, 8.0, 5.0)   # ICP gates, loose to tight
RANSAC_FT = 1.5
MIN_INLIERS = 6
MIN_BOXES = 6
CLOCK_TOL_S = 0.12
OFFSET_MAX, OFFSET_STEP = 1.5, 0.04


def frame_clock(window: dict, win_tm: dict, frame_in_window: int):
    """(period, clock) for a window frame via the WINDOW's own time map (linear inside the span)."""
    for s in win_tm["spans"]:
        if s["f0"] <= frame_in_window <= s["f1"]:
            u = (frame_in_window - s["f0"]) / max(s["f1"] - s["f0"], 1)
            return s["period"], s["c0"] + u * (s["c1"] - s["c0"])
    return None, None


def solve_window_offset(rows_pc: list, index: SportVUIndex, mirror: str) -> tuple[float, float]:
    """Residual clock offset for one window (the OCR clock is 1 s resolution): the offset in
    OFFSET_STEP steps over +-OFFSET_MAX that minimises the median nearest-SportVU-player distance
    of the pipeline's projected feet. Returns (offset_s, median_ft)."""
    fx, fy = MIRRORS[mirror]
    best = (0.0, np.inf)
    for off in np.arange(-OFFSET_MAX, OFFSET_MAX + 1e-9, OFFSET_STEP):
        med = []
        for q, c, p in rows_pc:
            i = index.at(q, c + off, CLOCK_TOL_S)
            if i is None:
                continue
            d = np.sqrt(((apply_mirror(p, fx, fy)[:, None, :] - index.q[q]["xy"][i][None]) ** 2).sum(-1)).min(1)
            med.append(float(np.median(d)))
        if med and np.median(med) < best[1]:
            best = (round(float(off), 2), round(float(np.median(med)), 2))
    return best


def _match(court: np.ndarray, sv_xy: np.ndarray, gate: float):
    from scipy.optimize import linear_sum_assignment
    D = np.sqrt(((court[:, None, :] - sv_xy[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(D)
    return [(int(i), int(j), float(D[i, j])) for i, j in zip(ri, ci) if D[i, j] <= gate]


def truth_for_frame(row: dict, sv_xy: np.ndarray, sv_pid: np.ndarray, mirror: str) -> dict:
    """ICP-style: Hungarian-match the feet projected through the CURRENT H to SportVU at a gate,
    refit H by RANSAC on the pairs, tighten the gate, repeat (GATES_FT). The first H is the
    pipeline's. A systematically stretched pipeline H (near-field overshoot) converges in 2..3
    steps; a grossly wrong one never gets MIN_INLIERS pairs and stays untestable."""
    import cv2
    fx, fy = MIRRORS[mirror]
    feet_px = np.array([b["foot_stab"] for b in row["boxes"]], np.float32)
    if len(feet_px) < MIN_BOXES or row["H"] is None:
        return {"status": "untestable_boxes" if len(feet_px) < MIN_BOXES else "untestable_noH", "n_boxes": int(len(feet_px))}
    sv = apply_mirror(sv_xy, fx, fy)             # SportVU into the pipeline's court frame
    H = np.array(row["H"], np.float64).reshape(3, 3)
    pairs, Ht, mask, pipeline_d = [], None, None, None
    for k, gate in enumerate(GATES_FT):
        court = cv2.perspectiveTransform(feet_px.reshape(-1, 1, 2), H).reshape(-1, 2)
        pairs = _match(court, sv, gate)
        if k == 0:
            pipeline_d = {i: d for i, _, d in pairs}
        if len(pairs) < MIN_INLIERS:
            return {"status": "untestable_match", "n_boxes": int(len(feet_px)), "n_matched": len(pairs), "iter": k}
        src = np.array([feet_px[i] for i, _, _ in pairs], np.float32).reshape(-1, 1, 2)
        dst = np.array([sv[j] for _, j, _ in pairs], np.float32).reshape(-1, 1, 2)
        Ht, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_FT if k == len(GATES_FT) - 1 else gate / 2)
        if Ht is None or int(mask.sum()) < MIN_INLIERS:
            return {"status": "untestable_inliers", "n_boxes": int(len(feet_px)), "n_matched": len(pairs),
                    "inliers": 0 if mask is None else int(mask.sum()), "iter": k}
        H = Ht
    proj = cv2.perspectiveTransform(src, Ht).reshape(-1, 2)
    res = np.sqrt(((proj - dst.reshape(-1, 2)) ** 2).sum(-1))
    inl = int(mask.sum())
    # d_pipeline_ft: the PIPELINE H's error for each finally-matched box (what B5 reports)
    court0 = cv2.perspectiveTransform(feet_px.reshape(-1, 1, 2), np.array(row["H"], np.float64).reshape(3, 3)).reshape(-1, 2)
    return {"status": "ok", "n_boxes": int(len(feet_px)), "n_matched": len(pairs), "inliers": inl,
            "H_truth": [float(v) for v in Ht.ravel()], "resid_ft": round(float(np.median(res[mask.ravel() > 0])), 3),
            "pairs": [{"box": i, "pid": int(sv_pid[j]), "d_pipeline_ft": round(float(np.hypot(*(court0[i] - sv[j]))), 2),
                       "inlier": bool(mask[k])} for k, (i, j, d) in enumerate(pairs)]}


def window_truth(w: dict, sc: dict, tm: dict, index: SportVUIndex, mirror: str) -> tuple:
    """(offset_s, offset_median_ft, rows) for one rebuilt window: the residual clock offset solved on
    the pipeline's projected feet, then truth_for_frame on every sidecar frame (status per frame)."""
    import cv2
    rows_pc = []
    for r in sc["frames"]:
        q, clock = frame_clock(w, tm, r["frame"])
        if q is None or r["H"] is None or len(r["boxes"]) < 4:
            continue
        feet = np.array([b["foot_stab"] for b in r["boxes"]], np.float32)
        rows_pc.append((q, clock, cv2.perspectiveTransform(feet.reshape(-1, 1, 2), np.array(r["H"]).reshape(3, 3)).reshape(-1, 2)))
    off, off_ft = solve_window_offset(rows_pc, index, mirror) if len(rows_pc) >= 20 else (0.0, None)
    rows = []
    for r in sc["frames"]:
        q, clock = frame_clock(w, tm, r["frame"])
        if q is None:
            rows.append({"frame": r["frame"], "status": "unmapped"}); continue
        clock = clock + off
        i = index.at(q, clock, CLOCK_TOL_S)
        if i is None:
            rows.append({"frame": r["frame"], "status": "no_moment", "q": q, "clock": round(clock, 2)}); continue
        t = truth_for_frame(r, index.q[q]["xy"][i], index.q[q]["pid"][i], mirror)
        t.update({"frame": r["frame"], "q": q, "clock": round(clock, 2), "moment": int(i), "state": r["state"]})
        rows.append(t)
    return off, off_ft, rows


def game_mirror(game: str) -> str:
    """The game's SportVU mirror: the phx sync report (B3), else the R1.2 direction report."""
    sync_p = config.REPORTS_DIR / ("sportvu_sync_%s.json" % game)
    if sync_p.exists():
        return json.loads(sync_p.read_text())["direction_resolution"]["mirror"]
    return json.loads((config.REPORTS_DIR / ("sportvu_direction_%s.json" % game)).read_text())["resolution"]["mirror"]


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "gsw_phx_2016"
    sv_game = sys.argv[2] if len(sys.argv) > 2 else "12.16.2015.PHX.at.GSW"
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    index = SportVUIndex(json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (sv_game + "_moments.json")).read_text())["moments"])
    mirror = game_mirror(game)
    windows = json.loads((BUILD_DIR / (game + "_windows.json")).read_text())
    rep = {"game": game, "sportvu_game": sv_game, "mirror": mirror, "windows": {}, "status_counts": {}}
    for w in windows:
        side = BUILD_DIR / (w["window"] + "_frames.json")
        if not side.exists():
            continue
        sc = json.loads(side.read_text()); tm = json.loads((BUILD_DIR / (w["window"] + "_timemap.json")).read_text())
        off, off_ft, rows = window_truth(w, sc, tm, index, mirror)
        (TRUTH_DIR / (w["window"] + "_truth.json")).write_text(json.dumps({"window": w, "offset_s": off, "offset_median_ft": off_ft, "frames": rows}))
        cnt = {}
        for r in rows:
            cnt[r["status"]] = cnt.get(r["status"], 0) + 1
            rep["status_counts"][r["status"]] = rep["status_counts"].get(r["status"], 0) + 1
        ok = [r for r in rows if r["status"] == "ok"]
        rep["windows"][w["window"]] = {"frames": len(rows), "offset_s": off, "offset_median_ft": off_ft, **cnt,
                                       "resid_ft_median": round(float(np.median([r["resid_ft"] for r in ok])), 3) if ok else None,
                                       "inliers_median": float(np.median([r["inliers"] for r in ok])) if ok else None}
        print("  %-24s %s" % (w["window"], rep["windows"][w["window"]]), flush=True)
    (config.REPORTS_DIR / ("sportvu_truth_%s.json" % game)).write_text(json.dumps(rep, indent=1))
    print("status counts:", rep["status_counts"])


if __name__ == "__main__":
    main()

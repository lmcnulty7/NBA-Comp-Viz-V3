"""
qc/track_qc.py: label-free quality checks on one build (FIX_PLAN Phase A).

Inputs are what build_trajectories already writes: the <clip>_frames.json sidecar (A0) and the
source video. Every check is a pure function of (frame image, sidecar row) so the same code runs
on a triage snippet locally and on a harvest section in Colab. Nothing here changes the pipeline.

A1  line_overlay(frame, H, boxes): how far the detected line ridges sit from the projected
    court template. Ridge pixels (same ridge_field as the solver, peak >= PEAK_MIN) outside every
    player box are measured against a distance transform of the template drawn through H.
    Reported: dist_px = median distance of ridge pixels to the nearest template line; score =
    share of ridge pixels within NEAR_PX of a template line; support = share of on-frame template
    samples with a ridge within NEAR_PX along their normal (template side). A right H reads a few
    px; a 20 px offset reads ~20; a wrong H reads 60+ with score < 0.08.
    Calibration note (2026-10-03, 55 triage renders): two template-side variants (hit rate at 3 px,
    nearest-ridge distance at 40 px) saturated at ~0.02 and ~17 px for every clip, because most
    template samples have no visible line nearby on 720p footage and production H's are typically
    10..40 px off the paint. Measuring from the ridge side separates clips (27 px good, 70..90 px
    for the clips a human called off). Non-line ridges (logos, crowd remnants, player edges) inflate
    the absolute scale per arena; use it as a ranking and threshold per arena until B6 calibrates it.
A2  geometry(frame, H, boxes, feet, bug_rect): three rules on the projected court.
    (a) scorebug: the far sideline or either baseline crosses the scorebug rectangle (union of the
        layout's clock and period boxes, padded BUG_PAD px). (b) floor mask: HSV maple band from
        gate/hsv_baseline.py; points 2 ft inside each sideline and baseline that are on frame must
        be >= FLOOR_INSIDE_MIN floor, points 6 ft outside the near sideline must be <= FLOOR_OUTSIDE_MAX
        floor. Arena caveat: the band is maple; OKC's blue paint and Cleveland's wine key read as
        "not floor", so the inside rule can fail on a right H there (reported per game, not hidden).
    (c) off-court feet: more than OFFCOURT_MAX boxes whose stabilized foot maps outside the court
        by more than OFFCOURT_FT (one is allowed: a ref or coach on the apron).
A3  physics(trajectories, sidecar, identity, fps, stride): per processed frame, (a) any track
    moving faster than MAX_SPEED_FTS between consecutive processed frames (raw court positions,
    pre-clean); (b) more than MAX_PER_TEAM boxes of one team on the frame (canonical ids, teams
    from the identity audit); (c) a camera cut (the pipeline's own rule: >= CUT_VANISH of the
    previous frame's ids gone, at least CUT_MIN_TRACKS of them) across which a kept id jumps more
    than CUT_JUMP_FT. Counts per rule per clip; frames carry `fails`.
A4  second-tracker disagreement (qc/second_tracker.py): COCO yolov8m person + ByteTrack over the
    same frames; per frame 1 - mutual IoU>=0.5 matches / max(n_pipeline, n_second); a frame fails
    when the disagreement exceeds SECOND_MAX. Boxes only the pipeline has are ghost candidates,
    boxes only the second tracker has are miss candidates (either tracker can be the wrong one).
"""
from __future__ import annotations

import numpy as np

from court.snap_track import CH_A, CH_B, CH_MID, match_lines, project, ridge_field

BUG_PAD = 20                  # px around the clock+period boxes taken as the scorebug rectangle
FLOOR_INSIDE_MIN = 0.70       # share of 2 ft-inside samples that must be floor coloured
FLOOR_OUTSIDE_MAX = 0.40      # share of 6 ft-outside-near-sideline samples that may be floor coloured
FLOOR_LO, FLOOR_HI = (8, 25, 90), (38, 210, 245)   # HSV maple band, gate/hsv_baseline.py defaults
OFFCOURT_FT = 3.0             # feet outside the court lines before a foot counts as off court
OFFCOURT_MAX = 1              # more off-court feet than this flags the frame
MIN_EDGE_SAMPLES = 8          # an edge with fewer on-frame samples is not judged
MAX_SPEED_FTS = 30.0          # ft/s; nobody on an NBA floor sustains this between two 10 Hz samples
MAX_PER_TEAM = 5              # boxes per team on a frame beyond this = ghosts or a team flip
CUT_VANISH, CUT_MIN_TRACKS = 0.80, 4   # detect/camera_cut.py's rule, replayed from the sidecar
CUT_JUMP_FT = 15.0            # a kept id moving this far across a cut is a stale id on a new player
SECOND_MAX = 0.30             # frame disagreement with the independent tracker above this fails
NEAR_PX = 6                   # within this of a template line counts as "on the line"
BOX_PAD = 4                   # px grown around player boxes before removing their ridge pixels
MIN_RIDGE_PX = 500            # fewer ridge pixels than this: score undefined (no line evidence in frame)
MIN_SAMPLES = 40              # fewer on-frame template samples than this: support undefined
LINE_SUPPORT_PX = 6           # template-side support: ridge peak within this along the normal


def _inside_boxes(pts: np.ndarray, boxes: list) -> np.ndarray:
    """Boolean mask: pixel point lies inside any xyxy box."""
    inside = np.zeros(len(pts), bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
    return inside


def line_overlay(frame_bgr: np.ndarray, H_px2ft, boxes: list, ridge=None) -> dict:
    """A1 for one frame: {score, dist_px, support, n_ridge, n_sampled}; None fields when undefined."""
    import cv2
    from court.snap_track import PEAK_MIN, _bilinear

    h, w = frame_bgr.shape[:2]
    H = np.asarray(H_px2ft, np.float64).reshape(3, 3)
    try:
        P = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}
    if ridge is None:
        ridge = ridge_field(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    if ridge is None:
        return {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}
    R = (ridge > PEAK_MIN).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        R[max(0, y1 - BOX_PAD):y2 + BOX_PAD, max(0, x1 - BOX_PAD):x2 + BOX_PAD] = 0
    n_ridge = int(R.sum())
    pa, pb = project(P, CH_A), project(P, CH_B)
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    ok &= (np.abs(pa) < 8 * max(w, h)).all(1) & (np.abs(pb) < 8 * max(w, h)).all(1)
    T = np.full((h, w), 255, np.uint8)
    for a, b in zip(pa[ok], pb[ok]):
        cv2.line(T, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 0, 1)
    if n_ridge < MIN_RIDGE_PX or (T == 0).sum() == 0:
        return {"score": None, "dist_px": None, "support": None, "n_ridge": n_ridge, "n_sampled": 0}
    dt = cv2.distanceTransform(T, cv2.DIST_L2, 3)
    d = dt[R > 0]
    # template-side support: on-frame samples outside boxes with a ridge peak within LINE_SUPPORT_PX
    mid = (pa + pb) / 2.0
    on = ok & (mid[:, 0] >= 1) & (mid[:, 0] < w - 1) & (mid[:, 1] >= 1) & (mid[:, 1] < h - 1)
    on &= ~_inside_boxes(mid, boxes)
    tang = pb - pa
    tlen = np.linalg.norm(tang, axis=1)
    on &= tlen > 1e-3
    idx = np.where(on)[0]
    support = None
    if len(idx) >= MIN_SAMPLES:
        tg = tang[idx] / tlen[idx, None]
        nrm = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
        Tn = np.arange(-LINE_SUPPORT_PX, LINE_SUPPORT_PX + 1, dtype=np.float32)
        resp = _bilinear(ridge, mid[idx][:, None, :] + Tn[None, :, None] * nrm[:, None, :])
        support = round(float((resp >= PEAK_MIN).any(axis=1).mean()), 4)
    return {"score": round(float((d <= NEAR_PX).mean()), 4),
            "dist_px": round(float(np.median(d)), 2),
            "support": support, "n_ridge": n_ridge, "n_sampled": int(len(idx))}


def _seg_hits_rect(a, b, rect, w, h) -> bool:
    """Does pixel segment a-b cross axis-aligned rect (x1, y1, x2, y2)? Sampled, after clipping to frame."""
    x1, y1, x2, y2 = rect
    n = int(max(2, np.hypot(b[0] - a[0], b[1] - a[1]) / 4))
    ts = np.linspace(0, 1, n)
    xs, ys = a[0] + ts * (b[0] - a[0]), a[1] + ts * (b[1] - a[1])
    on = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return bool(((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2) & on).any())


def _floor_share(mask, pts, w, h):
    """Share of pixel points (on frame) that are floor coloured; None if too few on frame."""
    ok = np.isfinite(pts).all(1)
    ok &= (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if ok.sum() < MIN_EDGE_SAMPLES:
        return None
    p = pts[ok].astype(int)
    return round(float(mask[p[:, 1], p[:, 0]].mean()), 4)


def geometry(frame_bgr: np.ndarray, H_px2ft, boxes: list, feet: list, bug_rect=None) -> dict:
    """A2 for one frame. feet = stabilized foot pixels, one per box. Returns the three rules'
    measurements and `fails` (list of rule names that fired)."""
    import cv2
    from court.court33 import COURT_LENGTH_FT as L, COURT_WIDTH_FT as W

    h, w = frame_bgr.shape[:2]
    H = np.asarray(H_px2ft, np.float64).reshape(3, 3)
    try:
        P = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return {"fails": ["no_H"]}
    out, fails = {}, []
    # edges in court ft: sidelines y=0 and y=W, baselines x=0 and x=L
    xs = np.arange(0, L + 1e-6, 2.0); ys = np.arange(0, W + 1e-6, 2.0)
    side = {"y0": np.stack([xs, np.zeros_like(xs)], 1), "yW": np.stack([xs, np.full_like(xs, W)], 1)}
    base = {"x0": np.stack([np.zeros_like(ys), ys], 1), "xL": np.stack([np.full_like(ys, L), ys], 1)}
    side_px = {k: project(P, v) for k, v in side.items()}
    far_key = min(side_px, key=lambda k: np.nanmean(side_px[k][:, 1]))   # smaller y = higher = far
    near_key = "yW" if far_key == "y0" else "y0"
    out["far_sideline"] = far_key
    # (a) scorebug
    if bug_rect is not None:
        hit = False
        for k, pts in list(side_px.items()) + [(k, project(P, v)) for k, v in base.items()]:
            if k == near_key:
                continue
            for a, b in zip(pts[:-1], pts[1:]):
                if np.isfinite([a, b]).all() and _seg_hits_rect(a, b, bug_rect, w, h):
                    hit = True; break
            if hit:
                break
        out["scorebug_hit"] = hit
        if hit:
            fails.append("scorebug")
    # (b) floor mask
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(FLOOR_LO, np.uint8), np.array(FLOOR_HI, np.uint8)) > 0
    inside = {
        "far_in": side[far_key] + np.array([0, 2.0 if far_key == "y0" else -2.0]),
        "near_in": side[near_key] + np.array([0, 2.0 if near_key == "y0" else -2.0]),
        "x0_in": base["x0"] + np.array([2.0, 0]), "xL_in": base["xL"] + np.array([-2.0, 0]),
    }
    near_out = side[near_key] + np.array([0, -6.0 if near_key == "y0" else 6.0])
    floor = {k: _floor_share(mask, project(P, v), w, h) for k, v in inside.items()}
    floor["near_out"] = _floor_share(mask, project(P, near_out), w, h)
    out["floor"] = floor
    low = [k for k, v in floor.items() if k != "near_out" and v is not None and v < FLOOR_INSIDE_MIN]
    # Rule (b) is MEASURED but does not fail a frame (2026-10-03, pass 3): the maple band misses
    # bright Oracle floors (V saturates at 255) and OKC's blue paint, a texture variant reads the
    # sideline itself as texture, and an adaptive per-frame colour band still put the near
    # sideline of the BEST H's in the crowd (the near-field overshoot). Shares are kept for B6.
    out["floor_low_edges"] = low
    out["floor_uncalibrated"] = True
    # (c) off-court feet
    n_off = 0
    if feet:
        c = cv2.perspectiveTransform(np.asarray(feet, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
        n_off = int(((c[:, 0] < -OFFCOURT_FT) | (c[:, 0] > L + OFFCOURT_FT) |
                     (c[:, 1] < -OFFCOURT_FT) | (c[:, 1] > W + OFFCOURT_FT) | ~np.isfinite(c).all(1)).sum())
    out["n_offcourt"] = n_off
    if n_off > OFFCOURT_MAX:
        fails.append("offcourt_feet")
    out["fails"] = fails
    return out


def physics(traj: dict, sidecar: dict, identity: dict, fps: float, stride: int) -> dict:
    """A3 over one build. Returns {"frames": {frame: {...}}, "counts": {...}}."""
    idmap = {int(k): int(v) for k, v in identity.get("idmap", {}).items()}
    team = {int(k): v for k, v in identity.get("team_by_track", {}).items()}
    dt = stride / fps
    per_frame = {r["frame"]: {"fails": [], "speed_max": 0.0, "speed_max_raw": 0.0, "team_counts": {}, "cut": False} for r in sidecar["frames"]}
    # (a) speeds between consecutive processed frames: cleaned (failing rule) and raw (measurement)
    for key, field in (("cleaned", "speed_max"), ("raw", "speed_max_raw")):
        for tid, tr in traj.items():
            pts = sorted(p[:3] for p in tr.get(key, []))
            for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
                if f1 - f0 != stride:
                    continue
                v = float(np.hypot(x1 - x0, y1 - y0)) / dt
                row = per_frame.get(f1)
                if row is not None:
                    row[field] = max(row[field], v)
    # (b) per-team counts and (c) cuts, replayed over the sidecar boxes
    prev_ids, prev_pos = set(), {}
    pos_by_frame = {}
    for tid, tr in traj.items():
        for f, x, y in tr.get("raw", []):
            pos_by_frame.setdefault(f, {})[int(tid)] = (x, y)
    for r in sidecar["frames"]:
        f = r["frame"]; row = per_frame[f]
        ids = {idmap.get(b["tid"], b["tid"]) for b in r["boxes"]}
        cnt = {}
        for cid in ids:
            t = team.get(cid)
            if t is not None:
                cnt[str(t)] = cnt.get(str(t), 0) + 1
        row["team_counts"] = cnt
        if any(c > MAX_PER_TEAM for c in cnt.values()):
            row["fails"].append("team_count")
        if row["speed_max"] > MAX_SPEED_FTS:
            row["fails"].append("speed")
        cut = len(prev_ids) >= CUT_MIN_TRACKS and len(prev_ids - ids) / len(prev_ids) >= CUT_VANISH
        row["cut"] = bool(cut)
        if cut:
            cur = pos_by_frame.get(f, {})
            jumped = [cid for cid in ids & prev_ids if cid in cur and cid in prev_pos
                      and np.hypot(cur[cid][0] - prev_pos[cid][0], cur[cid][1] - prev_pos[cid][1]) > CUT_JUMP_FT]
            if jumped:
                row["fails"].append("cut_jump")
                row["cut_jumped_ids"] = jumped
        prev_ids, prev_pos = ids, pos_by_frame.get(f, {})
    counts = {}
    for row in per_frame.values():
        for k in row["fails"]:
            counts[k] = counts.get(k, 0) + 1
    counts["cuts"] = sum(1 for row in per_frame.values() if row["cut"])
    return {"frames": per_frame, "counts": counts}


def score_build(video_path, sidecar: dict, checks=("overlay", "geometry"), bug_rect=None) -> list[dict]:
    """Run the per-frame checks over one build. Reads the video sequentially (never seeks)."""
    import cv2

    rows = []
    cap = cv2.VideoCapture(str(video_path))
    want = {r["frame"]: r for r in sidecar["frames"]}
    idx = 0
    while want:
        ret, frame = cap.read()
        if not ret:
            break
        r = want.pop(idx, None)
        if r is not None:
            out = {"frame": idx, "state": r["state"]}
            boxes = [b["bbox"] for b in r["boxes"]]
            if "overlay" in checks:
                out["overlay"] = line_overlay(frame, r["H"], boxes) if r["H"] is not None else {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}
            if "geometry" in checks:
                feet = [b["foot_stab"] for b in r["boxes"]]
                out["geometry"] = geometry(frame, r["H"], boxes, feet, bug_rect) if r["H"] is not None else {"fails": ["no_H"]}
            rows.append(out)
        idx += 1
    cap.release()
    for f in sorted(want):                      # frames the video ran out before reaching
        rows.append({"frame": f, "state": want[f]["state"], "overlay": {"score": None, "dist_px": None, "support": None, "n_ridge": 0, "n_sampled": 0}, "unread": True})
    return rows


def summarize(rows: list[dict], far: float = 40.0) -> dict:
    """Per-clip A1 summary over frames with an H: median dist_px, p90, median score and support,
    and share_far = share of frames whose lines sit more than `far` px from the template."""
    ov = [r["overlay"] for r in rows if r.get("overlay", {}).get("dist_px") is not None]
    dd = [o["dist_px"] for o in ov]; sc = [o["score"] for o in ov]; su = [o["support"] for o in ov if o["support"] is not None]
    has_h = sum(1 for r in rows if r["state"] != "LOST")
    rules = {}
    for r in rows:
        for f in r.get("geometry", {}).get("fails", []) + r.get("physics", {}).get("fails", []) + r.get("second", {}).get("fails", []):
            rules[f] = rules.get(f, 0) + 1
    return {"frames": len(rows), "frames_with_H": has_h, "frames_scored": len(ov),
            "rule_counts": rules,
            "frames_failing_any": sum(1 for r in rows if r.get("geometry", {}).get("fails") or r.get("physics", {}).get("fails") or r.get("second", {}).get("fails")),
            "second_disagreement_median": (lambda d: round(float(np.median(d)), 4) if d else None)([r["second"]["disagreement"] for r in rows if "second" in r]),
            "dist_px_median": round(float(np.median(dd)), 2) if dd else None,
            "dist_px_p90": round(float(np.percentile(dd, 90)), 2) if dd else None,
            "score_median": round(float(np.median(sc)), 4) if sc else None,
            "support_median": round(float(np.median(su)), 4) if su else None,
            "share_far": round(sum(d > far for d in dd) / len(dd), 4) if dd else None}

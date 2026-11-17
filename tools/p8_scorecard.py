"""tools/p8_scorecard.py: scorecard v2 for P8 (docs/research/P8_FEASIBILITY.md sections 2.4, 6 and 7 (4)).

    python -m tools.p8_scorecard game TAG --build r23 [--truth camera|camera_paint|b4]
    python -m tools.p8_scorecard combine --build r23 [--truth camera] [--tags a,b,...]
    python -m tools.p8_scorecard gate                  Phase B gate item 1: r23 camera vs b4 truth on the gate games
    python -m tools.p8_scorecard fix-ball-tolerance    once, before any ball model: reports/research/p8_constants.json

New code that imports the frozen scorer (sportvu/bench.py) and changes none of it. Per development game it scores a
build of the game's label windows against SportVU on every live wide frame whose camera truth was accepted:

  universe   live (the window's own OCR time map) and wide (gate v2 at its v2 threshold, bench.References) frames
             with an accepted camera-truth label (sportvu.autolabel's rules, run by tools.p8_frozen fit). The share
             of live wide frames that carry truth is printed beside every row. --truth camera_paint keeps only the
             labels the GSW gold-key paint check kept (GSW games only); --truth b4 uses the frozen B4 ICP truth
             (sportvu.truth.window_truth on the V3 label build), the old truth, for the Phase A gate comparison.
  frozen     each window is reduced to the per-window record of sportvu.bench.score_window and the rows are computed
  rows       by sportvu.bench.aggregate itself: position error, near-field bias, coverage, missed players, ghost
             boxes, team labels, identity (and the court line row, which the frozen scorer withholds).
  new rows   section 6.2 with its constants fixed below before any result: ball in the image, ball carrier,
             identity, identity (reported), shots, rebounds, shooter and rebounder position. A row the build gives no
             input for (no <window>_ball.json, no roster per track, no <window>_events.json) says "not measurable"
             and prints the truth side's n.
  breakdowns per observation: crowding (distance to the nearest other SportVU player), speed, court zone, camera
             third and time since the last camera cut; missed rate and position error per bin, ghost rate per third.
  intervals  window-cluster bootstrap (BOOT_N, seed BOOT_SEED) beside each headline rate and percentile; the
             combine command adds the game bootstrap.

The truth for events is the contract's play-by-play (basketball-reference), used as scoring truth only, never as a
training label (D84). Outputs go under the P8 workspace: scorecards/<tag>__<build>__<truth>.{json,txt} and
scorecards/obs/<tag>__<build>__<truth>.jsonl. tests/test_p8_scorecard.py pins this file, its constants and the
frozen files it imports by sha256 (reports/research/p8_scorecard_pin.json): a branch that changes any of them fails.
"""

import argparse
import datetime
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_scorecard"

# ── constants, fixed 2026-11-17 before any scorecard v2 result existed ──────────────────────────────────────────────
CONSTANTS = {
    # section 6.2 (docs/research/P8_FEASIBILITY.md), as written there
    "ball_f1_target": 0.75,
    "ball_tol_rule": "2 x the median camera-truth reprojection error of SportVU feet onto the box feet (px), over the development games, fixed by the Phase A baseline before any ball model runs",
    "holder_max_ft": 3.0,                 # one player within 3 ft of the ball ...
    "holder_ball_max_z_ft": 6.0,          # ... and the ball below 6 ft
    "carrier_target": 0.85,
    "identity_target": 0.80,
    "jersey_min_crops": 8,
    "switches_goal_per_possession": 5.0,
    "event_tol_s": 1.0,
    "event_live_wide_s": 0.5,             # an event counts when a live wide frame of its period lies within 0.5 s of it
    "shots_found_target": 0.95,
    "shots_false_max": 0.05,
    "rebounds_found_target": 0.90,
    "shooter_p50_ft": 3.0, "shooter_p90_ft": 6.0,
    "rebounder_p50_ft": 4.0,
    # breakdowns (section 2.4); bins are fixed here and never tuned
    "speed_dt_s": 0.2,                    # central difference over +-0.2 s of SportVU moments (25 Hz)
    "crowd_bins_ft": [3.0, 6.0, 10.0],
    "speed_bins_ftps": [3.0, 8.0, 14.0],
    "zone_bins_ft": [4.0, 16.0, 23.75],   # distance to the nearest rim: restricted, paint and short, mid, three
    "rim_x_ft": [5.25, 88.75], "rim_y_ft": 25.0,
    "cut_hist_corr": 0.6,                 # a cut: HSV histogram correlation below this between consecutive stride frames
    "cut_hist_bins": [30, 32], "cut_scale_px": [160, 90],
    "since_cut_bins_s": [1.0, 3.0, 10.0],
    "boot_n": 1000, "boot_seed": 0,
    "truth_modes": ["camera", "camera_paint", "b4"],
    "gate_games": ["gsw_bkn_2015", "nyk_bos_2016"],   # the b4 old-truth comparison: one R1.3 and one channel game
}
BOOT_N, BOOT_SEED = CONSTANTS["boot_n"], CONSTANTS["boot_seed"]
TARGETS_V2 = {
    "ball_in_image": "F1 >= 0.75 (tolerance: %s)" % CONSTANTS["ball_tol_rule"],
    "ball_carrier": ">= 85% of live frames with a SportVU holder",
    "identity_v2": ">= 80% of matched player-frames name the SportVU player",
    "identity_reported": "jersey tracklet accuracy on tracklets with >= 8 crops; goal <= 5 id switches per possession",
    "shots": ">= 95% of field-goal attempts found within 1.0 s; <= 5% false events per true event",
    "rebounds": ">= 90% found within 1.0 s",
    "shooter_position": "p50 <= 3.0 ft, p90 <= 6.0 ft",
    "rebounder_position": "p50 <= 4.0 ft",
    "swap_test": "per lane, set by the lane's owner before the run (D3); not part of this scorer",
}


# ── pure helpers (unit tested) ──────────────────────────────────────────────────────────────────────────────────────
def bin_of(value, edges, labels=None):
    """The bin label of a value: '<e0', 'e0-e1', ..., '>=en'; None for None."""
    if value is None or not np.isfinite(value):
        return None
    for k, e in enumerate(edges):
        if value < e:
            return ("<%g" % e) if k == 0 else ("%g-%g" % (edges[k - 1], e))
    return ">=%g" % edges[-1]


def third_of(y: float, near_is_high_y: bool) -> str:
    """sportvu.eval.region_of with y clamped to the court (a box foot or a SportVU player can sit just off it)."""
    from sportvu.eval import region_of
    return region_of(min(max(float(y), 0.0), 49.999), near_is_high_y)


def zone_of(x: float, y: float) -> str:
    d = min(np.hypot(x - rx, y - CONSTANTS["rim_y_ft"]) for rx in CONSTANTS["rim_x_ft"])
    return bin_of(float(d), CONSTANTS["zone_bins_ft"])


def crowding(sv: np.ndarray) -> np.ndarray:
    """Per player, the distance (ft) to the nearest other player."""
    D = np.sqrt(((sv[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D.min(1)


def holder(ball, sv: np.ndarray):
    """The index of the SportVU ball holder: exactly one player within holder_max_ft of the ball (court plane) and the
    ball below holder_ball_max_z_ft; None otherwise."""
    if ball is None or not np.isfinite(ball).all() or ball[2] >= CONSTANTS["holder_ball_max_z_ft"]:
        return None
    d = np.hypot(sv[:, 0] - ball[0], sv[:, 1] - ball[1])
    near = np.where(d <= CONSTANTS["holder_max_ft"])[0]
    return int(near[0]) if len(near) == 1 else None


def cuts_from_corr(frames: list, corr: list) -> list:
    """Cut frames: frames[k] where the histogram correlation with frames[k-1] is below cut_hist_corr."""
    return [frames[k] for k in range(1, len(frames)) if corr[k] is not None and corr[k] < CONSTANTS["cut_hist_corr"]]


def since_cut(frame: int, cuts: list, start: int, fps: float):
    """(seconds since the last cut at or before frame, censored): censored when no cut precedes it in the window."""
    prev = [c for c in cuts if c <= frame]
    if prev:
        return (frame - prev[-1]) / fps, False
    return (frame - start) / fps, True


def f1(tp: int, fp: int, fn: int):
    """F1 of the counts: 0.0 when nothing was right but something was predicted or missed; None with no events."""
    if tp + fp + fn == 0:
        return None
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return round(2 * p * r / (p + r), 3)


def universe(rows: list, live_wide: set, processed: dict) -> tuple:
    """(rows, n left out): truth rows on live wide frames that the scored build processed. A frame the build never
    processed is left out and counted, as the frozen bench does (its ok_rows need the frame in the sidecar)."""
    rows = [r for r in rows if r["frame"] in live_wide]
    kept = [r for r in rows if r["frame"] in processed]
    return kept, len(rows) - len(kept)


def caveats(in_sample: bool) -> list:
    return (["development game only: no CV test game, test video or test-arena game reaches this scorer outside score_once",
             "camera-truth candidates come from the B4 ICP matching that starts from V3's court H and boxes, so frames where V3 fails badly have no truth; the truth share is printed beside every row",
             "the camera truth's acceptance rules were set on the GSW training games (sportvu/autolabel.py) and are applied unchanged in new arenas; the gold-key paint check exists only at GSW",
             "events: play-by-play clock resolution is 1 s and the window OCR clock resolves 1 s; only events within 0.5 s of a live wide frame count",
             "the play-by-play comes from basketball-reference and is used as scoring truth only (D84)",
             "camera thirds clamp a point just off the court (y < 0 or y > 50 ft) to the nearest third; the frozen region rule (sportvu.eval.region_of) has no clamp, so third-level rows can differ from it there",
             "frames of the universe the build never processed are left out, as the frozen bench does; their count is printed"]
            + (["IN SAMPLE: the r23 court model was trained on this game's R1.3 labels, so its rows here are not a held-out measure; pooled rows report R1.3 and channel games separately"] if in_sample else []))


def live_wide_events(events: list, lw_clock: list) -> list:
    """The (period, clock) events with a live wide frame of the same period within event_live_wide_s."""
    tol = CONSTANTS["event_live_wide_s"]
    by_p = {}
    for p, c in lw_clock:
        by_p.setdefault(p, []).append(c)
    arr = {p: np.sort(np.array(v)) for p, v in by_p.items()}
    out = []
    for p, c in events:
        a = arr.get(p)
        if a is not None and len(a):
            k = int(np.searchsorted(a, c))
            near = min(abs(a[j] - c) for j in (k - 1, k) if 0 <= j < len(a))
            if near <= tol:
                out.append((p, c))
    return out


def match_events(truth: list, pred: list, tol: float) -> tuple:
    """Greedy one-to-one match of (period, clock) events within tol seconds, nearest first -> (found, false, pairs)."""
    cand = sorted(((abs(t[1] - p[1]), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred)
                   if t[0] == p[0] and abs(t[1] - p[1]) <= tol))
    ti, pj, pairs = set(), set(), []
    for d, i, j in cand:
        if i not in ti and j not in pj:
            ti.add(i); pj.add(j); pairs.append((i, j, d))
    return len(ti), len(pred) - len(pj), pairs


def boot_windows(groups: dict, stat, n: int = BOOT_N, seed: int = BOOT_SEED):
    """Cluster bootstrap over the keys of groups (window or game -> list of observations): [lo, hi] of the 95%
    interval of stat(pooled observations); None with fewer than 2 groups."""
    keys = [k for k, v in groups.items() if v]
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        pooled = [o for k in pick for o in groups[keys[k]]]
        v = stat(pooled)
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


def pct(a, q):
    return round(float(np.percentile(a, q)), 3) if len(a) else None


def project3d(C, rvec, f, w, h, pts):
    """Court points (x, y, height ft) -> px through the pinhole camera of sportvu.camera. Height is up; the camera
    frame's z axis points down (the camera position has a negative z), so height enters as -z."""
    import cv2
    from sportvu.camera import K_of
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    P = np.asarray(pts, np.float64).reshape(-1, 3) * np.array([1.0, 1.0, -1.0])
    cam = (R @ (P - np.asarray(C, np.float64).reshape(1, 3)).T)
    uv = K_of(f, w, h) @ cam
    return (uv[:2] / uv[2:3]).T, cam[2]


# ── inputs ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class Moments:
    """SportVUIndex plus the ball, shot clock and player tracks in the same per-quarter order."""

    def __init__(self, moments: list):
        from sportvu.sync import SportVUIndex
        self.index = SportVUIndex(moments)
        self.ball, self.order = {}, {}
        for q in self.index.q:
            ms = [m for m in moments if m["q"] == q and len(m["players"]) == 10]
            ms.sort(key=lambda m: -m["clock"])
            self.ball[q] = np.array([m["ball"] if m.get("ball") else [np.nan] * 3 for m in ms], np.float32)
        from sportvu.bench import shot_clock_index
        self.poss = shot_clock_index(moments)

    def speed(self, q: int, i: int, pid: int):
        d = self.index.q[q]
        k = int(round(CONSTANTS["speed_dt_s"] * 25))
        a, b = max(0, i - k), min(len(d["clock"]) - 1, i + k)
        dt = d["clock"][a] - d["clock"][b]
        if dt <= 0:
            return None
        pa, pb = np.where(d["pid"][a] == pid)[0], np.where(d["pid"][b] == pid)[0]
        if not len(pa) or not len(pb):
            return None
        return float(np.hypot(*(d["xy"][b][pb[0]] - d["xy"][a][pa[0]])) / dt)


def window_offset(P: dict, tag: str, stem: str):
    """The window's residual clock offset (s) solved at the camera fit (labels/<tag>_offsets.json); None if unknown."""
    p = P["labels"] / f"{tag}_offsets.json"
    return json.loads(p.read_text()).get(stem) if p.exists() else None


def truth_rows(P: dict, tag: str, window: dict, mode: str, index, mirror: str) -> tuple:
    """([{frame, q, moment, clock, H_truth}], offset_s) of one window under the truth mode. The offset is the
    window's own residual clock offset (b4: solved by window_truth; camera: solved at the fit), also when the window
    has no truth row; None when it was never solved."""
    stem = window["window"]
    if mode == "b4":
        from sportvu.truth import window_truth
        side_p, tm_p = (P["label_build"] / (stem + k) for k in ("_frames.json", "_timemap.json"))
        if not side_p.exists() or not tm_p.exists():
            return [], None
        off, _, rows = window_truth(window, json.loads(side_p.read_text()), json.loads(tm_p.read_text()), index, mirror)
        return [{"frame": r["frame"], "q": r["q"], "moment": r["moment"], "clock": r["clock"], "H_truth": r["H_truth"]}
                for r in rows if r["status"] == "ok"], off
    off = window_offset(P, tag, stem)
    p = P["labels"] / tag / (stem + ".jsonl")
    if not p.exists():
        return [], off
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    from sportvu.truth import CLOCK_TOL_S
    for r in rows:                                     # labels written before "moment" was stored: look it up
        if r.get("moment") is None:
            r["moment"] = index.at(r["q"], r["clock"], CLOCK_TOL_S)
    rows = [r for r in rows if r["moment"] is not None]
    if off is None and rows:
        off = rows[0].get("offset_s")
    if mode == "camera_paint":
        pp = P["labels"] / f"{tag}_paint.jsonl"
        if not pp.exists():
            return [], off
        keep = {(r["window"], r["frame"]) for r in map(json.loads, pp.read_text().splitlines()) if r.get("keep")}
        rows = [r for r in rows if (r["window"], r["frame"]) in keep]
    return rows, off


def frame_refs(P: dict, stem: str, video: Path, live: list, need_ref: set, stride: int, refs) -> dict:
    """Build-independent per-window references, cached: gate v2 wide score on live frames, referee-class detections
    on the given frames, and the cut signal on every stride frame."""
    import cv2
    from sportvu.bench import EMBED_BATCH
    from videoseq import SeqReader
    cp = P["cache"] / "scorecard" / (stem + ".json")
    cp.parent.mkdir(parents=True, exist_ok=True)
    c = json.loads(cp.read_text()) if cp.exists() else {}
    wide = {int(k): v for k, v in c.get("wide", {}).items()}
    refd = {int(k): v for k, v in c.get("referees", {}).items()}
    corr = {int(k): v for k, v in c.get("corr", {}).items()}
    cap = cv2.VideoCapture(str(video))
    W, H, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    strides = list(range(0, n, stride))
    need_w = [f for f in live if f not in wide]
    need_r = [f for f in sorted(need_ref) if f not in refd]
    need_c = strides if not corr else []
    read = sorted(set(need_w) | set(need_r) | set(need_c))
    complete = True
    if read:
        sr = SeqReader(cap)
        batch, bf, prev = [], [], None
        sw = set(need_w); srf = set(need_r); sc = set(need_c)

        def flush():
            if batch:
                g = refs.gate()
                for f, s in zip(bf, g.score_embeddings(g.backbone.embed_images(batch, batch_size=EMBED_BATCH))):
                    wide[f] = round(float(s), 4)
                batch.clear(); bf.clear()
        for f in read:
            ok, img = sr.read(f)
            if not ok:
                complete = False                 # a broken read: nothing of this pass is cached
                break
            if f in sw:
                batch.append(img[:, :, ::-1].copy()); bf.append(f)
                if len(batch) >= EMBED_BATCH:
                    flush()
            if f in srf:
                refd[f] = refs.referees(img)
            if f in sc:
                small = cv2.cvtColor(cv2.resize(img, tuple(CONSTANTS["cut_scale_px"])), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([small], [0, 1], None, CONSTANTS["cut_hist_bins"], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                corr[f] = None if prev is None else round(float(cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL)), 4)
                prev = hist
        flush()
        if complete:
            atomic_write(cp, json.dumps({"window": stem, "wide_threshold": refs.wide_thr,
                                         "wide": {str(k): v for k, v in sorted(wide.items())},
                                         "referees": {str(k): v for k, v in sorted(refd.items())},
                                         "corr": {str(k): v for k, v in sorted(corr.items())}}))
    cap.release()
    return {"wide": wide, "referees": refd, "corr": corr, "W": W, "H": H, "complete": complete}


# ── one window ──────────────────────────────────────────────────────────────────────────────────────────────────────
def score_window(P, tag, w, build_dir, mode, mom: Moments, mirror, refs, pbp_events) -> tuple:
    """(bench record, observations, box observations, v2 tallies) of one window."""
    import cv2
    from sportvu import bench as B
    from sportvu.camera import pose_from_h
    from sportvu.eval import FRAME_MARGIN_PX, GHOST_FT, MISS_FT, near_side, region_of
    from sportvu.sync import MIRRORS, apply_mirror
    from sportvu.truth import _match
    stem = w["window"]
    side_p = build_dir / (stem + "_frames.json")
    tm_p, video = P["label_build"] / (stem + "_timemap.json"), P["label_build"] / (stem + ".mp4")
    if not side_p.exists() or not tm_p.exists() or not video.exists():
        return {"window": stem, "skipped": "no sidecar" if not side_p.exists() else "no clip or time map"}, [], [], {}
    side = json.loads(side_p.read_text())
    srow = {r["frame"]: r for r in side["frames"]}
    tm = json.loads(tm_p.read_text())
    ident_p = build_dir / (stem + "_identity.json")
    ident = json.loads(ident_p.read_text()) if ident_p.exists() else {}
    idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
    team_of = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
    roster_of = {int(k): v for k, v in ident.get("roster_by_track", {}).items()}     # v2 input, when a build writes it
    fps, stride = float(tm["fps"]), int(tm["stride"])
    live = sorted({int(f) for f, _, _ in tm["frames"]})
    rows, off = truth_rows(P, tag, w, mode, mom.index, mirror)
    sidecar_cls = any("cls" in b for r in side["frames"] for b in r["boxes"])
    rf = frame_refs(P, stem, video, live, set() if sidecar_cls else {r["frame"] for r in rows}, stride, refs)
    if not rf["complete"]:
        return {"window": stem, "skipped": "clip read incomplete (nothing cached; a rerun reads it again)"}, [], [], {}
    wide, refd, Wd, Hd = rf["wide"], rf["referees"], rf["W"], rf["H"]
    cframes = sorted(rf["corr"]); cuts = cuts_from_corr(cframes, [rf["corr"][f] for f in cframes])
    lw = {f for f in live if wide.get(f, 0.0) >= refs.wide_thr}
    rows, not_processed = universe(rows, lw, srow)
    fx, fy = MIRRORS[mirror]
    sec = stride / fps
    rec = {"window": stem, "fps": fps, "stride": stride, "truth_frames": len(lw), "testable": len(rows), "matched": [],
           "line": [], "missed": 0, "in_frame": 0, "boxes": 0, "ghosts": 0, "ref_boxes": 0, "ref_ghosts": 0,
           "ref_near_player": 0, "team_rows": [], "poss_obs": {}, "poss_frames": {}, "truth_resid_px": [],
           "truth_status": {"ok": len(rows), "no_accepted_truth": len(lw) - len(rows)}, "jersey": None}
    acc = lambda f: srow.get(f) is not None and B.accepted(srow[f])
    rec["coverage"] = {"live_s": round(len(live) * sec, 1), "live_wide_s": round(len(lw) * sec, 1),
                       "accepted_live_wide_s": round(sum(acc(f) for f in lw) * sec, 1),
                       "accepted_live_not_wide_s": round(sum(acc(f) for f in live if f not in lw) * sec, 1)}
    obs, box_obs = [], []
    v2 = {"ball_frames": 0, "ball_in_frame": 0, "holder_frames": 0, "holder_named": 0, "holder_right": 0,
          "roster_obs": 0, "roster_right": 0, "reproj_px": [], "not_processed": not_processed}
    ball_pred = {}
    bp = build_dir / (stem + "_ball.json")
    if bp.exists():
        ball_pred = {int(k): v for k, v in json.loads(bp.read_text()).get("frames", {}).items()}
    v2["ball_tp"] = v2["ball_fp"] = v2["ball_fn"] = 0
    v2["ball_pred_written"] = bp.exists()
    carrier = {int(k): v for k, v in json.loads((build_dir / (stem + "_carrier.json")).read_text()).get("frames", {}).items()} \
        if (build_dir / (stem + "_carrier.json")).exists() else None
    for row in rows:
        f, q, i = row["frame"], row["q"], row["moment"]
        d = mom.index.q[q]
        sv = apply_mirror(d["xy"][i], fx, fy); pid = d["pid"][i]; team = d["team"][i]
        Ht = np.array(row["H_truth"], np.float64).reshape(3, 3)
        Pinv = np.linalg.inv(Ht)
        px = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32), Pinv).reshape(-1, 2)
        inside = (px[:, 0] > FRAME_MARGIN_PX) & (px[:, 0] < Wd - FRAME_MARGIN_PX) & (px[:, 1] > FRAME_MARGIN_PX) & (px[:, 1] < Hd - FRAME_MARGIN_PX)
        nh = near_side(Ht)
        r = srow[f]
        processed = True
        boxes = r["boxes"]
        feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
        ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2) if len(feet) else np.zeros((0, 2), np.float32)
        d_sv = np.sqrt(((sv[:, None, :] - ct[None, :, :]) ** 2).sum(-1)).min(1) if len(ct) else np.full(len(sv), np.inf)
        rec["in_frame"] += int(inside.sum())
        rec["missed"] += int((inside & (d_sv > MISS_FT)).sum())
        d_box = np.sqrt(((ct[:, None, :] - sv[None, :, :]) ** 2).sum(-1)).min(1) if len(ct) else np.zeros(0)
        for k, b in enumerate(boxes):
            is_ref = (b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else \
                any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
            ghost = bool(d_box[k] > GHOST_FT)
            rec["boxes"] += 1; rec["ghosts"] += ghost
            if is_ref:
                rec["ref_boxes"] += 1; rec["ref_ghosts"] += ghost; rec["ref_near_player"] += (not ghost)
            box_obs.append({"f": f, "ghost": ghost, "ref": bool(is_ref), "third": third_of(ct[k][1], nh)})
        pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
        err_of = {}
        if B.accepted(r) and r.get("H") is not None:
            Hp = np.array(r["H"], np.float64).reshape(3, 3)
            cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)
            for bi, j, _ in pairs:
                e = cp_[bi] - sv[j]
                err_of[j] = (float(np.hypot(*e)), float(e[1] if nh else -e[1]))
                rec["matched"].append({"err": err_of[j][0], "toward_camera": err_of[j][1],
                                       "region": third_of(sv[j][1], nh), "frame": int(f), "sv": int(j)})
            if f in lw:
                rec["line"].append({"frame": f, "line_px": B.line_error_px(r["H"], row["H_truth"], Wd, Hd)})
        seg = mom.poss[q][i]
        key = "%d_%d" % (q, seg)
        rec["poss_frames"][key] = rec["poss_frames"].get(key, 0) + 1
        track_of = {}
        for bi, j, dist in pairs:
            tid = boxes[bi]["tid"]; cid = idmap.get(tid, tid)
            track_of[j] = cid
            if team_of.get(cid) is not None:
                rec["team_rows"].append((cid, team_of[cid], int(team[j])))
            if dist <= B.ID_MATCH_FT:
                rec["poss_obs"].setdefault(key, {}).setdefault(str(int(pid[j])), []).append((f, cid, tid))
            if roster_of:
                v2["roster_obs"] += 1; v2["roster_right"] += str(roster_of.get(cid)) == str(int(pid[j]))
            v2["reproj_px"].append(float(np.hypot(*(px[j] - feet[bi]))))
        # observations: every in-frame SportVU player
        cr = crowding(sv)
        since, cens = since_cut(f, cuts, 0, fps)
        for j in np.where(inside)[0]:
            spd = mom.speed(q, i, int(pid[j]))
            obs.append({"w": stem, "f": int(f), "q": int(q), "pid": int(pid[j]), "x": round(float(sv[j][0]), 2),
                        "y": round(float(sv[j][1]), 2), "missed": bool(d_sv[j] > MISS_FT),
                        "err": None if j not in err_of else round(err_of[j][0], 3),
                        "toward": None if j not in err_of else round(err_of[j][1], 3),
                        "third": third_of(sv[j][1], nh), "crowd_ft": round(float(cr[j]), 2),
                        "speed_ftps": None if spd is None else round(spd, 2), "zone": zone_of(*sv[j]),
                        "since_cut_s": round(since, 2), "cut_censored": cens, "processed": processed})
        # ball and holder (SportVU ball, the frame's camera from its truth H)
        ball = mom.ball[q][i]
        if np.isfinite(ball).all():
            pose = pose_from_h(Ht, Wd, Hd)
            v2["ball_frames"] += 1
            if pose is not None:
                bxy = apply_mirror(ball[None, :2].copy(), fx, fy)[0]
                uv, depth = project3d(pose[2], pose[0], pose[1], Wd, Hd, [[bxy[0], bxy[1], float(ball[2])]])
                inb = depth[0] > 0 and 0 <= uv[0][0] < Wd and 0 <= uv[0][1] < Hd
                v2["ball_in_frame"] += bool(inb)
                if v2["ball_pred_written"]:
                    pr = ball_pred.get(f)
                    tol = BALL_TOL_PX
                    if inb and pr is not None and tol is not None and np.hypot(pr[0] - uv[0][0], pr[1] - uv[0][1]) <= tol:
                        v2["ball_tp"] += 1
                    else:
                        v2["ball_fp"] += pr is not None
                        v2["ball_fn"] += bool(inb)
            h = holder(np.array([*apply_mirror(ball[None, :2].copy(), fx, fy)[0], ball[2]]), sv)
            if h is not None:
                v2["holder_frames"] += 1
                if carrier is not None and f in carrier:
                    v2["holder_named"] += 1
                    v2["holder_right"] += carrier[f] is not None and track_of.get(h) == carrier[f]
    # events of the window: play-by-play attempts and rebounds near a live wide frame of the window (section 6.2)
    v2["window_offset_s"] = off
    if off is None:
        v2["window_without_offset"] = 1            # its events cannot be placed; counted in the event rows
        v2["events_truth"] = {"shot": [], "rebound": []}
    else:
        clock_of = {int(f_): (int(p_), float(c_) + off) for f_, p_, c_ in tm["frames"]}     # true clock = OCR + offset
        lw_clock = [clock_of[f_] for f_ in sorted(lw) if f_ in clock_of]
        v2["events_truth"] = {k: live_wide_events(pbp_events.get(k, []), lw_clock) for k in ("shot", "rebound")}
        ev_p = build_dir / (stem + "_events.json")
        if ev_p.exists():
            pred = json.loads(ev_p.read_text()).get("events", [])
            v2["events_pred"] = {k: live_wide_events([clock_of[e["frame"]] for e in pred
                                                      if e.get("kind") == k and e["frame"] in clock_of], lw_clock)
                                 for k in ("shot", "rebound")}
    return rec, obs, box_obs, v2


BALL_TOL_PX = None          # set from reports/research/p8_constants.json once the Phase A baseline fixes it


def load_fixed_constants() -> None:
    """The ball tolerance, fixed once on development games (fix-ball-tolerance); absent until then."""
    global BALL_TOL_PX
    from tools.p8_common import REPO
    p = REPO / "reports" / "research" / "p8_constants.json"
    if p.exists():
        BALL_TOL_PX = json.loads(p.read_text()).get("ball_tol_px")


def pbp_events(sv_game: str) -> dict:
    """Field-goal attempts and rebounds of the contract's play-by-play as (period, clock) (scoring truth only)."""
    import pandas as pd
    d = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "sportvu_contract" / f"{sv_game}_pbp.parquet",
                        columns=["period", "clock_s", "kind"])
    return {"shot": [(int(p), float(c)) for p, c, k in d.itertuples(index=False) if k in ("shot_made", "shot_missed")],
            "rebound": [(int(p), float(c)) for p, c, k in d.itertuples(index=False) if k == "rebound"]}


# ── aggregate ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _rate(o, key):
    return (sum(x[key] for x in o) / len(o)) if o else None


def breakdowns(obs: list, box_obs: list) -> dict:
    out = {}
    factors = {"crowding_ft": lambda o: bin_of(o["crowd_ft"], CONSTANTS["crowd_bins_ft"]),
               "speed_ftps": lambda o: bin_of(o["speed_ftps"], CONSTANTS["speed_bins_ftps"]),
               "zone_ft_to_rim": lambda o: o["zone"], "camera_third": lambda o: o["third"],
               "since_cut_s": lambda o: bin_of(o["since_cut_s"], CONSTANTS["since_cut_bins_s"]) + (" censored" if o["cut_censored"] else "")}
    for name, fn in factors.items():
        rows = {}
        for o in obs:
            b = fn(o)
            if b is not None:
                rows.setdefault(b, []).append(o)
        out[name] = {b: {"n": len(v), "missed_rate": round(_rate(v, "missed"), 3),
                         "matched_err_n": sum(1 for x in v if x["err"] is not None),
                         "err_p50": pct([x["err"] for x in v if x["err"] is not None], 50),
                         "err_p90": pct([x["err"] for x in v if x["err"] is not None], 90)} for b, v in sorted(rows.items())}
    gb = {}
    for b in box_obs:
        gb.setdefault(b["third"], []).append(b)
    out["ghost_by_third"] = {k: {"boxes": len(v), "ghost_rate_non_referee": round(sum(x["ghost"] and not x["ref"] for x in v) / max(1, sum(not x["ref"] for x in v)), 3)}
                             for k, v in sorted(gb.items())}
    return out


def v2_rows(v2s: list, events_pred_written: bool) -> dict:
    t = lambda k: sum(v.get(k, 0) for v in v2s)
    R = {}
    ball_written = any(v.get("ball_pred_written") for v in v2s)
    R["ball_in_image"] = {"target": TARGETS_V2["ball_in_image"], "truth_frames_ball_tracked": t("ball_frames"),
                          "truth_frames_ball_in_frame": t("ball_in_frame"), "tolerance_px": BALL_TOL_PX,
                          "value": f1(t("ball_tp"), t("ball_fp"), t("ball_fn")) if ball_written and BALL_TOL_PX else None,
                          "not_measurable": None if ball_written and BALL_TOL_PX else
                          ("the build writes no <window>_ball.json" if not ball_written else "the ball tolerance is not fixed yet")}
    R["ball_in_image"]["pass"] = None if R["ball_in_image"]["value"] is None else R["ball_in_image"]["value"] >= CONSTANTS["ball_f1_target"]
    named = t("holder_named")
    R["ball_carrier"] = {"target": TARGETS_V2["ball_carrier"], "truth_frames_with_holder": t("holder_frames"),
                         "value": round(t("holder_right") / t("holder_frames"), 3) if named and t("holder_frames") else None,
                         "not_measurable": None if named else "the build writes no <window>_carrier.json"}
    R["ball_carrier"]["pass"] = None if R["ball_carrier"]["value"] is None else R["ball_carrier"]["value"] >= CONSTANTS["carrier_target"]
    ro = t("roster_obs")
    R["identity_v2"] = {"target": TARGETS_V2["identity_v2"], "matched_player_frames": ro,
                        "value": round(t("roster_right") / ro, 3) if ro else None,
                        "not_measurable": None if ro else "the build writes no roster_by_track in <window>_identity.json"}
    R["identity_v2"]["pass"] = None if R["identity_v2"]["value"] is None else R["identity_v2"]["value"] >= CONSTANTS["identity_target"]
    for kind, target_key in (("shot", "shots"), ("rebound", "rebounds")):
        truth = sum(len(v.get("events_truth", {}).get(kind, [])) for v in v2s)
        found = false = 0
        if events_pred_written:
            for v in v2s:
                fd, fl, _ = match_events(v.get("events_truth", {}).get(kind, []), v.get("events_pred", {}).get(kind, []), CONSTANTS["event_tol_s"])
                found += fd; false += fl
        row = {"target": TARGETS_V2[target_key], "truth_events_near_live_wide_frames": truth,
               "windows_without_offset": t("window_without_offset"),
               "found_share": round(found / truth, 3) if events_pred_written and truth else None,
               "false_per_true": round(false / truth, 3) if events_pred_written and truth else None,
               "not_measurable": None if events_pred_written else "the build writes no <window>_events.json"}
        if row["found_share"] is None:
            row["pass"] = None
        elif kind == "shot":
            row["pass"] = row["found_share"] >= CONSTANTS["shots_found_target"] and row["false_per_true"] <= CONSTANTS["shots_false_max"]
        else:
            row["pass"] = row["found_share"] >= CONSTANTS["rebounds_found_target"]
        R[target_key] = row
    R["identity_reported"] = {"target": TARGETS_V2["identity_reported"], "value": None, "pass": None,
                              "not_measurable": "jersey tracklet accuracy: the build writes no jersey reads per tracklet",
                              "id_switches_per_possession_frozen": None}      # filled from the frozen identity row
    for k in ("shooter_position", "rebounder_position"):
        R[k] = {"target": TARGETS_V2[k], "value": None, "pass": None,
                "not_measurable": "needs a build that names the shooter or rebounder's track (E4 identity or E7 ball)"}
    R["swap_test"] = {"target": TARGETS_V2["swap_test"], "value": None, "pass": None, "not_measurable": "outside this scorer"}
    rp = [x for v in v2s for x in v.get("reproj_px", [])]
    R["truth_feet_reprojection_px"] = {"n": len(rp), "p50": pct(rp, 50), "p90": pct(rp, 90),
                                       "note": "SportVU feet through the truth camera vs the build's matched box feet; half the ball tolerance rule's input"}
    return R


def intervals(recs: list, obs: list, box_obs_by_w: dict) -> dict:
    by_w = {}
    for o in obs:
        by_w.setdefault(o["w"], []).append(o)
    m_by_w = {r["window"]: r["matched"] for r in recs if "skipped" not in r}
    errs = lambda pooled: [m["err"] for m in pooled]
    return {"position_p50_ft": boot_windows(m_by_w, lambda p: pct(errs(p), 50)),
            "position_p90_ft": boot_windows(m_by_w, lambda p: pct(errs(p), 90)),
            "missed_rate": boot_windows(by_w, lambda p: _rate(p, "missed")),
            "ghost_rate_non_referee": boot_windows(box_obs_by_w, lambda p: (sum(b["ghost"] and not b["ref"] for b in p) / max(1, sum(not b["ref"] for b in p))) if p else None),
            "method": "window-cluster bootstrap, %d resamples, seed %d, 95%% percentile interval" % (BOOT_N, BOOT_SEED)}


def score_game(tag: str, build: str, mode: str, root=None) -> dict:
    from sportvu import bench as B
    from tools import p8_frozen as F
    refuse_cv_test([tag], WHO)
    if mode not in CONSTANTS["truth_modes"]:
        raise SystemExit(f"{WHO}: unknown truth mode {mode}")
    g = F.game_of(tag)
    guard([g["sportvu"], g["video_id"]], WHO)
    P = F.install_paths(root)
    load_fixed_constants()
    build_dir = P["label_build"] if build == "v3" else P["builds"] / build
    mirror = F.mirror_of(P, tag)
    mom = Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = pbp_events(g["sportvu"])
    refs = B.References()
    recs, obs, box_obs_by_w, v2s = [], [], {}, []
    for w in F.label_windows(P, tag):
        rec, o, bo, v2 = score_window(P, tag, w, build_dir, mode, mom, mirror, refs, events)
        recs.append(rec); obs += o; box_obs_by_w[w["window"]] = bo
        if v2:
            v2s.append(v2)
    broken = [r["window"] for r in recs if str(r.get("skipped", "")).startswith("clip read incomplete")]
    if broken:
        raise SystemExit(f"{WHO}: {tag}: {len(broken)} window clip(s) could not be read through, e.g. {broken[0]}; no scorecard written")
    agg = B.aggregate(recs, refs.wide_thr)
    ev_written = any("events_pred" in v for v in v2s)
    rows_v2 = v2_rows(v2s, ev_written)
    rows_v2["identity_reported"]["id_switches_per_possession_frozen"] = agg["metrics"]["identity"]["id_switches_per_possession"]
    in_sample = build_in_sample(tag, build)
    box_obs = [b for v in box_obs_by_w.values() for b in v]
    rep = {"game": tag, "sportvu_game": g["sportvu"], "build": build, "truth": mode, "mirror": mirror,
           "date": datetime.date.today().isoformat(), "scorer": "scorecard v2 (tools/p8_scorecard.py)",
           "universe": {"live_wide_frames": agg["truth_frames"], "with_accepted_truth": agg["testable"],
                        "truth_share": None if not agg["truth_frames"] else round(agg["testable"] / agg["truth_frames"], 3),
                        "truth_frames_build_did_not_process": sum(v.get("not_processed", 0) for v in v2s)},
           "frozen_rows": agg["metrics"], "frozen_targets": B.TARGETS, "v2_rows": rows_v2,
           "build_in_sample": in_sample, "group": "r13" if tag in F.R13_TAGS else "channel",
           "intervals": intervals(recs, obs, box_obs_by_w), "breakdowns": breakdowns(obs, box_obs),
           "per_window": {r["window"]: ({"skipped": r["skipped"]} if "skipped" in r else
                                        {"live_wide": r["truth_frames"], "truth": r["testable"], "missed": r["missed"],
                                         "in_frame": r["in_frame"], "ghosts": r["ghosts"], "boxes": r["boxes"]}) for r in recs},
           "constants": CONSTANTS, "ball_tol_px": BALL_TOL_PX,
           "caveats": caveats(in_sample)}
    stem = f"{tag}__{build}__{mode}"
    out = P["scorecards"]
    (out / "obs").mkdir(parents=True, exist_ok=True)
    _gz_atomic(out / "obs" / (stem + ".jsonl.gz"), (json.dumps(o) + "\n" for o in obs))
    _gz_atomic(out / "obs" / (stem + ".reproj.json.gz"), [json.dumps([x for v in v2s for x in v.get("reproj_px", [])])])
    atomic_write(out / (stem + ".txt"), render(rep) + "\n")
    atomic_write(out / (stem + ".json"), json.dumps(rep, indent=1, default=str))     # last: the runner's done check
    return rep


def _gz_atomic(path: Path, lines) -> None:
    part = path.with_name(path.name + ".part")
    with gzip.open(part, "wt") as fh:
        for line in lines:
            fh.write(line)
    os.replace(part, path)


def build_in_sample(tag: str, build: str) -> bool:
    """True when the build's court model was trained on this game: r23 on the 4 R1.3 games (ROADMAP_LOG.md pass 14)."""
    from tools.p8_frozen import R13_TAGS
    return build == "r23" and tag in R13_TAGS


def render(rep: dict) -> str:
    M, V, I, U = rep["frozen_rows"], rep["v2_rows"], rep["intervals"], rep["universe"]
    v = lambda x: "n/a" if x is None else x
    pe, mp, gb = M["position_error"], M["missed_players"], M["ghost_boxes"]
    L = [f"SCORECARD V2  {rep['game']} vs SportVU {rep['sportvu_game']}  build {rep['build']}  truth {rep['truth']}"
         + ("  [IN SAMPLE: r23 trained on this game]" if rep.get("build_in_sample") else ""),
         f"universe: {U['with_accepted_truth']} of {U['live_wide_frames']} live wide frames carry accepted truth "
         f"(share {v(U['truth_share'])}); {U['truth_frames_build_did_not_process']} more the build never processed are left out", "",
         f"position error   p50 {v(pe['p50_ft'])} ft {v(I['position_p50_ft'])}, p90 {v(pe['p90_ft'])} ft {v(I['position_p90_ft'])}; n = {pe['matched_players']}",
         f"near-field bias  near {v(M['near_field_bias']['near_third_ft'])} ft, far {v(M['near_field_bias']['far_third_ft'])} ft",
         f"coverage         {v(M['coverage']['coverage'])}",
         f"missed players   {v(mp['rate'])} {v(I['missed_rate'])}; n = {mp['in_frame_players']}",
         f"ghost boxes      {v(gb['rate'])} non-referee {v(I['ghost_rate_non_referee'])}; n = {gb['boxes']}",
         f"team labels      {v(M['team_labels']['accuracy'])}; n = {M['team_labels']['n']}",
         f"identity         {v(M['identity']['id_switches_per_possession'])} switches / possession; {M['identity']['possessions']} possessions", "",
         "section 6.2 rows:"]
    for k, r in V.items():
        if k == "truth_feet_reprojection_px":
            L.append(f"  {k}: p50 {r['p50']} px, p90 {r['p90']} px, n = {r['n']}")
            continue
        val = r.get("value", r.get("found_share"))
        L.append(f"  {k}: {'not measurable: ' + r['not_measurable'] if r.get('not_measurable') else val}  (target {r['target']})")
    L += ["", "breakdowns (missed rate, position p50 by bin):"]
    for name, rows in rep["breakdowns"].items():
        if name == "ghost_by_third":
            L.append("  ghost rate by third: " + ", ".join(f"{k} {x['ghost_rate_non_referee']} (n {x['boxes']})" for k, x in rows.items()))
            continue
        L.append(f"  {name}: " + "; ".join(f"{b} missed {x['missed_rate']} p50 {x['err_p50']} (n {x['n']})" for b, x in rows.items()))
    L += ["", "intervals: " + I["method"], "caveats:"] + ["  - " + c for c in rep["caveats"]]
    return "\n".join(L)


def _load_obs(P: dict, t: str, build: str, mode: str) -> list:
    with gzip.open(P["scorecards"] / "obs" / f"{t}__{build}__{mode}.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


def combine(build: str, mode: str, tags=None, root=None) -> dict:
    """Pooled development rows, reported separately for the R1.3 games (r23 in sample there) and the channel games,
    with the game bootstrap beside the window bootstrap."""
    from tools import p8_frozen as F
    from tools.p8_common import p8_root
    P = F.layout(root or p8_root())
    tags = tags or sorted(F.games())
    refuse_cv_test(tags, WHO)
    reps = {t: json.loads((P["scorecards"] / f"{t}__{build}__{mode}.json").read_text()) for t in tags
            if (P["scorecards"] / f"{t}__{build}__{mode}.json").exists()}
    e = lambda p: pct([x["err"] for x in p if x["err"] is not None], 50)
    e90 = lambda p: pct([x["err"] for x in p if x["err"] is not None], 90)
    groups = {}
    for name, members in (("r13", [t for t in reps if t in F.R13_TAGS]), ("channel", [t for t in reps if t not in F.R13_TAGS])):
        if not members:
            continue
        obs_by_g = {t: _load_obs(P, t, build, mode) for t in members}
        by_w = {t + "/" + x["w"]: [] for t, o in obs_by_g.items() for x in o}
        for t, o in obs_by_g.items():
            for x in o:
                by_w[t + "/" + x["w"]].append(x)
        allo = [x for o in obs_by_g.values() for x in o]
        groups[name] = {"games": members, "build_in_sample": any(build_in_sample(t, build) for t in members),
                        "position_p50_ft": e(allo), "position_p90_ft": e90(allo), "missed_rate": _rate(allo, "missed"),
                        "game_bootstrap": {"position_p50_ft": boot_windows(obs_by_g, e), "position_p90_ft": boot_windows(obs_by_g, e90),
                                           "missed_rate": boot_windows(obs_by_g, lambda p: _rate(p, "missed"))},
                        "window_bootstrap": {"position_p50_ft": boot_windows(by_w, e), "position_p90_ft": boot_windows(by_w, e90),
                                             "missed_rate": boot_windows(by_w, lambda p: _rate(p, "missed"))}}
    out = {"build": build, "truth": mode, "groups": groups,
           "per_game": {t: {"p50": r["frozen_rows"]["position_error"]["p50_ft"], "p90": r["frozen_rows"]["position_error"]["p90_ft"],
                            "missed": r["frozen_rows"]["missed_players"]["rate"], "truth_share": r["universe"]["truth_share"],
                            "build_in_sample": r.get("build_in_sample")} for t, r in reps.items()},
           "note": "R1.3 and channel games are never pooled together: r23 was trained on the R1.3 games. Observation-level pooling; the per-game rows use the frozen aggregator"}
    atomic_write(P["scorecards"] / f"combined__{build}__{mode}.json", json.dumps(out, indent=1))
    return out


def gate(root=None) -> dict:
    """The first Phase B gate item (P8_FEASIBILITY.md section 9): on each gate game, r23's position p50 and p90 under
    the camera truth lie inside the window-bootstrap interval of the same rows under the old B4 truth. Refuses to run
    unless both scorecards of every gate game exist."""
    from tools import p8_frozen as F
    from tools.p8_common import p8_root
    P = F.layout(root or p8_root())
    games = CONSTANTS["gate_games"]
    refuse_cv_test(games, WHO)
    missing = [f"{t}__r23__{m}" for t in games for m in ("camera", "b4")
               if not (P["scorecards"] / f"{t}__r23__{m}.json").exists()]
    if missing:
        raise SystemExit(f"{WHO}: the gate needs the r23 camera and b4 scorecards of {games}; missing {missing} "
                         "(run tools.p8_night with --b4 gate)")
    out = {"rule": "r23 camera-truth p50 and p90 inside the b4-truth window-bootstrap interval, every gate game", "games": {}}
    for t in games:
        cam = json.loads((P["scorecards"] / f"{t}__r23__camera.json").read_text())
        b4 = json.loads((P["scorecards"] / f"{t}__r23__b4.json").read_text())
        row = {}
        for q, iv in (("p50_ft", "position_p50_ft"), ("p90_ft", "position_p90_ft")):
            v, lohi = cam["frozen_rows"]["position_error"][q], b4["intervals"][iv]
            row[q] = {"camera": v, "b4": b4["frozen_rows"]["position_error"][q], "b4_interval": lohi,
                      "inside": None if v is None or lohi is None else bool(lohi[0] <= v <= lohi[1])}
        row["build_in_sample"] = cam.get("build_in_sample")
        row["pass"] = all(row[q]["inside"] for q in ("p50_ft", "p90_ft"))
        out["games"][t] = row
    out["pass"] = all(r["pass"] for r in out["games"].values())
    atomic_write(P["scorecards"] / "gate_position.json", json.dumps(out, indent=1))
    return out


def fix_ball_tolerance(root=None, out_path=None) -> dict:
    """Writes reports/research/p8_constants.json once: ball_tol_px = 2 x the median reprojection error (px) of SportVU
    feet through the camera truth onto the matched box feet, pooled over every development game's r23 camera
    scorecard. Refuses when the file exists or a development game has no scorecard. Run before any ball model."""
    from tools import p8_frozen as F
    from tools.p8_common import REPO, p8_root
    out_p = Path(out_path) if out_path else REPO / "reports" / "research" / "p8_constants.json"
    if out_p.exists():
        raise SystemExit(f"{WHO}: {out_p.name} exists; the ball tolerance is fixed once and never rewritten")
    P = F.layout(root or p8_root())
    games = sorted(F.games())
    refuse_cv_test(games, WHO)
    paths = {t: P["scorecards"] / "obs" / f"{t}__r23__camera.reproj.json.gz" for t in games}
    missing = [t for t, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"{WHO}: every development game needs its r23 camera scorecard first; missing {missing}")
    vals = []
    per = {}
    for t, p in paths.items():
        with gzip.open(p, "rt") as fh:
            v = json.loads(fh.read())
        per[t] = {"n": len(v), "median_px": pct(v, 50)}
        vals += v
    med = float(np.median(vals))
    rec = {"fixed_on": datetime.date.today().isoformat(), "ball_tol_px": round(2 * med, 2),
           "rule": CONSTANTS["ball_tol_rule"], "median_reprojection_px": round(med, 3), "n": len(vals), "per_game": per,
           "source": "scorecards/obs/<game>__r23__camera.reproj.json.gz of the P8 workspace (development games only)"}
    atomic_write(out_p, json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("game")
    g.add_argument("tag")
    g.add_argument("--build", default="r23")
    g.add_argument("--truth", default="camera", choices=CONSTANTS["truth_modes"])
    c = sub.add_parser("combine")
    c.add_argument("--build", default="r23")
    c.add_argument("--truth", default="camera", choices=CONSTANTS["truth_modes"])
    c.add_argument("--tags", default=None)
    sub.add_parser("gate")
    sub.add_parser("fix-ball-tolerance")
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        print(json.dumps(gate(), indent=1))
        return 0
    if a.cmd == "fix-ball-tolerance":
        print(json.dumps(fix_ball_tolerance(), indent=1))
        return 0
    if a.cmd == "game":
        print(render(score_game(a.tag, a.build, a.truth)))
    else:
        print(json.dumps(combine(a.build, a.truth, a.tags.split(",") if a.tags else None), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

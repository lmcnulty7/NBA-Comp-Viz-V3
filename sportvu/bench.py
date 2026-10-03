"""
sportvu/bench.py: the Stage 1 scorecard (ROADMAP R0.3). One json + txt with every target metric
of CLAIMS.md "Stage 1 definition of done" for one build of a game's SportVU windows.

A build is a directory holding, per window, <window>_frames.json (the A0 sidecar) and
<window>_identity.json. The window videos, the window OCR time maps and the truth H come from the
B4 rebuild (data/sportvu/build, data/sportvu/truth) and are the same for every build, so builds are
scored on identical frames. The truth H is the camera of each testable frame: every build's feet
are matched to SportVU through it (Hungarian, MATCH_FT), never through the pairs B4 stored, which
belong to the V3 boxes. Each metric reports its n; the untestable share sits beside all of them.

  court line error   accepted testable wide frames: the template drawn through the build H vs the
                     template drawn through H_truth, symmetric median px (the larger of the two
                     directional medians); share of frames <= LINE_TARGET_PX. Beside it, the floor:
                     the truth fit's own residual in px (SportVU inliers projected back through
                     H_truth vs the V3 feet they were fitted on). The line field (painted-line
                     ridges) was tried as that floor and does not resolve a few px on 720p
                     footage (2026-10-03, window s00_w00: ridge support within 2 px 0.047 for
                     H_truth vs 0.045 for H_truth shifted 6 px), so it is not used.
  position error     |build H(foot) - SportVU| per matched player on accepted testable frames; p50,
                     p90, by camera third.
  near-field bias    median error toward the camera in the near third and in the far third.
  coverage           accepted live wide seconds / live wide seconds. Live: running clock in the
                     window's own OCR time map. Wide: gate v2 (models/trained_head_v2, threshold
                     thresholds.json "v2") on the same stride frames. Build independent, cached.
  missed players     testable frames the build processed: SportVU players inside the frame
                     (through H_truth) with no box foot within MISS_FT.
  ghost boxes        boxes more than GHOST_FT from every SportVU player, split by referee class:
                     the sidecar's per-box "cls" when a build writes one, else player_detector.pt's
                     referee class run on the frame (a detection with IoU >= REF_IOU). The target
                     rate excludes referee boxes.
  team labels        build team per canonical track vs the SportVU team of the matched player,
                     majority mapping per window (the build's labels are an arbitrary A/B).
  identity           id switches per possession. Possessions are SportVU shot-clock segments (a new
                     one where the shot clock jumps up by more than SHOT_RESET_S); per player, a
                     switch is a change of canonical track between consecutive matched observations
                     inside one possession, from pairs within ID_MATCH_FT only (a looser pair can be
                     a swap between two close players, which would read as a false switch).
                     Jersey-read rate when a build writes <window>_jersey.json.
  generalisation     the same scorecard on the held-out arena and era; needs those builds (R1).

  python -m sportvu.bench data/sportvu/build --name v3 [--game gsw_phx_2016]
      -> reports/scorecard/<game>__<name>.{json,txt}
"""
from __future__ import annotations
import argparse, datetime, json
from pathlib import Path
import numpy as np
import config
from sportvu.sync import SportVUIndex, apply_mirror, MIRRORS
from sportvu.rebuild import BUILD_DIR
from sportvu.truth import TRUTH_DIR, GATES_FT, _match
from sportvu.eval import MISS_FT, GHOST_FT, FRAME_MARGIN_PX, region_of, near_side

SCORECARD_DIR = config.REPORTS_DIR / "scorecard"
CACHE_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "bench"
MATCH_FT = GATES_FT[-1]       # B4's final gate: a box and a SportVU player pair within 5 ft under H_truth
LINE_TARGET_PX = 3.0
MIN_LINE_SAMPLES = 40         # fewer on-frame template samples: the frame's line error is undefined
ID_MATCH_FT = MISS_FT         # identity observations use only pairs this close under H_truth
REF_IOU = 0.5
REFEREE_CLASS = 1             # models/player_detector.pt: {0 player, 1 referee, 2 ball, 3 rim, 4 number}
SHOT_RESET_S = 1.0
MIN_POSS_FRAMES = 5           # a possession counts for identity with this many testable frames
EMBED_BATCH = 32

TARGETS = {
    "court_line_error": "<= 3 px (about 0.5 ft) on 95% of accepted wide frames",
    "position_error": "p50 <= 2.0 ft, p90 <= 5.0 ft",
    "near_field_bias": "median toward the camera within +-0.5 ft in the near and the far third",
    "coverage": ">= 50% of live wide seconds",
    "missed_players": "<= 5%",
    "ghost_boxes": "<= 5%, referees excluded by class",
    "team_labels": ">= 95%",
    "identity": "reported; no target yet (720p ceiling)",
    "generalisation": "held-out arena and era within 1.5x of the held-out game",
}


def accepted(row: dict) -> bool:
    """A build emits positions on a frame when it says so; V3 has no acceptance flag and emits
    whenever it has an H (TRACK, LINE_TRACK and HELD)."""
    return bool(row.get("accepted", row.get("H") is not None))


def sportvu_name(game: str) -> str:
    """Registry entry -> SportVU log name, e.g. gsw_phx_2016 -> 12.16.2015.PHX.at.GSW."""
    g = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())[game]
    y, m, d = g["date"].split("-")
    return "%s.%s.%s.%s.at.%s" % (m, d, y, g["away"], g["home"])


# ── geometry ──────────────────────────────────────────────────────────────────
def _template(P, w: int, h: int):
    """Template chords drawn through P (court ft -> px): a distance transform of the drawing and the
    on-frame chord midpoints. (None, empty) when nothing lands on the frame."""
    import cv2
    from court.snap_track import CH_A, CH_B, project
    pa, pb = project(P, CH_A), project(P, CH_B)
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    ok &= (np.abs(pa) < 8 * max(w, h)).all(1) & (np.abs(pb) < 8 * max(w, h)).all(1)
    T = np.full((h, w), 255, np.uint8)
    for a, b in zip(pa[ok], pb[ok]):
        cv2.line(T, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 0, 1)
    mid = ((pa + pb) / 2.0)[ok]
    mid = mid[(mid[:, 0] >= 0) & (mid[:, 0] <= w - 1) & (mid[:, 1] >= 0) & (mid[:, 1] <= h - 1)]
    if (T == 0).sum() == 0:
        return None, mid
    return cv2.distanceTransform(T, cv2.DIST_L2, 3), mid


def line_error_px(H_build, H_truth, w: int, h: int):
    """Symmetric median px distance between the template under the build H and under H_truth
    (both px -> ft). None when either template has fewer than MIN_LINE_SAMPLES on-frame samples."""
    try:
        Pb = np.linalg.inv(np.asarray(H_build, np.float64).reshape(3, 3))
        Pt = np.linalg.inv(np.asarray(H_truth, np.float64).reshape(3, 3))
    except np.linalg.LinAlgError:
        return None
    dtb, mb = _template(Pb, w, h)
    dtt, mt = _template(Pt, w, h)
    if dtb is None or dtt is None or len(mb) < MIN_LINE_SAMPLES or len(mt) < MIN_LINE_SAMPLES:
        return None
    ib, it = np.round(mb).astype(int), np.round(mt).astype(int)
    d1 = dtt[ib[:, 1], ib[:, 0]]
    d2 = dtb[it[:, 1], it[:, 0]]
    return float(max(np.median(d1), np.median(d2)))


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ── identity ──────────────────────────────────────────────────────────────────
def possession_ids(shot) -> np.ndarray:
    """Segment index per moment (play order): a new segment wherever the shot clock jumps up by more
    than SHOT_RESET_S. Moments without a shot clock (NaN/None) keep the current segment."""
    seg = np.zeros(len(shot), int)
    k, last = 0, None
    for i, s in enumerate(shot):
        if s is not None and np.isfinite(s):
            if last is not None and s > last + SHOT_RESET_S:
                k += 1
            last = s
        seg[i] = k
    return seg


def id_switches(seq: list) -> int:
    """Changes of track id between consecutive observations of one player."""
    return int(sum(1 for a, b in zip(seq, seq[1:]) if a != b))


def shot_clock_index(moments: list) -> dict:
    """Per quarter, the shot clock in SportVUIndex's moment order (same filter, same stable sort)."""
    out = {}
    for q in sorted({m["q"] for m in moments}):
        ms = [m for m in moments if m["q"] == q and len(m["players"]) == 10]
        ms.sort(key=lambda m: -m["clock"])
        out[q] = possession_ids([np.nan if m.get("shot") is None else float(m["shot"]) for m in ms])
    return out


# ── build-independent per-window references (cached) ─────────────────────────
class References:
    """Gate v2 wide scores on live frames and referee-class detections on testable frames, computed
    once per window and cached under data/sportvu/bench/. Models load only on a cache miss."""

    def __init__(self):
        self._gate = self._det = None
        self.device = config.get_device()
        thr = json.loads(config.THRESHOLDS_PATH.read_text())
        self.wide_thr = float(thr["v2"])

    def gate(self):
        if self._gate is None:
            from gate.backbones import get_backbone
            from gate.trained_head import TrainedHeadGate
            self._gate = TrainedHeadGate.load(config.HEAD_V2_PATH, backbone=get_backbone("clip", self.device), threshold=self.wide_thr)
        return self._gate

    def detector(self):
        if self._det is None:
            from ultralytics import YOLO
            self._det = YOLO(str(config.PLAYER_DETECTOR_WEIGHTS))
        return self._det

    def referees(self, frame_bgr) -> list:
        r = self.detector().predict(frame_bgr, classes=[REFEREE_CLASS], conf=config.PLAYER_CONF,
                                    device=self.device, verbose=False)[0]
        return [[round(float(v), 1) for v in b] + [round(float(c), 3)]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


# ── one window ────────────────────────────────────────────────────────────────
def score_window(wd: dict, build_dir: Path, refs: References, index: SportVUIndex, poss: dict, mirror: str) -> dict:
    """Every per-window count and per-observation list the scorecard aggregates."""
    import cv2
    from videoseq import SeqReader
    stem = wd["window"]
    side_p = build_dir / (stem + "_frames.json")
    truth_p = TRUTH_DIR / (stem + "_truth.json")
    if not side_p.exists() or not truth_p.exists():
        return {"window": stem, "skipped": "no sidecar" if not side_p.exists() else "no truth"}
    side = json.loads(side_p.read_text())
    srow = {r["frame"]: r for r in side["frames"]}
    sframes = np.array(sorted(srow))
    truth = json.loads(truth_p.read_text())
    tm = json.loads(((build_dir if (build_dir / (stem + "_timemap.json")).exists() else BUILD_DIR) / (stem + "_timemap.json")).read_text())
    ident_p = build_dir / (stem + "_identity.json")
    ident = json.loads(ident_p.read_text()) if ident_p.exists() else {}
    idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
    team_of = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
    video = build_dir / (stem + ".mp4")
    video = video if video.exists() else BUILD_DIR / (stem + ".mp4")
    fx, fy = MIRRORS[mirror]
    fps, stride = float(tm["fps"]), int(tm["stride"])

    def side_at(f: int):
        """The build's row for a stride frame (nearest within half the build's stride)."""
        if not len(sframes):
            return None
        k = int(np.argmin(np.abs(sframes - f)))
        return srow[int(sframes[k])] if abs(int(sframes[k]) - f) <= max(1, int(side.get("stride", stride)) // 2) else None

    # caches
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_p = CACHE_DIR / (stem + ".json")
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    wide = {int(k): v for k, v in cache.get("wide", {}).items()}
    refd = {int(k): v for k, v in cache.get("referees", {}).items()}
    live = sorted({int(f) for f, _, _ in tm["frames"]})
    ok_rows = [r for r in truth["frames"] if r["status"] == "ok" and r["frame"] in srow]
    testable = {r["frame"] for r in ok_rows}
    sidecar_cls = any("cls" in b for r in side["frames"] for b in r["boxes"])
    need_wide = [f for f in live if f not in wide]
    need_ref = [] if sidecar_cls else [f for f in sorted(testable) if f not in refd]
    read = sorted(set(need_wide) | set(need_ref))
    v3_side = srow if build_dir == BUILD_DIR else {r["frame"]: r for r in json.loads((BUILD_DIR / (stem + "_frames.json")).read_text())["frames"]}

    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video))
    W, Hh = int(cap.get(3)), int(cap.get(4))
    reader = SeqReader(cap)
    batch, batch_f = [], []
    need_wide_s, need_ref_s = set(need_wide), set(need_ref)

    def flush():
        if batch:
            g = refs.gate()
            sc = g.score_embeddings(g.backbone.embed_images(batch, batch_size=EMBED_BATCH))
            for f, s in zip(batch_f, sc):
                wide[f] = round(float(s), 4)
            batch.clear(); batch_f.clear()

    for f in read:
        ok, frame = reader.read(f)
        if not ok:
            break
        if f in need_wide_s:
            batch.append(frame[:, :, ::-1].copy()); batch_f.append(f)
            if len(batch) >= EMBED_BATCH:
                flush()
        if f in need_ref_s:
            refd[f] = refs.referees(frame)
    flush()
    cap.release()
    if need_wide or need_ref:
        cache_p.write_text(json.dumps({"window": stem, "wide_threshold": refs.wide_thr,
                                       "wide": {str(k): v for k, v in sorted(wide.items())},
                                       "referees": {str(k): v for k, v in sorted(refd.items())}}))

    out = {"window": stem, "fps": fps, "stride": stride, "truth_frames": len(truth["frames"]), "testable": len(ok_rows),
           "matched": [], "line": [], "missed": 0, "in_frame": 0, "boxes": 0, "ghosts": 0, "ref_boxes": 0, "ref_ghosts": 0,
           "ref_near_player": 0, "team_rows": [], "poss_obs": {}, "poss_frames": {}, "truth_resid_px": [],
           "truth_status": {}}
    for r in truth["frames"]:
        out["truth_status"][r["status"]] = out["truth_status"].get(r["status"], 0) + 1
    # coverage
    def emits(f: int) -> bool:
        r = side_at(f)
        return r is not None and accepted(r)

    lw = [f for f in live if wide.get(f, 0.0) >= refs.wide_thr]
    lw_set = set(lw)
    acc_lw = [f for f in lw if emits(f)]
    acc_lnw = [f for f in live if f not in lw_set and emits(f)]
    sec = stride / fps
    out["coverage"] = {"live_s": round(len(live) * sec, 1), "live_wide_s": round(len(lw) * sec, 1),
                       "accepted_live_wide_s": round(len(acc_lw) * sec, 1), "accepted_live_not_wide_s": round(len(acc_lnw) * sec, 1)}
    # truth-frame metrics
    for row in ok_rows:
        f = row["frame"]; r = srow[f]
        q, i = row["q"], row["moment"]
        sv = apply_mirror(index.q[q]["xy"][i], fx, fy); pid = index.q[q]["pid"][i]; team = index.q[q]["team"][i]
        Ht = np.array(row["H_truth"], np.float64).reshape(3, 3)
        # floor: the truth fit's residual in px on the V3 feet it was fitted on (inlier pairs)
        vb = v3_side[f]["boxes"]
        inl = [(p["box"], int(np.where(pid == p["pid"])[0][0])) for p in row["pairs"] if p["inlier"]]
        if inl:
            fp = np.array([vb[bi]["foot_stab"] for bi, _ in inl], np.float32)
            bp = cv2.perspectiveTransform(np.array([sv[j] for _, j in inl], np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
            out["truth_resid_px"].append(float(np.median(np.linalg.norm(bp - fp, axis=1))))
        feet = np.array([b["foot_stab"] for b in r["boxes"]], np.float32).reshape(-1, 2)
        ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2) if len(feet) else np.zeros((0, 2), np.float32)
        # missed and ghosts (detection, every processed testable frame)
        px = cv2.perspectiveTransform(sv.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Ht)).reshape(-1, 2)
        inside = (px[:, 0] > FRAME_MARGIN_PX) & (px[:, 0] < W - FRAME_MARGIN_PX) & (px[:, 1] > FRAME_MARGIN_PX) & (px[:, 1] < Hh - FRAME_MARGIN_PX)
        out["in_frame"] += int(inside.sum())
        if len(ct):
            d_sv = np.sqrt(((sv[:, None, :] - ct[None, :, :]) ** 2).sum(-1)).min(1)
            out["missed"] += int((inside & (d_sv > MISS_FT)).sum())
            d_box = np.sqrt(((ct[:, None, :] - sv[None, :, :]) ** 2).sum(-1)).min(1)
        else:
            out["missed"] += int(inside.sum()); d_box = np.zeros(0)
        for k, b in enumerate(r["boxes"]):
            is_ref = (b.get("cls") in ("referee", REFEREE_CLASS)) if sidecar_cls else \
                any(_iou(b["bbox"], d[:4]) >= REF_IOU for d in refd.get(f, []))
            ghost = bool(d_box[k] > GHOST_FT)
            out["boxes"] += 1; out["ghosts"] += ghost
            if is_ref:
                out["ref_boxes"] += 1; out["ref_ghosts"] += ghost; out["ref_near_player"] += (not ghost)
        # matching under H_truth (position, team, identity)
        pairs = _match(ct, sv, MATCH_FT) if len(ct) else []
        nh = near_side(Ht)
        seg = poss[q][i]
        if accepted(r) and r.get("H") is not None:
            Hp = np.array(r["H"], np.float64).reshape(3, 3)
            cp = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)
            for bi, j, _ in pairs:
                e = cp[bi] - sv[j]
                out["matched"].append({"err": float(np.hypot(*e)), "toward_camera": float(e[1] if nh else -e[1]),
                                       "region": region_of(float(sv[j][1]), nh)})
            if wide.get(f, 0.0) >= refs.wide_thr:
                le = line_error_px(r["H"], row["H_truth"], W, Hh)
                out["line"].append({"frame": f, "line_px": le})
        key = "%d_%d" % (q, seg)
        out["poss_frames"][key] = out["poss_frames"].get(key, 0) + 1
        for bi, j, d in pairs:
            tid = r["boxes"][bi]["tid"]; cid = idmap.get(tid, tid)
            if team_of.get(cid) is not None:
                out["team_rows"].append((cid, team_of[cid], int(team[j])))
            if d <= ID_MATCH_FT:
                out["poss_obs"].setdefault(key, {}).setdefault(str(int(pid[j])), []).append((f, cid, tid))
    jersey_p = build_dir / (stem + "_jersey.json")
    out["jersey"] = json.loads(jersey_p.read_text()) if jersey_p.exists() else None
    return out


# ── aggregate ─────────────────────────────────────────────────────────────────
def _pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


def aggregate(wins: list, wide_thr: float) -> dict:
    wins = [w for w in wins if "skipped" not in w]
    truth_frames = sum(w["truth_frames"] for w in wins); testable = sum(w["testable"] for w in wins)
    M = {}
    # court line error
    le = np.array([x["line_px"] for w in wins for x in w["line"] if x["line_px"] is not None])
    tr = np.array([x for w in wins for x in w["truth_resid_px"]])
    n_line = sum(len(w["line"]) for w in wins)
    M["court_line_error"] = {"frames": n_line, "frames_defined": int(len(le)),
                             "share_le_target": round(float((le <= LINE_TARGET_PX).mean()), 3) if len(le) else None,
                             "p50_px": _pct(le, 50), "p90_px": _pct(le, 90), "p95_px": _pct(le, 95),
                             "truth_fit_residual_p50_px": _pct(tr, 50), "truth_fit_residual_p90_px": _pct(tr, 90),
                             "pass": (float((le <= LINE_TARGET_PX).mean()) >= 0.95) if len(le) else None}
    # position error and near-field bias
    mt = [m for w in wins for m in w["matched"]]
    e = np.array([m["err"] for m in mt])
    reg = {}
    for k in ("far", "middle", "near"):
        ek = np.array([m["err"] for m in mt if m["region"] == k]); ck = np.array([m["toward_camera"] for m in mt if m["region"] == k])
        reg[k] = {"n": int(len(ek)), "p50": _pct(ek, 50), "p90": _pct(ek, 90),
                  "toward_camera_ft": round(float(np.median(ck)), 2) if len(ck) else None}
    M["position_error"] = {"matched_players": int(len(e)), "p50_ft": _pct(e, 50), "p90_ft": _pct(e, 90),
                           "mean_ft": round(float(e.mean()), 2) if len(e) else None, "by_region": reg,
                           "pass": (_pct(e, 50) <= 2.0 and _pct(e, 90) <= 5.0) if len(e) else None}
    nb, fb = reg["near"]["toward_camera_ft"], reg["far"]["toward_camera_ft"]
    M["near_field_bias"] = {"near_third_ft": nb, "far_third_ft": fb, "n_near": reg["near"]["n"], "n_far": reg["far"]["n"],
                            "pass": (abs(nb) <= 0.5 and abs(fb) <= 0.5) if nb is not None and fb is not None else None}
    # coverage
    cv = {k: round(sum(w["coverage"][k] for w in wins), 1) for k in wins[0]["coverage"]} if wins else {}
    cov = cv["accepted_live_wide_s"] / cv["live_wide_s"] if cv.get("live_wide_s") else None
    M["coverage"] = {**cv, "coverage": round(cov, 3) if cov is not None else None, "wide_reference": "gate v2 >= %.2f" % wide_thr,
                     "pass": (cov >= 0.50) if cov is not None else None}
    # missed, ghosts
    mi, inf = sum(w["missed"] for w in wins), sum(w["in_frame"] for w in wins)
    M["missed_players"] = {"missed": mi, "in_frame_players": inf, "rate": round(mi / inf, 3) if inf else None,
                           "pass": (mi / inf <= 0.05) if inf else None}
    bx, gh, rb, rg, rn = (sum(w[k] for w in wins) for k in ("boxes", "ghosts", "ref_boxes", "ref_ghosts", "ref_near_player"))
    nr_rate = (gh - rg) / (bx - rb) if bx - rb else None
    M["ghost_boxes"] = {"boxes": bx, "ghosts": gh, "rate_all": round(gh / bx, 3) if bx else None,
                        "referee_boxes": rb, "referee_ghosts": rg, "referee_boxes_within_ghost_ft_of_a_player": rn,
                        "non_referee_boxes": bx - rb, "non_referee_ghosts": gh - rg,
                        "rate": round(nr_rate, 3) if nr_rate is not None else None,
                        "pass": (nr_rate <= 0.05) if nr_rate is not None else None}
    # team labels: majority mapping per window
    n = c = 0
    for w in wins:
        for lab in {t[1] for t in w["team_rows"]}:
            sub = [t[2] for t in w["team_rows"] if t[1] == lab]
            n += len(sub); c += max(sub.count(v) for v in set(sub))
    M["team_labels"] = {"n": n, "correct": c, "accuracy": round(c / n, 3) if n else None, "pass": (c / n >= 0.95) if n else None}
    # identity
    P = sw = sw_frag = obs = 0; tracks_per = []
    for w in wins:
        for key, players in w["poss_obs"].items():
            if w["poss_frames"].get(key, 0) < MIN_POSS_FRAMES:
                continue
            P += 1
            for seq in players.values():
                seq = sorted(seq)
                sw += id_switches([s[1] for s in seq]); sw_frag += id_switches([s[2] for s in seq]); obs += len(seq)
                tracks_per.append(len({s[1] for s in seq}))
    jr = [w["jersey"] for w in wins if w.get("jersey") is not None]
    M["identity"] = {"possessions": P, "player_observations": obs, "id_switches": sw,
                     "id_switches_per_possession": round(sw / P, 2) if P else None,
                     "fragment_switches_per_possession": round(sw_frag / P, 2) if P else None,
                     "tracks_per_player_possession_mean": round(float(np.mean(tracks_per)), 2) if tracks_per else None,
                     "jersey_read_rate": None if not jr else jr,
                     "jersey_not_measurable": None if jr else "the build wrote no <window>_jersey.json: jersey OCR (CLAIMS C2) did not run on these windows (no saved crops)",
                     "pass": None}
    M["generalisation"] = {"value": None, "pass": None,
                           "not_measurable": "needs builds on the held-out arena (named after ROADMAP R1.1) and the held-out era; the era game (2013) has no public SportVU log"}
    causes = {}
    for w in wins:
        for k, n_ in w["truth_status"].items():
            if k != "ok":
                causes[k] = causes.get(k, 0) + n_
    return {"truth_frames": truth_frames, "testable": testable,
            "untestable_share": round(1 - testable / truth_frames, 3) if truth_frames else None,
            "untestable_by_cause": dict(sorted(causes.items(), key=lambda kv: -kv[1])), "metrics": M}


def render_txt(rep: dict) -> str:
    M = rep["metrics"]
    def v(x, fmt="%s"):
        return "n/a" if x is None else fmt % x
    def verdict(m):
        return {True: "MET", False: "not met", None: "-"}[m.get("pass")]
    cl, pe, nf, cv, mp, gb, tl, idn = (M[k] for k in ("court_line_error", "position_error", "near_field_bias", "coverage",
                                                       "missed_players", "ghost_boxes", "team_labels", "identity"))
    rows = [
        ("court line error", TARGETS["court_line_error"],
         "%s of frames <= 3 px; p50 %s px, p95 %s px (floor: truth fit residual p50 %s px, p90 %s px)" % (
             v(cl["share_le_target"] and 100 * cl["share_le_target"], "%.1f%%"),
             v(cl["p50_px"]), v(cl["p95_px"]), v(cl["truth_fit_residual_p50_px"]), v(cl["truth_fit_residual_p90_px"])),
         "%d frames" % cl["frames_defined"], verdict(cl)),
        ("position error", TARGETS["position_error"], "p50 %s ft, p90 %s ft" % (v(pe["p50_ft"]), v(pe["p90_ft"])),
         "%d players" % pe["matched_players"], verdict(pe)),
        ("near-field bias", TARGETS["near_field_bias"], "near %s ft, far %s ft (toward camera)" % (v(nf["near_third_ft"], "%+.2f"), v(nf["far_third_ft"], "%+.2f")),
         "%d / %d" % (nf["n_near"], nf["n_far"]), verdict(nf)),
        ("coverage", TARGETS["coverage"], "%s (%s of %s live wide s; %s s accepted on live non-wide)" % (
            "n/a" if cv["coverage"] is None else "%.1f%%" % (100 * cv["coverage"]), cv.get("accepted_live_wide_s"), cv.get("live_wide_s"), cv.get("accepted_live_not_wide_s")),
         "%s live s" % cv.get("live_s"), verdict(cv)),
        ("missed players", TARGETS["missed_players"], v(mp["rate"] and 100 * mp["rate"], "%.1f%%"), "%d in frame" % mp["in_frame_players"], verdict(mp)),
        ("ghost boxes", TARGETS["ghost_boxes"], "%s non-referee (%s with referees; %d referee boxes)" % (
            v(gb["rate"] and 100 * gb["rate"], "%.1f%%"), v(gb["rate_all"] and 100 * gb["rate_all"], "%.1f%%"), gb["referee_boxes"]),
         "%d boxes" % gb["boxes"], verdict(gb)),
        ("team labels", TARGETS["team_labels"], v(tl["accuracy"] and 100 * tl["accuracy"], "%.1f%%"), "%d" % tl["n"], verdict(tl)),
        ("identity", TARGETS["identity"], "%s id switches / possession (%s before linking); jersey: %s" % (
            v(idn["id_switches_per_possession"]), v(idn["fragment_switches_per_possession"]),
            "not measurable" if idn["jersey_read_rate"] is None else idn["jersey_read_rate"]),
         "%d possessions" % idn["possessions"], "reported"),
        ("generalisation", TARGETS["generalisation"], "not measurable", "-", "-"),
    ]
    L = ["STAGE 1 SCORECARD  %s (held-out game) vs SportVU %s  build %s (%s)" % (rep["game"], rep["sportvu_game"], rep["build"], rep["build_dir"]),
         "truth: %d testable of %d window frames; untestable %.1f%% (no truth H fits). Every row below is on testable frames only." % (
             rep["testable"], rep["truth_frames"], 100 * rep["untestable_share"]),
         "untestable by cause (sportvu/truth.py): " + ", ".join("%s %d" % kv for kv in rep["untestable_by_cause"].items()),
         ""]
    w0 = max(len(r[0]) for r in rows)
    for name, tgt, val, n, ver in rows:
        L.append("%-*s  %-8s  %s" % (w0, name, ver, val))
        L.append("%-*s            target %s; n = %s" % (w0, "", tgt, n))
    L += ["", "not measurable:", "  jersey-read rate: " + (idn["jersey_not_measurable"] or "-"),
          "  generalisation: " + M["generalisation"]["not_measurable"], "", "caveats:"]
    L += ["  - " + c for c in rep["caveats"]]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 scorecard for one build of a game's SportVU windows.")
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--game", default="gsw_phx_2016")
    ap.add_argument("--name", default=None, help="build label in the output name (default: the build dir's name)")
    args = ap.parse_args()
    build_dir = args.build_dir.resolve()
    name = args.name or build_dir.name
    sv_game = sportvu_name(args.game)
    moments = json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (sv_game + "_moments.json")).read_text())["moments"]
    index = SportVUIndex(moments)
    poss = shot_clock_index(moments)
    mirror = json.loads((config.REPORTS_DIR / ("sportvu_sync_%s.json" % args.game)).read_text())["direction_resolution"]["mirror"]
    wpath = build_dir / (args.game + "_windows.json")
    windows = json.loads((wpath if wpath.exists() else BUILD_DIR / (args.game + "_windows.json")).read_text())
    refs = References()
    wins = []
    for wd in windows:
        wins.append(score_window(wd, build_dir, refs, index, poss, mirror))
        print("  %s: %s" % (wd["window"], wins[-1].get("skipped") or "%d testable" % wins[-1]["testable"]), flush=True)
    agg = aggregate(wins, refs.wide_thr)
    rep = {"game": args.game, "sportvu_game": sv_game, "build": name,
           "build_dir": str(build_dir.relative_to(config.PROJECT_ROOT)) if build_dir.is_relative_to(config.PROJECT_ROOT) else str(build_dir),
           "mirror": mirror, "date": datetime.date.today().isoformat(), "targets": TARGETS, **agg,
           "per_window": {w["window"]: ({"skipped": w["skipped"]} if "skipped" in w else
                                        {"testable": w["testable"], "truth_frames": w["truth_frames"], "coverage": w["coverage"],
                                         "missed": w["missed"], "in_frame": w["in_frame"], "ghosts": w["ghosts"], "boxes": w["boxes"],
                                         "referee_boxes": w["ref_boxes"]}) for w in wins},
           "method": {"match_ft": MATCH_FT, "identity_match_ft": ID_MATCH_FT, "miss_ft": MISS_FT, "ghost_ft": GHOST_FT,
                      "line_target_px": LINE_TARGET_PX, "referee_iou": REF_IOU, "referee_conf": config.PLAYER_CONF,
                      "shot_reset_s": SHOT_RESET_S, "min_possession_frames": MIN_POSS_FRAMES, "wide_threshold_v2": refs.wide_thr},
           "caveats": [
               "testable frames are the B4 set: frames where the V3 H matched >= 6 players; every build is scored on them until ROADMAP R1.3 widens the truth, so the set leans toward frames V3 already handled",
               "the truth H is fitted through V3's own boxes (0.33 ft median residual on the feet); its px residual is the floor printed beside the line error, and only a floor: lines far from the fitted players extrapolate",
               "local windowed rebuild of the harvest sections (25 windows of running clock), not the production artifacts",
               "wide reference = gate v2 at its v2 threshold, which was trained on human-verified harvest frames including phx (in-sample here, so closer to the human label than a held-out gate); live = running clock read by OCR at 1 s resolution",
               "referee class = models/player_detector.pt's own referee class at the player confidence; its referee recall has never been measured against human labels, and a box with IoU >= 0.5 to a referee detection can be a player standing beside the referee (see referee_boxes_within_ghost_ft_of_a_player)",
               "id switches are counted between consecutive testable frames (sparse samples), so a switch and switch-back between samples is missed: a lower bound",
               "team accuracy uses the best A/B to team mapping per window, an upper bound on what a fixed mapping would score",
           ]}
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = "%s__%s" % (args.game, name)
    (SCORECARD_DIR / (stem + ".json")).write_text(json.dumps(rep, indent=1))
    txt = render_txt(rep)
    (SCORECARD_DIR / (stem + ".txt")).write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

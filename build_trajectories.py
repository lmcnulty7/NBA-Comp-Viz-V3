#!/usr/bin/env python
"""
build_trajectories.py — Component A: per-player court trajectories + correction.

Runs a clip through tracking (BoT-SORT) + homography (grid court tracker), projects
each tracked player's foot-point to court feet per frame, then cleans the per-player
paths (jump rejection + smoothing — court/trajectories.py). This is the bridge from
"perception" to "analytics": clean court-space trajectories per player.

Component B (identity + foot-point stability) is wired in here:
  • FootPointStabilizer (detect/footpoint.py) fixes occlusion-clipped box bottoms
    and jitter in PIXEL space before projection.            A/B lever: --no-stab
  • Offline fragment linking (detect/reid.py): CLIP torso embeddings + court-space
    motion gating merge the many BoT-SORT fragments of one player into one
    canonical identity, ACROSS occlusions / camera cuts / gate gaps.  --no-reid
  • Native BoT-SORT re-ID (botsort_reid.yaml) is a config lever: TRACKER_REID=0.
Physics metrics (impossible steps, p99 step, off-court %) are printed every run so
A/Bs are self-serve — these are the label-free numbers the DEVLOG tables use.

Outputs: data/tracking/<clip>_trajectories.json, <clip>_identity.json (merge log),
and a synced review video (broadcast | top-down minimap).

Examples
────────
  python build_trajectories.py --source ".../curry_q1_clip.mp4" --start 11520 --max-frames 200 --use-gate
  python build_trajectories.py --source ".../clip.mp4" --no-reid --no-stab   # Component-B off (baseline)
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from court.court33 import COURT_LENGTH_FT, COURT_WIDTH_FT, court33_segments, court_ft_to_px, draw_court_topdown
from court.trajectories import clean_trajectories
from videoseq import SeqReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_trajectories")

# generous court bound (ft) — drop horizon blow-ups; clean_paths handles the rest
X_LO, X_HI = -25.0, COURT_LENGTH_FT + 25
Y_LO, Y_HI = -25.0, COURT_WIDTH_FT + 25

IMPOSSIBLE_STEP_FT = 3.0   # per 0.1 s (≈30 ft/s) — the DEVLOG 2026-07-05 A/B metric


def physics_report(series: dict, fps: float, stride: int) -> dict:
    """Label-free physics metrics on RAW court positions.
    series: {tid: [(frame, x, y), ...]}. Steps are only measured between
    CONSECUTIVE processed frames (gap == stride), so merge gaps don't pollute it."""
    step_dt = stride / fps
    dists, n_pts, n_off = [], 0, 0
    for pts in series.values():
        for k, (f, x, y) in enumerate(pts):
            n_pts += 1
            if not (0 <= x <= COURT_LENGTH_FT and 0 <= y <= COURT_WIDTH_FT):
                n_off += 1
            if k and f - pts[k - 1][0] == stride:
                dists.append(float(np.hypot(x - pts[k - 1][1], y - pts[k - 1][2])))
    if not dists:
        return {"steps": 0}
    arr = np.asarray(dists)
    lim = IMPOSSIBLE_STEP_FT * (step_dt / 0.1)
    return {
        "steps": len(arr),
        "impossible_step_pct": round(100 * float((arr > lim).mean()), 1),
        "step_p50_ft": round(float(np.percentile(arr, 50)), 2),
        "step_p99_ft": round(float(np.percentile(arr, 99)), 1),
        "off_court_pct": round(100 * n_off / max(n_pts, 1), 1),
    }


def id_color(tid):
    rng = np.random.default_rng(tid * 9973 + 1)
    return tuple(int(c) for c in rng.integers(70, 256, size=3))


def draw_paths(per_track_series, title):
    """per_track_series: {tid: [(frame, x, y, ...)]} → top-down court image with paths."""
    img, scale, margin = draw_court_topdown()
    for tid, pts in per_track_series.items():
        xy = np.array([(p[1], p[2]) for p in pts], np.float32)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 2:
            continue
        px = court_ft_to_px(xy, scale, margin)
        cv2.polylines(img, [px], False, id_color(tid), 1, cv2.LINE_AA)
    cv2.putText(img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Build + clean per-player court trajectories.")
    ap.add_argument("--source", type=Path, default=config.VIDEO_DIR / "curry_q1_clip.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--use-gate", action="store_true", help="Skip non-court frames via the gate.")
    # Component B levers (identity + foot-point stability)
    ap.add_argument("--no-stab", action="store_true", help="Disable foot-point stabilization (A/B).")
    ap.add_argument("--no-reid", action="store_true", help="Disable offline fragment linking (A/B).")
    ap.add_argument("--no-teams", action="store_true", help="Disable team classification + linker veto (A/B).")
    ap.add_argument("--no-video", action="store_true", help="Skip the review-video render (batch runs).")
    ap.add_argument("--save-crops", action="store_true",
                    help="Persist the torso-crop pool to tracking/crops/<clip>_crops.tar "
                         "(one jpeg per crop, named <tid>_<seq>.jpg) — every later OCR/"
                         "team/re-ID experiment becomes a re-vote, not a re-track.")
    ap.add_argument("--pregate", action="store_true",
                    help="Coarse gate-only pass first; the full chain only runs inside "
                         "live segments (harvest-scale speedup; implies --use-gate).")
    ap.add_argument("--reid-sim", type=float, default=None, help="Override REID_SIM_MIN.")
    ap.add_argument("--reid-max-gap", type=float, default=None, help="Override REID_MAX_GAP_SEC.")
    # clean_paths params (defaults = the external project's basketball settings)
    ap.add_argument("--jump-sigma", type=float, default=3.5)
    ap.add_argument("--min-jump-dist", type=float, default=0.6)
    ap.add_argument("--max-jump-run", type=int, default=18)
    ap.add_argument("--smooth-window", type=int, default=9)
    ap.add_argument("--smooth-poly", type=int, default=2)
    args = ap.parse_args()

    config.set_seed()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    from detect import FootPointStabilizer, PlayerTracker
    from detect.reid import TorsoCropCollector, build_fragments, link_fragments, mean_embeddings
    from detect.teams import TeamClassifier, team_contact_sheet
    from gate.backbones import get_backbone
    from court import CourtMapper

    device = config.get_device()
    gate = None
    if args.use_gate or args.pregate:
        from gate.trained_head import TrainedHeadGate
        thr = json.loads(config.THRESHOLDS_PATH.read_text())["trained"]
        gate = TrainedHeadGate.load(config.HEAD_PATH, backbone=get_backbone("clip", device), threshold=thr)
    tracker = PlayerTracker(device=device)
    mapper = CourtMapper()
    stab = None if args.no_stab else FootPointStabilizer()
    collector = None if args.no_reid else TorsoCropCollector()

    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = SeqReader(cap)   # sequential grab(), never per-frame keyframe seeks

    # ── pre-gate: coarse live segments so the chain never scans dead footage ──
    intervals = None
    if args.pregate:
        pg_stride, pg_pad = config.pregate_params(fps)
        log.info("Pre-gate: coarse scan (stride %d @ %.2f fps) …", pg_stride, fps)
        hits, i = [], args.start
        while i < total:
            ret, frame = reader.read(i)
            if not ret:
                break             # sequential decode hit EOF/corruption — later reads can't succeed
            if gate.is_court_visible(frame):
                hits.append(i)
            i += pg_stride
        intervals = []
        for h in hits:
            a, b = h - pg_pad, h + pg_pad
            if intervals and a <= intervals[-1][1]:
                intervals[-1][1] = b
            else:
                intervals.append([a, b])
        live_frames = sum(b - a for a, b in intervals)
        log.info("Pre-gate: %d live segments, %.1f of %.1f min (%.0f%%)",
                 len(intervals), live_frames / fps / 60, (total - args.start) / fps / 60,
                 100 * live_frames / max(total - args.start, 1))

    raw = defaultdict(dict)       # {track_id: {frame_idx: (x, y)}}
    boxes_by_frame = {}           # {frame_idx: [(track_id, bbox)]}
    extrap = {}                   # {(track_id, frame_idx): bool}  — projection outside keypoint hull
    hull_by_frame = {}            # {frame_idx: keypoint convex hull (pixel)}
    hom_by_frame = {}             # {frame_idx: CourtHomography}  — for reprojecting the court model
    frame_order = []
    idx, n, n_H = args.start, 0, 0
    n_reads = 0   # successful frame reads — distinguishes I/O failure from gate rejection
    seg_i = 0   # pre-gate interval pointer
    log.info("Tracking + projecting %s from frame %d …", args.source.name, args.start)
    while idx < total and n < args.max_frames:
        if intervals is not None:
            while seg_i < len(intervals) and idx > intervals[seg_i][1]:
                seg_i += 1
            if seg_i >= len(intervals):
                break
            if idx < intervals[seg_i][0]:      # jump the dead gap entirely
                idx = intervals[seg_i][0]
        ret, frame = reader.read(idx)
        if not ret:
            break
        n_reads += 1
        if gate is not None and not gate.is_court_visible(frame):
            idx += args.stride
            continue
        tracks = tracker.update(frame, idx, idx / fps)
        mapper.update(frame)
        if collector is not None:
            collector.add(frame, tracks)
        feet = stab.stabilize(tracks) if stab is not None else \
            {t.track_id: (t.foot_point, False) for t in tracks}
        frame_order.append(idx)
        boxes_by_frame[idx] = [(t.track_id, [int(v) for v in t.bbox]) for t in tracks]
        hull_by_frame[idx] = mapper.keypoint_hull
        hom_by_frame[idx] = mapper.last_hom
        if mapper.has_homography:
            n_H += 1
            for t in tracks:
                foot, _ = feet[t.track_id]
                c = mapper.court_pos(foot)
                if np.isfinite(c).all() and X_LO <= c[0] <= X_HI and Y_LO <= c[1] <= Y_HI:
                    raw[t.track_id][idx] = (float(c[0]), float(c[1]))
                    extrap[(t.track_id, idx)] = mapper.is_extrapolated(foot)
        idx += args.stride
        n += 1
    cap.release()

    # A build that processed nothing must FAIL LOUDLY, never write "{}" and exit 0
    # (Colab run 2 did exactly that, silently — reads distinguish the cause).
    if n == 0:
        raise SystemExit(
            f"build processed 0 frames (successful reads: {n_reads}, total frames: {total}). "
            + ("Video unreadable by cv2 — unsupported codec (AV1 on Colab, run-7"
               " postmortem: ffprobe the file), FUSE/network-mounted storage, or"
               " corruption. Fix the file; do not rerun as-is." if n_reads == 0 else
               "All read frames were gate-rejected or outside pre-gate intervals — "
               "gate threshold/domain issue, or the section is genuinely dead footage."))

    n_fragments = len(raw)

    # ── Components B + C1: teams (veto) → offline fragment linking → canonical team ──
    idmap, merges, skipped = {}, [], []
    team_by_canon, team_cols = {}, {}
    team_clf = TeamClassifier()
    if collector is not None and raw:
        backbone = gate.backbone if gate is not None else get_backbone("clip", device)
        log.info("Embedding torso crops (re-ID + teams, %d tracks) …", len(collector.crops))
        crop_embs = collector.embed(backbone)
        # teams BEFORE linking: the linker uses them as a hard cross-team veto.
        # Teams cluster on Lab jersey COLOR of the crops; CLIP embs are re-ID-only.
        team_of = {}
        if not args.no_teams and team_clf.fit(collector.crops):
            team_of = {tid: t for tid, (t, _) in team_clf.assign(collector.crops).items()}
        frags = build_fragments(raw, fps, mean_embeddings(crop_embs))
        idmap, merges, skipped = link_fragments(
            frags, sim_min=args.reid_sim, max_gap_sec=args.reid_max_gap,
            team_of=team_of or None)
        if any(idmap[t] != t for t in idmap):
            merged = defaultdict(dict)
            for tid, d in raw.items():
                cid = idmap.get(tid, tid)
                for f, xy in d.items():
                    merged[cid].setdefault(f, xy)   # no-overlap gate ⇒ collisions shouldn't occur
            raw = merged
            extrap = {(idmap.get(tid, tid), f): v for (tid, f), v in extrap.items()}
            boxes_by_frame = {f: [(idmap.get(tid, tid), bb) for tid, bb in lst]
                              for f, lst in boxes_by_frame.items()}
        # canonical team = one vote over ALL member fragments' crops combined
        if team_of:
            canon_crops = defaultdict(list)
            for tid, lst in collector.crops.items():
                canon_crops[idmap.get(tid, tid)].extend(lst)
            team_by_canon = {cid: t for cid, (t, _) in team_clf.assign(canon_crops).items()}
            team_cols = team_clf.team_colors_bgr(collector.crops, team_of)
            config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
            sheet = config.VIZ_DIR / f"team_clusters_{args.source.stem}.png"
            team_contact_sheet(collector.crops, team_of, sheet)
            log.info("team contact sheet → %s", sheet)

    if args.save_crops and collector is not None and collector.crops:
        # jpeg tar, one member per crop, <fragment_tid>_<seq>.jpg — fragment ids
        # (pre-link) so consumers can re-vote teams/OCR and re-link independently
        import tarfile, io, time as _time
        crops_dir = config.TRACKING_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        tar_path = crops_dir / f"{args.source.stem}_crops.tar"
        n_saved = 0
        with tarfile.open(tar_path, "w") as tf:
            for tid, lst in sorted(collector.crops.items()):
                for i, crop in enumerate(lst):
                    ok, enc = cv2.imencode(".jpg", crop[:, :, ::-1],
                                           [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if not ok:
                        continue
                    info = tarfile.TarInfo(f"{tid}_{i:03d}.jpg")
                    info.size = len(enc)
                    info.mtime = int(_time.time())
                    tf.addfile(info, io.BytesIO(enc.tobytes()))
                    n_saved += 1
        log.info("crop archive: %d crops (%d fragments) → %s (%.1f MB)",
                 n_saved, len(collector.crops), tar_path,
                 tar_path.stat().st_size / 1e6)

    raw_series = {tid: [(f, x, y) for f, (x, y) in sorted(d.items())] for tid, d in raw.items()}

    # ── physics + identity diagnostics (the A/B numbers) ─────────────────────────
    phys = physics_report(raw_series, fps, args.stride)
    ppf = [sum(1 for tid in raw if f in raw[tid]) for f in frame_order]
    med_ppf = int(np.median([p for p in ppf if p])) if any(ppf) else 0
    identity = {
        "fragments": n_fragments,
        "canonical_tracks": len(raw),
        "merges": len(merges),
        "ambiguous_refused": sum(1 for s in skipped if s["reason"] == "ambiguous"),
        "team_vetoed": sum(1 for s in skipped if s["reason"] == "team_veto"),
        "median_players_per_frame": med_ppf,
        "id_churn_before": round(n_fragments / max(med_ppf, 1), 2),
        "id_churn_after": round(len(raw) / max(med_ppf, 1), 2),
        "footpoint_clip_fixes": stab.n_clip_fixes if stab is not None else None,
        "camera_cut_resets": tracker.n_resets,
    }
    # team diagnostics: track counts per team + per-frame balance (expect ~5v5 live)
    if team_by_canon:
        balance = [(sum(1 for tid in raw if f in raw[tid] and team_by_canon.get(tid) == 0),
                    sum(1 for tid in raw if f in raw[tid] and team_by_canon.get(tid) == 1))
                   for f in frame_order]
        balance = [b for b in balance if sum(b)]
        teams_diag = {
            "silhouette": team_clf.silhouette,
            "tracks_team_A": sum(1 for t in team_by_canon.values() if t == 0),
            "tracks_team_B": sum(1 for t in team_by_canon.values() if t == 1),
            "tracks_abstained": sum(1 for t in team_by_canon.values() if t is None),
            "frame_balance_median": [int(np.median([b[0] for b in balance])),
                                     int(np.median([b[1] for b in balance]))] if balance else None,
        }
        identity["teams"] = teams_diag
    log.info("── PHYSICS (raw, pre-clean) ──   %s", json.dumps(phys))
    log.info("── IDENTITY ──                   %s", json.dumps(identity))

    cleaned = clean_trajectories(
        raw, frame_order, jump_sigma=args.jump_sigma, min_jump_dist=args.min_jump_dist,
        max_jump_run=args.max_jump_run, smooth_window=args.smooth_window, smooth_poly=args.smooth_poly)

    n_edited = sum(1 for pts in cleaned.values() for p in pts if p[3])
    n_pts = sum(len(p) for p in cleaned.values())
    log.info("%d frames (%d with homography) · %d tracks · %d trajectory points · %d corrected (%.1f%%)",
             n, n_H, len(cleaned), n_pts, n_edited, 100 * n_edited / max(n_pts, 1))

    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.TRACKING_DIR / f"{args.source.stem}_trajectories.json"
    out_json.write_text(json.dumps({
        str(tid): {"team": team_by_canon.get(tid),
                   "raw": raw_series.get(tid, []),
                   "cleaned": [[f, x, y, ed] for f, x, y, ed in pts]}
        for tid, pts in cleaned.items()}, indent=2))

    # identity audit trail: every accepted merge with its evidence, every refusal
    # which team id is the light kit (for downstream evals that label light/dark)
    light_team = None
    if len(team_cols) == 2:
        light_team = max(team_cols, key=lambda t: sum(w * c for w, c in
                                                      zip((0.114, 0.587, 0.299), team_cols[t])))
    out_id = config.TRACKING_DIR / f"{args.source.stem}_identity.json"
    out_id.write_text(json.dumps({
        "physics_raw": phys, "identity": identity, "merges": merges,
        "refused": skipped,
        "idmap": {str(k): v for k, v in idmap.items() if k != v},
        "team_by_track": {str(k): v for k, v in team_by_canon.items()},
        "team_colors_bgr": {str(t): [int(v) for v in c] for t, c in team_cols.items()},
        "light_team": light_team,
    }, indent=2))
    log.info("identity audit → %s", out_id)

    # ── synced review video: broadcast (boxes+IDs) | top-down minimap (cleaned dots) ──
    lookup = {tid: {f: (x, y) for f, x, y, _ in pts} for tid, pts in cleaned.items()}
    pos = {f: i for i, f in enumerate(frame_order)}
    scale, margin, trail = 9.0, 15, 12

    def track_color(tid):
        """Team jersey color when the track's team is known, else the per-id color."""
        t = team_by_canon.get(tid)
        return team_cols.get(t, id_color(tid)) if t is not None else id_color(tid)

    def minimap(frame_idx):
        img, _, _ = draw_court_topdown(scale, margin)
        h_mm = img.shape[0]
        i = pos[frame_idx]
        for tid, fmap in lookup.items():
            if frame_idx not in fmap or not np.isfinite(fmap[frame_idx]).all():
                continue
            col = id_color(tid)
            tp = [fmap[frame_order[k]] for k in range(max(0, i - trail), i + 1)
                  if frame_order[k] in fmap and np.isfinite(fmap[frame_order[k]]).all()]
            if len(tp) >= 2:
                cv2.polylines(img, [court_ft_to_px(np.array(tp, np.float32), scale, margin)], False, col, 1, cv2.LINE_AA)
            cx, cy = (int(v) for v in court_ft_to_px(fmap[frame_idx], scale, margin)[0])
            # dot = team jersey color (id color if team unknown);
            # solid = position constrained by nearby landmarks; hollow ring = extrapolated (unreliable)
            dot = track_color(tid)
            if extrap.get((tid, frame_idx), True):
                cv2.circle(img, (cx, cy), 6, dot, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), 6, dot, -1, cv2.LINE_AA)
            cv2.putText(img, str(tid), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(img, f"frame {frame_idx}", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
        cv2.putText(img, "solid=trusted  hollow=extrapolated", (10, h_mm - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        return img

    if args.no_video:
        log.info("trajectories → %s (video render skipped)", out_json)
        return

    out_mp4 = config.TRACKING_DIR / f"{args.source.stem}_trajectories.mp4"
    if out_mp4.exists():
        out_mp4.unlink()
    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    render_reader = SeqReader(cap)
    writer = None
    log.info("Rendering synced review video (broadcast | court minimap) …")
    for f in frame_order:
        ret, frame = render_reader.read(f)
        if not ret:
            continue
        # reproject the court model (court→pixel) so the homography fit is visible:
        # green lines that don't sit on the real painted lines = the warp
        hom = hom_by_frame.get(f)
        if hom is not None and hom.is_valid:
            for P1, P2 in court33_segments():
                a = hom.to_pixel_batch(np.array([P1], np.float32))[0]
                b = hom.to_pixel_batch(np.array([P2], np.float32))[0]
                if np.isfinite([a[0], a[1], b[0], b[1]]).all():
                    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 1, cv2.LINE_AA)
        hull = hull_by_frame.get(f)
        if hull is not None:
            cv2.polylines(frame, [hull.astype(np.int32)], True, (0, 220, 255), 1, cv2.LINE_AA)
        for tid, bbox in boxes_by_frame.get(f, []):
            x1, y1, x2, y2 = bbox
            col = track_color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, str(tid), (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        mm = minimap(f)
        mm = cv2.resize(mm, (int(mm.shape[1] * frame.shape[0] / mm.shape[0]), frame.shape[0]))
        combo = np.hstack([frame, mm])
        if writer is None:
            h, w = combo.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), fps / args.stride, (w, h))
        writer.write(combo)
    cap.release()
    if writer:
        writer.release()

    log.info("trajectories → %s", out_json)
    log.info("review video (broadcast | minimap) → %s", out_mp4)


if __name__ == "__main__":
    main()

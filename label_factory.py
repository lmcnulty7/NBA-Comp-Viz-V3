#!/usr/bin/env python
"""
label_factory.py — self-training data harvester for the court detector.

Feed it games (YouTube URLs or local video files); it produces training-ready,
projection-labeled frames. Trust chain: only frames the CURRENT model solves AND
the line-snap verifies against the actual court pixels are accepted — the same
evidence standard that built the original snapped dataset. Everything else is
rejected. You spot-check a ranked sample; you never hand-label.

  harvest:  download (6 × 4-min sections spread across each game, 720p)
            → CLIP live-play gate → grid model + guarded line-snap per sampled frame
            → STRICT accept gate:  ≥12 confident grid kps, sane H, snap converged
              with ≥80 line matches, median residual ≤ 1.0 px, matches spread over
              ≥10% of the frame
            → per accepted frame: image + 33-pt & grid labels (identical format to
              generate_labels.py) + review overlay.
            Output: data/harvest/<video_id>/{frames,labels_33,labels_grid,review}/
                    + harvest_rank.tsv (worst-first, verify_projections-compatible)

  spot-check (you):
     /opt/anaconda3/bin/python verify_projections.py \\
         data/harvest/<vid>/harvest_rank.tsv data/harvest/<vid>/review \\
         data/harvest/<vid>/harvest_verdicts.tsv
     Frames you mark no_lineup are excluded at build time.

  build:    merge base datasets (court_pose33 / court_pose_grid) + all accepted
            harvest frames (minus your no_lineup verdicts) into *_v2 datasets +
            Colab zips. Harvest frames go to TRAIN ONLY — val stays the original
            279 verified frames so benchmarks remain comparable across retrains.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/anaconda3/bin/python label_factory.py harvest <url|file> [...]
  /opt/anaconda3/bin/python label_factory.py build
"""
from __future__ import annotations
import argparse, datetime, glob, json, os, re, shutil, subprocess, sys
import cv2, numpy as np

import config
from court.court33 import COURT_VERTICES_33, court33_segments, court33_curves
from court.grid import GRID_FT
from court.snap_track import (ridge_field, match_lines, project, h_sane,
                              MIN_SNAP_MATCH, RANSAC_PX)
from generate_labels import visible_court_bbox, keypoint_fields

cv2.setRNGSeed(42)
HARVEST_DIR = "data/harvest"
N_SECTIONS, SECTION_MIN = 6, 4          # per game: sections × minutes
STEP_S = 2.0                            # sample interval inside sections
KP_CONF = 0.5
# strict accept gate — poison prevention, err on rejection. The residual bar is
# RESOLUTION-NORMALIZED: 1.1 px at 640-px scale (the offline pipeline's frame size);
# a 1280-wide frame gets 2.2 px, which is the same physical accuracy. Calibrated on
# measured distributions from 2 unseen games (2026-07-05): accepts the best ~20%,
# whose label noise is at or below the original dataset's box-center noise.
MIN_KP, MIN_MATCH, MIN_SPREAD = 12, 25, 0.12
MAX_RES_640 = 1.1
SNAP_RADII, EVAL_R, MAX_CORNER = (8, 5), 6, 40.0


def video_id(src):
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{6,})", src)
    return m.group(1) if m else re.sub(r"\W+", "_", os.path.splitext(os.path.basename(src))[0])[:24]


def download_sections(url, vid):
    """6 × 4-min sections spread 12%..88% through the game → local mp4 paths."""
    out = subprocess.run(["yt-dlp", "--print", "%(duration)s", "--skip-download", url],
                         capture_output=True, text=True)
    dur = float(out.stdout.strip().splitlines()[-1])
    vdir = os.path.join(HARVEST_DIR, vid, "video")
    os.makedirs(vdir, exist_ok=True)
    cmd = ["yt-dlp", "-q", "--no-update", "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]",
           "-o", os.path.join(vdir, "s%(section_start)d.%(ext)s")]
    for k in range(N_SECTIONS):
        c = (0.12 + 0.76 * k / max(N_SECTIONS - 1, 1)) * dur
        a = max(0, int(c - SECTION_MIN * 30))
        cmd += ["--download-sections", f"*{a}-{a + SECTION_MIN * 60}"]
    subprocess.run(cmd + [url], check=True)
    return sorted(glob.glob(os.path.join(vdir, "*.mp4")))


GATE_THR = 0.35   # recall-oriented for harvesting: the pipeline threshold (0.70) was
                  # tuned on the user's own clips and rejects live frames of unfamiliar
                  # broadcasts (2024 DEN@MIN live frames score 0.47-0.66; non-court ~0.03).
                  # A gate false-positive is harmless here — the solve gate rejects it.


def load_gate():
    from gate.backbones import get_backbone
    from gate.trained_head import TrainedHeadGate
    return TrainedHeadGate.load(config.HEAD_PATH,
                                backbone=get_backbone("clip", config.get_device()),
                                threshold=GATE_THR)


def solve_frame(model, frame):
    """Grid model + guarded snap. Returns (P ft→px, metrics) if the strict gate
    passes, else (None, reject_reason)."""
    h, w = frame.shape[:2]
    r = model.predict(frame, device=config.get_device(), verbose=False)[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return None, "no_det"
    b = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[b].cpu().numpy()
    kc = (r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None
          else np.ones(len(xy)))
    keep = [(i, xy[i]) for i in range(len(xy))
            if kc[i] >= KP_CONF and 0 < xy[i][0] < w and 0 < xy[i][1] < h]
    if len(keep) < MIN_KP:
        return None, "few_kp"
    src = np.array([p for _, p in keep], np.float32).reshape(-1, 1, 2)
    dst = np.array([GRID_FT[i] for i, _ in keep], np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)
    if H is None or not h_sane(H, w, h):
        return None, "insane_H"
    ridge = ridge_field(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ridge is None:
        return None, "no_evidence"
    # anchors = the model's own grid keypoints. They stay in every refit so sparse
    # line matches can only POLISH the H, never warp it globally (the offline snap's
    # design — dropping the anchors made the refit overfit clustered line evidence).
    a_ft = np.array([GRID_FT[i] for i, _ in keep], np.float32)
    a_px = np.array([p for _, p in keep], np.float32)
    P = P0 = np.linalg.inv(H.astype(np.float64))
    for radius in SNAP_RADII:
        ft, px = match_lines(P, ridge, w, h, radius)
        if len(ft) < MIN_SNAP_MATCH:
            break
        Hn, _ = cv2.findHomography(np.concatenate([ft, a_ft]).reshape(-1, 1, 2),
                                   np.concatenate([px, a_px]).reshape(-1, 1, 2),
                                   cv2.RANSAC, RANSAC_PX)
        if Hn is None:
            break
        P = Hn.astype(np.float64)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), P @ np.linalg.inv(P0)).reshape(-1, 2)
    if not np.isfinite(moved).all() or np.linalg.norm(moved - corners, axis=1).max() > MAX_CORNER:
        P = P0          # snap ran away → keep the model's H; the residual gate decides
    ft, px = match_lines(P, ridge, w, h, EVAL_R)
    if len(ft) < MIN_MATCH:
        return None, "few_matches"
    res = float(np.median(np.linalg.norm(px - project(P, ft), axis=1)))
    lo, hi = px.min(0), px.max(0)
    spread = float((hi[0] - lo[0]) * (hi[1] - lo[1]) / (w * h))
    if res > MAX_RES_640 * max(w, h) / 640.0:
        return None, "residual"
    if spread < MIN_SPREAD:
        return None, "clustered"
    return P, {"res_px": res, "n_match": len(ft), "n_kp": len(keep), "spread": spread}


def draw_overlay(img, P, hud):
    h, w = img.shape[:2]
    for P1, P2 in court33_segments():
        seg = project(P, np.stack([P1, P2]))
        if np.isfinite(seg).all():
            cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    for poly in court33_curves():
        pp = project(P, poly)
        for a, b in zip(pp[:-1], pp[1:]):
            if (np.isfinite(a).all() and np.isfinite(b).all()
                    and max(abs(a[0]), abs(b[0])) < 4 * w and max(abs(a[1]), abs(b[1])) < 4 * h):
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def harvest(sources, max_frames):
    from ultralytics import YOLO
    model = YOLO(str(config.COURT_GRID_WEIGHTS))
    gate = load_gate()
    for srcarg in sources:
        vid = video_id(srcarg)
        vroot = os.path.join(HARVEST_DIR, vid)
        if os.path.exists(os.path.join(vroot, "harvest_manifest.tsv")):
            print(f"{vid}: already harvested — skipping (delete {vroot} to redo)")
            continue
        for sub in ("frames", "labels_33", "labels_grid", "review"):
            os.makedirs(os.path.join(vroot, sub), exist_ok=True)
        vids = ([srcarg] if os.path.exists(srcarg) else download_sections(srcarg, vid))
        rows, rejects = [], {}
        for vp in vids:
            cap = cv2.VideoCapture(vp)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            sec = int(re.search(r"s(\d+)", os.path.basename(vp)).group(1)) if re.search(r"s(\d+)", os.path.basename(vp)) else 0
            t = 0.0
            while t < dur and len(rows) < max_frames:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                t += STEP_S
                if not ok:
                    continue
                if not gate.is_court_visible(frame):
                    rejects["gate"] = rejects.get("gate", 0) + 1
                    continue
                P, m = solve_frame(model, frame)
                if P is None:
                    rejects[m] = rejects.get(m, 0) + 1
                    continue
                h, w = frame.shape[:2]
                stem = f"hv_{vid}_{sec + int(t - STEP_S):06d}"
                bbox = visible_court_bbox(P, w, h)
                if bbox is None:
                    rejects["bbox"] = rejects.get("bbox", 0) + 1
                    continue
                f33, n33 = keypoint_fields(P, COURT_VERTICES_33, w, h)
                fgr, ngr = keypoint_fields(P, GRID_FT, w, h)
                head = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                cv2.imwrite(os.path.join(vroot, "frames", stem + ".jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(vroot, "labels_33", stem + ".txt"), "w") as f:
                    f.write(head + " " + " ".join(f33) + "\n")
                with open(os.path.join(vroot, "labels_grid", stem + ".txt"), "w") as f:
                    f.write(head + " " + " ".join(fgr) + "\n")
                rows.append((stem, m["res_px"], m["n_match"], m["n_kp"], m["spread"], n33, ngr,
                             draw_overlay(frame.copy(), P,
                                          f"{stem}  res={m['res_px']:.2f}px  matches={m['n_match']}  kps={m['n_kp']}")))
            cap.release()
        rows.sort(key=lambda r: -r[1])                    # worst residual first
        with open(os.path.join(vroot, "harvest_rank.tsv"), "w") as f:
            f.write("rank\tfile\tframe\tres_px\tn_match\tn_kp\tspread\tn33\tngrid\n")
            for rank, (stem, res, nm, nk, sp, n33, ngr, ov) in enumerate(rows):
                fn = f"{rank:04d}_{stem}.jpg"
                cv2.imwrite(os.path.join(vroot, "review", fn), ov)
                f.write(f"{rank}\t{fn}\t{stem}\t{res:.3f}\t{nm}\t{nk}\t{sp:.3f}\t{n33}\t{ngr}\n")
        shutil.copy(os.path.join(vroot, "harvest_rank.tsv"),
                    os.path.join(vroot, "harvest_manifest.tsv"))
        with open(os.path.join(HARVEST_DIR, "index.tsv"), "a") as f:
            f.write(f"{vid}\t{srcarg}\t{datetime.date.today()}\t{len(rows)}\t{sum(rejects.values())}\n")
        print(f"{vid}: accepted {len(rows)}   rejected {sum(rejects.values())} {rejects}")
        print(f"  spot-check: /opt/anaconda3/bin/python verify_projections.py "
              f"{vroot}/harvest_rank.tsv {vroot}/review {vroot}/harvest_verdicts.tsv")


def build():
    from build_pose_datasets import FLIP_33, FLIP_GRID
    sets = [("data/court_pose33", "labels_33", "data/court_pose33_v2", 33, FLIP_33),
            ("data/court_pose_grid", "labels_grid", "data/court_pose_grid_v2", 91, FLIP_GRID)]
    # collect accepted harvest stems (minus user no_lineup verdicts)
    picks = []                                            # (vroot, stem)
    for vroot in sorted(glob.glob(os.path.join(HARVEST_DIR, "*"))):
        man = os.path.join(vroot, "harvest_manifest.tsv")
        if not os.path.isfile(man):
            continue
        bad = set()
        vfile = os.path.join(vroot, "harvest_verdicts.tsv")
        if os.path.exists(vfile):
            with open(vfile) as f:
                next(f)
                bad = {t.split("\t")[0] for t in f if t.rstrip().endswith("no_lineup")}
        with open(man) as f:
            next(f)
            for line in f:
                stem = line.split("\t")[2]
                if stem not in bad:
                    picks.append((vroot, stem))
        if bad:
            print(f"{os.path.basename(vroot)}: excluding {len(bad)} no_lineup frames")
    if not picks:
        raise SystemExit("no harvested frames found — run harvest first")

    for base, lblsub, out, nkp, flip in sets:
        for part in ("images/train", "images/val", "labels/train", "labels/val"):
            d = os.path.join(out, part)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        n_base = n_hv = 0
        for part in ("train", "val"):
            for img in glob.glob(os.path.join(base, "images", part, "*.jpg")):
                stem = os.path.splitext(os.path.basename(img))[0]
                os.link(img, os.path.join(out, "images", part, stem + ".jpg"))
                shutil.copy2(os.path.join(base, "labels", part, stem + ".txt"),
                             os.path.join(out, "labels", part, stem + ".txt"))
                n_base += 1
        for vroot, stem in picks:                          # harvest → TRAIN ONLY
            src = os.path.join(vroot, "frames", stem + ".jpg")
            lbl = os.path.join(vroot, lblsub, stem + ".txt")
            if not (os.path.exists(src) and os.path.exists(lbl)):
                continue
            os.link(src, os.path.join(out, "images", "train", stem + ".jpg"))
            shutil.copy2(lbl, os.path.join(out, "labels", "train", stem + ".txt"))
            n_hv += 1
        name = os.path.basename(out)
        for tag, root in (("data.yaml", os.path.abspath(out)), ("data_colab.yaml", f"/content/{name}")):
            with open(os.path.join(out, tag), "w") as f:
                f.write(f"path: {root}\ntrain: images/train\nval: images/val\n\n"
                        f"kpt_shape: [{nkp}, 3]\nflip_idx: {flip}\n\nnc: 1\nnames: ['court']\n")
        zp = f"{name}_colab.zip"
        if os.path.exists(zp):
            os.remove(zp)
        subprocess.run(["zip", "-qr", os.path.abspath(zp), "data_colab.yaml", "images", "labels"],
                       cwd=out, check=True)
        print(f"{out}: base {n_base} + harvest {n_hv} (train-only)  ->  {zp}")
    print("\nColab: upload the zips, set DATASET in train_pose_snapped_colab.ipynb, run cells 2-4.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    hp = sub.add_parser("harvest")
    hp.add_argument("sources", nargs="+", help="YouTube URLs or local video files")
    hp.add_argument("--max-frames", type=int, default=400, help="accepted-frame cap per game")
    sub.add_parser("build")
    args = ap.parse_args()
    if args.cmd == "harvest":
        harvest(args.sources, args.max_frames)
    else:
        build()


if __name__ == "__main__":
    main()

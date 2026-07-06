#!/usr/bin/env python
"""
unseen_benchmark.py — test the court detectors on genuinely UNSEEN broadcast footage
(YouTube game sections in data/unseen_videos/). No ground-truth labels exist, so the
scoring is unsupervised, against the image itself:

  • sane-H rate  — fraction of gated frames whose fitted homography passes the
    geometric sanity checks (no fold / collapse / explosion);
  • line residual — median distance from the projected court template to detected
    line pixels (the snapper's matched-offset metric, fixed search radius);
  • match count  — how many template samples found line evidence (low = the
    projection isn't near real lines, or the frame has no court).

Frames are sampled every SAMPLE_S seconds and pre-filtered by the HSV floor gate
(band 0.12–0.55 maple fraction). The gate was tuned on modern footage; its floor
fraction is logged per frame so era-specific rejections are visible.

Outputs:
  data/court_review/unseen_rank.tsv    — per-frame, per-model results (worst first
                                          by NEW-grid residual; verify-tool compatible)
  data/court_review/unseen_review/     — triptych renders old-aug | NEW-33 | NEW-grid
Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/anaconda3/bin/python unseen_benchmark.py
"""
from __future__ import annotations
import glob, os
import cv2, numpy as np
from ultralytics import YOLO
from court.court33 import COURT_VERTICES_33, court33_segments, court33_curves
from generate_labels import GRID_FT
from snap_projections import ridge_field, match_lines, project
from gate.hsv_baseline import HsvGate

cv2.setRNGSeed(42)
VID_DIR = "data/unseen_videos"
OUT_TSV = "data/court_review/unseen_rank.tsv"
OUT_DIR = "data/court_review/unseen_review"
SAMPLE_S = 2.0
EVAL_R = 10
KP_CONF = 0.5
GAMES = {"qAeUUwn-A8s": "2024 DEN@MIN", "92fLApYaCGI": "1998 CHI@UTA",
         "H56iR3188P0": "2013 SAS@MIA", "8Ap3jYl0nAA": "2019 TOR@PHI"}
MODELS = [("old-aug", "models/court_kp33_aug.pt", COURT_VERTICES_33),
          ("NEW-33", "models/court_kp33_snapped.pt", COURT_VERTICES_33),
          ("NEW-grid", "models/court_grid_snapped.pt", GRID_FT)]


def predict_kps(model, img):
    r = model.predict(img, verbose=False, device="mps")[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return None, None
    b = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[b].cpu().numpy()
    conf = (r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None
            else np.ones(len(xy)))
    return xy, conf


def fit_H(xy, conf, w, h, coords):
    keep = [(i, xy[i]) for i in range(len(xy))
            if conf[i] >= KP_CONF and 0 < xy[i][0] < w and 0 < xy[i][1] < h]
    if len(keep) < 4:
        return None, len(keep)
    src = np.array([p for _, p in keep], np.float32).reshape(-1, 1, 2)
    dst = np.array([coords[i] for i, _ in keep], np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, 0 if len(keep) == 4 else cv2.RANSAC, 3.0)
    return H, len(keep)


def h_sane(H_px, w, h):
    """Geometric sanity of H (px→ft): court vertices on-frame count + fold check."""
    try:
        P = np.linalg.inv(H_px.astype(np.float64))            # ft -> px
    except np.linalg.LinAlgError:
        return False
    p = project(P, COURT_VERTICES_33)
    ok = np.isfinite(p).all(axis=1)
    n_on = int(np.sum(ok & (p[:, 0] >= 0) & (p[:, 0] <= w) & (p[:, 1] >= 0) & (p[:, 1] <= h)))
    if n_on >= 30 or n_on <= 5:
        return False
    signs = set()
    for x, y in [(w * .01, h * .01), (w * .99, h * .01), (w * .99, h * .99), (w * .01, h * .99)]:
        d = H_px[2, 0] * x + H_px[2, 1] * y + H_px[2, 2]
        u = H_px[0, 0] * x + H_px[0, 1] * y + H_px[0, 2]
        v = H_px[1, 0] * x + H_px[1, 1] * y + H_px[1, 2]
        J = np.array([[H_px[0, 0] * d - u * H_px[2, 0], H_px[0, 1] * d - u * H_px[2, 1]],
                      [H_px[1, 0] * d - v * H_px[2, 0], H_px[1, 1] * d - v * H_px[2, 1]]]) / d ** 2
        signs.add(float(np.sign(np.linalg.det(J))))
    return len(signs) == 1


def line_res(P, ridge, w, h):
    ft, px = match_lines(P, ridge, w, h, EVAL_R)
    if len(ft) < 20:
        return None, len(ft)
    d = np.linalg.norm(px - project(P, ft), axis=1)
    return float(np.median(d)), len(ft)


def draw_court(img, P, color, thick=2):
    h, w = img.shape[:2]
    for P1, P2 in court33_segments():
        seg = project(P, np.stack([P1, P2]))
        if np.isfinite(seg).all():
            cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), color, thick, cv2.LINE_AA)
    for poly in court33_curves():
        px = project(P, poly)
        for a, b in zip(px[:-1], px[1:]):
            if (np.isfinite(a).all() and np.isfinite(b).all()
                    and max(abs(a[0]), abs(b[0])) < 4 * w and max(abs(a[1]), abs(b[1])) < 4 * h):
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUT_DIR, "*.jpg")):
        os.remove(old)
    gate = HsvGate()
    models = [(name, YOLO(path), coords) for name, path, coords in MODELS]

    rows = []
    for vp in sorted(glob.glob(os.path.join(VID_DIR, "*.mp4"))):
        base = os.path.splitext(os.path.basename(vp))[0]
        vid = base.split("_s")[0]
        game = GAMES.get(vid, vid)
        cap = cv2.VideoCapture(vp)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        n_gated = 0
        t = 0.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            t += SAMPLE_S
            if not ok:
                continue
            frac = gate.floor_fraction(img)
            if not gate.is_court_visible(img):
                continue
            n_gated += 1
            h, w = img.shape[:2]
            ridge = ridge_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            frame_id = f"{base}_t{int(t - SAMPLE_S):04d}"
            per = {}
            for name, model, coords in models:
                xy, conf = predict_kps(model, img)
                H, npts = fit_H(xy, conf, w, h, coords) if xy is not None else (None, 0)
                sane = H is not None and h_sane(H, w, h)
                res = nm = None
                if sane and ridge is not None:
                    try:
                        res, nm = line_res(np.linalg.inv(H.astype(np.float64)), ridge, w, h)
                    except np.linalg.LinAlgError:
                        sane = False
                per[name] = (H, sane, res, nm, npts)
            rows.append((frame_id, game, frac, img, per))
        cap.release()
        print(f"{base}: {n_gated} gated frames")

    # ── aggregate table ───────────────────────────────────────────────────────
    games = sorted(set(r[1] for r in rows))
    print(f"\n{'game':<14}{'frames':>7} | " + " | ".join(f"{n:^24}" for n, _, _ in MODELS))
    print(" " * 22 + "  " + " | ".join(f"{'saneH%':>7}{'res':>6}{'match':>6}   " for _ in MODELS))
    for g in games:
        sub = [r for r in rows if r[1] == g]
        line = f"{g:<14}{len(sub):>7} | "
        for name, _, _ in MODELS:
            sane = [r[4][name][1] for r in sub]
            res = [r[4][name][2] for r in sub if r[4][name][2] is not None]
            nm = [r[4][name][3] for r in sub if r[4][name][3] is not None]
            line += (f"{100*np.mean(sane):>6.0f}%{np.median(res) if res else float('nan'):>6.1f}"
                     f"{np.median(nm) if nm else 0:>6.0f}   | ")
        print(line)

    # ── triptych renders, worst-first by NEW-grid residual (None/insane first) ──
    def key(r):
        sane, res = r[4]["NEW-grid"][1], r[4]["NEW-grid"][2]
        return -(1000.0 if not sane else (999.0 if res is None else res))
    rows.sort(key=key)
    with open(OUT_TSV, "w") as f:
        f.write("rank\tfile\tframe\tgame\tfloor_frac\t" +
                "\t".join(f"{n}_sane\t{n}_res\t{n}_match\t{n}_pts" for n, _, _ in MODELS) + "\n")
        for rank, (frame_id, game, frac, img, per) in enumerate(rows):
            fn = f"{rank:04d}_{frame_id}.jpg"
            panels = []
            for name, _, _ in MODELS:
                H, sane, res, nm, npts = per[name]
                panel = img.copy()
                if H is not None:
                    try:
                        draw_court(panel, np.linalg.inv(H.astype(np.float64)),
                                   (255, 0, 255) if sane else (0, 0, 255), 2)
                    except np.linalg.LinAlgError:
                        pass
                hh, ww = panel.shape[:2]
                cv2.rectangle(panel, (0, 0), (ww, 24), (0, 0, 0), -1)
                tag = (f"{name}  {'SANE' if sane else 'INSANE/none'}  "
                       f"res={res:.1f}px m={nm}" if res is not None else
                       f"{name}  {'SANE' if sane else 'INSANE/none'}  pts={npts}")
                cv2.putText(panel, tag, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(cv2.resize(panel, (640, int(640 * hh / ww))))
            trip = np.hstack(panels)
            cv2.putText(trip, game, (trip.shape[1] // 2 - 60, trip.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUT_DIR, fn), trip)
            vals = "\t".join(f"{int(per[n][1])}\t{per[n][2] if per[n][2] is not None else 'nan'}"
                             f"\t{per[n][3] if per[n][3] is not None else 0}\t{per[n][4]}"
                             for n, _, _ in MODELS)
            f.write(f"{rank}\t{fn}\t{frame_id}\t{game}\t{frac:.3f}\t{vals}\n")
    print(f"\n{len(rows)} frames -> {OUT_TSV}\ntriptychs -> {OUT_DIR}/ (0000 = worst by NEW-grid)")


if __name__ == "__main__":
    main()

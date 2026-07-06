#!/usr/bin/env python
"""
evaluate_tracking.py — lightweight, honest evaluation of player DETECTION.

Reads hand-labeled boxes from data/tracking/box_truth/ (made with label_boxes.py),
runs the same YOLO detector the tracker uses, greedily matches predictions to
ground truth by IoU, and reports precision / recall / F1 at IoU thresholds.

Scope note: this measures DETECTION (are the right players boxed?). It does NOT
measure ID-switch rate / IDF1 — that needs box+ID ground truth across clips (the
heavier MOT labeling we opted out of). For ID stability, see the label-free
diagnostics printed by run_tracking.py (id_churn_ratio, camera_cut_resets).

Outputs: reports/tracking_metrics.json + .txt, and reports/viz/detection_eval.png
(green = matched GT (TP), yellow = missed GT (FN), red = false-positive prediction).

Examples
────────
  python evaluate_tracking.py
  python evaluate_tracking.py --iou 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

import config
from detect.detector import PlayerDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("evaluate_tracking")


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match(preds, gts, iou_thr):
    """Greedy match preds (conf-sorted) to gts. Returns (tp, fp, fn, matched_gt_idx,
    fp_pred_idx). preds: (N,5), gts: (M,4)."""
    matched_gt = set()
    tp_pred, fp_pred = [], []
    for pi, p in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            v = iou(p[:4], g)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gi)
            tp_pred.append(pi)
        else:
            fp_pred.append(pi)
    tp = len(matched_gt)
    fp = len(fp_pred)
    fn = len(gts) - tp
    return tp, fp, fn, matched_gt, fp_pred


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def make_eval_sheet(samples, out_path, cols=3, max_tiles=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = samples[:max_tiles]
    if not samples:
        return
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, s in zip(axes, samples):
        img = cv2.imread(s["image_path"])
        if img is None:
            continue
        gts, preds = s["gts"], s["preds"]
        matched, fp_idx = s["matched_gt"], set(s["fp_pred"])
        for gi, g in enumerate(gts):
            col = (60, 220, 60) if gi in matched else (60, 220, 255)  # TP green / FN yellow
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), col, 3)
        for pi, p in enumerate(preds):
            if pi in fp_idx:  # false-positive predictions in red
                cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 230), 2)
        ax.imshow(img[:, :, ::-1])
        ax.set_title(f"{s['frame']}\nTP={s['tp']} FN={s['fn']} FP={s['fp']}", fontsize=8)
    fig.suptitle("Detection eval — green=TP, yellow=missed(FN), red=false-positive", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detection precision/recall vs hand-labeled boxes.")
    ap.add_argument("--iou", type=float, default=0.5, help="Primary IoU match threshold (default 0.5).")
    ap.add_argument("--conf", type=float, default=None, help="Override detector confidence (default config.PLAYER_CONF).")
    ap.add_argument("--nms-iou", type=float, default=None, help="Override NMS IoU (higher keeps overlapping players).")
    ap.add_argument("--weights", type=str, default=None, help="Override detector weights (e.g. yolov8x.pt, yolo11x.pt).")
    ap.add_argument("--court-mask", action="store_true",
                    help="Drop detections whose foot-point projects off-court (needs trained court-kp model).")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    gt_files = sorted(config.TRACK_BOX_TRUTH.glob("*.json"))
    if not gt_files:
        raise SystemExit(
            f"No labeled boxes in {config.TRACK_BOX_TRUTH}. Run label_boxes.py first "
            "(this eval needs hand-drawn ground truth — it will not invent labels)."
        )

    log.info("Loading detector …")
    det = PlayerDetector(weights=args.weights, conf=args.conf, iou=args.nms_iou)

    mapper = None
    if args.court_mask:
        from court import CourtMapper
        log.info("Loading court-keypoint model for court-masking …")
        mapper = CourtMapper()

    n_masked_removed = n_no_homography = 0

    log.info("Evaluating detection on %d labeled frames …", len(gt_files))
    iou_levels = sorted({round(x, 2) for x in [0.3, args.iou, 0.7]})
    totals = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_levels}
    n_gt = n_pred = 0
    sheet_samples = []

    for gf in gt_files:
        rec = json.loads(gf.read_text())
        gts = np.array(rec["boxes"], dtype=np.float32) if rec["boxes"] else np.zeros((0, 4), np.float32)
        img = cv2.imread(rec["image_path"])
        if img is None:
            log.warning("missing image %s — skipping", rec["image_path"])
            continue
        preds = det.detect(img)
        if mapper is not None:
            mapper.update(img)
            if mapper.has_homography:
                keep = [p for p in preds
                        if mapper.is_inside_court([(p[0] + p[2]) / 2, p[3]])]
                n_masked_removed += len(preds) - len(keep)
                preds = np.array(keep, np.float32) if keep else np.zeros((0, 5), np.float32)
            else:
                n_no_homography += 1   # no court frame of reference → can't mask, keep all
        n_gt += len(gts)
        n_pred += len(preds)
        for t in iou_levels:
            tp, fp, fn, matched, fp_pred = match(preds, gts, t)
            totals[t]["tp"] += tp
            totals[t]["fp"] += fp
            totals[t]["fn"] += fn
            if t == args.iou:
                sheet_samples.append({
                    "frame": rec["frame"], "image_path": rec["image_path"],
                    "gts": gts.tolist(), "preds": preds[:, :4].tolist(),
                    "matched_gt": matched, "fp_pred": fp_pred,
                    "tp": tp, "fp": fp, "fn": fn,
                })

    results = {}
    for t in iou_levels:
        c = totals[t]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        results[f"iou_{t}"] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }

    out = {
        "frames_evaluated": len(gt_files),
        "total_gt_boxes": n_gt,
        "total_pred_boxes": n_pred,
        "avg_gt_players_per_frame": round(n_gt / len(gt_files), 2),
        "avg_pred_players_per_frame": round(n_pred / len(gt_files), 2),
        "detector": {"weights": Path(det.weights).name, "conf": det.conf, "iou_nms": det.iou},
        "court_mask": {"enabled": bool(mapper), "detections_removed": n_masked_removed,
                       "frames_without_homography": n_no_homography} if mapper else {"enabled": False},
        "metrics_by_iou": results,
        "note": "Detection only. ID-switch/IDF1 not measured (needs MOT labels). "
                "See run_tracking.py diagnostics for ID-stability proxies.",
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "tracking_metrics.json").write_text(json.dumps(out, indent=2))

    # human-readable
    lines = ["PLAYER DETECTION — EVAL (vs hand-labeled boxes)", "=" * 50,
             f"frames={out['frames_evaluated']}  GT boxes={n_gt}  pred boxes={n_pred}",
             f"avg players/frame: GT={out['avg_gt_players_per_frame']}  pred={out['avg_pred_players_per_frame']}",
             f"detector: {out['detector']['weights']} conf={out['detector']['conf']} iou_nms={out['detector']['iou_nms']}",
             (f"court-mask: ON — removed {n_masked_removed} off-court detections "
              f"({n_no_homography} frames had no homography)" if mapper else "court-mask: OFF"),
             "",
             f"  {'IoU':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"]
    for t in iou_levels:
        m = results[f"iou_{t}"]
        lines.append(f"  {t:>5} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
                     f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
    lines += ["", "FN here = a real player the detector missed (under-tracking).",
              "FP here = a detection with no real player (phantom/duplicate).",
              "ID-switch rate is NOT in this table (needs MOT labels); see run_tracking diagnostics."]
    (config.REPORTS_DIR / "tracking_metrics.txt").write_text("\n".join(lines))

    make_eval_sheet(sheet_samples, config.VIZ_DIR / "detection_eval.png")

    print("\n".join(lines))
    log.info("Wrote reports/tracking_metrics.{json,txt} and reports/viz/detection_eval.png")


if __name__ == "__main__":
    main()

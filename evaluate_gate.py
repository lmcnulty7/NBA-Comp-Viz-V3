#!/usr/bin/env python
"""
evaluate_gate.py — Phase 4 (metrics) + Phase 5 (visual verification).

Kept separate from training so reporting can be re-run without retraining.

LABELS COME EXCLUSIVELY FROM truth/ (via the saved split). predicted/ is never
read here — asserted at load time. If truth/ is empty, this errors out.

On the held-out TEST split, for the trained head, the CLIP zero-shot baseline,
and the recovered HSV baseline, it computes:
  • accuracy, precision, recall, F1 — per class AND overall
  • the 2×2 confusion matrix
  • the two error types labeled by DOWNSTREAM COST:
      FALSE NEGATIVE = live frame called dead → frame skipped → empty clips (the
                       current production bug)
      FALSE POSITIVE = dead frame called live → replay/ad processed → garbage events
  • a threshold sweep (table + PR-curve PNG) showing the FN/FP tradeoff, with a
    recommended low-FN threshold (tuned on VAL, not TEST)
  • a side-by-side of all approaches on the same TEST set

Outputs: reports/metrics.json, reports/metrics.txt, and contact sheets in
reports/viz/ (correct_samples.png, false_negatives.png, false_positives.png).

Examples
────────
  python evaluate_gate.py
  python evaluate_gate.py --no-viz
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import config
from gate.backbones import get_backbone
from gate.common import (
    confusion_counts,
    evaluate_scores,
    get_image_embeddings,
    metrics_from_counts,
)
from gate.hsv_baseline import HsvGate
from gate.trained_head import TrainedHeadGate
from gate.zero_shot import ZeroShotGate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("evaluate_gate")


# ── Test-split loading (truth/ only, asserted) ────────────────────────────────
def load_test_split():
    if not config.SPLIT_PATH.exists():
        raise SystemExit(f"No split at {config.SPLIT_PATH}. Run train_gate.py first.")
    rec = json.loads(config.SPLIT_PATH.read_text())
    paths = [Path(p) for p in rec["files"]["test"]]
    labels = np.array(rec["labels"]["test"], dtype=int)
    if len(paths) == 0:
        raise SystemExit("TEST split is empty — verify more frames into truth/ and re-run train_gate.py.")
    # Integrity guard: every test frame must live under truth/, never predicted/.
    truth_root = config.TRUTH_DIR.resolve()
    pred_root = config.PREDICTED_DIR.resolve()
    for p in paths:
        rp = p.resolve()
        if pred_root in rp.parents:
            raise AssertionError(f"INTEGRITY VIOLATION: TEST frame from predicted/ ({rp}).")
        if truth_root not in rp.parents:
            raise AssertionError(f"TEST frame not under truth/ ({rp}).")
    return paths, labels


def threshold_sweep(scores, y, steps=21):
    rows = []
    for t in np.linspace(0.0, 1.0, steps):
        c = confusion_counts(y, (scores >= t).astype(int))
        m = metrics_from_counts(c)
        rows.append({
            "threshold": round(float(t), 3),
            "FN_live_skipped": c["fn"],
            "FP_deadball_processed": c["fp"],
            "live_recall": round(m["live"]["recall"], 3),
            "dead_recall": round(m["dead"]["recall"], 3),
            "accuracy": round(m["accuracy"], 3),
            "macro_f1": round(m["macro_f1"], 3),
        })
    return rows


def pr_curve_png(curves: dict, out_path: Path, recommended: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(6, 5))
    for name, (scores, y) in curves.items():
        prec, rec, _ = precision_recall_curve(y, scores)
        plt.plot(rec, prec, label=name, linewidth=2)
        if name in recommended:
            m = recommended[name]
            plt.scatter([m["live"]["recall"]], [m["live"]["precision"]], s=60, zorder=5,
                        label=f"{name} @thr={m['threshold']:.2f}")
    plt.xlabel("Recall (live)")
    plt.ylabel("Precision (live)")
    plt.title("PR curve — live class (TEST)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


# ── Phase 5: contact sheets ───────────────────────────────────────────────────
def contact_sheet(items, title, out_path: Path, max_tiles=24, cols=6):
    """items: list of (path, true_int, pred_int, p_live). Green=correct, red=wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    items = items[:max_tiles]
    if not items:
        log.info("  (no tiles for '%s')", title)
        return
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (path, yt, yp, p) in zip(axes, items):
        img = cv2.imread(str(path))
        if img is not None:
            ax.imshow(img[:, :, ::-1])
        correct = yt == yp
        color = "#1a9850" if correct else "#d73027"
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color); sp.set_linewidth(4)
        ax.set_title(
            f"T:{config.CLASSES[yt]}  P:{config.CLASSES[yp]}\nP(live)={p:.2f}",
            fontsize=8, color=color,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("  wrote %s (%d tiles)", out_path, len(items))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the gate on the held-out TEST split (truth/ only).")
    ap.add_argument("--no-viz", action="store_true", help="Skip contact-sheet generation.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    paths, y = load_test_split()
    log.info("TEST set: %d frames (live=%d, dead=%d) — all human-verified from truth/.",
             len(paths), int((y == 1).sum()), int((y == 0).sum()))

    thresholds = json.loads(config.THRESHOLDS_PATH.read_text())
    cfg_rec = json.loads(config.CONFIG_RECORD_PATH.read_text()) if config.CONFIG_RECORD_PATH.exists() else {}
    head_backbone = cfg_rec.get("backbone", "clip")

    device = config.get_device()

    # ── Trained head (Approach B) on its own backbone embeddings ─────────────
    log.info("Loading %s backbone (head) + trained head …", head_backbone)
    bb_head = get_backbone(head_backbone, device)
    emb_head = get_image_embeddings(paths, bb_head, config.emb_cache_path(head_backbone))
    trained = TrainedHeadGate.load(config.HEAD_PATH, backbone=None, threshold=thresholds["trained"])
    pB = trained.score_embeddings(emb_head)

    # ── Zero-shot baseline (Approach A) — always CLIP (needs text) ───────────
    if head_backbone == "clip":
        bb_clip, emb_clip = bb_head, emb_head
    else:
        bb_clip = get_backbone("clip", device)
        emb_clip = get_image_embeddings(paths, bb_clip, config.emb_cache_path("clip"))
    zgate = ZeroShotGate(bb_clip, config.LIVE_PROMPTS, config.DEAD_PROMPTS)
    thrA = thresholds.get("zero_shot", 0.5)
    pA = zgate.score_embeddings(emb_clip)

    # ── HSV baseline (recovered shipped gate) at fixed band ───────────────────
    hsv = HsvGate(config.HSV_LO, config.HSV_HI, config.HSV_MIN_FLOOR_FRACTION, config.HSV_MAX_FLOOR_FRACTION)
    hsv_frac = np.array([hsv.floor_fraction(p) for p in paths])
    hsv_pred = hsv.predict_from_fraction(hsv_frac)

    # ── Metrics at operating points ───────────────────────────────────────────
    res = {
        "trained_head": evaluate_scores(pB, y, thresholds["trained"]),
        "zero_shot_clip": evaluate_scores(pA, y, thrA),
        "hsv_baseline": metrics_from_counts(confusion_counts(y, hsv_pred)),
    }
    res["hsv_baseline"]["threshold"] = f"band[{config.HSV_MIN_FLOOR_FRACTION},{config.HSV_MAX_FLOOR_FRACTION}]"

    # ── Threshold sweeps + PR curve ───────────────────────────────────────────
    sweeps = {"trained_head": threshold_sweep(pB, y), "zero_shot_clip": threshold_sweep(pA, y)}
    pr_curve_png(
        {"trained_head": (pB, y), "zero_shot_clip": (pA, y)},
        config.VIZ_DIR / "pr_curve.png",
        recommended={"trained_head": res["trained_head"], "zero_shot_clip": res["zero_shot_clip"]},
    )

    # ── Persist metrics.json + metrics.txt ────────────────────────────────────
    out = {
        "test_size": len(paths),
        "test_live": int((y == 1).sum()),
        "test_dead": int((y == 0).sum()),
        "operating_points": res,
        "thresholds": thresholds,
        "threshold_sweeps": sweeps,
        "error_glossary": {
            "false_negative": "LIVE frame called DEAD → frame skipped → empty/under-counted clips (current production bug)",
            "false_positive": "DEAD frame called LIVE → replay/closeup/ad processed → garbage events in box score",
        },
        "config_record": cfg_rec,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_JSON.write_text(json.dumps(out, indent=2))
    write_metrics_txt(out, config.METRICS_TXT)
    log.info("Wrote %s and %s", config.METRICS_JSON, config.METRICS_TXT)

    # ── Console side-by-side ──────────────────────────────────────────────────
    print_side_by_side(res)

    # ── Phase 5: visual verification (trained head) ──────────────────────────
    if not args.no_viz:
        log.info("Generating contact sheets in %s …", config.VIZ_DIR)
        yp = (pB >= thresholds["trained"]).astype(int)
        correct = [(paths[i], int(y[i]), int(yp[i]), float(pB[i])) for i in range(len(paths)) if y[i] == yp[i]]
        fns = [(paths[i], int(y[i]), int(yp[i]), float(pB[i])) for i in range(len(paths)) if y[i] == 1 and yp[i] == 0]
        fps = [(paths[i], int(y[i]), int(yp[i]), float(pB[i])) for i in range(len(paths)) if y[i] == 0 and yp[i] == 1]
        contact_sheet(correct, "Correctly classified (TEST) — trained head", config.VIZ_DIR / "correct_samples.png")
        contact_sheet(fns, "FALSE NEGATIVES — live frames SKIPPED (the empty-clip bug)", config.VIZ_DIR / "false_negatives.png")
        contact_sheet(fps, "FALSE POSITIVES — dead balls let through", config.VIZ_DIR / "false_positives.png")

    log.info("Done.")


def _fmt_block(name, m) -> str:
    c = m["confusion"]
    return (
        f"  {name}  (threshold={m['threshold']})\n"
        f"    accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}\n"
        f"    live: P={m['live']['precision']:.3f} R={m['live']['recall']:.3f} F1={m['live']['f1']:.3f} (n={m['live']['support']})\n"
        f"    dead: P={m['dead']['precision']:.3f} R={m['dead']['recall']:.3f} F1={m['dead']['f1']:.3f} (n={m['dead']['support']})\n"
        f"    confusion: TP(live→live)={c['tp']}  TN(dead→dead)={c['tn']}  "
        f"FP(dead→live)={c['fp']}  FN(live→dead)={c['fn']}\n"
        f"    >> FALSE NEGATIVES (live skipped, empty-clip bug) = {m['false_negatives_live_skipped']}\n"
        f"    >> FALSE POSITIVES (dead processed, garbage events) = {m['false_positives_deadball_processed']}\n"
    )


def write_metrics_txt(out: dict, path: Path) -> None:
    lines = []
    lines.append("COURT-VISIBILITY GATE — TEST METRICS")
    lines.append("=" * 60)
    lines.append(f"TEST frames: {out['test_size']} (live={out['test_live']}, dead={out['test_dead']}) — all human-verified.\n")
    lines.append("ERROR TYPES BY DOWNSTREAM COST")
    lines.append(f"  FALSE NEGATIVE = {out['error_glossary']['false_negative']}")
    lines.append(f"  FALSE POSITIVE = {out['error_glossary']['false_positive']}\n")
    lines.append("OPERATING POINTS")
    for name, m in out["operating_points"].items():
        lines.append(_fmt_block(name, m))
    lines.append("THRESHOLD SWEEP — trained_head (FN vs FP tradeoff)")
    lines.append(f"  {'thr':>5} {'FN':>4} {'FP':>4} {'live_R':>7} {'dead_R':>7} {'acc':>6} {'mF1':>6}")
    for r in out["threshold_sweeps"]["trained_head"]:
        lines.append(f"  {r['threshold']:>5.2f} {r['FN_live_skipped']:>4} {r['FP_deadball_processed']:>4} "
                     f"{r['live_recall']:>7.3f} {r['dead_recall']:>7.3f} {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f}")
    tuned = out["operating_points"]["trained_head"]["threshold"]
    lines.append(f"\n  RECOMMENDED threshold (tuned on VAL, minimizes FN s.t. FP cap): {tuned}")
    lines.append("  (Lower threshold → fewer FN/empty clips but more FP. See PR curve: reports/viz/pr_curve.png)")
    path.write_text("\n".join(lines))


def print_side_by_side(res: dict) -> None:
    print("\n=== SIDE-BY-SIDE on TEST (same frames) ===")
    print(f"{'approach':<16}{'acc':>7}{'liveR':>8}{'deadR':>8}{'mF1':>7}{'FN':>5}{'FP':>5}")
    for name, m in res.items():
        print(f"{name:<16}{m['accuracy']:>7.3f}{m['live']['recall']:>8.3f}{m['dead']['recall']:>8.3f}"
              f"{m['macro_f1']:>7.3f}{m['false_negatives_live_skipped']:>5}{m['false_positives_deadball_processed']:>5}")
    print("FN = live frames skipped (empty-clip bug). FP = dead balls processed (garbage events).\n")


if __name__ == "__main__":
    main()

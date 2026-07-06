"""
gate/common.py — shared utilities: image IO, embedding cache, metrics,
and threshold selection. Backbone-agnostic; imports nothing heavy at module
load (torch is only touched through a passed-in backbone object).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Image IO ──────────────────────────────────────────────────────────────────
def list_images(directory: Path) -> list[Path]:
    """All image files directly inside `directory`, sorted by name."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_bgr(path: Path) -> np.ndarray:
    """Load an image as a BGR uint8 array (OpenCV native), matching the old gate."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


# ── Embedding cache (shared by train_gate and evaluate_gate) ──────────────────
def get_image_embeddings(
    paths: Sequence[Path], backbone, cache_path: Path
) -> np.ndarray:
    """
    Return an (N, D) array of L2-normalized embeddings for `paths`, in order.

    Embeddings are cached on disk keyed by absolute path so the (expensive)
    encoder pass runs once and is reused across training and evaluation.
    """
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    keys = [str(Path(p).resolve()) for p in paths]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        missing_paths = [paths[i] for i in missing_idx]
        embs = backbone.embed_image_paths(missing_paths)
        for i, e in zip(missing_idx, embs):
            cache[keys[i]] = e.astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return np.stack([cache[k] for k in keys]).astype(np.float32)


# ── Metrics (positive class = live = 1) ───────────────────────────────────────
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Return TP/TN/FP/FN with the downstream-cost interpretation baked in.

    FN = a LIVE frame classified DEAD  → frame skipped → empty/under-counted clips
         (this is the current production bug: clips returning all-empty output).
    FP = a DEAD frame classified LIVE  → replay/closeup/ad processed → garbage events.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # dead → live
    fn = int(((y_pred == 0) & (y_true == 1)).sum())  # live → dead
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics_from_counts(c: dict) -> dict:
    """Per-class + overall metrics from a confusion-count dict."""
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn

    # live = positive
    p_live = _safe_div(tp, tp + fp)
    r_live = _safe_div(tp, tp + fn)
    f_live = _safe_div(2 * p_live * r_live, p_live + r_live)
    # dead = the other class as positive
    p_dead = _safe_div(tn, tn + fn)
    r_dead = _safe_div(tn, tn + fp)
    f_dead = _safe_div(2 * p_dead * r_dead, p_dead + r_dead)

    return {
        "n": n,
        "accuracy": _safe_div(tp + tn, n),
        "live": {"precision": p_live, "recall": r_live, "f1": f_live, "support": tp + fn},
        "dead": {"precision": p_dead, "recall": r_dead, "f1": f_dead, "support": tn + fp},
        "macro_f1": (f_live + f_dead) / 2,
        "confusion": dict(c),
        # explicit downstream-cost labels
        "false_negatives_live_skipped": fn,
        "false_positives_deadball_processed": fp,
    }


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    """Threshold P(live) scores and compute the full metric block."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    m = metrics_from_counts(c)
    m["threshold"] = float(threshold)
    return m


# ── Threshold selection (tuned on VAL) ────────────────────────────────────────
def _candidate_thresholds(scores: np.ndarray) -> list[float]:
    uniq = sorted(set(np.asarray(scores).tolist()))
    # midpoints + endpoints so every distinct partition is reachable
    cands = [0.0]
    for a, b in zip(uniq, uniq[1:]):
        cands.append((a + b) / 2)
    cands.append(1.0)
    return cands


def choose_threshold_max_f1(scores: np.ndarray, y_true: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in _candidate_thresholds(scores):
        m = evaluate_scores(scores, y_true, t)
        if m["live"]["f1"] > best_f1:
            best_f1, best_t = m["live"]["f1"], t
    return float(best_t)


def choose_threshold_min_fn(
    scores: np.ndarray, y_true: np.ndarray, max_fp_rate: float = 0.10
) -> float:
    """
    Pick the threshold that MINIMIZES false negatives (live frames skipped —
    the current empty-clip bug) subject to a cap on the dead-class false-positive
    rate. Ties broken by fewer FPs. Falls back to max-F1 if the cap is infeasible.
    """
    y_true = np.asarray(y_true).astype(int)
    n_dead = int((y_true == 0).sum())
    best_key, best_t = None, None
    for t in _candidate_thresholds(scores):
        c = confusion_counts(y_true, (scores >= t).astype(int))
        fp_rate = _safe_div(c["fp"], n_dead)
        if fp_rate <= max_fp_rate:
            key = (c["fn"], c["fp"])  # minimize FN, then FP
            if best_key is None or key < best_key:
                best_key, best_t = key, t
    if best_t is None:
        return choose_threshold_max_f1(scores, y_true)
    return float(best_t)

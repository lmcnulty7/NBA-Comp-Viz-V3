"""
gate/trained_head.py — Approach B: a lightweight head trained on frozen
backbone embeddings (default CLIP ViT-B/32). The encoder stays frozen; only a
logistic-regression (default) or 1-hidden-layer MLP head is trained on the
human-verified train split.

Interface (shared with the zero-shot gate):
    score(frame_bgr) -> P(live) in [0, 1]
    is_court_visible(frame_bgr, threshold=None) -> bool
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import load_bgr


def build_head(kind: str = "logreg", seed: int = 42):
    """Create an untrained sklearn head. kind ∈ {'logreg','mlp'}."""
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
        )
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            random_state=seed,
        )
    raise ValueError(f"Unknown head kind: {kind!r} (expected 'logreg' or 'mlp')")


class TrainedHeadGate:
    def __init__(self, clf, backbone=None, threshold: float = 0.5, meta: dict | None = None):
        self.clf = clf
        self.backbone = backbone
        self.threshold = threshold
        self.meta = meta or {}

    # ── Vectorized scoring over precomputed embeddings ────────────────────────
    def score_embeddings(self, emb: np.ndarray) -> np.ndarray:
        emb = np.atleast_2d(emb)
        # P(live) = probability of the positive (live=1) class
        classes = list(self.clf.classes_)
        live_col = classes.index(1)
        return self.clf.predict_proba(emb)[:, live_col]

    # ── Single-frame interface ────────────────────────────────────────────────
    def _embed_frame(self, frame_bgr) -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("No backbone attached; cannot embed raw frames.")
        from PIL import Image

        bb = self.backbone
        rgb = frame_bgr[:, :, ::-1]
        with bb.torch.no_grad():
            inputs = bb.processor(images=[Image.fromarray(rgb)], return_tensors="pt").to(bb.device)
            if hasattr(bb.model, "get_image_features"):
                from gate.backbones import as_tensor
                feats = as_tensor(bb.model.get_image_features(**inputs))
            else:  # DINOv2
                from gate.backbones import as_tensor
                feats = as_tensor(bb.model(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def score(self, frame_bgr) -> float:
        if not isinstance(frame_bgr, np.ndarray):
            frame_bgr = load_bgr(frame_bgr)  # accepts str / Path
        return float(self.score_embeddings(self._embed_frame(frame_bgr))[0])

    def is_court_visible(self, frame_bgr, threshold: float | None = None) -> bool:
        t = self.threshold if threshold is None else threshold
        return self.score(frame_bgr) >= t

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "threshold": self.threshold, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path, backbone=None, threshold: float | None = None) -> "TrainedHeadGate":
        import joblib

        blob = joblib.load(path)
        return cls(
            clf=blob["clf"],
            backbone=backbone,
            threshold=blob["threshold"] if threshold is None else threshold,
            meta=blob.get("meta", {}),
        )

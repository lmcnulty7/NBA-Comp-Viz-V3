"""
gate/zero_shot.py — Approach A: CLIP zero-shot court-visibility gate.

No training. Embeds the frame with CLIP and compares it to two sets of text
prompts ("live game footage" vs. "dead ball / replay / closeup / ad / crowd").
P(live) is a temperature-scaled softmax over the mean cosine similarity to each
prompt set — the standard CLIP zero-shot formulation.

Phase 0 note: the OLD project did NOT use CLIP for the gate (it used HSV floor-
color thresholding), so no prior prompts existed to recover. The prompt sets are
sensible defaults defined in config.py.

Interface (shared with the trained head):
    score(frame_bgr) -> P(live) in [0, 1]
    is_court_visible(frame_bgr, threshold=None) -> bool
"""
from __future__ import annotations

import numpy as np

from .common import load_bgr


class ZeroShotGate:
    def __init__(
        self,
        backbone,
        live_prompts: list[str],
        dead_prompts: list[str],
        temperature: float = 0.01,
        threshold: float = 0.5,
    ):
        if not getattr(backbone, "has_text", False):
            raise ValueError("Zero-shot gate requires a text-capable backbone (CLIP).")
        self.backbone = backbone
        self.temperature = temperature
        self.threshold = threshold
        self.live_protos = backbone.embed_texts(live_prompts)  # (Nl, D) normalized
        self.dead_protos = backbone.embed_texts(dead_prompts)  # (Nd, D) normalized

    # ── Vectorized scoring over precomputed image embeddings ──────────────────
    def score_embeddings(self, img_emb: np.ndarray) -> np.ndarray:
        """img_emb: (N, D) L2-normalized → P(live): (N,)."""
        img_emb = np.atleast_2d(img_emb)
        sim_live = (img_emb @ self.live_protos.T).mean(axis=1)  # mean cos sim
        sim_dead = (img_emb @ self.dead_protos.T).mean(axis=1)
        z = np.stack([sim_dead, sim_live], axis=1) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        return p[:, 1]  # P(live)

    # ── Single-frame interface ────────────────────────────────────────────────
    def score(self, frame_bgr) -> float:
        if not isinstance(frame_bgr, np.ndarray):
            frame_bgr = load_bgr(frame_bgr)  # accepts str / Path
        # Embed a single in-memory frame via PIL/RGB
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        img_emb = self._embed_pil(Image.fromarray(rgb))
        return float(self.score_embeddings(img_emb)[0])

    def _embed_pil(self, pil_img) -> np.ndarray:
        bb = self.backbone
        with bb.torch.no_grad():
            inputs = bb.processor(images=[pil_img], return_tensors="pt").to(bb.device)
            feats = bb.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def is_court_visible(self, frame_bgr, threshold: float | None = None) -> bool:
        t = self.threshold if threshold is None else threshold
        return self.score(frame_bgr) >= t

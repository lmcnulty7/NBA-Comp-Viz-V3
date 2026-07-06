"""
gate/backbones.py — frozen vision encoders that turn frames into embeddings.

Default: CLIP ViT-B/32 via transformers (open_clip is not installed; this is the
same model). CLIP also exposes a text encoder, which Approach A (zero-shot) needs.

Optional: DINOv2 (image-only) behind a flag, as a stronger embedding alternative
for the trained head (Approach B). DINOv2 has no text encoder, so it cannot be
used for the zero-shot baseline.

All embeddings are L2-normalized so a dot product is cosine similarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class ClipBackbone:
    """CLIP ViT-B/32 image + text encoder (transformers)."""

    name = "clip"
    embed_dim = 512

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.has_text = True

    def embed_image_paths(
        self, paths: Sequence[Path], batch_size: int = 32, progress: bool = True
    ) -> np.ndarray:
        from PIL import Image

        try:
            from tqdm import tqdm
        except Exception:  # pragma: no cover
            def tqdm(x, **k):
                return x

        out = []
        rng = range(0, len(paths), batch_size)
        bar = tqdm(rng, desc="CLIP image embeds", disable=not progress)
        with self.torch.no_grad():
            for i in bar:
                batch = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32) if out else np.zeros((0, self.embed_dim), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


class DinoBackbone:
    """DINOv2 image encoder (transformers). Image-only — no text/zero-shot."""

    name = "dinov2"
    embed_dim = 768

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.has_text = False

    def embed_image_paths(
        self, paths: Sequence[Path], batch_size: int = 32, progress: bool = True
    ) -> np.ndarray:
        from PIL import Image

        try:
            from tqdm import tqdm
        except Exception:  # pragma: no cover
            def tqdm(x, **k):
                return x

        out = []
        rng = range(0, len(paths), batch_size)
        bar = tqdm(rng, desc="DINOv2 image embeds", disable=not progress)
        with self.torch.no_grad():
            for i in bar:
                batch = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out_hs = self.model(**inputs).last_hidden_state[:, 0]  # CLS token
                feats = out_hs / out_hs.norm(dim=-1, keepdim=True)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32) if out else np.zeros((0, self.embed_dim), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("DINOv2 has no text encoder; zero-shot needs CLIP.")


def get_backbone(name: str, device: str):
    """Factory. name ∈ {'clip','dinov2'}."""
    import config

    if name == "clip":
        return ClipBackbone(config.CLIP_MODEL_NAME, device)
    if name == "dinov2":
        return DinoBackbone(config.DINOV2_MODEL_NAME, device)
    raise ValueError(f"Unknown backbone: {name!r} (expected 'clip' or 'dinov2')")

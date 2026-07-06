"""
court/line_seg.py — learned court-line segmentation (U-Net) + inference wrapper.

Replaces the noisy top-hat line detector in refine.py with a model that predicts
court-line pixels directly. Trained on the auto-generated masks from
make_line_dataset.py (court template projected through each frame's homography).

CourtLineSegmenter.mask(frame) -> uint8 0/255 mask at the frame's resolution,
a drop-in for refine.court_line_mask.
"""
from __future__ import annotations

import numpy as np

import config

# Segmentation input size (W, H), both divisible by 8 for the 3-level U-Net.
SEG_WH = (512, 288)


def _build_unet():
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, ch=(16, 32, 64, 128)):
            super().__init__()
            self.d1, self.d2, self.d3 = DoubleConv(3, ch[0]), DoubleConv(ch[0], ch[1]), DoubleConv(ch[1], ch[2])
            self.bott = DoubleConv(ch[2], ch[3])
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); self.u3 = DoubleConv(ch[3], ch[2])
            self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); self.u2 = DoubleConv(ch[2], ch[1])
            self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.u1 = DoubleConv(ch[1], ch[0])
            self.out = nn.Conv2d(ch[0], 1, 1)

        def forward(self, x):
            import torch
            c1 = self.d1(x); c2 = self.d2(self.pool(c1)); c3 = self.d3(self.pool(c2))
            b = self.bott(self.pool(c3))
            x = self.u3(torch.cat([self.up3(b), c3], 1))
            x = self.u2(torch.cat([self.up2(x), c2], 1))
            x = self.u1(torch.cat([self.up1(x), c1], 1))
            return self.out(x)

    return UNet()


class CourtLineSegmenter:
    def __init__(self, weights=None, device=None, thresh=0.5):
        import torch

        self.torch = torch
        self.device = device or config.get_device()
        self.thresh = thresh
        self.model = _build_unet().to(self.device).eval()
        w = weights or config.LINE_SEG_WEIGHTS
        self.model.load_state_dict(torch.load(str(w), map_location=self.device))

    def mask(self, frame) -> np.ndarray:
        """uint8 0/255 court-line mask at the frame's native resolution."""
        import cv2

        h, w = frame.shape[:2]
        x = cv2.resize(frame, SEG_WH)[:, :, ::-1].astype(np.float32) / 255.0
        t = self.torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.model(t))[0, 0].cpu().numpy()
        m = (p >= self.thresh).astype(np.uint8) * 255
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

#!/usr/bin/env python
"""
train_line_seg.py — train the court-line segmentation U-Net.

Trains on the auto-generated (image, line-mask) pairs from make_line_dataset.py.
Court lines are a tiny fraction of pixels, so we use BCE(pos_weight) + Dice and
report val Dice/IoU. Saves the best-val weights to models/court_line_seg.pt.

Usage
─────
  python make_line_dataset.py     # build the dataset first
  python train_line_seg.py
  python train_line_seg.py --epochs 120 --device cpu
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

import config
from court.line_seg import SEG_WH, _build_unet


class LineDS:
    def __init__(self, split, aug=False):
        self.imgs = sorted(glob.glob(str(config.LINE_DATASET / f"images/{split}/*.png")))
        self.aug = aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        mp = ip.replace("/images/", "/masks/")
        img = cv2.resize(cv2.imread(ip), SEG_WH)
        m = cv2.resize(cv2.imread(mp, 0), SEG_WH, interpolation=cv2.INTER_NEAREST)
        if self.aug:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]                       # horizontal flip
            if np.random.rand() < 0.5:
                f = 0.6 + 0.8 * np.random.rand()
                img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)  # brightness
        x = np.ascontiguousarray(img[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)
        y = np.ascontiguousarray((m > 127).astype(np.float32))[None]
        return x, y


def batches(ds, bs, shuffle):
    import torch
    idx = np.random.permutation(len(ds)) if shuffle else np.arange(len(ds))
    for k in range(0, len(idx), bs):
        xs, ys = zip(*[ds[i] for i in idx[k:k + bs]])
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def dice_iou(prob, y, thr=0.5):
    p = (prob >= thr).float()
    inter = (p * y).sum()
    dice = (2 * inter / (p.sum() + y.sum() + 1e-6)).item()
    iou = (inter / (p.sum() + y.sum() - inter + 1e-6)).item()
    return dice, iou


def main():
    ap = argparse.ArgumentParser(description="Train court-line segmentation U-Net.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    import torch
    device = args.device or config.get_device()

    tr, va = LineDS("train", aug=True), LineDS("val", aug=False)
    if len(tr) == 0:
        raise SystemExit("No training data — run make_line_dataset.py first.")
    print(f"train={len(tr)}  val={len(va)}  device={device}")

    model = _build_unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))  # lines are sparse

    def dice_loss(logits, y):
        p = torch.sigmoid(logits)
        inter = (p * y).sum((2, 3))
        return (1 - (2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)).mean()

    best = -1.0
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in batches(tr, args.batch, True):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        sched.step()

        model.eval()
        ds_, is_, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in batches(va, args.batch, False):
                x, y = x.to(device), y.to(device)
                p = torch.sigmoid(model(x))
                d, i = dice_iou(p, y)
                ds_ += d; is_ += i; n += 1
        vdice, viou = (ds_ / max(n, 1)), (is_ / max(n, 1))
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep:3d}  train_loss={tl/max(1,len(tr)//args.batch):.3f}  val_dice={vdice:.3f}  val_iou={viou:.3f}")
        if vdice > best:
            best = vdice
            torch.save(model.state_dict(), str(config.LINE_SEG_WEIGHTS))

    print(f"\nbest val Dice={best:.3f}  →  saved {config.LINE_SEG_WEIGHTS}")
    print("Next: wire it into refine (it auto-replaces the top-hat) and re-enable COURT_REFINE.")


if __name__ == "__main__":
    main()

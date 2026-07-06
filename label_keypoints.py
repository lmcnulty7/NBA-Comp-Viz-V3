#!/usr/bin/env python
"""
label_keypoints.py — guided court-keypoint labeler (the manual step for homography).

For each frame it walks you through the 14 court landmarks ONE AT A TIME, showing a
court diagram with the current target highlighted, so you never have to remember the
naming scheme — you just click where that point is, or skip it if it's off-screen /
not visible. Only label points you can locate confidently; ≥4 per frame is enough for
a homography, more is better.

Source = a seeded sample of confirmed-live frames from truth/live. One JSON per frame
is written to data/court/kp_labels/, so you can quit and resume anytime.

Controls
────────
  left-click   place the CURRENT highlighted keypoint, advance to the next
  k / SPACE    skip the current keypoint (off-screen / not visible)
  u            undo: step back and clear the last point
  b            back one keypoint (without clearing) to re-place it
  d            toggle the court-diagram inset (if it covers a point you need)
  n            save this frame and go to the next
  s            skip this whole frame (don't save)
  q / Esc      quit (progress saved)

Usage
─────
  python label_keypoints.py              # 120 frames from truth/live
  python label_keypoints.py --n 150
  python label_keypoints.py --review     # revisit all sampled frames to fix any
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import config
from court.geometry import KEYPOINT_NAMES, draw_court_diagram
from gate.common import list_images

WIN = "label_keypoints — click=place  k=skip  u=undo  b=back  d=diagram  n=save+next  s=skip  q=quit"
INSET_W, INSET_H = 330, 176


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided court-keypoint labeler.")
    ap.add_argument("--n", type=int, default=120, help="Frames to sample from truth/live (default 120).")
    ap.add_argument("--source", type=Path, default=config.TRUTH_LIVE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--review", action="store_true", help="Revisit all sampled frames (preloaded) to fix mistakes.")
    args = ap.parse_args()

    config.COURT_KP_LABELS.mkdir(parents=True, exist_ok=True)
    frames = list_images(args.source)
    if not frames:
        raise SystemExit(f"No frames in {args.source}.")
    random.seed(args.seed)
    sample = sorted(random.sample(frames, min(args.n, len(frames))), key=lambda p: p.name)

    if args.review:
        todo = sample
        print(f"REVIEW: paging all {len(todo)} frames (saved points preloaded).")
    else:
        todo = [p for p in sample if not (config.COURT_KP_LABELS / f"{p.stem}.json").exists()]
        if not todo:
            print(f"All {len(sample)} sampled frames labeled. Next: python train_court_kp.py "
                  "(or --review to fix any).")
            return
        print(f"{len(todo)} frame(s) to label (of {len(sample)} sampled).")

    state = {"k": 0, "placed": {}, "show_diag": True}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["k"] >= len(KEYPOINT_NAMES):
            return
        W = param["w"]
        if state["show_diag"] and x >= W - INSET_W - 5 and y <= INSET_H + 5:
            return  # ignore clicks on the diagram inset
        state["placed"][KEYPOINT_NAMES[state["k"]]] = (int(x), int(y))
        state["k"] += 1

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    i = 0
    saved = 0
    while i < len(todo):
        path = todo[i]
        base = cv2.imread(str(path))
        if base is None:
            i += 1
            continue
        H, W = base.shape[:2]
        existing = config.COURT_KP_LABELS / f"{path.stem}.json"
        placed = {}
        if existing.exists():
            kp = json.loads(existing.read_text()).get("keypoints", {})
            placed = {n: tuple(v) for n, v in kp.items() if v is not None}
        state["k"] = 0
        state["placed"] = placed
        cv2.setMouseCallback(WIN, on_mouse, {"w": W})

        while True:
            canvas = base.copy()
            # placed points
            for name, pt in state["placed"].items():
                if pt is not None:
                    cv2.circle(canvas, pt, 4, (60, 220, 60), -1)
                    cv2.putText(canvas, name.replace("home_", "H").replace("away_", "A").replace("paint_", "p").replace("three_", "3"),
                                (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 220, 60), 1)
            k = state["k"]
            target = KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else None
            # top bar
            cv2.rectangle(canvas, (0, 0), (W, 30), (0, 0, 0), -1)
            n_done = sum(1 for v in state["placed"].values() if v is not None)
            msg = (f"frame {i+1}/{len(todo)}  kp {min(k+1,14)}/14: "
                   f"{target if target else 'ALL DONE — press n to save'}   ({n_done} placed)  "
                   "click=place k=skip u=undo b=back n=save d=diagram q=quit")
            cv2.putText(canvas, msg, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
            # diagram inset
            if state["show_diag"]:
                diag = draw_court_diagram(highlight=target, w=INSET_W, h=INSET_H)
                canvas[5:5 + INSET_H, W - INSET_W - 5:W - 5] = diag
            cv2.imshow(WIN, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("k"), ord(" ")):
                if k < len(KEYPOINT_NAMES):
                    state["placed"][KEYPOINT_NAMES[k]] = None
                    state["k"] += 1
            elif key == ord("u"):
                if state["k"] > 0:
                    state["k"] -= 1
                    state["placed"].pop(KEYPOINT_NAMES[state["k"]], None)
            elif key == ord("b"):
                state["k"] = max(0, state["k"] - 1)
            elif key == ord("d"):
                state["show_diag"] = not state["show_diag"]
            elif key in (ord("n"), 13):  # save + next
                kp_out = {n: (list(state["placed"][n]) if state["placed"].get(n) else None)
                          for n in KEYPOINT_NAMES}
                rec = {"frame": path.name, "image_path": str(path), "width": W, "height": H,
                       "keypoints": kp_out,
                       "n_visible": sum(1 for v in kp_out.values() if v is not None)}
                existing.write_text(json.dumps(rec, indent=2))
                saved += 1
                i += 1
                break
            elif key == ord("s"):
                i += 1
                break
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"\nSaved {saved} frame(s) this session. "
                      f"Total: {len(list(config.COURT_KP_LABELS.glob('*.json')))} labeled.")
                print("Resume anytime; when done: python train_court_kp.py")
                return

    cv2.destroyAllWindows()
    n_total = len(list(config.COURT_KP_LABELS.glob("*.json")))
    print(f"\nDone. Saved {saved} this session; {n_total} frames labeled total.")
    print("Next: python train_court_kp.py")


if __name__ == "__main__":
    main()

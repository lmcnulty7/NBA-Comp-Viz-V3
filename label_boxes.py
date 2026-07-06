#!/usr/bin/env python
"""
label_boxes.py — draw ground-truth player boxes on a sample of live frames.

This is the (small) manual step for the lightweight tracking eval: hand-label
player bounding boxes on ~50 frames so `evaluate_tracking.py` can compute true
detection precision/recall. Boxes are drawn from scratch (NOT pre-filled with
model predictions) so the ground truth is unbiased by the detector being graded.

Source = a seeded sample of confirmed-live frames from truth/live (they're exactly
the "usable court footage" the detector runs on). One JSON per frame is written to
data/tracking/box_truth/, so you can quit and resume anytime.

Controls
────────
  left-drag    draw a box around one player (every player you can see)
  u            undo last box
  c            clear all boxes on this frame
  n / SPACE    save this frame's boxes and go to the next
  b            go BACK to the previous frame (to fix a skip/mistake)
  s            skip this frame (don't save it as ground truth)
  q / Esc      quit (progress saved)

Already-labeled frames reload their saved boxes when revisited (via b or on
re-run), so you can correct earlier work.

Label every clearly-visible player/person on the court (include refs — ref/player
separation is a later concern). Skip a frame if you're unsure rather than guess.

Usage
─────
  python label_boxes.py            # 50 frames from truth/live
  python label_boxes.py --n 80
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

import config
from gate.common import list_images

WIN = "label_boxes — drag=box  u=undo  c=clear  n/space=next  s=skip  q=quit"


def draw_state(base, boxes, cur):
    out = base.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 60), 2)
    if cur is not None:
        cv2.rectangle(out, (cur[0], cur[1]), (cur[2], cur[3]), (60, 220, 255), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, f"boxes: {len(boxes)}   drag=box  u=undo  c=clear  n=next  b=back  s=skip  q=quit",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 60), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-label player boxes for detection eval.")
    ap.add_argument("--n", type=int, default=50, help="How many live frames to sample (default 50).")
    ap.add_argument("--source", type=Path, default=config.TRUTH_LIVE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--review", action="store_true",
                    help="Revisit ALL sampled frames (saved boxes preloaded) to fix mistakes, "
                         "instead of only the still-unlabeled ones. Use n/b to page, edit, re-save.")
    args = ap.parse_args()

    config.TRACK_BOX_TRUTH.mkdir(parents=True, exist_ok=True)
    frames = list_images(args.source)
    if not frames:
        raise SystemExit(f"No frames in {args.source}. (truth/live should hold confirmed live frames.)")

    random.seed(args.seed)
    sample = sorted(random.sample(frames, min(args.n, len(frames))), key=lambda p: p.name)
    if args.review:
        todo = sample   # revisit everything so you can fix any saved frame
        print(f"REVIEW mode: paging through all {len(todo)} sampled frames (n=next, b=back, edit + n to re-save).")
    else:
        todo = [p for p in sample if not (config.TRACK_BOX_TRUTH / f"{p.stem}.json").exists()]
        if not todo:
            done = len(list(config.TRACK_BOX_TRUTH.glob("*.json")))
            print(f"All {len(sample)} sampled frames already labeled ({done} box files). "
                  "Run: python evaluate_tracking.py  (or --review to fix any frame).")
            return
        print(f"{len(todo)} frame(s) to label (of {len(sample)} sampled).")

    state = {"drawing": False, "ix": 0, "iy": 0, "cur": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["ix"], state["iy"] = x, y
            state["cur"] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["cur"] = (state["ix"], state["iy"], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            x1, y1 = state["ix"], state["iy"]
            x2, y2 = x, y
            state["cur"] = None
            bx = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if (bx[2] - bx[0]) >= 4 and (bx[3] - bx[1]) >= 4:  # ignore stray clicks
                param.append(bx)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    i = 0
    labeled = 0
    while i < len(todo):
        path = todo[i]
        base = cv2.imread(str(path))
        if base is None:
            i += 1
            continue
        # Reload saved boxes if this frame was already labeled (so b=back / re-runs
        # let you correct earlier work instead of starting blank).
        existing = config.TRACK_BOX_TRUTH / f"{path.stem}.json"
        if existing.exists():
            boxes: list[tuple[int, int, int, int]] = [
                tuple(int(v) for v in b) for b in json.loads(existing.read_text()).get("boxes", [])
            ]
        else:
            boxes = []
        cv2.setMouseCallback(WIN, on_mouse, boxes)
        state["cur"] = None

        while True:
            cv2.imshow(WIN, draw_state(base, boxes, state["cur"]))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), ord(" ")):
                rec = {"frame": path.name, "image_path": str(path),
                       "width": base.shape[1], "height": base.shape[0],
                       "boxes": [[int(v) for v in b] for b in boxes]}
                (config.TRACK_BOX_TRUTH / f"{path.stem}.json").write_text(json.dumps(rec, indent=2))
                labeled += 1
                i += 1
                break
            elif key == ord("u") and boxes:
                boxes.pop()
            elif key == ord("c"):
                boxes.clear()
            elif key == ord("s"):
                i += 1
                break
            elif key == ord("b"):
                i = max(0, i - 1)   # revisit previous frame (its saved boxes reload)
                break
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"\nLabeled {labeled} frame(s) this session. "
                      f"Total box files: {len(list(config.TRACK_BOX_TRUTH.glob('*.json')))}")
                print("Resume anytime; when done: python evaluate_tracking.py")
                return

    cv2.destroyAllWindows()
    n_done = len(list(config.TRACK_BOX_TRUTH.glob("*.json")))
    print(f"\nDone. Labeled {labeled} this session; {n_done} total frames with boxes.")
    print("Next: python evaluate_tracking.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
label_frames.py — fast keypress labeler for building truth/ (the human step).

Shows one frame at a time from data/visibility/_unsorted/ with CLIP's pre-sort
guess as a *hint*, and you decide with a single keystroke. Your keypress — not
the hint — is what creates the label:

    l = LIVE   → copy frame into truth/live
    d = DEAD   → copy frame into truth/dead
    s = SKIP   → leave it unlabeled (reappears next run)
    u = UNDO   → take back the previous label
    q / Esc    = quit (progress is saved as you go)

Notes
─────
• Source is the ORIGINAL frames in _unsorted/ (clean filenames). The CLIP hint is
  read from the predicted/ filenames but is only a suggestion — overrule it freely.
• Frames already present in truth/ are skipped, so you can quit and resume anytime.
• Copies (not moves) into truth/, leaving _unsorted/ intact as the raw pool.

Usage
─────
  python label_frames.py
  python label_frames.py --max 300      # cap this session
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

import config
from gate.common import list_images


def build_hint_map() -> dict[str, tuple[str, float]]:
    """original_filename -> (clip_guess, confidence) parsed from predicted/ names."""
    hints: dict[str, tuple[str, float]] = {}
    for guess, folder in (("live", config.PREDICTED_LIVE), ("dead", config.PREDICTED_DEAD)):
        for p in list_images(folder):
            # name format: plive{conf}__{original_filename}
            try:
                prefix, original = p.name.split("__", 1)
                conf = float(prefix.replace("plive", ""))
            except ValueError:
                original, conf = p.name, float("nan")
            hints[original] = (guess, conf)
    return hints


def already_labeled() -> set[str]:
    return {p.name for p in list_images(config.TRUTH_LIVE)} | {p.name for p in list_images(config.TRUTH_DEAD)}


def draw_banner(img, text_lines: list[str]):
    """Draw a translucent banner with status/legend across the top of the frame."""
    out = img.copy()
    h, w = out.shape[:2]
    band_h = 26 * len(text_lines) + 12
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for i, line in enumerate(text_lines):
        cv2.putText(out, line, (12, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (60, 255, 60), 2, cv2.LINE_AA)
    return out


def fit_to_screen(img, max_w=1280, max_h=760):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Keypress labeler: _unsorted/ → truth/{live,dead}.")
    ap.add_argument("--source", type=Path, default=config.UNSORTED_DIR,
                    help="Folder of frames to label (default: _unsorted/).")
    ap.add_argument("--max", type=int, default=None, help="Max frames to label this session.")
    args = ap.parse_args()

    config.TRUTH_LIVE.mkdir(parents=True, exist_ok=True)
    config.TRUTH_DEAD.mkdir(parents=True, exist_ok=True)

    hints = build_hint_map()
    done = already_labeled()
    queue = [p for p in list_images(args.source) if p.name not in done]
    if not queue:
        print(f"Nothing to label — every frame in {args.source} is already in truth/.")
        print(f"truth/: live={len(list_images(config.TRUTH_LIVE))}  dead={len(list_images(config.TRUTH_DEAD))}")
        return

    print(f"{len(queue)} frame(s) to label (already done: {len(done)}). "
          "Keys: [l]ive  [d]ead  [s]kip  [u]ndo  [q]uit")

    win = "label_frames — l=live  d=dead  s=skip  u=undo  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    undo_stack: list[tuple[Path, Path]] = []   # (source, copied_dest)
    n_live = len(list_images(config.TRUTH_LIVE))
    n_dead = len(list_images(config.TRUTH_DEAD))

    i = 0
    labeled_this_session = 0
    while i < len(queue):
        if args.max is not None and labeled_this_session >= args.max:
            print(f"Reached --max {args.max} for this session.")
            break

        path = queue[i]
        img = cv2.imread(str(path))
        if img is None:
            i += 1
            continue
        img = fit_to_screen(img)

        guess, conf = hints.get(path.name, ("?", float("nan")))
        conf_str = f"{conf:.2f}" if conf == conf else "n/a"
        banner = [
            f"[{i + 1}/{len(queue)}]  CLIP hint: {guess.upper()} ({conf_str})   truth so far: live={n_live} dead={n_dead}",
            "l=LIVE   d=DEAD   s=skip   u=undo   q=quit",
        ]
        cv2.imshow(win, draw_banner(img, banner))
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key == ord("l"):
            dest = config.TRUTH_LIVE / path.name
            shutil.copy2(path, dest)
            undo_stack.append((path, dest))
            n_live += 1; labeled_this_session += 1; i += 1
        elif key == ord("d"):
            dest = config.TRUTH_DEAD / path.name
            shutil.copy2(path, dest)
            undo_stack.append((path, dest))
            n_dead += 1; labeled_this_session += 1; i += 1
        elif key == ord("s"):
            i += 1
        elif key == ord("u"):
            if undo_stack:
                src, dest = undo_stack.pop()
                if dest.exists():
                    if dest.parent == config.TRUTH_LIVE: n_live -= 1
                    else: n_dead -= 1
                    dest.unlink()
                # step back to the frame we just undid
                while i > 0 and queue[i - 1] != src:
                    i -= 1
                i = max(0, i - 1)
                labeled_this_session = max(0, labeled_this_session - 1)
            else:
                print("Nothing to undo.")
        # any other key: redraw same frame

    cv2.destroyAllWindows()
    n_live = len(list_images(config.TRUTH_LIVE))
    n_dead = len(list_images(config.TRUTH_DEAD))
    print(f"\nLabeled this session: {labeled_this_session}")
    print(f"truth/ totals → live={n_live}  dead={n_dead}  (target ~{config.MIN_PER_CLASS_WARN}/class)")
    if min(n_live, n_dead) < config.MIN_PER_CLASS_WARN:
        print("Keep going / extract more frames before training for a meaningful TEST set.")
    else:
        print("Enough to train: python train_gate.py")


if __name__ == "__main__":
    main()

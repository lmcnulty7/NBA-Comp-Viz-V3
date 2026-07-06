#!/usr/bin/env python
"""
verify_projections.py — rapid-fire human verdicts on the court projections.

Shows the pre-rendered overlays from rank_projections.py (worst-first). One keystroke
per frame, auto-advance. Verdicts are saved after EVERY keypress (crash-safe) and the
tool resumes where you left off.

Keys:
  SPACE / g   = LINES UP (projection sits on the real court)          -> next
  x / b       = DOES NOT LINE UP                                       -> next
  u           = undo last verdict (steps back)
  s           = skip for now (no verdict)
  q           = quit (progress is already saved)

Verdicts -> data/court_review/projection_verdicts.tsv

Usage:
  /opt/anaconda3/bin/python verify_projections.py
      -> reviews the projection ranking (projection_rank.tsv / proj_review/)
  /opt/anaconda3/bin/python verify_projections.py <rank.tsv> <img_dir> <verdicts_out.tsv>
      -> review any other ranked render set, e.g. the line-snap results:
         verify_projections.py data/court_review/snap_rank.tsv data/court_review/snap_review data/court_review/snap_verdicts.tsv
"""
from __future__ import annotations
import os, sys
import cv2

ROOT = "data/court_review"
RANK_TSV = sys.argv[1] if len(sys.argv) > 3 else os.path.join(ROOT, "projection_rank.tsv")
IMG_DIR = sys.argv[2] if len(sys.argv) > 3 else os.path.join(ROOT, "proj_review")
VERDICTS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "projection_verdicts.tsv")
DISP_W = 1280

GOOD, BAD = "lineup", "no_lineup"


def main():
    order = []                      # [(frame_stem, filename, score)]
    with open(RANK_TSV) as f:
        next(f)
        for line in f:
            t = line.rstrip("\n").split("\t")
            order.append((t[2], t[1], t[4]))

    verdicts = {}
    if os.path.exists(VERDICTS):
        with open(VERDICTS) as f:
            next(f)
            for line in f:
                t = line.rstrip("\n").split("\t")
                if len(t) >= 2:
                    verdicts[t[0]] = t[1]

    def save():
        with open(VERDICTS, "w") as f:
            f.write("frame\tverdict\n")
            for k, v in verdicts.items():
                f.write(f"{k}\t{v}\n")

    idx = next((i for i, (stem, _, _) in enumerate(order) if stem not in verdicts), 0)
    hist = []
    cv2.namedWindow("VERIFY")

    while 0 <= idx < len(order):
        stem, fn, score = order[idx]
        img = cv2.imread(os.path.join(IMG_DIR, fn))
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        disp = cv2.resize(img, (DISP_W, int(h * DISP_W / w)))
        n_good = sum(1 for v in verdicts.values() if v == GOOD)
        n_bad = sum(1 for v in verdicts.values() if v == BAD)
        cur = verdicts.get(stem)
        hud = f"[{idx + 1}/{len(order)}]  lineup={n_good}  no_lineup={n_bad}" + (f"  (marked: {cur})" if cur else "")
        cv2.rectangle(disp, (0, disp.shape[0] - 46), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
        cv2.putText(disp, hud, (8, disp.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, "SPACE/g = lines up     x/b = does NOT line up     u=undo  s=skip  q=quit",
                    (8, disp.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 255, 140), 1, cv2.LINE_AA)
        cv2.imshow("VERIFY", disp)

        k = cv2.waitKey(0) & 0xFF
        if k in (ord(' '), ord('g')):
            verdicts[stem] = GOOD; save(); hist.append(idx); idx += 1
        elif k in (ord('x'), ord('b')):
            verdicts[stem] = BAD; save(); hist.append(idx); idx += 1
        elif k == ord('u') and hist:
            idx = hist.pop(); verdicts.pop(order[idx][0], None); save()
        elif k == ord('s'):
            idx += 1
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()
    n_good = sum(1 for v in verdicts.values() if v == GOOD)
    n_bad = sum(1 for v in verdicts.values() if v == BAD)
    print(f"verdicts: {len(verdicts)}/{len(order)}   lineup={n_good}   no_lineup={n_bad}")
    print(f"saved -> {VERDICTS}")


if __name__ == "__main__":
    main()

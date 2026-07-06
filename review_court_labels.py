#!/usr/bin/env python
"""
review_court_labels.py — manual verify/correct the densified 33-pt court labels.

Two windows:
  • MAIN  — the broadcast frame with the current points. GREEN = visible/verified,
            ORANGE = best-model fill to scrutinize, YELLOW RING = occluded (present but
            hidden, placed at inferred location). Index number by each dot.
  • REF   — the 33-point court diagram, so you know which index sits where.

3-state visibility (COCO convention): green=visible(2), yellow=occluded(1), deleted=absent(0).
Occluded-but-inferable points should be PLACED + marked occluded (o), NOT deleted — deleting
teaches the model to suppress a point that's really there. Only delete if truly unknowable.

Mouse (in MAIN window):
  • LEFT-drag a dot        → move it (turns GREEN = verified visible)
  • RIGHT-click a dot      → delete it (absent — out of frame / truly unknowable)
  • 'a' then LEFT-click    → ADD the prompted missing index at the cursor, then advance to
                             the next missing index. TAB skips an index. ESC leaves add-mode.

Keys:
  n / SPACE = save + next      p = save + prev      s = save
  o = toggle nearest dot OCCLUDED (hover over it, press o)
  u = undo last change         q = save + quit
Progress is saved to data/court_review/reviewed.txt — rerun to resume where you left off.

Usage:
  python review_court_labels.py                 # start at first un-reviewed image
  python review_court_labels.py --start 500     # jump to index 500
  python review_court_labels.py --only-fills     # only show images that still have fills
"""
from __future__ import annotations
import argparse, glob, os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, court_ft_to_px, draw_court_topdown

ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
REVIEWED = os.path.join(ROOT, "reviewed.txt")
REF_IMG = "data/court/_our33_reference.png"
DISP_W = 1180                     # main window display width
HIT = 16                         # px radius to grab/delete a point


def load_label(lp):
    pts = {}                     # idx -> [x, y, src]  (normalized x,y ; src 1=human 2=fill)
    for line in open(lp):
        t = line.split()
        if len(t) >= 4:
            pts[int(t[0])] = [float(t[1]), float(t[2]), int(t[3])]
    return pts


def save_label(lp, pts):
    with open(lp, "w") as f:
        for i in sorted(pts):
            x, y, s = pts[i]
            f.write(f"{i} {x:.6f} {y:.6f} {s}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--only-fills", action="store_true")
    args = ap.parse_args()

    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))]
    reviewed = set(open(REVIEWED).read().split()) if os.path.exists(REVIEWED) else set()

    if args.only_fills:
        stems = [s for s in stems if any(v[2] == 2 for v in load_label(os.path.join(LBL_DIR, s + ".txt")).values())]

    idx = args.start if args.start is not None else next((i for i, s in enumerate(stems) if s not in reviewed), 0)

    ref = cv2.imread(REF_IMG)
    if ref is not None:
        cv2.namedWindow("REF", cv2.WINDOW_NORMAL); cv2.imshow("REF", cv2.resize(ref, (520, int(ref.shape[0]*520/ref.shape[1]))))
    cv2.namedWindow("MAIN")

    state = {"pts": {}, "sel": None, "add": False, "add_i": None, "hist": [], "W": 1, "H": 1,
             "scale": 1.0, "mx": 0, "my": 0, "map": True}

    def missing():
        return [i for i in range(33) if i not in state["pts"]]

    def next_missing(after=-1):
        m = [i for i in missing() if i > after] or missing()
        return m[0] if m else None

    def to_norm(px, py):
        return px / (state["W"]*state["scale"]), py / (state["H"]*state["scale"])

    def nearest(px, py):
        best, bd = None, 1e9
        for i, (x, y, s) in state["pts"].items():
            dx, dy = x*state["W"]*state["scale"]-px, y*state["H"]*state["scale"]-py
            d = (dx*dx+dy*dy)**0.5
            if d < bd: bd, best = d, i
        return best if bd <= HIT else None

    def on_mouse(ev, px, py, flags, param):
        state["mx"], state["my"] = px, py
        if ev == cv2.EVENT_LBUTTONDOWN:
            if state["add"] and state["add_i"] is not None:
                state["hist"].append(dict(state["pts"]))
                nx, ny = to_norm(px, py); state["pts"][state["add_i"]] = [nx, ny, 2]
                state["add_i"] = next_missing(state["add_i"])
                if state["add_i"] is None: state["add"] = False
            else:
                state["sel"] = nearest(px, py)
                if state["sel"] is not None: state["hist"].append(dict(state["pts"]))
        elif ev == cv2.EVENT_MOUSEMOVE and state["sel"] is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            nx, ny = to_norm(px, py); p = state["pts"][state["sel"]]; p[0], p[1], p[2] = nx, ny, 2
        elif ev == cv2.EVENT_LBUTTONUP:
            state["sel"] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            i = nearest(px, py)
            if i is not None: state["hist"].append(dict(state["pts"])); state["pts"].pop(i)

    cv2.setMouseCallback("MAIN", on_mouse)

    def load(i):
        stem = stems[i]
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        state["H"], state["W"] = img.shape[:2]; state["scale"] = DISP_W/state["W"]
        state["pts"] = load_label(os.path.join(LBL_DIR, stem + ".txt")); state["hist"] = []; state["sel"] = None
        state["add"] = False; state["add_i"] = None
        return stem, cv2.resize(img, (DISP_W, int(state["H"]*state["scale"])))

    def commit(i):
        save_label(os.path.join(LBL_DIR, stems[i] + ".txt"), state["pts"])
        reviewed.add(stems[i]); open(REVIEWED, "w").write("\n".join(sorted(reviewed)))

    def court_inset():
        """Small top-down court in the corner, each vertex recolored to its current state —
        green=verified, orange=fill, yellow=occluded, gray=missing, magenta=add-target."""
        board, sc_i, mg_i = draw_court_topdown(3.0, 14)
        for i, (X, Y) in enumerate(COURT_VERTICES_33):
            p = court_ft_to_px((X, Y), sc_i, mg_i)[0]
            cx, cy = int(p[0]), int(p[1])
            if i in state["pts"]:
                s = state["pts"][i][2]
                col = (0, 255, 255) if s == 1 else (0, 255, 0) if s == 2 else (0, 165, 255)
                cv2.circle(board, (cx, cy), 4, col, -1)
            else:
                col = (110, 110, 110)
                cv2.circle(board, (cx, cy), 3, col, 1)
            if state["add"] and i == state["add_i"]:
                cv2.circle(board, (cx, cy), 7, (255, 0, 255), 2); col = (255, 0, 255)
            cv2.putText(board, str(i), (cx + 4, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(board, str(i), (cx + 4, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        return board

    stem, disp = load(idx)
    while True:
        canvas = disp.copy(); sc = state["scale"]
        for i, (x, y, s) in state["pts"].items():
            cx, cy = int(x*state["W"]*sc), int(y*state["H"]*sc)
            if s == 1:      col = (0, 255, 255); cv2.circle(canvas, (cx, cy), 6, col, 2)          # occluded: yellow ring
            elif s == 2:    col = (0, 255, 0);   cv2.circle(canvas, (cx, cy), 5, col, -1)         # visible/verified: green
            else:           col = (0, 165, 255); cv2.circle(canvas, (cx, cy), 5, col, -1)         # fill/unverified: orange
            tp = (cx+6, cy-6)                                                                    # black halo so the index reads against any background
            cv2.putText(canvas, str(i), tp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(i), tp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        nf = sum(1 for v in state["pts"].values() if v[2] == 3)
        hud = f"[{idx+1}/{len(stems)}] {stem[:26]}  pts={len(state['pts'])} unverified(orange)={nf}"
        if state["add"]: hud = f"ADD index {state['add_i']} -> click (TAB skip, ESC done)   " + hud
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, hud, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, "n=next p=prev a=add o=occluded m=map rclick=del u=undo q=quit",
                    (6, canvas.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        if state["map"]:                                           # live court map, top-right (toggle with 'm')
            ins = court_inset()
            ih, iw = ins.shape[:2]
            x0, y0 = canvas.shape[1] - iw - 8, 30
            if x0 >= 0 and y0 + ih <= canvas.shape[0]:
                cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + iw + 1, y0 + ih + 1), (255, 255, 255), 1)
                canvas[y0:y0 + ih, x0:x0 + iw] = ins
        cv2.imshow("MAIN", canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == 255: continue
        if k in (ord('n'), ord(' ')):
            commit(idx); idx = min(idx+1, len(stems)-1); stem, disp = load(idx)
        elif k == ord('p'):
            commit(idx); idx = max(idx-1, 0); stem, disp = load(idx)
        elif k == ord('s'): commit(idx)
        elif k == ord('a'):
            state["add"] = True; state["add_i"] = next_missing()
            if state["add_i"] is None: state["add"] = False
        elif k == 9:  # TAB
            if state["add"]: state["add_i"] = next_missing(state["add_i"] if state["add_i"] is not None else -1)
        elif k == 27:  # ESC
            state["add"] = False; state["add_i"] = None
        elif k == ord('o'):                      # toggle nearest point occluded<->visible
            i = nearest(state["mx"], state["my"])
            if i is not None:
                state["hist"].append(dict(state["pts"]))
                p = state["pts"][i]; p[2] = 1 if p[2] != 1 else 2
        elif k == ord('m'):                      # toggle the corner court map
            state["map"] = not state["map"]
        elif k == ord('u') and state["hist"]:
            state["pts"] = state["hist"].pop()
        elif k == ord('q'):
            commit(idx); break
    cv2.destroyAllWindows()
    print(f"reviewed {len(reviewed)}/{len(stems)} images")


if __name__ == "__main__":
    main()

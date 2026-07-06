#!/usr/bin/env python
"""
rank_projections.py — rank every court_review frame by how badly its pure-download
homography projection mismatches the ACTUAL court pixels.

Fit: H from code-2 points only (the box-center conversions of the downloaded dataset,
post swap-fix). Model fills (code 3) are ignored entirely.

Score (external evidence, not internal consistency): project the court-line template
through H, sample points every 2 ft along the on-frame segments, and measure each
sample's distance to the nearest line-like pixel (white top-hat + black top-hat, so
bright lines on dark floors AND dark lines on light floors both count). Mean capped
distance = mismatch score; high = the projected court is NOT sitting on real lines.
This is a SORTING heuristic for human review — the user's eyes are the judge.

Outputs:
  data/court_review/projection_rank.tsv   — rank order, scores
  data/court_review/proj_review/          — pre-rendered overlays named 0000_… (worst first);
                                            frames with no possible H come last, bannered.
"""
from __future__ import annotations
import glob, os
import cv2, numpy as np
from court.court33 import COURT_VERTICES_33, court33_segments, court33_curves

cv2.setRNGSeed(42)
ROOT = "data/court_review"
IMG_DIR, LBL_DIR = os.path.join(ROOT, "images"), os.path.join(ROOT, "labels")
OUT_DIR = os.path.join(ROOT, "proj_review")
TSV = os.path.join(ROOT, "projection_rank.tsv")
STEP_FT = 2.0      # sampling interval along court lines
CAP_PX = 30.0      # cap per-sample distance so one stray segment can't dominate
NEAR_PX = 6.0      # a sample within this of a line pixel counts as "supported"
MIN_SAMPLES = 30   # need at least this many on-frame samples to score


def load_code2(lp):
    pts = {}
    for line in open(lp):
        t = line.split()
        if len(t) >= 4 and int(t[3]) == 2:
            pts[int(t[0])] = (float(t[1]), float(t[2]))
    return pts


def fit_H(c2):
    if len(c2) < 4:
        return None
    idxs = list(c2)
    src = np.array([c2[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    dst = np.array([COURT_VERTICES_33[i] for i in idxs], np.float32).reshape(-1, 1, 2)
    method = 0 if len(idxs) == 4 else cv2.RANSAC
    H, _ = cv2.findHomography(src, dst, method, 3.0)
    return H


def line_distance_field(gray):
    """Distance (px) to the nearest line-like pixel. Top-hat pair catches bright lines
    on dark floors AND dark lines on light floors — color/paint agnostic.
    Crowd texture also fires the top-hat, densely — which would hand false support to
    garbage projections. Court lines are SPARSE thin structures, so gate the mask by
    local density: windows where responses are dense are texture (crowd), not lines."""
    g = cv2.medianBlur(gray, 5)                  # kill the dataset's salt-and-pepper speckle
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    ridge = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                    cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    mask = (ridge > 10).astype(np.uint8)
    density = cv2.boxFilter(mask.astype(np.float32), -1, (31, 31))
    mask[density > 0.20] = 0                     # dense response = texture, not a line
    if mask.sum() == 0:
        return None
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)


def h_sanity(H):
    """Geometric sanity of H (normalized image -> court ft), no pixels involved.
    Returns (n_vertices_on_frame, folded). A collapsed H stuffs nearly all 33 court
    vertices into the frame; an exploded one shows almost none; a folded one flips
    orientation inside the frame (projective fold) — all hallmarks of a garbage fit."""
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 33, True
    p = cv2.perspectiveTransform(COURT_VERTICES_33.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 2)
    ok = np.isfinite(p).all(axis=1)
    on = int(np.sum(ok & (p[:, 0] >= 0) & (p[:, 0] <= 1) & (p[:, 1] >= 0) & (p[:, 1] <= 1)))
    signs = set()
    for x, y in [(0.01, 0.01), (0.99, 0.01), (0.99, 0.99), (0.01, 0.99)]:
        d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        u, v = H[0, 0] * x + H[0, 1] * y + H[0, 2], H[1, 0] * x + H[1, 1] * y + H[1, 2]
        J = np.array([[H[0, 0] * d - u * H[2, 0], H[0, 1] * d - u * H[2, 1]],
                      [H[1, 0] * d - v * H[2, 0], H[1, 1] * d - v * H[2, 1]]]) / d ** 2
        signs.add(float(np.sign(np.linalg.det(J))))
    return on, len(signs) > 1


def sample_segments(Hinv, w, h):
    """Sample the court template every STEP_FT along each segment; return on-frame px."""
    out = []
    for P1, P2 in court33_segments():
        P1, P2 = np.asarray(P1, np.float32), np.asarray(P2, np.float32)
        n = max(2, int(np.linalg.norm(P2 - P1) / STEP_FT))
        ft = P1[None, :] + np.linspace(0, 1, n)[:, None] * (P2 - P1)[None, :]
        px = cv2.perspectiveTransform(ft.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        ok = np.isfinite(px).all(axis=1)
        px = px[ok]
        inside = (px[:, 0] >= 1) & (px[:, 0] < w - 1) & (px[:, 1] >= 1) & (px[:, 1] < h - 1)
        out.append(px[inside])
    return np.concatenate(out) if out else np.empty((0, 2), np.float32)


def render(stem, img, Hinv, c2, hud, banner=None):
    h, w = img.shape[:2]
    if Hinv is not None:
        for P1, P2 in court33_segments():
            seg = cv2.perspectiveTransform(np.array([[P1], [P2]], np.float32), Hinv).reshape(-1, 2)
            if np.isfinite(seg).all():
                cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
        # curved features (3-pt arcs, corner-3s, circles): project the sampled polylines.
        # Samples near the camera's horizon explode to huge coordinates — draw only
        # consecutive pairs that stay within a sane multiple of the frame.
        for poly in court33_curves():
            px = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            for a, b in zip(px[:-1], px[1:]):
                if (np.isfinite(a).all() and np.isfinite(b).all()
                        and max(abs(a[0]), abs(b[0])) < 4 * w and max(abs(a[1]), abs(b[1])) < 4 * h):
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    for i, (nx, ny) in c2.items():
        cv2.circle(img, (int(nx * w), int(ny * h)), 5, (0, 255, 0), -1)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if banner:
        cv2.putText(img, banner, (w // 2 - 170, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, banner, (w // 2 - 170, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUT_DIR, "*.jpg")):
        os.remove(old)

    scored, no_h = [], []
    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(LBL_DIR, "*.txt")))]
    for n, stem in enumerate(stems):
        if n % 300 == 0:
            print(f"  scoring {n}/{len(stems)} …")
        c2 = load_code2(os.path.join(LBL_DIR, stem + ".txt"))
        img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        H = fit_H(c2)
        Hinv = None
        if H is not None:
            S = np.array([[1.0 / w, 0, 0], [0, 1.0 / h, 0], [0, 0, 1]], np.float32)
            try:
                Hinv = np.linalg.inv(H @ S)
            except np.linalg.LinAlgError:
                Hinv = None
        med = support = None
        if Hinv is not None:
            pts = sample_segments(Hinv, w, h)
            if len(pts) >= MIN_SAMPLES:
                dt = line_distance_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                if dt is not None:
                    d = dt[pts[:, 1].astype(int), pts[:, 0].astype(int)]
                    med = float(np.median(np.minimum(d, CAP_PX)))
                    support = float((d < NEAR_PX).mean())
        if med is None:
            no_h.append((stem, img, Hinv, c2))
            continue
        n_on, fold = h_sanity(H)
        # primary rank = geometric insanity; secondary = median line distance
        if fold:
            final, tag = 3000 + n_on, "FOLDED H"
        elif n_on >= 30:
            final, tag = 2000 + n_on, "COLLAPSED H"
        elif n_on <= 6:
            final, tag = 1000 + (6 - n_on), "EXPLODED H"
        else:
            final, tag = med, ""
        scored.append((stem, img, Hinv, c2, final, med, support, n_on, tag))

    scored.sort(key=lambda r: -r[4])                      # worst mismatch first

    with open(TSV, "w") as f:
        f.write("rank\tfile\tframe\tn_code2\tflag\tverts_on\tmed_px\tsupport\n")
        rank = 0
        for stem, img, Hinv, c2, final, med, support, n_on, tag in scored:
            fn = f"{rank:04d}_{stem}.jpg"
            head = f"{tag}  " if tag else ""
            hud = (f"#{rank}  {head}verts_on={n_on}/33  med={med:.1f}px  support={support:.0%}  "
                   f"downloads={len(c2)}  {stem[:26]}")
            cv2.imwrite(os.path.join(OUT_DIR, fn), render(stem, img, Hinv, c2, hud))
            f.write(f"{rank}\t{fn}\t{stem}\t{len(c2)}\t{tag or '-'}\t{n_on}\t{med:.2f}\t{support:.3f}\n")
            rank += 1
        for stem, img, Hinv, c2 in no_h:
            fn = f"{rank:04d}_{stem}.jpg"
            hud = f"#{rank}  NO HOMOGRAPHY  downloads={len(c2)}  {stem[:40]}"
            cv2.imwrite(os.path.join(OUT_DIR, fn), render(stem, img, None, c2, hud, banner="NO HOMOGRAPHY"))
            f.write(f"{rank}\t{fn}\t{stem}\t{len(c2)}\tNO-H\t-\tnan\tnan\n")
            rank += 1

    flags = [r[8] for r in scored if r[8]]
    clean = np.array([r[5] for r in scored if not r[8]])
    print(f"\nscored {len(scored)} frames  (+{len(no_h)} NO-H at the end of the queue)")
    print(f"geometrically insane H: {len(flags)}  "
          f"(folded {sum(1 for t in flags if t == 'FOLDED H')}, "
          f"collapsed {sum(1 for t in flags if t == 'COLLAPSED H')}, "
          f"exploded {sum(1 for t in flags if t == 'EXPLODED H')})")
    if len(clean):
        print(f"sane frames' median line-dist (px): p10={np.percentile(clean,10):.1f}  "
              f"median={np.median(clean):.1f}  p90={np.percentile(clean,90):.1f}  max={clean.max():.1f}")
    print(f"rank TSV -> {TSV}")
    print(f"overlays -> {OUT_DIR}/  (0000 = worst)")


if __name__ == "__main__":
    main()

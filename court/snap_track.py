"""
court/snap_track.py — CourtTracker: grid detector + line-snap + temporal tracking.

The pipeline's court solver. Per frame:

  1. DETECT   — grid pose model → keypoints → RANSAC H (px → court ft);
  2. SANITY   — reject folded / collapsed / exploded H's (geometric checks, no pixels);
  3. SNAP     — refine a sane H onto the painted lines (ICP-style normal search on a
                density-gated top-hat ridge; sub-pixel; guarded);
  4. TRACK    — when the model fails (pans, motion blur), propagate the previous H by
                the camera's global shift (phase correlation) and let the line-snap
                re-lock it onto whatever lines are visible. The model re-anchors the
                track whenever it produces a sane H again.
  5. HOLD/LOST — if line evidence can't confirm the propagated H, it is HELD (valid
                but low-confidence: quality is set poor so masking won't trust it);
                after MAX_HELD consecutive holds the track is LOST (returns None).

States exposed via .state: "TRACK" | "LINE_TRACK" | "HELD" | "LOST".
Line-evidence primitives are the validated ones from the offline label pipeline
(snap_projections.py); constants retuned only where inference differs (larger
search radius when re-locking a propagated H).
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from .court33 import COURT_VERTICES_33, court33_segments, court33_curves
from .grid import GRID_FT
from .homography import CourtHomography

STEP_FT = 1.0                 # template sampling density (ft)
SNAP_RADII = (8, 5)           # normal-search radii when refining a fresh model H
TRACK_RADII = (14, 10, 6)     # wider first pass when re-locking a propagated H
PEAK_MIN = 12.0               # min ridge response for a line match
RANSAC_PX = 3.0               # line-refit inlier threshold (px)
MIN_SNAP_MATCH = 40           # snap accepted only with this many correspondences
MIN_TRACK_MATCH = 50          # stricter when there's no model H backing the frame
MAX_TRACK_RES = 2.5           # px — propagated H must re-lock at least this well
MAX_CORNER_SNAP = 40.0        # px — max corner motion a snap may introduce
MAX_CORNER_TRACK = 60.0       # px — max corner motion a track re-lock may introduce
PHASE_W = 320                 # downscale width for global-shift estimation


# ── template chords (built once) ─────────────────────────────────────────────
def _build_chords():
    chords = []
    for P1, P2 in court33_segments():
        P1, P2 = np.asarray(P1, np.float64), np.asarray(P2, np.float64)
        n = max(1, int(np.linalg.norm(P2 - P1) / STEP_FT))
        ts = np.linspace(0, 1, n + 1)
        for a, b in zip(ts[:-1], ts[1:]):
            chords.append((P1 + a * (P2 - P1), P1 + b * (P2 - P1)))
    for poly in court33_curves(step_deg=3.0):
        for a, b in zip(poly[:-1], poly[1:]):
            chords.append((np.asarray(a, np.float64), np.asarray(b, np.float64)))
    A = np.array([c[0] for c in chords], np.float32)
    B = np.array([c[1] for c in chords], np.float32)
    return A, B, ((A + B) / 2.0).astype(np.float32)


CH_A, CH_B, CH_MID = _build_chords()


def project(M, pts_ft):
    return cv2.perspectiveTransform(np.asarray(pts_ft, np.float32).reshape(-1, 1, 2),
                                    np.asarray(M, np.float64)).reshape(-1, 2)


def ridge_field(gray):
    """Density-gated line evidence: top-hat pair (bright+dark lines), speckle-denoised,
    crowd texture zeroed. Same recipe validated in the offline snap."""
    g = cv2.medianBlur(gray, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    ridge = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                    cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)).astype(np.float32)
    mask = (ridge > 10).astype(np.uint8)
    density = cv2.boxFilter(mask.astype(np.float32), -1, (31, 31))
    ridge[density > 0.20] = 0.0
    return ridge if (ridge > PEAK_MIN).any() else None


def _bilinear(imgf, xy):
    h, w = imgf.shape
    x, y = xy[..., 0], xy[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    fx, fy = x - x0c, y - y0c
    v = (imgf[y0c, x0c] * (1 - fx) * (1 - fy) + imgf[y0c, x0c + 1] * fx * (1 - fy)
         + imgf[y0c + 1, x0c] * (1 - fx) * fy + imgf[y0c + 1, x0c + 1] * fx * fy)
    v[~ok] = 0.0
    return v


def match_lines(P, ridge, w, h, radius):
    """Court-template samples → sub-pixel line peaks along each sample's normal.
    Returns (court_ft_midpoints, matched_px)."""
    pa, pb = project(P, CH_A), project(P, CH_B)
    mid = (pa + pb) / 2.0
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    ok &= (mid[:, 0] >= 1) & (mid[:, 0] < w - 1) & (mid[:, 1] >= 1) & (mid[:, 1] < h - 1)
    tang = pb - pa
    tlen = np.linalg.norm(tang, axis=1)
    ok &= tlen > 1e-3
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    tang = tang[idx] / tlen[idx, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    T = np.arange(-radius, radius + 1, dtype=np.float32)
    grid = mid[idx][:, None, :] + T[None, :, None] * nrm[:, None, :]
    resp = _bilinear(ridge, grid)
    k = resp.argmax(axis=1)
    peak = resp[np.arange(len(k)), k]
    inner = (k > 0) & (k < len(T) - 1) & (peak >= PEAK_MIN)
    if not inner.any():
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    idx, k = idx[inner], k[inner]
    nrm, r0 = nrm[inner], resp[inner, :]
    left = r0[np.arange(len(k)), k - 1]
    cen = r0[np.arange(len(k)), k]
    right = r0[np.arange(len(k)), k + 1]
    den = left - 2 * cen + right
    delta = np.clip(np.where(den < -1e-6, 0.5 * (left - right) / den, 0.0), -1.0, 1.0)
    t_star = T[k] + delta
    return CH_MID[idx], (mid[idx] + t_star[:, None] * nrm).astype(np.float32)


def h_sane(H_px2ft, w, h):
    """Geometric sanity: enough-but-not-all court vertices on frame, no projective fold."""
    try:
        P = np.linalg.inv(np.asarray(H_px2ft, np.float64))
    except np.linalg.LinAlgError:
        return False
    p = project(P, COURT_VERTICES_33)
    ok = np.isfinite(p).all(axis=1)
    n_on = int(np.sum(ok & (p[:, 0] >= 0) & (p[:, 0] <= w) & (p[:, 1] >= 0) & (p[:, 1] <= h)))
    if n_on >= 30 or n_on <= 5:
        return False
    H = np.asarray(H_px2ft, np.float64)
    signs = set()
    for x, y in [(w * .01, h * .01), (w * .99, h * .01), (w * .99, h * .99), (w * .01, h * .99)]:
        d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        v = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        J = np.array([[H[0, 0] * d - u * H[2, 0], H[0, 1] * d - u * H[2, 1]],
                      [H[1, 0] * d - v * H[2, 0], H[1, 1] * d - v * H[2, 1]]]) / d ** 2
        signs.add(float(np.sign(np.linalg.det(J))))
    return len(signs) == 1


def _corner_motion(P_new, P_ref, w, h):
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    try:
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                         P_new @ np.linalg.inv(P_ref)).reshape(-1, 2)
    except np.linalg.LinAlgError:
        return np.inf
    if not np.isfinite(moved).all():
        return np.inf
    return float(np.linalg.norm(moved - corners, axis=1).max())


class CourtTracker:
    """Grid model + line-snap + temporal propagation. update(frame) → CourtHomography|None."""

    def __init__(self, weights=None, device=None, kp_conf=None, max_held=None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights or config.COURT_GRID_WEIGHTS))
        self.device = device or config.get_device()
        self.kp_conf = config.COURT_GRID_KP_CONF if kp_conf is None else kp_conf
        self.max_held = config.COURT_TRACK_MAX_HELD if max_held is None else max_held
        self.state = "LOST"
        self.last_pts = np.empty((0, 2), np.float32)   # px points supporting the current H
        self._P = None                                  # court ft → px
        self._held = 0
        self._prev_small = None

    # ── pieces ────────────────────────────────────────────────────────────────
    def _detect_fit(self, frame, w, h):
        r = self.model.predict(frame, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None, np.empty((0, 2), np.float32)
        b = int(r.boxes.conf.argmax())
        xy = r.keypoints.xy[b].cpu().numpy()
        kc = (r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None
              else np.ones(len(xy)))
        keep = [(i, xy[i]) for i in range(len(xy))
                if kc[i] >= self.kp_conf and 0 < xy[i][0] < w and 0 < xy[i][1] < h]
        if len(keep) < config.MIN_KEYPOINTS_FOR_H:
            return None, np.empty((0, 2), np.float32)
        src = np.array([p for _, p in keep], np.float32).reshape(-1, 1, 2)
        dst = np.array([GRID_FT[i] for i, _ in keep], np.float32).reshape(-1, 1, 2)
        method = 0 if len(keep) == 4 else cv2.RANSAC
        H, _ = cv2.findHomography(src, dst, method, config.RANSAC_REPROJ_THRESHOLD)
        return H, np.array([p for _, p in keep], np.float32)

    def _snap(self, P, ridge, w, h, radii, max_corner):
        """Guarded line-snap. Returns (P, matched_px, res_px, res_ft, n_match).
        res_px = image-space alignment (for thresholds); res_ft = court-space
        residual (for CourtHomography.quality, gated in feet by the mapper)."""
        P0 = P.copy()
        px = np.empty((0, 2), np.float32)
        for radius in radii:
            ft, px = match_lines(P, ridge, w, h, radius)
            if len(ft) < MIN_SNAP_MATCH:
                break
            Hn, _ = cv2.findHomography(ft.reshape(-1, 1, 2), px.reshape(-1, 1, 2),
                                       cv2.RANSAC, RANSAC_PX)
            if Hn is None:
                break
            P = Hn.astype(np.float64)
        if _corner_motion(P, P0, w, h) > max_corner:
            P = P0
        ft, px = match_lines(P, ridge, w, h, 6)
        res_px = res_ft = None
        if len(ft) >= 10:
            res_px = float(np.median(np.linalg.norm(px - project(P, ft), axis=1)))
            try:
                res_ft = float(np.median(np.linalg.norm(
                    project(np.linalg.inv(P), px) - ft, axis=1)))
            except np.linalg.LinAlgError:
                pass
        return P, px, res_px, res_ft, len(ft)

    def _global_shift(self, small):
        if self._prev_small is None or self._prev_small.shape != small.shape:
            return 0.0, 0.0
        (dx, dy), _ = cv2.phaseCorrelate(self._prev_small, small)
        return dx, dy

    # ── main ─────────────────────────────────────────────────────────────────
    def update(self, frame) -> CourtHomography | None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = PHASE_W / w
        small = cv2.resize(gray, (PHASE_W, int(h * scale))).astype(np.float32)
        ridge = ridge_field(gray)

        H, kp_px = self._detect_fit(frame, w, h)
        if H is not None and h_sane(H, w, h):
            try:
                P = np.linalg.inv(H.astype(np.float64))
            except np.linalg.LinAlgError:
                P = None
            if P is not None:
                res_ft = None
                if ridge is not None:
                    P, mpx, _, res_ft, _ = self._snap(P, ridge, w, h, SNAP_RADII, MAX_CORNER_SNAP)
                    self.last_pts = mpx if len(mpx) >= 3 else kp_px
                else:
                    self.last_pts = kp_px
                self._P, self._held, self.state = P, 0, "TRACK"
                self._prev_small = small
                return CourtHomography.from_matrix(
                    np.linalg.inv(P), quality=res_ft, n_inliers=len(kp_px))

        # model failed → propagate previous H through the camera's global shift
        if self._P is not None and self._held < self.max_held:
            dx, dy = self._global_shift(small)
            T = np.array([[1, 0, dx / scale], [0, 1, dy / scale], [0, 0, 1]], np.float64)
            P0 = T @ self._P
            if ridge is not None:
                P, mpx, res_px, res_ft, n_m = self._snap(P0, ridge, w, h, TRACK_RADII, MAX_CORNER_TRACK)
                relocked = (n_m >= MIN_TRACK_MATCH and res_px is not None
                            and res_px <= MAX_TRACK_RES
                            and h_sane(np.linalg.inv(P), w, h))
                if relocked:
                    self._P, self._held, self.state = P, 0, "LINE_TRACK"
                    self.last_pts = mpx
                    self._prev_small = small
                    return CourtHomography.from_matrix(
                        np.linalg.inv(P), quality=res_ft, n_inliers=n_m)
            # couldn't confirm — hold the shifted H with poor quality (masking won't trust it)
            self._P = P0
            self._held += 1
            self.state = "HELD"
            self.last_pts = np.empty((0, 2), np.float32)
            self._prev_small = small
            return CourtHomography.from_matrix(np.linalg.inv(P0), quality=9.9, n_inliers=0)

        self.state = "LOST"
        self._P = None
        self.last_pts = np.empty((0, 2), np.float32)
        self._prev_small = small
        return None

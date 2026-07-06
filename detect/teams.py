"""
detect/teams.py — Component C1: unsupervised team classification.

Which team is each track on? Three consumers, all already waiting on this:
  • the fragment linker — a HARD VETO (never merge tracks from different teams),
    which lets the conservative ambiguity margin actually resolve candidates;
  • offense/defense assignment for the downstream defensive metrics;
  • non-player filtering (bench/crowd leakage the detector lets through tends
    to abstain rather than land in either team cluster).

Approach: k-means (k=2) on a tiny COLOR feature — the median Lab color of each
torso crop's central patch. Measured dead end first (keep for the record): we
tried clustering the re-ID CLIP embeddings on the theory that jersey color
dominates them; on real crops it split 25-vs-1 with silhouette 0.247 and the
contact sheet showed white and dark jerseys mixed in one cluster. On 30–70 px
broadcast crops, CLIP is dominated by scene statistics (floor, blur, lighting),
not jersey. Jersey color needs to be measured DIRECTLY: median Lab of the
central jersey patch separates a white kit from a dark kit on the L axis alone,
and the median is robust to the floor/skin pixels at the patch edges. CLIP
embeddings remain the re-ID feature (detect/reid.py) — each signal does the job
it's actually good at.

Fit is per-clip (team colors are constant within a game, not across games).
One POOLED color feature per track (see _track_feature — per-crop features were
measured too noisy), nearest-centroid assignment with an ABSTENTION rule: a
track whose pooled color sits nearly equidistant between the two kits
(TEAM_ABSTAIN_RATIO), or with too few crops, stays team=None — a non-player or
a contaminated track should stay out of both teams rather than pollute one.
Cluster ids 0/1 are arbitrary-but-deterministic (seeded k-means); they
mean "team A/B", not home/away — naming teams needs roster/jersey knowledge
that belongs to a later component.

Label-free sanity signals (reported, not asserted): silhouette score of the
2-way split, and per-frame team balance downstream (~5v5 on live possessions).
"""
from __future__ import annotations

import logging

import numpy as np

import config

log = logging.getLogger(__name__)


class TeamClassifier:
    def __init__(self, n_clusters: int = None, seed: int = None):
        self.k = n_clusters or config.TEAM_N_CLUSTERS
        self.seed = config.SEED if seed is None else seed
        self.km = None
        self.silhouette = None

    @staticmethod
    def _track_feature(crop_list: list[np.ndarray]) -> np.ndarray | None:
        """One (5,) color feature per TRACK: [L p25, L p50, L p75, a p50, b p50]
        over the pixels POOLED from all the track's crops.

        Why track-pooled (measured, 3rd iteration): individual broadcast crops
        are up to ~half floor/apron/occluding-opponent, so any per-crop color is
        a coin flip and per-crop voting abstained half the tracks. Pooling
        10–40 crops makes the track's own jersey the dominant pixel mass.
        Patch = inner 50% width × upper 60% height of each crop; pixels inside
        the maple-floor HSV band (config.HSV_LO/HI, the old gate's) are masked
        out first (also catches most skin — same hue range). L quartiles carry
        the light-vs-dark kit signal; a/b medians the hue."""
        import cv2

        pool = []
        for c in crop_list:
            h, w = c.shape[:2]
            patch = c[: max(1, int(0.6 * h)), w // 4: max(w // 4 + 1, 3 * w // 4)]
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            keep = (cv2.inRange(hsv, config.HSV_LO, config.HSV_HI) == 0).reshape(-1)
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).reshape(-1, 3)
            pool.append(lab[keep] if keep.sum() >= 0.25 * len(lab) else lab)
        if not pool:
            return None
        P = np.concatenate(pool, 0).astype(np.float32)
        return np.concatenate([np.percentile(P[:, 0], [25, 50, 75]),
                               np.median(P[:, 1:], axis=0)]).astype(np.float32)

    def fit(self, crops: dict[int, list[np.ndarray]]) -> bool:
        """crops: {track_id: [RGB torso crops]} (the re-ID collector's crops).
        Clusters TRACK features (one per track). Returns False when there isn't
        enough data to cluster."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        feats = [self._track_feature(lst) for lst in crops.values()
                 if len(lst) >= config.TEAM_MIN_CROPS]
        feats = [f for f in feats if f is not None]
        if len(feats) < max(self.k + 2, 4):
            log.info("teams: too few tracks with crops to cluster — skipping")
            return False
        X = np.stack(feats)
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(X)
        self.silhouette = round(float(silhouette_score(X, self.km.predict(X))), 3)
        log.info("teams: k-means fit on %d track color-features, silhouette=%.3f",
                 len(X), self.silhouette)
        return True

    def assign(self, crops: dict[int, list[np.ndarray]]) -> dict[int, tuple[int | None, float]]:
        """{track_id: (team | None, confidence)} — nearest team centroid; abstains
        (None) when the track sits nearly equidistant between the two centroids
        (distance ratio > TEAM_ABSTAIN_RATIO) or has < TEAM_MIN_CROPS crops."""
        out = {}
        for tid, lst in crops.items():
            f = self._track_feature(lst) if (self.km is not None
                                             and len(lst) >= config.TEAM_MIN_CROPS) else None
            if f is None:
                out[tid] = (None, 0.0)
                continue
            d = np.linalg.norm(self.km.cluster_centers_ - f, axis=1)
            near, far = int(d.argmin()), int(d.argmax())
            ratio = float(d[near] / max(d[far], 1e-6))
            conf = round(1.0 - ratio, 3)
            out[tid] = (near if ratio <= config.TEAM_ABSTAIN_RATIO else None, conf)
        return out

    @staticmethod
    def team_colors_bgr(crops: dict[int, list[np.ndarray]],
                        team_of: dict[int, int | None]) -> dict[int, tuple[int, int, int]]:
        """Median jersey color per team (for viz only). Crops are RGB; the median
        of each crop's central patch (inner 50%) approximates the jersey color."""
        px = {}
        for tid, lst in crops.items():
            team = team_of.get(tid)
            if team is None:
                continue
            for c in lst:
                h, w = c.shape[:2]
                patch = c[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
                if patch.size:
                    px.setdefault(team, []).append(np.median(patch.reshape(-1, 3), axis=0))
        return {t: tuple(int(v) for v in np.median(np.array(p), axis=0)[::-1])   # RGB → BGR
                for t, p in px.items()}


def team_contact_sheet(crops: dict[int, list[np.ndarray]],
                       team_of: dict[int, int | None],
                       out_path, per_team: int = 48) -> None:
    """Visual QA artifact: a grid of sample torso crops grouped by assigned team
    (A / B / abstained) so the clustering can be verified at a glance."""
    import cv2

    tile_w, tile_h, cols = 48, 72, 12
    rng = np.random.default_rng(config.SEED)
    groups: dict[str, list[np.ndarray]] = {"team A": [], "team B": [], "abstained": []}
    for tid, lst in crops.items():
        name = {0: "team A", 1: "team B"}.get(team_of.get(tid), "abstained")
        groups[name].extend(lst)
    rows = []
    for name, lst in groups.items():
        if not lst:
            continue
        pick = [lst[i] for i in rng.choice(len(lst), min(per_team, len(lst)), replace=False)]
        tiles = [cv2.resize(c[:, :, ::-1], (tile_w, tile_h)) for c in pick]  # RGB → BGR
        while len(tiles) % cols:
            tiles.append(np.zeros((tile_h, tile_w, 3), np.uint8))
        header = np.zeros((22, tile_w * cols, 3), np.uint8)
        cv2.putText(header, f"{name} ({len(lst)} crops)", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(header)
        for i in range(0, len(tiles), cols):
            rows.append(np.hstack(tiles[i:i + cols]))
    if rows:
        cv2.imwrite(str(out_path), np.vstack(rows))

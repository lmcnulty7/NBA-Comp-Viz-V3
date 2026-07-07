"""
detect/reid.py — Component B: appearance re-ID + offline track-fragment linking.

BoT-SORT is nearly perfect WITHIN a continuous camera shot; identity breaks at
occlusions (a track dies and the player re-appears as a new ID) and at every
camera cut / gate gap (the tracker is reset by design). Result: one player =
many track FRAGMENTS (the id-churn the diagnostics flag), and per-player stats
are impossible downstream.

This module links fragments OFFLINE, after the streaming pass, using two
independent signals:

  • Appearance — mean CLIP embedding of torso crops per fragment. We reuse the
    gate's frozen ViT-B/32 backbone (no new model, no training). Torso crop =
    inner upper box (jersey + number), avoiding floor/legs/other-player pixels.
    Caveat, by design: jersey color dominates CLIP crops, so TEAMMATES look
    alike — appearance alone must never decide a merge.

  • Motion feasibility — thanks to the court pipeline, fragment endpoints live
    in COURT FEET, which are comparable ACROSS camera shots (the payoff of the
    homography work). A link is only allowed if the implied speed between a's
    exit point and b's entry point is humanly possible (≤ ~30 ft/s + slack).

Candidate ranking (calibrated on the A/B window, DEVLOG 2026-07-05b): CLIP sims
cluster 0.89–0.98 across ALL candidate pairs — appearance can gate and tiebreak
but cannot rank. Court distance CAN (most fragments have one spatially clear
successor). So candidates are scored
    score = sim − REID_DIST_WEIGHT · dist / (max_speed·gap + slack)
(motion dominates, appearance breaks ties) and matching is deliberately
conservative: greedy on score, at most one successor and one predecessor per
fragment (fragments of one player form a temporal CHAIN), plus an ambiguity
margin on the score — when the top two candidates are within REID_AMBIG_MARGIN
(teammates converging), we refuse to merge. A false merge poisons two players'
trajectories; a missed merge just leaves an extra fragment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger(__name__)


# ── crop collection (streaming) ───────────────────────────────────────────────
class TorsoCropCollector:
    """Collect torso (jersey) crops per track during the streaming pass; embed
    them in one batched CLIP call at the end (embed())."""

    def __init__(self, every: int = None, max_per_track: int = None, min_px: int = 14):
        self.every = every or config.REID_CROP_EVERY
        self.cap = max_per_track or config.REID_MAX_CROPS
        self.min_px = min_px
        self.crops: dict[int, list[np.ndarray]] = {}
        self._n = 0

    def add(self, frame_bgr: np.ndarray, tracks) -> None:
        """Call once per PROCESSED frame (collects on every Nth call).

        Note (measured, 2026-07-06): occlusion-aware crop skipping (drop crops
        whose box overlaps another track's) was tried and REVERTED — in dense
        NBA play boxes overlap constantly, and starving tracks of crops cost
        more accuracy (track-level 87%→71%) than the contamination it removed."""
        self._n += 1
        if (self._n - 1) % self.every:
            return
        H, W = frame_bgr.shape[:2]
        for t in tracks:
            lst = self.crops.setdefault(t.track_id, [])
            if len(lst) >= self.cap:
                continue
            x1, y1, x2, y2 = t.bbox
            w, h = x2 - x1, y2 - y1
            # inner upper box: jersey + number, minimal background
            cx1, cx2 = int(max(0, x1 + 0.22 * w)), int(min(W, x2 - 0.22 * w))
            cy1, cy2 = int(max(0, y1 + 0.08 * h)), int(min(H, y1 + 0.55 * h))
            if cx2 - cx1 < self.min_px or cy2 - cy1 < self.min_px:
                continue
            lst.append(frame_bgr[cy1:cy2, cx1:cx2, ::-1].copy())   # BGR → RGB

    def embed(self, backbone) -> dict[int, np.ndarray]:
        """{track_id: (n, D) L2-normed PER-CROP embeddings}, only for tracks with
        ≥ REID_MIN_CROPS crops. Per-crop (not mean) because they have two consumers:
        re-ID links on the normed MEAN (mean_embeddings()); team classification
        k-means-VOTES over the individual crops (detect/teams.py)."""
        tids, flat = [], []
        for tid, lst in self.crops.items():
            if len(lst) >= config.REID_MIN_CROPS:
                tids.extend([tid] * len(lst))
                flat.extend(lst)
        if not flat:
            return {}
        embs = backbone.embed_images(flat, progress=False)
        arr_t = np.asarray(tids)
        return {tid: embs[arr_t == tid] for tid in set(tids)}


def mean_embeddings(crop_embs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-track L2-normed mean embedding — the re-ID linker's appearance feature."""
    out = {}
    for tid, m in crop_embs.items():
        v = m.mean(axis=0)
        out[tid] = (v / max(float(np.linalg.norm(v)), 1e-8)).astype(np.float32)
    return out


# ── fragments (offline) ───────────────────────────────────────────────────────
@dataclass
class Fragment:
    tid: int
    f0: int                     # first / last observed video frame
    f1: int
    t0: float                   # ... in seconds (real time, so gate gaps count)
    t1: float
    pos0: np.ndarray            # court-ft entry / exit points
    pos1: np.ndarray
    emb: np.ndarray | None      # mean torso embedding (None = too few crops)
    n_obs: int


def build_fragments(raw: dict, fps: float, embs: dict) -> list[Fragment]:
    """raw: {track_id: {frame_idx: (x_ft, y_ft)}} → time-sorted fragments."""
    frags = []
    for tid, d in raw.items():
        frames = sorted(d)
        if not frames:
            continue
        frags.append(Fragment(
            tid=tid, f0=frames[0], f1=frames[-1],
            t0=frames[0] / fps, t1=frames[-1] / fps,
            pos0=np.asarray(d[frames[0]], float), pos1=np.asarray(d[frames[-1]], float),
            emb=embs.get(tid), n_obs=len(frames)))
    return sorted(frags, key=lambda fr: fr.t0)


def link_fragments(frags: list[Fragment],
                   sim_min: float = None, max_gap_sec: float = None,
                   max_speed: float = None, slack_ft: float = None,
                   ambig_margin: float = None,
                   team_of: dict[int, int | None] = None):
    """Conservative fragment chaining.

    team_of (Component C1): hard veto — a candidate pair whose fragments belong
    to DIFFERENT teams is refused outright, before ambiguity is computed. This
    both blocks the worst merge error (cross-team) and shrinks candidate sets,
    so the ambiguity margin refuses less. Abstained tracks (None) are never vetoed.

    Returns (idmap, merges, skipped):
      idmap   {tid: canonical_tid} for every fragment (identity when unmerged)
      merges  accepted links, with the evidence (sim / gap / dist / implied speed)
      skipped candidates refused (reason: "team_veto" or "ambiguous") — the audit trail
    """
    sim_min = config.REID_SIM_MIN if sim_min is None else sim_min
    max_gap_sec = config.REID_MAX_GAP_SEC if max_gap_sec is None else max_gap_sec
    max_speed = config.REID_MAX_SPEED_FTS if max_speed is None else max_speed
    slack_ft = config.REID_DIST_SLACK_FT if slack_ft is None else slack_ft
    ambig_margin = config.REID_AMBIG_MARGIN if ambig_margin is None else ambig_margin

    # candidate pairs: temporal succession + appearance floor + motion feasibility.
    # score = sim − w·normalized-distance: motion ranks, appearance gates/tiebreaks.
    pairs = []   # (score, sim, gap, dist, a_tid, b_tid)
    vetoed = []
    for i, a in enumerate(frags):
        if a.emb is None:
            continue
        for b in frags[i + 1:]:
            gap = b.t0 - a.t1
            if gap <= 0:          # temporal overlap ⇒ two players on court simultaneously
                continue
            if gap > max_gap_sec:  # frags sorted by t0 ⇒ gap only grows from here
                break
            if b.emb is None:
                continue
            sim = float(a.emb @ b.emb)
            if sim < sim_min:
                continue
            reach = max_speed * gap + slack_ft
            dist = float(np.linalg.norm(a.pos1 - b.pos0))
            if dist > reach:
                continue
            if team_of is not None:
                ta, tb = team_of.get(a.tid), team_of.get(b.tid)
                if ta is not None and tb is not None and ta != tb:
                    # Continuity override (eval 2026-07-06): team assignment is
                    # ~87% accurate, so it must NOT outvote overwhelming
                    # continuity evidence — a near-identical appearance a few
                    # feet and a moment away is the same player regardless of
                    # what the (noisier) team signal says. The eval found the
                    # veto wrongly blocking sim 0.98 links at 1.4 ft / 1.6 s
                    # because one side's team was misassigned.
                    overwhelming = (sim >= config.REID_VETO_OVERRIDE_SIM
                                    and gap <= config.REID_VETO_OVERRIDE_GAP_S
                                    and dist <= config.REID_VETO_OVERRIDE_DIST_FT)
                    if not overwhelming:
                        vetoed.append({"a": a.tid, "b": b.tid, "sim": round(sim, 4),
                                       "gap_s": round(gap, 2), "dist_ft": round(dist, 1),
                                       "reason": "team_veto"})
                        continue
            score = sim - config.REID_DIST_WEIGHT * dist / reach
            pairs.append((score, sim, gap, dist, a.tid, b.tid))

    # ambiguity: if a fragment's best two candidates (either side) score too close,
    # refuse ALL its links — teammates converging is exactly this signature.
    def ambiguous(side_idx: int) -> set[int]:
        by: dict[int, list[float]] = {}
        for p in pairs:
            by.setdefault(p[side_idx], []).append(p[0])
        amb = set()
        for tid, scores in by.items():
            scores.sort(reverse=True)
            if len(scores) >= 2 and scores[0] - scores[1] < ambig_margin:
                amb.add(tid)
        return amb

    amb_a, amb_b = ambiguous(4), ambiguous(5)
    skipped = vetoed + [
        {"a": a, "b": b, "score": round(sc, 4), "sim": round(s, 4),
         "gap_s": round(g, 2), "dist_ft": round(d, 1), "reason": "ambiguous"}
        for sc, s, g, d, a, b in pairs if a in amb_a or b in amb_b]
    pairs = [p for p in pairs if p[4] not in amb_a and p[5] not in amb_b]

    # greedy: best similarity first; ≤1 successor and ≤1 predecessor per fragment
    parent = {fr.tid: fr.tid for fr in frags}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used_succ, used_pred, merges = set(), set(), []
    for score, sim, gap, dist, a, b in sorted(pairs, reverse=True):
        if a in used_succ or b in used_pred:
            continue
        used_succ.add(a)
        used_pred.add(b)
        parent[find(b)] = find(a)
        merges.append({"a": a, "b": b, "score": round(score, 4), "sim": round(sim, 4),
                       "gap_s": round(gap, 2), "dist_ft": round(dist, 1),
                       "speed_fts": round(dist / max(gap, 1e-6), 1)})

    # canonical id per chain = the tid of its earliest fragment
    t0_by_tid = {fr.tid: fr.t0 for fr in frags}
    roots: dict[int, int] = {}
    for fr in frags:
        r = find(fr.tid)
        if r not in roots or t0_by_tid[fr.tid] < t0_by_tid[roots[r]]:
            roots[r] = fr.tid
    idmap = {fr.tid: roots[find(fr.tid)] for fr in frags}

    if merges:
        log.info("re-ID: linked %d fragment pairs (%d refused as ambiguous)",
                 len(merges), len(skipped))
    return idmap, merges, skipped

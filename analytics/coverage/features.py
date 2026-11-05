"""analytics/coverage/features.py: the per-screen feature frame of the coverage classifier.

    python -m analytics.coverage.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]

Unit: one ball screen (hand-labelled or detected). The object being classified is the screen and the play right
after it, so the feature window is DECLARED as start_frame .. end_frame + 1.5 s (POST_FRAMES = 38 frames at
25 Hz); no frame beyond end_frame + POST_FRAMES enters (tested). Nothing from the label tables other than the
four player ids and the two frame bounds reaches a feature; the label (coverage), confidence and label_source
never do (FORBIDDEN, checked on every design matrix, plus analytics.schema.assert_time_forward on the frame).

Geometry. The screen point P is the screener's position at start_frame; u is the unit vector from P to the
basket the offence attacks (the basket nearer the handler at start_frame; screens happen in the half court);
v is perpendicular to u, oriented toward the nearer sideline (away from y = 25). Every position becomes
(depth, lat) = ((X - P) . u, (X - P) . v): depth grows toward the rim, lat toward the sideline. Roles: H the
ball handler, S the screener, D1 the on-ball defender, D2 the screener's defender. "end" is the last window
frame where H, D1 and D2 are all tracked; "early" is the first 0.6 s from start_frame; "late" the last 0.5 s
of the window. Speeds are displacement over 0.2 s divided by 0.2 s.

Detected screens carry no defender ids: D1 is the opponent nearest H at start_frame and D2 the opponent
nearest S other than D1 (assumption, recorded in reports/coverage.md). A screen whose handler has fewer than
two tracked window frames, or whose roles cannot all be placed at start_frame, is dropped and counted.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.coverage import CLASSES
from analytics.coverage.validate import add_input_args, load_inputs, season_of, validate_bundle

log = logging.getLogger("analytics.coverage")

HZ = 25
POST_S, EARLY_S, LATE_S = 1.5, 0.6, 0.5
POST_FRAMES = int(np.ceil(POST_S * HZ))        # 38
EARLY_FRAMES = int(np.ceil(EARLY_S * HZ))      # 15
LATE_FRAMES = int(np.ceil(LATE_S * HZ))        # 13
SPEED_LAG = 5                                  # 0.2 s
TRAP_FT = 6.0
BASKETS = ((5.25, 25.0), (88.75, 25.0))

FEATURES = (
    # at the screen frame (start_frame)
    "screen_basket_dist_ft", "handler_depth_t0_ft", "handler_lat_t0_ft", "d1_depth_t0_ft", "d1_lat_t0_ft",
    "d2_depth_t0_ft", "d2_lat_t0_ft", "d1_handler_dist_t0_ft", "d2_handler_dist_t0_ft", "d2_basket_dist_t0_ft",
    # after the screen (start_frame .. end_frame + 1.5 s)
    "d2_nearest_end", "def_dist_diff_end_ft", "d1_handler_dist_end_ft", "d2_handler_dist_end_ft",
    "d2_depth_end_ft", "d2_basket_dist_min_ft", "d2_handler_dist_min_early_ft", "d2_handler_dist_mean_early_ft",
    "d2_handler_dist_mean_post_ft", "d2_handler_dist_late_ft", "d2_handler_dist_min_post_ft",
    "d2_handler_dist_min_frac", "d2_handler_dist_after_min_ft", "d1_depth_rel_screener_pass_ft",
    "d1_lat_rel_handler_end_ft", "handler_lat_end_ft", "handler_depth_end_ft", "both_within_6ft_frac",
    "both_within_6ft_any", "d2_screen_dist_end_ft", "d2_screen_dist_max_ft", "handler_speed_mean_ftps",
    "handler_speed_max_ftps", "d1_speed_mean_ftps", "d2_speed_mean_ftps", "d1_handler_dist_mean_post_ft",
    "n_window_frames",
)
KEYS = ("sportvu_game", "possession", "action_id")
# Never in a design matrix: the label, its provenance and anything that names the outcome.
FORBIDDEN = frozenset({"coverage", "coverage_label", "confidence", "label_confidence", "label_source", "labeller_pass", "predicted",
                       "heldout", "fold"} | {"p_" + c for c in CLASSES} | set(schema.ALL_EXCLUDED))


def feature_columns() -> list:
    """The columns the models are fed, in FEATURES order."""
    return list(FEATURES)


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a forbidden column (label, provenance, fold, outcome)."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("analytics.coverage.features: forbidden columns in the design matrix: %s" % bad)


# ── per-game dense tracks ─────────────────────────────────────────────────────────────────────────
class GameTracks:
    """Dense (frame x player) position arrays of one game so a window is a slice.

    positions[k, j] is the (x, y) of player columns[j] at frame fmin + k, NaN where untracked."""

    def __init__(self, frames: pd.DataFrame):
        f = frames.drop_duplicates(["frame", "player_id"])
        self.fmin, self.fmax = int(f["frame"].min()), int(f["frame"].max())
        px = f.pivot(index="frame", columns="player_id", values="x").reindex(range(self.fmin, self.fmax + 1))
        py = f.pivot(index="frame", columns="player_id", values="y").reindex(range(self.fmin, self.fmax + 1))
        self.columns = {int(p): j for j, p in enumerate(px.columns)}
        self.xy = np.stack([px.to_numpy(dtype=float), py.to_numpy(dtype=float)], axis=-1)
        self.team = f.drop_duplicates("player_id").set_index("player_id")["team_id"].astype(int).to_dict()

    def window(self, pid: int, f0: int, f1: int) -> np.ndarray:
        """(f1 - f0 + 1, 2) positions of pid over frames f0..f1 (NaN outside the game or untracked)."""
        n = f1 - f0 + 1
        out = np.full((n, 2), np.nan)
        j = self.columns.get(int(pid))
        if j is None:
            return out
        lo, hi = max(f0, self.fmin), min(f1, self.fmax)
        if hi >= lo:
            out[lo - f0: hi - f0 + 1] = self.xy[lo - self.fmin: hi - self.fmin + 1, j]
        return out

    def at(self, f: int) -> dict:
        """{player_id: (x, y)} of every player tracked at frame f."""
        if f < self.fmin or f > self.fmax:
            return {}
        row = self.xy[f - self.fmin]
        return {p: row[j] for p, j in self.columns.items() if np.isfinite(row[j]).all()}



def infer_defenders(tracks: GameTracks, handler: int, screener: int, f0: int) -> tuple:
    """(on_ball_defender_id, screener_defender_id) at frame f0: the opponent nearest the handler and the
    opponent nearest the screener other than that one; (None, None) when fewer than two opponents are tracked."""
    at = tracks.at(f0)
    if handler not in at or screener not in at:
        return None, None
    off_team = tracks.team.get(int(handler))
    opp = {p: xy for p, xy in at.items() if tracks.team.get(p) != off_team}
    if len(opp) < 2:
        return None, None
    d1 = min(opp, key=lambda p: np.hypot(*(opp[p] - at[handler])))
    d2 = min((p for p in opp if p != d1), key=lambda p: np.hypot(*(opp[p] - at[screener])))
    return int(d1), int(d2)


# ── geometry ──────────────────────────────────────────────────────────────────────────────────────
def _axes(P: np.ndarray, basket: np.ndarray) -> tuple:
    u = basket - P
    nu = np.linalg.norm(u)
    u = u / nu if nu > 1e-9 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    side = np.sign(P[1] - 25.0)
    if side != 0 and v[1] * side < 0:
        v = -v
    return u, v


def _rel(X: np.ndarray, P: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple:
    d = X - P[None, :]
    return d @ u, d @ v


def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.hypot(A[:, 0] - B[:, 0], A[:, 1] - B[:, 1])


def _speed(X: np.ndarray) -> np.ndarray:
    if len(X) <= SPEED_LAG:
        return np.array([np.nan])
    return _dist(X[SPEED_LAG:], X[:-SPEED_LAG]) / (SPEED_LAG / HZ)


def _nanmean(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _nanmin(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmin(a)) if np.isfinite(a).any() else float("nan")


def _nanmax(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")


def screen_geometry(H: np.ndarray, S: np.ndarray, D1: np.ndarray, D2: np.ndarray, end_offset: int,
                    basket=None) -> dict | None:
    """Features of one screen from the four (n, 2) tracks over frames start_frame .. start_frame + n - 1, where
    end_offset = end_frame - start_frame. The tracks must already be cut to the declared window. None when the
    roles cannot be placed at the screen frame or the handler is tracked on fewer than two frames."""
    n = len(H)
    if n < 2 or not (np.isfinite(H[0]).all() and np.isfinite(S[0]).all() and np.isfinite(D1[0]).all() and np.isfinite(D2[0]).all()):
        return None
    if np.isfinite(H).all(axis=1).sum() < 2:
        return None
    if basket is None:
        basket = min(BASKETS, key=lambda b: np.hypot(H[0, 0] - b[0], H[0, 1] - b[1]))
    B = np.asarray(basket, dtype=float)
    P = S[0].copy()
    u, v = _axes(P, B)
    hd, hl = _rel(H, P, u, v)
    sd, sl = _rel(S, P, u, v)
    d1d, d1l = _rel(D1, P, u, v)
    d2d, d2l = _rel(D2, P, u, v)
    d1h, d2h = _dist(D1, H), _dist(D2, H)
    valid = np.isfinite(d1h) & np.isfinite(d2h)
    end = int(np.flatnonzero(valid)[-1]) if valid.any() else 0
    early = slice(0, min(n, EARLY_FRAMES))        # 15 frames = 0.6 s
    late = slice(max(0, n - LATE_FRAMES), n)
    d1s = _dist(D1, S)
    pas = int(np.nanargmin(d1s)) if np.isfinite(d1s).any() else 0
    both = (d1h <= TRAP_FT) & (d2h <= TRAP_FT)
    imin = int(np.nanargmin(d2h)) if np.isfinite(d2h).any() else 0
    d2b = _dist(D2, np.repeat(B[None, :], n, axis=0))
    d2p = _dist(D2, np.repeat(P[None, :], n, axis=0))
    out = {
        "screen_basket_dist_ft": float(np.linalg.norm(B - P)),
        "handler_depth_t0_ft": float(hd[0]), "handler_lat_t0_ft": float(hl[0]),
        "d1_depth_t0_ft": float(d1d[0]), "d1_lat_t0_ft": float(d1l[0]),
        "d2_depth_t0_ft": float(d2d[0]), "d2_lat_t0_ft": float(d2l[0]),
        "d1_handler_dist_t0_ft": float(d1h[0]), "d2_handler_dist_t0_ft": float(d2h[0]),
        "d2_basket_dist_t0_ft": float(d2b[0]),
        "d2_nearest_end": float(d2h[end] < d1h[end]) if valid[end] else float("nan"),
        "def_dist_diff_end_ft": float(d1h[end] - d2h[end]),
        "d1_handler_dist_end_ft": float(d1h[end]), "d2_handler_dist_end_ft": float(d2h[end]),
        "d2_depth_end_ft": float(d2d[end]),
        "d2_basket_dist_min_ft": _nanmin(d2b),
        "d2_handler_dist_min_early_ft": _nanmin(d2h[early]), "d2_handler_dist_mean_early_ft": _nanmean(d2h[early]),
        "d2_handler_dist_mean_post_ft": _nanmean(d2h), "d2_handler_dist_late_ft": _nanmean(d2h[late]),
        "d2_handler_dist_min_post_ft": _nanmin(d2h),
        "d2_handler_dist_min_frac": float(imin / max(1, n - 1)),
        "d2_handler_dist_after_min_ft": _nanmax(d2h[imin:]),
        "d1_depth_rel_screener_pass_ft": float(d1d[pas] - sd[pas]),
        "d1_lat_rel_handler_end_ft": float(d1l[end] - hl[end]),
        "handler_lat_end_ft": float(hl[end]), "handler_depth_end_ft": float(hd[end]),
        "both_within_6ft_frac": float(both[valid].mean()) if valid.any() else float("nan"),
        "both_within_6ft_any": float(both.any()),
        "d2_screen_dist_end_ft": float(d2p[end]), "d2_screen_dist_max_ft": _nanmax(d2p),
        "handler_speed_mean_ftps": _nanmean(_speed(H)), "handler_speed_max_ftps": _nanmax(_speed(H)),
        "d1_speed_mean_ftps": _nanmean(_speed(D1)), "d2_speed_mean_ftps": _nanmean(_speed(D2)),
        "d1_handler_dist_mean_post_ft": _nanmean(d1h),
        "n_window_frames": float(np.isfinite(H).all(axis=1).sum()),
    }
    out["_end_offset"] = end_offset
    return out


# ── the feature frame ─────────────────────────────────────────────────────────────────────────────
def features(candidates: pd.DataFrame, games: dict, counts: dict | None = None) -> pd.DataFrame:
    """Candidate table (validate.CANDIDATE_COLUMNS) + {game: dict(frames, ...)} -> the feature frame: KEYS,
    screen_frame, screen_end_frame, heldout, label_source, coverage_label (null when unlabelled), label_confidence
    (sure, unsure or null; bookkeeping for the trainer, never a feature), then FEATURES.
    Calls schema.assert_time_forward on the result. Dropped screens are counted in ``counts``."""
    counts = counts if counts is not None else {}
    rows = []
    dropped = {"no_frames": 0, "roles_untracked": 0, "defenders_not_inferable": 0}
    for gid, cand in candidates.groupby("sportvu_game", sort=True):
        game = games.get(gid)
        if game is None:
            dropped["no_frames"] += len(cand)
            continue
        tracks = GameTracks(game["frames"])
        for r in cand.itertuples(index=False):
            f0, f1 = int(r.start_frame), int(r.end_frame) + POST_FRAMES
            d1, d2 = r.on_ball_defender_id, r.screener_defender_id
            if pd.isna(d1) or pd.isna(d2):
                d1, d2 = infer_defenders(tracks, int(r.ball_handler_id), int(r.screener_id), int(r.start_frame))
                if d1 is None:
                    dropped["defenders_not_inferable"] += 1
                    continue
            H, S = tracks.window(int(r.ball_handler_id), f0, f1), tracks.window(int(r.screener_id), f0, f1)
            D1, D2 = tracks.window(int(d1), f0, f1), tracks.window(int(d2), f0, f1)
            geo = screen_geometry(H, S, D1, D2, int(r.end_frame) - int(r.start_frame))
            if geo is None:
                dropped["roles_untracked"] += 1
                continue
            geo.pop("_end_offset")
            row = {"sportvu_game": gid, "possession": int(r.possession), "action_id": int(r.action_id),
                   "screen_frame": int(r.start_frame), "screen_end_frame": int(r.end_frame), "heldout": bool(r.heldout),
                   "label_source": str(r.label_source), "coverage_label": None if pd.isna(r.coverage) else str(r.coverage),
                   "label_confidence": None if pd.isna(r.confidence) else str(r.confidence),
                   "on_ball_defender_id": int(d1), "screener_defender_id": int(d2)}
            row.update(geo)
            rows.append(row)
    cols = list(KEYS) + ["screen_frame", "screen_end_frame", "heldout", "label_source", "coverage_label", "label_confidence",
                         "on_ball_defender_id", "screener_defender_id"] + list(FEATURES)
    out = pd.DataFrame(rows, columns=cols)
    for k, v in dropped.items():
        counts["features_dropped_" + k] = int(v)
    counts["features_rows"] = int(len(out))
    counts["features_labelled"] = int(out["coverage_label"].notna().sum())
    for k, v in dropped.items():
        if v:
            log.info("features: dropped %d screen(s): %s", v, k)
    schema.assert_time_forward(out, "states_1hz")
    assert_no_leak(out[feature_columns()])
    return out


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """The design matrix (FEATURES as float) after the leakage check."""
    X = frame[feature_columns()].astype(float)
    assert_no_leak(X)
    return X


def write_features(feats: pd.DataFrame, out_root) -> list:
    """OUT_ROOT/analytics/coverage/<season>/coverage_features.parquet per season (held-out under heldout/<season>/)."""
    paths = []
    for (season, held), part in feats.groupby([feats["sportvu_game"].map(season_of), feats["heldout"]]):
        d = Path(out_root) / "analytics" / "coverage" / ("heldout" if held else "") / season
        d.mkdir(parents=True, exist_ok=True)
        p = d / "coverage_features.parquet"
        part.to_parquet(p, index=False)
        paths.append(p)
    return paths


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_input_args(ap)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stderr)
    bundle = load_inputs(a)
    cand, counts = validate_bundle(bundle)
    feats = features(cand, bundle["games"], counts)
    out_root = a.out or tempfile.mkdtemp(prefix="coverage_features_")
    paths = write_features(feats, out_root)
    print("feature frame: %d rows (%d labelled), %d features, written to %s" % (
        len(feats), counts["features_labelled"], len(FEATURES), ", ".join(map(str, paths)) or "nothing"))
    if a.fixtures:
        print("fixture output, not a result")
    return feats


if __name__ == "__main__":
    main()

"""analytics/epv/features.py: the time-forward EPV feature frame from states_1hz (docs/DATA_CONTRACT.md section 20).

    python -m analytics.epv.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--fixture-games 4]
                                     [--fixture-possessions 40] [--fixture-heldout-every 4]

The only source of columns is ``analytics.schema.feature_allowlist("states_1hz")``: ``build_features`` reads nothing
else from the states table, the targets (points_to_end, possession_end_kind) never enter, and the returned frame
passes ``analytics.schema.assert_time_forward`` before it leaves this module. Every feature of a state at time t is
computed from that state or from earlier states of the same possession (the expanding quantities), never from a
later frame.

FEATURES (one row per state):
  ball_x, ball_y, ball_z, ball_dist_basket   ball position and its floor distance to the attacked basket
  handler_dist_basket, handler_angle         the handler's distance to the basket and angle from the rim-to-rim axis
  handler_nearest_def_dist_ft                as in the table (nearest defender to the handler)
  off_dist_1 .. off_dist_5                   the five offensive players' distances to the basket, sorted ascending
  off_def_1 .. off_def_5                     their distances to their nearest defender, sorted ascending
  off_hull_area                              convex-hull area of the five offensive players, square feet
  shot_clock_s, elapsed_in_possession_s      from the table (shot clock null when the source has none)
  ball_dist_min_so_far                       the smallest ball distance to the basket over this possession up to t
  n_handler_switches                         handler changes over this possession up to t
  handler_fg_prior                           the handler's smoothed make probability in the zone of the handler's
                                             current distance (shots of the training games only, out-of-fold: a
                                             training game's own shots are left out of its zone counts), shrunk with
                                             K_SHOOTER pseudo-attempts towards the location-only xFG at the
                                             handler's position (analytics.features_xfg, a four-parameter league
                                             model fitted once on the whole training fold, the row's own game
                                             included; it is not left out per game)
  handler_zone_value                         2 or 3: the value of a shot from the handler's distance (>= 22 ft)
  lineup_tov_prior                           the offensive lineup's smoothed turnover rate per possession from the
                                             training games (out-of-fold the same way, K_LINEUP pseudo-possessions
                                             towards the training mean)

Attacked basket (the contract does not carry it): the basket nearer the expanding mean of the offensive centroid
over the possession's states up to t, so a possession that starts in the back court reads the wrong basket for its
first second or two and corrects itself; this uses no later frame. Zones by distance: rim < 4 ft, paint 4 to 14,
mid 14 to 22, three >= 22. ``lagged`` builds the last-N-states view the gated sequence model consumes.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema

BASKETS = np.array([[5.25, 25.0], [88.75, 25.0]])
HALF_X = 47.0
THREE_FT = 22.0
ZONE_EDGES = (4.0, 14.0, 22.0)
ZONES = ("rim", "paint", "mid", "three")
K_SHOOTER = 20.0
K_LINEUP = 30.0
KEYS = ["sportvu_game", "possession", "frame"]
GEOMETRY = (["ball_x", "ball_y", "ball_z", "ball_dist_basket", "handler_dist_basket", "handler_angle", "handler_nearest_def_dist_ft"]
            + ["off_dist_%d" % i for i in range(1, 6)] + ["off_def_%d" % i for i in range(1, 6)]
            + ["off_hull_area", "shot_clock_s", "elapsed_in_possession_s", "ball_dist_min_so_far", "n_handler_switches"])
PRIORS = ["handler_fg_prior", "handler_zone_value", "lineup_tov_prior"]
FEATURES = GEOMETRY + PRIORS
ALLOWED = schema.feature_allowlist("states_1hz")


def feature_columns() -> list:
    """The columns the models are fed, in order."""
    return list(FEATURES)


def zone_of(dist) -> np.ndarray:
    """The shot zone (rim, paint, mid, three) of each distance in feet (ZONE_EDGES)."""
    d = np.asarray(dist, dtype=float)
    return np.asarray(ZONES, dtype=object)[np.searchsorted(np.asarray(ZONE_EDGES), d, side="right")]


def _hull_area(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Convex-hull area per row of five points (scipy ConvexHull; 0 for collinear or degenerate sets, NaN with a missing point)."""
    from scipy.spatial import ConvexHull, QhullError
    out = np.zeros(len(px))
    pts = np.stack([px, py], axis=2)
    for i in range(len(px)):
        p = pts[i]
        if not np.isfinite(p).all():
            out[i] = np.nan
            continue
        try:
            out[i] = ConvexHull(p).volume
        except QhullError:
            out[i] = 0.0
    return out


def basket_side(states_allowed: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """Index (0 left, 1 right) of the attacked basket per state: the basket nearer the expanding mean of the
    offensive centroid over the possession's states up to and including t (sorted by elapsed time)."""
    cx = np.mean([states_allowed["off%d_x" % i].to_numpy(dtype=float) for i in range(1, 6)], axis=0)
    d = pd.DataFrame({"g": keys["sportvu_game"].astype(str).to_numpy(), "p": keys["possession"].to_numpy(),
                      "t": states_allowed["elapsed_in_possession_s"].to_numpy(dtype=float), "cx": cx})
    order = np.lexsort((d["t"].to_numpy(), d["p"].to_numpy(), d["g"].to_numpy()))
    ds = d.iloc[order]
    run = ds.groupby(["g", "p"], sort=False)["cx"].expanding().mean().to_numpy()
    side = np.empty(len(d), dtype=int)
    side[order] = (run >= HALF_X).astype(int)
    return side


def geometry(states: pd.DataFrame) -> pd.DataFrame:
    """KEYS plus the geometric features (no priors) plus the underscore auxiliary columns the priors read.
    Reads only the allow-listed columns of states."""
    missing = [c for c in ALLOWED if c not in states.columns]
    if missing:
        raise ValueError("features: states table lacks allow-listed column(s) %s" % missing)
    keys = states[[c for c in KEYS if c in states.columns]].reset_index(drop=True)
    src = states[ALLOWED].reset_index(drop=True)          # the only source of feature columns
    n = len(src)
    side = basket_side(src, keys)
    bk = BASKETS[side]
    bx, by, bz = (src[c].to_numpy(dtype=float) for c in ("ball_x", "ball_y", "ball_z"))
    ox = np.stack([src["off%d_x" % i].to_numpy(dtype=float) for i in range(1, 6)], axis=1)
    oy = np.stack([src["off%d_y" % i].to_numpy(dtype=float) for i in range(1, 6)], axis=1)
    dx = np.stack([src["def%d_x" % i].to_numpy(dtype=float) for i in range(1, 6)], axis=1)
    dy = np.stack([src["def%d_y" % i].to_numpy(dtype=float) for i in range(1, 6)], axis=1)
    oid = np.stack([src["off%d_id" % i].to_numpy(dtype=float) for i in range(1, 6)], axis=1)
    hid = pd.to_numeric(src["handler_id"], errors="coerce").to_numpy(dtype=float)
    h_idx = np.argmax(oid == hid[:, None], axis=1)
    has_h = np.isfinite(hid) & (oid == hid[:, None]).any(axis=1)
    hx = np.where(has_h, ox[np.arange(n), h_idx], np.nan)
    hy = np.where(has_h, oy[np.arange(n), h_idx], np.nan)
    toward = np.where(side == 0, 1.0, -1.0)                 # the direction from the basket into the court
    out = keys.copy()
    out["ball_x"], out["ball_y"], out["ball_z"] = bx, by, bz
    out["ball_dist_basket"] = np.hypot(bx - bk[:, 0], by - bk[:, 1])
    out["handler_dist_basket"] = np.hypot(hx - bk[:, 0], hy - bk[:, 1])
    out["handler_angle"] = np.arctan2(hy - bk[:, 1], (hx - bk[:, 0]) * toward)
    out["handler_nearest_def_dist_ft"] = src["handler_nearest_def_dist_ft"].to_numpy(dtype=float)
    od = np.hypot(ox - bk[:, [0]], oy - bk[:, [1]])
    od_sorted = np.sort(od, axis=1)
    for i in range(5):
        out["off_dist_%d" % (i + 1)] = od_sorted[:, i]
    dd = np.sqrt((ox[:, :, None] - dx[:, None, :]) ** 2 + (oy[:, :, None] - dy[:, None, :]) ** 2).min(axis=2)
    dd_sorted = np.sort(dd, axis=1)
    for i in range(5):
        out["off_def_%d" % (i + 1)] = dd_sorted[:, i]
    out["off_hull_area"] = _hull_area(ox, oy)
    out["shot_clock_s"] = src["shot_clock_s"].to_numpy(dtype=float)
    out["elapsed_in_possession_s"] = src["elapsed_in_possession_s"].to_numpy(dtype=float)
    # the possession's past: expanding quantities over states sorted by elapsed time
    t = out["elapsed_in_possession_s"].to_numpy()
    order = np.lexsort((t, keys["possession"].to_numpy(), keys["sportvu_game"].astype(str).to_numpy()))
    grp = pd.DataFrame({"g": keys["sportvu_game"].astype(str).to_numpy()[order], "p": keys["possession"].to_numpy()[order]})
    bd = pd.Series(out["ball_dist_basket"].to_numpy()[order])
    mn = bd.groupby([grp["g"].to_numpy(), grp["p"].to_numpy()], sort=False).cummin().to_numpy()
    hs = pd.Series(hid[order])
    prev = hs.groupby([grp["g"].to_numpy(), grp["p"].to_numpy()], sort=False).shift(1)
    switch = (prev.notna() & hs.notna() & (prev != hs)).astype(float)
    nsw = switch.groupby([grp["g"].to_numpy(), grp["p"].to_numpy()], sort=False).cumsum().to_numpy()
    tmp = np.empty(n)
    tmp[order] = mn
    out["ball_dist_min_so_far"] = tmp
    tmp2 = np.empty(n)
    tmp2[order] = nsw
    out["n_handler_switches"] = tmp2
    # auxiliary columns the priors read (dropped by build_features): underscore-prefixed, never features
    out["_side"], out["_hx"], out["_hy"], out["_handler_id"] = side, hx, hy, hid
    out["_lineup"] = ["-".join(str(int(v)) for v in row) for row in oid]
    return out


# ── priors (training folds only) ───────────────────────────────────────────────────────────────────
def shooter_id(shots: pd.DataFrame, rosters: pd.DataFrame | None) -> pd.Series:
    """The shooter's player id: shots.player_id when given (section 10), else rosters.bref_slug matched on
    (sportvu_game, player_slug) (section 10a), else null."""
    pid = pd.to_numeric(shots["player_id"], errors="coerce") if "player_id" in shots.columns else pd.Series(np.nan, index=shots.index)
    if pid.isna().any() and rosters is not None and "player_slug" in shots.columns and "bref_slug" in rosters.columns:
        m = rosters.dropna(subset=["bref_slug"]).drop_duplicates(["sportvu_game", "bref_slug"])
        lut = dict(zip(zip(m["sportvu_game"].astype(str), m["bref_slug"].astype(str)), m["player_id"].astype(int)))
        fill = [lut.get((g, s), np.nan) for g, s in zip(shots["sportvu_game"].astype(str), shots["player_slug"].astype(str))]
        pid = pid.where(pid.notna(), pd.Series(fill, index=shots.index, dtype=float))
    return pid


def shot_distance(shots: pd.DataFrame) -> np.ndarray:
    """Shot distance in feet: shot_distance_ft where given, else the length of (loc_x, loc_y) in tenths of a foot."""
    d = np.hypot(shots["loc_x"].to_numpy(dtype=float), shots["loc_y"].to_numpy(dtype=float)) / 10.0
    if "shot_distance_ft" in shots.columns:
        s = pd.to_numeric(shots["shot_distance_ft"], errors="coerce").to_numpy(dtype=float)
        d = np.where(np.isfinite(s), s, d)
    return d


def shooter_stats(shots: pd.DataFrame, rosters: pd.DataFrame | None) -> pd.DataFrame:
    """Per (sportvu_game, player_id, zone): attempts and makes. Per-game sufficient statistics, so a prior for any
    set of games is a sum and a game can be left out exactly."""
    if shots is None or len(shots) == 0:
        return pd.DataFrame(columns=["sportvu_game", "player_id", "zone", "attempts", "makes"])
    pid = shooter_id(shots, rosters)
    d = pd.DataFrame({"sportvu_game": shots["sportvu_game"].astype(str).to_numpy(), "player_id": pid.to_numpy(),
                      "zone": zone_of(shot_distance(shots)), "made": shots["made"].astype(bool).to_numpy()})
    d = d.dropna(subset=["player_id"])
    d["player_id"] = d["player_id"].astype(int)
    g = d.groupby(["sportvu_game", "player_id", "zone"], as_index=False).agg(attempts=("made", "size"), makes=("made", "sum"))
    return g


def lineup_stats(states: pd.DataFrame) -> pd.DataFrame:
    """Per (sportvu_game, lineup): possessions and turnovers, the lineup being the five offensive ids at the first
    state of each possession. Uses possession_end_kind, a target, so it may only ever see training games."""
    first = states.sort_values("elapsed_in_possession_s", kind="stable").drop_duplicates(["sportvu_game", "possession"])
    key = ["-".join(str(int(v)) for v in row) for row in first[["off%d_id" % i for i in range(1, 6)]].to_numpy()]
    d = pd.DataFrame({"sportvu_game": first["sportvu_game"].astype(str).to_numpy(), "lineup": key,
                      "tov": (first["possession_end_kind"].astype(str) == "turnover").to_numpy()})
    return d.groupby(["sportvu_game", "lineup"], as_index=False).agg(possessions=("tov", "size"), turnovers=("tov", "sum"))


def fit_xfg(shots_train: pd.DataFrame | None):
    """The location-only xFG (analytics.features_xfg + analytics.train_xfg.fit_models) on the training shots ->
    p(loc_x, loc_y) in tenths of a foot from the basket. A constant when there are too few shots. Fitted once per
    training fold on every fold-training shot (a league-wide four-parameter model), so a training row's own game is
    inside it; only the per-player zone counts are left out per game."""
    if shots_train is None or len(shots_train) < 30 or shots_train["made"].nunique() < 2:
        rate = float(shots_train["made"].mean()) if shots_train is not None and len(shots_train) else 0.45
        return lambda loc_x, loc_y, r=rate: np.full(len(loc_x), r)
    from analytics import features_xfg as fx
    from analytics import train_xfg as tx
    feats = fx.features(shots_train)          # its heldout column feeds train_xfg's own tripwire (a held-out row raises)
    model = tx.fit_models(feats)["logistic"]

    def predict(loc_x, loc_y, m=model):
        """The league xFG at shot locations given in tenths of a foot from the basket."""
        q = pd.DataFrame({"sportvu_game": "query", "made": False, "loc_x": np.asarray(loc_x, dtype=float),
                          "loc_y": np.asarray(loc_y, dtype=float), "period": 1, "clock_s": 0.0})
        return np.asarray(m(fx.features(q)), dtype=float)
    return predict


class Priors:
    """Priors from a set of training games: shooter zone stats, lineup turnover stats, the xFG, the training
    turnover rate. ``apply`` leaves a game's own statistics out when that game is among the training games."""

    def __init__(self, shooter: pd.DataFrame, lineup: pd.DataFrame, xfg, tov_rate: float, train_games: set):
        self.shooter, self.lineup, self.xfg, self.tov_rate, self.train_games = shooter, lineup, xfg, float(tov_rate), set(train_games)
        self._sh_tot = shooter.groupby(["player_id", "zone"])[["attempts", "makes"]].sum() if len(shooter) else None
        self._ln_tot = lineup.groupby("lineup")[["possessions", "turnovers"]].sum() if len(lineup) else None

    @classmethod
    def from_training(cls, states_train: pd.DataFrame, shots_train: pd.DataFrame | None, rosters: pd.DataFrame | None) -> "Priors":
        """Priors from the training states, the shots of those games and the rosters (shooter ids)."""
        games = set(states_train["sportvu_game"].astype(str))
        sh = shooter_stats(shots_train[shots_train["sportvu_game"].astype(str).isin(games)], rosters) if shots_train is not None and len(shots_train) else shooter_stats(None, rosters)
        ln = lineup_stats(states_train)
        tov = float(ln["turnovers"].sum() / max(ln["possessions"].sum(), 1)) if len(ln) else 0.13
        xfg = fit_xfg(shots_train[shots_train["sportvu_game"].astype(str).isin(games)] if shots_train is not None and len(shots_train) else None)
        return cls(sh, ln, xfg, tov, games)

    def _lookup(self, tot, per_game, keys, key_cols, val_cols, games):
        """Sum over training games minus the row's own game when that game is a training game."""
        n = len(games)
        out = np.zeros((n, len(val_cols)))
        if tot is None:                      # no statistics at all (no shots table): the prior is the shrinkage target
            return out
        idx = pd.MultiIndex.from_arrays([keys[c] for c in key_cols]) if len(key_cols) > 1 else pd.Index(keys[key_cols[0]])
        got = tot.reindex(idx)[val_cols].to_numpy(dtype=float)
        out = np.nan_to_num(got)
        own = per_game.set_index(["sportvu_game"] + key_cols)[val_cols]
        own_idx = pd.MultiIndex.from_arrays([games] + [keys[c] for c in key_cols])
        sub = np.nan_to_num(own.reindex(own_idx).to_numpy(dtype=float))
        is_train = np.isin(games, list(self.train_games))
        out = out - sub * is_train[:, None]
        return np.clip(out, 0, None)

    def apply(self, geo: pd.DataFrame) -> pd.DataFrame:
        """The three prior columns for a geometry frame (reads its auxiliary _handler_id, _hx, _hy, _side, _lineup)."""
        games = geo["sportvu_game"].astype(str).to_numpy()
        hid = geo["_handler_id"].to_numpy(dtype=float)
        hx, hy = geo["_hx"].to_numpy(dtype=float), geo["_hy"].to_numpy(dtype=float)
        side = geo["_side"].to_numpy(dtype=int)
        bk = BASKETS[side]
        dist = geo["handler_dist_basket"].to_numpy(dtype=float)
        zone = zone_of(np.nan_to_num(dist, nan=15.0))
        toward = np.where(side == 0, 1.0, -1.0)
        loc_y = np.nan_to_num((hx - bk[:, 0]) * toward * 10.0, nan=150.0)
        loc_x = np.nan_to_num((hy - bk[:, 1]) * 10.0, nan=0.0)
        p_loc = np.clip(self.xfg(loc_x, loc_y), 0.01, 0.99)
        keys = {"player_id": np.nan_to_num(hid, nan=-1).astype(int), "zone": zone}
        sh = self._lookup(self._sh_tot, self.shooter, keys, ["player_id", "zone"], ["attempts", "makes"], games)
        fg = (sh[:, 1] + K_SHOOTER * p_loc) / (sh[:, 0] + K_SHOOTER)
        fg = np.where(np.isfinite(hid), fg, p_loc)
        ln = self._lookup(self._ln_tot, self.lineup, {"lineup": geo["_lineup"].to_numpy(dtype=object)}, ["lineup"],
                          ["possessions", "turnovers"], games)
        tov = (ln[:, 1] + K_LINEUP * self.tov_rate) / (ln[:, 0] + K_LINEUP)
        return pd.DataFrame({"handler_fg_prior": fg, "handler_zone_value": np.where(np.nan_to_num(dist, nan=0.0) >= THREE_FT, 3.0, 2.0),
                             "lineup_tov_prior": tov}, index=geo.index)


def build_features(states: pd.DataFrame, priors: Priors | None = None, geo: pd.DataFrame | None = None) -> pd.DataFrame:
    """KEYS + FEATURES for every state (the prior columns NaN without priors). Passes assert_time_forward."""
    g = geometry(states) if geo is None else geo
    out = g.copy()
    if priors is not None:
        pr = priors.apply(g)
        for c in PRIORS:
            out[c] = pr[c].to_numpy()
    else:
        for c in PRIORS:
            out[c] = np.nan
    out = out[[c for c in KEYS if c in out.columns] + FEATURES].copy()
    schema.assert_time_forward(out, "states_1hz")
    return out


def lagged(feats: pd.DataFrame, n_lags: int = 5, columns: list | None = None) -> pd.DataFrame:
    """The last n_lags states' features per row (lag 0 is the row itself): columns <c>_lag<k> for k in 1..n_lags-1,
    taken from earlier states of the same possession and padded with the row's own values at the start of the
    possession. Only earlier states are read, so the view stays time-forward."""
    cols = list(columns or FEATURES)
    t = feats["elapsed_in_possession_s"].to_numpy(dtype=float)
    order = np.lexsort((t, feats["possession"].to_numpy(), feats["sportvu_game"].astype(str).to_numpy()))
    s = feats.iloc[order]
    grp = [s["sportvu_game"].astype(str).to_numpy(), s["possession"].to_numpy()]
    extra = {}
    for k in range(1, n_lags):
        sh = s[cols].groupby(grp, sort=False).shift(k)
        sh = sh.where(sh.notna(), s[cols])
        for c in cols:
            tmp = np.empty(len(feats))
            tmp[order] = sh[c].to_numpy(dtype=float)
            extra["%s_lag%d" % (c, k)] = tmp
    out = pd.concat([feats, pd.DataFrame(extra, index=feats.index)], axis=1)
    schema.assert_time_forward(out, "states_1hz")
    return out


def lagged_columns(n_lags: int = 5, columns: list | None = None) -> list:
    """The column names lagged() adds, after the base columns, in the order the sequence model is fed."""
    cols = list(columns or FEATURES)
    return cols + ["%s_lag%d" % (c, k) for k in range(1, n_lags) for c in cols]


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The features CLI flags."""
    from analytics.epv.validate import add_fixture_args
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="the synthetic season of analytics.fixtures (fixture output)")
    g.add_argument("--data", help="DATA_ROOT (see analytics.epv.validate)")
    ap.add_argument("--out", help="OUT_ROOT: analytics/epv_features/<season>/<game>.parquet (held-out under heldout/)")
    ap.add_argument("--season", default=None, help="only this season directory, e.g. 2015-16 (default: every season)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> list:
    """Writes the geometric features (priors are fold-dependent, so they are NaN here and filled by the trainer)."""
    from analytics.epv import validate as V
    a = parse_args(argv)
    clean, _reports = V.validate_all(V.load_inputs(a))
    st = clean["states_1hz"]
    if st is None or len(st) == 0:
        raise SystemExit("features: no states_1hz rows")
    feats = build_features(st)
    out_root = Path(a.out) if a.out else (Path(tempfile.gettempdir()) / "epv_fixture_root" if a.fixtures else Path(a.data))
    written = []
    held = st.set_index(KEYS)["heldout"] if "heldout" in st.columns else None
    for game, part in feats.groupby("sportvu_game", sort=True):
        h = bool(held.loc[game].iloc[0]) if held is not None else False
        path = out_root / "analytics" / "epv_features" / ("heldout" if h else "") / schema.season_of_game(game) / ("%s.parquet" % game)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        written.append(path)
        print("%s: %d states, %d features%s" % (game, len(part), len(FEATURES), " (held-out)" if h else ""))
    print("%s%d game(s) written under %s" % ("fixture output, not a result: " if a.fixtures else "", len(written), out_root))
    return written


if __name__ == "__main__":
    main()

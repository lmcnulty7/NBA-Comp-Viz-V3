"""analytics/matchups/features.py: defender to offensive player assignment at 1 Hz and the matchup_possessions table.

    python -m analytics.matchups.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

Assignment (positions only, nothing about the outcome enters). Inside each possession of the section 19 table
the section 4 frames are sampled at 1 Hz: for every whole second k of elapsed time (start_clock_s minus the
frame's game_clock_s) the frame whose elapsed time is nearest k. At each sampled frame the defenders (rows of
the team that is not offense_team_id) are assigned to the offensive players by scipy.optimize.linear_sum_assignment
on the cost

    cost[d, o] = dist(defender d, offensive player o) + 0.5 * dist(defender d, midpoint(o, basket))

where the basket is the one nearer the centroid of the offensive players over the possession's sampled
frames (the offence plays near the basket it attacks; the choice uses positions only). A defender's share of a
possession against an offensive player (partial_possession) is the fraction of the possession's sampled
frames at which it was assigned to that player; switches is the number of sampled frames (after the first) at
which the defender's assignment differed from the previous sampled frame, counted once per defender and
repeated on each of that defender's rows in the possession.

Targets (never features; the trainer drops them before schema.assert_time_forward):
    points_allowed_share   the possession's points times the share, on the rows of the offensive player who
                           took the possession's last field-goal attempt; 0 on every other row and on
                           possessions without a field-goal attempt (turnovers, free throws without a shot)
    xfg_expected_share     sum over that offensive player's field-goal attempts in the possession of
                           p(make) times the shot value (2 or 3) times the share; p comes from the xFG lane's
                           out-of-fold p_selected when given, else from the location-only fallback the
                           trainer builds with analytics.features_xfg

Shot value: 3 when the play-by-play description says 3-pt or 3PT, else 3 when the shooter is 22 ft or more
from the attacked basket at the release frame, else 2. A shot joins a possession by its play-by-play event
index lying in [start_event, end_event] of the same period.

Assumptions taken without asking: a frame with no defender or no offensive player rows is skipped (the
possession's share denominator counts only frames where an assignment was made); a possession with a null
offense_team_id is skipped and counted; a shot whose shooter_id is null attributes points to nobody; with
unequal squad sizes (tracking dropout) the rectangular assignment leaves the extra players unassigned.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from analytics import schema

log = logging.getLogger("analytics.matchups.features")

BASKETS = ((5.25, 25.0), (88.75, 25.0))
MIDPOINT_WEIGHT = 0.5
THREE_FT = 22.0
DEFENDED_FT = 4.0
ID_COLUMNS = ["sportvu_game", "possession", "def_player_id", "off_player_id"]
FEATURES = ["partial_possession", "switches"]
TARGETS = ["points_allowed_share", "xfg_expected_share"]
MATCHUP_COLUMNS = ID_COLUMNS + ["partial_possession", "points_allowed_share", "xfg_expected_share", "switches"]
FRAME_COLUMNS = ["sportvu_game", "possession", "k", "frame", "def_player_id", "off_player_id", "dist_ft", "changed"]
SHOT_COLUMNS = ["sportvu_game", "possession", "event", "shooter_id", "made", "value", "p", "release_frame", "k",
                "assigned_def_id", "def_dist_ft", "shooter_x", "shooter_y", "basket_x", "basket_y"]


def feature_columns() -> list:
    """The feature columns of the matchup_possessions table (ids and targets excluded)."""
    return list(FEATURES)


def feature_frame(mp: pd.DataFrame) -> pd.DataFrame:
    """Ids plus FEATURES of a matchup_possessions table, checked by schema.assert_time_forward (on the input
    too: a table that carries a contract target or post-outcome column is refused, not silently trimmed)."""
    schema.assert_time_forward(mp, "possessions")
    out = mp[ID_COLUMNS + FEATURES].copy()
    leaked = [c for c in out.columns if c in TARGETS]
    if leaked:
        raise ValueError("matchups: target column(s) %s in the feature frame" % leaked)
    schema.assert_time_forward(out, "possessions")
    return out


# ── assignment ─────────────────────────────────────────────────────────────────────────────────────
def basket_for(off_xy: np.ndarray) -> tuple:
    """The basket nearer the centroid of offensive positions (n x 2)."""
    c = np.asarray(off_xy, dtype=float).reshape(-1, 2).mean(axis=0)
    d = [np.hypot(c[0] - bx, c[1] - by) for bx, by in BASKETS]
    return BASKETS[int(np.argmin(d))]


def assignment_cost(def_xy, off_xy, basket) -> np.ndarray:
    """n_def x n_off cost: distance to the offensive player plus 0.5 times distance to the midpoint between
    that player and the basket."""
    D = np.asarray(def_xy, dtype=float).reshape(-1, 2)
    O = np.asarray(off_xy, dtype=float).reshape(-1, 2)
    b = np.asarray(basket, dtype=float)
    mid = (O + b) / 2.0
    d1 = np.hypot(D[:, None, 0] - O[None, :, 0], D[:, None, 1] - O[None, :, 1])
    d2 = np.hypot(D[:, None, 0] - mid[None, :, 0], D[:, None, 1] - mid[None, :, 1])
    return d1 + MIDPOINT_WEIGHT * d2


def assign(def_xy, off_xy, basket) -> np.ndarray:
    """Index of the offensive player each defender guards (-1 when unassigned), by linear_sum_assignment."""
    cost = assignment_cost(def_xy, off_xy, basket)
    out = np.full(cost.shape[0], -1, dtype=int)
    if cost.size == 0:
        return out
    rows, cols = linear_sum_assignment(cost)
    out[rows] = cols
    return out


def sample_frames(fl: pd.DataFrame, start_clock: float) -> np.ndarray:
    """1 Hz sample of a possession's frame-level rows (frame, game_clock_s): the frame nearest each whole second
    of elapsed time, in order, without repeats."""
    if len(fl) == 0:
        return np.array([], dtype=np.int64)
    clocks = fl["game_clock_s"].to_numpy(dtype=float)
    frames = fl["frame"].to_numpy(dtype=np.int64)
    elapsed = float(start_clock) - clocks
    picked, seen = [], set()
    for k in range(int(np.floor(max(elapsed.max(), 0.0))) + 1):
        i = int(np.argmin(np.abs(elapsed - k)))
        if frames[i] not in seen:
            seen.add(frames[i])
            picked.append(frames[i])
    return np.asarray(picked, dtype=np.int64)


class _Frames:
    """Column arrays of a frames table sorted by frame, with slices by frame id."""

    def __init__(self, frames: pd.DataFrame):
        f = frames.sort_values("frame", kind="stable")
        self.frame = f["frame"].to_numpy(dtype=np.int64)
        self.period = f["period"].to_numpy(dtype=np.int64)
        self.clock = f["game_clock_s"].to_numpy(dtype=float)
        self.team = f["team_id"].to_numpy(dtype=np.int64)
        self.pid = f["player_id"].to_numpy(dtype=np.int64)
        self.x = f["x"].to_numpy(dtype=float)
        self.y = f["y"].to_numpy(dtype=float)
        first = np.r_[True, self.frame[1:] != self.frame[:-1]]
        self.fl = pd.DataFrame({"frame": self.frame[first], "period": self.period[first], "game_clock_s": self.clock[first]})

    def rows(self, frame: int) -> slice:
        a = int(np.searchsorted(self.frame, frame, "left"))
        b = int(np.searchsorted(self.frame, frame, "right"))
        return slice(a, b)

    def span(self, start_frame: int, end_frame: int, period: int) -> pd.DataFrame:
        fl = self.fl
        m = (fl["frame"] >= start_frame) & (fl["frame"] <= end_frame) & (fl["period"] == period)
        return fl[m]

    def position(self, frame: int, player_id: int):
        s = self.rows(int(frame))
        m = self.pid[s] == int(player_id)
        if not m.any():
            return None
        i = s.start + int(np.flatnonzero(m)[0])
        return float(self.x[i]), float(self.y[i])


def frame_assignments(frames: pd.DataFrame, possessions: pd.DataFrame) -> tuple:
    """Per sampled frame assignments -> (matchup_frames table, per-possession info, counts).

    matchup_frames has FRAME_COLUMNS; info maps possession -> dict(n_sampled, basket, sampled frames);
    counts reports skipped possessions and frames."""
    F = _Frames(frames)
    game = str(possessions["sportvu_game"].iloc[0]) if len(possessions) else ""
    rows, info = [], {}
    counts = {"possessions": int(len(possessions)), "no_offense": 0, "no_frames": 0, "frames_skipped": 0, "frames_sampled": 0}
    for p in possessions.itertuples(index=False):
        if pd.isna(p.offense_team_id):
            counts["no_offense"] += 1
            continue
        off_team = int(p.offense_team_id)
        fl = F.span(int(p.start_frame), int(p.end_frame), int(p.period))
        sampled = sample_frames(fl, float(p.start_clock_s))
        if len(sampled) == 0:
            counts["no_frames"] += 1
            continue
        # basket from the offensive centroid over the sampled frames (positions only)
        off_pts = []
        per_frame = []
        for fr in sampled:
            s = F.rows(int(fr))
            is_off = F.team[s] == off_team
            o = np.flatnonzero(is_off)
            d = np.flatnonzero(~is_off)
            if len(o) == 0 or len(d) == 0:
                per_frame.append(None)
                counts["frames_skipped"] += 1
                continue
            oxy = np.c_[F.x[s][o], F.y[s][o]]
            dxy = np.c_[F.x[s][d], F.y[s][d]]
            off_pts.append(oxy)
            per_frame.append((F.pid[s][o], oxy, F.pid[s][d], dxy))
        if not off_pts:
            counts["no_frames"] += 1
            continue
        basket = basket_for(np.vstack(off_pts))
        prev = {}
        k = 0
        n_sampled = 0
        for fr, item in zip(sampled, per_frame):
            if item is None:
                continue
            opid, oxy, dpid, dxy = item
            a = assign(dxy, oxy, basket)
            cur = {}
            for j, oi in enumerate(a):
                if oi < 0:
                    continue
                did, oid = int(dpid[j]), int(opid[oi])
                dist = float(np.hypot(dxy[j, 0] - oxy[oi, 0], dxy[j, 1] - oxy[oi, 1]))
                changed = np.nan if k == 0 or did not in prev else float(prev[did] != oid)
                rows.append((game, int(p.possession), k, int(fr), did, oid, dist, changed))
                cur[did] = oid
            prev = cur
            k += 1
            n_sampled += 1
        counts["frames_sampled"] += n_sampled
        info[int(p.possession)] = {"n_sampled": n_sampled, "basket": basket, "frames": sampled,
                                   "period": int(p.period), "start_event": int(p.start_event), "end_event": int(p.end_event),
                                   "points": int(p.points), "off_team": off_team}
    mf = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    return mf, info, counts


# ── shots inside possessions ────────────────────────────────────────────────────────────────────────
def shot_value(desc, dist_ft) -> int:
    d = str(desc or "").lower()
    if "3-pt" in d or "3pt" in d:
        return 3
    if "2-pt" in d or "2pt" in d:
        return 2
    return 3 if (dist_ft is not None and np.isfinite(dist_ft) and dist_ft >= THREE_FT) else 2


def possession_shots(game: dict, mf: pd.DataFrame, info: dict, xfg: pd.DataFrame | None = None,
                     default_p: float = 0.45) -> pd.DataFrame:
    """One row per field-goal attempt that falls in a possession (SHOT_COLUMNS), with the shooter's
    position at the release frame, the assigned defender at the sampled frame at or before the release and
    the distance to it, the shot value and the xFG probability p."""
    sf = game["shot_frames"]
    events = game.get("events")
    F = _Frames(game["frames"])
    desc = {}
    ev_frame = {}
    if events is not None and len(events):
        desc = dict(zip(events["event"].astype(int), events["desc"].astype(str)))
        ev_frame = dict(zip(events["event"].astype(int), events["frame"].astype(int)))
    pmap = {}
    if xfg is not None and len(xfg):
        pmap = {(str(g), int(e)): float(p) for g, e, p in zip(xfg["sportvu_game"], xfg["event"], xfg["p_selected"])}
    gid = str(sf["sportvu_game"].iloc[0]) if len(sf) else (str(game["possessions"]["sportvu_game"].iloc[0]) if len(game["possessions"]) else "")
    by_pos = {pos: g for pos, g in mf.groupby("possession")} if len(mf) else {}
    rows = []
    for s in sf.itertuples(index=False):
        ev = int(s.event)
        pos = None
        for pid, inf in info.items():
            if inf["period"] == int(s.period) and inf["start_event"] <= ev <= inf["end_event"]:
                pos = pid
                break
        if pos is None:
            continue
        inf = info[pos]
        shooter = None if pd.isna(s.shooter_id) else int(s.shooter_id)
        rel = int(s.release_frame)
        if rel == schema.NOT_FOUND:
            rel = int(ev_frame.get(ev, schema.NOT_FOUND))
        frames_s = inf["frames"]
        if rel == schema.NOT_FOUND:
            rel = int(frames_s[-1])
        before = frames_s[frames_s <= rel]
        k_frame = int(before[-1]) if len(before) else int(frames_s[0])
        sx = sy = np.nan
        if shooter is not None:
            xy = F.position(rel, shooter) or F.position(k_frame, shooter)
            if xy is not None:
                sx, sy = xy
        bx, by = inf["basket"]
        dist = float(np.hypot(sx - bx, sy - by)) if np.isfinite(sx) else np.nan
        value = shot_value(desc.get(ev), dist)
        def_id, def_dist = np.nan, np.nan
        g = by_pos.get(pos)
        if g is not None and shooter is not None:
            m = g[(g["frame"] == k_frame) & (g["off_player_id"] == shooter)]
            if len(m):
                def_id, def_dist = float(m["def_player_id"].iloc[0]), float(m["dist_ft"].iloc[0])
        p = pmap.get((gid, ev), default_p)
        k = int(np.flatnonzero(frames_s == k_frame)[0]) if (frames_s == k_frame).any() else 0
        rows.append((gid, pos, ev, shooter, bool(s.made), value, p, int(s.release_frame), k, def_id, def_dist, sx, sy, bx, by))
    out = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    if len(out):
        out["assigned_def_id"] = out["assigned_def_id"].astype("Int64")
        out["shooter_id"] = out["shooter_id"].astype("Int64")
    return out


def matchup_possessions(mf: pd.DataFrame, info: dict, shots: pd.DataFrame, game: str) -> pd.DataFrame:
    """Aggregate matchup_frames to the matchup_possessions table (MATCHUP_COLUMNS)."""
    if len(mf) == 0:
        return pd.DataFrame(columns=MATCHUP_COLUMNS)
    n_sampled = mf.groupby("possession")["frame"].nunique()
    pair = mf.groupby(["possession", "def_player_id", "off_player_id"], as_index=False).agg(n=("frame", "size"))
    pair["partial_possession"] = pair["n"] / pair["possession"].map(n_sampled)
    sw = mf.groupby(["possession", "def_player_id"])["changed"].sum(min_count=1).fillna(0).astype(int)
    pair["switches"] = [int(sw.get((p, d), 0)) for p, d in zip(pair["possession"], pair["def_player_id"])]
    # targets
    last_shooter, expected = {}, {}
    if len(shots):
        ss = shots.dropna(subset=["shooter_id"]).sort_values("event")
        for pos, g in ss.groupby("possession"):
            last_shooter[pos] = int(g["shooter_id"].iloc[-1])
            for sid, gg in g.groupby("shooter_id"):
                expected[(pos, int(sid))] = float((gg["p"] * gg["value"]).sum())
    pts = pair["possession"].map(lambda p: info[p]["points"])
    is_shooter = [last_shooter.get(p) == o for p, o in zip(pair["possession"], pair["off_player_id"])]
    pair["points_allowed_share"] = np.where(is_shooter, pts * pair["partial_possession"], 0.0)
    pair["xfg_expected_share"] = [expected.get((p, o), 0.0) * s for p, o, s in
                                  zip(pair["possession"], pair["off_player_id"], pair["partial_possession"])]
    pair.insert(0, "sportvu_game", game)
    out = pair[MATCHUP_COLUMNS].copy()
    out = out.astype({"possession": "int32", "def_player_id": "int32", "off_player_id": "int32", "partial_possession": "float32",
                      "points_allowed_share": "float32", "xfg_expected_share": "float32", "switches": "int16"})
    out["sportvu_game"] = out["sportvu_game"].astype("string")
    return out.reset_index(drop=True)


def build(game: dict, xfg: pd.DataFrame | None = None, default_p: float = 0.45) -> dict:
    """One game dict (frames, events, possessions, shot_frames) -> dict(matchup_possessions, matchup_frames,
    shots, counts). The feature frame of the result passes schema.assert_time_forward."""
    mf, info, counts = frame_assignments(game["frames"], game["possessions"])
    gid = str(game["possessions"]["sportvu_game"].iloc[0]) if len(game["possessions"]) else ""
    shots = possession_shots(game, mf, info, xfg, default_p)
    mp = matchup_possessions(mf, info, shots, gid)
    if len(mp):
        feature_frame(mp)
    return {"matchup_possessions": mp, "matchup_frames": mf, "shots": shots, "counts": counts}


# ── shot locations for the xFG fallback ─────────────────────────────────────────────────────────────
def shot_locations(shots: pd.DataFrame, heldout: bool = False) -> pd.DataFrame:
    """A section 10a-shaped table from possession_shots output, for analytics.features_xfg.features:
    loc_x, loc_y in tenths of a foot relative to the attacked basket (loc_y along the court axis, loc_x across it).
    Rows whose shooter position is unknown carry NaN locations; the trainer gives them the constant rate."""
    sx, sy = shots["shooter_x"].to_numpy(dtype=float), shots["shooter_y"].to_numpy(dtype=float)
    bx, by = shots["basket_x"].to_numpy(dtype=float), shots["basket_y"].to_numpy(dtype=float)
    inward = np.where(bx < 47.0, 1.0, -1.0)
    out = pd.DataFrame({"sportvu_game": shots["sportvu_game"].astype(str).to_numpy(),
                        "event_id": shots["event"].to_numpy(dtype=int),
                        "period": 0, "clock_s": 0.0,
                        "made": shots["made"].astype(bool).to_numpy(),
                        "loc_x": (sy - by) * 10.0 * inward, "loc_y": np.abs(sx - bx) * 10.0,
                        "shot_type": np.where(shots["value"].to_numpy() == 3, "3PT", "2PT"),
                        "heldout": bool(heldout)})
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def run_games(games: list, xfg: pd.DataFrame | None = None, default_p: float = 0.45) -> dict:
    """build() over a list of game dicts -> dict(matchup_possessions, matchup_frames, shots, counts) concatenated."""
    parts = {"matchup_possessions": [], "matchup_frames": [], "shots": []}
    counts = []
    for g in games:
        r = build(g, xfg, default_p)
        for k in parts:
            parts[k].append(r[k])
        counts.append({"sportvu_game": str(g["possessions"]["sportvu_game"].iloc[0]), **r["counts"]})
    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in parts.items()}
    out["counts"] = pd.DataFrame(counts)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="run on the synthetic season (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT with sportvu_contract/ and analytics/<table>/<season>/")
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes analytics/matchups/<season>/<game>.parquet, reads analytics/xfg/")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    from analytics.matchups import validate as V
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    a = parse_args(argv)
    games = V.fixture_games(a.seed) if a.fixtures else V.load_games(a.data, a.season)
    xfg = V.load_xfg(a.out, a.season) if a.out else None
    res = run_games(games, xfg)
    mp = res["matchup_possessions"]
    print("matchup_possessions: %d rows, %d games, %d defenders; sampled frames %d"
          % (len(mp), mp["sportvu_game"].nunique() if len(mp) else 0, mp["def_player_id"].nunique() if len(mp) else 0,
             int(res["counts"]["frames_sampled"].sum()) if len(res["counts"]) else 0))
    print(res["counts"].to_string(index=False))
    if a.out:
        for gid, g in mp.groupby("sportvu_game"):
            held = V.is_heldout(gid) or any(bool(x.get("heldout")) for x in games if str(x["possessions"]["sportvu_game"].iloc[0]) == gid)
            d = Path(a.out) / "analytics" / "matchups" / ("heldout" if held else "") / a.season
            d.mkdir(parents=True, exist_ok=True)
            g.to_parquet(d / (str(gid) + ".parquet"), index=False)
        print("wrote %s" % (Path(a.out) / "analytics" / "matchups"))
    return res


if __name__ == "__main__":
    main()

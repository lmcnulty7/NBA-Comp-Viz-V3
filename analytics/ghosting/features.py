"""analytics/ghosting/features.py: the defender-frame panel at 5 Hz and the lagged feature window.

    python -m analytics.ghosting.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
        [--fixture-games 4] [--fixture-possessions 10] [--fixture-heldout-every 0]

Unit: one defender at one sampled frame of a possession. The 25 Hz frames (section 4) are sampled every
STRIDE = 5 frames (DT = 0.2 s) from the possession's start_frame; a possession (section 19) is used when
gap is false, offense_team_id is set, every sampled frame holds the ten players of the first sampled frame
split five and five by team, the ball is present, and consecutive sampled frames are exactly STRIDE frame
ids apart with a game clock that advances by at most DT + CLOCK_TOL_S (ids are contiguous after the Step 1
dedupe, so a source gap shows as a clock jump; a stopped clock with frames still flowing is a dead ball
and allowed); the sequence is cut at the first sampled frame that breaks this (a dropped frame inside a
possession is never sampled as 0.2 s), and sequences shorter than MIN_STEPS
steps are dropped (cut and skip counts are returned per game).

Assignment: for each defender, the nearest offensive player at the possession's first sampled frame
(defender-centric nearest, so two defenders may share an assignment; no one-to-one matching).

Feature vector F(s) "as of step s", computed by ``base_features`` from the state at s and the defender's
position two steps (0.4 s) earlier (BASE_FEATURES order):
    rel_asg_dx, rel_asg_dy            assignment minus defender
    rel_off1..5_dx/dy                 each offensive player minus defender (slots sorted by player_id)
    rel_ball_dx, rel_ball_dy          ball minus defender
    rel_def1..4_dx/dy                 the other four defenders minus defender (slot order, self removed)
    ball_basket_dist_ft               ball to the attacking basket
    def_basket_dist_ft                defender to the attacking basket
    def_vx, def_vy                    defender velocity over the last 0.4 s (two sampled steps)
    def_x, def_y                      defender position (absolute, feet)
    shot_clock_s, elapsed_s           shot clock and seconds since the possession's start_frame
    attack_right                      1 when the ball is in the right half (x > 47); the attacking basket is
                                      the one on that side (a past-only proxy for the offence's direction)
A null shot clock is filled with max(24 minus elapsed, 0).

The feature frame (``feature_frame``) has one row per defender-step t >= MIN_STEPS - 1 with
    ids        sportvu_game, possession, frame (the frame of t), step, def_slot, def_player_id, off_player_id,
               panel_row (position of step t in the panel, for the roll-forward)
    features   F(t-1) under the plain names (``feature_columns``, the linear baseline's inputs) and
               F(t-k) as <name>_lag<k> for k = 2..N_LAGS (``window_columns``: lag 1 first in columns, then 2, ...;
               reshaped to (n, N_LAGS, len(BASE_FEATURES)), the GRU reverses it to oldest first;
               lags before the sequence start repeat F(0))
    targets    target_dx, target_dy = defender position at t minus at t-1 (the displacement in feet)
Nothing measured at t or later enters the features; ``schema.assert_time_forward`` runs on every frame
returned, and ``shift_features_forward`` is the leakage probe the tests use (it moves every window one
step later, so the lag-1 block is measured at the target frame; a linear model then recovers the target
exactly and the error collapses).

Memory: ``load_games`` builds each game's panel in the same pass that reads and cleans its frames
(validate.load_games with the ``attach_panel`` hook), so only one game's frames are in memory at a time;
``build_panel`` then uses those panels. A game dict without a panel is built from its frames, read one game
at a time. The CLI goes further: it writes each game's feature frame inside that pass and keeps nothing.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from analytics import schema
from analytics.ghosting import validate as V

log = logging.getLogger("analytics.ghosting")

HZ = 25
SAMPLE_HZ = 5
STRIDE = HZ // SAMPLE_HZ
DT = 1.0 / SAMPLE_HZ
N_LAGS = 5
VEL_STEPS = 2                       # 0.4 s
MIN_STEPS = VEL_STEPS + 2           # first target row is step VEL_STEPS + 1
HORIZONS = (0.2, 1.0)
HORIZON_STEPS = {0.2: 1, 1.0: 5}
BASKETS = np.array([[5.25, 25.0], [88.75, 25.0]])
MID_X = 47.0
SHOT_CLOCK_S = 24.0
CLOCK_TOL_S = 0.5 * DT              # a sampled step whose clock advances by more than DT + this cuts the sequence
                                    # (a smaller step is a stopped clock with frames still flowing, allowed)

OFF_XY = [f"off{i}_{a}" for i in range(1, 6) for a in ("x", "y")]
ODEF_XY = [f"odef{j}_{a}" for j in range(1, 5) for a in ("x", "y")]
CTX_COLUMNS = ["asg_x", "asg_y"] + OFF_XY + ["ball_x", "ball_y"] + ODEF_XY + ["shot_clock_s", "elapsed_s"]
_C = {c: i for i, c in enumerate(CTX_COLUMNS)}

BASE_FEATURES = (
    ["rel_asg_dx", "rel_asg_dy"]
    + [f"rel_off{i}_{a}" for i in range(1, 6) for a in ("dx", "dy")]
    + ["rel_ball_dx", "rel_ball_dy"]
    + [f"rel_def{j}_{a}" for j in range(1, 5) for a in ("dx", "dy")]
    + ["ball_basket_dist_ft", "def_basket_dist_ft", "def_vx", "def_vy", "def_x", "def_y",
       "shot_clock_s", "elapsed_s", "attack_right"])
FEATURES = tuple(BASE_FEATURES)
ID_COLUMNS = ["sportvu_game", "possession", "frame", "step", "def_slot", "def_player_id", "off_player_id", "panel_row"]
TARGETS = ["target_dx", "target_dy"]
PANEL_COLUMNS = (["sportvu_game", "possession", "def_slot", "step", "frame", "period", "game_clock_s",
                  "def_player_id", "off_player_id", "asg_slot", "def_x", "def_y"] + CTX_COLUMNS + ["ball_z"])


def feature_columns() -> list[str]:
    """The lag-1 features (the linear baseline's design)."""
    return list(BASE_FEATURES)


def lag_name(name: str, k: int) -> str:
    """The window column of feature ``name`` at lag k: the plain name for lag 1, else <name>_lag<k>."""
    return name if k == 1 else f"{name}_lag{k}"


def window_columns(n_lags: int = N_LAGS) -> list[str]:
    """Lag 1 block first, then lag 2, ... lag n_lags (the deep candidate's design)."""
    return [lag_name(f, k) for k in range(1, n_lags + 1) for f in BASE_FEATURES]


# ----------------------------------------------------------------------------- the feature function

def base_features(ctx: np.ndarray, def_xy: np.ndarray, def_back: np.ndarray, back_s=VEL_STEPS * DT) -> np.ndarray:
    """F as of one step for n rows: ctx (n, len(CTX_COLUMNS)) holds the state of everything except the
    defender, def_xy (n, 2) the defender's position at that step, def_back (n, 2) its position back_s
    seconds earlier (VEL_STEPS steps, or fewer at the sequence start; scalar or (n,) array, floor DT).
    Returns (n, len(BASE_FEATURES)) in BASE_FEATURES order."""
    ctx = np.asarray(ctx, dtype=float)
    def_xy = np.asarray(def_xy, dtype=float).reshape(-1, 2)
    def_back = np.asarray(def_back, dtype=float).reshape(-1, 2)
    back_s = np.maximum(np.asarray(back_s, dtype=float), DT).reshape(-1, 1)
    n = len(ctx)
    cols = []
    cols.append(ctx[:, [_C["asg_x"], _C["asg_y"]]] - def_xy)
    for i in range(1, 6):
        cols.append(ctx[:, [_C[f"off{i}_x"], _C[f"off{i}_y"]]] - def_xy)
    ball = ctx[:, [_C["ball_x"], _C["ball_y"]]]
    cols.append(ball - def_xy)
    for j in range(1, 5):
        cols.append(ctx[:, [_C[f"odef{j}_x"], _C[f"odef{j}_y"]]] - def_xy)
    right = (ball[:, 0] > MID_X).astype(float)
    basket = BASKETS[right.astype(int)]
    cols.append(np.linalg.norm(ball - basket, axis=1).reshape(n, 1))
    cols.append(np.linalg.norm(def_xy - basket, axis=1).reshape(n, 1))
    cols.append((def_xy - def_back) / back_s)
    cols.append(def_xy)
    cols.append(ctx[:, [_C["shot_clock_s"], _C["elapsed_s"]]])
    cols.append(right.reshape(n, 1))
    out = np.concatenate(cols, axis=1)
    assert out.shape[1] == len(BASE_FEATURES)
    return out


# ----------------------------------------------------------------------------- the panel

def _possession_arrays(sub: pd.DataFrame, offense_team_id: int, start_clock: float):
    """Sampled-step arrays of one possession from its sorted frame rows, or (None, reason)."""
    fr = sub["frame"].to_numpy()
    starts = np.flatnonzero(np.r_[True, fr[1:] != fr[:-1]])
    sizes = np.diff(np.r_[starts, len(fr)])
    pid = sub["player_id"].to_numpy()
    team = sub["team_id"].to_numpy()
    xy = sub[["x", "y"]].to_numpy(dtype=float)
    ball = sub[["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)
    sc = sub["shot_clock_s"].to_numpy(dtype=float)
    gc = sub["game_clock_s"].to_numpy(dtype=float)
    period = sub["period"].to_numpy()
    lineup = None
    cut = ""
    O, D, B, S, G, FR, P = [], [], [], [], [], [], []
    for s0, n in zip(starts, sizes):
        if n != 10:
            cut = "lineup"
            break
        if FR and (fr[s0] != FR[-1] + STRIDE or (G[-1] - gc[s0]) > DT + CLOCK_TOL_S):
            cut = "frame_gap"              # a dropped frame or a clock jump inside the possession: never sampled as 0.2 s
            break
        ids = pid[s0:s0 + 10]
        if lineup is None:
            off = team[s0:s0 + 10] == offense_team_id
            if off.sum() != 5:
                return None, "first frame without five on five"
            lineup, off_mask = ids.copy(), off.copy()
        elif not np.array_equal(ids, lineup):
            cut = "lineup"
            break
        if not np.all(np.isfinite(ball[s0, :2])):
            cut = "ball"
            break
        O.append(xy[s0:s0 + 10][off_mask])
        D.append(xy[s0:s0 + 10][~off_mask])
        B.append(ball[s0])
        S.append(sc[s0])
        G.append(gc[s0])
        FR.append(fr[s0])
        P.append(period[s0])
    if len(FR) < MIN_STEPS:
        return None, "shorter than %d steps" % MIN_STEPS
    T = len(FR)
    cut_frames = int(fr[starts[len(FR)]]) if len(FR) < len(starts) else -1   # first sampled frame not used
    O, D, B = np.stack(O), np.stack(D), np.stack(B)
    elapsed = float(start_clock) - np.asarray(G, dtype=float)
    elapsed = np.where(np.isfinite(elapsed), elapsed, np.arange(T) * DT)
    S = np.asarray(S, dtype=float)
    S = np.where(np.isfinite(S), S, np.maximum(SHOT_CLOCK_S - elapsed, 0.0))
    return {"O": O, "D": D, "B": B, "shot_clock": S, "elapsed": elapsed, "game_clock": np.asarray(G, dtype=float),
            "frame": np.asarray(FR), "period": np.asarray(P), "off_ids": lineup[off_mask], "def_ids": lineup[~off_mask],
            "cut": cut, "cut_frame": cut_frames}, ""


def build_panel_game(frames: pd.DataFrame, possessions: pd.DataFrame, sportvu_game: str) -> tuple[pd.DataFrame, dict]:
    """The 5 Hz defender panel of one game -> (panel, counts). Rows are contiguous per (possession, def_slot)
    and ordered by step."""
    counts = {"possessions": int(len(possessions)), "used": 0, "skipped_gap": 0, "skipped_no_offense": 0,
              "skipped_short": 0, "skipped_lineup": 0, "cut_frame_gap": 0, "cut_lineup": 0, "cut_ball": 0, "steps": 0}
    f = frames.sort_values(["frame", "player_id"], kind="stable")
    fcol = f["frame"].to_numpy()
    blocks = []
    for r in possessions.sort_values("possession").itertuples(index=False):
        if bool(r.gap):
            counts["skipped_gap"] += 1
            continue
        if pd.isna(r.offense_team_id):
            counts["skipped_no_offense"] += 1
            continue
        sampled = np.arange(int(r.start_frame), int(r.end_frame) + 1, STRIDE)
        sub = f.iloc[np.flatnonzero(np.isin(fcol, sampled))]
        arr, why = _possession_arrays(sub, int(r.offense_team_id), float(r.start_clock_s))
        if arr is None:
            counts["skipped_short" if "short" in why else "skipped_lineup"] += 1
            continue
        O, D, B, T = arr["O"], arr["D"], arr["B"], len(arr["frame"])
        if arr["cut"]:
            counts["cut_" + arr["cut"]] += 1
        d0 = np.linalg.norm(D[0][:, None, :] - O[0][None, :, :], axis=2)
        asg = d0.argmin(axis=1)
        for d in range(5):
            others = [j for j in range(5) if j != d]
            ctx = np.concatenate([O[:, asg[d], :], O.reshape(T, 10), B[:, :2], D[:, others, :].reshape(T, 8),
                                  arr["shot_clock"].reshape(T, 1), arr["elapsed"].reshape(T, 1)], axis=1)
            blk = {"sportvu_game": np.full(T, str(sportvu_game), dtype=object), "possession": np.full(T, int(r.possession)),
                   "def_slot": np.full(T, d), "step": np.arange(T), "frame": arr["frame"], "period": arr["period"],
                   "game_clock_s": arr["game_clock"], "def_player_id": np.full(T, int(arr["def_ids"][d])),
                   "off_player_id": np.full(T, int(arr["off_ids"][asg[d]])), "asg_slot": np.full(T, int(asg[d])),
                   "def_x": D[:, d, 0], "def_y": D[:, d, 1]}
            for i, c in enumerate(CTX_COLUMNS):
                blk[c] = ctx[:, i]
            blk["ball_z"] = B[:, 2]
            blocks.append(pd.DataFrame(blk))
        counts["used"] += 1
        counts["steps"] += T
    if not blocks:
        return pd.DataFrame({c: pd.Series([], dtype=float) for c in PANEL_COLUMNS}), counts
    panel = pd.concat(blocks, ignore_index=True)[PANEL_COLUMNS]
    return panel, counts


def attach_panel(game: dict, frames: pd.DataFrame) -> None:
    """validate.load_games' on_frames hook: build the game's panel from its cleaned frames and keep it on the
    dict (``panel``, ``panel_counts``), so the frames can be dropped."""
    game["panel"], game["panel_counts"] = build_panel_game(frames, game["possessions"], game["sportvu_game"])


def load_games(args) -> list[dict]:
    """validate.load_games with each game's panel built while its frames are in memory, one game at a time."""
    return V.load_games(args, on_frames=attach_panel)


def build_panel(games: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Panel over game dicts (sportvu_game, frames, possessions, heldout) -> (panel with a heldout column,
    {game: counts}). ``panel_row`` is the row position, the key the roll-forward uses. A panel kept on a dict
    by attach_panel is used and taken off the dict (so it is not held twice); otherwise the game's frames are
    read and the panel built, one game at a time."""
    parts, counts = [], {}
    for g in games:
        if g.get("panel") is not None:
            p, c = g.pop("panel"), g["panel_counts"]
        else:
            p, c = build_panel_game(V.frames_of(g), g["possessions"], g["sportvu_game"])
        p["heldout"] = bool(g.get("heldout", False))
        parts.append(p)
        counts[g["sportvu_game"]] = c
        log.info("%s panel: %s", g["sportvu_game"], c)
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PANEL_COLUMNS + ["heldout"])
    panel["panel_row"] = np.arange(len(panel))
    return panel, counts


# ----------------------------------------------------------------------------- the feature frame

def panel_features(panel: pd.DataFrame) -> np.ndarray:
    """F(s) for every panel row (n, len(BASE_FEATURES)); at the sequence start the velocity is over the steps
    that exist, min(step, VEL_STEPS) * DT with a floor of DT (rows are contiguous per sequence with step
    counting from 0)."""
    n = len(panel)
    if n == 0:
        return np.zeros((0, len(BASE_FEATURES)))
    step = panel["step"].to_numpy()
    idx = np.arange(n)
    nback = np.minimum(step, VEL_STEPS)
    ctx = panel[CTX_COLUMNS].to_numpy(dtype=float)
    dxy = panel[["def_x", "def_y"]].to_numpy(dtype=float)
    return base_features(ctx, dxy, dxy[idx - nback], back_s=nback * DT)


def feature_frame(panel: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """One row per defender-step t >= MIN_STEPS - 1 with the lagged window and the displacement target.
    Runs schema.assert_time_forward before returning."""
    F = panel_features(panel)
    n = len(panel)
    step = panel["step"].to_numpy()
    idx = np.arange(n)
    keep = idx[step >= MIN_STEPS - 1]
    out = {c: panel[c].to_numpy()[keep] for c in ID_COLUMNS if c != "panel_row"}
    out["panel_row"] = (panel["panel_row"].to_numpy() if "panel_row" in panel.columns else idx)[keep]
    cols = {}
    for k in range(1, n_lags + 1):
        src = keep - np.minimum(step[keep], k)
        block = F[src]
        for j, f in enumerate(BASE_FEATURES):
            cols[lag_name(f, k)] = block[:, j]
    dxy = panel[["def_x", "def_y"]].to_numpy(dtype=float)
    disp = dxy[keep] - dxy[keep - 1]
    frame = pd.DataFrame({**out, **cols, "target_dx": disp[:, 0], "target_dy": disp[:, 1]})
    frame["heldout"] = panel["heldout"].to_numpy()[keep] if "heldout" in panel.columns else False
    schema.assert_time_forward(frame, "states_1hz")
    return frame


def shift_features_forward(frame: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """The leakage probe: every window moves one step later, so the lag-1 block is F(t), measured at the
    target frame, and the lag-2 block is F(t-1). Rows whose next step is missing are dropped. Only a test
    calls this; a trainer never does."""
    f = frame.reset_index(drop=True)
    keys = ["sportvu_game", "possession", "def_slot"]
    nxt = f.groupby(keys, sort=False)["step"].shift(-1)
    ok = (nxt == f["step"] + 1).to_numpy()
    cols = window_columns(n_lags)
    shifted = f[cols].shift(-1)
    out = f.loc[ok].copy()
    out[cols] = shifted.loc[ok].to_numpy()
    return out.reset_index(drop=True)


def build(games: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """-> (panel, feature frame, counts) for a list of game dicts."""
    panel, counts = build_panel(games)
    return panel, feature_frame(panel), counts


# ----------------------------------------------------------------------------- CLI

def parse_args(argv=None):
    """The features CLI's command line (validate.add_input_args plus --out)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    V.add_input_args(ap)
    ap.add_argument("--out", help="OUT_ROOT (default DATA_ROOT): write analytics/ghosting/<season>/ghost_features_<game>.parquet there "
                                  "(held-out under analytics/ghosting/heldout/<season>/)")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """CLI: build the panel and the feature frame, print the counts, write ghost_features per game under --out.
    Streams: each game's panel and feature frame are built, written and dropped while its frames are in memory
    (the feature frame is per sequence, so this matches one frame over all games; panel_row stays global)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = parse_args(argv)
    out_root = a.out or a.data
    counts, rows, written, offset = {}, {}, [], [0]

    def one_game(g, frames):
        p, c = build_panel_game(frames, g["possessions"], g["sportvu_game"])
        log.info("%s panel: %s", g["sportvu_game"], c)
        p["heldout"] = bool(g.get("heldout", False))
        p["panel_row"] = offset[0] + np.arange(len(p))
        offset[0] += len(p)
        frame = feature_frame(p)
        counts[g["sportvu_game"]], rows[g["sportvu_game"]] = c, int(len(frame))
        if out_root:
            path = V.output_path(out_root, "ghost_features", g["sportvu_game"], g["heldout"], V.SEASON)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            written.append(path)

    games = V.load_games(a, on_frames=one_game)
    print("features (%d, lag 1): %s" % (len(BASE_FEATURES), ", ".join(BASE_FEATURES)))
    print("window columns: %d (%d lags)" % (len(window_columns()), N_LAGS))
    print("| game | held out | possessions | used | skipped gap | skipped no offence | skipped short | skipped lineup | steps | feature rows |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g in games:
        c = counts[g["sportvu_game"]]
        print("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
            g["sportvu_game"], g["heldout"], c["possessions"], c["used"], c["skipped_gap"], c["skipped_no_offense"],
            c["skipped_short"], c["skipped_lineup"], c["steps"], rows[g["sportvu_game"]]))
    for p in written:
        print("wrote", p)
    return {"rows": int(sum(rows.values())), "counts": counts}


if __name__ == "__main__":
    main(sys.argv[1:])

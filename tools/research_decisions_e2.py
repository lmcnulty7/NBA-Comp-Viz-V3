"""tools/research_decisions_e2.py: the decisions research experiment decisions-E2 (PLAN.md D120, D122).

    python3 -m tools.research_decisions_e2 e2-pass [--artifacts DIR] [--max-minutes 25]
    python3 -m tools.research_decisions_e2 e2-fold --fold K [--artifacts DIR]
    python3 -m tools.research_decisions_e2 e2-dev  [--artifacts DIR]

Pre-registration: docs/research/decisions.md, "decisions-E2", with its amendment (D122), which fixes every definition
used here before any number existed. The question: valued on one scale (the EPV of the state each option leads to),
does the chosen option's value predict the points that followed, and does regret carry a signal the frozen values do
not?

e2-pass   One pass over the development games' frames, one game at a time, resumable: the frozen decision contexts
          joined to the decisions-E1 gate's frozen decisions; for each decision the state rows to score (the actual
          state at the decision frame, one per pass option, the modelled drive state, the actual state 1 s later);
          the features of the completion, drive and choice models; the pass attempts the completion model learns
          from; the outcome. Development games only; held-out games are never read.
e2-fold   One fold, checkpointed: the frozen EPV sequence model refit on the fold's training games with the frozen
          functions and gated against the stored out-of-fold epv_points; the fold's state rows scored; the
          completion, drive and choice models fitted on the other folds; the new option values of the fold's
          decisions.
e2-dev    The development statistics and decision. Writes reports/research/decisions-E2-dev.json.

Nothing here writes under DATA_ROOT or reports/ except the research results; no lane command line is run.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, _plain, _sha256, artifacts_dir, assert_safe_output, training_games_only
from tools.research_testset import dev_games, refuse_test

WHO = "tools.research_decisions_e2"
EXPERIMENT = "decisions-E2"
RESULTS = REPO_ROOT / "reports" / "research"
SEASON = "2015-16"
SEED, BOOT, FOLDS = 0, 2000, 5
PASS_MAX_S, RELEASE_TOL = 1.5, 2
LANE_FT, NEAR_FT, DRIVE_WIN_S = 3.0, 6.0, 1.0
EPV_TOL = 1e-5
SLOPE_BAND = (0.8, 1.2)
WORK = "decisions_e2"
SHOTS_FILE = "collect/bref_shots/shotchart/2015-16.parquet"
EVENT_KEYS = ["sportvu_game", "possession", "frame", "player_id", "decision_kind", "event", "receiver_id"]
OPTIONS = ("hold", "pass1", "pass2", "pass3", "pass4", "drive", "after")
COMPLETION_FEATURES = ["pass_ft", "receiver_def_ft", "lane_def_ft", "lane_n_def", "receiver_basket_ft"]
DRIVE_FEATURES = ["speed_toward_basket_ftps", "dist_to_basket_ft", "nearest_def_dist_ft", "n_def_ahead", "shot_clock_s"]


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / WORK)


def _frozen():
    from analytics.decisions import features as DF
    from analytics.epv import features as EF
    from analytics.epv import states as ES
    if (DF.HOLD_FT, DF.HOLD_Z_FT, DF.MIN_HOLD_FRAMES, DF.PASS_MAX_S, DF.DRIVE_FT, DF.DRIVE_MIN_FT) != \
            (2.5, 8.0, 3, PASS_MAX_S, 8.0, 3.0):
        raise RuntimeError(f"{WHO}: the frozen decision thresholds changed")
    return DF, EF, ES


# ----------------------------------------------------------------------------- development inputs

def dev_game_list(root: Path) -> list:
    """The development games with stored states, decisions and frames (the decisions-E1 gate's games)."""
    from analytics.eval import is_heldout_game
    states = {p.stem for p in (root / "analytics" / "states_1hz" / SEASON).glob("*.parquet")}
    frames = {p.stem for p in (root / "sportvu_contract").glob("*.parquet") if not p.name.endswith("_pbp.parquet")}
    games = dev_games(sorted(g for g in states & frames if not is_heldout_game(g)), WHO)
    training_games_only(games, WHO)
    return games


def frozen_decisions(artifacts=None):
    """The decisions-E1 gate's frozen decisions of the development games (the frozen lane refit with frozen code,
    reproduced exactly; tools.research_decisions)."""
    import pandas as pd
    from tools import research_decisions as RD
    RD.require_gate(artifacts)
    ev = pd.read_parquet(RD.gate_dir(artifacts) / "events.parquet")
    training_games_only(ev["sportvu_game"].astype(str).unique(), WHO)
    return ev


# ----------------------------------------------------------------------------- holders and pass attempts

def holder_segments(frames, possessions, fps: float):
    """Holding segments of every player (the frozen thresholds: nearest player within 2.5 ft of the ball in the
    floor plane while the ball is under 8 ft, a segment of at least 3 frames) -> rows (player_id, team_id, start,
    end, possession), inside possession spans only."""
    import numpy as np
    import pandas as pd
    DF, _, _ = _frozen()
    f = frames[["frame", "player_id", "team_id", "x", "y", "ball_x", "ball_y", "ball_z"]].sort_values(["frame", "player_id"])
    d = np.hypot(f["x"].to_numpy(float) - f["ball_x"].to_numpy(float), f["y"].to_numpy(float) - f["ball_y"].to_numpy(float))
    f = f.assign(d=np.where(np.isfinite(d), d, np.inf))
    near = f.loc[f.groupby("frame")["d"].idxmin()]
    near = near[(near["d"] <= DF.HOLD_FT) & (near["ball_z"].to_numpy(float) < DF.HOLD_Z_FT)]
    fr = near["frame"].to_numpy(np.int64)
    pid = near["player_id"].to_numpy(np.int64)
    if not len(fr):
        return pd.DataFrame(columns=["player_id", "team_id", "start", "end", "possession"])
    brk = np.r_[True, (pid[1:] != pid[:-1]) | (fr[1:] - fr[:-1] > 1)]
    seg = np.cumsum(brk) - 1
    s = pd.DataFrame({"seg": seg, "frame": fr, "player_id": pid, "team_id": near["team_id"].to_numpy(np.int64)})
    g = s.groupby("seg").agg(player_id=("player_id", "first"), team_id=("team_id", "first"), start=("frame", "min"),
                             end=("frame", "max"), n=("frame", "size"))
    g = g[g["n"] >= DF.MIN_HOLD_FRAMES].drop(columns="n").reset_index(drop=True)
    p = possessions.sort_values("start_frame")
    starts, ends = p["start_frame"].to_numpy(np.int64), p["end_frame"].to_numpy(np.int64)
    k = np.searchsorted(starts, g["start"].to_numpy(np.int64), side="right") - 1
    ok = (k >= 0) & (g["start"].to_numpy(np.int64) <= ends[np.clip(k, 0, len(ends) - 1)])
    g["possession"] = np.where(ok, p["possession"].to_numpy()[np.clip(k, 0, len(p) - 1)], -1)
    return g[g["possession"] >= 0].reset_index(drop=True)


def lane_stats(px, py, rx, ry, dxs, dys):
    """Smallest defender distance to the passer-receiver segment and the number of defenders within LANE_FT of it,
    per row (defenders as (n, 5) arrays)."""
    import numpy as np
    vx, vy = (rx - px)[:, None], (ry - py)[:, None]
    L2 = np.maximum(vx ** 2 + vy ** 2, 1e-9)
    t = np.clip(((dxs - px[:, None]) * vx + (dys - py[:, None]) * vy) / L2, 0.0, 1.0)
    dist = np.hypot(dxs - (px[:, None] + t * vx), dys - (py[:, None] + t * vy))
    dist = np.where(np.isfinite(dist), dist, np.inf)
    return dist.min(axis=1), (dist <= LANE_FT).sum(axis=1)


def frame_positions(frames, frame_ids, offense_team):
    """Per requested (frame, offense team): offence and defence positions as (n, 5) arrays (sorted by player id, as
    the frozen state builder sorts them), offence ids, and a mask of rows with five on five."""
    import numpy as np
    import pandas as pd
    req = pd.DataFrame({"_i": np.arange(len(frame_ids)), "frame": np.asarray(frame_ids, np.int64),
                        "off": np.asarray(offense_team, np.int64)})
    sub = frames[frames["frame"].isin(set(req["frame"]))][["frame", "team_id", "player_id", "x", "y"]]
    m = req.merge(sub, on="frame", how="inner")
    m["is_off"] = m["team_id"].to_numpy() == m["off"].to_numpy()
    cnt = m.groupby("_i").agg(n=("player_id", "size"), n_off=("is_off", "sum"))
    good = cnt.index[(cnt["n"] == 10) & (cnt["n_off"] == 5)]
    m = m[m["_i"].isin(good)].sort_values(["_i", "is_off", "player_id"], ascending=[True, False, True], kind="stable")
    n = len(req)
    ox = np.full((n, 5), np.nan); oy = np.full((n, 5), np.nan); dx = np.full((n, 5), np.nan); dy = np.full((n, 5), np.nan)
    oid = np.full((n, 5), -1, dtype=np.int64); did = np.full((n, 5), -1, dtype=np.int64)
    if len(m):
        i = m["_i"].to_numpy().reshape(-1, 10)[:, 0]
        X, Y = m["x"].to_numpy(float).reshape(-1, 10), m["y"].to_numpy(float).reshape(-1, 10)
        ids = m["player_id"].to_numpy(np.int64).reshape(-1, 10)
        ox[i], oy[i], dx[i], dy[i] = X[:, :5], Y[:, :5], X[:, 5:], Y[:, 5:]
        oid[i], did[i] = ids[:, :5], ids[:, 5:]
    ok = np.zeros(n, dtype=bool)
    ok[np.asarray(good, dtype=int)] = True
    return {"ox": ox, "oy": oy, "dx": dx, "dy": dy, "oid": oid, "did": did, "ok": ok}


def pass_attempts(frames, possessions, shot_frames, baskets: dict, fps: float, game: str):
    """Pass attempts (amendment, commitment 5): an offensive holder's segment followed within 1.5 s by another
    holder's, with no shot release in [end - 2, next start]: completed when the next holder is a teammate; failed
    when a defender, the intended receiver being the offensive teammate nearest the ball at that defender's first
    holding frame. Features at the passer's last holding frame."""
    import numpy as np
    import pandas as pd
    from analytics import schema
    seg = holder_segments(frames, possessions, fps)
    poss = possessions.set_index("possession")
    rel = pd.to_numeric(shot_frames["release_frame"], errors="coerce").to_numpy(float) if shot_frames is not None else np.zeros(0)
    rel = rel[np.isfinite(rel) & (rel != schema.NOT_FOUND)]
    rows, counts = [], {"no_next_holder": 0, "shot_between": 0}
    for p, s in seg.groupby("possession"):
        off_team = poss.loc[p, "offense_team_id"]
        if pd.isna(off_team):
            continue
        s = s.sort_values("start")
        a, b = s.iloc[:-1].reset_index(drop=True), s.iloc[1:].reset_index(drop=True)
        for i in range(len(a)):
            if int(a.loc[i, "team_id"]) != int(off_team):
                continue
            gap = (int(b.loc[i, "start"]) - int(a.loc[i, "end"])) / fps
            if gap > PASS_MAX_S or b.loc[i, "player_id"] == a.loc[i, "player_id"]:
                if gap > PASS_MAX_S:
                    counts["no_next_holder"] += 1
                continue
            lo, hi = int(a.loc[i, "end"]) - RELEASE_TOL, int(b.loc[i, "start"])
            if ((rel >= lo) & (rel <= hi)).any():
                counts["shot_between"] += 1
                continue
            later = s.iloc[i + 2:]
            regains = bool((later["team_id"].to_numpy() == int(off_team)).any())
            rows.append({"possession": int(p), "frame": int(a.loc[i, "end"]), "next_frame": int(b.loc[i, "start"]),
                         "offense_regains": regains,
                         "possession_turnover": str(poss.loc[p, "end_kind"]) == "turnover",
                         "gap_s": gap,
                         "passer": int(a.loc[i, "player_id"]), "next_holder": int(b.loc[i, "player_id"]),
                         "completed": int(int(b.loc[i, "team_id"]) == int(off_team)), "offense_team_id": int(off_team)})
    att = pd.DataFrame(rows)
    if not len(att):
        return att, counts
    pos = frame_positions(frames, att["frame"], att["offense_team_id"])
    nxt = frame_positions(frames, att["next_frame"], att["offense_team_id"])
    fl = frames.drop_duplicates("frame").set_index("frame")
    out = []
    for r in range(len(att)):
        if not pos["ok"][r]:
            continue
        oid = pos["oid"][r]
        pi = np.flatnonzero(oid == att.loc[r, "passer"])
        if not len(pi):
            continue
        if att.loc[r, "completed"]:
            ri = np.flatnonzero(oid == att.loc[r, "next_holder"])
        else:
            if not nxt["ok"][r]:
                continue
            bx, by = fl.loc[att.loc[r, "next_frame"], "ball_x"], fl.loc[att.loc[r, "next_frame"], "ball_y"]
            d = np.hypot(nxt["ox"][r] - bx, nxt["oy"][r] - by)
            d[nxt["oid"][r] == att.loc[r, "passer"]] = np.inf
            ri = np.flatnonzero(oid == nxt["oid"][r][int(np.argmin(d))])
        if not len(ri):
            continue
        pi, ri = int(pi[0]), int(ri[0])
        per = int(fl.loc[att.loc[r, "frame"], "period"])
        bk = baskets.get((int(att.loc[r, "offense_team_id"]), per))
        if bk is None:
            continue
        px, py, rx, ry = pos["ox"][r, pi], pos["oy"][r, pi], pos["ox"][r, ri], pos["oy"][r, ri]
        lane_d, lane_n = lane_stats(np.array([px]), np.array([py]), np.array([rx]), np.array([ry]),
                                    pos["dx"][r:r + 1], pos["dy"][r:r + 1])
        out.append({"sportvu_game": game, "possession": int(att.loc[r, "possession"]), "frame": int(att.loc[r, "frame"]),
                    "passer": int(att.loc[r, "passer"]), "receiver": int(oid[ri]), "completed": int(att.loc[r, "completed"]),
                    "pass_ft": float(np.hypot(rx - px, ry - py)),
                    "receiver_def_ft": float(np.nanmin(np.hypot(pos["dx"][r] - rx, pos["dy"][r] - ry))),
                    "lane_def_ft": float(lane_d[0]), "lane_n_def": int(lane_n[0]),
                    "offense_regains": bool(att.loc[r, "offense_regains"]),
                    "possession_turnover": bool(att.loc[r, "possession_turnover"]), "gap_s": float(att.loc[r, "gap_s"]),
                    "ball_travel_ft": float(np.hypot(fl.loc[att.loc[r, "next_frame"], "ball_x"] - px,
                                                     fl.loc[att.loc[r, "next_frame"], "ball_y"] - py)),
                    "receiver_basket_ft": float(np.hypot(rx - bk[0], ry - bk[1]))})
    return pd.DataFrame(out), counts


# ----------------------------------------------------------------------------- state rows

def state_rows(frames, possessions, game: str, frame_ids, possession_ids):
    """The frozen state builder's row (analytics.epv.states.build_states) at any requested frame of a possession:
    the ten players split by the possession's offence and sorted by player id, the ball, the handler (the offensive
    player nearest the ball) and its nearest-defender distance, the clocks, and the elapsed time on the game clock
    (continuous here; build_states takes whole seconds). Rows that are not five on five or have no ball are NaN in
    every position and flagged not ok."""
    import numpy as np
    import pandas as pd
    _, _, ES = _frozen()
    fl = ES.frame_level(frames).set_index("frame")
    poss = possessions.set_index("possession")
    fids = np.asarray(frame_ids, np.int64)
    pids = np.asarray(possession_ids, np.int64)
    off = poss.loc[pids, "offense_team_id"].to_numpy(float)
    pos = frame_positions(frames, fids, np.nan_to_num(off, nan=-1).astype(np.int64))
    fli = fl.reindex(fids)
    bx, by, bz = (fli[c].to_numpy(float) for c in ("ball_x", "ball_y", "ball_z"))
    d_ball = np.hypot(pos["ox"] - bx[:, None], pos["oy"] - by[:, None])
    d_ball = np.where(np.isfinite(d_ball), d_ball, np.inf)
    h = np.argmin(d_ball, axis=1)
    n = len(fids)
    hx, hy = pos["ox"][np.arange(n), h], pos["oy"][np.arange(n), h]
    dd = np.hypot(pos["dx"] - hx[:, None], pos["dy"] - hy[:, None])
    any_def = np.isfinite(dd).any(axis=1)                 # False when the frame is not five on five: the row is not ok
    hdef = np.full(n, np.nan)
    hdef[any_def] = np.nanmin(dd[any_def], axis=1)
    ok = pos["ok"] & np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz) & np.isfinite(off)
    out = pd.DataFrame({"sportvu_game": game, "possession": pids, "frame": fids,
                        "period": fli["period"].to_numpy(), "game_clock_s": fli["game_clock_s"].to_numpy(float),
                        "shot_clock_s": fli["shot_clock_s"].to_numpy(float),
                        "elapsed_in_possession_s": poss.loc[pids, "start_clock_s"].to_numpy(float) - fli["game_clock_s"].to_numpy(float),
                        "offense_team_id": off, "handler_id": pos["oid"][np.arange(n), h].astype(float),
                        "ball_x": bx, "ball_y": by, "ball_z": bz})
    for i in range(5):
        out[f"off{i + 1}_id"] = pos["oid"][:, i]
    for i in range(5):
        out[f"off{i + 1}_x"], out[f"off{i + 1}_y"] = pos["ox"][:, i], pos["oy"][:, i]
    for i in range(5):
        out[f"def{i + 1}_id"] = pos["did"][:, i]
    for i in range(5):
        out[f"def{i + 1}_x"], out[f"def{i + 1}_y"] = pos["dx"][:, i], pos["dy"][:, i]
    out["handler_nearest_def_dist_ft"] = hdef
    out["points_to_end"], out["possession_end_kind"] = 0, "other"        # targets: never read by a feature
    out["ok"] = ok
    out["not_ok_reason"] = np.where(ok, "", np.where(~pos["ok"], "not five on five",
                                     np.where(~np.isfinite(off), "no offence", "no ball")))
    return out


def with_handler(rows, slot: "np.ndarray", ball_xy=None):
    """State rows with the offensive player in `slot` (0..4) as handler and the ball at his position (or at
    ball_xy), and the handler's nearest-defender distance recomputed."""
    import numpy as np
    r = rows.copy()
    n = len(r)
    ox = np.stack([r[f"off{i + 1}_x"].to_numpy(float) for i in range(5)], axis=1)
    oy = np.stack([r[f"off{i + 1}_y"].to_numpy(float) for i in range(5)], axis=1)
    oid = np.stack([r[f"off{i + 1}_id"].to_numpy() for i in range(5)], axis=1)
    dx = np.stack([r[f"def{i + 1}_x"].to_numpy(float) for i in range(5)], axis=1)
    dy = np.stack([r[f"def{i + 1}_y"].to_numpy(float) for i in range(5)], axis=1)
    hx, hy = ox[np.arange(n), slot], oy[np.arange(n), slot]
    r["handler_id"] = oid[np.arange(n), slot].astype(float)
    r["ball_x"], r["ball_y"] = (hx, hy) if ball_xy is None else ball_xy
    r["handler_nearest_def_dist_ft"] = np.nanmin(np.hypot(dx - hx[:, None], dy - hy[:, None]), axis=1)
    return r


def drive_rows(rows, slot, basket_x, basket_y):
    """The modelled state 1 s later on a drive (amendment, commitment 4): the handler and the ball moved 8 ft toward
    the basket along the line to it, never nearer than 3 ft, everyone else where they are, the clocks advanced 1 s."""
    import numpy as np
    DF, _, _ = _frozen()
    r = rows.copy()
    n = len(r)
    hx = np.stack([r[f"off{i + 1}_x"].to_numpy(float) for i in range(5)], axis=1)[np.arange(n), slot]
    hy = np.stack([r[f"off{i + 1}_y"].to_numpy(float) for i in range(5)], axis=1)[np.arange(n), slot]
    vx, vy = hx - basket_x, hy - basket_y
    dd = np.hypot(vx, vy)
    scale = np.maximum(dd - DF.DRIVE_FT, DF.DRIVE_MIN_FT) / np.maximum(dd, 1e-9)
    nx, ny = basket_x + vx * scale, basket_y + vy * scale
    for i in range(5):
        m = slot == i
        r.loc[m, f"off{i + 1}_x"] = nx[m]
        r.loc[m, f"off{i + 1}_y"] = ny[m]
    r["ball_x"] = r["ball_x"].to_numpy(float) + (nx - hx)
    r["ball_y"] = r["ball_y"].to_numpy(float) + (ny - hy)
    r = with_handler(r, slot, (r["ball_x"].to_numpy(float), r["ball_y"].to_numpy(float)))
    r["elapsed_in_possession_s"] = r["elapsed_in_possession_s"].to_numpy(float) + DRIVE_WIN_S
    r["shot_clock_s"] = np.maximum(r["shot_clock_s"].to_numpy(float) - DRIVE_WIN_S, 0.0)
    r["game_clock_s"] = r["game_clock_s"].to_numpy(float) - DRIVE_WIN_S
    return r


# ----------------------------------------------------------------------------- the frame pass

def _join_events(ctx, ev):
    """The frozen contexts joined to the gate's frozen decisions on the decision keys (unique per game)."""
    import pandas as pd
    keys = EVENT_KEYS[1:]
    ints = {k: "int64" for k in keys if k != "decision_kind"}
    ctx, ev = ctx.reset_index(drop=True), ev.reset_index(drop=True)      # positional: no index alignment below
    c = ctx.astype(ints).assign(decision_kind=ctx["decision_kind"].astype(str).to_numpy())
    e = ev.astype(ints).assign(decision_kind=ev["decision_kind"].astype(str).to_numpy())
    c = c.drop(columns=[x for x in c.columns if x in e.columns and x not in keys])
    m = e.merge(c, on=keys, how="left", validate="one_to_one", indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError(f"{WHO}: a frozen decision has no context")
    return m.drop(columns="_merge")


def game_pass(root: Path, g: str, ev):
    """Everything one game contributes (one game of frames in memory): decision rows with features, labels and the
    outcome; the state rows to score; the pass attempts; a check of the state builder against the stored states."""
    import numpy as np
    import pandas as pd
    from analytics.decisions import validate as DV
    DF, _, _ = _frozen()
    box = {}

    def per_game(c):
        raw = c["frames"]() if callable(c["frames"]) else c["frames"]
        ctx = DF.decision_contexts({**c, "frames": raw}, {})        # the frozen function on the bundle as loaded
        frames = raw.sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)
        poss, sf = c["possessions"], c["shot_frames"]
        fps = DF.fps_of(frames)
        fv = frames["frame"].to_numpy()
        baskets = DF.baskets_by_team_period(frames, fv, poss)
        att, att_counts = pass_attempts(frames, poss, sf, baskets, fps, g)
        dec = _join_events(ctx, ev)
        dec.insert(0, "decision_id", np.arange(len(dec)))
        actual = state_rows(frames, poss, g, dec["frame"], dec["possession"])
        oid = np.stack([actual[f"off{i + 1}_id"].to_numpy() for i in range(5)], axis=1)
        slot_of = lambda ids: np.array([int(np.flatnonzero(oid[r] == ids[r])[0]) if (oid[r] == ids[r]).any() else -1
                                        for r in range(len(ids))])
        h = slot_of(dec["player_id"].to_numpy())
        rows = []
        keep = actual["ok"].to_numpy() & (h >= 0)
        a = actual.assign(decision_id=dec["decision_id"].to_numpy(), option="hold")
        rows.append(a[keep])
        for k in range(1, 5):
            tm = dec[f"tm{k}_id"].to_numpy()
            s = slot_of(tm)
            m = keep & (tm >= 0) & (s >= 0)
            if m.any():
                rows.append(with_handler(actual[m], s[m]).assign(decision_id=dec["decision_id"].to_numpy()[m],
                                                                 option=f"pass{k}"))
        rows.append(drive_rows(actual[keep], h[keep], dec["basket_x"].to_numpy(float)[keep],
                               dec["basket_y"].to_numpy(float)[keep]).assign(decision_id=dec["decision_id"].to_numpy()[keep],
                                                                             option="drive"))
        end = poss.set_index("possession").loc[dec["possession"], "end_frame"].to_numpy(np.int64)
        f1 = dec["frame"].to_numpy(np.int64) + int(round(fps * DRIVE_WIN_S))
        has = keep & (f1 <= end) & np.isin(f1, fv)
        if has.any():
            after = state_rows(frames, poss, g, f1[has], dec["possession"].to_numpy()[has])
            after = after.assign(decision_id=dec["decision_id"].to_numpy()[has], option="after")
            rows.append(after[after["ok"].to_numpy()])
        cf = pd.concat(rows, ignore_index=True)
        # decision-level features
        dxs = np.stack([actual[f"def{i + 1}_x"].to_numpy(float) for i in range(5)], axis=1)
        dys = np.stack([actual[f"def{i + 1}_y"].to_numpy(float) for i in range(5)], axis=1)
        hx, hy = dec["x"].to_numpy(float), dec["y"].to_numpy(float)
        bx, by = dec["basket_x"].to_numpy(float), dec["basket_y"].to_numpy(float)
        for k in range(1, 5):
            rx, ry = dec[f"tm{k}_x"].to_numpy(float), dec[f"tm{k}_y"].to_numpy(float)
            ld, ln = lane_stats(hx, hy, rx, ry, dxs, dys)
            dec[f"tm{k}_lane_def_ft"], dec[f"tm{k}_lane_n_def"] = ld, ln
            dec[f"tm{k}_basket_ft"] = np.hypot(rx - bx, ry - by)
        ahead = ((dxs - hx[:, None]) * (bx - hx)[:, None] + (dys - hy[:, None]) * (by - hy)[:, None]) > 0
        near = np.hypot(dxs - hx[:, None], dys - hy[:, None]) <= NEAR_FT
        dec["n_def_ahead"] = (ahead & near).sum(axis=1)
        dec["shot_clock_s"] = actual["shot_clock_s"].to_numpy(float)
        seg = holder_segments(frames, poss, fps)
        dseg = seg[seg["team_id"].to_numpy() != poss.set_index("possession").loc[seg["possession"], "offense_team_id"].to_numpy()]
        lost = np.zeros(len(dec), dtype=bool)
        win = int(round(fps * DRIVE_WIN_S))
        for r, (p, f) in enumerate(zip(dec["possession"].to_numpy(), dec["frame"].to_numpy())):
            s = dseg[dseg["possession"] == p]
            lost[r] = bool(((s["start"] > f) & (s["start"] <= f + win)).any())
        pinfo = poss.set_index("possession").loc[dec["possession"]]
        tov = (pinfo["end_kind"].astype(str).to_numpy() == "turnover") & (pinfo["end_frame"].to_numpy() <= dec["frame"].to_numpy() + win)
        dec["drive_kept"] = np.where(dec["decision_kind"].astype(str) == "drive", (~(lost | tov)).astype(float), np.nan)
        dec["state_ok"] = keep
        dec["not_ok_reason"] = np.where(keep, "", np.where(h < 0, "decision player not on the floor",
                                                           actual["not_ok_reason"].to_numpy()))
        box.update(dec=dec, cf=cf, att=att, att_counts=att_counts, fps=fps)

    bundles = DV.load_root_bundles(root, SEASON, root, games=[g])
    if len(bundles) != 1 or bundles[0]["heldout"]:
        raise SystemExit(f"{WHO}: {g} is not one development game")
    DV.stream_bundles(bundles, per_game)
    # the outcome and the builder check, from the stored states (the EPV lane's input)
    stored = pd.read_parquet(root / "analytics" / "states_1hz" / SEASON / f"{g}.parquet")
    box["dec"]["points_to_end"] = outcomes(box["dec"], stored)
    return box


def outcomes(dec, stored):
    """Commitment 2: points_to_end of the possession's last stored state at or before the decision frame; NaN when
    there is none (a possession with no stored state at all, or a decision before its first state), and such a
    decision is left out of both arms and counted."""
    import numpy as np
    out = np.full(len(dec), np.nan)
    pos = {p: i for i, p in enumerate(dec.index)}
    for p, d in dec.groupby("possession"):
        s = stored[stored["possession"] == p].sort_values("frame")
        if not len(s):
            continue
        idx = np.searchsorted(s["frame"].to_numpy(), d["frame"].to_numpy(), side="right") - 1
        val = np.where(idx >= 0, s["points_to_end"].to_numpy(float)[np.clip(idx, 0, None)], np.nan)
        out[[pos[i] for i in d.index]] = val
    return out


def builder_check(root: Path, g: str) -> dict:
    """state_rows at the stored states' frames against the stored states (positions, ids, handler, its defender
    distance): the frozen builder's row, reproduced. Elapsed differs by design (continuous against whole seconds)."""
    import numpy as np
    import pandas as pd
    from analytics.decisions import validate as DV
    stored = pd.read_parquet(root / "analytics" / "states_1hz" / SEASON / f"{g}.parquet")
    res = {}

    def per_game(c):
        fr = c["frames"]() if callable(c["frames"]) else c["frames"]
        mine = state_rows(fr, c["possessions"], g, stored["frame"], stored["possession"])
        cols = [f"{s}{i}_{a}" for s in ("off", "def") for i in range(1, 6) for a in ("x", "y")] + \
               ["ball_x", "ball_y", "ball_z", "handler_nearest_def_dist_ft"]
        a, b = mine[cols].to_numpy(float), stored[cols].to_numpy(float)
        ids = [f"{s}{i}_id" for s in ("off", "def") for i in range(1, 6)] + ["handler_id"]
        res.update(rows=int(len(stored)), max_abs_diff=float(np.nanmax(np.abs(a - b))) if len(a) else 0.0,
                   id_mismatch=int((mine[ids].to_numpy(float) != stored[ids].to_numpy(float)).sum()),
                   elapsed_max_diff=float(np.nanmax(np.abs(mine["elapsed_in_possession_s"].to_numpy(float)
                                                        - stored["elapsed_in_possession_s"].to_numpy(float)))))

    DV.stream_bundles(DV.load_root_bundles(root, SEASON, root, games=[g]), per_game)
    return res


def frame_pass(artifacts=None, max_minutes: float = 25.0, log=print) -> dict:
    import os
    import numpy as np
    root = Path(os.environ["DATA_ROOT"])
    games = dev_game_list(root)
    ev = frozen_decisions(artifacts)
    ev_games = set(ev["sportvu_game"].astype(str))
    out = work_dir(artifacts) / "games"
    out.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    for g in games:
        if (out / f"{g}.json").exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        if g not in ev_games:
            (out / f"{g}.json").write_text(json.dumps({"game": g, "decisions": 0}) + "\n")
            continue
        box = game_pass(root, g, ev[ev["sportvu_game"].astype(str) == g])
        box["dec"].to_parquet(out / f"{g}.decisions.parquet", index=False)
        box["cf"].to_parquet(out / f"{g}.states.parquet", index=False)
        box["att"].to_parquet(out / f"{g}.attempts.parquet", index=False)
        rec = {"game": g, "decisions": int(len(box["dec"])), "state_rows": int(len(box["cf"])),
               "attempts": int(len(box["att"])), "attempt_counts": box["att_counts"], "fps": box["fps"],
               "state_ok": int(box["dec"]["state_ok"].sum()),
               "not_ok_by_reason": {str(k): int(v) for k, v in
                                    box["dec"].loc[~box["dec"]["state_ok"], "not_ok_reason"].value_counts().items()},
               "no_outcome": int(box["dec"]["points_to_end"].isna().sum())}
        if done % 50 == 0:
            rec["builder_check"] = builder_check(root, g)
        (out / f"{g}.json").write_text(json.dumps(_plain(rec)) + "\n")         # written last: the game is done
        done += 1
        if log and done % 25 == 0:
            log(f"[research] {done} games this run, {time.time() - started:.0f} s")
    have = sum((out / f"{g}.json").exists() for g in games)
    return {"games": len(games), "written_this_run": done, "present": have, "complete": have == len(games),
            "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2)}


# ----------------------------------------------------------------------------- EPV fold refits

def epv_training(root: Path) -> dict:
    """The frozen EPV loader's training tables from the development games only: analytics.epv.validate.load_root
    reads the held-out directories too, so the same tables are built here from the non-held-out directories and
    the season's shot file (every game in it a development game), then cleaned and split by the frozen code."""
    import pandas as pd
    from analytics.epv import train as T
    from analytics.epv import validate as V
    tables = {}
    for table in V.TABLE_DIRS:
        parts = V._read_dir(root / "analytics" / table, False, SEASON)
        tables[table] = pd.concat(parts, ignore_index=True) if parts else None
    shots = pd.read_parquet(root / SHOTS_FILE)
    ids = sorted(shots["sportvu_game"].astype(str).unique())
    if len(dev_games(ids, WHO)) != len(ids):
        raise SystemExit(f"{WHO}: the shot file holds a game that is not a development game")
    tables["shots"] = V.with_heldout(shots, False)
    st = tables["states_1hz"]
    tables["games"] = sorted(set(zip(st["sportvu_game"].astype(str), st["heldout"].astype(bool))))
    tables["sources"] = {}
    d = T.split_tables(tables)
    if len(d["states_heldout"]):
        raise SystemExit(f"{WHO}: a held-out state reached the development tables")
    return d


def epv_folds(games) -> list:
    from analytics.eval import grouped_folds
    return [(sorted(tr), sorted(te)) for tr, te in grouped_folds(sorted(games), n_folds=FOLDS, seed=SEED)]


def sequences(stored, cf):
    """Each state row of cf as the last state of its own sequence (commitment 4): the possession's stored states with
    an earlier frame, then the row, its elapsed time raised to the last earlier state's when the whole-second grid
    puts that state later. Returns the stacked states (one possession id per sequence) and the positions of the
    rows of cf in it."""
    import numpy as np
    import pandas as pd
    cols = [c for c in stored.columns if c in cf.columns]
    st = stored.sort_values(["possession", "frame"], kind="stable").reset_index(drop=True)
    key = (st["possession"].to_numpy(np.int64) << 32) + st["frame"].to_numpy(np.int64)
    p, f = cf["possession"].to_numpy(np.int64), cf["frame"].to_numpy(np.int64)
    lo = np.searchsorted(key, p << 32, side="left")
    hi = np.searchsorted(key, (p << 32) + f, side="left")
    lens = hi - lo
    n = len(cf)
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    idx = np.repeat(lo - starts, lens) + np.arange(int(lens.sum()))
    prior = st.iloc[idx][cols].reset_index(drop=True)
    prior["possession"] = np.repeat(np.arange(n), lens)
    el = st["elapsed_in_possession_s"].to_numpy(float)
    last_el = np.where(lens > 0, el[np.maximum(hi - 1, 0)], -np.inf)
    own = cf[cols].reset_index(drop=True).copy()
    own["possession"] = np.arange(n)
    own["elapsed_in_possession_s"] = np.maximum(own["elapsed_in_possession_s"].to_numpy(float), last_el)
    seq = pd.concat([prior, own], ignore_index=True)       # a tie in elapsed keeps the row last (stable sorts)
    return seq, len(prior) + np.arange(n)


def score_states(model, priors, stored, cf, max_rows: int = 400_000):
    """Expected points of every row of cf, each the last state of its sequence, by the frozen features (geometry,
    the fold's priors, 5 lags) and the fold model."""
    import numpy as np
    from analytics.epv import features as EF
    from analytics.epv import train as T
    cols = EF.lagged_columns(T.N_LAGS)
    out = np.full(len(cf), np.nan)
    n_rows = stored.groupby("possession").size().reindex(cf["possession"]).fillna(0).to_numpy() + 1
    cum = np.cumsum(n_rows)
    c0 = 0
    while c0 < len(cf):
        base = cum[c0 - 1] if c0 else 0
        c1 = max(int(np.searchsorted(cum, base + max_rows, side="right")), c0 + 1)
        seq, last = sequences(stored, cf.iloc[c0:c1])
        X = EF.lagged(EF.build_features(seq, priors), T.N_LAGS)
        out[c0:c1] = np.asarray(model.predict(X[cols].to_numpy(dtype=float)), dtype=float)[last]
        c0 = c1
    return out


def stored_epv(root: Path, games) -> "pd.DataFrame":
    import pandas as pd
    return pd.concat([pd.read_parquet(root / "analytics" / "epv" / SEASON / f"{g}.parquet",
                                      columns=["sportvu_game", "possession", "frame", "epv_points", "fold"])
                      for g in games], ignore_index=True)


def fold_dir(artifacts, k: int) -> Path:
    d = work_dir(artifacts) / f"fold{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_job(k: int, artifacts=None, max_minutes: float = 25.0, log=print) -> dict:
    """Fold k, checkpointed (amendment, commitment 3): the frozen EPV sequence model refit on the fold's training
    games and saved; the gate (its expected points on the fold's stored states against the stored out-of-fold
    epv_points); then the fold's decisions' state rows scored, one game at a time, resumable."""
    import os
    import numpy as np
    import pandas as pd
    import torch
    from analytics.epv import deep as D
    from analytics.epv import features as EF
    from analytics.epv import train as T
    root = Path(os.environ["DATA_ROOT"])
    started = time.time()
    d = epv_training(root)
    st, shots, rosters = d["states_train"], d["shots_train"], d["rosters"]
    games = st["sportvu_game"].astype(str).to_numpy()
    folds = epv_folds(np.unique(games))
    tr_g, te_g = folds[k]
    out = fold_dir(artifacts, k)
    (work_dir(artifacts) / "folds.json").write_text(json.dumps({"folds": [te for _, te in folds]}) + "\n")
    tr, te = np.isin(games, tr_g), np.isin(games, te_g)
    priors = EF.Priors.from_training(st.loc[tr], shots, rosters)
    model_path = out / "epv_model.pt"
    rec_path = out / "gate.json"
    if not model_path.exists():
        geo = EF.geometry(st)
        X_tr, X_te, _ = T.fold_features(geo.loc[tr], geo.loc[te], st.loc[tr], shots, rosters)
        del geo
        X_tr, X_te = EF.lagged(X_tr, T.N_LAGS), EF.lagged(X_te, T.N_LAGS)
        cols = EF.lagged_columns(T.N_LAGS)
        t0 = time.time()
        model = D.fit_sequence(X_tr, st.loc[tr, "points_to_end"].to_numpy(float), cols, seed=SEED, n_steps=T.N_LAGS)
        fit_s = time.time() - t0
        del X_tr
        pred = X_te[["sportvu_game", "possession", "frame"]].assign(mine=model.predict(X_te[cols].to_numpy(float)))
        del X_te
        ref = stored_epv(root, te_g)
        m = pred.merge(ref, on=["sportvu_game", "possession", "frame"], how="outer", indicator=True)
        both = m["_merge"] == "both"
        diff = np.abs(m.loc[both, "mine"].to_numpy(float) - m.loc[both, "epv_points"].to_numpy(float))
        gate = {"fold": k, "train_games": len(tr_g), "test_games": len(te_g), "test_states": int(te.sum()),
                "matched": int(both.sum()), "unmatched": int((~both).sum()),
                "stored_fold_is_k": bool((m.loc[both, "fold"] == k).all()),
                "max_abs_diff": float(diff.max()) if len(diff) else float("nan"), "tolerance": EPV_TOL,
                "fit_seconds": round(fit_s, 1)}
        gate["passed"] = bool(gate["unmatched"] == 0 and gate["stored_fold_is_k"] and gate["max_abs_diff"] <= EPV_TOL)
        torch.save(model, model_path)
        gate["model_sha256"] = _sha256(model_path)
        rec_path.write_text(json.dumps(_plain(gate), indent=2) + "\n")
        if log:
            log(f"[research] fold {k} gate: {gate}")
    gate = json.loads(rec_path.read_text())
    if not gate["passed"] or gate["model_sha256"] != _sha256(model_path):
        raise SystemExit(f"{WHO}: fold {k} EPV gate did not pass: {gate}")
    model = torch.load(model_path, weights_only=False)
    gdir = work_dir(artifacts) / "games"
    done = 0
    for g in te_g:
        target = out / f"{g}.epv.parquet"
        if target.exists() or not (gdir / f"{g}.states.parquet").exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        cf = pd.read_parquet(gdir / f"{g}.states.parquet")
        stored = st[games == g]
        e = score_states(model, priors, stored, cf)
        cf[["decision_id", "option"]].assign(epv=e).to_parquet(target, index=False)
        done += 1
    want = [g for g in te_g if (gdir / f"{g}.states.parquet").exists()]
    have = sum((out / f"{g}.epv.parquet").exists() for g in want)
    return {"fold": k, "gate": gate, "games_scored_this_run": done, "present": have, "wanted": len(want),
            "complete": have == len(want), "seconds": round(time.time() - started, 1),
            "peak_gb": round(peak_rss_gb(), 2)}


# ----------------------------------------------------------------------------- development statistics

def _logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def load_dev(artifacts=None):
    """The frame pass and the fold outputs of every development game, joined: one row per decision with its EPVs
    (wide), the pass attempts, and the fold of every game."""
    import numpy as np
    import pandas as pd
    w = work_dir(artifacts)
    folds = json.loads((w / "folds.json").read_text())["folds"]
    fold_of = {g: k for k, gs in enumerate(folds) for g in gs}
    for k in range(FOLDS):
        require_fold(artifacts, k)
    decs, epvs, atts = [], [], []
    for g in sorted(fold_of):
        if not (w / "games" / f"{g}.decisions.parquet").exists():
            continue
        dd = pd.read_parquet(w / "games" / f"{g}.decisions.parquet")
        e = pd.read_parquet(w / f"fold{fold_of[g]}" / f"{g}.epv.parquet")
        wide = e.pivot(index="decision_id", columns="option", values="epv").reindex(columns=list(OPTIONS))
        wide.columns = [f"epv_{c}" for c in wide.columns]
        decs.append(dd.join(wide, on="decision_id").assign(fold=fold_of[g]))
        a = pd.read_parquet(w / "games" / f"{g}.attempts.parquet")
        atts.append(a.assign(fold=fold_of[g]) if len(a) else a)
    dec = pd.concat(decs, ignore_index=True)
    att = completion_label(pd.concat([a for a in atts if len(a)], ignore_index=True))
    training_games_only(dec["sportvu_game"].astype(str).unique(), WHO)
    refuse_test(sorted(dec["sportvu_game"].astype(str).unique()), WHO)
    return dec, att, fold_of


def completion_label(att):
    """Commitment 5 as amended (D123): completed when the next holder is a teammate; failed when it is a defender, the
    offence never holds the ball again in the possession, and the possession ends in a turnover; every other attempt
    is left out (label NaN) and counted."""
    import numpy as np
    a = att.copy()
    done = a["completed"].to_numpy(int) == 1
    failed = (~done) & (~a["offense_regains"].to_numpy(bool)) & a["possession_turnover"].to_numpy(bool)
    a["label"] = np.where(done, 1.0, np.where(failed, 0.0, np.nan))
    return a


def attempt_counts(att) -> dict:
    """The D123 counts: labelled, completion rate, excluded, overall and by game."""
    import numpy as np
    lab = att["label"].to_numpy(float)
    by = att.assign(excluded=~np.isfinite(lab)).groupby("sportvu_game")["excluded"].sum() if len(att) else None
    return {"attempts": int(len(att)), "labelled": int(np.isfinite(lab).sum()),
            "completed": int((lab == 1).sum()), "failed": int((lab == 0).sum()),
            "completion_rate": float(np.nanmean(lab)), "excluded": int((~np.isfinite(lab)).sum()),
            "excluded_by_game": {str(g): int(n) for g, n in by.items()} if by is not None else {}}


def require_fold(artifacts, k: int) -> dict:
    d = fold_dir(artifacts, k)
    gate = json.loads((d / "gate.json").read_text())
    if not gate["passed"] or gate["model_sha256"] != _sha256(d / "epv_model.pt"):
        raise SystemExit(f"{WHO}: fold {k} EPV gate did not pass")
    return gate


def option_models(dec, att):
    """Out of fold by game (the EPV folds): P(completion) per pass option, P(drive keeps the ball), and the choice
    probabilities of the three kinds (commitments 5 to 7). Adds p_pass<k>, p_keep, p_choice_<kind>."""
    import numpy as np
    from analytics.decisions import features as DF
    dec = dec.copy()
    for k in range(1, 5):
        dec[f"p_pass{k}"] = np.nan
    dec["p_keep"] = np.nan
    for kind in DF.KINDS:
        dec[f"p_choice_{kind}"] = np.nan
    fits = {}
    for f in range(FOLDS):
        te = dec["fold"].to_numpy() == f
        a = att[att["fold"] != f].dropna(subset=COMPLETION_FEATURES + ["label"])
        comp = _logistic().fit(a[COMPLETION_FEATURES].to_numpy(float), a["label"].to_numpy(int))
        for k in range(1, 5):
            X = np.column_stack([dec[f"tm{k}_pass_ft"], dec[f"tm{k}_def_ft"], dec[f"tm{k}_lane_def_ft"],
                                 dec[f"tm{k}_lane_n_def"], dec[f"tm{k}_basket_ft"]]).astype(float)
            ok = te & np.isfinite(X).all(axis=1) & (dec[f"tm{k}_id"].to_numpy() >= 0)
            if ok.any():
                dec.loc[ok, f"p_pass{k}"] = comp.predict_proba(X[ok])[:, 1]
        dr = dec[(dec["fold"] != f) & dec["drive_kept"].notna()].dropna(subset=DRIVE_FEATURES)
        keep = _logistic().fit(dr[DRIVE_FEATURES].to_numpy(float), dr["drive_kept"].to_numpy(int))
        X = dec[DRIVE_FEATURES].to_numpy(float)
        ok = te & np.isfinite(X).all(axis=1)
        dec.loc[ok, "p_keep"] = keep.predict_proba(X[ok])[:, 1]
        ch_tr = dec[(dec["fold"] != f)].dropna(subset=DF.FEATURES)
        choice = _logistic().fit(ch_tr[DF.FEATURES].to_numpy(float), ch_tr["decision_kind"].astype(str).to_numpy())
        X = dec[DF.FEATURES].to_numpy(float)
        ok = te & np.isfinite(X).all(axis=1)
        pr = choice.predict_proba(X[ok])
        for j, kind in enumerate(choice.classes_):
            dec.loc[ok, f"p_choice_{kind}"] = pr[:, j]
        fits[f] = {"attempts": int(len(a)), "attempt_completion_rate": float(a["label"].mean()),
                   "drives": int(len(dr)), "drive_kept_rate": float(dr["drive_kept"].mean()),
                   "completion_coef": comp[-1].coef_[0].tolist(), "keep_coef": keep[-1].coef_[0].tolist()}
    return dec, fits


def values(dec):
    """One value scale (commitment 4) -> per decision: v_shot, v_hold, v_pass<k>, v_drive, chosen, best, the best
    option, regret; and the frozen arm's chosen, hold and regret beside them."""
    import numpy as np
    v = dec.copy()
    v["v_shot"] = v["shot_value"].to_numpy(float)
    v["v_hold"] = v["epv_hold"].to_numpy(float)
    for k in range(1, 5):
        v[f"v_pass{k}"] = v[f"p_pass{k}"].to_numpy(float) * v[f"epv_pass{k}"].to_numpy(float)
    v["v_drive"] = v["p_keep"].to_numpy(float) * v["epv_drive"].to_numpy(float)
    kind = v["decision_kind"].astype(str).to_numpy()
    chosen = np.full(len(v), np.nan)
    chosen[kind == "shot"] = v["v_shot"].to_numpy()[kind == "shot"]
    chosen[kind == "drive"] = v["v_drive"].to_numpy()[kind == "drive"]
    rec = v["receiver_id"].to_numpy()
    for k in range(1, 5):
        m = (kind == "pass") & (v[f"tm{k}_id"].to_numpy() == rec)
        chosen[m] = v[f"v_pass{k}"].to_numpy()[m]
    opts = ["v_shot", "v_hold"] + [f"v_pass{k}" for k in range(1, 5)] + ["v_drive"]
    M = v[opts].to_numpy(float)
    have = np.isfinite(M).any(axis=1)
    best = np.where(have, np.nanmax(np.where(np.isfinite(M), M, -np.inf), axis=1), np.nan)
    v["chosen"], v["best"] = chosen, best
    v["best_option"] = np.where(have, np.array(opts)[np.argmax(np.where(np.isfinite(M), M, -np.inf), axis=1)], "")
    v["regret"] = chosen - best
    v["frozen_chosen"], v["frozen_hold"] = v["chosen_value"].to_numpy(float), v["hold_value"].to_numpy(float)
    v["frozen_regret"] = v["decision_quality"].to_numpy(float)
    v["delta_after"] = v["epv_after"].to_numpy(float) - v["epv_hold"].to_numpy(float)
    return v


def _game_moments(v, cols, y="points_to_end"):
    """Per game: X'X and X'y of [1, cols] (sufficient statistics of least squares), stacked (G, p, p) and (G, p)."""
    import numpy as np
    g = v["sportvu_game"].astype(str).to_numpy()
    games, gi = np.unique(g, return_inverse=True)
    X = np.column_stack([np.ones(len(v))] + [v[c].to_numpy(float) for c in cols])
    yv = v[y].to_numpy(float)
    p = X.shape[1]
    XX = np.zeros((len(games), p, p))
    Xy = np.zeros((len(games), p))
    np.add.at(XX, gi, X[:, :, None] * X[:, None, :])
    np.add.at(Xy, gi, X * yv[:, None])
    return games, XX, Xy


def _coef(XX, Xy, w):
    import numpy as np
    A = np.tensordot(w, XX, axes=(-1, 0))
    b = np.tensordot(w, Xy, axes=(-1, 0))
    return np.linalg.solve(A, b[..., None])[..., 0] if A.ndim == 3 else np.linalg.solve(A, b)


def boot_weights(n_games: int, n: int = BOOT, seed: int = SEED):
    """Game bootstrap as count weights (n, G): the same draws for every statistic and both arms."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_games, size=(n, n_games))
    W = np.zeros((n, n_games))
    np.add.at(W, (np.repeat(np.arange(n), n_games), idx.ravel()), 1.0)
    return W


def _interval(point, draws) -> dict:
    import numpy as np
    d = np.asarray(draws, dtype=float)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def metric_set(v, W=None, n: int = BOOT, seed: int = SEED) -> dict:
    """The primary metric (commitment 8) for both arms on the same decisions, with the paired difference, and the
    ablation (commitment 9), each with its game bootstrap interval of fixed per-decision values."""
    import numpy as np
    out = {}
    games = np.unique(v["sportvu_game"].astype(str))
    W = boot_weights(len(games), n, seed) if W is None else W
    one = np.ones(len(games))
    co = {}
    for arm, (ch, hold, reg) in {"new": ("chosen", "v_hold", "regret"),
                                 "frozen": ("frozen_chosen", "frozen_hold", "frozen_regret")}.items():
        _, XX, Xy = _game_moments(v, [ch])
        s_pt, s_b = _coef(XX, Xy, one)[1], _coef(XX, Xy, W)[:, 1]
        _, XX, Xy = _game_moments(v, [hold, reg])
        r_pt, r_b = _coef(XX, Xy, one)[2], _coef(XX, Xy, W)[:, 2]
        co[arm] = (r_pt, r_b)
        out[arm] = {"slope": _interval(s_pt, s_b), "regret_coef": _interval(r_pt, r_b),
                    "mean_chosen": float(v[ch].mean()), "mean_regret": float(v[reg].mean())}
    out["paired_regret_coef_new_minus_frozen"] = _interval(co["new"][0] - co["frozen"][0], co["new"][1] - co["frozen"][1])
    ab = v[np.isfinite(v["delta_after"].to_numpy(float))]
    ga = np.unique(ab["sportvu_game"].astype(str))
    pos = np.searchsorted(games, ga)
    _, XX, Xy = _game_moments(ab, ["v_hold", "delta_after"])
    out["ablation_delta_epv_coef"] = _interval(_coef(XX, Xy, np.ones(len(ga)))[2], _coef(XX, Xy, W[:, pos])[:, 2])
    out["ablation_rows"] = int(len(ab))
    return out


def overlap(v) -> dict:
    """Commitment 7 (report only): the share of decisions whose best option's kind has a choice probability below
    0.05; decisions whose best option is hold are counted apart."""
    import numpy as np
    kind_of = {"v_shot": "shot", "v_drive": "drive", **{f"v_pass{k}": "pass" for k in range(1, 5)}}
    bo = v["best_option"].astype(str).to_numpy()
    hold = bo == "v_hold"
    p = np.full(len(v), np.nan)
    for opt, kind in kind_of.items():
        m = bo == opt
        p[m] = v[f"p_choice_{kind}"].to_numpy(float)[m]
    ok = ~hold & np.isfinite(p)
    return {"decisions": int(len(v)), "best_is_hold": int(hold.sum()), "best_is_hold_share": float(hold.mean()),
            "best_kind_rated": int(ok.sum()), "extrapolated": int((p[ok] < 0.05).sum()),
            "extrapolated_share": float((p[ok] < 0.05).mean()) if ok.any() else float("nan"),
            "best_option_counts": {k: int(c) for k, c in zip(*np.unique(bo, return_counts=True))}}


def secondary(v, artifacts=None) -> dict:
    """The pre-registration's secondary readings, point estimates, report only: decisions-E1's statistics
    (stability, R squared on kind shares, split-half) of the new regret and of the frozen regret on the same
    decisions (tools.research_decisions, its frozen-unit table and fixed player sets)."""
    import numpy as np
    from tools import research_decisions as RD
    events, ortg, _ = RD._gate_tables(artifacts)
    games = sorted(v["sportvu_game"].astype(str).unique())
    out = {}
    for arm, col in (("new", "regret"), ("frozen", "frozen_regret")):
        ev = v[["sportvu_game", "player_id", "decision_kind", "zone"]].assign(decision_quality=v[col].to_numpy(float))
        T = RD.Table(ev, ortg, games)
        s = RD.statistics(T, np.ones(T.G), RD.player_sets(T), np.random.default_rng(SEED), RD.INNER_BOOT)
        out[arm] = {k: s["frozen"][k] for k in ("stability", "r2_on_kind_shares", "split_half_r", "discrimination")}
    return out


def decide(m: dict) -> dict:
    new = m["new"]
    checks = {"regret_coef_above_zero": new["regret_coef"]["lo"] > 0,
              "paired_difference_above_zero": m["paired_regret_coef_new_minus_frozen"]["lo"] > 0,
              "slope_in_band": SLOPE_BAND[0] <= new["slope"]["point"] <= SLOPE_BAND[1]}
    return {"checks": checks, "adopt_on_development": all(checks.values()),
            "set_a": "scored once under D122" if all(checks.values()) else "not scored (D122: development does not adopt)"}


def development(artifacts=None, log=print) -> dict:
    import numpy as np
    dec, att, fold_of = load_dev(artifacts)
    dec, fits = option_models(dec, att)
    v = values(dec)
    need = ["chosen", "v_hold", "regret", "frozen_chosen", "frozen_hold", "frozen_regret", "points_to_end"]
    finite = np.isfinite(v[need].to_numpy(float)).all(axis=1) & v["state_ok"].to_numpy(bool)
    counts = {"decisions": int(len(v)), "finite_both_arms": int(finite.sum()),
              "by_kind": {k: int(c) for k, c in v.loc[finite, "decision_kind"].astype(str).value_counts().items()},
              "left_out_by_kind": {k: int(c) for k, c in v.loc[~finite, "decision_kind"].astype(str).value_counts().items()},
              "games": int(v.loc[finite, "sportvu_game"].nunique()), "pass_attempts_d123": attempt_counts(att)}
    vf = v[finite].reset_index(drop=True)
    m = metric_set(vf)
    out = {"experiment": EXPERIMENT, "stage": "development games (D122: development decides)", "counts": counts,
           "metrics": m, "decision": decide(m), "overlap": overlap(vf), "option_models": fits,
           "secondary_report_only": secondary(vf, artifacts),
           "folds": {str(k): require_fold(artifacts, k) for k in range(FOLDS)},
           "pass_counts": _pass_counts(artifacts)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    vf[["sportvu_game", "possession", "frame", "decision_kind", "fold", "points_to_end", "chosen", "v_hold", "regret",
        "best_option", "frozen_chosen", "frozen_hold", "frozen_regret", "delta_after"]].to_parquet(
        work_dir(artifacts) / "dev_values.parquet", index=False)
    return out


def _pass_counts(artifacts=None) -> dict:
    gdir = work_dir(artifacts) / "games"
    tot = {"games": 0, "decisions": 0, "state_ok": 0, "attempts": 0, "no_next_holder": 0, "shot_between": 0}
    checks = []
    for f in sorted(gdir.glob("*.json")):
        r = json.loads(f.read_text())
        tot["games"] += 1
        for k in ("decisions", "state_ok", "attempts"):
            tot[k] += int(r.get(k, 0))
        for k in ("no_next_holder", "shot_between"):
            tot[k] += int(r.get("attempt_counts", {}).get(k, 0))
        if "builder_check" in r:
            checks.append(r["builder_check"])
    tot["builder_checks"] = checks
    return tot


# ----------------------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m " + WHO)
    ap.add_argument("stage", choices=("e2-pass", "e2-fold", "e2-dev"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    if a.stage == "e2-fold" and a.fold not in range(FOLDS):
        ap.error("--fold 0..4")
    name = f"{EXPERIMENT} {a.stage}" + (f" {a.fold}" if a.fold is not None else "")
    with heavy_job(name):
        if a.stage == "e2-pass":
            r = frame_pass(a.artifacts, a.max_minutes)
        elif a.stage == "e2-fold":
            r = fold_job(a.fold, a.artifacts, a.max_minutes)
        else:
            r = development(a.artifacts)
            r = {"decision": r["decision"], "metrics": r["metrics"], "counts": r["counts"]}
    print(json.dumps(_plain(r), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

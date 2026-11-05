"""analytics/rebounds/features.py: candidate rows for the rebound model.

    python -m analytics.rebounds.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0] [--games 8]
        [--possessions 40]
writes the candidate frame of every game to OUT_ROOT/analytics/rebounds/features/<season>/<game>.parquet (held-out
games under heldout/<season>/), under a temporary OUT_ROOT when --out is absent, and prints the row counts.

Unit: one missed field-goal attempt (a row of the section 18 rebound_frames table with a rim frame). Each miss
becomes one candidate row per player on court at the rim frame (ten in a full frame) plus one "none" row
(candidate_player_id = -1, the team rebound). The target column is ``is_rebounder``: true on the row of the
play-by-play rebounder (``pbp_rebounder_id``), or on the none row when the play-by-play credits the team.

Which misses are kept (``select_misses``):
    kept, label = pbp_rebounder_id     pbp_rebounder_id not null and confidence_flag not "unmatched"
    kept, label = none                 rim_frame found, a rebound event follows, and the play-by-play names no
                                       player (team rebound: pbp_rebounder_name and pbp_rebounder_id null)
    dropped, counted                   rim_frame -1; no rebound event; confidence_flag "unmatched" with a named
                                       rebounder (roster join failed); identity_match null otherwise; offence
                                       unknown; fewer than two players in the rim frame; the labelled rebounder
                                       not on court in the rim frame
Assumption recorded: the lane spec drops every row with identity_match null, which would also drop every team
rebound (its identity_match is null by construction, section 18). Team rebounds are kept as the "none" class
when a rebound event follows the miss; every other null identity_match is dropped and counted.

Features (all measured at or before the rim frame; the 0.4 s look-back is the frame nearest 0.4 s of game clock
earlier, within 0.2 s, else the speed is NaN):
    distance_to_rim_ft           player to the basket nearest the ball at the rim frame
    distance_to_ball_landing_ft  player to the ball's horizontal position at the rim frame
    angle_to_rim                 absolute angle (radians) of the player seen from the basket, 0 straight in front
    speed_toward_rim_ftps        decrease of distance_to_rim over the 0.4 s before the rim frame, per second
    is_offense                   the player's team shot the ball (section 5 ``team`` of the miss, else possessions)
    nearest_opponent_dist_ft     box-out proxy: distance to the nearest player of the other team at the rim frame
    n_opponents_closer_to_rim    opponents nearer the basket than the player at the rim frame
    ball_apex_ft                 highest ball z in the 2 s of game clock before the rim frame (miss-level)
    shot_distance_ft             ball to basket at the section 17 release_frame; else the ball's horizontal
                                 distance 1 s before the rim frame (miss-level)
    is_none                      1 on the none row; its player features are 0 and its miss-level features copied
    prior_rate                   the candidate's smoothed rebound rate from training folds only (filled by
                                 train.py out of fold; NaN here)

Leakage rules: no rebounder_id, pbp_rebounder_id, pbp_rebounder_name, identity_match, confidence,
rebound_frame or any frame after rim_frame enters a feature; ``analytics.schema.assert_time_forward`` runs on
every frame ``build`` returns; ``assert_no_leak`` refuses a design matrix with a FORBIDDEN column.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analytics import schema

BASKETS = ((5.25, 25.0), (88.75, 25.0))
NONE_ID = -1
LOOKBACK_S = 0.4
LOOKBACK_TOL_S = 0.2
APEX_WINDOW_S = 2.0
SHOT_FALLBACK_S = 1.0
PRIOR = "prior_rate"
KEYS = ["sportvu_game", "event", "candidate_player_id"]
TARGET = "is_rebounder"
STATIC_FEATURES = ["distance_to_rim_ft", "distance_to_ball_landing_ft", "angle_to_rim", "speed_toward_rim_ftps",
                   "is_offense", "nearest_opponent_dist_ft", "n_opponents_closer_to_rim", "ball_apex_ft",
                   "shot_distance_ft", "is_none"]
FEATURES = STATIC_FEATURES + [PRIOR]
FORBIDDEN = frozenset(schema.ALL_EXCLUDED) | frozenset({"rebounder_id", "pbp_rebounder_id", "pbp_rebounder_name",
                                                       "rebound_frame", "identity_match", "confidence",
                                                       "confidence_flag", TARGET, "label_id"})
DROP_REASONS = ("no_rim_frame", "no_rebound_event", "unmatched_named_rebounder", "identity_match_null",
                "offense_unknown", "rim_frame_not_in_frames", "too_few_players", "rebounder_not_on_court")


def feature_columns() -> list:
    """The columns the models are fed, in order (prior_rate is filled per fold by train.py)."""
    return list(FEATURES)


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a target, post-outcome or label column."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("rebounds.features: forbidden columns in the design matrix: %s" % bad)


# ── which misses, which label ─────────────────────────────────────────────────────────────────────
def select_misses(rebound_frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """-> (kept misses with a label_id column, counts). Only id and index columns leave here: the label is
    read once into label_id and the target columns are dropped."""
    rf = rebound_frames
    counts = {"n_in": int(len(rf))}
    rim_ok = rf["rim_frame"].to_numpy() >= 0
    has_ev = rf["rebound_event"].notna().to_numpy()
    pbp_id = rf["pbp_rebounder_id"]
    named = rf["pbp_rebounder_name"].notna().to_numpy()
    unmatched = (rf["confidence_flag"].astype("string") == "unmatched").fillna(False).to_numpy()
    ident_null = rf["identity_match"].isna().to_numpy()
    team = rim_ok & has_ev & ~named & pbp_id.isna().to_numpy()
    player = rim_ok & has_ev & pbp_id.notna().to_numpy() & ~unmatched & ~ident_null
    keep = team | player
    counts["no_rim_frame"] = int((~rim_ok).sum())
    counts["no_rebound_event"] = int((rim_ok & ~has_ev).sum())
    counts["unmatched_named_rebounder"] = int((rim_ok & has_ev & named & pbp_id.isna().to_numpy()).sum())
    counts["identity_match_null"] = int((rim_ok & has_ev & pbp_id.notna().to_numpy() & ~keep).sum())
    counts["team_rebounds_kept"] = int(team.sum())
    counts["player_rebounds_kept"] = int(player.sum())
    out = rf.loc[keep, ["sportvu_game", "event", "period", "clock_s", "rim_frame"]].copy()
    out["label_id"] = np.where(team[keep], NONE_ID, pbp_id[keep].fillna(NONE_ID).astype(int).to_numpy()).astype(int)
    return out.reset_index(drop=True), counts


def offense_side(misses: pd.DataFrame, events: pd.DataFrame | None, possessions: pd.DataFrame | None,
                 frames: pd.DataFrame) -> pd.Series:
    """'home' or 'away' per miss: the section 5 team of the miss event, else the possession's offense_team_id
    mapped through the frames' team_id and home columns; NA when neither is known."""
    side = pd.Series(pd.NA, index=misses.index, dtype="string")
    if events is not None and "team" in events.columns:
        m = events.drop_duplicates("event").set_index("event")["team"].astype("string")
        side = misses["event"].map(m).astype("string")
        side = side.where(side.isin(["home", "away"]), pd.NA)
    if side.isna().any() and possessions is not None and len(possessions):
        team_home = frames.drop_duplicates("team_id").set_index("team_id")["home"].astype(bool)
        pos = possessions[["start_frame", "end_frame", "offense_team_id"]].dropna()
        fill = []
        for r in misses["rim_frame"].to_numpy():
            hit = pos[(pos["start_frame"] <= r) & (pos["end_frame"] >= r)]
            if len(hit) and int(hit["offense_team_id"].iloc[0]) in team_home.index:
                fill.append("home" if bool(team_home.loc[int(hit["offense_team_id"].iloc[0])]) else "away")
            else:
                fill.append(pd.NA)
        side = side.where(side.notna(), pd.Series(fill, index=misses.index, dtype="string"))
    return side


# ── per-game build ────────────────────────────────────────────────────────────────────────────────
def _per_frame(frames: pd.DataFrame) -> pd.DataFrame:
    pf = frames[["frame", "period", "game_clock_s", "ball_x", "ball_y", "ball_z"]].drop_duplicates("frame", keep="first")
    return pf.sort_values("frame", kind="stable").set_index("frame")


def _nearest_earlier(pf: pd.DataFrame, period: int, rim_frame: int, clock_target: float, tol: float):
    """The frame of ``period`` before rim_frame whose clock is nearest clock_target, within tol; else None."""
    sub = pf[(pf["period"] == period) & (pf.index < rim_frame)]
    if sub.empty:
        return None
    d = (sub["game_clock_s"].to_numpy(float) - clock_target)
    k = int(np.argmin(np.abs(d)))
    return int(sub.index[k]) if abs(d[k]) <= tol else None


def build_game(frames: pd.DataFrame, rebound_frames: pd.DataFrame, events: pd.DataFrame | None = None,
               shot_frames: pd.DataFrame | None = None, possessions: pd.DataFrame | None = None,
               heldout: bool = False) -> tuple[pd.DataFrame, dict]:
    """Candidate rows of one game -> (frame, counts). The frame holds KEYS, period, clock_s, heldout, n_candidates,
    the target is_rebounder and the STATIC_FEATURES (prior_rate NaN)."""
    misses, counts = select_misses(rebound_frames)
    for k in DROP_REASONS:
        counts.setdefault(k, 0)
    if misses.empty:
        return _empty(), {**counts, "misses_kept": 0, "rows": 0}
    fr = frames.sort_values("frame", kind="stable").reset_index(drop=True)
    pf = _per_frame(fr)
    side = offense_side(misses, events, possessions, fr)
    release = None
    if shot_frames is not None and len(shot_frames):
        release = shot_frames.drop_duplicates("event").set_index("event")["release_frame"]
    fr_frame = fr["frame"].to_numpy()
    fr_pid = fr["player_id"].to_numpy(); fr_home = fr["home"].to_numpy(bool)
    fr_x = fr["x"].to_numpy(float); fr_y = fr["y"].to_numpy(float)
    rows = []
    for i_miss, m in enumerate(misses.itertuples(index=False)):
        r = int(m.rim_frame)
        if r not in pf.index:
            counts["rim_frame_not_in_frames"] += 1
            continue
        s = side.iloc[i_miss]
        if pd.isna(s):
            counts["offense_unknown"] += 1
            continue
        lo, hi = np.searchsorted(fr_frame, r, "left"), np.searchsorted(fr_frame, r, "right")
        pids, home = fr_pid[lo:hi], fr_home[lo:hi]
        px, py = fr_x[lo:hi], fr_y[lo:hi]
        if len(pids) < 2:
            counts["too_few_players"] += 1
            continue
        if m.label_id != NONE_ID and m.label_id not in set(pids.tolist()):
            counts["rebounder_not_on_court"] += 1
            continue
        row_r = pf.loc[r]
        period, clock = int(row_r["period"]), float(row_r["game_clock_s"])
        ball = (float(row_r["ball_x"]), float(row_r["ball_y"]))
        if not np.isfinite(ball).all():
            ball = (np.nan, np.nan)
            basket = BASKETS[1] if np.nanmean(px) > 47 else BASKETS[0]
        else:
            basket = min(BASKETS, key=lambda b: np.hypot(ball[0] - b[0], ball[1] - b[1]))
        bx, by = basket
        sign = 1.0 if bx < 47 else -1.0
        d_rim = np.hypot(px - bx, py - by)
        d_ball = np.hypot(px - ball[0], py - ball[1])
        angle = np.abs(np.arctan2(py - by, sign * (px - bx)))
        is_off = home == (s == "home")
        # look-back frame for the speed toward the rim
        speed = np.full(len(pids), np.nan)
        back = _nearest_earlier(pf, period, r, clock + LOOKBACK_S, LOOKBACK_TOL_S)
        if back is not None:
            dt = float(pf.loc[back, "game_clock_s"]) - clock
            blo, bhi = np.searchsorted(fr_frame, back, "left"), np.searchsorted(fr_frame, back, "right")
            then = {int(p): (x, y) for p, x, y in zip(fr_pid[blo:bhi], fr_x[blo:bhi], fr_y[blo:bhi])}
            if dt > 0:
                for i, p in enumerate(pids):
                    if int(p) in then:
                        x0, y0 = then[int(p)]
                        speed[i] = (np.hypot(x0 - bx, y0 - by) - d_rim[i]) / dt
        # box-out proxies
        near_opp = np.full(len(pids), np.nan); n_closer = np.zeros(len(pids), dtype=int)
        for i in range(len(pids)):
            opp = home != home[i]
            if opp.any():
                near_opp[i] = float(np.min(np.hypot(px[opp] - px[i], py[opp] - py[i])))
                n_closer[i] = int((d_rim[opp] < d_rim[i]).sum())
        # miss-level features, all before or at the rim frame
        win = pf[(pf["period"] == period) & (pf.index <= r) & (pf["game_clock_s"] <= clock + APEX_WINDOW_S)]
        apex = float(np.nanmax(win["ball_z"].to_numpy(float))) if len(win) and win["ball_z"].notna().any() else np.nan
        shot_d = np.nan
        rel = int(release.get(m.event, -1)) if release is not None else -1
        if rel >= 0 and rel in pf.index and rel <= r:
            shot_d = float(np.hypot(float(pf.loc[rel, "ball_x"]) - bx, float(pf.loc[rel, "ball_y"]) - by))
        if not np.isfinite(shot_d):
            f1 = _nearest_earlier(pf, period, r, clock + SHOT_FALLBACK_S, LOOKBACK_TOL_S)
            if f1 is not None and np.isfinite(float(pf.loc[f1, "ball_x"])):
                shot_d = float(np.hypot(float(pf.loc[f1, "ball_x"]) - bx, float(pf.loc[f1, "ball_y"]) - by))
        n_cand = len(pids) + 1
        base = {"sportvu_game": str(m.sportvu_game), "event": int(m.event), "period": period, "clock_s": clock,
                "heldout": bool(heldout), "n_candidates": n_cand, "ball_apex_ft": apex, "shot_distance_ft": shot_d}
        for i, p in enumerate(pids):
            rows.append({**base, "candidate_player_id": int(p), TARGET: bool(int(p) == m.label_id),
                         "distance_to_rim_ft": float(d_rim[i]), "distance_to_ball_landing_ft": float(d_ball[i]),
                         "angle_to_rim": float(angle[i]), "speed_toward_rim_ftps": float(speed[i]),
                         "is_offense": int(is_off[i]), "nearest_opponent_dist_ft": float(near_opp[i]),
                         "n_opponents_closer_to_rim": int(n_closer[i]), "is_none": 0})
        rows.append({**base, "candidate_player_id": NONE_ID, TARGET: bool(m.label_id == NONE_ID),
                     "distance_to_rim_ft": 0.0, "distance_to_ball_landing_ft": 0.0, "angle_to_rim": 0.0,
                     "speed_toward_rim_ftps": 0.0, "is_offense": 0, "nearest_opponent_dist_ft": 0.0,
                     "n_opponents_closer_to_rim": 0, "is_none": 1})
    out = pd.DataFrame(rows, columns=_columns()) if rows else _empty()
    out[PRIOR] = np.nan
    out = out.astype({"event": "int32", "candidate_player_id": "int32", "period": "int16", "is_offense": "int8",
                      "n_opponents_closer_to_rim": "int16", "is_none": "int8", "n_candidates": "int16"})
    counts["misses_kept"] = int(out[["sportvu_game", "event"]].drop_duplicates().shape[0])
    counts["rows"] = int(len(out))
    schema.assert_time_forward(out, "rebound_frames")
    assert_no_leak(out.drop(columns=[TARGET]))
    return out, counts


def _columns() -> list:
    return ["sportvu_game", "event", "candidate_player_id", "period", "clock_s", "heldout", "n_candidates",
            TARGET] + STATIC_FEATURES


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([], dtype="float64") for c in _columns()}).astype(
        {"sportvu_game": "string", "event": "int32", "candidate_player_id": "int32", "heldout": "bool", TARGET: "bool"})


def build(bundles: Iterable[dict]) -> tuple[pd.DataFrame, dict]:
    """Candidate rows of several games -> (frame, counts per game and in total). Each bundle is a dict with
    frames, rebound_frames and optionally events, shot_frames, possessions, heldout (analytics.rebounds.validate
    builds them from DATA_ROOT or the fixtures)."""
    parts, per_game = [], {}
    for b in bundles:
        df, counts = build_game(b["frames"], b["rebound_frames"], b.get("events"), b.get("shot_frames"),
                                b.get("possessions"), bool(b.get("heldout", False)))
        game = str(b.get("sportvu_game") or (df["sportvu_game"].iat[0] if len(df) else "?"))
        per_game[game] = counts
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else _empty()
    total = {k: int(sum(c.get(k, 0) for c in per_game.values())) for k in
             ["n_in", "team_rebounds_kept", "player_rebounds_kept", "misses_kept", "rows"] + list(DROP_REASONS)}
    schema.assert_time_forward(out, "rebound_frames")
    return out, {"per_game": per_game, "total": total}


# ── command line ──────────────────────────────────────────────────────────────────────────────────
def write_features(cands: pd.DataFrame, out_root) -> list:
    """One parquet per game under OUT_ROOT/analytics/rebounds/features/<season>/ (held-out under heldout/)."""
    from analytics.rebound_frames import season_of_game
    paths = []
    for game, part in cands.groupby("sportvu_game", sort=True):
        try:
            season = season_of_game(str(game))
        except ValueError:
            season = "unknown"
        held = bool(part["heldout"].any())
        p = Path(out_root) / "analytics" / "rebounds" / "features" / ("heldout" if held else "") / season / (str(game) + ".parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(p, index=False)
        paths.append(str(p))
    return paths


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="synthetic games (fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT (analytics.rebounds.validate docstring)")
    ap.add_argument("--out", help="OUT_ROOT; a temporary directory when absent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=8, help="fixture games (every fourth held out)")
    ap.add_argument("--possessions", type=int, default=40, help="fixture possessions per game")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    import tempfile
    from analytics.rebounds import validate as V
    a = parse_args(argv)
    bundles = V.bundles_from_fixtures(a.seed, a.games, 4, a.possessions) if a.fixtures else V.bundles_from_root(a.data)
    cleaned = [V.clean_bundle(b)[0] for b in bundles]
    cands, counts = build(cleaned)
    out_root = a.out or tempfile.mkdtemp(prefix="rebounds_features_")
    paths = write_features(cands, out_root)
    t = counts["total"]
    L = ["# Rebound candidate rows (analytics.rebounds.features)" + (" (fixture output, not a result)" if a.fixtures else ""), "",
         "%d game(s): %d misses in, %d kept (%d team rebounds as none), %d candidate rows; dropped: %s."
         % (len(counts["per_game"]), t["n_in"], t["misses_kept"], t["team_rebounds_kept"], t["rows"],
            ", ".join("%s %d" % (k, t[k]) for k in DROP_REASONS)),
         "Features: %s." % ", ".join(feature_columns()),
         "Written %d file(s) under %s/analytics/rebounds/features/." % (len(paths), out_root), ""]
    print("\n".join(L))
    return {"candidates": cands, "counts": counts, "paths": paths, "out_root": out_root}


if __name__ == "__main__":
    main()


__all__ = ["FEATURES", "STATIC_FEATURES", "PRIOR", "KEYS", "TARGET", "NONE_ID", "FORBIDDEN", "DROP_REASONS",
           "feature_columns", "assert_no_leak", "select_misses", "offense_side", "build_game", "build",
           "write_features", "main"]

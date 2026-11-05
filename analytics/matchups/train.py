"""analytics/matchups/train.py: defender matchup values with a gate, a ridge adjustment and internal validation.

    python -m analytics.matchups.train --fixtures | --data DATA_ROOT [--out OUT_ROOT] --report reports/matchups.md
                                       [--seed 0] [--bootstrap 200] [--folds 5] [--n-min 300]

With --out the season-wide tables go to OUT_ROOT/analytics/matchups/tables/<season>/matchup_defenders.parquet and
matchup_possessions_train.parquet, apart from the per-game files of analytics.matchups.features.

Steps
  1. load the games (analytics.matchups.validate), set the held-out games aside (heldout/ directory or a
     held-out token of sportvu/splits.json), call analytics.eval.refuse_heldout on the training game ids
  2. assign defenders at 1 Hz (analytics.matchups.features) and join the field-goal attempts; the xFG
     probability of each attempt is the xFG lane's out-of-fold p_selected when OUT_ROOT/analytics/xfg/<season>/
     holds it with a verified fold column, else the location-only fallback: analytics.features_xfg features of
     the shooter's position at the release frame, a logistic regression (analytics.train_xfg.fit_models)
     out-of-fold under grouped folds by game; the held-out game gets the model refit on all training games
  3. matchup_possessions -> defender x offensive player pairs and defenders: points allowed per 100 matchup
     possessions minus expected per 100 (positive = more points allowed than expected = worse), a
     300-possession gate (analytics.eval.gated on the defender's distinct possessions) before any ranking,
     bootstrap intervals by game of the difference
  4. ridge regression (sklearn Ridge) of possession points on defender and offensive-player indicator
     columns (each defender's column holds its share of the possession's sampled frames, each offensive
     player's column the share of frames it was guarded); alpha from a grid by out-of-fold mean squared
     error under grouped folds by game; ridge_value = 100 times the defender coefficient (per 100 possessions,
     positive = worse). The out-of-fold MSE of ridge (at the chosen alpha, so slightly optimistic), of the raw
     model (possession points predicted as the sum over the defenders on the floor of their training-fold
     points allowed per matchup possession, times their share) and of the constant (training-fold mean points)
     are compared with paired bootstrap intervals by game
  5. internal validation: assignment stability (share of defender sampled frames whose assignment is the same
     as at the previous sampled frame) and defended FG% consistency (FG% of attempts whose assigned defender
     is within 4 ft at the sampled frame at or before the release vs beyond), both with bootstrap by game
  6. the held-out game(s): matchups built with the training-fitted xFG, scored once by the ridge, raw and
     constant models fit on all training games; stability and defended FG% of that game; never fitted on

Leakage: assignments use positions only; the possession's points and the shot outcome enter only as targets;
schema.assert_time_forward runs on the feature frame (ids plus partial_possession and switches); the ridge
design is built from partial_possession only; the xFG lane's p_selected is used only when its fold column is present and
non-negative on every training row.

Fewer games than folds: the fold count becomes the number of games (leave-one-game-out). One training game:
in-sample fits, labelled in-sample in the report. All numbers of a --fixtures run are fixture output, not
a result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analytics import schema
from analytics.eval import bootstrap_by_game, gated, grouped_folds, paired_bootstrap_by_game, refuse_heldout
from analytics.matchups import features as F
from analytics.matchups import validate as V

log = logging.getLogger("analytics.matchups.train")

WHO = "train_matchups"
FIXTURE_HELDOUT_EVERY = V.FIXTURE_HELDOUT_EVERY
N_MIN = 300
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
MODELS = ("constant", "raw", "ridge")
DEFENDER_COLUMNS = ["def_player_id", "n_possessions", "points_allowed_per100", "expected_per100", "diff_per100",
                    "ridge_value", "lo", "hi", "insufficient"]
MAC_CMD = ("python -m analytics.matchups.train --data \"<DATA_ROOT>\" --out \"<OUT_ROOT>\" --report reports/matchups.md "
           "--seed 0 --bootstrap 1000 --folds 5 --n-min 300")


# ── step 1 and 2: inputs, assignment, xFG ──────────────────────────────────────────────────────────
def split(games: list) -> tuple:
    train = [g for g in games if not g.get("heldout")]
    held = [g for g in games if g.get("heldout")]
    refuse_heldout(sorted(str(g["sportvu_game"]) for g in train), WHO)
    return train, held


def assign_games(games: list) -> dict:
    """frame_assignments and possession_shots (p at the default) per game -> {game id: parts}."""
    out = {}
    for g in games:
        mf, info, counts = F.frame_assignments(g["frames"], g["possessions"])
        shots = F.possession_shots(g, mf, info, None)
        out[str(g["sportvu_game"])] = {"mf": mf, "info": info, "counts": counts, "shots": shots, "game": g}
    return out


def _xfg_frame(parts: dict, heldout: bool) -> pd.DataFrame:
    from analytics.features_xfg import features as xfg_features
    shots = pd.concat([p["shots"] for p in parts.values()], ignore_index=True) if parts else pd.DataFrame(columns=F.SHOT_COLUMNS)
    if len(shots) == 0:
        return pd.DataFrame()
    return xfg_features(F.shot_locations(shots, heldout))


def fallback_xfg(train_parts: dict, held_parts: dict, n_folds: int, seed: int) -> tuple:
    """Location-only xFG probabilities -> (predictions frame with sportvu_game, event, p_selected, fold; note)."""
    from analytics.train_xfg import fit_models
    tr = _xfg_frame(train_parts, False)
    he = _xfg_frame(held_parts, True)
    if len(tr) == 0:
        return pd.DataFrame(columns=["sportvu_game", "event", "p_selected", "fold"]), "no training attempts"
    ok = np.isfinite(tr["distance_ft"].to_numpy(dtype=float))
    tr_ok = tr[ok].reset_index(drop=True)
    games = sorted(tr_ok["sportvu_game"].unique())
    k = min(n_folds, len(games))
    p = np.full(len(tr), np.nan)
    fold = np.full(len(tr), -1)
    rate = float(tr["made"].mean())
    if len(games) >= 2:
        for fi, (tr_g, te_g) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
            m_tr = tr_ok["sportvu_game"].isin(tr_g)
            model = fit_models(tr_ok[m_tr])["logistic"]
            te = tr["sportvu_game"].isin(te_g).to_numpy()
            te_ok = te & ok
            p[te_ok] = model(tr[te_ok])
            p[te & ~ok] = float(tr_ok.loc[m_tr, "made"].mean())
            fold[te] = fi
        note = "location-only logistic, out-of-fold under %d grouped folds by game" % k
    else:
        model = fit_models(tr_ok)["logistic"]
        p[ok] = model(tr_ok)
        p[~ok] = rate
        fold[:] = 0
        note = "location-only logistic, IN-SAMPLE (one training game)"
    out = pd.DataFrame({"sportvu_game": tr["sportvu_game"].astype(str), "event": tr["event"].astype(int), "p_selected": p, "fold": fold})
    if len(he):
        model = fit_models(tr_ok)["logistic"]
        hok = np.isfinite(he["distance_ft"].to_numpy(dtype=float))
        ph = np.full(len(he), rate)
        if hok.any():
            ph[hok] = model(he[hok])
        out = pd.concat([out, pd.DataFrame({"sportvu_game": he["sportvu_game"].astype(str), "event": he["event"].astype(int),
                                            "p_selected": ph, "fold": -1})], ignore_index=True)
    return out, note


def matchup_tables(parts: dict, xfg: pd.DataFrame, default_p: float) -> dict:
    """Apply the xFG probabilities and aggregate -> dict(matchup_possessions, matchup_frames, shots)."""
    pmap = {(str(g), int(e)): float(p) for g, e, p in zip(xfg["sportvu_game"], xfg["event"], xfg["p_selected"])} if len(xfg) else {}
    mps, mfs, shots = [], [], []
    for gid, pt in parts.items():
        s = pt["shots"].copy()
        if len(s):
            s["p"] = [pmap.get((gid, int(e)), default_p) for e in s["event"]]
        mp = F.matchup_possessions(pt["mf"], pt["info"], s, gid)
        if len(mp):
            F.feature_frame(mp)
        mps.append(mp)
        mfs.append(pt["mf"])
        shots.append(s)
    cat = lambda L, cols: pd.concat(L, ignore_index=True) if any(len(x) for x in L) else pd.DataFrame(columns=cols)
    return {"matchup_possessions": cat(mps, F.MATCHUP_COLUMNS), "matchup_frames": cat(mfs, F.FRAME_COLUMNS),
            "shots": cat(shots, F.SHOT_COLUMNS)}


def default_probability(xfg: pd.DataFrame, fallback: float = 0.45) -> float:
    """Mean p_selected over the training rows (out-of-fold, not held out): the p of an attempt the xFG table lacks."""
    if xfg is None or len(xfg) == 0:
        return fallback
    tr = xfg[(pd.to_numeric(xfg["fold"], errors="coerce").fillna(-1) >= 0) & ~V.xfg_heldout_mask(xfg)]
    return float(tr["p_selected"].mean()) if len(tr) else fallback


# ── step 3: defender table ─────────────────────────────────────────────────────────────────────────
def pair_table(mp: pd.DataFrame) -> pd.DataFrame:
    """defender x offensive player: matchup possessions (sum of shares), points allowed, expected, per 100."""
    g = mp.groupby(["def_player_id", "off_player_id"], as_index=False).agg(
        matchup_possessions=("partial_possession", "sum"), points_allowed=("points_allowed_share", "sum"),
        expected=("xfg_expected_share", "sum"), n_games=("sportvu_game", "nunique"))
    den = g["matchup_possessions"].replace(0, np.nan)
    g["points_allowed_per100"] = 100 * g["points_allowed"] / den
    g["expected_per100"] = 100 * g["expected"] / den
    g["diff_per100"] = g["points_allowed_per100"] - g["expected_per100"]
    return g


def _diff_per100(d: pd.DataFrame) -> float:
    den = float(d["partial_possession"].sum())
    return float("nan") if den == 0 else 100.0 * float(d["points_allowed_share"].sum() - d["xfg_expected_share"].sum()) / den


def defender_table(mp: pd.DataFrame, n_min: int = N_MIN, n_boot: int = 200, seed: int = 0,
                   ridge_values: dict | None = None) -> pd.DataFrame:
    """matchup_defenders (DEFENDER_COLUMNS): per defender, the gate on distinct possessions, per-100 metrics and
    the bootstrap-by-game interval of diff_per100."""
    if len(mp) == 0:
        return pd.DataFrame(columns=DEFENDER_COLUMNS)
    dp = mp[["sportvu_game", "possession", "def_player_id"]].drop_duplicates()
    gate = gated(dp, "def_player_id", n_min).groupby("def_player_id")["insufficient"].first()
    rows = []
    for did, d in mp.groupby("def_player_id"):
        n_pos = int(dp[dp["def_player_id"] == did].shape[0])
        den = float(d["partial_possession"].sum())
        pa = 100 * float(d["points_allowed_share"].sum()) / den if den else float("nan")
        ex = 100 * float(d["xfg_expected_share"].sum()) / den if den else float("nan")
        multi = d["sportvu_game"].nunique() >= 2
        ci = bootstrap_by_game(d, "sportvu_game", _diff_per100, n=n_boot, seed=seed) if (multi and n_boot > 0) else \
            {"point": pa - ex, "lo": float("nan"), "hi": float("nan")}
        rows.append({"def_player_id": int(did), "n_possessions": n_pos, "points_allowed_per100": pa, "expected_per100": ex,
                     "diff_per100": pa - ex, "ridge_value": (ridge_values or {}).get(int(did), float("nan")),
                     "lo": ci["lo"], "hi": ci["hi"], "insufficient": bool(gate.get(did, True))})
    out = pd.DataFrame(rows, columns=DEFENDER_COLUMNS)
    return out.sort_values(["insufficient", "diff_per100"]).reset_index(drop=True)


# ── step 4: ridge adjustment and the baseline comparison ───────────────────────────────────────────
def possession_points(games: list) -> dict:
    out = {}
    for g in games:
        p = g["possessions"]
        for gid, pos, pts in zip(p["sportvu_game"].astype(str), p["possession"].astype(int), p["points"].astype(int)):
            out[(gid, pos)] = int(pts)
    return out


def possession_design(mp: pd.DataFrame, points: dict) -> dict:
    """One row per possession: sparse indicator design (defender columns then offensive columns), y and groups."""
    key = list(zip(mp["sportvu_game"].astype(str), mp["possession"].astype(int)))
    rows_idx, uniq = pd.factorize(pd.Series(key), sort=True)
    defs = sorted(mp["def_player_id"].astype(int).unique())
    offs = sorted(mp["off_player_id"].astype(int).unique())
    dcol = {d: i for i, d in enumerate(defs)}
    ocol = {o: len(defs) + i for i, o in enumerate(offs)}
    share = mp["partial_possession"].to_numpy(dtype=float)
    dc = np.array([dcol[int(d)] for d in mp["def_player_id"]])
    oc = np.array([ocol[int(o)] for o in mp["off_player_id"]])
    n = len(uniq)
    X = sp.coo_matrix((np.r_[share, share], (np.r_[rows_idx, rows_idx], np.r_[dc, oc])), shape=(n, len(defs) + len(offs))).tocsr()
    X.sum_duplicates()
    y = np.array([points.get(k, 0) for k in uniq], dtype=float)
    groups = np.array([k[0] for k in uniq])
    columns = ["D%d" % d for d in defs] + ["O%d" % o for o in offs]
    return {"X": X, "y": y, "groups": groups, "defs": defs, "offs": offs, "columns": columns, "keys": list(uniq)}


def _fit_ridge(X, y, alpha: float):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha, fit_intercept=True).fit(X, y)


def raw_model(mp_train: pd.DataFrame) -> dict:
    """Training-fold points allowed per matchup possession per defender, plus the fold mean per unit share."""
    g = mp_train.groupby("def_player_id").agg(pts=("points_allowed_share", "sum"), den=("partial_possession", "sum"))
    den = float(mp_train["partial_possession"].sum())
    mean_rate = float(mp_train["points_allowed_share"].sum()) / den if den else 0.0
    rate = {int(d): (float(r.pts) / float(r.den) if r.den else mean_rate) for d, r in g.iterrows()}
    return {"rate": rate, "mean_rate": mean_rate}


def raw_predict(model: dict, mp_test: pd.DataFrame, keys: list) -> np.ndarray:
    r = model["rate"]
    contrib = mp_test["partial_possession"].to_numpy(dtype=float) * np.array([r.get(int(d), model["mean_rate"]) for d in mp_test["def_player_id"]])
    s: dict = {}
    for g, p, c in zip(mp_test["sportvu_game"].astype(str), mp_test["possession"].astype(int), contrib):
        s[(g, p)] = s.get((g, p), 0.0) + float(c)
    return np.array([s.get(k, 0.0) for k in keys])


def ridge_cv(mp: pd.DataFrame, points: dict, n_folds: int, seed: int, alphas=ALPHAS) -> dict:
    """Alpha by grouped CV by game, out-of-fold predictions of constant, raw and ridge, the final ridge fit."""
    D = possession_design(mp, points)
    games = sorted(set(D["groups"]))
    k = min(n_folds, len(games))
    n = len(D["y"])
    oof = {m: np.full(n, np.nan) for m in MODELS}
    oof_alpha = {a: np.full(n, np.nan) for a in alphas}
    if len(games) >= 2:
        folds = grouped_folds(games, n_folds=k, seed=seed)
        mode = "out-of-fold under %d grouped folds by game (seed %d)" % (k, seed)
    else:
        folds = [(games, games)]
        mode = "IN-SAMPLE (one training game)"
    gm = {"sportvu_game": mp["sportvu_game"].astype(str).to_numpy()}
    for tr_g, te_g in folds:
        te = np.isin(D["groups"], te_g)
        tr = ~te if len(games) >= 2 else te
        m_tr = np.isin(gm["sportvu_game"], tr_g) if len(games) >= 2 else np.ones(len(mp), bool)
        m_te = np.isin(gm["sportvu_game"], te_g)
        oof["constant"][te] = float(D["y"][tr].mean())
        rm = raw_model(mp[m_tr])
        oof["raw"][te] = raw_predict(rm, mp[m_te], [kk for kk, t in zip(D["keys"], te) if t])
        for a in alphas:
            oof_alpha[a][te] = _fit_ridge(D["X"][tr], D["y"][tr], a).predict(D["X"][te])
    mse_alpha = {a: float(np.mean((oof_alpha[a] - D["y"]) ** 2)) for a in alphas}
    best = min(alphas, key=lambda a: mse_alpha[a])
    oof["ridge"] = oof_alpha[best]
    final = _fit_ridge(D["X"], D["y"], best)
    coef = final.coef_
    ridge_values = {d: 100.0 * float(coef[i]) for i, d in enumerate(D["defs"])}
    pred = pd.DataFrame({"sportvu_game": D["groups"], "y": D["y"], **{"p_" + m: oof[m] for m in MODELS}})
    return {"alpha": best, "mse_alpha": mse_alpha, "pred": pred, "ridge_values": ridge_values, "model": final,
            "design": D, "mode": mode, "folds": k, "raw": raw_model(mp)}


def _mse(d: pd.DataFrame, col: str) -> float:
    return float(np.mean((d[col] - d["y"]) ** 2))


def compare_models(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """MSE per model with bootstrap by game, and paired differences ridge minus raw, ridge minus constant."""
    out = {"mse": {}, "paired": {}}
    multi = pred["sportvu_game"].nunique() >= 2
    for m in MODELS:
        if multi and n_boot > 0:
            out["mse"][m] = bootstrap_by_game(pred, "sportvu_game", lambda d, c="p_" + m: _mse(d, c), n=n_boot, seed=seed)
        else:
            out["mse"][m] = {"point": _mse(pred, "p_" + m), "lo": float("nan"), "hi": float("nan")}
    for a, b in (("ridge", "raw"), ("ridge", "constant")):
        if multi and n_boot > 0:
            out["paired"]["%s_minus_%s" % (a, b)] = paired_bootstrap_by_game(
                pred, "sportvu_game", lambda d, c="p_" + a: _mse(d, c), lambda d, c="p_" + b: _mse(d, c), n=n_boot, seed=seed)
        else:
            diff = _mse(pred, "p_" + a) - _mse(pred, "p_" + b)
            out["paired"]["%s_minus_%s" % (a, b)] = {"diff": diff, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    return out


# ── step 5: internal validation ────────────────────────────────────────────────────────────────────
def _stability(d: pd.DataFrame) -> float:
    c = d["changed"].dropna()
    return float("nan") if len(c) == 0 else 1.0 - float(c.mean())


def stability(mf: pd.DataFrame, n_boot: int, seed: int) -> dict:
    d = mf.dropna(subset=["changed"])
    if len(d) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if d["sportvu_game"].nunique() >= 2 and n_boot > 0:
        r = bootstrap_by_game(d, "sportvu_game", _stability, n=n_boot, seed=seed)
    else:
        r = {"point": _stability(d), "lo": float("nan"), "hi": float("nan")}
    return {**r, "n": int(len(d))}


def _fg(d: pd.DataFrame, near: bool) -> float:
    m = d["def_dist_ft"] <= F.DEFENDED_FT if near else d["def_dist_ft"] > F.DEFENDED_FT
    return float("nan") if m.sum() == 0 else float(d.loc[m, "made"].astype(float).mean())


def defended_fg(shots: pd.DataFrame, n_boot: int, seed: int) -> dict:
    d = shots.dropna(subset=["def_dist_ft"])
    out = {"n_near": int((d["def_dist_ft"] <= F.DEFENDED_FT).sum()), "n_far": int((d["def_dist_ft"] > F.DEFENDED_FT).sum())}
    nan = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if len(d) == 0:
        return {**out, "near": nan, "far": nan, "far_minus_near": {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}}
    multi = d["sportvu_game"].nunique() >= 2 and n_boot > 0
    if multi:
        out["near"] = bootstrap_by_game(d, "sportvu_game", lambda x: _fg(x, True), n=n_boot, seed=seed)
        out["far"] = bootstrap_by_game(d, "sportvu_game", lambda x: _fg(x, False), n=n_boot, seed=seed)
        out["far_minus_near"] = paired_bootstrap_by_game(d, "sportvu_game", lambda x: _fg(x, False), lambda x: _fg(x, True), n=n_boot, seed=seed)
    else:
        out["near"] = {**nan, "point": _fg(d, True)}
        out["far"] = {**nan, "point": _fg(d, False)}
        out["far_minus_near"] = {"diff": _fg(d, False) - _fg(d, True), "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    return out


# ── step 6: held-out, scored once ──────────────────────────────────────────────────────────────────
def score_heldout(held_tables: dict, held_games: list, cv: dict, n_boot: int, seed: int) -> dict:
    mp = held_tables["matchup_possessions"]
    if len(mp) == 0:
        return {}
    D = possession_design(mp, possession_points(held_games))
    # align the held-out design to the training columns (unseen players get no column: the intercept carries them)
    cols = {c: i for i, c in enumerate(cv["design"]["columns"])}
    X = sp.lil_matrix((len(D["y"]), len(cols)))
    Xc = D["X"].tocoo()
    for r, c, v in zip(Xc.row, Xc.col, Xc.data):
        j = cols.get(D["columns"][c])
        if j is not None:
            X[r, j] = v
    pred = pd.DataFrame({"sportvu_game": D["groups"], "y": D["y"],
                         "p_constant": float(cv["pred"]["y"].mean()),
                         "p_raw": raw_predict(cv["raw"], mp, D["keys"]),
                         "p_ridge": cv["model"].predict(X.tocsr())})
    return {"games": sorted(set(D["groups"])), "n_possessions": int(len(D["y"])), "n_rows": int(len(mp)),
            "mse": {m: _mse(pred, "p_" + m) for m in MODELS},
            "stability": stability(held_tables["matchup_frames"], 0, seed),
            "defended_fg": defended_fg(held_tables["shots"], 0, seed)}


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def _f(v, nd=2) -> str:
    try:
        return "n/a" if v is None or not np.isfinite(float(v)) else ("%%.%df" % nd) % float(v)
    except (TypeError, ValueError):
        return "n/a"


def _ci(d: dict, key="point", nd=3) -> str:
    if d is None or not np.isfinite(float(d.get(key, np.nan))):
        return "n/a"
    if not np.isfinite(float(d.get("lo", np.nan))):
        return _f(d[key], nd)
    return "%s [%s, %s]" % (_f(d[key], nd), _f(d["lo"], nd), _f(d["hi"], nd))


def _defender_rows(t: pd.DataFrame) -> list:
    return ["| %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
        r.def_player_id, r.n_possessions, _f(r.points_allowed_per100, 1), _f(r.expected_per100, 1), _f(r.diff_per100, 1),
        _f(r.ridge_value, 1), _f(r.lo, 1), _f(r.hi, 1), "yes" if r.insufficient else "no") for r in t.itertuples(index=False)]


def write_report(path: Path, ctx: dict) -> str:
    fixture = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fixture else ""
    L = ["# Matchups: individual defender assessment (analytics/matchups)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fixture:
        L += ["**Every number below is fixture output, not a result**: the inputs are the synthetic season of "
              "tests/_fx_matchups.py (or analytics.fixtures), drawn from a seeded generator. The real run is the Mac command at the end.", ""]
    L += ["## Inputs" + tag, "", "| game | held out | possessions | sampled frames | matchup rows | attempts |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ctx["inputs"]:
        L.append("| %s | %s | %d | %d | %d | %d |" % (r["sportvu_game"], "yes" if r["heldout"] else "no", r["possessions"], r["frames_sampled"], r["rows"], r["attempts"]))
    L += ["", "Training: %d game(s) (%s). Held out, set aside before any fit: %d game(s) (%s). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json."
          % (len(ctx["train_ids"]), ", ".join(ctx["train_ids"]) or "none", len(ctx["held_ids"]), ", ".join(ctx["held_ids"]) or "none"),
          "xFG probabilities: %s." % ctx["xfg_note"],
          "Skipped: %d possession(s) without an offence id, %d without frames, %d sampled frame(s) without both squads." % (ctx["skipped"]["no_offense"], ctx["skipped"]["no_frames"], ctx["skipped"]["frames_skipped"]), "",
          "## Protocol", "",
          "Assignment at 1 Hz by `scipy.optimize.linear_sum_assignment` on distance to the offensive player plus 0.5 times distance to the midpoint between that player and the attacked basket (one basket per offense team and period, by majority over the possessions' last-third offensive centroids); positions only. "
          "Share = fraction of the possession's sampled frames assigned. Points enter only as targets; `schema.assert_time_forward` passed on the feature frame (%s); the ridge design is built from partial_possession only." % ", ".join(F.feature_columns()),
          "Gate: %d distinct possessions per defender (`analytics.eval.gated`) before any ranking. Intervals: 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)." % (ctx["n_min"], ctx["bootstrap"], ctx["seed"]), "",
          "## Assignment stability" + tag, "",
          "Share of defender sampled frames whose assignment equals the previous sampled frame: %s over %d defender-frames (%d games)." % (_ci(ctx["stability"]), ctx["stability"]["n"], len(ctx["train_ids"])),
          "Switches per possession per defender: mean %s, share of defender-possessions with no switch %s." % (_f(ctx["switch_mean"], 2), _f(ctx["no_switch_share"], 3)), "",
          "## Defended FG% consistency" + tag, "",
          "| assigned defender | attempts | FG% |", "| --- | --- | --- |",
          "| within %.0f ft | %d | %s |" % (F.DEFENDED_FT, ctx["defended"]["n_near"], _ci(ctx["defended"]["near"])),
          "| beyond %.0f ft | %d | %s |" % (F.DEFENDED_FT, ctx["defended"]["n_far"], _ci(ctx["defended"]["far"])),
          "", "Beyond minus within: %s (paired by game; excludes zero: %s)." % (_ci(ctx["defended"]["far_minus_near"], "diff"), ctx["defended"]["far_minus_near"]["excludes_zero"]), "",
          "## Baseline comparison: possession points, %s" % ctx["cv"]["mode"] + tag, "",
          "Ridge alpha %s chosen from %s by out-of-fold MSE (%s)." % (_f(ctx["cv"]["alpha"], 1), list(ALPHAS), ", ".join("%s: %s" % (_f(a, 1), _f(m, 4)) for a, m in ctx["cv"]["mse_alpha"].items())), "",
          "| model | out-of-fold MSE |", "| --- | --- |"]
    for m in MODELS:
        L.append("| %s | %s |" % (m, _ci(ctx["compare"]["mse"][m], "point", 4)))
    for k, d in ctx["compare"]["paired"].items():
        L.append("| %s | %s (excludes zero: %s) |" % (k.replace("_", " "), _ci(d, "diff", 4), d["excludes_zero"]))
    L += ["", "The ridge MSE is at the alpha chosen on the same folds, so it is slightly optimistic; the raw and constant models carry no such choice."]
    if ctx["n_defenders_multi_game"] == 0:
        L.append("Every defender is in one game on this input: raw and ridge out-of-fold equal the constant, so the comparison above carries no information about the defenders.")
    else:
        L.append("%d of %d defenders appear in two or more training games (only those can move an out-of-fold prediction)." % (ctx["n_defenders_multi_game"], ctx["n_defenders"]))
    L += ["",
          "## Defenders" + tag, "",
          "%d defenders, %d pass the %d-possession gate. Sorted by diff_per100 (points allowed minus expected per 100 matchup possessions; positive = worse); ridge_value is the ridge coefficient per 100 possessions (positive = worse)."
          % (ctx["n_defenders"], ctx["n_gated"], ctx["n_min"]), ""]
    head = ["| def_player_id | n_possessions | points_allowed_per100 | expected_per100 | diff_per100 | ridge_value | lo | hi | insufficient |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ok = ctx["defenders"][~ctx["defenders"]["insufficient"]]
    if len(ok):
        L += ["### Top %d (fewest points allowed over expected, gated)" % min(10, len(ok)), ""] + head + _defender_rows(ok.head(10))
        L += ["", "### Bottom %d (gated)" % min(10, len(ok)), ""] + head + _defender_rows(ok.tail(10).iloc[::-1])
    else:
        L += ["No defender passes the gate, so there is no ranking. The ungated rows, every one flagged insufficient, for inspection only:", ""] + head + \
             _defender_rows(ctx["defenders"].head(10))
    L += ["", "## Held-out game(s), scored once" + tag, ""]
    h = ctx["heldout"]
    if h:
        L += ["%s: %d possessions, %d matchup rows; never fitted or tuned on; point estimates only." % (", ".join(h["games"]), h["n_possessions"], h["n_rows"]), "",
              "| model | MSE |", "| --- | --- |"] + ["| %s | %s |" % (m, _f(h["mse"][m], 4)) for m in MODELS] + \
             ["", "Assignment stability %s; defended FG%% within %.0f ft %s (%d attempts) vs beyond %s (%d attempts)." % (
                 _f(h["stability"]["point"], 3), F.DEFENDED_FT, _f(h["defended_fg"]["near"]["point"], 3), h["defended_fg"]["n_near"],
                 _f(h["defended_fg"]["far"]["point"], 3), h["defended_fg"]["n_far"])]
    else:
        L.append("No held-out game in the inputs.")
    L += ["", "## Status", ""]
    if fixture:
        L.append("Fixture output, not a result. The real run, on the Mac with the full DATA_ROOT (frames, play-by-play, possessions, shot_frames, rosters for 2015-16) and the xFG lane's predictions under OUT_ROOT/analytics/xfg/2015-16/, is")
    else:
        L.append("Run on the DATA_ROOT given; rerun with")
    L += ["", "    " + MAC_CMD, "",
          "## Assumptions", "",
          "- The attacked basket is the one nearer the offensive centroid over the possession's sampled frames (positions only).",
          "- A possession's points go to the rows of the offensive player who took its last field-goal attempt; possessions without an attempt (turnovers, free throws without a shot) count 0 points and 0 expected for every matchup.",
          "- Shot value is 3 when the play-by-play text says 3-pt, else by 22 ft from the basket at the release frame.",
          "- switches counts, per defender and possession, the sampled frames at which its assignment changed; the count repeats on each of that defender's pair rows.",
          "- The gate counts distinct possessions in which the defender was on the floor; n_possessions is that count, the per-100 denominators are the summed shares.",
          "- Unseen players in the held-out game have no ridge column; the intercept carries them.",
          "- No NBA.com matchup table exists for 2015-16, so validation is internal (stability, defended FG% consistency, held-out MSE); nothing here is checked against an external matchup record.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def run(games: list, xfg: pd.DataFrame | None, n_folds: int, seed: int, n_boot: int, n_min: int, fixtures: bool, command: str) -> dict:
    train, held = split(games)
    if not train:
        raise SystemExit("%s: no training games after setting the held-out games aside" % WHO)
    train_parts = assign_games(train)
    held_parts = assign_games(held)
    if xfg is not None:
        note = "the xFG lane's out-of-fold p_selected (%d rows, fold column verified)" % len(xfg)
    else:
        xfg, note = fallback_xfg(train_parts, held_parts, n_folds, seed)
    default_p = default_probability(xfg)
    tt = matchup_tables(train_parts, xfg, default_p)
    ht = matchup_tables(held_parts, xfg, default_p)
    mp = tt["matchup_possessions"]
    if len(mp) == 0:
        raise SystemExit("%s: no matchup rows from the training games" % WHO)
    cv = ridge_cv(mp, possession_points(train), n_folds, seed)
    compare = compare_models(cv["pred"], n_boot, seed)
    defenders = defender_table(mp, n_min, n_boot, seed, cv["ridge_values"])
    pairs = pair_table(mp)
    dsw = mp.groupby(["sportvu_game", "possession", "def_player_id"])["switches"].first()
    inputs = []
    for gid, pt in list(train_parts.items()) + list(held_parts.items()):
        t = tt if gid in train_parts else ht
        inputs.append({"sportvu_game": gid, "heldout": gid in held_parts, "possessions": pt["counts"]["possessions"],
                       "frames_sampled": pt["counts"]["frames_sampled"],
                       "rows": int((t["matchup_possessions"]["sportvu_game"] == gid).sum()), "attempts": int(len(pt["shots"]))})
    skipped = {k: int(sum(pt["counts"][k] for pt in list(train_parts.values()) + list(held_parts.values()))) for k in ("no_offense", "no_frames", "frames_skipped")}
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": command, "fixtures": fixtures,
           "inputs": inputs, "train_ids": sorted(train_parts), "held_ids": sorted(held_parts), "xfg_note": note, "skipped": skipped,
           "n_min": n_min, "bootstrap": n_boot, "seed": seed, "stability": stability(tt["matchup_frames"], n_boot, seed),
           "switch_mean": float(dsw.mean()) if len(dsw) else float("nan"), "no_switch_share": float((dsw == 0).mean()) if len(dsw) else float("nan"),
           "defended": defended_fg(tt["shots"], n_boot, seed), "cv": cv, "compare": compare, "defenders": defenders, "pairs": pairs,
           "n_defenders": int(len(defenders)), "n_gated": int((~defenders["insufficient"]).sum()),
           "n_defenders_multi_game": int((mp.groupby("def_player_id")["sportvu_game"].nunique() >= 2).sum()),
           "heldout": score_heldout(ht, held, cv, n_boot, seed) if held else {},
           "tables": {"train": tt, "heldout": ht}, "xfg": xfg}
    return ctx


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="the synthetic season (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", default=None, help="OUT_ROOT: reads analytics/xfg/<season>/ predictions, writes analytics/matchups/")
    ap.add_argument("--report", default="reports/matchups.md")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--n-games", type=int, default=8, help="fixtures only")
    ap.add_argument("--heldout-every", type=int, default=FIXTURE_HELDOUT_EVERY, help="fixtures only: every k-th game is held out")
    ap.add_argument("--n-possessions", type=int, default=None, help="fixtures only: possessions per game (default: the generator's season)")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    a = parse_args(argv)
    games = V.fixture_games(a.seed, a.n_games, a.heldout_every, a.n_possessions) if a.fixtures else V.load_games(a.data, a.season)
    if not games:
        raise SystemExit("%s: no games found" % WHO)
    V.validate_games(games)
    xfg = V.load_xfg(a.out, a.season) if a.out else None
    cmd = "python -m analytics.matchups.train " + " ".join(argv if argv is not None else sys.argv[1:])
    ctx = run(games, xfg, a.folds, a.seed, a.bootstrap, a.n_min, a.fixtures, cmd)
    text = write_report(Path(a.report), ctx)
    print(text)
    if a.out:
        d = Path(a.out) / "analytics" / "matchups" / "tables" / a.season     # season-wide tables, apart from the per-game files
        d.mkdir(parents=True, exist_ok=True)
        ctx["defenders"].to_parquet(d / "matchup_defenders.parquet", index=False)
        ctx["tables"]["train"]["matchup_possessions"].to_parquet(d / "matchup_possessions_train.parquet", index=False)
        print("wrote %s" % d)
    return ctx


if __name__ == "__main__":
    main()

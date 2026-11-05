"""analytics/playcalling/train.py: PPP by action type and coverage, the ridge adjustment and the recommendation
table, every number an association with a bootstrap interval by game and the 300-possession gate.

    python -m analytics.playcalling.train --fixtures | --data DATA_ROOT [--out OUT_ROOT] --report reports/playcalling.md
                                          [--folds 5] [--seed 0] [--bootstrap 200] [--n-min 300] [--season 2015-16]

Steps
  0. Build the action table (analytics.playcalling.features), set every held-out row aside (heldout column: a
     sportvu/splits.json token, the CLE arena or a heldout/ folder), call analytics.eval.refuse_heldout on the
     training game ids and analytics.schema.assert_time_forward on the feature frame.
  1. PPP by action type: mean ppp with the 2.5 and 97.5 percentiles of bootstrap resamples of whole games, and
     analytics.eval.gated at n_min rows (300 by default; each action row is one possession bearing that action).
  2. PPP by action type x coverage, the same way (coverage is the hand screen label, else the prediction, else
     "none"). The same two tables for epv_gain when epv_states exist (reported, not written).
  3. Ridge regression of possession points on indicator columns of action type, coverage, action type x
     coverage, offensive lineup and defensive lineup. Alpha from the grid ALPHAS by grouped cross-validation by
     game; the reported out-of-fold MSE is nested (alpha chosen inside each outer fold on its training games
     only). Baselines: the training-fold mean, and the training-fold PPP by action type (table 1 as a
     predictor). Paired bootstrap by game of the MSE differences.
  4. The recommendation table: per coverage, every action type's adjusted PPP (ridge prediction with the
     lineup effects averaged over the training rows) with a bootstrap interval from refits on resampled games,
     n, the gate, and recommended = the highest adjusted PPP among the sufficient cells of that coverage.
     Labelled association: it ranks what happened after each call against each coverage, holding lineups
     fixed in a linear sense; it does not say what would happen if the call were changed.
  5. Held-out rows, when any: scored once by the final ridge and the two baselines (MSE), never fitted on.

Fewer games than folds: n_folds becomes the number of games. One game: in-sample metrics, labelled so.
Deep models are out of scope (baselines first; ridge is the only fitted model).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analytics import schema
from analytics.eval import bootstrap_by_game, gated, grouped_folds, paired_bootstrap_by_game, refuse_heldout
from analytics.playcalling import ACTION_TYPES, COVERAGES, N_MIN, SEASON, WHO
from analytics.playcalling import features as F

ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
BASELINES = ("mean", "by_action", "ridge")
RIDGE_GROUPS = ("action_type", "coverage", "interaction", "off_lineup", "def_lineup")
FULL_RUN_CMD = ("python -m analytics.playcalling.train --data DATA_ROOT --out OUT_ROOT --report reports/playcalling.md "
                "--folds 5 --seed 0 --bootstrap 1000 --n-min 300")
ASSOCIATION_LINE = ("Association only, not causal: every number below is an observational comparison of what followed "
                    "each action against each coverage, with a bootstrap interval by game and a sample gate. Nothing here "
                    "estimates what would happen if a different play were called.")


# ── split ───────────────────────────────────────────────────────────────────────────────────────────
def training_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, heldout) rows of the action table; the guard runs on the training game ids and the leakage check on
    the training feature frame."""
    held = table["heldout"].astype(bool)
    train, heldout = table[~held].reset_index(drop=True), table[held].reset_index(drop=True)
    refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    F.features(train)
    return train, heldout


# ── tables 1 and 2 ──────────────────────────────────────────────────────────────────────────────────
def ppp_by_action(table: pd.DataFrame, by_coverage: bool = False, outcome: str = "ppp", n_boot: int = 200, seed: int = 0,
                  n_min: int = N_MIN) -> pd.DataFrame:
    """Mean outcome per action type (and coverage) with bootstrap intervals by game and the gate.
    Columns: action_type, coverage (null when not by coverage), n, ppp, lo, hi, insufficient."""
    t = table[table[outcome].notna()].copy()
    t["_entity"] = t["action_type"].astype(str) + ("|" + t["coverage"].astype(str) if by_coverage else "")
    t = gated(t, "_entity", n_min)
    rows = []
    for ent, grp in t.groupby("_entity", sort=True):
        d = pd.DataFrame({"g": grp["sportvu_game"].astype(str).to_numpy(), "y": grp[outcome].astype(float).to_numpy()})
        r = bootstrap_by_game(d, "g", lambda s: float(s["y"].mean()), n=n_boot, seed=seed)
        if d["g"].nunique() < 2:
            r["lo"], r["hi"] = float("nan"), float("nan")
        at, cov = (ent.split("|", 1) if by_coverage else (ent, None))
        rows.append({"action_type": at, "coverage": cov, "n": int(len(grp)), "ppp": r["point"], "lo": r["lo"], "hi": r["hi"],
                     "insufficient": bool(grp["insufficient"].iloc[0])})
    out = pd.DataFrame(rows, columns=["action_type", "coverage", "n", "ppp", "lo", "hi", "insufficient"])
    return out.sort_values(["coverage", "ppp"], ascending=[True, False], na_position="first").reset_index(drop=True)


# ── ridge ───────────────────────────────────────────────────────────────────────────────────────────
class Design:
    """Indicator columns of the ridge, fitted on a training frame and applied to any frame (unseen levels get no
    column). Levels are per group so coefficients can be read back by name."""

    def __init__(self, train: pd.DataFrame):
        self.levels = {g: sorted(self._col(train, g).unique().tolist()) for g in RIDGE_GROUPS}
        self.index = {}
        k = 0
        for g in RIDGE_GROUPS:
            for lv in self.levels[g]:
                self.index[(g, lv)] = k
                k += 1
        self.n_cols = k

    @staticmethod
    def _col(frame: pd.DataFrame, g: str) -> pd.Series:
        if g == "interaction":
            return frame["action_type"].astype(str) + "|" + frame["coverage"].astype(str)
        return frame[g].fillna("unknown").astype(str)

    def matrix(self, frame: pd.DataFrame) -> sp.csr_matrix:
        rows, cols = [], []
        for g in RIDGE_GROUPS:
            v = self._col(frame, g).to_numpy()
            for i, lv in enumerate(v):
                j = self.index.get((g, lv))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows))
        return sp.csr_matrix((data, (rows, cols)), shape=(len(frame), self.n_cols))

    def group_slice(self, g: str) -> list[int]:
        return [self.index[(g, lv)] for lv in self.levels[g]]


def fit_ridge(train: pd.DataFrame, alpha: float):
    """-> (design, model). The guard against held-out rows reaching a fit runs here, on every fit."""
    from sklearn.linear_model import Ridge
    if train["heldout"].any():
        raise ValueError("%s: a held-out row reached fit_ridge" % WHO)
    d = Design(train)
    X = d.matrix(train)
    y = train["ppp"].astype(float).to_numpy()
    m = Ridge(alpha=alpha, fit_intercept=True, solver="auto").fit(X, y)
    return d, m


def predict_ridge(model, frame: pd.DataFrame) -> np.ndarray:
    d, m = model
    return np.asarray(m.predict(d.matrix(frame)), dtype=float)


def fit_baselines(train: pd.DataFrame) -> dict:
    if train["heldout"].any():
        raise ValueError("%s: a held-out row reached fit_baselines" % WHO)
    mean = float(train["ppp"].mean())
    by_action = train.groupby("action_type")["ppp"].mean().to_dict()
    return {"mean": lambda f, r=mean: np.full(len(f), r),
            "by_action": lambda f, ba=by_action, r=mean: f["action_type"].map(ba).fillna(r).to_numpy(dtype=float)}


def mse(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def choose_alpha(train: pd.DataFrame, n_folds: int, seed: int, alphas=ALPHAS) -> tuple[float, dict]:
    """Alpha with the lowest grouped-CV MSE by game -> (alpha, {alpha: mse}). One game: the middle of the grid."""
    games = sorted(train["sportvu_game"].astype(str).unique())
    if len(games) < 2:
        return float(alphas[len(alphas) // 2]), {}
    k = min(n_folds, len(games))
    folds = grouped_folds(games, n_folds=k, seed=seed)
    scores = {}
    for a in alphas:
        err, n = 0.0, 0
        for tr_g, te_g in folds:
            tr, te = train[train["sportvu_game"].isin(tr_g)], train[train["sportvu_game"].isin(te_g)]
            p = predict_ridge(fit_ridge(tr, a), te)
            err += float(((te["ppp"].to_numpy(dtype=float) - p) ** 2).sum())
            n += len(te)
        scores[float(a)] = err / max(n, 1)
    best = min(scores, key=lambda a: (scores[a], a))
    return best, scores


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int, inner_folds: int = 3) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions of the mean, by-action and ridge models -> (frame with p_<model>, protocol note).
    The ridge alpha is chosen inside each outer fold (nested grouped CV)."""
    games = sorted(train["sportvu_game"].astype(str).unique())
    pred = train.copy()
    if len(games) < 2:
        a, _ = choose_alpha(train, n_folds, seed)
        base = fit_baselines(train)
        for m in ("mean", "by_action"):
            pred["p_" + m] = base[m](train)
        pred["p_ridge"] = predict_ridge(fit_ridge(train, a), train)
        return pred, {"mode": "in-sample", "folds": 0, "games": 1, "alphas": [a],
                      "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE"}
    k = min(n_folds, len(games))
    for m in BASELINES:
        pred["p_" + m] = np.nan
    alphas = []
    for tr_g, te_g in grouped_folds(games, n_folds=k, seed=seed):
        tr_m, te_m = train["sportvu_game"].isin(tr_g), train["sportvu_game"].isin(te_g)
        tr, te = train[tr_m], train[te_m]
        base = fit_baselines(tr)
        for m in ("mean", "by_action"):
            pred.loc[te_m, "p_" + m] = base[m](te)
        a, _ = choose_alpha(tr, inner_folds, seed)
        alphas.append(a)
        pred.loc[te_m, "p_ridge"] = predict_ridge(fit_ridge(tr, a), te)
    assert pred[["p_" + m for m in BASELINES]].notna().all().all()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d), ridge alpha chosen by %d inner grouped folds" % (k, seed, inner_folds)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games" % (len(games), n_folds)
    return pred, {"mode": "out-of-fold", "folds": k, "games": len(games), "alphas": alphas, "text": text}


def score(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{model: {point, lo, hi}} of the MSE with bootstrap by game, plus paired differences ridge minus baseline."""
    d = pd.DataFrame({"g": pred["sportvu_game"].astype(str).to_numpy(), "y": pred["ppp"].astype(float).to_numpy()})
    for m in BASELINES:
        d["p_" + m] = pred["p_" + m].astype(float).to_numpy()
    one = d["g"].nunique() < 2
    out = {"mse": {}, "paired": {}}
    for m in BASELINES:
        r = bootstrap_by_game(d, "g", lambda s, m=m: mse(s["y"], s["p_" + m]), n=n_boot, seed=seed)
        if one:
            r["lo"], r["hi"] = float("nan"), float("nan")
        out["mse"][m] = r
    for m in ("mean", "by_action"):
        if one:
            out["paired"][m] = {"diff": mse(d["y"], d["p_ridge"]) - mse(d["y"], d["p_" + m]), "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
        else:
            out["paired"][m] = paired_bootstrap_by_game(d, "g", lambda s: mse(s["y"], s["p_ridge"]), lambda s, m=m: mse(s["y"], s["p_" + m]),
                                                        n=n_boot, seed=seed)
    return out


# ── table 4 ─────────────────────────────────────────────────────────────────────────────────────────
def _adjusted_grid(model, ref: pd.DataFrame) -> dict:
    """{(coverage, action_type): adjusted ppp} from one fitted ridge: intercept + mean lineup contribution over the
    reference rows + the action, coverage and interaction coefficients."""
    d, m = model
    w = np.asarray(m.coef_, dtype=float)
    X = d.matrix(ref)
    lu = d.group_slice("off_lineup") + d.group_slice("def_lineup")
    base = float(m.intercept_) + (float(np.mean(X[:, lu] @ w[lu])) if lu else 0.0)
    out = {}
    for c in d.levels["coverage"]:
        for a in d.levels["action_type"]:
            v = base + w[d.index[("action_type", a)]] + w[d.index[("coverage", c)]]
            j = d.index.get(("interaction", a + "|" + c))
            out[(c, a)] = v + (w[j] if j is not None else 0.0)
    return out


def adjusted_table(train: pd.DataFrame, alpha: float, n_boot: int = 200, seed: int = 0, n_min: int = N_MIN) -> pd.DataFrame:
    """playcalling_adjusted: coverage, action_type, adjusted_ppp, lo, hi, n, insufficient, recommended."""
    model = fit_ridge(train, alpha)
    point = _adjusted_grid(model, train)
    counts = train.groupby(["coverage", "action_type"]).size().to_dict()
    games = sorted(train["sportvu_game"].astype(str).unique())
    draws = {k: [] for k in point}
    if len(games) >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        groups = {g: train[train["sportvu_game"].astype(str) == g] for g in games}
        for _ in range(n_boot):
            pick = rng.integers(0, len(games), size=len(games))
            s = pd.concat([groups[games[i]] for i in pick], ignore_index=True)
            grid = _adjusted_grid(fit_ridge(s, alpha), train)
            for k in point:
                draws[k].append(grid.get(k, np.nan))
    rows = []
    for (c, a), v in point.items():
        if counts.get((c, a), 0) == 0:          # a cell never observed has no row (its value is pure extrapolation)
            continue
        arr = np.array(draws[(c, a)], dtype=float)
        lo, hi = (np.nanpercentile(arr, [2.5, 97.5]) if arr.size and np.isfinite(arr).any() else (np.nan, np.nan))
        n = int(counts.get((c, a), 0))
        rows.append({"coverage": c, "action_type": a, "adjusted_ppp": float(v), "lo": float(lo), "hi": float(hi), "n": n,
                     "insufficient": n < n_min})
    out = pd.DataFrame(rows, columns=["coverage", "action_type", "adjusted_ppp", "lo", "hi", "n", "insufficient"])
    out["recommended"] = False
    for c, grp in out.groupby("coverage"):
        ok = grp[~grp["insufficient"]]
        if len(ok):
            out.loc[ok["adjusted_ppp"].idxmax(), "recommended"] = True
    return out.sort_values(["coverage", "adjusted_ppp"], ascending=[True, False]).reset_index(drop=True)


# ── report ──────────────────────────────────────────────────────────────────────────────────────────
def _f(v, nd=3) -> str:
    try:
        return "n/a" if v is None or not np.isfinite(float(v)) else ("%%.%df" % nd) % float(v)
    except (TypeError, ValueError):
        return "n/a"


def _ci(r: dict, nd=3) -> str:
    return "%s [%s, %s]" % (_f(r["point"], nd), _f(r["lo"], nd), _f(r["hi"], nd))


def _md_table(df: pd.DataFrame, cols, nd=3) -> list[str]:
    L = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df[cols].itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append(_f(v, nd))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("all")
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return L


def write_report(path: Path, ctx: dict) -> str:
    fixture = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fixture else ""
    L = ["# Play calling: " + ASSOCIATION_LINE, "",
         "Written %s by `%s`.%s" % (ctx["written"], ctx["command"], " FIXTURE OUTPUT, NOT A RESULT: the inputs are synthetic." if fixture else ""), "",
         "## Inputs", ""]
    for name, entry in ctx["validation"].items():
        if isinstance(entry, dict):
            L.append("- %s: %s" % (name, ", ".join("%s=%s" % (k, json.dumps(v, default=str)) for k, v in entry.items() if k != "schema")))
        else:
            L.append("- %s: %s" % (name, json.dumps(entry, default=str)))
    L += ["", "Build: %s." % ", ".join("%s=%s" % (k, json.dumps(v, default=str)) for k, v in ctx["build"].items()),
          "Training rows: %d actions from %d game(s). Held-out rows set aside before any fit: %d actions from %d game(s) (%s)."
          % (ctx["train_rows"], ctx["train_games"], ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids; `analytics.schema.assert_time_forward` passed on the feature frame (%s)."
          % (ctx["train_games"], ", ".join(F.feature_columns())), "",
          "## The caveat", "",
          "The unit is an action event; the outcomes are the containing possession's points (PPP) and, when epv_states exist, the EPV gained across the action. "
          "Which action gets called depends on the defence, the lineup, the clock and the score, so a difference in PPP between action types is an association, not the effect of the call. "
          "Intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d); a group with fewer than %d rows is marked insufficient and its number is not to be quoted."
          % (ctx["bootstrap"], ctx["seed"], ctx["n_min"]), "",
          "## 1. PPP by action type%s" % tag, ""]
    L += _md_table(ctx["ppp_by_action"], ["action_type", "n", "ppp", "lo", "hi", "insufficient"])
    L += ["", "## 2. PPP by action type x coverage%s" % tag, ""]
    L += _md_table(ctx["ppp_by_action_coverage"], ["coverage", "action_type", "n", "ppp", "lo", "hi", "insufficient"])
    if ctx.get("epv_by_action") is not None and len(ctx["epv_by_action"]):
        L += ["", "## 2b. EPV gained across the action, by action type and by action type x coverage%s" % tag, "",
              "epv_gain = epv_points at the action's end frame minus at its start frame (EPV lane output); %d rows carried it." % ctx["epv_rows"], ""]
        L += _md_table(ctx["epv_by_action"].rename(columns={"ppp": "epv_gain"}), ["action_type", "n", "epv_gain", "lo", "hi", "insufficient"])
        L += [""]
        L += _md_table(ctx["epv_by_action_coverage"].rename(columns={"ppp": "epv_gain"}), ["coverage", "action_type", "n", "epv_gain", "lo", "hi", "insufficient"])
    else:
        L += ["", "## 2b. EPV gained across the action", "", "No epv_states were given, so no EPV outcome is reported."]
    sc = ctx["scores"]
    L += ["", "## 3. Ridge adjustment%s" % tag, "",
          "Possession points regressed on indicator columns of action type, coverage, action type x coverage, offensive lineup and defensive lineup "
          "(lineups from the possession's first frame). Alpha grid %s; %s. Outer-fold alphas: %s. Final alpha on all training rows: %s (grouped-CV MSE by alpha: %s)."
          % (list(ALPHAS), ctx["protocol"], ctx["fold_alphas"], _f(ctx["alpha"], 1),
             ", ".join("%s: %s" % (_f(a, 1), _f(m, 4)) for a, m in ctx["alpha_scores"].items()) or "n/a"), "",
          "| model | out-of-fold MSE | lo | hi |", "| --- | --- | --- | --- |"]
    for m in BASELINES:
        r = sc["mse"][m]
        L.append("| %s | %s | %s | %s |" % (m, _f(r["point"], 4), _f(r["lo"], 4), _f(r["hi"], 4)))
    L += ["", "Paired bootstrap by game of the MSE difference (ridge minus baseline; negative favours the ridge):", ""]
    for m in ("mean", "by_action"):
        r = sc["paired"][m]
        L.append("- ridge minus %s: %s [%s, %s], interval excludes zero: %s" % (m, _f(r["diff"], 4), _f(r["lo"], 4), _f(r["hi"], 4), "yes" if r["excludes_zero"] else "no"))
    L += ["", "## 4. Recommendation table (association)%s" % tag, "",
          "Per coverage, the adjusted PPP of each action type: the ridge prediction with the lineup effects averaged over the training rows. "
          "recommended marks the highest adjusted PPP among the cells with at least %d rows; a coverage with no sufficient cell has no recommendation. "
          "This is the action type that was followed by the most points against that coverage after a linear lineup adjustment, not the action that would score most if called." % ctx["n_min"], ""]
    L += _md_table(ctx["adjusted"], ["coverage", "action_type", "adjusted_ppp", "lo", "hi", "n", "insufficient", "recommended"])
    rec = ctx["adjusted"][ctx["adjusted"]["recommended"]]
    if len(rec):
        L += ["", "Recommended (association, gated): " + "; ".join("%s: %s (%s [%s, %s], n=%d)" % (r.coverage, r.action_type, _f(r.adjusted_ppp), _f(r.lo), _f(r.hi), r.n) for r in rec.itertuples())]
    else:
        L += ["", "No coverage has a cell with at least %d rows, so no recommendation is made." % ctx["n_min"]]
    L += ["", "## Held-out rows", ""]
    if ctx["heldout_rows"]:
        L += ["%d actions from %d held-out game(s) (%s), scored ONCE by the final models refit on all %d training rows; never fitted or tuned on."
              % (ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]), ctx["train_rows"]), "",
              "| model | held-out MSE |", "| --- | --- |"]
        for m in BASELINES:
            L.append("| %s | %s |" % (m, _f(ctx["heldout_scores"][m], 4)))
        L.append("")
        L.append("Point estimates only (no interval over %d game(s))." % ctx["heldout_games"])
    else:
        L.append("No held-out rows were present in the inputs.")
    L += ["", "## Status", ""]
    if fixture:
        L.append("This is a fixture run on synthetic inputs: fixture output, not a result. The real run awaits the actions, coverage and EPV lane outputs "
                 "and the possessions tables on the owner's Mac, with the exact command")
    else:
        L.append("The real run of this protocol is")
    L += ["", "    " + FULL_RUN_CMD, ""]
    if ctx["written"]:
        L += ["Tables written: %s." % ", ".join(ctx["written_files"]) if ctx["written_files"] else "No tables written (no --out)."]
    text = "\n".join(L) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ────────────────────────────────────────────────────────────────────────────────────────────
def run(table: pd.DataFrame, n_folds: int = 5, seed: int = 0, n_boot: int = 200, n_min: int = N_MIN) -> dict:
    """The whole protocol on a built action table -> context dict (tables and scores)."""
    train, heldout = training_split(table)
    if not len(train):
        raise SystemExit("%s: no training rows after setting the held-out rows aside" % WHO)
    ctx = {"train_rows": int(len(train)), "train_games": int(train["sportvu_game"].nunique()),
           "heldout_rows": int(len(heldout)), "heldout_games": int(heldout["sportvu_game"].nunique()),
           "heldout_game_ids": sorted(heldout["sportvu_game"].astype(str).unique()), "seed": seed, "bootstrap": n_boot, "n_min": n_min}
    ctx["ppp_by_action"] = ppp_by_action(train, False, "ppp", n_boot, seed, n_min)
    ctx["ppp_by_action_coverage"] = ppp_by_action(train, True, "ppp", n_boot, seed, n_min)
    ctx["epv_rows"] = int(train["epv_gain"].notna().sum())
    if ctx["epv_rows"]:
        ctx["epv_by_action"] = ppp_by_action(train, False, "epv_gain", n_boot, seed, n_min)
        ctx["epv_by_action_coverage"] = ppp_by_action(train, True, "epv_gain", n_boot, seed, n_min)
    else:
        ctx["epv_by_action"] = ctx["epv_by_action_coverage"] = None
    pred, note = out_of_fold(train, n_folds, seed)
    ctx["pred"], ctx["protocol"], ctx["mode"], ctx["fold_alphas"] = pred, note["text"], note["mode"], note["alphas"]
    ctx["scores"] = score(pred, n_boot, seed)
    ctx["alpha"], ctx["alpha_scores"] = choose_alpha(train, n_folds, seed)
    ctx["adjusted"] = adjusted_table(train, ctx["alpha"], n_boot, seed, n_min)
    ctx["heldout_scores"] = {}
    if len(heldout):
        base, model = fit_baselines(train), fit_ridge(train, ctx["alpha"])
        y = heldout["ppp"].astype(float).to_numpy()
        ctx["heldout_scores"] = {"mean": mse(y, base["mean"](heldout)), "by_action": mse(y, base["by_action"](heldout)),
                                 "ridge": mse(y, predict_ridge(model, heldout))}
    ctx["train"], ctx["heldout"] = train, heldout
    return ctx


def write_tables(ctx: dict, table: pd.DataFrame, out_root, season: str) -> list[Path]:
    written = F.write_action_outcomes(table, out_root, season)
    both = pd.concat([ctx["ppp_by_action"], ctx["ppp_by_action_coverage"]], ignore_index=True)
    for name, df in (("ppp_by_action", both), ("playcalling_adjusted", ctx["adjusted"])):
        p = F.out_path(out_root, season, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        written.append(p)
    return written


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic inputs: the report is labelled fixture output, not a result")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", default=None, help="OUT_ROOT (upstream outputs in, this lane's tables out); default DATA_ROOT")
    ap.add_argument("--report", default="reports/playcalling.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--season", default=SEASON)
    ap.add_argument("--fixture-games", type=int, default=8)
    ap.add_argument("--fixture-possessions", type=int, default=40)
    ap.add_argument("--fixture-generator", choices=("auto", "local"), default="auto", help="auto: analytics.fixtures when present")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    from analytics.playcalling import validate as V
    a = parse_args(argv)
    if a.fixtures:
        inputs = V.fixture_inputs(a.seed, n_games=a.fixture_games, n_possessions=a.fixture_possessions, generator=a.fixture_generator)
    else:
        inputs = V.load_data_root(a.data, a.out or a.data, a.season)
    inputs, vlog = V.validate_inputs(inputs)
    table, blog = F.build_action_table(inputs["frames"], inputs["possessions"], inputs["actions"], inputs.get("screens"),
                                       inputs.get("coverage_predictions"), inputs.get("epv_states"), inputs.get("heldout_games", ()))
    ctx = run(table, a.folds, a.seed, a.bootstrap, a.n_min)
    out_root = a.out or a.data
    ctx["written_files"] = [str(p) for p in write_tables(ctx, table, out_root, a.season)] if out_root else []
    ctx.update({"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "command": "python -m analytics.playcalling.train " + " ".join(argv if argv is not None else sys.argv[1:]),
                "validation": vlog, "build": blog, "fixtures": bool(a.fixtures)})
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()

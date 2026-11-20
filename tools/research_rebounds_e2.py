"""tools/research_rebounds_e2.py: rebounds-E2, train on the whole miss with a within-miss softmax objective
(docs/research/rebounds.md, "rebounds-E2"; PLAN.md D124).

Written before any data is read (D124): the objective, its guards, the binary ablation and the development statistic,
tested on synthetic candidate frames only. Nothing here runs on DATA_ROOT yet. rebounds-E2 runs only after rebounds-E1
decides, because E2's baseline is the E1 model if E1 is adopted, else the frozen hgb with its single tau (both fixed in
the pre-registration). The baseline's out-of-fold predictions will come from rebounds-E1's frozen refit and gate, reused,
not rewritten; until then the baseline is a parameter (baseline_fit) and the tests use the frozen candidate as a
stand-in.

Arms (the same candidate rows, features and folds):
    baseline  the frozen hgb with tau (or the E1 model, if E1 is adopted), from E1's refit
    softmax   XGBoost with a custom objective: per miss, the softmax over that miss's candidate rows, gradient
              p_i - y_i and hessian p_i (1 - p_i) (Cao et al. 2007, one winner per list); depth 3, learning rate 0.1,
              80 trees, as the frozen booster; every other XGBoost setting at its default; fixed, no tuning
    binary    the ablation: the same XGBoost settings with the binary objective, then the frozen tau procedure (a
              temperature fitted on inner out-of-fold predictions over the frozen inner grouped folds), so a gain is
              not a library difference

The features are the frozen design (analytics.rebounds.train.design: the static features plus the out-of-fold prior
rate). Probabilities of every arm are normalised within the miss and checked to sum to 1 per miss. The objective needs
each miss's rows contiguous: check_blocks refuses a frame whose miss rows are split or whose miss has not exactly one
rebounder, before any fit.

Development statistic (pre-registered): out-of-fold per-miss log loss under 5 grouped folds by game (seed 0); the paired
difference softmax minus baseline with a bootstrap by game, 2000 resamples, 95 percent interval; adoption at most
-0.010 with the interval excluding 0. Top-1 and the favourite's calibration slope are reported.
"""

import numpy as np
import pandas as pd

WHO = "tools.research_rebounds_e2"
EXPERIMENT = "rebounds-E2"
SEED, FOLDS, BOOT = 0, 5, 2000
THRESHOLD = -0.010              # pre-registered: paired difference at most -0.010, interval excluding 0
XGB_PARAMS = {"max_depth": 3, "eta": 0.1, "base_score": 0.0, "tree_method": "hist", "nthread": 1, "seed": SEED,
              "disable_default_eval_metric": 1}
ROUNDS = 80
HESS_FLOOR = 1e-16
ARMS = ("baseline", "softmax", "binary")
PAIRS = (("softmax", "baseline"), ("binary", "baseline"), ("softmax", "binary"))


def _T():
    from analytics.rebounds import train as T
    return T


def _F():
    from analytics.rebounds import features as F
    return F


# ----------------------------------------------------------------------------- blocks and the guard

def check_blocks(frame: pd.DataFrame) -> np.ndarray:
    """The block starts (an indptr over rows, length misses + 1) of a candidate frame whose rows of each miss are
    contiguous and hold exactly one rebounder; ValueError otherwise. A sort that split a miss would mix misses inside
    the softmax silently, so every fit and every prediction goes through here first."""
    T, F = _T(), _F()
    key = T.miss_key(frame)
    if len(key) == 0:
        raise ValueError(f"{WHO}: no candidate rows")
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate([[0], change, [len(key)]])
    if len(starts) - 1 != len(np.unique(key)):
        raise ValueError(f"{WHO}: the rows of a miss are not contiguous ({len(starts) - 1} runs, "
                         f"{len(np.unique(key))} misses)")
    y = frame[F.TARGET].astype(int).to_numpy()
    per = np.add.reduceat(y, starts[:-1])
    if not (per == 1).all():
        raise ValueError(f"{WHO}: every miss needs exactly one rebounder row; found {sorted(set(per.tolist()))}")
    return starts


def softmax_blocks(scores, starts) -> np.ndarray:
    """Softmax of the scores within each contiguous block (max-shifted)."""
    s = np.asarray(scores, dtype=float)
    idx = starts[:-1]
    sizes = np.diff(starts)
    mx = np.repeat(np.maximum.reduceat(s, idx), sizes)
    e = np.exp(s - mx)
    return e / np.repeat(np.add.reduceat(e, idx), sizes)


def check_sums(p, starts, tol: float = 1e-9) -> None:
    """ValueError unless the probabilities of every miss sum to 1."""
    sums = np.add.reduceat(np.asarray(p, dtype=float), starts[:-1])
    if not np.allclose(sums, 1.0, atol=tol):
        raise ValueError(f"{WHO}: a miss's probabilities do not sum to 1 (worst {float(np.max(np.abs(sums - 1)))})")


# ----------------------------------------------------------------------------- the objective

def softmax_grad_hess(margins, y, starts):
    """Gradient p - y and hessian p (1 - p) of the within-miss softmax cross entropy, per row (the diagonal of the
    per-miss Hessian, as XGBoost's own multi-class softmax objective uses)."""
    p = softmax_blocks(margins, starts)
    return p - y, np.maximum(p * (1.0 - p), HESS_FLOOR)


def make_objective(y, starts):
    """An XGBoost custom objective over the training rows in their (checked, contiguous) order."""
    y = np.asarray(y, dtype=float)

    def objective(preds, dtrain):
        return softmax_grad_hess(preds, y, starts)
    return objective


def softmax_loss(margins, y, starts) -> float:
    """The mean per-miss softmax cross entropy (for the tests: its gradient is what the objective returns)."""
    p = softmax_blocks(margins, starts)
    return float(-np.log(np.clip(p[np.asarray(y, dtype=bool)], 1e-300, 1.0)).mean())


# ----------------------------------------------------------------------------- the arms

def _design(train: pd.DataFrame, test: pd.DataFrame | None, seed: int = SEED):
    """The frozen design of the training rows (out-of-fold prior) and of the test rows (prior of all training rows)."""
    T = _T()
    if train["heldout"].astype(bool).any():
        raise ValueError(f"{WHO}: a held-out row reached a fit")
    from analytics.eval import refuse_heldout
    refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    rates = T.player_rates(train)
    prior_tr, _ = T.oof_prior(train, seed)
    X_tr = T.design(train, prior_tr)
    X_te = None if test is None else T.design(test, T.apply_prior(test, rates))
    return X_tr, X_te


def fit_softmax(train: pd.DataFrame, seed: int = SEED):
    """The softmax arm on one training frame -> predict(frame) -> p within each miss (checked)."""
    import xgboost as xgb
    F = _F()
    starts = check_blocks(train)
    X, _ = _design(train, None, seed)
    y = train[F.TARGET].astype(int).to_numpy()
    d = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=list(X.columns))
    booster = xgb.train({**XGB_PARAMS, "seed": seed}, d, num_boost_round=ROUNDS, obj=make_objective(y, starts))
    rates = _T().player_rates(train)

    def predict(frame):
        T = _T()
        st = check_blocks(frame)
        Xt = T.design(frame, T.apply_prior(frame, rates))
        m = booster.predict(xgb.DMatrix(Xt.to_numpy(dtype=float), feature_names=list(Xt.columns)), output_margin=True)
        p = softmax_blocks(m, st)
        check_sums(p, st)
        return p
    return predict


def _fit_binary_raw(train: pd.DataFrame, seed: int = SEED):
    """XGBoost, binary objective, the same settings -> predict_q(frame) -> raw probabilities (not normalised)."""
    import xgboost as xgb
    T, F = _T(), _F()
    X, _ = _design(train, None, seed)
    y = train[F.TARGET].astype(int).to_numpy()
    params = {**XGB_PARAMS, "seed": seed, "objective": "binary:logistic", "base_score": 0.5}
    booster = xgb.train(params, xgb.DMatrix(X.to_numpy(dtype=float), label=y, feature_names=list(X.columns)),
                        num_boost_round=ROUNDS)
    rates = T.player_rates(train)

    def predict_q(frame):
        Xt = T.design(frame, T.apply_prior(frame, rates))
        return booster.predict(xgb.DMatrix(Xt.to_numpy(dtype=float), feature_names=list(Xt.columns)))
    return predict_q


def binary_tau(train: pd.DataFrame, seed: int = SEED) -> float:
    """The frozen tau procedure on the binary XGBoost arm: inner out-of-fold raw probabilities over the frozen inner
    grouped folds of the training games, then analytics.rebounds.train.fit_temperature (bounded exponent)."""
    from analytics.eval import grouped_folds
    T, F = _T(), _F()
    games = sorted(train["sportvu_game"].astype(str).unique())
    if len(games) < 2:
        return 1.0
    k = min(T.INNER_FOLDS, len(games))
    g = train["sportvu_game"].astype(str)
    q = np.full(len(train), np.nan)
    for tr_games, te_games in grouped_folds(games, n_folds=k, seed=seed):
        tr, te = g.isin(tr_games).to_numpy(), g.isin(te_games).to_numpy()
        q[te] = _fit_binary_raw(train[tr], seed)(train[te])
    return T.fit_temperature(q, T.miss_key(train), train[F.TARGET].astype(int).to_numpy())


def fit_binary(train: pd.DataFrame, seed: int = SEED):
    """The ablation arm: binary XGBoost with the frozen tau, normalised within each miss (checked)."""
    from analytics.eval import EPS
    check_blocks(train)
    predict_q = _fit_binary_raw(train, seed)
    tau = binary_tau(train, seed)

    def predict(frame):
        T = _T()
        st = check_blocks(frame)
        q = np.clip(predict_q(frame), EPS, 1.0)
        p = T.normalise_within(np.exp(tau * np.log(q)), T.miss_key(frame))
        check_sums(p, st)
        return p
    predict.tau = tau
    return predict


def frozen_baseline_fit(train: pd.DataFrame, seed: int = SEED):
    """A stand-in baseline for the synthetic tests only: the frozen hgb with tau (analytics.rebounds.train). The real
    run takes the baseline's out-of-fold predictions from rebounds-E1's frozen refit and gate (D124)."""
    return _T().fit_candidates(train, seed, only="hgb")["hgb"]


# ----------------------------------------------------------------------------- development statistic

def out_of_fold(cands: pd.DataFrame, baseline_fit=frozen_baseline_fit, n_folds: int = FOLDS, seed: int = SEED):
    """Out-of-fold probabilities of the three arms under grouped folds by game (the frozen folds: analytics.eval
    .grouped_folds over the sorted game ids, seed 0). The frame is sorted by game and event first, keeping each
    miss's rows together, and the blocks are checked."""
    from analytics.eval import grouped_folds
    T = _T()
    frame = cands.sort_values(["sportvu_game", "event"], kind="stable").reset_index(drop=True)
    check_blocks(frame)
    games = sorted(frame["sportvu_game"].astype(str).unique())
    g = frame["sportvu_game"].astype(str)
    out = frame.copy()
    for a in ARMS:
        out["p_" + a] = np.nan
    out["fold"] = -1
    fits = {"baseline": baseline_fit, "softmax": fit_softmax, "binary": fit_binary}
    taus = []
    for k, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=min(n_folds, len(games)), seed=seed)):
        tr, te = g.isin(tr_games).to_numpy(), g.isin(te_games).to_numpy()
        train, test = frame[tr].reset_index(drop=True), frame[te].reset_index(drop=True)
        for a in ARMS:
            model = fits[a](train, seed)
            out.loc[te, "p_" + a] = model(test)
            if a == "binary":
                taus.append(float(model.tau))
        out.loc[te, "fold"] = k
    if out[["p_" + a for a in ARMS]].isna().any().any():
        raise ValueError(f"{WHO}: a row has no out-of-fold probability")
    starts = check_blocks(out)
    for a in ARMS:
        check_sums(out["p_" + a].to_numpy(), starts)
    return out, {"folds": int(out["fold"].max()) + 1, "binary_tau_by_fold": taus}


def summarize(pred: pd.DataFrame, n_boot: int = BOOT, seed: int = SEED) -> dict:
    """Per arm: per-miss log loss and top-1 with 95 percent intervals from one bootstrap of games, the favourite's
    calibration slope (point estimate); the paired per-miss log loss differences on the same resamples; the verdict."""
    from analytics.eval import calibration_slope_fast
    T = _T()
    pms = {a: T.per_miss(pred, "p_" + a) for a in ARMS}
    games = pms[ARMS[0]]["game"].to_numpy()
    for a in ARMS[1:]:
        if not (pms[a]["game"].to_numpy() == games).all():
            raise ValueError(f"{WHO}: the arms' misses are not aligned")
    codes, uniq = pd.factorize(games, sort=True)
    G = len(uniq)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n_boot)]).astype(float)
    n_g = np.bincount(codes, minlength=G).astype(float)
    Wn = W @ n_g

    def ci(per_miss_values):
        v = np.asarray(per_miss_values, dtype=float)
        s = np.bincount(codes, weights=v, minlength=G)
        boots = (W @ s) / Wn
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"value": float(v.mean()), "lo": float(lo), "hi": float(hi)}

    loss = {a: -pms[a]["logp"].to_numpy() for a in ARMS}
    out = {"misses": int(len(games)), "games": int(G), "resamples": n_boot, "seed": seed, "arms": {}, "paired": {}}
    for a in ARMS:
        pm = pms[a]
        out["arms"][a] = {"per_miss_log_loss": ci(loss[a]), "top1": ci(pm["hit"]),
                          "favourite_calibration_slope_point_estimate":
                              float(calibration_slope_fast(pm["top_is_rebounder"].astype(float).to_numpy(),
                                                           pm["p_top"].to_numpy()))}
    for x, z in PAIRS:
        out["paired"][f"{x}_minus_{z}"] = ci(loss[x] - loss[z])
    out["verdict"] = verdict(out["paired"]["softmax_minus_baseline"])
    return out


def verdict(d: dict) -> dict:
    """The pre-registered rule on the development games: softmax minus baseline at most -0.010 with its 95 percent
    interval excluding 0 (that is, its upper end below 0)."""
    size = d["value"] <= THRESHOLD
    excl = d["hi"] < 0
    adopt = bool(size and excl)
    if adopt:
        reading = "met on development"
    elif not excl and d["lo"] > 0:
        reading = "not met: the softmax arm is worse"
    elif not excl:
        reading = "not met: the interval includes zero"
    else:
        reading = "not met: the gain is real (interval below zero) but smaller than 0.010"
    return {"difference": d["value"], "lo": d["lo"], "hi": d["hi"], "threshold": THRESHOLD,
            "size_at_most_threshold": bool(size), "interval_excludes_zero": bool(excl), "met": adopt,
            "reading": reading}

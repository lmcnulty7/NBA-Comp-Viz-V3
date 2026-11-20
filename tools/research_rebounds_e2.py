"""tools/research_rebounds_e2.py: rebounds-E2, train on the whole miss with a within-miss softmax objective
(docs/research/rebounds.md, "rebounds-E2"; PLAN.md D124).

Written before any data is read (D124): the objective, its guards, the binary ablation and the development statistic,
tested on synthetic candidate frames only. The only data read so far is the Set A feasibility check (e2-feasibility),
on the frozen stored out-of-fold predictions of the development games, before any E2 number. rebounds-E2 runs only after rebounds-E1
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

from pathlib import Path

import numpy as np
import pandas as pd

WHO = "tools.research_rebounds_e2"
EXPERIMENT = "rebounds-E2"
SEED, FOLDS, BOOT = 0, 5, 2000
THRESHOLD = -0.010              # pre-registered: paired difference at most -0.010, interval excluding 0
# nthread 1 for the booster and for every DMatrix: DMatrix construction runs its own OpenMP loop, which crashed the
# full test suite when several libomp copies were loaded (2026-11-20); one thread changes no value
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
    d = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=list(X.columns), nthread=1)
    booster = xgb.train({**XGB_PARAMS, "seed": seed}, d, num_boost_round=ROUNDS, obj=make_objective(y, starts))
    rates = _T().player_rates(train)

    def predict(frame):
        T = _T()
        st = check_blocks(frame)
        Xt = T.design(frame, T.apply_prior(frame, rates))
        m = booster.predict(xgb.DMatrix(Xt.to_numpy(dtype=float), feature_names=list(Xt.columns), nthread=1),
                            output_margin=True)
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
    booster = xgb.train(params, xgb.DMatrix(X.to_numpy(dtype=float), label=y, feature_names=list(X.columns), nthread=1),
                        num_boost_round=ROUNDS)
    rates = T.player_rates(train)

    def predict_q(frame):
        Xt = T.design(frame, T.apply_prior(frame, rates))
        return booster.predict(xgb.DMatrix(Xt.to_numpy(dtype=float), feature_names=list(Xt.columns), nthread=1))
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


# ----------------------------------------------------------------------------- Set A feasibility (no E2 number)

def feasibility(root, n: int = BOOT, seed: int = SEED, test_games: int = 10, set_a_misses: int = 820) -> dict:
    """Before any E2 number: how wide would a paired difference of two whole models be on Set A? Read from the frozen
    stored out-of-fold predictions of the development games (analytics/rebounds/<season>/<game>.parquet, fold >= 0;
    the rows rebounds-E1's feasibility used), with the frozen logit and the frozen hgb (p_selected) as two different
    models of the same misses: a proxy for a model replacement, which E2 is. Reports the per-miss SD of their
    per-miss log loss difference, the half-width of that difference over random 10-game sets of development games,
    the development half-width (bootstrap of all games), and the SD-based projection at about 820 Set A misses."""
    import glob
    from analytics.eval import EPS
    from tools.research_testset import refuse_test
    files = sorted(glob.glob(str(Path(root) / "analytics" / "rebounds" / "2015-16" / "*.parquet")))
    cols = ["sportvu_game", "event", "candidate_player_id", "p_logit", "p_selected", "is_rebounder", "fold", "heldout"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    d = d[(d["fold"] >= 0) & ~d["heldout"].astype(bool)]
    refuse_test(d["sportvu_game"].astype(str).unique(), WHO)
    true = d[d["is_rebounder"].astype(bool)]
    if true.duplicated(["sportvu_game", "event"]).any():
        raise ValueError(f"{WHO}: a miss has two rebounder rows in the stored predictions")
    diff = (np.log(np.clip(true["p_selected"].to_numpy(float), EPS, 1)) -
            np.log(np.clip(true["p_logit"].to_numpy(float), EPS, 1)))      # logit loss minus hgb loss, per miss
    games = true["sportvu_game"].astype(str).to_numpy()
    codes, uniq = pd.factorize(games, sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    s = np.bincount(codes, weights=diff, minlength=G)
    rng = np.random.default_rng(seed)
    sets = [s[c].sum() / cnt[c].sum() for c in (rng.choice(G, size=test_games, replace=False) for _ in range(n))]
    lo, hi = np.percentile(sets, [2.5, 97.5])
    idx = rng.integers(0, G, size=(n, G))
    blo, bhi = np.percentile(s[idx].sum(axis=1) / cnt[idx].sum(axis=1), [2.5, 97.5])
    sd = float(np.std(diff, ddof=1))
    se_iid = sd / np.sqrt(set_a_misses)
    return {"proxy": "frozen logit minus frozen hgb (p_selected), per-miss log loss, stored out-of-fold rows",
            "games": int(G), "misses": int(len(diff)), "mean_difference": float(diff.mean()),
            "per_miss_sd_of_difference": sd,
            "random_10_game_sets_half_width": float((hi - lo) / 2),
            "development_half_width": float((bhi - blo) / 2),
            "set_a_misses_assumed": set_a_misses,
            "set_a_half_width_iid": float(1.96 * se_iid),
            "threshold": THRESHOLD}


# ----------------------------------------------------------------------------- the run on rebounds-E1's refit

RESULTS = None                 # reports/research by default; tests point it elsewhere
FOLD_MINUTES = 8.0             # a fold starts only when at least this much of the job's budget is left


def work_dir(artifacts=None) -> Path:
    from tools.frozen_common import artifacts_dir, assert_safe_output
    return assert_safe_output(artifacts_dir(artifacts) / "research" / "rebounds_e2")


def load_e1(e1_artifacts) -> tuple:
    """rebounds-E1's gated refit, reused (D124): its candidate rows (checked against the frozen record), its gate A
    record (must have passed) and its five fold pieces (training and test rows, the baseline's p on the test rows)."""
    import json
    from tools.frozen_common import artifacts_dir
    from tools.research_testset import refuse_test
    folder = artifacts_dir(e1_artifacts) / "research" / "rebounds_e1"
    gate = json.loads((folder / "gate.json").read_text())
    if gate.get("passed") is not True:
        raise SystemExit(f"{WHO}: rebounds-E1's gate A did not pass, so its refit is not used")
    rec = json.loads((folder / "candidates_train.json").read_text())
    if not rec.get("matches_frozen_record"):
        raise SystemExit(f"{WHO}: rebounds-E1's candidate rows do not match the frozen record")
    cands = pd.read_parquet(folder / "candidates_train.parquet")
    if len(cands) != gate["rows_mine"] or len(cands) != rec["rows"]:
        raise SystemExit(f"{WHO}: rebounds-E1's candidate rows changed after its gate")
    refuse_test(cands["sportvu_game"].astype(str).unique(), WHO)
    folds = [dict(np.load(folder / "folds" / f"fold_{k}.npz")) for k in range(FOLDS)]
    test_all = np.sort(np.concatenate([f["test_rows"] for f in folds]))
    if not np.array_equal(test_all, np.arange(len(cands))):
        raise SystemExit(f"{WHO}: rebounds-E1's folds do not cover every candidate row exactly once")
    return cands, gate, folds


def _ordered(frame: pd.DataFrame) -> np.ndarray:
    """A stable order that puts each miss's rows together (the frame's own order when they already are)."""
    return np.argsort((frame["sportvu_game"].astype(str) + "#" + frame["event"].astype(str)).to_numpy(), kind="stable")


def fold_arms(cands: pd.DataFrame, fold: dict, seed: int = SEED) -> dict:
    """The softmax arm and the binary ablation fitted on one fold's training rows, predicting its test rows (in the
    fold's test_rows order)."""
    train = cands.iloc[fold["train_rows"]].reset_index(drop=True)
    test = cands.iloc[fold["test_rows"]].reset_index(drop=True)
    o_tr, o_te = _ordered(train), _ordered(test)
    train, test_sorted = train.iloc[o_tr].reset_index(drop=True), test.iloc[o_te].reset_index(drop=True)
    out = {}
    soft = fit_softmax(train, seed)(test_sorted)
    binary_model = fit_binary(train, seed)
    binary = binary_model(test_sorted)
    for name, p in (("p_softmax", soft), ("p_binary", binary)):
        back = np.empty(len(p))
        back[o_te] = p                                  # back to the fold's test_rows order
        out[name] = back
    out["tau_binary"] = float(binary_model.tau)
    return out


def fold_stage(e1_artifacts, artifacts=None, max_minutes: float = 25.0, log=print, head: str = "") -> dict:
    """Resumable: one outer fold a piece, written once; a fold starts only with FOLD_MINUTES of budget left."""
    import os
    import time
    cands, _gate, folds = load_e1(e1_artifacts)
    folder = work_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = []
    for k in range(FOLDS):
        f = folder / f"fold_{k}.npz"
        if f.exists():
            continue
        if max_minutes - (time.time() - t0) / 60 < FOLD_MINUTES:
            break
        r = fold_arms(cands, folds[k])
        part = folder / f"fold_{k}.part.npz"
        np.savez(part, test_rows=folds[k]["test_rows"], code_commit=np.array(head), **r)
        os.replace(part, f)
        done.append(k)
        log(f"[rebounds-E2] fold {k} done {round(time.time() - t0)} s, binary tau {r['tau_binary']:.4f}", flush=True)
    left = [k for k in range(FOLDS) if not (folder / f"fold_{k}.npz").exists()]
    return {"done_this_job": done, "left": left, "seconds": round(time.time() - t0, 1)}


def projected_set_a(pred: pd.DataFrame, test_games: int = 10, n: int = BOOT, seed: int = SEED) -> dict:
    """Commitment 3 (D125): the Set A half-width projected from the development per-game spread of the softmax minus
    baseline difference: random 10-game sets of development games, and the development bootstrap scaled."""
    T = _T()
    a, b = T.per_miss(pred, "p_softmax"), T.per_miss(pred, "p_baseline")
    diff = (-a["logp"].to_numpy()) - (-b["logp"].to_numpy())
    codes, uniq = pd.factorize(a["game"].to_numpy(), sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    s = np.bincount(codes, weights=diff, minlength=G)
    rng = np.random.default_rng(seed)
    vals = [s[c].sum() / cnt[c].sum() for c in (rng.choice(G, size=test_games, replace=False) for _ in range(n))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    idx = rng.integers(0, G, size=(n, G))
    blo, bhi = np.percentile(s[idx].sum(axis=1) / cnt[idx].sum(axis=1), [2.5, 97.5])
    return {"random_10_game_sets_half_width": float((hi - lo) / 2),
            "dev_half_width_scaled": float((bhi - blo) / 2 * np.sqrt(G / test_games)), "development_games": int(G)}


def dev_stage(e1_artifacts, artifacts=None, head: str = "") -> dict:
    """Assemble the out-of-fold frame (baseline p from rebounds-E1's fold pieces, the two arms from this experiment's)
    and write reports/research/rebounds-E2-dev.json."""
    import json
    from tools.frozen_common import REPO_ROOT, _plain
    cands, gate, folds = load_e1(e1_artifacts)
    folder = work_dir(artifacts)
    pred = cands.copy()
    for c in ("p_baseline", "p_softmax", "p_binary"):
        pred[c] = np.nan
    pred["fold"] = -1
    taus, commits = [], set()
    for k in range(FOLDS):
        mine = dict(np.load(folder / f"fold_{k}.npz"))
        if not np.array_equal(mine["test_rows"], folds[k]["test_rows"]):
            raise SystemExit(f"{WHO}: fold {k}'s rows differ from rebounds-E1's")
        rows = folds[k]["test_rows"]
        pred.loc[rows, "p_baseline"] = folds[k]["p_base"]
        pred.loc[rows, "p_softmax"] = mine["p_softmax"]
        pred.loc[rows, "p_binary"] = mine["p_binary"]
        pred.loc[rows, "fold"] = k
        taus.append(float(mine["tau_binary"]))
        commits.add(str(mine["code_commit"]))
    if pred[["p_baseline", "p_softmax", "p_binary"]].isna().any().any():
        raise SystemExit(f"{WHO}: a candidate row has no out-of-fold probability")
    order = _ordered(pred)
    pred = pred.iloc[order].reset_index(drop=True)
    starts = check_blocks(pred)
    for a in ARMS:
        check_sums(pred["p_" + a].to_numpy(), starts)
    s = summarize(pred)
    base_ll = s["arms"]["baseline"]["per_miss_log_loss"]["value"]
    if round(base_ll, 6) != round(gate["per_miss_log_loss"], 6):
        raise SystemExit(f"{WHO}: the baseline here ({base_ll}) is not rebounds-E1's gated baseline")
    out = {"experiment": EXPERIMENT, "code_commit": head, "fold_code_commits": sorted(commits),
           "stage": ("development games (604), out of fold under rebounds-E1's five folds; baseline the frozen hgb with "
                     "tau from rebounds-E1's gated refit (reused); the development games decide (D125's pattern)"),
           "pre_registration": "docs/research/rebounds.md, rebounds-E2 and its amendment",
           "e1_gate_a": gate, "candidate_rows": int(len(pred)), "binary_tau_by_fold": taus,
           **s, "projected_set_a": projected_set_a(pred)}
    path = Path(RESULTS or REPO_ROOT / "reports" / "research") / f"{EXPERIMENT}-dev.json"
    path.write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out["verdict"]


def _head() -> str:
    import subprocess
    from tools.frozen_common import REPO_ROOT
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


# ----------------------------------------------------------------------------- Set A confirmation (D127)

REHEARSAL_GAMES = 10
SEASON = "2015-16"
SET_A_TABLES = ("shot_frames", "rosters", "possessions")      # rebound_frames is built here, never read on Set A


def clean_head() -> str:
    """HEAD, after checking that the whole working tree is committed."""
    import subprocess
    from tools.frozen_common import REPO_ROOT
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty}")
    return _head()


def game_paths(root, game: str) -> dict:
    root = Path(root)
    out = {"frames": root / "sportvu_contract" / f"{game}.parquet",
           "events": root / "sportvu_contract" / f"{game}_pbp.parquet"}
    out.update({t: root / "analytics" / t / SEASON / f"{game}.parquet" for t in SET_A_TABLES})
    return out


def missing_tables(root, games) -> dict:
    """Per game, the inputs that are not there (checked before any value is read)."""
    out = {g: sorted(k for k, q in game_paths(root, g).items() if not q.exists()) for g in games}
    return {g: m for g, m in out.items() if m}


def game_bundle(root, game: str) -> dict:
    """The frozen bundle of one game (as analytics.rebounds.validate.bundles_from_root forms it), except that the
    rebound_frames table is built here by the frozen analytics.rebound_frames (read_frames, read_events on the joined
    play-by-play, locate_rebounds) and typed through its own arrow schema, as phase0 writes it, instead of read."""
    from analytics import rebound_frames as RF
    from analytics.rebounds import validate as V
    paths = game_paths(root, game)
    table, rf_frames = RF.read_frames(paths["frames"])
    events = RF.read_events(game, paths["events"], table)
    rebounds = RF.to_arrow(RF.locate_rebounds(rf_frames, events, game)).to_pandas()
    del table, rf_frames
    b = {"sportvu_game": game, "heldout": False, "frames": V.read_frames(paths["frames"]),
         "events": pd.read_parquet(paths["events"]), "rebound_frames": rebounds}
    for name in SET_A_TABLES:
        b[name] = pd.read_parquet(paths[name])
    return b


def build_candidates(root, games, counts: dict | None = None) -> pd.DataFrame:
    """The candidate rows of the given games, one game's frames at a time, through the frozen clean_bundle and
    features.build; refuses a held-out row. With counts (a dict), the frozen build's per-game counts are stored in it:
    misses in and kept, rows, and every drop reason (unmatched_named_rebounder among them) and info count; counting
    only."""
    from analytics.rebounds import features as Fz
    from analytics.rebounds import train as T
    from analytics.rebounds import validate as V

    def cleaned():
        for g in games:
            cb, _counts = V.clean_bundle(game_bundle(root, g))
            yield cb
    cands, fc = Fz.build(cleaned())
    if counts is not None:
        for g in games:
            c = fc["per_game"].get(g, {})
            counts[g] = {k: int(c.get(k, 0)) for k in ("n_in", "misses_kept", "rows") + tuple(Fz.DROP_REASONS)
                         + tuple(Fz.INFO_COUNTS)}
    rows, held = T.split_train_heldout(cands)
    if len(held):
        raise SystemExit(f"{WHO}: a held-out row came out of the scorer's path")
    return rows


def fit_dev_models(cands: pd.DataFrame, games) -> dict:
    """The three arms fitted on the given development games: the frozen hgb with tau (fit_candidates, the frozen
    code whose out-of-fold fits passed rebounds-E1's gate A), the softmax arm and the binary ablation."""
    train = cands[cands["sportvu_game"].astype(str).isin(set(games))].reset_index(drop=True)
    ordered = train.iloc[_ordered(train)].reset_index(drop=True)
    # the baseline on the rows in their build order, as rebounds-E1's gated fold fits saw them; only the two arms
    # that need each miss's rows together are fitted on the sorted rows
    return {"baseline": frozen_baseline_fit(train), "softmax": fit_softmax(ordered), "binary": fit_binary(ordered)}


def reading(lo: float, hi: float) -> str:
    """D108's readings: confirms (wholly below 0), contradiction (wholly above 0, stops the adoption), else
    inconclusive at 10 games."""
    if hi < 0:
        return "confirms"
    if lo > 0:
        return "contradiction"
    return "inconclusive at 10 games"


def t_interval_of_game_means(games, diff) -> dict:
    """Report only: a 95 percent t interval on the per-game mean differences, one value per game."""
    from scipy import stats
    d = pd.DataFrame({"g": games, "d": diff}).groupby("g")["d"].mean().to_numpy()
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean_of_game_means": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half),
            "games": int(len(d))}


def score_rows(rows: pd.DataFrame, models: dict, n_boot: int = BOOT) -> dict:
    """Every arm on rows built by the scorer's path; the paired confirmation and its reading."""
    T = _T()
    frame = rows.iloc[_ordered(rows)].reset_index(drop=True)
    starts = check_blocks(frame)
    pred = frame.copy()
    for a in ARMS:
        pred["p_" + a] = models[a](frame)
        check_sums(pred["p_" + a].to_numpy(), starts)
    s = summarize(pred, n_boot=n_boot)
    a, b = T.per_miss(pred, "p_softmax"), T.per_miss(pred, "p_baseline")
    diff = (-a["logp"].to_numpy()) - (-b["logp"].to_numpy())
    d = s["paired"]["softmax_minus_baseline"]
    s.pop("verdict")
    return {**s, "candidate_rows": int(len(frame)),
            "t_interval_softmax_minus_baseline_per_game": t_interval_of_game_means(a["game"].to_numpy(), diff),
            "reading": reading(d["lo"], d["hi"])}


def key_tuples(rows) -> list:
    """(game, event, candidate player) of each row as typed values (the none row's id is -1, never NaN)."""
    return list(zip(rows["sportvu_game"].astype(str), rows["event"].astype(int),
                    pd.to_numeric(rows["candidate_player_id"]).astype(int)))


def same_column(a, b) -> bool:
    """The same values in the same order: numbers exactly, with the same NaN placement; anything else value by value,
    a missing value matching only a missing value. No tolerance."""
    if len(a) != len(b):
        return False
    try:
        return bool(np.array_equal(pd.to_numeric(a).to_numpy(float), pd.to_numeric(b).to_numpy(float), equal_nan=True))
    except (TypeError, ValueError):
        return [None if pd.isna(x) else x for x in a] == [None if pd.isna(x) else x for x in b]


def rehearsal_check(built: pd.DataFrame, cached: pd.DataFrame) -> dict:
    rows = len(built) == len(cached) and key_tuples(built) == key_tuples(cached)
    shared = sorted(set(built.columns) & set(cached.columns))
    return {"rows_match_cache": bool(rows), "rows": int(len(built)), "rows_cached": int(len(cached)),
            "shared_columns_match_cache": {c: bool(rows and same_column(built[c], cached[c])) for c in shared}}


def rehearsal_passed(record: dict) -> bool:
    cols = record.get("shared_columns_match_cache", {})
    return bool(record.get("rows_match_cache")) and bool(cols) and all(cols.values())


def rehearsal_path(artifacts=None) -> Path:
    return work_dir(artifacts) / "rebounds_e2_rehearsal.json"


def rehearse(e1_artifacts, artifacts=None) -> dict:
    """The scorer's path on 10 development games drawn with seed 0 in place of Set A (not a result; artifacts only):
    the arms fitted on the other development games, the stand-in games' rows built from DATA_ROOT by the scorer's own
    path (rebound frames built, not read), which must equal rebounds-E1's gated candidate rows of those games."""
    import json
    import os
    import random
    from tools.frozen_common import _plain
    from tools.research_testset import dev_games
    head = clean_head()
    cands, _gate, _folds = load_e1(e1_artifacts)
    games = dev_games(sorted(cands["sportvu_game"].astype(str).unique()), WHO)
    stand_in = sorted(random.Random(SEED).sample(games, REHEARSAL_GAMES))
    models = fit_dev_models(cands, [g for g in games if g not in set(stand_in)])
    built = build_candidates(Path(os.environ["DATA_ROOT"]), stand_in)
    cached = cands[cands["sportvu_game"].astype(str).isin(set(stand_in))].reset_index(drop=True)
    check = rehearsal_check(built.reset_index(drop=True), cached)
    out = {"what": "rehearsal of the rebounds-E2 scorer on 10 development games in place of Set A; not a result",
           "code_commit": head, "stand_in_games": len(stand_in), **check, **score_rows(built, models)}
    work_dir(artifacts).mkdir(parents=True, exist_ok=True)
    rehearsal_path(artifacts).write_text(json.dumps(_plain(out), indent=2) + "\n")
    if not rehearsal_passed(out):
        raise SystemExit(f"{WHO}: the scorer's rows differ from rebounds-E1's development rows")
    return {k: out[k] for k in ("code_commit", "rows_match_cache", "rows", "rows_cached")}


def check_rehearsal(path, head: str) -> dict:
    import json
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{WHO}: rehearse the scorer first")
    record = json.loads(path.read_text())
    if not rehearsal_passed(record):
        raise SystemExit(f"{WHO}: the rehearsal did not pass")
    if record.get("code_commit") != head:
        raise SystemExit(f"{WHO}: the rehearsal ran at {record.get('code_commit')}, not at this HEAD {head}")
    return record


def make_scorer(models: dict, artifacts=None, provenance: dict = None):
    """The scorer score_once runs. Set A is reached only here: the research root, its inputs checked before any value
    is read, all 10 games or none, a non-finite number refuses the reading."""
    from tools import research_testset as RT
    from tools.research_testset import TRACKING
    from tools.research_xfg import research_data_root

    def scorer(test_games):
        if RT._SCORING is None or RT._SCORING[1] != TRACKING:
            raise SystemExit(f"{WHO}: the test data is read only inside the scorer that score_once runs")
        root = research_data_root(artifacts)
        games = sorted(str(g) for g in test_games)
        head = {"experiment": EXPERIMENT, "stage": ("fitted on every development game, scored once on the D71 test "
                                                    "set as a D127 confirmation with D108's readings"),
                **(provenance or {})}
        missing = missing_tables(root, games)
        if missing:
            return {**head, "refused": True, "reading": f"refused: {len(missing)} test game(s) lack an input",
                    "missing": missing}
        counts = {}
        rows = build_candidates(root, games, counts)
        head["build_counts_by_game"] = counts
        without = sorted(set(games) - set(rows["sportvu_game"].astype(str)))
        if without:
            return {**head, "refused": True, "reading": f"refused: {len(without)} test game(s) gave no candidate row",
                    "games_without_rows": without}
        try:
            result = score_rows(rows, models)
        except ValueError as exc:                # a block or sum guard: a refusal on record, never a silent crash
            return {**head, "refused": True, "reading": f"refused: a guard failed: {exc}"}
        bad = [k for k, v in result["paired"].items() if not all(np.isfinite([v["value"], v["lo"], v["hi"]]))]
        if bad:
            return {**head, "refused": True, "reading": f"refused: non-finite values in {bad}"}
        return {**head, **result}

    return scorer


def final(e1_artifacts, artifacts=None) -> dict:
    """The one Set A score (D127): only when development adopts and the rehearsal passed at this HEAD."""
    import hashlib
    import json
    from tools.frozen_common import REPO_ROOT, _plain
    from tools.research_testset import TRACKING, dev_games, score_once
    from tools.research_xfg import test_root_ready
    head = clean_head()
    results = Path(RESULTS or REPO_ROOT / "reports" / "research")
    dev_path = results / f"{EXPERIMENT}-dev.json"
    dev_out = json.loads(dev_path.read_text())
    if not dev_out["verdict"]["met"]:
        raise SystemExit(f"{WHO}: development does not adopt, so Set A is not scored (D127)")
    check_rehearsal(rehearsal_path(artifacts), head)
    if not test_root_ready(artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games")
    cands, gate, _folds = load_e1(e1_artifacts)
    models = fit_dev_models(cands, dev_games(sorted(cands["sportvu_game"].astype(str).unique()), WHO))
    sha = lambda q: hashlib.sha256(Path(q).read_bytes()).hexdigest()
    provenance = {"code_commit": head, "dev_json_sha256": sha(dev_path),
                  "rehearsal_json_sha256": sha(rehearsal_path(artifacts))}
    out = score_once(EXPERIMENT, TRACKING, make_scorer(models, artifacts, provenance))
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"e1_gate_a": gate, "projected_set_a": dev_out["projected_set_a"],
                        "rule": ("D127 with D108's readings: confirms if the 95 percent interval of softmax minus "
                                 "baseline lies wholly below 0, inconclusive at 10 games if it straddles 0, "
                                 "contradiction if wholly above 0 (stops the adoption); the t interval on the per-game "
                                 "means beside it, report only")}}
    (results / f"{EXPERIMENT}-final.json").write_text(json.dumps(_plain(record), indent=2) + "\n")
    return out


def main(argv=None) -> int:
    import argparse
    import json
    import os
    from export.heavy import heavy_job
    from tools.frozen_common import _plain
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e2-feasibility", "e2-folds", "e2-dev", "e2-rehearse", "e2-final"))
    ap.add_argument("--e1-artifacts", default=None, help="rebounds-E1's artifacts folder (its gated refit)")
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research rebounds {a.command}"):
        if a.command == "e2-feasibility":
            out = feasibility(Path(os.environ["DATA_ROOT"]))
        elif a.command == "e2-folds":
            out = fold_stage(a.e1_artifacts, a.artifacts, a.max_minutes, head=_head())
        elif a.command == "e2-dev":
            out = dev_stage(a.e1_artifacts, a.artifacts, head=_head())
        elif a.command == "e2-rehearse":
            out = rehearse(a.e1_artifacts, a.artifacts)
        else:
            out = final(a.e1_artifacts, a.artifacts)
            out = {k: out.get(k) for k in ("reading", "refused", "misses", "games", "paired")}
    print(json.dumps(_plain(out), indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

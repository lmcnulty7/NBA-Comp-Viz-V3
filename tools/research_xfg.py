"""tools/research_xfg.py: the xFG research experiments (PLAN.md D70, D71, D74).

    python3 -m tools.research_xfg features [--artifacts DIR]
    python3 -m tools.research_xfg e1-gate  [--artifacts DIR]
    python3 -m tools.research_xfg e1-dev   [--artifacts DIR]
    python3 -m tools.research_xfg e1-final [--artifacts DIR]

features    Builds the frozen xFG feature frame of the 2015-16 training rows once, with the frozen code unchanged
            (analytics.xfg.validate and analytics.xfg.train.assemble, the same path tools.frozen_models uses), and
            writes it to <artifacts>/research/xfg_features_train.parquet. Held-out rows are set aside by the frozen
            assemble and are never written; their labels are blanked before any feature is built. The research
            experiments split this frame into development games and the D71 test games through
            tools.research_testset only.

e1-gate     xfg-E1 gate (docs/research/xfg.md): the isotonic arm, refit here with the frozen functions on all 607
            training games under the frozen outer folds, must reproduce the stored out-of-fold p_selected
            (fold >= 0) within the xfg tolerance. Training rows only; no held-out row and no metric.
e1-dev      xfg-E1 on the development games (every training game except the D71 test games): 5 grouped outer folds
            (seed 0), the frozen isotonic arm against cross Venn-Abers (CVAP, Vovk, Petej, Fedorova 2015) on the
            same 3 inner folds, and the single-split IVAP ablation. Writes reports/research/xfg-E1-dev.json.
e1-final    The one final scoring call: both arms fitted on all development games, scored once on the D71 test
            games through tools.research_testset.score_once. Writes reports/research/xfg-E1-final.json.

Artifacts go only under the artifacts folder (never under DATA_ROOT, never under reports/); the research results go
to reports/research/. Everything runs under the heavy-job lock.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import (MemoryGuard, REPO_ROOT, TOLERANCE_CAP, artifacts_dir, assert_safe_output,
                                 begin_rebuild, require_passed, training_games_only, write_check)
from tools.research_testset import TRACKING, dev_games, refuse_test, score_once

WHO = "tools.research_xfg"
FEATURES_FILE = "xfg_features_train.parquet"
RESULTS = REPO_ROOT / "reports" / "research"
SEED, OUTER_FOLDS, BOOT = 0, 5, 2000
THRESHOLD = 0.0005             # pre-registered: adopt CVAP if the 95 percent upper bound of (CVAP - isotonic) <= this
POINT_ESTIMATE_WIDTH = 0.01    # pre-registered site rule: p95 of p1 - p0 under this means "point estimate"
STORED_KEYS = ["sportvu_game", "event", "player_id"]


def research_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research")


def features_path(artifacts=None) -> Path:
    return research_dir(artifacts) / FEATURES_FILE


def build_features(artifacts=None) -> dict:
    """Build and write the training feature frame -> a summary record."""
    import os
    from analytics.xfg import train as T
    from tools.frozen_models import xfg_tables
    root = Path(os.environ["DATA_ROOT"])
    out = features_path(artifacts)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    guard = MemoryGuard()
    train, heldout = T.assemble(xfg_tables(root, guard))
    del heldout                                           # set aside by the frozen assemble; never written
    games = sorted(train["sportvu_game"].astype(str).unique())
    training_games_only(games, WHO)
    train.to_parquet(out, index=False)
    record = {"file": out.name, "rows": int(len(train)), "games": len(games),
              "with_contest": int(train["has_contest"].sum()), "seconds": round(time.time() - started, 1),
              "peak_gb": round(peak_rss_gb(), 2)}
    (out.parent / "xfg_features_train.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


# ----------------------------------------------------------------------------- Venn-Abers

class VennAbers:
    """Inductive Venn-Abers predictor (Vovk and Petej 2014) on one calibration set of (score, label) pairs.
    p0p1(s) gives, for every test score, the isotonic fit at s of the calibration set plus (s, 0) and plus (s, 1).
    Exact: one weighted pool-adjacent-violators fit per distinct insertion position (ties share a position)."""

    def __init__(self, scores, labels):
        import numpy as np
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        u, inv = np.unique(s, return_inverse=True)
        self.u = u
        self.w = np.bincount(inv).astype(float)
        self.sy = np.bincount(inv, weights=y)

    def _fit_at(self, pos: int, exact: bool, label: float) -> float:
        import numpy as np
        from sklearn._isotonic import _inplace_contiguous_isotonic_regression as pav
        if exact:
            w = self.w.copy()
            sy = self.sy.copy()
            w[pos] += 1.0
            sy[pos] += label
            idx = pos
        else:
            w = np.insert(self.w, pos, 1.0)
            sy = np.insert(self.sy, pos, label)
            idx = pos
        means = np.ascontiguousarray(sy / w, dtype=np.float64)
        pav(means, np.ascontiguousarray(w, dtype=np.float64))
        return float(means[idx])

    def p0p1(self, scores):
        import numpy as np
        s = np.asarray(scores, dtype=float)
        pos = np.searchsorted(self.u, s, side="left")
        exact = (pos < len(self.u)) & (self.u[np.minimum(pos, len(self.u) - 1)] == s)
        keys = pos * 2 + exact.astype(int)
        p0 = np.empty(len(s))
        p1 = np.empty(len(s))
        for key in np.unique(keys):
            m = keys == key
            ps, ex = int(key // 2), bool(key % 2)
            p0[m] = self._fit_at(ps, ex, 0.0)
            p1[m] = self._fit_at(ps, ex, 1.0)
        return p0, p1


def _gm(a):
    """Geometric mean over axis 0 (0 when any entry is 0)."""
    import numpy as np
    with np.errstate(divide="ignore"):
        return np.exp(np.mean(np.log(np.asarray(a, dtype=float)), axis=0))


def merge_cvap(p0s, p1s):
    """The log-loss minimax merge of Vovk, Petej, Fedorova 2015, Section 4: GM(p1) / (GM(1 - p0) + GM(p1))."""
    import numpy as np
    g1 = np.exp(np.mean(np.log(np.asarray(p1s)), axis=0))
    g0 = np.exp(np.mean(np.log(1.0 - np.asarray(p0s)), axis=0))
    return g1 / (g0 + g1)


# ----------------------------------------------------------------------------- the frozen candidate, kept open

def fit_parts(train, seed: int = SEED) -> dict:
    """hgb_contest_prior exactly as analytics.xfg.train._fit_candidate fits it (the frozen functions, unchanged),
    keeping what the frozen closure discards: the 3 inner boosters with their prior tables, and the raw inner
    out-of-fold probabilities (the calibration sets)."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from analytics.xfg import train as T
    from analytics.xfg.features import PRIOR_FEATURE, feature_columns
    if train["heldout"].any():
        raise ValueError(f"{WHO}: a held-out row reached the fit")
    T.refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    cols = feature_columns(with_prior=True)
    games = train["sportvu_game"].astype(str)
    y = train["made"].astype(int).to_numpy()
    k = min(T.INNER_FOLDS, games.nunique())
    oof = np.full(len(train), np.nan)
    inner_fold = np.full(len(train), -1)
    inner = []
    with T.fit_threads(1):
        for j, (_tr, te_g) in enumerate(T.grouped_folds(games.unique(), k, seed)):
            m = games.isin(te_g).to_numpy()
            rows = train[~m].copy()
            rows[PRIOR_FEATURE] = T.oof_prior(rows, T.INNER_FOLDS, seed)
            test = train[m].copy()
            table = T.prior_table(rows)
            test[PRIOR_FEATURE] = T.apply_prior(table, test)
            h = HistGradientBoostingClassifier(random_state=seed, **T.HGB_PARAMS).fit(T._design(rows, cols),
                                                                                    rows["made"].astype(int))
            oof[m] = h.predict_proba(T._design(test, cols))[:, 1]
            inner_fold[m] = j
            inner.append((h, table))
        full = train.copy()
        full[PRIOR_FEATURE] = T.oof_prior(full, T.INNER_FOLDS, seed)
        hgb = HistGradientBoostingClassifier(random_state=seed, **T.HGB_PARAMS).fit(T._design(full, cols), y)
    iso = IsotonicRegression(y_min=T.ISO_CLIP, y_max=1 - T.ISO_CLIP, out_of_bounds="clip").fit(oof, y)
    return {"cols": cols, "hgb": hgb, "iso": iso, "prior": T.prior_table(train), "inner": inner, "oof": oof,
            "inner_fold": inner_fold, "y": y}


def predict_arms(parts: dict, frame) -> dict:
    """Both arms on a frame: the frozen isotonic p, the CVAP merged p with its (p0, p1) per inner fold, the IVAP p
    of inner fold 0 (the ablation), and the raw full-booster score."""
    import numpy as np
    from analytics.xfg import train as T
    from analytics.xfg.features import PRIOR_FEATURE
    f = frame.copy()
    with T.fit_threads(1):
        f[PRIOR_FEATURE] = T.apply_prior(parts["prior"], f)
        raw = parts["hgb"].predict_proba(T._design(f, parts["cols"]))[:, 1]
        p_iso = parts["iso"].predict(raw)
        p0s, p1s = [], []
        for j, (h, table) in enumerate(parts["inner"]):
            g = frame.copy()
            g[PRIOR_FEATURE] = T.apply_prior(table, g)
            s = h.predict_proba(T._design(g, parts["cols"]))[:, 1]
            m = parts["inner_fold"] == j
            va = VennAbers(parts["oof"][m], parts["y"][m])
            p0, p1 = va.p0p1(s)
            p0s.append(p0)
            p1s.append(p1)
    p0s, p1s = np.array(p0s), np.array(p1s)
    return {"p_iso": p_iso, "p_cvap": merge_cvap(p0s, p1s), "p_ivap": p1s[0] / (1 - p0s[0] + p1s[0]),
            "lo": _gm(p0s),
            "hi": np.exp(np.mean(np.log(p1s), axis=0)), "widths_by_fold": p1s - p0s}


# ----------------------------------------------------------------------------- metrics

def ece_equal_mass(y, p, bins: int = 10) -> float:
    import numpy as np
    order = np.argsort(p, kind="mergesort")
    out = 0.0
    for idx in np.array_split(order, bins):
        out += len(idx) * abs(float(np.mean(p[idx])) - float(np.mean(y[idx])))
    return out / len(y)


def ece_debiased(y, p, bins: int = 10) -> float:
    """Debiased squared calibration error with equal-mass bins (Brocker 2012; Ferro and Fricker 2012, as Roelofs et
    al. 2022 use it): the signed estimate of the squared error (it can be negative when the bias correction exceeds
    the raw gap, which means no calibration error is detectable)."""
    import numpy as np
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, bins):
        n = len(idx)
        ybar, pbar = float(np.mean(y[idx])), float(np.mean(p[idx]))
        total += n / len(y) * ((ybar - pbar) ** 2 - ybar * (1 - ybar) / max(n - 1, 1))
    return float(total)


def ece_noise_floor(p, sims: int = 200, seed: int = SEED) -> dict:
    """ECE (10 equal-mass bins) of labels drawn from p itself (perfect calibration), at the same n."""
    import numpy as np
    rng = np.random.default_rng(seed)
    vals = [ece_equal_mass(rng.random(len(p)) < p, p) for _ in range(sims)]
    return {"mean": float(np.mean(vals)), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def per_shot_loss(y, p):
    import numpy as np
    from analytics.eval import EPS
    q = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def paired_by_game(games, a, b, n: int = BOOT, seed: int = SEED) -> dict:
    """Mean per-shot difference a - b with a 95 percent percentile interval from a bootstrap of games."""
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"g": games, "d": a - b}).groupby("g")["d"].agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(s.sum() / c.sum()), "lo": float(lo), "hi": float(hi), "games": int(len(s)), "resamples": n}


def _arm_metrics(y, p) -> dict:
    from analytics.eval import brier, calibration_slope
    return {"log_loss": float(per_shot_loss(y, p).mean()), "brier": brier(y, p),
            "calibration_slope": calibration_slope(y, p), "ece_equal_mass_10": ece_equal_mass(y, p),
            "ece_debiased_squared_signed": ece_debiased(y, p)}


def arm_intervals(games, y, p, n: int = BOOT, seed: int = SEED) -> dict:
    """95 percent percentile intervals of every per-arm metric from a bootstrap of games (n resamples, seed 0)."""
    import numpy as np
    g = np.asarray(games)
    uniq = np.unique(g)
    rows = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in _arm_metrics(y[:2], p[:2]) }
    for _ in range(n):
        idx = np.concatenate([rows[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        m = _arm_metrics(y[idx], p[idx])
        for k in vals:
            vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def per_game_t_interval(games, a, b) -> dict:
    """Sensitivity check fixed before the test score (PLAN.md D80): a t interval on the per-game mean differences
    a - b, one value per game, for when the games are few."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    d = pd.DataFrame({"g": games, "d": a - b}).groupby("g")["d"].mean().to_numpy()
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean_of_game_means": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half), "games": int(len(d))}


def summarize(games, y, arms: dict, with_intervals: bool = True) -> dict:
    import numpy as np
    out = {"shots": int(len(y)), "games": int(len(set(games))), "arms": {}, "arm_intervals_95": {}}
    for name in ("p_iso", "p_cvap", "p_ivap"):
        out["arms"][name] = _arm_metrics(y, arms[name])
        if with_intervals:
            out["arm_intervals_95"][name] = arm_intervals(games, y, arms[name])
    out["ece_noise_floor_of_p_iso"] = ece_noise_floor(arms["p_iso"])
    loss = {k: per_shot_loss(y, arms[k]) for k in ("p_iso", "p_cvap", "p_ivap")}
    out["paired_cvap_minus_iso_shot_weighted"] = paired_by_game(games, loss["p_cvap"], loss["p_iso"])
    out["paired_ivap_minus_iso_shot_weighted"] = paired_by_game(games, loss["p_ivap"], loss["p_iso"])
    out["t_interval_cvap_minus_iso_per_game"] = per_game_t_interval(games, loss["p_cvap"], loss["p_iso"])
    width = arms["hi"] - arms["lo"]
    deciles = np.quantile(arms["p_cvap"], np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(deciles, arms["p_cvap"], side="right") - 1, 0, 9)
    out["interval_width"] = {"definition": "GM(p1) - GM(p0) over the 3 inner folds (PLAN.md D80)",
                             "median": float(np.median(width)), "p95": float(np.percentile(width, 95)),
                             "max": float(width.max()),
                             "median_by_decile_of_p": [float(np.median(width[which == d])) for d in range(10)],
                             "per_fold_p95": [float(np.percentile(w, 95)) for w in arms["widths_by_fold"]]}
    upper = out["paired_cvap_minus_iso_shot_weighted"]["hi"]
    out["verdict"] = {"threshold": THRESHOLD, "upper_95": upper, "passes_threshold": bool(upper <= THRESHOLD),
                      "width_p95_under_0_01": bool(out["interval_width"]["p95"] < POINT_ESTIMATE_WIDTH)}
    return out


# ----------------------------------------------------------------------------- commands

def _features(artifacts=None):
    import pandas as pd
    frame = pd.read_parquet(features_path(artifacts))
    training_games_only(frame["sportvu_game"].astype(str).unique(), WHO)
    return frame


def _outer(frame, games, seed: int = SEED):
    """Out-of-fold arms over grouped outer folds of `games` -> (rows, arms concatenated in row order)."""
    import numpy as np
    from analytics.eval import grouped_folds
    rows = frame[frame["sportvu_game"].astype(str).isin(set(games))].reset_index(drop=True)
    g = rows["sportvu_game"].astype(str)
    out = {k: np.full(len(rows), np.nan) for k in ("p_iso", "p_cvap", "p_ivap", "lo", "hi")}
    out["widths_by_fold"] = np.full((3, len(rows)), np.nan)
    fold = np.full(len(rows), -1)
    for k, (_tr, te) in enumerate(grouped_folds(g.unique(), min(OUTER_FOLDS, g.nunique()), seed)):
        m = g.isin(te).to_numpy()
        parts = fit_parts(rows[~m].reset_index(drop=True), seed)
        arms = predict_arms(parts, rows[m].reset_index(drop=True))
        for key in ("p_iso", "p_cvap", "p_ivap", "lo", "hi"):
            out[key][m] = arms[key]
        out["widths_by_fold"][:, m] = arms["widths_by_fold"]
        fold[m] = k
        print(f"[research] outer fold {k}: {int(m.sum())} test shots", flush=True)
    out["fold"] = fold
    return rows, out


def gate(artifacts=None) -> dict:
    import os
    import numpy as np
    import pandas as pd
    from tools.frozen_models import key_mismatches
    folder = begin_rebuild(research_dir(artifacts) / "xfg_e1_gate", "xfg")
    frame = _features(artifacts)
    stored = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet",
                             columns=STORED_KEYS + ["p_selected", "fold"])
    stored = stored[stored["fold"] >= 0].reset_index(drop=True)          # training rows only
    rows, arms = _outer(frame, frame["sportvu_game"].astype(str).unique())
    keys = key_mismatches(stored, rows) if len(stored) == len(rows) else {"length": [len(stored), len(rows)]}
    if any(keys.values()):
        raise SystemExit(f"{WHO}: the cached rows do not line up with the stored predictions: {keys}")
    diff = float(np.max(np.abs(arms["p_iso"] - stored["p_selected"].to_numpy())))
    folds = int((arms["fold"] != stored["fold"].to_numpy()).sum())
    pd.DataFrame({"p_iso": arms["p_iso"], "fold": arms["fold"]}).to_parquet(folder / "gate_oof.parquet", index=False)
    record = {"name": "xfg", "compared": "the isotonic arm refit by tools.research_xfg on all training games under the "
              "frozen outer folds against the stored out-of-fold p_selected (fold >= 0); no held-out row, no metric",
              "n": int(len(rows)), "n_expected": int(len(stored)), "max_abs_diff": diff,
              "tolerance": TOLERANCE_CAP["xfg"], "fold_mismatches": folds, "passed": folds == 0}
    write_check(folder, record, ["gate_oof.parquet"])
    return require_passed(folder, "xfg")


def dev(artifacts=None) -> dict:
    require_passed(research_dir(artifacts) / "xfg_e1_gate", "xfg")
    frame = _features(artifacts)
    games = dev_games(frame["sportvu_game"].astype(str).unique(), WHO)
    refuse_test(games, WHO)
    rows, arms = _outer(frame, games)
    out = {"experiment": "xfg-E1", "stage": "development games, 5 grouped outer folds, seed 0",
           **summarize(rows["sportvu_game"].astype(str).to_numpy(), rows["made"].astype(int).to_numpy(), arms)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "xfg-E1-dev.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def research_data_root(artifacts=None) -> Path:
    """Where tools.research_ingest put the untouched test games in the DATA_ROOT layout: V3_RESEARCH_DATA, else
    _research-data/sportvu_untouched beside the artifacts folder."""
    import os
    named = os.environ.get("V3_RESEARCH_DATA")
    return (Path(named).expanduser().resolve() if named
            else artifacts_dir(artifacts).parent / "_research-data" / "sportvu_untouched")


def test_features(test_games, artifacts=None):
    """The frozen xFG features of the given games, built with the frozen code from the research data root. Called
    only inside the final scoring call, with the games score_once hands it."""
    from analytics.xfg import train as T
    from analytics.xfg import validate as V
    root = research_data_root(artifacts)
    tables, _summaries = V.clean(V.load_data_root(root, "2015-16"))
    rows, held = T.assemble(tables)
    if len(held):
        raise SystemExit(f"{WHO}: a spent held-out game is in the research data root")
    rows = rows[rows["sportvu_game"].astype(str).isin(set(test_games))].reset_index(drop=True)
    missing = sorted(set(test_games) - set(rows["sportvu_game"].astype(str)))
    if missing:
        raise SystemExit(f"{WHO}: no shots built for {len(missing)} test game(s); score them all or none")
    return rows


def final(artifacts=None) -> dict:
    require_passed(research_dir(artifacts) / "xfg_e1_gate", "xfg")
    if not (RESULTS / "xfg-E1-dev.json").exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test set is scored after the development result exists")
    frame = _features(artifacts)
    games = dev_games(frame["sportvu_game"].astype(str).unique(), WHO)
    rows = frame[frame["sportvu_game"].astype(str).isin(set(games))].reset_index(drop=True)
    parts = fit_parts(rows)

    def scorer(test_games):
        test = test_features(test_games, artifacts)
        arms = predict_arms(parts, test)
        return {"experiment": "xfg-E1", "stage": "fitted on every development game, scored once on the D71 test set",
                "has_contest_share": {"test": float(test["has_contest"].mean()),
                                      "development": float(rows["has_contest"].mean())},
                **summarize(test["sportvu_game"].astype(str).to_numpy(), test["made"].astype(int).to_numpy(), arms)}

    out = score_once("xfg-E1", TRACKING, scorer)
    (RESULTS / "xfg-E1-final.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("features", "e1-gate", "e1-dev", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    run = {"features": build_features, "e1-gate": gate, "e1-dev": dev, "e1-final": final}[a.command]
    with heavy_job(f"research xfg {a.command}"):
        out = run(a.artifacts)
    print(json.dumps(out.get("verdict", out) if isinstance(out, dict) else out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

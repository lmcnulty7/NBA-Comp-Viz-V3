"""tools/research_winprob_e2.py: winprob-E2, a forward-in-time temperature for the over-confidence
(docs/research/winprob.md section 6; PLAN.md D71, D79, D89, D91).

    python3 -m export.heavy --name winprob-E2 -- python3 -m tools.research_winprob_e2 e2-dev   [--artifacts DIR]
    python3 -m export.heavy --name winprob-E2 -- python3 -m tools.research_winprob_e2 e2-final [--artifacts DIR]

winprob-E1 was adopted as a research result, so, as the pre-registration says, E2 runs on top of the E1 model: the
"no temperature" arm is E1 (the frozen logistic's recipe plus mu_hat and mu_drift from Elo), not the frozen logistic.
Everything E1 built is reused unchanged (tools.research_winprob and tools.research_winprob_elo): the cached
development state frame, the Elo parameters chosen on 2020-21 to 2023-24, 2024-25 read as Elo state only (D89), and
the season scoring path that the scorer runs on 2025-26.

The temperature is one number a that multiplies the logit of the E1 probability, p' = sigmoid(a * logit(p)), with no
intercept; it is fitted by log loss (a above 1 sharpens, below 1 softens; T = 1 / a). Arms:

    e1          the E1 model, no temperature
    e2          E1 with the forward temperature: fitted on the predictions of a model fitted on 2020-21 to 2022-23
                for 2023-24 (the pre-registered split)
    e2_grouped  the ablation: E1 with a temperature fitted on out-of-fold predictions under the frozen protocol
                (5 grouped folds by game inside the development seasons, seed 0)

Development stage (dev seasons only): nested forward chaining. For a check season s, e2's temperature comes from the
season before s (a model fitted on the seasons before s - 1 predicting s - 1) and is applied to a model fitted on the
seasons before s predicting s; e2_grouped's temperature comes from grouped folds over the seasons before s. That
gives two check seasons, 2022-23 and 2023-24 (2021-22 has no earlier forward season). Writes
reports/research/winprob-E2-dev.json.

Final: the three arms fitted on every development row, the forward temperature from 2023-24 as pre-registered, the
grouped temperature from all development seasons, scored once on 2025-26 through score_once. Writes
reports/research/winprob-E2-final.json.

Metric (pre-registered): the mean over the five regulation buckets of ECE (10 equal-width bins, the frozen
analytics.eval.ece), primary; mean bucket Brier as a guard; a pathwise peak-on-loss check, descriptive. Adoption: e2
minus e1 mean bucket ECE has its paired 95 percent interval below zero, and the upper end of the paired interval of
e2 minus e1 mean bucket Brier is at most BRIER_GUARD.
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from export.heavy import heavy_job, peak_rss_gb
from tools import research_testset as R
from tools import research_winprob as W
from tools import research_winprob_elo as E
from tools.frozen_common import MemoryGuard, _plain
from tools.research_testset import dev_games, score_once, scoring_data_root

WHO = "tools.research_winprob_e2"
EXPERIMENT = "winprob-E2"
RESULTS = W.RESULTS
SEED, BOOT, FOLDS = W.SEED, W.BOOT, W.FOLDS
BRIER_GUARD = 0.0005            # pre-registered: upper end of the paired Brier interval at or below +0.0005
BINS = 10
NEAR_ONE = 0.02                 # fixed before any E2 result: a temperature with |log a| at or below this is "near 1"
PEAK_LEVELS = (0.9, 0.95)
ARMS = ("e1", "e2", "e2_grouped")
PAIRS = (("e2", "e1"), ("e2_grouped", "e1"), ("e2", "e2_grouped"))
FORWARD_FIT_SEASON = "2023-24"  # pre-registered: fit on 2020-21 to 2022-23, predict 2023-24
CHECK_SEASONS = ("2022-23", "2023-24")
COEF_TOLERANCE = 1e-9


# ----------------------------------------------------------------------------- temperature

def _logit(p):
    from analytics.eval import EPS
    q = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(q) - np.log1p(-q)


def apply_temperature(p, a: float):
    return 1.0 / (1.0 + np.exp(-a * _logit(p)))


def fit_temperature(y, p) -> dict:
    """The logit multiplier a minimising log loss of sigmoid(a * logit(p)) (bounded search on [0.2, 5])."""
    from scipy.optimize import minimize_scalar
    from analytics.eval import log_loss
    y = np.asarray(y, dtype=float)
    z = _logit(p)

    def loss(a):
        q = 1.0 / (1.0 + np.exp(-a * z))
        return log_loss(y, q)
    r = minimize_scalar(loss, bounds=(0.2, 5.0), method="bounded", options={"xatol": 1e-6})
    a = float(r.x)
    return {"a": a, "T": 1.0 / a, "rows": int(len(y)), "log_loss_before": float(loss(1.0)),
            "log_loss_after": float(r.fun), "near_one": bool(abs(np.log(a)) <= NEAR_ONE)}


# ----------------------------------------------------------------------------- the E1 model, refit

def fit_e1(train: pd.DataFrame):
    """The E1 arm (tools.research_winprob._fit_logistic on the E1 columns), development rows only."""
    ids = sorted(train["game_id"].astype(str).unique())
    if train["heldout"].astype(bool).any() or dev_games(ids, WHO, R.PBP) != ids:
        raise SystemExit(f"{WHO}: a row outside development reached the fit")
    return W._fit_logistic(train, W.arm_columns()["e1"])


def grouped_oof(train: pd.DataFrame) -> np.ndarray:
    """E1's out-of-fold predictions under 5 grouped folds by game (seed 0) inside the given rows."""
    from analytics.eval import grouped_folds
    g = train["game_id"].astype(str)
    p = np.full(len(train), np.nan)
    for tr, te in grouped_folds(sorted(g.unique()), n_folds=FOLDS, seed=SEED):
        mtr, mte = g.isin(tr).to_numpy(), g.isin(te).to_numpy()
        model, _ = fit_e1(train[mtr])
        p[mte] = model(train[mte])
    return p


def _coef_diff(info: dict, recorded: dict) -> float:
    a = np.asarray(info["coef_standardised"] + [info["intercept"]])
    b = np.asarray(recorded["coef_standardised"] + [recorded["intercept"]])
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")


# ----------------------------------------------------------------------------- metrics

def summarize(rows: pd.DataFrame, n_boot: int = BOOT, seed: int = SEED) -> dict:
    """Per arm: mean bucket ECE (10 equal-width bins, analytics.eval.ece's definition), per-bucket ECE, mean bucket
    Brier, Brier and log loss with 95 percent percentile intervals from one bootstrap of whole games; the paired
    differences on the same resamples; slopes as point estimates; the pathwise peak-on-loss check; the verdict."""
    from analytics.eval import EPS, calibration_slope_fast, ece
    F = W._F()
    names = list(F.BUCKET_NAMES)
    reg = [b[0] for b in F.BUCKETS]
    R5 = len(reg)
    codes, uniq = pd.factorize(rows["game_id"].astype(str), sort=True)
    G, B = len(uniq), len(names)
    bidx = pd.Categorical(rows["time_bucket"].astype(str), categories=names).codes.astype(int)
    if (bidx < 0).any():
        raise SystemExit(f"{WHO}: a row has an unknown time bucket")
    y = rows["home_win"].astype(float).to_numpy()
    cell = codes * B + bidx
    n_gb = np.bincount(cell, minlength=G * B).reshape(G, B).astype(float)
    cnt = n_gb.sum(axis=1)
    Wt = W._boot_weights(G, n_boot, seed)
    Wn, Wc = Wt @ n_gb, Wt @ cnt
    out = {"rows": int(len(rows)), "games": int(G), "resamples": n_boot, "seed": seed,
           "rows_by_bucket": {b: int(n_gb[:, i].sum()) for i, b in enumerate(names)}, "arms": {}, "paired": {}}
    bad = int((~np.isfinite(rows[["p_" + a for a in ARMS]].to_numpy(dtype=float))).sum())
    out["non_finite_values"] = bad
    if bad:
        out["verdict"] = {"refused": True, "reading": f"refused: {bad} non-finite value(s) reached the verdict"}
        return out
    point, boot = {}, {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for arm in ARMS:
            p = np.clip(rows["p_" + arm].astype(float).to_numpy(), EPS, 1 - EPS)
            k = np.clip(np.floor(p * BINS), 0, BINS - 1).astype(int)
            c3 = cell * BINS + k
            sp = np.bincount(c3, weights=p, minlength=G * B * BINS).reshape(G, B * BINS)
            sy = np.bincount(c3, weights=y, minlength=G * B * BINS).reshape(G, B * BINS)
            ece_pt = (np.abs(sp.sum(0) - sy.sum(0)).reshape(B, BINS).sum(axis=1)) / n_gb.sum(axis=0)
            ece_bt = np.abs(Wt @ sp - Wt @ sy).reshape(n_boot, B, BINS).sum(axis=2) / Wn
            se = np.bincount(cell, weights=(p - y) ** 2, minlength=G * B).reshape(G, B)
            ll = np.bincount(codes, weights=-(y * np.log(p) + (1 - y) * np.log1p(-p)), minlength=G)
            br_pt, br_bt = se.sum(0) / n_gb.sum(0), (Wt @ se) / Wn
            point[arm] = {"ece": ece_pt, "mean_ece": float(ece_pt[:R5].mean()), "brier_b": br_pt,
                          "mean_brier": float(br_pt[:R5].mean()), "brier": float(se.sum() / cnt.sum()),
                          "log_loss": float(ll.sum() / cnt.sum())}
            boot[arm] = {"ece": ece_bt, "mean_ece": ece_bt[:, :R5].mean(axis=1), "brier_b": br_bt,
                         "mean_brier": br_bt[:, :R5].mean(axis=1), "brier": (Wt @ se.sum(axis=1)) / Wc,
                         "log_loss": (Wt @ ll) / Wc}
            for i, b in enumerate(names):          # the vectorised ECE equals the frozen function's
                m = bidx == i
                if m.sum() and abs(ece(y[m], p[m], BINS) - ece_pt[i]) > 1e-9:
                    raise SystemExit(f"{WHO}: the ECE of {arm} in {b} disagrees with analytics.eval.ece")
            a = {"mean_bucket_ece": W._ci(point[arm]["mean_ece"], boot[arm]["mean_ece"]),
                 "bucket_ece": {b: W._ci(ece_pt[i], ece_bt[:, i]) for i, b in enumerate(names)},
                 "mean_bucket_brier": W._ci(point[arm]["mean_brier"], boot[arm]["mean_brier"]),
                 "bucket_brier": {b: W._ci(br_pt[i], br_bt[:, i]) for i, b in enumerate(names)},
                 "brier": W._ci(point[arm]["brier"], boot[arm]["brier"]),
                 "log_loss": W._ci(point[arm]["log_loss"], boot[arm]["log_loss"]),
                 "calibration_slope_point_estimate": {}}
            for i, b in enumerate(names):
                m = bidx == i
                if m.sum():
                    a["calibration_slope_point_estimate"][b] = float(calibration_slope_fast(y[m], p[m]))
            out["arms"][arm] = a
        for x, z in PAIRS:
            px, pz, bx, bz = point[x], point[z], boot[x], boot[z]
            out["paired"][f"{x}_minus_{z}"] = {
                "mean_bucket_ece": W._ci(px["mean_ece"] - pz["mean_ece"], bx["mean_ece"] - bz["mean_ece"]),
                "bucket_ece": {b: W._ci(px["ece"][i] - pz["ece"][i], bx["ece"][:, i] - bz["ece"][:, i])
                               for i, b in enumerate(names)},
                "mean_bucket_brier": W._ci(px["mean_brier"] - pz["mean_brier"], bx["mean_brier"] - bz["mean_brier"]),
                "bucket_brier": {b: W._ci(px["brier_b"][i] - pz["brier_b"][i], bx["brier_b"][:, i] - bz["brier_b"][:, i])
                                 for i, b in enumerate(names)},
                "log_loss": W._ci(px["log_loss"] - pz["log_loss"], bx["log_loss"] - bz["log_loss"])}
    out["peak_on_loss"] = peak_on_loss(rows, codes, G, Wt)
    out["verdict"] = verdict(out["paired"])
    return out


def peak_on_loss(rows: pd.DataFrame, codes, G: int, Wt) -> dict:
    """Descriptive, in the spirit of Pipping-Gamon and Wyner (2026), not their exact benchmark. For a level x and each
    game and side, the first event where that side's win probability reaches x: if the forecast is calibrated, the
    side then loses with probability 1 - p at that event (the martingale property at a stopping time). Observed
    losses after reaching x are compared with that expected count, with an interval of observed / expected from the
    same bootstrap of games. Rows must be in event order within each game."""
    y = rows["home_win"].astype(int).to_numpy()
    out = {}
    for arm in ARMS:
        p = rows["p_" + arm].astype(float).to_numpy()
        res = {}
        for x in PEAK_LEVELS:
            obs, exp = np.zeros(G), np.zeros(G)
            for side_p, side_lost in ((p, y == 0), (1.0 - p, y == 1)):
                hit = side_p >= x
                first = pd.Series(np.where(hit, np.arange(len(p)), np.iinfo(np.int64).max)).groupby(codes).min()
                ok = first.to_numpy() < np.iinfo(np.int64).max
                idx = first.to_numpy()[ok]
                g = first.index.to_numpy()[ok]
                exp[g] += 1.0 - side_p[idx]
                obs[g] += side_lost[idx].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_bt = (Wt @ obs) / (Wt @ exp)
            res[str(x)] = {"games_reaching": int((exp > 0).sum()), "observed_losses": float(obs.sum()),
                           "expected_losses": float(exp.sum()),
                           "observed_over_expected": W._ci(obs.sum() / exp.sum() if exp.sum() else float("nan"),
                                                           ratio_bt)}
        out[arm] = res
    return out


def verdict(paired: dict) -> dict:
    """The pre-registered rule, read on e2 minus e1: adopt when the paired interval of mean bucket ECE lies below
    zero and the upper end of the paired interval of mean bucket Brier is at most BRIER_GUARD."""
    d = paired["e2_minus_e1"]
    e, b = d["mean_bucket_ece"], d["mean_bucket_brier"]
    ece_falls = bool(e["hi"] < 0)
    brier_ok = bool(b["hi"] <= BRIER_GUARD)
    adopt = ece_falls and brier_ok
    if adopt:
        reading = "pass"
    elif not ece_falls and e["lo"] > 0:
        reading = "fail: ECE rises"
    elif not ece_falls:
        reading = "fail: the ECE interval includes zero"
    else:
        reading = "fail: ECE falls but the Brier guard is broken"
    g = paired["e2_minus_e2_grouped"]["mean_bucket_ece"]
    return {"ece_difference": e["point"], "ece_lo": e["lo"], "ece_hi": e["hi"], "ece_falls": ece_falls,
            "brier_difference": b["point"], "brier_lo": b["lo"], "brier_hi": b["hi"], "brier_guard": BRIER_GUARD,
            "brier_guard_holds": brier_ok, "adopt": adopt, "reading": reading,
            "ablation": {"e2_minus_e2_grouped_ece": g["point"], "lo": g["lo"], "hi": g["hi"],
                         "forward_beats_grouped": bool(g["hi"] < 0)}}


# ----------------------------------------------------------------------------- scoring one season

def score(root, season: str, allowed, elo_start, model, a_fwd: float, a_grp: float):
    """tools.research_winprob.score_season, unchanged, with E1 in every arm slot; the temperatures are applied here.
    -> (rows with p_e1, p_e2, p_e2_grouped, pregame, info)."""
    models = {"baseline": model, "e1": model, "ablation": model}
    rows, pre, info, _ = W.score_season(root, season, allowed, elo_start, models)
    rows = rows.drop(columns=["p_e1", "p_ablation"]).rename(columns={"p_baseline": "p_e1"})
    rows["p_e2"] = apply_temperature(rows["p_e1"].to_numpy(), a_fwd)
    rows["p_e2_grouped"] = apply_temperature(rows["p_e1"].to_numpy(), a_grp)
    return rows, pre, info


# ----------------------------------------------------------------------------- development

def _season_lt(states, season):
    return states[states["season"].astype(str) < season]


def _season_eq(states, season):
    return states[states["season"].astype(str) == season]


def dev(artifacts=None) -> dict:
    gate_record = W.require_gate(artifacts)
    guard = MemoryGuard()
    e1_dev = json.loads((RESULTS / "winprob-E1-dev.json").read_text())
    e1_final = json.loads((RESULTS / "winprob-E1-final.json").read_text())
    root = W.data_root()
    schedule = E.dev_schedule(root)
    params = E.choose_params(schedule)["chosen"]
    if params != e1_dev["elo"]["chosen"]:
        raise SystemExit(f"{WHO}: the Elo choice differs from winprob-E1's")
    states = W.attach_mu(W.load_dev_states(artifacts), E.advance(E.Elo(**params), schedule))
    # gate: the E1 arm refit here equals the E1 fits winprob-E1 recorded (forward folds and the final fit)
    gate_e1 = {}
    models, fwd_preds = {}, {}
    for season in E.CHECK_SEASONS:
        model, info = fit_e1(_season_lt(states, season))
        gate_e1[season] = _coef_diff(info, e1_dev["fits_by_season"][season]["e1"])
        models[season] = model
        allowed = sorted(_season_eq(states, season)["game_id"].astype(str).unique())
        rows, pre, sinfo, _ = W.score_season(root, season, allowed, E.state_after(params, schedule, season),
                                             {"baseline": model, "e1": model, "ablation": model})
        fwd_preds[season] = (rows, sinfo)
        guard.check(season)
    _, info_all = fit_e1(states)
    gate_e1["all_development"] = _coef_diff(info_all, e1_final["setup"]["fits"]["e1"])
    if max(gate_e1.values()) > COEF_TOLERANCE:
        raise SystemExit(f"{WHO}: the E1 refit does not reproduce winprob-E1's fits: {gate_e1}")
    # forward temperatures: fitted on season s's forward predictions
    t_fwd = {s: fit_temperature(fwd_preds[s][0]["home_win"], fwd_preds[s][0]["p_baseline"]) for s in E.CHECK_SEASONS}
    # grouped temperatures: on out-of-fold predictions under the frozen protocol inside the seasons before s
    t_grp = {}
    for season in CHECK_SEASONS + ("all_development",):
        part = states if season == "all_development" else _season_lt(states, season)
        t_grp[season] = fit_temperature(part["home_win"], grouped_oof(part))
        print(f"[e2 dev] grouped temperature before {season}: {t_grp[season]['a']:.4f}", flush=True)
    seasons, pooled = {}, []
    prev = {"2022-23": "2021-22", "2023-24": "2022-23"}
    for season in CHECK_SEASONS:
        rows, sinfo = fwd_preds[season]
        a_f, a_g = t_fwd[prev[season]]["a"], t_grp[season]["a"]
        r = rows.drop(columns=["p_e1", "p_ablation"]).rename(columns={"p_baseline": "p_e1"})
        r["p_e2"] = apply_temperature(r["p_e1"].to_numpy(), a_f)
        r["p_e2_grouped"] = apply_temperature(r["p_e1"].to_numpy(), a_g)
        seasons[season] = {"temperature_forward_from": prev[season], "a_forward": a_f, "a_grouped": a_g,
                           "info": {k: sinfo[k] for k in ("counts", "schedule_agreement", "elo_prefix_violations")},
                           "summary": summarize(r)}
        pooled.append(r)
    pooled_summary = summarize(pd.concat(pooled, ignore_index=True))
    mb = pooled_summary["paired"]["e2_minus_e1"]["mean_bucket_ece"]
    half = (mb["hi"] - mb["lo"]) / 2
    out = {"experiment": EXPERIMENT,
           "stage": ("development seasons 2020-21 to 2023-24 only, on top of the winprob-E1 model; nested forward "
                     "chaining: the check seasons 2022-23 and 2023-24 each take e2's temperature from the season "
                     "before (fitted on forward predictions) and e2_grouped's from grouped folds over the seasons "
                     "before; paired bootstrap by game, 2000 resamples, seed 0"),
           "pre_registration": "docs/research/winprob.md section 6, winprob-E2",
           "gate_frozen_logistic": gate_record,
           "gate_e1_refit_max_coef_diff": gate_e1, "elo_params": params,
           "temperatures": {"forward_by_season_fitted_on": t_fwd, "grouped_before_season": t_grp,
                            "final_forward_pre_registered": t_fwd[FORWARD_FIT_SEASON],
                            "final_grouped": t_grp["all_development"], "near_one_rule": f"|log a| <= {NEAR_ONE}"},
           "by_season": seasons, "pooled_nested_forward": pooled_summary,
           "power": {"development_games": pooled_summary["games"], "development_half_width_ece": half,
                     "test_games": W.TEST_GAMES_EXPECTED,
                     "scaled_half_width_at_test_game_count": half * float(np.sqrt(pooled_summary["games"]
                                                                                 / W.TEST_GAMES_EXPECTED))},
           "peak_gb": round(peak_rss_gb(), 2)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "winprob-E2-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


# ----------------------------------------------------------------------------- the one final score

def final(artifacts=None) -> dict:
    head = W.clean_head()
    gate_record = W.require_gate(artifacts)
    dev_path = RESULTS / "winprob-E2-dev.json"
    if not dev_path.exists():
        raise SystemExit(f"{WHO}: run e2-dev first; the test season is scored after the development result exists")
    dev_out = json.loads(dev_path.read_text())
    root = W.data_root()
    schedule = E.dev_schedule(root)
    params = E.choose_params(schedule)["chosen"]
    if params != dev_out["elo_params"]:
        raise SystemExit(f"{WHO}: the Elo choice differs from the development record")
    states = W.attach_mu(W.load_dev_states(artifacts), E.advance(E.Elo(**params), schedule))
    model, fit_info = fit_e1(states)
    a_fwd = dev_out["temperatures"]["final_forward_pre_registered"]["a"]
    a_grp = dev_out["temperatures"]["final_grouped"]["a"]
    elo, state_record = E.state_through_2024_25(params, schedule, root)
    states_sha = json.loads((W.research_dir(artifacts) / "winprob_states_dev.json").read_text())["sha256"]
    gate_files = json.loads((W.research_dir(artifacts) / "winprob_e1_gate" / "check.json").read_text())["files"]
    del states

    def scorer(test_games):
        troot = scoring_data_root(R.PBP)
        rows, pre, info = score(troot, R.PBP_TEST_SEASON, test_games, elo, model, a_fwd, a_grp)
        results = summarize(rows)
        if info["elo_prefix_violations"]:
            results["verdict"] = {"refused": True, "computed_but_not_read": results["verdict"],
                                  "reading": (f"refused: the test-season Elo prefix check found "
                                              f"{len(info['elo_prefix_violations'])} violation(s)")}
        return {"experiment": EXPERIMENT,
                "stage": ("E1 fitted on every development row (2020-21 to 2023-24), temperatures fitted on development "
                          "seasons only, Elo carried through 2024-25 as state (D89) and updated game by game through "
                          "the test season; scored once"),
                "code_commit": head, "dev_states_sha256": states_sha, "gate_files_sha256": gate_files,
                "temperatures": {"a_forward": a_fwd, "a_grouped": a_grp},
                "test_season": info, "results": results}

    out = score_once(EXPERIMENT, R.PBP, scorer)
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"gate_frozen_logistic": gate_record, "elo_params": params, "elo_state": state_record,
                        "fit_e1": fit_info, "temperatures": dev_out["temperatures"]}}
    (RESULTS / "winprob-E2-final.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e2-dev", "e2-final"))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    run = {"e2-dev": dev, "e2-final": final}[a.command]
    started = time.time()
    with heavy_job(f"research winprob {a.command}"):
        out = run(a.artifacts)
    v = (out.get("pooled_nested_forward") or out.get("scored", {}).get("results") or {}).get("verdict", {})
    print(json.dumps(_plain(v), indent=2), f"\n{time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

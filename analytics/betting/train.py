"""analytics/betting/train.py: the betting market candidates under the held-out season protocol.

    python -m analytics.betting.train --fixtures | --data DATA_ROOT --report reports/betting.md [--out OUT_ROOT]
        [--seed 0] [--bootstrap 200] [--folds 5] [--heldout-season 2024-25]
        [--fixture-games 40] [--fixture-seasons 2020-21,2021-22]

Protocol
  1. validate and join (analytics.betting.validate), build betting_games (analytics.betting.features); rows that the
     collectors marked heldout (sportvu/splits.json games) are dropped before anything.
  2. The held-out unit is a whole season: --heldout-season when present, else the latest season when there are at
     least two, else none (then only the cross-validation block is reported). Its rows are set aside before any fit;
     seasons later than it are dropped before the split and listed in the report as ignored.
     analytics.eval.refuse_heldout runs on the training game ids; features.assert_no_leak runs on every design matrix.
  3. Candidates, each giving a cover probability p_cover and a margin forecast:
       market    p_cover 0.5, margin forecast minus the closing home spread (the line itself)
       logistic  logistic regression of home_cover on FEATURES (standardised, C=1); margin forecast the market's
       ridge     ridge regression of home_margin on FEATURES (standardised, alpha=1); p_cover = Phi((mu + spread) / sigma)
                 with sigma the standard deviation of the training residuals (the normal residual conversion)
     The spread never enters a feature matrix; it enters only the ridge conversion and the CLV target.
  4. Selection inside the training seasons: out-of-fold predictions under grouped folds by game (analytics.eval.
     grouped_folds), out-of-season by construction once the held-out season is aside; the cover model with the lowest
     out-of-fold log loss and the margin model with the lowest out-of-fold MSE are selected (pushes and rows without
     a spread carry no cover label and are excluded from the cover metrics). Intervals: bootstrap by game.
  5. Staking rule, fixed here before any scoring (never tuned): flat one unit on the home cover when p_cover exceeds
     STAKE_THRESHOLD, on the away cover when 1 - p_cover exceeds it, nothing otherwise. A win returns WIN_RETURN =
     1 - 2 * VIG units (VIG 4.5 percent a side, about -110), a loss costs one unit, a push returns the stake.
     ROI = total profit / total units staked.
  6. The held-out season is scored once per run by the selected models refit on every training row: log loss, Brier,
     MSE, profit and ROI with a bootstrap interval by game. The report labels it held-out, scored once.

Outputs under OUT_ROOT/analytics/betting/<season>/: betting_games.parquet, betting_predictions.parquet,
betting_summary.parquet (docs/contract/betting.md). A --fixtures run writes a report that says fixture output, not
a result. With --fixtures and no --out the tables go to a temporary directory (nothing derived from odds is ever
written into the repository).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analytics.betting import features as _features
from analytics.betting import validate as _validate
from analytics.eval import bootstrap_by_game, brier, fit_threads, fmt_ci, fmt_num, grouped_folds, log_loss, refuse_heldout

WHO = "betting.train"
CANDIDATES = ("market", "logistic", "ridge")

# ── the staking rule, fixed before any scoring ─────────────────────────────────────────────────────
STAKE_THRESHOLD = 0.55     # bet only when the model's cover probability for a side exceeds this
VIG = 0.045                # the book's margin per side; break-even probability is 1 / (1 + WIN_RETURN) = 0.5236
WIN_RETURN = 1.0 - 2.0 * VIG   # 0.91 units won on a one-unit winning bet (about -110)
UNIT = 1.0
FIXTURE_TAG = "(fixture output, not a result)"
MAC_CMD = ("python -m analytics.betting.train --data \"$DATA_ROOT\" --out \"$OUT_ROOT\" --report reports/betting.md "
           "--seed 0 --bootstrap 1000 --heldout-season 2024-25")


def stakes(p_cover) -> np.ndarray:
    """+UNIT on the home cover, -UNIT on the away cover, 0 otherwise (sign is the side)."""
    p = np.asarray(p_cover, dtype=float)
    return np.where(p > STAKE_THRESHOLD, UNIT, np.where(1.0 - p > STAKE_THRESHOLD, -UNIT, 0.0))


def win_return(price=None, n: int | None = None) -> np.ndarray:
    """Units won per unit staked: from the collected American spread price when present and finite (-110 -> 0.909,
    +100 -> 1.0), else the nominal WIN_RETURN (the 4.5 percent vig a side)."""
    if price is None:
        return np.full(int(n), WIN_RETURN)
    p = np.asarray(price, dtype=float)
    ret = np.where(p < 0, 100.0 / np.abs(np.where(p < 0, p, -100.0)), p / 100.0)
    return np.where(np.isfinite(p) & (np.abs(p) >= 100), ret, WIN_RETURN)


def pnl(stake, home_margin, closing_spread_home, price=None) -> np.ndarray:
    """Profit of each stake against the result: home covers when margin + spread > 0, push when it is 0.
    No spread or no stake gives 0. A win pays win_return(price) per unit (the collected spread price when present,
    the nominal WIN_RETURN otherwise); a loss costs the unit."""
    s = np.asarray(stake, dtype=float)
    cover = np.asarray(home_margin, dtype=float) + np.asarray(closing_spread_home, dtype=float)
    out = np.zeros(len(s))
    live = (s != 0) & np.isfinite(cover)
    win = live & (np.sign(cover) == np.sign(s))
    lose = live & (cover != 0) & ~win
    ret = win_return(price, len(s))
    out[win] = np.abs(s[win]) * ret[win]
    out[lose] = -np.abs(s[lose])
    return out


def _price_of(frame: pd.DataFrame):
    """The collected home spread price when the frame carries it, else None (nominal return)."""
    return frame["spread_price_home"].to_numpy(dtype=float) if "spread_price_home" in frame.columns else None


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def fit_models(train: pd.DataFrame) -> dict:
    """Fit every candidate on one training frame -> {name: (predict_p(frame), predict_margin(frame))}.
    Every fit of the trainer goes through here; a held-out row raises."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if train["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_models" % WHO)
    cols = _features.feature_columns(train)
    market_margin = lambda frame: -frame["closing_spread_home"].astype(float).to_numpy()
    out = {"market": (lambda frame: np.full(len(frame), 0.5), market_margin)}
    lab = train[train["home_cover"].notna()]
    y = lab["home_cover"].astype(int).to_numpy()
    if len(lab) >= 2 and len(np.unique(y)) == 2:
        with fit_threads():
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(_features.design(lab, cols).to_numpy(), y)
        out["logistic"] = (lambda frame, m=lr: m.predict_proba(_features.design(frame, cols).to_numpy())[:, 1], market_margin)
    else:
        out["logistic"] = out["market"]
    mg = train[train["home_margin"].notna()]
    if len(mg) >= 2:
        with fit_threads():
            rg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(_features.design(mg, cols).to_numpy(), mg["home_margin"].astype(float).to_numpy())
        resid = mg["home_margin"].astype(float).to_numpy() - rg.predict(_features.design(mg, cols).to_numpy())
        sigma = float(max(np.std(resid, ddof=1) if len(resid) > 1 else 12.0, 1.0))

        def ridge_margin(frame, m=rg):
            return m.predict(_features.design(frame, cols).to_numpy())

        def ridge_p(frame, s=sigma):
            mu = ridge_margin(frame)
            spread = frame["closing_spread_home"].astype(float).to_numpy()
            p = norm.cdf((mu + spread) / s)
            return np.where(np.isfinite(p), p, 0.5)
        out["ridge"] = (ridge_p, ridge_margin)
    else:
        out["ridge"] = out["market"]
    return out


def predict_frame(models: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """p_<candidate>, margin_<candidate> for every candidate on frame's index."""
    pred = pd.DataFrame(index=frame.index)
    for c in CANDIDATES:
        p_fn, m_fn = models[c]
        pred["p_" + c] = np.clip(np.asarray(p_fn(frame), dtype=float), 1e-6, 1 - 1e-6)
        pred["margin_" + c] = np.asarray(m_fn(frame), dtype=float)
    return pred


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold candidate predictions under grouped folds by game -> (train with p_*, margin_*, fold; note)."""
    games = sorted(train["game_id"].astype(str).unique())
    pred = train.copy()
    pred["fold"] = -1
    if len(games) < 2:
        p = predict_frame(fit_models(train), train)
        for c in p.columns:
            pred[c] = p[c]
        return pred, {"mode": "in-sample", "folds": 0, "text": "one training game: metrics are IN-SAMPLE"}
    k = min(n_folds, len(games))
    for c in CANDIDATES:
        pred["p_" + c], pred["margin_" + c] = np.nan, np.nan
    for i, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr, te = train["game_id"].astype(str).isin(tr_games), train["game_id"].astype(str).isin(te_games)
        p = predict_frame(fit_models(train[tr]), train[te])
        for c in p.columns:
            pred.loc[te, c] = p[c]
        pred.loc[te, "fold"] = i
    text = "out-of-fold under %d grouped folds by game inside the training seasons (seed %d)" % (k, seed)
    return pred, {"mode": "out-of-fold", "folds": k, "text": text}


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────
def _mse(y, m) -> float:
    y, m = np.asarray(y, dtype=float), np.asarray(m, dtype=float)
    ok = np.isfinite(y) & np.isfinite(m)
    return float(np.mean((y[ok] - m[ok]) ** 2)) if ok.any() else float("nan")


def _roi(d: pd.DataFrame) -> float:
    staked = float(np.abs(d["stake"]).sum())
    return float(d["pnl"].sum() / staked) if staked > 0 else float("nan")


def _boot(df: pd.DataFrame, fn, n: int, seed: int) -> dict:
    """bootstrap_by_game without the all-NaN percentile warning (a season with no bet has no ROI)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return bootstrap_by_game(df, "game", fn, n=n, seed=seed)


def score_candidates(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{candidate: {log_loss, brier, mse, roi, n_bets, pnl: {point, lo, hi}}} with bootstrap by game."""
    out = {}
    for c in CANDIDATES:
        lab = pred[pred["home_cover"].notna()]
        d = pd.DataFrame({"game": lab["game_id"].astype(str).to_numpy(), "y": lab["home_cover"].astype(float).to_numpy(),
                          "p": lab["p_" + c].astype(float).to_numpy()})
        res = {}
        if len(d) >= 1:
            res["log_loss"] = bootstrap_by_game(d, "game", lambda s: log_loss(s["y"], s["p"]), n=n_boot, seed=seed)
            res["brier"] = bootstrap_by_game(d, "game", lambda s: brier(s["y"], s["p"]), n=n_boot, seed=seed)
        else:
            res["log_loss"] = res["brier"] = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
        m = pd.DataFrame({"game": pred["game_id"].astype(str).to_numpy(), "y": pred["home_margin"].astype(float).to_numpy(),
                          "m": pred["margin_" + c].astype(float).to_numpy()})
        res["mse"] = bootstrap_by_game(m, "game", lambda s: _mse(s["y"], s["m"]), n=n_boot, seed=seed)
        st = stakes(pred["p_" + c])
        b = pd.DataFrame({"game": pred["game_id"].astype(str).to_numpy(), "stake": st,
                          "pnl": pnl(st, pred["home_margin"], pred["closing_spread_home"], _price_of(pred))})
        res["n_bets"] = int((b["stake"] != 0).sum())
        res["pnl"] = bootstrap_by_game(b, "game", lambda s: float(s["pnl"].sum()), n=n_boot, seed=seed)
        res["roi"] = _boot(b, _roi, n_boot, seed)
        out[c] = res
    return out


def select(scores: dict) -> dict:
    """cover_model: lowest out-of-fold log loss; margin_model: lowest out-of-fold MSE among market and ridge."""
    cover = min(CANDIDATES, key=lambda c: (np.nan_to_num(scores[c]["log_loss"]["point"], nan=np.inf), CANDIDATES.index(c)))
    margin = min(("market", "ridge"), key=lambda c: (np.nan_to_num(scores[c]["mse"]["point"], nan=np.inf), CANDIDATES.index(c)))
    return {"cover_model": cover, "margin_model": margin}


def predictions_table(pred: pd.DataFrame, chosen: dict, heldout: bool) -> pd.DataFrame:
    """betting_predictions rows (docs/contract/betting.md) from the chosen candidates' columns."""
    p = pred["p_" + chosen["cover_model"]].astype(float).to_numpy()
    m = pred["margin_" + chosen["margin_model"]].astype(float).to_numpy()
    st = stakes(p)
    spread = pred["closing_spread_home"].astype(float).to_numpy()
    return pd.DataFrame({"game_id": pred["game_id"].astype(str).to_numpy(), "season": pred["season"].astype(str).to_numpy(),
                         "p_cover": p, "margin_forecast": m, "clv": m - (-spread), "stake": st,
                         "pnl": pnl(st, pred["home_margin"], spread, _price_of(pred)), "fold": pred["fold"].astype(int).to_numpy(),
                         "heldout": heldout, "cover_model": chosen["cover_model"], "margin_model": chosen["margin_model"]})


def summary_rows(pt: pd.DataFrame, games: pd.DataFrame, mode: str, n_boot: int, seed: int) -> list[dict]:
    """betting_summary rows per season: n_games, log_loss, brier, mse, n_bets, pnl, roi, lo, hi (ROI interval)."""
    rows = []
    g = games.set_index("game_id")
    for season, s in pt.groupby("season", sort=True):
        y = g.loc[s["game_id"], "home_cover"].astype(float).to_numpy()
        ym = g.loc[s["game_id"], "home_margin"].astype(float).to_numpy()
        ok = np.isfinite(y)
        b = pd.DataFrame({"game": s["game_id"].to_numpy(), "stake": s["stake"].to_numpy(), "pnl": s["pnl"].to_numpy()})
        roi = _boot(b, _roi, n_boot, seed)
        rows.append({"season": str(season), "mode": mode, "n_games": int(len(s)), "n_labelled": int(ok.sum()),
                     "log_loss": float(log_loss(y[ok], s["p_cover"].to_numpy()[ok])) if ok.any() else float("nan"),
                     "brier": float(brier(y[ok], s["p_cover"].to_numpy()[ok])) if ok.any() else float("nan"),
                     "mse": _mse(ym, s["margin_forecast"].to_numpy()), "n_bets": int((s["stake"] != 0).sum()),
                     "pnl": float(s["pnl"].sum()), "roi": roi["point"], "lo": roi["lo"], "hi": roi["hi"]})
    return rows


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def write_report(path: Path, ctx: dict) -> str:
    """The markdown report of a run (ctx from run); every heading over fixture numbers carries the fixture tag."""
    fx = ctx["fixtures"]
    tag = " %s" % FIXTURE_TAG if fx else ""
    L = ["# Betting market model (analytics/betting/train.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture output, not a result.** This run used the synthetic games and odds of analytics.fixtures (synth_stints, "
              "synth_odds); it proves "
              "the pipeline runs end to end and nothing else. Odds history is a paid feed (The Odds API through "
              "collect/extras.py --odds) that has not been collected.", ""]
    L += ["## Inputs", "", "| source | value |", "| --- | --- |"]
    for k, v in ctx["files"].items():
        L.append("| %s | %s |" % (k, ", ".join(v) if v else "none"))
    cl, jn = ctx["counts"]["closing"], ctx["counts"]["join"]
    L += ["", "## Join coverage (odds to games: team nickname key, game date within one day of the UTC tip minus 8 hours)%s" % tag, "",
          "| count | value |", "| --- | --- |",
          "| odds rows | %d |" % cl["rows"], "| odds games (game_key) | %d |" % cl["games"],
          "| closing rows after resolving duplicates | %d |" % cl["closing_rows"],
          "| duplicate closing snapshots resolved (nearest tip kept) | %d |" % cl["duplicate_snapshots"],
          "| games with more than one book (median point, mean de-vigged probability) | %d |" % cl["multi_book_games"],
          "| games with no is_close snapshot (latest pre-tip used) | %d |" % cl["fallback_preclose"],
          "| games with no pre-tip snapshot (dropped) | %d |" % cl["no_pretip_snapshot"],
          "| spread out of [-30, 30] (nulled) | %d |" % cl["spread_out_of_range"],
          "| total out of [150, 300] (nulled) | %d |" % cl["total_out_of_range"],
          "| games in the games table | %d |" % jn["games"], "| matched to a closing line | %d (%.1f percent) |" % (jn["matched"], 100 * jn["coverage"]),
          "| games without odds | %d |" % jn["games_without_odds"], "| odds without a game | %d |" % jn["odds_without_game"],
          "| matched with spread / total / moneyline | %d / %d / %d |" % (jn["matched_with_spread"], jn["matched_with_total"], jn["matched_with_moneyline"]),
          "| team names given / inferred from dates / ids unresolved | %d / %d / %d |" % (jn.get("team_names_given", 0), jn.get("team_names_inferred", 0), jn.get("team_ids_unresolved", 0)),
          "| collector held-out rows dropped | %d |" % ctx["counts"]["games_heldout_dropped"], "",
          "## Protocol%s" % tag, "",
          "Unit: one game. Targets: home_cover (home margin plus closing home spread above 0; null on a push or without a spread) and "
          "clv = margin forecast minus the market margin forecast (minus the closing home spread).",
          "Features (previous games only): %s. Guard: analytics.schema.assert_time_forward plus the betting target list and a "
          "correlation check against the final margin (over %.2f refused) on every design matrix." % (", ".join(ctx["features"]), _features.LEAK_CORR),
          "Held-out unit: season %s, %d games set aside before any fit; `analytics.eval.refuse_heldout` checked %d training game ids."
          % (ctx["heldout_season"] or "none", ctx["heldout_games"], ctx["train_games"]),
          "Training seasons: %s (%d games, %d with a cover label). Seasons later than the held-out season, ignored (never fitted, never "
          "scored): %s." % (", ".join(ctx["train_seasons"]), ctx["train_games"], ctx["train_labelled"], ", ".join(ctx["ignored_seasons"]) or "none"),
          "Selection: %s; cover model by out-of-fold log loss, margin model by out-of-fold MSE; intervals are 2.5 and 97.5 percentiles of %d "
          "bootstrap resamples of whole games (seed %d)." % (ctx["protocol"], ctx["bootstrap"], ctx["seed"]), "",
          "Staking rule, fixed in code before scoring (train.py constants): flat %.0f unit on the home cover when p_cover > %.2f, on the away "
          "cover when 1 - p_cover > %.2f, else no bet; a win returns the collected spread price when the row carries one, else the nominal "
          "%.2f units (vig %.1f percent a side, about -110); a loss costs one unit, a push returns the stake; ROI is profit over units "
          "staked. betting_games carries no spread price, so the run reports the nominal return." % (UNIT, STAKE_THRESHOLD, STAKE_THRESHOLD, WIN_RETURN, 100 * VIG), "",
          "## Candidates (%s, %d games)%s" % (ctx["mode"], ctx["train_games"], tag), "",
          "| candidate | cover log loss | Brier | margin MSE | bets | profit (units) | ROI |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in CANDIDATES:
        s = ctx["scores"][c]
        L.append("| %s | %s | %s | %s | %d | %s | %s |" % (c, fmt_ci(s["log_loss"], 4), fmt_ci(s["brier"], 4), fmt_ci(s["mse"], 1), s["n_bets"],
                                                        fmt_ci(s["pnl"], 2), fmt_ci(s["roi"], 3)))
    L += ["", "Selected: cover model **%s**, margin model **%s**." % (ctx["chosen"]["cover_model"], ctx["chosen"]["margin_model"]), "",
          "## Profit and loss per season, selected models%s" % tag, "",
          "| season | mode | games | labelled | log loss | Brier | margin MSE | bets | profit | ROI | lo | hi |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in ctx["summary"]:
        L.append("| %s | %s | %d | %d | %s | %s | %s | %d | %s | %s | %s | %s |" % (
            r["season"], r["mode"], r["n_games"], r["n_labelled"], fmt_num(r["log_loss"], 4), fmt_num(r["brier"], 4), fmt_num(r["mse"], 1), r["n_bets"],
            fmt_num(r["pnl"], 2), fmt_num(r["roi"], 3), fmt_num(r["lo"], 3), fmt_num(r["hi"], 3)))
    L.append("")
    if ctx["heldout_season"]:
        L.append("The row marked held-out is season %s scored ONCE by the selected models refit on every training row; it was never "
                 "fitted or tuned on. Training-season rows are out-of-fold under the grouped folds." % ctx["heldout_season"])
    else:
        L.append("No held-out season: the input carried one season, so only the out-of-fold block is reported.")
    L += ["", "## Outputs", "", "Tables under %s: betting_games, betting_predictions, betting_summary "
          "(docs/contract/betting.md)." % ctx["out_root_label"], "", "## Status", ""]
    if fx:
        L.append("Fixture output, not a result. The real run on the Mac, once odds have been bought and collected with "
                 "`ODDS_API_KEY=... python -m collect.extras --odds --seasons ...`, is one command:")
    else:
        L.append("Run on the collected odds table with the command")
    L += ["", "    " + MAC_CMD, "",
          "Standing caveat: odds history is a paid feed not yet collected; until it is, every number here is fixture output.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The train CLI: --fixtures or --data, the report and output paths, the protocol knobs and the fixture knobs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic fixtures of analytics.fixtures "
                     "(fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with collect/extras/odds and collect/stints")
    ap.add_argument("--out", default=None, help="OUT_ROOT (fixtures default to a temporary directory)")
    ap.add_argument("--report", default="reports/betting.md", help="markdown report path (default reports/betting.md)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the folds, the bootstrap and the fixtures (default 0)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples of whole games (default 200)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by game inside the training seasons (default 5)")
    ap.add_argument("--heldout-season", default="2024-25", help="the season set aside and scored once (default 2024-25)")
    _validate.add_fixture_args(ap)
    return ap.parse_args(argv)


def run(data: dict, a, fixtures: bool, argv) -> dict:
    """The whole protocol on the loaded data and the parsed arguments -> ctx (what write_report renders)."""
    res = _validate.validate(data["odds"], data["games"], data.get("team_names"))
    joined = res["joined"]
    held_season = _features.pick_heldout_season(joined["season"].unique(), a.heldout_season)
    joined, ignored = _features.drop_later_seasons(joined, held_season)      # never fit on seasons after the held-out one
    out_root = a.out or (tempfile.mkdtemp(prefix="betting_fixture_") if fixtures else None)
    if out_root is None:
        raise SystemExit("%s: --out OUT_ROOT is required with --data" % WHO)
    bg = _features.build_games(joined, held_season, out_root)
    train, heldout = bg[~bg["heldout"]].reset_index(drop=True), bg[bg["heldout"]].reset_index(drop=True)
    if len(train) == 0:
        raise SystemExit("%s: no training rows" % WHO)
    refuse_heldout(sorted(train["game_id"].astype(str).unique()), WHO)
    _features.assert_no_leak(_features.design(train), train["home_margin"])
    pred, note = out_of_fold(train, a.folds, a.seed)
    scores = score_candidates(pred, a.bootstrap, a.seed)
    chosen = select(scores)
    pt = predictions_table(pred, chosen, heldout=False)
    summary = summary_rows(pt, bg, "cv", a.bootstrap, a.seed)
    if len(heldout):                                     # the held-out season, scored once
        models = fit_models(train)
        hp = predict_frame(models, heldout)
        hp = pd.concat([heldout, hp], axis=1)
        hp["fold"] = -1
        ht = predictions_table(hp, chosen, heldout=True)
        pt = pd.concat([pt, ht], ignore_index=True)
        summary += summary_rows(ht, bg, "heldout", a.bootstrap, a.seed)
    _features.write_games(bg, out_root)
    for season, s in pt.groupby("season", sort=True):
        d = Path(out_root) / "analytics" / "betting" / str(season)
        d.mkdir(parents=True, exist_ok=True)
        s.reset_index(drop=True).to_parquet(d / "betting_predictions.parquet", index=False)
        pd.DataFrame([r for r in summary if r["season"] == str(season)]).to_parquet(d / "betting_summary.parquet", index=False)
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python -m analytics.betting.train " + " ".join(argv if argv is not None else sys.argv[1:]),
           "fixtures": fixtures, "files": data["files"], "counts": res["counts"], "features": _features.feature_columns(train),
           "heldout_season": held_season, "heldout_games": int(len(heldout)), "train_games": int(len(train)),
           "train_labelled": int(train["home_cover"].notna().sum()), "train_seasons": sorted(train["season"].astype(str).unique()),
           "ignored_seasons": ignored,
           "protocol": note["text"], "mode": note["mode"], "bootstrap": a.bootstrap, "seed": a.seed, "scores": scores,
           "chosen": chosen, "summary": summary, "out_root": str(out_root), "games": bg, "predictions": pt,
           "out_root_label": ("`%s/analytics/betting/<season>/`" % a.out) if a.out else "a temporary directory (no --out given), `analytics/betting/<season>/`"}
    return ctx


def main(argv=None) -> dict:
    """Run the trainer from the command line, write the report and return ctx."""
    a = parse_args(argv)
    data = _validate.fixture_data(a.seed, n_games=a.fixture_games, seasons=_validate.fixture_seasons(a.fixture_seasons)) \
        if a.fixtures else _validate.load_data(a.data)
    ctx = run(data, a, a.fixtures, argv)
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()

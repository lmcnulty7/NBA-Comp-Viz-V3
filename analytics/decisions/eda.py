"""analytics/decisions/eda.py: the tables of notebooks/eda_decisions.ipynb (no plots; pandas frames only).

    res = fixture_run(seed=0, n_games=6, n_possessions=30)      # or train.run(bundles, ...) on real bundles
    kinds_per_possession(res["events"])                          # decisions per possession by kind
    value_distribution(res["events"])                            # quantiles of chosen and best values by kind
    quality_by_kind(res["events"])                               # mean, sd, n of decision quality by kind
    correlation_table(res)                                       # per-game Spearman against on-court offensive rating
    shuffled_check(res)                                          # the same under a within-game permutation

Every table built from fixtures is fixture output, not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.decisions import train as T
from analytics.decisions import validate as V

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def fixture_run(seed: int = 0, n_games: int = 6, n_possessions: int = 30, heldout_every: int = 3,
                folds: int = 3, n_boot: int = 50) -> dict:
    """train.run on synthetic games (small by default so a notebook re-runs in seconds)."""
    bundles = V.load_fixture_bundles(seed, n_games, heldout_every, n_possessions)
    return T.run(bundles, folds, seed, n_boot)


def kinds_per_possession(events: pd.DataFrame) -> pd.DataFrame:
    """Per game: possessions with decisions, mean decisions per possession and the mean count by kind."""
    if not len(events):
        return pd.DataFrame(columns=["sportvu_game", "possessions", "decisions_per_possession", "shot", "pass", "drive"])
    k = events.groupby(["sportvu_game", "possession", "decision_kind"]).size().unstack(fill_value=0)
    for c in ("shot", "pass", "drive"):
        if c not in k.columns:
            k[c] = 0
    k = k.reset_index()
    g = k.groupby("sportvu_game").agg(possessions=("possession", "nunique"), shot=("shot", "mean"), pass_=("pass", "mean"),
                                      drive=("drive", "mean")).reset_index()
    g = g.rename(columns={"pass_": "pass"})
    g["decisions_per_possession"] = g[["shot", "pass", "drive"]].sum(axis=1)
    return g[["sportvu_game", "possessions", "decisions_per_possession", "shot", "pass", "drive"]]


def value_distribution(events: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of chosen_value and best_value (expected points) by decision kind."""
    rows = []
    for kind, d in events.groupby("decision_kind"):
        for col in ("chosen_value", "best_value"):
            q = d[col].quantile(QUANTILES)
            rows.append({"decision_kind": kind, "value": col, "n": int(len(d)), "mean": float(d[col].mean()),
                         **{"q%02d" % int(p * 100): float(v) for p, v in q.items()}})
    return pd.DataFrame(rows)


def quality_by_kind(events: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and count of decision quality by kind, plus the share of decisions at the best option."""
    if not len(events):
        return pd.DataFrame(columns=["decision_kind", "n", "mean_quality", "sd_quality", "share_best"])
    g = events.groupby("decision_kind").agg(n=("decision_quality", "size"), mean_quality=("decision_quality", "mean"),
                                            sd_quality=("decision_quality", "std"),
                                            share_best=("decision_quality", lambda s: float((s >= -1e-9).mean()))).reset_index()
    return g


def best_option_table(events: pd.DataFrame) -> pd.DataFrame:
    """Counts of the best option by decision kind (shot, pass_to_<rank>, hold)."""
    if not len(events):
        return pd.DataFrame()
    return events.groupby(["decision_kind", "best_option"]).size().unstack(fill_value=0)


def correlation_table(res: dict) -> pd.DataFrame:
    """Per game: Spearman between the players' summed quality and their on-court offensive rating, under each prior."""
    parts = []
    for prior, v in res["validation"].items():
        d = v["per_game"].copy()
        d.insert(0, "prior", prior)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["prior", "sportvu_game", "rho", "n_players"])


def correlation_summary(res: dict) -> pd.DataFrame:
    """The observed and shuffled validation summaries (mean rho, interval, sign test) per prior, one row each."""
    rows = []
    for prior, v in res["validation"].items():
        for name in ("summary", "shuffled"):
            rows.append({"prior": prior, "which": "observed" if name == "summary" else "shuffled", **v[name]})
    return pd.DataFrame(rows)


def shuffled_check(res: dict, n_shuffles: int = 20, seed: int = 0) -> pd.DataFrame:
    """Mean Spearman under n_shuffles within-game permutations of the players' summed quality, next to the observed one."""
    ev, ortg = res["events"], res["ortg"]
    pg = T.player_game_quality(ev)
    obs = T.per_game_spearman(pg, ortg)["rho"].mean()
    vals = [T.per_game_spearman(T.shuffled_quality(pg, seed + i), ortg)["rho"].mean() for i in range(n_shuffles)]
    vals = np.array(vals, dtype=float)
    return pd.DataFrame([{"observed_mean_rho": float(obs), "shuffled_mean": float(np.nanmean(vals)),
                          "shuffled_sd": float(np.nanstd(vals)), "n_shuffles": n_shuffles,
                          "share_shuffles_at_or_above_observed": float(np.nanmean(vals >= obs))}])

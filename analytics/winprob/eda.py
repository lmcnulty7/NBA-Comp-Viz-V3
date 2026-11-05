"""analytics/winprob/eda.py: the tables of notebooks/eda_winprob.ipynb (no plots; pandas only).

Every function takes the winprob_states frame of analytics.winprob.features.build_states and returns a
DataFrame. The notebook runs them on the synthetic fixtures, so its output is fixture output, not a result.

    margin_distribution       quantiles of the margin per time bucket
    time_distribution         rows per period and per time bucket
    win_rate_by_margin_bucket home win rate by margin bin (rows) and time bucket (columns), with counts
    spread_coverage           share of rows and games with a closing spread, per season
    shuffled_label_check      a logistic regression on margin and time under grouped folds scores no better
                              than the base rate once home_win is shuffled across games (a leakage smoke
                              check: a feature that encodes the outcome would survive the shuffle only if it
                              were rebuilt from the shuffled labels, which the state builder cannot do)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import brier, grouped_cv, refuse_heldout
from analytics.winprob import features as F

MARGIN_BINS = (-100, -20, -15, -10, -6, -3, -1, 0, 2, 5, 9, 14, 19, 100)


def margin_distribution(states: pd.DataFrame) -> pd.DataFrame:
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = []
    for b in F.BUCKET_NAMES:
        s = states.loc[states["time_bucket"] == b, "margin"].astype(float)
        if len(s) == 0:
            continue
        r = {"time_bucket": b, "n": int(len(s)), "mean_abs_margin": float(s.abs().mean())}
        for qq, v in zip(q, s.quantile(q).tolist()):
            r["q%02d" % int(qq * 100)] = float(v)
        rows.append(r)
    return pd.DataFrame(rows)


def time_distribution(states: pd.DataFrame) -> pd.DataFrame:
    a = states.groupby("period").size().rename("rows").reset_index().assign(kind="period").rename(columns={"period": "value"})
    b = states.groupby("time_bucket").size().rename("rows").reset_index().assign(kind="time_bucket").rename(columns={"time_bucket": "value"})
    out = pd.concat([a, b], ignore_index=True)
    out["value"] = out["value"].astype(str)
    out["share"] = out["rows"] / max(len(states), 1)
    return out[["kind", "value", "rows", "share"]]


def win_rate_by_margin_bucket(states: pd.DataFrame) -> pd.DataFrame:
    s = states.copy()
    s["margin_bin"] = pd.cut(s["margin"].astype(float), bins=MARGIN_BINS, right=False)
    tab = s.pivot_table(index="margin_bin", columns="time_bucket", values=F.TARGET, aggfunc="mean", observed=False)
    n = s.pivot_table(index="margin_bin", columns="time_bucket", values=F.TARGET, aggfunc="size", observed=False)
    cols = [b for b in F.BUCKET_NAMES if b in tab.columns]
    tab, n = tab[cols], n[cols]
    out = tab.copy()
    out.columns = ["win_rate_%s" % c for c in cols]
    for c in cols:
        out["n_%s" % c] = n[c].fillna(0).astype(int)
    out.index = out.index.astype(str)
    return out.reset_index()


def spread_coverage(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, s in states.groupby("season", sort=True):
        g = s.drop_duplicates("game_id")
        rows.append({"season": season, "games": int(len(g)), "games_with_spread": int((g["spread_missing"] == 0).sum()),
                     "row_share_with_spread": float(1 - s["spread_missing"].mean()),
                     "possession_known_share": float(s["possession_home"].notna().mean())})
    return pd.DataFrame(rows)


def shuffled_label_check(states: pd.DataFrame, seed: int = 0, n_folds: int = 5) -> pd.DataFrame:
    """Out-of-fold Brier of a logistic regression on the real labels and on labels shuffled across games. Rows flagged
    heldout are dropped first, refuse_heldout runs on the game ids and assert_time_forward on the columns (the only fit
    outside analytics.winprob.train).
    Expected: the shuffled run sits at the base-rate Brier (about 0.25); far below it would mean a feature
    carries the (shuffled) outcome, which no time-forward feature can."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = ["margin", "seconds_remaining", "margin_x_sqrt_time"]
    states = states[~states["heldout"].astype(bool)].reset_index(drop=True)
    refuse_heldout(sorted(states["game_id"].astype(str).unique()), "winprob.eda")
    schema.assert_time_forward(states[cols], "states_1hz")
    rng = np.random.default_rng(seed)
    games = states["game_id"].astype(str)
    uniq = sorted(games.unique())
    real = states[F.TARGET].astype(int).to_numpy()
    by_game = states.drop_duplicates("game_id").set_index("game_id")[F.TARGET].astype(int)
    perm = pd.Series(by_game.to_numpy()[rng.permutation(len(by_game))], index=by_game.index)
    shuffled = games.map(perm).astype(int).to_numpy()
    rows = []
    k = min(n_folds, len(uniq))
    for name, y in (("real labels", real), ("shuffled labels (across games)", shuffled)):
        d = states[cols].astype(float).copy()
        d["game_id"] = games.to_numpy()
        d["y"] = y
        fit = lambda tr: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(tr[cols].to_numpy(), tr["y"].to_numpy())
        pred = lambda m, te: m.predict_proba(te[cols].to_numpy())[:, 1]
        oof = grouped_cv(d, "game_id", fit, pred, n_folds=k, seed=seed)
        rows.append({"labels": name, "rows": int(len(d)), "games": len(uniq), "base_rate": float(np.mean(y)),
                     "brier_base_rate": float(np.mean(y) * (1 - np.mean(y))), "brier_oof_logistic": brier(y, oof["p"].to_numpy())})
    return pd.DataFrame(rows)


def all_tables(states: pd.DataFrame, seed: int = 0) -> dict:
    return {"margin_distribution": margin_distribution(states), "time_distribution": time_distribution(states),
            "win_rate_by_margin_bucket": win_rate_by_margin_bucket(states), "spread_coverage": spread_coverage(states),
            "shuffled_label_check": shuffled_label_check(states, seed)}


def main(argv=None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    d = F.load_fixtures(a.seed) if a.fixtures else F.load_data(a.data)
    states = F.build_states(d["playbyplay"], d["games"], d["odds"])
    tabs = all_tables(states, a.seed)
    for name, t in tabs.items():
        print("## %s%s\n" % (name, " (fixture output, not a result)" if a.fixtures else ""))
        print(t.to_string(index=False))
        print()
    return tabs


if __name__ == "__main__":
    main()

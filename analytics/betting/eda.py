"""analytics/betting/eda.py: the tables behind notebooks/eda_betting.ipynb. Every function returns a DataFrame or a
dict; render_markdown turns a frame into a markdown table (no plotting library, no tabulate).

    tables = run_all(seed=0)              # the fixtures; run_all(data_root=...) for the collected odds
    print(render_markdown(tables["cover_rate_by_spread_bin"]))

    python -m analytics.betting.eda --fixtures | --data DATA_ROOT [--seed 0] [--heldout-season 2024-25]
        [--fixture-games 40] [--fixture-seasons 2020-21,2021-22]

Tables
    spread_distribution       closing home spread: count, mean, sd, quantiles, share of favourites at home
    margin_distribution       final home margin: the same, plus the margin minus the market forecast (the line's error);
                              training seasons only (held-out outcomes are never looked at)
    cover_rate_by_spread_bin  home cover rate per spread bin (a fair line gives about 0.5 everywhere); training seasons only
    join_coverage             the validate counts as one two-column table
    shuffled_label_check      out-of-fold log loss of the logistic and ridge candidates on the true labels and on
                              labels shuffled within season: a pipeline that leaks shows a gap on shuffled labels too
                              (the shuffled run should sit at the market's 0.6931, give or take noise)

Fixture numbers are fixture output, not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.betting import features as _features
from analytics.betting import validate as _validate
from analytics.eval import md_table

SPREAD_BINS = (-30, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 30)


def load_games(seed: int = 0, data_root=None, heldout_season: str = "2024-25", n_games: int = 40,
               seasons=("2020-21", "2021-22")) -> dict:
    """-> dict(games=betting_games frame, counts=validate counts, fixtures=bool)."""
    data = _validate.load_data(data_root) if data_root else _validate.fixture_data(seed, n_games=n_games, seasons=seasons)
    res = _validate.validate(data["odds"], data["games"], data.get("team_names"))
    held = _features.pick_heldout_season(res["joined"]["season"].unique(), heldout_season)
    joined, ignored = _features.drop_later_seasons(res["joined"], held)
    return {"games": _features.build_games(joined, held), "counts": res["counts"], "fixtures": data_root is None,
            "heldout_season": held, "ignored_seasons": ignored}


def _describe(s: pd.Series, name: str) -> pd.DataFrame:
    s = pd.to_numeric(s, errors="coerce").dropna()
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if len(s) else pd.Series([np.nan] * 5, index=[0.05, 0.25, 0.5, 0.75, 0.95])
    return pd.DataFrame([{"column": name, "n": int(len(s)), "mean": float(s.mean()) if len(s) else np.nan,
                          "sd": float(s.std()) if len(s) > 1 else np.nan, "q05": q[0.05], "q25": q[0.25], "median": q[0.5],
                          "q75": q[0.75], "q95": q[0.95], "min": float(s.min()) if len(s) else np.nan,
                          "max": float(s.max()) if len(s) else np.nan}])


def spread_distribution(bg: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the closing home spread over every season."""
    t = _describe(bg["closing_spread_home"], "closing_spread_home")
    t["home_favourite_share"] = float((bg["closing_spread_home"] < 0).sum() / max(bg["closing_spread_home"].notna().sum(), 1))
    return t


def training_rows(bg: pd.DataFrame) -> pd.DataFrame:
    """bg without the held-out season: outcome tables never look at held-out results."""
    return bg[~bg["heldout"].astype(bool)] if "heldout" in bg.columns else bg


def margin_distribution(bg: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the final home margin and of the margin minus the market, training seasons only."""
    bg = training_rows(bg)
    err = bg["home_margin"] + bg["closing_spread_home"]      # positive when the home side beat the line
    return pd.concat([_describe(bg["home_margin"], "home_margin"), _describe(err, "home_margin_minus_market")], ignore_index=True)


def cover_rate_by_spread_bin(bg: pd.DataFrame) -> pd.DataFrame:
    """Home cover rate per closing-spread bin, training seasons only."""
    lab = training_rows(bg)
    lab = lab[lab["home_cover"].notna()].copy()
    lab["bin"] = pd.cut(lab["closing_spread_home"], bins=list(SPREAD_BINS), include_lowest=True)
    out = lab.groupby("bin", observed=False).agg(n=("home_cover", "size"), cover_rate=("home_cover", "mean"),
                                                mean_margin=("home_margin", "mean"), mean_spread=("closing_spread_home", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def join_coverage(counts: dict) -> pd.DataFrame:
    """The validate counts as one two-column table (count, value)."""
    rows = [{"count": "closing." + k, "value": v} for k, v in counts["closing"].items()]
    rows += [{"count": "join." + k, "value": v} for k, v in counts["join"].items()]
    rows.append({"count": "games_heldout_dropped", "value": counts["games_heldout_dropped"]})
    out = pd.DataFrame(rows)
    out["value"] = out["value"].astype(object)      # keep the ints as ints next to the coverage fraction
    return out


def shuffled_label_check(bg: pd.DataFrame, seed: int = 0, folds: int = 5) -> pd.DataFrame:
    """Out-of-fold cover log loss per candidate on the true labels and on labels shuffled within season (the margin
    and spread shuffled together so the cover label stays consistent with them). Training-season rows only."""
    from analytics.betting import train as _train
    train = bg[~bg["heldout"].astype(bool)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    shuffled = train.copy()
    for _s, idx in shuffled.groupby("season").groups.items():
        idx = np.asarray(list(idx))
        perm = rng.permutation(idx)
        for c in ("home_margin", "closing_spread_home", "home_cover"):
            shuffled.loc[idx, c] = train.loc[perm, c].to_numpy()
    rows = []
    for label, frame in (("true", train), ("shuffled", shuffled)):
        pred, _ = _train.out_of_fold(frame, folds, seed)
        lab = pred[pred["home_cover"].notna()]
        for c in _train.CANDIDATES:
            rows.append({"labels": label, "candidate": c, "n": int(len(lab)),
                         "log_loss": float(_train.log_loss(lab["home_cover"], lab["p_" + c])),
                         "mse": _train._mse(pred["home_margin"], pred["margin_" + c])})
    return pd.DataFrame(rows)


def render_markdown(df: pd.DataFrame, nd: int = 4) -> str:
    """A table as markdown (analytics.eval.md_table): floats to nd decimals, NaN blank."""
    return "\n".join(md_table(df, nd))


def run_all(seed: int = 0, data_root=None, heldout_season: str = "2024-25", n_games: int = 40,
            seasons=("2020-21", "2021-22")) -> dict:
    """Every EDA table of the notebook -> dict(games, fixtures, heldout_season, and one frame per table)."""
    d = load_games(seed, data_root, heldout_season, n_games=n_games, seasons=seasons)
    bg = d["games"]
    return {"games": bg, "fixtures": d["fixtures"], "heldout_season": d["heldout_season"],
            "spread_distribution": spread_distribution(bg), "margin_distribution": margin_distribution(bg),
            "cover_rate_by_spread_bin": cover_rate_by_spread_bin(bg), "join_coverage": join_coverage(d["counts"]),
            "shuffled_label_check": shuffled_label_check(bg, seed)}


TABLES = ("spread_distribution", "margin_distribution", "cover_rate_by_spread_bin", "join_coverage", "shuffled_label_check")


def main(argv=None) -> dict:
    """Print every EDA table as markdown from the command line (fixture output under --fixtures) and return them."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic fixtures of analytics.fixtures "
                     "(fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with collect/extras/odds and collect/stints")
    ap.add_argument("--seed", type=int, default=0, help="seed of the fixtures and the shuffled-label check (default 0)")
    ap.add_argument("--heldout-season", default="2024-25", help="the season no outcome table reads (default 2024-25)")
    _validate.add_fixture_args(ap)
    a = ap.parse_args(argv)
    t = run_all(a.seed, None if a.fixtures else a.data, a.heldout_season, n_games=a.fixture_games,
                seasons=_validate.fixture_seasons(a.fixture_seasons))
    for k in TABLES:
        print("\n## %s%s\n" % (k, " (fixture output, not a result)" if t["fixtures"] else ""))
        print(render_markdown(t[k]))
    return t


if __name__ == "__main__":
    main()

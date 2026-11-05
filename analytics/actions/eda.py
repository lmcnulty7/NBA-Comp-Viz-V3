"""analytics/actions/eda.py: exploratory tables for the action-recognition lane (no plots: tables and markdown).

    python -m analytics.actions.eda --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]

Functions (notebooks/eda_actions.ipynb calls them and prints the markdown):
    candidate_counts_per_possession(cands, n_possessions)   distribution of candidates per possession, by rule class
    class_balance(cands)                                    label_class counts and shares, rule class against label class
    feature_distributions_by_class(cands, feats)            mean, std, median and quartiles of every feature per label class
    shuffled_label_check(cands, feats, seed, n_folds)       out-of-fold ball_screen F1 of the hgb model on the real labels and
                                                            on labels permuted within each game: a model that scores well on
                                                            permuted labels is reading something it should not
    markdown_table(df)                                      a markdown table without the tabulate dependency

The shuffled check fits through train.fit_models, so the same held-out guard applies: the frame it sees has passed
analytics.schema.assert_time_forward (called again here) and the candidates carry no held-out game. On fixtures every
table is fixture output, not a result.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.actions import CLASSES
from analytics.actions import features as F

log = logging.getLogger("analytics.actions.eda")


def markdown_table(df: pd.DataFrame, nd: int = 3) -> str:
    from analytics.actions.train import md_table
    return "\n".join(md_table(df.reset_index() if df.index.name is not None else df, nd))


def candidate_counts_per_possession(cands: pd.DataFrame, n_possessions: int) -> pd.DataFrame:
    """One row per rule class plus a total: possessions with 0, 1, 2 and 3+ candidates, mean per possession."""
    rows = []
    for cls in list(F.DETECT_CLASSES) + ["all"]:
        c = cands if cls == "all" else cands[cands["rule_class"] == cls]
        per = c.groupby(["sportvu_game", "possession"]).size()
        n_with = int(len(per))
        counts = {"0": max(0, n_possessions - n_with), "1": int((per == 1).sum()), "2": int((per == 2).sum()), "3+": int((per >= 3).sum())}
        rows.append({"rule_class": cls, "candidates": int(len(c)), "possessions_with_0": counts["0"], "with_1": counts["1"],
                     "with_2": counts["2"], "with_3_or_more": counts["3+"], "mean_per_possession": len(c) / max(1, n_possessions)})
    return pd.DataFrame(rows)


def class_balance(cands: pd.DataFrame) -> dict:
    """dict(classes=DataFrame(label_class, n, share), rule_vs_label=crosstab)."""
    vc = cands["label_class"].value_counts().reindex(CLASSES, fill_value=0)
    classes = pd.DataFrame({"label_class": vc.index, "n": vc.to_numpy(), "share": vc.to_numpy() / max(1, len(cands))})
    ct = pd.crosstab(cands["rule_class"], cands["label_class"]).reindex(index=F.DETECT_CLASSES, columns=CLASSES, fill_value=0)
    return {"classes": classes, "rule_vs_label": ct.reset_index()}


def feature_distributions_by_class(cands: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Long table: feature, label_class, n, mean, std, q25, median, q75 (NaN-aware)."""
    key = F.ID_COLUMNS
    m = cands[key + ["label_class"]].merge(feats, on=key, how="inner")
    rows = []
    for feat in F.FEATURES:
        for cls, grp in m.groupby("label_class"):
            v = grp[feat].astype(float)
            rows.append({"feature": feat, "label_class": cls, "n": int(v.notna().sum()), "mean": float(v.mean()), "std": float(v.std()),
                         "q25": float(v.quantile(0.25)), "median": float(v.median()), "q75": float(v.quantile(0.75))})
    return pd.DataFrame(rows)


def shuffled_label_check(cands: pd.DataFrame, feats: pd.DataFrame, seed: int = 0, n_folds: int = 5, n_rounds: int = 1) -> pd.DataFrame:
    """Out-of-fold hgb ball_screen F1 and accuracy on the real labels and on labels permuted within each game."""
    from analytics.actions import train as T
    schema.assert_time_forward(feats, "possessions")
    c = cands[~cands["heldout"].astype(bool)].reset_index(drop=True)
    f = feats[~cands["heldout"].astype(bool).to_numpy()].reset_index(drop=True)
    rows = []
    rng = np.random.default_rng(seed)
    variants = [("real", c)]
    for r in range(n_rounds):
        s = c.copy()
        s["label_class"] = s.groupby("sportvu_game")["label_class"].transform(lambda x: x.to_numpy()[rng.permutation(len(x))])
        variants.append(("shuffled_%d" % r, s))
    for name, cc in variants:
        probs, _fold, note = T.out_of_fold(cc, f, n_folds, seed)
        yhat = T.predicted(probs["hgb"])
        y = cc["label_class"].astype(str).to_numpy(dtype=object)
        rows.append({"labels": name, "mode": note["mode"], "f1_ball_screen": T.f1_for(y, yhat, "ball_screen"),
                     "accuracy": float((y == yhat).mean()), "base_rate_ball_screen": float((y == "ball_screen").mean())})
    return pd.DataFrame(rows)


def run(games: list, seed: int = 0, n_folds: int = 5) -> dict:
    """All EDA tables on a list of game dicts (held-out games are dropped first; EDA never looks at them)."""
    from analytics.actions import validate as V
    games = [g for g in games if not (g.get("heldout") or V.is_heldout_id(g["sportvu_game"]))]
    cands, feats, recall, skipped = F.build_season(games)
    n_poss = int(sum(len(g["possessions"]) for g in games))
    out = {"counts": candidate_counts_per_possession(cands, n_poss), "balance": class_balance(cands),
           "distributions": feature_distributions_by_class(cands, feats), "recall": recall,
           "shuffled": shuffled_label_check(cands, feats, seed, n_folds), "n_games": len(games), "n_possessions": n_poss}
    return out


def to_markdown(res: dict, fixtures: bool) -> str:
    tag = " (fixture output, not a result)" if fixtures else ""
    L = ["# EDA: action candidates%s" % tag, "", "%d training game(s), %d possessions." % (res["n_games"], res["n_possessions"]), "",
         "## Candidates per possession", "", markdown_table(res["counts"]), "", "## Class balance", "", markdown_table(res["balance"]["classes"]),
         "", markdown_table(res["balance"]["rule_vs_label"]), "", "## Candidate recall", "", markdown_table(res["recall"]), "",
         "## Feature distributions by label class", "", markdown_table(res["distributions"]), "",
         "## Shuffled-label check", "", markdown_table(res["shuffled"]), ""]
    return "\n".join(L)


def main(argv=None) -> dict:
    from analytics.actions import validate as V
    ap = argparse.ArgumentParser(prog="python -m analytics.actions.eda", description=__doc__.split("\n")[0])
    V.add_input_args(ap)
    ap.add_argument("--out", help="OUT_ROOT: writes analytics/actions/eda.md under it")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run(V.load_inputs(a), seed=a.seed, n_folds=a.folds)
    text = to_markdown(res, bool(a.fixtures))
    if a.out:
        dest = Path(a.out) / "analytics" / "actions" / "eda.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        res["path"] = str(dest)
    print(text)
    return res


if __name__ == "__main__":
    main()

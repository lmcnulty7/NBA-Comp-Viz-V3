"""analytics/decisions/train.py: the value model, the decision events and the validation of decision quality.

    python -m analytics.decisions.train --fixtures | --data DATA_ROOT [--out OUT_ROOT] --report reports/decisions.md
                                        [--folds 5] [--seed 0] [--bootstrap 200] [--season 2015-16] [--skip-heldout]

Protocol
  1. Load and validate the bundles (analytics.decisions.validate). Held-out games (under heldout/ or carrying a
     held-out token of sportvu/splits.json) are set aside before any fit; analytics.eval.refuse_heldout runs on the
     training game ids. On --data each game's frames are a loader and are read one game at a time
     (validate.stream_bundles): steps 2 and the fit rows of step 3 are built in that pass, then the frames dropped.
  2. Decision contexts per game (analytics.decisions.features): shots, passes and drive starts with their
     decision-frame geometry; schema.assert_time_forward on every context frame and on the FEATURES subset;
     assert_decisions_before_outcome on every game.
  3. The value model: the location-only xFG (analytics.features_xfg: distance, angle, is_three, is_corner), a
     logistic regression with standardisation (C=1) fit on the shot_frames rows with a release frame, the shooter's
     position at the release relative to the basket of the possession, target made. The fit frame is the design
     matrix only (assert_time_forward and features_xfg.assert_no_leak). Fit under grouped folds by game: the decisions
     of a game are valued by the model fit on the other folds, so a shot's own outcome never informs its value. Held-out
     games are valued once by the model refit on every training game (never with --skip-heldout: the held-out games
     are then set aside the same way but never valued or written, and the report says "held-out games not scored
     (--skip-heldout)"). Fewer games than folds: n_folds = number of games; one game: in-sample, labelled.
  4. Option values and decision quality (features.option_values). The prior of the option values is reported, not
     fitted: candidate 1 is the location-only xFG, candidate 2 the xFG lane's p_selected for the shot events when its
     table is present (contest adjustment then not applied on those events; passes and drives keep candidate 1).
     The rule: candidate 2 is used when the xFG table covers at least XFG_MIN_COVERAGE of the shot decisions, else
     candidate 1; both are computed and reported when candidate 2 exists. The EPV lane's epv_points is the hold
     option when present.
  5. Validation: per game, the players' summed decision quality against their on-court offensive rating (points per
     100 offensive possessions on the floor, from possessions and frames); Spearman per game over players with at
     least MIN_DECISIONS_GAME decisions, a sign test over games (scipy binomtest, two sided) and a bootstrap by game
     of the mean Spearman. A shuffled check (quality permuted across players within each game) is reported next to it.
  6. The player table: n_decisions, mean quality with a bootstrap interval over the player's games (NaN with one
     game), insufficient when under N_MIN = 300 decisions (analytics.eval.gated). Players are ranked on training
     games only.
Outputs under OUT_ROOT/analytics/decisions/<season>/: decision_events/<game>.parquet (held-out under
heldout/<season>/decision_events/), decision_players.parquet, and the markdown report. With --fixtures and no --out
the parquet files go to a temporary directory (never into the repo) and the report says "fixture output, not a result".
The report lists the written files relative to OUT_ROOT.

Assumptions (also in reports/decisions.md): the contest adjustment, openness and pass-risk forms of features.py; the
drive value; the attacked basket from the offence's mean x; players with no shot in the fit set still get values
(the model is a function of location only); the shuffled check uses the same seed as the folds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analytics import schema
from analytics.decisions import features as F
from analytics.decisions import validate as V
from analytics.eval import (add_skip_heldout_arg, bootstrap_by_game, fit_threads, fmt_num, gated, grouped_folds,
                            refuse_heldout, skip_heldout_lines)
from analytics.features_xfg import assert_no_leak, feature_columns as xfg_feature_columns, features as xfg_features

WHO = "train_decisions"
N_MIN = 300
IN_SAMPLE_FOLD, HELDOUT_FOLD = -2, -1      # fold convention: >= 0 out-of-fold, -1 held-out, -2 in-sample
MIN_DECISIONS_GAME = 3
MIN_PLAYERS_GAME = 3
XFG_MIN_COVERAGE = 0.5
MAC_CMD = ("python -m analytics.decisions.train --data \"$DATA_ROOT\" --out \"$OUT_ROOT\" --season 2015-16 "
           "--report reports/decisions.md --folds 5 --seed 0 --bootstrap 1000")
EVENT_COLUMNS = ["sportvu_game", "possession", "frame", "player_id", "decision_kind", "chosen_value", "best_value",
                 "best_option", "decision_quality", "heldout", "event", "receiver_id", "shot_value", "hold_value",
                 "drive_value", "prior", "fold"]
EVENT_INT32 = ["possession", "frame", "player_id", "event", "receiver_id", "fold"]
PLAYER_COLUMNS = ["player_id", "n_decisions", "mean_quality", "lo", "hi", "insufficient", "n_games", "sum_quality",
                  "mean_quality_shot", "mean_quality_pass", "mean_quality_drive", "n_shot", "n_pass", "n_drive"]
PLAYER_INT32 = ["player_id", "n_decisions", "n_games", "n_shot", "n_pass", "n_drive"]


def _cast_events(df: pd.DataFrame) -> pd.DataFrame:
    """The contract dtypes of decision_events: int32 ids, float64 values, string kinds, bool heldout."""
    out = df.copy()
    for c in EVENT_INT32:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c]).astype("int32")
    for c in ("sportvu_game", "decision_kind", "best_option", "prior"):
        if c in out.columns:
            out[c] = out[c].astype("string")
    if "heldout" in out.columns:
        out["heldout"] = out["heldout"].astype(bool)
    return out


def _cast_players(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PLAYER_INT32:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c]).astype("int32")
    if "insufficient" in out.columns:
        out["insufficient"] = out["insufficient"].astype(bool)
    return out


# ── the value model (location-only xFG) ────────────────────────────────────────────────────────────
def shot_fit_rows(bundle: dict) -> pd.DataFrame:
    """One row per localised shot of the game: the shooter's position at the release, in the section 10
    convention relative to the basket of the possession that holds the release, with the target made."""
    sf, frames, poss = bundle["shot_frames"], V.frames_of(bundle), bundle["possessions"]
    rel = pd.to_numeric(sf["release_frame"]).to_numpy()
    ok = rel != schema.NOT_FOUND
    if not ok.any():
        return pd.DataFrame(columns=["sportvu_game", "event", "made", "period", "clock_s", "loc_x", "loc_y", "heldout"])
    s = sf[ok].copy()
    pid = s["shooter_at_release_id"].where(s["shooter_at_release_id"].notna(), s["shooter_id"])
    s["pid"] = pid
    s = s[s["pid"].notna()]
    s["pid"] = s["pid"].astype(int)
    s["release_frame"] = s["release_frame"].astype(int)
    fr = frames[["frame", "player_id", "x", "y"]]
    m = s.merge(fr, left_on=["release_frame", "pid"], right_on=["frame", "player_id"], how="inner")
    if not len(m):
        return pd.DataFrame(columns=["sportvu_game", "event", "made", "period", "clock_s", "loc_x", "loc_y", "heldout"])
    # the basket nearer the shooter at the release (a shot is taken at the attacked basket)
    sx = m["x"].to_numpy(float)
    bx = np.where(sx > F.HALF_X, F.RIM_RIGHT[0], F.RIM_LEFT[0])
    loc_x, loc_y = F.loc_tenths(sx, m["y"].to_numpy(float), bx, np.full(len(m), F.RIM_LEFT[1]))
    return pd.DataFrame({"sportvu_game": bundle["sportvu_game"], "event": m["event"].to_numpy(), "made": m["made"].astype(bool).to_numpy(),
                         "period": m["period"].to_numpy(), "clock_s": m["clock_s"].to_numpy(float), "loc_x": loc_x,
                         "loc_y": loc_y, "heldout": bool(bundle["heldout"])})


class ValueModel:
    """p(make) at a court position from the four location features (constant when the fit set has one class)."""

    def __init__(self, model=None, rate: float = 0.45, n_fit: int = 0):
        self.model, self.rate, self.n_fit = model, float(rate), int(n_fit)

    def p_loc(self, loc_x, loc_y) -> np.ndarray:
        """p(make) at court positions given as loc_x, loc_y (tenths of a foot from the basket)."""
        loc_x, loc_y = np.asarray(loc_x, float), np.asarray(loc_y, float)
        n = len(loc_x)
        if n == 0:
            return np.zeros(0)
        if self.model is None:
            return np.full(n, self.rate)
        shots = pd.DataFrame({"sportvu_game": "", "made": False, "loc_x": loc_x, "loc_y": loc_y, "period": 1, "clock_s": 0.0})
        X = xfg_features(shots)[xfg_feature_columns()].astype(float)
        return self.model.predict_proba(X.to_numpy())[:, 1]

    def __call__(self, x, y, bx, by) -> np.ndarray:
        lx, ly = F.loc_tenths(x, y, bx, by)
        return self.p_loc(lx, ly)


def fit_value_model(rows: pd.DataFrame) -> ValueModel:
    """Fit on the shot rows (never a held-out row): design = the four location features only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(rows) and rows["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_value_model" % WHO)
    if len(rows) == 0:
        return ValueModel(None, 0.45, 0)
    feats = xfg_features(rows)
    X = feats[xfg_feature_columns()].astype(float)
    schema.assert_time_forward(X, "shot_frames")
    assert_no_leak(X)
    y = rows["made"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return ValueModel(None, float(y.mean()), len(y))
    with fit_threads():
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X.to_numpy(), y)
    return ValueModel(lr, float(y.mean()), len(y))


# ── per-game valuation ─────────────────────────────────────────────────────────────────────────────
def value_game(contexts: pd.DataFrame, bundle: dict, model: ValueModel, fold: int, with_xfg: bool = False,
               counts: Optional[dict] = None) -> dict:
    """{prior: decision events} for the game under candidate 1 and, with with_xfg, candidate 2 (computed for every
    game when any bundle carries an xFG table: rows without a joined p_selected fall back to candidate 1, so no game
    vanishes under partial coverage). Upstream rows are filtered by features.upstream_rows."""
    held = bool(bundle["heldout"])
    hold = F.hold_values(contexts, bundle.get("epv"), heldout=held, counts=counts)
    out = {"location": F.option_values(contexts, model, hold=hold, prior="location")}
    ov = F.xfg_override(contexts, bundle.get("xfg"), heldout=held, counts=counts)
    n_shot = int((contexts["decision_kind"] == "shot").sum())
    if with_xfg:
        out["xfg_selected"] = F.option_values(contexts, model, hold=hold, p_shot_override=ov, prior="xfg_selected")
    for ev in out.values():
        ev["fold"] = fold
    return {"events": out, "n_shot": n_shot, "n_shot_overridden": int(ov.notna().sum()), "n_hold": int(hold.notna().sum())}


# ── validation against on-court offensive rating ───────────────────────────────────────────────────
def player_game_quality(events: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: n_decisions, sum_quality and mean_quality. The validation ranks players by mean quality
    (quality is regret, at most 0, so a sum would mostly rank by decision count)."""
    g = events.groupby(["sportvu_game", "player_id"]).agg(n_decisions=("decision_quality", "size"),
                                                           sum_quality=("decision_quality", "sum"),
                                                           mean_quality=("decision_quality", "mean")).reset_index()
    return g


def per_game_spearman(pg: pd.DataFrame, ortg: pd.DataFrame, min_dec: int = MIN_DECISIONS_GAME,
                      column: str = "mean_quality") -> pd.DataFrame:
    """One row per game: rho between the players' mean decision quality and their on-court offensive rating,
    over players with at least min_dec decisions in the game."""
    from scipy.stats import spearmanr
    m = pg.merge(ortg, on=["sportvu_game", "player_id"], how="inner")
    m = m[m["n_decisions"] >= min_dec]
    rows = []
    for g, d in m.groupby("sportvu_game"):
        if len(d) < MIN_PLAYERS_GAME or d[column].nunique() < 2 or d["ortg"].nunique() < 2:
            rows.append({"sportvu_game": g, "rho": np.nan, "n_players": int(len(d))})
            continue
        rho = spearmanr(d[column], d["ortg"]).correlation
        rows.append({"sportvu_game": g, "rho": float(rho), "n_players": int(len(d))})
    return pd.DataFrame(rows, columns=["sportvu_game", "rho", "n_players"])


def summarise_rhos(per_game: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Mean Spearman over games, the two-sided sign test and the bootstrap interval by game of the mean."""
    from scipy.stats import binomtest
    d = per_game.dropna(subset=["rho"])
    out = {"n_games": int(len(d)), "mean_rho": float(d["rho"].mean()) if len(d) else float("nan"),
           "n_positive": int((d["rho"] > 0).sum()), "n_negative": int((d["rho"] < 0).sum()),
           "sign_p": float("nan"), "lo": float("nan"), "hi": float("nan")}
    k, n = out["n_positive"], out["n_positive"] + out["n_negative"]
    if n:
        out["sign_p"] = float(binomtest(k, n, 0.5).pvalue)
    if len(d) >= 2:
        b = bootstrap_by_game(d, "sportvu_game", lambda s: s["rho"].mean(), n=n_boot, seed=seed)
        out["lo"], out["hi"] = b["lo"], b["hi"]
    return out


def shuffled_quality(pg: pd.DataFrame, seed: int, min_dec: int = MIN_DECISIONS_GAME, column: str = "mean_quality") -> pd.DataFrame:
    """The player-game quality permuted across the players of each game (the null the validation is compared
    with), after the same n >= min_dec filter the observed statistic uses, so both see the same players."""
    rng = np.random.default_rng(seed)
    out = pg[pg["n_decisions"] >= min_dec].copy()
    for g, idx in out.groupby("sportvu_game").indices.items():
        vals = out[column].to_numpy()[idx]
        out.loc[out.index[idx], column] = vals[rng.permutation(len(idx))]
    return out


def validate_quality(events: pd.DataFrame, ortg: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Per-game Spearman of mean quality against on-court offensive rating, summarised, plus the within-game shuffled check."""
    pg = player_game_quality(events)
    per = per_game_spearman(pg, ortg)
    shuf = per_game_spearman(shuffled_quality(pg, seed), ortg)
    return {"per_game": per, "summary": summarise_rhos(per, n_boot, seed), "shuffled": summarise_rhos(shuf, n_boot, seed)}


# ── player table ───────────────────────────────────────────────────────────────────────────────────
def player_table(events: pd.DataFrame, n_boot: int, seed: int, n_min: int = N_MIN) -> pd.DataFrame:
    """Per player: decision counts and mean quality (overall and by kind) with a bootstrap interval by game, gated at n_min and ranked."""
    if not len(events):
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    g = gated(events[["player_id", "sportvu_game", "decision_kind", "decision_quality"]], "player_id", n_min)
    rows = []
    for pid, d in g.groupby("player_id"):
        r = {"player_id": int(pid), "n_decisions": int(len(d)), "mean_quality": float(d["decision_quality"].mean()),
             "sum_quality": float(d["decision_quality"].sum()), "n_games": int(d["sportvu_game"].nunique()),
             "insufficient": bool(d["insufficient"].iloc[0]), "lo": float("nan"), "hi": float("nan")}
        for kind in F.KINDS:                                    # mean quality by kind (all are regret, at most 0)
            dk = d.loc[d["decision_kind"] == kind, "decision_quality"]
            r["mean_quality_" + kind] = float(dk.mean()) if len(dk) else float("nan")
            r["n_" + kind] = int(len(dk))
        if r["n_games"] >= 2:
            b = bootstrap_by_game(d, "sportvu_game", lambda s: s["decision_quality"].mean(), n=n_boot, seed=seed)
            r["lo"], r["hi"] = b["lo"], b["hi"]
        rows.append(r)
    out = pd.DataFrame(rows, columns=PLAYER_COLUMNS)
    # ranked on mean quality among the players past the gate; insufficient rows carry no rank and sort last by id
    out = out.sort_values(["insufficient", "mean_quality", "player_id"], ascending=[True, False, True]).reset_index(drop=True)
    enough = ~out["insufficient"].to_numpy(dtype=bool)
    rank = np.full(len(out), np.nan)
    rank[enough] = np.arange(1, int(enough.sum()) + 1)
    out["rank"] = rank
    ins = out[out["insufficient"]].sort_values("player_id")
    return pd.concat([out[~out["insufficient"]], ins], ignore_index=True)


# ── the run ────────────────────────────────────────────────────────────────────────────────────────
def run(bundles: list, n_folds: int = 5, seed: int = 0, n_boot: int = 200, skip_heldout: bool = False) -> dict:
    """Everything train.py computes, as a dict (the report and the parquet files are written by main). With
    skip_heldout the held-out games are set aside as always but never valued, so events_heldout is empty.
    The frames are read in one pass, one game at a time (validate.stream_bundles): each game is validated, its
    contexts, on-court rating and shot fit rows are built, then its frames are dropped."""
    train_ids = sorted(b["sportvu_game"] for b in bundles if not b["heldout"])
    refuse_heldout(train_ids, WHO)
    if not train_ids:
        raise SystemExit("%s: no training game after setting the held-out games aside" % WHO)
    contexts, counts, ortg, fit_rows = {}, {}, [], {}

    def per_game(b):
        """Everything train needs from one game's frames, built while they are in memory."""
        c = {}
        ctx = F.decision_contexts(b, c)
        F.assert_decisions_before_outcome(ctx, b["shot_frames"], b["possessions"])
        F.feature_frame(ctx)
        contexts[b["sportvu_game"]] = ctx
        counts[b["sportvu_game"]] = c
        ortg.append(F.on_court_offensive_rating(V.frames_of(b), b["possessions"], b["sportvu_game"]))
        if not b["heldout"]:
            fit_rows[b["sportvu_game"]] = shot_fit_rows(b)

    bundles, val_reports, _ = V.stream_bundles(bundles, per_game)
    train = [b for b in bundles if not b["heldout"]]
    held = [b for b in bundles if b["heldout"]]
    ortg = pd.concat(ortg, ignore_index=True) if ortg else pd.DataFrame(columns=["sportvu_game", "player_id", "n_possessions", "points", "ortg"])
    # folds by game
    k = min(n_folds, len(train_ids))
    folds = grouped_folds(train_ids, n_folds=k, seed=seed) if len(train_ids) > 1 else [(train_ids, train_ids)]
    mode = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if len(train_ids) == 1:
        mode = "IN-SAMPLE: one training game, the value model is fit and applied on the same game"
    elif k < n_folds:
        mode += "; fewer games (%d) than the %d folds asked, so leave-one-game-out" % (len(train_ids), n_folds)
    by_game = {b["sportvu_game"]: b for b in bundles}
    has_xfg = any(b.get("xfg") is not None for b in bundles)
    events = {"location": [], "xfg_selected": []}
    diag = {"n_shot": 0, "n_shot_overridden": 0, "n_hold": 0, "fit_rows": 0}
    in_sample = len(train_ids) == 1
    for fi, (tr, te) in enumerate(folds):
        rows = pd.concat([fit_rows[g] for g in tr], ignore_index=True) if tr else fit_rows[te[0]].iloc[:0]
        model = fit_value_model(rows)
        for g in te:
            # fold convention: >= 0 out-of-fold, -1 held-out (valued by the full refit), -2 in-sample (one training game)
            r = value_game(contexts[g], by_game[g], model, IN_SAMPLE_FOLD if in_sample else fi, with_xfg=has_xfg, counts=counts[g])
            for name, ev in r["events"].items():
                events[name].append(ev)
            diag["n_shot"] += r["n_shot"]
            diag["n_shot_overridden"] += r["n_shot_overridden"]
            diag["n_hold"] += r["n_hold"]
    full_rows = pd.concat(list(fit_rows.values()), ignore_index=True)
    diag["fit_rows"] = int(len(full_rows))
    full_model = fit_value_model(full_rows)
    held_events = {"location": [], "xfg_selected": []}
    for b in ([] if skip_heldout else held):
        r = value_game(contexts[b["sportvu_game"]], b, full_model, -1, with_xfg=has_xfg, counts=counts[b["sportvu_game"]])
        for name, ev in r["events"].items():
            held_events[name].append(ev)
    cat = lambda L: _cast_events(pd.concat(L, ignore_index=True)[EVENT_COLUMNS]) if L else pd.DataFrame(columns=EVENT_COLUMNS)
    ev_train = {n: cat(L) for n, L in events.items()}
    ev_held = {n: cat(L) for n, L in held_events.items()}
    # pooled coverage: overridden shot decisions over all shot decisions of the training games
    cover = float(diag["n_shot_overridden"] / diag["n_shot"]) if diag["n_shot"] else 0.0
    prior = "xfg_selected" if (has_xfg and cover >= XFG_MIN_COVERAGE) else "location"
    validation = {n: validate_quality(ev_train[n], ortg, n_boot, seed) for n in ("location", "xfg_selected") if len(ev_train[n])}
    players = player_table(ev_train[prior], n_boot, seed)
    return {"events": ev_train[prior], "events_heldout": ev_held[prior],
            "events_by_prior": ev_train, "players": players, "validation": validation, "prior": prior,
            "xfg_coverage": cover, "has_xfg": has_xfg, "n_hold": diag["n_hold"], "fit_rows": diag["fit_rows"],
            "ortg": ortg, "counts": counts, "train_ids": train_ids, "heldout_ids": sorted(b["sportvu_game"] for b in held),
            "mode": mode, "folds": k, "val_reports": val_reports, "full_model": full_model, "contexts": contexts,
            "seed": seed, "n_boot": n_boot, "skip_heldout": bool(skip_heldout)}


# ── outputs ────────────────────────────────────────────────────────────────────────────────────────
def write_outputs(res: dict, out_root, season: str) -> list:
    """Write decision_events per game (held-out games under heldout/) and decision_players; returns the paths."""
    base = Path(out_root) / "analytics" / "decisions"
    written = []
    for ev, held in ((res["events"], False), (res["events_heldout"], True)):
        if not len(ev):
            continue
        d = base / ("heldout/" + season if held else season) / "decision_events"
        d.mkdir(parents=True, exist_ok=True)
        for g, part in ev.groupby("sportvu_game"):
            p = d / ("%s.parquet" % g)
            part.reset_index(drop=True).to_parquet(p, index=False)
            written.append(p)
    d = base / season
    d.mkdir(parents=True, exist_ok=True)
    p = d / "decision_players.parquet"
    _cast_players(res["players"]).to_parquet(p, index=False)
    written.append(p)
    return written


def _val_lines(name: str, v: dict) -> list:
    """One validation table row: games, mean rho with its interval, the sign test and the shuffled mean."""
    s, sh = v["summary"], v["shuffled"]
    L = ["| %s | %d | %s | %s | %s | %d / %d | %s | %s |" % (name, s["n_games"], fmt_num(s["mean_rho"]), fmt_num(s["lo"]), fmt_num(s["hi"]),
                                                            s["n_positive"], s["n_positive"] + s["n_negative"], fmt_num(s["sign_p"]),
                                                            fmt_num(sh["mean_rho"]))]
    return L


def write_report(path: Path, res: dict, ctx: dict) -> str:
    """Write the markdown report to path and return its text; with ctx["fixtures"] every section is labelled fixture output."""
    fixture = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fixture else ""
    ev, held = res["events"], res["events_heldout"]
    L = ["# Decision quality (analytics/decisions)%s" % tag, "",
         "Written %s by `%s`." % (ctx["written_at"], ctx["command"]), ""]
    if fixture:
        L += ["**Fixture output, not a result.** Every number below comes from synthetic games "
              "(analytics/fixtures.py); the real run is the Mac command at the end.", ""]
    L += ["## Inputs" + tag, "", "| game | held out | frames | possessions | shots | schema problems | duplicates dropped | epv rows | xfg rows |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["val_reports"]:
        L.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d |" % (
            r["sportvu_game"], "yes" if r["heldout"] else "no", r["rows"].get("frames", 0), r["rows"].get("possessions", 0),
            r["rows"].get("shot_frames", 0), sum(len(v) for v in r["schema"].values()), sum(r["duplicates"].values()),
            r["rows"].get("epv", 0), r["rows"].get("xfg", 0)))
    L += ["", "Training games: %d (%s). Held-out games set aside before any fit: %d (%s). Guard: `analytics.eval.refuse_heldout` "
          "checked the training ids against sportvu/splits.json." % (len(res["train_ids"]), ", ".join(res["train_ids"]) or "none",
                                                                       len(res["heldout_ids"]), ", ".join(res["heldout_ids"]) or "none"),
          "Value model fit rows (localised shots of the training games): %d. Protocol: %s." % (res["fit_rows"], res["mode"]), "",
          "## Event counts (training games)" + tag, "", "| game | shots | passes | drives | possessions used | skipped |",
          "| --- | --- | --- | --- | --- | --- |"]
    for g in res["train_ids"]:
        d = ev[ev["sportvu_game"] == g]
        c = res["counts"].get(g, {})
        skipped = {k: v for k, v in c.items() if k != "possessions_used"}
        L.append("| %s | %d | %d | %d | %d | %s |" % (g, int((d["decision_kind"] == "shot").sum()), int((d["decision_kind"] == "pass").sum()),
                                                     int((d["decision_kind"] == "drive").sum()), c.get("possessions_used", 0), skipped or "none"))
    L += ["", "Total: %d decisions (%s)." % (len(ev), ev["decision_kind"].value_counts().to_dict() if len(ev) else {}),
          "Hold option (EPV lane) present on %d decisions; xFG lane coverage of shot decisions %s." % (res["n_hold"], fmt_num(res["xfg_coverage"], 2)), "",
          "## Value-function assumptions" + tag, "",
          "- Shot option: points (3 at 22 ft or more, else 2) times the location-only xFG probability at the player's position, "
          "minus %.2f per foot of nearest-defender distance below %.0f ft when no contest model exists (floor %.2f)." % (F.CONTEST_PER_FT, F.CONTEST_FT, F.P_FLOOR),
          "- Pass option: points times the location-only probability at the teammate's position (no contest term: openness already carries the "
          "defender distance) times openness (1 at %.0f ft or more of nearest-defender distance, linearly down to %.1f at 0 ft) "
          "times pass risk (%.2f minus %.2f per foot of pass length over %.0f ft)." % (F.OPEN_FT, F.OPEN_MIN, F.PASS_RISK_BASE, F.PASS_RISK_PER_FT, F.PASS_RISK_FT),
          "- Hold option: the EPV lane's epv_points of the last state at or before the decision frame; absent when the EPV table is absent.",
          "- Drive option: the shot value %.0f ft nearer the rim along the line to the basket (never nearer than %.0f ft), with the decision-frame "
          "defender distance, times %.2f; it is in every decision's option set and is the chosen value of a drive, so every quality is regret "
          "(at most 0) and best_option is one of shot, pass_to_<rank>, hold, drive." % (F.DRIVE_FT, F.DRIVE_MIN_FT, F.DRIVE_RISK),
          "- Attacked basket: per (offence team, period) by majority vote over that team's possessions, each voting with the end where its "
          "offence's mean x lies; the value-model fit uses the basket nearer the shooter at the release.",
          "- Event detection: holder within %.1f ft of the ball under %.0f ft; pass arrives within %.1f s with no release in between; "
          "drive = the first frame at which the holder's speed toward the basket is measured above %.0f ft/s in a run lasting %.1f s." % (
              F.HOLD_FT, F.HOLD_Z_FT, F.PASS_MAX_S, F.DRIVE_FTPS, F.DRIVE_S),
          "- Upstream rows: a training game refuses EPV or xFG rows with fold == -1, a held-out game refuses rows not flagged heldout "
          "(refused counts: %s)." % ({g: {k: v for k, v in c.items() if k.endswith("_refused")} for g, c in res["counts"].items() if any(k.endswith("_refused") for k in c)} or "none"),
          "- Leakage: option values use the decision frame and earlier frames only; the value model is out-of-fold by game; "
          "`schema.assert_time_forward` ran on every context frame; `assert_decisions_before_outcome` on every game.",
          "- Prior of the option values: candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. "
          "Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least %.2f. Selected: **%s**%s." % (
              XFG_MIN_COVERAGE, res["prior"], "" if res["has_xfg"] else " (no xFG table present)"), "",
          "## Player table (training games, gated at %d decisions; ranked on mean quality, by-kind means are regret at most 0)%s" % (N_MIN, tag), ""]
    players = res["players"]
    ranked = players[~players["insufficient"]] if len(players) else players
    n_ins = int(players["insufficient"].sum()) if len(players) else 0
    head = ["| rank | player_id | n_decisions | n_games | mean quality | lo | hi | shot | pass | drive |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    if len(ranked):
        L += head
        for r in ranked.head(25).itertuples(index=False):
            L.append("| %d | %d | %d | %d | %s | %s | %s | %s | %s | %s |" % (r.rank, r.player_id, r.n_decisions, r.n_games, fmt_num(r.mean_quality),
                                                                         fmt_num(r.lo), fmt_num(r.hi), fmt_num(r.mean_quality_shot), fmt_num(r.mean_quality_pass), fmt_num(r.mean_quality_drive)))
    else:
        L.append("No player passes the %d-decision gate: nobody is ranked." % N_MIN)
    L += ["", "%d of %d players are under the %d-decision gate and carry no rank (listed unranked, by id, first 10):" % (n_ins, len(players), N_MIN), "",
          "| player_id | n_decisions | n_games | mean quality | shot | pass | drive |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in players[players["insufficient"]].head(10).itertuples(index=False):
        L.append("| %d | %d | %d | %s | %s | %s | %s |" % (r.player_id, r.n_decisions, r.n_games, fmt_num(r.mean_quality), fmt_num(r.mean_quality_shot),
                                                       fmt_num(r.mean_quality_pass), fmt_num(r.mean_quality_drive)))
    L += ["", "## Validation: mean decision quality per player-game against on-court offensive rating" + tag, "",
          "Spearman per game between the players' mean decision quality (a sum would rank by decision count, since quality is regret) and "
          "their points per 100 offensive possessions on the floor, over players with at least %d decisions in the game; sign test over "
          "games (two sided); bootstrap of the mean Spearman by game (n=%d, seed %d); shuffled = the same statistic with mean quality permuted "
          "across those players within each game." % (MIN_DECISIONS_GAME, res["n_boot"], res["seed"]), "",
          "| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, v in res["validation"].items():
        L += _val_lines(n, v)
    if res.get("skip_heldout"):
        L += ["", "## Held-out games" + tag, ""] + skip_heldout_lines(len(res["heldout_ids"]))
    elif len(held):
        L += ["", "## Held-out games (valued once by the model refit on all training games; never fitted or tuned on)" + tag, ""]
        L += ["%d decisions from %d held-out game(s): %s." % (len(held), held["sportvu_game"].nunique(), ", ".join(sorted(held["sportvu_game"].unique()))),
              "", "| kind | n | mean quality |", "| --- | --- | --- |"]
        for kd, d in held.groupby("decision_kind"):
            L.append("| %s | %d | %s |" % (kd, len(d), fmt_num(d["decision_quality"].mean())))
        hv = validate_quality(held, res["ortg"], res["n_boot"], res["seed"])
        L += ["", "| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        L += _val_lines(res["prior"] + " (held out)", hv)
    else:
        L += ["", "## Held-out games (valued once by the model refit on all training games; never fitted or tuned on)" + tag, ""]
        L.append("No held-out game in this run.")
    L += ["", "## Outputs (relative to OUT_ROOT)" + tag, ""]
    L += ["- %s" % p for p in ctx["written_paths"]] or ["- none written"]
    L += ["", "## Status" + tag, ""]
    if fixture:
        L.append("Fixture output, not a result. The real run, on the Mac where the full data lives:")
    else:
        L.append("The run of this protocol on the Mac is")
    L += ["", "    " + MAC_CMD, "",
          "What needs real data: the Phase 0 tables of every game (possessions, shot_frames, states_1hz under DATA_ROOT/analytics), "
          "the frames and play-by-play under DATA_ROOT/sportvu_contract, and, optionally, the EPV and xFG lanes' outputs under OUT_ROOT.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The train CLI's arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true", help="train on synthetic games from analytics.fixtures (fixture output, not a result)")
    ap.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT for the parquet outputs and the upstream lanes (default DATA_ROOT; a temp dir with --fixtures)")
    ap.add_argument("--season", default=V.DEFAULT_SEASON, help="season directory under DATA_ROOT/analytics (default %(default)s)")
    ap.add_argument("--report", default="reports/decisions.md", help="markdown report path (default %(default)s)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by game for the value model (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the folds, the bootstrap and the fixtures (default %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples by game (default %(default)s)")
    V.add_fixture_args(ap)
    add_skip_heldout_arg(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Run the protocol, write the parquet outputs and the report; returns the run dict."""
    a = parse_args(argv)
    if not a.fixtures and not a.data:
        raise SystemExit("%s: pass --fixtures or --data DATA_ROOT" % WHO)
    if a.fixtures:
        bundles = V.load_fixture_bundles(a.seed, a.n_games, a.heldout_every, a.n_possessions)
        out_root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="decisions_fixture_"))
        season = "fixture" if a.season == V.DEFAULT_SEASON else a.season
    else:
        bundles = V.load_root_bundles(a.data, a.season, a.out)
        out_root = Path(a.out) if a.out else Path(a.data)
        season = a.season
    res = run(bundles, a.folds, a.seed, a.bootstrap, skip_heldout=a.skip_heldout)
    written = write_outputs(res, out_root, season)
    ctx = {"written_paths": [p.relative_to(out_root).as_posix() for p in written], "fixtures": a.fixtures,
           "written_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python -m analytics.decisions.train " + " ".join(argv if argv is not None else sys.argv[1:])}
    text = write_report(Path(a.report), res, ctx)
    print(text)
    res["written"] = written
    return res


if __name__ == "__main__":
    main()

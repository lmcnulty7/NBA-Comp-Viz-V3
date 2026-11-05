"""analytics/decisions/train.py: the value model, the decision events and the validation of decision quality.

    python -m analytics.decisions.train --fixtures | --data DATA_ROOT [--out OUT_ROOT] --report reports/decisions.md
                                        [--folds 5] [--seed 0] [--bootstrap 200] [--season 2015-16]

Protocol
  1. Load and validate the bundles (analytics.decisions.validate). Held-out games (under heldout/ or carrying a
     held-out token of sportvu/splits.json) are set aside before any fit; analytics.eval.refuse_heldout runs on the
     training game ids.
  2. Decision contexts per game (analytics.decisions.features): shots, passes and drive starts with their
     decision-frame geometry; schema.assert_time_forward on every context frame and on the FEATURES subset;
     assert_decisions_before_outcome on every game.
  3. The value model: the location-only xFG (analytics.features_xfg: distance, angle, is_three, is_corner), a
     logistic regression with standardisation (C=1) fit on the shot_frames rows with a release frame, the shooter's
     position at the release relative to the basket of the possession, target made. The fit frame is the design
     matrix only (assert_time_forward and features_xfg.assert_no_leak). Fit under grouped folds by game: the decisions
     of a game are valued by the model fit on the other folds, so a shot's own outcome never informs its value. Held-out
     games are valued once by the model refit on every training game. Fewer games than folds: n_folds = number of
     games; one game: in-sample, labelled.
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

import numpy as np
import pandas as pd

from analytics import schema
from analytics.decisions import features as F
from analytics.decisions import validate as V
from analytics.eval import bootstrap_by_game, gated, grouped_folds, refuse_heldout
from analytics.features_xfg import assert_no_leak, feature_columns as xfg_feature_columns, features as xfg_features

WHO = "train_decisions"
N_MIN = 300
MIN_DECISIONS_GAME = 3
MIN_PLAYERS_GAME = 3
XFG_MIN_COVERAGE = 0.5
MAC_CMD = ("python -m analytics.decisions.train --data \"$DATA_ROOT\" --out \"$OUT_ROOT\" --season 2015-16 "
           "--report reports/decisions.md --folds 5 --seed 0 --bootstrap 1000")
EVENT_COLUMNS = ["sportvu_game", "possession", "frame", "player_id", "decision_kind", "chosen_value", "best_value",
                 "best_option", "decision_quality", "heldout", "event", "receiver_id", "shot_value", "hold_value",
                 "prior", "fold"]
PLAYER_COLUMNS = ["player_id", "n_decisions", "mean_quality", "lo", "hi", "insufficient", "n_games", "sum_quality"]


# ── the value model (location-only xFG) ────────────────────────────────────────────────────────────
def shot_fit_rows(bundle: dict) -> pd.DataFrame:
    """One row per localised shot of the game: the shooter's position at the release, in the section 10
    convention relative to the basket of the possession that holds the release, with the target made."""
    sf, frames, poss = bundle["shot_frames"], bundle["frames"], bundle["possessions"]
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
    # basket of the possession holding the release (mean offence x), else the nearer basket
    p = poss.dropna(subset=["offense_team_id"])
    bx = np.full(len(m), np.nan)
    fsorted = frames if frames["frame"].is_monotonic_increasing else frames.sort_values("frame", kind="stable")
    fv = fsorted["frame"].to_numpy()
    for i, r in enumerate(m.itertuples(index=False)):
        hit = p[(p["start_frame"] <= r.release_frame) & (p["end_frame"] >= r.release_frame)]
        if len(hit):
            row = hit.iloc[0]
            lo = int(np.searchsorted(fv, int(row.start_frame), side="left"))
            hi = int(np.searchsorted(fv, int(row.end_frame), side="right"))
            sub = fsorted.iloc[lo:hi]
            offx = sub.loc[sub["team_id"] == row.offense_team_id, "x"]
            bx[i] = F.basket_of(float(offx.mean()))[0] if len(offx) else np.nan
        if np.isnan(bx[i]):
            bx[i] = F.RIM_RIGHT[0] if r.x > F.HALF_X else F.RIM_LEFT[0]
    loc_x, loc_y = F.loc_tenths(m["x"].to_numpy(float), m["y"].to_numpy(float), bx, np.full(len(m), F.RIM_LEFT[1]))
    return pd.DataFrame({"sportvu_game": bundle["sportvu_game"], "event": m["event"].to_numpy(), "made": m["made"].astype(bool).to_numpy(),
                         "period": m["period"].to_numpy(), "clock_s": m["clock_s"].to_numpy(float), "loc_x": loc_x,
                         "loc_y": loc_y, "heldout": bool(bundle["heldout"])})


class ValueModel:
    """p(make) at a court position from the four location features (constant when the fit set has one class)."""

    def __init__(self, model=None, rate: float = 0.45, n_fit: int = 0):
        self.model, self.rate, self.n_fit = model, float(rate), int(n_fit)

    def p_loc(self, loc_x, loc_y) -> np.ndarray:
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
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X.to_numpy(), y)
    return ValueModel(lr, float(y.mean()), len(y))


# ── per-game valuation ─────────────────────────────────────────────────────────────────────────────
def value_game(contexts: pd.DataFrame, bundle: dict, model: ValueModel, fold: int) -> dict:
    """{prior: decision events} for the game under candidate 1 and, when the xFG table joins, candidate 2."""
    hold = F.hold_values(contexts, bundle.get("epv"))
    out = {"location": F.option_values(contexts, model, hold=hold, prior="location")}
    ov = F.xfg_override(contexts, bundle.get("xfg"))
    n_shot = int((contexts["decision_kind"] == "shot").sum())
    cover = float(ov.notna().sum() / n_shot) if n_shot else 0.0
    if ov.notna().any():
        out["xfg_selected"] = F.option_values(contexts, model, hold=hold, p_shot_override=ov, prior="xfg_selected")
    for ev in out.values():
        ev["fold"] = fold
    return {"events": out, "xfg_coverage": cover, "n_hold": int(hold.notna().sum())}


# ── validation against on-court offensive rating ───────────────────────────────────────────────────
def player_game_quality(events: pd.DataFrame) -> pd.DataFrame:
    g = events.groupby(["sportvu_game", "player_id"]).agg(n_decisions=("decision_quality", "size"),
                                                           sum_quality=("decision_quality", "sum")).reset_index()
    return g


def per_game_spearman(pg: pd.DataFrame, ortg: pd.DataFrame, min_dec: int = MIN_DECISIONS_GAME) -> pd.DataFrame:
    """One row per game: rho between the players' summed quality and their on-court offensive rating."""
    from scipy.stats import spearmanr
    m = pg.merge(ortg, on=["sportvu_game", "player_id"], how="inner")
    m = m[m["n_decisions"] >= min_dec]
    rows = []
    for g, d in m.groupby("sportvu_game"):
        if len(d) < MIN_PLAYERS_GAME or d["sum_quality"].nunique() < 2 or d["ortg"].nunique() < 2:
            rows.append({"sportvu_game": g, "rho": np.nan, "n_players": int(len(d))})
            continue
        rho = spearmanr(d["sum_quality"], d["ortg"]).correlation
        rows.append({"sportvu_game": g, "rho": float(rho), "n_players": int(len(d))})
    return pd.DataFrame(rows, columns=["sportvu_game", "rho", "n_players"])


def summarise_rhos(per_game: pd.DataFrame, n_boot: int, seed: int) -> dict:
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


def shuffled_quality(pg: pd.DataFrame, seed: int) -> pd.DataFrame:
    """sum_quality permuted across the players of each game (the null the validation is compared with)."""
    rng = np.random.default_rng(seed)
    out = pg.copy()
    for g, idx in out.groupby("sportvu_game").indices.items():
        vals = out["sum_quality"].to_numpy()[idx]
        out.loc[out.index[idx], "sum_quality"] = vals[rng.permutation(len(idx))]
    return out


def validate_quality(events: pd.DataFrame, ortg: pd.DataFrame, n_boot: int, seed: int) -> dict:
    pg = player_game_quality(events)
    per = per_game_spearman(pg, ortg)
    shuf = per_game_spearman(shuffled_quality(pg, seed), ortg)
    return {"per_game": per, "summary": summarise_rhos(per, n_boot, seed), "shuffled": summarise_rhos(shuf, n_boot, seed)}


# ── player table ───────────────────────────────────────────────────────────────────────────────────
def player_table(events: pd.DataFrame, n_boot: int, seed: int, n_min: int = N_MIN) -> pd.DataFrame:
    if not len(events):
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    g = gated(events[["player_id", "sportvu_game", "decision_quality"]], "player_id", n_min)
    rows = []
    for pid, d in g.groupby("player_id"):
        r = {"player_id": int(pid), "n_decisions": int(len(d)), "mean_quality": float(d["decision_quality"].mean()),
             "sum_quality": float(d["decision_quality"].sum()), "n_games": int(d["sportvu_game"].nunique()),
             "insufficient": bool(d["insufficient"].iloc[0]), "lo": float("nan"), "hi": float("nan")}
        if r["n_games"] >= 2:
            b = bootstrap_by_game(d, "sportvu_game", lambda s: s["decision_quality"].mean(), n=n_boot, seed=seed)
            r["lo"], r["hi"] = b["lo"], b["hi"]
        rows.append(r)
    out = pd.DataFrame(rows, columns=PLAYER_COLUMNS)
    return out.sort_values(["insufficient", "mean_quality"], ascending=[True, False]).reset_index(drop=True)


# ── the run ────────────────────────────────────────────────────────────────────────────────────────
def run(bundles: list, n_folds: int = 5, seed: int = 0, n_boot: int = 200) -> dict:
    """Everything train.py computes, as a dict (the report and the parquet files are written by main)."""
    bundles, val_reports = V.validate_bundles(bundles)
    train = [b for b in bundles if not b["heldout"]]
    held = [b for b in bundles if b["heldout"]]
    train_ids = sorted(b["sportvu_game"] for b in train)
    refuse_heldout(train_ids, WHO)
    if not train:
        raise SystemExit("%s: no training game after setting the held-out games aside" % WHO)
    contexts, counts, ortg = {}, {}, []
    for b in bundles:
        c = {}
        ctx = F.decision_contexts(b, c)
        F.assert_decisions_before_outcome(ctx, b["shot_frames"], b["possessions"])
        F.feature_frame(ctx)
        contexts[b["sportvu_game"]] = ctx
        counts[b["sportvu_game"]] = c
        ortg.append(F.on_court_offensive_rating(b["frames"], b["possessions"], b["sportvu_game"]))
    ortg = pd.concat(ortg, ignore_index=True) if ortg else pd.DataFrame(columns=["sportvu_game", "player_id", "n_possessions", "points", "ortg"])
    fit_rows = {b["sportvu_game"]: shot_fit_rows(b) for b in train}
    # folds by game
    k = min(n_folds, len(train_ids))
    folds = grouped_folds(train_ids, n_folds=k, seed=seed) if len(train_ids) > 1 else [(train_ids, train_ids)]
    mode = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if len(train_ids) == 1:
        mode = "IN-SAMPLE: one training game, the value model is fit and applied on the same game"
    elif k < n_folds:
        mode += "; fewer games (%d) than the %d folds asked, so leave-one-game-out" % (len(train_ids), n_folds)
    by_game = {b["sportvu_game"]: b for b in bundles}
    events = {"location": [], "xfg_selected": []}
    diag = {"xfg_coverage": [], "n_hold": 0, "fit_rows": 0}
    for fi, (tr, te) in enumerate(folds):
        rows = pd.concat([fit_rows[g] for g in tr], ignore_index=True) if tr else fit_rows[te[0]].iloc[:0]
        model = fit_value_model(rows)
        for g in te:
            r = value_game(contexts[g], by_game[g], model, fi)
            for name, ev in r["events"].items():
                events[name].append(ev)
            diag["xfg_coverage"].append(r["xfg_coverage"])
            diag["n_hold"] += r["n_hold"]
    full_rows = pd.concat(list(fit_rows.values()), ignore_index=True)
    diag["fit_rows"] = int(len(full_rows))
    full_model = fit_value_model(full_rows)
    held_events = {"location": [], "xfg_selected": []}
    for b in held:
        r = value_game(contexts[b["sportvu_game"]], b, full_model, -1)
        for name, ev in r["events"].items():
            held_events[name].append(ev)
    cat = lambda L: pd.concat(L, ignore_index=True)[EVENT_COLUMNS] if L else pd.DataFrame(columns=EVENT_COLUMNS)
    ev_train = {n: cat(L) for n, L in events.items()}
    ev_held = {n: cat(L) for n, L in held_events.items()}
    cover = float(np.mean(diag["xfg_coverage"])) if diag["xfg_coverage"] else 0.0
    has_xfg = len(ev_train["xfg_selected"]) > 0
    prior = "xfg_selected" if (has_xfg and cover >= XFG_MIN_COVERAGE) else "location"
    validation = {n: validate_quality(ev_train[n], ortg, n_boot, seed) for n in ("location", "xfg_selected") if len(ev_train[n])}
    players = player_table(ev_train[prior], n_boot, seed)
    return {"events": ev_train[prior], "events_heldout": ev_held[prior] if len(ev_held[prior]) else ev_held["location"],
            "events_by_prior": ev_train, "players": players, "validation": validation, "prior": prior,
            "xfg_coverage": cover, "has_xfg": has_xfg, "n_hold": diag["n_hold"], "fit_rows": diag["fit_rows"],
            "ortg": ortg, "counts": counts, "train_ids": train_ids, "heldout_ids": sorted(b["sportvu_game"] for b in held),
            "mode": mode, "folds": k, "val_reports": val_reports, "full_model": full_model, "contexts": contexts,
            "seed": seed, "n_boot": n_boot}


# ── outputs ────────────────────────────────────────────────────────────────────────────────────────
def write_outputs(res: dict, out_root, season: str) -> list:
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
    res["players"].to_parquet(p, index=False)
    written.append(p)
    return written


def _f(v, nd=4) -> str:
    try:
        return "n/a" if v is None or not np.isfinite(float(v)) else ("%%.%df" % nd) % float(v)
    except (TypeError, ValueError):
        return str(v)


def _val_lines(name: str, v: dict) -> list:
    s, sh = v["summary"], v["shuffled"]
    L = ["| %s | %d | %s | %s | %s | %d / %d | %s | %s |" % (name, s["n_games"], _f(s["mean_rho"]), _f(s["lo"]), _f(s["hi"]),
                                                            s["n_positive"], s["n_positive"] + s["n_negative"], _f(s["sign_p"]),
                                                            _f(sh["mean_rho"]))]
    return L


def write_report(path: Path, res: dict, ctx: dict) -> str:
    fixture = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fixture else ""
    ev, held = res["events"], res["events_heldout"]
    L = ["# Decision quality (analytics/decisions)%s" % tag, "",
         "Written %s by `%s`." % (ctx["written_at"], ctx["command"]), ""]
    if fixture:
        L += ["**Fixture output, not a result.** Every number below comes from synthetic games "
              "(tests/_fx_decisions.py or analytics/fixtures.py); the real run is the Mac command at the end.", ""]
    L += ["## Inputs", "", "| game | held out | frames | possessions | shots | schema problems | duplicates dropped | epv rows | xfg rows |",
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
          "## Event counts (training games)", "", "| game | shots | passes | drives | possessions used | skipped |",
          "| --- | --- | --- | --- | --- | --- |"]
    for g in res["train_ids"]:
        d = ev[ev["sportvu_game"] == g]
        c = res["counts"].get(g, {})
        skipped = {k: v for k, v in c.items() if k != "possessions_used"}
        L.append("| %s | %d | %d | %d | %d | %s |" % (g, int((d["decision_kind"] == "shot").sum()), int((d["decision_kind"] == "pass").sum()),
                                                     int((d["decision_kind"] == "drive").sum()), c.get("possessions_used", 0), skipped or "none"))
    L += ["", "Total: %d decisions (%s)." % (len(ev), ev["decision_kind"].value_counts().to_dict() if len(ev) else {}),
          "Hold option (EPV lane) present on %d decisions; xFG lane coverage of shot decisions %s." % (res["n_hold"], _f(res["xfg_coverage"], 2)), "",
          "## Value-function assumptions", "",
          "- Shot option: points (3 at 22 ft or more, else 2) times the location-only xFG probability at the player's position, "
          "minus %.2f per foot of nearest-defender distance below %.0f ft when no contest model exists (floor %.2f)." % (F.CONTEST_PER_FT, F.CONTEST_FT, F.P_FLOOR),
          "- Pass option: the teammate's shot value times openness (1 at %.0f ft or more of nearest-defender distance, linearly down to %.1f at 0 ft) "
          "times pass risk (%.2f minus %.2f per foot of pass length over %.0f ft)." % (F.OPEN_FT, F.OPEN_MIN, F.PASS_RISK_BASE, F.PASS_RISK_PER_FT, F.PASS_RISK_FT),
          "- Hold option: the EPV lane's epv_points of the last state at or before the decision frame; absent when the EPV table is absent.",
          "- Drive (chosen value only): the shot value at the position %.0f s ahead along the decision-frame velocity, never nearer than %.0f ft, "
          "times %.2f; a drive is not in the option set, so its quality can exceed 0." % (F.DRIVE_LOOKAHEAD_S, F.DRIVE_MIN_FT, F.DRIVE_RISK),
          "- Attacked basket: the end of the court where the offence's mean x over the possession lies.",
          "- Event detection: holder within %.1f ft of the ball under %.0f ft; pass arrives within %.1f s with no release in between; "
          "drive at over %.0f ft/s toward the basket for %.1f s." % (F.HOLD_FT, F.HOLD_Z_FT, F.PASS_MAX_S, F.DRIVE_FTPS, F.DRIVE_S),
          "- Leakage: option values use the decision frame and earlier frames only; the value model is out-of-fold by game; "
          "`schema.assert_time_forward` ran on every context frame; `assert_decisions_before_outcome` on every game.",
          "- Prior of the option values: candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. "
          "Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least %.2f. Selected: **%s**%s." % (
              XFG_MIN_COVERAGE, res["prior"], "" if res["has_xfg"] else " (no xFG table present)"), "",
          "## Player table (training games, gated at %d decisions; insufficient rows are not ranked)" % N_MIN, "",
          "| player_id | n_decisions | n_games | mean quality | lo | hi | insufficient |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["players"].head(25).itertuples(index=False):
        L.append("| %d | %d | %d | %s | %s | %s | %s |" % (r.player_id, r.n_decisions, r.n_games, _f(r.mean_quality), _f(r.lo), _f(r.hi), "yes" if r.insufficient else "no"))
    n_ins = int(res["players"]["insufficient"].sum()) if len(res["players"]) else 0
    L += ["", "%d of %d players are under the %d-decision gate." % (n_ins, len(res["players"]), N_MIN), "",
          "## Validation: summed decision quality per player-game against on-court offensive rating", "",
          "Spearman per game over players with at least %d decisions in the game; sign test over games (two sided); "
          "bootstrap of the mean Spearman by game (n=%d, seed %d); shuffled = quality permuted across the players of each game." % (
              MIN_DECISIONS_GAME, res["n_boot"], res["seed"]), "",
          "| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, v in res["validation"].items():
        L += _val_lines(n, v)
    L += ["", "## Held-out games (valued once by the model refit on all training games; never fitted or tuned on)", ""]
    if len(held):
        L += ["%d decisions from %d held-out game(s): %s." % (len(held), held["sportvu_game"].nunique(), ", ".join(sorted(held["sportvu_game"].unique()))),
              "", "| kind | n | mean quality |", "| --- | --- | --- |"]
        for kd, d in held.groupby("decision_kind"):
            L.append("| %s | %d | %s |" % (kd, len(d), _f(d["decision_quality"].mean())))
        hv = validate_quality(held, res["ortg"], res["n_boot"], res["seed"])
        L += ["", "| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        L += _val_lines(res["prior"] + " (held out)", hv)
    else:
        L.append("No held-out game in this run.")
    L += ["", "## Outputs", ""]
    L += ["- %s" % p for p in ctx["written_paths"]] or ["- none written"]
    L += ["", "## Status", ""]
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--out", help="OUT_ROOT for the parquet outputs and the upstream lanes (default DATA_ROOT; a temp dir with --fixtures)")
    ap.add_argument("--season", default=V.DEFAULT_SEASON)
    ap.add_argument("--report", default="reports/decisions.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--n-games", type=int, default=8, help="fixture games")
    ap.add_argument("--n-possessions", type=int, default=40, help="fixture possessions per game")
    ap.add_argument("--heldout-every", type=int, default=4, help="fixture: every k-th game is held out")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
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
    res = run(bundles, a.folds, a.seed, a.bootstrap)
    written = write_outputs(res, out_root, season)
    ctx = {"written_paths": [str(p) for p in written], "fixtures": a.fixtures,
           "written_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python -m analytics.decisions.train " + " ".join(argv if argv is not None else sys.argv[1:])}
    text = write_report(Path(a.report), res, ctx)
    print(text)
    res["written"] = written
    return res


if __name__ == "__main__":
    main()

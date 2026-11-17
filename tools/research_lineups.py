"""tools/research_lineups.py: the lineups research experiment lineups-E1 (PLAN.md D70, D71, D79, D84, D89).

    python3 -m tools.research_lineups e1-gate  [--artifacts DIR]
    python3 -m tools.research_lineups e1-dev   [--artifacts DIR]
    python3 -m tools.research_lineups e1-final [--artifacts DIR]

Pre-registration: docs/research/lineups.md, section 6, "lineups-E1". The question: do RAPM player ratings fitted on
the development seasons predict each game's margin, summed over its stints (Sill 2010), better than the frozen
plus_minus candidate, and better than home court alone?

e1-gate     Builds the stints of every development season (2020-21 to 2023-24, tools.research_testset.dev_games; the
            spent season 2024-25 is never read) from DATA_ROOT play-by-play with the frozen builder
            (analytics.lineups.stints.build_stints, no games table, the starters table, as the frozen stints step
            ran), and the frozen features. Gate A: the rebuilt stints equal the stored stints of each season. Gate B:
            the frozen candidates (analytics.lineups.train.fit_candidates, unchanged) refit on every development
            season choose alpha 3000 and reproduce the stored player ratings of that fit
            (DATA_ROOT/analytics/lineups/2023-24/player_rapm.parquet, fit seasons 2020-21 to 2023-24) within
            GATE_TOLERANCE. Writes the feature and game tables under <artifacts>/research/lineups_e1/ with gate.json.
            The frozen command lines are never run; nothing is written under DATA_ROOT or reports/.
e1-dev      The pre-registered dry run inside 2023-24 (fit October to February, score March and April), a
            season-ahead check labelled as not pre-registered (fit 2020-21 and 2021-22, score 2023-24: the same
            one-season gap as the test), and the settings of the test fit (alpha by the frozen CV, the replacement
            penalty by grouped CV on every development season). Writes reports/research/lineups-E1-dev.json.
e1-final    The one final scoring call: every arm fitted on every development season, the 2025-26 stints built from
            the pinned files with the same frozen code inside the scorer, scored once through
            tools.research_testset.score_once. Writes reports/research/lineups-E1-final.json.

Definitions, fixed in this file before any test score (the pre-registration leaves them open):
- Stints and possessions: the frozen builder and features; a stint's possessions are duration_s / 28.8 (frozen).
- Arms. A stint's predicted margin in points is possessions / 100 x its predicted net rating per 100, and a game's
  prediction is the sum over its stints (Sill's construction).
  floor             the home-court intercept only: the possession-weighted mean net rating per 100 of the fit stints
                    (the intercept-only weighted least squares fit).
  plus_minus        the frozen candidate: each five-man lineup's raw net rating pooled over the fit seasons
                    (fit_candidates()["plus_minus"]); a stint's rating is the floor intercept plus the home lineup's
                    rating minus the away lineup's; a lineup never seen in the fit (or with no rating) counts 0.
  rapm              the frozen candidate (fit_candidates()["rapm"]): its own intercept plus the home players'
                    coefficients minus the away players'; a player with no rating counts 0 (frozen behaviour,
                    analytics/lineups/train.py predict_lineups).
  rapm_replacement  the ablation (Watts, Pipping-Gamon and Wyner 2026, simplified as pre-registered to one shared
                    indicator per side): players under MIN_PER_APPEARANCE minutes per game played in the fit stints
                    lose their own column; two count columns (replacement players in the home lineup, in the away
                    lineup) get their own coefficients, penalised by rho x alpha with rho chosen by grouped CV by game
                    (5 folds, seed 0, out-of-fold game-margin RMSE over the fit games, ties to the larger rho; when the
                    top of RHO_GRID wins, the grid doubles past it until the choice is interior or RHO_MAX, as Watts
                    et al. extend theirs). At
                    prediction every player without his own coefficient (low minutes or never seen) counts as a
                    replacement player. The player penalty is the rapm arm's alpha. The 10-minute cut is the value
                    Watts et al. selected, fixed here and not tuned.
  player_plus_minus a context arm, not pre-registered, reading no verdict: each player's raw on-court net rating
                    per 100 possessions over the fit stints (his side's points minus the other side's, over his
                    possessions); a stint's rating is the floor intercept plus the mean of the five home players'
                    ratings minus the mean of the five away players'. The mean, not the sum: each player's raw rating
                    already measures his whole lineup's margin, so a sum over five would count it five times. A
                    player never seen counts 0, as in rapm.
- Margins. Primary: the whole game, the sum of points_home - points_away over every stint the frozen builder makes
  (the final margin of the play-by-play); stints the frozen validation drops (not five against five) are predicted
  by each arm's intercept alone. Sensitivity, reading no verdict: the stints the validation keeps only. A game the
  frozen builder drops whole (shifted period labels) has no stint and is not scored; the count is reported.
- Metric: game-margin RMSE per arm and out-of-sample R squared against the floor, 1 - SSE(arm) / SSE(floor). Every
  interval is a 95 percent percentile interval of a 2000-resample bootstrap of games (seed 0), all arms paired on the
  same resampled games. Verdict (pre-registered): rapm is validated when the interval of RMSE(rapm) -
  RMSE(plus_minus) lies wholly below zero and rapm's R squared against the floor is above zero with its interval
  above zero.

Artifacts go only under the artifacts folder (never under DATA_ROOT, never under reports/); the research results go
to reports/research/. Everything runs under the heavy-job lock.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, artifacts_dir, assert_safe_output
from tools.research_testset import PBP, PBP_DEV_SEASONS, PBP_SPENT_SEASONS, PBP_TEST_SEASON, dev_games, refuse_test, \
    score_once, scoring_data_root

WHO = "tools.research_lineups"
EXPERIMENT = "lineups-E1"
RESULTS = REPO_ROOT / "reports" / "research"
SEED, BOOT, FOLDS = 0, 2000, 5
FROZEN_ALPHA = 3000.0                 # the frozen rapm alpha on 2020-21 to 2023-24 (reports/lineups.md)
GATE_TOLERANCE = 1e-6                 # largest absolute difference of a refit player rating against the stored one
SPENT = PBP_SPENT_SEASONS[0]          # 2024-25: the frozen held-out season, passed to the frozen split guard
MIN_PER_APPEARANCE = 10.0             # Watts et al. 2026, Section 4: their selected cut; fixed, not tuned here
RHO_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0)
RHO_MAX = 10240.0                     # the grid doubles past its top while the top is chosen, up to this value
ARMS = ("floor", "plus_minus", "rapm", "rapm_replacement", "player_plus_minus")
CONTEXT_ARMS = {"player_plus_minus": "not pre-registered, reads no verdict (added by the lead before the test score)"}
DRY_SEASON = "2023-24"
DRY_FIT_MONTHS, DRY_SCORE_MONTHS = (10, 11, 12, 1, 2), (3, 4)
AHEAD_FIT, AHEAD_TARGET = ("2020-21", "2021-22"), "2023-24"
GAME_COLUMNS = ["game_id", "game_date", "season", "actual_all", "actual_valid", "poss_all", "poss_valid",
                "stints_all", "stints_valid"]


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / "lineups_e1")


# ----------------------------------------------------------------------------- stints, features, games

def season_tables(playbyplay, starters):
    """Frozen stints and features of one season's play-by-play -> (feats, games, dropped game ids). feats is the
    frozen feature frame (validated stints); games has one row per game with stints (GAME_COLUMNS)."""
    import pandas as pd
    from analytics.lineups.features import features, possessions
    from analytics.lineups.stints import build_stints
    stints = build_stints(playbyplay, pd.DataFrame(), starters)
    dropped = list(stints.attrs.get("dropped_shifted_period_games", []))
    feats = features(stints, pd.DataFrame())
    return feats, game_table(stints, feats), dropped, stints


def game_table(stints, feats):
    """One row per game of the built stints: the whole-game margin and possessions, and the same over the stints the
    frozen validation keeps (feats)."""
    import pandas as pd
    from analytics.lineups.features import possessions
    s = pd.DataFrame({"game_id": stints["game_id"].astype(str),
                      "game_date": stints["game_date"].astype(str).str[:10],
                      "season": stints["season"].astype(str),
                      "m": stints["points_home"].astype(float) - stints["points_away"].astype(float),
                      "p": possessions(stints).to_numpy()})
    g = s.groupby("game_id", sort=True).agg(game_date=("game_date", "first"), season=("season", "first"),
                                            actual_all=("m", "sum"), poss_all=("p", "sum"), stints_all=("m", "size"))
    v = feats.groupby(feats["game_id"].astype(str)).agg(actual_valid=("points_diff", "sum"),
                                                        poss_valid=("possessions", "sum"),
                                                        stints_valid=("points_diff", "size"))
    g = g.join(v, how="left")
    g[["actual_valid", "poss_valid"]] = g[["actual_valid", "poss_valid"]].fillna(0.0)
    g["stints_valid"] = g["stints_valid"].fillna(0).astype(int)
    return g.reset_index()[GAME_COLUMNS]


def read_dev_season(data_root, season: str):
    """The play-by-play and starters of one development season from DATA_ROOT, restricted to dev_games."""
    from analytics.lineups import read_season_tables
    if season not in PBP_DEV_SEASONS:
        raise SystemExit(f"{WHO}: {season} is not a development season ({PBP_DEV_SEASONS})")
    pbp = read_season_tables(data_root, "playbyplay", [season])
    starters = read_season_tables(data_root, "starters", [season])
    keep = set(dev_games(pbp["game_id"].astype(str).unique(), WHO, PBP))
    pbp = pbp[pbp["game_id"].astype(str).isin(keep)].reset_index(drop=True)
    starters = starters[starters["game_id"].astype(str).isin(keep)].reset_index(drop=True)
    return pbp, starters


STINT_COMPARE = ["game_id", "stint", "start_s", "end_s", "duration_s", "period_start", "period_end", "home_1", "home_2",
                 "home_3", "home_4", "home_5", "away_1", "away_2", "away_3", "away_4", "away_5", "n_home", "n_away",
                 "lineup_ok", "points_home", "points_away", "score_home", "score_away"]


def stints_equal(built, stored) -> dict:
    """Gate A: the rebuilt stints against the stored ones, on STINT_COMPARE, row for row."""
    import pandas as pd
    a = built[STINT_COMPARE].sort_values(["game_id", "stint"]).reset_index(drop=True)
    b = stored[STINT_COMPARE].sort_values(["game_id", "stint"]).reset_index(drop=True)
    if len(a) != len(b):
        return {"rows_built": int(len(a)), "rows_stored": int(len(b)), "equal": False, "mismatched_cells": None}
    bad = 0
    for c in STINT_COMPARE:
        x, y = a[c], b[c]
        if c == "game_id":
            bad += int((x.astype(str) != y.astype(str)).sum())
        else:
            bad += int((~((x.astype("float64") == y.astype("float64")) | (x.isna() & y.isna()))).sum())
    return {"rows_built": int(len(a)), "rows_stored": int(len(b)), "equal": bad == 0, "mismatched_cells": bad}


def schedule_agreement(games, data_root, season: str) -> dict:
    """Data check on a development season: the share of games whose whole-game stint margin equals the final margin
    of the frozen schedule table."""
    import pandas as pd
    p = Path(data_root) / "collect" / "bref_schedule" / "games" / f"{season}.parquet"
    if not p.exists():
        return {"checked": 0}
    s = pd.read_parquet(p, columns=["game_id", "home_score", "away_score"]).drop_duplicates("game_id")
    s["final"] = s["home_score"].astype(float) - s["away_score"].astype(float)
    m = games.merge(s[["game_id", "final"]], on="game_id", how="inner")
    m = m[m["final"].notna()]
    return {"checked": int(len(m)), "equal": int((m["final"] == m["actual_all"]).sum()),
            "share_equal": float((m["final"] == m["actual_all"]).mean()) if len(m) else None,
            "mean_abs_gap_points": float((m["final"] - m["actual_all"]).abs().mean()) if len(m) else None}


# ----------------------------------------------------------------------------- the gate record

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_gate(folder: Path, record: dict, files) -> dict:
    record = dict(record)
    record["files"] = {Path(f).name: _sha(folder / Path(f).name) for f in files}
    record["passed"] = bool(record["stints_equal_every_season"] and record["alpha"] == FROZEN_ALPHA
                            and record["players_compared"] == record["players_stored"] and record["players_stored"] > 0
                            and record["max_abs_diff"] is not None and record["max_abs_diff"] <= GATE_TOLERANCE
                            and record["files"])
    record["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (folder / "gate.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def require_gate(artifacts=None) -> dict:
    folder = work_dir(artifacts)
    p = folder / "gate.json"
    if not p.exists():
        raise SystemExit(f"{WHO}: run e1-gate first")
    record = json.loads(p.read_text())
    if not record.get("passed"):
        raise SystemExit(f"{WHO}: the gate did not pass, so nothing here is used: {record}")
    for name, sha in record["files"].items():
        if not (folder / name).exists() or _sha(folder / name) != sha:
            raise SystemExit(f"{WHO}: {name} changed after the gate was written")
    return record


def load_dev(artifacts=None, seasons=PBP_DEV_SEASONS):
    """The gated feature and game tables of the given development seasons (refuse_test on every game id)."""
    import pandas as pd
    folder = work_dir(artifacts)
    feats = pd.concat([pd.read_parquet(folder / f"features_{s}.parquet") for s in seasons], ignore_index=True)
    games = pd.concat([pd.read_parquet(folder / f"games_{s}.parquet") for s in seasons], ignore_index=True)
    refuse_test(games["game_id"].astype(str).unique(), WHO, PBP)
    refuse_test(feats["game_id"].astype(str).unique(), WHO, PBP)
    return feats, games


# ----------------------------------------------------------------------------- the arms

def floor_intercept(feats) -> float:
    """The intercept-only weighted least squares fit: possession-weighted mean net rating per 100."""
    return float(feats["points_diff"].sum() / feats["possessions"].sum() * 100.0)


def minutes_per_appearance(feats):
    """player id -> mean minutes on court per game played, over the given stints (possessions x 28.8 s)."""
    import pandas as pd
    from analytics.lineups import PLAYER_COLUMNS, SECONDS_PER_POSS
    parts = [pd.DataFrame({"player": feats[c].astype("int64").to_numpy(), "game_id": feats["game_id"].astype(str),
                           "poss": feats["possessions"].to_numpy()}) for c in PLAYER_COLUMNS]
    m = pd.concat(parts, ignore_index=True).groupby(["player", "game_id"])["poss"].sum() * SECONDS_PER_POSS / 60.0
    return m.groupby(level="player").mean()


def replacement_design(feats, retained: list):
    """Sparse [1 | retained players (+1 home, -1 away) | replacement count home | replacement count away]."""
    import numpy as np
    import scipy.sparse as sp
    from analytics.lineups.features import assert_no_leak, feature_columns
    assert_no_leak(feats[feature_columns()])
    n = len(feats)
    pix = {p: i + 1 for i, p in enumerate(retained)}
    rh, ra = 1 + len(retained), 2 + len(retained)
    rows, cols, vals = [np.arange(n)], [np.zeros(n, dtype=int)], [np.ones(n)]
    for side, sign, rcol in (("home", 1.0, rh), ("away", -1.0, ra)):
        for k in range(1, 6):
            c = np.array([pix.get(int(p), -1) for p in feats[f"{side}_{k}"].to_numpy()])
            m = c >= 0
            rows.append(np.flatnonzero(m)); cols.append(c[m]); vals.append(np.full(int(m.sum()), sign))
            rows.append(np.flatnonzero(~m)); cols.append(np.full(int((~m).sum()), rcol)); vals.append(np.ones(int((~m).sum())))
    X = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, 3 + len(retained))).tocsr()        # duplicates (several replacement players) add up
    return X


def _penalty(n_cols: int, alpha: float, rho: float):
    import numpy as np
    d = np.full(n_cols, alpha)
    d[0] = 0.0
    d[-2:] = rho * alpha
    return d


def _solve(A, b):
    """The ridge solution; the minimum-norm least squares one when the system is singular (rho 0 with a count column
    that is all zero, as when every player in the fit is retained)."""
    import numpy as np
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _game_sums(games_of_rows, values):
    import pandas as pd
    return pd.Series(values).groupby(pd.Series(games_of_rows)).sum()


def choose_rho(X, y, w, games, alpha: float, grid=RHO_GRID, n_folds: int = FOLDS, seed: int = SEED):
    """Grouped CV by game: the rho with the smallest out-of-fold game-margin RMSE over the fit stints (a game's margin
    and prediction summed over its stints, in points); ties go to the larger rho. -> (rho, {rho: rmse})."""
    import numpy as np
    from analytics.eval import grouped_folds
    from analytics.lineups.train import normal_equations
    A, b = normal_equations(X, y, w)
    sse = {r: 0.0 for r in grid}
    n_games = len(np.unique(games))
    for _tr, te in grouped_folds(np.unique(games), n_folds=n_folds, seed=seed):
        m = np.isin(games, te)
        A_te, b_te = normal_equations(X[m], y[m], w[m])
        actual = _game_sums(games[m], w[m] * y[m] / 100.0)
        for r in grid:
            beta = _solve(A - A_te + np.diag(_penalty(A.shape[0], alpha, r)), b - b_te)
            pred = _game_sums(games[m], w[m] * np.asarray(X[m] @ beta).ravel() / 100.0)
            sse[r] += float(((actual - pred) ** 2).sum())
    rmse = {float(r): float(np.sqrt(v / n_games)) for r, v in sse.items()}
    best = min(rmse, key=lambda r: (rmse[r], -r))
    if best == max(grid) and best * 2 <= RHO_MAX:          # Watts et al.: extend the grid until the choice is interior
        return choose_rho(X, y, w, games, alpha, tuple(grid) + (best * 2,), n_folds, seed)
    return best, rmse


def fit_replacement(feats, alpha: float, rho=None) -> dict:
    import numpy as np
    from analytics.lineups.features import TARGET
    from analytics.lineups.train import normal_equations
    mpa = minutes_per_appearance(feats)
    retained = sorted(int(p) for p in mpa.index[mpa.to_numpy() >= MIN_PER_APPEARANCE])
    X = replacement_design(feats, retained)
    y = feats[TARGET].to_numpy(dtype=float)
    w = feats["possessions"].to_numpy(dtype=float)
    games = feats["game_id"].astype(str).to_numpy()
    cv = {}
    if rho is None:
        rho, cv = choose_rho(X, y, w, games, alpha)
    A, b = normal_equations(X, y, w)
    beta = _solve(A + np.diag(_penalty(A.shape[0], alpha, rho)), b)
    return {"retained": retained, "beta": beta, "rho": float(rho), "cv_game_rmse": cv, "alpha": float(alpha),
            "players_seen": int(len(mpa)), "players_retained": len(retained),
            "replacement_home": float(beta[-2]), "replacement_away": float(beta[-1])}


def player_plus_minus(feats):
    """player id -> raw on-court net rating per 100 possessions over the given stints (signed from the player's side)."""
    import pandas as pd
    parts = []
    for side, sign in (("home", 1.0), ("away", -1.0)):
        for k in range(1, 6):
            parts.append(pd.DataFrame({"player": feats[f"{side}_{k}"].astype("int64").to_numpy(),
                                       "diff": sign * feats["points_diff"].to_numpy(dtype=float),
                                       "poss": feats["possessions"].to_numpy(dtype=float)}))
    t = pd.concat(parts, ignore_index=True).groupby("player")[["diff", "poss"]].sum()
    return (t["diff"] / t["poss"] * 100.0).to_dict()


def fit_arms(feats, target_season, rho=None) -> dict:
    """Every arm on the fit stints. The frozen candidates come from analytics.lineups.train.fit_candidates (its split
    guards run inside: no target-season row, no 2024-25 row, refuse_heldout)."""
    from analytics.eval import fit_threads
    from analytics.lineups.train import fit_candidates
    refuse_test(feats["game_id"].astype(str).unique(), WHO, PBP)
    with fit_threads():
        fits = fit_candidates(feats, target_season, SPENT)
        rep = fit_replacement(feats, fits["rapm"]["alpha"], rho)
    return {"floor": floor_intercept(feats), "plus_minus": fits["plus_minus"], "rapm": fits["rapm"],
            "rapm_replacement": rep, "player_plus_minus": player_plus_minus(feats),
            "fit_seasons": fits["fit_seasons"], "n_fit_stints": fits["n_rows"],
            "n_fit_games": int(feats["game_id"].nunique()), "n_players": len(fits["players"])}


def stint_ratings(model: dict, feats) -> dict:
    """arm -> the predicted net rating per 100 of every stint (numpy arrays in feats row order)."""
    import numpy as np
    from analytics.lineups.train import design
    out = {"floor": np.full(len(feats), model["floor"])}
    pm = model["plus_minus"]
    h = feats["lineup_home"].map(pm).astype(float).fillna(0.0).to_numpy()
    a = feats["lineup_away"].map(pm).astype(float).fillna(0.0).to_numpy()
    out["plus_minus"] = model["floor"] + h - a
    r = model["rapm"]
    beta = np.array([r["intercept"]] + [r["player_coef"][p] for p in r["players"]])
    out["rapm"] = np.asarray(design(feats, r["players"], []) @ beta).ravel()
    rep = model["rapm_replacement"]
    out["rapm_replacement"] = np.asarray(replacement_design(feats, rep["retained"]) @ rep["beta"]).ravel()
    ppm = model["player_plus_minus"]
    side = {s: np.mean([feats[f"{s}_{k}"].astype("int64").map(ppm).fillna(0.0).to_numpy() for k in range(1, 6)], axis=0)
            for s in ("home", "away")}
    out["player_plus_minus"] = model["floor"] + side["home"] - side["away"]
    return out


def intercepts(model: dict) -> dict:
    return {"floor": model["floor"], "plus_minus": model["floor"], "rapm": model["rapm"]["intercept"],
            "rapm_replacement": float(model["rapm_replacement"]["beta"][0]), "player_plus_minus": model["floor"]}


def game_predictions(model: dict, feats, games):
    """One row per game: actual_all, actual_valid and, per arm, pred_all and pred_valid in points."""
    import numpy as np
    import pandas as pd
    rat = stint_ratings(model, feats)
    gid = feats["game_id"].astype(str).to_numpy()
    out = games.copy().set_index("game_id")
    icp = intercepts(model)
    for arm in ARMS:
        pts = pd.Series(feats["possessions"].to_numpy() * rat[arm] / 100.0).groupby(pd.Series(gid)).sum()
        valid = out.index.map(pts).to_numpy(dtype=float)
        valid = np.where(np.isnan(valid), 0.0, valid)
        out[f"valid_{arm}"] = valid
        out[f"all_{arm}"] = valid + icp[arm] * (out["poss_all"] - out["poss_valid"]).to_numpy() / 100.0
    return out.reset_index()


def coverage(model: dict, feats) -> dict:
    """Counts only: how much of the scored stints the ratings know (no outcome)."""
    import numpy as np
    from analytics.lineups import PLAYER_COLUMNS
    w = feats["possessions"].to_numpy()
    rated = set(model["rapm"]["players"])
    retained = set(model["rapm_replacement"]["retained"])
    unseen = np.zeros(len(feats))
    nonret = np.zeros(len(feats))
    for c in PLAYER_COLUMNS:
        ids = feats[c].astype("int64").to_numpy()
        unseen += np.array([p not in rated for p in ids])
        nonret += np.array([p not in retained for p in ids])
    seen_lineup = (feats["lineup_home"].isin(model["plus_minus"].dropna().index).to_numpy().astype(float)
                   + feats["lineup_away"].isin(model["plus_minus"].dropna().index).to_numpy().astype(float)) / 2.0
    return {"stints": int(len(feats)), "possessions": float(w.sum()),
            "share_of_player_possessions_unseen_in_fit": float((unseen * w).sum() / (10 * w.sum())),
            "share_of_player_possessions_counted_as_replacement": float((nonret * w).sum() / (10 * w.sum())),
            "share_of_lineup_possessions_with_a_fit_lineup_rating": float((seen_lineup * w).sum() / w.sum())}


# ----------------------------------------------------------------------------- metrics

def score_games(pred, n: int = BOOT, seed: int = SEED) -> dict:
    """RMSE per arm, R squared against the floor, and the paired differences, for the whole-game margin (primary)
    and the kept-stints margin (sensitivity); every interval from one shared bootstrap of games."""
    import numpy as np
    G = len(pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n, G))

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {"games": int(G), "resamples": n, "seed": seed, "context_arms": CONTEXT_ARMS}
    for variant, actual in (("whole_game", "actual_all"), ("kept_stints", "actual_valid")):
        prefix = "all" if variant == "whole_game" else "valid"
        y = pred[actual].to_numpy(dtype=float)
        sq = {a: (y - pred[f"{prefix}_{a}"].to_numpy(dtype=float)) ** 2 for a in ARMS}
        point = {a: float(np.sqrt(sq[a].mean())) for a in ARMS}
        boot = {a: np.sqrt(sq[a][idx].mean(axis=1)) for a in ARMS}
        sse_b = {a: sq[a][idx].sum(axis=1) for a in ARMS}
        res = {"rmse": {a: {"point": point[a], "ci95": ci(boot[a])} for a in ARMS}, "r2_vs_floor": {}, "paired": {}}
        for a in ARMS[1:]:
            res["r2_vs_floor"][a] = {"point": float(1 - sq[a].sum() / sq["floor"].sum()),
                                     "ci95": ci(1 - sse_b[a] / sse_b["floor"])}
        for name, a, b in (("rapm_minus_plus_minus", "rapm", "plus_minus"), ("rapm_minus_floor", "rapm", "floor"),
                           ("plus_minus_minus_floor", "plus_minus", "floor"),
                           ("replacement_minus_rapm", "rapm_replacement", "rapm"),
                           ("rapm_minus_player_plus_minus", "rapm", "player_plus_minus"),
                           ("player_plus_minus_minus_floor", "player_plus_minus", "floor")):
            lo, hi = ci(boot[a] - boot[b])
            res["paired"][name] = {"point": point[a] - point[b], "ci95": [lo, hi], "excludes_zero": bool(hi < 0 or lo > 0)}
        out[variant] = res
    p = out["whole_game"]
    d, r2 = p["paired"]["rapm_minus_plus_minus"], p["r2_vs_floor"]["rapm"]
    out["verdict"] = {"rule": "RMSE(rapm) - RMSE(plus_minus) interval wholly below 0, and R squared of rapm against "
                              "the floor above 0 with its interval above 0 (whole-game margin)",
                      "diff_ci_below_zero": bool(d["ci95"][1] < 0),
                      "r2_above_zero_with_interval": bool(r2["point"] > 0 and r2["ci95"][0] > 0),
                      "passes": bool(d["ci95"][1] < 0 and r2["point"] > 0 and r2["ci95"][0] > 0)}
    return out


def model_record(model: dict) -> dict:
    rep = model["rapm_replacement"]
    return {"fit_seasons": model["fit_seasons"], "fit_stints": model["n_fit_stints"], "fit_games": model["n_fit_games"],
            "players": model["n_players"], "alpha_rapm": model["rapm"]["alpha"],
            "cv_mse_rapm": {str(k): v for k, v in model["rapm"]["cv_mse"].items()},
            "intercept_rapm": model["rapm"]["intercept"], "intercept_floor": model["floor"],
            "lineups_with_plus_minus": int(model["plus_minus"].notna().sum()),
            "players_with_player_plus_minus": len(model["player_plus_minus"]),
            "replacement": {"min_per_appearance": MIN_PER_APPEARANCE, "players_seen": rep["players_seen"],
                            "players_retained": rep["players_retained"], "rho": rep["rho"],
                            "penalty": rep["rho"] * rep["alpha"], "cv_game_rmse": {str(k): v for k, v in rep["cv_game_rmse"].items()},
                            "coef_home_count": rep["replacement_home"], "coef_away_count": rep["replacement_away"],
                            "intercept": float(rep["beta"][0])}}


def evaluate(model: dict, feats, games) -> dict:
    pred = game_predictions(model, feats, games)
    return {"model": model_record(model), "coverage": coverage(model, feats),
            "games_without_kept_stints": int((pred["stints_valid"] == 0).sum()),
            "margin_whole_game": {"mean": float(pred["actual_all"].mean()), "sd": float(pred["actual_all"].std(ddof=1))},
            "share_of_possessions_in_kept_stints": float(pred["poss_valid"].sum() / pred["poss_all"].sum()),
            **score_games(pred)}


# ----------------------------------------------------------------------------- commands

def gate(artifacts=None) -> dict:
    import gc
    import numpy as np
    import pandas as pd
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gate.json").write_text(json.dumps({"passed": False, "status": "gate started"}) + "\n")
    seasons, files, feats_all = {}, [], []
    for season in PBP_DEV_SEASONS:
        t0 = time.time()
        pbp, starters = read_dev_season(root, season)
        feats, games, dropped, stints = season_tables(pbp, starters)
        stored = pd.read_parquet(root / "analytics" / "lineups" / season / "stints.parquet", columns=STINT_COMPARE)
        eq = stints_equal(stints, stored)
        feats.to_parquet(folder / f"features_{season}.parquet", index=False)
        games.to_parquet(folder / f"games_{season}.parquet", index=False)
        files += [f"features_{season}.parquet", f"games_{season}.parquet"]
        seasons[season] = {"playbyplay_games": int(pbp["game_id"].nunique()), "games_with_stints": int(len(games)),
                           "dropped_shifted_period_games": len(dropped), "stints": int(len(stints)),
                           "kept_stints": int(len(feats)), "stints_vs_stored": eq,
                           "schedule_agreement": schedule_agreement(games, root, season),
                           "seconds": round(time.time() - t0, 1)}
        print(f"[lineups-E1] {season}: {seasons[season]}", flush=True)
        feats_all.append(feats)
        del pbp, starters, stints, stored
        gc.collect()
    feats = pd.concat(feats_all, ignore_index=True)
    from analytics.eval import fit_threads
    from analytics.lineups.train import fit_candidates
    with fit_threads():
        fits = fit_candidates(feats, SPENT, SPENT)
    stored = pd.read_parquet(root / "analytics" / "lineups" / DRY_SEASON / "player_rapm.parquet",
                             columns=["player_id", "rapm", "fit_seasons"])
    if set(stored["fit_seasons"]) != {",".join(PBP_DEV_SEASONS)}:
        raise SystemExit(f"{WHO}: the stored player table is not the fit on {PBP_DEV_SEASONS}")
    coef = fits["rapm"]["player_coef"]
    common = [int(p) for p in stored["player_id"] if int(p) in coef]
    s = stored.set_index("player_id")["rapm"]
    diff = float(np.max(np.abs([coef[p] - s[p] for p in common]))) if common else None
    record = {"name": "lineups", "compared": ("rapm player coefficients of fit_candidates refit on the rebuilt stints of "
                                              "2020-21 to 2023-24 against the stored player_rapm of that fit "
                                              "(DATA_ROOT analytics/lineups/2023-24); rebuilt stints against the stored "
                                              "stints of each season"),
              "seasons": seasons, "stints_equal_every_season": all(v["stints_vs_stored"]["equal"] for v in seasons.values()),
              "alpha": fits["rapm"]["alpha"], "alpha_expected": FROZEN_ALPHA, "players_stored": int(len(stored)),
              "players_fit": len(coef), "players_compared": len(common), "max_abs_diff": diff, "tolerance": GATE_TOLERANCE,
              "peak_memory_gb": round(peak_rss_gb(), 2)}
    return write_gate(folder, record, files)


def dev(artifacts=None) -> dict:
    g = require_gate(artifacts)
    t0 = time.time()
    # 1. the pre-registered dry run inside 2023-24
    feats, games = load_dev(artifacts, [DRY_SEASON])
    month_f = feats["game_date"].astype(str).str[5:7].astype(int)
    month_g = games["game_date"].astype(str).str[5:7].astype(int)
    fit_games = set(games.loc[month_g.isin(DRY_FIT_MONTHS), "game_id"])
    score_games_ = games[month_g.isin(DRY_SCORE_MONTHS)].reset_index(drop=True)
    model = fit_arms(feats[feats["game_id"].isin(fit_games) & month_f.isin(DRY_FIT_MONTHS)].reset_index(drop=True), None)
    dry = evaluate(model, feats[feats["game_id"].isin(set(score_games_["game_id"]))].reset_index(drop=True), score_games_)
    print(f"[lineups-E1] dry run: {dry['verdict']} {round(time.time() - t0)} s", flush=True)
    # 2. season-ahead check with the test's one-season gap (not pre-registered; context and power only)
    feats_all, games_all = load_dev(artifacts)
    fit_rows = feats_all[feats_all["season"].astype(str).isin(AHEAD_FIT)].reset_index(drop=True)
    model = fit_arms(fit_rows, AHEAD_TARGET)
    tgt = games_all[games_all["season"].astype(str) == AHEAD_TARGET].reset_index(drop=True)
    ahead = evaluate(model, feats_all[feats_all["season"].astype(str) == AHEAD_TARGET].reset_index(drop=True), tgt)
    print(f"[lineups-E1] season-ahead check: {ahead['verdict']} {round(time.time() - t0)} s", flush=True)
    # 3. the settings of the test fit: every arm on every development season
    final_model = fit_arms(feats_all, PBP_TEST_SEASON)
    out = {"experiment": EXPERIMENT, "stage": "development seasons only (2020-21 to 2023-24); no 2024-25 or 2025-26 row read",
           "gate": {k: g[k] for k in ("stints_equal_every_season", "alpha", "alpha_expected", "players_stored",
                                       "players_compared", "max_abs_diff", "tolerance", "passed")},
           "data_checks": g["seasons"],
           "dry_run": {"what": "pre-registered: fit on 2023-24 games of October to February, scored on 2023-24 games of "
                               "March and April", **dry},
           "season_ahead_check": {"what": ("not pre-registered, context and power only: fit on 2020-21 and 2021-22, "
                                           "scored on every 2023-24 game (the one-season gap of the test)"), **ahead},
           "test_fit": {"what": "every arm fitted on every development season; the fit e1-final uses",
                        **model_record(final_model)},
           "power": power(dry, ahead),
           "seconds": round(time.time() - t0, 1), "peak_memory_gb": round(peak_rss_gb(), 2)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lineups-E1-dev.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def power(dry: dict, ahead: dict, test_games: int = 1316) -> dict:
    """The interval half-widths each development check implies for the test season's game count, scaled by the square
    root of the game counts (an estimate, not a result)."""
    import math
    out = {"test_games_listed": test_games, "what": ("half-widths of the whole-game intervals of each development check, "
                                                    "and the same scaled to the test game count by the square root "
                                                    "of the game counts (an estimate, not a result)")}
    for name, r in (("dry_run", dry), ("season_ahead_check", ahead)):
        w = r["whole_game"]
        parts = {"rapm_minus_plus_minus": w["paired"]["rapm_minus_plus_minus"]["ci95"],
                 "rapm_minus_floor": w["paired"]["rapm_minus_floor"]["ci95"], "r2_rapm": w["r2_vs_floor"]["rapm"]["ci95"]}
        scale = math.sqrt(r["games"] / test_games)
        out[name] = {"games": r["games"], **{k: {"half_width": (hi - lo) / 2, "projected_test_half_width": (hi - lo) / 2 * scale}
                                             for k, (lo, hi) in parts.items()}}
    return out


def make_scorer(model: dict):
    """The scorer score_once runs: the 2025-26 stints built from the pinned files with the frozen code, every arm
    scored once. The test data is reached only here, through scoring_data_root."""
    from analytics.lineups import read_season_tables

    def scorer(test_games):
        root = scoring_data_root(PBP)
        pbp = read_season_tables(root, "playbyplay", [PBP_TEST_SEASON])
        starters = read_season_tables(root, "starters", [PBP_TEST_SEASON])
        listed = set(map(str, test_games))
        pbp = pbp[pbp["game_id"].astype(str).isin(listed)].reset_index(drop=True)
        starters = starters[starters["game_id"].astype(str).isin(listed)].reset_index(drop=True)
        feats, games, dropped, stints = season_tables(pbp, starters)
        res = evaluate(model, feats, games)
        return {"experiment": EXPERIMENT, "stage": "fitted on every development season, scored once on the 2025-26 season",
                "test_games_listed": len(listed), "games_with_playbyplay": int(pbp["game_id"].nunique()),
                "games_dropped_shifted_periods": len(dropped), "stints": int(len(stints)), "kept_stints": int(len(feats)),
                **res}

    return scorer


def final(artifacts=None, testset_path=None, ledger_path=None, data_root=None) -> dict:
    require_gate(artifacts)
    devfile = RESULTS / "lineups-E1-dev.json"
    if not devfile.exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test set is scored after the development result exists")
    planned = json.loads(devfile.read_text())["test_fit"]
    feats, _games = load_dev(artifacts)
    model = fit_arms(feats, PBP_TEST_SEASON)
    rec = model_record(model)
    for key in ("alpha_rapm", "players", "fit_stints"):
        if rec[key] != planned[key]:
            raise SystemExit(f"{WHO}: the test fit differs from the one e1-dev recorded ({key}); nothing scored")
    if rec["replacement"]["rho"] != planned["replacement"]["rho"]:
        raise SystemExit(f"{WHO}: the replacement penalty differs from e1-dev's; nothing scored")
    scorer = make_scorer(model)
    out = score_once(EXPERIMENT, PBP, scorer, testset_path=testset_path, ledger_path=ledger_path, data_root=data_root)
    (RESULTS / "lineups-E1-final.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-gate", "e1-dev", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    run = {"e1-gate": gate, "e1-dev": dev, "e1-final": final}[a.command]
    with heavy_job(f"research lineups {a.command}"):
        out = run(a.artifacts)
    print(json.dumps(out.get("verdict", {k: out[k] for k in out if k != "seasons"}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

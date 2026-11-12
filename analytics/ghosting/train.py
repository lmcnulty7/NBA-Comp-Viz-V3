"""analytics/ghosting/train.py: the three ghosting candidates under the full evaluation protocol.

    python -m analytics.ghosting.train --fixtures | --data DATA_ROOT --report reports/ghosting.md
        [--folds 5] [--seed 0] [--bootstrap 200] [--out OUT_ROOT] [--threads 1] [--skip-heldout]
        [--fixture-games 4] [--fixture-possessions 10] [--fixture-heldout-every 0]

Steps: load and validate the games one at a time, building each game's panel while its frames are in memory
(analytics.ghosting.features.load_games); set every held-out game aside (its own
heldout flag, or analytics.eval.is_heldout_game on its id); call
analytics.eval.refuse_heldout on the training game ids; build the 5 Hz panel and the feature frame
(analytics.ghosting.features, which runs schema.assert_time_forward); fit the candidates under grouped
folds by sportvu_game and roll each one forward from every origin row; score out-of-fold mean position
error in feet at the 0.2 s and 1.0 s horizons with bootstrap intervals by game; run the paired bootstrap
of 3 vs 2 and 2 vs 1; keep the deep candidate only when its 1.0 s error beats the linear baseline with an
interval excluding zero; build the ghost_frames and ghost_possessions tables (docs/contract/ghosting.md)
for the selected candidate; validate the graded deviation against points allowed (per-game Spearman, sign
test by game); score the held-out games once with the candidates refit on every training row.

Candidates (the displacement from t-1 to t is the regression target, both axes at once):
  1 persistence   zero displacement: the ghost stays at the defender's position at t-1
  2 ridge         StandardScaler + Ridge(alpha=1) on the lag-1 features (features.feature_columns)
  3 mlp           StandardScaler + MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200, early_stopping=True)
                  on the window of the last five sampled feature vectors (features.window_columns), the
                  stand-in for the GRU: a torch GRU reading the same window reshaped to (n, 5, n_features)
                  (lag 1 first in columns; the GRU reverses to oldest first) is the drop-in
  --deep torch    the GRU of analytics/ghosting/deep.py fills slot 3 instead (one deep candidate per run, the same
                  window, target, roll-forward and keep rule); --deep mlp, the default, keeps the stand-in, and
                  --deep torch without torch falls back to it with "torch not available; MLP stand-in used"
Roll-forward: at each of the five sampled steps the candidate's predicted displacement moves its ghost,
the ghost's own position and velocity re-enter the features, and the offence, the ball and the other
defenders stay actual (the question is where this defender would have been given what everyone else did).
The 0.2 s error is the first step, the 1.0 s error the fifth; rows whose sequence ends before the fifth
step have no 1.0 s row.

Deviation: deviation_ft = dist(actual, assignment) minus dist(ghost, assignment) at the target frame;
positive means the defender was farther from his assignment than the ghost. The ghost_possessions mean
is over the 1.0 s rows of the selected candidate. Points (a TARGET of section 19) are joined to the
possession table only after every fit, for the validation.

Fewer games than folds: n_folds becomes the number of games (leave-one-game-out). One game: in-sample,
labelled, with no interval and no keep decision (the deep candidate is dropped by default). Fold convention
in the tables: >= 0 out-of-fold, -1 held-out scored once (HELDOUT_FOLD), -2 in-sample (IN_SAMPLE_FOLD).
The held-out games are scored by the selected candidate (plus persistence as a labelled floor) only.

Threads: every fit runs under analytics.eval.fit_threads(threads) with THREADS = 1 by default, because the
build container has 4 cores shared by 21 lanes and a multi-threaded BLAS spins for minutes there (an MLP fit
measured 100 s with the default pools against 1.8 s single-threaded). ``--threads N`` passes N on the Mac.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.deep_torch import add_deep_arg, resolve_deep
from analytics.eval import (bootstrap_by_game, fit_threads, fmt_ci, fmt_num, grouped_folds, is_heldout_game,
                            paired_bootstrap_by_game, refuse_heldout, add_skip_heldout_arg, skip_heldout_lines)
from analytics.ghosting import deep as D
from analytics.ghosting import features as F
from analytics.ghosting import validate as V

log = logging.getLogger("analytics.ghosting")

WHO = "ghosting.train"
CANDIDATES = ("persistence", "ridge", "mlp")
CANDIDATE_LABELS = {"persistence": "1 persistence (position at t minus 1)",
                    "ridge": "2 ridge on the lag-1 features (linear baseline)",
                    "mlp": "3 MLPRegressor 64x64 on the 5-step window (GRU stand-in)"}
HELDOUT_FOLD, IN_SAMPLE_FOLD = -1, -2        # fold convention: >= 0 out-of-fold, -1 held-out scored once, -2 in-sample
HIDDEN = (64, 64)
MAX_ITER = 200
THREADS = 1
GHOST_FRAMES_COLUMNS = ["sportvu_game", "possession", "frame", "def_player_id", "off_player_id", "actual_x", "actual_y",
                        "ghost_x", "ghost_y", "deviation_ft", "horizon_s", "fold", "heldout"]
GHOST_POSSESSIONS_COLUMNS = ["sportvu_game", "possession", "mean_deviation_ft", "points"]
GHOST_FRAMES_DTYPES = {"sportvu_game": "string", "possession": "int32", "frame": "int32", "def_player_id": "int32",
                       "off_player_id": "int32", "actual_x": "float32", "actual_y": "float32", "ghost_x": "float32",
                       "ghost_y": "float32", "deviation_ft": "float32", "horizon_s": "float32", "fold": "int32", "heldout": "bool"}
GHOST_POSSESSIONS_DTYPES = {"sportvu_game": "string", "possession": "int32", "mean_deviation_ft": "float32", "points": "int16"}
MAC_CMD = ("python -m analytics.ghosting.train --data \"<Drive>/nba_harvest\" --report reports/ghosting.md "
           "--folds 5 --seed 0 --bootstrap 1000 --out \"<Drive>/nba_harvest\"")


# ----------------------------------------------------------------------------- models

def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """The design matrix of ``columns`` after schema.assert_time_forward."""
    X = frame[columns]
    schema.assert_time_forward(X, "states_1hz")
    return X.to_numpy(dtype=float)


def fit_ridge(frame: pd.DataFrame, columns: list[str] | None = None, alpha: float = 1.0):
    """StandardScaler + Ridge on ``columns`` (default the lag-1 features) -> predict(X) of displacements."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = list(columns) if columns is not None else F.feature_columns()
    y = frame[F.TARGETS].to_numpy(dtype=float)
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(_design(frame, cols), y)
    return m.predict


def fit_mlp(frame: pd.DataFrame, seed: int = 0, columns: list[str] | None = None):
    """StandardScaler + MLPRegressor (two layers of 64, early stopping) on the window columns."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = list(columns) if columns is not None else F.window_columns()
    y = frame[F.TARGETS].to_numpy(dtype=float)
    m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER, early_stopping=True,
                                                     n_iter_no_change=10, random_state=seed)).fit(_design(frame, cols), y)
    return m.predict


def fit_candidates(train: pd.DataFrame, seed: int = 0, threads: int = THREADS, deep: str = "mlp") -> dict:
    """{candidate: predict(window_matrix) -> displacement (n, 2)}. Every fit of the trainer goes through here,
    under analytics.eval.fit_threads(threads). The window matrix is (n, N_LAGS * n_base) in window_columns order;
    the ridge reads its first block. ``deep`` fills the one deep slot ("mlp"): the sklearn stand-in, or with
    "torch" the GRU of analytics.ghosting.deep over the same window."""
    if "heldout" in train.columns and bool(train["heldout"].any()):
        raise ValueError("%s: a held-out row reached fit_candidates" % WHO)
    if len(train) < 10:
        raise ValueError("%s: %d training rows is too few to fit" % (WHO, len(train)))
    nb = len(F.BASE_FEATURES)
    with fit_threads(threads):
        ridge = fit_ridge(train)
        mlp = fit_mlp(train, seed=seed) if deep == "mlp" else D.fit_gru(train, seed=seed, threads=threads)
    return {"persistence": lambda W: np.zeros((len(W), 2)),
            "ridge": lambda W, m=ridge: m(W[:, :nb]),
            "mlp": lambda W, m=mlp: m(W)}


# ----------------------------------------------------------------------------- roll-forward

def _sequence_end(panel: pd.DataFrame) -> np.ndarray:
    """For every panel row the position of the last row of its (game, possession, defender) sequence."""
    step = panel["step"].to_numpy()
    last = panel.groupby(["sportvu_game", "possession", "def_slot"], sort=False)["step"].transform("max").to_numpy()
    return np.arange(len(panel)) + (last - step)


def roll_forward(models: dict, rows: pd.DataFrame, panel: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Ghosts of every candidate from every origin row of ``rows`` (feature-frame rows) at both horizons.
    Returns one row per (origin, horizon, candidate) with the error and the deviation in feet."""
    if len(rows) == 0:
        return pd.DataFrame(columns=["sportvu_game", "possession", "frame", "def_player_id", "off_player_id", "horizon_s",
                                     "candidate", "actual_x", "actual_y", "ghost_x", "ghost_y", "error_ft", "deviation_ft",
                                     "fold", "heldout"])
    ctx = panel[F.CTX_COLUMNS].to_numpy(dtype=float)
    dxy = panel[["def_x", "def_y"]].to_numpy(dtype=float)
    asg = panel[["asg_x", "asg_y"]].to_numpy(dtype=float)
    frame_id = panel["frame"].to_numpy()
    seq_end = _sequence_end(panel)
    n_panel = len(panel)
    p = rows["panel_row"].to_numpy().astype(int)
    nb = len(F.BASE_FEATURES)
    W0 = rows[F.window_columns()].to_numpy(dtype=float)
    cands = [c for c in CANDIDATES if c in models]       # the held-out run passes only persistence and the selected candidate
    state = {}
    for c in cands:
        state[c] = {"ghost": dxy[p - 1].copy(), "hist": [dxy[p - 2].copy(), dxy[p - 1].copy()],
                    "win": [W0[:, k * nb:(k + 1) * nb].copy() for k in range(F.N_LAGS)]}
    max_steps = max(F.HORIZON_STEPS.values())
    by_steps = {v: k for k, v in F.HORIZON_STEPS.items()}
    out = []
    for j in range(max_steps):
        q = np.minimum(p + j, n_panel - 1)
        valid = (p + j) <= seq_end[p]
        for c in cands:
            s = state[c]
            W = np.concatenate(s["win"], axis=1)
            s["ghost"] = s["ghost"] + models[c](W)
            if (j + 1) in by_steps:
                h = by_steps[j + 1]
                g = s["ghost"][valid]
                a = dxy[q][valid]
                t = asg[q][valid]
                out.append(pd.DataFrame({
                    "sportvu_game": rows["sportvu_game"].to_numpy()[valid], "possession": rows["possession"].to_numpy()[valid],
                    "frame": frame_id[q][valid], "def_player_id": rows["def_player_id"].to_numpy()[valid],
                    "off_player_id": rows["off_player_id"].to_numpy()[valid], "horizon_s": h, "candidate": c,
                    "actual_x": a[:, 0], "actual_y": a[:, 1], "ghost_x": g[:, 0], "ghost_y": g[:, 1],
                    "error_ft": np.linalg.norm(g - a, axis=1),
                    "deviation_ft": np.linalg.norm(a - t, axis=1) - np.linalg.norm(g - t, axis=1),
                    "fold": fold, "heldout": rows["heldout"].to_numpy()[valid] if "heldout" in rows.columns else False}))
            if j + 1 < max_steps:
                Fnew = F.base_features(ctx[q], s["ghost"], s["hist"][0])
                s["hist"] = [s["hist"][1], s["ghost"].copy()]
                s["win"] = [Fnew] + s["win"][:-1]
    return pd.concat(out, ignore_index=True)


def out_of_fold(frame: pd.DataFrame, panel: pd.DataFrame, n_folds: int, seed: int,
                threads: int = THREADS, deep: str = "mlp") -> tuple[pd.DataFrame, dict]:
    """Out-of-fold ghosts of every candidate -> (errors frame, protocol note)."""
    games = sorted(frame["sportvu_game"].unique())
    if len(games) == 1:
        models = fit_candidates(frame, seed, threads, deep)
        err = roll_forward(models, frame, panel, fold=IN_SAMPLE_FOLD)
        note = {"mode": "in-sample", "folds": 0, "games": 1,
                "text": "one training game: no grouped fold is possible, so the errors are IN-SAMPLE (fit and scored on the same rows)"}
        return err, note
    k = min(n_folds, len(games))
    parts = []
    for i, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr = frame[frame["sportvu_game"].isin(tr_games)]
        te = frame[frame["sportvu_game"].isin(te_games)]
        models = fit_candidates(tr, seed, threads, deep)
        parts.append(roll_forward(models, te, panel, fold=i))
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return pd.concat(parts, ignore_index=True), {"mode": "out-of-fold", "folds": k, "games": len(games), "text": text}


# ----------------------------------------------------------------------------- scoring

def _wide(err: pd.DataFrame, horizon: float) -> pd.DataFrame:
    """One row per origin at ``horizon`` with one error column per candidate."""
    e = err[err["horizon_s"] == horizon]
    keys = ["sportvu_game", "possession", "frame", "def_player_id"]
    w = e.pivot_table(index=keys, columns="candidate", values="error_ft", aggfunc="first").reset_index()
    w.columns.name = None
    return w


def score(err: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{horizon: {candidate: {point, lo, hi}}} of the mean error in feet, bootstrap by game (no interval
    with a single game)."""
    out = {}
    for h in F.HORIZONS:
        w = _wide(err, h)
        out[h] = {}
        for c in CANDIDATES:
            if c not in w.columns or len(w) == 0:
                out[h][c] = {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "rows": 0}
                continue
            r = bootstrap_by_game(w, "sportvu_game", lambda d, c=c: d[c].mean(), n=n_boot, seed=seed)
            if w["sportvu_game"].nunique() < 2:
                r = {**r, "lo": float("nan"), "hi": float("nan")}
            out[h][c] = {**r, "rows": int(len(w))}
    return out


def paired(err: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{pair: {horizon: {diff, lo, hi, excludes_zero}}} for mlp vs ridge and ridge vs persistence (a
    negative diff favours the first; no interval with a single game)."""
    out = {}
    for a, b in (("mlp", "ridge"), ("ridge", "persistence")):
        out[a + "_vs_" + b] = {}
        for h in F.HORIZONS:
            w = _wide(err, h)
            if len(w) == 0 or a not in w.columns or b not in w.columns:
                out[a + "_vs_" + b][h] = {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
                continue
            r = paired_bootstrap_by_game(w, "sportvu_game", lambda d, a=a: d[a].mean(), lambda d, b=b: d[b].mean(),
                                         n=n_boot, seed=seed)
            if w["sportvu_game"].nunique() < 2:
                r = {**r, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
            out[a + "_vs_" + b][h] = r
    return out


def decide(pairs: dict) -> dict:
    """The keep-or-drop decision for the deep candidate: kept only when its 1.0 s error is below the ridge's
    with a paired interval excluding zero."""
    r = pairs["mlp_vs_ridge"][1.0]
    keep = bool(np.isfinite(r["diff"]) and r["diff"] < 0 and r["excludes_zero"])
    return {"keep_mlp": keep, "selected": "mlp" if keep else "ridge", "diff_1s": r["diff"], "lo": r["lo"], "hi": r["hi"],
            "excludes_zero": bool(r["excludes_zero"])}


# ----------------------------------------------------------------------------- the contract tables and the validation

def ghost_tables(err: pd.DataFrame, selected: str, possessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ghost_frames (both horizons, selected candidate) and ghost_possessions (mean 1.0 s deviation joined
    to points; the join happens here, after every fit)."""
    e = err[err["candidate"] == selected]
    gf = e[GHOST_FRAMES_COLUMNS].copy().reset_index(drop=True).astype(GHOST_FRAMES_DTYPES)
    one = e[e["horizon_s"] == 1.0]
    gp = one.groupby(["sportvu_game", "possession"], as_index=False)["deviation_ft"].mean()
    gp = gp.rename(columns={"deviation_ft": "mean_deviation_ft"})
    pts = possessions[["sportvu_game", "possession", "points"]].drop_duplicates(["sportvu_game", "possession"])
    gp = gp.merge(pts, on=["sportvu_game", "possession"], how="left")
    gp["points"] = gp["points"].fillna(0).astype(int)
    return gf, gp[GHOST_POSSESSIONS_COLUMNS].astype(GHOST_POSSESSIONS_DTYPES)


def sign_test(gp: pd.DataFrame, min_possessions: int = 3) -> dict:
    """Per-game Spearman of mean_deviation_ft against points over possessions, then a sign test over games
    with a finite correlation (one-sided: more positive than negative). -> dict(per_game, n_pos, n_neg,
    n_zero, p_value)."""
    from scipy.stats import binomtest, spearmanr
    per = []
    for g, d in gp.groupby("sportvu_game"):
        rho = float("nan")
        if len(d) >= min_possessions and d["points"].nunique() > 1 and d["mean_deviation_ft"].nunique() > 1:
            rho = float(spearmanr(d["mean_deviation_ft"], d["points"]).statistic)
        per.append({"sportvu_game": g, "possessions": int(len(d)), "spearman": rho})
    per = pd.DataFrame(per, columns=["sportvu_game", "possessions", "spearman"])
    fin = per["spearman"].to_numpy(dtype=float)
    fin = fin[np.isfinite(fin)]
    n_pos, n_neg = int((fin > 0).sum()), int((fin < 0).sum())
    p = float(binomtest(n_pos, n_pos + n_neg, 0.5, alternative="greater").pvalue) if n_pos + n_neg else float("nan")
    return {"per_game": per, "n_pos": n_pos, "n_neg": n_neg, "n_zero": int((fin == 0).sum()), "p_value": p}


# ----------------------------------------------------------------------------- the run

def split_games(games: list[dict]) -> tuple[list[dict], list[dict]]:
    """(training games, held-out games); refuse_heldout runs on the training ids."""
    held = [g for g in games if g.get("heldout") or is_heldout_game(g["sportvu_game"])]
    train = [g for g in games if g not in held]
    for g in held:
        g["heldout"] = True
    refuse_heldout(sorted(g["sportvu_game"] for g in train), WHO)
    return train, held


def run(games: list[dict], n_folds: int = 5, seed: int = 0, n_boot: int = 200, threads: int = THREADS,
        skip_heldout: bool = False, deep: str = "mlp") -> dict:
    """The whole protocol on game dicts -> ctx (scores, pairs, decision, errors, ghost tables, held-out), with BLAS
    and OpenMP pinned to ``threads`` (analytics.eval.fit_threads). ``deep`` ("mlp" or "torch", resolved by
    analytics.deep_torch.resolve_deep: torch absent falls back to "mlp") picks the model of the deep slot; the
    keep rule (decide) is the same either way."""
    info = resolve_deep(deep, WHO)
    with fit_threads(threads):
        ctx = _run(games, n_folds, seed, n_boot, threads, skip_heldout, info["used"])
    ctx["deep"] = info
    return ctx


def _run(games: list[dict], n_folds: int, seed: int, n_boot: int, threads: int = THREADS, skip_heldout: bool = False,
         deep: str = "mlp") -> dict:
    """run() without the thread pin."""
    train_games, held_games = split_games(games)
    if not train_games:
        raise SystemExit("%s: no training games after setting the held-out games aside" % WHO)
    panel, frame, counts = F.build(train_games + held_games)
    schema.assert_time_forward(frame, "states_1hz")
    tr = frame[~frame["heldout"].astype(bool)].reset_index(drop=True)
    he = frame[frame["heldout"].astype(bool)].reset_index(drop=True)
    if len(tr) < 10:
        raise SystemExit("%s: %d training rows is too few" % (WHO, len(tr)))
    err, note = out_of_fold(tr, panel, n_folds, seed, threads, deep)
    scores = score(err, n_boot, seed)
    pairs = paired(err, n_boot, seed)
    decision = decide(pairs)
    poss = pd.concat([g["possessions"] for g in train_games + held_games], ignore_index=True)
    gf, gp = ghost_tables(err, decision["selected"], poss)
    val = sign_test(gp)
    ctx = {"train_games": [g["sportvu_game"] for g in train_games], "heldout_games": [g["sportvu_game"] for g in held_games],
           "counts": counts, "panel_rows": int(len(panel)), "train_rows": int(len(tr)), "heldout_rows": int(len(he)),
           "note": note, "scores": scores, "pairs": pairs, "decision": decision, "errors": err, "ghost_frames": gf,
           "ghost_possessions": gp, "validation": val, "folds": n_folds, "seed": seed, "bootstrap": n_boot,
           "panel": panel, "frame": frame, "heldout_scores": {}, "heldout_validation": None, "skip_heldout": skip_heldout}
    if len(he) and not skip_heldout:       # --skip-heldout: set aside above, never rolled forward or scored
        # the held-out game sees only the selected candidate (and persistence as a labelled floor): the other
        # candidate is never scored on it, so nothing can be chosen on held-out numbers
        models = fit_candidates(tr, seed, threads, deep)
        keep = [c for c in ("persistence", decision["selected"]) if c in models]
        herr = roll_forward({c: models[c] for c in keep}, he, panel, fold=HELDOUT_FOLD)
        ctx["heldout_errors"] = herr
        ctx["heldout_candidates"] = keep
        ctx["heldout_scores"] = {h: {c: float(herr[(herr["horizon_s"] == h) & (herr["candidate"] == c)]["error_ft"].mean())
                                     for c in keep} for h in F.HORIZONS}
        hgf, hgp = ghost_tables(herr, decision["selected"], poss)
        ctx["ghost_frames"] = pd.concat([gf, hgf], ignore_index=True)
        ctx["ghost_possessions"] = pd.concat([gp, hgp], ignore_index=True)
        ctx["heldout_validation"] = sign_test(hgp)
    return ctx


def write_tables(out_root, ctx: dict, season: str = V.SEASON) -> list[Path]:
    """ghost_frames and ghost_possessions per game under OUT_ROOT/analytics/ghosting/<season>/<table>_<game>.parquet
    (held-out under OUT_ROOT/analytics/ghosting/heldout/<season>/); OUT_ROOT defaults to DATA_ROOT."""
    written = []
    for table, df in (("ghost_frames", ctx["ghost_frames"]), ("ghost_possessions", ctx["ghost_possessions"])):
        for g, d in df.groupby("sportvu_game"):
            held = g in ctx["heldout_games"]
            p = V.output_path(out_root, table, str(g), held, season)
            p.parent.mkdir(parents=True, exist_ok=True)
            d.reset_index(drop=True).to_parquet(p, index=False)
            written.append(p)
    return written


# ----------------------------------------------------------------------------- report

def write_report(path: Path, ctx: dict, fixtures: bool, command: str) -> str:
    """Write the markdown report of a run context to path; returns the text. On fixtures every heading of a section
    that carries a number says it is fixture output, not a result."""
    tag = " (fixture output, not a result)" if fixtures else ""
    deep = ctx.get("deep") or {"requested": "mlp", "used": "mlp", "notice": ""}
    LBL = {**CANDIDATE_LABELS, **({"mlp": D.LABEL} if deep["used"] == "torch" else {})}
    L = ["# Ghosting: defender position from the past five sampled frames (analytics/ghosting/train.py)%s" % tag, "",
         "Written %s by `%s`." % (dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command), ""]
    if fixtures:
        L += ["**Fixture output, not a result.** Every number below comes from synthetic games (analytics.fixtures); it "
              "shows the pipeline runs end to end and nothing about NBA defenders.", ""]
    L += ["## Inputs%s" % tag, "", "| game | held out | possessions | used | skipped (gap, no offence, short, lineup) | sampled steps |",
          "| --- | --- | --- | --- | --- | --- |"]
    for g in ctx["train_games"] + ctx["heldout_games"]:
        c = ctx["counts"][g]
        L.append("| %s | %s | %d | %d | %d, %d, %d, %d | %d |" % (g, g in ctx["heldout_games"], c["possessions"], c["used"],
                                                                 c["skipped_gap"], c["skipped_no_offense"], c["skipped_short"],
                                                                 c["skipped_lineup"], c["steps"]))
    L += ["", "Training rows: %d defender-steps from %d game(s) (%s). Held-out rows set aside before any fit: %d from %d game(s) (%s)."
          % (ctx["train_rows"], len(ctx["train_games"]), ", ".join(ctx["train_games"]), ctx["heldout_rows"],
             len(ctx["heldout_games"]), ", ".join(ctx["heldout_games"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids against sportvu/splits.json; "
          "`schema.assert_time_forward` passed on the feature frame (%d columns, %d lag-1 features, %d window columns)."
          % (len(ctx["train_games"]), ctx["frame"].shape[1], len(F.feature_columns()), len(F.window_columns())), "",
          "## Protocol%s" % tag, "",
          "Unit: defender at a sampled frame (5 Hz, every %d frames of the 25 Hz tracking) inside a possession; assignment = nearest "
          "offensive player at the possession's first sampled frame. Features at t minus 1 (and t minus 2 to t minus 5 for the window): "
          "%s." % (F.STRIDE, ", ".join(F.feature_columns())),
          "Target: displacement of the defender from t minus 1 to t. The 1.0 s horizon rolls the ghost forward five steps with its own "
          "position fed back and everyone else actual.",
          "Candidates: " + "; ".join(LBL[c] for c in CANDIDATES) + ".",
          *([] if deep["requested"] == "mlp" else
            ["Deep candidate (--deep torch): %s." % (deep["notice"] if deep["notice"] else
             "torch GRU, hidden %d, over the same 5-step window reshaped to (n, %d, %d) oldest step first, the same "
             "target and the same roll-forward; Adam, batch %d, at most %d epochs, early stopping after %d epochs on an "
             "inner split of the training games, then refit on every training row for the best epoch count; kept only by the rule below" % (D.GRU_HIDDEN, F.N_LAGS, len(F.BASE_FEATURES),
                                                                                 D.BATCH, D.MAX_EPOCHS, D.PATIENCE))]),
          "Metrics: %s; mean position error in feet; intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)%s."
          % (ctx["note"]["text"], ctx["bootstrap"], ctx["seed"], "; no interval with a single game" if len(ctx["train_games"]) < 2 else ""),
          "", "## Candidates by horizon (%s, feet)%s" % (ctx["note"]["mode"], tag), "",
          "| candidate | 0.2 s error | rows | 1.0 s error | rows |", "| --- | --- | --- | --- | --- |"]
    for c in CANDIDATES:
        s2, s1 = ctx["scores"][0.2][c], ctx["scores"][1.0][c]
        L.append("| %s | %s | %d | %s | %d |" % (LBL[c], fmt_ci(s2), s2["rows"], fmt_ci(s1), s1["rows"]))
    L += ["", "## Paired bootstrap by game (difference of mean error, feet; negative favours the first)%s" % tag, "",
          "| pair | horizon | diff | lo | hi | excludes zero |", "| --- | --- | --- | --- | --- | --- |"]
    for pair, byh in ctx["pairs"].items():
        for h, r in byh.items():
            name = pair.replace("_", " ") if deep["used"] == "mlp" else pair.replace("mlp_", "deep (GRU) ").replace("_", " ")
            L.append("| %s | %.1f s | %s | %s | %s | %s |" % (name, h, fmt_num(r["diff"]), fmt_num(r["lo"]), fmt_num(r["hi"]), r["excludes_zero"]))
    d = ctx["decision"]
    L += ["", "## Decision%s" % tag, "",
          "The deep candidate (3) is **%s**: its 1.0 s error minus the linear baseline's is %s with paired interval [%s, %s], which %s zero. "
          "The rule is fixed in advance: keep 3 only when the difference is negative and the interval excludes zero. Selected ghost: %s%s."
          % ("kept" if d["keep_mlp"] else "dropped", fmt_num(d["diff_1s"]), fmt_num(d["lo"]), fmt_num(d["hi"]),
             "excludes" if d["excludes_zero"] else "does not exclude", LBL[d["selected"]], tag), "",
          "## Validation: graded deviation against points allowed (%s rows, selected ghost, 1.0 s horizon)%s" % (ctx["note"]["mode"], tag), "",
          "deviation_ft = distance(actual, assignment) minus distance(ghost, assignment); per possession the mean over its 1.0 s rows; "
          "per game the Spearman correlation of that mean with the points the offence scored; then a sign test over games "
          "(one-sided, more positive than negative correlations).", "",
          "| game | possessions | Spearman |", "| --- | --- | --- |"]
    for r in ctx["validation"]["per_game"].itertuples():
        L.append("| %s | %d | %s |" % (r.sportvu_game, r.possessions, fmt_num(r.spearman)))
    v = ctx["validation"]
    L += ["", "Sign test: %d positive, %d negative, %d zero; one-sided p = %s%s." % (v["n_pos"], v["n_neg"], v["n_zero"], fmt_num(v["p_value"], 4), tag), ""]
    if ctx.get("skip_heldout"):
        L += ["## Held-out games%s" % tag, ""] + skip_heldout_lines(len(ctx["heldout_games"])) + [""]
    elif ctx["heldout_games"]:
        L += ["## Held-out games (the selected candidate refit on all %d training rows, scored once; never fitted or tuned on)%s"
              % (ctx["train_rows"], tag), "",
              "%d defender-steps from %d held-out game(s): %s. Only the selected candidate and the persistence floor are scored here; "
              "the dropped candidate never sees the held-out game." % (ctx["heldout_rows"], len(ctx["heldout_games"]), ", ".join(ctx["heldout_games"])), "",
              "| candidate | 0.2 s error | 1.0 s error |", "| --- | --- | --- |"]
        for c in ctx["heldout_candidates"]:
            L.append("| %s | %s | %s |" % (LBL[c], fmt_num(ctx["heldout_scores"][0.2][c]), fmt_num(ctx["heldout_scores"][1.0][c])))
        hv = ctx["heldout_validation"]
        L += ["", "Held-out numbers are point estimates (no bootstrap over %d game(s)). Held-out sign test: %d positive, %d negative, p = %s."
              % (len(ctx["heldout_games"]), hv["n_pos"], hv["n_neg"], fmt_num(hv["p_value"], 4)), ""]
    else:
        L += ["## Held-out games", "", "None in this run; the Mac run scores the held-out games once after the decision above.", ""]
    L += ["## Tables%s" % tag, "",
          "ghost_frames: %d rows; ghost_possessions: %d rows (docs/contract/ghosting.md)." % (len(ctx["ghost_frames"]), len(ctx["ghost_possessions"])), "",
          "## Assumptions and what needs real data", "",
          "- Assignment is the nearest offensive player at the possession's first sampled frame (defender-centric, no one-to-one matching); "
          "the attacking side is the half the ball is in at t minus 1; a null shot clock is max(24 minus elapsed, 0); possessions with a "
          "clock gap or no offence are skipped; a sequence is cut at the first sampled frame without the first frame's ten players or without a ball.",
          "- The 1.0 s roll-forward keeps the offence, the ball and the other defenders actual and feeds only this defender's ghost back.",
          "- Fits run under `analytics.eval.fit_threads(--threads)`, single-threaded by default because the build container shares 4 cores "
          "across 21 lanes; `--threads` raises it on the Mac.",
          "- Real data: the Step 1 frames and the section 19 possessions of every tracked game on the owner's Mac (DATA_ROOT layout); the "
          "held-out games are %s. The sign test needs dozens of "
          % ("set aside and left unscored until a run without --skip-heldout" if ctx.get("skip_heldout")
             else "scored once, after the decision, by the candidates refit on every training row") +
          "games to say anything; with the fixtures it is a pipeline check.", "",
          "## Next on the Mac", "", "    " + MAC_CMD, "",
          "Torch GRU note: the deep candidate here is sklearn's MLPRegressor over the flattened 5-step window because torch is not installed "
          "in the build container. The window columns are ordered lag 1 first, so `X.reshape(n, %d, %d)` is the (batch, time, features) tensor "
          "a `torch.nn.GRU` consumes (reverse the time axis for oldest-first); it replaces `fit_mlp` with no change to the features, the "
          "roll-forward or the decision rule, and is kept under the same paired-bootstrap test." % (F.N_LAGS, len(F.BASE_FEATURES)), ""]
    if deep["used"] == "torch":      # the note above describes the stand-in; this run used the GRU itself
        L[-2] = ("Torch GRU note: the deep candidate here is the torch GRU of analytics/ghosting/deep.py (--deep torch) over the same "
                 "window reshaped to (n, %d, %d), oldest step first, with no change to the features, the roll-forward or the decision "
                 "rule." % (F.N_LAGS, len(F.BASE_FEATURES)))
    elif deep["notice"]:
        L[-2] += " This run asked for --deep torch: %s." % deep["notice"]
    text = "\n".join(L)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    """The trainer's command line (validate.add_input_args plus the protocol flags)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    V.add_input_args(ap)
    ap.add_argument("--report", default="reports/ghosting.md", help="markdown report path (default reports/ghosting.md)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by sportvu_game (default 5; fewer games: one per game)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples of whole games for every interval (default 200)")
    ap.add_argument("--out", help="OUT_ROOT (default DATA_ROOT): write ghost_frames and ghost_possessions per game under "
                                  "OUT_ROOT/analytics/ghosting/<season>/")
    ap.add_argument("--threads", type=int, default=THREADS,
                    help="BLAS and OpenMP threads for the fits, passed to analytics.eval.fit_threads (default 1, the shared build container)")
    add_skip_heldout_arg(ap)
    add_deep_arg(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """CLI: run the protocol, write the report and the per-game tables (under --out, else DATA_ROOT); returns ctx."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = parse_args(argv)
    games = F.load_games(a)          # panels built game by game; frames are never all in memory
    ctx = run(games, n_folds=a.folds, seed=a.seed, n_boot=a.bootstrap, threads=max(int(a.threads), 1),
              skip_heldout=a.skip_heldout, deep=a.deep)
    cmd = "python -m analytics.ghosting.train " + " ".join(argv if argv is not None else sys.argv[1:])
    text = write_report(Path(a.report), ctx, fixtures=bool(a.fixtures), command=cmd)
    print(text)
    out_root = a.out or a.data
    if out_root:
        for p in write_tables(out_root, ctx):
            print("wrote", p)
    return ctx


if __name__ == "__main__":
    main(sys.argv[1:])

"""tools/research_rebounds.py: the rebounds research experiment rebounds-E1 (PLAN.md D70, D124, D125).

    python3 -m export.heavy --name rebounds-E1 -- python3 -m tools.research_rebounds e1-build
    python3 -m export.heavy --name rebounds-E1 -- python3 -m tools.research_rebounds e1-fit [--max-minutes 26]
    python3 -m export.heavy --name rebounds-E1 -- python3 -m tools.research_rebounds e1-dev

Pre-registration: docs/research/rebounds.md, section 6, "rebounds-E1", with its D125 amendment: the development
games decide; Set A is scored once as a D108 confirmation only if development adopts (no Set A path is written
before that).

e1-build  The candidate rows of every development game (tools.research_testset.dev_games), built with the frozen
          code unchanged (analytics.rebounds.validate.clean_bundle, analytics.rebounds.features.build), one game's
          frames at a time. Only the training directory is listed; nothing under a heldout/ directory is read. The
          frozen run stored no candidate-row files, so they are rebuilt here (the pre-registration assumed they were
          stored). Checked against the frozen record (537,175 rows, 49,288 misses, 604 games).
e1-fit    Resumable, one outer fold a piece (the frozen outer folds: analytics.eval.grouped_folds over the training
          games, 5 folds, seed 0): the frozen candidates fitted by analytics.rebounds.train.fit_candidates, unchanged,
          on the fold's training rows (the baseline: hgb with its single tau), and the raw hgb q (predict_proba before
          any normalisation) of (1) the fold's outer-test rows from that fit and (2) its training rows out of fold,
          from hgb fits on the inner grouped folds of inner_temperature (analytics.rebounds.train._fit with the
          temperature off, the frozen inner fit). After the fifth piece, gate A: the baseline out-of-fold
          probabilities must equal the stored p_selected of every training candidate row and give the stored per-miss
          log loss 1.6009.
e1-dev    The 5-parameter fit and the ablation on the inner rows of each fold, applied to its outer-test rows; the
          metrics and the paired game bootstrap; writes reports/research/rebounds-E1-dev.json.

Definitions, fixed in this file before any development number:
- E1: within a miss, p proportional to exp(a_g log q + b_g), g the candidate type (offense player, defense player,
  none), q the raw hgb probability; five parameters (three slopes; offsets for defense and none, offense 0). Fitted
  per outer fold on that fold's inner out-of-fold rows (the rows the frozen tau is fitted on) by the per-miss log
  loss, L-BFGS from a = tau, b = 0.
- Ablation (pre-registered): one shared slope a and the two offsets (three parameters), fitted the same way.
- The baseline is the frozen p: q ** tau normalised within the miss.
- Metric: out-of-fold per-miss multinomial log loss (analytics.rebounds.train.per_miss); the paired difference
  E1 minus baseline with a 2000-resample bootstrap of games (seed 0). Verdict (pre-registered, D125): the difference
  at most -0.005 with its 95 percent interval excluding zero. Also reported: top-1, the favourite's calibration slope
  and ECE (10 equal-mass bins), team-level accuracy on misses whose rebounder is a player (offense against defense
  by summed probability), the fitted parameters per fold, and the projected Set A half-width (D125).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, _plain, artifacts_dir, assert_safe_output
from tools.research_testset import dev_games, refuse_test

WHO = "tools.research_rebounds"
EXPERIMENT = "rebounds-E1"
RESULTS = REPO_ROOT / "reports" / "research"
SEASON = "2015-16"
SEED, BOOT, FOLDS = 0, 2000, 5
FROZEN_RECORD = {"rows": 537175, "misses": 49288, "games": 604}
FROZEN_LOGLOSS = 1.6009                    # reports/rebounds.md, out-of-fold per-miss log loss of hgb (4 decimals)
GATE_TOL = 1e-9
THRESHOLD = -0.005
FOLD_MINUTES = 10.0


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / "rebounds_e1")


# ----------------------------------------------------------------------------- candidate rows

def training_games(root: Path) -> list:
    """Development games with a rebound_frames table and frames in the training directories (heldout/ never listed)."""
    rf = sorted(p.stem for p in (root / "analytics" / "rebound_frames" / SEASON).glob("*.parquet"))
    have = [g for g in rf if (root / "sportvu_contract" / f"{g}.parquet").exists()]
    return dev_games(have, WHO)


def bundles(root: Path, games):
    """The frozen bundle (analytics.rebounds.validate.bundles_from_root) of each given game, one at a time."""
    import pandas as pd
    from analytics.rebounds import validate as V
    for g in games:
        frames = root / "sportvu_contract" / f"{g}.parquet"
        events = root / "sportvu_contract" / f"{g}_pbp.parquet"
        b = {"sportvu_game": g, "heldout": False, "frames": V.read_frames(frames),
             "events": pd.read_parquet(events) if events.exists() else None}
        for t in V.TABLES:
            p = root / "analytics" / t / SEASON / f"{g}.parquet"
            b[t] = pd.read_parquet(p) if p.exists() else None
        yield b


def build(artifacts=None) -> dict:
    """e1-build."""
    from analytics.rebounds import features as F
    from analytics.rebounds import train as T
    from analytics.rebounds import validate as V
    root = Path(os.environ["DATA_ROOT"])
    games = training_games(root)
    t0 = time.time()

    def cleaned():
        for b in bundles(root, games):
            cb, _counts = V.clean_bundle(b)
            yield cb
    cands, _fc = F.build(cleaned())
    train, held = T.split_train_heldout(cands)
    if len(held):
        raise SystemExit(f"{WHO}: a held-out row came out of the development games")
    rec = {"rows": int(len(train)), "misses": int(train[["sportvu_game", "event"]].drop_duplicates().shape[0]),
           "games": int(train["sportvu_game"].nunique())}
    rec["matches_frozen_record"] = rec == FROZEN_RECORD
    if not rec["matches_frozen_record"]:
        raise SystemExit(f"{WHO}: the rebuilt candidate rows are not the frozen record's: {rec} against {FROZEN_RECORD}")
    folder = work_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    train.to_parquet(folder / "candidates_train.parquet", index=False)
    rec.update(seconds=round(time.time() - t0, 1), peak_gb=round(peak_rss_gb(), 2))
    (folder / "candidates_train.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def load_candidates(artifacts=None):
    import pandas as pd
    folder = work_dir(artifacts)
    rec = json.loads((folder / "candidates_train.json").read_text())
    if not rec.get("matches_frozen_record"):
        raise SystemExit(f"{WHO}: run e1-build first")
    t = pd.read_parquet(folder / "candidates_train.parquet")
    refuse_test(t["sportvu_game"].astype(str).unique(), WHO)
    return t


# ----------------------------------------------------------------------------- the fold pieces

def raw_q(predict, frame):
    """The raw hgb probability (predict_proba, before temperature or normalisation) of a frozen hgb closure."""
    from analytics.eval import fit_threads
    from analytics.rebounds import train as T
    m, rates, _tau = predict.__defaults__
    X = T.design(frame, T.apply_prior(frame, rates))
    with fit_threads(1):
        return m.predict_proba(X.to_numpy())[:, 1]


def fold_piece(train, k: int, folds) -> dict:
    """The frozen candidates on outer fold k's training rows; raw q of its test rows and of its training rows out of
    fold over the inner folds of inner_temperature."""
    import numpy as np
    from analytics.eval import grouped_folds
    from analytics.rebounds import train as T
    tr_g, te_g = folds[k]
    g = train["sportvu_game"].astype(str)
    tr, te = g.isin(tr_g).to_numpy(), g.isin(te_g).to_numpy()
    trn, tst = train[tr].reset_index(drop=True), train[te].reset_index(drop=True)
    models = T.fit_candidates(trn, SEED)
    hgb = models["hgb"]
    tau = float(hgb.__defaults__[2])
    p_base = hgb(tst)
    q_test = raw_q(hgb, tst)
    q_inner = np.full(len(trn), np.nan)
    gk = trn["sportvu_game"].astype(str)
    games_k = sorted(gk.unique())
    for itr, ite in grouped_folds(games_k, n_folds=min(T.INNER_FOLDS, len(games_k)), seed=SEED):
        a, b = gk.isin(itr).to_numpy(), gk.isin(ite).to_numpy()
        f = T._fit(trn[a], SEED, only="hgb", temperature=False)["hgb"]
        q_inner[b] = raw_q(f, trn[b])
    if not np.isfinite(q_inner).all():
        raise SystemExit(f"{WHO}: an inner row of fold {k} has no out-of-fold q")
    return {"test_rows": np.flatnonzero(te), "train_rows": np.flatnonzero(tr), "p_base": np.asarray(p_base, dtype=float),
            "q_test": q_test, "q_inner": q_inner, "tau": tau}


def fit_pieces(artifacts=None, max_minutes: float = 26.0, log=print, head: str = "") -> dict:
    import numpy as np
    from analytics.eval import grouped_folds
    t0 = time.time()
    folder = work_dir(artifacts) / "folds"
    folder.mkdir(parents=True, exist_ok=True)
    train = load_candidates(artifacts)
    folds = grouped_folds(sorted(train["sportvu_game"].astype(str).unique()), n_folds=FOLDS, seed=SEED)
    done = []
    for k in range(FOLDS):
        f = folder / f"fold_{k}.npz"
        if f.exists():
            continue
        if max_minutes - (time.time() - t0) / 60 < FOLD_MINUTES:
            break
        r = fold_piece(train, k, folds)
        part = folder / f"fold_{k}.part.npz"
        np.savez(part, **r, code_commit=np.array(head))
        os.replace(part, f)
        done.append(k)
        log(f"[rebounds-E1] fold {k} done {round(time.time() - t0)} s, tau {r['tau']:.4f}, peak {peak_rss_gb():.1f} GB",
            flush=True)
    left = [k for k in range(FOLDS) if not (folder / f"fold_{k}.npz").exists()]
    out = {"done_this_job": done, "left": left, "seconds": round(time.time() - t0, 1)}
    if not left:
        out["gate"] = gate(train, artifacts)
    return out


def gate(train, artifacts=None) -> dict:
    """Gate A: the baseline out-of-fold p against the stored p_selected of every training candidate row, and the
    per-miss log loss against the frozen 1.6009."""
    import numpy as np
    import pandas as pd
    from analytics.rebounds import train as T
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts) / "folds"
    p = np.full(len(train), np.nan)
    fold = np.full(len(train), -1)
    for k in range(FOLDS):
        z = np.load(folder / f"fold_{k}.npz")
        p[z["test_rows"]] = z["p_base"]
        fold[z["test_rows"]] = k
    games = sorted(train["sportvu_game"].astype(str).unique())
    stored = pd.concat([pd.read_parquet(root / "analytics" / "rebounds" / SEASON / f"{g}.parquet",
                                        columns=["sportvu_game", "event", "candidate_player_id", "p_selected", "fold"])
                        for g in games], ignore_index=True)
    stored = stored[stored["fold"] >= 0]
    mine = train[["sportvu_game", "event", "candidate_player_id"]].copy()
    mine["p"], mine["fold_mine"] = p, fold
    for t in (mine, stored):
        t["sportvu_game"] = t["sportvu_game"].astype(str)
    m = mine.merge(stored, on=["sportvu_game", "event", "candidate_player_id"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    pred = train.copy()
    pred["p_hgb"] = p
    ll = T.mlogloss(T.per_miss(pred, "p_hgb"))
    rec = {"rows_mine": int(len(mine)), "rows_stored": int(len(stored)), "matched": int(len(both)),
           "only_mine": int((m["_merge"] == "left_only").sum()), "only_stored": int((m["_merge"] == "right_only").sum()),
           "fold_mismatches": int((both["fold_mine"].to_numpy() != both["fold"].to_numpy()).sum()),
           "max_abs_diff": float(np.max(np.abs(both["p"].to_numpy() - both["p_selected"].to_numpy()))) if len(both) else None,
           "tolerance": GATE_TOL, "per_miss_log_loss": ll, "frozen_log_loss": FROZEN_LOGLOSS}
    rec["passed"] = bool(rec["only_mine"] == 0 and rec["only_stored"] == 0 and rec["fold_mismatches"] == 0
                         and rec["max_abs_diff"] is not None and rec["max_abs_diff"] <= GATE_TOL
                         and round(ll, 4) == FROZEN_LOGLOSS)
    (work_dir(artifacts) / "gate.json").write_text(json.dumps(_plain(rec), indent=2) + "\n")
    return rec


def require_gate(artifacts=None) -> dict:
    p = work_dir(artifacts) / "gate.json"
    if not p.exists():
        raise SystemExit(f"{WHO}: run e1-fit to the end first (gate A)")
    rec = json.loads(p.read_text())
    if not rec.get("passed"):
        raise SystemExit(f"{WHO}: gate A did not pass, so nothing here is used: {rec}")
    return rec


# ----------------------------------------------------------------------------- the type temperatures

def type_index(frame):
    """0 offense player, 1 defense player, 2 none, per candidate row."""
    import numpy as np
    none = frame["is_none"].astype(int).to_numpy() == 1
    off = frame["is_offense"].astype(int).to_numpy() == 1
    return np.where(none, 2, np.where(off, 0, 1))


def typed_p(q, types, key, params, shared: bool = False):
    """p within each miss proportional to exp(a_g log q + b_g); params (a_off, a_def, a_none, b_def, b_none), or with
    shared=True (a, b_def, b_none)."""
    import numpy as np
    from analytics.rebounds import train as T
    from analytics.eval import EPS
    lq = np.log(np.clip(np.asarray(q, dtype=float), EPS, 1.0))
    if shared:
        a = np.full(3, params[0])
        b = np.array([0.0, params[1], params[2]])
    else:
        a = np.asarray(params[:3])
        b = np.array([0.0, params[3], params[4]])
    return T.softmax_within(a[types] * lq + b[types], key)


def fit_types(q, types, key, y, tau: float, shared: bool = False) -> dict:
    """Minimize the per-miss log loss of typed_p over its parameters (L-BFGS, numerical gradient), from a = tau, b = 0."""
    import numpy as np
    from scipy.optimize import minimize
    from analytics.eval import EPS
    y = np.asarray(y).astype(bool)
    x0 = np.array([tau, 0.0, 0.0]) if shared else np.array([tau, tau, tau, 0.0, 0.0])

    def loss(x):
        p = typed_p(q, types, key, x, shared)
        return -float(np.log(np.clip(p[y], EPS, 1.0)).mean())
    r = minimize(loss, x0, method="L-BFGS-B")
    return {"params": [float(v) for v in r.x], "inner_log_loss": float(r.fun), "start_log_loss": float(loss(x0)),
            "converged": bool(r.success)}


# ----------------------------------------------------------------------------- development

def game_bootstrap(groups, values: dict, n: int = BOOT, seed: int = SEED) -> dict:
    import numpy as np
    import pandas as pd
    codes, uniq = pd.factorize(pd.Series(groups), sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n, G))
    out = {}
    for k, v in values.items():
        s = np.bincount(codes, weights=v, minlength=G)
        b = s[idx].sum(axis=1) / cnt[idx].sum(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        out[k] = {"point": float(s.sum() / cnt.sum()), "ci95": [float(lo), float(hi)]}
    return out


def projected_half_width(groups, diff, test_games: int = 10, n: int = BOOT, seed: int = SEED) -> dict:
    """D125: the half-width a 10-game set would give, from the development per-game spread of the difference."""
    import numpy as np
    import pandas as pd
    codes, uniq = pd.factorize(pd.Series(groups), sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    s = np.bincount(codes, weights=diff, minlength=G)
    rng = np.random.default_rng(seed)
    vals = [s[p].sum() / cnt[p].sum() for p in (rng.choice(G, size=test_games, replace=False) for _ in range(n))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    idx = rng.integers(0, G, size=(n, G))
    blo, bhi = np.percentile(s[idx].sum(axis=1) / cnt[idx].sum(axis=1), [2.5, 97.5])
    return {"random_10_game_sets_half_width": float((hi - lo) / 2),
            "dev_half_width_scaled": float((bhi - blo) / 2 * math.sqrt(G / test_games)), "development_games": int(G)}


def favourite_calibration(pm) -> dict:
    """Slope (analytics.eval.calibration_slope_fast) and ECE over 10 equal-mass bins of the favourite's probability."""
    import numpy as np
    from analytics.eval import calibration_slope_fast
    y = pm["top_is_rebounder"].astype(float).to_numpy()
    p = pm["p_top"].to_numpy(dtype=float)
    order = np.argsort(p, kind="mergesort")
    ece = sum(len(i) * abs(float(p[i].mean()) - float(y[i].mean())) for i in np.array_split(order, 10)) / len(p)
    return {"slope": float(calibration_slope_fast(y, p)), "ece_equal_mass_10": float(ece)}


def team_accuracy(frame, p) -> float:
    """On misses whose rebounder is a player: the share where the side (offense or defense) with the larger summed
    probability is the rebounder's side."""
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"key": frame["sportvu_game"].astype(str) + "#" + frame["event"].astype(str),
                      "off": frame["is_offense"].astype(int).to_numpy(), "none": frame["is_none"].astype(int).to_numpy(),
                      "y": frame["is_rebounder"].astype(bool).to_numpy(), "p": p})
    players = d[d["none"] == 0]
    mass = players.groupby(["key", "off"])["p"].sum().unstack(fill_value=0.0)
    who = players[players["y"]].set_index("key")["off"]
    mass = mass.loc[mass.index.intersection(who.index)]
    pred_off = (mass.get(1, 0.0) > mass.get(0, 0.0)).astype(int)
    return float((pred_off == who.reindex(mass.index)).mean())


def dev(artifacts=None) -> dict:
    import numpy as np
    from analytics.rebounds import train as T
    head = clean_head()
    g = require_gate(artifacts)
    train = load_candidates(artifacts)
    folder = work_dir(artifacts) / "folds"
    n = len(train)
    P = {a: np.full(n, np.nan) for a in ("baseline", "e1", "ablation")}
    per_fold = []
    types_all = type_index(train)
    y_all = train["is_rebounder"].astype(int).to_numpy()
    for k in range(FOLDS):
        z = np.load(folder / f"fold_{k}.npz")
        trr, ter = z["train_rows"], z["test_rows"]
        trn, tst = train.iloc[trr].reset_index(drop=True), train.iloc[ter].reset_index(drop=True)
        key_in, key_te = T.miss_key(trn), T.miss_key(tst)
        tau = float(z["tau"])
        f5 = fit_types(z["q_inner"], types_all[trr], key_in, y_all[trr], tau)
        f3 = fit_types(z["q_inner"], types_all[trr], key_in, y_all[trr], tau, shared=True)
        P["baseline"][ter] = z["p_base"]
        P["e1"][ter] = typed_p(z["q_test"], types_all[ter], key_te, f5["params"])
        P["ablation"][ter] = typed_p(z["q_test"], types_all[ter], key_te, f3["params"], shared=True)
        per_fold.append({"fold": k, "tau": tau, "e1": f5, "ablation": f3, "code_commit": str(z["code_commit"])})
    if any(not np.isfinite(P[a]).all() for a in P):
        raise SystemExit(f"{WHO}: a candidate row lacks an out-of-fold probability")
    pm = {}
    for a in P:
        pred = train.copy()
        pred["p_" + a] = P[a]
        pm[a] = T.per_miss(pred, "p_" + a)
    keys = [(r.game, r.event) for r in pm["baseline"].itertuples(index=False)]
    for a in ("e1", "ablation"):
        if [(r.game, r.event) for r in pm[a].itertuples(index=False)] != keys:
            raise SystemExit(f"{WHO}: the per-miss rows of {a} are not paired with the baseline's")
    groups = pm["baseline"]["game"].to_numpy()
    loss = {a: -pm[a]["logp"].to_numpy() for a in pm}
    st = game_bootstrap(groups, {**{f"log_loss_{a}": loss[a] for a in loss}, **{f"top1_{a}": pm[a]["hit"].to_numpy() for a in pm},
                                 "e1_minus_baseline": loss["e1"] - loss["baseline"],
                                 "ablation_minus_baseline": loss["ablation"] - loss["baseline"],
                                 "e1_minus_ablation": loss["e1"] - loss["ablation"]})
    d = st["e1_minus_baseline"]
    out = {"experiment": EXPERIMENT, "code_commit": head,
           "stage": "development games, out of fold under the frozen 5 grouped folds by game (seed 0); no Set A game "
                    "read; no held-out row fitted or scored",
           "gate_a": g, "misses": int(len(groups)), "games": int(len(set(groups))), "candidate_rows": int(n),
           "per_fold": per_fold, "log_loss": {a: st[f"log_loss_{a}"] for a in loss},
           "top1": {a: st[f"top1_{a}"] for a in pm},
           "favourite_calibration": {a: favourite_calibration(pm[a]) for a in pm},
           "team_accuracy_player_rebounds": {a: team_accuracy(train, P[a]) for a in P},
           "e1_minus_baseline": d, "ablation_minus_baseline": st["ablation_minus_baseline"],
           "e1_minus_ablation": st["e1_minus_ablation"],
           "projected_set_a": projected_half_width(groups, loss["e1"] - loss["baseline"]),
           "verdict": {"rule": ("D125: E1 minus baseline per-miss log loss at most -0.005 with its 95 percent interval "
                                "excluding zero, out of fold on the development games"),
                       "at_most_threshold": bool(d["point"] <= THRESHOLD),
                       "interval_excludes_zero": bool(d["ci95"][1] < 0 or d["ci95"][0] > 0),
                       "passes": bool(d["point"] <= THRESHOLD and (d["ci95"][1] < 0 or d["ci95"][0] > 0))},
           "peak_gb": round(peak_rss_gb(), 2)}
    bad = [k for k, v in _walk(out) if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        raise SystemExit(f"{WHO}: non-finite value(s), for example {bad[0]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def _walk(v, path=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _walk(x, f"{path}.{k}")
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            yield from _walk(x, f"{path}[{i}]")
    else:
        yield path, (v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes)) else v)


def clean_head() -> str:
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty[:300]}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-build", "e1-fit", "e1-dev"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=26.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research rebounds {a.command}"):
        head = clean_head()
        if a.command == "e1-build":
            out = {"code_commit": head, **build(a.artifacts)}
        elif a.command == "e1-fit":
            out = {"code_commit": head, **fit_pieces(a.artifacts, a.max_minutes, head=head)}
        else:
            out = dev(a.artifacts)
    print(json.dumps(_plain(out.get("verdict", out)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""tools/research_gravity.py: the gravity research experiment gravity-E1 (PLAN.md D106, D107).

    python3 -m tools.research_gravity e1-pass  [--artifacts DIR] [--max-minutes 25]
    python3 -m tools.research_gravity e1-gate  [--artifacts DIR]
    python3 -m tools.research_gravity e1-dev   [--artifacts DIR]
    python3 -m tools.research_gravity e1-final [--artifacts DIR]

Pre-registration: docs/research/gravity.md, "gravity-E1", with its amendment (D107), which fixes every definition
used here before any number existed. The question: is a shot more open (a larger nearest-defender distance at the
release) when the shooter's on-floor teammates have more gravity, beyond the shot's location and the shooter?

e1-pass   One pass over the development games' frames, one game at a time, resumable (--max-minutes keeps a job
          inside the lock budget): the frozen gravity lane's validate_game and build_game (unchanged) give the game's
          1 Hz player rows; the cleaned frames at each attempt's release frame give the shooter's team and the players
          on the floor. Development games only (tools.research_testset.dev_games); held-out games are never read.
e1-gate   The frozen ridge expectation out of fold under the lane's own folds, rebuilt from those rows with the frozen
          functions, must reproduce the stored gravity_frames (training rows), the stored gravity_players means and
          the printed ridge MSE. Writes the gravity frames and the teammates table with their hashes.
e1-dev    The development decision (D107): the out-of-fold MSE difference, candidate minus baseline, and the
          cross-fitted effect per SD of teammates' gravity sum, with what is reported beside. Writes
          reports/research/gravity-E1-dev.json.
e1-final  Set A once through tools.research_testset.score_once, as D107's consistency check. Runs only on the lead's
          go.

Nothing here writes under DATA_ROOT or reports/ except the research results; the lane's command line is never run.
The fits are ridges and a three-parameter logistic on development attempts (rule 5: tools/, training games only).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, _plain, _sha256, artifacts_dir, assert_safe_output, training_games_only
from tools.research_testset import TRACKING, dev_games, refuse_test, score_once

WHO = "tools.research_gravity"
EXPERIMENT = "gravity-E1"
RESULTS = REPO_ROOT / "reports" / "research"
LANE_REPORT = REPO_ROOT / "reports" / "gravity.md"
SEASON = "2015-16"
SEED, BOOT, FOLDS = 0, 2000, 5
RIDGE_ALPHA = 1.0
EFFECT_MIN = 0.2                 # ft of openness per SD of teammates' gravity sum (pre-registered)
EXPECTED_TOL, PLAYER_TOL = 1e-4, 1e-6
SHOTS_FILE = "collect/bref_shots/shotchart/2015-16.parquet"
XFG_FEATURES = "xfg_features_train.parquet"
WORK = "gravity_e1"
CHECK = "check.json"
LOC = ["distance_ft", "angle_rad", "is_three", "is_corner"]
SKILL_SHOT_COLUMNS = ["sportvu_game", "made", "loc_x", "loc_y", "shot_distance_ft", "heldout", "split"]
TEAMMATES = 4


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / WORK)


def _frozen():
    from analytics.gravity import features as GF
    from analytics.gravity import train as GT
    from analytics.gravity import validate as GV
    if GT.RIDGE_ALPHA != RIDGE_ALPHA or GT.SHIPPED != "ridge":
        raise RuntimeError(f"{WHO}: the frozen gravity expectation changed")
    return GF, GT, GV


# ----------------------------------------------------------------------------- development inputs

def dev_game_list(root: Path) -> list:
    from analytics.eval import is_heldout_game
    poss = {p.stem for p in (root / "analytics" / "possessions" / SEASON).glob("*.parquet")}
    frames = {p.stem for p in (root / "sportvu_contract").glob("*.parquet") if not p.name.endswith("_pbp.parquet")}
    games = dev_games(sorted(g for g in poss & frames if not is_heldout_game(g)), WHO)
    training_games_only(games, WHO)
    return games


def game_dict(root: Path, g: str) -> dict:
    """A development game as the frozen loader builds it (validate.load_data_root), without the season shot file,
    which validate_game and build_game do not change."""
    import pandas as pd
    _, _, GV = _frozen()
    sub = SEASON
    return {"sportvu_game": g, "heldout": False, "frames": GV._frames_loader(root / "sportvu_contract" / f"{g}.parquet"),
            "possessions": pd.read_parquet(root / "analytics" / "possessions" / sub / f"{g}.parquet"),
            "events": GV._read_if(root / "sportvu_contract" / f"{g}_pbp.parquet"),
            "rosters": GV._read_if(root / "analytics" / "rosters" / sub / f"{g}.parquet"),
            "shot_frames": GV._read_if(root / "analytics" / "shot_frames" / sub / f"{g}.parquet"),
            "rebound_frames": None, "states_1hz": None, "shots": None}


def xfg_frame(artifacts=None):
    """The frozen xFG feature frame of the training rows (tools.research_xfg, frozen code), checked against the
    stored xFG predictions' training rows."""
    import os
    import pandas as pd
    path = artifacts_dir(artifacts) / "research" / XFG_FEATURES
    f = pd.read_parquet(path)
    stored = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "analytics" / "xfg" / SEASON / "xfg_predictions.parquet",
                             columns=["sportvu_game", "event", "player_id", "fold"])
    stored = stored[stored["fold"] >= 0]
    key = lambda d: sorted(zip(d["sportvu_game"].astype(str), d["event"].astype(int), d["player_id"].astype("Int64")))
    if len(f) != len(stored) or key(f) != key(stored):
        raise SystemExit(f"{WHO}: the xFG feature frame does not match the stored xFG training rows")
    training_games_only(f["sportvu_game"].astype(str).unique(), WHO)
    return f, _sha256(path)


def release_rows(frames, shots) -> list:
    """For each attempt (event, shooter player_id, release_frame), the shooter's team and the players on the floor at
    the release frame from the cleaned frames -> rows with the shooter's teammates (commitment 3)."""
    import numpy as np
    fr = frames[["frame", "team_id", "player_id"]]
    by = {f: g for f, g in fr[fr["frame"].isin(set(shots["release_frame"].astype(int)))].groupby("frame")}
    out = []
    for r in shots.itertuples(index=False):
        g = by.get(int(r.release_frame))
        row = {"sportvu_game": r.sportvu_game, "event": int(r.event), "shooter": int(r.player_id), "status": "ok",
               "teammates": []}
        if g is None:
            row["status"] = "no_frame"
        else:
            me = g[g["player_id"] == int(r.player_id)]
            if not len(me):
                row["status"] = "shooter_not_in_frame"
            else:
                team = me["team_id"].iloc[0]
                mates = sorted(int(p) for p in g.loc[(g["team_id"] == team) & (g["player_id"] != int(r.player_id)),
                                                      "player_id"])
                row["teammates"] = mates
                if len(mates) != TEAMMATES:
                    row["status"] = "not_four_teammates"
        out.append(row)
    return out


def _release_frame_table(rows):
    import pandas as pd
    t = pd.DataFrame(rows)
    for i in range(TEAMMATES):
        t[f"tm{i + 1}"] = [r[i] if len(r) == TEAMMATES else -1 for r in t["teammates"]]
    return t.drop(columns=["teammates"])


def frame_pass(artifacts=None, max_minutes: float = 25.0, log=print) -> dict:
    """The frozen per-game gravity rows and the release-frame teammates of every development game; resumable."""
    import os
    import pandas as pd
    GF, _, GV = _frozen()
    root = Path(os.environ["DATA_ROOT"])
    games = dev_game_list(root)
    shots, _ = xfg_frame(artifacts)
    shots = shots[shots["has_contest"].astype(bool) & shots["release_frame"].notna() & shots["player_id"].notna()]
    out = work_dir(artifacts) / "games"
    out.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    for g in games:
        if (out / f"{g}.rows.parquet").exists() and (out / f"{g}.release.parquet").exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        cleaned, _ = GV.validate_game(game_dict(root, g))
        pf, _, _ = GF.build_game(cleaned)
        rel = _release_frame_table(release_rows(cleaned["frames"], shots[shots["sportvu_game"].astype(str) == g])
                                   or [{"sportvu_game": g, "event": -1, "shooter": -1, "status": "none", "teammates": []}])
        del cleaned
        pf.to_parquet(out / f"{g}.rows.parquet.part", index=False)
        (out / f"{g}.rows.parquet.part").rename(out / f"{g}.rows.parquet")
        rel.to_parquet(out / f"{g}.release.parquet.part", index=False)
        (out / f"{g}.release.parquet.part").rename(out / f"{g}.release.parquet")
        done += 1
        if log and done % 100 == 0:
            log(f"[research] {done} games this run, {time.time() - started:.0f} s")
    have = sum((out / f"{g}.rows.parquet").exists() for g in games)
    return {"games": len(games), "written_this_run": done, "present": have, "complete": have == len(games),
            "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2)}


# ----------------------------------------------------------------------------- the gate

def skill_shots(root: Path, games: list):
    """The season shot rows of the development games (the zone prior's input), read with a filter so no other game's
    row is loaded."""
    import pandas as pd
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(root / SHOTS_FILE).schema.names)
    s = pd.read_parquet(root / SHOTS_FILE, columns=[c for c in SKILL_SHOT_COLUMNS if c in names],
                        filters=[("sportvu_game", "in", list(games))])
    if not set(s["sportvu_game"].astype(str)) <= set(games):
        raise SystemExit(f"{WHO}: a shot row outside the development games was read")
    return s


def oof_ridge(pf, shots, n_folds: int = FOLDS, seed: int = SEED):
    """The frozen out_of_fold of analytics.gravity.train with the shipped ridge only (the same folds, zone prior
    and fit_models), so expected_ft is the frozen lane's; the HGB diagnostic is not refit."""
    import numpy as np
    from analytics.eval import grouped_folds
    GF, GT, _ = _frozen()
    ids = sorted(pf["sportvu_game"].unique())
    pred = pf.copy()
    pred["fold"] = -1
    pred["p_ridge"] = np.nan
    sg = shots["sportvu_game"].astype(str)
    for fold, (tr_ids, te_ids) in enumerate(grouped_folds(ids, n_folds=min(n_folds, len(ids)), seed=seed)):
        tr, te = pred["sportvu_game"].isin(tr_ids), pred["sportvu_game"].isin(te_ids)
        table = GF.skill_table(shots[sg.isin(set(tr_ids))])
        model = GT.fit_models(GF.attach_skill(pred[tr], table), seed, which=("ridge",))["ridge"]
        pred.loc[te, "p_ridge"] = model(GF.attach_skill(pred[te], table))
        pred.loc[te, "fold"] = fold
    return pred


def printed_ridge_mse(text: str) -> str:
    import re
    m = re.search(r"\| ridge \| (\d+\.\d+) ", text)
    if not m:
        raise SystemExit(f"{WHO}: the frozen report does not print the ridge MSE")
    return m.group(1)


def _record(folder: Path, record: dict, files) -> None:
    record = _plain(dict(record))
    record["files"] = {f: _sha256(folder / f) for f in files}
    record["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (folder / CHECK).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def require_gate(artifacts=None) -> dict:
    folder = work_dir(artifacts)
    path = folder / CHECK
    if not path.exists():
        raise RuntimeError(f"{EXPERIMENT}: no gate record (run e1-gate first)")
    record = json.loads(path.read_text())
    if record.get("passed") is not True or not record.get("files"):
        raise RuntimeError(f"{EXPERIMENT}: the rebuild did not reproduce the frozen gravity tables, so it is not used")
    for name, digest in record["files"].items():
        if not (folder / name).exists() or _sha256(folder / name) != digest:
            raise RuntimeError(f"{EXPERIMENT}: {name} changed after the gate record was written")
    return record


def gate(artifacts=None) -> dict:
    import os
    import numpy as np
    import pandas as pd
    GF, GT, _ = _frozen()
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts)
    (folder / CHECK).write_text(json.dumps({"passed": False, "status": "rebuild started"}) + "\n")
    games = dev_game_list(root)
    gdir = folder / "games"
    missing = [g for g in games if not (gdir / f"{g}.rows.parquet").exists()]
    if missing:
        raise SystemExit(f"{WHO}: {len(missing)} development games are not built yet; run e1-pass")
    pf, _, _ = GF.combine([(pd.read_parquet(gdir / f"{g}.rows.parquet"), pd.DataFrame(), {}) for g in games])
    pred = oof_ridge(pf, skill_shots(root, games))
    del pf
    mse_ridge = float(np.mean((pred[GF.TARGET].to_numpy(float) - pred["p_ridge"].to_numpy(float)) ** 2))
    gf = GT.gravity_frames(pred, "ridge")
    del pred
    stored = pd.read_parquet(root / "analytics" / "gravity" / "gravity_frames.parquet")
    stored = stored[~stored["heldout"].astype(bool)].reset_index(drop=True)
    keys = ["sportvu_game", "possession", "frame", "player_id"]
    a = gf.sort_values(keys, kind="stable").reset_index(drop=True)
    b = stored.sort_values(keys, kind="stable").reset_index(drop=True)
    same_keys = len(a) == len(b) and all((a[k].to_numpy() == b[k].to_numpy()).all() for k in keys)
    exp_diff = float(np.max(np.abs(a["expected_ft"].to_numpy(float) - b["expected_ft"].to_numpy(float)))) if same_keys else None
    fold_same = bool(same_keys and (a["fold"].to_numpy() == b["fold"].to_numpy()).all())
    del a, b, stored
    players = GT.gravity_players(gf, n_boot=0)
    sp = pd.read_parquet(root / "analytics" / "gravity" / "gravity_players.parquet").set_index("player_id")
    same_players = set(players["player_id"].astype(int)) == set(sp.index.astype(int))
    pl = players.set_index("player_id").loc[sp.index] if same_players else None
    player_diff = float(np.max(np.abs(pl["gravity_ft"].to_numpy(float) - sp["gravity_ft"].to_numpy(float)))) \
        if same_players else float("inf")
    printed = printed_ridge_mse(LANE_REPORT.read_text())
    mse_ok = abs(mse_ridge - float(printed)) <= 0.5 * 10 ** -len(printed.split(".")[1])
    gf[["sportvu_game", "player_id", "gravity_ft", "nearest_def_dist_ft", "fold"]].to_parquet(
        folder / "gravity_frames.parquet", index=False)
    rel = pd.concat([pd.read_parquet(gdir / f"{g}.release.parquet") for g in games], ignore_index=True)
    rel = rel[rel["event"] >= 0]
    rel.to_parquet(folder / "release_teammates.parquet", index=False)
    passed = bool(same_keys and exp_diff is not None and exp_diff <= EXPECTED_TOL and fold_same
                  and len(players) == len(sp) and player_diff <= PLAYER_TOL and mse_ok)
    record = {"experiment": EXPERIMENT, "name": "gravity",
              "compared": "the frozen ridge expectation out of fold, rebuilt with the frozen functions on the "
                          "development games, against the stored gravity_frames (training rows) and gravity_players; "
                          "the ridge out-of-fold MSE against reports/gravity.md",
              "games": len(games), "rows": int(len(gf)), "keys_equal": same_keys, "folds_equal": fold_same,
              "expected_ft_max_abs_diff": exp_diff, "players": int(len(players)), "players_expected": int(len(sp)),
              "player_gravity_max_abs_diff": player_diff,
              "ridge_mse": {"recomputed": mse_ridge, "printed": printed, "ok": mse_ok},
              "tolerances": {"expected_ft": EXPECTED_TOL, "player_gravity_ft": PLAYER_TOL},
              "release_rows": int(len(rel)), "release_status": rel["status"].value_counts().to_dict(),
              "passed": passed}
    _record(folder, record, ["gravity_frames.parquet", "release_teammates.parquet"])
    if not passed:
        print(json.dumps(_plain(record), indent=2, default=str))
        raise SystemExit(f"{WHO}: the rebuild did not reproduce the frozen gravity tables; nothing downstream runs")
    return require_gate(artifacts)


# ----------------------------------------------------------------------------- the attempts and their teammates

def player_game_sums(gf):
    """Per (player, game): sums and counts of gravity_ft and of nearest_def_dist_ft, with the game's fold."""
    return gf.groupby(["player_id", "sportvu_game"]).agg(g_sum=("gravity_ft", "sum"), n=("gravity_ft", "size"),
                                                          d_sum=("nearest_def_dist_ft", "sum"),
                                                          fold=("fold", "first")).reset_index()


def player_means(pgs, exclude_fold=None) -> dict:
    """{player_id: (mean gravity, mean raw nearest-defender distance)} over the games outside exclude_fold."""
    d = pgs if exclude_fold is None else pgs[pgs["fold"] != exclude_fold]
    s = d.groupby("player_id")[["g_sum", "n", "d_sum"]].sum()
    return {int(p): (r.g_sum / r.n, r.d_sum / r.n) for p, r in s.iterrows() if r.n > 0}


def attach_teammates(att, rel, means_of_fold) -> tuple:
    """The attempts with teammates' gravity sum and maximum and the raw-distance ablation's sum and maximum; drops
    and counts every attempt that commitment 3 or 4 excludes. means_of_fold(fold) -> player means."""
    import numpy as np
    import pandas as pd
    m = att.merge(rel, left_on=["sportvu_game", "event", "player_id"], right_on=["sportvu_game", "event", "shooter"],
                  how="left")
    counts = {"attempts": int(len(att)), "no_release_row": int(m["status"].isna().sum())}
    m = m[m["status"].notna()]
    for s, n in m["status"].value_counts().items():
        if s != "ok":
            counts[str(s)] = int(n)
    m = m[m["status"] == "ok"].copy()
    cache, cols = {}, {k: [] for k in ("g_sum", "g_max", "d_sum", "d_max", "keep")}
    for r in m.itertuples(index=False):
        f = int(r.fold)
        if f not in cache:
            cache[f] = means_of_fold(f)
        means = cache[f]
        tm = [int(getattr(r, f"tm{i + 1}")) for i in range(TEAMMATES)]
        vals = [means.get(p) for p in tm]
        if any(v is None for v in vals):
            cols["keep"].append(False)
            for k in ("g_sum", "g_max", "d_sum", "d_max"):
                cols[k].append(np.nan)
            continue
        g, d = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        cols["keep"].append(True)
        cols["g_sum"].append(g.sum()); cols["g_max"].append(g.max())
        cols["d_sum"].append(d.sum()); cols["d_max"].append(d.max())
    for k, v in cols.items():
        m[k] = v
    counts["teammate_without_gravity"] = int((~m["keep"]).sum())
    m = m[m["keep"]].drop(columns=["keep"]).reset_index(drop=True)
    counts["used"] = int(len(m))
    return m, counts


# ----------------------------------------------------------------------------- the models

class Design:
    """Standardised location features plus a shooter indicator per shooter of the training rows, and optional extra
    columns standardised on the training rows (commitment 5)."""

    def __init__(self, train, extra=()):
        import numpy as np
        self.extra = list(extra)
        cols = LOC + self.extra
        X = train[cols].to_numpy(float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.shooters = {int(p): i for i, p in enumerate(sorted(train["player_id"].astype(int).unique()))}

    def __call__(self, frame):
        import numpy as np
        from scipy import sparse
        Xn = (frame[LOC + self.extra].to_numpy(float) - self.mu) / self.sd
        idx = np.array([self.shooters.get(int(p), -1) for p in frame["player_id"]])
        rows = np.flatnonzero(idx >= 0)
        S = sparse.csr_matrix((np.ones(len(rows)), (rows, idx[rows])), shape=(len(frame), len(self.shooters)))
        return sparse.hstack([sparse.csr_matrix(Xn), S]).tocsr()


def ridge_predict(train, test, target: str, extra=()):
    from sklearn.linear_model import Ridge
    D = Design(train, extra)
    m = Ridge(alpha=RIDGE_ALPHA).fit(D(train), train[target].to_numpy(float))
    return m.predict(D(test))


def logistic_offset(train_off, train_X, train_y, test_off, test_X):
    """Unpenalised logistic with an offset and an intercept, by maximum likelihood (scipy); returns test p."""
    import numpy as np
    from scipy.optimize import minimize
    A = np.column_stack([np.ones(len(train_X)), train_X])

    def nll(b):
        z = train_off + A @ b
        return float(np.sum(np.logaddexp(0, z) - train_y * z))

    def grad(b):
        z = train_off + A @ b
        return A.T @ (1 / (1 + np.exp(-z)) - train_y)

    b = minimize(nll, np.zeros(A.shape[1]), jac=grad, method="BFGS").x
    z = test_off + np.column_stack([np.ones(len(test_X)), test_X]) @ b
    return 1 / (1 + np.exp(-z))


def logloss(y, p):
    import numpy as np
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def slope(y_res, X_res, w=None):
    """Least-squares coefficients of y_res on [1, X_res] (weights w: resample counts) -> the X coefficients."""
    import numpy as np
    A = np.column_stack([np.ones(len(y_res)), X_res])
    if w is not None:
        A, y_res = A * np.sqrt(w)[:, None], y_res * np.sqrt(w)
    return np.linalg.lstsq(A, y_res, rcond=None)[0][1:]


def game_bootstrap(games, per_shot: dict, n: int = BOOT, seed=SEED) -> dict:
    """Shot-weighted means of per-shot values with percentile intervals from resampling games (paired: every value
    uses the same draws)."""
    import numpy as np
    g = np.asarray(games)
    uniq, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n, len(uniq)))
    for b in range(n):
        np.add.at(counts[b], rng.integers(0, len(uniq), len(uniq)), 1)
    w = counts[:, inv]
    out = {}
    for k, v in per_shot.items():
        v = np.asarray(v, float)
        means = (w @ v) / w.sum(axis=1)
        out[k] = {"value": float(v.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}
    return {"games": int(len(uniq)), "shots": int(len(g)), "resamples": n, "intervals": out}


def effect_bootstrap(games, y_res, X_res, n: int = BOOT, seed=SEED, names=("sum", "max")) -> dict:
    """The effect slopes (by default gravity sum and gravity maximum) with percentile intervals from resampling
    games."""
    import numpy as np
    g = np.asarray(games)
    uniq, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    point = slope(y_res, X_res)
    draws = []
    for _ in range(n):
        c = np.zeros(len(uniq))
        np.add.at(c, rng.integers(0, len(uniq), len(uniq)), 1)
        draws.append(slope(y_res, X_res, c[inv]))
    draws = np.array(draws)
    return {name: {"value": float(point[i]), "lo": float(np.percentile(draws[:, i], 2.5)),
                   "hi": float(np.percentile(draws[:, i], 97.5))} for i, name in enumerate(names)}


def per_game_t(values) -> dict:
    """Report-only (D107, as D80): a 95 percent t interval on per-game values (one per game)."""
    import numpy as np
    from scipy import stats
    v = np.asarray(values, float)
    if len(v) < 2:
        return {"mean_of_game_values": float(v.mean()) if len(v) else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "games": int(len(v))}
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean_of_game_values": float(v.mean()), "lo": float(v.mean() - half), "hi": float(v.mean() + half),
            "games": int(len(v))}


# ----------------------------------------------------------------------------- development

def load_dev(artifacts=None) -> dict:
    """The development attempts with their teammates' out-of-fold gravity, after the gate."""
    import numpy as np
    import pandas as pd
    require_gate(artifacts)
    folder = work_dir(artifacts)
    gf = pd.read_parquet(folder / "gravity_frames.parquet")
    rel = pd.read_parquet(folder / "release_teammates.parquet")
    games = sorted(gf["sportvu_game"].astype(str).unique())
    training_games_only(games, WHO)
    refuse_test(games, WHO)
    shots, xfg_sha = xfg_frame(artifacts)
    fold_of = gf.drop_duplicates("sportvu_game").set_index("sportvu_game")["fold"].astype(int).to_dict()
    att = shots[shots["has_contest"].astype(bool) & shots["release_frame"].notna() & shots["player_id"].notna()
                & shots["sportvu_game"].astype(str).isin(set(games))].copy()
    att["sportvu_game"] = att["sportvu_game"].astype(str)
    att["fold"] = att["sportvu_game"].map(fold_of).astype(int)
    pgs = player_game_sums(gf)
    data, counts = attach_teammates(att, rel, lambda f: player_means(pgs, f))
    counts["shots_in_feature_frame"] = int(len(shots))
    counts["shots_outside_gravity_games"] = int((~shots["sportvu_game"].astype(str).isin(set(games))).sum())
    sd = {"g_sum": float(data["g_sum"].std()), "g_max": float(data["g_max"].std()),
          "d_sum": float(data["d_sum"].std()), "d_max": float(data["d_max"].std())}
    mean = {k: float(data[k].mean()) for k in sd}
    for k in sd:
        data["z_" + k] = (data[k] - mean[k]) / sd[k]
    return {"data": data, "counts": counts, "sd": sd, "mean": mean, "pgs": pgs, "xfg_sha256": xfg_sha,
            "games": games}


def oof_models(data) -> dict:
    """Out-of-fold predictions: baseline, candidate (gravity), ablation (raw distance), the nuisance ridges of the
    cross-fitted effect, and the offset logistic beside."""
    import numpy as np
    from analytics import train_xfg as TX
    y = data["closest_defender_dist_ft"].to_numpy(float)
    out = {k: np.full(len(data), np.nan) for k in ("base", "cand", "abl", "zg_sum", "zg_max", "zd_sum", "zd_max",
                                                   "p_loc", "p_off")}
    for f in sorted(data["fold"].unique()):
        tr, te = data[data["fold"] != f], data[data["fold"] == f]
        m = (data["fold"] == f).to_numpy()
        out["base"][m] = ridge_predict(tr, te, "closest_defender_dist_ft")
        out["cand"][m] = ridge_predict(tr, te, "closest_defender_dist_ft", ("z_g_sum", "z_g_max"))
        out["abl"][m] = ridge_predict(tr, te, "closest_defender_dist_ft", ("z_d_sum", "z_d_max"))
        for k in ("g_sum", "g_max", "d_sum", "d_max"):
            out["z" + k[0] + "_" + k[2:]][m] = ridge_predict(tr, te, "z_" + k)
        trf, tef = tr.assign(heldout=False), te.assign(heldout=False)
        p_loc = TX.fit_models(trf)["logistic"](tef)
        p_loc_tr = np.full(len(tr), np.nan)
        for f2 in sorted(tr["fold"].unique()):                       # an out-of-fold offset on the training side too
            a, b = tr[tr["fold"] != f2].assign(heldout=False), tr[tr["fold"] == f2].assign(heldout=False)
            p_loc_tr[(tr["fold"] == f2).to_numpy()] = TX.fit_models(a)["logistic"](b)
        logit = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
        out["p_loc"][m] = p_loc
        out["p_off"][m] = logistic_offset(logit(p_loc_tr), tr[["z_g_sum", "z_g_max"]].to_numpy(float),
                                          tr["made"].astype(int).to_numpy(), logit(p_loc),
                                          te[["z_g_sum", "z_g_max"]].to_numpy(float))
    out["y"] = y
    return out


def dev(artifacts=None) -> dict:
    import numpy as np
    d = load_dev(artifacts)
    data = d["data"]
    o = oof_models(data)
    y, games = o["y"], data["sportvu_game"].to_numpy()
    sq = {k: (y - o[k]) ** 2 for k in ("base", "cand", "abl")}
    made = data["made"].astype(int).to_numpy()
    per_shot = {"mse_baseline": sq["base"], "mse_candidate": sq["cand"], "mse_ablation": sq["abl"],
                "candidate_minus_baseline": sq["cand"] - sq["base"], "ablation_minus_baseline": sq["abl"] - sq["base"],
                "logloss_offset_alone": logloss(made, o["p_loc"]), "logloss_offset_plus_gravity": logloss(made, o["p_off"]),
                "logloss_change": logloss(made, o["p_off"]) - logloss(made, o["p_loc"])}
    boot = game_bootstrap(games, per_shot, BOOT, [SEED, 1])
    y_res = y - o["base"]
    eff = effect_bootstrap(games, y_res, np.column_stack([data["z_g_sum"] - o["zg_sum"], data["z_g_max"] - o["zg_max"]]),
                           BOOT, [SEED, 2])
    eff_abl = effect_bootstrap(games, y_res, np.column_stack([data["z_d_sum"] - o["zd_sum"],
                                                              data["z_d_max"] - o["zd_max"]]), BOOT, [SEED, 3])
    diff = boot["intervals"]["candidate_minus_baseline"]
    decision = {"mse_interval_below_zero": bool(diff["hi"] < 0), "effect_per_sd": eff["sum"]["value"],
                "effect_at_least": EFFECT_MIN, "effect_ok": bool(eff["sum"]["value"] >= EFFECT_MIN)}
    decision["adopted"] = bool(decision["mse_interval_below_zero"] and decision["effect_ok"])
    out = {"experiment": EXPERIMENT, "stage": "development games, 5 grouped folds by game (the frozen lane's), seed 0",
           "games": len(d["games"]), "counts": d["counts"], "xfg_features_sha256": d["xfg_sha256"],
           "gravity_sum_sd_ft": d["sd"]["g_sum"], "standardisation": {"mean": d["mean"], "sd": d["sd"]},
           "bootstrap": boot, "effect_per_sd_gravity": eff, "effect_per_sd_ablation_raw_distance": eff_abl,
           "decision": decision}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return {"decision": decision, "mse_difference": diff}


def posthoc(artifacts=None) -> dict:
    """Post hoc, after the development result and its review; no rule reads it. The sum-only cross-fitted slope:
    the release-distance residual on the gravity-sum residual alone (the maximum not held fixed), with the same
    out-of-fold nuisance ridges on the same development folds. It answers whether a total effect of the sum could
    reach 0.2 ft per SD, which the committed estimand (sum with the maximum held fixed) does not. Development data
    only; bootstrap by game, 2000 resamples, seed sequence [0, 4]."""
    import numpy as np
    d = load_dev(artifacts)
    data = d["data"]
    o = oof_models(data)
    eff = effect_bootstrap(data["sportvu_game"].to_numpy(), o["y"] - o["base"],
                           (data["z_g_sum"] - o["zg_sum"]).to_numpy()[:, None], BOOT, [SEED, 4], names=("sum_only",))
    out = {"experiment": EXPERIMENT, "stage": "post hoc, development games only, after the development result and its "
                                              "review; read by no rule",
           "sum_only_effect_per_sd": eff["sum_only"], "seed_sequence": [SEED, 4], "resamples": BOOT,
           "attempts": int(len(data)), "games": int(data["sportvu_game"].nunique())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-posthoc.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


# ----------------------------------------------------------------------------- Set A (D107)

def reading(dev_effect: float, lo: float, hi: float) -> str:
    """D107's reading of the test effect interval, in the fixed order (amendment)."""
    if not (lo <= dev_effect <= hi):
        return "inconsistent"
    if (dev_effect > 0 and lo > 0) or (dev_effect < 0 and hi < 0):
        return "confirms"
    return "consistent"


def non_finite(value, path: str = "") -> list:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in non_finite(v, f"{path}.{k}" if path else str(k))]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in non_finite(v, f"{path}[{i}]")]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def refused(reading_text: str, **known) -> dict:
    keep = {k: v for k, v in known.items() if not non_finite(v)}
    return {"verdict": {"refused": True, "reading": reading_text}, **keep}


def test_summary(test, dev_data, dev_effect: float, n: int = BOOT) -> dict:
    """D107's consistency check on attempts `test` (with z features on the development scale): every model fit on
    all development attempts."""
    import numpy as np
    y = test["closest_defender_dist_ft"].to_numpy(float)
    base = ridge_predict(dev_data, test, "closest_defender_dist_ft")
    cand = ridge_predict(dev_data, test, "closest_defender_dist_ft", ("z_g_sum", "z_g_max"))
    nz = {k: ridge_predict(dev_data, test, "z_" + k) for k in ("g_sum", "g_max")}
    games = test["sportvu_game"].to_numpy()
    d = (y - cand) ** 2 - (y - base) ** 2
    boot = game_bootstrap(games, {"candidate_minus_baseline": d}, n, [SEED, 21])
    y_res = y - base
    X_res = np.column_stack([test["z_g_sum"] - nz["g_sum"], test["z_g_max"] - nz["g_max"]])
    eff = effect_bootstrap(games, y_res, X_res, n, [SEED, 22])
    per_game_d, per_game_slope = [], []
    for g in np.unique(games):
        m = games == g
        per_game_d.append(float(d[m].mean()))
        if m.sum() >= 4:
            per_game_slope.append(float(slope(y_res[m], X_res[m])[0]))
    e = eff["sum"]
    return {"games": int(len(np.unique(games))), "attempts": int(len(test)),
            "mse_candidate_minus_baseline": boot["intervals"]["candidate_minus_baseline"],
            "effect_per_sd_gravity": eff, "development_effect": dev_effect,
            "reading": reading(dev_effect, e["lo"], e["hi"]),
            "pre_score_sensitivity_t": {"mse_difference": per_game_t(per_game_d),
                                        "effect_per_sd_sum": per_game_t(per_game_slope)}}


def clean_head() -> str:
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def make_scorer(d: dict, dev_effect: float, artifacts=None, provenance: dict = None):
    """The scorer score_once runs (D107). The test data is reached only here, while score_once runs a tracking
    scorer. Before any value, it refuses unless every test game has frames and at least one used attempt; a summary
    with no attempt or any non-finite number is a refused verdict, never a NaN."""
    import pandas as pd
    _, _, GV = _frozen()
    from tools import research_testset as RT
    from tools.research_xfg import research_data_root, test_features

    def scorer(test_games):
        if RT._SCORING is None or RT._SCORING[1] != TRACKING:
            raise SystemExit(f"{WHO}: the test data is read only inside the scorer that score_once runs")
        head = {"experiment": EXPERIMENT, "stage": "D107 consistency check, scored once on the D71 test set",
                **(provenance or {})}
        root = research_data_root(artifacts)
        games = sorted(str(g) for g in test_games)
        no_frames = [g for g in games if not (root / "sportvu_contract" / f"{g}.parquet").exists()]
        if no_frames:
            return {**head, **refused(f"{len(no_frames)} test game(s) have no frames", games_without_frames=no_frames)}
        try:
            feats = test_features(games, artifacts)
        except SystemExit as exc:
            return {**head, **refused(f"the frozen xFG features could not be built for every test game: {exc}")}
        att = feats[feats["has_contest"].astype(bool) & feats["release_frame"].notna() & feats["player_id"].notna()].copy()
        att["sportvu_game"] = att["sportvu_game"].astype(str)
        rows = []
        for g in games:
            frames, _, _ = GV.clean_frames(pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet"))
            rows += release_rows(frames, att[att["sportvu_game"] == g])
            del frames
        rel = _release_frame_table(rows) if rows else None
        if rel is None:
            return {**head, **refused("no test attempt has a release-frame defender distance")}
        att["fold"] = -1
        means = player_means(d["pgs"])
        test, counts = attach_teammates(att, rel, lambda f: means)
        for k in d["sd"]:
            test["z_" + k] = (test[k] - d["mean"][k]) / d["sd"][k]
        used = set(test["sportvu_game"])
        empty = [g for g in games if g not in used]
        if empty:
            return {**head, **refused(f"{len(empty)} test game(s) have no used attempt", counts=counts,
                                      games_without_attempts=empty)}
        summary = {"counts": counts, **test_summary(test, d["data"], dev_effect)}
        bad = non_finite(summary)
        if bad:
            return {**head, **refused(f"{len(bad)} non-finite value(s) in the summary, for example {bad[0]}",
                                      counts=counts)}
        return {**head, **summary}

    return scorer


def final(artifacts=None) -> dict:
    head = clean_head()
    from tools.research_xfg import test_root_ready
    gate_record = require_gate(artifacts)
    dev_path = RESULTS / f"{EXPERIMENT}-dev.json"
    if not dev_path.exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test set is scored after the development result exists")
    dev_out = json.loads(dev_path.read_text())
    d = load_dev(artifacts)
    if not test_root_ready(artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games; run tools.research_ingest first")
    dev_effect = float(dev_out["effect_per_sd_gravity"]["sum"]["value"])
    provenance = {"code_commit": head, "gate_files_sha256": gate_record["files"], "dev_json_sha256": _sha256(dev_path)}
    scorer = make_scorer(d, dev_effect, artifacts, provenance)
    out = score_once(EXPERIMENT, TRACKING, scorer)
    rule = {"development_decides": dev_out["decision"], "set_a": "D107 consistency check: inconsistent if the test "
            "effect interval excludes the development estimate; otherwise confirms if it excludes zero on that side; "
            "otherwise consistent; only inconsistent stops an adoption", "source": "PLAN.md D107"}
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"gate": gate_record, "rule": rule}}
    (RESULTS / f"{EXPERIMENT}-final.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-pass", "e1-gate", "e1-dev", "e1-posthoc", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research gravity {a.command}"):
        if a.command == "e1-pass":
            out = frame_pass(a.artifacts, a.max_minutes)
        else:
            out = {"e1-gate": gate, "e1-dev": dev, "e1-posthoc": posthoc, "e1-final": final}[a.command](a.artifacts)
    if a.command == "e1-final":
        out = {"reading": out.get("reading"), "verdict": out.get("verdict")}
    print(json.dumps(_plain(out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

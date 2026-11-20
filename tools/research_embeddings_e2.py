"""tools/research_embeddings_e2.py: embeddings-E2, a spatial shot basis by non-negative matrix factorization
(docs/research/embeddings.md, "embeddings-E2" and its amendment; PLAN.md D103, D116, D118).

    python3 -m tools.research_embeddings_e2 e2-hist  --artifacts DIR [--max-minutes 25]
    python3 -m tools.research_embeddings_e2 e2-gate  --artifacts DIR
    python3 -m tools.research_embeddings_e2 e2-range --artifacts DIR [--max-minutes 25]
    python3 -m tools.research_embeddings_e2 e2-fit   --artifacts DIR
    python3 -m tools.research_embeddings_e2 e2-split --artifacts DIR
    python3 -m tools.research_embeddings_e2 e2-logo  --artifacts DIR [--max-minutes 25]
    python3 -m tools.research_embeddings_e2 e2-dev   --artifacts DIR

Builds on embeddings-E1 (tools/research_embeddings.py): its gated partial sums, its frozen fit (the control arm), its
queries, pools, halves, possession floors, retrieval rule, position-only retriever and intervals are reused unchanged.

e2-hist   Reads every development game's tracking frames once, one game at a time, and writes per player the 1 ft
          half-court histogram of the offensive frames in the attacked half (positions mirrored as the frozen lane
          mirrors them), plus the frozen six zone counts from the same mirrored positions (for the gate). Resumable;
          stops after --max-minutes. Development games only; the 17 spent held-out games are never read.
e2-gate   The new data path against the frozen one: per (game, player) the six zone counts equal embeddings-E1's gated
          frozen partials exactly, each court histogram sums to them, and each shot histogram sums to the frozen
          attempts. Writes the histogram tables once, with their hashes in the gate record.
e2-range  The pre-registered game bootstrap of the split-half rates (resample games, rebuild both halves, re-transform
          and re-embed with the fixed fits), shown as a range and not read (D86, D118). Resumable in chunks.
e2-fit    Fits the two bases (sklearn NMF, KL loss, multiplicative updates, nndsvda, seed 0) and the frozen pipeline
          on each arm; caches the bases with the gate's file hashes.
e2-split  The deciding split-half statistic (player bootstrap), cached.
e2-logo   Leave one game out (reported, decides nothing), resumable in chunks of games, cached.
e2-dev    Assembles reports/research/embeddings-E2-dev.json and the verdict from the cached stages; fits nothing.
Every stage logs its timing and stays under the lock's 30-minute budget (--max-minutes where it can run long).

Set A is scored only if development adopts (D118); its scorer is not part of this module yet.
Nothing here writes under DATA_ROOT or reports/ except the research results. The only fits are NMF and the frozen
pipeline on development rows (rule 5: tools/, training games only).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from export.heavy import heavy_job, peak_rss_gb
from tools import research_embeddings as E1
from tools.frozen_common import REPO_ROOT, _plain, _sha256, assert_safe_output, artifacts_dir, training_games_only
from tools.research_testset import refuse_test

WHO = "tools.research_embeddings_e2"
EXPERIMENT = "embeddings-E2"
RESULTS = REPO_ROOT / "reports" / "research"
SEED, BOOT = E1.SEED, E1.BOOT
GX, GY = 47, 50
CELLS = GX * GY
MAPS = 10
THRESHOLD = 0.05                 # pre-registered: NMF minus frozen at least +0.05, interval above zero (D118)
TOL, MAX_ITER, CHECK_EVERY = 1e-6, 2000, 10
NMF_MAX_ITER = 1000
RANGE_CHUNK = 50
WORK = "embeddings_e2"
CHECK = "check.json"
SPACES = ("frozen", "nmf", "nmf_shot")
PAIRS = (("nmf", "frozen"), ("nmf_shot", "frozen"), ("nmf", "nmf_shot"))


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / WORK)


def cell_index(x, y) -> np.ndarray:
    """The 1 ft cell (floor(x), floor(y)) of the 47 by 50 half court, edge cells for coordinates outside it."""
    cx = np.clip(np.floor(np.asarray(x, dtype=float)), 0, GX - 1).astype(np.int64)
    cy = np.clip(np.floor(np.asarray(y, dtype=float)), 0, GY - 1).astype(np.int64)
    return cx * GY + cy


# ----------------------------------------------------------------------------- the histogram pass

def court_histogram(frames, possessions):
    """One game -> (histogram rows: player_id, cell, n; zone rows: player_id and the frozen six zone counts). The
    offence, the attacked basket and the mirroring repeat analytics.embeddings.features.tracking_partials line for line
    (the gate checks the zone counts against the frozen partials)."""
    import pandas as pd
    F, _ = E1._frozen()
    need = ["game", "period", "frame", "ts_ms", "team_id", "player_id", "x", "y", "ball_x", "ball_y", "ball_z"]
    f = frames[need].copy()
    f = f.drop_duplicates(["frame", "player_id"]).sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)
    game = str(f["game"].iloc[0]) if len(f) else ""
    poss = possessions[possessions["sportvu_game"].astype(str) == game] if "sportvu_game" in possessions.columns \
        else possessions
    frame_ids = f["frame"].to_numpy(dtype=np.int64)
    poss_idx, off_team = F._possession_of_frames(frame_ids, poss)
    team = f["team_id"].to_numpy(dtype=np.int64)
    is_off = (poss_idx >= 0) & (team == off_team)
    x, y = f["x"].to_numpy(dtype=float), f["y"].to_numpy(dtype=float)
    bx = f["ball_x"].to_numpy(dtype=float)
    period = f["period"].to_numpy(dtype=np.int64)
    att = pd.DataFrame({"period": period[is_off], "team": team[is_off], "bx": bx[is_off]})
    att_left = att.groupby(["period", "team"])["bx"].mean() < F.HALF_X
    attack_left = att_left.reindex(pd.MultiIndex.from_arrays([period, team])).to_numpy()
    attack_left = np.where(pd.isna(attack_left), True, attack_left).astype(bool)
    xa = np.where(attack_left, x, F.COURT_X - x)
    ya = np.where(attack_left, y, F.COURT_Y - y)
    band_x = np.floor(xa / F.BAND_X).astype(int)
    band_y = (ya >= F.RIM_Y).astype(int)
    in_half = is_off & (xa >= 0) & (xa < F.HALF_X)
    pid = f["player_id"].to_numpy(dtype=np.int64)
    zones = pd.DataFrame({"player_id": pid})
    for i, bxn in enumerate(F.ZONE_X):
        for j, byn in enumerate(F.ZONE_Y):
            zones["zone_%s_%s_n" % (bxn, byn)] = (in_half & (band_x == i) & (band_y == j)).astype(int)
    zones = zones.groupby("player_id").sum().reset_index()
    cells = cell_index(xa[in_half], ya[in_half])
    hist = pd.DataFrame({"player_id": pid[in_half], "cell": cells}).groupby(["player_id", "cell"]).size()
    hist = hist.rename("n").reset_index()
    hist["sportvu_game"] = game
    zones["sportvu_game"] = game
    return hist, zones


def hist_pass(artifacts=None, max_minutes: float = 25.0, log=print) -> dict:
    import pandas as pd
    root = Path(os.environ["DATA_ROOT"])
    games = E1.dev_game_list(root)
    out = work_dir(artifacts) / "hist"
    out.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    for g in games:
        target = out / f"{g}.zones.parquet"
        if target.exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        frames = pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet")
        poss = pd.read_parquet(root / "analytics" / "possessions" / E1.SEASON / f"{g}.parquet")
        hist, zones = court_histogram(frames, poss)
        del frames
        hist.to_parquet(out / f"{g}.hist.parquet", index=False)
        zones.to_parquet(out / f"{g}.zones.parquet.part", index=False)
        (out / f"{g}.zones.parquet.part").rename(target)
        done += 1
        if log and done % 50 == 0:
            log(f"[e2] {done} games this run, {time.time() - started:.0f} s")
    have = sum((out / f"{g}.zones.parquet").exists() for g in games)
    return {"games": len(games), "written_this_run": done, "present": have, "complete": have == len(games),
            "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2)}


def shot_histogram(shots, rosters):
    """Per (game, player, cell) attempt counts, shooters resolved and rows kept exactly as the frozen shot features
    (and embeddings-E1's shot partials) keep them. Reads only locations, never an outcome column."""
    import pandas as pd
    F, _ = E1._frozen()
    s = shots.copy()
    s["player_id"] = F.resolve_shooter_ids(s, rosters)
    s = s[s["player_id"].notna()].copy()
    s["player_id"] = s["player_id"].astype(int)
    s["cell"] = cell_index(s["court_x"], s["court_y"])
    out = s.groupby(["sportvu_game", "player_id", "cell"]).size().rename("n").reset_index()
    return out


# ----------------------------------------------------------------------------- the gate

def gate(artifacts=None) -> dict:
    import pandas as pd
    F, _ = E1._frozen()
    e1_gate = E1.require_gate(artifacts)
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts)
    (folder / CHECK).write_text(json.dumps({"passed": False, "status": "gate started"}) + "\n")
    games = E1.dev_game_list(root)
    hdir = folder / "hist"
    missing = [g for g in games if not (hdir / f"{g}.zones.parquet").exists()]
    if missing:
        raise SystemExit(f"{WHO}: {len(missing)} development games have no histogram yet; run e2-hist")
    zones = pd.concat([pd.read_parquet(hdir / f"{g}.zones.parquet") for g in games], ignore_index=True)
    court = pd.concat([pd.read_parquet(hdir / f"{g}.hist.parquet") for g in games], ignore_index=True)
    e1dir = E1.work_dir(artifacts)
    track = pd.read_parquet(e1dir / "tracking_partials.parquet")
    sp = pd.read_parquet(e1dir / "shot_partials.parquet")
    zcols = [f"{z}_n" for z in F.ZONE_FEATURES]
    key = ["sportvu_game", "player_id"]
    m = track[key + zcols].merge(zones[key + zcols], on=key, how="outer", suffixes=("_frozen", "_mine"),
                                 indicator=True)
    rows_only_one_side = int((m["_merge"] != "both").sum())
    zone_diff = int(sum((m[f"{c}_frozen"] != m[f"{c}_mine"]).sum() for c in zcols)) if not rows_only_one_side else -1
    ctot = court.groupby(key)["n"].sum()
    ztot = zones.set_index(key)[zcols].sum(axis=1)
    court_vs_zones = int((ctot.reindex(ztot.index).fillna(0).astype(int) != ztot.astype(int)).sum())
    tables = E1.dev_tables(root, games)
    shots = shot_histogram(tables["shots"], tables["rosters"])
    stot = shots.groupby(key)["n"].sum()
    att = sp.set_index(key)["attempts"]
    shots_vs_attempts = int((stot.reindex(att.index).fillna(0).astype(int) != att.astype(int)).sum()) + \
        int(len(stot.index.difference(att.index)))
    court.to_parquet(folder / "court_hist.parquet", index=False)
    shots.to_parquet(folder / "shot_hist.parquet", index=False)
    passed = bool(rows_only_one_side == 0 and zone_diff == 0 and court_vs_zones == 0 and shots_vs_attempts == 0)
    record = {"experiment": EXPERIMENT, "games": len(games), "player_games": int(len(zones)),
              "compared": ("the six frozen zone counts recounted from the histogram pass's own mirrored positions "
                           "against embeddings-E1's gated frozen partials, per (game, player); each court histogram "
                           "against those six counts; each shot histogram against the frozen attempts"),
              "rows_on_one_side_only": rows_only_one_side, "zone_count_mismatches": zone_diff,
              "court_histograms_not_summing_to_zone_counts": court_vs_zones,
              "shot_histograms_not_summing_to_attempts": shots_vs_attempts,
              "court_frames_counted": int(court["n"].sum()), "shots_counted": int(shots["n"].sum()),
              "e1_gate_files": e1_gate["files"], "passed": passed}
    E1._record(folder, record, ["court_hist.parquet", "shot_hist.parquet"])
    if not passed:
        print(json.dumps(_plain(record), indent=2))
        raise SystemExit(f"{WHO}: the histogram pass does not reproduce the frozen counts; nothing downstream runs")
    return require_gate(artifacts)


def require_gate(artifacts=None) -> dict:
    folder = work_dir(artifacts)
    path = folder / CHECK
    if not path.exists():
        raise RuntimeError(f"{EXPERIMENT}: no gate record (run e2-gate first)")
    record = json.loads(path.read_text())
    if record.get("passed") is not True or not record.get("files"):
        raise RuntimeError(f"{EXPERIMENT}: the histogram pass did not reproduce the frozen counts")
    for name, digest in record["files"].items():
        if _sha256(folder / name) != digest:
            raise RuntimeError(f"{EXPERIMENT}: {name} changed after the gate record was written")
    return record


# ----------------------------------------------------------------------------- the basis and its transform

def kl_transform(X, H, tol: float = TOL, max_iter: int = MAX_ITER, chunk: int = 2000):
    """Weights W >= 0 minimising KL(X || W H) with the basis H held fixed, by multiplicative updates on W only, from
    the same constant start for every row; each row stops on its own when its objective changes by at most tol
    (relative) over CHECK_EVERY updates, so a row's weights do not depend on the other rows of the batch. All-zero
    rows get zero weights. X: rows (n x CELLS), sparse or dense; computed densely in chunks of rows."""
    from scipy import sparse
    X = sparse.csr_matrix(X, dtype=float)
    n, k = X.shape[0], H.shape[0]
    dead = np.flatnonzero(H.sum(axis=0) <= 0)
    if len(dead):
        hit = np.asarray((X[:, dead] > 0).sum(axis=1)).ravel() > 0
        if hit.any():                     # KL is infinite there: the row would come out NaN, so refuse, never impute
            raise ValueError(f"{WHO}: {int(hit.sum())} row(s) have mass in a cell where every basis map is zero")
    W = np.zeros((n, k))
    iters = np.zeros(n, dtype=int)
    left = 0
    Hsum = H.sum(axis=1)
    HT = H.T
    for s in range(0, n, chunk):
        Xd = X[s:s + chunk].toarray()
        m = len(Xd)
        pos = Xd > 0
        has = pos.any(axis=1)
        Wc = np.full((m, k), np.sqrt((1.0 / X.shape[1]) / k))
        active = has.copy()
        prev = np.full(m, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            xlogx = np.where(pos, Xd * np.log(np.where(pos, Xd, 1.0)), 0.0).sum(axis=1) - Xd.sum(axis=1)
            for it in range(max_iter):
                if not active.any():
                    break
                a = np.flatnonzero(active)
                WH = Wc[a] @ H
                Wc[a] = Wc[a] * ((np.where(pos[a], Xd[a] / WH, 0.0)) @ HT) / Hsum
                iters[s + a] += 1
                if (it + 1) % CHECK_EVERY == 0:
                    WH = Wc[a] @ H
                    obj = xlogx[a] - np.where(pos[a], Xd[a] * np.log(np.where(pos[a], WH, 1.0)), 0.0).sum(axis=1) \
                        + Wc[a] @ Hsum
                    done = np.abs(prev[a] - obj) <= tol * np.maximum(np.abs(obj), 1e-12)
                    prev[a] = obj
                    active[a[done]] = False
        Wc[~has] = 0.0
        W[s:s + m] = Wc
        left += int(active.sum())
    return W, {"rows": int(n), "max_iterations": int(iters.max()) if n else 0, "not_converged": left}


def shares(W) -> np.ndarray:
    """Weights divided by their sum; a row with no weight (no frame or no attempt) is missing (NaN)."""
    s = W.sum(axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return np.where(s > 0, W / s, np.nan)


def rates(Hrows):
    """Histogram rows over their totals (sparse); zero rows stay zero."""
    from scipy import sparse
    Hrows = sparse.csr_matrix(Hrows, dtype=float)
    tot = np.asarray(Hrows.sum(axis=1)).ravel()
    inv = np.where(tot > 0, 1.0 / np.where(tot > 0, tot, 1.0), 0.0)
    return sparse.diags(inv) @ Hrows


def fit_basis(Hrows) -> dict:
    """sklearn NMF on the rates of the given rows: KL loss, multiplicative updates, nndsvda, seed 0."""
    from sklearn.decomposition import NMF
    X = rates(Hrows)
    nonzero = np.asarray(X.sum(axis=1)).ravel() > 0
    model = NMF(n_components=MAPS, init="nndsvda", solver="mu", beta_loss="kullback-leibler", max_iter=NMF_MAX_ITER,
                random_state=SEED)
    model.fit(X[nonzero])
    H = model.components_.copy()
    mine, info = kl_transform(X[nonzero], H)
    theirs = model.transform(X[nonzero])
    return {"H": H, "rows": int(nonzero.sum()), "n_iter": int(model.n_iter_),
            "converged": bool(model.n_iter_ < NMF_MAX_ITER), "reconstruction_err": float(model.reconstruction_err_),
            "transform_check": {"max_abs_diff_of_shares_vs_sklearn_transform":
                                float(np.nanmax(np.abs(shares(mine) - shares(theirs)))), **info}}


# ----------------------------------------------------------------------------- features of the arms

class Hists:
    """The court and shot histograms aligned to embeddings-E1's Parts rows (game, player), as sparse matrices, so the
    histogram of any weighted set of a player's games is a weighted row sum, like the partial sums."""

    def __init__(self, P, court, shots):
        from scipy import sparse
        gi = {g: i for i, g in enumerate(P.games)}
        pi = {int(p): i for i, p in enumerate(P.players)}
        lookup = np.full(P.G * P.P, -1, dtype=np.int64)
        lookup[P.g.astype(np.int64) * P.P + P.p] = np.arange(P.R)
        self.P = P
        self.mats = {}
        for name, t in (("court", court), ("shot", shots)):
            t = t[t["sportvu_game"].astype(str).isin(gi) & t["player_id"].astype(int).isin(pi)]
            g = t["sportvu_game"].astype(str).map(gi).to_numpy()
            p = t["player_id"].astype(int).map(pi).to_numpy()
            rows = lookup[g.astype(np.int64) * P.P + p.astype(np.int64)]
            if (rows < 0).any():
                raise SystemExit(f"{WHO}: a {name} histogram row has no partial-sum row")
            self.mats[name] = sparse.csr_matrix((t["n"].to_numpy(dtype=float), (rows, t["cell"].to_numpy())),
                                                shape=(P.R, CELLS))

    def player_sums(self, name: str, w=None):
        """Per-player histograms (P x CELLS) under row weights w (R,), all ones by default."""
        from scipy import sparse
        w = np.ones(self.P.R) if w is None else np.asarray(w, dtype=float)
        return (self.P.A.T @ sparse.diags(w) @ self.mats[name]).tocsr()

    def game_rows(self, name: str, g: int):
        """The rows of one game (players of that game, per Parts.p) as (player indices, histograms)."""
        rows = np.flatnonzero(self.P.g == g)
        return self.P.p[rows], self.mats[name][rows]


def arm_columns():
    F, _ = E1._frozen()
    order = E1.feature_order()
    court = [f"court_basis_{i}" for i in range(MAPS)]
    shot = [f"shot_basis_{i}" for i in range(MAPS)]
    keep_nmf = [c for c in order if c not in F.ZONE_FEATURES and c not in [f"shot_share_{z}" for z in F.SHOT_ZONES]]
    keep_shot = [c for c in order if c not in [f"shot_share_{z}" for z in F.SHOT_ZONES]]
    return {"nmf": (keep_nmf, court + shot), "nmf_shot": (keep_shot, shot)}


def arm_features(X19, court_shares, shot_shares) -> dict:
    """The NMF arm's 27 and the ablation's 23 feature columns (names, values) from the frozen 19 and the shares."""
    order = E1.feature_order()
    cols = arm_columns()
    idx = {c: i for i, c in enumerate(order)}
    out = {}
    for arm, (keep, new) in cols.items():
        parts = [X19[..., [idx[c] for c in keep]]]
        parts.append(np.concatenate([court_shares, shot_shares], axis=-1) if arm == "nmf" else shot_shares)
        out[arm] = (keep + new, np.concatenate(parts, axis=-1))
    return out


def fit_arm(names, X) -> dict:
    """The frozen pipeline (median imputation, standardisation, PCA to 8) on an arm's columns of the fit rows."""
    import pandas as pd
    F, T = E1._frozen()
    df = pd.DataFrame(X, columns=names)
    F.assert_no_leak(df)
    Z, transform, _ = T.standardise(df)
    pca = T.fit_pca(Z, SEED)
    return {"names": names, "transform": transform, "embed": pca["embed"],
            "explained": [float(v) for v in pca["explained_variance_ratio"]]}


def embed(fit: dict, X) -> np.ndarray:
    import pandas as pd
    shape = X.shape[:-1]
    flat = pd.DataFrame(X.reshape(-1, X.shape[-1]), columns=fit["names"])
    E = fit["embed"](fit["transform"](flat))
    return np.asarray(E, dtype=float).reshape(shape + (E.shape[1],))


class Model:
    """Everything the retrieval reads: E1's frozen fit, the two bases and the two arm fits."""

    def __init__(self, d: dict, hists: Hists, basis: dict = None, log=print):
        self.d, self.hists = d, hists
        P = d["P"]
        fit_rows = d["fit"]["rows"]
        self.fit_idx = np.searchsorted(P.players, fit_rows["player_id"].to_numpy(dtype=np.int64))
        full_court = hists.player_sums("court")[self.fit_idx]
        full_shot = hists.player_sums("shot")[self.fit_idx]
        self.transform_record = {name: {"calls": 0, "rows": 0, "not_converged": 0, "max_iterations": 0}
                                 for name in ("court", "shot")}
        started = time.time()
        self.basis = basis or {"court": fit_basis(full_court), "shot": fit_basis(full_shot)}
        if log and basis is None:
            log(f"[e2] two NMF bases fitted, {time.time() - started:.0f} s")
        X19 = fit_rows[E1.feature_order()].to_numpy(dtype=float)
        cs, ss = self.shares_of(full_court, full_shot)
        feats = arm_features(X19, cs, ss)
        self.fits = {arm: fit_arm(*feats[arm]) for arm in ("nmf", "nmf_shot")}

    def shares_of(self, court_rows, shot_rows):
        """Shares of both bases; every call's convergence is added to transform_record, and a weight row that is
        not finite is refused (never imputed)."""
        out = []
        for name, rows in (("court", court_rows), ("shot", shot_rows)):
            W, info = kl_transform(rates(rows), self.basis[name]["H"])
            if not np.isfinite(W).all():
                raise SystemExit(f"{WHO}: a {name} weight row is not finite; refused, not imputed")
            rec = self.transform_record[name]
            rec["calls"] += 1
            rec["rows"] += info["rows"]
            rec["not_converged"] += info["not_converged"]
            rec["max_iterations"] = max(rec["max_iterations"], info["max_iterations"])
            out.append(shares(W))
        return out[0], out[1]

    def represent(self, X19, court_rows, shot_rows) -> dict:
        """Every retrieval space for rows given as frozen features (n x 19) and histograms (n x CELLS)."""
        cs, ss = self.shares_of(court_rows, shot_rows)
        feats = arm_features(X19, cs, ss)
        out = {"frozen": E1.represent(self.d["fit"], X19)["pca"]}
        for arm in ("nmf", "nmf_shot"):
            out[arm] = embed(self.fits[arm], feats[arm][1])
        return out


def query_values(hit: dict, pos_prob) -> dict:
    vals = {s: hit[s].astype(float) for s in SPACES}
    vals["position"] = np.asarray(pos_prob, dtype=float)
    for a, b in PAIRS:
        vals[f"{a}_minus_{b}"] = vals[a] - vals[b]
    return vals


# ----------------------------------------------------------------------------- development statistics

def _halves(model: Model):
    P = model.d["P"]
    half = P.halves()
    masks = [(half == h).astype(float) for h in (0, 1)]
    Xh, nh = zip(*(E1.features_from_sums(P.sums(m[None, :])) for m in masks))
    idx = np.flatnonzero((nh[0][0] >= E1.N_MIN) & (nh[1][0] >= E1.N_MIN))
    return masks, Xh, idx


def split_half(model: Model, n: int = BOOT) -> dict:
    """The deciding statistic (D118): each player's two date-alternated halves retrieve each other, both directions,
    players with at least 300 possessions in each half; a bootstrap of the queries by player, pool fixed."""
    d, H, P = model.d, model.hists, model.d["P"]
    masks, Xh, idx = _halves(model)
    if not len(idx):
        raise SystemExit(f"{WHO}: no player has enough possessions in both halves")
    reps = [model.represent(Xh[h][0][idx], H.player_sums("court", masks[h])[idx], H.player_sums("shot", masks[h])[idx])
            for h in (0, 1)]
    own = np.arange(len(idx))
    hit = {s: np.concatenate([E1.hits(reps[0][s], reps[1][s], own), E1.hits(reps[1][s], reps[0][s], own)])
           for s in SPACES}
    pos = E1.positions_of(P.players[idx], d["positions"])
    vals = query_values(hit, np.concatenate([E1.position_probability(pos, pos)] * 2))
    boot = E1.cluster_bootstrap(np.concatenate([P.players[idx]] * 2), vals, n, [SEED, 11])
    return {"players": int(len(idx)), "queries": int(2 * len(idx)), "chance": float(E1.K / len(idx)),
            "rates_with_player_bootstrap": boot["intervals"],
            "note": "the deciding interval (D118, E1 amendment commitment 6): queries resampled by player, pool fixed"}


def game_range(model: Model, artifacts=None, n: int = BOOT, max_minutes: float = 25.0, log=print) -> dict:
    """The pre-registered game bootstrap of the split-half rates, shown and not read (D86, D118): resample games,
    rebuild both halves (features and histograms), re-transform and re-embed with the fixed fits. Resumable: the
    draws of each finished chunk of RANGE_CHUNK resamples are kept under the work folder."""
    d, H, P = model.d, model.hists, model.d["P"]
    masks, _, idx = _halves(model)
    own = np.arange(len(idx))
    pos = E1.positions_of(P.players[idx], d["positions"])
    pprob = np.concatenate([E1.position_probability(pos, pos)] * 2)
    folder = work_dir(artifacts) / "range"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng([SEED, 12])
    for b0 in range(0, n, RANGE_CHUNK):
        B = min(RANGE_CHUNK, n - b0)
        draws_g = [rng.integers(0, P.G, P.G) for _ in range(B)]       # drawn in order, so a resume draws the same
        path = folder / f"chunk_{b0:05d}.json"
        if path.exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        rows = []
        for draw in draws_g:
            wg = np.bincount(draw, minlength=P.G).astype(float)
            wr = wg[P.g]
            reps = []
            for h in (0, 1):
                m = wr * masks[h]
                X, _ = E1.features_from_sums(P.sums(m[None, :]))
                reps.append(model.represent(X[0][idx], H.player_sums("court", m)[idx], H.player_sums("shot", m)[idx]))
            hb = {s: np.concatenate([E1.hits(reps[0][s], reps[1][s], own), E1.hits(reps[1][s], reps[0][s], own)])
                  for s in SPACES}
            v = query_values(hb, pprob)
            rows.append({k: float(np.mean(x)) for k, x in v.items()})
        path.write_text(json.dumps(rows) + "\n")
        if log:
            log(f"[e2] game range {b0 + B} of {n}, {time.time() - started:.0f} s")
    chunks = sorted(folder.glob("chunk_*.json"))
    draws = [r for p in chunks for r in json.loads(p.read_text())]
    out = {"resamples_done": len(draws), "resamples_planned": n, "complete": len(draws) == n}
    if draws:
        out["range"] = {k: {"lo": float(np.percentile([r[k] for r in draws], 2.5)),
                            "hi": float(np.percentile([r[k] for r in draws], 97.5))} for k in draws[0]}
    return out


LOGO_CHUNK = 100


def logo_games(model: Model, g0: int, g1: int) -> dict:
    """Leave one game out for the development games g0 to g1 - 1: per query (game, player) the hit of every space and
    the position retriever's probability, and the pool size of each game with queries."""
    from scipy import sparse
    d, H, P = model.d, model.hists, model.d["P"]
    GS = P.game_sums()
    tot = {c: v.sum(axis=0) for c, v in GS.items()}
    full = {name: H.player_sums(name) for name in ("court", "shot")}
    pos_all = E1.positions_of(P.players, d["positions"])
    out = {s: [] for s in SPACES}
    out.update(position=[], game=[], player=[], pool=[])
    for g in range(g0, min(g1, P.G)):
        one = {c: GS[c][g] for c in GS}
        rest = {c: tot[c] - GS[c][g] for c in GS}
        Xr, nr = E1.features_from_sums(rest)
        Xo, _ = E1.features_from_sums(one)
        pool = np.flatnonzero(nr >= E1.N_MIN)
        q = np.flatnonzero((one["n_possessions"] >= E1.ONE_GAME_MIN) & (nr >= E1.N_MIN))
        if not len(q):
            continue
        game_h = {}
        for name in ("court", "shot"):
            pidx, rows = H.game_rows(name, g)
            lift = sparse.csr_matrix((np.ones(len(pidx)), (pidx, np.arange(len(pidx)))), shape=(P.P, len(pidx)))
            game_h[name] = (lift @ rows).tocsr()
        rest_h = {name: (full[name] - game_h[name]).tocsr() for name in game_h}
        pool_rep = model.represent(Xr[pool], rest_h["court"][pool], rest_h["shot"][pool])
        q_rep = model.represent(Xo[q], game_h["court"][q], game_h["shot"][q])
        where = {int(i): j for j, i in enumerate(pool)}
        own = np.array([where[int(i)] for i in q])
        for s in SPACES:
            out[s].append(E1.hits(q_rep[s], pool_rep[s], own).astype(float))
        out["position"].append(E1.position_probability(pos_all[q], pos_all[pool]))
        out["game"].append(np.full(len(q), g))
        out["player"].append(P.players[q].astype(np.int64))
        out["pool"].append(np.array([len(pool)]))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def logo_summary(parts: dict, n: int = BOOT) -> dict:
    """D103's matched comparator from the per-query values of every development game, reported and deciding nothing
    (D118); bootstrap of the development games."""
    hit = {s: parts[s].astype(bool) for s in SPACES}
    v = query_values(hit, parts["position"])
    g_of = parts["game"].astype(int)
    boot = E1.cluster_bootstrap(g_of, v, n, [SEED, 13])
    return {"games_with_queries": int(len(np.unique(g_of))), "queries": int(len(g_of)),
            "players": int(len(np.unique(parts["player"]))),
            "pool_size": {"mean": float(np.mean(parts["pool"])), "min": int(np.min(parts["pool"])),
                          "max": int(np.max(parts["pool"]))},
            "rates_with_game_bootstrap": boot["intervals"],
            "note": "D103's matched comparator, reported beside the verdict and deciding nothing (D118); the fits saw "
                    "every development game, stated not repaired"}


def leave_one_game_out(model: Model, n: int = BOOT, log=print) -> dict:
    return logo_summary(logo_games(model, 0, model.d["P"].G), n)


def stage_logo(artifacts=None, max_minutes: float = 25.0) -> dict:
    """Leave one game out in chunks of LOGO_CHUNK games, each kept under the work folder; resumable. When every chunk
    exists, writes the summary."""
    started = time.time()
    model = load(artifacts)
    folder = work_dir(artifacts) / "logo"
    folder.mkdir(parents=True, exist_ok=True)
    G = model.d["P"].G
    for g0 in range(0, G, LOGO_CHUNK):
        path = folder / f"chunk_{g0:04d}.npz"
        if path.exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        np.savez(path, **logo_games(model, g0, g0 + LOGO_CHUNK))
        _log(f"leave one game out, games {g0} to {min(g0 + LOGO_CHUNK, G) - 1}", started)
    chunks = [folder / f"chunk_{g0:04d}.npz" for g0 in range(0, G, LOGO_CHUNK)]
    if not all(c.exists() for c in chunks):
        return {"chunks_done": sum(c.exists() for c in chunks), "chunks": len(chunks)}
    loaded = [dict(np.load(c)) for c in chunks]
    parts = {k: np.concatenate([x[k] for x in loaded]) for k in loaded[0]}
    out = logo_summary(parts)
    out["seconds_last_run"] = round(time.time() - started, 1)
    out["transform_record_last_run"] = model.transform_record
    (work_dir(artifacts) / "logo.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return {k: out["rates_with_game_bootstrap"][k] for k in ("frozen", "nmf", "nmf_minus_frozen")}


def verdict(split: dict) -> dict:
    d = split["rates_with_player_bootstrap"]["nmf_minus_frozen"]
    size = d["value"] >= THRESHOLD
    above = d["lo"] > 0
    adopt = bool(size and above)
    if adopt:
        reading = "adopted on development; Set A is scored once as a consistency check (D118)"
    elif not above:
        reading = "not adopted on development: the interval of NMF minus frozen does not lie above zero"
    else:
        reading = "not adopted on development: the gain is real (interval above zero) but smaller than +0.05"
    return {"rule": "split-half top-5 self-retrieval, NMF minus frozen, at least +0.05 with the player-bootstrap "
                    "interval above zero (D118)", "difference": d["value"], "lo": d["lo"], "hi": d["hi"],
            "threshold": THRESHOLD, "size_at_least_threshold": bool(size), "interval_above_zero": bool(above),
            "adopt": adopt, "reading": reading, "set_a": "scored once as a consistency check" if adopt else "not scored"}


def _log(msg: str, started: float) -> None:
    print(f"[e2] {msg}, {time.time() - started:.0f} s", flush=True)


def load(artifacts=None, fitted: bool = True) -> Model:
    """The gated inputs and the model; with fitted, the bases are read from e2-fit's cache (whose record names the
    gate's file hashes) instead of being fitted again."""
    import pandas as pd
    started = time.time()
    gate_record = require_gate(artifacts)
    d = E1.load_dev(artifacts)
    _log("embeddings-E1 development inputs loaded", started)
    folder = work_dir(artifacts)
    court = pd.read_parquet(folder / "court_hist.parquet")
    shots = pd.read_parquet(folder / "shot_hist.parquet")
    games = sorted(set(court["sportvu_game"].astype(str)) | set(shots["sportvu_game"].astype(str)))
    training_games_only(games, WHO)
    refuse_test(games, WHO)
    if not set(games) <= set(d["games"]):
        raise SystemExit(f"{WHO}: a histogram game is not a development game")
    hists = Hists(d["P"], court, shots)
    del court, shots
    _log("histograms assembled", started)
    basis = None
    if fitted:
        rec = json.loads((folder / "fit.json").read_text())
        if rec["gate_files"] != gate_record["files"] or _sha256(folder / "bases.npz") != rec["bases_sha256"]:
            raise SystemExit(f"{WHO}: the cached bases do not belong to this gate; run e2-fit again")
        z = np.load(folder / "bases.npz")
        basis = {name: {**rec["bases"][name], "H": z[name]} for name in ("court", "shot")}
    model = Model(d, hists, basis)
    _log("arm fits ready", started)
    return model


def stage_fit(artifacts=None) -> dict:
    started = time.time()
    model = load(artifacts, fitted=False)
    folder = work_dir(artifacts)
    np.savez(folder / "bases.npz", court=model.basis["court"]["H"], shot=model.basis["shot"]["H"])
    rec = {"gate_files": require_gate(artifacts)["files"], "bases_sha256": _sha256(folder / "bases.npz"),
           "bases": {name: {k: v for k, v in b.items() if k != "H"} for name, b in model.basis.items()},
           "arm_fits": {arm: {"columns": f["names"], "explained_variance_ratio": f["explained"]}
                        for arm, f in model.fits.items()}, "seconds": round(time.time() - started, 1)}
    (folder / "fit.json").write_text(json.dumps(_plain(rec), indent=2, allow_nan=False) + "\n")
    return rec


def stage_split(artifacts=None) -> dict:
    started = time.time()
    model = load(artifacts)
    out = split_half(model)
    out["seconds"] = round(time.time() - started, 1)
    out["transform_record"] = model.transform_record
    (work_dir(artifacts) / "split.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return {k: out["rates_with_player_bootstrap"][k] for k in ("frozen", "nmf", "nmf_minus_frozen")}


def dev(artifacts=None) -> dict:
    """Assembles the development json from the cached stages (e2-fit, e2-split, e2-logo, e2-range); fits nothing."""
    folder = work_dir(artifacts)
    gate_record = require_gate(artifacts)
    fit = json.loads((folder / "fit.json").read_text())
    if fit["gate_files"] != gate_record["files"]:
        raise SystemExit(f"{WHO}: the cached fits do not belong to this gate")
    split = json.loads((folder / "split.json").read_text())
    logo = json.loads((folder / "logo.json").read_text())
    e1_dev = json.loads((RESULTS / "embeddings-E1-dev.json").read_text())
    frozen_check = {
        "split_half": [split["rates_with_player_bootstrap"]["frozen"]["value"],
                       e1_dev["split_half"]["rates_with_player_bootstrap"]["pca"]["value"]],
        "leave_one_game_out": [logo["rates_with_game_bootstrap"]["frozen"]["value"],
                               e1_dev["leave_one_game_out"]["rates_with_game_bootstrap"]["pca"]["value"]]}
    if any(a != b for a, b in frozen_check.values()):
        raise SystemExit(f"{WHO}: the frozen arm does not reproduce embeddings-E1's development rates: {frozen_check}")
    chunks = sorted((folder / "range").glob("chunk_*.json")) if (folder / "range").exists() else []
    draws = [r for p in chunks for r in json.loads(p.read_text())]
    rng_out = {"resamples_done": len(draws), "resamples_planned": BOOT}
    if draws:
        rng_out["range"] = {k: {"lo": float(np.percentile([r[k] for r in draws], 2.5)),
                                "hi": float(np.percentile([r[k] for r in draws], 97.5))} for k in draws[0]}
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    out = {"experiment": EXPERIMENT, "code_commit": head,
           "stage": "development games (gates passed); development decides (PLAN.md D118)",
           "pre_registration": "docs/research/embeddings.md, embeddings-E2 and its amendment (D118)",
           "gate": {k: v for k, v in gate_record.items() if k != "files"},
           "bases": fit["bases"], "arm_fits": fit["arm_fits"], "frozen_arm_equals_embeddings_E1": frozen_check,
           "split_half": split, "split_half_game_bootstrap_range_not_read": rng_out,
           "leave_one_game_out": logo, "verdict": verdict(split)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out["verdict"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e2-hist", "e2-gate", "e2-fit", "e2-split", "e2-logo", "e2-range", "e2-dev"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research embeddings {a.command}"):
        if a.command == "e2-hist":
            out = hist_pass(a.artifacts, a.max_minutes)
        elif a.command == "e2-logo":
            out = stage_logo(a.artifacts, a.max_minutes)
        elif a.command == "e2-range":
            out = game_range(load(a.artifacts), a.artifacts, max_minutes=a.max_minutes)
            out.pop("range", None)
        else:
            out = {"e2-gate": gate, "e2-fit": stage_fit, "e2-split": stage_split, "e2-dev": dev}[a.command](a.artifacts)
    print(json.dumps(_plain(out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

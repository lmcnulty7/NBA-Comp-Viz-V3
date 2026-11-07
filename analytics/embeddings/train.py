"""analytics/embeddings/train.py: player style embeddings (unsupervised, unit player-season) under the evaluation protocol.

    python -m analytics.embeddings.train --fixtures | --data DATA_ROOT --report reports/embeddings.md
                                        [--out OUT_ROOT] [--folds 5] [--seed 0] [--bootstrap 200] [--n-min 300]
                                        [--fixture-games 8] [--fixture-possessions 40] [--fixture-heldout-every 4]

DATA_ROOT layout (the owner's Mac):
    DATA_ROOT/sportvu_contract/<game>.parquet (+ heldout/)            section 4 frames, one file per game
    DATA_ROOT/analytics/possessions/<season>/<game>.parquet (+ heldout/<season>/)   section 19
    DATA_ROOT/analytics/rosters/<season>/<game>.parquet (+ heldout/<season>/)       section 16 (optional; resolves bref shooter ids)
    DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet            section 10a (or collect/shots/shotchartdetail/, section 10)
    DATA_ROOT/collect/tracking/aggregates/<season>.parquet            section 11 long table (optional: absent means tracking
                                                                      and shot features only)
    DATA_ROOT/collect/positions.parquet                               player_id, season, position in {G, F, C} (optional,
      or DATA_ROOT/collect/tracking/positions/<season>.parquet        needed input for the 5-NN position metric)

Protocol:
  0. Speed and distance steps are consecutive frames of the same game and period with 0 < dt <= 2 s on ts_ms (not the
     game clock); frames outside a possession span (dead balls) are included, so dead-ball movement counts.
  1. load_inputs drops every held-out game before any table is read into a feature: files under heldout/, ids that
     analytics.eval.is_heldout_game flags (sportvu/splits.json tokens and the held-out arena), shot rows with heldout true. The training game
     ids then pass analytics.eval.refuse_heldout, and the feature frame passes analytics.schema.assert_time_forward.
  2. Features per player-season (analytics.embeddings.features): tracking partial sums per game, combined over games,
     joined to the shot mix and to the same-season NBA.com aggregates when present. Rows under n_min possessions are
     flagged insufficient and are neither fitted nor evaluated.
  3. Standardise: median imputation then StandardScaler, both fit on the sufficient rows.
  4. Candidates: (1) PCA to 8 components; (2) an autoencoder (sklearn MLPRegressor reconstructing its standardised input,
     hidden layers 32, 8, 32, tanh; the 8-dim bottleneck is the second hidden layer's activation), trained ONLY when
     PCA's 5-NN position accuracy is under 0.60, and kept only when it beats PCA on that metric. Without positions the gate
     cannot be evaluated and PCA stands.
  5. Evaluation, with bootstrap intervals over players (analytics.eval.bootstrap_by_game with player_id as the group, on
     the per-player out-of-sample outcomes):
       5-NN position accuracy: leave-one-out, majority of the 5 nearest player-seasons' positions (ties go to the nearest),
       against the majority-class baseline;
       next-season R squared: ridge (alpha 1) from the 2015-16 embedding to a 2016-17 aggregate stat (the first of
       NEXT_SEASON_TARGETS present), out-of-fold under grouped K-fold by player_id (--folds, default 5), against the fold-mean baseline.
     The next-season stat is a target only; no 2016-17 value enters the embedding (aggregate_features reads one season).
  6. Outputs under OUT_ROOT/analytics/embeddings/<season>/: player_season_features, player_embeddings (both candidates
     when both were fitted, the chosen one first) and embedding_neighbours (the chosen method), docs/contract/embeddings.md.

A --fixtures run writes the analytics.fixtures DATA_ROOT (write_fixture_root: league rosters, with aggregates and
positions keyed by the same players) to a temporary directory and loads it through load_inputs, writes its tables to a
temporary directory unless --out is given, lowers n_min to FIXTURE_N_MIN (the fixture games are short) and labels the
report "fixture output, not a result".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.embeddings import features as F
from analytics.eval import bootstrap_by_game, fmt_ci, heldout_mask, is_heldout_game, refuse_heldout

log = logging.getLogger("analytics.embeddings.train")

WHO = "embeddings.train"
METHODS = ("pca", "autoencoder")
N_DIM = 8
K_NN = 5
AE_GATE = 0.60
AE_LAYERS = (32, N_DIM, 32)
FIXTURE_N_MIN = 20
NEXT_SEASON_TARGETS = (("leaguedashptstats", "drives"), ("leaguedashptstats", "touches"),
                       ("leaguedashptstats", "passes_made"), ("leaguehustlestatsplayer", "contested_shots"))
EMB_COLS = ["e%d" % i for i in range(1, N_DIM + 1)]
POSITION_MAP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}   # part before the first hyphen
MAC_CMD = ("python -m analytics.embeddings.train --data /path/to/DATA_ROOT --report reports/embeddings.md "
           "--out /path/to/DATA_ROOT --seed 0 --bootstrap 1000")


# ----------------------------------------------------------------------------- held-out detection

def heldout_shot_mask(shots: pd.DataFrame) -> pd.Series:
    """True for shot rows of a held-out game: by id (analytics.eval.heldout_mask), a heldout flag or a heldout split."""
    m = pd.Series(heldout_mask(shots["sportvu_game"].astype(str)), index=shots.index)
    if "heldout" in shots.columns:
        m = m | shots["heldout"].fillna(False).astype(bool)
    if "split" in shots.columns:
        m = m | shots["split"].astype("string").fillna("").str.startswith("heldout").astype(bool)
    return m


# ----------------------------------------------------------------------------- inputs

def _read_table_dir(root: Path, table: str) -> dict:
    """{game: frame} over analytics/<table>/<season>/*.parquet, skipping heldout/."""
    out = {}
    base = root / "analytics" / table
    if not base.exists():
        return out
    for p in sorted(base.rglob("*.parquet")):
        if "heldout" in p.relative_to(base).parts:
            continue
        out[p.stem] = pd.read_parquet(p)
    return out


def load_inputs(data_root) -> dict:
    """The DATA_ROOT layout -> dict(games, shots, aggregates, positions, dropped, source). Frames are loaded lazily per
    game through games[i]["frames"]() so one game sits in memory at a time."""
    root = Path(data_root)
    dropped = {"heldout_frame_files": 0, "heldout_games_by_id": [], "games_without_possessions": [], "heldout_shot_rows": 0}
    fdir = root / "sportvu_contract"
    frame_files = sorted(p for p in fdir.glob("*.parquet") if not p.name.endswith("_pbp.parquet")) if fdir.exists() else []
    hdir = fdir / "heldout"
    dropped["heldout_frame_files"] = len([p for p in hdir.glob("*.parquet") if not p.name.endswith("_pbp.parquet")]) if hdir.exists() else 0
    poss = _read_table_dir(root, "possessions")
    rosters = _read_table_dir(root, "rosters")
    games = []
    for p in frame_files:
        gid = p.stem
        if is_heldout_game(gid):
            dropped["heldout_games_by_id"].append(gid)
            continue
        if gid not in poss:
            dropped["games_without_possessions"].append(gid)
            continue
        games.append({"sportvu_game": gid, "frames": (lambda q=p: pd.read_parquet(q)),
                      "possessions": poss[gid], "rosters": rosters.get(gid)})
    shots = None
    for rel in ("collect/bref_shots/shotchart/2015-16.parquet", "collect/shots/shotchartdetail/2015-16.parquet"):
        if (root / rel).exists():
            shots = pd.read_parquet(root / rel)
            break
    if shots is not None:
        m = heldout_shot_mask(shots)
        dropped["heldout_shot_rows"] = int(m.sum())
        shots = shots[~m].reset_index(drop=True)
    adir = root / "collect" / "tracking" / "aggregates"
    aggregates = None
    if adir.exists() and list(adir.glob("*.parquet")):
        aggregates = pd.concat([pd.read_parquet(p) for p in sorted(adir.glob("*.parquet"))], ignore_index=True)
        if "heldout" in aggregates.columns:
            aggregates = aggregates[~aggregates["heldout"].fillna(False).astype(bool)]
    positions = None
    pdir = root / "collect" / "tracking" / "positions"
    if (root / "collect" / "positions.parquet").exists():
        positions = pd.read_parquet(root / "collect" / "positions.parquet")
    elif pdir.exists() and list(pdir.glob("*.parquet")):
        positions = pd.concat([pd.read_parquet(p) for p in sorted(pdir.glob("*.parquet"))], ignore_index=True)
    if positions is not None:
        positions = normalise_positions(positions, pd.concat(rosters.values(), ignore_index=True) if rosters else None)
    return {"games": games, "shots": shots, "aggregates": aggregates, "positions": positions, "dropped": dropped,
            "source": str(root)}


def collapse_position(value) -> object:
    """basketball-reference position -> G, F or C by the part before the first hyphen (SF-SG -> F, C-PF -> C); NaN otherwise."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return POSITION_MAP.get(str(value).strip().upper().split("-")[0], np.nan)


def normalise_positions(positions: pd.DataFrame, rosters: pd.DataFrame | None) -> pd.DataFrame:
    """A positions table -> (player_id, season, position in G/F/C), one row per player-season.

    The real file (collect/positions.py: player_slug, player, season, position as PG, SG, SF, PF, C, hyphenated) has no
    player_id: it is joined to the section 16 rosters by collect.positions.join_rosters (bref_slug = player_slug, season
    from sportvu_game). A table that already carries player_id is used as is. Positions outside POSITION_MAP are dropped
    with a logged count."""
    pos = positions.copy()
    if "player_id" not in pos.columns and "player_slug" in pos.columns:
        if rosters is None or len(rosters) == 0:
            log.warning("%s: positions carry player_slug but no rosters are present to resolve player_id; positions dropped", WHO)
            return pd.DataFrame(columns=["player_id", "season", "position"])
        if "games" not in pos.columns:
            pos["games"] = 0
        from collect.positions import join_rosters
        pos = join_rosters(pos, rosters[["sportvu_game", "player_id", "bref_slug"]].drop_duplicates())
    out = pos[["player_id", "season", "position"]].copy()
    out["position"] = out["position"].map(collapse_position)
    n_bad = int(out["position"].isna().sum())
    if n_bad:
        log.warning("%s: %d position row(s) outside %s dropped", WHO, n_bad, sorted(POSITION_MAP))
    out = out[out["position"].notna() & out["player_id"].notna()].copy()
    out["player_id"] = pd.to_numeric(out["player_id"]).astype(int)
    out["season"] = out["season"].astype(str)
    return out.drop_duplicates(["player_id", "season"]).reset_index(drop=True)


def fixture_inputs(seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40, root=None) -> dict:
    """Write the analytics.fixtures DATA_ROOT (write_fixture_root) under root, a temporary directory by default, and
    load it through load_inputs, so a fixture run exercises the same loader as the real one."""
    from analytics.fixtures import write_fixture_root
    root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="emb_fixture_"))
    manifest = write_fixture_root(root, seed=seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    inputs = load_inputs(root)
    inputs["fixture"] = {"games": manifest["counts"], "heldout": manifest["heldout"]}
    inputs["fixture_root"] = str(root)
    inputs["source"] = "fixtures (analytics.fixtures, seed %d), written under a temporary directory" % seed
    return inputs


def add_input_args(ap: argparse.ArgumentParser) -> None:
    """The input flags every embeddings CLI shares: the source (--fixtures or --data), the seed and the fixture knobs
    (canonical --fixture-* names; --n-games and --heldout-every are kept as aliases)."""
    ap.add_argument("--fixtures", action="store_true", help="synthetic season from analytics.fixtures (fixture output, not a result)")
    ap.add_argument("--data", help="DATA_ROOT (layout in analytics/embeddings/train.py)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed, also the seed of the fits and the bootstrap (default 0)")
    ap.add_argument("--fixture-games", "--n-games", dest="fixture_games", type=int, default=8,
                    help="--fixtures: games in the synthetic season (default 8)")
    ap.add_argument("--fixture-possessions", dest="fixture_possessions", type=int, default=40,
                    help="--fixtures: possessions per synthetic game (default 40)")
    ap.add_argument("--fixture-heldout-every", "--heldout-every", dest="fixture_heldout_every", type=int, default=4,
                    help="--fixtures: every k-th game is held out, 0 for none (default 4)")


def inputs_from_args(a) -> dict:
    """fixture_inputs or load_inputs, as the parsed add_input_args flags ask."""
    if a.fixtures:
        return fixture_inputs(a.seed, a.fixture_games, a.fixture_heldout_every, a.fixture_possessions)
    return load_inputs(a.data)


# ----------------------------------------------------------------------------- features

def training_games(inputs: dict) -> list:
    """The sorted training game ids of inputs; SystemExit when one is held out (analytics.eval.refuse_heldout)."""
    ids = sorted(g["sportvu_game"] for g in inputs["games"])
    refuse_heldout(ids, WHO)
    return ids


def build_features(inputs: dict, n_min: int = F.N_MIN) -> pd.DataFrame:
    """player_season_features over the training games of ``inputs`` (one game of frames in memory at a time)."""
    ids = training_games(inputs)
    parts, rosters = [], []
    for g in inputs["games"]:
        frames = g["frames"]()
        parts.append(F.tracking_partials(frames, g["possessions"]))
        if g.get("rosters") is not None:
            rosters.append(g["rosters"])
        del frames
    tracking = F.combine_partials(parts)
    shots = inputs.get("shots")
    if shots is not None and len(shots):
        shots = shots[shots["sportvu_game"].astype(str).isin(ids)]
        if heldout_shot_mask(shots).any():
            raise SystemExit("%s: a held-out shot row reached the feature builder" % WHO)
    ros = pd.concat(rosters, ignore_index=True) if rosters else None
    feats = F.player_season_features(tracking, shots, ros, inputs.get("aggregates"), n_min=n_min)
    schema.assert_time_forward(feats, "possessions")
    return feats


# ----------------------------------------------------------------------------- standardise and candidates

def standardise(X: pd.DataFrame) -> tuple:
    """Median imputation then StandardScaler fit on X -> (Z, transform(frame) -> Z, columns)."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = list(X.columns)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = pipe.fit_transform(X.to_numpy(dtype=float))
    Z = np.nan_to_num(Z)   # a constant column scales to 0 (StandardScaler leaves it at 0 already); NaN stays impossible
    return Z, (lambda frame, m=pipe: np.nan_to_num(m.transform(frame[cols].to_numpy(dtype=float)))), cols


def fit_pca(Z: np.ndarray, seed: int = 0) -> dict:
    """PCA to N_DIM components (fewer when the matrix is smaller) -> dict(name, embed(Z) -> E, explained_variance_ratio)."""
    from sklearn.decomposition import PCA
    k = int(min(N_DIM, Z.shape[0], Z.shape[1]))
    m = PCA(n_components=k, random_state=seed).fit(Z)
    return {"name": "pca", "embed": (lambda A, m=m: _pad(m.transform(A))), "explained_variance_ratio": m.explained_variance_ratio_.tolist()}


def _pad(E: np.ndarray) -> np.ndarray:
    """E widened with zero columns to N_DIM (PCA on a small matrix has fewer components)."""
    if E.shape[1] == N_DIM:
        return E
    out = np.zeros((E.shape[0], N_DIM))
    out[:, :E.shape[1]] = E
    return out


def fit_autoencoder(Z: np.ndarray, seed: int = 0, max_iter: int = 400) -> dict:
    """sklearn MLPRegressor reconstructing Z with layers AE_LAYERS (tanh); the bottleneck is the activation of the
    second hidden layer -> dict(name, embed, reconstruction_mse)."""
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(hidden_layer_sizes=AE_LAYERS, activation="tanh", solver="adam", max_iter=max_iter,
                     random_state=seed, learning_rate_init=0.01, early_stopping=False).fit(Z, Z)

    def embed(A, m=m):
        """The bottleneck activation (second hidden layer) of A."""
        h = A
        for W, b in zip(m.coefs_[:2], m.intercepts_[:2]):
            h = np.tanh(h @ W + b)
        return h

    mse = float(np.mean((m.predict(Z) - Z) ** 2))
    return {"name": "autoencoder", "embed": embed, "reconstruction_mse": mse, "n_iter": int(m.n_iter_)}


# ----------------------------------------------------------------------------- metrics

def knn_position_predictions(E: np.ndarray, labels, k: int = K_NN) -> np.ndarray:
    """Leave-one-out k-NN label per row (Euclidean): the majority among the k nearest other rows, ties broken by the
    nearest neighbour among the tied labels."""
    from scipy.spatial.distance import cdist
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    if n < 2:
        return labels.copy()
    D = cdist(E, E)
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)
    out = np.empty(n, dtype=object)
    for i in range(n):
        nn = np.argsort(D[i], kind="stable")[:kk]
        votes = {}
        for rank, j in enumerate(nn):
            lab = labels[j]
            c, first = votes.get(lab, (0, rank))
            votes[lab] = (c + 1, first)
        out[i] = max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
    return out


def knn_position_accuracy(E: np.ndarray, labels, k: int = K_NN) -> float:
    """Share of rows whose leave-one-out k-NN label equals their own (NaN under two rows)."""
    labels = np.asarray(labels, dtype=object)
    if len(labels) < 2:
        return float("nan")
    return float(np.mean(knn_position_predictions(E, labels, k) == labels))


def majority_baseline(labels) -> float:
    """Share of the most frequent label (the accuracy of always guessing it)."""
    labels = pd.Series(list(labels))
    return float(labels.value_counts(normalize=True).iloc[0]) if len(labels) else float("nan")


def r2(y, p) -> float:
    """1 - SSE / SST of p against y (NaN when y is constant)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst == 0 else float(1.0 - np.sum((y - p) ** 2) / sst)


def next_season_predictions(E: np.ndarray, y: np.ndarray, groups, seed: int = 0, n_folds: int = 5) -> dict:
    """Out-of-fold ridge (alpha 1) and fold-mean predictions under grouped K-fold by player -> dict(ridge, mean, folds)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    k = int(min(n_folds, n_groups))
    pr, pm = np.full(len(y), np.nan), np.full(len(y), np.nan)
    if k < 2:
        return {"ridge": pr, "mean": pm, "folds": k}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    for tr, te in GroupKFold(n_splits=k).split(E[order], y[order], groups[order]):
        tr_i, te_i = order[tr], order[te]
        m = Ridge(alpha=1.0).fit(E[tr_i], y[tr_i])
        pr[te_i] = m.predict(E[te_i])
        pm[te_i] = float(np.mean(y[tr_i]))
    return {"ridge": pr, "mean": pm, "folds": k}


def next_season_target(aggregates: pd.DataFrame | None, season: str, next_season: str) -> tuple:
    """(player_id -> value of the first NEXT_SEASON_TARGETS stat present for next_season, its name) or (None, None)."""
    if aggregates is None or len(aggregates) == 0 or "player_id" not in aggregates.columns:
        return None, None
    a = aggregates[aggregates["season"].astype(str) == next_season]
    if "season_type" in a.columns and (a["season_type"] == "Regular Season").any():
        a = a[a["season_type"].fillna("Regular Season") == "Regular Season"]
    for endpoint, metric in NEXT_SEASON_TARGETS:
        m = a[(a["endpoint"].astype(str).str.lower() == endpoint) & (a["metric"].astype(str).str.lower() == metric)]
        if len(m):
            s = m.groupby(pd.to_numeric(m["player_id"]).astype(int))["value"].mean()
            return s, "%s.%s (%s)" % (endpoint, metric, next_season)
    return None, None


def following_season(season: str) -> str:
    """The season after ``season`` ("2015-16" -> "2016-17")."""
    y = int(season[:4]) + 1
    return "%d-%02d" % (y, (y + 1) % 100)


def _interval(df: pd.DataFrame, fn, n_boot: int, seed: int) -> dict:
    """fn on df with a bootstrap interval over players (no interval with one player)."""
    if len(df) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    r = bootstrap_by_game(df, "player_id", fn, n=n_boot, seed=seed)
    if df["player_id"].nunique() < 2:
        r["lo"], r["hi"] = float("nan"), float("nan")
    return r


def evaluate(E: np.ndarray, rows: pd.DataFrame, positions: pd.DataFrame | None, aggregates: pd.DataFrame | None,
             n_boot: int, seed: int, n_folds: int = 5) -> dict:
    """Both metrics for one embedding of ``rows`` (player_id, season aligned to E) with bootstrap intervals by player."""
    out = {"knn": None, "next": None}
    nan = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if positions is not None and len(positions):
        pos = positions.drop_duplicates(["player_id", "season"])[["player_id", "season", "position"]]
        j = rows[["player_id", "season"]].merge(pos, how="left", on=["player_id", "season"])
        m = j["position"].notna().to_numpy()
        if m.sum() >= 2:
            labels = j.loc[m, "position"].astype(str).to_numpy()
            pred = knn_position_predictions(E[m], labels)
            d = pd.DataFrame({"player_id": j.loc[m, "player_id"].to_numpy(), "correct": (pred == labels).astype(float),
                              "label": labels})
            out["knn"] = {"n": int(m.sum()), "accuracy": _interval(d, lambda s: float(s["correct"].mean()), n_boot, seed),
                          "majority": _interval(d, lambda s: majority_baseline(s["label"]), n_boot, seed),
                          "majority_class": pd.Series(labels).value_counts().index[0]}
    seasons = sorted(rows["season"].astype(str).unique())
    if seasons:
        season = seasons[0]
        target, name = next_season_target(aggregates, season, following_season(season))
        if target is not None:
            sel = (rows["season"].astype(str) == season).to_numpy()
            y = rows.loc[sel, "player_id"].map(target).to_numpy(dtype=float)
            has = np.isfinite(y)
            if has.sum() >= 4:
                idx = np.flatnonzero(sel)[has]
                pids = rows["player_id"].to_numpy()[idx]
                p = next_season_predictions(E[idx], y[has], pids, seed=seed, n_folds=n_folds)
                if p["folds"] >= 2:
                    d = pd.DataFrame({"player_id": pids, "y": y[has], "ridge": p["ridge"], "mean": p["mean"]})
                    out["next"] = {"n": int(has.sum()), "target": name, "folds": p["folds"],
                                   "r2": _interval(d, lambda s: r2(s["y"], s["ridge"]), n_boot, seed),
                                   "r2_mean": _interval(d, lambda s: r2(s["y"], s["mean"]), n_boot, seed)}
    if out["knn"] is None:
        out["knn"] = {"n": 0, "accuracy": dict(nan), "majority": dict(nan), "majority_class": None}
    if out["next"] is None:
        out["next"] = {"n": 0, "target": None, "folds": 0, "r2": dict(nan), "r2_mean": dict(nan)}
    return out


# ----------------------------------------------------------------------------- decision and tables

def decide(pca_acc: float, ae_acc) -> dict:
    """The PCA-or-autoencoder rule: the autoencoder is trained only when PCA's 5-NN accuracy is under AE_GATE, and kept
    only when it beats PCA on that metric. NaN PCA accuracy (no positions) keeps PCA. A fifth branch covers an
    autoencoder that was trained but whose accuracy is undefined (ae_acc None or NaN): PCA stands, ae_trained true."""
    if not np.isfinite(pca_acc):
        return {"chosen": "pca", "ae_trained": False, "reason": "no position labels: the gate cannot be evaluated, PCA stands"}
    if pca_acc >= AE_GATE:
        return {"chosen": "pca", "ae_trained": False,
                "reason": "PCA 5-NN accuracy %.3f is at or above the %.2f gate: the autoencoder is not trained" % (pca_acc, AE_GATE)}
    if ae_acc is None or not np.isfinite(ae_acc):
        return {"chosen": "pca", "ae_trained": True, "reason": "autoencoder trained but its accuracy is undefined; PCA stands"}
    if ae_acc > pca_acc:
        return {"chosen": "autoencoder", "ae_trained": True,
                "reason": "PCA 5-NN accuracy %.3f is under the %.2f gate and the autoencoder's %.3f beats it" % (pca_acc, AE_GATE, ae_acc)}
    return {"chosen": "pca", "ae_trained": True,
            "reason": "PCA 5-NN accuracy %.3f is under the %.2f gate but the autoencoder's %.3f does not beat it" % (pca_acc, AE_GATE, ae_acc)}


def embeddings_table(rows: pd.DataFrame, E: np.ndarray, method: str) -> pd.DataFrame:
    """player_embeddings rows of one method: player_id, season, method, e1 .. e8."""
    t = rows[["player_id", "season"]].copy().reset_index(drop=True)
    t["method"] = method
    for i, c in enumerate(EMB_COLS):
        t[c] = E[:, i].astype(float)
    return t


def neighbours_table(rows: pd.DataFrame, E: np.ndarray, method: str, k: int = K_NN) -> pd.DataFrame:
    """embedding_neighbours: the k nearest other player-seasons of every row, rank 1 nearest."""
    from scipy.spatial.distance import cdist
    n = len(rows)
    if n < 2:
        return pd.DataFrame(columns=["player_id", "season", "method", "neighbour_rank", "neighbour_player_id",
                                     "neighbour_season", "distance"])
    D = cdist(E, E)
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)
    pid = rows["player_id"].to_numpy()
    sea = rows["season"].astype(str).to_numpy()
    recs = []
    for i in range(n):
        nn = np.argsort(D[i], kind="stable")[:kk]
        for r, j in enumerate(nn, start=1):
            recs.append((int(pid[i]), sea[i], method, r, int(pid[j]), sea[j], float(D[i, j])))
    return pd.DataFrame(recs, columns=["player_id", "season", "method", "neighbour_rank", "neighbour_player_id",
                                       "neighbour_season", "distance"])


# ----------------------------------------------------------------------------- run

def run(inputs: dict, seed: int = 0, n_boot: int = 200, n_min: int = F.N_MIN, fixture: bool = False, n_folds: int = 5) -> dict:
    """The protocol of the module docstring on loaded inputs -> ctx (features, candidates, decision, tables)."""
    feats = build_features(inputs, n_min)
    cols = F.available_features(feats)
    fit_rows = feats[~feats["insufficient"]].reset_index(drop=True)
    ctx = {"features": feats, "feature_columns": cols, "fixture_root": inputs.get("fixture_root"), "n_rows": int(len(feats)), "n_fit": int(len(fit_rows)),
           "n_insufficient": int(feats["insufficient"].sum()), "n_min": n_min, "seed": seed, "bootstrap": n_boot,
           "fixture": fixture, "folds": n_folds, "games": sorted(g["sportvu_game"] for g in inputs["games"]), "dropped": inputs["dropped"],
           "source": inputs.get("source", ""), "has_aggregates": inputs.get("aggregates") is not None,
           "has_positions": inputs.get("positions") is not None, "candidates": {}, "tables": {}}
    if len(fit_rows) < 3 or not cols:
        ctx["decision"] = {"chosen": None, "ae_trained": False, "reason": "fewer than 3 sufficient player-seasons: nothing fitted"}
        ctx["tables"]["player_season_features"] = feats
        return ctx
    X = fit_rows[cols]
    F.assert_no_leak(X)
    Z, transform, _ = standardise(X)
    pca = fit_pca(Z, seed)
    E_pca = pca["embed"](Z)
    ev_pca = evaluate(E_pca, fit_rows, inputs.get("positions"), inputs.get("aggregates"), n_boot, seed, n_folds)
    ctx["candidates"]["pca"] = {"eval": ev_pca, "explained_variance_ratio": pca["explained_variance_ratio"]}
    embeddings = {"pca": E_pca}
    pca_acc = ev_pca["knn"]["accuracy"]["point"]
    ae_acc = None
    if np.isfinite(pca_acc) and pca_acc < AE_GATE:
        ae = fit_autoencoder(Z, seed)
        E_ae = ae["embed"](Z)
        ev_ae = evaluate(E_ae, fit_rows, inputs.get("positions"), inputs.get("aggregates"), n_boot, seed, n_folds)
        ctx["candidates"]["autoencoder"] = {"eval": ev_ae, "reconstruction_mse": ae["reconstruction_mse"], "n_iter": ae["n_iter"]}
        embeddings["autoencoder"] = E_ae
        ae_acc = ev_ae["knn"]["accuracy"]["point"]
    ctx["decision"] = decide(pca_acc, ae_acc)
    chosen = ctx["decision"]["chosen"]
    order = [chosen] + [m for m in embeddings if m != chosen]
    ctx["tables"]["player_season_features"] = feats
    ctx["tables"]["player_embeddings"] = pd.concat([embeddings_table(fit_rows, embeddings[m], m) for m in order], ignore_index=True)
    ctx["tables"]["embedding_neighbours"] = neighbours_table(fit_rows, embeddings[chosen], chosen)
    return ctx


def write_tables(ctx: dict, out_root) -> dict:
    """OUT_ROOT/analytics/embeddings/<season>/<table>.parquet; returns {table: [paths]}."""
    written = {}
    for name, t in ctx["tables"].items():
        if t is None or len(t) == 0:
            continue
        for season, part in t.groupby("season"):
            d = Path(out_root) / "analytics" / "embeddings" / str(season)
            d.mkdir(parents=True, exist_ok=True)
            p = d / ("%s.parquet" % name)
            part.to_parquet(p, index=False)
            written.setdefault(name, []).append(str(p))
    return written


# ----------------------------------------------------------------------------- report

def write_report(path, ctx: dict, command: str) -> str:
    """Write the markdown report of a run context to path; returns the text. On fixtures every heading of a section
    that carries a number says it is fixture output, not a result."""
    fx = ctx["fixture"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Player style embeddings (analytics/embeddings/train.py)%s" % tag, "",
         "Written %s by `%s`." % (dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command), ""]
    if fx:
        L += ["**Every number below is fixture output, not a result**: the inputs are synthetic games from "
              "analytics.fixtures with planted archetypes. The real run is the Mac command at the end.", ""]
    L += ["## Inputs%s" % tag, "",
          "Source: %s." % ctx["source"],
          "Training games (%d): %s." % (len(ctx["games"]), ", ".join(ctx["games"]) or "none"),
          "Held-out material set aside before any feature: %d frame file(s) under heldout/, %d game id(s) dropped by "
          "sportvu/splits.json or the CLE arena (%s), %d shot row(s) flagged held out, %d game(s) without a possessions table (%s)."
          % (ctx["dropped"]["heldout_frame_files"], len(ctx["dropped"]["heldout_games_by_id"]),
             ", ".join(ctx["dropped"]["heldout_games_by_id"]) or "none", ctx["dropped"]["heldout_shot_rows"],
             len(ctx["dropped"]["games_without_possessions"]), ", ".join(ctx["dropped"]["games_without_possessions"][:5]) or "none"),
          "Guards: `analytics.eval.refuse_heldout` on the training game ids, `analytics.schema.assert_time_forward` on the feature frame.",
          "NBA.com aggregates: %s. Positions table: %s." % ("present" if ctx["has_aggregates"] else "absent (tracking and shot features only)",
                                                          "present" if ctx["has_positions"] else "absent (5-NN position accuracy not computable)"),
          "Player-seasons: %d, of which %d sufficient (at least %d possessions) and %d insufficient%s."
          % (ctx["n_rows"], ctx["n_fit"], ctx["n_min"], ctx["n_insufficient"],
             "; the contract gate is 300, lowered for the short fixture games" if ctx["n_min"] != F.N_MIN else ""),
          "Features used (%d): %s." % (len(ctx["feature_columns"]), ", ".join(ctx["feature_columns"])), ""]
    if ctx["decision"]["chosen"] is None:
        L += ["## Status%s" % tag, "", ctx["decision"]["reason"], "", "    " + MAC_CMD, ""]
        text = "\n".join(L)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        return text
    L += ["## Candidates%s" % tag, "",
          "Protocol: median imputation and standardisation fit on the sufficient rows; PCA to %d components; the autoencoder "
          "(MLPRegressor %s, tanh, bottleneck %d) only when PCA's 5-NN accuracy is under %.2f, kept only when it beats PCA. "
          "Intervals: 2.5 and 97.5 percentiles of %d bootstrap resamples of players (seed %d). The imputer, scaler and embedding "
          "are fit once on all sufficient rows, unsupervised; only the ridge fit and the k-NN vote are out-of-sample."
          % (N_DIM, AE_LAYERS, N_DIM, AE_GATE, ctx["bootstrap"], ctx["seed"]), "",
          "| method | 5-NN position accuracy | majority baseline | next-season R squared (ridge) | mean baseline R squared | note |",
          "| --- | --- | --- | --- | --- | --- |"]
    for m in METHODS:
        c = ctx["candidates"].get(m)
        if c is None:
            L.append("| %s | not trained | | | | gate not reached |" % m)
            continue
        ev = c["eval"]
        note = "explained variance %s" % ", ".join("%.2f" % v for v in c["explained_variance_ratio"]) if m == "pca" \
            else "reconstruction mse %.3f, %d iterations" % (c["reconstruction_mse"], c["n_iter"])
        L.append("| %s | %s | %s | %s | %s | %s |" % (m, fmt_ci(ev["knn"]["accuracy"]), fmt_ci(ev["knn"]["majority"]),
                                                     fmt_ci(ev["next"]["r2"]), fmt_ci(ev["next"]["r2_mean"]), note))
    ev = ctx["candidates"]["pca"]["eval"]
    L += ["", "5-NN: leave-one-out over %d labelled player-seasons, majority class %s. Next season: %s, %d player(s), "
          "out-of-fold under %d grouped folds by player_id." % (ev["knn"]["n"], ev["knn"]["majority_class"], ev["next"]["target"],
                                                                 ev["next"]["n"], ev["next"]["folds"]), "",
          "## Decision%s" % tag, "", "Chosen: **%s**. %s." % (ctx["decision"]["chosen"], ctx["decision"]["reason"]),
          "The gate compares the 5-NN position accuracy reported in the candidate table above (leave-one-out point estimate).", "",
          "## Outputs%s" % tag, "",
          "Tables (docs/contract/embeddings.md): player_season_features (%d rows), player_embeddings (%d rows), embedding_neighbours (%d rows)."
          % (len(ctx["tables"]["player_season_features"]), len(ctx["tables"]["player_embeddings"]), len(ctx["tables"]["embedding_neighbours"]))]
    if ctx.get("written"):
        if ctx.get("out_is_temp"):
            L.append("Written under a temporary directory (no --out given).")
        else:
            L.append("Written under OUT_ROOT/analytics/embeddings/<season>/ (OUT_ROOT is the --out value): %s."
                     % ", ".join(sorted(ctx["written"])))
    L += ["", "## Assumptions (decided without asking; also in the module docstrings)", "",
          "- The attacked basket of a (game, period, team) is the side where the ball spends that team's offensive frames; "
          "court zones are shares of the player's OFFENSIVE frames in a 3 by 2 grid of that half, backcourt frames in the denominator only.",
          "- The handler is the player nearest the ball within 3 ft while the ball is under 8 ft; speed uses the section 4 gap rule (dt <= 2 s) and drops steps over 30 ft/s.",
          "- Shots are restricted to the tracked training games (attempts per possession needs the tracking possessions); a null basketball-reference "
          "shooter id is resolved through the rosters by player_slug = bref_slug; a table without zone_basic gets zones derived from court_x, court_y, shot_type.",
          "- NBA.com aggregates are matched by metric name under the listed endpoints; play-type frequency is poss_pct, or a poss count normalised over the "
          "play types; a missing play type is 0, a missing metric NaN (median-imputed at fit). The 2016-17 file supplies the next-season target only.",
          "- Positions (player_id, season, position in G, F, C) are a needed input at collect/positions.parquet or collect/tracking/positions/<season>.parquet; "
          "without them the 5-NN metric is n/a and PCA stands (the gate cannot be evaluated).",
          "- Insufficient rows (under %d possessions; 300 in the contract) are written to player_season_features but never fitted, embedded or scored." % ctx["n_min"],
          "- The NBA.com season aggregates and the positions table are season-level and include the held-out games' minutes; no "
          "held-out row is fitted, but those season totals are not held-out-free. There is no flag to drop the aggregate block yet: "
          "a tracked-era run without it is the run without collect/tracking/aggregates present (the block is then NaN and unused).",
          "- Fixture runs use analytics.fixtures.write_fixture_root: league rosters persist across games and the aggregates and positions are "
          "keyed by the same players (archetype from the roster slot), so both metrics run on them. Fixture output is never a result.",
          "", "## Mac command", "", "On the owner's Mac, with the full 2015-16 tracking games, possessions, shots, the NBA.com aggregates when "
          "collected and collect/positions.parquet (player_id, season, position):", "", "    " + MAC_CMD, ""]
    if fx:
        L.append("The numbers above are fixture output, not a result.")
    text = "\n".join(L) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return text


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    """The trainer's command line (add_input_args plus the protocol flags)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_input_args(ap)
    ap.add_argument("--report", default="reports/embeddings.md", help="markdown report path (default reports/embeddings.md)")
    ap.add_argument("--out", help="OUT_ROOT for the three tables (a fixture run uses a temporary directory when absent)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by player_id for the next-season ridge (fewer when fewer players)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples of players for every interval (default 200)")
    ap.add_argument("--n-min", type=int, default=None, help="possession gate (300; %d for --fixtures)" % FIXTURE_N_MIN)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """CLI: run the protocol, write the tables and the report; returns the run context."""
    a = parse_args(argv)
    if not a.fixtures and not a.data:
        raise SystemExit("%s: one of --fixtures or --data is required" % WHO)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    inputs = inputs_from_args(a)
    n_min = a.n_min if a.n_min is not None else (FIXTURE_N_MIN if a.fixtures else F.N_MIN)
    ctx = run(inputs, seed=a.seed, n_boot=a.bootstrap, n_min=n_min, fixture=a.fixtures, n_folds=a.folds)
    out_root = a.out if a.out else (tempfile.mkdtemp(prefix="emb_out_") if a.fixtures else None)
    if out_root:
        ctx["written"] = write_tables(ctx, out_root)
        ctx["out_is_temp"] = not bool(a.out)
    cmd = "python -m analytics.embeddings.train " + " ".join(argv if argv is not None else sys.argv[1:])
    text = write_report(a.report, ctx, cmd)
    print(text)
    summary = {k: ctx[k] for k in ("n_rows", "n_fit", "n_insufficient", "decision")}
    summary["written"] = ctx.get("written", {})
    print(json.dumps(summary, indent=1, default=str))
    return ctx


if __name__ == "__main__":
    main()

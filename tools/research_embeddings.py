"""tools/research_embeddings.py: the embeddings research experiment embeddings-E1 (PLAN.md D102, D103).

    python3 -m tools.research_embeddings e1-partials [--artifacts DIR] [--max-minutes 25]
    python3 -m tools.research_embeddings e1-gate     [--artifacts DIR]
    python3 -m tools.research_embeddings e1-dryrun   [--artifacts DIR]
    python3 -m tools.research_embeddings e1-dev      [--artifacts DIR]
    python3 -m tools.research_embeddings e1-final    [--artifacts DIR]

Pre-registration: docs/research/embeddings.md, "embeddings-E1", with its amendment (PLAN.md D103), which fixes every
definition used here before any number existed. In short: does the frozen PCA style embedding find the same player
again in games it was not built from?

e1-partials Reads every development game's tracking frames once, one game at a time, and writes the frozen lane's
            per-game partial sums (analytics.embeddings.features.tracking_partials, unchanged) under
            <artifacts>/research/embeddings_e1/partials/. Resumable: a game already written is skipped, and the run
            stops after --max-minutes so a job stays inside the lock budget. Development games only
            (tools.research_testset.dev_games); the 17 spent held-out games are never read.
e1-gate     The baseline: the frozen features rebuilt from those partials with the frozen functions
            (combine_partials, player_season_features) must equal the stored frozen player_season_features of
            tools.frozen_models (D45) on every row, and the frozen pipeline (standardise, fit_pca) refit on the
            sufficient rows must equal the stored PCA embedding; the 5-NN position accuracy must equal the printed one.
            The partial tables are written once, with their hashes in the gate record.
e1-dryrun   The one-game scoring path run on three development games against the development pool (mechanics only:
            those games are inside the pool, so its numbers mean nothing and are not kept).
e1-dev      The development stage: split-half self-retrieval with its comparators and ablation, the matched
            leave-one-game-out comparator of the one-game test, and Franks discrimination per component. Writes
            reports/research/embeddings-E1-dev.json.
e1-final    The one-game test once on Set A through tools.research_testset.score_once (PLAN.md D103). Runs only on the
            lead's go.

Nothing here writes under DATA_ROOT or reports/ except the research results; the lane's command line is never run.
The only fits are the frozen pipeline (median imputation, standardisation, PCA to 8) on development rows, for the
embedding and for the two feature blocks of the ablation (rule 5: tools/, training games only).
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

WHO = "tools.research_embeddings"
EXPERIMENT = "embeddings-E1"
RESULTS = REPO_ROOT / "reports" / "research"
LANE_REPORT = REPO_ROOT / "reports" / "embeddings.md"
SEASON = "2015-16"
SEED, BOOT, CHUNK = 0, 2000, 100
K = 5
N_MIN = 300                     # the lane's sufficiency rule (analytics.embeddings.features.N_MIN)
ONE_GAME_MIN = 50               # D103: possessions in the one game
TEST_LO_MIN = 0.10              # D103: the lower 95 percent end must be above this and above the position retriever
FEATURE_TOL, EMBED_TOL = 1e-9, 1e-6
SHOTS_FILE = "collect/bref_shots/shotchart/2015-16.parquet"
WORK = "embeddings_e1"
CHECK = "check.json"
SPACES = ("pca", "raw", "movement", "shot")
COMPARATORS = ("raw", "position", "movement", "shot")


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / WORK)


def game_date(game: str) -> tuple:
    month, day, year = (int(x) for x in str(game).split(".")[:3])
    return (year, month, day, str(game))


def _frozen():
    from analytics.embeddings import features as F
    from analytics.embeddings import train as T
    if F.N_MIN != N_MIN:
        raise RuntimeError(f"{WHO}: the frozen sufficiency rule changed")
    return F, T


def feature_order() -> list:
    F, _ = _frozen()
    return list(F.TRACKING_FEATURES) + list(F.SHOT_FEATURES)


def shot_parts_columns() -> list:
    F, _ = _frozen()
    return ["attempts", "threes"] + [f"n_{z}" for z in F.SHOT_ZONES]


# ----------------------------------------------------------------------------- development inputs

def dev_game_list(root: Path) -> list:
    """The development games: frame files with a possessions table, neither spent nor in the D71 test set."""
    from analytics.eval import is_heldout_game
    frames = {p.stem for p in (root / "sportvu_contract").glob("*.parquet") if not p.name.endswith("_pbp.parquet")}
    poss = {p.stem for p in (root / "analytics" / "possessions" / SEASON).glob("*.parquet")}
    games = dev_games(sorted(g for g in frames & poss if not is_heldout_game(g)), WHO)
    training_games_only(games, WHO)
    return games


def partials_pass(artifacts=None, max_minutes: float = 25.0, log=print) -> dict:
    """The frozen per-game tracking partial sums of every development game, one game of frames at a time; resumable."""
    import os
    import pandas as pd
    F, _ = _frozen()
    root = Path(os.environ["DATA_ROOT"])
    games = dev_game_list(root)
    out = work_dir(artifacts) / "partials"
    out.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    for g in games:
        target = out / f"{g}.parquet"
        if target.exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        frames = pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet")
        poss = pd.read_parquet(root / "analytics" / "possessions" / SEASON / f"{g}.parquet")
        part = F.tracking_partials(frames, poss)
        del frames
        part.to_parquet(out / f"{g}.parquet.part", index=False)
        (out / f"{g}.parquet.part").rename(target)
        done += 1
        if log and done % 50 == 0:
            log(f"[research] {done} games this run, {time.time() - started:.0f} s")
    have = sum((out / f"{g}.parquet").exists() for g in games)
    return {"games": len(games), "written_this_run": done, "present": have, "complete": have == len(games),
            "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2)}


def dev_tables(root: Path, games: list) -> dict:
    """Rosters, shots (read with a filter, so no other game's row is materialised) and positions of the development
    games."""
    import pandas as pd
    F, T = _frozen()
    rosters = pd.concat([pd.read_parquet(p) for g in games
                         for p in [root / "analytics" / "rosters" / SEASON / f"{g}.parquet"] if p.exists()],
                        ignore_index=True)
    shots = pd.read_parquet(root / SHOTS_FILE, filters=[("sportvu_game", "in", list(games))])
    if T.heldout_shot_mask(shots).any() or not set(shots["sportvu_game"].astype(str)) <= set(games):
        raise SystemExit(f"{WHO}: a shot row outside the development games was read")
    positions = T.normalise_positions(pd.read_parquet(root / "collect" / "positions.parquet"), rosters)
    return {"rosters": rosters, "shots": shots.reset_index(drop=True), "positions": positions}


def shot_partials(shots, rosters):
    """Per (game, player): attempts, threes and attempts per zone, with the frozen shooter resolution and zones."""
    import pandas as pd
    F, _ = _frozen()
    s = shots.copy()
    s["player_id"] = F.resolve_shooter_ids(s, rosters)
    s = s[s["player_id"].notna()].copy()
    s["player_id"] = s["player_id"].astype(int)
    s["zone"] = F.derive_zone_basic(s)
    s["three"] = s["shot_type"].astype(str).str.upper().str.startswith("3").astype(int)
    out = s.groupby(["sportvu_game", "player_id"]).agg(attempts=("three", "size"), threes=("three", "sum"))
    for z in F.SHOT_ZONES:
        out[f"n_{z}"] = s.assign(_z=(s["zone"] == z).astype(int)).groupby(["sportvu_game", "player_id"])["_z"].sum()
    return out.reset_index()


# ----------------------------------------------------------------------------- features from partial sums

class Parts:
    """Per (game, player) partial sums of tracking and shots, so the frozen features of any weighted set of a
    player's games are ratios of weighted sums. Players are those with a tracking row (the frozen base table)."""

    def __init__(self, track, shots, games):
        import numpy as np
        import pandas as pd
        from scipy import sparse
        F, _ = _frozen()
        self.games = sorted((str(g) for g in games), key=game_date)
        gi = {g: i for i, g in enumerate(self.games)}
        tcols = ["n_possessions"] + list(F.PARTIAL_SUMS)
        scols = shot_parts_columns()
        t = track[track["sportvu_game"].astype(str).isin(gi)][["sportvu_game", "player_id"] + tcols]
        s = shots[shots["sportvu_game"].astype(str).isin(gi)][["sportvu_game", "player_id"] + scols]
        s = s[s["player_id"].isin(set(t["player_id"]))]
        m = t.merge(s, on=["sportvu_game", "player_id"], how="outer").fillna(0.0)
        m["g"] = m["sportvu_game"].astype(str).map(gi)
        m = m.sort_values(["g", "player_id"]).reset_index(drop=True)
        self.players = np.sort(t["player_id"].astype(np.int64).unique())
        self.g = m["g"].to_numpy(dtype=int)
        self.p = np.searchsorted(self.players, m["player_id"].to_numpy(dtype=np.int64))
        self.cols = tcols + scols
        self.M = m[self.cols].to_numpy(dtype=float)
        self.R, self.P, self.G = len(m), len(self.players), len(self.games)
        self.A = sparse.csr_matrix((np.ones(self.R), (np.arange(self.R), self.p)), shape=(self.R, self.P))

    def sums(self, Wr) -> dict:
        """Per-player sums of every column under row weights Wr (B x R) -> {column: B x P}."""
        import numpy as np
        Wr = np.atleast_2d(Wr)
        out = {}
        for j, c in enumerate(self.cols):
            out[c] = np.asarray((self.A.T @ (Wr * self.M[:, j][None, :]).T).T)
        return out

    def game_sums(self) -> dict:
        """{column: G x P} sums of each single game (0 where the player has no row)."""
        import numpy as np
        out = {}
        for j, c in enumerate(self.cols):
            a = np.zeros((self.G, self.P))
            np.add.at(a, (self.g, self.p), self.M[:, j])
            out[c] = a
        return out

    def halves(self) -> "np.ndarray":
        """Half 0 or 1 of each row: each player's games sorted by date and alternated (amendment, commitment 5)."""
        import numpy as np
        half = np.zeros(self.R, dtype=int)
        for p in range(self.P):
            rows = np.flatnonzero(self.p == p)
            order = rows[np.argsort(self.g[rows], kind="stable")]     # self.games is sorted by date
            half[order] = np.arange(len(order)) % 2
        return half


def features_from_sums(S: dict) -> tuple:
    """The 19 frozen features (feature_order) and n_possessions from summed partials of any shape: ratios exactly as
    combine_partials and shot_features form them (a zero denominator gives NaN; no attempt gives NaN shot features)."""
    import numpy as np
    F, _ = _frozen()
    with np.errstate(all="ignore"):
        nz = lambda a: np.where(a == 0, np.nan, a)
        n = S["n_possessions"]
        off = nz(S["off_frames"])
        feats = {"avg_speed_ftps": S["speed_sum"] / nz(S["speed_n"]), "dist_per_poss_ft": S["dist_sum"] / nz(n)}
        for z in F.ZONE_FEATURES:
            feats[z] = S[z + "_n"] / off
        feats["ball_share"] = S["handler_frames"] / off
        feats["nearest_teammate_ft"] = S["teammate_sum"] / nz(S["teammate_n"])
        feats["offball_ball_dist_ft"] = S["offball_sum"] / nz(S["offball_n"])
        att = nz(S["attempts"])
        for z in F.SHOT_ZONES:
            feats[f"shot_share_{z}"] = S[f"n_{z}"] / att
        feats["shot_share_3pt"] = S["threes"] / att
        feats["attempts_per_poss"] = np.where(np.isnan(att), np.nan, S["attempts"] / nz(n))
    return np.stack([feats[c] for c in feature_order()], axis=-1), n


# ----------------------------------------------------------------------------- the frozen fit

def frozen_fit(feats):
    """The frozen pipeline on the sufficient rows of a player_season_features frame: standardise (median imputation,
    StandardScaler) and PCA to 8, plus the same pipeline on the movement and the shot block (the ablation)."""
    F, T = _frozen()
    cols = F.available_features(feats)
    if cols != feature_order():
        raise SystemExit(f"{WHO}: the frozen feature set is not the 19 tracking and shot features: {cols}")
    rows = feats[~feats["insufficient"]].reset_index(drop=True)
    X = rows[cols]
    F.assert_no_leak(X)
    fits = {}
    for name, block in (("full", cols), ("movement", list(F.TRACKING_FEATURES)), ("shot", list(F.SHOT_FEATURES))):
        Z, transform, _ = T.standardise(rows[block])
        pca = T.fit_pca(Z, SEED)
        fits[name] = {"cols": block, "transform": transform, "embed": pca["embed"], "Z": Z, "E": pca["embed"](Z),
                      "explained": pca["explained_variance_ratio"]}
    return {"rows": rows, "fits": fits}


def represent(fit: dict, X) -> dict:
    """Every retrieval space of feature arrays X (... x 19): pca, raw (standardised, no PCA), movement, shot."""
    import numpy as np
    import pandas as pd
    order = feature_order()
    shape = X.shape[:-1]
    flat = pd.DataFrame(X.reshape(-1, X.shape[-1]), columns=order)
    out = {}
    f = fit["fits"]["full"]
    Z = f["transform"](flat)
    out["raw"] = Z.reshape(shape + (Z.shape[1],))
    E = f["embed"](Z)
    out["pca"] = E.reshape(shape + (E.shape[1],))
    for name in ("movement", "shot"):
        b = fit["fits"][name]
        E = b["embed"](b["transform"](flat))
        out[name] = E.reshape(shape + (E.shape[1],))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ----------------------------------------------------------------------------- retrieval

def hits(Q, C, own) -> "np.ndarray":
    """Top-K self-retrieval: query i hits when fewer than K pool rows are strictly closer than its own row own[i]."""
    import numpy as np
    from scipy.spatial.distance import cdist
    if len(Q) == 0:
        return np.zeros(0, dtype=bool)
    D = cdist(Q, C)
    d_own = D[np.arange(len(Q)), own]
    return (D < d_own[:, None]).sum(axis=1) < K


def position_probability(query_pos, pool_pos) -> "np.ndarray":
    """The position-only retriever's exact hit probability per query (amendment, commitment 3)."""
    import numpy as np
    pool_pos = np.asarray(pool_pos, dtype=object)
    n_pool = len(pool_pos)
    out = np.empty(len(query_pos))
    for i, c in enumerate(query_pos):
        if isinstance(c, str):
            n = int((pool_pos == c).sum())
            out[i] = min(K, n) / n if n else min(K, n_pool) / n_pool
        else:
            out[i] = min(K, n_pool) / n_pool
    return out


def positions_of(players, positions) -> "np.ndarray":
    import numpy as np
    pos = positions[positions["season"].astype(str) == SEASON].drop_duplicates("player_id").set_index("player_id")["position"]
    return np.array([pos.get(int(p), np.nan) for p in players], dtype=object)


def cluster_bootstrap(groups, values: dict, n: int = BOOT, seed=SEED) -> dict:
    """Percentile intervals of the mean of each per-query value array, resampling the groups (games or players)
    with replacement; differences are paired because every value uses the same draw."""
    import numpy as np
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n, len(uniq)))
    for b in range(n):
        np.add.at(counts[b], rng.integers(0, len(uniq), len(uniq)), 1)
    w = counts[:, inv]                                  # n x queries
    out = {}
    for k, v in values.items():
        v = np.asarray(v, dtype=float)
        means = (w @ v) / w.sum(axis=1)
        out[k] = {"value": float(v.mean()), "lo": float(np.percentile(means, 2.5)),
                  "hi": float(np.percentile(means, 97.5))}
    return {"groups": int(len(uniq)), "queries": int(len(groups)), "resamples": n, "intervals": out}


def query_values(hit: dict, pos_prob) -> dict:
    """The per-query values every summary reads: hits per space, the position retriever, and the paired differences
    of the embedding against each comparator."""
    import numpy as np
    vals = {s: hit[s].astype(float) for s in SPACES}
    vals["position"] = np.asarray(pos_prob, dtype=float)
    for c in COMPARATORS:
        vals[f"pca_minus_{c}"] = vals["pca"] - vals[c]
    return vals


# ----------------------------------------------------------------------------- the gate

def _stored(artifacts=None) -> dict:
    from tools.frozen_models import load_embeddings
    return load_embeddings(artifacts)


def printed_accuracy(text: str) -> str:
    import re
    m = re.search(r"PCA 5-NN accuracy (\d+\.\d+) is at or above", text)
    if not m:
        raise SystemExit(f"{WHO}: the frozen report does not print the PCA 5-NN accuracy")
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
        raise RuntimeError(f"{EXPERIMENT}: the rebuild did not reproduce the frozen embedding, so it is not used")
    for name, digest in record["files"].items():
        if not (folder / name).exists() or _sha256(folder / name) != digest:
            raise RuntimeError(f"{EXPERIMENT}: {name} changed after the gate record was written")
    return record


def _compare_frames(a, b, cols) -> dict:
    import numpy as np
    x, y = a[cols].to_numpy(dtype=float), b[cols].to_numpy(dtype=float)
    nan = int((np.isnan(x) != np.isnan(y)).sum())
    both = np.isfinite(x) & np.isfinite(y)
    return {"n": int(len(a)), "nan_mismatch": nan,
            "max_abs_diff": float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0}


def gate(artifacts=None) -> dict:
    import os
    import numpy as np
    import pandas as pd
    F, T = _frozen()
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts)
    (folder / CHECK).write_text(json.dumps({"passed": False, "status": "rebuild started"}) + "\n")
    games = dev_game_list(root)
    part_dir = folder / "partials"
    missing = [g for g in games if not (part_dir / f"{g}.parquet").exists()]
    if missing:
        raise SystemExit(f"{WHO}: {len(missing)} development games have no partials yet; run e1-partials")
    parts = [pd.read_parquet(part_dir / f"{g}.parquet") for g in games]
    tables = dev_tables(root, games)
    # the frozen path, unchanged: combine the per-game partials, join the shot mix, flag sufficiency
    tracking = F.combine_partials(parts)
    feats = F.player_season_features(tracking, tables["shots"], tables["rosters"], None, n_min=N_MIN)
    stored = _stored(artifacts)
    s_feats = stored["player_season_features"].sort_values(["player_id", "season"]).reset_index(drop=True)
    feats = feats.sort_values(["player_id", "season"]).reset_index(drop=True)
    order = feature_order()
    keys_equal = bool((feats["player_id"].to_numpy() == s_feats["player_id"].to_numpy()).all()) if len(feats) == len(s_feats) else False
    frozen_vs_stored = _compare_frames(feats, s_feats, order) if keys_equal else {"n": len(feats), "n_expected": len(s_feats)}
    counts_equal = keys_equal and bool((feats["n_possessions"].to_numpy() == s_feats["n_possessions"].to_numpy()).all()
                                       and (feats["insufficient"].to_numpy() == s_feats["insufficient"].to_numpy()).all())
    # the experiment's own path (sums of partials) against the frozen path
    track = pd.concat(parts, ignore_index=True)
    sp = shot_partials(tables["shots"], tables["rosters"])
    P = Parts(track, sp, games)
    X, n = features_from_sums(P.sums(np.ones((1, P.R))))
    mine = pd.DataFrame(X[0], columns=order)
    mine["player_id"] = P.players
    mine["n_possessions"] = n[0]
    mine = mine.sort_values("player_id").reset_index(drop=True)
    mine_vs_frozen = _compare_frames(mine, feats, order) if len(mine) == len(feats) and \
        bool((mine["player_id"].to_numpy() == feats["player_id"].to_numpy()).all()) else {"n": len(mine), "n_expected": len(feats)}
    mine_counts = len(mine) == len(feats) and bool((mine["n_possessions"].to_numpy() == feats["n_possessions"].to_numpy()).all())
    fit = frozen_fit(feats)
    E = fit["fits"]["full"]["E"]
    s_emb = stored["player_embeddings"]
    s_emb = s_emb[s_emb["method"] == "pca"].set_index("player_id").loc[fit["rows"]["player_id"].to_numpy()]
    emb_diff = float(np.max(np.abs(E - s_emb[T.EMB_COLS].to_numpy(dtype=float))))
    labels = positions_of(fit["rows"]["player_id"], tables["positions"])
    lab = np.array([isinstance(x, str) for x in labels])
    acc = T.knn_position_accuracy(E[lab], labels[lab])
    printed = printed_accuracy(LANE_REPORT.read_text())
    acc_ok = abs(acc - float(printed)) <= 0.5 * 10 ** -len(printed.split(".")[1])
    track.to_parquet(folder / "tracking_partials.parquet", index=False)
    sp.to_parquet(folder / "shot_partials.parquet", index=False)
    passed = bool(keys_equal and counts_equal and frozen_vs_stored.get("nan_mismatch") == 0
                  and frozen_vs_stored.get("max_abs_diff", 1) <= FEATURE_TOL and mine_counts
                  and mine_vs_frozen.get("nan_mismatch") == 0 and mine_vs_frozen.get("max_abs_diff", 1) <= FEATURE_TOL
                  and emb_diff <= EMBED_TOL and acc_ok)
    record = {"experiment": EXPERIMENT, "name": "embeddings",
              "compared": "the frozen features rebuilt from per-game partials of the development games with the frozen "
                          "functions, against the stored player_season_features of tools.frozen_models (D45); the "
                          "experiment's sums-of-partials features against the frozen path; the refit PCA embedding "
                          "against the stored one; the 5-NN position accuracy against reports/embeddings.md",
              "games": len(games), "rows": int(len(feats)), "fit_rows": int(len(fit["rows"])),
              "frozen_vs_stored": frozen_vs_stored, "counts_equal": counts_equal,
              "experiment_path_vs_frozen": mine_vs_frozen, "experiment_counts_equal": mine_counts,
              "embedding_max_abs_diff": emb_diff, "tolerances": {"features": FEATURE_TOL, "embedding": EMBED_TOL},
              "knn_accuracy": {"recomputed": acc, "printed": printed, "ok": acc_ok}, "passed": passed}
    _record(folder, record, ["tracking_partials.parquet", "shot_partials.parquet"])
    if not passed:
        print(json.dumps(_plain(record), indent=2, default=str))
        raise SystemExit(f"{WHO}: the rebuild did not reproduce the frozen embedding; nothing downstream runs")
    return require_gate(artifacts)


def load_dev(artifacts=None) -> dict:
    """Everything the statistics read, after the gate: the partials, the frozen features and fit (refit and checked
    again against the stored embedding), and positions."""
    import os
    import numpy as np
    import pandas as pd
    F, T = _frozen()
    require_gate(artifacts)
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts)
    track = pd.read_parquet(folder / "tracking_partials.parquet")
    sp = pd.read_parquet(folder / "shot_partials.parquet")
    games = sorted(track["sportvu_game"].astype(str).unique(), key=game_date)
    training_games_only(games, WHO)
    refuse_test(games, WHO)
    if sorted(games) != dev_game_list(root):
        raise SystemExit(f"{WHO}: the partials are not exactly the development games")
    P = Parts(track, sp, games)
    X, n = features_from_sums(P.sums(np.ones((1, P.R))))
    feats = pd.DataFrame(X[0], columns=feature_order())
    feats.insert(0, "season", SEASON)
    feats.insert(0, "player_id", P.players)
    feats["n_possessions"] = n[0]
    feats["insufficient"] = n[0] < N_MIN
    fit = frozen_fit(feats)
    stored = _stored(artifacts)["player_embeddings"]
    stored = stored[stored["method"] == "pca"].set_index("player_id").loc[fit["rows"]["player_id"].to_numpy()]
    if float(np.max(np.abs(fit["fits"]["full"]["E"] - stored[T.EMB_COLS].to_numpy(dtype=float)))) > EMBED_TOL:
        raise SystemExit(f"{WHO}: the refit no longer reproduces the stored embedding")
    positions = T.normalise_positions(pd.read_parquet(root / "collect" / "positions.parquet"),
                                      dev_tables(root, games)["rosters"])
    return {"P": P, "fit": fit, "feats": feats, "positions": positions, "games": games}


# ----------------------------------------------------------------------------- development statistics

def split_half(d: dict, n: int = BOOT, log=print) -> dict:
    """Commitment 5 and 6: each player's two date-alternated halves retrieve each other; player bootstrap (valid)
    and the pre-registered game bootstrap with rebuilt features."""
    import numpy as np
    P, fit = d["P"], d["fit"]
    half = P.halves()
    masks = [(half == h).astype(float)[None, :] for h in (0, 1)]
    Xh, nh = zip(*(features_from_sums(P.sums(m)) for m in masks))
    ok = (nh[0][0] >= N_MIN) & (nh[1][0] >= N_MIN)
    idx = np.flatnonzero(ok)
    reps = [represent(fit, Xh[h][0][idx]) for h in (0, 1)]
    pos = positions_of(P.players[idx], d["positions"])
    own = np.arange(len(idx))
    hit = {s: np.concatenate([hits(reps[0][s], reps[1][s], own), hits(reps[1][s], reps[0][s], own)]) for s in SPACES}
    pprob = np.concatenate([position_probability(pos, pos)] * 2)
    vals = query_values(hit, pprob)
    players = np.concatenate([P.players[idx]] * 2)
    by_player = cluster_bootstrap(players, vals, n, [SEED, 11])
    # the pre-registered game bootstrap: resample games, rebuild both halves, re-embed with the fixed fit
    rng = np.random.default_rng([SEED, 12])
    draws = {k: [] for k in vals}
    started = time.time()
    for b0 in range(0, n, CHUNK):
        B = min(CHUNK, n - b0)
        Wg = np.zeros((B, P.G))
        for b in range(B):
            np.add.at(Wg[b], rng.integers(0, P.G, P.G), 1)
        Wr = Wg[:, P.g]
        Xb = [features_from_sums(P.sums(Wr * m))[0][:, idx] for m in masks]
        for b in range(B):
            r = [represent(fit, Xb[h][b]) for h in (0, 1)]
            hb = {s: np.concatenate([hits(r[0][s], r[1][s], own), hits(r[1][s], r[0][s], own)]) for s in SPACES}
            v = query_values(hb, pprob)
            for k in draws:
                draws[k].append(float(np.mean(v[k])))
        if log:
            log(f"[research] split-half game bootstrap {b0 + B} of {n}, {time.time() - started:.0f} s")
    game_range = {k: {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))} for k, v in draws.items()}
    return {"players": int(len(idx)), "queries": int(2 * len(idx)), "chance": float(K / len(idx)),
            "players_with_position": int(sum(isinstance(x, str) for x in pos)),
            "rates_with_player_bootstrap": by_player["intervals"],
            "pre_registered_game_bootstrap_range": game_range,
            "game_range_note": "pre-registered (resample games, rebuild the halves, re-embed); a game resample adds "
                               "sampling noise to every multi-game player value, so this range is not read as a "
                               "confidence interval (amendment, commitment 6; PLAN.md D86)",
            "player_bootstrap_note": "valid interval fixed in the amendment: queries resampled by player, pool fixed"}


def one_game_queries(one_sums: dict, pool_rep: dict, pool_players, pool_pos, query_players, query_pos, fit) -> dict:
    """One-game self-retrieval for the queries of one game: query features from that game's sums alone, against a
    pool of full-season (or rest-of-season) representations whose row for each query's player is its own."""
    import numpy as np
    X, _ = features_from_sums(one_sums)
    rep = represent(fit, X)
    where = {int(p): i for i, p in enumerate(pool_players)}
    own = np.array([where[int(p)] for p in query_players])
    hit = {s: hits(rep[s], pool_rep[s], own) for s in SPACES}
    return {"hit": hit, "position": position_probability(query_pos, pool_pos)}


def leave_one_game_out(d: dict, n: int = BOOT, log=print) -> dict:
    """The matched development comparator of the one-game test (D103): each game's players with at least 50
    possessions retrieve their rest-of-season row among every player's rest-of-season rows (pool: at least 300
    rest-of-season possessions); bootstrap of the development games."""
    import numpy as np
    P, fit = d["P"], d["fit"]
    GS = P.game_sums()
    tot = {c: v.sum(axis=0) for c, v in GS.items()}
    pos_all = positions_of(P.players, d["positions"])
    rows = {k: [] for k in ("game", "player")}
    vals = {s: [] for s in SPACES}
    vals["position"] = []
    pool_sizes = []
    started = time.time()
    for g in range(P.G):
        one = {c: GS[c][g] for c in GS}
        rest = {c: tot[c] - GS[c][g] for c in GS}
        Xr, nr = features_from_sums(rest)
        pool = np.flatnonzero(nr >= N_MIN)
        q = np.flatnonzero((one["n_possessions"] >= ONE_GAME_MIN) & (nr >= N_MIN))
        if not len(q):
            continue
        pool_rep = represent(fit, Xr[pool])
        r = one_game_queries({c: one[c][q] for c in one}, pool_rep, P.players[pool], pos_all[pool], P.players[q],
                             pos_all[q], fit)
        for s in SPACES:
            vals[s].append(r["hit"][s].astype(float))
        vals["position"].append(r["position"])
        rows["game"].append(np.full(len(q), g))
        rows["player"].append(P.players[q])
        pool_sizes.append(len(pool))
        if log and (g + 1) % 150 == 0:
            log(f"[research] leave-one-game-out {g + 1} of {P.G}, {time.time() - started:.0f} s")
    games_of = np.concatenate(rows["game"])
    hit = {s: np.concatenate(vals[s]).astype(bool) for s in SPACES}
    v = query_values(hit, np.concatenate(vals["position"]))
    boot = cluster_bootstrap(games_of, v, n, [SEED, 13])
    return {"games_with_queries": int(len(np.unique(games_of))), "queries": int(len(games_of)),
            "players": int(len(np.unique(np.concatenate(rows["player"])))),
            "pool_size": {"mean": float(np.mean(pool_sizes)), "min": int(np.min(pool_sizes)), "max": int(np.max(pool_sizes))},
            "chance": float(np.mean([K / s for s in pool_sizes])),
            "rates_with_game_bootstrap": boot["intervals"],
            "note": "matched to the one-game test (D103); the development fit saw every game, stated not repaired; the "
                    "game bootstrap resamples queries by game and adds no noise to any value"}


def discrimination(d: dict, n: int = BOOT, log=print) -> dict:
    """Franks discrimination per component of the full-season PCA embedding (commitment 7): BV from n game
    resamples (rebuild, re-embed with the fixed fit); interval by a bootstrap of players."""
    import numpy as np
    P, fit = d["P"], d["fit"]
    rows = fit["rows"]
    idx = np.searchsorted(P.players, rows["player_id"].to_numpy(dtype=np.int64))
    E = fit["fits"]["full"]["E"]
    rng = np.random.default_rng([SEED, 14])
    sumsq = np.zeros_like(E)
    s1 = np.zeros_like(E)
    cnt = np.zeros_like(E)
    started = time.time()
    for b0 in range(0, n, CHUNK):
        B = min(CHUNK, n - b0)
        Wg = np.zeros((B, P.G))
        for b in range(B):
            np.add.at(Wg[b], rng.integers(0, P.G, P.G), 1)
        X, _ = features_from_sums(P.sums(Wg[:, P.g]))
        Eb = represent(fit, X[:, idx])["pca"]                    # B x players x 8
        ok = np.isfinite(Eb)
        s1 += np.where(ok, Eb, 0).sum(axis=0)
        sumsq += np.where(ok, Eb ** 2, 0).sum(axis=0)
        cnt += ok.sum(axis=0)
        if log:
            log(f"[research] discrimination resamples {b0 + B} of {n}, {time.time() - started:.0f} s")
    BV = (sumsq - s1 ** 2 / cnt) / (cnt - 1)

    def disc(Xc, bv):
        tot = np.mean((Xc - Xc.mean(axis=0)) ** 2, axis=0)
        return 1 - bv.mean(axis=0) / tot

    point = disc(E, BV)
    prng = np.random.default_rng([SEED, 15])
    draws = np.array([disc(E[i], BV[i]) for i in (prng.integers(0, len(E), len(E)) for _ in range(n))])
    return {"players": int(len(E)), "explained_variance_ratio": fit["fits"]["full"]["explained"],
            "components": [{"component": k + 1, "discrimination": float(point[k]),
                            "lo": float(np.percentile(draws[:, k], 2.5)), "hi": float(np.percentile(draws[:, k], 97.5))}
                           for k in range(E.shape[1])],
            "note": "BV from 2000 game resamples; interval by player bootstrap (amendment, commitment 7); no verdict"}


def dev(artifacts=None) -> dict:
    d = load_dev(artifacts)
    out = {"experiment": EXPERIMENT, "stage": "development games (gate passed); no adoption rule reads this stage",
           "games": len(d["games"]), "player_seasons": int(len(d["feats"])), "fit_rows": int(len(d["fit"]["rows"])),
           "split_half": split_half(d), "leave_one_game_out": leave_one_game_out(d), "discrimination": discrimination(d),
           "one_game_test_rule": {"min_possessions": ONE_GAME_MIN, "lower_end_above": TEST_LO_MIN,
                                  "and_above": "the upper end of the position-only retriever's interval",
                                  "source": "PLAN.md D103"}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return {"split_half": out["split_half"]["rates_with_player_bootstrap"]["pca"],
            "leave_one_game_out": out["leave_one_game_out"]["rates_with_game_bootstrap"]["pca"]}


# ----------------------------------------------------------------------------- the one-game test

def one_game_summary(P_test, d: dict, n: int = BOOT) -> dict:
    """The D103 one-game test on the games of P_test against the full-season development pool (the frozen fit
    rows): queries with at least 50 possessions in their game and a development embedding (commitment 1)."""
    import numpy as np
    fit = d["fit"]
    pool_players = fit["rows"]["player_id"].to_numpy(dtype=np.int64)
    pool_pos = positions_of(pool_players, d["positions"])
    X_pool = fit["rows"][feature_order()].to_numpy(dtype=float)
    pool_rep = represent(fit, X_pool)
    GS = P_test.game_sums()
    pool_set = set(pool_players.tolist())
    hitv = {s: [] for s in SPACES}
    hitv["position"] = []
    games_of = []
    counted = {"player_games": 0, "with_50_possessions": 0, "no_development_embedding": 0}
    for g in range(P_test.G):
        n_g = GS["n_possessions"][g]
        present = n_g > 0
        counted["player_games"] += int(present.sum())
        enough = np.flatnonzero(n_g >= ONE_GAME_MIN)
        counted["with_50_possessions"] += int(len(enough))
        q = np.array([i for i in enough if int(P_test.players[i]) in pool_set], dtype=int)
        counted["no_development_embedding"] += int(len(enough) - len(q))
        if not len(q):
            continue
        r = one_game_queries({c: GS[c][g][q] for c in GS}, pool_rep, pool_players, pool_pos, P_test.players[q],
                             positions_of(P_test.players[q], d["positions"]), fit)
        for s in SPACES:
            hitv[s].append(r["hit"][s])
        hitv["position"].append(r["position"])
        games_of.append(np.full(len(q), g))
    if not games_of:
        return {"games": P_test.G, "counts": counted, "queries": 0, "games_with_queries": 0,
                "pool": int(len(pool_players))}
    hit = {s: np.concatenate(hitv[s]).astype(bool) for s in SPACES}
    v = query_values(hit, np.concatenate(hitv["position"]))
    g_of = np.concatenate(games_of)
    boot = cluster_bootstrap(g_of, v, n, [SEED, 21])
    iv = boot["intervals"]
    rule = {"lower_end": iv["pca"]["lo"], "position_upper_end": iv["position"]["hi"], "threshold": TEST_LO_MIN,
            "passes": bool(iv["pca"]["lo"] > TEST_LO_MIN and iv["pca"]["lo"] > iv["position"]["hi"])}
    return {"games": P_test.G, "counts": counted, "queries": boot["queries"], "games_with_queries": boot["groups"],
            "pool": int(len(pool_players)), "chance": float(K / len(pool_players)),
            "rates_with_game_bootstrap": iv, "rule": rule,
            "pre_score_sensitivity_t_interval_of_game_means": game_mean_t_interval(g_of, v)}


def game_mean_t_interval(groups, values: dict) -> dict:
    """Sensitivity check proposed before the score, reported only (as D80 for xfg-E1): a 95 percent t interval on the
    per-game mean of each value, one value per game, because a percentile bootstrap of few games under-covers. No
    rule reads it; if it and the bootstrap disagree on the rule, the report says so."""
    import numpy as np
    from scipy import stats
    groups = np.asarray(groups)
    out = {}
    for k, v in values.items():
        means = np.array([np.mean(np.asarray(v, dtype=float)[groups == g]) for g in np.unique(groups)])
        half = float(stats.t.ppf(0.975, len(means) - 1) * means.std(ddof=1) / np.sqrt(len(means))) \
            if len(means) > 1 else float("nan")
        out[k] = {"mean_of_game_means": float(means.mean()), "lo": float(means.mean() - half),
                  "hi": float(means.mean() + half), "games": int(len(means))}
    return out


def dryrun(artifacts=None, n_games: int = 3) -> dict:
    """The one-game path on the first n development games, mechanics only (they are inside the pool)."""
    import pandas as pd
    d = load_dev(artifacts)
    games = d["games"][:n_games]
    folder = work_dir(artifacts)
    track = pd.read_parquet(folder / "tracking_partials.parquet")
    sp = pd.read_parquet(folder / "shot_partials.parquet")
    P_small = Parts(track[track["sportvu_game"].isin(games)], sp[sp["sportvu_game"].isin(games)], games)
    s = one_game_summary(P_small, d, n=50)
    return {"games": len(games), "rows": int(P_small.R), "counts": s["counts"], "queries": s["queries"],
            "keys": sorted(s["rates_with_game_bootstrap"]), "note": "mechanics only; these games are in the pool"}


SHOT_COLUMNS = ("sportvu_game", "player_id", "player_slug", "shot_type", "court_x", "court_y", "zone_basic",
                "heldout", "split", "season")    # never the outcome columns (made, and the like)


def clean_head() -> str:
    """HEAD, after checking that the whole working tree is committed (no change, no untracked file), so the score
    belongs to one commit of the code, the frozen code and the development record."""
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def non_finite(value, path: str = "") -> list:
    """Paths of every float in a JSON-like value that is NaN or infinite (the ledger must never hold one)."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in non_finite(v, f"{path}.{k}" if path else str(k))]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in non_finite(v, f"{path}[{i}]")]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def refused(reading: str, **known) -> dict:
    """A refused verdict: what is known (finite only) and why no rule is read."""
    keep = {k: v for k, v in known.items() if not non_finite(v)}
    return {"verdict": {"refused": True, "reading": reading}, **keep}


def make_scorer(d: dict, artifacts=None, provenance: dict = None):
    """The scorer score_once runs: the D103 one-game test on the games it is handed. The test data is reached only
    here, and only while score_once runs a scorer of the tracking source. Before any value is computed, and without
    reading an outcome column, it refuses unless every test game has frames, possessions, a roster and at least one
    shot row, and no shot row is flagged held out. A summary with no query or with any non-finite number is
    returned as a refused verdict, never as NaN."""
    import pandas as pd
    import pyarrow.parquet as pq
    from analytics.embeddings import features as F
    from analytics.embeddings import train as T
    from tools import research_testset as RT
    from tools.research_xfg import research_data_root

    def scorer(test_games):
        if RT._SCORING is None or RT._SCORING[1] != TRACKING:
            raise SystemExit(f"{WHO}: the test data is read only inside the scorer that score_once runs")
        root = research_data_root(artifacts)
        games = sorted(str(g) for g in test_games)
        head = {"experiment": EXPERIMENT, "stage": "the D103 one-game test, scored once on the D71 test set",
                **(provenance or {})}
        paths = {g: {"frames": root / "sportvu_contract" / f"{g}.parquet",
                     "possessions": root / "analytics" / "possessions" / SEASON / f"{g}.parquet",
                     "rosters": root / "analytics" / "rosters" / SEASON / f"{g}.parquet"} for g in games}
        missing = {g: sorted(k for k, q in ps.items() if not q.exists()) for g, ps in paths.items()}
        missing = {g: m for g, m in missing.items() if m}
        if missing:
            return {**head, **refused(f"{len(missing)} test game(s) lack a needed table", missing=missing)}
        names = set(pq.ParquetFile(root / SHOTS_FILE).schema.names)
        shots = pd.read_parquet(root / SHOTS_FILE, columns=[c for c in SHOT_COLUMNS if c in names],
                                filters=[("sportvu_game", "in", games)])
        if T.heldout_shot_mask(shots).any():
            return {**head, **refused("a test shot row is flagged held out")}
        no_shots = sorted(set(games) - set(shots["sportvu_game"].astype(str)))
        if no_shots:
            return {**head, **refused(f"{len(no_shots)} test game(s) have no shot row", games_without_shots=no_shots)}
        parts, rosters = [], []
        for g in games:
            frames = pd.read_parquet(paths[g]["frames"])
            parts.append(F.tracking_partials(frames, pd.read_parquet(paths[g]["possessions"])))
            del frames
            rosters.append(pd.read_parquet(paths[g]["rosters"]))
        sp = shot_partials(shots, pd.concat(rosters, ignore_index=True))
        P_test = Parts(pd.concat(parts, ignore_index=True), sp, games)
        if sorted(P_test.games) != games or len(set(P_test.g.tolist())) != len(games):
            return {**head, **refused("not every test game has tracking partials; scored all or none")}
        summary = one_game_summary(P_test, d)
        if not summary["queries"]:
            return {**head, **refused("no test player qualifies (at least 50 possessions and a development "
                                      "embedding)", counts=summary["counts"], games=summary["games"])}
        bad = non_finite(summary)
        if bad:
            return {**head, **refused(f"{len(bad)} non-finite value(s) in the summary, for example {bad[0]}",
                                      counts=summary["counts"], queries=summary["queries"],
                                      games_with_queries=summary["games_with_queries"])}
        return {**head, **summary}

    return scorer


def final(artifacts=None) -> dict:
    head = clean_head()
    from tools.research_xfg import test_root_ready
    gate_record = require_gate(artifacts)
    dev_path = RESULTS / f"{EXPERIMENT}-dev.json"
    if not dev_path.exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test set is scored after the development result exists")
    d = load_dev(artifacts)
    if not test_root_ready(artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games; run tools.research_ingest first")
    provenance = {"code_commit": head, "gate_files_sha256": gate_record["files"], "dev_json_sha256": _sha256(dev_path)}
    scorer = make_scorer(d, artifacts, provenance)
    out = score_once(EXPERIMENT, TRACKING, scorer)
    rule = {"min_possessions": ONE_GAME_MIN, "lower_end_above": TEST_LO_MIN,
            "and_above": "the upper end of the position-only retriever's interval", "source": "PLAN.md D103"}
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"gate": gate_record, "one_game_test_rule": rule}}
    (RESULTS / f"{EXPERIMENT}-final.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-partials", "e1-gate", "e1-dryrun", "e1-dev", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research embeddings {a.command}"):
        if a.command == "e1-partials":
            out = partials_pass(a.artifacts, a.max_minutes)
        else:
            out = {"e1-gate": gate, "e1-dryrun": dryrun, "e1-dev": dev, "e1-final": final}[a.command](a.artifacts)
    if a.command == "e1-final":
        out = {"rule": out.get("rule"), "verdict": out.get("verdict"), "queries": out.get("queries")}
    print(json.dumps(_plain(out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

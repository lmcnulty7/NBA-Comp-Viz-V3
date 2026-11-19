"""tools/research_matchups_e2.py: the matchups research experiment matchups-E2 (PLAN.md D70, D71, D111, D112).

    python3 -m export.heavy --name matchups-E2 -- python3 -m tools.research_matchups_e2 e2-cache [--max-minutes 25]
    python3 -m export.heavy --name matchups-E2 -- python3 -m tools.research_matchups_e2 e2-dev   [--max-minutes 25]
    python3 -m export.heavy --name matchups-E2 -- python3 -m tools.research_matchups_e2 e2-dryrun
    python3 -m export.heavy --name matchups-E2 -- python3 -m tools.research_matchups_e2 e2-final

Pre-registration: docs/research/matchups.md, section 6, "matchups-E2", with its D112 amendment: the development
out-of-fold result decides; Set A is scored once as a confirmation read as PLAN.md D80; a pre-shot context arm reads
no verdict. It reuses matchups-E1 (tools.research_matchups): its gated assignment pass (the frozen Hungarian
assignment of every development game, equal to the stored matchup table), its part (a) rows, designs, penalized
logistic fits, nested CV, bootstrap and projections.

e2-cache   One pass over the development games, one game's frames at a time (validated as the frozen trainer does):
           at every 1 Hz sampled frame of matchups-E1 (the frozen sampling), one row per defender with the defender's
           position, the five attackers' ids and positions, the ball and the attacked basket, relative to the basket.
e2-dev     EM for the HMM on the development games only, Viterbi assignments (HMM) and per-frame most likely
           assignments (the emission-only ablation), their shares, and matchups-E1 part (a) under the Hungarian, HMM and
           emission-only shares, out of fold, in cached pieces keyed by the code commit. Writes
           reports/research/matchups-E2-dev.json when every piece is done.
e2-dryrun  The Set A path on three development games read from DATA_ROOT (mechanics only).
e2-final   The one Set A score through tools.research_testset.score_once.

The model (Franks, Miller, Bornn and Goldsberry 2015, section 3), fixed in this file before any development number:
- A defender's chain is his run of consecutive sampled frames in one possession while the five attackers stay the
  same (a sampled frame with other than five attackers is left out and breaks the chain). Hidden state: which of the
  five attackers he guards. Emission: his position is normal (isotropic, variance s2) around
  m_k = gO * attacker_k + gB * ball + gH * basket, gO + gB + gH = 1, all >= 0. Transition: stay with probability rho,
  switch to each other attacker with (1 - rho) / 4. Initial state uniform.
- EM on the development chains only: forward-backward E-step in log space; M-step: rho from the expected stays, the
  weights by least squares on the expected states with the convex constraint (the best point on the simplex when the
  free solution leaves it), s2 from the expected residuals. Start: Franks et al.'s means (0.62, 0.11, 0.27), rho 0.9,
  s2 from the start residuals; stop when the log likelihood gains less than 1e-7 of its size, at most EM_MAX_ITER.
- HMM assignment: the Viterbi path of each chain with the fitted parameters. Emission-only ablation: the attacker of
  highest emission likelihood at each frame (the same fitted weights and variance, no transition). Doubles allowed.
- Shares: per (possession, defender, attacker), the frames a defender is assigned to the attacker over the
  possession's frames with five attackers (the denominator of every defender of the possession).
- Pre-shot context (D112): the HMM decoded on each chain cut to the sampled frames at least 2 sampled frames before
  the sampled frame at or before the release of the possession's last attempt (matchups-E1's cut), so no frame at or
  after the release enters the decoding; compared with matchups-E1's pre-shot Hungarian shares on the same rows.
- Metric: the E1 part (a) gain (no-defender minus defender log loss) under each assignment. The rows and the
  no-defender model are the same under every assignment, so the difference of gains, HMM minus Hungarian, is the
  per-row Hungarian defender-model loss minus the HMM defender-model loss. Out of fold under the 5 grouped folds by
  game of matchups-E1 (seed 0), penalties by inner grouped CV per fold. Verdict (pre-registered): the HMM's gain is
  larger, the difference above zero with its 95 percent interval (2000-resample bootstrap of games) wholly above zero.
- Reported beside it: switches per possession per defender and the share of defender-possessions without a switch
  under each assignment, and the defended FG% gap (beyond 4 ft minus within) under each.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools import research_matchups as E1
from tools.frozen_common import REPO_ROOT, _plain, artifacts_dir, assert_safe_output
from tools.research_testset import TRACKING, refuse_test, score_once

WHO = "tools.research_matchups_e2"
EXPERIMENT = "matchups-E2"
RESULTS = E1.RESULTS
SEASON = E1.SEASON
K = 5
START = {"gO": 0.62, "gB": 0.11, "gH": 0.27, "rho": 0.9}
EM_MAX_ITER, EM_TOL = 100, 1e-7
BATCH_CELLS = 4_000_000
CACHE_COLUMNS = ["sportvu_game", "possession", "k", "frame", "def_id", "dx", "dy", "bx", "by"] + \
    [f"o{i}" for i in range(K)] + [f"ox{i}" for i in range(K)] + [f"oy{i}" for i in range(K)]


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / "matchups_e2")


def cache_dir(artifacts=None) -> Path:
    return work_dir(artifacts) / "frames"


# ----------------------------------------------------------------------------- the 1 Hz cache

def frame_rows(frames, info: dict, gid: str):
    """One row per (sampled frame, defender) of matchups-E1's sampling, CACHE_COLUMNS; k is matchups-E1's index of
    the sampled frame (it counts the frames with both squads, as frame_assignments does). Frames with other than five
    attackers are counted in k and left out of the rows."""
    import numpy as np
    import pandas as pd
    f = frames.sort_values("frame", kind="stable")
    fr = f["frame"].to_numpy(dtype=np.int64)
    team = f["team_id"].to_numpy(dtype=np.int64)
    pid = f["player_id"].to_numpy(dtype=np.int64)
    x, y = f["x"].to_numpy(dtype=float), f["y"].to_numpy(dtype=float)
    bxa, bya = f["ball_x"].to_numpy(dtype=float), f["ball_y"].to_numpy(dtype=float)
    out = []
    counts = {"frames": 0, "frames_not_five_attackers": 0}
    for pos, inf in info.items():
        hx, hy = inf["basket"]
        k = 0
        for frame in inf["frames"]:
            a, b = int(np.searchsorted(fr, frame, "left")), int(np.searchsorted(fr, frame, "right"))
            if a == b:
                continue
            is_off = team[a:b] == inf["off_team"]
            o, d = np.flatnonzero(is_off), np.flatnonzero(~is_off)
            if len(o) == 0 or len(d) == 0:
                continue
            counts["frames"] += 1
            if len(o) != K:
                counts["frames_not_five_attackers"] += 1
                k += 1
                continue
            order = np.argsort(pid[a:b][o])
            oi = o[order]
            ball = (bxa[a] - hx, bya[a] - hy)
            for j in d:
                out.append((gid, int(pos), k, int(frame), int(pid[a + j]), x[a + j] - hx, y[a + j] - hy, ball[0], ball[1],
                            *[int(v) for v in pid[a:b][oi]], *(x[a:b][oi] - hx), *(y[a:b][oi] - hy)))
            k += 1
    rows = pd.DataFrame(out, columns=CACHE_COLUMNS)          # positions relative to the attacked basket
    return rows, counts


def clean_frames(g: dict):
    """A game's frames, validated and cleaned as the frozen trainer does (validate.validate_row)."""
    from analytics.matchups import validate as V
    frames = V.frames_of(g)
    tmp = {**g, "frames": frames}
    V.validate_row(tmp)
    frames = tmp["frames"]
    del tmp
    return frames


def cache_pass(artifacts=None, max_minutes: float = 25.0, limit=None) -> dict:
    """e2-cache: every development game not yet cached, one at a time, until max_minutes."""
    import gc
    import json as _json
    import pandas as pd
    root = Path(os.environ["DATA_ROOT"])
    games = E1.development_games(root)
    folder = cache_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    e1_parts = E1.games_dir(artifacts)
    todo = [g for g in games if not (folder / f"{g}.parquet").exists()]
    if limit:
        todo = todo[:limit]
    t0, done = time.time(), 0
    for gid in todo:
        if (time.time() - t0) / 60 > max_minutes:
            break
        (g,) = E1.game_dicts(root, [gid])
        info = E1.info_dict(pd.read_parquet(e1_parts / f"{gid}.info.parquet"))
        frames = clean_frames(g)
        rows, counts = frame_rows(frames, info, gid)
        del frames, g
        part = folder / f"{gid}.parquet.part"
        rows.to_parquet(part, index=False)
        os.replace(part, folder / f"{gid}.parquet")
        (folder / f"{gid}.counts.json").write_text(_json.dumps(counts) + "\n")
        done += 1
        gc.collect()
    left = sum(1 for g in games if not (folder / f"{g}.parquet").exists())
    return {"development_games": len(games), "done_this_run": done, "left": left,
            "seconds": round(time.time() - t0, 1), "peak_gb": round(peak_rss_gb(), 2)}


# ----------------------------------------------------------------------------- chains

class Chains:
    """Defender chains bucketed by length: per bucket Xh (n, T, 2), D (n, T, K, 2), Bh (n, T, 2) relative to the basket,
    and the keys to map decoded states back (row index of each cell, attacker ids per chain)."""

    def __init__(self, rows):
        import numpy as np
        r = rows.sort_values(["sportvu_game", "possession", "def_id", "k"], kind="stable").reset_index(drop=True)
        offs = r[[f"o{i}" for i in range(K)]].to_numpy(dtype=np.int64)
        same = np.r_[False, (r["sportvu_game"].to_numpy()[1:] == r["sportvu_game"].to_numpy()[:-1])
                     & (r["possession"].to_numpy()[1:] == r["possession"].to_numpy()[:-1])
                     & (r["def_id"].to_numpy()[1:] == r["def_id"].to_numpy()[:-1])
                     & (offs[1:] == offs[:-1]).all(axis=1)]
        chain = np.cumsum(~same) - 1
        self.rows = r
        self.chain = chain
        n_chains = int(chain.max()) + 1 if len(chain) else 0
        lengths = np.bincount(chain, minlength=n_chains)
        start = np.r_[0, np.cumsum(lengths)[:-1]]
        Xh = np.c_[r["dx"].to_numpy(), r["dy"].to_numpy()].astype(np.float64)
        Bh = np.c_[r["bx"].to_numpy(), r["by"].to_numpy()].astype(np.float64)
        O = np.stack([np.c_[r[f"ox{i}"].to_numpy(), r[f"oy{i}"].to_numpy()] for i in range(K)], axis=1).astype(np.float64)
        self.buckets = []
        for T in np.unique(lengths):
            ids = np.flatnonzero(lengths == T)
            for part in np.array_split(ids, max(1, int(math.ceil(len(ids) * T * K / BATCH_CELLS)))):
                cells = (start[part][:, None] + np.arange(T)[None, :])          # (n, T) row indices
                self.buckets.append({"T": int(T), "cells": cells, "Xh": Xh[cells], "Bh": Bh[cells], "D": O[cells]})
        self.n_cells = int(len(r))
        self.n_chains = n_chains


def log_emission(b: dict, p: dict):
    """log N(x; gO * o_k + gB * ball, s2 I) relative to the basket (gH * basket is 0 there): (n, T, K)."""
    import numpy as np
    mu = p["gO"] * b["D"] + p["gB"] * b["Bh"][:, :, None, :]
    r2 = ((b["Xh"][:, :, None, :] - mu) ** 2).sum(axis=3)
    return -np.log(2 * np.pi * p["s2"]) - r2 / (2 * p["s2"])


def _log_trans(rho: float):
    import numpy as np
    A = np.full((K, K), math.log((1 - rho) / (K - 1)))
    np.fill_diagonal(A, math.log(rho))
    return A


def forward_backward(logE, rho: float):
    """-> (log likelihood per chain, state posteriors (n, T, K), expected stays per chain)."""
    import numpy as np
    from scipy.special import logsumexp
    n, T, _ = logE.shape
    A = _log_trans(rho)
    al = np.empty_like(logE)
    al[:, 0] = math.log(1 / K) + logE[:, 0]
    for t in range(1, T):
        al[:, t] = logsumexp(al[:, t - 1][:, :, None] + A[None], axis=1) + logE[:, t]
    be = np.zeros_like(logE)
    for t in range(T - 2, -1, -1):
        be[:, t] = logsumexp(A[None] + (logE[:, t + 1] + be[:, t + 1])[:, None, :], axis=2)
    ll = logsumexp(al[:, -1], axis=1)
    post = np.exp(al + be - ll[:, None, None])
    stays = np.zeros(n)
    for t in range(1, T):
        stays += np.exp(al[:, t - 1] + math.log(rho) + logE[:, t] + be[:, t] - ll[:, None]).sum(axis=1)
    return ll, post, stays


def viterbi(logE, rho: float):
    """The most likely state path (n, T)."""
    import numpy as np
    n, T, _ = logE.shape
    A = _log_trans(rho)
    delta = math.log(1 / K) + logE[:, 0]
    back = np.zeros((n, T, K), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, :, None] + A[None]
        back[:, t] = np.argmax(cand, axis=1)
        delta = np.max(cand, axis=1) + logE[:, t]
    path = np.zeros((n, T), dtype=np.int64)
    path[:, -1] = np.argmax(delta, axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[np.arange(n), t, path[:, t]]
    return path


def simplex_weights(S, v):
    """argmin over gO, gB >= 0, gO + gB <= 1 of a 2 x 2 quadratic g'Sg - 2 v'g (gH = 1 - gO - gB)."""
    import numpy as np

    def f(g):
        return float(g @ S @ g - 2 * v @ g)
    cands = []
    try:
        g = np.linalg.solve(S, v)
        if g.min() >= 0 and g.sum() <= 1:
            return g
    except np.linalg.LinAlgError:
        pass
    for i in (0, 1):                                       # edges g_i = 0
        j = 1 - i
        gj = min(max(v[j] / S[j, j], 0.0), 1.0) if S[j, j] > 0 else 0.0
        g = np.zeros(2); g[j] = gj; cands.append(g)
    a = S[0, 0] - 2 * S[0, 1] + S[1, 1]                    # edge gO + gB = 1
    t = min(max(((v[0] - v[1]) - (S[0, 1] - S[1, 1])) / a, 0.0), 1.0) if a > 0 else 0.5
    cands.append(np.array([t, 1 - t]))
    cands.append(np.zeros(2))
    return min(cands, key=f)


def em(chains: Chains, log=print) -> dict:
    """EM on the development chains -> fitted parameters and the log-likelihood path. One E-step per iteration; the
    M-step solves the weights, then the variance from the same posteriors (sufficient statistics), then rho."""
    import numpy as np
    p = dict(START)
    xx = sum(float((b["Xh"] ** 2).sum()) for b in chains.buckets)
    r2 = sum(float(((b["Xh"][:, :, None, :] - (p["gO"] * b["D"] + p["gB"] * b["Bh"][:, :, None, :])) ** 2)
                   .sum(axis=3).mean(axis=2).sum()) for b in chains.buckets)
    p["s2"] = r2 / (2 * chains.n_cells)
    path, prev = [], None
    for it in range(EM_MAX_ITER):
        S, v, stays, trans, ll_total = np.zeros((2, 2)), np.zeros(2), 0.0, 0.0, 0.0
        for b in chains.buckets:
            ll, post, st = forward_backward(log_emission(b, p), p["rho"])
            ll_total += float(ll.sum())
            stays += float(st.sum())
            trans += float(len(ll) * (b["T"] - 1))
            w = post[..., None]
            D, Bh, Xh = b["D"], b["Bh"][:, :, None, :], b["Xh"][:, :, None, :]
            S[0, 0] += float((w * D * D).sum())
            S[0, 1] += float((w * D * Bh).sum())
            v[0] += float((w * D * Xh).sum())
            S[1, 1] += float((b["Bh"] ** 2).sum())           # the posteriors of a cell sum to one
            v[1] += float((b["Bh"] * b["Xh"]).sum())
        S[1, 0] = S[0, 1]
        path.append(ll_total)
        g = simplex_weights(S, v)
        s2 = (xx - 2 * float(g @ v) + float(g @ S @ g)) / (2 * chains.n_cells)
        p.update(gO=float(g[0]), gB=float(g[1]), gH=float(1 - g.sum()), s2=float(s2),
                 rho=float(min(max(stays / trans, 1e-6), 1 - 1e-6)) if trans else p["rho"])
        log(f"[matchups-E2] EM {it}: loglik {ll_total:.1f} {p}", flush=True)
        if prev is not None and abs(ll_total - prev) <= EM_TOL * abs(ll_total):
            break
        prev = ll_total
    return {"params": p, "iterations": len(path), "loglik_path": path, "chains": chains.n_chains,
            "cells": chains.n_cells, "converged": len(path) < EM_MAX_ITER}


def decode(chains: Chains, p: dict, mode: str):
    """Assigned attacker id per cell: mode "hmm" (Viterbi) or "emission" (per-frame argmax) -> rows with def_id,
    off_id, sportvu_game, possession, k."""
    import numpy as np
    r = chains.rows
    off_ids = r[[f"o{i}" for i in range(K)]].to_numpy(dtype=np.int64)
    state = np.full(len(r), -1, dtype=np.int64)
    for b in chains.buckets:
        logE = log_emission(b, p)
        s = viterbi(logE, p["rho"]) if mode == "hmm" else np.argmax(logE, axis=2)
        state[b["cells"].ravel()] = s.ravel()
    out = r[["sportvu_game", "possession", "k", "frame", "def_id"]].copy()
    out["off_id"] = off_ids[np.arange(len(r)), state]
    return out


def shares_of(assigned, rows_all):
    """Shares per (possession, defender, attacker) over the possession's frames with five attackers."""
    denom = rows_all.groupby(["sportvu_game", "possession"])["k"].nunique()
    pair = assigned.groupby(["sportvu_game", "possession", "def_id", "off_id"], as_index=False).size()
    pair["partial_possession"] = pair["size"] / [denom[(g, p)] for g, p in zip(pair["sportvu_game"], pair["possession"])]
    return pair.rename(columns={"def_id": "def_player_id", "off_id": "off_player_id"}).drop(columns=["size"])


def switches(assigned) -> dict:
    """Switches per possession per defender and the share of defender-possessions with none."""
    a = assigned.sort_values(["sportvu_game", "possession", "def_id", "k"])
    same = (a["sportvu_game"].shift() == a["sportvu_game"]) & (a["possession"].shift() == a["possession"]) & \
        (a["def_id"].shift() == a["def_id"])
    a = a.assign(sw=(same & (a["off_id"].shift() != a["off_id"])).astype(int))
    per = a.groupby(["sportvu_game", "possession", "def_id"])["sw"].sum()
    return {"mean_switches": float(per.mean()), "no_switch_share": float((per == 0).mean()),
            "defender_possessions": int(len(per))}


def hungarian_switches(mf) -> dict:
    """The same statistics from matchups-E1's frozen assignment (its changed flags)."""
    per = mf.groupby(["sportvu_game", "possession", "def_player_id"])["changed"].sum(min_count=1).fillna(0)
    return {"mean_switches": float(per.mean()), "no_switch_share": float((per == 0).mean()),
            "defender_possessions": int(len(per))}


def defended_fg(assigned, rows_all, shots, n: int = E1.BOOT, seed: int = E1.SEED) -> dict:
    """FG% of attempts whose assigned defender (the nearest when several guard the shooter) at the sampled frame at or
    before the release is within 4 ft against beyond, and their paired difference by game."""
    import numpy as np
    import pandas as pd
    s = shots.dropna(subset=["shooter_id"])[["sportvu_game", "possession", "event", "k", "shooter_id", "made"]].copy()
    s["sportvu_game"] = s["sportvu_game"].astype(str)
    a = assigned.merge(s, left_on=["sportvu_game", "possession", "k", "off_id"],
                       right_on=["sportvu_game", "possession", "k", "shooter_id"], how="inner")
    pos = rows_all[["sportvu_game", "possession", "k", "def_id", "dx", "dy"] + [f"o{i}" for i in range(K)]
                   + [f"ox{i}" for i in range(K)] + [f"oy{i}" for i in range(K)]]
    a = a.merge(pos, on=["sportvu_game", "possession", "k", "def_id"], how="left")
    o = a[[f"o{i}" for i in range(K)]].to_numpy()
    j = (o == a["off_id"].to_numpy()[:, None]).argmax(axis=1)
    ox = a[[f"ox{i}" for i in range(K)]].to_numpy()[np.arange(len(a)), j]
    oy = a[[f"oy{i}" for i in range(K)]].to_numpy()[np.arange(len(a)), j]
    a["dist"] = np.hypot(a["dx"] - ox, a["dy"] - oy)
    near = a.sort_values("dist").drop_duplicates(["sportvu_game", "event"])
    near["far"] = near["dist"] > 4.0
    g = near.groupby(["sportvu_game", "far"])["made"].agg(["sum", "count"]).unstack(fill_value=0)
    games = g.index.to_numpy()
    sn, cn = g[("sum", False)].to_numpy(float), g[("count", False)].to_numpy(float)
    sf, cf = g[("sum", True)].to_numpy(float), g[("count", True)].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), size=(n, len(games)))
    diff = sf[idx].sum(1) / cf[idx].sum(1) - sn[idx].sum(1) / cn[idx].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    del pd
    return {"attempts_within_4ft": int(cn.sum()), "attempts_beyond": int(cf.sum()),
            "fg_within": float(sn.sum() / cn.sum()), "fg_beyond": float(sf.sum() / cf.sum()),
            "beyond_minus_within": {"point": float(sf.sum() / cf.sum() - sn.sum() / cn.sum()),
                                    "ci95": [float(lo), float(hi)]}}


# ----------------------------------------------------------------------------- development

def load_cache(artifacts, games):
    import pandas as pd
    folder = cache_dir(artifacts)
    missing = [g for g in games if not (folder / f"{g}.parquet").exists()]
    if missing:
        raise SystemExit(f"{WHO}: run e2-cache first; {len(missing)} game(s) are not cached")
    r = pd.concat([pd.read_parquet(folder / f"{g}.parquet") for g in games], ignore_index=True)
    refuse_test(r["sportvu_game"].unique(), WHO)
    return complete_cells(r)


POSITION_COLUMNS = ["dx", "dy", "bx", "by"] + [f"ox{i}" for i in range(K)] + [f"oy{i}" for i in range(K)]


def complete_cells(rows):
    """The cells with every position known (a sampled frame without the ball or a player position is left out, as a
    frame without five attackers is); the count left out is kept in rows.attrs."""
    ok = rows[POSITION_COLUMNS].notna().all(axis=1)
    out = rows[ok].reset_index(drop=True)
    out.attrs["cells_left_out_missing_position"] = int((~ok).sum())
    return out


def pre_shot_cut(rows, shots):
    """The cache rows of each possession at least E1.PRE_SHOT_GAP sampled frames before the sampled frame at or before
    the release of its last attempt (matchups-E1's cut)."""
    sh = shots.sort_values(["sportvu_game", "possession", "event"])
    last_k = sh.groupby([sh["sportvu_game"].astype(str), "possession"])["k"].last() - E1.PRE_SHOT_GAP
    lim = [last_k.get((g, p), -1) for g, p in zip(rows["sportvu_game"], rows["possession"])]
    keep = (rows["k"].to_numpy() <= lim) & (pd_array(lim) >= 0)
    return rows[keep].reset_index(drop=True)


def pd_array(x):
    import numpy as np
    return np.asarray(x)


class DevE2:
    """The development tables, built once per run and only when a piece needs them."""

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self._base = self._cache = False

    def base(self):
        if self._base:
            return self
        from analytics.matchups import train as T
        root = Path(os.environ["DATA_ROOT"])
        self.games = E1.development_games(root)
        xfg = E1.stored_xfg(root)
        self.mp, self.shots, zones = E1.tables(self.artifacts, self.games, xfg, T.default_probability(xfg))
        self.a_rows, self.a_shares, self.a_counts = E1.part_a_rows(self.mp, self.shots, zones)
        self.pre_rows, self.pre_shares = E1.pre_shot_shares(E1.stored_mf_shots(self.artifacts, self.games), self.a_rows)
        self._base = True
        return self

    def cache(self):
        if self._cache:
            return self
        self.base()
        self.rows = load_cache(self.artifacts, self.games)
        self.pre_cells = pre_shot_cut(self.rows, self.shots)
        self._cache = True
        return self


def assignments(d: DevE2, params: dict, folder: Path) -> dict:
    """HMM and emission-only assignments of every development cell, and the HMM of the pre-shot cut, written once."""
    import pandas as pd
    out = {}
    for name in ("hmm", "emission", "hmm_pre"):
        f = folder / f"assigned_{name}.parquet"
        if not f.exists():
            d.cache()
            ch = Chains(d.pre_cells if name == "hmm_pre" else d.rows)
            decode(ch, params, "emission" if name == "emission" else "hmm").to_parquet(f, index=False)
        out[name] = pd.read_parquet(f)
    return out


def designs(d: DevE2, assigned: dict) -> dict:
    """Part (a) designs on the same rows under every assignment (the no-defender model is shared)."""
    d.cache()
    sh = {"hmm": shares_of(assigned["hmm"], d.rows), "emission": shares_of(assigned["emission"], d.rows),
          "hmm_pre": shares_of(assigned["hmm_pre"], d.pre_cells)}
    keep = lambda t, rows: t.merge(rows[["sportvu_game", "possession", "off_player_id"]], how="inner",
                                   on=["sportvu_game", "possession", "off_player_id"])
    return {"hungarian": E1.Design("a", d.a_rows, d.a_shares),
            "hmm": E1.Design("a", d.a_rows, keep(sh["hmm"], d.a_rows)),
            "emission": E1.Design("a", d.a_rows, keep(sh["emission"], d.a_rows)),
            "no_defender": E1.Design("a", d.a_rows, d.a_shares, with_defenders=False),
            "pre_hungarian": E1.Design("a", d.pre_rows, d.pre_shares),
            "pre_hmm": E1.Design("a", d.pre_rows, keep(sh["hmm_pre"], d.pre_rows)),
            "pre_no_defender": E1.Design("a", d.pre_rows, d.pre_shares, with_defenders=False)}


MODELS = ("hungarian", "hmm", "emission", "no_defender", "pre_hungarian", "pre_hmm", "pre_no_defender")
PIECES = ("em",) + tuple(f"oof_{m}" for m in MODELS) + tuple(f"lambda_{m}" for m in MODELS) + ("reported",)


def dev(artifacts=None, max_minutes: float = 25.0, pieces_commit=None) -> dict:
    """The development stage in cached pieces keyed by the code commit; the run that finds every piece done writes
    reports/research/matchups-E2-dev.json. pieces_commit names an earlier commit whose cached pieces are used (when
    only code outside the pieces changed since); the record names both commits."""
    import numpy as np
    t0 = time.time()
    head = E1.clean_head()
    gate = E1.require_gate(artifacts)
    cache = work_dir(artifacts) / "dev_cache" / (pieces_commit or head)
    cache.mkdir(parents=True, exist_ok=True)
    d = DevE2(artifacts)
    D = None
    for name in PIECES:
        f = cache / (f"{name}.npz" if name.startswith("oof_") else f"{name}.json")
        if f.exists():
            continue
        if (time.time() - t0) / 60 > max_minutes:
            return {"status": "stopped for the lock budget; run e2-dev again to resume",
                    "done": sorted(p.name for p in cache.iterdir()), "seconds": round(time.time() - t0, 1)}
        if name == "em":
            d.cache()
            res = em(Chains(d.rows))
            f.write_text(json.dumps(_plain(res), indent=2, allow_nan=False) + "\n")
        else:
            params = json.loads((cache / "em.json").read_text())["params"]
            if D is None:
                D = designs(d, assignments(d, params, cache))
            if name.startswith("oof_"):
                m = name[4:]
                r = E1.oof(D[m])
                np.savez(f, z=r["z"], y=D[m].y, groups=np.asarray(D[m].groups, dtype=str),
                         fold_lambdas=np.array(r["fold_lambdas"]))
            elif name.startswith("lambda_"):
                m = name[7:]
                lam, curve = E1.choose_lambda(D[m], np.arange(len(D[m].y)))
                f.write_text(json.dumps(_plain({"lambda": lam, "curve": curve}), indent=2) + "\n")
            else:
                f.write_text(json.dumps(_plain(reported(d, assignments(d, params, cache))), indent=2,
                                        allow_nan=False) + "\n")
        print(f"[matchups-E2] piece {name} done {round(time.time() - t0)} s", flush=True)
    out = assemble(cache, head, gate)
    out["pieces_commit"] = pieces_commit or head
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def reported(d: DevE2, assigned: dict) -> dict:
    """Switches and the defended FG% gap under each assignment (reported beside the verdict, never read by it)."""
    import pandas as pd
    d.cache()
    mf = pd.concat([pd.read_parquet(E1.games_dir(d.artifacts) / f"{g}.mf.parquet") for g in d.games], ignore_index=True)
    hung = {"sportvu_game": mf["sportvu_game"].astype(str), "possession": mf["possession"], "k": mf["k"],
            "def_id": mf["def_player_id"], "off_id": mf["off_player_id"]}
    hung = pd.DataFrame(hung)
    return {"switches": {"hungarian": hungarian_switches(mf), "hmm": switches(assigned["hmm"]),
                         "emission": switches(assigned["emission"])},
            "defended_fg": {"hungarian": defended_fg(hung, d.rows, d.shots), "hmm": defended_fg(assigned["hmm"], d.rows, d.shots),
                            "emission": defended_fg(assigned["emission"], d.rows, d.shots)}}


def losses(cache: Path, m: str):
    import numpy as np
    z = np.load(cache / f"oof_{m}.npz", allow_pickle=True)          # our own file: the game ids are strings
    return E1.row_loss(z["y"], z["z"]), z["groups"].astype(str), z["fold_lambdas"]


def assemble(cache: Path, head: str, gate: dict) -> dict:
    """The development record from the cached pieces."""
    import numpy as np
    L, G, lam = {}, {}, {}
    for m in MODELS:
        L[m], G[m], lam[m] = losses(cache, m)
    for a, b in (("hungarian", "hmm"), ("hungarian", "emission"), ("hungarian", "no_defender"),
                 ("pre_hungarian", "pre_hmm"), ("pre_hungarian", "pre_no_defender")):
        if not (G[a] == G[b]).all():
            raise SystemExit(f"{WHO}: the rows of {a} and {b} differ; the comparison is not paired")
    main = E1.game_bootstrap(G["hungarian"], {
        "gain_hungarian": L["no_defender"] - L["hungarian"], "gain_hmm": L["no_defender"] - L["hmm"],
        "gain_emission": L["no_defender"] - L["emission"],
        "hmm_minus_hungarian": L["hungarian"] - L["hmm"],
        "emission_minus_hungarian": L["hungarian"] - L["emission"],
        "hmm_minus_emission": L["emission"] - L["hmm"]})
    pre = E1.game_bootstrap(G["pre_hungarian"], {
        "gain_hungarian": L["pre_no_defender"] - L["pre_hungarian"], "gain_hmm": L["pre_no_defender"] - L["pre_hmm"],
        "hmm_minus_hungarian": L["pre_hungarian"] - L["pre_hmm"]})
    d = main["hmm_minus_hungarian"]
    em_rec = json.loads((cache / "em.json").read_text())
    return {"experiment": EXPERIMENT, "code_commit": head,
            "stage": "development games, out of fold under matchups-E1's 5 grouped folds by game (seed 0); no Set A or "
                     "held-out row read",
            "matchups_e1_gate_passed": gate["passed"],
            "em": {k: em_rec[k] for k in ("params", "iterations", "chains", "cells", "converged")},
            "em_loglik_first_last": [em_rec["loglik_path"][0], em_rec["loglik_path"][-1]],
            "rows": main["rows"], "games": main["games"],
            "fold_lambdas": {m: [float(x) for x in lam[m]] for m in MODELS},
            "gains": {k: main[k] for k in ("gain_hungarian", "gain_hmm", "gain_emission")},
            "hmm_minus_hungarian": d,
            "ablation": {"emission_minus_hungarian": main["emission_minus_hungarian"],
                         "hmm_minus_emission": main["hmm_minus_emission"]},
            "pre_shot_context": {"what": "context, reads no verdict (D112): pre-shot shares under both assignments",
                                 "rows": pre["rows"], **{k: pre[k] for k in ("gain_hungarian", "gain_hmm",
                                                                             "hmm_minus_hungarian")}},
            "projected_set_a": E1.projected_half_width(G["hungarian"], L["hungarian"] - L["hmm"]),
            "reported": json.loads((cache / "reported.json").read_text()),
            "test_fit": {m: json.loads((cache / f"lambda_{m}.json").read_text())["lambda"] for m in MODELS},
            "verdict": {"rule": ("the E1 part (a) gain under the HMM larger than under the Hungarian assignment: the "
                                 "difference above zero with its 95 percent interval wholly above zero, on the "
                                 "development games (D112); Set A is a later confirmation read as D80"),
                        "passes": bool(d["point"] > 0 and d["ci95"][0] > 0)},
            "peak_gb": round(peak_rss_gb(), 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e2-cache", "e2-dev"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pieces-commit", default=None, help="e2-dev: use the cached pieces of this earlier commit")
    a = ap.parse_args(argv)
    with heavy_job(f"research matchups-E2 {a.command}"):
        if a.command == "e2-cache":
            out = cache_pass(a.artifacts, a.max_minutes, a.limit)
        else:
            out = dev(a.artifacts, a.max_minutes, a.pieces_commit)
    print(json.dumps(_plain(out.get("verdict", out)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

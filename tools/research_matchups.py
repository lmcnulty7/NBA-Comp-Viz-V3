"""tools/research_matchups.py: the matchups research experiment matchups-E1 (PLAN.md D70, D71, D77, D106, D108).

    python3 -m export.heavy --name matchups-E1 -- python3 -m tools.research_matchups e1-assign [--max-minutes 25]
    python3 -m export.heavy --name matchups-E1 -- python3 -m tools.research_matchups e1-gate
    python3 -m export.heavy --name matchups-E1 -- python3 -m tools.research_matchups e1-dryrun
    python3 -m export.heavy --name matchups-E1 -- python3 -m tools.research_matchups e1-dev
    python3 -m export.heavy --name matchups-E1 -- python3 -m tools.research_matchups e1-final

Pre-registration: docs/research/matchups.md, section 6, "matchups-E1", with its D108 amendment: the development
out-of-fold result decides each part; Set A is scored once as a confirmation read as PLAN.md D80.

e1-assign   Resumable, one game at a time (frames read once and dropped): for every development game
            (tools.research_testset.dev_games; the 17 spent held-out games are never loaded, their directory is
            never listed) the frozen assignment exactly as the frozen trainer runs it (analytics.matchups.train
            assign_games with validate=True: validate.validate_row, features.frame_assignments,
            features.possession_shots), plus each attacker's court-zone mix over the possession's sampled frames
            (positions only). Writes per-game parquet files under <artifacts>/research/matchups_e1/games/.
e1-gate     Gate A: the matchup_possessions rebuilt from those parts with the stored out-of-fold xFG equal the
            stored matchup_possessions_train table of every game. Gate B: the frozen ridge_cv refit on them
            reproduces the stored ridge_value of every defender and the out-of-fold MSEs printed in
            reports/matchups.md. Writes gate.json.
e1-dryrun   The Set A path (the scorer's own code: load, assign, xFG from the frozen model, rows, losses) on three
            development games read from DATA_ROOT, mechanics only.
e1-dev      Parts (a) and (b), the ablation and the stability check on the development games, out of fold under 5
            grouped folds by game (seed 0), penalties by inner grouped CV on each fold's training games; the
            projected Set A half-widths (D108). Writes reports/research/matchups-E1-dev.json.
e1-final    The one score on Set A through tools.research_testset.score_once. Writes
            reports/research/matchups-E1-final.json.

Definitions, fixed in this file before any development number (the pre-registration leaves them open):
- Part (a), who shoots. Rows: one per (possession, attacker) for every possession whose last field-goal attempt has a
  known shooter (the frozen lane's attribution of a possession's attempt). Target: this attacker took that
  attempt. Columns: the attacker's indicator (1), the defenders' shares of the possession's sampled frames spent
  guarding this attacker (the frozen partial_possession), and the attacker's court-zone mix over the possession's
  sampled frames (shares in the zones of analytics.epv.features.zone_of: rim, paint, mid, three, by distance to the
  attacked basket). No-defender model: the same without the defender columns. Both have a free intercept.
- Part (b), make given xFG. Rows: field-goal attempts with an assigned defender at the sampled frame at or before
  release (the frozen possession_shots) and an xFG probability. Model: logit P(make) = logit(xFG) + b0 + the
  assigned defender's coefficient. No-defender model: logit(xFG) + b0. On development games xFG is the stored
  out-of-fold p_selected (fold >= 0); on Set A it is the frozen xFG model (tools.frozen_models.load_xfg, fitted on
  the training games, which never read Set A) on the frozen features of the test attempts.
- Penalized logistic regression: L2 on every column but the intercept, one penalty per model, chosen by inner
  grouped CV (5 folds by game, seed 0) on each outer fold's training games over LOG_C_GRID, by out-of-fold log loss
  (ties to the stronger penalty); fitted by L-BFGS to a gradient tolerance of 1e-8.
- Metric: mean out-of-fold log loss per row; the paired difference defender model minus no-defender model with a
  2000-resample bootstrap of games (seed 0). A part passes when the difference is below zero with its interval
  wholly below zero.
- Ablation: the frozen possession design (defender and attacker share columns) on possession points minus the
  xFG-expected points of the possession's attempts, ridge with the frozen nested alpha grid, against the same
  design without defender columns; out-of-fold MSE, paired.
- Stability, reported only: defender coefficients of each part fitted on odd and on even development games (in
  date order), Spearman over defenders with at least 300 distinct possessions (a) or 100 attempts (b) in each half.
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
from tools.research_testset import TRACKING, dev_games, refuse_test, score_once

WHO = "tools.research_matchups"
EXPERIMENT = "matchups-E1"
RESULTS = REPO_ROOT / "reports" / "research"
SEASON = "2015-16"
SEED, BOOT, FOLDS = 0, 2000, 5
ZONES = ("rim", "paint", "mid", "three")


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / "matchups_e1")


def games_dir(artifacts=None) -> Path:
    return work_dir(artifacts) / "games"


# ----------------------------------------------------------------------------- loading

def game_dicts(root, games) -> list:
    """Game dicts in the frozen layout (analytics.matchups.validate.load_games, the training directory only) for the
    given game ids; frames stay a loader. Nothing under a heldout/ directory is listed or read."""
    import pandas as pd
    from analytics import schema
    from analytics.eval import is_heldout_game
    from analytics.matchups import validate as V
    root = Path(root)
    out = []
    for gid in games:
        if is_heldout_game(gid):
            raise SystemExit(f"{WHO}: {gid} is a spent held-out game")
        pq = root / "analytics" / "possessions" / SEASON / f"{gid}.parquet"
        sv = root / "sportvu_contract"
        if not pq.exists() or not (sv / f"{gid}.parquet").exists():
            raise SystemExit(f"{WHO}: {gid} lacks its possessions or frames under the root")
        g = {"sportvu_game": gid, "heldout": False, "frames": V._frames_loader(sv / f"{gid}.parquet"),
             "events": V._read(sv / f"{gid}_pbp.parquet"), "possessions": pd.read_parquet(pq)}
        for table in ("shot_frames", "rosters", "rebound_frames"):
            t = V._read(root / "analytics" / table / SEASON / f"{gid}.parquet")
            g[table] = t if t is not None else schema.empty(table)
        out.append(g)
    return out


def development_games(root) -> list:
    """The development games: games with a possessions table in the training directory, through dev_games."""
    names = sorted(p.stem for p in (Path(root) / "analytics" / "possessions" / SEASON).glob("*.parquet"))
    return dev_games(names, WHO)


# ----------------------------------------------------------------------------- the per-game assignment pass

def zone_mix(frames, info: dict):
    """(possession, attacker) -> shares of the possession's sampled frames in each zone of ZONES (distance to the
    attacked basket, analytics.epv.features.zone_of), positions only."""
    import numpy as np
    import pandas as pd
    from analytics.epv.features import zone_of
    from analytics.matchups import features as F
    Fr = F._Frames(frames)
    rows = []
    for pos, inf in info.items():
        bx, by = inf["basket"]
        counts = {}
        n = 0
        for fr in inf["frames"]:
            s = Fr.rows(int(fr))
            o = np.flatnonzero(Fr.team[s] == inf["off_team"])
            if len(o) == 0:
                continue
            n += 1
            d = np.hypot(Fr.x[s][o] - bx, Fr.y[s][o] - by)
            for pid, dist in zip(Fr.pid[s][o], d):
                z = zone_of(float(dist))
                key = (int(pid), z)
                counts[key] = counts.get(key, 0) + 1
        if n == 0:
            continue
        players = {p for p, _z in counts}
        for p in players:
            rows.append((int(pos), p, *[counts.get((p, z), 0) / n for z in ZONES]))
    return pd.DataFrame(rows, columns=["possession", "off_player_id"] + [f"zone_{z}" for z in ZONES])


def assign_game(g: dict) -> dict:
    """The frozen trainer's per-game step (analytics.matchups.train.assign_games with validate=True) on one game
    dict, plus the zone mix from the same frames -> the parts written per game."""
    import pandas as pd
    from analytics.matchups import features as F
    from analytics.matchups import validate as V
    frames = V.frames_of(g)
    tmp = {**g, "frames": frames}
    V.validate_row(tmp)
    frames = tmp["frames"]
    del tmp
    mf, info, counts = F.frame_assignments(frames, g["possessions"])
    shots = F.possession_shots(g, mf, info, None, frames=frames)
    zones = zone_mix(frames, info)
    del frames
    gid = str(g["sportvu_game"])
    inf = pd.DataFrame([{"possession": int(p), "n_sampled": v["n_sampled"], "basket_x": v["basket"][0],
                         "basket_y": v["basket"][1], "period": v["period"], "start_event": v["start_event"],
                         "end_event": v["end_event"], "points": v["points"], "off_team": v["off_team"],
                         "frames": [int(x) for x in v["frames"]]} for p, v in info.items()])
    zones.insert(0, "sportvu_game", gid)
    return {"mf": mf, "info": inf, "shots": shots, "zones": zones, "counts": {"sportvu_game": gid, **counts}}


def info_dict(inf) -> dict:
    """The per-possession info of frame_assignments back from its parquet form."""
    import numpy as np
    return {int(r.possession): {"n_sampled": int(r.n_sampled), "basket": (float(r.basket_x), float(r.basket_y)),
                                "period": int(r.period), "start_event": int(r.start_event), "end_event": int(r.end_event),
                                "points": int(r.points), "off_team": int(r.off_team),
                                "frames": np.asarray(r.frames, dtype=np.int64)} for r in inf.itertuples(index=False)}


PART_NAMES = ("mf", "info", "shots", "zones")


def write_parts(folder: Path, gid: str, parts: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k in PART_NAMES:
        part = folder / f"{gid}.{k}.parquet.part"
        parts[k].to_parquet(part, index=False)
        os.replace(part, folder / f"{gid}.{k}.parquet")
    (folder / f"{gid}.counts.json").write_text(json.dumps(_plain(parts["counts"])) + "\n")


def done_games(folder: Path) -> set:
    return {p.name[:-len(".counts.json")] for p in folder.glob("*.counts.json")
            if all((folder / f"{p.name[:-len('.counts.json')]}.{k}.parquet").exists() for k in PART_NAMES)}


def assign_pass(artifacts=None, max_minutes: float = 25.0, limit=None, log=print) -> dict:
    """e1-assign: every development game not yet done, one at a time, until max_minutes."""
    import gc
    root = Path(os.environ["DATA_ROOT"])
    folder = games_dir(artifacts)
    games = development_games(root)
    todo = [g for g in games if g not in done_games(folder)]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    done = 0
    for gid in todo:
        if (time.time() - t0) / 60 > max_minutes:
            break
        (g,) = game_dicts(root, [gid])
        write_parts(folder, gid, assign_game(g))
        done += 1
        del g
        gc.collect()
        log(f"[matchups-E1] {gid} done ({done} this run, {round(time.time() - t0)} s)", flush=True) \
            if log is print else log(gid)
    left = len([g for g in games if g not in done_games(folder)])
    return {"development_games": len(games), "done_this_run": done, "left": left,
            "seconds": round(time.time() - t0, 1), "peak_gb": round(peak_rss_gb(), 2)}


# ----------------------------------------------------------------------------- gate

STORED_MP = Path("analytics") / "matchups" / "tables" / SEASON / "matchup_possessions_train.parquet"
STORED_DEF = Path("analytics") / "matchups" / "tables" / SEASON / "matchup_defenders.parquet"
GATE_TOL = 1e-6
RIDGE_TOL = 1e-6
PRINTED = {"constant": 1.2842, "raw": 1.3244, "ridge": 1.2854, "alpha": 300.0}   # reports/matchups.md


def stored_xfg(root):
    """The xFG lane's stored predictions with the frozen loader's rules (analytics.matchups.validate.load_xfg: the
    fold column verified on training rows); held-out rows are dropped at once (xfg_heldout_mask), no test game."""
    from analytics.matchups import validate as V
    x = V.load_xfg(root, SEASON)
    if x is None:
        raise SystemExit(f"{WHO}: no verified out-of-fold xFG under DATA_ROOT")
    x = x[~V.xfg_heldout_mask(x).to_numpy()].reset_index(drop=True)
    refuse_test(x["sportvu_game"].astype(str).unique(), WHO)
    return x


def load_parts(artifacts, games) -> dict:
    """{game: parts} of the assignment pass, with info back as a dict."""
    import pandas as pd
    folder = games_dir(artifacts)
    out = {}
    for g in games:
        parts = {k: pd.read_parquet(folder / f"{g}.{k}.parquet") for k in PART_NAMES}
        parts["info"] = info_dict(parts["info"])
        parts["counts"] = json.loads((folder / f"{g}.counts.json").read_text())
        out[g] = parts
    return out


def tables(artifacts, games, xfg, default_p):
    """matchup_possessions, shots (with p) and zones of the games, one game's frames-table at a time."""
    import pandas as pd
    from analytics.matchups import train as T
    mps, shots, zones = [], [], []
    for g in games:
        pt = load_parts(artifacts, [g])[g]
        t = T.matchup_tables({g: {"mf": pt["mf"], "info": pt["info"], "shots": pt["shots"]}}, xfg, default_p)
        mps.append(t["matchup_possessions"])
        shots.append(t["shots"])
        zones.append(pt["zones"])
    return (pd.concat(mps, ignore_index=True), pd.concat(shots, ignore_index=True),
            pd.concat(zones, ignore_index=True))


def gate(artifacts=None) -> dict:
    import numpy as np
    import pandas as pd
    from analytics.matchups import features as F
    from analytics.matchups import train as T
    root = Path(os.environ["DATA_ROOT"])
    folder = work_dir(artifacts)
    (folder / "gate.json").write_text(json.dumps({"passed": False, "status": "gate started"}) + "\n")
    games = development_games(root)
    missing = sorted(set(games) - done_games(games_dir(artifacts)))
    if missing:
        raise SystemExit(f"{WHO}: run e1-assign first; {len(missing)} game(s) have no parts")
    xfg = stored_xfg(root)
    default_p = T.default_probability(xfg)
    mp, shots, zones = tables(artifacts, games, xfg, default_p)
    stored = pd.read_parquet(root / STORED_MP)
    keys = ["sportvu_game", "possession", "def_player_id", "off_player_id"]
    a = mp.assign(sportvu_game=mp["sportvu_game"].astype(str)).sort_values(keys).reset_index(drop=True)
    b = stored.assign(sportvu_game=stored["sportvu_game"].astype(str)).sort_values(keys).reset_index(drop=True)
    same_keys = len(a) == len(b) and all((a[k].astype(str).to_numpy() == b[k].astype(str).to_numpy()).all() for k in keys)
    diffs = {c: float(np.max(np.abs(a[c].to_numpy(dtype=float) - b[c].to_numpy(dtype=float)))) if same_keys else None
             for c in ("partial_possession", "points_allowed_share", "xfg_expected_share", "switches")}
    pts = T.possession_points([{"possessions": pd.read_parquet(root / "analytics" / "possessions" / SEASON / f"{g}.parquet")}
                               for g in games])
    cv = T.ridge_cv(mp, pts, FOLDS, SEED)
    mse = {m: float(np.mean((cv["pred"]["p_" + m] - cv["pred"]["y"]) ** 2)) for m in T.MODELS}
    sd = pd.read_parquet(root / STORED_DEF).set_index("def_player_id")["ridge_value"]
    same_defenders = set(int(d) for d in sd.index) == set(int(d) for d in cv["ridge_values"])
    ridge_diff = float(np.nanmax(np.abs(np.array([cv["ridge_values"].get(int(d), np.nan) for d in sd.index])
                                        - sd.to_numpy(dtype=float)))) if same_defenders else None
    printed_ok = all(abs(round(mse[m], 4) - PRINTED[m]) < 1e-9 for m in ("constant", "raw", "ridge")) \
        and cv["alpha"] == PRINTED["alpha"]
    compare = T.compare_models(cv["pred"], BOOT, SEED)
    record = {"compared": ("matchup_possessions rebuilt by the frozen assignment and tables against the stored "
                           "matchup_possessions_train; the frozen ridge_cv refit on them against the stored ridge_value "
                           "of every defender and the out-of-fold MSEs printed in reports/matchups.md"),
              "games": len(games), "rows_rebuilt": int(len(a)), "rows_stored": int(len(b)), "keys_equal": bool(same_keys),
              "max_abs_diff": diffs, "tolerance": GATE_TOL, "xfg_rows": int(len(xfg)), "default_p": default_p,
              "ridge": {"alpha": cv["alpha"], "fold_alphas": cv["fold_alphas"], "oof_mse": mse,
                        "printed": PRINTED, "printed_equal": bool(printed_ok),
                        "defenders_stored": int(len(sd)), "defenders_refit": len(cv["ridge_values"]),
                        "same_defender_set": bool(same_defenders), "max_abs_diff_ridge_value": ridge_diff,
                        "tolerance": RIDGE_TOL},
              "frozen_reference_possession_points": {
                  "what": "the frozen ridge on possession points, refit with frozen code on the development games "
                          "(the lane's original target), 2000-resample game bootstrap",
                  "mse": {m: {"point": v["point"], "ci95": [v["lo"], v["hi"]]} for m, v in compare["mse"].items()},
                  "ridge_minus_constant": {"point": compare["paired"]["ridge_minus_constant"]["diff"],
                                           "ci95": [compare["paired"]["ridge_minus_constant"]["lo"],
                                                    compare["paired"]["ridge_minus_constant"]["hi"]]}},
              "attempts": int(len(shots)), "attempts_with_assigned_defender": int(shots["assigned_def_id"].notna().sum())}
    record["passed"] = bool(same_keys and all(v is not None and v <= GATE_TOL for v in diffs.values())
                            and printed_ok and same_defenders and ridge_diff is not None and ridge_diff <= RIDGE_TOL)
    record["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (folder / "gate.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return record


def require_gate(artifacts=None) -> dict:
    p = work_dir(artifacts) / "gate.json"
    if not p.exists():
        raise SystemExit(f"{WHO}: run e1-gate first")
    record = json.loads(p.read_text())
    if not record.get("passed"):
        raise SystemExit(f"{WHO}: the gate did not pass, so nothing here is used")
    return record


# ----------------------------------------------------------------------------- rows of the two parts

LAMBDA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
LAMBDA_MAX, LAMBDA_MIN = 1e6, 1e-3
PRE_SHOT_GAP = 2           # the pre-shot diagnostic keeps sampled frames at least this many seconds before the release
ZONE_COLUMNS = [f"zone_{z}" for z in ZONES[:-1]]       # "three" is the reference zone (the four shares sum to 1)
STAB_MIN_POSSESSIONS, STAB_MIN_ATTEMPTS = 300, 100


def part_a_rows(mp, shots, zones):
    """Part (a): one row per (possession, attacker) of every possession whose last attempt has a known shooter ->
    (rows frame, defender shares frame, counts). rows: game, possession, off_player_id, y, zone columns."""
    import pandas as pd
    s = shots.sort_values(["sportvu_game", "possession", "event"])
    last = s.groupby(["sportvu_game", "possession"], as_index=False).tail(1)
    known = last[last["shooter_id"].notna()][["sportvu_game", "possession", "shooter_id"]]
    z = zones.merge(known, on=["sportvu_game", "possession"], how="inner")
    z["y"] = (z["off_player_id"].astype("int64") == z["shooter_id"].astype("int64")).astype(float)
    hit = z.groupby(["sportvu_game", "possession"])["y"].sum()
    keep = hit[hit == 1].index                          # the shooter is among the tracked attackers
    z = z.set_index(["sportvu_game", "possession"]).loc[keep].reset_index()
    counts = {"possessions_with_known_last_shooter": int(len(known)), "possessions_kept": int(len(keep)),
              "possessions_shooter_not_tracked": int(len(known) - len(keep)), "rows": int(len(z))}
    shares = mp[["sportvu_game", "possession", "def_player_id", "off_player_id", "partial_possession"]].copy()
    shares["sportvu_game"] = shares["sportvu_game"].astype(str)
    shares = shares.merge(z[["sportvu_game", "possession", "off_player_id"]], how="inner",
                          on=["sportvu_game", "possession", "off_player_id"])
    return z.drop(columns=["shooter_id"]).reset_index(drop=True), shares, counts


def part_b_rows(shots, pmap: dict):
    """Part (b): attempts with an assigned defender at release and an xFG probability -> (rows, counts)."""
    import numpy as np
    s = shots.copy()
    s["sportvu_game"] = s["sportvu_game"].astype(str)
    s["p_xfg"] = [pmap.get((g, int(e)), np.nan) for g, e in zip(s["sportvu_game"], s["event"])]
    has_def = s["assigned_def_id"].notna()
    has_p = s["p_xfg"].notna()
    rows = s[has_def & has_p][["sportvu_game", "possession", "event", "assigned_def_id", "made", "p_xfg"]].copy()
    rows["y"] = rows["made"].astype(float)
    counts = {"attempts": int(len(s)), "with_assigned_defender": int(has_def.sum()),
              "without_xfg": int((has_def & ~has_p).sum()), "rows": int(len(rows))}
    return rows.drop(columns=["made"]).reset_index(drop=True), counts


class Design:
    """A sparse design over one part's rows with fixed column maps (players never seen get no column)."""

    def __init__(self, part: str, rows, shares=None, attackers=None, defenders=None, with_defenders=True):
        import numpy as np
        import scipy.sparse as sp
        self.part, self.with_defenders = part, with_defenders
        n = len(rows)
        if part == "a":
            self.attackers = attackers if attackers is not None else sorted(rows["off_player_id"].astype(int).unique())
            self.defenders = defenders if defenders is not None else sorted(shares["def_player_id"].astype(int).unique())
            ai = {p: i for i, p in enumerate(self.attackers)}
            di = {p: i for i, p in enumerate(self.defenders)}
            nz, na = len(ZONE_COLUMNS), len(self.attackers)
            nd = len(self.defenders) if with_defenders else 0
            self.n_free = 1 + nz                               # intercept and zone columns are not penalized
            cols = [np.zeros(n, dtype=int)]
            vals = [np.ones(n)]
            rws = [np.arange(n)]
            for j, c in enumerate(ZONE_COLUMNS):
                rws.append(np.arange(n)); cols.append(np.full(n, 1 + j)); vals.append(rows[c].to_numpy(dtype=float))
            ac = np.array([ai.get(int(p), -1) for p in rows["off_player_id"]])
            m = ac >= 0
            rws.append(np.flatnonzero(m)); cols.append(1 + nz + ac[m]); vals.append(np.ones(int(m.sum())))
            if with_defenders:
                key = {(g, int(q), int(o)): i for i, (g, q, o) in
                       enumerate(zip(rows["sportvu_game"], rows["possession"], rows["off_player_id"]))}
                r = np.array([key.get((g, int(q), int(o)), -1) for g, q, o in
                              zip(shares["sportvu_game"], shares["possession"], shares["off_player_id"])])
                dc = np.array([di.get(int(d), -1) for d in shares["def_player_id"]])
                m = (r >= 0) & (dc >= 0)
                rws.append(r[m]); cols.append(1 + nz + na + dc[m]); vals.append(shares["partial_possession"].to_numpy(dtype=float)[m])
            self.shape = (n, 1 + nz + na + nd)
            self.offset = np.zeros(n)
        else:
            self.defenders = defenders if defenders is not None else sorted(rows["assigned_def_id"].astype(int).unique())
            di = {p: i for i, p in enumerate(self.defenders)}
            nd = len(self.defenders) if with_defenders else 0
            self.n_free = 1
            rws, cols, vals = [np.arange(n)], [np.zeros(n, dtype=int)], [np.ones(n)]
            if with_defenders:
                dc = np.array([di.get(int(d), -1) for d in rows["assigned_def_id"]])
                m = dc >= 0
                rws.append(np.flatnonzero(m)); cols.append(1 + dc[m]); vals.append(np.ones(int(m.sum())))
            self.shape = (n, 1 + nd)
            p = np.clip(rows["p_xfg"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            self.offset = np.log(p) - np.log1p(-p)
        self.X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rws), np.concatenate(cols))), shape=self.shape)
        self.y = rows["y"].to_numpy(dtype=float)
        self.groups = rows["sportvu_game"].astype(str).to_numpy()

    def like(self, rows, shares=None):
        """The same column maps on other rows (test rows: unseen players get no column)."""
        return Design(self.part, rows, shares, getattr(self, "attackers", None), self.defenders, self.with_defenders)


def _penalty_vector(n_cols: int, n_free: int, lam: float):
    import numpy as np
    v = np.full(n_cols, lam)
    v[:n_free] = 0.0
    return v


def fit_logit(X, y, offset, pen, beta0=None):
    """Penalized logistic regression: minimize sum of log loss + 0.5 * sum pen_j beta_j^2 by L-BFGS."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit, log1p

    def f(b):
        z = offset + X @ b
        loss = np.sum(np.logaddexp(0.0, z) - y * z)
        g = X.T @ (expit(z) - y) + pen * b
        return loss + 0.5 * np.sum(pen * b * b), g
    del log1p
    x0 = np.zeros(X.shape[1]) if beta0 is None else beta0
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-8})
    if not np.all(np.isfinite(r.x)):
        raise SystemExit(f"{WHO}: the logistic fit did not converge to finite values")
    return r.x


def row_loss(y, z):
    import numpy as np
    return np.logaddexp(0.0, z) - y * z


def choose_lambda(D, rows_idx, n_folds: int = FOLDS, seed: int = SEED, grid=LAMBDA_GRID) -> tuple:
    """Inner grouped CV by game over rows_idx: the penalty with the lowest out-of-fold log loss (ties to the larger;
    the grid grows by threes past whichever end wins, up to LAMBDA_MAX and down to LAMBDA_MIN). Each fold's fits are
    warm-started from its fit at the nearest penalty already done (the optimum does not depend on the start)."""
    import numpy as np
    from analytics.eval import grouped_folds
    X, y, off, g = D.X[rows_idx], D.y[rows_idx], D.offset[rows_idx], D.groups[rows_idx]
    games = np.unique(g)
    folds = grouped_folds(games, n_folds=min(n_folds, len(games)), seed=seed)
    masks = [np.isin(g, te) for _tr, te in folds]
    curve, betas = {}, {}
    grid = sorted(grid, reverse=True)
    while True:
        for lam in grid:
            if lam in curve:
                continue
            near = min(betas, key=lambda l: abs(math.log(l) - math.log(lam))) if betas else None
            total, fitted = 0.0, []
            for i, m in enumerate(masks):
                b = fit_logit(X[~m], y[~m], off[~m], _penalty_vector(X.shape[1], D.n_free, lam),
                              None if near is None else betas[near][i])
                fitted.append(b)
                total += float(row_loss(y[m], off[m] + X[m] @ b).sum())
            curve[lam], betas[lam] = total / len(y), fitted
        best = min(curve, key=lambda l: (curve[l], -l))
        if best == max(curve) and best * 3 <= LAMBDA_MAX:
            grid = [best * 3]
            continue
        if best == min(curve) and best / 3 >= LAMBDA_MIN:
            grid = [best / 3]
            continue
        return float(best), {str(k): v for k, v in sorted(curve.items())}


def oof(D, n_folds: int = FOLDS, seed: int = SEED, log=print) -> dict:
    """Out-of-fold linear predictors under 5 grouped outer folds by game, the penalty of each fold by inner grouped CV
    on its training games (the no-defender model of part (b) has no penalized column and needs none)."""
    import numpy as np
    from analytics.eval import grouped_folds
    z = np.full(len(D.y), np.nan)
    lams = []
    for k, (_tr, te) in enumerate(grouped_folds(np.unique(D.groups), n_folds=n_folds, seed=seed)):
        m = np.isin(D.groups, te)
        tr_idx = np.flatnonzero(~m)
        if D.X.shape[1] > D.n_free:
            lam, _curve = choose_lambda(D, tr_idx, n_folds, seed)
        else:
            lam = 0.0
        b = fit_logit(D.X[tr_idx], D.y[tr_idx], D.offset[tr_idx], _penalty_vector(D.X.shape[1], D.n_free, lam))
        z[m] = D.offset[m] + D.X[m] @ b
        lams.append(lam)
        log(f"[matchups-E1] part {D.part} {'defender' if D.with_defenders else 'no-defender'} fold {k}: lambda {lam}",
            flush=True)
    return {"z": z, "fold_lambdas": lams}


def game_bootstrap(groups, values: dict, n: int = BOOT, seed: int = SEED) -> dict:
    """Mean of each row value (and of the paired differences given as values) with a percentile interval from a
    bootstrap of games; every value resampled on the same games. values: name -> per-row array."""
    import numpy as np
    import pandas as pd
    codes, uniq = pd.factorize(pd.Series(groups), sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    sums = {k: np.bincount(codes, weights=v, minlength=G) for k, v in values.items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n, G))
    out = {}
    for k, s in sums.items():
        boots = s[idx].sum(axis=1) / cnt[idx].sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[k] = {"point": float(s.sum() / cnt.sum()), "ci95": [float(lo), float(hi)]}
    out["games"] = int(G)
    out["rows"] = int(len(groups))
    return out


def projected_half_width(groups, diff, test_games: int = 10, n: int = BOOT, seed: int = SEED) -> dict:
    """D108: the half-width a 10-game set would give, from the development per-game variance: the 2.5 to 97.5
    percentile spread of the row-weighted mean difference over random sets of 10 development games (without
    replacement), halved; and the development interval half-width scaled by the square root of the game counts."""
    import numpy as np
    import pandas as pd
    codes, uniq = pd.factorize(pd.Series(groups), sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    s = np.bincount(codes, weights=diff, minlength=G)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(G, size=test_games, replace=False)
        vals.append(s[pick].sum() / cnt[pick].sum())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    idx = rng.integers(0, G, size=(n, G))
    blo, bhi = np.percentile(s[idx].sum(axis=1) / cnt[idx].sum(axis=1), [2.5, 97.5])
    return {"random_10_game_sets_half_width": float((hi - lo) / 2),
            "dev_half_width_scaled": float((bhi - blo) / 2 * math.sqrt(G / test_games)), "development_games": int(G)}


def part_result(name: str, D_def, D_nodef, log=print) -> dict:
    """Out-of-fold log loss of both models of a part, their paired difference and the D108 projection."""
    r_def, r_no = oof(D_def, log=log), oof(D_nodef, log=log)
    l_def, l_no = row_loss(D_def.y, r_def["z"]), row_loss(D_nodef.y, r_no["z"])
    stats = game_bootstrap(D_def.groups, {"log_loss_defender": l_def, "log_loss_no_defender": l_no,
                                          "defender_minus_no_defender": l_def - l_no})
    d = stats["defender_minus_no_defender"]
    return {"part": name, "rows": stats["rows"], "games": stats["games"],
            "fold_lambdas": {"defender": r_def["fold_lambdas"], "no_defender": r_no["fold_lambdas"]},
            "log_loss": {"defender": stats["log_loss_defender"], "no_defender": stats["log_loss_no_defender"]},
            "defender_minus_no_defender": d,
            "passes": bool(d["point"] < 0 and d["ci95"][1] < 0),
            "projected_set_a": projected_half_width(D_def.groups, l_def - l_no)}


def stored_mf_shots(artifacts, games):
    """(game, matchup_frames, shots) of the assignment pass, one game at a time."""
    import pandas as pd
    folder = games_dir(artifacts)
    for g in games:
        yield g, pd.read_parquet(folder / f"{g}.mf.parquet"), pd.read_parquet(folder / f"{g}.shots.parquet")


def pre_shot_shares(mf_shots, rows):
    """Context for part (a), not pre-registered: the defender shares recomputed from the sampled frames at least
    PRE_SHOT_GAP seconds before the sampled frame at or before the last attempt's release, so a defender who reaches
    the shooter only at the shot gains no share. mf_shots yields (game, matchup_frames, shots). -> (rows of the
    possessions that keep a frame, shares)."""
    import pandas as pd
    keep_rows, shares = [], []
    want = rows[["sportvu_game", "possession"]].drop_duplicates()
    for g, mf, sh in mf_shots:
        sh = sh.sort_values(["possession", "event"])
        last_k = sh.groupby("possession")["k"].last()
        poss = set(want.loc[want["sportvu_game"] == g, "possession"].astype(int))
        mf = mf[mf["possession"].isin(poss)]
        lim = mf["possession"].map(last_k).fillna(-1).astype(int) - PRE_SHOT_GAP
        mf = mf[(mf["k"] <= lim) & (lim >= 0)]
        if len(mf) == 0:
            continue
        n = mf.groupby("possession")["k"].nunique()
        pair = mf.groupby(["possession", "def_player_id", "off_player_id"], as_index=False).size()
        pair["partial_possession"] = pair["size"] / pair["possession"].map(n)
        pair.insert(0, "sportvu_game", g)
        shares.append(pair.drop(columns=["size"]))
        keep_rows.append(pd.DataFrame({"sportvu_game": g, "possession": sorted(n.index.astype(int))}))
    kept = pd.concat(keep_rows, ignore_index=True)
    r = rows.merge(kept, on=["sportvu_game", "possession"], how="inner")
    sh = pd.concat(shares, ignore_index=True)
    sh = sh.merge(r[["sportvu_game", "possession", "off_player_id"]], how="inner",
                  on=["sportvu_game", "possession", "off_player_id"])
    return r.reset_index(drop=True), sh


def ablation(mp, shots, pts: dict, log=print) -> dict:
    """The defender design on possession points minus xFG-expected points, against the same design without defender
    columns; frozen ridge functions and alpha grid, nested CV."""
    import numpy as np
    import pandas as pd
    from analytics.eval import grouped_folds
    from analytics.matchups import train as T
    D = T.possession_design(mp, pts)
    s = shots.copy()
    s["sportvu_game"] = s["sportvu_game"].astype(str)
    exp = (s["p"] * s["value"]).groupby([s["sportvu_game"], s["possession"].astype(int)]).sum()
    y = D["y"] - np.array([float(exp.get(k, 0.0)) for k in D["keys"]])
    nd = len(D["defs"])
    Xs = {"defender": D["X"], "no_defender": D["X"][:, nd:]}
    groups = D["groups"]
    pred = {k: np.full(len(y), np.nan) for k in Xs}
    lams = {k: [] for k in Xs}
    for tr_g, te_g in grouped_folds(sorted(set(groups)), n_folds=FOLDS, seed=SEED):
        te = np.isin(groups, te_g)
        for k, X in Xs.items():
            a, _ = T.choose_alpha(X[~te], y[~te], groups[~te], FOLDS, SEED)
            pred[k][te] = T._fit_ridge(X[~te], y[~te], a).predict(X[te])
            lams[k].append(a)
    sq = {k: (pred[k] - y) ** 2 for k in Xs}
    stats = game_bootstrap(groups, {"mse_defender": sq["defender"], "mse_no_defender": sq["no_defender"],
                                    "defender_minus_no_defender": sq["defender"] - sq["no_defender"]})
    log(f"[matchups-E1] ablation done", flush=True)
    return {"what": ("ridge on possession points minus the xFG-expected points of the possession's attempts, the frozen "
                     "possession design with and without defender columns, nested alpha over the frozen grid"),
            "possessions": stats["rows"], "games": stats["games"], "fold_alphas": lams,
            "mse": {"defender": stats["mse_defender"], "no_defender": stats["mse_no_defender"]},
            "defender_minus_no_defender": stats["defender_minus_no_defender"]}


def stability(D_full, rows, shares, part: str, lam: float, dates: dict, mp=None) -> dict:
    """Split-half: defender coefficients fitted on odd and on even development games in date order; Spearman over
    defenders with enough data in each half. Reported, never a selection input."""
    import numpy as np
    from scipy.stats import spearmanr
    games = sorted(set(D_full.groups), key=lambda g: (dates[g], g))
    halves = [set(games[0::2]), set(games[1::2])]
    coefs, enough = [], []
    for h in halves:
        m = np.isin(D_full.groups, list(h))
        b = fit_logit(D_full.X[m], D_full.y[m], D_full.offset[m], _penalty_vector(D_full.X.shape[1], D_full.n_free, lam))
        first = D_full.shape[1] - len(D_full.defenders)
        coefs.append(dict(zip(D_full.defenders, b[first:])))
        if part == "a":
            dp = mp[mp["sportvu_game"].astype(str).isin(h)][["sportvu_game", "possession", "def_player_id"]].drop_duplicates()
            n = dp.groupby("def_player_id").size()
            enough.append(set(int(d) for d in n.index[n >= STAB_MIN_POSSESSIONS]))
        else:
            r = rows[rows["sportvu_game"].isin(h)]
            n = r.groupby("assigned_def_id").size()
            enough.append(set(int(d) for d in n.index[n >= STAB_MIN_ATTEMPTS]))
    both = sorted(enough[0] & enough[1])
    rho = spearmanr([coefs[0][d] for d in both], [coefs[1][d] for d in both]).statistic if len(both) > 2 else float("nan")
    return {"defenders": len(both), "spearman": float(rho), "penalty": lam,
            "gate": f"{STAB_MIN_POSSESSIONS} distinct possessions" if part == "a" else f"{STAB_MIN_ATTEMPTS} attempts",
            "halves": "odd and even development games in date order"}


def game_date_of(g: str):
    m, d, y = g.split(".")[:3]
    return (int(y), int(m), int(d))


class DevData:
    """The development tables and designs, built once per run and only when a piece needs them."""

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self._built = False

    def build(self):
        if self._built:
            return self
        import pandas as pd
        from analytics.matchups import train as T
        root = Path(os.environ["DATA_ROOT"])
        self.root, self.games = root, development_games(root)
        xfg = stored_xfg(root)
        self.mp, self.shots, zones = tables(self.artifacts, self.games, xfg, T.default_probability(xfg))
        pmap = {(str(a), int(e)): float(p) for a, e, p, f in zip(xfg["sportvu_game"], xfg["event"], xfg["p_selected"],
                                                                 xfg["fold"]) if f >= 0}
        self.a_rows, self.a_shares, self.a_counts = part_a_rows(self.mp, self.shots, zones)
        self.b_rows, self.b_counts = part_b_rows(self.shots, pmap)
        self.pre_rows, self.pre_shares = pre_shot_shares(stored_mf_shots(self.artifacts, self.games), self.a_rows)
        self.D = {"a": Design("a", self.a_rows, self.a_shares),
                  "a0": Design("a", self.a_rows, self.a_shares, with_defenders=False),
                  "b": Design("b", self.b_rows), "b0": Design("b", self.b_rows, with_defenders=False),
                  "p": Design("a", self.pre_rows, self.pre_shares),
                  "p0": Design("a", self.pre_rows, self.pre_shares, with_defenders=False)}
        self.pts = T.possession_points([{"possessions": pd.read_parquet(root / "analytics" / "possessions" / SEASON /
                                                                        f"{x}.parquet")} for x in self.games])
        self._built = True
        return self


def _piece(name: str, d: DevData):
    """One cached piece of the development stage."""
    import numpy as np
    d.build()
    D = d.D
    if name == "counts":
        return {"a": d.a_counts, "b": d.b_counts,
                "pre_shot_possessions": int(d.pre_rows[["sportvu_game", "possession"]].drop_duplicates().shape[0])}
    if name == "part_a":
        return part_result("a", D["a"], D["a0"])
    if name == "part_b":
        return part_result("b", D["b"], D["b0"])
    if name == "pre_shot":
        r = part_result("a", D["p"], D["p0"])
        return {"what": ("context, not pre-registered, reads no verdict: part (a) with the defender shares taken only "
                         f"from sampled frames at least {PRE_SHOT_GAP} s before the sampled frame at or before the "
                         "release of the possession's last attempt (possessions without such a frame left out)"),
                "possessions": int(d.pre_rows[["sportvu_game", "possession"]].drop_duplicates().shape[0]), **r}
    if name == "ablation":
        return ablation(d.mp, d.shots, d.pts)
    if name.startswith("lambda_"):
        key = {"lambda_a": "a", "lambda_a_no_defender": "a0", "lambda_b": "b", "lambda_pre_shot": "p",
               "lambda_pre_shot_no_defender": "p0"}[name]
        lam, curve = choose_lambda(D[key], np.arange(len(D[key].y)))
        return {"lambda": lam, "curve": curve}
    if name == "stability":
        raise SystemExit("stability needs the final penalties; it is computed in the assembly")
    raise ValueError(name)


PIECES = ("counts", "part_a", "part_b", "pre_shot", "ablation", "lambda_a", "lambda_a_no_defender", "lambda_b",
          "lambda_pre_shot", "lambda_pre_shot_no_defender")


def dev(artifacts=None, max_minutes: float = 25.0) -> dict:
    """The development stage in cached pieces (keyed by the code commit): each run computes pieces until max_minutes,
    then stops; the run that finds every piece done assembles reports/research/matchups-E1-dev.json."""
    t0 = time.time()
    head = clean_head()
    g = require_gate(artifacts)
    cache = work_dir(artifacts) / "dev_cache" / head
    cache.mkdir(parents=True, exist_ok=True)
    d = DevData(artifacts)
    for name in PIECES:
        f = cache / f"{name}.json"
        if f.exists():
            continue
        if (time.time() - t0) / 60 > max_minutes:
            return {"status": "stopped for the lock budget; run e1-dev again to resume",
                    "done": sorted(p.stem for p in cache.glob("*.json")), "seconds": round(time.time() - t0, 1)}
        out = _piece(name, d)
        f.write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
        print(f"[matchups-E1] piece {name} done {round(time.time() - t0)} s", flush=True)
    P = {name: json.loads((cache / f"{name}.json").read_text()) for name in PIECES}
    stab_file = cache / "stability.json"
    if not stab_file.exists():
        d.build()
        dates = {x: game_date_of(x) for x in d.games}
        stab = {"a": stability(d.D["a"], d.a_rows, d.a_shares, "a", P["lambda_a"]["lambda"], dates, d.mp),
                "b": stability(d.D["b"], d.b_rows, None, "b", P["lambda_b"]["lambda"], dates)}
        stab_file.write_text(json.dumps(_plain(stab), indent=2, allow_nan=False) + "\n")
    stab = json.loads(stab_file.read_text())
    out = {"experiment": EXPERIMENT, "code_commit": head,
           "stage": "development games, out of fold under 5 grouped folds by game (seed 0); no Set A or held-out row read",
           "gate": {k: g[k] for k in ("games", "rows_rebuilt", "rows_stored", "keys_equal", "max_abs_diff", "ridge",
                                      "passed")},
           "frozen_reference_possession_points": g["frozen_reference_possession_points"],
           "rows": P["counts"], "part_a": P["part_a"], "part_b": P["part_b"], "ablation": P["ablation"],
           "part_a_pre_shot_context": P["pre_shot"], "stability": stab,
           "test_fit": {"what": "penalties of the final fits on every development game (grouped CV, 5 folds, seed 0)",
                        **{k: P[k]["lambda"] for k in PIECES if k.startswith("lambda_")},
                        "curves": {k: P[k]["curve"] for k in PIECES if k.startswith("lambda_")}},
           "verdict": {"rule": ("D108: each part passes when the out-of-fold log loss difference, defender model minus "
                                "no-defender model, is below zero with its 95 percent interval wholly below zero, on the "
                                "development games; Set A is a later confirmation read as D80"),
                       "part_a_passes": P["part_a"]["passes"], "part_b_passes": P["part_b"]["passes"]},
           "peak_gb": round(peak_rss_gb(), 2)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def clean_head() -> str:
    """HEAD, refused unless the whole working tree is clean."""
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty[:300]}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


# ----------------------------------------------------------------------------- the Set A path

PATH_TABLES = {"frames": ("sportvu_contract", "{g}.parquet"), "possessions": ("analytics/possessions/2015-16", "{g}.parquet"),
               "shot_frames": ("analytics/shot_frames/2015-16", "{g}.parquet"), "rosters": ("analytics/rosters/2015-16", "{g}.parquet")}


def missing_tables(root, games) -> dict:
    """game -> the needed tables it lacks (paths only, nothing read)."""
    out = {}
    for g in games:
        miss = [k for k, (d, f) in PATH_TABLES.items() if not (Path(root) / d / f.format(g=g)).exists()]
        if miss:
            out[g] = miss
    return out


def final_models(artifacts, lambdas: dict) -> dict:
    """Every model fitted on all development rows with the penalties the development stage chose -> templates (column
    maps) and coefficients."""
    import numpy as np
    from analytics.matchups import train as T
    root = Path(os.environ["DATA_ROOT"])
    games = development_games(root)
    xfg = stored_xfg(root)
    default_p = T.default_probability(xfg)
    mp, shots, zones = tables(artifacts, games, xfg, default_p)
    pmap = {(str(a), int(e)): float(p) for a, e, p, f in zip(xfg["sportvu_game"], xfg["event"], xfg["p_selected"],
                                                             xfg["fold"]) if f >= 0}
    a_rows, a_shares, _ = part_a_rows(mp, shots, zones)
    b_rows, _ = part_b_rows(shots, pmap)
    pre_rows, pre_shares = pre_shot_shares(stored_mf_shots(artifacts, games), a_rows)
    designs = {"a": (Design("a", a_rows, a_shares), lambdas["lambda_a"]),
               "a0": (Design("a", a_rows, a_shares, with_defenders=False), lambdas["lambda_a_no_defender"]),
               "b": (Design("b", b_rows), lambdas["lambda_b"]),
               "b0": (Design("b", b_rows, with_defenders=False), 0.0),
               "p": (Design("a", pre_rows, pre_shares), lambdas["lambda_pre_shot"]),
               "p0": (Design("a", pre_rows, pre_shares, with_defenders=False), lambdas["lambda_pre_shot_no_defender"])}
    out = {"default_p": default_p}
    for k, (D, lam) in designs.items():
        out[k] = {"template": D, "lambda": lam,
                  "beta": fit_logit(D.X, D.y, D.offset, _penalty_vector(D.shape[1], D.n_free, lam))}
        D.X = D.X[:0]                                      # keep the column maps, drop the development rows
    del np
    return out


def t_interval(groups, diff) -> dict:
    """Report-only sensitivity check (PLAN.md D80): a t interval on the per-game mean differences."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    m = pd.Series(diff).groupby(pd.Series(groups)).mean().to_numpy()
    if len(m) < 2:
        return {"games": int(len(m)), "mean_of_game_means": float(m.mean()) if len(m) else None, "lo": None, "hi": None}
    half = float(stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m)))
    return {"games": int(len(m)), "mean_of_game_means": float(m.mean()), "lo": float(m.mean() - half),
            "hi": float(m.mean() + half)}


def reading(ci, dev_passed=True, context=False) -> str:
    """D108 and D80: the Set A reading of a defender-minus-no-defender interval. A part that failed on development is
    never "confirmed" by Set A; a context arm gets no D108 reading."""
    if context:
        return "context, reads no verdict"
    lo, hi = ci
    if hi < 0:
        if not dev_passed:
            return "wholly below zero; does not overturn the development fail (D108)"
        return "confirms (interval wholly below zero)"
    if lo > 0:
        return "contradiction (interval wholly above zero): blocks any site defender table for this part"
    return "inconclusive at 10 games (the interval straddles zero; not evidence against)"


def evaluate_games(root, games, pmap: dict, models: dict, dev_passed=None) -> dict:
    """Parts (a) and (b) and the pre-shot context on the given games of a root, with the final models: the frozen
    assignment per game (frames read once each), the rows, the per-row log losses of both models of each part, a
    bootstrap of the games and a t interval of per-game means. Mechanics shared by the scorer and the dry run."""
    import numpy as np
    import pandas as pd
    from analytics.matchups import train as T
    mps, shots, zones, mfs = [], [], [], []
    xfg_df = pd.DataFrame([{"sportvu_game": g, "event": e, "p_selected": p} for (g, e), p in pmap.items()],
                          columns=["sportvu_game", "event", "p_selected"])
    for g in game_dicts(root, games):
        pt = assign_game(g)
        gid = str(g["sportvu_game"])
        t = T.matchup_tables({gid: {"mf": pt["mf"], "info": info_dict(pt["info"]), "shots": pt["shots"]}}, xfg_df,
                             models["default_p"])
        mps.append(t["matchup_possessions"]); shots.append(t["shots"]); zones.append(pt["zones"])
        mfs.append((gid, pt["mf"], pt["shots"]))
        del g, pt
    mp, sh, zn = (pd.concat(x, ignore_index=True) for x in (mps, shots, zones))
    a_rows, a_shares, a_counts = part_a_rows(mp, sh, zn)
    b_rows, b_counts = part_b_rows(sh, pmap)
    pre_rows, pre_shares = pre_shot_shares(iter(mfs), a_rows)
    counts = {"games": len(games), "matchup_rows": int(len(mp)), "attempts": int(len(sh)), "a": a_counts, "b": b_counts,
              "pre_shot_possessions": int(pre_rows[["sportvu_game", "possession"]].drop_duplicates().shape[0]),
              "games_with_part_a_rows": int(a_rows["sportvu_game"].nunique()),
              "games_with_part_b_rows": int(b_rows["sportvu_game"].nunique()),
              "part_b_rows_per_game_min": int(b_rows.groupby("sportvu_game").size().min()) if len(b_rows) else 0,
              "defenders_in_part_b_without_a_development_coefficient": int(
                  len(set(b_rows["assigned_def_id"].astype(int)) - set(models["b"]["template"].defenders)))}
    dev_passed = dev_passed or {"a": True, "b": True}
    out = {"counts": counts, "parts": {}}
    for name, rows, shares, kd, k0 in (("a", a_rows, a_shares, "a", "a0"), ("b", b_rows, None, "b", "b0"),
                                       ("a_pre_shot_context", pre_rows, pre_shares, "p", "p0")):
        if len(rows) == 0:
            out["parts"][name] = None
            continue
        loss = {}
        for k in (kd, k0):
            D = models[k]["template"].like(rows, shares)
            loss[k] = row_loss(D.y, D.offset + D.X @ models[k]["beta"])
        groups = rows["sportvu_game"].astype(str).to_numpy()
        stats = game_bootstrap(groups, {"log_loss_defender": loss[kd], "log_loss_no_defender": loss[k0],
                                        "defender_minus_no_defender": loss[kd] - loss[k0]})
        d = stats["defender_minus_no_defender"]
        out["parts"][name] = {"rows": stats["rows"], "games": stats["games"],
                              "log_loss": {"defender": stats["log_loss_defender"],
                                           "no_defender": stats["log_loss_no_defender"]},
                              "defender_minus_no_defender": d,
                              "t_interval_of_game_means": t_interval(groups, loss[kd] - loss[k0]),
                              "reading": reading(d["ci95"], dev_passed.get(name, True),
                                                 context=name == "a_pre_shot_context")}
    del np
    return out


def non_finite(value, path: str = "") -> list:
    """Paths of every float in a JSON-like value that is NaN or infinite."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in non_finite(v, f"{path}.{k}" if path else str(k))]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in non_finite(v, f"{path}[{i}]")]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def make_scorer(models: dict, xfg_artifacts=None, provenance: dict = None, xfg_p=None, xfg_model=None,
                dev_passed=None):
    """The scorer score_once runs on Set A. The test data is reached only here: the research data root and the frozen
    xFG features of the test attempts (tools.research_xfg). Before anything is computed it refuses unless every test
    game has frames, possessions, shot rows and a roster; it refuses unless every game gives rows to both parts; a
    summary with any non-finite number is returned as a refused verdict. xfg_model is the gated frozen xFG model,
    loaded by final() before score_once; xfg_p replaces it in the rehearsal test only."""
    import numpy as np
    from tools import research_testset as RT
    if xfg_p is None and xfg_model is None:
        raise SystemExit(f"{WHO}: no xFG model was given to the scorer")

    def scorer(test_games):
        if RT._SCORING is None or RT._SCORING[1] != TRACKING:
            raise SystemExit(f"{WHO}: the test data is read only inside the scorer that score_once runs")
        from tools.research_xfg import research_data_root, test_features
        root = research_data_root(xfg_artifacts)
        games = sorted(str(g) for g in test_games)
        head = {"experiment": EXPERIMENT, "stage": "Set A, scored once as the D108 confirmation", **(provenance or {})}
        miss = missing_tables(root, games)
        if miss:
            return {**head, "verdict": {"refused": True, "reading": f"{len(miss)} test game(s) lack a needed table"},
                    "missing": miss}
        if xfg_p is not None:
            pmap = xfg_p(games)
        else:
            feats = test_features(games, xfg_artifacts)
            p = xfg_model.predict(feats)
            pmap = {(str(g), int(e)): float(v) for g, e, v in zip(feats["sportvu_game"], feats["event"], np.asarray(p))}
        res = evaluate_games(root, games, pmap, models, dev_passed)
        c = res["counts"]
        if c["games_with_part_a_rows"] != len(games) or c["games_with_part_b_rows"] != len(games):
            return {**head, "verdict": {"refused": True, "reading": "not every test game gives rows to both parts; "
                                                                   "scored all or none"}, "counts": c}
        bad = non_finite(res)
        if bad:
            return {**head, "verdict": {"refused": True, "reading": f"{len(bad)} non-finite value(s), for example "
                                                                   f"{bad[0]}"}, "counts": c}
        return {**head, **res}

    return scorer


def dryrun(artifacts=None, xfg_artifacts=None, n_games: int = 3) -> dict:
    """The Set A path (evaluate_games and the frozen xFG model) on the first development games, read from DATA_ROOT,
    with models fitted on every development game: mechanics only, these games are in sample."""
    import pandas as pd
    from tools.frozen_models import load_xfg
    from tools.research_xfg import features_path
    require_gate(artifacts)
    dev_json = json.loads((RESULTS / f"{EXPERIMENT}-dev.json").read_text())
    models = final_models(artifacts, dev_json["test_fit"])
    root = Path(os.environ["DATA_ROOT"])
    games = development_games(root)[:n_games]
    feats = pd.read_parquet(features_path(xfg_artifacts), filters=[("sportvu_game", "in", games)])
    p = load_xfg(xfg_artifacts).predict(feats)
    pmap = {(str(g), int(e)): float(v) for g, e, v in zip(feats["sportvu_game"], feats["event"], p)}
    dev_passed = {"a": bool(dev_json["verdict"]["part_a_passes"]), "b": bool(dev_json["verdict"]["part_b_passes"])}
    res = evaluate_games(root, games, pmap, models, dev_passed)
    return {"games": games, "counts": res["counts"], "parts": sorted(k for k, v in res["parts"].items() if v),
            "readings": {k: v["reading"] for k, v in res["parts"].items() if v},
            "non_finite": non_finite(res), "reassignment_equal": reassignment_equal(artifacts, root, games[:2]),
            "note": "mechanics only; these games are in the development fit"}


def reassignment_equal(artifacts, root, games) -> dict:
    """The assignment pass rerun on a few games against the stored parts (every part, the zone mix included)."""
    import pandas as pd
    folder = games_dir(artifacts)
    out = {}
    for g in game_dicts(root, games):
        gid = str(g["sportvu_game"])
        pt = assign_game(g)
        same = {}
        for k in PART_NAMES:
            stored = pd.read_parquet(folder / f"{gid}.{k}.parquet")
            new = pt[k].reset_index(drop=True).copy()
            stored = stored.reset_index(drop=True)
            if "frames" in new.columns:                    # a list column: compare as tuples
                new["frames"] = [tuple(int(v) for v in x) for x in new["frames"]]
                stored["frames"] = [tuple(int(v) for v in x) for x in stored["frames"]]
            try:
                pd.testing.assert_frame_equal(new, stored, check_dtype=False)
                same[k] = True
            except AssertionError:
                same[k] = False
        out[gid] = same
    return out


def final(artifacts=None, xfg_artifacts=None) -> dict:
    head = clean_head()
    from tools.research_xfg import test_root_ready
    gate_record = require_gate(artifacts)
    dev_path = RESULTS / f"{EXPERIMENT}-dev.json"
    if not dev_path.exists():
        raise SystemExit(f"{WHO}: run e1-dev first; Set A is scored after the development result exists")
    dev_json = json.loads(dev_path.read_text())
    if not test_root_ready(xfg_artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games; run tools.research_ingest first")
    from tools.frozen_models import load_xfg
    xfg_model = load_xfg(xfg_artifacts)                # refuses here, before the ledger, if it is missing or stale
    models = final_models(artifacts, dev_json["test_fit"])
    provenance = {"code_commit": head, "dev_code_commit": dev_json["code_commit"],
                  "development_verdict": dev_json["verdict"]}
    dev_passed = {"a": bool(dev_json["verdict"]["part_a_passes"]), "b": bool(dev_json["verdict"]["part_b_passes"])}
    scorer = make_scorer(models, xfg_artifacts, provenance, xfg_model=xfg_model, dev_passed=dev_passed)
    out = score_once(EXPERIMENT, TRACKING, scorer)
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"gate": {k: gate_record[k] for k in ("games", "rows_rebuilt", "keys_equal", "passed")},
                        "lambdas": {k: models[k]["lambda"] for k in ("a", "a0", "b", "b0", "p", "p0")},
                        "projected_set_a": {"a": dev_json["part_a"]["projected_set_a"],
                                            "b": dev_json["part_b"]["projected_set_a"]}}}
    (RESULTS / f"{EXPERIMENT}-final.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-assign", "e1-gate", "e1-dev", "e1-dryrun", "e1-final"))
    ap.add_argument("--xfg-artifacts", default=os.environ.get("V3_XFG_ARTIFACTS"),
                    help="the artifacts folder holding the gated frozen xFG model (tools.frozen_models) and the xFG "
                         "research features (default: V3_XFG_ARTIFACTS)")
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    with heavy_job(f"research matchups {a.command}"):
        if a.command == "e1-assign":
            out = assign_pass(a.artifacts, a.max_minutes, a.limit)
        elif a.command in ("e1-dryrun", "e1-final"):
            out = {"e1-dryrun": dryrun, "e1-final": final}[a.command](a.artifacts, a.xfg_artifacts)
        elif a.command == "e1-dev":
            out = dev(a.artifacts, a.max_minutes)
        else:
            out = {"e1-gate": gate}[a.command](a.artifacts)
    print(json.dumps(_plain(out.get("verdict", out)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

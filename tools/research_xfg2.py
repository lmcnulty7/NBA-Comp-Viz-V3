"""tools/research_xfg2.py: xfg-E2, second defender and contest geometry (docs/research/xfg.md, xfg-E2 and its
commitments; PLAN.md D121).

    python3 -m tools.research_xfg2 features [--artifacts DIR] [--max-minutes M]   resumable, one game at a time
    python3 -m tools.research_xfg2 gate     [--artifacts DIR]                     the baseline refit reproduces the frozen
    python3 -m tools.research_xfg2 dev      [--artifacts DIR]                     development games, 5 grouped folds

features  The three release-frame features (second_defender_dist_ft, contest_angle_deg, defender_speed_to_line_ftps)
          for every row of the cached frozen feature frame (tools.research_xfg features, training rows only) with
          has_contest 1, from that game's tracking frames, one game at a time; each game's result is written as it
          finishes, so a job that reaches --max-minutes stops and the next resumes. No held-out game: the frozen guard
          runs on every game id, and the cached frame holds training rows only.
gate      The baseline arm (the frozen hgb_contest_prior, isotonic step on 3 inner folds) refit here with the frozen
          functions on all training games under the frozen outer folds must reproduce the stored out-of-fold
          p_selected (fold >= 0) within the xfg tolerance, as xfg-E1's gate. This module's fit is shared by the three
          arms, so the gate checks the code the new arms use.
dev       Every development game (tools.research_testset.dev_games), 5 grouped outer folds, seed 0: the baseline, the
          new arm (the three columns added) and the ablation (second_defender_dist_ft only); paired per-shot log loss
          differences with a 2000-resample game bootstrap; per-arm Brier, calibration slope and ECE with intervals.
          Writes reports/research/xfg-E2-dev.json.

rehearse  The scorer's path on 10 development games in place of Set A (not a result; artifacts only).
final     The one Set A score, score_once("xfg-E2"), only when development adopts and the rehearsal ran; it waits for
          the lead's go. Writes reports/research/xfg-E2-final.json. Everything runs under the heavy-job lock; artifacts go only under the artifacts folder.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools import research_xfg as RX
from tools.frozen_common import (REPO_ROOT, TOLERANCE_CAP, begin_rebuild, require_passed, training_games_only,
                                 write_check)
from tools.research_testset import dev_games, refuse_test

WHO = "tools.research_xfg2"
SEED, OUTER_FOLDS, BOOT = 0, 5, 2000
THRESHOLD = -0.001          # pre-registered: adopt when new minus baseline is at most this and the interval excludes 0
NEW_FEATURES = ["second_defender_dist_ft", "contest_angle_deg", "defender_speed_to_line_ftps"]
ABLATION = ["second_defender_dist_ft"]
SPEED_FRAMES, FPS = 10, 25
RESULTS = REPO_ROOT / "reports" / "research"


def out_dir(artifacts=None) -> Path:
    # v2: the first job read the frames without the frozen cleaning; its folder is left as it is, unused
    return RX.research_dir(artifacts) / "xfg_e2_features_v2"


# ----------------------------------------------------------------------------- the features of one game

def nearer_rim(x: float) -> tuple:
    from analytics.decisions.features import RIM_LEFT, RIM_RIGHT
    return RIM_RIGHT if x > 47.0 else RIM_LEFT


def line_distance(px, py, sx, sy, rx, ry) -> float:
    """Perpendicular distance from (px, py) to the line through the shooter (sx, sy) and the rim (rx, ry)."""
    vx, vy = rx - sx, ry - sy
    norm = math.hypot(vx, vy)
    if norm == 0:
        return float("nan")
    return abs(vx * (py - sy) - vy * (px - sx)) / norm


def angle_deg(sx, sy, rx, ry, dx, dy) -> float:
    """The angle at the shooter between the line to the rim and the line to the defender, 0 to 180 degrees."""
    a = (rx - sx, ry - sy)
    b = (dx - sx, dy - sy)
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return float("nan")
    c = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(c))


def row_features(G, release_frame: int, shooter: int, rims: dict) -> dict:
    """The three features of one attempt from GameFrames G (the frozen class), NaN where not measurable."""
    import numpy as np
    nan = {c: float("nan") for c in NEW_FEATURES}
    i = G.pos.get(int(release_frame))
    j = G.col.get(int(shooter))
    if i is None or j is None or not np.isfinite(G.x[i, j]):
        return nan
    sx, sy = float(G.x[i, j]), float(G.y[i, j])
    opp = (G.team != G.team[j]) & np.isfinite(G.x[i])
    if not opp.any():
        return nan
    d = np.where(opp, np.hypot(G.x[i] - sx, G.y[i] - sy), np.inf)
    order = np.argsort(d, kind="stable")
    k = int(order[0])
    out = dict(nan)
    if opp.sum() >= 2:
        out["second_defender_dist_ft"] = float(d[order[1]])
    rx, ry = rims.get((int(G.team[j]), int(G.period[i]))) or nearer_rim(sx)
    out["contest_angle_deg"] = angle_deg(sx, sy, rx, ry, float(G.x[i, k]), float(G.y[i, k]))
    i0 = i - SPEED_FRAMES
    if i0 >= 0 and G.period[i0] == G.period[i] and G.frames[i] - G.frames[i0] == SPEED_FRAMES \
            and np.isfinite(G.x[i0, j]) and np.isfinite(G.x[i0, k]):
        d1 = line_distance(float(G.x[i, k]), float(G.y[i, k]), sx, sy, rx, ry)
        d0 = line_distance(float(G.x[i0, k]), float(G.y[i0, k]), sx, sy, rx, ry)
        out["defender_speed_to_line_ftps"] = (d0 - d1) / (SPEED_FRAMES / FPS)
    return out


def game_features(rows, frames, possessions) -> dict:
    """{row index: features} for the rows of one game (rows of the cached frame with has_contest 1)."""
    import numpy as np
    from analytics.decisions.features import baskets_by_team_period
    from analytics.xfg.features import GameFrames
    f = frames.sort_values("frame", kind="stable").reset_index(drop=True)
    rims = baskets_by_team_period(f, f["frame"].to_numpy(), possessions)
    G = GameFrames(f)
    return {int(i): row_features(G, int(r.release_frame), int(r.player_id), rims) for i, r in rows.iterrows()}


def new_features_of_game(root: Path, frame, game: str) -> dict:
    """{row index of `frame`: the three features} for one game of a data root in the DATA_ROOT layout: the rows with
    has_contest 1, a release frame and a shooter, from the game's frames read with the lane's columns and cleaning
    (as the frozen feature build reads them) and its possessions table (the target-rim vote). The development build
    and the Set A scorer both call this."""
    import pandas as pd
    from analytics import schema as contract_tables
    from analytics.xfg import validate as V
    rows = frame[(frame["sportvu_game"].astype(str) == game) & (frame["has_contest"] == 1)
                 & (frame["release_frame"] >= 0) & frame["player_id"].notna()]
    if not len(rows):
        return {}
    frames, _summary = V.clean_frames(pd.read_parquet(Path(root) / "sportvu_contract" / f"{game}.parquet",
                                                      columns=V.FRAME_COLUMNS))
    poss = pd.read_parquet(contract_tables.table_path(Path(root) / "analytics", "possessions", "2015-16", game))
    return game_features(rows, frames, poss)


def add_new_features(root: Path, rows):
    """`rows` (a frozen feature frame of some games) with the three columns built game by game."""
    import numpy as np
    rows = rows.reset_index(drop=True)
    for c in NEW_FEATURES:
        rows[c] = np.nan
    for game in sorted(rows["sportvu_game"].astype(str).unique()):
        for i, v in new_features_of_game(root, rows, game).items():
            for c in NEW_FEATURES:
                rows.at[i, c] = v[c]
    return rows


def build_features(artifacts=None, max_minutes: float = 25.0) -> dict:
    """Resumable: one parquet per game under out_dir; stops before max_minutes."""
    import os
    import pandas as pd
    from analytics.eval import is_heldout_game
    started = time.time()
    root = Path(os.environ["DATA_ROOT"])
    frame = RX._features(artifacts)
    games = sorted(frame["sportvu_game"].astype(str).unique())
    training_games_only(games, WHO)
    folder = out_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in folder.glob("*.parquet")}
    built = 0
    for game in games:
        if game in done:
            continue
        if (time.time() - started) / 60 > max_minutes:
            break
        if is_heldout_game(game):
            raise SystemExit(f"{WHO}: {game} is a held-out game")
        feats = new_features_of_game(root, frame, game)
        out = pd.DataFrame([{"row": i, **v} for i, v in feats.items()],
                           columns=["row"] + NEW_FEATURES)
        out.insert(0, "sportvu_game", game)
        out.to_parquet(folder / f"{game}.parquet", index=False)
        built += 1
    done = {p.stem for p in folder.glob("*.parquet")}
    return {"games": len(games), "done": len(done), "built_this_job": built,
            "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2),
            "complete": len(done) == len(games)}


def with_new_features(artifacts=None):
    """The cached frozen frame with the three new columns joined by row (NaN where not built: has_contest 0)."""
    import numpy as np
    import pandas as pd
    frame = RX._features(artifacts)
    folder = out_dir(artifacts)
    parts = [pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))]
    missing = sorted(set(frame["sportvu_game"].astype(str)) - {p.stem for p in folder.glob("*.parquet")})
    if missing:
        raise SystemExit(f"{WHO}: features missing for {len(missing)} game(s); run the features command to the end")
    new = pd.concat(parts, ignore_index=True).set_index("row")
    for c in NEW_FEATURES:
        frame[c] = np.nan
        frame.loc[new.index, c] = new[c].astype(float).to_numpy()
    return frame


# ----------------------------------------------------------------------------- the fit shared by the three arms

def fit_arm(train, cols, seed: int = SEED) -> dict:
    """hgb_contest_prior with the given design columns, as analytics.xfg.train fits the frozen candidate (the frozen
    functions, unchanged): 3 inner boosters give out-of-fold scores for the isotonic step; a full booster on every
    training row (out-of-fold shooter prior) gives the raw score."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from analytics.xfg import train as T
    from analytics.xfg.features import PRIOR_FEATURE
    if train["heldout"].any():
        raise ValueError(f"{WHO}: a held-out row reached the fit")
    T.refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    games = train["sportvu_game"].astype(str)
    y = train["made"].astype(int).to_numpy()
    k = min(T.INNER_FOLDS, games.nunique())
    oof = np.full(len(train), np.nan)
    with T.fit_threads(1):
        for _j, (_tr, te_g) in enumerate(T.grouped_folds(games.unique(), k, seed)):
            m = games.isin(te_g).to_numpy()
            rows = train[~m].copy()
            rows[PRIOR_FEATURE] = T.oof_prior(rows, T.INNER_FOLDS, seed)
            test = train[m].copy()
            test[PRIOR_FEATURE] = T.apply_prior(T.prior_table(rows), test)
            h = HistGradientBoostingClassifier(random_state=seed, **T.HGB_PARAMS).fit(T._design(rows, cols),
                                                                                    rows["made"].astype(int))
            oof[m] = h.predict_proba(T._design(test, cols))[:, 1]
        full = train.copy()
        full[PRIOR_FEATURE] = T.oof_prior(full, T.INNER_FOLDS, seed)
        hgb = HistGradientBoostingClassifier(random_state=seed, **T.HGB_PARAMS).fit(T._design(full, cols), y)
    iso = IsotonicRegression(y_min=T.ISO_CLIP, y_max=1 - T.ISO_CLIP, out_of_bounds="clip").fit(oof, y)
    return {"cols": cols, "hgb": hgb, "iso": iso, "prior": T.prior_table(train)}


def predict_arm(parts: dict, frame):
    from analytics.xfg import train as T
    from analytics.xfg.features import PRIOR_FEATURE
    f = frame.copy()
    with T.fit_threads(1):
        f[PRIOR_FEATURE] = T.apply_prior(parts["prior"], f)
        return parts["iso"].predict(parts["hgb"].predict_proba(T._design(f, parts["cols"]))[:, 1])


def arm_columns() -> dict:
    from analytics.xfg.features import feature_columns
    base = feature_columns(with_prior=True)
    return {"baseline": base, "new": base + NEW_FEATURES, "ablation": base + ABLATION}


def outer(frame, games, arms: dict, seed: int = SEED):
    """Out-of-fold p of each arm over grouped outer folds of `games` -> (rows, {arm: p}, fold)."""
    import numpy as np
    from analytics.eval import grouped_folds
    rows = frame[frame["sportvu_game"].astype(str).isin(set(games))].reset_index(drop=True)
    g = rows["sportvu_game"].astype(str)
    out = {a: np.full(len(rows), np.nan) for a in arms}
    fold = np.full(len(rows), -1)
    for k, (_tr, te) in enumerate(grouped_folds(g.unique(), min(OUTER_FOLDS, g.nunique()), seed)):
        m = g.isin(te).to_numpy()
        for a, cols in arms.items():
            out[a][m] = predict_arm(fit_arm(rows[~m].reset_index(drop=True), cols, seed), rows[m].reset_index(drop=True))
        fold[m] = k
        print(f"[xfg-E2] outer fold {k}: {int(m.sum())} test shots", flush=True)
    return rows, out, fold


# ----------------------------------------------------------------------------- stages

def gate(artifacts=None) -> dict:
    import os
    import numpy as np
    import pandas as pd
    from tools.frozen_models import key_mismatches
    folder = begin_rebuild(RX.research_dir(artifacts) / "xfg_e2_gate", "xfg")
    frame = RX._features(artifacts)
    stored = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "analytics" / "xfg" / "2015-16" / "xfg_predictions.parquet",
                             columns=RX.STORED_KEYS + ["p_selected", "fold"])
    stored = stored[stored["fold"] >= 0].reset_index(drop=True)
    rows, arms, fold = outer(frame, frame["sportvu_game"].astype(str).unique(), {"baseline": arm_columns()["baseline"]})
    keys = key_mismatches(stored, rows) if len(stored) == len(rows) else {"length": [len(stored), len(rows)]}
    if any(keys.values()):
        raise SystemExit(f"{WHO}: the cached rows do not line up with the stored predictions: {keys}")
    diff = float(np.max(np.abs(arms["baseline"] - stored["p_selected"].to_numpy())))
    mismatches = int((fold != stored["fold"].to_numpy()).sum())
    pd.DataFrame({"p": arms["baseline"], "fold": fold}).to_parquet(folder / "gate_oof.parquet", index=False)
    record = {"name": "xfg", "compared": "the baseline arm of tools.research_xfg2 (the fit the new arms share), refit on all "
              "training games under the frozen outer folds, against the stored out-of-fold p_selected (fold >= 0); no "
              "held-out row, no metric", "n": int(len(rows)), "n_expected": int(len(stored)), "max_abs_diff": diff,
              "tolerance": TOLERANCE_CAP["xfg"], "fold_mismatches": mismatches, "passed": mismatches == 0}
    write_check(folder, record, ["gate_oof.parquet"])
    return require_passed(folder, "xfg")


def dev(artifacts=None) -> dict:
    import numpy as np
    head = clean_head()
    require_passed(RX.research_dir(artifacts) / "xfg_e2_gate", "xfg")
    frame = with_new_features(artifacts)
    games = dev_games(frame["sportvu_game"].astype(str).unique(), WHO)
    refuse_test(games, WHO)
    rows, arms, _fold = outer(frame, games, arm_columns())
    g = rows["sportvu_game"].astype(str).to_numpy()
    y = rows["made"].astype(int).to_numpy()
    loss = {a: RX.per_shot_loss(y, p) for a, p in arms.items()}
    out = {"experiment": "xfg-E2", "stage": "development games, 5 grouped outer folds, seed 0",
           "code_commit": head, "shots": int(len(y)), "games": int(len(set(g))),
           "new_feature_coverage": {c: float(np.isfinite(rows[c].to_numpy(float)).mean()) for c in NEW_FEATURES},
           "arms": {a: RX._arm_metrics(y, p) for a, p in arms.items()},
           "arm_intervals_95": {a: RX.arm_intervals(g, y, p) for a, p in arms.items()},
           "paired_new_minus_baseline": RX.paired_by_game(g, loss["new"], loss["baseline"]),
           "paired_ablation_minus_baseline": RX.paired_by_game(g, loss["ablation"], loss["baseline"]),
           "paired_new_minus_ablation": RX.paired_by_game(g, loss["new"], loss["ablation"]),
           "per_game_sd_new_minus_baseline": per_game_sd(g, loss["new"], loss["baseline"])}
    paired = out["paired_new_minus_baseline"]
    out["verdict"] = {"threshold": THRESHOLD, "diff": paired["diff"], "lo": paired["lo"], "hi": paired["hi"],
                      "adopts_on_development": bool(paired["diff"] <= THRESHOLD and paired["hi"] < 0),
                      "set_a": "scored once as a D108 confirmation (D121)" if paired["diff"] <= THRESHOLD and paired["hi"] < 0
                      else "not scored (D121, D109)"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "xfg-E2-dev.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


# ----------------------------------------------------------------------------- the Set A confirmation (D121)

EXPERIMENT = "xfg-E2"
REHEARSAL_GAMES = 10


def reading(lo: float, hi: float) -> str:
    """D121 with D108's readings: confirms (wholly below 0), contradiction (wholly above 0, stops the adoption),
    else inconclusive at 10 games."""
    if hi < 0:
        return "confirms"
    if lo > 0:
        return "contradiction"
    return "inconclusive at 10 games"


def score_rows(rows, parts: dict) -> dict:
    """Both arms on rows built by the scorer's path; every number of the confirmation."""
    import numpy as np
    g = rows["sportvu_game"].astype(str).to_numpy()
    y = rows["made"].astype(int).to_numpy()
    p = {a: predict_arm(parts[a], rows) for a in ("baseline", "new")}
    if not all(np.isfinite(v).all() for v in p.values()):
        raise SystemExit(f"{WHO}: a prediction is not finite")
    loss = {a: RX.per_shot_loss(y, v) for a, v in p.items()}
    paired = RX.paired_by_game(g, loss["new"], loss["baseline"])
    return {"shots": int(len(y)), "games": int(len(set(g))),
            "has_contest_share": float(rows["has_contest"].mean()),
            "new_feature_coverage": {c: float(np.isfinite(rows[c].to_numpy(float)).mean()) for c in NEW_FEATURES},
            "arms": {a: RX._arm_metrics(y, v) for a, v in p.items()},
            "arm_intervals_95": {a: RX.arm_intervals(g, y, v) for a, v in p.items()},
            "paired_new_minus_baseline": paired,
            "t_interval_new_minus_baseline_per_game": RX.per_game_t_interval(g, loss["new"], loss["baseline"]),
            "reading": reading(paired["lo"], paired["hi"])}


def fit_dev_arms(frame, games) -> dict:
    rows = frame[frame["sportvu_game"].astype(str).isin(set(games))].reset_index(drop=True)
    cols = arm_columns()
    return {a: fit_arm(rows, cols[a]) for a in ("baseline", "new")}


def clean_head() -> str:
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_of(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rehearse(artifacts=None) -> dict:
    """The scorer's path on 10 development games drawn with seed 0 in place of Set A, both arms fitted on the other
    development games, rows built from DATA_ROOT by the frozen feature code and add_new_features. Its rows and new
    features must equal the cached development ones of those games. Not a result: written under the artifacts
    folder."""
    import os
    import random
    from analytics.xfg import train as T
    from tools.frozen_common import MemoryGuard
    from tools.frozen_models import xfg_tables
    head = clean_head()
    require_passed(RX.research_dir(artifacts) / "xfg_e2_gate", "xfg")
    frame = with_new_features(artifacts)
    games = dev_games(frame["sportvu_game"].astype(str).unique(), WHO)
    stand_in = sorted(random.Random(SEED).sample(games, REHEARSAL_GAMES))
    parts = fit_dev_arms(frame, [g for g in games if g not in set(stand_in)])
    root = Path(os.environ["DATA_ROOT"])
    # the tables cut to the stand-in games, as the research root holds only the test games for the scorer
    tables = xfg_tables(root, MemoryGuard())                          # held-out labels blanked
    for name in ("shots", "shot_frames"):
        tables[name] = tables[name][tables[name]["sportvu_game"].astype(str).isin(set(stand_in))]
    built, held = T.assemble(tables)
    del held                                                          # held rows never used
    built = built[built["sportvu_game"].astype(str).isin(set(stand_in))].reset_index(drop=True)
    built = add_new_features(root, built)
    cached = frame[frame["sportvu_game"].astype(str).isin(set(stand_in))].reset_index(drop=True)
    check = rehearsal_check(built, cached)
    out = {"what": "rehearsal of the xfg-E2 scorer on 10 development games in place of Set A; not a result",
           "code_commit": head, "stand_in_games": len(stand_in), **check, **score_rows(built, parts)}
    (RX.research_dir(artifacts) / "xfg_e2_rehearsal.json").write_text(json.dumps(out, indent=2) + "\n")
    if not rehearsal_passed(out):
        raise SystemExit(f"{WHO}: the scorer's rows or columns differ from the development ones: {check}")
    return out


def key_tuples(rows) -> list:
    """(game, event, shooter) of each row as typed values, -1 for a row with no shooter id. The first rehearsal
    compared the keys as strings: a missing shooter id becomes a float NaN in that array, and NaN never equals NaN,
    so rows with the same keys were refused (diagnosis on the 10 stand-in games, 2026-11-19)."""
    import pandas as pd
    shooter = pd.to_numeric(rows["player_id"]).astype("Int64")
    return list(zip(rows["sportvu_game"].astype(str), rows["event"].astype(int),
                    [-1 if pd.isna(v) else int(v) for v in shooter]))


def same_column(a, b) -> bool:
    """Two columns hold the same values in the same order: numbers exactly, with the same NaN placement; anything else
    value by value, a missing value matching only a missing value. No tolerance."""
    import numpy as np
    import pandas as pd
    if len(a) != len(b):
        return False
    try:
        return bool(np.array_equal(pd.to_numeric(a).to_numpy(float), pd.to_numeric(b).to_numpy(float), equal_nan=True))
    except (TypeError, ValueError):
        return [None if pd.isna(x) else x for x in a] == [None if pd.isna(x) else x for x in b]


def rehearsal_check(built, cached) -> dict:
    """The rehearsal's exact check: the same rows in the same order (typed keys); for each new feature, and for every
    other column the two frames share (the frozen features and made), the same values with the same NaN placement,
    exactly (no tolerance)."""
    rows = len(built) == len(cached) and key_tuples(built) == key_tuples(cached)
    same = {c: bool(rows and same_column(built[c], cached[c])) for c in NEW_FEATURES}
    shared = sorted((set(built.columns) & set(cached.columns)) - set(NEW_FEATURES))
    others = {c: bool(rows and same_column(built[c], cached[c])) for c in shared}
    return {"rows_match_cache": bool(rows), "new_features_match_cache": same, "shared_columns_match_cache": others}


def rehearsal_passed(record: dict) -> bool:
    return bool(record.get("rows_match_cache")) and all(record.get("new_features_match_cache", {}).values()) \
        and bool(record.get("new_features_match_cache")) and all(record.get("shared_columns_match_cache", {}).values()) \
        and bool(record.get("shared_columns_match_cache"))


def make_scorer(parts: dict, artifacts=None):
    """The scorer score_once runs. The test data is reached only here."""
    from tools import research_testset as RT
    from tools.research_testset import TRACKING
    from tools.research_xfg import research_data_root, test_features

    def scorer(test_games):
        if RT._SCORING is None or RT._SCORING[1] != TRACKING:
            raise SystemExit(f"{WHO}: the test data is read only inside the scorer that score_once runs")
        rows = add_new_features(research_data_root(artifacts), test_features(sorted(test_games), artifacts))
        return {"experiment": EXPERIMENT, "stage": "fitted on every development game, scored once on the D71 test "
                "set as a D121 confirmation", **score_rows(rows, parts)}

    return scorer


def check_rehearsal(path, head: str) -> dict:
    """SystemExit unless the rehearsal json exists, every one of its checks passed, and it ran at commit `head`."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{WHO}: rehearse the scorer first")
    record = json.loads(path.read_text())
    if not rehearsal_passed(record):
        raise SystemExit(f"{WHO}: the rehearsal did not pass; fix and rehearse again before any score")
    if record.get("code_commit") != head:
        raise SystemExit(f"{WHO}: the rehearsal ran at {record.get('code_commit')}, not at this HEAD {head}; rehearse "
                         "again at this commit")
    return record


def final(artifacts=None) -> dict:
    """The one Set A score (D121): only when development adopts and the rehearsal ran."""
    head = clean_head()
    from tools.research_testset import TRACKING, score_once
    gate_record = require_passed(RX.research_dir(artifacts) / "xfg_e2_gate", "xfg")
    dev_path = RESULTS / "xfg-E2-dev.json"
    dev_out = json.loads(dev_path.read_text())
    if not dev_out["verdict"]["adopts_on_development"]:
        raise SystemExit(f"{WHO}: development does not adopt, so Set A is not scored (D121, D109)")
    check_rehearsal(RX.research_dir(artifacts) / "xfg_e2_rehearsal.json", head)
    rehearsal_sha = sha256_of(RX.research_dir(artifacts) / "xfg_e2_rehearsal.json")
    dev_sha = sha256_of(dev_path)
    if not RX.test_root_ready(artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games; run tools.research_ingest first")
    frame = with_new_features(artifacts)
    parts = fit_dev_arms(frame, dev_games(frame["sportvu_game"].astype(str).unique(), WHO))
    scorer = make_scorer(parts, artifacts)
    out = score_once(EXPERIMENT, TRACKING, scorer)
    record = {"experiment": EXPERIMENT, "code_commit": head, "dev_json_sha256": dev_sha,
              "rehearsal_json_sha256": rehearsal_sha,
              "gate": gate_record, "rule": "D121 with D108's readings: confirms if the 95 percent interval of new "
              "minus baseline lies wholly below 0, inconclusive at 10 games if it straddles 0, contradiction if "
              "wholly above 0 (stops the adoption); the t interval on the per-game means is reported beside it",
              "projected_10_game_half_width": dev_out["per_game_sd_new_minus_baseline"]["projected_10_game_half_width"],
              "scored": out}
    (RESULTS / "xfg-E2-final.json").write_text(json.dumps(record, indent=2) + "\n")
    return out


def per_game_sd(games, a, b) -> dict:
    """The spread of the per-game mean differences, and the half-width a 10-game test would have from it (normal
    approximation), reported before any Set A score (D121)."""
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"g": games, "d": a - b}).groupby("g")["d"].agg(["mean", "count"])
    sd = float(d["mean"].std(ddof=1))
    return {"games": int(len(d)), "sd_of_game_means": sd, "mean_shots_per_game": float(d["count"].mean()),
            "projected_10_game_half_width": float(1.96 * sd / math.sqrt(10))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("features", "gate", "dev", "rehearse", "final"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research xfg-E2 {a.command}"):
        if a.command == "features":
            out = build_features(a.artifacts, a.max_minutes)
        elif a.command == "gate":
            out = gate(a.artifacts)
        elif a.command == "dev":
            out = dev(a.artifacts)
        elif a.command == "rehearse":
            out = rehearse(a.artifacts)
        else:
            out = final(a.artifacts)
    print(json.dumps(out.get("verdict", out) if isinstance(out, dict) else out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

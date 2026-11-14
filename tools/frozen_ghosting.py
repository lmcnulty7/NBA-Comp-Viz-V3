"""tools/frozen_ghosting.py: the frozen ghosting model as an object a later process can load (PLAN.md D45, D49).

freeze-v1 fitted the selected ghosting model and rolled it over the held-out games in one process, and saved
nothing. This module reruns that fit unchanged on the training games, saves the two fitted objects, and keeps them
only when they reproduce the stored held-out ghost positions.

    python3 -m tools.frozen_ghosting project    measure the same code on two subsets of training games and project
                                                the memory of the full fit (rebuild refuses to start without it)
    python3 -m tools.frozen_ghosting rebuild    fit, save, check, write check.json
    python3 -m tools.frozen_ghosting check      rerun only the comparison on the saved model

The model (analytics/ghosting, frozen at freeze-v1): StandardScaler, then scikit-learn's MLPRegressor (two hidden
layers of 64, early stopping, seed 0) on the window of the last five sampled feature vectors, fitted on every 5th
sampled step (1 Hz) of every training game, stored as float32. It maps a defender's past to his next 0.2 s step,
so a ghost is where an average defender would have moved, never where the best one would.

Fit (`_fit`): the frozen trainer's own pass over the training games (validate.iter_data_root, train.split_games,
then per game validate.validate_game with train.prepare_game, one game's frames in memory at a time, fit rows
spilled to a temporary folder), then train.fit_frame and train.fit_candidates(frame, seed 0, 1 thread, "mlp"), the
call the frozen run made for its refit on every fit row. Held-out frames are never read by the fit.

Saved under <artifacts>/ghosting/: mlp.pkl (the fitted MLPRegressor and StandardScaler, taken out of the closure
fit_candidates returns, and the ids of the training games) and meta.json.

Check (`check`): the saved model is loaded from disk and, for each held-out game, one game at a time, the frozen
code rebuilds the panel and the feature frame from the tracking frames and rolls the model forward (the path
`predict_ghosts` gives a later process). Every stored ghost position at both horizons must be reproduced within
1e-3 ft. Predictions are compared with predictions: no outcome column is read and nothing is scored. The frozen
roll-forward computes each row's distance to the real defender as a by-product; it is dropped at once, never
summed and never written.

Use: `load(artifacts).predict_ghosts(frames, possessions, game)`.
"""

import gc
import json
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import fit_threads, is_heldout_game
from analytics.ghosting import features as F
from analytics.ghosting import train as T
from analytics.ghosting import validate as V
from export.heavy import heavy_job
from tools import frozen_common as C
from tools import frozen_epv as M      # the memory watch, the comparison and the projection are shared

NAME = "ghosting"
WHO = "tools.frozen_ghosting"
SEED = 0                     # the frozen run: --seed 0
DEEP = "mlp"                 # the frozen run kept the scikit-learn MLP (PLAN.md D8)
TOLERANCE = C.TOLERANCE_CAP[NAME]      # 1e-3 ft (PLAN.md D49)
UNIT = "ft"
LIMIT_GB = M.LIMIT_GB
MODEL_FILE, META_FILE = "mlp.pkl", "meta.json"
FILES = (MODEL_FILE, META_FILE)
# The frozen record (reports/ghosting.md): the fit rows of the final model and the ghost rows it wrote.
FROZEN_RECORD = {"train_games": 604, "fit_rows": 7147100, "heldout_games": 17, "ghost_rows": 1741995}
PROJECTION_GAMES = (50, 110)           # both above 500,000 fit rows, the scaler's chunk, so the line is straight
OUT_COLUMNS = ["sportvu_game", "possession", "frame", "def_player_id", "ghost_x", "ghost_y", "horizon_s"]
KEYS = ["possession", "frame", "def_player_id", "horizon_tenths"]
# What the frozen panel builder and its cleaning read from the possessions table: no outcome column.
POSSESSION_COLUMNS = ("possession", "offense_team_id", "start_frame", "end_frame", "start_clock_s", "gap")
# What the frozen panel builder reads from the frames table.
MODEL_FRAME_COLUMNS = ("frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "team_id",
                       "player_id", "x", "y")
SPILL_GB = {"full": 12.0, "subset": 3.0}       # free disk the spilled fit rows need, with a margin


# ---------------------------------------------------------------------------------------------------- the fit

def _prepare(train_games: list, spill: Path, limit_gb: float) -> None:
    """Pass 1 of the frozen trainer over the training games, one game's frames in memory at a time: clean, build
    the panel and the feature frame, keep the 1 Hz float32 fit rows in the spill folder."""
    guard = C.MemoryGuard(limit_gb)
    for i, game in enumerate(train_games, 1):
        game["validation"] = V.validate_game(game, on_frames=lambda g, frames: T.prepare_game(g, frames, spill))
        guard.check("prepare " + game["sportvu_game"])
        if i % 50 == 0 or i == len(train_games):
            M.say(WHO, f"prepared {i} of {len(train_games)} training games")


def _fit_models(train_games: list, spill: Path):
    """The frozen refit on every fit row: one memory-mapped fit frame, then train.fit_candidates as the frozen run
    called it. Returns the fitted MLPRegressor and StandardScaler and the number of fit rows."""
    frame = T.fit_frame(train_games, spill)
    rows = int(len(frame))
    with fit_threads(T.THREADS):
        models = T.fit_candidates(frame, seed=SEED, threads=T.THREADS, deep=DEEP)
    mlp, scaler = _parts_of(models["mlp"])
    return mlp, scaler, models["mlp"], rows


def _parts_of(predict):
    """The fitted MLPRegressor and StandardScaler inside the closure train.fit_candidates returns for the deep
    slot: `lambda W, m=mlp: m(W)` around `lambda W: model.predict(sc.transform(W))`."""
    inner = predict.__defaults__[0]
    cells = dict(zip(inner.__code__.co_freevars, (cell.cell_contents for cell in inner.__closure__ or ())))
    mlp, scaler = cells.get("model"), cells.get("sc")
    if type(mlp).__name__ != "MLPRegressor" or type(scaler).__name__ != "StandardScaler":
        raise RuntimeError(f"{WHO}: the fitted model is not the closure the frozen code is known to return")
    return mlp, scaler


def _fit(data_root, limit_gb: float, n_games=None, expect=None, every_s: float = 1.0, on_trip=None, cleanup=None) -> dict:
    """Load, prepare and fit. `n_games` limits the fit to the first training games (the projection); `expect`
    holds the frozen record's counts and stops the run before the fit when the inputs are not the ones it names.
    `cleanup` receives an action that removes the spill folder, for a run that has to leave abruptly."""
    games = V.iter_data_root(data_root)
    train_games, held_games = T.split_games(games)
    counts = {"train_games": len(train_games), "heldout_games": len(held_games)}
    if expect and counts["train_games"] != expect["train_games"]:
        raise SystemExit(f"{WHO}: {counts['train_games']} training games found, the frozen record has {expect['train_games']}")
    if n_games:
        train_games = train_games[:n_games]
    ids = [g["sportvu_game"] for g in train_games]
    C.training_games_only(ids, WHO)
    need = SPILL_GB["subset" if n_games else "full"]
    free = shutil.disk_usage(tempfile.gettempdir()).free / 1e9
    if free < need:
        raise SystemExit(f"{WHO}: {free:.0f} GB free in the temporary folder, the fit rows need {need:.0f} GB")
    stages = {}
    with tempfile.TemporaryDirectory(prefix="ghosting_fit_") as tmp:
        spill = Path(tmp)
        if cleanup is not None:
            cleanup.append(lambda: shutil.rmtree(spill, ignore_errors=True))
        with M.stage("prepare", stages, limit_gb, every_s, on_trip, WHO):
            _prepare(train_games, spill, limit_gb)
        fit_rows = int(sum(g["fit_rows"]["rows"] for g in train_games if g.get("fit_rows")))
        counts.update(fit_rows=fit_rows, feature_rows=int(sum(g["feature_rows"] for g in train_games)),
                      panel={k: int(sum(g["panel_counts"][k] for g in train_games)) for k in train_games[0]["panel_counts"]})
        if expect and not n_games and fit_rows != expect["fit_rows"]:
            raise SystemExit(f"{WHO}: {fit_rows} fit rows found, the frozen record has {expect['fit_rows']}")
        M.say(WHO, f"fit on {fit_rows} fit rows of {len(train_games)} training games")
        with M.stage("fit", stages, limit_gb, every_s, on_trip, WHO):
            mlp, scaler, predict, rows = _fit_models(train_games, spill)
    return {"mlp": mlp, "scaler": scaler, "predict": predict, "stages": stages, "counts": counts, "rows": rows, "games": ids}


# --------------------------------------------------------------------------------------------- save and read

def _save(folder: Path, fitted: dict, code: dict) -> None:
    """Write the model, read it back from disk, and confirm that it gives the fitted closure's displacements bit
    for bit before the metadata is written."""
    mlp, scaler = fitted["mlp"], fitted["scaler"]
    with open(folder / MODEL_FILE, "wb") as handle:
        pickle.dump({"mlp": mlp, "scaler": scaler, "train_games": list(fitted["games"])}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    again = _read(folder)
    rng = np.random.default_rng(0)
    window = scaler.mean_ + scaler.scale_ * rng.normal(size=(2000, len(F.window_columns())))
    if not np.array_equal(again.displacement(window), fitted["predict"](window)):
        raise RuntimeError(f"{WHO}: the model read back from disk does not equal the fitted model")
    meta = {"name": NAME, "model": "StandardScaler, then MLPRegressor %s with early stopping, on the %d window "
                                   "columns (the last %d sampled steps); slot name mlp in the reports"
                                   % (list(T.HIDDEN), len(F.window_columns()), F.N_LAGS),
            "frozen_tag": "freeze-v1", "code": code,
            "calls": ["analytics.ghosting.validate.iter_data_root", "analytics.ghosting.train.split_games",
                      "analytics.ghosting.validate.validate_game with analytics.ghosting.train.prepare_game, per game",
                      "analytics.ghosting.train.fit_frame",
                      "analytics.ghosting.train.fit_candidates(frame, seed 0, threads 1, 'mlp')"],
            "settings": {"seed": SEED, "deep": DEEP, "hidden": list(T.HIDDEN), "max_iter": T.MAX_ITER,
                         "threads": T.THREADS, "fit_every": F.FIT_EVERY, "fit_dtype": np.dtype(F.FIT_DTYPE).name,
                         "scaler_chunk": T.SCALER_CHUNK, "n_lags": F.N_LAGS},
            "train": {"games": len(fitted["games"]), "fit_rows": fitted["rows"]}, "counts": fitted["counts"],
            "final_fit": {"n_iter": int(mlp.n_iter_), "loss": float(mlp.loss_), "n_layers": int(mlp.n_layers_),
                          "weights": int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
                          "weights_dtype": str(mlp.coefs_[0].dtype)},
            "saved_equals_fitted": True, "versions": M.versions(), "stages": fitted["stages"],
            "machine_at_save": M.system_memory()}
    (folder / META_FILE).write_text(json.dumps(M.plain(meta), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(folder: Path) -> "GhostPredictor":
    """The predictor from the saved model, whatever the check says. `load` is the door for everyone else."""
    with open(Path(folder) / MODEL_FILE, "rb") as handle:
        parts = pickle.load(handle)
    return GhostPredictor(parts["mlp"], parts["scaler"], parts["train_games"])


# ---------------------------------------------------------------------------------------------- the predictor

def _possessions_for(possessions: pd.DataFrame, game: str) -> pd.DataFrame:
    """The possession columns the frozen panel builder and its cleaning read, for one game. No outcome column is
    among them, so none is read."""
    missing = [c for c in POSSESSION_COLUMNS if c not in possessions.columns]
    if missing:
        raise ValueError(f"possessions lack column(s) {missing}")
    rows = possessions
    if "sportvu_game" in possessions.columns:
        ids = possessions["sportvu_game"].astype(str)
        if (ids == game).any():
            rows = possessions[(ids == game).to_numpy()]
        elif ids.nunique() > 1:
            raise ValueError(f"possessions hold several games and none of them is {game!r}")
    return rows[list(POSSESSION_COLUMNS)].reset_index(drop=True).copy()


def _frames_for(frames: pd.DataFrame, game: str) -> pd.DataFrame:
    """The frames with the two bookkeeping columns the frozen cleaning asks for (game, n_players) added when a
    caller's table lacks them. The model reads neither."""
    missing = [c for c in MODEL_FRAME_COLUMNS if c not in frames.columns]
    if missing:
        raise ValueError(f"frames lack column(s) {missing}")
    out = frames
    if "game" not in out.columns:
        out = out.assign(game=game)
    if "n_players" not in out.columns:
        out = out.assign(n_players=out.groupby("frame")["player_id"].transform("size").astype("int8"))
    return out


def _no_ghosts() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([], dtype=T.GHOST_FRAMES_DTYPES[c]) for c in OUT_COLUMNS})


class GhostPredictor:
    """The frozen ghosting model, ready to predict. Nothing here fits: the panel and the features are built by the
    frozen code and the saved model is rolled forward by the frozen roll-forward."""

    def __init__(self, mlp, scaler, train_games):
        self.mlp, self.scaler = mlp, scaler
        self.train_games = frozenset(str(g) for g in train_games)
        self.check = None

    def displacement(self, window: np.ndarray) -> np.ndarray:
        """The model's 0.2 s step (dx, dy) in feet for rows of the window matrix: the two calls of the frozen
        closure, on the same two fitted objects."""
        return self.mlp.predict(self.scaler.transform(window))

    def predict_ghosts(self, frames: pd.DataFrame, possessions: pd.DataFrame, game: str) -> pd.DataFrame:
        """Ghost positions for the defenders of one game's possessions, from its tracking frames.

        `frames` is one game's frames table (docs/DATA_CONTRACT.md section 4: one row per player per frame at
        25 Hz; frame, period, game_clock_s, shot_clock_s, ball_x, ball_y, ball_z, team_id, player_id, x, y).
        `possessions` is its possessions table (section 19); only possession, offense_team_id, start_frame,
        end_frame, start_clock_s and gap are read. Returns sportvu_game, possession, frame, def_player_id,
        ghost_x, ghost_y, horizon_s with the dtypes of the stored table: one row per defender, sampled frame and
        horizon, the 0.2 s rows first, then the 1.0 s rows (select them with horizon_s > 0.5). `frame` is the
        frame the ghost is for.

        Each ghost is one defender's answer with the other nine players real: where an average defender would be
        0.2 s or 1.0 s after his real position, given how everyone else moved. A possession is used only when its
        whole span lies inside `frames` (the frozen cleaning drops it otherwise), when it has no clock gap, and
        from its fourth sampled frame on; the 1.0 s rows start at the eighth. A training game is refused: the
        model was fitted on it, so its ghosts there would be in-sample."""
        game = str(game)
        if game in self.train_games:
            raise ValueError(f"{game} is a training game of this model; its ghosts there would be in-sample")
        one = {"sportvu_game": game, "heldout": bool(is_heldout_game(game)), "frames": _frames_for(frames, game),
               "possessions": _possessions_for(possessions, game)}
        counts = V.validate_game(one)["possessions"]
        try:
            panel, rows = T.game_rows(one)
        except IndexError:                # the frozen builder's answer to a game without a usable possession
            panel, _ = F.build_panel_game(V.frames_of(one), one["possessions"], game)
            if len(panel):
                raise
            rows = panel
        out = _no_ghosts()
        if len(rows):
            with fit_threads(T.THREADS):
                rolled = T.roll_forward({"mlp": self.displacement}, rows, panel, fold=T.HELDOUT_FOLD)
            out = rolled.loc[rolled["candidate"] == "mlp", OUT_COLUMNS].reset_index(drop=True)
            out = out.astype({c: T.GHOST_FRAMES_DTYPES[c] for c in OUT_COLUMNS})
        out.attrs.update(rows=int(len(out)), possessions=int(out["possession"].nunique()),
                         possessions_kept=int(counts["kept"]), possessions_outside_frames=int(counts["span_outside_frames_dropped"]),
                         possessions_with_gap=int(counts["gap"]))
        return out


# -------------------------------------------------------------------------------------------------- the check

def _heldout_games(data_root: Path) -> list:
    """The held-out games that have a stored ghost table, in sorted order."""
    base = data_root / "analytics" / "ghosting" / "heldout" / V.SEASON
    games = sorted(p.stem[len("ghost_frames_"):] for p in base.glob("ghost_frames_*.parquet")) if base.is_dir() else []
    if not games:
        raise SystemExit(f"{WHO}: no stored held-out ghost tables under analytics/ghosting/heldout")
    return games


def _keyed(table: pd.DataFrame) -> pd.DataFrame:
    """A ghost table with the horizon as a whole number of tenths of a second, so that it can be a join key."""
    return table.assign(horizon_tenths=np.rint(table["horizon_s"].to_numpy(dtype=float) * 10).astype(int))


def check(data_root=None, artifacts=None, expect=True, limit_gb: float = LIMIT_GB) -> dict:
    """Compare the saved model with the frozen record and write check.json. Reads the held-out tables without
    their outcome columns, one game's frames at a time; fits nothing, scores nothing. `expect` is True for the
    counts of FROZEN_RECORD (the real data root), a record of counts, or None for a root that is not the frozen
    one (the fixtures)."""
    import pyarrow.parquet as pq
    expect = M.expected(expect, FROZEN_RECORD)
    data_root = M.data_root_of(data_root)
    folder = C.assert_safe_output(C.artifacts_dir(artifacts) / NAME, data_root)
    missing = [name for name in FILES if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"{WHO}: no rebuilt model to check ({', '.join(missing)} missing); run rebuild first")
    started = time.time()
    predictor = _read(folder)
    games = _heldout_games(data_root)
    guard = C.MemoryGuard(limit_gb)
    parts, by_horizon, stored_rows = {}, {2: {}, 10: {}}, 0
    for game in games:
        stored = _keyed(pd.read_parquet(V.output_path(data_root, "ghost_frames", game, True),
                                        columns=["possession", "frame", "def_player_id", "horizon_s", "ghost_x", "ghost_y"]))
        ppath = schema.table_path(data_root / "analytics", "possessions", V.SEASON, game, heldout=True)
        names = pq.read_schema(ppath).names
        possessions = pd.read_parquet(ppath, columns=[c for c in ("sportvu_game",) + POSSESSION_COLUMNS if c in names])
        frames = pd.read_parquet(data_root / "sportvu_contract" / "heldout" / f"{game}.parquet", columns=V.FRAME_COLUMNS)
        predicted = _keyed(predictor.predict_ghosts(frames, possessions, game))
        del frames
        parts[game] = M.compare(predicted, stored, KEYS, ["ghost_x", "ghost_y"], TOLERANCE)
        for tenths in by_horizon:
            by_horizon[tenths][game] = M.compare(predicted[predicted["horizon_tenths"] == tenths],
                                                 stored[stored["horizon_tenths"] == tenths], KEYS, ["ghost_x", "ghost_y"],
                                                 TOLERANCE, worst=0)
        stored_rows += int(len(stored))
        M.say(WHO, f"{game}: {parts[game]['n']} of {len(stored)} stored rows compared, max abs diff "
                   f"{parts[game]['max_abs_diff']}, passed {parts[game]['passed']}")
        del stored, predicted
        gc.collect()
        guard.check("check " + game)
    total = M.combine(parts)
    horizons = {f"{tenths / 10:.1f} s": {k: v for k, v in M.combine(results).items() if k != "per_game"}
                for tenths, results in by_horizon.items()}
    found = {"heldout_games": len(games), "ghost_rows": stored_rows}
    as_recorded = not expect or all(found[k] == expect[k] for k in found)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    record = {"name": NAME, "unit": UNIT,
              "compared": "ghost positions of the rebuilt MLP for the held-out games, rolled forward from their tracking "
                          "frames, against the stored ghost_x and ghost_y at both horizons "
                          "($DATA_ROOT/analytics/ghosting/heldout); predictions with predictions, no outcome column "
                          "read, nothing scored",
              "n": total["n"], "n_expected": stored_rows, "max_abs_diff": total["max_abs_diff"], "tolerance": TOLERANCE,
              "passed": bool(total["passed"] and as_recorded),
              "positions": total, "by_horizon": horizons,
              "found": found, "frozen_record": expect, "matches_frozen_record": bool(as_recorded),
              "fit": {"train": meta.get("train"), "final_fit": meta.get("final_fit"), "stages": meta.get("stages"),
                      "code": meta.get("code")},
              "versions": M.versions(), "check_seconds": round(time.time() - started, 1), "machine": M.system_memory()}
    M.write_check(folder, record, files=FILES)
    return C.read_check(folder)


# ------------------------------------------------------------------------------------------- rebuild and load

def rebuild(data_root=None, artifacts=None, limit_gb: float = LIMIT_GB, expect=True,
            require_projection: bool = True) -> dict:
    """Rerun the frozen final fit on the training games, save the model, run the check, and return its record.

    Writes only under <artifacts>/ghosting (the fit rows go to a temporary folder that is removed). Take the
    heavy-job lock around this call (the command line does). After a MemoryError the process must end. `expect`
    as in `check`."""
    expect = M.expected(expect, FROZEN_RECORD)
    data_root = M.data_root_of(data_root)
    folder = C.assert_safe_output(C.artifacts_dir(artifacts) / NAME, data_root)
    if require_projection:
        M.need_projection(folder, limit_gb, WHO)
    code = M.require_frozen_code(WHO)
    C.begin_rebuild(folder, NAME)
    cleanup = []
    on_trip = M.stopper(folder, NAME, TOLERANCE, cleanup)
    M.say(WHO, f"rebuild starts; machine {M.system_memory()}")
    try:
        fitted = _fit(data_root, limit_gb, expect=expect, on_trip=on_trip, cleanup=cleanup)
        with M.stage("save", fitted["stages"], limit_gb, on_trip=on_trip, who=WHO):
            _save(folder, fitted, code)
    except BaseException as error:
        M.write_stopped(folder, NAME, TOLERANCE, M.stopped_reason(error))
        raise
    del fitted
    gc.collect()
    return check(data_root, artifacts, expect=expect, limit_gb=limit_gb)


def load(artifacts=None) -> GhostPredictor:
    """The frozen ghosting predictor, or RuntimeError when there is no rebuilt model, its check did not pass, a
    file changed after the check, or the libraries are not the ones the check ran under."""
    folder = C.artifacts_dir(artifacts) / NAME
    record = C.require_passed(folder, NAME)
    M.same_versions(record, NAME)
    predictor = _read(folder)
    predictor.check = record
    return predictor


# -------------------------------------------------------------------------------------------- the projection

def subset(data_root, n_games: int, limit_gb: float = LIMIT_GB) -> dict:
    """The same prepare and fit on the first `n_games` training games, measured. Nothing is saved."""
    fitted = _fit(M.data_root_of(data_root), limit_gb, n_games=n_games, every_s=0.25)
    return {"games": len(fitted["games"]), "rows": fitted["rows"], "stages": fitted["stages"],
            "n_iter": int(fitted["mlp"].n_iter_)}


def project(data_root=None, artifacts=None, games=PROJECTION_GAMES, limit_gb: float = LIMIT_GB) -> dict:
    """Measure the fit on subsets of training games, each in its own process, project the memory and the time of
    the full fit, and write the projection. Preparing a game costs the same memory at every size; the fit grows
    with the fit rows."""
    data_root = M.data_root_of(data_root)
    folder = C.assert_safe_output(C.artifacts_dir(artifacts) / NAME, data_root)
    runs = M.run_subsets("tools.frozen_ghosting", data_root, games, limit_gb)
    record = M.extrapolate(runs, FROZEN_RECORD["fit_rows"], limit_gb, fixed=("prepare",), scaled=("fit",))
    per_game = max(r["stages"]["prepare"]["seconds"] / r["games"] for r in runs)
    record["projected_seconds"] = round(per_game * FROZEN_RECORD["train_games"] + record["fit_seconds"]["projected"])
    record.update(name=NAME, what="the frozen ghosting final fit, measured on the first training games and "
                                  "projected to every fit row")
    M.write_projection(folder, record)
    return record


# ------------------------------------------------------------------------------------------- the command line

def main(argv=None) -> int:
    args = M.parser_for("tools.frozen_ghosting", "Rebuild and check the frozen ghosting model (PLAN.md D45).",
                        PROJECTION_GAMES).parse_args(argv)
    with M.terminating(), heavy_job(f"frozen_ghosting {args.command}", log=lambda m: print(m, file=sys.stderr, flush=True)):
        if args.command == "subset":
            print("SUBSET " + json.dumps(M.plain(subset(args.data, args.games, args.limit_gb))), flush=True)
            return 0
        if args.command == "project":
            record = project(args.data, args.artifacts, [int(k) for k in args.games.split(",")], args.limit_gb)
            print(json.dumps(record, indent=2))
            return 0 if record["ok"] else 2
        try:
            record = rebuild(args.data, args.artifacts, args.limit_gb) if args.command == "rebuild" else \
                check(args.data, args.artifacts, limit_gb=args.limit_gb)
        except MemoryError as error:
            print(f"{WHO}: stopped, {error}", file=sys.stderr, flush=True)
            return M.MEMORY_EXIT
        print(json.dumps(record, indent=2))
        return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

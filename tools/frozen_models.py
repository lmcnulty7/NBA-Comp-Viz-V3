"""tools/frozen_models.py: rebuild what freeze-v1 computed and never stored (PLAN.md D45, first half).

    python3 -m tools.frozen_models absence    [--artifacts DIR]
    python3 -m tools.frozen_models embeddings [--artifacts DIR]
    python3 -m tools.frozen_models xfg        [--artifacts DIR]
    python3 -m tools.frozen_models status     [--artifacts DIR]

Each rebuild reruns frozen code unchanged, with the frozen settings (seed 0, 5 folds, 200 bootstrap resamples), on
training games only. It reads DATA_ROOT (the environment variable), writes only under <artifacts>/<name>/
(--artifacts, else V3_ARTIFACTS, else artifacts/ in the repository; never under DATA_ROOT, never under reports/),
holds the machine-wide heavy-job lock while it runs, and ends with check.json (tools.frozen_common). A consumer
reads an artifact only through load_absence, load_embeddings or load_xfg, which refuse anything whose check did
not pass or whose files changed after the check.

absence     The out-of-fold absence probabilities of the load lane, which the lane computed in memory and never
            wrote. Reads the stored training table $DATA_ROOT/analytics/load/<season>/player_games.parquet and
            calls analytics.load.train unchanged (training_frames, absence_rows, out_of_fold with fit_absence).
            Writes oof.parquet: sportvu_game, player_id, game_date, p_logistic, absent_next, fold.
            Check: the frozen scorer on the written rows gives the numbers reports/load.md prints for the absence
            candidates (the log loss and the Brier score of the logistic model first of all), each within half of
            its last printed digit. Training-fold numbers only: no held-out row is read.

embeddings  The three tables of the embeddings lane, which had no output root at freeze-v1. Runs the frozen trainer
            itself in a child process: python3 -m analytics.embeddings.train --data $DATA_ROOT --seed 0
            --out <folder> --report <folder>/report.md. With those two options the trainer writes its tables under
            <folder>/analytics/embeddings/<season>/ and its report at <folder>/report.md, and nowhere else. The
            tables are then copied to the top of the folder, where the check vouches for them by hash.
            Check: the player counts and the 5-NN position accuracy with its interval in the new report are the
            ones in reports/embeddings.md, and every generated section of the two reports is the same line for line.

xfg         The frozen xFG model: hgb_contest_prior fitted once on every training row. Calls analytics.xfg unchanged
            (validate.load_data_root, validate.clean, train.assemble, train.fit_candidate), never the lane's main,
            which would overwrite the only copy of the stored predictions. Writes model.pkl, the fitted parts of the
            frozen predictor. Check: the saved parts, read back from disk, predict the held-out feature rows as the
            stored p_selected of the held-out rows of $DATA_ROOT/analytics/xfg/<season>/xfg_predictions.parquet,
            within 1e-6. Predictions against predictions: the label of every held-out attempt is overwritten before
            a feature is built, the stored outcome column is never read, and no metric is computed.

status      Lists the artifacts and whether each one can be used.
"""

import argparse
import contextlib
import hashlib
import inspect
import itertools
import json
import math
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from export.schemas import find_private
from tools.frozen_common import (CHECK_NAME, REPO_ROOT, TOLERANCE_CAP, MemoryGuard, artifacts_dir, assert_safe_output,
                                 begin_rebuild, read_check, require_passed, training_games_only, write_check)

WHO = "tools.frozen_models"
NAMES = ("absence", "embeddings", "xfg")
SEASON = "2015-16"
# The frozen settings: every group 1 command line named --seed 0 and left folds and bootstrap at the trainers'
# defaults. tests/test_frozen_models.py holds these against the frozen parsers.
SEED, FOLDS, BOOTSTRAP = 0, 5, 200
FROZEN_TAG = "freeze-v1"
CHANGED_SINCE_FREEZE = ("analytics/results.py",)     # the stage 9 parser, the one allowed difference (CLAUDE.md rule 3)

ABSENCE_FILE = "oof.parquet"
ABSENCE_COLUMNS = ["sportvu_game", "player_id", "game_date", "p_logistic", "absent_next", "fold"]
FOLD_LABEL_TOLERANCE = 1e-12

EMBEDDING_TABLES = ("player_season_features", "player_embeddings", "embedding_neighbours")
REPORT_NAME, LOG_NAME = "report.md", "train.log"
HAND_ADDED_SECTION = "Stage 8"      # written by hand into every committed lane report after the stage 8 run
OUTPUT_LINE = "Written under "      # the line a trainer adds to its Outputs section when it has an output root

XFG_CANDIDATE = "hgb_contest_prior"
XFG_TOLERANCE = TOLERANCE_CAP["xfg"]      # fixed for every rebuild in tools.frozen_common (PLAN.md D49)
MODEL_NAME = "model.pkl"
STORED_COLUMNS = ["sportvu_game", "event", "player_id", "p_selected", "fold", "heldout"]    # never the outcome


# ----------------------------------------------------------------------------- shared frame

def _say(message: str) -> None:
    print(message, flush=True)


def data_root() -> Path:
    """The folder DATA_ROOT names. SystemExit when the variable is not set or is not a folder."""
    value = os.environ.get("DATA_ROOT")
    if not value:
        raise SystemExit(f"{WHO}: DATA_ROOT is not set")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{WHO}: DATA_ROOT is not a folder that exists")
    return root


def versions() -> dict:
    """The interpreter and library versions a rebuild ran under. A fitted object is only trusted under the same ones."""
    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}


def frozen_code() -> dict:
    """How analytics/ and sportvu/ of this checkout differ from the tag freeze-v1 (the working tree against the tag).
    `verified` is false when git or the tag is not available; `unexpected` lists every changed file but the one
    difference the plan allows."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=120)

    try:
        diff, head = git("diff", "--name-only", FROZEN_TAG, "--", "analytics", "sportvu"), git("rev-parse", "HEAD")
    except (OSError, subprocess.TimeoutExpired):
        return {"tag": FROZEN_TAG, "verified": False}
    if diff.returncode != 0 or head.returncode != 0:
        return {"tag": FROZEN_TAG, "verified": False}
    changed = sorted(line.strip() for line in diff.stdout.splitlines() if line.strip())
    return {"tag": FROZEN_TAG, "verified": True, "head": head.stdout.strip(), "changed": changed,
            "unexpected": [path for path in changed if path not in CHANGED_SINCE_FREEZE]}


def require_frozen_code() -> dict:
    """frozen_code(), or SystemExit when the frozen packages were changed or are imported from another checkout."""
    import analytics
    if Path(analytics.__file__).resolve().parents[1] != REPO_ROOT:
        raise SystemExit(f"{WHO}: analytics is imported from another checkout; run from the repository root")
    state = frozen_code()
    if state.get("unexpected"):
        raise SystemExit(f"{WHO}: frozen code differs from {FROZEN_TAG} in {state['unexpected']}; nothing is rebuilt")
    return state


def public_text(text: str, artifacts_root, root) -> tuple:
    """`text` with the local folders named instead of spelled out -> (text, how many private strings are left).
    The artifacts folder becomes $V3_ARTIFACTS; DATA_ROOT, the repository and the home folder follow the frozen
    rule of analytics.report_paths, the one run_all applied to every committed report."""
    from analytics.report_paths import redact
    text = redact(text.replace(str(artifacts_root), "$V3_ARTIFACTS"), data_root=str(root))
    return text, len(find_private(text))


def _shown(path) -> str:
    """A path as a check may record it: relative to the repository, else the file name alone."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if REPO_ROOT in path.parents else path.name


def compare_printed(what: str, printed, recomputed, required: bool = False) -> dict:
    """One number as a report prints it against its recomputed value. They agree when the difference is at most
    half of the last printed digit; a number the report does not give (None, or n/a) agrees only with a value
    that is not finite."""
    row = {"what": what, "reported": printed, "recomputed": None, "abs_diff": None, "half_unit": None,
           "agrees": False, "required": bool(required)}
    try:
        value = float(recomputed)
    except (TypeError, ValueError):
        return row
    if printed is None or printed == "n/a":
        row["agrees"] = not math.isfinite(value)
        return row
    try:
        reported = float(printed)
    except ValueError:
        return row
    if not math.isfinite(value):
        return row
    half_unit = 0.5 * 10 ** -len(str(printed).partition(".")[2])
    row.update(recomputed=value, abs_diff=abs(value - reported), half_unit=half_unit)
    row["agrees"] = row["abs_diff"] <= half_unit
    return row


def compare_same(what: str, committed, rebuilt, required: bool = False) -> dict:
    """One value as two reports print it. They agree only when the printed text is identical."""
    row = {"what": what, "committed": committed, "rebuilt": rebuilt, "abs_diff": None,
           "agrees": committed == rebuilt, "required": bool(required)}
    try:
        row["abs_diff"] = abs(float(committed) - float(rebuilt))
    except (TypeError, ValueError):
        pass
    return row


def _exact(what: str, reported, recomputed) -> dict:
    return {"what": what, "reported": reported, "recomputed": recomputed, "agrees": reported == recomputed}


def _numbers(rows: list) -> dict:
    """n, max_abs_diff and tolerance of a list of compare_printed or compare_same rows, as write_check wants them."""
    diffs = [row["abs_diff"] for row in rows if row["abs_diff"] is not None]
    units = [row.get("half_unit", 0.0) for row in rows if row["abs_diff"] is not None]
    complete = len(diffs) == len(rows) or all(row["agrees"] for row in rows if row["abs_diff"] is None)
    return {"n": len(rows), "max_abs_diff": max(diffs) if diffs and complete else None,
            "tolerance": max(units) if units else None}


def _stopped(folder: Path, name: str, error: BaseException, root: Path) -> None:
    """Record that a rebuild stopped before its check. The folder stays not passed."""
    message, left = public_text(str(error), folder.parent, root)
    write_check(folder, {"name": name, "compared": "nothing: the rebuild stopped before its check", "n": 0,
                         "max_abs_diff": None, "tolerance": None, "passed": False, "status": "stopped",
                         "error": type(error).__name__ + (": " + message[:500] if message and not left else "")})


def _rebuild(name: str, artifacts, body) -> dict:
    """The frame every rebuild shares: the output guard, the frozen-code check, the heavy-job lock, the not-passed
    marker, the memory guard and check.json. `body(folder, root, guard)` returns (check record, files it vouches
    for). Returns the check as written."""
    root = data_root()
    folder = assert_safe_output(artifacts_dir(artifacts) / name)
    code = require_frozen_code()
    with heavy_job(f"{WHO} {name}"):
        begin_rebuild(folder, name)
        started = time.time()
        try:
            record, files = body(folder, root, MemoryGuard())
        except BaseException as error:
            _stopped(folder, name, error, root)
            raise
        record = dict(record, name=name, status="finished", season=SEASON, seed=SEED, code=code, versions=versions(),
                      seconds=round(time.time() - started, 1))
        path = write_check(folder, record, files=files)
        if find_private(path.read_text(encoding="utf-8")):
            error = RuntimeError("the check record carried a local path or an address")
            _stopped(folder, name, error, root)
            raise error
    return read_check(folder)


# ----------------------------------------------------------------------------- absence

def absence_oof(player_games) -> tuple:
    """The frozen out-of-fold absence predictions for a stored training player_games table -> (pred, note).

    `pred` is what analytics.load.train.out_of_fold returns: the player-games with a known absent_next, with p_base
    and p_logistic. SystemExit when the table holds a held-out game id or a row flagged held out: the rebuild reads
    training games only."""
    from analytics.load import train as L
    training_games_only(player_games["sportvu_game"].unique(), f"{WHO} absence")
    if player_games["heldout"].astype(bool).any():
        raise SystemExit(f"{WHO} absence: the training table carries rows flagged held out")
    train, heldout = L.training_frames(player_games)      # the frozen split and guards: held-out ids, time-forward, no leak
    if len(heldout) or len(train) != len(player_games):
        raise SystemExit(f"{WHO} absence: the frozen split set {len(heldout)} row(s) aside as held out")
    pred, note = L.out_of_fold(L.absence_rows(train), L.fit_absence, ("base", "logistic"), FOLDS, SEED)
    if note.get("mode") != "out-of-fold":
        raise SystemExit(f"{WHO} absence: {note.get('text')}")
    return pred, note


def fold_labels(games, n_folds: int, seed: int = SEED) -> dict:
    """game id -> the index of the grouped fold that holds the game out, in the order analytics.eval.grouped_folds
    yields the folds: the order the frozen out_of_fold loop fits in, and the numbering analytics.eval.grouped_cv
    writes."""
    from analytics.eval import grouped_folds
    ids = sorted({str(game) for game in games})
    return {game: k for k, (_train, test) in enumerate(grouped_folds(ids, n_folds=n_folds, seed=seed)) for game in test}


def refit_by_fold(rows, fold) -> tuple:
    """(p_base, p_logistic) of every row from the frozen fit on the rows outside its fold: what the fold labels
    claim the out-of-fold predictions are. Used only to verify the labels."""
    import numpy as np
    from analytics.load import train as L
    fold = np.asarray(fold)
    base, logistic = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for k in sorted({int(value) for value in fold}):
        test = fold == k
        models = L.fit_absence(rows[~test])
        base[test], logistic[test] = models["base"](rows[test]), models["logistic"](rows[test])
    return base, logistic


def absence_table(pred, labels: dict):
    """The rows oof.parquet holds (ABSENCE_COLUMNS), in the order of `pred`."""
    import numpy as np
    import pandas as pd
    games = pred["sportvu_game"].astype(str).to_numpy()
    table = pd.DataFrame({"sportvu_game": games, "player_id": pred["player_id"].to_numpy(),
                          "game_date": pred["game_date"].astype(str).to_numpy(),
                          "p_logistic": pred["p_logistic"].to_numpy(dtype=float),
                          "absent_next": pred["absent_next"].astype(bool).to_numpy(),
                          "fold": np.array([labels[game] for game in games], dtype="int16")})
    p = table["p_logistic"].to_numpy()
    if not len(table) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or (table["fold"] < 0).any():
        raise RuntimeError(f"{WHO} absence: the out-of-fold rows are empty or hold a value that is not a probability")
    if table.duplicated(["sportvu_game", "player_id"]).any():
        raise RuntimeError(f"{WHO} absence: a player-game appears twice")
    return table[ABSENCE_COLUMNS]


def _cell(cell: str) -> dict:
    """A report cell such as `0.2764 [0.2644, 0.2883]` as printed -> {point, lo, hi}; lo and hi are None when the
    report gives no interval."""
    from analytics import results as R
    metric = R.metric("cell", cell)
    lo, hi = [part.strip() for part in metric.interval.strip("[]").split(",")] if metric.interval else (None, None)
    return {"point": metric.value, "lo": lo, "hi": hi}


def parse_load_report(text: str) -> dict:
    """What a load report prints for the absence candidates (the section "Candidate 3"), as printed. These are
    out-of-fold numbers; the held-out sections of the report are never read."""
    from analytics import results as R
    from analytics.load.train import ABS_ROWS
    section = R.Doc(text).need("Candidate 3: absence from the next game", level=2)
    head = section.search(r"^(?P<protocol>.+?)\. (?P<rows>\d+) player-games with a known absent_next from (?P<games>\d+) "
                          r"games, base rate (?P<rate>[0-9.]+)\. Logistic C", "the sentence with the row counts")
    out = {"protocol": head["protocol"], "rows": int(head["rows"]), "games": int(head["games"]), "base_rate": head["rate"]}
    table = section.table("model", "log loss", "Brier", "ECE (10 bins)")
    for key, label in ABS_ROWS.items():
        row = table.one("model", lambda cell, label=label: R.clean(cell) == label, f"the {key} row")
        out[key] = {"log_loss": _cell(table.cell(row, "log loss")), "brier": _cell(table.cell(row, "Brier")),
                    "ece": _cell(table.cell(row, "ECE (10 bins)"))}
    paired = section.search(r"log loss\(logistic\) - log loss\(base\): (\S+) \[(\S+), (\S+)\], excludes zero: (True|False)\. "
                            r"Calibration slope of the logistic model: (\S+) \(", "the paired bootstrap sentence")
    out["paired"] = {"diff": paired[1], "lo": paired[2], "hi": paired[3]}
    out["excludes_zero"] = paired[4] == "True"
    out["calibration_slope"] = paired[5]
    out["selected"] = section.search(r"^Selected candidate: (.+?)\.$", "the selected candidate").group(1)
    try:
        bins = section.table("bin", "n", "mean p", "mean absent")
        out["calibration_n"] = [int(bins.cell(row, "n")) for row in bins.rows]
    except R.ParseError:
        out["calibration_n"] = None       # one class only: the frozen writer prints no calibration table
    return out


def check_absence(written, pred, reported: dict, note: dict) -> dict:
    """The frozen scorer on the written rows against what the load report prints for the absence candidates.

    `written` is oof.parquet read back, `pred` the frame the frozen out_of_fold returned (it carries the features
    the fold check refits on), `reported` is parse_load_report of the committed report. Every printed number must
    agree within half of its last digit, the counts, the protocol sentence and the selected candidate exactly, and
    the written fold labels must be the folds the frozen loop fitted."""
    import numpy as np
    from analytics.load import train as L
    record = {"compared": "the frozen scorer (analytics.load.train.score_absence) on the rows of oof.parquet, with the "
                          "numbers reports/load.md prints for the absence candidates; each within half of its last "
                          "printed digit", "unit": "numbers as reports/load.md prints them (4 decimals)",
              "n": 0, "max_abs_diff": None, "tolerance": None, "passed": False}
    same_rows = len(written) == len(pred) and bool(
        (written["sportvu_game"].astype(str).to_numpy() == pred["sportvu_game"].astype(str).to_numpy()).all()
        and (written["player_id"].to_numpy() == pred["player_id"].to_numpy()).all())
    record["rows_written"] = int(len(written))
    if not same_rows:
        return dict(record, problem="the rows read back from oof.parquet are not the rows that were fitted")
    base, logistic = refit_by_fold(pred, written["fold"].to_numpy())
    fitted = pred["p_logistic"].to_numpy(dtype=float)
    label_diff = float(np.max(np.abs(np.concatenate([base - pred["p_base"].to_numpy(dtype=float), logistic - fitted,
                                                     written["p_logistic"].to_numpy(dtype=float) - fitted]))))
    got = L.score_absence(written.assign(p_base=base), BOOTSTRAP, SEED)
    values = [compare_printed("base rate", reported["base_rate"], got["base_rate"])]
    for model in ("base", "logistic"):
        for metric in ("log_loss", "brier", "ece"):
            for part in ("point", "lo", "hi"):
                required = model == "logistic" and metric in ("log_loss", "brier") and part == "point"
                values.append(compare_printed(f"{model} {metric} {part}", reported[model][metric][part],
                                              got[model][metric][part], required))
    for part in ("diff", "lo", "hi"):
        values.append(compare_printed(f"paired log loss, logistic minus base, {part}", reported["paired"][part],
                                      got["diff"][part]))
    values.append(compare_printed("calibration slope of the logistic model", reported["calibration_slope"],
                                  got["logistic"]["calibration_slope"]))
    selected = "logistic" if got["logistic"]["log_loss"]["point"] <= got["base"]["log_loss"]["point"] else "base"
    bins = None if got["calibration"] is None else [int(n) for n in got["calibration"]["n"]]
    exact = [_exact("player-games with a known absent_next", reported["rows"], got["rows"]),
             _exact("games", reported["games"], got["games"]),
             _exact("protocol sentence", reported["protocol"], note["text"]),
             _exact("paired interval excludes zero", reported["excludes_zero"], bool(got["diff"]["excludes_zero"])),
             _exact("selected candidate", reported["selected"], L.ABS_ROWS[selected]),
             _exact("rows per calibration bin of the logistic model", reported["calibration_n"], bins)]
    record.update(_numbers(values))
    record.update(values=values, exact=exact, rows=got["rows"], games=got["games"], folds=note["folds"],
                  bootstrap=BOOTSTRAP,
                  fold_labels={"what": "the frozen fit on the rows outside each written fold predicts that fold as the "
                                       "frozen out_of_fold did, and as oof.parquet says",
                               "max_abs_diff": label_diff, "tolerance": FOLD_LABEL_TOLERANCE,
                               "agrees": label_diff <= FOLD_LABEL_TOLERANCE})
    record["passed"] = bool(all(row["agrees"] for row in values) and all(row["agrees"] for row in exact)
                            and label_diff <= FOLD_LABEL_TOLERANCE)
    return record


def _absence(folder: Path, root: Path, guard: MemoryGuard, report: Path) -> tuple:
    import pandas as pd
    reported = parse_load_report(report.read_text(encoding="utf-8"))
    pred, note = absence_oof(pd.read_parquet(root / "analytics" / "load" / SEASON / "player_games.parquet"))
    guard.check("absence fits")
    path = assert_safe_output(folder / ABSENCE_FILE)
    absence_table(pred, fold_labels(pred["sportvu_game"], note["folds"])).to_parquet(path, index=False)
    record = check_absence(pd.read_parquet(path), pred, reported, note)
    record.update(source=f"$DATA_ROOT/analytics/load/{SEASON}/player_games.parquet", report=_shown(report),
                  columns=ABSENCE_COLUMNS,
                  fold_note="fold is the grouped fold that held the row's game out, over the games of these rows only; "
                            "use it as the out-of-fold flag, never as a join key across lanes")
    return record, [ABSENCE_FILE]


def rebuild_absence(artifacts=None, report=None) -> dict:
    """Rebuild <artifacts>/absence (oof.parquet and check.json) -> the check record."""
    report = Path(report) if report else REPO_ROOT / "reports" / "load.md"
    return _rebuild("absence", artifacts, lambda folder, root, guard: _absence(folder, root, guard, report))


def load_absence(artifacts=None):
    """The out-of-fold absence rows (ABSENCE_COLUMNS), only when their rebuild reproduced the frozen report."""
    import pandas as pd
    folder = artifacts_dir(artifacts) / "absence"
    require_passed(folder, "absence")
    return pd.read_parquet(folder / ABSENCE_FILE)


# ----------------------------------------------------------------------------- embeddings

def embeddings_command(folder: Path, root: Path) -> list:
    """The frozen trainer's command line. The output root and the report are always named and always inside
    `folder`: without --out the trainer writes no table, and without --report it overwrites reports/embeddings.md."""
    return [sys.executable, "-m", "analytics.embeddings.train", "--data", str(root), "--seed", str(SEED),
            "--out", str(assert_safe_output(folder)), "--report", str(assert_safe_output(folder / REPORT_NAME))]


def _rss_gb(pid: int) -> float:
    """Resident memory of a process right now in GB (from ps); 0 when it cannot be read."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip()) / 1e6
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


@contextlib.contextmanager
def _never_outlive(process):
    """Kill a child when this process is told to stop, so the child never runs on without the heavy-job lock."""
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, stop)
    except ValueError:              # not the main thread: the default handlers stay
        pass
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            process.kill()
            process.wait()


def run_child(command: list, guard: MemoryGuard, cwd, poll_s: float = 2.0, say=_say) -> tuple:
    """Run a command to its end -> (exit code, its output, its peak memory in GB). The child is killed and
    MemoryError raised when its memory in use goes above the guard."""
    process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    started, told, peak = time.time(), time.time(), 0.0
    with _never_outlive(process):
        while True:
            try:
                output, _ = process.communicate(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                used = _rss_gb(process.pid)
                peak = max(peak, used)
                if used > guard.limit_gb:
                    process.kill()
                    process.communicate()
                    raise MemoryError(f"the child process used {used:.1f} GB, above the {guard.limit_gb:.1f} GB guard") from None
                if time.time() - told >= 30:
                    told = time.time()
                    say(f"[frozen] child running for {told - started:.0f} s, {used:.1f} GB in use")
    return process.returncode, output, max(peak, peak_rss_gb(children=True))


def trained_games(text: str) -> list:
    """The training game ids an embeddings report lists in its Inputs section."""
    from analytics import results as R
    listed = R.Doc(text).need("Inputs", level=2).search(r"^Training games \((\d+)\): (.*)\.$", "the training games")
    games = [game.strip() for game in listed[2].split(",") if game.strip() and game.strip() != "none"]
    if len(games) != int(listed[1]):
        raise R.ParseError(f"{len(games)} training game ids listed, {listed[1]} counted")
    return games


def parse_embeddings_report(text: str) -> dict:
    """The player counts, the chosen method and its 5-NN position accuracy with its interval, the majority baseline
    and the table row counts of an embeddings report, as printed."""
    from analytics import results as R
    doc = R.Doc(text)
    counts = doc.need("Inputs", level=2).search(
        r"^Player-seasons: (\d+), of which (\d+) sufficient \(at least (\d+) possessions\) and (\d+) insufficient",
        "the player counts")
    chosen = doc.need("Decision", level=2).search(r"^Chosen: \*\*([^*]+)\*\*", "the chosen method").group(1)
    candidates = doc.need("Candidates", level=2)
    table = candidates.table("method", "5-NN position accuracy", "majority baseline")
    row = table.one("method", lambda cell: cell == chosen, "the row of the chosen method")
    labelled = candidates.search(r"^5-NN: leave-one-out over (\d+) labelled player-seasons", "the labelled count")
    rows = doc.need("Outputs", level=2).search(
        r"player_season_features \((\d+) rows\), player_embeddings \((\d+) rows\), embedding_neighbours \((\d+) rows\)",
        "the table row counts")
    return {"player_seasons": counts[1], "sufficient": counts[2], "possession_gate": counts[3], "insufficient": counts[4],
            "chosen": chosen, "accuracy": _cell(table.cell(row, "5-NN position accuracy")),
            "majority": _cell(table.cell(row, "majority baseline")), "labelled": labelled[1],
            "table_rows": dict(zip(EMBEDDING_TABLES, (rows[1], rows[2], rows[3])))}


def generated_sections(text: str) -> dict:
    """title -> body lines of every level 2 section of a lane report, without the output-location line and without
    trailing blank lines."""
    from analytics import results as R
    out = {}
    for section in R.Doc(text).sections:
        if section.level != 2:
            continue
        lines = [line.rstrip() for line in section.lines if not line.startswith(OUTPUT_LINE)]
        while lines and not lines[-1]:
            lines.pop()
        out[section.title] = lines
    return out


def check_embeddings(rebuilt: str, committed: str, rows_on_disk: dict = None) -> dict:
    """A rebuilt embeddings report against the committed one.

    The player counts, the 5-NN position accuracy and its interval must be printed identically (the required
    values), and so must the majority baseline, the labelled count and the table row counts. Every level 2 section
    of the rebuilt report must equal the same section of the committed report line for line; the committed report
    may hold one more section, the one added by hand after the stage 8 run. `rows_on_disk`, when given, are the
    row counts of the written tables and must be the reported ones."""
    from analytics import results as R
    record = {"compared": "the player counts and the 5-NN position accuracy with its interval in the rebuilt report, "
                          "with reports/embeddings.md; printed values must be identical, and every generated section "
                          "the same line for line", "unit": "numbers as reports/embeddings.md prints them",
              "n": 0, "max_abs_diff": None, "tolerance": 0.0, "passed": False}
    try:
        new, old = parse_embeddings_report(rebuilt), parse_embeddings_report(committed)
    except R.ParseError as error:
        return dict(record, problem=f"a report could not be parsed: {error}")
    values = [compare_same("player-seasons", old["player_seasons"], new["player_seasons"], True),
              compare_same("sufficient player-seasons", old["sufficient"], new["sufficient"], True),
              compare_same("insufficient player-seasons", old["insufficient"], new["insufficient"], True),
              compare_same("possession gate", old["possession_gate"], new["possession_gate"])]
    for part in ("point", "lo", "hi"):
        values.append(compare_same(f"5-NN position accuracy {part}", old["accuracy"][part], new["accuracy"][part], True))
    for part in ("point", "lo", "hi"):
        values.append(compare_same(f"majority baseline {part}", old["majority"][part], new["majority"][part]))
    values.append(compare_same("labelled player-seasons", old["labelled"], new["labelled"]))
    for name in EMBEDDING_TABLES:
        values.append(compare_same(f"{name} rows", old["table_rows"][name], new["table_rows"][name]))
    exact = [_exact("chosen method", old["chosen"], new["chosen"])]
    if rows_on_disk is not None:
        exact.append(_exact("rows of the written tables", {name: int(old["table_rows"][name]) for name in EMBEDDING_TABLES},
                            {name: int(rows_on_disk.get(name, -1)) for name in EMBEDDING_TABLES}))
    a, b = generated_sections(rebuilt), generated_sections(committed)
    unexpected = [title for title in b if title not in a and not title.startswith(HAND_ADDED_SECTION)]
    unexpected += [title for title in a if title not in b]
    differing = [{"section": title, "line": i, "committed": (x or "")[:160], "rebuilt": (y or "")[:160]}
                 for title in a if title in b
                 for i, (x, y) in enumerate(itertools.zip_longest(b[title], a[title])) if x != y]
    record.update(_numbers(values))
    record.update(values=values, exact=exact,
                  sections={"compared": [title for title in a if title in b], "only_in_committed": [t for t in b if t not in a],
                            "unexpected": unexpected, "differing_lines": len(differing), "first_differences": differing[:5]})
    record["passed"] = bool(all(row["agrees"] for row in values) and all(row["agrees"] for row in exact)
                            and not unexpected and not differing)
    return record


def _outside_state(root: Path) -> tuple:
    """What the frozen trainer must leave alone: the committed report and the lane's folder under DATA_ROOT."""
    report = REPO_ROOT / "reports" / "embeddings.md"
    digest = hashlib.sha256(report.read_bytes()).hexdigest() if report.is_file() else None
    return digest, (root / "analytics" / "embeddings").exists()


def _embeddings(folder: Path, root: Path, guard: MemoryGuard, committed: Path) -> tuple:
    import pyarrow.parquet as pq
    committed_text = committed.read_text(encoding="utf-8")
    parse_embeddings_report(committed_text)         # stop before the long run when the frozen record cannot be read
    before = _outside_state(root)
    code, output, child_peak = run_child(embeddings_command(folder, root), guard, REPO_ROOT)
    if _outside_state(root) != before:
        raise RuntimeError("the frozen trainer changed reports/embeddings.md or wrote under DATA_ROOT")
    report = assert_safe_output(folder / REPORT_NAME)
    leftovers = 0
    if report.is_file():
        text, left = public_text(report.read_text(encoding="utf-8"), folder.parent, root)
        report.write_text(text, encoding="utf-8")
        leftovers += left
    log, left = public_text(output or "", folder.parent, root)
    assert_safe_output(folder / LOG_NAME).write_text(log if not left else "log withheld: it carried a local path\n",
                                                     encoding="utf-8")
    if code != 0 or not report.is_file():
        raise RuntimeError(f"the frozen embeddings trainer ended with exit code {code}; see {LOG_NAME}")
    if leftovers:
        raise RuntimeError(f"{REPORT_NAME} still carries a local path after redaction")
    rows = {}
    for name in EMBEDDING_TABLES:
        source = folder / "analytics" / "embeddings" / SEASON / f"{name}.parquet"
        if not source.is_file():
            raise RuntimeError(f"the frozen embeddings trainer wrote no {name} table for {SEASON}")
        target = assert_safe_output(folder / f"{name}.parquet")
        shutil.copyfile(source, target)
        rows[name] = pq.read_metadata(target).num_rows
    rebuilt = report.read_text(encoding="utf-8")
    fitted_on = training_games_only(trained_games(rebuilt), f"{WHO} embeddings")     # the ids the trainer says it fitted on
    record = check_embeddings(rebuilt, committed_text, rows)
    record.update(training_games=fitted_on, report=_shown(committed), child_peak_memory_gb=round(child_peak, 2),
                  command=f"python3 -m analytics.embeddings.train --data $DATA_ROOT --seed {SEED} --out "
                          f"$V3_ARTIFACTS/embeddings --report $V3_ARTIFACTS/embeddings/{REPORT_NAME}",
                  tables={name: {"file": f"{name}.parquet", "rows": rows[name]} for name in EMBEDDING_TABLES},
                  trainer_layout=f"analytics/embeddings/{SEASON}/ holds the tables as the trainer wrote them; the "
                                 "copies at the top of the folder are the ones this check vouches for")
    return record, [REPORT_NAME, LOG_NAME] + [f"{name}.parquet" for name in EMBEDDING_TABLES]


def rebuild_embeddings(artifacts=None, report=None) -> dict:
    """Rebuild <artifacts>/embeddings (the three tables, the trainer's report and log, check.json) -> the check record."""
    report = Path(report) if report else REPO_ROOT / "reports" / "embeddings.md"
    return _rebuild("embeddings", artifacts, lambda folder, root, guard: _embeddings(folder, root, guard, report))


def load_embeddings(artifacts=None) -> dict:
    """{table name: frame} for the three embeddings tables, only when their rebuild reproduced the frozen report."""
    import pandas as pd
    folder = artifacts_dir(artifacts) / "embeddings"
    require_passed(folder, "embeddings")
    return {name: pd.read_parquet(folder / f"{name}.parquet") for name in EMBEDDING_TABLES}


# ----------------------------------------------------------------------------- xfg

def blank_heldout_labels(shots):
    """The shots table with the label of every held-out attempt overwritten with False. The rebuild needs the
    held-out features for its check and must never read a held-out outcome; training rows are untouched."""
    import numpy as np
    from analytics.eval import is_heldout_game
    games = shots["sportvu_game"].astype(str)
    by_id = games.map({game: is_heldout_game(game) for game in games.unique()}).to_numpy(dtype=bool)
    held = shots["heldout"].fillna(False).astype(bool).to_numpy() | by_id
    out = shots.copy()
    out["made"] = np.where(held, False, out["made"].astype(bool).to_numpy())
    return out


def _guarded(loader, guard: MemoryGuard, say=_say):
    """A frames loader that checks the memory guard before every game and reports its progress."""
    seen = [0]

    def load(game):
        used = guard.check(f"before the frames of game {seen[0] + 1}")
        seen[0] += 1
        if seen[0] % 100 == 0:
            say(f"[frozen] xfg features: frames of {seen[0]} games read, {used:.1f} GB in use")
        return loader(game)

    return load


def xfg_tables(root: Path, guard: MemoryGuard) -> dict:
    """The lane's tables as its trainer loads and cleans them, with two changes that touch no training row: held-out
    labels are blanked, and the frames loader checks the memory guard before each game."""
    from analytics.xfg import validate as V
    tables, _summaries = V.clean(V.load_data_root(root, SEASON))
    tables["shots"] = blank_heldout_labels(tables["shots"])
    tables["frames"] = _guarded(tables["frames"], guard)
    return tables


def xfg_parts(predict) -> dict:
    """The fitted parts of the predictor analytics.xfg.train.fit_candidate returns for the selected candidate: the
    booster, the isotonic map, the shooter prior table and the model columns (the defaults of the frozen closure)."""
    names = list(inspect.signature(predict).parameters)
    defaults = predict.__defaults__ or ()
    if names != ["frame", "h", "iso", "pt", "c"] or len(defaults) != 4:
        raise RuntimeError(f"{WHO} xfg: the frozen predictor is not the {XFG_CANDIDATE} closure")
    booster, iso, prior, columns = defaults
    return {"candidate": XFG_CANDIDATE, "seed": SEED, "hgb": booster, "iso": iso, "prior": prior, "columns": list(columns)}


def fit_xfg(train, seed: int = SEED) -> tuple:
    """The frozen fit of the selected candidate on training rows -> (the frozen predictor, its fitted parts).
    SystemExit when a game id is held out; the frozen fit runs the same guard again on what reaches it."""
    from analytics.xfg import train as T
    training_games_only(train["sportvu_game"].astype(str).unique(), f"{WHO} xfg")
    predict = T.fit_candidate(XFG_CANDIDATE, train, seed=seed)
    return predict, xfg_parts(predict)


class FrozenXfg:
    """The frozen xFG model, hgb_contest_prior fitted once on every training row with seed 0, as rebuilt.

    predict(frame) takes a feature frame as analytics.xfg.features.build returns it: the 14 model columns of
    analytics.xfg.features.FEATURES, plus player_id (null means the zone prior) and zone. It returns one
    probability per row, as the frozen predictor does: the shooter prior from all training rows, the booster, then
    the isotonic map. Nothing is fitted."""

    def __init__(self, parts: dict, check: dict = None):
        self.hgb, self.iso, self.prior = parts["hgb"], parts["iso"], parts["prior"]
        self.columns = list(parts["columns"])
        self.candidate, self.seed, self.check = parts.get("candidate"), parts.get("seed"), check

    def needs(self) -> list:
        """The columns predict reads from a frame."""
        from analytics.xfg.features import PRIOR_FEATURE
        return [column for column in self.columns if column != PRIOR_FEATURE] + ["player_id", "zone"]

    def predict(self, frame):
        import numpy as np
        from analytics.eval import fit_threads
        from analytics.xfg import train as T
        missing = [column for column in self.needs() if column not in frame.columns]
        if missing:
            raise KeyError(f"the feature frame lacks {missing}")
        rows = frame.copy()
        rows[T.PRIOR_FEATURE] = T.apply_prior(self.prior, rows)
        with fit_threads(1):
            p = self.hgb.predict_proba(T._design(rows, self.columns))[:, 1]
        return np.asarray(self.iso.predict(p) if self.iso is not None else p, dtype=float)


def _save_parts(folder: Path, parts: dict) -> Path:
    path, part = assert_safe_output(folder / MODEL_NAME), assert_safe_output(folder / (MODEL_NAME + ".part"))
    with open(part, "wb") as handle:
        pickle.dump(parts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(part, path)
    return path


def _read_parts(folder: Path) -> dict:
    with open(Path(folder) / MODEL_NAME, "rb") as handle:
        return pickle.load(handle)


def key_mismatches(a, b) -> dict:
    """How many rows of two equally long frames differ in sportvu_game, in event and in player_id (two nulls agree)."""
    import numpy as np
    import pandas as pd

    def players(frame):
        return pd.to_numeric(frame["player_id"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    pa, pb = players(a), players(b)
    return {"sportvu_game": int((a["sportvu_game"].astype(str).to_numpy() != b["sportvu_game"].astype(str).to_numpy()).sum()),
            "event": int((a["event"].to_numpy(dtype="int64") != b["event"].to_numpy(dtype="int64")).sum()),
            "player_id": int((~((pa == pb) | (np.isnan(pa) & np.isnan(pb)))).sum())}


def check_predictions(stored, rebuilt, p, tolerance: float = XFG_TOLERANCE) -> dict:
    """Rebuilt predictions `p` for the rows of `rebuilt` against the stored p_selected of `stored`, row for row.

    The two frames must hold the same attempts in the same order (sportvu_game, event, player_id); when they do
    not, nothing is compared and the check fails. Every stored row is compared (n must reach n_expected, the
    stored row count). A prediction that is not finite fails. Returns the fields write_check needs and the
    pattern of the difference: rows over the tolerance, quantiles, the split by has_contest and by whether the
    attempt matched a tracked event, and the worst rows when the check fails."""
    import numpy as np
    out = {"n": 0, "n_expected": int(len(stored)), "max_abs_diff": None, "tolerance": tolerance, "passed": False,
           "alignment": {"stored_rows": int(len(stored)), "rebuilt_rows": int(len(rebuilt)), "keys_equal": False}}
    if not len(stored) or len(stored) != len(rebuilt) or len(p) != len(rebuilt):
        return out
    mismatches = key_mismatches(stored, rebuilt)
    out["alignment"].update(keys_equal=not any(mismatches.values()), mismatches=mismatches)
    if any(mismatches.values()):
        return out
    was = stored["p_selected"].to_numpy(dtype=float)
    now = np.asarray(p, dtype=float)
    diff = np.abs(now - was)
    finite = np.isfinite(diff)
    over = ~(diff <= tolerance)                     # a value that is not finite counts as over
    out.update(n=int(len(diff)), not_finite=int((~finite).sum()), rows_over_tolerance=int(over.sum()),
               max_abs_diff=float(diff.max()) if finite.all() else None,
               passed=bool(finite.all() and not over.any()))
    if finite.any():
        out["abs_diff_quantiles"] = {name: float(np.quantile(diff[finite], q))
                                     for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))}
    groups = {"tracked_event": rebuilt["event"].to_numpy(dtype="int64") >= 0}
    if "has_contest" in rebuilt.columns:
        groups["has_contest"] = rebuilt["has_contest"].to_numpy(dtype="int64") == 1
    out["by_group"] = {}
    for name, mask in groups.items():
        for label, part in (("yes", mask), ("no", ~mask)):
            if part.any():
                out["by_group"][f"{name} {label}"] = {
                    "n": int(part.sum()), "rows_over_tolerance": int(over[part].sum()),
                    "max_abs_diff": float(np.max(diff[part & finite])) if (part & finite).any() else None}
    if not out["passed"]:
        worst = np.argsort(-np.where(finite, diff, np.inf), kind="stable")[:10]
        games = rebuilt["sportvu_game"].astype(str).to_numpy()
        events = rebuilt["event"].to_numpy(dtype="int64")
        out["worst_rows"] = [{"sportvu_game": games[i], "event": int(events[i]), "stored": float(was[i]),
                              "rebuilt": float(now[i]) if np.isfinite(now[i]) else None} for i in worst]
        out["games_over_tolerance"] = int(len(set(games[over])))
    return out


def _xfg(folder: Path, root: Path, guard: MemoryGuard) -> tuple:
    import numpy as np
    import pandas as pd
    from analytics.xfg import train as T
    stored = pd.read_parquet(root / "analytics" / "xfg" / SEASON / "xfg_predictions.parquet", columns=STORED_COLUMNS)
    held = stored["heldout"].astype(bool).to_numpy()
    if ((stored["fold"].to_numpy() == T.FOLD_HELDOUT) != held).any():
        raise RuntimeError("fold and heldout disagree in the stored predictions")
    train, heldout = T.assemble(xfg_tables(root, guard))     # the frozen feature build; held-out rows are set aside
    if heldout["made"].astype(bool).any():
        raise RuntimeError("a held-out label reached the feature frame")
    heldout = heldout.drop(columns=["made"])
    guard.check("xfg features built")
    _say(f"[frozen] xfg: {len(train)} training rows, {len(heldout)} held-out feature rows; fitting")
    predict, parts = fit_xfg(train)
    _save_parts(folder, parts)
    model = FrozenXfg(_read_parts(folder))               # the check runs on what a later process will load
    p = model.predict(heldout) if len(heldout) else np.empty(0)
    record = {"compared": f"predictions of the rebuilt {XFG_CANDIDATE} (the saved parts, read back from {MODEL_NAME}) for "
                          "the held-out feature rows, with the stored p_selected of the held-out rows of "
                          f"$DATA_ROOT/analytics/xfg/{SEASON}/xfg_predictions.parquet; no outcome column is read",
              "unit": "probability"}
    record.update(check_predictions(stored[held].reset_index(drop=True), heldout, p))
    stored_train = stored[~held].reset_index(drop=True)
    training = {"rows": int(len(train)), "games": int(train["sportvu_game"].nunique()), "stored_rows": int(len(stored_train)),
                "with_contest": int(train["has_contest"].sum())}
    if len(stored_train) == len(train):
        training["key_mismatches_against_stored"] = key_mismatches(stored_train, train)
    gaps = [model.predict(train) - np.asarray(predict(train), dtype=float)]
    if len(heldout):
        gaps.append(p - np.asarray(predict(heldout), dtype=float))
    closure = float(np.max(np.abs(np.concatenate(gaps))))
    record.update(training=training, candidate=XFG_CANDIDATE, model={"file": MODEL_NAME, "columns": model.needs()},
                  heldout={"rows": int(len(heldout)), "games": int(heldout["sportvu_game"].nunique()),
                           "with_contest": int(heldout["has_contest"].sum())},
                  saved_against_frozen_predictor={"what": "the saved parts against the frozen closure on every "
                                                          "training and held-out feature row", "max_abs_diff": closure})
    record["passed"] = bool(record["passed"] and closure == 0.0)
    return record, [MODEL_NAME]


def rebuild_xfg(artifacts=None) -> dict:
    """Rebuild <artifacts>/xfg (model.pkl and check.json) -> the check record."""
    return _rebuild("xfg", artifacts, _xfg)


def load_xfg(artifacts=None) -> FrozenXfg:
    """The frozen xFG model, only when its rebuild reproduced the stored held-out predictions, the model file is the
    one that check hashed, and scikit-learn is the version the check ran under. Nothing is fitted here."""
    folder = artifacts_dir(artifacts) / "xfg"
    check = require_passed(folder, "xfg")
    if MODEL_NAME not in (check.get("files") or {}):
        raise RuntimeError(f"xfg: the check does not vouch for {MODEL_NAME}")
    built, now = (check.get("versions") or {}).get("scikit_learn"), versions()["scikit_learn"]
    if built != now:
        raise RuntimeError(f"xfg: rebuilt under scikit-learn {built}, this is {now}; run the rebuild again")
    return FrozenXfg(_read_parts(folder), check)


# ----------------------------------------------------------------------------- listing and command line

def status(artifacts=None) -> list:
    """One row per artifact: whether its folder holds a check, whether it passed and can be used (require_passed:
    the check passed and its files did not change since), and what it compared. The three artifacts of this
    module always appear; any other folder that holds a check.json follows them."""
    root = artifacts_dir(artifacts)
    others = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in NAMES and (p / CHECK_NAME).exists()) \
        if root.is_dir() else []
    rows = []
    for name in list(NAMES) + others:
        record, reason = read_check(root / name), "no rebuilt artifact"
        passed = False
        if record is not None:
            try:
                require_passed(root / name, name)
                passed, reason = True, ""
            except RuntimeError as error:
                reason = str(error)
        record = record or {}
        rows.append({"name": name, "exists": bool(record), "passed": passed, "reason": reason,
                     "status": record.get("status"), "n": record.get("n"), "max_abs_diff": record.get("max_abs_diff"),
                     "tolerance": record.get("tolerance"), "written": record.get("written"),
                     "compared": record.get("compared")})
    return rows


REBUILDS = {"absence": rebuild_absence, "embeddings": rebuild_embeddings, "xfg": rebuild_xfg}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tools.frozen_models", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("absence", "the out-of-fold absence probabilities of the load lane"),
                       ("embeddings", "the three tables of the embeddings lane"),
                       ("xfg", "the frozen xFG model object"),
                       ("status", "list the artifacts and whether each one can be used")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--artifacts",
                             help="the artifacts folder (default: V3_ARTIFACTS, else artifacts/ in the repository)")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(args.artifacts), indent=2))
        return 0
    record = REBUILDS[args.command](args.artifacts)
    print(json.dumps(record, indent=2))
    return 0 if record.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

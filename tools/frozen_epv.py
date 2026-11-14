"""tools/frozen_epv.py: the frozen EPV model as an object a later process can load (PLAN.md D45, D49).

freeze-v1 fitted the selected EPV model and scored the held-out games in one process, and saved nothing. This module
reruns that fit unchanged on the training games, saves the parts, and keeps them only when they reproduce the stored
held-out predictions.

    python3 -m tools.frozen_epv project    measure the same code on two subsets of training games and project the
                                           memory of the full fit (rebuild refuses to start without it)
    python3 -m tools.frozen_epv rebuild    fit, save, check, write check.json
    python3 -m tools.frozen_epv check      rerun only the comparison on the saved parts

The model (analytics/epv, frozen at freeze-v1): the torch GRU with 16 hidden units over the last 5 states of a
possession, with an isotonic map from its expected points to P(points > 0). The reports call that slot `mlp`; the
frozen run used `--deep torch --seed 0`.

Fit (`_fit_calibrated`): the calls the frozen trainer makes when it refits the selected candidate on every training
row, in its order and on the training games only: validate.load_root, train.split_tables, features.geometry,
features.Priors.from_training, features.build_features, features.lagged, train.Calibrated(...).fit. The trainer's
own function for that refit also scores the held-out games, so it is never called here; only its fitting calls
are. Nothing changes a setting; held-out rows are set aside by the frozen split and never reach a fit.

Saved under <artifacts>/epv/: gru_state.pt (the GRU weights), parts.pkl (the imputer, the two scalers, the isotonic
map, and the priors of the training games: shooter zone counts, lineup turnover counts, the turnover rate and the
fitted location model the priors shrink towards), meta.json (what was fitted, on what, with which library versions).

Check (`check`): the saved parts are loaded from disk and predict (1) the stored 1 Hz states of the held-out games
in one call, as the frozen run did, and (2) the same games again from their tracking frames, through
`predict_states`, the path a later process uses. Both must equal the stored epv_points within 1e-5 points on every
stored row. Predictions are compared with predictions: no outcome column is read (the held-out tables are read
with a column list that leaves the outcome columns out) and nothing is scored.

Use: `load(artifacts).predict_states(frames, possessions, game)`. `load` returns a predictor only when the check
passed, the files are the ones it covered, and the libraries are the ones it ran under.

Memory: the full fit holds several copies of a 1.7 GB matrix. tools.frozen_common.MemoryGuard is applied between
stages, and while a stage runs a watch thread applies the same rule to readings of this process, because the frozen
fitting loops have no place to call a guard. The memory helpers below are shared with tools.frozen_ghosting.
"""

import argparse
import contextlib
import ctypes
import ctypes.util
import gc
import json
import os
import pickle
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import deep_torch as DT
from analytics import schema
from analytics.epv import deep as D
from analytics.epv import features as F
from analytics.epv import states as S
from analytics.epv import train as T
from analytics.epv import validate as V
from analytics.eval import fit_threads
from export.heavy import heavy_job
from tools import frozen_common as C

NAME = "epv"
WHO = "tools.frozen_epv"
SEED = 0                     # the frozen run: --seed 0
HGB_ITER = 100               # the trainer's default; part of the frozen call, unused by the sequence slot
DEEP = "torch"               # the frozen run: --deep torch
TOLERANCE = C.TOLERANCE_CAP[NAME]      # 1e-5 points (PLAN.md D49)
UNIT = "points"
LIMIT_GB = 11.5              # the default of tools.frozen_common.MemoryGuard
SURGE_GB = 4.0               # the watch also ends a run whose footprint passes the guard by this much
STATE_FILE, PARTS_FILE, META_FILE = "gru_state.pt", "parts.pkl", "meta.json"
FILES = (STATE_FILE, PARTS_FILE, META_FILE)
# The frozen record (reports/epv.md): the rows the final model was fitted on and the rows it scored.
FROZEN_RECORD = {"train_games": 604, "train_states": 1739119, "heldout_games": 17, "heldout_states": 44703}
PROJECTION_GAMES = (60, 150)
KEYS = ["sportvu_game", "possession", "frame"]
# What the frozen states builder reads from the possessions table, apart from the two outcome columns.
POSSESSION_COLUMNS = ("possession", "offense_team_id", "start_frame", "end_frame", "start_clock_s")
FRAME_COLUMNS = ("frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "team_id",
                 "player_id", "x", "y")
MEMORY_EXIT = 70             # exit status of a run the memory watch ended
ALLOWED_FROZEN_DIFF = ("analytics/results.py",)      # CLAUDE.md rule 3


# ------------------------------------------------------------------------------------------------ memory (shared)

_PROC = {"lib": None, "tried": False}


def _rusage_slots():
    """rusage_info_v4 of this process as 64-bit slots (macOS), else None. Slot 8 is resident memory, 9 the
    physical footprint, 35 the largest footprint since the last interval reset."""
    if sys.platform != "darwin":
        return None
    try:
        if not _PROC["tried"]:
            _PROC["tried"] = True
            _PROC["lib"] = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
        if _PROC["lib"] is None:
            return None
        slots = (ctypes.c_uint64 * 64)()
        if _PROC["lib"].proc_pid_rusage(os.getpid(), 4, ctypes.byref(slots)) != 0:
            return None
        return slots
    except Exception:
        return None


def memory_now() -> dict:
    """Memory of this process in GB, read without starting another process. `rss` is resident memory, the number
    tools.frozen_common.MemoryGuard judges. `footprint` is the macOS physical footprint, which also counts the
    pages the system has compressed or swapped out: on a machine under memory pressure resident memory reads low
    and the footprint is the size of the job (None where the system does not report it)."""
    slots = _rusage_slots()
    if slots is None:
        return {"rss": C.current_rss_gb(), "footprint": None}
    return {"rss": slots[8] / 1e9, "footprint": slots[9] / 1e9}


def reset_footprint_interval() -> None:
    """Start a new interval for the system's own record of the largest footprint (macOS)."""
    try:
        if _rusage_slots() is not None:
            _PROC["lib"].proc_reset_footprint_interval(os.getpid())
    except Exception:
        pass


def footprint_interval_peak():
    """The largest footprint since the last reset, in GB, as the system recorded it (None when not reported)."""
    slots = _rusage_slots()
    return None if slots is None else slots[35] / 1e9


def system_memory() -> dict:
    """What the machine says about itself: the free memory level in percent and the swap in use (None elsewhere)."""
    out = {"free_percent": None, "swap_used_gb": None}
    if sys.platform != "darwin":
        return out
    try:
        level = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=10)
        out["free_percent"] = int(level.stdout.strip())
        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        out["swap_used_gb"] = round(float(swap.split("used =")[1].split()[0].rstrip("M")) / 1024, 2)
    except Exception:
        pass
    return out


def plain(value):
    """numpy scalars and containers as plain JSON values; NaN and infinity become None."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def say(who: str, message: str) -> None:
    """One progress line on stderr with the time and the memory of this process."""
    now = memory_now()
    footprint = "" if now["footprint"] is None else f", footprint {now['footprint']:.1f} GB"
    print(f"[{who} {time.strftime('%H:%M:%S')} rss {now['rss']:.1f} GB{footprint}] {message}", file=sys.stderr, flush=True)


def held_level(times, values, window_s: float):
    """The highest level that was held for at least `window_s` seconds: over every window of that length, the
    smallest reading in it, and of those the largest. A record shorter than the window gives its smallest
    reading. macOS takes freed arrays back a few seconds late, so a peak of the footprint can be memory that is
    already free; the held level leaves such peaks out. It is reported beside the peak and never replaces it."""
    t, v = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    if t[-1] - t[0] < window_s:
        return float(v.min())
    best = -np.inf
    for i in range(len(v)):
        if t[-1] - t[i] < window_s:
            break
        j = int(np.searchsorted(t, t[i] + window_s, side="right"))
        best = max(best, float(v[i:j].min()))
    return best


def stop_process(watch) -> None:
    """The default action of a tripped watch: say why and leave at once with status MEMORY_EXIT. The frozen code
    has no place where another thread can stop it cleanly, and an abrupt exit is better than a machine that
    swaps to a halt. The heavy-job lock is a file lock, so it is released with the process."""
    print(f"[memory watch] {watch.reason}; the process exits now with status {MEMORY_EXIT}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(MEMORY_EXIT)


class MemoryWatch:
    """tools.frozen_common.MemoryGuard inside a stage that has no place to call it. A background thread reads the
    memory of this process every `every_s` seconds and applies the guard's rule: resident memory now, or its peak
    so far, above `limit_gb` ends the run. It also ends a run whose footprint is above `limit_gb + surge_gb`,
    which resident memory cannot show on a machine that is compressing. The thread only reads counters, so the
    frozen code path is the same with and without it.

    Ending the run means `on_trip(watch)`, by default stop_process. If `on_trip` returns, leaving the block raises
    MemoryError."""

    def __init__(self, limit_gb: float = LIMIT_GB, where: str = "", every_s: float = 1.0, surge_gb: float = SURGE_GB,
                 on_trip=None):
        self.limit_gb, self.where, self.every_s, self.surge_gb = float(limit_gb), where, float(every_s), float(surge_gb)
        self.on_trip = on_trip or stop_process
        self.samples = []            # (seconds since the start, rss, footprint or None)
        self.tripped = None
        self.reason = ""

    def __enter__(self):
        self._started = time.monotonic()
        self._peak_before = C.rss_gb()
        self._stop = threading.Event()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="memory-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self._sample()
        if self.tripped is not None:
            raise MemoryError(self.reason) from None
        return False

    def _sample(self) -> dict:
        now = memory_now()
        self.samples.append((time.monotonic() - self._started, now["rss"], now["footprint"]))
        return now

    def _over(self, now: dict) -> str:
        """Why this reading ends the run, or an empty string."""
        rss = max(now["rss"], C.rss_gb())                    # the guard's rule: in use now, or the peak so far
        if rss > self.limit_gb:
            return f"resident memory reached {rss:.1f} GB, above the {self.limit_gb:.1f} GB guard ({self.where})"
        footprint = now["footprint"] or 0.0
        if footprint > self.limit_gb + self.surge_gb:
            return (f"the memory footprint reached {footprint:.1f} GB, more than {self.surge_gb:.1f} GB above the "
                    f"{self.limit_gb:.1f} GB guard ({self.where})")
        return ""

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            now = self._sample()
            reason = self._over(now)
            if reason:
                self.tripped, self.reason = max(now["rss"], now["footprint"] or 0.0), reason
                self.on_trip(self)
                return

    def summary(self, window_s: float = 5.0) -> dict:
        """Peaks of the readings so far in GB. The resident peak also takes the process's own record of its
        largest resident size when that rose during the block. Footprint values are None where not reported."""
        t = [s[0] for s in self.samples]
        rss = max(s[1] for s in self.samples)
        peak_now = C.rss_gb()
        if peak_now > self._peak_before:
            rss = max(rss, peak_now)
        footprint = [s[2] for s in self.samples if s[2] is not None]
        has = bool(footprint) and len(footprint) == len(self.samples)
        return {"samples": len(self.samples), "peak_rss_gb": round(rss, 2),
                "peak_footprint_gb": round(max(footprint), 2) if has else None,
                "held_footprint_gb": round(held_level(t, footprint, window_s), 2) if has else None}


@contextlib.contextmanager
def stage(name: str, stages: dict, limit_gb: float = LIMIT_GB, every_s: float = 1.0, on_trip=None, who: str = WHO):
    """One stage of a rebuild under the memory watch. Stores its seconds and memory peaks in stages[name], then
    applies tools.frozen_common.MemoryGuard before the next stage."""
    say(who, f"{name}: start")
    reset_footprint_interval()
    started = time.time()
    watch = MemoryWatch(limit_gb, where=name, every_s=every_s, on_trip=on_trip)
    try:
        with watch:
            yield watch
    finally:
        record = watch.summary()
        record["seconds"] = round(time.time() - started, 1)
        system_peak = footprint_interval_peak()
        if system_peak is not None and record["peak_footprint_gb"] is not None:
            record["peak_footprint_gb"] = round(max(record["peak_footprint_gb"], system_peak), 2)
        stages[name] = record
        say(who, f"{name}: {record['seconds']} s, peak rss {record['peak_rss_gb']} GB, "
                 f"peak footprint {record['peak_footprint_gb']} GB")
    gc.collect()
    C.MemoryGuard(limit_gb).check("after " + name)


# ------------------------------------------------------------------------------------- comparison (shared)

def compare(predicted: pd.DataFrame, stored: pd.DataFrame, keys: list, values: list, tolerance: float,
            worst: int = 5) -> dict:
    """Predictions against stored predictions, joined on `keys`. `n` is the number of stored rows that were
    compared; a pass needs every stored row predicted, no extra row, no duplicate key, every difference finite
    and none above `tolerance`. Nothing here reads an outcome: the two tables hold keys and predictions only."""
    a = predicted[keys + values].reset_index(drop=True).copy()
    b = stored[keys + values].reset_index(drop=True).copy()
    for key in keys:
        numeric = pd.api.types.is_numeric_dtype(a[key]) and pd.api.types.is_numeric_dtype(b[key])
        a[key] = a[key].astype("int64") if numeric else a[key].astype(str)
        b[key] = b[key].astype("int64") if numeric else b[key].astype(str)
    duplicates = int(a.duplicated(keys).sum() + b.duplicated(keys).sum())
    merged = b.merge(a, on=keys, how="outer", suffixes=("_stored", "_new"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    new = both[[v + "_new" for v in values]].to_numpy(dtype=float)
    old = both[[v + "_stored" for v in values]].to_numpy(dtype=float)
    diff = np.abs(new - old)
    finite = np.isfinite(diff)
    good = diff[finite]
    out = {"n": int(len(both)), "values": int(diff.size), "rows_stored": int(len(b)), "rows_predicted": int(len(a)),
           "rows_missing": int((merged["_merge"] == "left_only").sum()),
           "rows_extra": int((merged["_merge"] == "right_only").sum()),
           "duplicate_keys": duplicates, "not_finite": int((~finite).sum()),
           "max_abs_diff": float(good.max()) if good.size else None,
           "mean_abs_diff": float(good.mean()) if good.size else None,
           "values_exact": int((good == 0).sum()), "tolerance": tolerance}
    out["passed"] = bool(out["n"] > 0 and not duplicates and not out["rows_missing"] and not out["rows_extra"]
                         and not out["not_finite"] and out["max_abs_diff"] <= tolerance)
    if good.size and out["max_abs_diff"] > 0 and worst:
        row_max = np.where(finite, diff, -1.0).max(axis=1)
        top = np.argsort(-row_max)[:worst]
        out["worst"] = [{**{k: both.iloc[i][k] for k in keys}, "abs_diff": float(row_max[i])} for i in top]
    return out


def combine(parts: dict) -> dict:
    """Per-game results of `compare` as one result, with the per-game numbers kept under `per_game`."""
    out = {k: int(sum(p[k] for p in parts.values()))
           for k in ("n", "values", "rows_stored", "rows_predicted", "rows_missing", "rows_extra", "duplicate_keys",
                     "not_finite", "values_exact")}
    largest = [p["max_abs_diff"] for p in parts.values() if p["max_abs_diff"] is not None]
    out["max_abs_diff"] = max(largest) if largest else None
    weights = [(p["mean_abs_diff"], p["values"] - p["not_finite"]) for p in parts.values() if p["mean_abs_diff"] is not None]
    total = sum(w for _, w in weights)
    out["mean_abs_diff"] = float(sum(m * w for m, w in weights) / total) if total else None
    out["games"] = len(parts)
    out["passed"] = bool(parts) and all(p["passed"] for p in parts.values())
    out["per_game"] = {game: {"n": p["n"], "max_abs_diff": p["max_abs_diff"], "passed": p["passed"]}
                       for game, p in parts.items()}
    worst = [dict(w, game=game) for game, p in parts.items() for w in p.get("worst", [])]
    if worst:
        out["worst"] = sorted(worst, key=lambda w: -w["abs_diff"])[:5]
    return out


# ------------------------------------------------------------------------------------- projection (shared)

def _line(rows, values, rows_full: float) -> dict:
    """A value at `rows_full` from measured (rows, value) points: a straight line through them with two or more
    distinct sizes and a positive slope, else in proportion to the rows of the largest run. `proportional` is
    always given: it scales the constant part too, so it is the upper figure."""
    x, y = np.asarray(rows, dtype=float), np.asarray(values, dtype=float)
    proportional = float(y[-1] * rows_full / x[-1])
    if len(np.unique(x)) >= 2:
        slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
        if slope > 0:
            return {"method": "line", "per_million_rows": float(slope * 1e6), "intercept": float(intercept),
                    "projected": float(intercept + slope * rows_full), "proportional": proportional}
    return {"method": "proportional", "projected": proportional, "proportional": proportional}


def extrapolate(runs: list, rows_full: int, limit_gb: float, fixed=(), scaled=("fit",), surge_gb: float = SURGE_GB) -> dict:
    """The memory and the time of the full fit, projected from measured subset runs.

    Each run holds `rows` and `stages` (the records `stage` writes). Stages named in `scaled` grow with the rows
    and are projected along a line through the runs; stages named in `fixed` cost the same at every size and are
    taken as measured.

    `projected_demand_gb` is what the gate compares with the guard: the footprint the job holds (its peaks that
    last under five seconds left out, see held_level), because that is the memory the job needs whatever the
    machine does with it. Resident memory is not projected into the gate: a small run on an idle machine keeps
    everything resident, the file cache of a memory map included, while the full run on a busy machine has its
    cold pages compressed, so a line through small runs says little about the resident size of the full one. It
    is reported as `projected_rss_gb`, the figure of an idle machine. Where the system reports no footprint, the
    resident peak is the demand. `ok` also needs the projected footprint peak within `limit_gb + surge_gb`, the
    level at which the watch ends a run."""
    runs = sorted(runs, key=lambda r: r["rows"])

    def level(run, names, key):
        found = [run["stages"][s][key] for s in names if s in run["stages"] and run["stages"][s].get(key) is not None]
        return max(found) if found else 0.0

    rows = [r["rows"] for r in runs]
    keys = ("peak_rss_gb", "peak_footprint_gb", "held_footprint_gb")
    lines = {key: _line(rows, [level(r, scaled, key) for r in runs], rows_full) for key in keys}
    seconds = _line(rows, [level(r, scaled, "seconds") for r in runs], rows_full)
    at_any_size = {key: max([level(r, fixed, key) for r in runs] or [0.0]) for key in keys}
    fixed_s = max([sum(r["stages"][s]["seconds"] for s in fixed if s in r["stages"]) for r in runs] or [0.0])
    rss = max(lines["peak_rss_gb"]["projected"], at_any_size["peak_rss_gb"])
    footprint = max(lines["peak_footprint_gb"]["projected"], at_any_size["peak_footprint_gb"])
    held = max(lines["held_footprint_gb"]["projected"], at_any_size["held_footprint_gb"])
    reported = all(s.get("held_footprint_gb") is not None for r in runs for s in r["stages"].values())
    demand = held if reported else rss
    return {"rows_full": int(rows_full), "limit_gb": float(limit_gb), "surge_gb": float(surge_gb),
            "runs": [{"games": r.get("games"), "rows": r["rows"], "stages": r["stages"]} for r in runs],
            "lines": lines, "fixed_stages": at_any_size,
            "projected_demand_gb": round(demand, 2),
            "demand_is": "held footprint" if reported else "resident peak (no footprint reported)",
            "projected_held_footprint_gb": round(held, 2), "projected_footprint_gb": round(footprint, 2),
            "projected_rss_gb": round(rss, 2),
            "fit_seconds": seconds, "projected_seconds": round(fixed_s + seconds["projected"]),
            "ok": bool(demand <= limit_gb and footprint <= limit_gb + surge_gb)}


def projection_path(folder) -> Path:
    """Where the projection of an artifact folder lives: in a subfolder, so that writing it never makes a file of
    the folder newer than its check."""
    return Path(folder) / "projection" / "projection.json"


def write_projection(folder, record: dict) -> Path:
    path = C.assert_safe_output(projection_path(folder))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), machine=system_memory())
    path.write_text(json.dumps(plain(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_projection(folder):
    path = projection_path(folder)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def need_projection(folder, limit_gb: float, who: str) -> dict:
    """The projection of this artifact, or SystemExit when there is none or it is above the guard: the full fit is
    not started on a guess."""
    record = read_projection(folder)
    if record is None:
        raise SystemExit(f"{who}: no memory projection yet; run the project command first")
    if not record.get("ok") or record.get("projected_demand_gb", float("inf")) > limit_gb:
        raise SystemExit(f"{who}: the projected memory of the full fit ({record.get('projected_demand_gb')} GB held, "
                         f"{record.get('projected_footprint_gb')} GB at its peak) is above the {limit_gb:.1f} GB guard; "
                         "not started")
    return record


def run_subsets(module: str, data_root, games, limit_gb: float) -> list:
    """Run `python3 -m <module> subset --games K` once per K, each in its own process so that each has its own
    memory record, and return what they measured."""
    runs = []
    for k in games:
        command = [sys.executable, "-B", "-m", module, "subset", "--games", str(int(k)), "--data", str(data_root),
                   "--limit-gb", str(limit_gb)]
        done = subprocess.run(command, cwd=C.REPO_ROOT, stdout=subprocess.PIPE, text=True)
        if done.returncode == MEMORY_EXIT:
            raise SystemExit(f"{module}: the subset run with {k} games was ended by the memory watch; not projected")
        if done.returncode != 0:
            raise SystemExit(f"{module}: the subset run with {k} games failed with status {done.returncode}")
        lines = [line for line in done.stdout.splitlines() if line.startswith("SUBSET ")]
        if not lines:
            raise SystemExit(f"{module}: the subset run with {k} games reported nothing")
        runs.append(json.loads(lines[-1][len("SUBSET "):]))
    return runs


# ------------------------------------------------------------------------------------- provenance (shared)

def versions(torch: bool = False) -> dict:
    """The interpreter, the platform and the libraries a result depends on. Reproduction is bound to them."""
    import scipy
    import sklearn
    out = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
    if torch:
        out["torch"] = DT.require_torch(WHO).__version__
    return out


def same_versions(record: dict, name: str, torch: bool = False) -> None:
    """RuntimeError when the libraries differ from the ones the check ran under: the check does not vouch for
    another environment, and rerunning it there takes about a minute."""
    then, now = record.get("versions") or {}, versions(torch)
    changed = sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k))
    if changed:
        raise RuntimeError(f"{name}: checked under other versions of {changed}; rerun the check command here first")


def frozen_code_state() -> dict:
    """What git says about the frozen lanes: HEAD, the tracked files under analytics/ and sportvu/ that differ
    from the freeze-v1 tag, and the ones modified in the working tree. A value is None when git cannot answer."""
    def git(*args):
        try:
            done = subprocess.run(["git", *args], cwd=C.REPO_ROOT, capture_output=True, text=True, check=True, timeout=60)
            return done.stdout.strip()
        except Exception:
            return None

    diff = git("diff", "--name-only", "freeze-v1", "HEAD", "--", "analytics", "sportvu")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "analytics", "sportvu")
    return {"head": git("rev-parse", "HEAD"),
            "differs_from_freeze_v1": None if diff is None else diff.splitlines(),
            "uncommitted": None if dirty is None else [line[3:] for line in dirty.splitlines()]}


def require_frozen_code(who: str) -> dict:
    """The git state of the frozen lanes, or SystemExit when lane code differs from freeze-v1: a fit of changed
    code is not the frozen model whatever its check says."""
    state = frozen_code_state()
    changed = [p for p in state["differs_from_freeze_v1"] or [] if p not in ALLOWED_FROZEN_DIFF] + (state["uncommitted"] or [])
    if changed:
        raise SystemExit(f"{who}: lane code differs from freeze-v1 ({', '.join(changed)}); not rebuilt")
    return state


def write_check(folder, record: dict, files=()) -> Path:
    """tools.frozen_common.write_check, with its notices sent to stderr so that stdout holds the record alone."""
    with contextlib.redirect_stdout(sys.stderr):
        return C.write_check(folder, record, files=files)


def write_stopped(folder, name: str, tolerance: float, reason: str) -> None:
    """Record why a rebuild ended without a check. The record cannot pass: nothing was compared."""
    write_check(folder, {"name": name, "compared": "nothing: the rebuild ended before its check", "n": 0,
                         "max_abs_diff": None, "tolerance": tolerance, "status": reason})


def stopped_reason(error: BaseException) -> str:
    """What a stopped rebuild writes into its record: the message of a stop this code raised itself, and only the
    type of anything else (an unexpected error can carry a local path, which must not reach a data file)."""
    if isinstance(error, (MemoryError, SystemExit)):
        return f"{type(error).__name__}: {error}"
    return f"{type(error).__name__} (see the output of the run)"


def stopper(folder, name: str, tolerance: float, cleanup=()):
    """The on_trip action of a real rebuild: record why, run the cleanups, leave the process."""
    def on_trip(watch):
        try:
            write_stopped(folder, name, tolerance, "ended by the memory watch: " + watch.reason)
        except BaseException:
            pass
        for action in cleanup:
            try:
                action()
            except BaseException:
                pass
        stop_process(watch)
    return on_trip


def expected(expect, frozen_record: dict):
    """The counts a run is held to: the frozen record when `expect` is True, a record the caller gives, or none."""
    return frozen_record if expect is True else (expect or None)


def data_root_of(data_root=None) -> Path:
    root = data_root or os.environ.get("DATA_ROOT")
    if not root:
        raise SystemExit("give the data root or set DATA_ROOT")
    return Path(root).expanduser()


@contextlib.contextmanager
def terminating():
    """Inside the block SIGTERM raises SystemExit, so that a stopped job leaves through its cleanup (temporary
    folders are removed); the previous handler comes back after. Outside the main thread it does nothing."""
    def handler(signum, frame):
        raise SystemExit(128 + signum)

    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, handler) if main_thread else None
    try:
        yield
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------------------------------------------- the fit

def _load_training(data_root) -> dict:
    """The frozen loader and the frozen split. Returns the training tables in a box, so that the fit can let go of
    them one by one, with the counts of what was set aside."""
    tables = V.load_root(data_root)
    d = T.split_tables(tables)
    held = d["states_heldout"]
    counts = {"train_states": int(len(d["states_train"])), "train_games": int(len(d["train_games"])),
              "heldout_states": int(len(held)), "heldout_games": int(held["sportvu_game"].nunique()),
              "validation": {name: {k: r[k] for k in ("rows_in", "duplicates_dropped", "rows_dropped", "rows_out")}
                             for name, r in d["reports"].items()}}
    return {"states": d["states_train"], "shots": d["shots_train"], "rosters": d["rosters"], "counts": counts}


def _first_games(box: dict, n_games: int) -> None:
    """Keep the rows of the first `n_games` training games (sorted ids). Only the projection uses this."""
    st = box["states"]
    keep = sorted(st["sportvu_game"].astype(str).unique())[:n_games]
    box["states"] = st[st["sportvu_game"].astype(str).isin(keep)].reset_index(drop=True)


def _fit_calibrated(box: dict):
    """The fitting calls of the frozen trainer's refit on every training row (analytics/epv/train.py, the function
    that runs after the selection), unchanged and in its order, without its scoring half. The states and the
    geometry are released once the lagged frame exists; the frozen code does not read them again."""
    st = box.pop("states")
    shots_train, rosters = box.pop("shots"), box.pop("rosters")
    geo = F.geometry(st)
    schema.assert_time_forward(geo, "states_1hz")
    priors = F.Priors.from_training(st, shots_train, rosters)
    X = F.lagged(F.build_features(st, priors, geo=geo), T.N_LAGS)
    y = st["points_to_end"].to_numpy(dtype=float)
    games = st["sportvu_game"].to_numpy()
    del geo, st, shots_train, rosters
    gc.collect()
    cols = F.lagged_columns(T.N_LAGS)
    c = T.Calibrated(T.SEQUENCE, cols, SEED, HGB_ITER, DEEP).fit(X, y, games)
    sample = X.iloc[:4096].copy()
    return c, priors, sample


def _fit(data_root, limit_gb: float, n_games=None, expect=None, every_s: float = 1.0, on_trip=None) -> dict:
    """Load, split and fit. `n_games` limits the fit to the first training games (the projection); `expect` holds
    the frozen record's counts and stops the run before the fit when the inputs are not the ones it names."""
    with fit_threads():      # the frozen runs loaded scikit-learn's OpenMP runtime before torch; keep that order
        pass
    stages = {}
    with stage("load", stages, limit_gb, every_s, on_trip):
        box = _load_training(data_root)
    counts = box.pop("counts")
    if expect:
        wrong = {k: (counts[k], expect[k]) for k in ("train_states", "train_games") if counts[k] != expect[k]}
        if wrong:
            raise SystemExit(f"{WHO}: the training inputs are not the frozen record's (found, expected): {wrong}")
    if n_games:
        _first_games(box, n_games)
    games = sorted(box["states"]["sportvu_game"].astype(str).unique())
    C.training_games_only(games, WHO)
    rows = int(len(box["states"]))
    say(WHO, f"fit on {rows} states of {len(games)} training games")
    with stage("fit", stages, limit_gb, every_s, on_trip):
        c, priors, sample = _fit_calibrated(box)
    return {"c": c, "priors": priors, "sample": sample, "stages": stages, "counts": counts, "rows": rows,
            "games": games, "history": dict(c.model.history_)}


# --------------------------------------------------------------------------------------------- save and read

class XfgPrior:
    """The location-only make probability the priors shrink towards, as an object that can be rebuilt from saved
    parts. analytics.epv.features.fit_xfg returns a closure, which cannot be pickled; this holds the same fitted
    pipeline and makes the same frozen calls on it (features_xfg.features, then train_xfg.design)."""

    def __init__(self, pipeline=None, rate=None):
        self.pipeline, self.rate = pipeline, rate

    def __call__(self, loc_x, loc_y):
        if self.pipeline is None:
            return np.full(len(loc_x), self.rate)
        from analytics import features_xfg as fx
        from analytics import train_xfg as tx
        q = pd.DataFrame({"sportvu_game": "query", "made": False, "loc_x": np.asarray(loc_x, dtype=float),
                          "loc_y": np.asarray(loc_y, dtype=float), "period": 1, "clock_s": 0.0})
        return np.asarray(self.pipeline.predict_proba(tx.design(fx.features(q)).to_numpy())[:, 1], dtype=float)


def _xfg_parts(xfg) -> dict:
    """The fitted pipeline (or the constant rate) inside the closure analytics.epv.features.fit_xfg returned."""
    inner = xfg.__defaults__[0]
    if xfg.__name__ == "predict":                 # predict(loc_x, loc_y, m=model); model is lambda frame, m=pipeline
        inner = inner.__defaults__[0]
    if isinstance(inner, float):
        return {"pipeline": None, "rate": float(inner)}
    if not hasattr(inner, "predict_proba"):
        raise RuntimeError(f"{WHO}: the xFG prior is not the closure the frozen code is known to return")
    return {"pipeline": inner, "rate": None}


def _save(folder: Path, fitted: dict, code: dict) -> None:
    """Write the parts, read them back from disk, and confirm that the saved model gives the fitted model's
    predictions bit for bit before the metadata is written."""
    torch = DT.require_torch(WHO)
    c, priors = fitted["c"], fitted["priors"]
    m = c.model
    xfg = _xfg_parts(priors.xfg)
    rng = np.random.default_rng(0)
    loc_x, loc_y = rng.uniform(-250.0, 250.0, 4000), rng.uniform(-50.0, 900.0, 4000)
    if not np.array_equal(XfgPrior(**xfg)(loc_x, loc_y), priors.xfg(loc_x, loc_y)):
        raise RuntimeError(f"{WHO}: the saved xFG prior does not equal the frozen closure")
    parts = {"n_steps": m.n_steps, "n_features": m.n_features, "hidden": m.hidden, "seed": m.seed,
             "threads": m.threads, "imputer": m.imp_, "xs": m.xs_, "ys": m.ys_, "iso": c.iso,
             "p_const": c.p_const, "columns": list(c.columns), "hgb_iter": c.hgb_iter,
             "priors": {"shooter": priors.shooter, "lineup": priors.lineup, "tov_rate": priors.tov_rate,
                        "train_games": sorted(priors.train_games), "xfg": xfg}}
    torch.save(m.net_.state_dict(), folder / STATE_FILE)
    with open(folder / PARTS_FILE, "wb") as handle:
        pickle.dump(parts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    again = _read(folder)
    sample = fitted["sample"]
    epv_a, p_a = c.predict(sample)
    epv_b, p_b = again.calibrated.predict(sample)
    if not (np.array_equal(epv_a, epv_b) and np.array_equal(p_a, p_b)):
        raise RuntimeError(f"{WHO}: the model read back from disk does not equal the fitted model")
    meta = {"name": NAME, "model": "torch GRU (hidden %d) over the last %d states, isotonic map to P(points > 0); "
                                   "slot name mlp in the reports" % (m.hidden, m.n_steps),
            "frozen_tag": "freeze-v1", "code": code,
            "calls": ["analytics.epv.validate.load_root", "analytics.epv.train.split_tables",
                      "analytics.epv.features.geometry", "analytics.epv.features.Priors.from_training",
                      "analytics.epv.features.build_features", "analytics.epv.features.lagged",
                      "analytics.epv.train.Calibrated(SEQUENCE, lagged_columns(N_LAGS), seed 0, 100, 'torch').fit"],
            "settings": {"seed": SEED, "deep": DEEP, "n_lags": T.N_LAGS, "inner_folds": T.INNER_FOLDS,
                         "hidden": D.GRU_HIDDEN, "max_epochs": D.MAX_EPOCHS, "patience": D.PATIENCE, "batch": D.BATCH,
                         "lr": D.LR, "weight_decay": D.WEIGHT_DECAY, "threads": D.THREADS},
            "train": {"games": len(fitted["games"]), "states": fitted["rows"]}, "counts": fitted["counts"],
            "final_fit": fitted["history"], "saved_equals_fitted": True, "xfg_prior_equals_frozen_closure": True,
            "isotonic_thresholds": None if c.iso is None else int(len(c.iso.X_thresholds_)),
            "versions": versions(torch=True), "stages": fitted["stages"], "machine_at_save": system_memory()}
    (folder / META_FILE).write_text(json.dumps(plain(meta), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(folder: Path) -> "EpvPredictor":
    """The predictor from the saved parts, whatever the check says. `load` is the door for everyone else."""
    torch = DT.require_torch(WHO)
    with open(Path(folder) / PARTS_FILE, "rb") as handle:
        parts = pickle.load(handle)
    net = D._net(torch, parts["n_steps"], parts["n_features"], parts["hidden"])
    net.load_state_dict(torch.load(Path(folder) / STATE_FILE, map_location="cpu", weights_only=True))
    net.eval()
    m = D.GRUSequenceRegressor(n_steps=parts["n_steps"], hidden=parts["hidden"], seed=parts["seed"], threads=parts["threads"])
    m.n_features, m.net_, m.imp_, m.xs_, m.ys_ = parts["n_features"], net, parts["imputer"], parts["xs"], parts["ys"]
    c = T.Calibrated(T.SEQUENCE, parts["columns"], parts["seed"], parts["hgb_iter"], DEEP)
    c.model, c.iso, c.p_const = m, parts["iso"], parts["p_const"]
    pr = parts["priors"]
    priors = F.Priors(pr["shooter"], pr["lineup"], XfgPrior(**pr["xfg"]), pr["tov_rate"], set(pr["train_games"]))
    return EpvPredictor(c, priors)


# ---------------------------------------------------------------------------------------------- the predictor

def _possessions_for(possessions: pd.DataFrame, game: str) -> pd.DataFrame:
    """The possession columns the frozen states builder reads, for one game. The builder also copies `points` and
    `end_kind` into the two target columns of the states table; constants stand in for them here, so no outcome
    is read. The features never read those targets, so a prediction cannot depend on them."""
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
    out = rows[list(POSSESSION_COLUMNS)].reset_index(drop=True).copy()
    out["points"], out["end_kind"] = 0, "other"
    return out


class EpvPredictor:
    """The frozen EPV model, ready to predict. Nothing here fits: the states and the features are built by the
    frozen code and pushed through the saved weights."""

    def __init__(self, calibrated, priors):
        self.calibrated, self.priors = calibrated, priors
        self.train_games = frozenset(priors.train_games)
        self.p_score_checked = False
        self.check = None

    def predict_table(self, states: pd.DataFrame):
        """(expected points, P(points > 0)) for the rows of a states table (docs/DATA_CONTRACT.md section 20; the
        keys and the allow-listed columns are enough), in its row order."""
        lagged = F.lagged(F.build_features(states, self.priors), T.N_LAGS)
        return self.calibrated.predict(lagged)

    def predict_states(self, frames: pd.DataFrame, possessions: pd.DataFrame, game: str, extra: bool = False) -> pd.DataFrame:
        """EPV at every 1 Hz state of one game's possessions, from its tracking frames.

        `frames` is one game's frames table (docs/DATA_CONTRACT.md section 4: one row per player per frame; frame,
        period, game_clock_s, shot_clock_s, ball_x, ball_y, ball_z, team_id, player_id, x, y). `possessions` is its
        possessions table (section 19); only possession, offense_team_id, start_frame, end_frame and
        start_clock_s are read. Returns sportvu_game, possession, frame, epv_points, one row per state, ordered by
        possession and time. With `extra`: elapsed_in_possession_s, p_score (when its own check passed) and
        from_start.

        A state's features run from the first state of its possession, so give the frames of a possession from
        its start_frame: a possession the frames cover only in part still gets values for the seconds they cover,
        computed from those states alone, and `from_start` is false for it. A training game is refused: the model
        was fitted on it, so its values there would be in-sample."""
        game = str(game)
        if game in self.train_games:
            raise ValueError(f"{game} is a training game of this model; its predictions there would be in-sample")
        poss = _possessions_for(possessions, game)
        states, report = V.check_states(S.build_states(frames, poss, game, events=None))
        out = pd.DataFrame({"sportvu_game": pd.Series([], dtype=str), "possession": pd.Series([], dtype=np.int32),
                            "frame": pd.Series([], dtype=np.int32), "epv_points": pd.Series([], dtype=float)})
        p = np.zeros(0)
        if len(states):
            epv, p = self.predict_table(states)
            out = pd.DataFrame({"sportvu_game": states["sportvu_game"].astype(str).to_numpy(),
                                "possession": states["possession"].to_numpy(dtype=np.int32),
                                "frame": states["frame"].to_numpy(dtype=np.int32),
                                "epv_points": np.asarray(epv, dtype=float)})
        if extra:
            out["elapsed_in_possession_s"] = states["elapsed_in_possession_s"].to_numpy(dtype=np.float32)
            if self.p_score_checked:
                out["p_score"] = np.asarray(p, dtype=float)
            present = np.isin(poss["start_frame"].to_numpy(), frames["frame"].unique())
            covered = dict(zip(poss["possession"].to_numpy().tolist(), present.tolist()))
            out["from_start"] = np.array([covered.get(int(k), False) for k in out["possession"].to_numpy()], dtype=bool)
        out.attrs.update(states=int(len(out)), possessions=int(out["possession"].nunique()),
                         rows_dropped=int(report.get("rows_dropped", 0)))
        return out


# -------------------------------------------------------------------------------------------------- the check

def _heldout_games(data_root: Path) -> list:
    """The held-out games that have a stored prediction file, in sorted order (the per-season possession file of
    the same folder is not a game and holds an outcome column, so it is never opened)."""
    base = data_root / "analytics" / "epv" / "heldout"
    if not base.is_dir():
        raise SystemExit(f"{WHO}: no stored held-out predictions under analytics/epv/heldout")
    return sorted(p.stem for season in base.iterdir() if season.is_dir() for p in season.glob("*.parquet")
                  if p.name != "epv_possessions.parquet")


def _stored_predictions(data_root: Path, games: list) -> pd.DataFrame:
    """The stored held-out predictions: keys, epv_points and p_score. points_to_end is never read."""
    parts = []
    for game in games:
        path = data_root / "analytics" / "epv" / "heldout" / schema.season_of_game(game) / f"{game}.parquet"
        parts.append(pd.read_parquet(path, columns=KEYS + ["epv_points", "p_score"]))
    return pd.concat(parts, ignore_index=True)


def _stored_states(data_root: Path) -> pd.DataFrame:
    """The stored 1 Hz states of the held-out games in the frozen loader's order (season folders, then files, by
    name), cleaned by the frozen check, without the two outcome columns."""
    import pyarrow.parquet as pq
    base = data_root / "analytics" / "states_1hz" / "heldout"
    wanted = KEYS + list(F.ALLOWED)
    parts = []
    for season in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(season.glob("*.parquet")):
            names = pq.read_schema(path).names
            df = pd.read_parquet(path, columns=[c for c in wanted if c in names])
            if "sportvu_game" not in df.columns:
                df["sportvu_game"] = path.stem
            parts.append(df)
    if not parts:
        raise SystemExit(f"{WHO}: no stored held-out states under analytics/states_1hz/heldout")
    states, _report = V.check_states(pd.concat(parts, ignore_index=True))
    return states


def _from_frames(predictor: "EpvPredictor", data_root: Path, games: list, stored: pd.DataFrame, limit_gb: float) -> dict:
    """The same held-out games through predict_states, one game's frames at a time."""
    import pyarrow.parquet as pq
    parts = {}
    for game in games:
        fpath = data_root / "sportvu_contract" / "heldout" / f"{game}.parquet"
        ppath = schema.table_path(data_root / "analytics", "possessions", schema.season_of_game(game), game, heldout=True)
        names = pq.read_schema(ppath).names
        possessions = pd.read_parquet(ppath, columns=[c for c in ("sportvu_game",) + POSSESSION_COLUMNS if c in names])
        frames = pd.read_parquet(fpath, columns=list(FRAME_COLUMNS))
        predicted = predictor.predict_states(frames, possessions, game)
        del frames
        parts[game] = compare(predicted, stored[stored["sportvu_game"].astype(str) == game], KEYS, ["epv_points"], TOLERANCE)
        C.MemoryGuard(limit_gb).check("check from frames, " + game)
    return combine(parts)


def check(data_root=None, artifacts=None, expect=True, limit_gb: float = LIMIT_GB) -> dict:
    """Compare the saved model with the frozen record and write check.json. Reads the held-out tables without
    their outcome columns, fits nothing, scores nothing. `expect` is True for the counts of FROZEN_RECORD (the
    real data root), a record of counts, or None for a root that is not the frozen one (the fixtures)."""
    expect = expected(expect, FROZEN_RECORD)
    data_root = data_root_of(data_root)
    folder = C.assert_safe_output(C.artifacts_dir(artifacts) / NAME, data_root)
    missing = [name for name in FILES if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"{WHO}: no rebuilt model to check ({', '.join(missing)} missing); run rebuild first")
    started = time.time()
    predictor = _read(folder)
    games = _heldout_games(data_root)
    stored = _stored_predictions(data_root, games)
    states = _stored_states(data_root)
    epv, p = predictor.predict_table(states)          # every held-out state in one call, as the frozen run did
    predicted = pd.DataFrame({"sportvu_game": states["sportvu_game"].astype(str).to_numpy(),
                              "possession": states["possession"].to_numpy(), "frame": states["frame"].to_numpy(),
                              "epv_points": epv, "p_score": p})
    main = compare(predicted, stored, KEYS, ["epv_points"], TOLERANCE)
    p_score = compare(predicted, stored, KEYS, ["p_score"], TOLERANCE)
    by_game = {}
    for game in games:
        mine = predicted[predicted["sportvu_game"] == game]
        theirs = stored[stored["sportvu_game"].astype(str) == game]
        by_game[game] = compare(mine, theirs, KEYS, ["epv_points"], TOLERANCE, worst=0)
    main["per_game"] = {g: {"n": r["n"], "max_abs_diff": r["max_abs_diff"], "passed": r["passed"]} for g, r in by_game.items()}
    say(WHO, f"stored states: n {main['n']}, max abs diff {main['max_abs_diff']}, passed {main['passed']}")
    from_frames = _from_frames(predictor, data_root, games, stored, limit_gb)
    say(WHO, f"from frames: n {from_frames['n']}, max abs diff {from_frames['max_abs_diff']}, passed {from_frames['passed']}")
    found = {"heldout_games": len(games), "heldout_states": int(len(stored))}
    as_recorded = not expect or all(found[k] == expect[k] for k in found)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    record = {"name": NAME, "unit": UNIT,
              "compared": "predictions of the rebuilt GRU for the stored 1 Hz states of the held-out games against the "
                          "stored epv_points ($DATA_ROOT/analytics/epv/heldout); predictions with predictions, no "
                          "outcome column read, nothing scored",
              "n": main["n"], "n_expected": int(len(stored)), "max_abs_diff": main["max_abs_diff"], "tolerance": TOLERANCE,
              "passed": bool(main["passed"] and from_frames["passed"] and as_recorded),
              "stored_states": main, "from_frames": from_frames, "p_score": p_score,
              "found": found, "frozen_record": expect, "matches_frozen_record": bool(as_recorded),
              "fit": {"train": meta.get("train"), "final_fit": meta.get("final_fit"), "stages": meta.get("stages"),
                      "code": meta.get("code")},
              "versions": versions(torch=True), "check_seconds": round(time.time() - started, 1),
              "machine": system_memory()}
    write_check(folder, record, files=FILES)
    return C.read_check(folder)


# ------------------------------------------------------------------------------------------- rebuild and load

def rebuild(data_root=None, artifacts=None, limit_gb: float = LIMIT_GB, expect=True,
            require_projection: bool = True) -> dict:
    """Rerun the frozen final fit on the training games, save the parts, run the check, and return its record.

    Writes only under <artifacts>/epv. Take the heavy-job lock around this call (the command line does). After a
    MemoryError the process must end: the fit was cut short. `expect` as in `check`."""
    expect = expected(expect, FROZEN_RECORD)
    data_root = data_root_of(data_root)
    folder = C.assert_safe_output(C.artifacts_dir(artifacts) / NAME, data_root)
    if require_projection:
        need_projection(folder, limit_gb, WHO)
    code = require_frozen_code(WHO)
    C.begin_rebuild(folder, NAME)
    on_trip = stopper(folder, NAME, TOLERANCE)
    say(WHO, f"rebuild starts; machine {system_memory()}")
    try:
        fitted = _fit(data_root, limit_gb, expect=expect, on_trip=on_trip)
        with stage("save", fitted["stages"], limit_gb, on_trip=on_trip):
            _save(folder, fitted, code)
    except BaseException as error:
        write_stopped(folder, NAME, TOLERANCE, stopped_reason(error))
        raise
    del fitted
    gc.collect()
    return check(data_root, artifacts, expect=expect, limit_gb=limit_gb)


def load(artifacts=None) -> EpvPredictor:
    """The frozen EPV predictor, or RuntimeError when there is no rebuilt model, its check did not pass, a file
    changed after the check, or the libraries are not the ones the check ran under."""
    folder = C.artifacts_dir(artifacts) / NAME
    record = C.require_passed(folder, NAME)
    same_versions(record, NAME, torch=True)
    predictor = _read(folder)
    predictor.p_score_checked = bool((record.get("p_score") or {}).get("passed"))
    predictor.check = record
    return predictor


# -------------------------------------------------------------------------------------------- the projection

def subset(data_root, n_games: int, limit_gb: float = LIMIT_GB) -> dict:
    """The same load and fit on the first `n_games` training games, measured. Nothing is saved."""
    fitted = _fit(data_root_of(data_root), limit_gb, n_games=n_games, every_s=0.25)
    return {"games": len(fitted["games"]), "rows": fitted["rows"], "rows_full": fitted["counts"]["train_states"],
            "stages": fitted["stages"], "final_fit": fitted["history"]}


def project(data_root=None, artifacts=None, games=PROJECTION_GAMES, limit_gb: float = LIMIT_GB) -> dict:
    """Measure the fit on subsets of training games, each in its own process, project the memory and the time of
    the full fit, and write the projection. The load stage is the same at every size; the fit stage grows with
    the rows."""
    data_root = data_root_of(data_root)
    folder = C.assert_safe_output(C.artifacts_dir(artifacts) / NAME, data_root)
    runs = run_subsets("tools.frozen_epv", data_root, games, limit_gb)
    record = extrapolate(runs, runs[-1]["rows_full"], limit_gb, fixed=("load",), scaled=("fit",))
    record.update(name=NAME, what="the frozen EPV final fit, measured on the first training games and projected "
                                  "to every training state")
    write_projection(folder, record)
    return record


# ------------------------------------------------------------------------------------------- the command line

def parser_for(module: str, description: str, default_games) -> argparse.ArgumentParser:
    """The command line both rebuild modules share: project, rebuild, check, and the internal subset."""
    parser = argparse.ArgumentParser(prog=f"python3 -m {module}", description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("project", "measure the fit on subsets of training games and project the full fit"),
                       ("rebuild", "fit on the training games, save, check, write check.json"),
                       ("check", "rerun only the comparison on the saved model"),
                       ("subset", "one measured subset run (used by project)")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--data", default=None, help="the data root (default: DATA_ROOT)")
        sub.add_argument("--artifacts", default=None, help="the artifacts folder (default: V3_ARTIFACTS, else artifacts/)")
        sub.add_argument("--limit-gb", type=float, default=LIMIT_GB, help="the memory guard in GB")
        if name == "project":
            sub.add_argument("--games", default=",".join(str(k) for k in default_games),
                             help="training games per subset run, comma separated")
        if name == "subset":
            sub.add_argument("--games", type=int, required=True, help="training games in this run")
    return parser


def main(argv=None) -> int:
    args = parser_for("tools.frozen_epv", "Rebuild and check the frozen EPV model (PLAN.md D45).", PROJECTION_GAMES).parse_args(argv)
    with terminating(), heavy_job(f"frozen_epv {args.command}", log=lambda m: print(m, file=sys.stderr, flush=True)):
        if args.command == "subset":
            print("SUBSET " + json.dumps(plain(subset(args.data, args.games, args.limit_gb))), flush=True)
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
            return MEMORY_EXIT
        print(json.dumps(record, indent=2))
        return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

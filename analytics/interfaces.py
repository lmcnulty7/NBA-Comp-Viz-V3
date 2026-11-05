"""analytics/interfaces.py: the registry of the five cross-lane interfaces and the integration helpers (lane I6).

An interface is a parquet table one extension lane (the producer) writes under OUT_ROOT and one or more other
lanes (the consumers) read. The registry below records, for each, the producer, the consumers, the required
columns with their dtype class, the key columns and the path pattern under OUT_ROOT. It imports nothing from the
extension packages, so it works while a lane is absent or half-written.

    from analytics import interfaces
    interfaces.INTERFACES["xfg_predictions"]            # the registry entry (a dict)
    interfaces.interface_path(out_root, "epv_states", "2015-16", "12.16.2015.PHX.at.GSW")
    interfaces.find_outputs(out_root, "player_rapm")    # every file matching the entry's glob under OUT_ROOT
    interfaces.check_frame("xfg_predictions", df)       # AssertionError listing missing or mistyped columns
    interfaces.discover_lanes()                         # every analytics/<ext>/ with validate, features, train

Dtype classes: ``string`` (object or pandas string dtype), ``int`` (any integer dtype, nullable Int64 included),
``float`` (float, or an integer dtype, which is accepted as a degenerate float), ``bool`` (bool or nullable
boolean). A nullable column may hold nulls; the dtype class still has to match.

Where the producer lane has documented its columns (docs/contract/<ext>.md, or the docstring and constants of
its train module when the contract page is not written yet) the entry matches that. Where it has not, the entry
records the columns the consumers already code against, and the assumption is stated in the entry's ``note``
and in docs/contract/interfaces.md:

  xfg_predictions      analytics/xfg/train.py PREDICTION_COLUMNS: sportvu_game, event, player_id, p_location,
                       p_contest, p_selected, made, fold, heldout, one file per season
                       (analytics/xfg/<season>/xfg_predictions.parquet). Consumers (decisions) join p_selected
                       by (sportvu_game, event). gravity and epv use the location-only baseline of
                       analytics.train_xfg today and are listed as consumers of the lane's shot-quality column.
  epv_states           not documented by the epv lane yet (no docs/contract/epv.md, no train module). The
                       consumers (decisions, playcalling) read OUT_ROOT/analytics/epv/<season>/<game>.parquet
                       with sportvu_game, possession, frame, epv_points; that is the assumption recorded here.
                       The lane's states_1hz table (analytics/states_1hz/) is the contract section 20 table,
                       not this interface.
  actions_detected     not documented yet (no docs/contract/actions.md, no train module). The consumers
                       (coverage, playcalling) read OUT_ROOT/analytics/actions/<season>/<game>.parquet with the
                       hand-label columns sportvu_game, possession, action_type, start_frame, end_frame,
                       ball_handler_id, screener_id (nullable), label_source; action_id is optional and
                       defaults to the row's position in the file. Files under analytics/actions/<season>/
                       features/ are the lane's feature frames, not this interface, and are skipped.
  coverage_predictions not documented yet (no docs/contract/coverage.md, no train module). The consumer
                       (playcalling) reads OUT_ROOT/analytics/coverage/**/*.parquet with sportvu_game,
                       possession, action_id, predicted and p_<class> columns in [0, 1]. The file name
                       coverage_predictions.parquet per season is the assumption; coverage_features.parquet
                       (the lane's feature frame) is skipped.
  player_rapm          analytics/lineups/train.py docstring: season, player_id, rapm, lo, hi, fit_seasons, one
                       file per season (analytics/lineups/<season>/player_rapm.parquet). betting reads team_id
                       and as_of when present and keeps its RAPM columns null otherwise; both are optional here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_DIR = Path(__file__).resolve().parent
STANDARD_STEPS = ("validate", "features", "train")

# The fifteen extension lanes of the analytics phase (the names of analytics/<ext>/ packages).
EXPECTED_LANES = ("xfg", "rebounds", "matchups", "gravity", "ghosting", "embeddings", "epv", "actions",
                  "coverage", "decisions", "playcalling", "lineups", "winprob", "load", "betting")

COVERAGE_CLASSES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over")


def _iface(name, producer, consumers, columns, keys, pattern, glob, optional=(), per_game=False, note=""):
    return {"name": name, "producer": producer, "consumers": list(consumers),
            "columns": dict(columns), "optional": dict(optional), "keys": list(keys),
            "pattern": pattern, "glob": glob, "per_game": per_game, "note": note}


INTERFACES = {
    "xfg_predictions": _iface(
        "xfg_predictions", "xfg", ["decisions", "gravity", "epv"],
        columns={"sportvu_game": "string", "event": "int", "p_selected": "float", "made": "bool",
                 "fold": "int", "heldout": "bool"},
        optional={"player_id": "int", "p_location": "float", "p_contest": "float"},
        keys=["sportvu_game", "event"],
        pattern="analytics/xfg/<season>/xfg_predictions.parquet",
        glob="analytics/xfg/**/xfg_predictions*.parquet",
        note="matches analytics/xfg/train.py PREDICTION_COLUMNS; one file per season, held-out rows flagged"),
    "epv_states": _iface(
        "epv_states", "epv", ["decisions", "playcalling"],
        columns={"sportvu_game": "string", "possession": "int", "frame": "int", "epv_points": "float"},
        optional={"heldout": "bool", "fold": "int"},
        keys=["sportvu_game", "possession", "frame"],
        pattern="analytics/epv/<season>/<game>.parquet",
        glob="analytics/epv/**/*.parquet", per_game=True,
        note="assumed from the consumers (decisions, playcalling); the epv lane has not documented it yet"),
    "actions_detected": _iface(
        "actions_detected", "actions", ["coverage", "playcalling"],
        columns={"sportvu_game": "string", "possession": "int", "action_type": "string", "start_frame": "int",
                 "end_frame": "int", "ball_handler_id": "int", "label_source": "string"},
        optional={"screener_id": "int", "action_id": "int", "labeller_pass": "string", "heldout": "bool"},
        keys=["sportvu_game", "possession", "start_frame"],
        pattern="analytics/actions/<season>/<game>.parquet",
        glob="analytics/actions/**/*.parquet", per_game=True,
        note="assumed from the consumers (coverage ACTION_COLUMNS, playcalling); files under a features/ "
             "directory are the lane's feature frames and are skipped"),
    "coverage_predictions": _iface(
        "coverage_predictions", "coverage", ["playcalling"],
        columns={"sportvu_game": "string", "possession": "int", "action_id": "int", "predicted": "string"},
        optional={**{"p_" + c: "float" for c in COVERAGE_CLASSES}, "heldout": "bool", "fold": "int"},
        keys=["sportvu_game", "possession", "action_id"],
        pattern="analytics/coverage/<season>/coverage_predictions.parquet",
        glob="analytics/coverage/**/coverage_predictions*.parquet",
        note="assumed from the consumer (playcalling): predicted plus p_<class> in [0, 1]; "
             "coverage_features.parquet is the lane's feature frame and is not matched"),
    "player_rapm": _iface(
        "player_rapm", "lineups", ["matchups", "betting"],
        columns={"season": "string", "player_id": "int", "rapm": "float", "lo": "float", "hi": "float"},
        optional={"fit_seasons": "string", "team_id": "int", "as_of": "string"},
        keys=["season", "player_id"],
        pattern="analytics/lineups/<season>/player_rapm.parquet",
        glob="analytics/lineups/**/player_rapm*.parquet",
        note="matches analytics/lineups/train.py; betting uses team_id and as_of only when present"),
}

INTERFACE_NAMES = tuple(INTERFACES)
_SKIP_DIRS = {"features", "eda"}        # sub-directories of a producer's folder that are never the interface


# ── paths ──────────────────────────────────────────────────────────────────────────────────────────
def get(name: str) -> dict:
    if name not in INTERFACES:
        raise KeyError("unknown interface %r; known: %s" % (name, ", ".join(INTERFACE_NAMES)))
    return INTERFACES[name]


def interface_path(out_root, name: str, season: str, game: str | None = None, heldout: bool = False) -> Path:
    """The file the producer writes for (season, game) under OUT_ROOT. Season-level interfaces ignore `game`;
    held-out games of per-game interfaces sit under heldout/<season>/ (the Phase 0 convention)."""
    e = get(name)
    season_dir = ("heldout/" + season) if heldout else season
    rel = e["pattern"].replace("<season>", season_dir)
    if e["per_game"]:
        if not game:
            raise ValueError("%s is written per game: pass the game id" % name)
        rel = rel.replace("<game>", str(game))
    return Path(out_root) / rel


def find_outputs(out_root, name: str) -> list[Path]:
    """Every parquet file under OUT_ROOT that matches the interface's glob (feature and eda folders skipped)."""
    e = get(name)
    root = Path(out_root)
    out = []
    for p in sorted(root.glob(e["glob"])):
        if not p.is_file():
            continue
        rel_parts = set(p.relative_to(root).parts[:-1])
        if rel_parts & _SKIP_DIRS:
            continue
        out.append(p)
    return out


# ── frames ─────────────────────────────────────────────────────────────────────────────────────────
def _dtype_ok(s: pd.Series, cls: str) -> bool:
    dt = s.dtype
    if cls == "string":
        return pd.api.types.is_string_dtype(dt) or dt == object
    if cls == "int":
        return pd.api.types.is_integer_dtype(dt)
    if cls == "float":
        return pd.api.types.is_float_dtype(dt) or pd.api.types.is_integer_dtype(dt)
    if cls == "bool":
        return pd.api.types.is_bool_dtype(dt)
    raise ValueError("unknown dtype class %r" % cls)


def check_frame(name: str, df: pd.DataFrame) -> list[str]:
    """Raise AssertionError naming every missing required column and every column of the wrong dtype class
    (required and present optional ones). Returns notes: optional columns absent, empty frames, key duplicates
    (duplicates are a note rather than a failure because a producer may legitimately write out-of-fold rows
    twice under different folds; the consumers de-duplicate on their key)."""
    e = get(name)
    assert isinstance(df, pd.DataFrame), "%s: expected a DataFrame, got %s" % (name, type(df).__name__)
    missing = [c for c in e["columns"] if c not in df.columns]
    mistyped = []
    for c, cls in list(e["columns"].items()) + list(e["optional"].items()):
        if c in df.columns and not _dtype_ok(df[c], cls):
            mistyped.append("%s (expected %s, got %s)" % (c, cls, df[c].dtype))
    problems = []
    if missing:
        problems.append("missing columns: %s" % ", ".join(missing))
    if mistyped:
        problems.append("mistyped columns: %s" % "; ".join(mistyped))
    assert not problems, "interface %s (producer %s, consumers %s): %s. Required: %s" % (
        name, e["producer"], ", ".join(e["consumers"]), "; ".join(problems),
        ", ".join("%s:%s" % kv for kv in e["columns"].items()))
    notes = []
    absent = [c for c in e["optional"] if c not in df.columns]
    if absent:
        notes.append("optional columns absent: %s" % ", ".join(absent))
    if len(df) == 0:
        notes.append("empty frame")
    keys = [k for k in e["keys"] if k in df.columns]
    if keys and len(df) and df.duplicated(keys).any():
        notes.append("%d duplicate rows on key %s" % (int(df.duplicated(keys).sum()), keys))
    if name == "coverage_predictions":
        pcols = [c for c in df.columns if c.startswith("p_")]
        if pcols and len(df):
            v = df[pcols].to_numpy(dtype=float)
            if np.nanmin(v) < -1e-9 or np.nanmax(v) > 1 + 1e-9:
                raise AssertionError("coverage_predictions: p_<class> columns outside [0, 1]")
    return notes


def check_file(name: str, path) -> list[str]:
    """check_frame on a parquet file; the path is prefixed to the error."""
    try:
        df = pd.read_parquet(path)
    except Exception as exc:                                              # noqa: BLE001
        raise AssertionError("interface %s: cannot read %s: %s" % (name, path, exc)) from exc
    try:
        return check_frame(name, df)
    except AssertionError as exc:
        raise AssertionError("%s: %s" % (path, exc)) from None


# ── lanes ──────────────────────────────────────────────────────────────────────────────────────────
def lane_dir(ext: str) -> Path:
    return ANALYTICS_DIR / ext


def lane_steps_present(ext: str) -> dict:
    """{step: bool} for validate, features, train of analytics/<ext>/."""
    d = lane_dir(ext)
    return {s: (d / (s + ".py")).is_file() for s in STANDARD_STEPS}


def discover_lanes(analytics_dir=None) -> list[str]:
    """Every analytics/<ext>/ package (an __init__.py beside validate.py, features.py and train.py), sorted."""
    base = Path(analytics_dir) if analytics_dir else ANALYTICS_DIR
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        if (d / "__init__.py").is_file() and all((d / (s + ".py")).is_file() for s in STANDARD_STEPS):
            out.append(d.name)
    return out


def missing_lanes(analytics_dir=None) -> dict:
    """{ext: reason} for every expected lane that discover_lanes does not return."""
    found = set(discover_lanes(analytics_dir))
    out = {}
    for ext in EXPECTED_LANES:
        if ext in found:
            continue
        d = (Path(analytics_dir) if analytics_dir else ANALYTICS_DIR) / ext
        if not d.is_dir():
            out[ext] = "analytics/%s/ does not exist" % ext
            continue
        absent = [f for f in ("__init__.py",) + tuple(s + ".py" for s in STANDARD_STEPS) if not (d / f).is_file()]
        out[ext] = "analytics/%s/ is missing %s" % (ext, ", ".join(absent))
    return out


def fixture_module(ext: str) -> Path | None:
    """tests/_fx_<ext>.py when present, else analytics/fixtures.py when present, else None."""
    for p in (ROOT / "tests" / ("_fx_%s.py" % ext), ANALYTICS_DIR / "fixtures.py"):
        if p.is_file():
            return p
    return None


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def producers_present() -> dict:
    """{interface: bool} whether the producer lane's train module exists."""
    return {n: lane_steps_present(e["producer"])["train"] for n, e in INTERFACES.items()}


__all__ = ["INTERFACES", "INTERFACE_NAMES", "EXPECTED_LANES", "STANDARD_STEPS", "get", "interface_path",
           "find_outputs", "check_frame", "check_file", "discover_lanes", "missing_lanes", "lane_steps_present",
           "fixture_module", "module_available", "producers_present"]

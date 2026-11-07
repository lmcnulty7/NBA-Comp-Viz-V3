"""analytics.registry: the catalogue of extension packages, Phase 0 steps and collectors.

This module is plain data plus two small helpers. It imports nothing from the extension packages,
so it works while the packages are absent or half-written: `available(name)` asks importlib whether a
package can be found, `order_for(names)` sorts selected extensions so every dependency runs first.

Every extension package `analytics.<name>` exposes the same command-line convention, one module per step:

    python -m analytics.<name>.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.<name>.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.<name>.train    --fixtures | --data DATA_ROOT --report reports/<name>.md [--seed 0] [--bootstrap 200]

Two packages carry one extra step between validate and features: `analytics.epv.states` builds the
1 Hz states table (docs/DATA_CONTRACT.md section 20) and `analytics.lineups.stints` builds lineup stints.
Each step module exposes `main(argv=None) -> int`, which run_all.py calls in-process.

`real_data_inputs` are paths relative to DATA_ROOT (collector tables) or to OUT_ROOT for the tables other
lanes write under `analytics/` (Phase 0 tables, states_1hz, epv_states, lineups RAPM, coverage predictions):
the consuming lanes look under OUT_ROOT first, then DATA_ROOT, so the simplest setup is no --out at all
(OUT_ROOT = DATA_ROOT). DATA_ROOT is the parent folder of `sportvu_contract/` (the Step 1
contract dataset, docs/DATA_CONTRACT.md section 3: manifest.json, <game>.parquet, heldout/), of the
collector outputs under `collect/` and of the Phase 0 tables under `analytics/`; on the Mac that is the
Drive nba_harvest folder. `<game>` is a SportVU game code such as 12.16.2015.PHX.at.GSW, `<season>` is 2015-16.
"""
from __future__ import annotations

import importlib.util
from typing import Iterable

PHASES = {
    0: "freeze and reframe: clock gaps, rosters, shot and rebound frames, possessions",
    1: "location-only models on the Phase 0 tables",
    2: "tracking geometry: matchups, gravity, ghosting, player embeddings",
    3: "possession state: EPV states, action recognition, defensive coverage",
    4: "decisions on top of the state models, play calling, lineup stints",
    5: "product and reattachment: win probability, load, betting",
}

# Paths that recur in real_data_inputs (relative to DATA_ROOT).
_FRAMES = "sportvu_contract/<game>.parquet"
_PBP = "sportvu_contract/<game>_pbp.parquet"
_MANIFEST = "sportvu_contract/manifest.json"
_ROSTERS = "analytics/rosters/<season>/<game>.parquet"
_SHOT_FRAMES = "analytics/shot_frames/<season>/<game>.parquet"
_REBOUND_FRAMES = "analytics/rebound_frames/<season>/<game>.parquet"
_POSSESSIONS = "analytics/possessions/<season>/<game>.parquet"
_STATES = "analytics/states_1hz/<season>/<game>.parquet"          # written by analytics.epv.states (OUT_ROOT)
_EPV_STATES = "analytics/epv_states/<season>/<game>.parquet"      # written by analytics.epv (OUT_ROOT)
_RAPM = "analytics/lineups/<season>/player_rapm*.parquet"         # written by analytics.lineups (OUT_ROOT)
_COVERAGE_PRED = "analytics/coverage/<season>/coverage_predictions/<game>.parquet"
_BREF_SHOTS = "collect/bref_shots/shotchart/<season>.parquet"
_NBA_SHOTS = "collect/shots/shotchartdetail/<season>.parquet"
# bref_boxscores/{starters,players}/ and bref_seasons/{shotchart,playbyplay}/ are compacted by the collectors
# into collect/stints/, which is what the lanes read; bref_schedule writes collect/bref_schedule/games/.
_STINTS = "collect/stints/stints/<season>.parquet"
_MULTI_PBP = "collect/stints/playbyplay/<season>.parquet"
_STINT_GAMES = "collect/stints/games/<season>.parquet"
_STINT_STARTERS = "collect/stints/starters/<season>.parquet"
_POSITIONS = "collect/positions.parquet"
_TRACKING_POSITIONS = "collect/tracking/positions/<season>.parquet"
_TRACKING_AGG = "collect/tracking/aggregates/<season>.parquet"
_ODDS = "collect/extras/odds/<season>.parquet"
_INJURIES = "collect/extras/injuries/<season>.parquet"

STANDARD_STEPS = ["validate", "features", "train"]


def _ext(name, steps, depends_on, phase, inputs, notes):
    return {"name": name, "package": "analytics." + name, "steps": list(steps),
            "depends_on": list(depends_on), "phase": phase,
            "real_data_inputs": list(inputs), "notes": notes}


EXTENSIONS = [
    _ext("xfg", STANDARD_STEPS, [], 1,
         [_BREF_SHOTS, _NBA_SHOTS, _SHOT_FRAMES, _MANIFEST],
         "expected field goal: location plus tracking features on top of analytics.train_xfg; "
         "report reports/xfg.md; its shot-quality column feeds matchups and decisions"),
    _ext("rebounds", STANDARD_STEPS, [], 1,
         [_REBOUND_FRAMES, _SHOT_FRAMES, _FRAMES, _ROSTERS],
         "rebound probability per player at the rebound frame; position and box-out geometry"),
    _ext("matchups", STANDARD_STEPS, ["xfg"], 2,
         [_FRAMES, _POSSESSIONS, _ROSTERS, _SHOT_FRAMES],
         "defender assignment per frame (Hungarian on tracked positions); defender credit uses xfg"),
    _ext("gravity", STANDARD_STEPS, [], 2,
         [_FRAMES, _POSSESSIONS, _SHOT_FRAMES, _ROSTERS],
         "offensive gravity: defender displacement attributable to each off-ball player"),
    _ext("ghosting", STANDARD_STEPS, [], 2,
         [_FRAMES, _POSSESSIONS, _ROSTERS],
         "ghost defence: where a league-average defender would stand, versus where the real one stood"),
    _ext("embeddings", STANDARD_STEPS, [], 2,
         [_FRAMES, _POSSESSIONS, _ROSTERS, _POSITIONS, _TRACKING_POSITIONS],
         "player and lineup embeddings from movement; the coursework foundation-model angle"),
    _ext("epv", ["validate", "states", "features", "train"], [], 3,
         [_FRAMES, _PBP, _POSSESSIONS, _ROSTERS, _STATES],
         "expected possession value; analytics.epv.states writes the 1 Hz states table "
         "(docs/DATA_CONTRACT.md section 20) that features and the later phases read"),
    _ext("actions", STANDARD_STEPS, [], 3,
         [_FRAMES, _PBP, _POSSESSIONS, _ROSTERS],
         "action recognition from tracking: screens, cuts, drives, handoffs per possession"),
    _ext("coverage", STANDARD_STEPS, ["actions"], 3,
         [_FRAMES, _POSSESSIONS, _ROSTERS],
         "defensive coverage classification (switch, drop, hedge) on the screens actions found"),
    _ext("decisions", STANDARD_STEPS, ["xfg", "epv"], 4,
         [_STATES, _SHOT_FRAMES, _PBP, _POSSESSIONS],
         "shot versus pass decisions scored against EPV and xfg at the moment of the choice"),
    _ext("playcalling", STANDARD_STEPS, ["epv", "actions", "coverage"], 4,
         [_EPV_STATES, _POSSESSIONS, _PBP, _COVERAGE_PRED],
         "play-type recognition and value from the action sequences, coverage predictions and EPV traces"),
    _ext("lineups", ["validate", "stints", "features", "train"], [], 4,
         [_STINTS, _MULTI_PBP, _STINT_GAMES, _STINT_STARTERS],
         "lineup stints and adjusted plus-minus; analytics.lineups.stints builds the stint table first"),
    _ext("winprob", STANDARD_STEPS, [], 5,
         [_MULTI_PBP, _STINT_GAMES, _ODDS],
         "in-game win probability from score, clock and possession; calibration on held-out seasons"),
    _ext("load", STANDARD_STEPS, [], 5,
         [_TRACKING_AGG, _INJURIES, "sportvu_contract/", _ROSTERS],
         "player load: distance, speed and minutes against rest days and injury reports"),
    _ext("betting", STANDARD_STEPS, ["lineups"], 5,
         [_ODDS, _STINT_GAMES, _MULTI_PBP, _RAPM],
         "closing-line comparison of lineup-adjusted ratings; odds only if the odds table was pursued"),
]

EXTENSION_NAMES = [e["name"] for e in EXTENSIONS]

# Phase 0, run before any extension. argv templates are formatted with contract (DATA_ROOT/sportvu_contract,
# or DATA_ROOT itself when it holds manifest.json), data_root, out_root, report_dir and steps.
PHASE0_STEPS = [
    {"name": "gaps", "module": "analytics.gaps",
     "argv": ["--manifest", "{contract}/manifest.json", "--out", "{report_dir}/sportvu_gaps.md"],
     "notes": "clock gaps per game from the Step 1 manifest; flags games over the threshold"},
    {"name": "run_phase0", "module": "analytics.run_phase0",
     "argv": ["--data", "{contract}", "--out", "{out_root}", "--steps", "{steps}"],
     "notes": "rosters, shot frames, rebound frames and possessions for every game of the manifest "
              "(--data is the contract folder; gaps is left out because it already ran)"},
]

_NBA_BLOCKED = ("stats.nba.com refuses non-browser clients from cloud and many home networks; "
                "run from a browser session or take the tables from the Drive contract")

COLLECTORS = [
    {"name": "drive", "module": "collect.drive", "reachable": True, "reason": "local Google Drive folder, no network",
     "command": "python -m collect.drive --sync-from [DATA_ROOT] [--only bref_shots ...] [--dry-run]",
     "notes": "copies DATA_ROOT/collect (default: the Drive nba_harvest folder) to the local data/collect; "
              "--sync-to copies local data/collect to DATA_ROOT/collect; --where prints the resolved Drive root. "
              "The Step 1 contract itself stays at DATA_ROOT/sportvu_contract"},
    {"name": "bref_schedule", "module": "collect.bref_schedule", "reachable": True, "reason": "basketball-reference, 17 requests a minute",
     "command": "python -m collect.bref_schedule --seasons 2015-16 --out DATA_ROOT/collect",
     "notes": "schedule and game codes per season; one page per month"},
    {"name": "bref_seasons", "module": "collect.bref_seasons", "reachable": True, "reason": "basketball-reference, 17 requests a minute",
     "command": "python -m collect.bref_seasons --seasons 2015-16 --out DATA_ROOT/collect",
     "notes": "season totals and advanced tables per team and player"},
    {"name": "bref_boxscores", "module": "collect.bref_boxscores", "reachable": True, "reason": "basketball-reference, 17 requests a minute",
     "command": "python -m collect.bref_boxscores --seasons 2015-16 --out DATA_ROOT/collect",
     "notes": "one box score page per game (1,230 a season, about 72 minutes at 17 a minute)"},
    {"name": "bref_lineups", "module": "collect.bref_lineups", "reachable": True, "reason": "basketball-reference, 17 requests a minute",
     "command": "python -m collect.bref_lineups --seasons 2015-16 --out DATA_ROOT/collect",
     "notes": "five-man lineup tables per team"},
    {"name": "positions", "module": "collect.positions", "reachable": True, "reason": "basketball-reference, 17 requests a minute",
     "command": "python -m collect.positions --seasons 2015-16 --out DATA_ROOT/collect",
     "notes": "listed positions and heights per player-season"},
    {"name": "bref_shots", "module": "collect.bref_shots", "reachable": True, "reason": "basketball-reference, 17 requests a minute",
     "command": "python -m collect.bref_shots --contract-dir DATA_ROOT [--fetch-pbp]",
     "notes": "shot charts per game; the only shot source that is reachable without a browser"},
    {"name": "pbp", "module": "collect.pbp", "reachable": True, "reason": "basketball-reference fallback (--br-only); the NBA.com path is blocked",
     "command": "python -m collect.pbp --contract-dir DATA_ROOT --br-only",
     "notes": "play-by-play for games whose Step 1 join failed"},
    {"name": "shots", "module": "collect.shots", "reachable": False, "reason": _NBA_BLOCKED,
     "command": "python -m collect.shots --seasons 2015-16", "notes": "shotchartdetail per team-season"},
    {"name": "tracking", "module": "collect.tracking", "reachable": False, "reason": _NBA_BLOCKED,
     "command": "python -m collect.tracking --seasons 2015-16", "notes": "player tracking aggregates"},
    {"name": "context", "module": "collect.context", "reachable": False, "reason": _NBA_BLOCKED,
     "command": "python -m collect.context --seasons 2015-16", "notes": "synergy, hustle and matchup aggregates"},
    {"name": "stints", "module": "collect.stints", "reachable": False, "reason": _NBA_BLOCKED,
     "command": "python -m collect.stints --seasons 2015-16", "notes": "rotations and multi-season play-by-play"},
    {"name": "extras", "module": "collect.extras", "reachable": False, "reason": "odds need ODDS_API_KEY; injuries come from a downloaded CSV",
     "command": "python -m collect.extras --injuries FILE.csv", "notes": "optional odds and injury tables"},
    {"name": "probe", "module": "collect.probe", "reachable": False, "reason": _NBA_BLOCKED,
     "command": "python -m collect.probe", "notes": "endpoint probe; diagnostic only"},
]

BREF_RPM_NOTE = ("basketball-reference allows 17 requests a minute; every bref collector throttles itself "
                 "and backs off on the first 429. Do not run two bref collectors at once.")


def get(name: str) -> dict:
    """The registry entry for an extension, or KeyError naming the known extensions."""
    for e in EXTENSIONS:
        if e["name"] == name:
            return e
    raise KeyError("unknown extension %r; known: %s" % (name, ", ".join(EXTENSION_NAMES)))


def available(name: str, extensions: Iterable[dict] | None = None) -> bool:
    """True when the extension's package can be found by importlib (without importing it)."""
    entries = list(extensions) if extensions is not None else EXTENSIONS
    package = None
    for e in entries:
        if e["name"] == name:
            package = e["package"]
            break
    if package is None:
        package = "analytics." + name
    return module_available(package)


def module_available(module: str) -> bool:
    """True when `module` resolves with importlib.util.find_spec; False on any lookup failure."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def order_for(names: Iterable[str], extensions: Iterable[dict] | None = None) -> list[str]:
    """The selected names in an order where every selected dependency comes before its dependents.

    Ties keep registry order, so the result is stable. Dependencies that are not selected are not
    added (run_all prints a note about them). Raises ValueError on an unknown name or a cycle."""
    entries = list(extensions) if extensions is not None else EXTENSIONS
    by_name = {e["name"]: e for e in entries}
    selected = []
    for n in names:
        if n not in by_name:
            raise ValueError("unknown extension %r; known: %s" % (n, ", ".join(by_name)))
        if n not in selected:
            selected.append(n)
    registry_pos = {e["name"]: i for i, e in enumerate(entries)}
    selected.sort(key=registry_pos.__getitem__)
    chosen = set(selected)
    remaining = list(selected)
    out: list[str] = []
    while remaining:
        progressed = False
        for n in list(remaining):
            deps = [d for d in by_name[n].get("depends_on", []) if d in chosen]
            if all(d in out for d in deps):
                out.append(n)
                remaining.remove(n)
                progressed = True
                break
        if not progressed:
            raise ValueError("dependency cycle among: " + ", ".join(remaining))
    return out


def missing_dependencies(names: Iterable[str], extensions: Iterable[dict] | None = None) -> dict[str, list[str]]:
    """For each selected name, the dependencies that are not themselves selected."""
    entries = list(extensions) if extensions is not None else EXTENSIONS
    by_name = {e["name"]: e for e in entries}
    chosen = set(names)
    out = {}
    for n in chosen:
        miss = [d for d in by_name[n].get("depends_on", []) if d not in chosen]
        if miss:
            out[n] = miss
    return out


def with_dependencies(names: Iterable[str], extensions: Iterable[dict] | None = None) -> list[str]:
    """The selected names plus every transitive dependency, in dependency order."""
    entries = list(extensions) if extensions is not None else EXTENSIONS
    by_name = {e["name"]: e for e in entries}
    todo = list(names)
    seen: list[str] = []
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        if n not in by_name:
            raise ValueError("unknown extension %r; known: %s" % (n, ", ".join(by_name)))
        seen.append(n)
        todo.extend(by_name[n].get("depends_on", []))
    return order_for(seen, entries)

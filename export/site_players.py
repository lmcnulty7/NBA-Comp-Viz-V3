"""export/site_players.py: players.json, the player views of the prep tab (PLAN.md D47, D51, C9).

Every value comes from training games only. A lane table that carries a fold column is cut to its out-of-fold rows
(fold >= 0) and the cut is checked in code: a held-out row (fold -1), an in-sample row (fold -2), a row flagged held
out or a held-out game id stops the build. The values are sums and shares of stored rows and stored predictions.
Intervals resample the player's own games (export.site_common.bootstrap_by_game). No model runs here.

Definitions (PLAN.md D47 and D51, fixed before any player value was computed):
  shot-making   makes above expected per 100 attempts, expected = the lane's location and contest candidate
                (p_contest), which does not know the shooter and was not the selected model. Its out-of-fold
                calibration table is read from reports/xfg.md and travels as typed data; each player also gets the
                part of his value that this table explains for his mix of shots. An attempt counts for a player
                only when the tracked attempt (shot_frames) names him as the shooter resolved from the play-by-play.
                The others are left out and counted by reason: the game has no shot_frames table, the table has no
                row for the event, or the name match did not resolve the shooter (for those the lane stored whoever
                stood nearest the ball). fg_pct is descriptive.
  gravity       the lane's stored player table and its own 300-possession gate (descriptive).
  rebounding    rebounds per 100 chances (descriptive), with the selected model's out-of-fold expectation beside it.
  absence       the share of games after which the player missed his team's next tracked game (descriptive), and
                the mean out-of-fold probability of the frozen logistic when a rebuilt copy passed its check.
  similar       the lane's nearest neighbours (descriptive), when a rebuilt embedding passed its check.
A rebuilt output is used only when its check passed, vouches for the files read, and equals its tracked copy in
reports/export/frozen_checks.json (PLAN.md D49). Otherwise the view falls back to what the frozen run stored.
Only an interval-carrying value is typed as an estimate by the contract; the plain numbers beside it (fg_pct,
expected_fg_pct, baseline_error_per_100, frozen_model_residual_per_100, model_expected_per_100) are point estimates
and the metric notes say so.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analytics import results as R
from analytics.eval import is_heldout_game
from analytics.possessions import game_meta
from collect.positions import join_rosters
from export import schemas as S
from export import site_common as C
from export import site_reports
from tools.frozen_common import require_passed

ExportProblem = C.ExportProblem

MIN_ATTEMPTS = 100
MIN_GRAVITY_POSSESSIONS = 300
MIN_REBOUND_CHANCES = 300
MIN_ABSENCE_GAMES = 10
MIN_SIMILAR_POSSESSIONS = 300
N_NEIGHBOURS = 5
BASELINE = "contest_logistic"              # the lane's location and contest candidate (reports/xfg.md)
XFG_REPORT = "reports/xfg.md"
ABSENCE_ARTIFACT, ABSENCE_FILE = "absence", "oof.parquet"
EMBEDDINGS_ARTIFACT = "embeddings"
ABSENCE_PROBABILITY_COLUMNS = ("p_logistic", "p_selected")
NEIGHBOURS_FILE = "embedding_neighbours.parquet"
FEATURES_FILE = "player_season_features.parquet"
CHECK_UNITS = {"absence": "log loss", "embeddings": "accuracy"}
CHECKS_COPY = "reports/export/frozen_checks.json"
# What the tracked copy of a check record must repeat from the artifact's own check.json to vouch for it.
TRACKED_FIELDS = ("name", "compared", "n", "n_expected", "max_abs_diff", "tolerance", "written")
# Why an out-of-fold attempt counts for no player, in the order the note states them.
LEFT_OUT_REASONS = (
    ("no_shot_frames_table", "belong to {games} training game(s) with no shot_frames table, so there is no tracked "
                             "attempt to join"),
    ("no_shot_frames_row", "have no row in their game's shot_frames table"),
    ("shooter_not_resolved", "have a tracked attempt whose shooter the play-by-play name match did not resolve"),
    ("another_shooter", "have a tracked attempt that names another shooter than the stored prediction"),
)

# The embeddings lane's own collapse of a listed position to G, F or C (analytics/embeddings/train.py: POSITION_MAP
# and collapse_position, the part before the first hyphen decides). export/ may not import a lane's train module,
# so the mapping is repeated here and a test keeps it equal to the lane's.
POSITION_MAP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}

# metric -> the lane whose verdict it shows
METRIC_LANES = {"shot_making": "xfg", "gravity": "gravity", "rebounding": "rebounds", "absence_risk": "load",
                "similar": "embeddings"}


@dataclass
class PlayersInfo:
    """What the run reports about players.json."""

    n_players: int = 0
    n_shown: int = 0
    n_hidden: int = 0
    shown_by_metric: dict = field(default_factory=dict)
    attempts_total: int = 0
    attempts_left_out: int = 0
    attempts_left_out_by_reason: dict = field(default_factory=dict)
    games_with_attempts: int = 0
    games_without_shot_frames: int = 0
    absence: str = ""
    similar: str = ""
    training_games: int = 0


# ------------------------------------------------------------------------------------------------- guards


def training_rows(table: pd.DataFrame, what: str) -> pd.DataFrame:
    """The out-of-fold rows of a lane table (fold >= 0). ExportProblem when the cut lets through a row that is
    flagged held out or belongs to a held-out game, or when nothing is left."""
    if "fold" not in table.columns:
        raise ExportProblem(f"{what}: no fold column, so out-of-fold rows cannot be told apart")
    fold = pd.to_numeric(table["fold"], errors="coerce")
    kept = table[(fold >= 0).to_numpy()]
    if len(kept) == 0:
        raise ExportProblem(f"{what}: no out-of-fold row (fold >= 0)")
    if not pd.to_numeric(kept["fold"]).ge(0).all():
        raise ExportProblem(f"{what}: a row with fold below 0 passed the out-of-fold cut")
    return training_games(kept, what)


def training_games(table: pd.DataFrame, what: str) -> pd.DataFrame:
    """The table, after checking that no row is flagged held out and no game id is a held-out game."""
    if "heldout" in table.columns and table["heldout"].fillna(False).astype(bool).any():
        raise ExportProblem(f"{what}: a row flagged held out would feed a player value")
    if "sportvu_game" in table.columns:
        held = sorted(g for g in table["sportvu_game"].astype(str).unique() if is_heldout_game(g))
        if held:
            raise ExportProblem(f"{what}: held-out game(s) {held[:3]} would feed a player value")
    return table


def _training_files(folder: Path, what: str) -> list:
    """The per-game files of a training folder, each named by a game id that is not held out."""
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    if not files:
        raise ExportProblem(f"no files under {what}")
    held = [p.stem for p in files if is_heldout_game(p.stem)]
    if held:
        raise ExportProblem(f"{what} holds held-out game(s) {held[:3]}")
    return files


def _concat(files: list, columns: list, what: str) -> pd.DataFrame:
    try:
        return pd.concat([pq.read_table(p, columns=columns).to_pandas() for p in files], ignore_index=True)
    except Exception as exc:  # the message may hold a local path, so it is not repeated
        raise ExportProblem(f"cannot read {what} ({type(exc).__name__}); columns asked: {columns}") from None


# ------------------------------------------------------------------------------------------------- baseline


def read_baseline(repo_root=C.REPO_ROOT):
    """(the contract's Baseline, gap per calibration bin, attempts per bin) of the shot-making expectation, read
    from reports/xfg.md: the candidate table gives its out-of-fold calibration error and says it was not selected,
    the calibration section gives its out-of-fold table. gap[k] = observed minus predicted in bin k, NaN where the
    bin is empty."""
    path = Path(repo_root) / XFG_REPORT
    if not path.exists():
        raise ExportProblem(f"{XFG_REPORT} is missing")
    doc = R.Doc(path.read_text(encoding="utf-8"))
    try:
        section = doc.need("Candidate table", level=2)
        table = section.table("candidate", "ECE (10 bins)", "selected")
        row = table.one("candidate", lambda c: R.clean(c) == BASELINE, f"candidate {BASELINE}")
        ece = R.metric("ECE (10 bins)", table.cell(row, "ECE (10 bins)"))
        selected = R.clean(table.cell(row, "selected")).lower() == "yes"
        method = doc.search(r"Intervals: ([^.\n]+)\.", "the sentence that names the interval method").group(1)
        calibration = doc.need("Calibration", level=2).find(BASELINE, level=3)
        if calibration is None or "out-of-fold" not in calibration.title:
            raise R.ParseError(f"no out-of-fold calibration table of {BASELINE}")
        bins = calibration.table("lo", "hi", "n", "mean_p", "mean_y")
    except R.ParseError as exc:
        raise ExportProblem(f"{XFG_REPORT}: {exc}") from None
    attempts = re.search(r"(\d+) attempts", section.title)
    ece_lo, ece_hi = site_reports.interval_of(ece)
    if ece_lo is None or not attempts:
        raise ExportProblem(f"{XFG_REPORT}: the calibration error of {BASELINE} has no interval or no attempt count")
    rows, gaps = [], []
    for k, line in enumerate(bins.rows):
        lo, hi, n = float(bins.cell(line, "lo")), float(bins.cell(line, "hi")), int(bins.cell(line, "n"))
        if abs(lo - k / len(bins.rows)) > 1e-9 or abs(hi - (k + 1) / len(bins.rows)) > 1e-9:
            raise ExportProblem(f"{XFG_REPORT}: the calibration table of {BASELINE} is not in equal-width bins")
        if n == 0:
            gaps.append(float("nan"))
            continue
        predicted, observed = float(bins.cell(line, "mean_p")), float(bins.cell(line, "mean_y"))
        rows.append(S.CalibrationRow(lo=lo, hi=hi, n=n, predicted=predicted, observed=observed))
        gaps.append(observed - predicted)
    baseline = S.Baseline(
        name=BASELINE, selected=selected,
        ece=S.Estimate(value=float(ece.value), lo=ece_lo, hi=ece_hi, interval=method,
                       point_estimate=False, n=int(attempts.group(1)), n_unit="attempts"),
        calibration=rows,
        source=C.source(XFG_REPORT, f"candidate table and out-of-fold calibration table of {BASELINE}", "out_of_fold"))
    return baseline, np.array(gaps), [int(bins.cell(line, "n")) for line in bins.rows]


def calibration_bin(p, n_bins: int) -> np.ndarray:
    """The equal-width bin of each probability, as the lane's calibration table counts it (analytics.eval)."""
    return np.clip(np.floor(np.asarray(p, dtype=float) * n_bins), 0, n_bins - 1).astype(int)


# ------------------------------------------------------------------------------------------------- tables


def _rosters(data_root: Path):
    """(per-player frame, all training roster rows, number of training games, first and last day). The per-player
    frame is indexed by player_id: name, teams in date order, games."""
    folder = data_root / "analytics" / "rosters" / C.SEASON
    files = _training_files(folder, C.data_path("analytics/rosters", C.SEASON))
    rows = _concat(files, ["sportvu_game", "player_id", "player", "team", "bref_slug"], "training rosters")
    rows = training_games(rows, "training rosters")
    real = rows[rows["player_id"] > 0].copy()            # id 0 is a nameless placeholder in some frames
    if real["player"].isna().any():
        raise ExportProblem("training rosters: a player id above 0 has no name")
    if real.groupby("player_id")["player"].nunique().gt(1).any():
        raise ExportProblem("training rosters: a player id carries two names")
    real["date"] = [game_meta(g)["date"] for g in real["sportvu_game"].astype(str)]
    real = real.sort_values(["player_id", "date", "sportvu_game"], kind="stable")
    players = real.groupby("player_id").agg(
        name=("player", "first"), games=("sportvu_game", "nunique"),
        teams=("team", lambda s: list(dict.fromkeys(s.astype(str)))))
    dates = sorted(set(real["date"]))
    return players, rows, len(files), (dates[0], dates[-1])


def collapse_position(value):
    """A listed position (PG, SF-SG, C-PF) as G, F or C by the part before the first hyphen; None when unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return POSITION_MAP.get(str(value).strip().upper().split("-")[0])


def _positions(data_root: Path, roster_rows: pd.DataFrame) -> dict:
    """player_id -> G, F or C: the listed position of the season, joined to player ids through bref_slug
    (collect.positions.join_rosters) and collapsed with the embeddings lane's mapping."""
    table = C.read_table(data_root / "collect" / "positions.parquet", what=C.data_path("collect/positions.parquet"))
    if "games" not in table.columns:
        table = table.assign(games=0)
    joined = join_rosters(table, roster_rows[["sportvu_game", "player_id", "bref_slug"]].drop_duplicates())
    joined = joined[joined["season"].astype(str) == C.SEASON]
    out = {}
    for player_id, position in zip(joined["player_id"], joined["position"]):
        letter = collapse_position(position)
        if letter is not None and int(player_id) > 0:
            out.setdefault(int(player_id), letter)
    return out


def _shot_making(data_root: Path, gaps: np.ndarray, report_counts: list):
    """(per-player frame, counts). Per player: attempts, makes over the contest candidate per 100 with its
    interval, make share, expected share, the part the candidate's own calibration error explains for his shot mix,
    and the frozen model's residual per 100. The counts say how many out-of-fold attempts there are, from how many
    games, and how many count for no player, by reason (LEFT_OUT_REASONS)."""
    path = data_root / "analytics" / "xfg" / C.SEASON / "xfg_predictions.parquet"
    cols = ["sportvu_game", "event", "player_id", "p_contest", "p_selected", "made", "fold", "heldout"]
    rows = training_rows(C.read_table(path, cols, C.data_path("analytics/xfg", C.SEASON, "xfg_predictions.parquet")),
                         "xfg_predictions")
    if rows["p_contest"].isna().any() or rows["p_selected"].isna().any():
        raise ExportProblem("xfg_predictions: an out-of-fold attempt has no prediction")
    rows = rows.assign(bin=calibration_bin(rows["p_contest"], len(gaps)))
    counts = np.bincount(rows["bin"], minlength=len(gaps)).tolist()
    if counts != list(report_counts):
        raise ExportProblem(f"xfg_predictions: the out-of-fold rows fall in the calibration bins as {counts}, "
                            f"{XFG_REPORT} counts {list(report_counts)}: the stored predictions are not the report's")
    folder = data_root / "analytics" / "shot_frames" / C.SEASON
    files = _training_files(folder, C.data_path("analytics/shot_frames", C.SEASON))
    located = training_games(_concat(files, ["sportvu_game", "event", "shooter_id"], "training shot_frames"),
                             "training shot_frames")
    if located.duplicated(["sportvu_game", "event"]).any():
        raise ExportProblem("training shot_frames: an event is listed twice")
    merged = rows.merge(located, on=["sportvu_game", "event"], how="left", indicator="_joined")
    same = (merged["player_id"].astype("Float64") == merged["shooter_id"].astype("Float64")).fillna(False).astype(bool)
    same = same.to_numpy()
    attempt_games = merged["sportvu_game"].astype(str)
    no_table = ~attempt_games.isin({p.stem for p in files}).to_numpy()
    no_row = (merged["_joined"] == "left_only").to_numpy() & ~no_table
    no_shooter = merged["shooter_id"].isna().to_numpy() & ~no_table & ~no_row
    by_reason = {"no_shot_frames_table": int(no_table.sum()), "no_shot_frames_row": int(no_row.sum()),
                 "shooter_not_resolved": int(no_shooter.sum()),
                 "another_shooter": int((~same & ~no_table & ~no_row & ~no_shooter).sum())}
    if (same & (no_table | no_row | no_shooter)).any() or sum(by_reason.values()) != int((~same).sum()):
        raise ExportProblem("xfg_predictions: the attempts left out of shot-making do not split by reason")
    counts = {"total": int(len(rows)), "left_out": int((~same).sum()), "by_reason": by_reason,
              "games": int(attempt_games.nunique()), "games_without_table": int(attempt_games[no_table].nunique())}
    kept = merged[same].drop(columns="_joined").copy()
    kept["player_id"] = kept["player_id"].astype("int64")
    kept["made"] = kept["made"].astype(float)
    kept["p_contest"] = kept["p_contest"].astype(float)
    kept["over"] = kept["made"] - kept["p_contest"]
    kept["over_frozen"] = kept["made"] - kept["p_selected"].astype(float)
    kept["gap"] = gaps[kept["bin"].to_numpy()]
    if kept["gap"].isna().any():
        raise ExportProblem(f"xfg_predictions: an attempt falls in a calibration bin that {XFG_REPORT} leaves empty")
    per_game = kept.groupby(["player_id", "sportvu_game"], as_index=False).agg(
        n=("made", "size"), makes=("made", "sum"), over=("over", "sum"), over_frozen=("over_frozen", "sum"),
        expected=("p_contest", "sum"), gap=("gap", "sum"))
    out = C.bootstrap_by_game(per_game, "over", "n", scale=100.0)
    totals = per_game.groupby("player_id")[["n", "makes", "over_frozen", "expected", "gap"]].sum()
    out["fg_pct"] = totals["makes"] / totals["n"]
    out["expected_fg_pct"] = totals["expected"] / totals["n"]
    out["baseline_error"] = 100.0 * totals["gap"] / totals["n"]
    out["frozen_residual"] = 100.0 * totals["over_frozen"] / totals["n"]
    return out, counts


def _gravity(data_root: Path, players: pd.DataFrame) -> pd.DataFrame:
    path = data_root / "analytics" / "gravity" / "gravity_players.parquet"
    table = training_games(C.read_table(path, what=C.data_path("analytics/gravity/gravity_players.parquet")),
                           "gravity_players")
    need = ["player_id", "n_possessions", "n_games", "gravity_ft", "lo", "hi", "insufficient"]
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ExportProblem(f"gravity_players lacks {missing}")
    table = table[table["player_id"] > 0].set_index("player_id")
    if table.index.duplicated().any():
        raise ExportProblem("gravity_players: a player is listed twice")
    gate = table["n_possessions"] < MIN_GRAVITY_POSSESSIONS
    if (gate != table["insufficient"].astype(bool)).any():
        raise ExportProblem("gravity_players: the stored gate is not the 300-possession gate")
    # The table is built from training games only. A count above the player's training games would say otherwise.
    games = players["games"].reindex(table.index)
    if games.isna().any() or (table["n_games"] > games).any():
        raise ExportProblem("gravity_players: a player has more games than he has training games")
    return table


def _rebounding(data_root: Path) -> pd.DataFrame:
    folder = data_root / "analytics" / "rebounds" / C.SEASON
    files = _training_files(folder, C.data_path("analytics/rebounds", C.SEASON))
    cols = ["sportvu_game", "candidate_player_id", "p_selected", "is_rebounder", "fold", "heldout"]
    rows = training_rows(_concat(files, cols, "training rebound predictions"), "rebound predictions")
    rows = rows[rows["candidate_player_id"] > 0]          # -1 is the team rebound row, 0 a nameless placeholder
    if rows["p_selected"].isna().any():
        raise ExportProblem("rebound predictions: an out-of-fold row has no prediction")
    rows = rows.assign(player_id=rows["candidate_player_id"].astype("int64"), got=rows["is_rebounder"].astype(float),
                       expected=rows["p_selected"].astype(float))
    per_game = rows.groupby(["player_id", "sportvu_game"], as_index=False).agg(
        n=("got", "size"), got=("got", "sum"), expected=("expected", "sum"))
    out = C.bootstrap_by_game(per_game, "got", "n", scale=100.0)
    totals = per_game.groupby("player_id")[["n", "expected"]].sum()
    out["expected_per_100"] = 100.0 * totals["expected"] / totals["n"]
    return out


def _absence_rows(data_root: Path) -> pd.DataFrame:
    """One row per training player-game with a known absent_next."""
    path = data_root / "analytics" / "load" / C.SEASON / "player_games.parquet"
    table = training_games(C.read_table(path, ["sportvu_game", "player_id", "absent_next", "heldout"],
                                        C.data_path("analytics/load", C.SEASON, "player_games.parquet")),
                           "load player_games")
    known = table[(table["player_id"] > 0) & table["absent_next"].notna()].copy()
    if known.duplicated(["player_id", "sportvu_game"]).any():
        raise ExportProblem("load player_games: a player-game is listed twice")
    known["player_id"] = known["player_id"].astype("int64")
    known["absent"] = known["absent_next"].astype(bool).astype(float)
    known["one"] = 1.0
    return known


def _tracked_check(checks_copy: Path, name: str, record: dict) -> str:
    """Why the tracked copy of the check records does not vouch for this artifact's check ("" when it does). The
    artifacts live outside git, so a value built on one traces to a tracked file only through this copy
    (PLAN.md D49): it must exist, hold one passed record for the artifact, and repeat the artifact's own check.json
    in TRACKED_FIELDS."""
    if not Path(checks_copy).exists():
        return (f"{CHECKS_COPY} is missing, so the check of {name} has no tracked copy (PLAN.md D49): run "
                "python3 -m tools.frozen_common and commit the file")
    try:
        entries = [e for e in json.loads(Path(checks_copy).read_text(encoding="utf-8"))["checks"] if e.get("name") == name]
    except (ValueError, KeyError, TypeError, AttributeError):
        return f"{CHECKS_COPY} cannot be read as a list of check records"
    if len(entries) != 1:
        return f"{CHECKS_COPY} holds {len(entries)} records of {name}, not one"
    entry = entries[0]
    if entry.get("passed") is not True or entry.get("reasons"):
        return f"{CHECKS_COPY} does not say that the {name} rebuild passed"
    differs = [key for key in TRACKED_FIELDS if entry.get(key) != record.get(key)]
    if differs:
        return (f"{CHECKS_COPY}: the record of {name} is not the artifact's own check (it differs in {differs}): "
                "run python3 -m tools.frozen_common again and commit the file")
    return ""


def _passed_check(artifacts: Path, name: str, files: tuple, checks_copy: Path):
    """(check record, why not). A rebuilt output is used only when its check passed, vouches for the files that
    are read, and has a tracked copy that equals it (PLAN.md D45, D49). A passed check that does not cover the
    files the export needs is a problem; a check with no matching tracked copy leaves the output unused."""
    try:
        record = require_passed(artifacts / name, name)
    except RuntimeError as exc:
        return None, str(exc)
    missing = [f for f in files if f not in (record.get("files") or {})]
    if missing:
        raise ExportProblem(f"artifacts/{name}: the check passed but does not vouch for {missing}; it vouches for "
                            f"{sorted(record.get('files') or {})}")
    why = _tracked_check(checks_copy, name, record)
    return (None, why) if why else (record, "")


def _rebuild_check(record: dict, name: str) -> S.RebuildCheck:
    try:
        return S.RebuildCheck(
            name=name, compared=str(record["compared"]), n=int(record["n"]), n_expected=record.get("n_expected"),
            max_abs_diff=float(record["max_abs_diff"]), tolerance=float(record["tolerance"]),
            passed=bool(record["passed"]), unit=str(record.get("unit") or CHECK_UNITS[name]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportProblem(f"artifacts/{name}/check.json cannot be shown as a check record: {exc}") from None


def _absence_model(artifacts: Path, known: pd.DataFrame, checks_copy: Path):
    """(per-player frame or None, check record or None, why not). The out-of-fold probabilities of the frozen
    logistic, rebuilt outside export/ (PLAN.md D45), are used only when the rebuild's check passed and is tracked."""
    record, why = _passed_check(artifacts, ABSENCE_ARTIFACT, (ABSENCE_FILE,), checks_copy)
    if record is None:
        return None, None, why
    name = f"artifacts/{ABSENCE_ARTIFACT}/{ABSENCE_FILE}"
    table = C.read_table(artifacts / ABSENCE_ARTIFACT / ABSENCE_FILE, what=name)
    column = next((c for c in ABSENCE_PROBABILITY_COLUMNS if c in table.columns), None)
    if column is None or not {"sportvu_game", "player_id"} <= set(table.columns):
        raise ExportProblem(f"{name} needs sportvu_game, player_id and one of {list(ABSENCE_PROBABILITY_COLUMNS)}")
    if "fold" in table.columns:
        if pd.to_numeric(table["fold"], errors="coerce").lt(0).any():
            raise ExportProblem(f"{name} holds rows with fold below 0")
        table = training_rows(table, name)
    table = training_games(table, name)
    table = table[(table["player_id"] > 0) & table[column].notna()].copy()
    table["player_id"] = table["player_id"].astype("int64")
    if table.duplicated(["player_id", "sportvu_game"]).any():
        raise ExportProblem(f"{name}: a player-game is listed twice")
    keys = set(zip(known["player_id"], known["sportvu_game"].astype(str)))
    if not set(zip(table["player_id"], table["sportvu_game"].astype(str))) <= keys:
        raise ExportProblem(f"{name} holds player-games that have no known absence label")
    table["p"] = table[column].astype(float)
    if ((table["p"] < 0) | (table["p"] > 1)).any():
        raise ExportProblem(f"{name}: a probability lies outside 0 to 1")
    table["one"] = 1.0
    return C.bootstrap_by_game(table, "p", "one"), record, ""


def _similar(artifacts: Path, checks_copy: Path):
    """(neighbours by player, possessions and gate by player, check record, why not)."""
    record, why = _passed_check(artifacts, EMBEDDINGS_ARTIFACT, (NEIGHBOURS_FILE, FEATURES_FILE), checks_copy)
    if record is None:
        return None, None, None, why
    folder = artifacts / EMBEDDINGS_ARTIFACT

    def table(name: str) -> pd.DataFrame:
        out = C.read_table(folder / name, what=f"artifacts/{EMBEDDINGS_ARTIFACT}/{name}")
        return out[out["season"].astype(str) == C.SEASON] if "season" in out.columns else out

    near = table(NEIGHBOURS_FILE)
    need = {"player_id", "neighbour_rank", "neighbour_player_id", "distance"}
    if not need <= set(near.columns):
        raise ExportProblem(f"{NEIGHBOURS_FILE} lacks {sorted(need - set(near.columns))}")
    if "neighbour_season" in near.columns:
        near = near[near["neighbour_season"].astype(str) == C.SEASON]
    near = near[(near["player_id"] > 0) & (near["neighbour_rank"] <= N_NEIGHBOURS)]
    near = near.sort_values(["player_id", "neighbour_rank"], kind="stable")
    feats = table(FEATURES_FILE)
    need = {"player_id", "n_possessions", "insufficient"}
    if not need <= set(feats.columns):
        raise ExportProblem(f"{FEATURES_FILE} lacks {sorted(need - set(feats.columns))}")
    feats = feats[feats["player_id"] > 0].drop_duplicates("player_id").set_index("player_id")
    return near, feats, record, ""


# ------------------------------------------------------------------------------------------------- notes


def _row_value(doc, lane: str, field_name: str, starts: str = "") -> str:
    """The value of the first row of a lane's generated table with this field name (and this start)."""
    for name, value in doc.rows.get(lane, []):
        if name == field_name and value.startswith(starts):
            return value
    raise ExportProblem(f"reports/results.md: lane {lane} has no row {field_name!r} starting {starts!r}")


def _left_out_note(info: PlayersInfo) -> str:
    """Which out-of-fold attempts count for no player, and why, with every count taken from the tables."""
    parts = [f"{info.attempts_left_out_by_reason[key]} {words.format(games=info.games_without_shot_frames)}"
             for key, words in LEFT_OUT_REASONS if info.attempts_left_out_by_reason.get(key)]
    return (f"Attempts left out: {info.attempts_left_out} of {info.attempts_total} out-of-fold attempts, which come "
            f"from {info.games_with_attempts} training games (more than the tracked training games, because a "
            "few training games have shot-chart rows and no tracking tables). An attempt counts for a player only when the tracked "
            "attempt names him as the shooter resolved from the play-by-play."
            + (" Of those left out, " + "; ".join(parts) + "." if parts else ""))


def _metric_notes(doc, info: PlayersInfo, absence_model: bool, similar_ready: bool) -> list:
    interval = C.interval_label()
    notes = {
        "shot_making": [
            f"Makes above expected per 100 attempts. Expected is the lane's location and contest model ({BASELINE}), "
            "which does not know who is shooting. It is a candidate the lane did not select.",
            "That model has a structured error of its own: its out-of-fold calibration table is given in "
            "shot_making_baseline. baseline_error_per_100 is the part of a player's value that this table explains "
            "for his mix of shots: his attempts in each calibration bin times that bin's observed minus predicted "
            "rate, per 100 attempts. Read the two numbers together.",
            "The frozen xFG model includes a shooter prior, so its own residual is near zero by construction. It is "
            "given for reference only.",
            _left_out_note(info),
            "fg_pct is descriptive: the share of the player's counted attempts that went in.",
            f"Interval: {interval}. It belongs to over_expected_per_100. fg_pct, expected_fg_pct, "
            "baseline_error_per_100 and frozen_model_residual_per_100 are point estimates: no interval was computed "
            "for them.",
        ],
        "gravity": [
            "Descriptive. Feet the nearest defender stands closer to the player than his location alone explains. "
            "Values are the lane's stored player table: training games only, expectation out of fold by game.",
            f"The lane's own gate, as its report states it: {_row_value(doc, 'gravity', 'player gate')}.",
            f"Interval: {C.interval_label('stored by the gravity lane')}.",
        ],
        "rebounding": [
            "The rate is descriptive: rebounds per 100 chances in training games. A chance is a missed shot with a "
            "located rim frame while the player was on the floor, so a total here is not a box-score total.",
            "The model expectation is the selected model's out-of-fold probability. It already uses the player's "
            "rebounding rate in other games, so the gap between the two is not a skill number.",
            f"Interval: {interval}. It belongs to rate_per_100. model_expected_per_100 is a point estimate: no "
            "interval was computed for it.",
        ],
        "absence_risk": [
            "The observed rate is descriptive: the share of the player's tracked games after which he did not appear "
            "in his team's next tracked game. Rest, a coach's decision, an injury and a tracking miss all read the "
            "same: neither number is an injury forecast.",
            ("Model probability: the mean out-of-fold probability of the frozen logistic. The frozen run did not "
             "store it. It was rebuilt by rerunning the frozen code on training games and is used because the rebuild "
             "passed its check against the frozen record (PLAN.md D45, D49; the record is in checks and its tracked "
             f"copy is {CHECKS_COPY}).")
            if absence_model else
            ("The frozen model's out-of-fold probabilities were not stored, and no rebuilt copy with a passed and "
             "tracked check against the frozen record was found, so only the observed rate is shown."),
            f"The lane's out-of-fold result for absence: {_row_value(doc, 'load', 'out-of-fold, selected', 'absent_next')} "
            "(reports/load.md).",
            f"Interval: {interval}.",
        ],
        "similar": [
            "Descriptive. The nearest players in the lane's style embedding, fitted on training games. A neighbour "
            "and a distance are point estimates: they carry no interval.",
            f"As the lane's report states: {_row_value(doc, 'embeddings', 'note', 'Player-seasons')}",
        ] + ([] if similar_ready else [
            "The embedding tables were not stored at the freeze, and no rebuilt copy with a passed and tracked check "
            "against the frozen record was found, so no similar players are shown."]),
    }
    out = []
    for metric, lane in METRIC_LANES.items():
        if lane not in doc.summary:
            raise ExportProblem(f"reports/results.md: the Summary table has no lane {lane}")
        out.append(S.MetricNote(
            metric=metric, verdict=doc.summary[lane]["verdict"], notes=notes[metric],
            sources=[C.source(site_reports.RESULTS_MD, f"verdict sentence of lane {lane} (Summary table) and its "
                              "generated rows", "not_split"),
                     C.source(doc.records[lane].path, f"lane report of {lane}", "not_split")]))
    return out


RULES = (
    ("shot_making", MIN_ATTEMPTS, "attempts",
     "Set in PLAN.md D47 before any player value was computed. The lane has no gate for shot attempts."),
    ("gravity", MIN_GRAVITY_POSSESSIONS, "possessions", "The gravity lane's own gate, stored in its player table."),
    ("rebounding", MIN_REBOUND_CHANCES, "chances",
     "Set in PLAN.md D47 before any player value was computed. The lane has no gate for rebound chances."),
    ("absence_risk", MIN_ABSENCE_GAMES, "games", "Set in PLAN.md D47 before any player value was computed."),
    ("similar", MIN_SIMILAR_POSSESSIONS, "possessions", "The embeddings lane's own gate."),
)


# ------------------------------------------------------------------------------------------------- the file


def _count(frame, player_id: int, column: str = "n") -> int:
    return int(frame.at[player_id, column]) if frame is not None and player_id in frame.index else 0


def build_players(data_root, artifacts, repo_root=C.REPO_ROOT, doc=None, checks_copy=None):
    """-> (PlayersFile, PlayersInfo). Reads training tables under the data root and, when their checks passed and
    equal their tracked copy (`checks_copy`, default reports/export/frozen_checks.json of the repository), the
    rebuilt absence and embedding outputs under the artifacts folder."""
    data_root, artifacts, repo_root = Path(data_root), Path(artifacts), Path(repo_root)
    checks_copy = Path(checks_copy) if checks_copy else repo_root / CHECKS_COPY
    doc = doc or site_reports.read_results(repo_root)
    baseline, gaps, report_counts = read_baseline(repo_root)
    players, roster_rows, n_games, (first_day, last_day) = _rosters(data_root)
    positions = _positions(data_root, roster_rows)
    shots, attempt_counts = _shot_making(data_root, gaps, report_counts)
    gravity = _gravity(data_root, players)
    rebounds = _rebounding(data_root)
    known = _absence_rows(data_root)
    observed = C.bootstrap_by_game(known, "absent", "one")
    absent_counts = known.groupby("player_id")["absent"].sum()
    model, absence_record, why_no_model = _absence_model(artifacts, known, checks_copy)
    near, feats, similar_record, why_no_similar = _similar(artifacts, checks_copy)

    for name, frame in (("xfg_predictions", shots), ("gravity_players", gravity), ("rebound predictions", rebounds),
                        ("load player_games", observed)):
        unknown = sorted(set(frame.index) - set(players.index))
        if unknown:
            raise ExportProblem(f"{name}: player id(s) {unknown[:3]} are in no training roster")

    interval = C.interval_label()
    slugs = {int(p): C.slugify(n) for p, n in players["name"].items()}
    taken = pd.Series(slugs).value_counts()
    slugs = {p: (s if s and taken[s] == 1 else f"{s}-{p}".strip("-")) for p, s in slugs.items()}
    names = {int(p): str(n) for p, n in players["name"].items()}

    views = []
    info = PlayersInfo(n_players=int(len(players)), attempts_total=attempt_counts["total"],
                       attempts_left_out=attempt_counts["left_out"],
                       attempts_left_out_by_reason=attempt_counts["by_reason"],
                       games_with_attempts=attempt_counts["games"],
                       games_without_shot_frames=attempt_counts["games_without_table"], training_games=n_games, absence="model_out_of_fold" if model is not None else
                       f"observed_only ({why_no_model})", similar="available" if near is not None else
                       f"not available ({why_no_similar})")
    shown_by_metric = {metric: 0 for metric in METRIC_LANES}
    for player_id in sorted(int(p) for p in players.index):
        attempts = _count(shots, player_id)
        show = attempts >= MIN_ATTEMPTS
        shot_making = S.ShotMaking(
            shown=show, attempts=attempts, min_attempts=MIN_ATTEMPTS,
            over_expected_per_100=C.estimate(shots.at[player_id, "point"], shots.at[player_id, "lo"],
                                             shots.at[player_id, "hi"], 2, attempts, "attempts", interval) if show else None,
            fg_pct=C.rnd(shots.at[player_id, "fg_pct"], 4) if show else None,
            expected_fg_pct=C.rnd(shots.at[player_id, "expected_fg_pct"], 4) if show else None,
            baseline_error_per_100=C.rnd(shots.at[player_id, "baseline_error"], 2) if show else None,
            frozen_model_residual_per_100=C.rnd(shots.at[player_id, "frozen_residual"], 2) if show else None)

        in_gravity = player_id in gravity.index
        possessions = _count(gravity, player_id, "n_possessions")
        show = in_gravity and not bool(gravity.at[player_id, "insufficient"])
        grav = S.Gravity(
            shown=show, possessions=possessions, min_possessions=MIN_GRAVITY_POSSESSIONS,
            games=_count(gravity, player_id, "n_games"),
            gravity_ft=C.estimate(gravity.at[player_id, "gravity_ft"], gravity.at[player_id, "lo"],
                                  gravity.at[player_id, "hi"], 3, possessions, "possessions",
                                  C.interval_label("stored by the gravity lane")) if show else None)

        chances = _count(rebounds, player_id)
        show = chances >= MIN_REBOUND_CHANCES
        rebounding = S.Rebounding(
            shown=show, chances=chances, min_chances=MIN_REBOUND_CHANCES,
            rate_per_100=C.estimate(rebounds.at[player_id, "point"], rebounds.at[player_id, "lo"],
                                    rebounds.at[player_id, "hi"], 2, chances, "chances", interval) if show else None,
            model_expected_per_100=C.rnd(rebounds.at[player_id, "expected_per_100"], 2) if show else None)

        games = _count(observed, player_id)
        show = games >= MIN_ABSENCE_GAMES
        has_model = model is not None and player_id in model.index
        absence = S.AbsenceRisk(
            shown=show, games=games, min_games=MIN_ABSENCE_GAMES,
            absent=int(absent_counts.get(player_id, 0)), kind="model_out_of_fold" if has_model else "observed_only",
            model_probability=C.estimate(model.at[player_id, "point"], model.at[player_id, "lo"],
                                         model.at[player_id, "hi"], 4, _count(model, player_id), "games",
                                         interval) if show and has_model else None,
            observed_rate=C.estimate(observed.at[player_id, "point"], observed.at[player_id, "lo"],
                                     observed.at[player_id, "hi"], 4, games, "games", interval) if show else None)

        available = near is not None
        seen = _count(feats, player_id, "n_possessions") if available else 0
        mine = near[near["player_id"] == player_id] if available else None
        show = bool(available and len(mine) and player_id in feats.index and not bool(feats.at[player_id, "insufficient"])
                    and seen >= MIN_SIMILAR_POSSESSIONS)
        neighbours = []
        if show:
            for row in mine.itertuples(index=False):
                other = int(row.neighbour_player_id)
                if other not in names:
                    raise ExportProblem(f"{NEIGHBOURS_FILE}: neighbour {other} is in no training roster")
                neighbours.append(S.SimilarPlayer(rank=int(row.neighbour_rank), player_id=other, name=names[other],
                                                  distance=C.rnd(row.distance, 4)))
        similar = S.Similar(shown=show, available=available, possessions=seen,
                            min_possessions=MIN_SIMILAR_POSSESSIONS, neighbours=neighbours)

        metrics = {"shot_making": shot_making, "gravity": grav, "rebounding": rebounding, "absence_risk": absence,
                   "similar": similar}
        if not any(m.shown for m in metrics.values()):
            continue
        for metric, value in metrics.items():
            shown_by_metric[metric] += int(value.shown)
        views.append(S.PlayerView(
            player_id=player_id, slug=slugs[player_id], name=names[player_id], teams=list(players.at[player_id, "teams"]),
            position=positions.get(player_id), games=int(players.at[player_id, "games"]), shot_making=shot_making,
            gravity=grav, rebounding=rebounding, absence_risk=absence, similar=similar))

    info.n_shown, info.n_hidden, info.shown_by_metric = len(views), int(len(players)) - len(views), shown_by_metric

    def lane_table(*parts: str) -> str:
        return C.data_path("analytics", *parts)

    sources = [
        C.source(lane_table("xfg", C.SEASON, "xfg_predictions.parquet"), "shot-making: out-of-fold rows (fold >= 0)",
                 "out_of_fold"),
        C.source(lane_table("shot_frames", C.SEASON, "<game>.parquet"), "shooter of each attempt, training games",
                 "training"),
        C.source(lane_table("gravity", "gravity_players.parquet"), "the gravity lane's player table, training games",
                 "training"),
        C.source(lane_table("rebounds", C.SEASON, "<game>.parquet"), "rebounding: out-of-fold rows (fold >= 0)",
                 "out_of_fold"),
        C.source(lane_table("load", C.SEASON, "player_games.parquet"), "absence labels of training games", "training"),
        C.source(lane_table("rosters", C.SEASON, "<game>.parquet"), "names, teams and games of training games",
                 "training"),
        C.source(C.data_path("collect/positions.parquet"), "listed position of the season, joined through bref_slug",
                 "not_split"),
    ]
    checks = []
    for name, record, what, rows in (
            (ABSENCE_ARTIFACT, absence_record, "out-of-fold absence probabilities of the frozen logistic, rebuilt "
             "outside git (PLAN.md D45); its check record is tracked", "out_of_fold"),
            (EMBEDDINGS_ARTIFACT, similar_record, "nearest neighbours of the frozen embedding, rebuilt on training "
             "games outside git (PLAN.md D45); its check record is tracked", "training")):
        if record is None:
            continue
        checks.append(_rebuild_check(record, name))
        file_name = ABSENCE_FILE if name == ABSENCE_ARTIFACT else NEIGHBOURS_FILE
        sources.append(C.source(f"artifacts/{name}/{file_name}", what, rows))
        sources.append(C.source(CHECKS_COPY, f"the tracked copy of the {name} rebuild's check record, equal to the "
                                "artifact's own check", "not_split"))
    window = f"{n_games} training games, {first_day} to {last_day}; the held-out games feed no player value"
    file = S.PlayersFile(
        schema_version=S.SCHEMA_VERSION, season=C.SEASON, window=window, n_players=info.n_players, n_shown=info.n_shown,
        n_hidden=info.n_hidden, rules=[S.MinSampleRule(metric=m, min_n=n, unit=u, reason=r) for m, n, u, r in RULES],
        metrics=_metric_notes(doc, info, model is not None, near is not None), shot_making_baseline=baseline,
        checks=checks, players=views, sources=C.merge_sources(sources))
    return file, info

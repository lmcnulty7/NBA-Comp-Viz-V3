"""export/demo_bridge.py: the demo bridge (PLAN.md M2; D16, D27, D42, D50).

    python3 -m export.demo_bridge --cv-root DIR --out DIR --media DIR [--artifacts DIR] [--stills DIR]
                                  [--game gsw_phx_2016] [--build r23] [--no-media]

On the windows of the held-out game that the adopted broadcast build rebuilt, the bridge turns broadcast tracking
into contract frames, runs the frozen models once on sensor positions and once on broadcast positions, and reports
how far the answers move. It is an illustration (D16): player identity, team and the ball come from the sensor
data on both sides, so the differences measure position error only, and the output says so.

Inputs
- DATA_ROOT (the environment variable): the sensor frames of the game and its possessions, shot_frames and shot
  chart rows, read without any outcome column, and the stored held-out predictions of the frozen models.
- --cv-root: the local outputs of the broadcast build (ignored by git): <build>/<window>_frames.json (boxes,
  stabilised feet, the row's homography), build/<window>_timemap.json, truth/<window>_truth.json (the synced sensor
  moment of each row), replay_offsets.json (the decode offset of each seek segment) and the clips build/<window>.mp4.
- reports/scorecard/<game>__<build>.json: the matching distance (method.match_ft) and the mirror. Neither the
  matching distance nor any other constant here is tuned: the demo game is the held-out game (D50).
- <artifacts>/xfg, epv, ghosting: the rebuilt frozen models (tools/frozen_models.py, tools/frozen_epv.py,
  tools/frozen_ghosting.py), each used only when its check passed. A model that is missing gets a summary with
  available false and a note, and no items.

Method (the same statements are written into demo/swap.json)
1. A box's court position is its stabilised foot point through that row's homography.
2. Every row with a synced sensor moment is matched to the sensor players of that moment: one-to-one assignment
   (sportvu.truth._match), kept within the scorecard's matching distance. A matched player takes the broadcast
   position, an unmatched sensor player keeps the sensor position, an unmatched box is dropped.
3. The broadcast frames table is the sensor frames table with positions replaced inside the windows. Between two
   consecutive broadcast samples of one player at most MAX_GAP_S apart the position is interpolated linearly onto
   the sensor frames; every other player-frame inside a window keeps the sensor position and counts as filled.
4. Fill share = filled player-frames over all player-frames inside the windows, overall and per possession.
5. The frozen models predict on both tables. Each sensor-side value must equal the stored held-out prediction
   within the tolerance of the model's rebuild check; otherwise the run stops.
6. Scenes for the cold open are chosen by a rule that reads no position error: the longest stretches in which
   every row has a synced moment.

Outputs
- <out>/demo/swap.json and <out>/demo/frames.json, validated against export/schemas.py.
- <artifacts>/demo/possessions.json: the possessions that overlap a window (the site data marks them), and
  <artifacts>/demo/media_check.json: what was measured on the clips and stills.
- <media>/<scene>.mp4 (private, never committed) and <stills>/<scene>.jpg (public).

Nothing here fits, tunes, calibrates or scores anything, and no outcome column is read.
"""

import argparse
import importlib
import json
import os
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from export import schemas as S
from export.heavy import heavy_job
from tools.frozen_common import TOLERANCE_CAP, artifacts_dir, assert_safe_output, read_check

REPO_ROOT = Path(__file__).resolve().parents[1]
WHO = "export.demo_bridge"

GAME = "gsw_phx_2016"            # the held-out game of the tracking scorecard (sportvu/splits.json, heldout_game)
BUILD = "r23"                    # the adopted build (PLAN.md D5, D42)
MAX_GAP_S = 0.5                  # longest gap between two broadcast samples that is interpolated (D50)
SCENES_MAX = 4                   # scenes of the cold open (D50)
SCENE_MAX_S = 12.0               # a scene runs at most this long from the start of its stretch (D50)
CLIP_CAP_S = 15.0                # PLAN.md M2: no clip longer than this
TOTAL_CAP_S = 60.0               # PLAN.md M2: all clips together
CLIP_MAX_WIDTH = 1280            # clips are at most this wide
GHOST_HORIZON_S = 1.0            # the ghosting comparison uses the 1.0 s ghosts
GHOST_STRIDE = 5                 # the ghosting lane samples every fifth sensor frame of a possession
SENSOR_HZ = 25.0                 # sensor frames a second (docs/DATA_CONTRACT.md section 4)
COURT_MARGIN_FT = 10.0           # an unmatched box is put on the court map only this close to the court
CLOCK_TOL_S = 0.12               # sportvu.truth.CLOCK_TOL_S, used when that module cannot be imported
CLOCK_ROUND_S = 0.005            # the truth files store clocks to 0.01 s
SHIFTS = (-30, -12, 0, 12, 30)   # frame shifts of the motion check on a clip (0 is the alignment in the data)
SCENE_RULE = ("Scenes are chosen by a rule that reads no position error: among all stretches of one window in "
              "which every row of the build has a synced sensor moment and follows the row before it by one "
              "stride, the {n} longest are taken, the earlier window first on a tie, and each scene is the "
              "first {s:g} s of its stretch.")
UNITS = {"xfg": "probability", "epv": "points", "ghosting": "ft"}
LOADERS = {"xfg": ("tools.frozen_models", "load_xfg"), "epv": ("tools.frozen_epv", "load"),
           "ghosting": ("tools.frozen_ghosting", "load")}
MIRRORS = {"identity": (False, False), "flip_x": (True, False), "flip_y": (False, True), "flip_xy": (True, True)}
FRAME_COLUMNS = ["game", "period", "frame", "game_clock_s", "shot_clock_s", "n_players", "ball_x", "ball_y", "ball_z",
                 "team_id", "team", "home", "player_id", "x", "y"]
# No outcome column: points and end_kind stay on disk.
POSSESSION_COLUMNS = ["sportvu_game", "possession", "period", "start_frame", "end_frame", "start_clock_s",
                      "start_event", "end_event", "offense_team_id", "gap"]
LABEL = "made"                   # the outcome of a shot: never read, a constant stands in for it


class BridgeProblem(Exception):
    """An input is not what the bridge requires, or one of its checks failed. The run stops."""


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _r(value, digits: int):
    """A plain rounded float (negative zero written as zero), or None for a missing value."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits) + 0.0


def _clean(error) -> str:
    """An error as one line that is safe to write into a data file."""
    text = " ".join(str(error).split()) or type(error).__name__
    return type(error).__name__ if S.find_private(text) else text


def dump(model: S.Strict) -> str:
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"), allow_nan=False) + "\n"


# ------------------------------------------------------------------------------------------------- inputs

def read_scorecard(game: str, build: str, repo=REPO_ROOT) -> dict:
    """What the bridge takes from the tracked scorecard of a build: the matching distance, the mirror, the folder
    of the build and the whole-game position error that a scene's error is shown beside."""
    rel = f"reports/scorecard/{game}__{build}.json"
    path = Path(repo) / rel
    if not path.exists():
        raise BridgeProblem(f"{rel} is missing: the matching distance is read from it")
    card = _json(path)
    method = card.get("method") or {}
    if "match_ft" not in method:
        raise BridgeProblem(f"{rel} has no method.match_ft")
    position = (card.get("metrics") or {}).get("position_error") or {}
    return {"path": rel, "game": str(card["game"]), "game_id": str(card["sportvu_game"]), "build": str(card["build"]),
            "build_dir": str(card["build_dir"]), "mirror": str(card.get("mirror") or "identity"),
            "match_ft": float(method["match_ft"]), "p50_ft": position.get("p50_ft"), "p90_ft": position.get("p90_ft"),
            "matched_players": position.get("matched_players"), "testable": card.get("testable"),
            "truth_frames": card.get("truth_frames")}


def build_folder(build_dir: str) -> str:
    """The build's folder below the broadcast outputs folder (the scorecard names it from the repository root)."""
    parts = Path(build_dir).parts
    return str(Path(*parts[2:])) if len(parts) > 2 and parts[:2] == ("data", "sportvu") else Path(build_dir).name


@dataclass
class Window:
    """One rebuilt window: the build's rows (about ten a second) and the synced sensor moment of each row."""

    order: int
    name: str
    fps: float
    stride: int
    offset_s: float
    rows: list                    # the sidecar rows: frame, H, boxes
    frame: np.ndarray             # frame number of each row in the window clip
    synced: np.ndarray            # the row has a synced sensor moment
    period: np.ndarray            # 0 where not synced
    clock: np.ndarray             # NaN where not synced
    moment: np.ndarray            # -1 where not synced
    spans: list                   # the running-clock spans of the window's time map
    decode: list                  # [[first frame of a seek segment, frames the decoder ran late], ...] or None
    uncalibrated: list
    pos: np.ndarray = None        # position of the synced sensor frame in Sensor.ids, -1 where not synced
    kept: np.ndarray = None       # synced, and the one row kept for its sensor frame
    court: list = None            # per row: court position of each box in the sensor frame (NaN when not finite)
    pair_row: list = None         # per row: table row of the sensor player matched to each box, -1 when unmatched
    pair_ft: list = None          # per row: matched distance of each box in feet, NaN when unmatched

    def decode_offset(self, frame: int) -> int:
        """Frames the decoder ran late when the build read `frame`: the row was computed on clip frame
        frame + this."""
        if self.decode is None:
            raise BridgeProblem(f"{self.name}: no decode offsets, so its boxes cannot be placed on the clip")
        late = 0
        for start, value in self.decode:
            if start <= frame:
                late = int(value)
        return late


def load_windows(cv_root, game: str, folder: str) -> list:
    """The windows of a game as the build's sidecars, the time maps and the truth files describe them."""
    cv_root = Path(cv_root)
    plan = cv_root / "build" / f"{game}_windows.json"
    if not plan.exists():
        raise BridgeProblem(f"no window plan for {game} in the broadcast outputs folder (build/{game}_windows.json)")
    offsets_path = cv_root / "replay_offsets.json"
    offsets = _json(offsets_path) if offsets_path.exists() else {}
    windows = []
    for order, entry in enumerate(_json(plan)):
        name = entry["window"]
        side = _json(cv_root / folder / f"{name}_frames.json")
        truth = _json(cv_root / "truth" / f"{name}_truth.json")
        timemap = _json(cv_root / "build" / f"{name}_timemap.json")
        rows, marks = side["frames"], truth["frames"]
        frames = [int(r["frame"]) for r in rows]
        if frames != [int(t["frame"]) for t in marks]:
            raise BridgeProblem(f"{name}: the build's rows and the truth rows are not the same frames")
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise BridgeProblem(f"{name}: the build's rows are not in frame order")
        has = ["moment" in t for t in marks]
        decode = offsets.get(name)
        windows.append(Window(
            order=order, name=name, fps=float(timemap["fps"]), stride=int(side["stride"]),
            offset_s=float(truth.get("offset_s") or 0.0), rows=rows, frame=np.array(frames, dtype=np.int64),
            synced=np.array(has, dtype=bool),
            period=np.array([int(t["q"]) if ok else 0 for t, ok in zip(marks, has)], dtype=np.int64),
            clock=np.array([float(t["clock"]) if ok else np.nan for t, ok in zip(marks, has)], dtype=float),
            moment=np.array([int(t["moment"]) if ok else -1 for t, ok in zip(marks, has)], dtype=np.int64),
            spans=list(timemap.get("spans") or []),
            decode=None if decode is None else [[int(a), int(b)] for a, b in decode.get("offsets") or []],
            uncalibrated=[] if decode is None else list(decode.get("uncalibrated") or [])))
    if not windows:
        raise BridgeProblem(f"the window plan of {game} is empty")
    return windows


def _label_free(path, wanted=None) -> pd.DataFrame:
    """A table without its outcome column. The footer must say that no row lacks the outcome, so the frozen cleaning
    (which drops a row without one) keeps the same rows when a constant stands in for it."""
    import pyarrow.parquet as pq
    handle = pq.ParquetFile(path)
    names = handle.schema_arrow.names
    if LABEL in names:
        column = names.index(LABEL)
        for group in range(handle.metadata.num_row_groups):
            stats = handle.metadata.row_group(group).column(column).statistics
            if stats is None or not stats.has_null_count or stats.null_count:
                raise BridgeProblem(f"{Path(path).name}: cannot tell from the footer that no row lacks its outcome")
    table = pd.read_parquet(path, columns=[c for c in (wanted or names) if c != LABEL and c in names])
    table[LABEL] = False
    return table


def load_tables(data_root, game_id: str) -> dict:
    """The sensor-side tables of the demo game. No outcome column is read: the shot tables come without `made`
    (a constant stands in, the frozen feature code asks for the column and never uses it), the possessions without
    points and end_kind."""
    from analytics.schema import season_of_game
    root, season = Path(data_root), season_of_game(game_id)
    lane = root / "analytics"
    frames_path = root / "sportvu_contract" / "heldout" / f"{game_id}.parquet"
    if not frames_path.exists():
        raise BridgeProblem(f"no held-out sensor frames for {game_id} under $DATA_ROOT/sportvu_contract/heldout")
    shots = _label_free(root / "collect" / "bref_shots" / "heldout" / "shotchart" / f"{season}.parquet")
    shots = shots[shots["sportvu_game"].astype(str) == game_id].reset_index(drop=True)
    shots["heldout"] = True       # as the frozen loader marks every row of a file under heldout/
    return {
        "season": season,
        "frames": pd.read_parquet(frames_path, columns=FRAME_COLUMNS),
        "possessions": pd.read_parquet(lane / "possessions" / "heldout" / season / f"{game_id}.parquet",
                                       columns=POSSESSION_COLUMNS),
        "shot_frames": _label_free(lane / "shot_frames" / "heldout" / season / f"{game_id}.parquet"),
        "shots": shots,
    }


def stored_predictions(data_root, game_id: str, kind: str) -> pd.DataFrame:
    """The stored held-out predictions of one frozen model for the demo game: keys and predictions, never an
    outcome column."""
    import pyarrow.parquet as pq
    from analytics.schema import season_of_game
    lane, season = Path(data_root) / "analytics", season_of_game(game_id)
    if kind == "xfg":
        table = pq.read_table(lane / "xfg" / season / "xfg_predictions.parquet",
                              columns=["sportvu_game", "event", "player_id", "p_selected", "fold", "heldout"],
                              filters=[("sportvu_game", "=", game_id)]).to_pandas()
        return table.reset_index(drop=True)
    if kind == "epv":
        return pd.read_parquet(lane / "epv" / "heldout" / season / f"{game_id}.parquet",
                               columns=["possession", "frame", "epv_points"])
    if kind == "ghosting":
        return pd.read_parquet(lane / "ghosting" / "heldout" / season / f"ghost_frames_{game_id}.parquet",
                               columns=["possession", "frame", "def_player_id", "horizon_s", "ghost_x", "ghost_y"])
    raise KeyError(kind)


def data_sources(game_id: str, season: str, card: dict, kinds) -> dict:
    """The source records of the files the bridge reads, by short name."""
    held = "heldout"
    out = {
        "scorecard": S.Source(path=card["path"], what="the matching distance, the mirror and the whole-game position "
                              "error of the build", rows=held),
        "sidecars": S.Source(path=f"{card['build_dir']}/<window>_frames.json", what="boxes, feet and homographies "
                             "of the broadcast build per window (local files, not in the repository)", rows=held),
        "truth": S.Source(path="data/sportvu/truth/<window>_truth.json", what="the synced sensor moment of every row "
                          "of the build (local files, not in the repository)", rows=held),
        "timemaps": S.Source(path="data/sportvu/build/<window>_timemap.json", what="the game clock of every window "
                             "frame (local files, not in the repository)", rows=held),
        "frames": S.Source(path=f"$DATA_ROOT/sportvu_contract/heldout/{game_id}.parquet",
                           what="sensor positions, identity, team and ball", rows=held),
        "possessions": S.Source(path=f"$DATA_ROOT/analytics/possessions/heldout/{season}/{game_id}.parquet",
                                what="possession spans", rows=held),
        "shot_frames": S.Source(path=f"$DATA_ROOT/analytics/shot_frames/heldout/{season}/{game_id}.parquet",
                                what="release frame and shooter of each attempt", rows=held),
        "shots": S.Source(path=f"$DATA_ROOT/collect/bref_shots/heldout/shotchart/{season}.parquet",
                          what="shot location of each attempt", rows=held),
        "xfg": S.Source(path=f"$DATA_ROOT/analytics/xfg/{season}/xfg_predictions.parquet",
                        what="stored held-out xFG predictions that the sensor-side values must equal", rows=held),
        "epv": S.Source(path=f"$DATA_ROOT/analytics/epv/heldout/{season}/{game_id}.parquet",
                        what="stored held-out EPV predictions that the sensor-side values must equal", rows=held),
        "ghosting": S.Source(path=f"$DATA_ROOT/analytics/ghosting/heldout/{season}/ghost_frames_{game_id}.parquet",
                             what="stored held-out ghost positions that the sensor-side values must equal", rows=held),
        "checks": S.Source(path="reports/export/frozen_checks.json", what="the tracked copy of the rebuild checks of "
                           "the frozen models", rows="not_split"),
    }
    drop = [k for k in ("xfg", "epv", "ghosting") if k not in kinds]
    return {name: source for name, source in out.items() if name not in drop}


# ------------------------------------------------------------------------------------------- sensor frames

class Sensor:
    """The sensor frames table of one game (docs/DATA_CONTRACT.md section 4: one row per player per frame) with
    the lookups the bridge needs. The table itself is never changed."""

    def __init__(self, table: pd.DataFrame):
        need = ("frame", "period", "game_clock_s", "team_id", "player_id", "x", "y")
        missing = [c for c in need if c not in table.columns]
        if missing:
            raise BridgeProblem(f"the sensor frames table lacks column(s) {missing}")
        frame = table["frame"].to_numpy(dtype=np.int64)
        if not len(frame) or (np.diff(frame) < 0).any():
            raise BridgeProblem("the sensor frames table is empty or not ordered by frame")
        self.table = table
        self.frame = frame
        self.player = table["player_id"].to_numpy(dtype=np.int64)
        self.team_id = table["team_id"].to_numpy(dtype=np.int64)
        self.x = table["x"].to_numpy(dtype=np.float32)
        self.y = table["y"].to_numpy(dtype=np.float32)
        self.ids, self.first = np.unique(frame, return_index=True)
        self.count = np.diff(np.append(self.first, len(frame)))
        self.row_pos = np.repeat(np.arange(len(self.ids)), self.count)
        self.period = table["period"].to_numpy(dtype=np.int64)[self.first]
        self.clock = table["game_clock_s"].to_numpy(dtype=float)[self.first]
        # The synced moment of a truth row is an index into the quarter's frames that hold exactly ten players,
        # in table order (sportvu.sync.SportVUIndex).
        ten = self.count == 10
        self.moments = {int(q): np.flatnonzero(ten & (self.period == q)) for q in np.unique(self.period)}
        order = np.lexsort((frame, self.player))
        players, start = np.unique(self.player[order], return_index=True)
        stop = np.append(start[1:], len(order))
        self._by_player = {int(p): order[a:b] for p, a, b in zip(players, start, stop)}
        for rows in self._by_player.values():
            if (np.diff(frame[rows]) <= 0).any():
                raise BridgeProblem("the sensor frames table holds a player twice in one frame")
        if "team" in table.columns:
            pairs = table[["team_id", "team"]].drop_duplicates()
            self.team_name = {int(i): str(n) for i, n in zip(pairs["team_id"], pairs["team"])}
        else:
            self.team_name = {}

    def rows_at(self, pos: int) -> np.ndarray:
        """Table rows of the frame at position `pos`."""
        return np.arange(self.first[pos], self.first[pos] + self.count[pos])

    def rows_between(self, first_id: int, last_id: int) -> tuple:
        """Half-open range of table rows whose frame id lies in [first_id, last_id]."""
        return (int(np.searchsorted(self.frame, first_id, "left")), int(np.searchsorted(self.frame, last_id, "right")))

    def player_rows(self, player: int, frame_ids) -> np.ndarray:
        """Table row of `player` at each frame id, -1 where he has no row."""
        rows = self._by_player.get(int(player))
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        out = np.full(len(frame_ids), -1, dtype=np.int64)
        if rows is None or not len(frame_ids):
            return out
        have = self.frame[rows]
        j = np.minimum(np.searchsorted(have, frame_ids), len(have) - 1)
        hit = have[j] == frame_ids
        out[hit] = rows[j[hit]]
        return out

    def rows_of(self, frame_ids, players) -> np.ndarray:
        """Table row of each (frame id, player) pair, -1 where the pair has no row."""
        frame_ids, players = np.asarray(frame_ids, dtype=np.int64), np.asarray(players, dtype=np.int64)
        out = np.full(len(frame_ids), -1, dtype=np.int64)
        for player in np.unique(players):
            mine = players == player
            out[mine] = self.player_rows(int(player), frame_ids[mine])
        return out


# -------------------------------------------------------------------------------- sync, match, interpolate

def timemap_clock(spans: list, frame: int):
    """(period, clock) of a window frame from the window's time map: linear inside a running-clock span."""
    for span in spans:
        if span["f0"] <= frame <= span["f1"]:
            share = (frame - span["f0"]) / max(span["f1"] - span["f0"], 1)
            return int(span["period"]), span["c0"] + share * (span["c1"] - span["c0"])
    return None, None


def sync_rows(sensor: Sensor, windows: list, clock_tol_s: float = CLOCK_TOL_S) -> dict:
    """Give every synced row its frame of the sensor table and check the mapping on the game clocks.

    Mapping: the moment index of a truth row counts the quarter's ten-player frames in table order, so the sensor
    frame is that frame of the quarter. Checks, on every synced row: the frame lies in the row's quarter; its game
    clock is the row's clock within the sync tolerance; no other frame of the quarter is nearer to the row's clock
    by more than the rounding of the stored clock; and the window's time map plus its offset gives the same clock.
    When two rows of a window are synced to the same sensor frame, the row whose clock is nearer to that frame is
    kept (the earlier one on a tie) and the other is not used for positions."""
    report = {"rows": 0, "synced": 0, "used": 0, "max_clock_gap_s": 0.0, "max_nearest_margin_s": 0.0,
              "max_timemap_gap_s": 0.0}
    for w in windows:
        report["rows"] += len(w.frame)
        w.pos = np.full(len(w.frame), -1, dtype=np.int64)
        w.kept = np.zeros(len(w.frame), dtype=bool)
        idx = np.flatnonzero(w.synced)
        if not len(idx):
            continue
        for i in idx:
            frames = sensor.moments.get(int(w.period[i]))
            if frames is None or not 0 <= w.moment[i] < len(frames):
                raise BridgeProblem(f"{w.name} frame {w.frame[i]}: moment {w.moment[i]} of quarter {w.period[i]} "
                                    "is not a frame of the sensor table")
            w.pos[i] = frames[w.moment[i]]
            period, clock = timemap_clock(w.spans, int(w.frame[i]))
            if period != w.period[i]:
                raise BridgeProblem(f"{w.name} frame {w.frame[i]}: the time map and the truth file disagree on the period")
            report["max_timemap_gap_s"] = max(report["max_timemap_gap_s"], abs(clock + w.offset_s - w.clock[i]))
        gap = np.abs(sensor.clock[w.pos[idx]] - w.clock[idx])
        for q in np.unique(w.period[idx]):
            mine = w.period[idx] == q
            clocks = sensor.clock[sensor.moments[int(q)]]          # falling through the quarter
            j = np.searchsorted(-clocks, -w.clock[idx][mine])
            before, after = np.clip(j - 1, 0, len(clocks) - 1), np.clip(j, 0, len(clocks) - 1)
            nearest = np.minimum(np.abs(clocks[before] - w.clock[idx][mine]), np.abs(clocks[after] - w.clock[idx][mine]))
            report["max_nearest_margin_s"] = max(report["max_nearest_margin_s"], float((gap[mine] - nearest).max()))
        report["max_clock_gap_s"] = max(report["max_clock_gap_s"], float(gap.max()))
        best = {}
        for i, pos, g in zip(idx.tolist(), w.pos[idx].tolist(), gap.tolist()):
            if pos not in best or g < best[pos][0] - 1e-9:
                best[pos] = (g, i)
        w.kept[[i for _, i in best.values()]] = True
        if (np.diff(w.pos[w.kept]) <= 0).any():
            raise BridgeProblem(f"{w.name}: the synced sensor frames do not move forward with the video")
        report["synced"] += len(idx)
        report["used"] += int(w.kept.sum())
    if not report["synced"]:
        raise BridgeProblem("no row of the build has a synced sensor moment")
    if report["max_clock_gap_s"] > clock_tol_s + CLOCK_ROUND_S + 1e-6:
        raise BridgeProblem(f"frame mapping: a synced row is {report['max_clock_gap_s']:.3f} s away from its sensor "
                            f"frame on the game clock, more than the sync tolerance {clock_tol_s} s")
    if report["max_nearest_margin_s"] > 2 * CLOCK_ROUND_S + 1e-6:
        raise BridgeProblem("frame mapping: a synced row has a nearer sensor frame than the one it is mapped to")
    if report["max_timemap_gap_s"] > CLOCK_ROUND_S + 1e-6:
        raise BridgeProblem("the time maps and the truth files give different clocks for the same row")
    report["duplicates"] = report["synced"] - report["used"]
    return report


def hungarian(court: np.ndarray, sensor_xy: np.ndarray, gate: float) -> list:
    """One-to-one assignment of court positions to sensor positions that minimises the summed distance, kept where
    the distance is within `gate` feet -> [(box index, sensor index, distance)]. The rule of sportvu.truth._match,
    used when that module cannot be imported."""
    from scipy.optimize import linear_sum_assignment
    distance = np.sqrt(((court[:, None, :] - sensor_xy[None, :, :]) ** 2).sum(-1))
    box, player = linear_sum_assignment(distance)
    return [(int(i), int(j), float(distance[i, j])) for i, j in zip(box, player) if distance[i, j] <= gate]


def matcher() -> tuple:
    """(the matcher, its name, the sync tolerance): the scorecard's own matcher when the frozen module imports."""
    try:
        from sportvu import truth
        return truth._match, "sportvu.truth._match", float(truth.CLOCK_TOL_S)
    except ImportError:
        return hungarian, "export.demo_bridge.hungarian (the rule of sportvu.truth._match)", CLOCK_TOL_S


def mirror_xy(xy, name: str) -> np.ndarray:
    """Court positions through one of the four mirrors of the court (its own inverse)."""
    if name not in MIRRORS:
        raise BridgeProblem(f"unknown mirror {name!r}")
    flip_x, flip_y = MIRRORS[name]
    out = np.array(xy, dtype=float, copy=True).reshape(-1, 2)
    if flip_x:
        out[:, 0] = S.COURT_LENGTH_FT - out[:, 0]
    if flip_y:
        out[:, 1] = S.COURT_WIDTH_FT - out[:, 1]
    return out


def court_positions(row: dict) -> np.ndarray:
    """Court position of every box of a sidecar row: its stabilised foot point through the row's homography
    (pixel to court feet), as the scorecard projects it. NaN for a row without a homography and for a point that
    does not land at a finite place."""
    import cv2
    boxes = row.get("boxes") or []
    out = np.full((len(boxes), 2), np.nan)
    if not boxes or row.get("H") is None:
        return out
    feet = np.array([b["foot_stab"] for b in boxes], dtype=np.float32).reshape(-1, 1, 2)
    court = cv2.perspectiveTransform(feet, np.array(row["H"], dtype=np.float64).reshape(3, 3)).reshape(-1, 2).astype(float)
    court[~np.isfinite(court).all(axis=1)] = np.nan
    return court


def match_rows(sensor: Sensor, windows: list, match_ft: float, mirror: str = "identity", match=None) -> dict:
    """Match the boxes of every synced row to the sensor players of its moment. Positions are compared in the
    sensor frame (the build's court positions go through the game's mirror first)."""
    match = match or matcher()[0]
    counts = {"boxes": 0, "matched": 0, "player_slots": 0}
    for w in windows:
        w.court, w.pair_row, w.pair_ft = ([None] * len(w.frame) for _ in range(3))
        for i in np.flatnonzero(w.synced):
            court = mirror_xy(court_positions(w.rows[i]), mirror)
            rows = sensor.rows_at(int(w.pos[i]))
            pair_row, pair_ft = np.full(len(court), -1, dtype=np.int64), np.full(len(court), np.nan)
            finite = np.flatnonzero(np.isfinite(court).all(axis=1))
            if len(finite):
                players = np.column_stack([sensor.x[rows], sensor.y[rows]]).astype(float)
                for box, player, distance in match(court[finite], players, match_ft):
                    pair_row[finite[box]], pair_ft[finite[box]] = rows[player], distance
            w.court[i], w.pair_row[i], w.pair_ft[i] = court, pair_row, pair_ft
            counts["boxes"] += len(court)
            counts["matched"] += int((pair_row >= 0).sum())
            counts["player_slots"] += len(rows)
    return counts


def player_path(video_frames, frame_ids, xy, fps: float, max_gap_s: float = MAX_GAP_S) -> tuple:
    """One player's broadcast positions on the sensor frames -> (frame ids, positions).

    `video_frames`, `frame_ids` and `xy` are his matched samples in time order: the window frame of the row, the
    sensor frame it is synced to, and the broadcast court position. Every sample keeps its position at its own
    frame. Between two consecutive samples whose rows are at most `max_gap_s` apart in video time, the sensor
    frames in between get the straight line between the two positions. Nothing is written past a longer gap."""
    video_frames = np.asarray(video_frames, dtype=float)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(frame_ids):
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2))
    ids, out = [frame_ids[:1]], [xy[:1]]
    for k in range(1, len(frame_ids)):
        a, b = frame_ids[k - 1], frame_ids[k]
        if b - a > 1 and (video_frames[k] - video_frames[k - 1]) / fps <= max_gap_s:
            between = np.arange(a + 1, b)
            share = ((between - a) / (b - a))[:, None]
            ids.append(between)
            out.append(xy[k - 1] + share * (xy[k] - xy[k - 1]))
        ids.append(frame_ids[k:k + 1])
        out.append(xy[k:k + 1])
    return np.concatenate(ids), np.concatenate(out)


@dataclass
class State:
    """Everything the comparison and the scenes read: the sensor table, the windows, and the broadcast positions."""

    card: dict
    game_id: str
    tables: dict
    sensor: Sensor
    windows: list
    sync: dict
    matches: dict
    matcher_name: str
    x: np.ndarray                 # broadcast table: x of every row (the sensor value where nothing was matched)
    y: np.ndarray
    used: np.ndarray              # the row holds a broadcast position
    window_of: np.ndarray         # window of each sensor frame position, -1 outside every window
    row_window: np.ndarray        # the same for each table row
    sample_id: np.ndarray         # sensor frame id of every matched pair of a used row
    sample_ft: np.ndarray         # and its matched distance in feet
    broadcast_table: pd.DataFrame = None
    data_root: Path = None

    def window_at(self, frame_id: int) -> int:
        j = int(np.searchsorted(self.sensor.ids, frame_id))
        return int(self.window_of[j]) if j < len(self.sensor.ids) and self.sensor.ids[j] == frame_id else -1

    def fill(self, rows) -> tuple:
        """(player-frames, filled player-frames) of a set of table rows given as a slice or an index array."""
        used = self.used[rows]
        return int(len(used)), int((~used).sum())


def broadcast_positions(sensor: Sensor, windows: list, max_gap_s: float = MAX_GAP_S) -> dict:
    """The positions of the broadcast frames table: the sensor positions, replaced inside the windows where the
    build has a matched box or two matched boxes close enough in time to interpolate between."""
    x, y = sensor.x.copy(), sensor.y.copy()
    used = np.zeros(len(x), dtype=bool)
    window_of = np.full(len(sensor.ids), -1, dtype=np.int64)
    sample_id, sample_ft = [], []
    for w in windows:
        idx = np.flatnonzero(w.synced)
        if not len(idx):
            continue
        lo, hi = int(w.pos[idx].min()), int(w.pos[idx].max())
        if (window_of[lo:hi + 1] >= 0).any():
            raise BridgeProblem(f"{w.name} covers sensor frames that another window covers")
        window_of[lo:hi + 1] = w.order
        samples = {}
        for i in np.flatnonzero(w.kept):
            frame_id = int(sensor.ids[w.pos[i]])
            for box in np.flatnonzero(w.pair_row[i] >= 0):
                row = int(w.pair_row[i][box])
                samples.setdefault(int(sensor.player[row]), []).append(
                    (float(w.frame[i]), frame_id, w.court[i][box, 0], w.court[i][box, 1]))
                sample_id.append(frame_id)
                sample_ft.append(float(w.pair_ft[i][box]))
        for player, rows in samples.items():
            arr = np.array(rows, dtype=float)
            ids, xy = player_path(arr[:, 0], arr[:, 1].astype(np.int64), arr[:, 2:4], w.fps, max_gap_s)
            target = sensor.player_rows(player, ids)
            ok = target >= 0
            x[target[ok]], y[target[ok]] = xy[ok, 0], xy[ok, 1]
            used[target[ok]] = True
    return {"x": x, "y": y, "used": used, "window_of": window_of,
            "sample_id": np.array(sample_id, dtype=np.int64), "sample_ft": np.array(sample_ft, dtype=float)}


def prepare(card: dict, tables: dict, windows: list, max_gap_s: float = MAX_GAP_S, data_root=None) -> State:
    """Sync, match and interpolate: the state every later step reads."""
    match, name, clock_tol = matcher()
    sensor = Sensor(tables["frames"])
    sync = sync_rows(sensor, windows, clock_tol)
    matches = match_rows(sensor, windows, card["match_ft"], card["mirror"], match)
    out = broadcast_positions(sensor, windows, max_gap_s)
    state = State(card=card, game_id=card["game_id"], tables=tables, sensor=sensor, windows=windows, sync=sync,
                  matches=matches, matcher_name=name, x=out["x"], y=out["y"], used=out["used"],
                  window_of=out["window_of"], row_window=out["window_of"][sensor.row_pos],
                  sample_id=out["sample_id"], sample_ft=out["sample_ft"],
                  data_root=None if data_root is None else Path(data_root))
    state.broadcast_table = sensor.table.assign(x=state.x, y=state.y)
    return state


def overall_fill(state: State) -> dict:
    inside = state.row_window >= 0
    n, filled = int(inside.sum()), int((inside & ~state.used).sum())
    if not n:
        raise BridgeProblem("no sensor frame lies inside a window")
    return {"n_frames": int((state.window_of >= 0).sum()), "n_player_frames": n, "n_filled": filled,
            "fill_share": filled / n, "n_windows": int(len(np.unique(state.window_of[state.window_of >= 0])))}


def possession_fill(state: State) -> list:
    """Fill share of every possession that shares at least one frame with a window, over its frames inside the
    windows, and the median matched distance of the broadcast samples in its span."""
    out = []
    for p in state.tables["possessions"].sort_values("possession", kind="stable").itertuples():
        r0, r1 = state.sensor.rows_between(int(p.start_frame), int(p.end_frame))
        inside = state.row_window[r0:r1] >= 0
        n = int(inside.sum())
        if not n:
            continue
        filled = int((inside & ~state.used[r0:r1]).sum())
        window = state.windows[int(np.bincount(state.row_window[r0:r1][inside]).argmax())]
        distance = state.sample_ft[(state.sample_id >= p.start_frame) & (state.sample_id <= p.end_frame)]
        out.append(S.SwapPossession(possession=int(p.possession), window=window.name, n_player_frames=n, n_filled=filled,
                                    fill_share=_r(filled / n, 6),
                                    matched_p50_ft=_r(np.median(distance), 2) if len(distance) else None))
    return out


# ------------------------------------------------------------------------------------------------- models

@dataclass
class Model:
    """A frozen model as the bridge sees it: a predictor, or the reason there is none, and its rebuild check."""

    kind: str
    predictor: object = None
    check: dict = None
    note: str = None


@dataclass
class Compared:
    """One model run on both tables: the items, and the counts the method statements quote."""

    kind: str
    items: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    note: str = None


def load_model(kind: str, artifacts=None) -> Model:
    """The rebuilt frozen model of one kind, through its own loader (which refuses a model whose check did not
    pass). The loader's module is imported by name, so a checkout without it still runs the other models."""
    module_name, loader = LOADERS[kind]
    folder = artifacts_dir(artifacts) / kind
    check = read_check(folder) if folder.is_dir() else None
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        return Model(kind, check=check, note=f"{module_name} could not be imported in this checkout ({_clean(error)}), "
                                            f"so the frozen {kind} model was not run")
    try:
        predictor = getattr(module, loader)(artifacts)
    except (RuntimeError, OSError, AttributeError) as error:
        return Model(kind, check=check, note=f"the rebuilt {kind} model cannot be used: {_clean(error)}")
    return Model(kind, predictor=predictor, check=check)


def rebuild_check(kind: str, record) -> S.RebuildCheck:
    """A rebuild's check record as the contract carries it, or None when there is no record with numbers (a
    rebuild that stopped before its comparison has none)."""
    if not record:
        return None
    try:
        expected = record.get("n_expected")
        return S.RebuildCheck(name=kind, compared=str(record["compared"]), n=int(record["n"]),
                              n_expected=None if expected is None else int(expected),
                              max_abs_diff=float(record["max_abs_diff"]), tolerance=float(record["tolerance"]),
                              passed=bool(record.get("passed")), unit=str(record.get("unit") or UNITS[kind]))
    except (KeyError, TypeError, ValueError):
        return None


def _same_as_stored(kind: str, gap, n: int) -> float:
    """The largest gap between sensor-side values and the stored held-out predictions, or BridgeProblem when it is
    above the tolerance of the model's rebuild check."""
    tolerance = TOLERANCE_CAP[kind]
    if not n or gap is None or not np.isfinite(gap) or gap > tolerance:
        raise BridgeProblem(f"{kind}: the sensor-side values do not equal the stored held-out predictions "
                            f"(largest gap {gap} over {n} values, tolerance {tolerance}); nothing is written")
    return float(gap)


def _possession_of(possessions: pd.DataFrame, event: int, period: int):
    """The possession whose event span holds a play-by-play event (the rule of the frozen lanes), or None."""
    hit = possessions[(possessions["start_event"] <= event) & (possessions["end_event"] >= event)
                      & (possessions["period"] == period)]
    return hit.iloc[0] if len(hit) else None


def _same_attempts(features: pd.DataFrame, stored: pd.DataFrame) -> bool:
    """True when two tables hold the same attempts in the same order: event and shooter (two nulls agree)."""
    if len(features) != len(stored):
        return False
    mine, theirs = (pd.to_numeric(t["player_id"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    for t in (features, stored))
    same_player = (mine == theirs) | (np.isnan(mine) & np.isnan(theirs))
    same_event = features["event"].to_numpy(dtype=np.int64) == stored["event"].to_numpy(dtype=np.int64)
    return bool(same_player.all() and same_event.all())


def xfg_swap(model: Model, state: State) -> Compared:
    """xFG for every attempt whose release frame and lookback lie inside one window, from sensor positions and
    from broadcast positions. Features come from the frozen feature code on each frames table, after the frozen
    cleaning of each table; the shot location comes from the shot chart and is the same on both sides."""
    from analytics.xfg import features as XF
    from analytics.xfg import validate as XV
    tables, sensor = state.tables, state.sensor
    shots, _ = XV.clean_shots(tables["shots"])
    shot_frames, _ = XV.clean_contract(tables["shot_frames"], "shot_frames")
    sides, dropped = {}, {}
    for side, table in (("sensor", sensor.table), ("broadcast", state.broadcast_table)):
        frames, summary = XV.clean_frames(table[XV.FRAME_COLUMNS])
        if summary["schema_problems"]:
            raise BridgeProblem(f"xfg: the frozen cleaning refused the {side} frames: {summary['schema_problems']}")
        dropped[side] = int(summary["rows_in"] - summary["rows_out"])
        features = XF.build(shots, shot_frames, {state.game_id: frames})
        sides[side] = (features, np.asarray(model.predictor.predict(features), dtype=float))
    (fs, ps), (fb, pb) = sides["sensor"], sides["broadcast"]
    stored = stored_predictions(state.data_root, state.game_id, "xfg")
    if not _same_attempts(fs, stored):
        raise BridgeProblem("xfg: the attempts of the feature frame are not the stored held-out attempts, row for row")
    if not stored["heldout"].astype(bool).all() or (stored["fold"].to_numpy(dtype=np.int64) != -1).any():
        raise BridgeProblem("xfg: the stored predictions of the demo game are not held-out rows")
    was = stored["p_selected"].to_numpy(dtype=float)
    gap = _same_as_stored("xfg", float(np.max(np.abs(ps - was))) if len(was) else None, len(was))
    release = fs["release_frame"].to_numpy(dtype=np.int64)
    measured = (fs["has_contest"].to_numpy(dtype=np.int64) == 1) & (release >= 0)
    touch = np.rint(np.nan_to_num(fs["touch_time_s"].to_numpy(dtype=float)) * XF.FPS).astype(np.int64)
    out = Compared("xfg")
    in_window = lost = 0
    for i in np.flatnonzero(measured):
        window = state.window_at(int(release[i]))
        if window < 0:
            continue
        in_window += 1
        first = int(release[i]) - max(XF.SPEED_FRAMES, int(touch[i]))
        r0, r1 = sensor.rows_between(first, int(release[i]))
        if r1 <= r0 or sensor.frame[r0] != first or (state.row_window[r0:r1] != window).any():
            continue
        event, period = int(fs.at[i, "event"]), int(fs.at[i, "period"])
        possession = _possession_of(tables["possessions"], event, period)
        n, filled = state.fill(slice(r0, r1))
        lost += int(fb.at[i, "has_contest"] != 1)
        a, b = _r(ps[i], 6), _r(pb[i], 6)
        number = None if possession is None else int(possession["possession"])
        since = None if possession is None else _r((release[i] - possession["start_frame"]) / SENSOR_HZ, 2)
        out.items.append(S.SwapItem(kind="xfg", window=state.windows[window].name, possession=number, key=f"event-{event}",
                                    t=since, sensor=a, broadcast=b, abs_diff=_r(abs(b - a), 6), fill_share=_r(filled / n, 6)))
    out.facts = {"attempts": int(len(fs)), "measured": int(measured.sum()), "release_in_window": in_window,
                 "items": len(out.items), "stored_rows": int(len(stored)), "stored_gap": gap,
                 "lost_contest": lost, "rows_dropped_by_cleaning": dropped}
    return out


def epv_swap(model: Model, state: State) -> Compared:
    """EPV for every 1 Hz state whose frame lies inside a window, from sensor positions and from broadcast
    positions. The states are the frozen ones (they depend on the clocks and the possession table, not on
    positions), and each value reads its possession from the first state on."""
    tables, sensor, keys = state.tables, state.sensor, ["possession", "frame"]

    def states(table, name):
        rows = model.predictor.predict_states(table, tables["possessions"], state.game_id)
        return pd.DataFrame({"possession": rows["possession"].to_numpy(dtype=np.int64),
                             "frame": rows["frame"].to_numpy(dtype=np.int64),
                             name: rows["epv_points"].to_numpy(dtype=float)})

    a, b = states(sensor.table, "sensor"), states(state.broadcast_table, "broadcast")
    stored = stored_predictions(state.data_root, state.game_id, "epv")
    stored = pd.DataFrame({"possession": stored["possession"].to_numpy(dtype=np.int64),
                           "frame": stored["frame"].to_numpy(dtype=np.int64),
                           "stored": stored["epv_points"].to_numpy(dtype=float)})
    both = a.merge(stored, on=keys, how="outer", indicator=True)
    if a.duplicated(keys).any() or (both["_merge"] != "both").any():
        raise BridgeProblem("epv: the states of the sensor table are not the stored held-out states")
    gap = _same_as_stored("epv", float(np.max(np.abs(both["sensor"] - both["stored"]))) if len(both) else None, len(both))
    pair = a.merge(b, on=keys, how="inner").sort_values(keys, kind="stable").reset_index(drop=True)
    start = dict(zip(tables["possessions"]["possession"].astype(int), tables["possessions"]["start_frame"].astype(int)))
    out = Compared("epv")
    seen = {}
    for row in pair.itertuples():
        r0, r1 = sensor.rows_between(int(row.frame), int(row.frame))
        n, filled = state.fill(slice(r0, r1))
        total = seen.setdefault(int(row.possession), [0, 0])
        total[0] += n
        total[1] += filled
        window = state.window_at(int(row.frame))
        if window < 0:
            continue
        s, c = _r(row.sensor, 6), _r(row.broadcast, 6)
        out.items.append(S.SwapItem(
            kind="epv", window=state.windows[window].name, possession=int(row.possession),
            key=f"state-{int(row.possession)}-{int(row.frame)}",
            t=_r((row.frame - start[int(row.possession)]) / SENSOR_HZ, 2) if int(row.possession) in start else None,
            sensor=s, broadcast=c, abs_diff=_r(abs(c - s), 6), fill_share=_r(total[1] / total[0], 6)))
    out.facts = {"states": int(len(a)), "items": len(out.items), "stored_rows": int(len(stored)), "stored_gap": gap,
                 "states_on_one_side_only": int(len(a) + len(b) - 2 * len(pair))}
    return out


def ghost_swap(model: Model, state: State) -> Compared:
    """One item per possession: the mean distance in feet between the real defenders and their ghosts at the
    GHOST_HORIZON_S horizon, over the ghost rows whose frame lies inside a window, once with every position from
    the sensor table and once from the broadcast table."""
    tables, sensor, keys = state.tables, state.sensor, ["possession", "frame", "def_player_id"]

    def ghosts(table, x, y, name):
        rows = model.predictor.predict_ghosts(table, tables["possessions"], state.game_id)
        rows = rows[np.abs(rows["horizon_s"].to_numpy(dtype=float) - GHOST_HORIZON_S) < 0.05]
        out = pd.DataFrame({k: rows[k].to_numpy(dtype=np.int64) for k in keys})
        out[name + "_x"], out[name + "_y"] = rows["ghost_x"].to_numpy(dtype=float), rows["ghost_y"].to_numpy(dtype=float)
        at = sensor.rows_of(out["frame"].to_numpy(), out["def_player_id"].to_numpy())
        if (at < 0).any():
            raise BridgeProblem("ghosting: a ghost row names a defender who has no row in that frame")
        out[name] = np.hypot(x[at] - out[name + "_x"].to_numpy(), y[at] - out[name + "_y"].to_numpy())
        return out

    a = ghosts(sensor.table, sensor.x, sensor.y, "sensor")
    b = ghosts(state.broadcast_table, state.x, state.y, "broadcast")
    stored = stored_predictions(state.data_root, state.game_id, "ghosting")
    stored = stored[np.abs(stored["horizon_s"].to_numpy(dtype=float) - GHOST_HORIZON_S) < 0.05]
    stored = pd.DataFrame({**{k: stored[k].to_numpy(dtype=np.int64) for k in keys},
                           "ghost_x": stored["ghost_x"].to_numpy(dtype=float),
                           "ghost_y": stored["ghost_y"].to_numpy(dtype=float)})
    both = a.merge(stored, on=keys, how="outer", indicator=True)
    if a.duplicated(keys).any() or (both["_merge"] != "both").any():
        raise BridgeProblem("ghosting: the ghost rows of the sensor table are not the stored held-out rows")
    off = np.abs(both[["sensor_x", "sensor_y"]].to_numpy() - both[["ghost_x", "ghost_y"]].to_numpy())
    gap = _same_as_stored("ghosting", float(off.max()) if len(both) else None, len(both))
    pair = a.merge(b, on=keys, how="inner")
    one_side = int(len(a) + len(b) - 2 * len(pair))
    pair["window"] = state.row_window[sensor.rows_of(pair["frame"].to_numpy(), pair["def_player_id"].to_numpy())]
    pair = pair[pair["window"] >= 0]
    start = dict(zip(tables["possessions"]["possession"].astype(int), tables["possessions"]["start_frame"].astype(int)))
    out = Compared("ghosting")
    for possession, rows in pair.groupby("possession", sort=True):
        first, last = start[int(possession)], int(rows["frame"].max())
        r0, r1 = sensor.rows_between(first, last)
        grid = r0 + np.flatnonzero((sensor.frame[r0:r1] - first) % GHOST_STRIDE == 0)   # the possession's 5 Hz frames
        n, filled = state.fill(grid)
        s, c = _r(rows["sensor"].mean(), 6), _r(rows["broadcast"].mean(), 6)
        out.items.append(S.SwapItem(
            kind="ghosting", window=state.windows[int(np.bincount(rows["window"].to_numpy()).argmax())].name,
            possession=int(possession), key=f"possession-{int(possession)}", t=None,
            sensor=s, broadcast=c, abs_diff=_r(abs(c - s), 6), fill_share=_r(filled / n, 6)))
    out.facts = {"ghost_rows": int(len(a)), "rows_in_windows": int(len(pair)), "items": len(out.items),
                 "stored_rows": int(len(stored)), "stored_gap": gap, "rows_on_one_side_only": one_side}
    return out


SWAPS = {"xfg": xfg_swap, "epv": epv_swap, "ghosting": ghost_swap}


def compare(models: dict, state: State) -> dict:
    """Run every available model on both tables -> {kind: Compared}. A model that is not available gets a
    Compared with its note and no items."""
    out = {}
    for kind in S.MODEL_KINDS:
        model = models.get(kind) or Model(kind, note=f"no rebuilt {kind} model was given")
        if model.predictor is None:
            out[kind] = Compared(kind, note=model.note or f"the frozen {kind} model is not available")
            continue
        if rebuild_check(kind, model.check) is None or not model.check.get("passed"):
            raise BridgeProblem(f"{kind}: a predictor without a passed rebuild check cannot be shown")
        out[kind] = SWAPS[kind](model, state)
        if not out[kind].items:
            out[kind].note = f"no {kind} item lies inside a window"
    return out


# --------------------------------------------------------------------------------------------- swap.json

def _summary(result: Compared) -> S.SwapSummary:
    kind, unit = result.kind, UNITS[result.kind]
    if result.note:
        return S.SwapSummary(kind=kind, n=0, unit=unit, available=False, note=result.note)
    diffs = np.array([item.abs_diff for item in result.items], dtype=float)
    facts = result.facts
    note = (f"Sensor-side values equal the stored held-out predictions on all {facts['stored_rows']} stored rows of "
            f"the game: largest gap {facts['stored_gap']:.3g} {unit}, tolerance {TOLERANCE_CAP[kind]:g}.")
    return S.SwapSummary(kind=kind, n=len(diffs), mean_abs_diff=_r(diffs.mean(), 6), median_abs_diff=_r(np.median(diffs), 6),
                         unit=unit, available=True, note=note)


def method_statements(state: State, fill: dict, results: dict, max_gap_s: float) -> list:
    """The method as plain statements. Every number is computed here from the data."""
    card, sync, matches = state.card, state.sync, state.matches
    n_rows = sum(len(w.frame) for w in state.windows)
    text = [
        "An illustration, not a result: the frozen models are run once on sensor positions and once on broadcast "
        "positions of the same game. Player identity, team and the ball come from the sensor data on both sides, so "
        "a difference measures position error only.",
        f"Broadcast build {card['build']} on {len(state.windows)} windows of the held-out game {card['game']} "
        f"({state.game_id}): {n_rows} rows at about ten a second, {sync['synced']} of them with a synced sensor moment.",
        "Court position of a box: its stabilised foot point through that row's homography.",
        f"Matching: on every synced row the boxes are assigned one to one to the ten sensor players of that moment "
        f"({state.matcher_name}), and a pair is kept when it is within {card['match_ft']:g} ft, the matching "
        f"distance of the scorecard ({card['path']}, method.match_ft). The distance was read, not tuned.",
        "This is not the scorecard's own measurement: the scorecard matches through a truth homography on the rows "
        "that have one, the bridge matches every synced row through the build's own homography. Pairs farther apart "
        "than the matching distance are not matched, so the matched distance is biased low.",
        f"{matches['matched']} of {matches['boxes']} boxes on synced rows were matched ({matches['player_slots']} "
        "sensor player slots). A matched player takes the broadcast position at that row, an unmatched box is dropped."
        + (f" Median matched distance: {np.median(state.sample_ft):.2f} ft." if len(state.sample_ft) else ""),
        f"Between two consecutive broadcast samples of one player at most {max_gap_s:g} s apart the position is "
        f"interpolated linearly onto the 25 Hz sensor frames. The {max_gap_s:g} s was fixed before the bridge ran.",
        "Every other player-frame inside the windows keeps the sensor position and is counted as filled. A window "
        "is the stretch of sensor frames from its first to its last synced row, gaps included.",
        f"Fill share: {fill['n_filled']} of {fill['n_player_frames']} player-frames inside the windows are filled "
        f"({fill['n_frames']} sensor frames in {fill['n_windows']} windows).",
        f"Frame mapping: sensor frame = the quarter's frame with the synced moment's index. Checked on all "
        f"{sync['synced']} synced rows against the game clock: largest gap {sync['max_clock_gap_s']:.3f} s, and no "
        "other frame of the quarter is nearer to a row's clock."
        + (f" {sync['duplicates']} rows share their sensor frame with another row and are not used for positions."
           if sync["duplicates"] else ""),
        "Where a player switches between a broadcast position and a filled sensor position his path jumps by the "
        "position error, and a speed measured across the switch includes that jump.",
        "Fill share of an item: the share of filled positions among the player-frames the model read for it. xFG: "
        "all ten players from the start of the lookback to the release. EPV: the possession's states up to this "
        "one. Ghosting: the possession's 5 Hz frames up to its last compared ghost. A frame outside every window "
        "counts as filled.",
    ]
    xfg, epv, ghost = (results[k] for k in S.MODEL_KINDS)
    if not xfg.note:
        f = xfg.facts
        text.append(
            f"xFG: {f['measured']} of the game's {f['attempts']} attempts have contest features on the sensor table, "
            f"{f['release_in_window']} of them with the release inside a window, {f['items']} with the lookback "
            "inside the same window too. The lookback is what the frozen feature code reads on the sensor table: the "
            "release frame, the frame 0.4 s before it and the touch window. Shot location comes from the shot chart "
            "and is the same on both sides."
            + (f" On broadcast positions {f['lost_contest']} of these attempts lost their contest features."
               if f["lost_contest"] else "")
            + (f" The frozen cleaning dropped {f['rows_dropped_by_cleaning']['broadcast']} rows of the broadcast "
               f"table and {f['rows_dropped_by_cleaning']['sensor']} of the sensor table."))
    if not epv.note:
        f = epv.facts
        text.append(f"EPV: one item per 1 Hz state whose frame lies inside a window, {f['items']} of the game's "
                    f"{f['states']} states. A state's value reads its possession from the first state on, so a "
                    "possession that began before the window keeps sensor positions there."
                    + (f" {f['states_on_one_side_only']} states exist on one table only and are left out."
                       if f["states_on_one_side_only"] else ""))
    if not ghost.note:
        f = ghost.facts
        text.append(f"Ghosting: one item per possession, the mean distance between the real defenders and their "
                    f"{GHOST_HORIZON_S:g} s ghosts over the ghost rows inside the windows ({f['rows_in_windows']} of "
                    f"{f['ghost_rows']} rows), an average defense, never the best one. On the broadcast side both "
                    "the defender and what the model read come from the broadcast table.")
    for result in results.values():
        if result.note:
            text.append(f"{result.kind}: not available. {result.note}")
    text.append("Nothing was fitted, tuned or scored, and no outcome was read: predictions are compared with "
                "predictions. A sensor-side value that did not equal the stored held-out prediction would have "
                "stopped the run.")
    return text


def build_swap(state: State, results: dict, models: dict, max_gap_s: float = MAX_GAP_S) -> S.DemoSwap:
    fill = overall_fill(state)
    checks = [c for c in (rebuild_check(k, (models.get(k) or Model(k)).check) for k in S.MODEL_KINDS) if c is not None]
    shown = [k for k in S.MODEL_KINDS if not results[k].note]
    sources = data_sources(state.game_id, state.tables["season"], state.card, shown)
    if not checks:
        sources.pop("checks")
    items = [item for kind in S.MODEL_KINDS for item in results[kind].items]
    return S.DemoSwap(
        schema_version=S.SCHEMA_VERSION, label="illustration", game_id=state.game_id, game=state.card["game"],
        build=state.card["build"], n_windows=fill["n_windows"], n_frames=fill["n_frames"],
        fill_share=_r(fill["fill_share"], 6), match_ft=state.card["match_ft"], max_gap_s=max_gap_s,
        summaries=[_summary(results[k]) for k in S.MODEL_KINDS], items=items, possessions=possession_fill(state),
        checks=checks, method=method_statements(state, fill, results, max_gap_s), sources=list(sources.values()))


def demo_possessions(state: State, swap: S.DemoSwap) -> dict:
    """What <artifacts>/demo/possessions.json holds: the possessions of the game that share a frame with a window."""
    return {"what": "Possessions of the held-out demo game that share at least one sensor frame with a window of the "
                    "broadcast build. Written by export/demo_bridge.py; the site data marks them.",
            "game_id": state.game_id, "game": state.card["game"], "build": state.card["build"],
            "possessions": [{"game_id": state.game_id, "possession": p.possession, "window": p.window,
                             "n_player_frames": p.n_player_frames, "fill_share": p.fill_share} for p in swap.possessions]}


# ------------------------------------------------------------------------------------------------ scenes

def scene_inputs(windows: list) -> list:
    """What the scene rule may read: for every window its order, frame rate, stride, and for every row its frame
    number and whether it has a synced moment. No position, no homography, no distance."""
    return [{"order": w.order, "fps": w.fps, "stride": w.stride, "frame": [int(f) for f in w.frame],
             "synced": [bool(s) for s in w.synced]} for w in windows]


def stretches(frame: list, synced: list, stride: int) -> list:
    """Runs of at least two consecutive rows, each with a synced moment and one stride after the row before it
    -> [(index of the first row, index of the last row)]."""
    out, first = [], None
    for i in range(len(frame)):
        follows = first is not None and frame[i] - frame[i - 1] == stride
        if synced[i] and follows:
            continue
        if first is not None and i - 1 > first:
            out.append((first, i - 1))
        first = i if synced[i] else None
    if first is not None and len(frame) - 1 > first:
        out.append((first, len(frame) - 1))
    return out


def choose_scenes(inputs: list, max_scenes: int = SCENES_MAX, scene_s: float = SCENE_MAX_S) -> list:
    """The scene rule (D50): the longest stretches, the earlier window first on a tie, each cut to the first
    `scene_s` seconds of its stretch. Reads only what scene_inputs gives: no error value can reach it."""
    found = []
    for w in inputs:
        for first, last in stretches(w["frame"], w["synced"], w["stride"]):
            found.append(((w["frame"][last] - w["frame"][first]) / w["fps"], w["order"], first, last))
    found.sort(key=lambda s: (-s[0], s[1], s[2]))
    by_order = {w["order"]: w for w in inputs}
    scenes = []
    for rank, (_, order, first, stretch_last) in enumerate(found[:max_scenes], 1):
        w = by_order[order]
        last = first
        while last < stretch_last and (w["frame"][last + 1] - w["frame"][first] + 1) / w["fps"] <= scene_s:
            last += 1
        n = w["frame"][last] - w["frame"][first] + 1
        scenes.append({"id": f"scene-{rank}", "window": order, "first": first, "last": last, "n_frames": n,
                       "duration_s": n / w["fps"]})
    if any(s["duration_s"] > CLIP_CAP_S for s in scenes) or sum(s["duration_s"] for s in scenes) > TOTAL_CAP_S:
        raise BridgeProblem("the scenes break the clip caps of PLAN.md M2")
    return scenes


def scene_plan(state: State, scene: dict, cv_root=None) -> dict:
    """Where a scene sits in its window clip. The build computed row `frame` on clip frame frame + late (its
    decoder ran `late` frames behind after a seek), so the scene's clip starts at source frame first + late and
    the rows keep their spacing. The offsets were calibrated on the first build of these clips; they hold for
    this build only when its boxes are that build's boxes, which is checked here when `cv_root` is given."""
    w = state.windows[scene["window"]]
    first, last = scene["first"], scene["last"]
    late = {w.decode_offset(int(w.frame[i])) for i in range(first, last + 1)}
    if len(late) != 1:
        raise BridgeProblem(f"{scene['id']}: the decode offset changes inside the scene")
    segment = max([start for start, _ in w.decode if start <= w.frame[first]], default=None)
    if segment in w.uncalibrated:
        raise BridgeProblem(f"{scene['id']}: the decode offset of this stretch was never calibrated")
    if cv_root is not None and build_folder(state.card["build_dir"]) != "build":
        reference = Path(cv_root) / "build" / f"{w.name}_frames.json"
        if not reference.exists():
            raise BridgeProblem(f"{scene['id']}: the build the decode offsets were calibrated on is missing")
        rows = {int(r["frame"]): [b["bbox"] for b in r["boxes"]] for r in _json(reference)["frames"]}
        for i in range(first, last + 1):
            if rows.get(int(w.frame[i])) != [b["bbox"] for b in w.rows[i]["boxes"]]:
                raise BridgeProblem(f"{scene['id']}: this build's boxes are not the boxes the decode offsets were "
                                    "calibrated on, so the offsets cannot place them on the clip")
    late = late.pop()
    return dict(scene, late=late, source_first=int(w.frame[first]) + late,
                still_row=first + (last - first) // 2)          # the still is the middle row of the scene


def plan_scenes(state: State, cv_root=None) -> list:
    return [scene_plan(state, scene, cv_root) for scene in choose_scenes(scene_inputs(state.windows))]


def scene_frames(state: State, scene: dict, scale: float = 1.0) -> tuple:
    """The sampled frames of a scene and the matched distances behind its error figures."""
    w, sensor = state.windows[scene["window"]], state.sensor
    frames, distances, off_court = [], [], 0
    for i in range(scene["first"], scene["last"] + 1):
        court, pair_row, pair_ft = w.court[i], w.pair_row[i], w.pair_ft[i]
        boxes, points = [], []
        for b, box in enumerate(w.rows[i]["boxes"]):
            x1, y1, x2, y2 = (float(v) * scale for v in box["bbox"])
            row = int(pair_row[b])
            player = int(sensor.player[row]) if row >= 0 else None
            boxes.append(S.DemoBox(x=_r(x1, 1), y=_r(y1, 1), w=_r(x2 - x1, 1), h=_r(y2 - y1, 1), track=int(box["tid"]),
                                   status="matched" if row >= 0 else "unmatched", player_id=player))
            near = np.isfinite(court[b]).all() \
                and -COURT_MARGIN_FT <= court[b, 0] <= S.COURT_LENGTH_FT + COURT_MARGIN_FT \
                and -COURT_MARGIN_FT <= court[b, 1] <= S.COURT_WIDTH_FT + COURT_MARGIN_FT
            if row >= 0 or near:
                team = sensor.team_name.get(int(sensor.team_id[row])) if row >= 0 else None
                points.append(S.DemoPoint(player_id=player, team=team, x=_r(court[b, 0], 2), y=_r(court[b, 1], 2)))
            else:
                off_court += 1
        truth = [S.DemoPoint(player_id=int(sensor.player[r]), team=sensor.team_name.get(int(sensor.team_id[r])),
                             x=_r(sensor.x[r], 2), y=_r(sensor.y[r], 2)) for r in sensor.rows_at(int(w.pos[i]))]
        matched = pair_ft[pair_row >= 0]
        distances.extend(matched.tolist())
        frames.append(S.DemoFrame(t=_r((w.frame[i] - w.frame[scene["first"]]) / w.fps, 3), boxes=boxes, broadcast=points,
                                  truth=truth, error_ft=_r(np.median(matched), 2) if len(matched) else None))
    return frames, np.array(distances, dtype=float), off_court


def build_frames(state: State, scenes: list, clips: dict) -> S.DemoFrames:
    """demo/frames.json. `clips` gives, per scene id, the clip's width, height and frame rate and the factor from
    source pixels to clip pixels."""
    card = state.card
    out, off_court = [], 0
    for scene in scenes:
        w, clip = state.windows[scene["window"]], clips[scene["id"]]
        frames, distances, dropped = scene_frames(state, scene, clip["scale"])
        off_court += dropped
        first = scene["first"]
        out.append(S.DemoScene(
            id=scene["id"], window=w.name, period=int(w.period[first]), start_clock_s=_r(w.clock[first], 2),
            clip=f"{scene['id']}.mp4", still=f"stills/{scene['id']}.jpg",
            still_t=frames[scene["still_row"] - first].t, duration_s=_r(scene["duration_s"], 3), fps=_r(clip["fps"], 3),
            width=int(clip["width"]), height=int(clip["height"]), n_matched=int(len(distances)),
            p50_error_ft=_r(np.median(distances), 2) if len(distances) else None,
            p90_error_ft=_r(np.quantile(distances, 0.9), 2) if len(distances) else None, frames=frames))
    notes = [
        "Each scene is a stretch of one window of the held-out game. The private site plays the clip, the public "
        "site shows the still.",
        "frames holds the build's rows, about ten a second. t is the time of a row in seconds from the first frame "
        "of the clip; the boxes of a row belong to the clip frame at that time. fps, width and height describe the clip.",
        f"A box is matched when it was assigned to a sensor player within {card['match_ft']:g} ft, the matching "
        "distance of the scorecard; otherwise it is unmatched (the scorecard calls such a box a ghost box). "
        "broadcast holds the court position of every matched box and of every unmatched box that lands within "
        f"{COURT_MARGIN_FT:g} ft of the court; truth holds the ten sensor players of the synced moment.",
        "error_ft, p50_error_ft and p90_error_ft are distances between matched pairs through the build's own "
        "homography. Pairs farther apart than the matching distance are not matched, so these values are biased low "
        "and are not the scorecard's measurement.",
    ]
    if card.get("p50_ft") is not None and card.get("p90_ft") is not None:
        basis = (f" over {card['matched_players']} matched players on {card['testable']} testable rows"
                 if card.get("matched_players") is not None and card.get("testable") is not None else "")
        notes.append(f"Show a scene's error beside the whole-game value of the scorecard for build {card['build']}: "
                     f"p50 {card['p50_ft']:g} ft and p90 {card['p90_ft']:g} ft{basis} ({card['path']}).")
    if off_court:
        notes.append(f"{off_court} unmatched boxes land farther than {COURT_MARGIN_FT:g} ft from the court and have "
                     "no court position in broadcast.")
    notes.append("The build read some stretches a few frames late. The clip of a scene starts at the frame the build "
                 "really read for the scene's first row, so t needs no correction. The still is the frame of the "
                 "scene's middle row, at still_t.")
    sources = data_sources(state.game_id, state.tables["season"], card, ())
    keep = ("scorecard", "sidecars", "truth", "timemaps", "frames")
    listed = [sources[name] for name in keep] + [
        S.Source(path="data/sportvu/build/<window>.mp4", what="the window clips the scenes are cut from (local files, "
                 "not in the repository)", rows="heldout"),
        S.Source(path="data/sportvu/replay_offsets.json", what="frames the build's decoder ran late per seek segment "
                 "(local file, not in the repository)", rows="heldout")]
    return S.DemoFrames(schema_version=S.SCHEMA_VERSION, game_id=state.game_id, game=card["game"], build=card["build"],
                        total_clip_s=_r(sum(s["duration_s"] for s in scenes), 3),
                        scene_rule=SCENE_RULE.format(n=SCENES_MAX, s=SCENE_MAX_S), scenes=out, notes=notes, sources=listed)


# ------------------------------------------------------------------------------------- clips, stills, checks

def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise BridgeProblem(f"{name} is not on PATH, so clips and stills cannot be made")
    return path


def clip_shape(source, max_width: int = CLIP_MAX_WIDTH) -> dict:
    """Frame rate of a window clip and the size its scenes are cut to: at most `max_width` wide, never enlarged,
    the height even. `scale` takes a source pixel to a clip pixel."""
    import cv2
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise BridgeProblem(f"cannot open the window clip {Path(source).name}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise BridgeProblem(f"the window clip {Path(source).name} has no readable size or frame rate")
    out_width = min(max_width, width)
    out_height = int(round(height * out_width / width / 2.0)) * 2
    return {"width": out_width, "height": out_height, "fps": fps, "scale": out_width / width, "source_frames": frames}


def probe(path) -> dict:
    """What ffprobe says about a clip: its one video stream, its duration and how many audio streams it has."""
    done = subprocess.run([_tool("ffprobe"), "-v", "error", "-show_entries",
                           "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,pix_fmt:format=duration",
                           "-of", "json", str(path)], capture_output=True, text=True, check=True)
    doc = json.loads(done.stdout)
    video = [s for s in doc.get("streams", []) if s.get("codec_type") == "video"]
    if len(video) != 1:
        raise BridgeProblem(f"{Path(path).name} does not hold exactly one video stream")
    top, bottom = (float(v) for v in (video[0].get("avg_frame_rate") or "0/1").split("/"))
    return {"codec": video[0].get("codec_name"), "width": int(video[0]["width"]), "height": int(video[0]["height"]),
            "pix_fmt": video[0].get("pix_fmt"), "fps": top / bottom if bottom else 0.0,
            "frames": int(video[0]["nb_frames"]) if str(video[0].get("nb_frames", "")).isdigit() else None,
            "duration_s": float(doc["format"]["duration"]),
            "audio_streams": sum(s.get("codec_type") == "audio" for s in doc.get("streams", []))}


def index_first(path) -> bool:
    """True when an mp4 carries its index (the moov box) before its media (the mdat box), which lets a browser
    start playing before the whole file has arrived."""
    order = []
    with open(path, "rb") as handle:
        while True:
            head = handle.read(8)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head)
            order.append(kind.decode("latin-1"))
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0] - 8
            if size < 8:
                break
            handle.seek(size - 8, 1)
    return "moov" in order and "mdat" in order and order.index("moov") < order.index("mdat")


def cut_clip(source, target, first: int, n_frames: int, max_width: int = CLIP_MAX_WIDTH) -> None:
    """Source frames [first, first + n_frames) as an H.264 mp4 without audio, at most `max_width` wide, with the
    index at the front. Frames are selected by number, so the cut is exact; none is dropped or repeated."""
    chain = f"trim=start_frame={first}:end_frame={first + n_frames},setpts=PTS-STARTPTS,scale='min({max_width},iw)':-2"
    subprocess.run([_tool("ffmpeg"), "-y", "-v", "error", "-i", str(source), "-an", "-vf", chain, "-fps_mode", "passthrough",
                    "-map_metadata", "-1", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(target)], check=True)


def cut_still(source, target, frame: int, max_width: int = CLIP_MAX_WIDTH) -> None:
    """One source frame, selected by number, as a JPEG at most `max_width` wide."""
    chain = f"select=eq(n\\,{frame}),scale='min({max_width},iw)':-2"
    subprocess.run([_tool("ffmpeg"), "-y", "-v", "error", "-i", str(source), "-vf", chain, "-fps_mode", "passthrough",
                    "-frames:v", "1", "-q:v", "3", "-update", "1", str(target)], check=True)


def read_gray(path, wanted, size=None) -> dict:
    """Frames of a video by number -> {number: gray image}, decoded in order from the first frame (never a seek,
    which can land late). `size` (width, height) resizes every frame."""
    import cv2
    wanted = {int(i) for i in wanted}
    out, last = {}, max(wanted, default=-1)
    cap = cv2.VideoCapture(str(path))
    try:
        for number in range(last + 1):
            ok, image = cap.read()
            if not ok:
                break
            if number in wanted:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                out[number] = cv2.resize(gray, size, interpolation=cv2.INTER_AREA) if size else gray
    finally:
        cap.release()
    return out


def motion_scores(frames: dict, rows: list, shifts=SHIFTS, step: int = 2) -> dict:
    """How well boxes sit on moving regions, for each frame shift.

    `rows` is [(clip frame number of a row, its boxes as (x, y, w, h) in the pixels of `frames`)]. Motion at
    frame n is |image(n + step) - image(n - step)|. A box scores the mean motion inside it minus the mean motion
    in the band around it (half the box width wide). For shift d every box of a row is laid on the motion of
    frame n + d; the result is the mean score over all boxes. Boxes that follow the players score highest at
    the shift that is the true alignment. Only rows that have every shifted frame are used, so all shifts are
    scored on the same boxes."""
    import cv2
    sums, n_boxes = {int(d): 0.0 for d in shifts}, 0
    for number, boxes in rows:
        need = [number + d + s for d in shifts for s in (-step, step)]
        if not boxes or any(f not in frames for f in need):
            continue
        for d in shifts:
            motion = cv2.absdiff(frames[number + d + step], frames[number + d - step]).astype(np.float32)
            height, width = motion.shape
            for x, y, w, h in boxes:
                x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
                band = max(1, (x1 - x0) // 2)
                ax0, ay0, ax1, ay1 = max(0, x0 - band), max(0, y0 - band), min(width, x1 + band), min(height, y1 + band)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                inner, outer = motion[y0:y1, x0:x1], motion[ay0:ay1, ax0:ax1]
                around = (outer.sum() - inner.sum()) / max(outer.size - inner.size, 1)
                sums[int(d)] += float(inner.mean() - around)
                n_boxes += d == shifts[0]
    return {"boxes": int(n_boxes), "scores": {str(d): (sums[d] / n_boxes if n_boxes else None) for d in sums}}


def nearest_shift(images: dict, reference: dict, first: int, shifts=(-2, -1, 0, 1, 2)) -> dict:
    """Mean absolute difference between image j of a cut and reference frame first + j + shift, for each shift,
    over the given images. The cut is exact when shift 0 is the smallest."""
    out = {}
    for shift in shifts:
        diffs = [float(np.abs(image.astype(np.int16) - reference[first + j + shift].astype(np.int16)).mean())
                 for j, image in images.items() if first + j + shift in reference]
        out[str(shift)] = float(np.mean(diffs)) if diffs else None
    return out


def _best(scores: dict, largest: bool):
    known = {k: v for k, v in scores.items() if v is not None}
    if not known:
        return None
    return (max if largest else min)(known, key=known.get)


def make_media(state: State, scenes: list, clips: dict, cv_root, media, stills, every: int = 2) -> dict:
    """Cut the clip and the still of every scene and measure them. Returns the record written to
    <artifacts>/demo/media_check.json; raises BridgeProblem when a clip breaks a cap or a check fails.

    Checks per scene: the clip is H.264 without audio, no wider than the cap, with its index first, as long as
    planned and within the caps; frame j of the clip is source frame first + j (compared as images at shifts of
    up to two frames); the still is the source frame of its row; and the boxes sit on moving regions at the
    alignment in the data better than at any other tested shift."""
    import cv2
    media, stills, cv_root = Path(media), Path(stills), Path(cv_root)
    media.mkdir(parents=True, exist_ok=True)
    stills.mkdir(parents=True, exist_ok=True)
    report = {"what": "Measurements on the clips and stills of the demo scenes (export/demo_bridge.py).",
              "shifts_are": "frames added to the alignment in the data; 0 must win", "scenes": {}}
    for scene in scenes:
        w, shape = state.windows[scene["window"]], clips[scene["id"]]
        source = cv_root / "build" / f"{w.name}.mp4"
        clip, still = media / f"{scene['id']}.mp4", stills / f"{scene['id']}.jpg"
        first, n = scene["source_first"], scene["n_frames"]
        if first + n > shape["source_frames"]:
            raise BridgeProblem(f"{scene['id']}: the scene runs past the end of its window clip")
        cut_clip(source, clip, first, n)
        still_at = int(w.frame[scene["still_row"]] - w.frame[scene["first"]])
        cut_still(source, still, first + still_at)
        seen = probe(clip)
        problems = []
        if seen["codec"] != "h264" or seen["audio_streams"] or seen["pix_fmt"] != "yuv420p":
            problems.append("the clip is not H.264 4:2:0 video without audio")
        if (seen["width"], seen["height"]) != (shape["width"], shape["height"]) or seen["width"] > CLIP_MAX_WIDTH:
            problems.append("the clip does not have the planned size")
        if seen["frames"] != n or abs(seen["duration_s"] - scene["duration_s"]) > 1.0 / shape["fps"]:
            problems.append("the clip does not have the planned length")
        if seen["duration_s"] > min(SCENE_MAX_S, CLIP_CAP_S) + 1e-6:
            problems.append("the clip is longer than the cap")
        if not index_first(clip):
            problems.append("the clip's index is not at its front")
        size = (shape["width"] // 2, shape["height"] // 2)
        rows = [(int(w.frame[i] - w.frame[scene["first"]]),
                 [(b["bbox"][0] * shape["scale"] / 2, b["bbox"][1] * shape["scale"] / 2,
                   (b["bbox"][2] - b["bbox"][0]) * shape["scale"] / 2, (b["bbox"][3] - b["bbox"][1]) * shape["scale"] / 2)
                  for b in w.rows[i]["boxes"]]) for i in range(scene["first"], scene["last"] + 1, every)]
        probes = sorted({int(round(v)) for v in np.linspace(0, n - 1, 9)})
        wanted = set(probes) | {number + d + s for number, _ in rows for d in SHIFTS for s in (-2, 2)}
        frames = read_gray(clip, {f for f in wanted if 0 <= f < n}, size)
        reference = read_gray(source, {first + j + s for j in probes + [still_at] for s in (-2, -1, 0, 1, 2)
                                       if first + j + s >= 0}, size)
        cut = nearest_shift({j: frames[j] for j in probes if j in frames}, reference, first)
        motion = motion_scores(frames, rows)
        picture = cv2.imread(str(still), cv2.IMREAD_GRAYSCALE)
        if picture is None or (picture.shape[1], picture.shape[0]) != (shape["width"], shape["height"]):
            problems.append("the still is not a picture of the clip's size")
            still_cut = {}
        else:
            still_cut = nearest_shift({still_at: cv2.resize(picture, size, interpolation=cv2.INTER_AREA)}, reference, first)
            if _best(still_cut, largest=False) != "0":
                problems.append("the still is not the frame of its row")
        if _best(cut, largest=False) != "0":
            problems.append("the clip is not cut at the planned frame")
        if not motion["boxes"] or _best(motion["scores"], largest=True) != "0":
            problems.append("the boxes do not sit best on the moving regions at the alignment in the data")
        report["scenes"][scene["id"]] = {
            "window": w.name, "decoder_late_frames": scene["late"], "source_first_frame": first, "frames": n,
            "clip": seen, "index_first": index_first(clip), "clip_bytes": clip.stat().st_size,
            "still_bytes": still.stat().st_size, "cut_mean_abs_diff_by_shift": cut,
            "still_mean_abs_diff_by_shift": still_cut, "motion_inside_minus_around_by_shift": motion["scores"],
            "motion_boxes": motion["boxes"], "problems": problems, "passed": not problems}
    total = sum(s["clip"]["duration_s"] for s in report["scenes"].values())
    report["total_clip_s"] = round(total, 3)
    report["caps"] = {"scene_s": SCENE_MAX_S, "clip_s": CLIP_CAP_S, "total_s": TOTAL_CAP_S, "width": CLIP_MAX_WIDTH}
    report["passed"] = total <= TOTAL_CAP_S and all(s["passed"] for s in report["scenes"].values())
    return report


# --------------------------------------------------------------------------------------------- the run

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(cv_root, out, artifacts=None, media=None, stills=None, game: str = GAME, build: str = BUILD, data_root=None,
        models=None, clips: bool = True, repo=REPO_ROOT, say=print) -> dict:
    """The whole bridge. `models` ({kind: Model}) replaces the rebuilt models of the artifacts folder (tests give
    stand-ins); `clips` false skips the clips and stills and writes the data only. Returns what was written and
    the headline numbers."""
    from analytics.eval import is_heldout_game
    started = time.time()
    data_root = Path(data_root or os.environ.get("DATA_ROOT") or "")
    if not str(data_root) or not data_root.is_dir():
        raise BridgeProblem("DATA_ROOT is not set or is not a folder")
    cv_root = Path(cv_root)
    if not cv_root.is_dir():
        raise BridgeProblem("the broadcast outputs folder (--cv-root) does not exist")
    card = read_scorecard(game, build, repo)
    if not is_heldout_game(card["game_id"]):
        raise BridgeProblem(f"{card['game_id']} is not a held-out game: the frozen models were fitted on it")
    if clips and (media is None or stills is None):
        raise BridgeProblem("clips and stills need their two folders (media and stills)")
    out = assert_safe_output(out, data_root)
    demo = assert_safe_output(artifacts_dir(artifacts) / "demo", data_root)
    targets = [out, demo]
    if clips:
        media, stills = assert_safe_output(media, data_root), assert_safe_output(stills, data_root)
        targets += [media, stills]
    for target in targets:
        if target == cv_root.resolve() or cv_root.resolve() in target.parents:
            raise BridgeProblem("an output folder lies inside the broadcast outputs folder, which is read only")
    windows = load_windows(cv_root, card["game"], build_folder(card["build_dir"]))
    tables = load_tables(data_root, card["game_id"])
    state = prepare(card, tables, windows, MAX_GAP_S, data_root)
    fill = overall_fill(state)
    say(f"[bridge] {len(windows)} windows, {state.sync['synced']} synced rows, {state.matches['matched']} matched "
        f"boxes, fill share {fill['fill_share']:.3f} over {fill['n_player_frames']} player-frames")
    if models is None:
        models = {kind: load_model(kind, artifacts) for kind in S.MODEL_KINDS}
    for kind in S.MODEL_KINDS:
        if models.get(kind) is None or models[kind].predictor is None:
            say(f"[bridge] {kind}: not available: {(models.get(kind) or Model(kind)).note}")
    results = compare(models, state)
    swap = build_swap(state, results, models)
    scenes = plan_scenes(state, cv_root)
    shapes = {scene["id"]: clip_shape(cv_root / "build" / f"{state.windows[scene['window']].name}.mp4") for scene in scenes}
    frames = build_frames(state, scenes, shapes)
    check = None
    if clips:
        check = make_media(state, scenes, shapes, cv_root, media, stills)
        _write(demo / "media_check.json", json.dumps(check, indent=2, allow_nan=False) + "\n")
        if not check["passed"]:
            bad = {name: s["problems"] for name, s in check["scenes"].items() if s["problems"]}
            raise BridgeProblem(f"the clips or stills failed a check, so no data file was written: {bad}")
    written = {}
    for rel, model in (("demo/swap.json", swap), ("demo/frames.json", frames)):
        text = dump(model)
        private = S.find_private(text)
        if private:
            raise BridgeProblem(f"{rel} would hold a private string: {private}")
        _write(out / rel, text)
        problems = S.validate_file(out, rel)
        if problems:
            raise BridgeProblem(f"{rel} does not validate against the contract: {problems[:3]}")
        written[rel] = len(text.encode("utf-8"))
    _write(demo / "possessions.json", json.dumps(demo_possessions(state, swap), indent=2, allow_nan=False) + "\n")
    report = {"game_id": state.game_id, "build": card["build"], "match_ft": card["match_ft"], "max_gap_s": MAX_GAP_S,
              "sync": state.sync, "matches": state.matches, "fill": fill,
              "summaries": [s.model_dump(mode="json") for s in swap.summaries],
              "facts": {kind: result.facts for kind, result in results.items()},
              "possessions": len(swap.possessions),
              "scenes": [{"id": s.id, "window": s.window, "duration_s": s.duration_s, "frames": len(s.frames),
                          "n_matched": s.n_matched, "p50_error_ft": s.p50_error_ft, "p90_error_ft": s.p90_error_ft}
                         for s in frames.scenes],
              "total_clip_s": frames.total_clip_s, "media_checked": None if check is None else check["passed"],
              "written": written, "seconds": round(time.time() - started, 1)}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m export.demo_bridge", description=__doc__.split("\n")[0])
    parser.add_argument("--cv-root", default=os.environ.get("V3_CV_ROOT") or str(REPO_ROOT / "data" / "sportvu"),
                        help="the local outputs of the broadcast build (default: V3_CV_ROOT, else data/sportvu)")
    parser.add_argument("--out", default=str(REPO_ROOT / "web" / "data"),
                        help="the site data root; demo/swap.json and demo/frames.json are written under it")
    parser.add_argument("--artifacts", help="the artifacts folder (default: V3_ARTIFACTS, else artifacts/)")
    parser.add_argument("--media", default=str(REPO_ROOT / "web" / "media-private"),
                        help="the folder of private clips (never committed)")
    parser.add_argument("--stills", default=str(REPO_ROOT / "web" / "public" / "stills"),
                        help="the folder of public stills")
    parser.add_argument("--game", default=GAME, help="the game tag of the scorecard (default %(default)s)")
    parser.add_argument("--build", default=BUILD, help="the broadcast build (default %(default)s, PLAN.md D42)")
    parser.add_argument("--no-media", action="store_true", help="write the data files only, no clip and no still")
    args = parser.parse_args(argv)
    try:
        with heavy_job("demo bridge"):
            report = run(args.cv_root, args.out, args.artifacts, args.media, args.stills, args.game, args.build,
                         clips=not args.no_media)
    except BridgeProblem as problem:
        print(f"{WHO}: {problem}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""export/site_data.py and its helper modules (PLAN.md M1).

Two kinds of test, both fast and without DATA_ROOT:
- on a small synthetic data root in the real layout (tests/_fx_site_data.py): the tree validates, only held-out
  games become games, only out-of-fold rows feed player values, the featured picks follow D48, the minimum-sample
  logic, ghost and EPV alignment by frame, the shot fallbacks, no private string in any output;
- on the committed reports: every text and number of results.json equals reports/results.md, and every value of
  scorecard.json equals the scorecard files.
"""

import copy
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics import results as R
from analytics.possessions import GAP_S
from export import sample
from export import schemas as S
from export import site_common as C
from export import site_data
from export import site_games
from export import site_players
from export import site_reports
from tests import _fx_site_data as FX
from tools import frozen_common

REPO = Path(__file__).resolve().parents[1]
QUIET = lambda message: None  # noqa: E731


@dataclass
class Built:
    base: Path
    data: Path
    artifacts: Path
    out: Path
    tables: dict
    problems: list
    info: dict
    checks: Path                      # the tracked copy of the rebuild check records of the synthetic artifacts


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The synthetic data root, its rebuilt outputs, and one full run of the export on them."""
    base = tmp_path_factory.mktemp("site_data")
    tables = FX.write_data_root(base / "data", site_players.read_baseline(REPO)[2])
    FX.write_artifacts(base / "artifacts", base / "data", tables)
    checks = FX.tracked_checks(base / "artifacts")
    problems, info = site_data.run(base / "out", base / "data", base / "artifacts", log=QUIET, allow_fixture_root=True,
                                   checks_copy=checks)
    return Built(base, base / "data", base / "artifacts", base / "out", tables, problems, info, checks)


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slug(game: str) -> str:
    return site_games.game_of(game).slug


def _possession(built, game: str, number: int) -> dict:
    return _json(built.out / "games" / _slug(game) / f"{number}.json")


def _index(built, game: str) -> dict:
    return _json(built.out / "games" / _slug(game) / "possessions.json")


def _row(built, game: str, number: int) -> dict:
    return next(r for r in _index(built, game)["possessions"] if r["possession"] == number)


def _list(values, digits: int) -> list:
    """Rounded values with None for a missing one, the way a contract file writes a track."""
    arr = np.round(np.asarray(values, dtype=float), digits) + 0.0
    return [None if not np.isfinite(v) else float(v) for v in arr]


def _span(built, game: str, number: int):
    poss = built.tables["heldout"][game]["possessions"]
    return poss[poss["possession"] == number].iloc[0]


# ------------------------------------------------------------------------------------------- the tree as a whole


def test_the_tree_validates_against_the_contract(built):
    assert built.problems == []
    assert S.validate_tree(built.out, allow_missing=S.STAGED_FILES) == []
    for rel in ("manifest.json", "players.json", "scorecard.json", "results.json"):
        assert (built.out / rel).exists(), rel
    manifest = S.Manifest.model_validate(_json(built.out / "manifest.json"))
    assert manifest.schema_version == S.SCHEMA_VERSION and manifest.freeze_tag == "freeze-v1"
    assert manifest.n_games == 2 and manifest.n_possessions == len(FX.PLAN_A) + len(FX.PLAN_B)
    # a tree built from a test's data root says so in its build id; synthetic is reserved for export/sample.py
    assert manifest.synthetic is False and manifest.build_id.startswith("test-fixture-")


def test_only_heldout_games_become_games(built, tmp_path):
    manifest = _json(built.out / "manifest.json")
    assert [(g["game_id"], g["split"], g["slug"]) for g in manifest["games"]] == [
        (FX.HELD_A, "heldout_game", "2015-12-16-phx-at-gsw"), (FX.HELD_B, "heldout_arena", "2016-01-05-fxa-at-cle")]
    assert sorted(p.name for p in (built.out / "games").iterdir()) == ["2015-12-16-phx-at-gsw", "2016-01-05-fxa-at-cle"]
    for game in FX.TRAIN_GAMES[:2] + ["12.25.2015.CLE.at.GSW"]:
        with pytest.raises(C.ExportProblem, match="not a held-out game"):
            site_games.game_of(game)
    # a training game that sits in the held-out folder stops the games step before anything is written
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    held = data / "analytics" / "possessions" / "heldout" / FX.SEASON
    shutil.copy(held / f"{FX.HELD_A}.parquet", held / f"{FX.TRAIN_GAMES[0]}.parquet")
    problems, _info = site_data.run(tmp_path / "out", data, built.artifacts, only=["games"], log=QUIET,
                                    allow_fixture_root=True)
    assert any("is not a held-out game" in line for line in problems)
    assert not (tmp_path / "out" / "games").exists()


def test_model_outputs_that_are_not_heldout_rows_are_refused(built, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    path = data / "analytics" / "epv" / "heldout" / FX.SEASON / f"{FX.HELD_B}.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "fold"] = 2                                   # an out-of-fold row among the held-out states
    table.to_parquet(path, index=False)
    ctx = site_games.load_shot_context(data, [FX.HELD_A, FX.HELD_B])
    with pytest.raises(C.ExportProblem, match="not held-out rows"):
        site_games.build_game(data, site_games.game_of(FX.HELD_B), ctx)


def test_no_private_string_in_any_output(built):
    files = sorted(p for p in built.out.rglob("*") if p.is_file())
    assert len(files) == 4 + 2 + len(FX.PLAN_A) + len(FX.PLAN_B)
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert S.find_private(text) == [], path.name
        for private in (str(built.base), str(built.data), Path.home().name, str(REPO)):
            assert private not in text, (path.name, private)
        assert "NaN" not in text and "Infinity" not in text


def test_manifest_names_every_file_its_sources_and_the_size_of_each_game(built):
    manifest = S.Manifest.model_validate(_json(built.out / "manifest.json"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    assert manifest.repo_commit in (head, head + "+uncommitted")
    by_path = {entry.path: entry for entry in manifest.files}
    assert {"players.json", "scorecard.json", "results.json"} <= set(by_path)
    assert all(entry.sources for entry in manifest.files)
    for game in manifest.games:
        folder = built.out / "games" / game.slug
        family = by_path[f"games/{game.slug}/<possession>.json"]
        members = sorted(p for p in folder.iterdir() if p.name != "possessions.json")
        assert family.model == "Possession" and family.count == len(members) == game.n_possessions
        assert family.bytes == sum(p.stat().st_size for p in members)
        assert by_path[f"games/{game.slug}/possessions.json"].model == "PossessionIndex"
        assert game.bytes == sum(p.stat().st_size for p in folder.iterdir())
        # the family names the tables of its own game, read for held-out rows
        paths = {s.path for s in family.sources}
        assert f"$DATA_ROOT/sportvu_contract/heldout/{game.game_id}.parquet" in paths
        assert all(s.rows in ("heldout", "not_split") for s in family.sources)
    assert sum(g.n_featured for g in manifest.games) == len(built.info["games"]["featured"]["picks"])


# ------------------------------------------------------------------------------------------- possession files


def test_frames_sit_on_the_ghost_grid_and_copy_the_tracking_table(built):
    for game in (FX.HELD_A, FX.HELD_B):
        tables = built.tables["heldout"][game]
        frames, teams = tables["frames"], dict(zip(tables["rosters"]["team_id"], tables["rosters"]["team"]))
        for span in tables["possessions"].itertuples(index=False):
            doc = _possession(built, game, span.possession)
            grid = np.arange(span.start_frame, span.end_frame + 1, 5)
            assert doc["frame0"] == span.start_frame and doc["hz"] == 5.0
            assert doc["t"] == [round(float(v), 2) for v in (grid - span.start_frame) / 25]
            on_grid = frames[frames["frame"].isin(grid)]
            level = on_grid.drop_duplicates("frame").set_index("frame").reindex(grid)
            assert doc["game_clock_s"] == _list(level["game_clock_s"], 2)
            assert doc["shot_clock_s"] == _list(level["shot_clock_s"], 2)
            for axis in "xyz":
                assert doc["ball"][axis] == _list(level[f"ball_{axis}"], 1)
            tracks = {p["player_id"]: p for p in doc["players"]}
            assert set(tracks) == set(on_grid["player_id"])
            for player_id, part in on_grid.groupby("player_id"):
                part = part.set_index("frame")
                assert tracks[player_id]["x"] == _list(part["x"].reindex(grid), 1)
                assert tracks[player_id]["y"] == _list(part["y"].reindex(grid), 1)
                assert tracks[player_id]["offense"] == (part["team_id"].iloc[0] == span.offense_team_id)
                assert tracks[player_id]["name"] == FX.player_name(player_id)
            assert doc["offense"] == teams[span.offense_team_id]
            assert doc["defense"] == next(a for t, a in teams.items() if t != span.offense_team_id)
            assert (doc["result"], doc["points"], doc["gap"], doc["period"]) == (
                span.end_kind, span.points, bool(span.gap), span.period)
            # away attacks the right rim in odd periods in the synthetic games
            away_on_offense = doc["offense"] == FX.PLANS[game][0]
            assert doc["attack_x"] == (88.75 if away_on_offense == (span.period % 2 == 1) else 5.25)


def test_degenerate_possessions_are_kept_and_flagged(built):
    single = _possession(built, FX.HELD_A, 10)
    assert single["t"] == [0.0] and any("Single frame" in n for n in single["notes"])
    assert _row(built, FX.HELD_A, 10)["n_frames"] == 1
    gap = _possession(built, FX.HELD_A, 8)
    assert gap["gap"] is True and any("Clock gap" in n for n in gap["notes"]) and gap["ghosts"] is None
    assert any(f"jumps by more than {GAP_S:g} s inside" in n for n in gap["notes"])   # the lane's own threshold
    assert any("No ghost defenders" in n for n in gap["notes"])
    lost = _possession(built, FX.HELD_A, 17)                   # one defender has no rows in the second half
    missing = [p for p in lost["players"] if None in p["x"]]
    assert len(missing) == 1 and missing[0]["x"][0] is not None and missing[0]["x"][-1] is None
    assert any("fewer than 10 tracked players" in n for n in lost["notes"])
    hole = _possession(built, FX.HELD_A, 18)                   # one sampled frame has no rows; no ball later on
    assert all(p["x"][4] is None for p in hole["players"]) and hole["game_clock_s"][4] is None
    assert any("Frames missing from the tracking table: 1 of 46." in n for n in hole["notes"])
    assert hole["ball"]["x"][-1] is None and hole["shot_clock_s"][1] is None   # frame 0 belongs to the possession before
    assert any("Frames with no ball position" in n for n in hole["notes"])
    assert _possession(built, FX.HELD_A, 0)["notes"] == []     # a clean possession carries no note
    totals = built.info["games"]["totals"]
    assert (totals["possessions"], totals["one_frame"], totals["gap"]) == (30, 1, 1)
    assert totals["no_ghosts"] == 3 and totals["frames_missing"] == 1


def test_ghosts_align_by_frame_and_use_the_one_second_rows(built):
    for game in (FX.HELD_A, FX.HELD_B):
        tables = built.tables["heldout"][game]
        stored = tables["ghost_frames"]
        one_second = stored[stored["horizon_s"] > 0.5]
        deviation = dict(zip(tables["ghost_possessions"]["possession"], tables["ghost_possessions"]["mean_deviation_ft"]))
        for span in tables["possessions"].itertuples(index=False):
            doc = _possession(built, game, span.possession)
            rows = one_second[one_second["possession"] == span.possession]
            assert _row(built, game, span.possession)["has_ghosts"] == (len(rows) > 0)
            if not len(rows):
                assert doc["ghosts"] is None
                continue
            ghosts = doc["ghosts"]
            assert ghosts["label"] == "average defense" and ghosts["horizon_s"] == 1.0
            assert ghosts["mean_deviation_ft"] == round(float(deviation[span.possession]), 2)
            grid = np.arange(span.start_frame, span.end_frame + 1, 5)
            dropped = 0
            for track in ghosts["tracks"]:
                mine = rows[rows["def_player_id"] == track["player_id"]].set_index("frame")
                x, y = mine["ghost_x"].reindex(grid), mine["ghost_y"].reindex(grid)
                far = (x < -10) | (x > 104) | (y < -10) | (y > 60)
                dropped += int(far.sum())
                assert track["x"] == _list(x.mask(far), 1) and track["y"] == _list(y.mask(far), 1)
                assert track["x"][:7] == [None] * 7            # the 1.0 s rows start at step 7; step 3 is the 0.2 s rows
            assert ghosts["dropped_points"] == dropped
    far = _possession(built, FX.HELD_A, 13)
    assert far["ghosts"]["dropped_points"] == 1 and any("not drawn (a display rule): 1." in n for n in far["notes"])
    assert all(FX.FAR_GHOST_X not in t["x"] for t in far["ghosts"]["tracks"])
    cut = _possession(built, FX.HELD_A, 19)["ghosts"]["tracks"][0]["x"]
    assert cut[-1] is None and cut[10] is not None             # the stored sequence stops half way


def test_epv_curve_is_the_stored_states_by_frame_and_is_not_clipped(built):
    lowest = []
    for game in (FX.HELD_A, FX.HELD_B):
        tables = built.tables["heldout"][game]
        for span in tables["possessions"].itertuples(index=False):
            doc = _possession(built, game, span.possession)
            states = tables["epv"][tables["epv"]["possession"] == span.possession].sort_values("frame")
            assert doc["epv"]["t"] == [round(float(v), 2) for v in (states["frame"] - span.start_frame) / 25]
            assert doc["epv"]["value"] == _list(states["epv_points"], 3)
            row = _row(built, game, span.possession)
            assert (row["epv_start"], row["epv_end"]) == (doc["epv"]["value"][0], doc["epv"]["value"][-1])
            lowest.append(min(doc["epv"]["value"]))
            assert any(round(t * 25) % 5 for t in doc["epv"]["t"]) or len(states) < 3   # states sit off the 5-frame grid
    assert min(lowest) < 0                                     # a negative value is written as stored


def test_a_shot_or_an_epv_time_can_lie_just_after_the_last_frame(built):
    """The file ends on the last grid frame at or before the possession's end; a play-by-play shot time and an EPV
    state use the exact frame, so they can lie up to 4 frames (0.16 s) after it. The values are kept as stored."""
    span = _span(built, FX.HELD_B, 3)                          # 277 frames long: not a whole number of grid steps
    assert (span["end_frame"] - span["start_frame"]) % C.STRIDE == 2
    doc = _possession(built, FX.HELD_B, 3)
    assert doc["t"][-1] == 11.0 and _row(built, FX.HELD_B, 3)["duration_s"] == 11.0
    assert (doc["shots"][0]["t_source"], doc["shots"][0]["t"]) == ("play_by_play", 11.08)
    assert doc["epv"]["t"][-1] == 11.08
    slack = (C.STRIDE - 1) / C.FPS
    late = 0
    for game, (_away, _home, plan, _extra) in FX.PLANS.items():
        for number in range(len(plan)):
            doc = _possession(built, game, number)
            times = [s["t"] for s in doc["shots"] if s["t"] is not None] + doc["epv"]["t"]
            assert all(0 <= t <= doc["t"][-1] + slack + 1e-9 for t in times), (game, number)
            late += sum(t > doc["t"][-1] for t in times)
        assert any(f"can lie up to {slack:g} s after the last frame" in n for n in _index(built, game)["notes"])
    assert late == 2 and slack == 0.16


def test_ghost_coverage_counts_stored_rows_before_the_display_drop():
    """PLAN.md D48: the featured rule reads no model error. A ghost that flies off the court is dropped from the
    drawing, and still counts as a stored ghost for the 80 percent rule."""
    n = 10
    grid = site_games.Grid(start=100, end=145, n=n, rows=np.array([], dtype=int), slot=np.array([], dtype=int),
                           per_frame=np.zeros(n, dtype=int), level={})
    rows = pd.DataFrame({"frame": 100 + 5 * np.arange(n), "def_player_id": 7, "ghost_x": [50.0] * 4 + [130.0] * 6,
                         "ghost_y": 25.0})
    ghosts, share, dropped = site_games.ghost_tracks(rows, grid, {7}, 1.0, "t")
    assert share == 1.0 >= site_games.FEATURED_MIN_GHOST_SHARE and dropped == 6 == ghosts.dropped_points
    assert ghosts.tracks[0].x == [50.0] * 4 + [None] * 6       # the far points are still not drawn
    rows.loc[0, "ghost_x"] = np.nan                            # a stored row with no position is not a ghost
    assert site_games.ghost_tracks(rows, grid, {7}, 1.0, "t")[1] == 0.9
    assert site_games.ghost_tracks(rows.iloc[:5], grid, {7}, 1.0, "t")[1] == 0.4   # frames with no stored row


def _chart_in_court(chart_row, attack_x: float, mirrored: bool):
    """The same transform as the export's, written out: the chart draws every shot at the left rim."""
    x, y = float(chart_row["court_x"]), float(chart_row["court_y"])
    y = 50.0 - y if mirrored else y
    return (94.0 - x, 50.0 - y) if attack_x > 47 else (x, y)


def test_shot_time_and_place_fall_back_in_the_stated_order(built):
    tables = built.tables["heldout"][FX.HELD_A]
    frames, attempts, events, chart = tables["frames"], tables["shot_frames"], tables["pbp"], tables["chart"]

    def shot_of(number: int, k: int = 0) -> dict:
        return _possession(built, FX.HELD_A, number)["shots"][k]

    def chart_row(event: int):
        attempt = attempts[attempts["event"] == event].iloc[0]
        return chart[(chart["period"] == attempt["period"]) & np.isclose(chart["clock_s"], float(attempt["clock_s"]))].iloc[0]

    # a located release: time and place from tracking
    span, shot = _span(built, FX.HELD_A, 0), shot_of(0)
    attempt = attempts[attempts["event"] == shot["event"]].iloc[0]
    at_release = frames[(frames["frame"] == attempt["release_frame"]) & (frames["player_id"] == attempt["shooter_id"])].iloc[0]
    assert (shot["t_source"], shot["location_source"]) == ("release_frame", "tracking_at_release")
    assert shot["t"] == round((attempt["release_frame"] - span["start_frame"]) / 25, 2)
    assert (shot["x"], shot["y"]) == (round(float(at_release["x"]), 1), round(float(at_release["y"]), 1))
    assert (shot["shooter_id"], shot["shooter"]) == (9101, FX.player_name(9101))
    assert (shot["made"], shot["three"], shot["xfg"], shot["has_contest"]) == (True, False, 0.3, True)
    assert shot_of(2)["three"] is True

    # no release: the time is the play-by-play frame, the place the shot chart marker (orientation verified here)
    span, shot = _span(built, FX.HELD_A, 5), shot_of(5)
    event_frame = int(events.loc[events["event"] == shot["event"], "frame"].iloc[0])
    assert (shot["t_source"], shot["location_source"], shot["has_contest"]) == ("play_by_play", "shot_chart", False)
    assert shot["t"] == round((event_frame - span["start_frame"]) / 25, 2)
    doc = _possession(built, FX.HELD_A, 5)
    x, y = _chart_in_court(chart_row(shot["event"]), doc["attack_x"], mirrored=True)
    assert (shot["x"], shot["y"]) == (round(x, 1), round(y, 1))
    assert any("was not located" in n for n in doc["notes"]) and any("shot chart marker" in n for n in doc["notes"])

    # an unresolved shooter: a release found by the ball alone is not used; the shooter comes from the shot chart
    shot = shot_of(6)
    assert (shot["t_source"], shot["location_source"]) == ("play_by_play", "shot_chart")
    assert (shot["shooter_id"], shot["shooter"], shot["has_contest"]) == (9102, FX.player_name(9102), True)

    # two attempts in one possession, in event order; the index carries the last one
    both = _possession(built, FX.HELD_A, 9)["shots"]
    assert [s["made"] for s in both] == [False, True] and both[0]["event"] < both[1]["event"]
    assert _row(built, FX.HELD_A, 9)["xfg"] == both[1]["xfg"] == 0.71

    # the ball outside the court at the release: the lane measured no contest features there
    assert (shot_of(11)["t_source"], shot_of(11)["has_contest"]) == ("release_frame", False)
    # an attempt with no prediction is not a shot of the file, and the index carries no xFG for it
    last = _possession(built, FX.HELD_A, 14)
    assert last["shots"] == [] and any("no xFG prediction" in n for n in last["notes"])
    assert _row(built, FX.HELD_A, 14)["xfg"] is None and _row(built, FX.HELD_A, 14)["xfg_bucket"] is None
    # a release before the possession's first frame is outside it
    assert (shot_of(15)["t_source"], shot_of(15)["location_source"]) == ("play_by_play", "shot_chart")
    # no release and no play-by-play frame: no time, the chart still gives the place
    assert (shot_of(16)["t"], shot_of(16)["t_source"], shot_of(16)["location_source"]) == (None, "none", "shot_chart")

    # game B: the shooters stand in the middle, the orientation cannot be verified, so no chart location is used
    other = _possession(built, FX.HELD_B, 3)["shots"][0]
    assert (other["t_source"], other["location_source"], other["x"], other["y"]) == ("play_by_play", "none", None, None)
    orientation = built.info["games"]["orientation"]
    assert orientation[_slug(FX.HELD_A)] == {"verified": True, "mirrored": True, "n_votes": 13, "n_agree": 13,
                                              "n_shots": 13, "median_ft": 1.1}
    assert orientation[_slug(FX.HELD_B)]["verified"] is False
    assert any("not verified in this game" in n for n in _index(built, FX.HELD_B)["notes"])

    # the prediction whose event lies in no possession is counted, not shown
    shown = {s["event"] for number in range(len(FX.PLAN_A)) for s in _possession(built, FX.HELD_A, number)["shots"]}
    assert len(shown) == len(tables["xfg"]) - 1
    assert any("1 of them fall in no possession" in n for n in _index(built, FX.HELD_A)["notes"])
    totals = built.info["games"]["totals"]
    assert (totals["predictions"], totals["predictions_unplaced"], totals["attempts_without_prediction"]) == (26, 1, 1)
    assert (totals["shots_release"], totals["shots_event_time"], totals["shots_no_time"]) == (20, 4, 1)
    assert (totals["shots_chart_location"], totals["shots_no_location"], totals["shooter_from_chart"]) == (4, 1, 1)


def test_orientation_check_reads_positions_only():
    def samples(mirrored: bool, n: int, lateral: float = 12.0):
        out = []
        for k in range(n):
            attack_x, cy = (88.75, 25.0 + lateral) if k % 2 else (5.25, 25.0 - lateral)
            x, y = site_games.chart_to_court(15.0 + k, cy, attack_x, mirrored)
            out.append((x + 0.5, y - 0.5, 15.0 + k, cy, attack_x))
        return out

    assert site_games.check_orientation(samples(True, 12))["mirrored"] is True
    assert site_games.check_orientation(samples(False, 12))["mirrored"] is False
    assert site_games.check_orientation(samples(True, 9))["verified"] is False           # too few shots
    assert site_games.check_orientation(samples(True, 40, lateral=2.0))["verified"] is False   # all near the middle
    mixed = site_games.check_orientation(samples(True, 10) + samples(False, 10))
    assert mixed["verified"] is False and mixed["mirrored"] is None
    assert site_games.chart_to_court(10.0, 20.0, 5.25, False) == (10.0, 20.0)
    assert site_games.chart_to_court(10.0, 20.0, 88.75, False) == (84.0, 30.0)
    assert site_games.chart_to_court(10.0, 20.0, 5.25, True) == (10.0, 30.0)
    assert site_games.chart_to_court(10.0, 20.0, 88.75, True) == (84.0, 20.0)


def test_attack_direction_is_checked_against_the_ball(built, tmp_path):
    notes = _index(built, FX.HELD_A)["notes"]
    assert any("agrees with it in 18 of 18 possessions" in n for n in notes)
    # the ball in the other half for a whole period: the inferred direction cannot be trusted, the game stops
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    path = data / "sportvu_contract" / "heldout" / f"{FX.HELD_A}.parquet"
    frames = pd.read_parquet(path)
    first = frames["period"] == 1
    frames.loc[first, "ball_x"] = (94.0 - frames.loc[first, "ball_x"]).astype("float32")
    frames.to_parquet(path, index=False)
    ctx = site_games.load_shot_context(data, [FX.HELD_A, FX.HELD_B])
    with pytest.raises(C.ExportProblem, match="disagrees with where the ball ends up"):
        site_games.build_game(data, site_games.game_of(FX.HELD_A), ctx)


# ------------------------------------------------------------------------------------------- the index, featured


def _all_predictions(built) -> np.ndarray:
    held = pd.concat([t["xfg"] for t in built.tables["heldout"].values()])
    return np.round(held["p_selected"].to_numpy(dtype=float), 3)


def test_index_carries_the_last_attempt_and_its_fifth(built):
    edges = [round(float(e), 4) for e in np.quantile(_all_predictions(built), [0.2, 0.4, 0.6, 0.8])]
    for game, (_away, _home, plan, _extra) in FX.PLANS.items():
        index = S.PossessionIndex.model_validate(_index(built, game))
        assert index.xfg_bucket_edges == edges
        assert [row.possession for row in index.possessions] == list(range(len(plan)))
        for row, spec in zip(index.possessions, plan):
            doc = _possession(built, game, row.possession)
            predicted = [s for s in spec.shots if s.predicted]
            last_is_predicted = bool(spec.shots) and spec.shots[-1].predicted
            assert row.xfg == (predicted[-1].xfg if last_is_predicted else None)
            assert row.xfg_bucket == (None if row.xfg is None else 1 + sum(row.xfg > e for e in edges))
            # the file ends on the last grid frame at or before the possession's end
            assert row.n_frames == len(doc["t"]) == int(spec.seconds * FX.FPS) // FX.STRIDE + 1
            assert row.duration_s == doc["t"][-1] == FX.frames_to_seconds(row.n_frames)
            assert row.demo == ((game, row.possession) in FX.DEMO)
    assert built.info["games"]["demo_flagged"] == len(FX.DEMO)
    assert built.info["games"]["xfg_bucket_rows"] == len(_all_predictions(built))


def test_xfg_fifths_are_over_every_heldout_prediction(built, tmp_path):
    """The contract says xfg_bucket is the fifth among all held-out attempts: the rows the lane report scores, also
    those of a held-out game that has predictions and no possessions table. The note states both counts."""
    n = len(_all_predictions(built))
    notes = " ".join(_index(built, FX.HELD_A)["notes"])
    assert f"the {n} held-out attempts with a prediction, from 2 held-out games (edges in xfg_bucket_edges)." in notes
    assert "no possessions table" not in notes                 # every held-out prediction is in a game shown
    games = built.info["games"]
    assert (games["xfg_bucket_rows"], games["xfg_bucket_games"], games["xfg_rows_in_games_shown"],
            games["xfg_games_not_shown"]) == (n, 2, n, [])

    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    path = data / "analytics" / "xfg" / FX.SEASON / "xfg_predictions.parquet"
    table = pd.read_parquet(path)
    other = "01.07.2016.FXA.at.CLE"                            # held out (a game at CLE), with no possessions table
    extra = table[table["heldout"]].head(6).assign(sportvu_game=other, event=-1, p_selected=np.float32(0.9))
    pd.concat([table, extra], ignore_index=True).to_parquet(path, index=False)
    ctx = site_games.load_shot_context(data, [FX.HELD_A, FX.HELD_B])
    every = np.append(_all_predictions(built), [0.9] * 6)
    assert ctx.edges == [round(float(e), 4) for e in np.quantile(every, [0.2, 0.4, 0.6, 0.8])]
    assert ctx.edges != _index(built, FX.HELD_A)["xfg_bucket_edges"]
    assert (ctx.n_bucket_rows, ctx.n_bucket_games, ctx.n_rows_shown, ctx.games_not_shown) == (n + 6, 3, n, [other])
    assert set(ctx.rows["sportvu_game"]) == {FX.HELD_A, FX.HELD_B}      # the shots shown are still the games' own
    result = site_games.build_game(data, site_games.game_of(FX.HELD_B), ctx)
    note = next(text for text in result.notes if text.startswith("xFG:"))
    assert f"the {n + 6} held-out attempts with a prediction, from 3 held-out games" in note
    assert f"6 of those attempts belong to 1 held-out game(s) with no possessions table ({other})" in note
    assert f"The other {n} are in the games shown here." in note

    # a row flagged held out that belongs to a training game stops the build
    wrong = table.copy()
    wrong.loc[wrong.index[wrong["heldout"]][0], "sportvu_game"] = FX.TRAIN_GAMES[0]
    wrong.to_parquet(path, index=False)
    with pytest.raises(C.ExportProblem, match="not held-out games"):
        site_games.load_shot_context(data, [FX.HELD_A, FX.HELD_B])


def test_featured_picks_follow_the_written_rule(built, tmp_path):
    edges = _index(built, FX.HELD_A)["xfg_bucket_edges"]
    cells = {f"{result} fifth {fifth}": [] for result in ("shot_made", "shot_missed") for fifth in range(1, 6)}
    for game, (_away, _home, plan, _extra) in FX.PLANS.items():
        for number, spec in enumerate(plan):
            if FX.eligible(spec):                              # D48, stated over the plan of the synthetic games
                fifth = 1 + sum(spec.shots[-1].xfg > e for e in edges)
                cells[f"{spec.end_kind} fifth {fifth}"].append((game, number))
    featured = built.info["games"]["featured"]
    assert featured["eligible"] == {cell: len(ids) for cell, ids in cells.items()}
    assert sum(featured["eligible"].values()) == 15
    # the draw, written out: one generator, made then missed, fifth 1 to 5, ids in sorted order, one draw per cell
    rng = np.random.default_rng(0)
    expected = {}
    for cell, ids in cells.items():
        if ids:
            expected[cell] = sorted(ids)[int(rng.integers(len(ids)))]
    assert {cell: tuple(pick) for cell, pick in featured["picks"].items()} == expected
    assert featured["empty"] == [cell for cell, ids in cells.items() if not ids] == ["shot_missed fifth 5"]
    flagged = {}
    for game in FX.PLANS:
        for row in _index(built, game)["possessions"]:
            if row["featured"]:
                assert row["featured_by"] == "rule"
                flagged[row["featured_cell"]] = (game, row["possession"])
            else:
                assert row["featured_by"] is None and row["featured_cell"] is None
    assert flagged == expected
    assert any("9 of 10 cells have a pick" in n for n in _index(built, FX.HELD_B)["notes"])
    # deterministic: a second run writes the same index files
    again = site_data.run_games(built.data, built.artifacts, tmp_path / "out", log=QUIET)
    assert again["featured"] == featured
    for game in FX.PLANS:
        rel = Path("games") / _slug(game) / "possessions.json"
        assert (tmp_path / "out" / rel).read_bytes() == (built.out / rel).read_bytes()


def test_pick_featured_reports_an_empty_cell_and_takes_no_draw_for_it():
    candidates = [("b", 2, "shot_made fifth 1"), ("a", 7, "shot_made fifth 1"), ("a", 3, "shot_missed fifth 4")]
    drawn = site_games.pick_featured(candidates)
    rng = np.random.default_rng(0)
    first = sorted([("a", 7), ("b", 2)])[int(rng.integers(2))]
    assert drawn["picks"] == {"shot_made fifth 1": first, "shot_missed fifth 4": ("a", 3)}
    assert len(drawn["empty"]) == 8 and "shot_made fifth 2" in drawn["empty"]
    assert site_games.pick_featured(list(reversed(candidates))) == drawn           # the input order does not matter
    with pytest.raises(C.ExportProblem):
        site_games.pick_featured([("a", 1, "turnover fifth 1")])


def test_a_hand_pick_replaces_the_rules_pick_and_is_marked(built, tmp_path):
    cell = "shot_made fifth 1"
    rule_pick = tuple(built.info["games"]["featured"]["picks"][cell])
    fifth = lambda row: row["result"] == "shot_made" and row["xfg_bucket"] == 1  # noqa: E731
    others = [(game, row["possession"]) for game in FX.PLANS for row in _index(built, game)["possessions"]
              if fifth(row) and (game, row["possession"]) != rule_pick]
    hand = others[0]
    swaps = tmp_path / "swaps.json"
    swaps.write_text(json.dumps([{"cell": cell, "game_id": hand[0], "possession": hand[1], "why": "a clearer play"}]))
    problems, info = site_data.run(tmp_path / "out", built.data, built.artifacts, log=QUIET, allow_fixture_root=True,
                                   swaps=swaps)
    assert problems == [] and info["games"]["featured"]["by_hand"] == [cell]
    marked = {}
    for game in FX.PLANS:
        for row in _json(tmp_path / "out" / "games" / _slug(game) / "possessions.json")["possessions"]:
            if row["featured"]:
                marked[(game, row["possession"])] = (row["featured_cell"], row["featured_by"])
    assert marked[hand] == (cell, "hand") and rule_pick not in marked
    assert sorted(by for _cell, by in marked.values()) == ["hand"] + ["rule"] * 8
    assert any("1 of them were put there by hand" in n for n in
               _json(tmp_path / "out" / "games" / _slug(FX.HELD_A) / "possessions.json")["notes"])
    # a hand pick must belong to its cell, be in the run, and a cell takes one swap
    rows = {(game, row["possession"]): row for game in FX.PLANS for row in _index(built, game)["possessions"]}
    picks = built.info["games"]["featured"]["picks"]
    for bad in ([{"cell": "shot_missed fifth 1", "game_id": hand[0], "possession": hand[1]}],
                [{"cell": cell, "game_id": hand[0], "possession": 4}],
                [{"cell": cell, "game_id": hand[0], "possession": 999}],
                [{"cell": "turnover fifth 1", "game_id": hand[0], "possession": hand[1]}],
                [{"cell": cell, "game_id": hand[0], "possession": hand[1]}] * 2):
        with pytest.raises(C.ExportProblem, match="featured swap"):
            site_games.featured_marks(picks, bad, rows)
    swaps.write_text("{}")
    with pytest.raises(C.ExportProblem, match="--swaps"):
        site_data.read_swaps(swaps)


def test_demo_list_shapes(tmp_path):
    assert site_games.demo_possessions(tmp_path) == set()
    folder = tmp_path / "demo"
    folder.mkdir()
    for doc in ([{"game_id": FX.HELD_A, "possession": 2}], {"possessions": [{"sportvu_game": FX.HELD_A, "possession": 2}]},
                {FX.HELD_A: [2]}):
        (folder / "possessions.json").write_text(json.dumps(doc))
        assert site_games.demo_possessions(tmp_path) == {(FX.HELD_A, 2)}
    (folder / "possessions.json").write_text(json.dumps([{"possession": 2}]))
    with pytest.raises(C.ExportProblem):
        site_games.demo_possessions(tmp_path)


# ------------------------------------------------------------------------------------------- players.json


def _kept_attempts(built) -> pd.DataFrame:
    """Out-of-fold attempts whose shooter the name match resolved, straight from the synthetic tables."""
    training = built.tables["training"]
    rows = training["xfg"][training["xfg"]["fold"] >= 0]
    located = pd.concat(training["shot_frames"].values(), ignore_index=True)[["sportvu_game", "event", "shooter_id"]]
    merged = rows.merge(located, on=["sportvu_game", "event"], how="left")
    same = (merged["player_id"].astype("Float64") == merged["shooter_id"].astype("Float64")).fillna(False)
    kept = merged[same.to_numpy(dtype=bool)].copy()
    kept["player_id"] = kept["player_id"].astype(int)
    return kept


def _report_gaps() -> np.ndarray:
    """Observed minus predicted per calibration bin of the contest candidate, read from reports/xfg.md."""
    text = (REPO / "reports" / "xfg.md").read_text(encoding="utf-8")
    section = text.split("### contest_logistic (out-of-fold)")[1].split("###")[0]
    gaps = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].isdigit():
            gaps.append(float(cells[5]) - float(cells[4]) if cells[3] != "0" else np.nan)
    return np.array(gaps)


def test_player_values_come_from_out_of_fold_rows_only(built):
    from analytics.gravity.train import gravity_players       # the lane whose game bootstrap D47 names

    file = S.PlayersFile.model_validate(_json(built.out / "players.json"))
    players = {p.player_id: p for p in file.players}
    assert (file.n_players, file.n_shown, file.n_hidden) == (8, 7, 1)          # the id 0 placeholder is not a player
    assert sorted(players) == [9301, 9302, 9303, 9304, 9305, 9501, 9502]       # 9306 has no value above a minimum
    assert file.window == "12 training games, 2015-11-01 to 2015-11-12; the held-out games feed no player value"

    kept = _kept_attempts(built)
    attempts = kept.groupby("player_id").size().to_dict()
    # 9301: not the 5 in-sample decoys (fold -2), not the attempts credited to him by a ball-only fallback;
    # 9303: not his held-out attempts (fold -1) of the held-out game he plays in
    assert attempts == {9301: 240, 9302: 90, 9303: 120, 9304: 150, 9501: 110}
    held_ids = pd.concat([t["xfg"] for t in built.tables["heldout"].values()])["player_id"].dropna().astype(int)
    assert 9303 in set(held_ids)
    info = built.info["players"]
    assert info["attempts_total"] == 101853
    assert info["attempts_left_out"] == 101853 - len(kept)
    # the attempts left out, by reason: the filler game has no shot_frames table at all (no name match was ever
    # tried there), and for player 9306 the name match failed
    out_of_fold = built.tables["training"]["xfg"]
    out_of_fold = out_of_fold[out_of_fold["fold"] >= 0]
    n_filler = int((out_of_fold["sportvu_game"] == FX.FILLER_GAME).sum())
    assert n_filler > 0 and info["attempts_left_out_by_reason"] == {
        "no_shot_frames_table": n_filler, "no_shot_frames_row": 0, "shooter_not_resolved": 200, "another_shooter": 0}
    assert n_filler + 200 == info["attempts_left_out"]
    assert (info["games_with_attempts"], info["games_without_shot_frames"]) == (out_of_fold["sportvu_game"].nunique(), 1)
    note = " ".join(next(m for m in file.metrics if m.metric == "shot_making").notes)
    assert (f"Attempts left out: {info['attempts_left_out']} of 101853 out-of-fold attempts, which come from "
            f"{info['games_with_attempts']} training games (more than the tracked training games") in note
    assert f"{n_filler} belong to 1 training game(s) with no shot_frames table" in note
    assert "200 have a tracked attempt whose shooter the play-by-play name match did not resolve" in note
    assert "no row in their game's shot_frames table" not in note        # a reason with no attempt is not stated

    gaps = _report_gaps()
    lane = gravity_players(pd.DataFrame({
        "player_id": kept["player_id"], "sportvu_game": kept["sportvu_game"], "possession": np.arange(len(kept)),
        "gravity_ft": 100.0 * (kept["made"].astype(float) - kept["p_contest"].astype(float)), "heldout": False}),
        n_min=1, n_boot=200, seed=0).set_index("player_id")
    for player_id, n in attempts.items():
        view, mine = players[player_id].shot_making, kept[kept["player_id"] == player_id]
        assert (view.attempts, view.min_attempts, view.shown) == (n, 100, n >= 100)
        if not view.shown:
            assert view.over_expected_per_100 is None and view.fg_pct is None and view.baseline_error_per_100 is None
            continue
        made, p = mine["made"].astype(float), mine["p_contest"].astype(float)
        estimate = view.over_expected_per_100
        assert estimate.value == pytest.approx(100 * (made - p).mean(), abs=0.006)
        assert estimate.value == pytest.approx(lane.at[player_id, "gravity_ft"], abs=0.006)
        assert (estimate.n, estimate.n_unit) == (n, "attempts")
        if mine["sportvu_game"].nunique() == 1:                # one game: no interval, and it says so
            assert estimate.point_estimate is True and estimate.lo is None and estimate.interval is None
        else:                                                  # the interval is the gravity lane's own draw
            assert estimate.point_estimate is False and "200 resamples, seed 0" in estimate.interval
            assert estimate.lo == pytest.approx(lane.at[player_id, "lo"], abs=0.006)
            assert estimate.hi == pytest.approx(lane.at[player_id, "hi"], abs=0.006)
        assert view.fg_pct == pytest.approx(made.mean(), abs=6e-5)
        assert view.expected_fg_pct == pytest.approx(p.mean(), abs=6e-5)
        assert view.frozen_model_residual_per_100 == pytest.approx(100 * (made - mine["p_selected"].astype(float)).mean(), abs=0.006)
        bins = np.floor(p.to_numpy() * len(gaps)).astype(int)
        assert view.baseline_error_per_100 == pytest.approx(100 * gaps[bins].mean(), abs=0.006)
    assert players[9304].shot_making.over_expected_per_100.point_estimate is True
    assert players[9305].shot_making.attempts == 0 and players[9305].shot_making.shown is False


def test_minimum_samples_and_the_other_player_views(built):
    file = S.PlayersFile.model_validate(_json(built.out / "players.json"))
    players = {p.player_id: p for p in file.players}
    assert {r.metric: (r.min_n, r.unit) for r in file.rules} == {
        "shot_making": (100, "attempts"), "gravity": (300, "possessions"), "rebounding": (300, "chances"),
        "absence_risk": (10, "games"), "similar": (300, "possessions")}
    assert built.info["players"]["shown_by_metric"] == {"shot_making": 4, "gravity": 4, "rebounding": 2,
                                                         "absence_risk": 4, "similar": 2}

    # gravity: the stored table and its own gate; a player with one game is a point estimate
    for player_id, (possessions, games, value, lo, hi) in FX.GRAVITY.items():
        if player_id not in players:
            continue
        view = players[player_id].gravity
        assert (view.possessions, view.games, view.shown) == (possessions, games, possessions >= 300)
        if view.shown:
            assert view.gravity_ft.value == round(value, 3)
            assert (view.gravity_ft.lo, view.gravity_ft.hi) == ((None, None) if np.isnan(lo) else (lo, hi))
            assert view.gravity_ft.point_estimate == bool(np.isnan(lo))
        else:
            assert view.gravity_ft is None
    assert players[9305].gravity.possessions == 0 and players[9305].gravity.shown is False

    # rebounding: chances, rebounds per 100 and the model's expectation; 200 chances stay hidden
    for player_id, (games, chances, got, p) in FX.REBOUND_CHANCES.items():
        view, total = players[player_id].rebounding, chances * len(games)
        assert (view.chances, view.shown) == (total, total >= 300)
        if view.shown:
            assert view.rate_per_100.value == round(100 * got / chances, 2)
            assert view.model_expected_per_100 == round(100 * p, 2)
            assert view.rate_per_100.lo == view.rate_per_100.hi == view.rate_per_100.value   # every game is alike
        else:
            assert view.rate_per_100 is None and view.model_expected_per_100 is None

    # absence: the observed rate of training games, and the rebuilt out-of-fold probability beside it
    for player_id, games in FX.PLAYS_IN.items():
        if player_id not in players:
            continue
        view = players[player_id].absence_risk
        known = [g for g in games if g != max(games)]          # the last game has no next game
        absent = [g for g in known if g in FX.ABSENT_AFTER.get(player_id, ())]
        assert (view.games, view.absent, view.shown) == (len(known), len(absent), len(known) >= 10)
        assert view.kind == ("model_out_of_fold" if known else "observed_only")   # no rebuilt row without a known game
        if view.shown:
            assert view.observed_rate.value == round(len(absent) / len(known), 4)
            assert view.model_probability.value == pytest.approx(
                np.mean([FX.absence_probability(player_id, g) for g in known]), abs=6e-5)
            assert view.model_probability.n == len(known) and view.model_probability.point_estimate is False
        else:
            assert view.observed_rate is None and view.model_probability is None

    # similar players: the lane's neighbours for a player above its gate
    for player_id, (possessions, insufficient) in FX.EMBEDDING_POSSESSIONS.items():
        view = players[player_id].similar
        assert (view.available, view.possessions, view.shown) == (True, possessions, not insufficient)
        assert [(n.rank, n.player_id, n.name) for n in view.neighbours] == (
            [] if insufficient else [(k + 1, other, FX.player_name(other)) for k, other in enumerate(FX.NEIGHBOURS[player_id])])
    assert players[9303].similar.shown is False and players[9303].similar.available is True

    # names, teams in date order, positions through bref_slug
    assert players[FX.TRADED].teams == ["FXA", "FXB"] and players[9301].teams == ["FXA"]
    assert {p: players[p].position for p in (9301, 9302, 9303, 9304, 9305, 9501)} == {
        9301: "G", 9302: "F", 9303: "C", 9304: None, 9305: None, 9501: "G"}
    assert players[9301].games == 12 and players[9304].games == 1
    assert len({p.slug for p in file.players}) == len(file.players) and players[9301].slug == "fixture-player-9301"

    # the rebuilt outputs travel with their passed checks; no source of this file reads a held-out row
    assert sorted((c.name, c.passed) for c in file.checks) == [("absence", True), ("embeddings", True)]
    sources = S._all_sources(file)
    assert sources and all(s.rows != "heldout" for s in sources)
    assert {s.rows for s in file.sources} == {"out_of_fold", "training", "not_split"}


def test_shot_making_baseline_is_read_from_the_xfg_report(built):
    file = S.PlayersFile.model_validate(_json(built.out / "players.json"))
    baseline = file.shot_making_baseline
    text = (REPO / "reports" / "xfg.md").read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if row.startswith("| contest_logistic |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert (baseline.name, baseline.selected) == ("contest_logistic", False) and cells[-1] == ""
    assert cells[3] == "%.4f [%.4f, %.4f]" % (baseline.ece.value, baseline.ece.lo, baseline.ece.hi)
    assert baseline.ece.interval in text and baseline.ece.interval.startswith("bootstrap by game")
    assert f"{baseline.ece.n} attempts" in text
    section = text.split("### contest_logistic (out-of-fold)")[1].split("###")[0]
    for row in baseline.calibration:
        assert re.search(r"\| %.2f \| %.2f \| %d \| %.4f \| %.4f \|" % (row.lo, row.hi, row.n, row.predicted, row.observed),
                         section), row
    assert len(baseline.calibration) == 9 and sum(r.n for r in baseline.calibration) == baseline.ece.n
    note = next(m for m in file.metrics if m.metric == "shot_making")
    assert "baseline_error_per_100" in " ".join(note.notes)


def test_metric_notes_quote_the_write_up_and_type_no_number(built):
    file = S.PlayersFile.model_validate(_json(built.out / "players.json"))
    results = (REPO / "reports" / "results.md").read_text(encoding="utf-8")
    reports = results + "".join((REPO / "reports" / f"{lane}.md").read_text(encoding="utf-8")
                                for lane in site_players.METRIC_LANES.values())
    # numbers a note may hold besides those of a report: the export's own settings, and counts of its own tables
    own = {"95", "200", "0", "100", "300", "10", "45", "49", "47", "51", "52"}
    info = built.info["players"]
    own |= {str(info["attempts_total"]), str(info["attempts_left_out"]), str(info["games_with_attempts"]),
            str(info["games_without_shot_frames"])} | {str(n) for n in info["attempts_left_out_by_reason"].values()}
    for note in file.metrics:
        lane = site_players.METRIC_LANES[note.metric]
        assert f"| {lane} |" in results and f"| {note.verdict} |" in results   # the verdict cell, word for word
        assert [s.path for s in note.sources] == ["reports/results.md", f"reports/{lane}.md"]
        for text in note.notes:
            for number in re.findall(r"\d+(?:\.\d+)?", text):
                assert number in own or re.search(r"(?<![\d.])%s(?![\d])" % re.escape(number), reports), (note.metric, number)


def test_metric_notes_label_observed_rates_and_plain_numbers(built):
    """A value shows its interval or says it is a point estimate, and an observed rate is labelled descriptive.
    The contract types the numbers beside an estimate as plain floats, so the metric notes carry both statements."""
    file = S.PlayersFile.model_validate(_json(built.out / "players.json"))
    notes = {m.metric: " ".join(m.notes) for m in file.metrics}
    plain = {metric: [name for name, info in model.model_fields.items() if "float" in str(info.annotation)]
             for metric, model in (("shot_making", S.ShotMaking), ("rebounding", S.Rebounding))}
    assert plain == {"shot_making": ["fg_pct", "expected_fg_pct", "baseline_error_per_100",
                                     "frozen_model_residual_per_100"], "rebounding": ["model_expected_per_100"]}
    for metric, names in plain.items():                        # a new plain number must be named here too
        sentence = next(n for n in file.metrics if n.metric == metric).notes[-1]
        assert sentence.startswith("Interval: ") and "no interval was computed" in sentence
        assert all(name in sentence.split("point estimate")[0] for name in names), metric
    assert "It belongs to over_expected_per_100." in notes["shot_making"]
    assert "It belongs to rate_per_100." in notes["rebounding"]
    assert "fg_pct is descriptive" in notes["shot_making"]
    assert notes["rebounding"].startswith("The rate is descriptive")
    for metric in ("gravity", "absence_risk", "similar"):      # the other observed tables say it as well
        assert "escriptive" in notes[metric], metric


def test_players_without_rebuilt_outputs_show_what_is_stored(built, tmp_path):
    file, info = site_players.build_players(built.data, tmp_path / "no_artifacts")
    assert info.absence.startswith("observed_only (") and info.similar.startswith("not available (")
    assert file.checks == []
    for player in file.players:
        assert player.absence_risk.kind == "observed_only" and player.absence_risk.model_probability is None
        assert (player.similar.available, player.similar.shown, player.similar.neighbours) == (False, False, [])
    assert next(p for p in file.players if p.player_id == 9301).absence_risk.observed_rate.value == round(3 / 11, 4)
    notes = {m.metric: " ".join(m.notes) for m in file.metrics}
    assert "only the observed rate is shown" in notes["absence_risk"] and "no similar players are shown" in notes["similar"]
    assert S._content_problems("players.json", file, synthetic=False) == []


def test_a_rebuilt_output_is_used_only_with_a_check_that_covers_it(built, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(built.data))
    record = dict(name="absence", compared="fixture", n=2, max_abs_diff=1e-5, tolerance=5e-5)
    # a check that did not pass: the output is not used, and the file says observed_only
    failed = tmp_path / "failed"
    FX.write_artifacts(failed, built.data, built.tables, embeddings=False, demo=False)
    frozen_common.write_check(failed / "absence", dict(record, max_abs_diff=1e-3), files=["oof.parquet"])
    file, info = site_players.build_players(built.data, failed)
    assert info.absence.startswith("observed_only") and file.checks == []
    # a passed check that vouches for another file: a problem, not a silent fallback
    other = tmp_path / "other"
    FX.write_artifacts(other, built.data, built.tables, embeddings=False, demo=False)
    (other / "absence" / "notes.txt").write_text("x")
    frozen_common.write_check(other / "absence", record, files=["notes.txt"])
    with pytest.raises(C.ExportProblem, match="does not vouch for"):
        site_players.build_players(built.data, other)
    # a held-out game among the rebuilt rows stops the build
    leaked = tmp_path / "leaked"
    FX.write_artifacts(leaked, built.data, built.tables, embeddings=False, demo=False)
    oof = pd.read_parquet(leaked / "absence" / "oof.parquet")
    oof.loc[0, "sportvu_game"] = FX.HELD_B
    oof.to_parquet(leaked / "absence" / "oof.parquet", index=False)
    frozen_common.write_check(leaked / "absence", record, files=["oof.parquet"])
    frozen_common.publish_checks(leaked, out=FX.tracked_checks(leaked))
    with pytest.raises(C.ExportProblem, match="held-out game"):
        site_players.build_players(built.data, leaked, checks_copy=FX.tracked_checks(leaked))


def test_a_rebuilt_output_needs_the_tracked_copy_of_its_check(built, tmp_path):
    """PLAN.md D49: the artifacts live outside git, so a value built on one must trace to the tracked copy of its
    check record. Without a copy that equals the artifact's own check, the output is not used."""
    used = S.PlayersFile.model_validate(_json(built.out / "players.json"))
    paths = [s.path for s in used.sources]
    assert site_players.CHECKS_COPY == "reports/export/frozen_checks.json" and site_players.CHECKS_COPY in paths
    assert "artifacts/absence/oof.parquet" in paths and "artifacts/embeddings/embedding_neighbours.parquet" in paths
    assert built.info["players"]["absence"] == "model_out_of_fold" and built.info["players"]["similar"] == "available"

    # no tracked copy: both rebuilt outputs passed their checks, and neither is used
    file, info = site_players.build_players(built.data, built.artifacts, checks_copy=tmp_path / "absent.json")
    assert info.absence.startswith("observed_only (reports/export/frozen_checks.json is missing")
    assert info.similar.startswith("not available (reports/export/frozen_checks.json is missing")
    assert file.checks == []
    assert not any(s.path.startswith("artifacts/") or s.path == site_players.CHECKS_COPY for s in S._all_sources(file))
    assert all(p.absence_risk.kind == "observed_only" and p.absence_risk.model_probability is None
               and not p.similar.available and not p.similar.neighbours for p in file.players)
    notes = {m.metric: " ".join(m.notes) for m in file.metrics}
    assert "only the observed rate is shown" in notes["absence_risk"] and "no similar players are shown" in notes["similar"]

    # a tracked copy that holds the record of another run of the absence rebuild: absence falls back, the
    # embeddings output, whose record still equals its check, is used
    doc = _json(built.checks)
    stale = copy.deepcopy(doc)
    next(e for e in stale["checks"] if e["name"] == "absence")["written"] = "2000-01-01T00:00:00Z"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(stale))
    file, info = site_players.build_players(built.data, built.artifacts, checks_copy=path)
    assert info.absence.startswith("observed_only (reports/export/frozen_checks.json: the record of absence is not")
    assert "['written']" in info.absence and info.similar == "available"
    assert [c.name for c in file.checks] == ["embeddings"]
    assert not any(s.path.startswith("artifacts/absence") for s in file.sources)

    # the rule itself, field by field
    record = frozen_common.read_check(built.artifacts / "absence")
    assert site_players._tracked_check(built.checks, "absence", record) == ""
    for change, why in (({"n": record["n"] + 1}, "['n']"), ({"max_abs_diff": 0.0}, "['max_abs_diff']"),
                        ({"tolerance": 1e-9}, "['tolerance']"), ({"compared": "something else"}, "['compared']"),
                        ({"passed": False}, "does not say that the absence rebuild passed"),
                        ({"reasons": ["nothing was compared"]}, "does not say that the absence rebuild passed"),
                        ({"name": "epv"}, "holds 0 records of absence")):
        other = copy.deepcopy(doc)
        next(e for e in other["checks"] if e["name"] == "absence").update(change)
        path.write_text(json.dumps(other))
        assert why in site_players._tracked_check(path, "absence", record), change
    path.write_text(json.dumps({"checks": doc["checks"] + doc["checks"]}))
    assert "holds 2 records of absence" in site_players._tracked_check(path, "absence", record)
    path.write_text("[]")
    assert "cannot be read" in site_players._tracked_check(path, "absence", record)


def test_the_out_of_fold_cut_is_checked_in_code(built, tmp_path):
    table = pd.DataFrame({"sportvu_game": FX.TRAIN_GAMES[:4], "fold": [0, -1, -2, 3], "heldout": [False, True, False, False]})
    assert site_players.training_rows(table, "t")["fold"].tolist() == [0, 3]
    with pytest.raises(C.ExportProblem, match="flagged held out"):
        site_players.training_rows(table.assign(heldout=[True, True, False, False]), "t")
    with pytest.raises(C.ExportProblem, match="held-out game"):
        site_players.training_rows(table.assign(sportvu_game=[FX.HELD_A] + FX.TRAIN_GAMES[1:4]), "t")
    with pytest.raises(C.ExportProblem, match="no fold column"):
        site_players.training_rows(table.drop(columns="fold"), "t")
    with pytest.raises(C.ExportProblem, match="no out-of-fold row"):
        site_players.training_rows(table.assign(fold=-1), "t")
    # in the data: a held-out row among the rebound predictions, a held-out game among the training rosters
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    path = data / "analytics" / "rebounds" / FX.SEASON / f"{FX.TRAIN_GAMES[3]}.parquet"
    rebounds = pd.read_parquet(path)
    rebounds.loc[0, "heldout"] = True
    rebounds.to_parquet(path, index=False)
    with pytest.raises(C.ExportProblem, match="rebound predictions: a row flagged held out"):
        site_players.build_players(data, built.artifacts)
    rebounds.loc[0, "heldout"] = False
    rebounds.to_parquet(path, index=False)
    rosters = data / "analytics" / "rosters" / FX.SEASON
    shutil.copy(rosters / f"{FX.TRAIN_GAMES[0]}.parquet", rosters / f"{FX.HELD_B}.parquet")
    with pytest.raises(C.ExportProblem, match="holds held-out game"):
        site_players.build_players(data, built.artifacts)


def test_stored_predictions_must_reproduce_the_reports_calibration_table(built, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    path = data / "analytics" / "xfg" / FX.SEASON / "xfg_predictions.parquet"
    table = pd.read_parquet(path)
    row = table.index[(table["fold"] >= 0) & (table["p_contest"] > 0.3) & (table["p_contest"] < 0.4)][0]
    table.loc[row, "p_contest"] = np.float32(0.55)             # one attempt moves to another calibration bin
    table.to_parquet(path, index=False)
    with pytest.raises(C.ExportProblem, match="not the report's"):
        site_players.build_players(data, built.artifacts)


def test_position_mapping_is_the_embeddings_lanes(built):
    from analytics.embeddings import train as lane             # export/ may not import a lane's train module

    assert site_players.POSITION_MAP == lane.POSITION_MAP
    for value in ("PG", "SG", "SF", "PF", "C", "G", "F", "SF-SG", "C-PF", " pf ", "X", None, float("nan")):
        theirs = lane.collapse_position(value)
        assert site_players.collapse_position(value) == (None if isinstance(theirs, float) else theirs)
    training = built.tables["training"]
    rosters = pd.concat(training["rosters"].values(), ignore_index=True)
    theirs = lane.normalise_positions(training["positions"], rosters)
    theirs = theirs[theirs["season"] == FX.SEASON]
    mine = site_players._positions(built.data, rosters)
    assert mine == {int(p): v for p, v in zip(theirs["player_id"], theirs["position"])}


def test_bootstrap_by_game_is_seeded_and_needs_two_games():
    table = pd.DataFrame({"player_id": [1, 1, 1, 2, 3, 3], "sportvu_game": list("abcabc"),
                          "s": [1.0, 2.0, 6.0, 4.0, 0.0, 5.0], "n": [2.0, 2.0, 4.0, 2.0, 5.0, 5.0]})
    first, second = C.bootstrap_by_game(table, "s", "n", scale=100.0), C.bootstrap_by_game(table.iloc[::-1], "s", "n", scale=100.0)
    pd.testing.assert_frame_equal(first, second)               # the row order does not matter
    assert first.at[1, "point"] == pytest.approx(100 * 9 / 8) and first.at[1, "lo"] <= first.at[1, "hi"]
    assert np.isnan(first.at[2, "lo"]) and first.at[2, "n_games"] == 1
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 3, size=(200, 3))
    s, n = np.array([1.0, 2.0, 6.0]), np.array([2.0, 2.0, 4.0])
    lo, hi = np.percentile(100 * s[draws].sum(axis=1) / n[draws].sum(axis=1), [2.5, 97.5])
    assert (first.at[1, "lo"], first.at[1, "hi"]) == (pytest.approx(lo), pytest.approx(hi))
    with pytest.raises(C.ExportProblem):
        C.estimate(float("nan"), 0, 1, 2, 10, "games", "x")
    assert C.estimate(1.234, float("nan"), float("nan"), 2, 10, "games", "x").point_estimate is True


# ------------------------------------------------------------------------------------------- the command


def test_demo_files_are_listed_and_never_touched(built, tmp_path):
    out = tmp_path / "out"
    sample.write_sample(tmp_path / "sample")
    (out / "demo").mkdir(parents=True)
    before = {}
    for name in ("swap.json", "frames.json"):
        shutil.copy(tmp_path / "sample" / "demo" / name, out / "demo" / name)
        before[name] = (out / "demo" / name).read_bytes()
    _problems, info = site_data.run(out, None, built.artifacts, only=["scorecard", "results"], log=QUIET)
    assert info["manifest"]["demo_files"] == ["demo/swap.json", "demo/frames.json"]
    manifest = S.Manifest.model_validate(_json(out / "manifest.json"))
    listed = {entry.path: entry for entry in manifest.files}
    for name, data in before.items():
        assert (out / "demo" / name).read_bytes() == data
        assert listed[f"demo/{name}"].bytes == len(data) and listed[f"demo/{name}"].sources
    assert listed["demo/swap.json"].model == "DemoSwap" and listed["demo/frames.json"].model == "DemoFrames"


def test_stale_files_are_reported_and_never_deleted(built, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(built.out, out)
    slug = _slug(FX.HELD_A)
    stale = out / "games" / slug / "999.json"
    stale.write_text((out / "games" / slug / "0.json").read_text().replace('"possession":0,', '"possession":999,'))
    with pytest.raises(C.ExportProblem, match="999.json"):
        site_data.run_games(built.data, built.artifacts, out, log=QUIET)
    assert stale.exists()
    stale.unlink()
    old = out / "games" / "2015-01-01-old-at-run"
    old.mkdir()
    (old / "possessions.json").write_text("{}")
    with pytest.raises(C.ExportProblem, match="2015-01-01-old-at-run"):
        site_data.run_games(built.data, built.artifacts, out, log=QUIET)
    assert (old / "possessions.json").exists()


def test_a_subset_of_games_and_of_steps(built, tmp_path):
    out = tmp_path / "out"
    problems, info = site_data.run(out, built.data, built.artifacts, games=[_slug(FX.HELD_B)], log=QUIET,
                                   allow_fixture_root=True)
    assert problems == [] and info["games"]["partial"] is True and info["manifest"]["n_games"] == 1
    # the fifths come from all held-out games, also in a run of one game
    assert _json(out / "games" / _slug(FX.HELD_B) / "possessions.json")["xfg_bucket_edges"] == \
        _index(built, FX.HELD_B)["xfg_bucket_edges"]
    # another game into the same folder would leave two runs side by side: refused, nothing removed
    problems, _info = site_data.run(out, built.data, built.artifacts, games=[FX.HELD_A], only=["games"], log=QUIET,
                                    allow_fixture_root=True)
    assert any("not written by this run" in line for line in problems)
    assert (out / "games" / _slug(FX.HELD_B) / "0.json").exists() and not (out / "games" / _slug(FX.HELD_A)).exists()
    problems, _info = site_data.run(out, built.data, built.artifacts, games=["11.01.2015.FXA.at.FXB"], only=["games"],
                                    log=QUIET, allow_fixture_root=True)
    assert any("are not held-out games on the data root" in line for line in problems)
    before = (out / "players.json").read_bytes()
    problems, info = site_data.run(out, built.data, built.artifacts, only=["results"], log=QUIET, allow_fixture_root=True)
    assert problems == [] and info["steps"] == ["results"] and (out / "players.json").read_bytes() == before


def test_real_data_built_from_uncommitted_code_is_a_problem(built, tmp_path, monkeypatch):
    """The manifest of real data must name a commit that holds the code and the reports that built it."""
    def manifest_problems(problems):
        return [line for line in problems if line.startswith("manifest: repo_commit is")]

    monkeypatch.setattr(C, "repo_commit", lambda: "0" * 40 + C.UNCOMMITTED)
    problems, info = site_data.run(tmp_path / "real", None, built.artifacts, only=["scorecard", "results"], log=QUIET)
    assert len(manifest_problems(problems)) == 1 and "Commit them and run again" in manifest_problems(problems)[0]
    assert info["manifest"]["repo_commit"] == "0" * 40 + "+uncommitted"
    assert _json(tmp_path / "real" / "manifest.json")["repo_commit"].endswith("+uncommitted")   # said, never hidden
    # a tree a test builds from its synthetic root is not real data (its build id says so): no such problem
    problems, info = site_data.run(tmp_path / "fixture", built.data, built.artifacts, only=["results"], log=QUIET,
                                   allow_fixture_root=True)
    assert manifest_problems(problems) == [] and info["manifest"]["build_id"].startswith("test-fixture-")
    assert info["manifest"]["repo_commit"].endswith("+uncommitted")
    monkeypatch.setattr(C, "repo_commit", lambda: "0" * 40)
    problems, _info = site_data.run(tmp_path / "clean", None, built.artifacts, only=["scorecard", "results"], log=QUIET)
    assert manifest_problems(problems) == []


def test_the_command_refuses_a_test_data_root_and_unsafe_outputs(built, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))      # never the machine-wide lock in a test
    out = tmp_path / "out"
    status = site_data.main(["--out", str(out), "--data", str(built.data), "--artifacts", str(built.artifacts)])
    assert status == 1 and not out.exists()
    printed = capsys.readouterr().out
    assert "written by a test" in printed
    assert (tmp_path / "heavy.lock").exists() and "[heavy] 'export.site_data' done" in printed   # the run held the lock
    problems, _info = site_data.run(built.data / "site", built.data, built.artifacts, log=QUIET, allow_fixture_root=True)
    assert problems == ["refusing to write the site data under data"]
    problems, _info = site_data.run(REPO / "reports" / "site", None, built.artifacts, only=["results"], log=QUIET)
    assert problems == ["refusing to write the site data under reports"] and not (REPO / "reports" / "site").exists()
    problems, _info = site_data.run(out, None, built.artifacts, only=["players"], log=QUIET)
    assert problems == ["no data root: pass --data or set DATA_ROOT"]
    assert (site_data.SCORECARD_BUILD, site_data.CANDIDATE_BUILD) == ("r23", "r23_cam")


# ------------------------------------------------------------------------------------------- the committed reports


@pytest.fixture(scope="module")
def results_md():
    text = (REPO / "reports" / "results.md").read_text(encoding="utf-8")
    before, block, _after = R._split_existing(text)
    return before, block


def test_results_json_equals_reports_results_md(results_md):
    before, block = results_md
    results = site_reports.build_results(REPO)
    assert S._content_problems("results.json", results, synthetic=False) == []
    summary = [line for line in before.splitlines() if line.startswith("| ") and not line.startswith(("| lane", "| ---"))]
    assert [lane.lane for lane in results.lanes] == [line.split("|")[1].strip() for line in summary]
    assert len(results.lanes) == 11
    sections = dict(re.findall(r"^### (\w+)(?: \(fixture output, not a result\))?\n\n(.*?)(?=^### |^<!-- results:end)", block,
                               re.M | re.S))
    for lane, line in zip(results.lanes, summary):
        # question, model name and verdict: the cells of the hand-written row, word for word
        assert line.startswith(f"| {lane.lane} | {lane.question} | {lane.selected_model} | ")
        assert line.endswith(f" | {lane.verdict} |")
        assert lane.verdict_source == "reports/results.md" and lane.report == f"reports/{lane.lane}.md"
        table = sections[lane.lane]
        # the generated lane table, row by row and in order
        assert [f"| {row.field} | {row.value} |" for row in lane.rows] == [
            row for row in table.splitlines() if row.startswith("| ") and not row.startswith(("| field", "| ---"))]
        assert f"| status | {lane.status} |" in table
        for field, metric in (("out-of-fold metric", lane.out_of_fold), ("held-out", lane.heldout)):
            if metric is None:
                assert f"| {field} | absent |" in table
                continue
            assert f"| {field} | {metric.text} |" in table
            rest = metric.text[len(metric.name) + 1:]          # the value and what follows it, as the report wrote them
            assert metric.text.startswith(metric.name + " ") and float(rest.split()[0]) == metric.value
            interval = re.match(r"\S+ \[(\S+), (\S+)\]", rest)
            assert (metric.lo, metric.hi) == ((float(interval.group(1)), float(interval.group(2))) if interval else (None, None))
            assert metric.point_estimate == (interval is None)
    # fixture reports are pending and carry no number
    assert [p.lane for p in results.pending] == re.findall(r"^### (\w+) \(fixture output, not a result\)$", block, re.M)
    assert [p.lane for p in results.pending] == ["actions", "coverage", "playcalling", "betting"]
    assert all(not re.search(r"\d", p.reason) and "fixture output" in p.reason for p in results.pending)
    assert results.generated == re.search(r"^Generated (\S+) by", block, re.M).group(1)
    assert results.freeze_tag == "freeze-v1" and "`freeze-v1`" in before
    assert " ".join(before.split()).count(results.notes[0]) == 1          # the sentence on intervals, copied


def test_verdict_class_comes_from_the_generated_tables_and_agrees_with_the_write_up():
    results = site_reports.build_results(REPO)
    assert {lane.lane: lane.verdict_class for lane in results.lanes} == {
        "xfg": "works", "rebounds": "works", "matchups": "no_signal", "gravity": "works", "ghosting": "works",
        "embeddings": "works", "epv": "works", "decisions": "failed", "lineups": "inconclusive", "winprob": "works",
        "load": "works"}
    doc = site_reports.read_results(REPO)
    bases = site_reports.class_bases(doc)
    for lane in results.lanes:                                 # no class is a default: something decided each one
        tables = site_reports.class_from_tables(doc.records[lane.lane])
        words = site_reports.class_from_words(lane.verdict)
        assert lane.verdict_class in (tables, words) and {tables, words} <= {lane.verdict_class, None}, lane.lane
        assert bases[lane.lane] == ("write-up" if tables is None else "tables" if words else "tables, write-up silent")
    # the lanes whose table holds nothing to decide by are named in the file, and so is the silent sentence
    assert [lane for lane, basis in bases.items() if basis == "write-up"] == ["xfg", "embeddings"]
    assert [lane for lane, basis in bases.items() if basis == "tables, write-up silent"] == ["gravity"]
    assert any(note.endswith("classed by the word of the write-up's own sentence instead: xfg, embeddings.")
               for note in results.notes)
    assert any(note.endswith("because the write-up's sentence names no class: gravity.") for note in results.notes)

    def record(name, value, interval="", candidates=(), held=None):
        return R.LaneResult("x", "reports/x.md", oof=R.Metric(name, value, interval),
                            candidates=[(n, R.Metric(name, v)) for n, v in candidates],
                            heldout=R.HeldOut(R.HO_SCORED, R.Metric(name, *held)) if held else R.HeldOut(R.HO_ABSENT))

    assert site_reports.class_from_tables(record("mean Spearman", "0.20", "[0.05, 0.31]")) == "works"
    assert site_reports.class_from_tables(record("mean Spearman", "0.01", "[-0.01, 0.03]")) == "failed"
    assert site_reports.class_from_tables(record("Spearman", "0.1", held=("0.06", "[-0.4, 0.5]"))) == "inconclusive"
    assert site_reports.class_from_tables(record("Spearman", "0.1")) == "inconclusive"
    assert site_reports.class_from_tables(record("Spearman", "0.1", held=("0.3", "[0.1, 0.5]"))) == "works"
    floor = (("constant", "1.2842"), ("raw", "1.3244"))
    assert site_reports.class_from_tables(record("points MSE", "1.2854", candidates=floor)) == "no_signal"
    assert site_reports.class_from_tables(record("points MSE", "1.2100", candidates=floor)) == "works"
    no_floor = record("log loss", "0.63", candidates=(("location_logistic", "0.60"),))
    assert site_reports.class_from_tables(no_floor) is None    # nothing to compare with: not `works` by default
    majority = (("1 majority (the common class)", "0.41"),)
    assert site_reports.class_from_tables(record("accuracy", "0.40", candidates=majority)) == "no_signal"
    assert site_reports.class_from_tables(record("accuracy", "0.77", candidates=majority)) == "works"
    # the words of a verdict sentence: a sentence that names no class is not read as works
    assert site_reports.class_from_words("works; beats the baseline") == "works"
    assert site_reports.class_from_words("passes its 0.60 gate; no held-out score exists") == "works"
    assert site_reports.class_from_words("no defender signal beats the constant") == "no_signal"
    assert site_reports.class_from_words("**failed**: no validated signal") == "failed"
    assert site_reports.class_from_words("expectation holds on held-out") is None
    # what decides: the tables when they can, else the sentence's word, and never a default
    assert site_reports.verdict_class("x", no_floor, "works; fine") == ("works", "write-up")
    assert site_reports.verdict_class("x", record("points MSE", "1.2100", candidates=floor), "holds up") == (
        "works", "tables, write-up silent")
    assert site_reports.verdict_class("x", record("points MSE", "1.2854", candidates=floor), "no signal") == (
        "no_signal", "tables")
    with pytest.raises(C.ExportProblem, match="not defaulted"):
        site_reports.verdict_class("x", no_floor, "better than the location baseline")
    with pytest.raises(C.ExportProblem, match="lane x: the generated tables read as no_signal"):
        site_reports.verdict_class("x", record("points MSE", "1.2854", candidates=floor), "works well")
    # the write-up and the tables must agree, or nothing is written
    doc.summary["epv"] = dict(doc.summary["epv"], verdict="failed: a sentence the tables do not support")
    with pytest.raises(C.ExportProblem, match="lane epv"):
        site_reports.build_results(REPO, doc)
    # a lane that nothing decides stops the build
    doc = site_reports.read_results(REPO)
    doc.summary["xfg"] = dict(doc.summary["xfg"], verdict="beats the location-only baseline on held-out")
    with pytest.raises(C.ExportProblem, match="lane xfg.*not defaulted"):
        site_reports.build_results(REPO, doc)


def _walk(block: dict, dotted: str):
    for key in dotted.split("."):
        block = block[key]
    return block


def test_scorecard_json_equals_the_scorecard_files():
    adopted, candidate = site_data.SCORECARD_BUILD, site_data.CANDIDATE_BUILD
    card = site_reports.build_scorecard(adopted, candidate, REPO)
    assert S._content_problems("scorecard.json", card, synthetic=False) == []
    assert [(b.build, b.role) for b in card.builds] == [(adopted, "adopted"), (candidate, "candidate")]
    labels = site_reports.ROW_LABELS
    for build in card.builds:
        files = {game: _json(REPO / "reports" / "scorecard" / f"{game}__{build.build}.json")
                 for game in site_reports.SCORECARD_GAMES}
        texts = {game: (REPO / "reports" / "scorecard" / f"{game}__{build.build}.txt").read_text()
                 for game in site_reports.SCORECARD_GAMES}
        targets = files["gsw_phx_2016"]["targets"]
        # the adopted build has one row per target; the candidate is a court replay, shown on the court rows only
        expected = list(targets) if build.role == "adopted" else list(site_reports.COURT_ROWS)
        assert [row.key for row in build.rows] == expected
        assert [g.game for g in build.games] == list(site_reports.SCORECARD_GAMES)
        for game in build.games:
            file = files[game.game]
            assert (game.game_id, game.split, game.truth_frames, game.testable, game.untestable_share) == (
                file["sportvu_game"], file["split"], file["truth_frames"], file["testable"], file["untestable_share"])
        for row in build.rows:
            assert row.target == targets[row.key]
            assert [cell.game for cell in row.cells] == list(site_reports.SCORECARD_GAMES)
            for cell in row.cells:
                block = files[cell.game]["metrics"][row.key]
                assert cell.passed is block["pass"] and cell.split == files[cell.game]["split"]
                assert cell.status == {True: "met", False: "not met"}.get(block["pass"], cell.status)
                if block["pass"] is None:
                    measurable = not (block.get("not_measurable") or str(block.get("pass_withheld", "")).startswith("not measurable"))
                    assert cell.status == ("reported" if measurable else "not measurable")
                numbers = block.get("value") or {} if row.key == "generalisation" else block
                for key, value in cell.values.items():
                    assert _walk(numbers, key) == value, (build.build, cell.game, row.key, key)
                assert cell.values or row.key == "generalisation"
                stated = re.search(r"^%s\s{2,}(?:MET|not met|n/a|reported|-)\s+(.*)$" % re.escape(labels[row.key]),
                                   texts[cell.game], re.M).group(1)
                assert cell.text == stated.rstrip() and cell.interval is None
                if cell.n is not None:
                    assert cell.n in [float(v) for v in block.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for game, file in files.items():
            for caveat in file["caveats"]:
                assert caveat in build.caveats or f"{game}: {caveat}" in build.caveats
            for key, value in file["method"].items():
                assert card.method[key] == str(value)
    assert any("best of the three training seeds" in c for c in card.builds[0].caveats)       # PLAN.md D52
    assert "is the best of its three seeds" in (REPO / "ROADMAP_LOG.md").read_text(encoding="utf-8")
    compare = _json(REPO / "reports" / "scorecard" / f"compare__{adopted}_replay__{candidate}.json")
    assert len(card.paired) == 3 * 4
    for diff in card.paired:
        stored = compare["games"][diff.game]["paired"][candidate][diff.metric]
        assert (diff.baseline, diff.build) == (f"{adopted}_replay", candidate) == tuple(compare["builds"])
        assert (diff.diff, [diff.lo, diff.hi]) == (stored["diff"], stored["ci95"])
        assert diff.excludes_zero == (diff.lo > 0 or diff.hi < 0)
        block = compare["games"][diff.game]["paired"][candidate]
        assert card.method[f"paired_n_{diff.game}"] == f"{block['windows']} windows, {block['observations']} observations"
    assert all(s.path.startswith("reports/scorecard/") or s.path == "ROADMAP_LOG.md" for s in card.sources)


def test_scorecard_text_must_equal_the_line_of_the_text_file(tmp_path):
    repo = tmp_path / "repo"
    (repo / "reports" / "scorecard").mkdir(parents=True)
    names = [f"{game}__{build}" for game in site_reports.SCORECARD_GAMES for build in ("r23", "r23_cam")]
    for name in names + ["compare__r23_replay__r23_cam"]:     # named files only: reports/ is never globbed
        for suffix in (".json", ".txt"):
            shutil.copy(REPO / "reports" / "scorecard" / (name + suffix), repo / "reports" / "scorecard" / (name + suffix))
    shutil.copy(REPO / "ROADMAP_LOG.md", repo / "ROADMAP_LOG.md")
    site_reports.build_scorecard("r23", "r23_cam", repo)
    path = repo / "reports" / "scorecard" / "cle_nyk_2015__r23.json"
    doc = _json(path)
    doc["metrics"]["position_error"]["p50_ft"] = 9.99          # the json and the text file no longer agree
    path.write_text(json.dumps(doc))
    with pytest.raises(C.ExportProblem, match="is not the line of the text file"):
        site_reports.build_scorecard("r23", "r23_cam", repo)


def test_a_stale_results_file_stops_the_build(tmp_path):
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    for name in ["results"] + site_reports._lane_reports():
        shutil.copy(REPO / "reports" / f"{name}.md", repo / "reports" / f"{name}.md")
    site_reports.read_results(repo)
    path = repo / "reports" / "xfg.md"
    edited = path.read_text().replace("0.6294", "0.6194")      # the lane report changes, results.md was not regenerated
    assert edited != path.read_text()
    path.write_text(edited)
    with pytest.raises(C.ExportProblem, match="does not equal what the lane reports render to"):
        site_reports.read_results(repo)

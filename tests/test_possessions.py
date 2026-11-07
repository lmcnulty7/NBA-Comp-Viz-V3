"""tests/test_possessions.py: analytics.possessions on synthetic frames and a scripted event list."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import possessions as P  # noqa: E402

HOME, AWAY = 10, 20
HZ, PERIOD_S = 25, 720.0
JUMP_AT, JUMP_S = 400.0, 5.0


def synthetic_frames(period: int = 1) -> pd.DataFrame:
    """25 Hz frame-level clocks over one 720 s period with one 5 s jump at 400 s, 10 rows per frame."""
    clocks = PERIOD_S - np.arange(int(PERIOD_S * HZ) + 1) / HZ
    clocks = clocks[~((clocks < JUMP_AT) & (clocks > JUMP_AT - JUMP_S))]
    n = len(clocks)
    fl = pd.DataFrame({"frame": np.arange(n, dtype=np.int32), "period": np.int8(period),
                       "game_clock_s": clocks.astype(np.float32)})
    rows = fl.loc[fl.index.repeat(10)].reset_index(drop=True)
    rows["team_id"] = np.where(np.arange(len(rows)) % 10 < 5, HOME, AWAY).astype(np.int32)
    rows["home"] = rows["team_id"] == HOME
    rows["player_id"] = (np.arange(len(rows)) % 10 + 1).astype(np.int32)
    rows["x"] = rows["y"] = np.float32(0.0)
    rows["ball_x"] = rows["ball_y"] = rows["ball_z"] = np.float32(0.0)
    return rows


SCRIPT = [  # (clock_s, team, kind, points, desc)
    (720.0, None, "period_start", 0, "Start of 1st quarter"),
    (700.0, "home", "shot_made", 2, "A makes 2-pt jump shot"),
    (680.0, "away", "shot_missed", 0, "B misses 3-pt jump shot"),
    (678.0, "home", "rebound", 0, "Defensive rebound by A"),
    (660.0, "home", "shot_missed", 0, "A misses 2-pt layup"),
    (658.0, "home", "rebound", 0, "Offensive rebound by C"),
    (650.0, "home", "shot_made", 3, "C makes 3-pt jump shot"),
    (630.0, "away", "turnover", 0, "Turnover by B (bad pass)"),
    (610.0, "away", "foul", 0, "Shooting foul by B (drawn by A)"),
    (610.0, "home", "ft_made", 1, "A makes free throw 1 of 2"),
    (610.0, "home", "ft_made", 1, "A makes free throw 2 of 2"),
    (380.0, "away", "shot_made", 2, "B makes 2-pt layup"),
    (300.0, "home", "shot_missed", 0, "A misses 2-pt jump shot"),
    (298.0, "away", "rebound", 0, "Defensive rebound by B"),
    (0.0, None, "period_end", 0, "End of 1st quarter"),
]


def scripted_events(frames: pd.DataFrame, script=SCRIPT, period: int = 1, no_frame=()) -> pd.DataFrame:
    fl = P.frame_level(frames)
    clocks = fl["game_clock_s"].to_numpy(dtype=float)
    rows = []
    for i, (clock, team, kind, points, desc) in enumerate(script):
        frame = -1 if i in no_frame else int(fl["frame"].iloc[int(np.argmin(np.abs(clocks - clock)))])
        rows.append({"event": i, "period": period, "clock_s": clock, "team": team, "desc": desc,
                     "points": points, "score": None, "kind": kind, "frame": frame})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frames():
    return synthetic_frames()


@pytest.fixture(scope="module")
def poss(frames):
    return P.segment(frames, scripted_events(frames, no_frame=(7,)), "synthetic")


def test_count_and_end_kinds(poss):
    assert len(poss) == 8
    assert poss["end_kind"].tolist() == ["shot_made", "shot_missed", "shot_made", "turnover", "free_throws",
                                         "shot_made", "shot_missed", "period_end"]
    assert poss["possession"].tolist() == list(range(8))
    assert poss.attrs["n_dropped"] == 0


def test_points_and_offense(poss):
    assert poss["points"].tolist() == [2, 0, 3, 0, 2, 2, 0, 0]
    assert poss["offense_team_id"].tolist() == [HOME, AWAY, HOME, AWAY, HOME, AWAY, HOME, AWAY]
    assert poss["end_event"].tolist() == [1, 3, 6, 7, 10, 11, 13, 14]
    assert poss["start_event"].tolist() == [0, 2, 4, 7, 8, 11, 12, 14]


def test_gap_only_on_the_jump(poss):
    assert poss["gap"].tolist() == [False, False, False, False, False, True, False, False]
    spanning = poss[poss["gap"]].iloc[0]
    assert spanning["start_clock_s"] > JUMP_AT > spanning["end_clock_s"]


def test_frames_and_clocks(poss, frames):
    fl = P.frame_level(frames)
    assert (poss["start_clock_s"] >= poss["end_clock_s"]).all()
    assert (poss["n_frames"] == poss["end_frame"] - poss["start_frame"] + 1).all()
    assert poss["start_frame"].iloc[0] == 0 and poss["end_frame"].iloc[-1] == fl["frame"].iloc[-1]
    # consecutive possessions share the boundary frame, except after the turnover (frame -1): its end is the
    # nearest frame by clock and the next possession starts at the next frame with a smaller clock
    starts, ends = poss["start_frame"].tolist(), poss["end_frame"].tolist()
    assert [s - e for s, e in zip(starts[1:], ends[:-1])] == [0, 0, 0, 1, 0, 0, 0]
    assert abs(poss["end_clock_s"].iloc[3] - 630.0) < 1.0 / HZ
    assert poss["period"].tolist() == [1] * 8


def test_schema(poss):
    assert list(poss.columns) == P.COLUMNS
    assert str(poss["possession"].dtype) == "int32" and str(poss["period"].dtype) == "int8"
    assert str(poss["start_clock_s"].dtype) == "float32" and str(poss["points"].dtype) == "int16"
    assert str(poss["offense_team_id"].dtype) == "Int32" and str(poss["gap"].dtype) == "bool"
    assert set(poss["end_kind"]) <= set(P.END_KINDS)


def test_summary(poss):
    s = P.summary(poss)
    assert s["n"] == 8 and s["points_total"] == 9 and s["n_gap"] == 1 and s["n_dropped"] == 0
    assert s["by_end_kind"] == {"shot_made": 3, "shot_missed": 2, "free_throws": 1, "turnover": 1, "period_end": 1}
    assert s["share_offense_known"] == 1.0
    assert abs(s["mean_duration_s"] - 720.0 / 8) < 1.0
    json.dumps(s)


def test_missed_last_free_throw_and_synthesized_period_end(frames):
    script = [
        (700.0, "away", "foul", 0, "Shooting foul by B"),
        (700.0, "home", "ft_made", 1, "A makes free throw 1 of 2"),
        (700.0, "home", "ft_missed", 0, "A misses free throw 2 of 2"),
        (698.0, "home", "rebound", 0, "Offensive rebound by C"),
        (690.0, "home", "shot_missed", 0, "C misses 2-pt layup"),
        (689.0, "away", "rebound", 0, "Defensive rebound by B"),
        (660.0, "away", "foul", 0, "Shooting foul by B"),
        (660.0, "home", "ft_missed", 0, "A misses free throw 1 of 1"),
        (659.0, "away", "rebound", 0, "Defensive rebound by B"),
        (640.0, "away", "shot_made", 2, "B makes 2-pt layup"),
        (640.0, "home", "foul", 0, "Shooting foul by A"),
        (640.0, "away", "ft_made", 1, "B makes free throw 1 of 1"),
        (600.0, "home", "shot_missed", 0, "A misses 3-pt jump shot"),
    ]
    poss = P.segment(frames, scripted_events(frames, script), "synthetic")
    assert poss["end_kind"].tolist() == ["shot_missed", "free_throws", "free_throws", "period_end"]
    assert poss["points"].tolist() == [1, 0, 3, 0]
    assert poss["offense_team_id"].tolist() == [HOME, HOME, AWAY, HOME]
    assert poss["end_event"].iloc[-1] == 12 and poss["end_frame"].iloc[-1] == P.frame_level(frames)["frame"].iloc[-1]


def test_no_stub_possession_after_a_closing_last_event(frames):
    script = [(700.0, "home", "shot_made", 2, "A makes 2-pt jump shot")]
    poss = P.segment(frames, scripted_events(frames, script), "synthetic")
    assert poss["end_kind"].tolist() == ["shot_made"] and poss["points"].tolist() == [2]


def test_free_throws_by_the_other_team_split_the_possession(frames):
    script = [
        (700.0, "home", "shot_missed", 0, "A misses 2-pt jump shot"),
        (698.0, "home", "rebound", 0, "Offensive rebound by C"),
        (690.0, "home", "foul", 0, "Loose ball foul by C (drawn by B)"),
        (690.0, "away", "ft_made", 1, "B makes free throw 1 of 2"),
        (690.0, "away", "ft_made", 1, "B makes free throw 2 of 2"),
        (670.0, "home", "shot_made", 2, "A makes 2-pt layup"),
        (650.0, "home", "foul", 0, "Technical foul by A"),
        (650.0, "away", "ft_made", 1, "B makes technical free throw"),
        (640.0, "away", "shot_made", 2, "B makes 2-pt layup"),
    ]
    poss = P.segment(frames, scripted_events(frames, script), "synthetic")
    assert poss["end_kind"].tolist()[:3] == ["other", "free_throws", "shot_made"]
    assert poss["points"].tolist()[:3] == [0, 2, 2]
    assert poss["offense_team_id"].tolist()[:3] == [HOME, AWAY, HOME]
    assert poss["end_event"].tolist()[:3] == [2, 4, 5]
    # a technical free throw never ends or splits a possession; its point goes to the shooting team
    assert poss["end_kind"].tolist()[3:] == ["shot_made"] and poss["points"].tolist()[3:] == [3]


def test_game_meta_and_output_path(tmp_path):
    m = P.game_meta("12.16.2015.PHX.at.GSW")
    assert m == {"game": "12.16.2015.PHX.at.GSW", "date": "2015-12-16", "away": "PHX", "home": "GSW", "season": "2015-16"}
    assert P.game_meta("01.18.2016.GSW.at.CLE")["season"] == "2015-16"
    assert P.output_path("12.16.2015.PHX.at.GSW", tmp_path) == tmp_path / "heldout" / "2015-16" / "12.16.2015.PHX.at.GSW.parquet"
    assert P.output_path("11.14.2015.BKN.at.GSW", tmp_path) == tmp_path / "2015-16" / "11.14.2015.BKN.at.GSW.parquet"
    assert P.output_path("12.23.2015.NYK.at.CLE", tmp_path).parent.parent.name == "heldout"

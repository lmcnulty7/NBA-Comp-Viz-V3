"""tests/test_shot_frames.py: analytics.shot_frames on a synthetic 25 Hz game (no real tracking data)."""
import numpy as np
import pandas as pd
import pytest

from analytics import shot_frames as sfm

HZ, N_FRAMES = 25, 1100
PLAYERS = [  # (player_id, name, home, team_id, team, resting position)
    (1, "Stephen Curry", True, 100, "GSW", (20.0, 25.0)),
    (2, "Klay Thompson", True, 100, "GSW", (30.0, 10.0)),
    (3, "Eric Bledsoe", False, 200, "PHX", (75.0, 30.0)),
    (4, "Brandon Knight", False, 200, "PHX", (60.0, 40.0)),
]
# scripted shots: (shooter index, carry start, release frame, apex frame, basket)
SHOTS = [(0, 100, 150, 175, (5.25, 25.0)), (2, 400, 450, 475, (88.75, 25.0)), (1, 600, 650, 675, (5.25, 25.0))]


def clock_of(frame: int) -> float:
    return 720.0 - frame / HZ


def script_ball():
    """Ball position per frame: parked at mid court at z 5 unless a shot is scripted."""
    bx, by, bz = np.full(N_FRAMES, 47.0), np.full(N_FRAMES, 25.0), np.full(N_FRAMES, 5.0)
    for shooter, carry, rel, apex, (gx, gy) in SHOTS:
        px, py = PLAYERS[shooter][5]
        for f in range(carry, rel):                           # carried: at the shooter's feet, z 4
            bx[f], by[f], bz[f] = px, py, 4.0
        flight = 48                                           # rel .. rel + 47 in flight, then held under the rim
        for k in range(flight):
            f = rel + k
            t = k / (flight - 1)
            bx[f], by[f] = px + (gx - px) * t, py + (gy - py) * t
            bz[f] = 18.0 - 10.5 * ((f - apex) / 25.0) ** 2   # 7.5 at release, 18 at apex, below 10 from apex + 22
        for f in range(rel + flight, rel + flight + 30):
            bx[f], by[f], bz[f] = gx, gy, 2.0
    return bx, by, bz


@pytest.fixture(scope="module")
def frames():
    bx, by, bz = script_ball()
    rows = []
    for f in range(N_FRAMES):
        for pid, name, home, tid, team, (x, y) in PLAYERS:
            rows.append({"game": "synthetic", "period": 1, "frame": f, "game_clock_s": clock_of(f),
                         "ball_x": bx[f], "ball_y": by[f], "ball_z": bz[f], "team_id": tid, "team": team,
                         "home": home, "player_id": pid, "player": name, "x": x, "y": y})
    df = pd.DataFrame(rows)
    return df.astype({"period": "int8", "frame": "int32", "game_clock_s": "float32", "ball_x": "float32",
                      "ball_y": "float32", "ball_z": "float32", "team_id": "int32", "player_id": "int32",
                      "x": "float32", "y": "float32"})


@pytest.fixture(scope="module")
def events():
    rows = [
        (1, 720.0, None, "Jump ball", 0, "other"),
        (1, 712.0, "home", "S. Curry makes 2-pt jump shot from 15 ft", 2, "shot_made"),
        (1, 700.0, "away", "E. Bledsoe misses 2-pt jump shot from 14 ft", 0, "shot_missed"),
        (1, 699.0, "home", "Defensive rebound by K. Thompson", 0, "rebound"),
        (1, 692.0, "home", "Z. Nobody misses 3-pt jump shot from 25 ft", 0, "shot_missed"),
        (1, 682.0, "home", "S. Curry misses 3-pt jump shot from 27 ft", 0, "shot_missed"),
    ]
    df = pd.DataFrame(rows, columns=["period", "clock_s", "team", "desc", "points", "kind"])
    df.insert(0, "event", range(len(df)))
    df["frame"] = -1
    return df


@pytest.fixture(scope="module")
def located(frames, events):
    return sfm.locate_shots(frames, events, "synthetic")


def test_missing_name_gives_empty_last_name():
    """A nameless player (player_id 0, NaN name in real frames) yields an empty name instead of an AttributeError."""
    for missing in (float("nan"), None):
        assert sfm._norm(missing) == "" and sfm._last_name(missing) == ""


def test_one_row_per_shot_with_contract_columns(located):
    assert list(located.columns) == sfm.COLUMNS
    assert located["event"].tolist() == [1, 2, 4, 5]
    assert located["made"].tolist() == [True, False, False, False]
    assert str(located["shooter_id"].dtype) == "Int32" and str(located["identity_match"].dtype) == "boolean"
    assert set(located["confidence_flag"]) <= {"high", "low", "unmatched"}
    assert ((located["confidence"] >= 0) & (located["confidence"] <= 1)).all()


@pytest.mark.parametrize("event, shot", [(1, SHOTS[0]), (2, SHOTS[1])])
def test_scripted_shots_are_located(located, event, shot):
    shooter, _carry, rel, apex, _basket = shot
    row = located[located["event"] == event].iloc[0]
    assert abs(int(row["release_frame"]) - rel) <= 2
    assert abs(int(row["apex_frame"]) - apex) <= 2
    assert int(row["outcome_frame"]) > int(row["apex_frame"])
    assert int(row["shooter_id"]) == PLAYERS[shooter][0]
    assert int(row["shooter_at_release_id"]) == PLAYERS[shooter][0]
    assert bool(row["identity_match"]) is True
    assert row["confidence_flag"] == "high"
    assert row["confidence"] >= 0.6


def test_unknown_shooter_is_unmatched_but_located_by_ball(located):
    row = located[located["event"] == 4].iloc[0]
    assert pd.isna(row["shooter_id"])
    assert row["confidence_flag"] == "unmatched"
    assert abs(int(row["release_frame"]) - SHOTS[2][2]) <= 2
    assert int(row["shooter_at_release_id"]) == PLAYERS[SHOTS[2][0]][0]
    assert pd.isna(row["identity_match"])


def test_no_ball_near_anyone_gives_no_release(located):
    row = located[located["event"] == 5].iloc[0]
    assert int(row["release_frame"]) == -1 and int(row["apex_frame"]) == -1 and int(row["outcome_frame"]) == -1
    assert row["confidence_flag"] == "unmatched"
    assert float(row["confidence"]) == 0.0


def test_validate_counts(located):
    v = sfm.validate(located)
    assert v == {"n_shots": 4, "n_release_found": 3, "share_release_found": 0.75, "n_identity_known": 2,
                 "identity_agreement": 1.0, "share_high": 0.5}
    empty = sfm.validate(located.iloc[0:0])
    assert empty["n_shots"] == 0 and empty["share_release_found"] is None


def test_shooter_name_matching():
    roster = pd.DataFrame({"player_id": [1, 2, 3, 4, 5], "home": [True, True, False, False, False],
                           "player": ["Stephen Curry", "Klay Thompson", "Eric Bledsoe", "Brandon Knight",
                                      "Marcus Thompson"]})
    assert sfm.parse_shooter("K. Thompson makes 2-pt layup from 1 ft") == ("K", "Thompson")
    assert sfm.parse_shooter("L. Mbah a Moute misses 3-pt jump shot") == ("L", "Mbah a Moute")
    assert sfm.parse_shooter("Defensive rebound by K. Thompson") == (None, None)
    assert sfm.match_shooter("K", "Thompson", roster, "home") == 2
    assert sfm.match_shooter("M", "Thompson", roster, None) == 5     # initial breaks the tie
    assert sfm.match_shooter("E", "Bledsoe", roster, "home") == 3    # falls back to the whole roster
    assert sfm.match_shooter("Z", "Nobody", roster, "home") is None
    assert sfm.match_shooter(None, None, roster, "home") is None


def test_season_and_report_line(tmp_path):
    assert sfm.season_of_game("12.16.2015.PHX.at.GSW") == "2015-16"
    assert sfm.season_of_game("01.22.2016.IND.at.GSW") == "2015-16"
    rp = tmp_path / "r.md"
    sfm.append_report(rp, "g1", True, {"n_shots": 2, "n_release_found": 1, "share_release_found": 0.5,
                                        "n_identity_known": 0, "identity_agreement": None, "share_high": 0.0})
    text = rp.read_text()
    assert text.startswith("# Shot localisation") and "| g1 | held-out" in text and "n/a" in text
    assert chr(0x2014) not in text

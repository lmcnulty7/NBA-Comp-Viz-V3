"""tests/test_rebound_frames.py: analytics.rebound_frames on a scripted synthetic game.

One period at 25 Hz, two players per team, ten ball segments: a missed shot that rises above 10 ft, comes
down at the home basket and bounces to Bob Jones, who holds it; then a second miss at the other basket that
rolls away from everybody (a team rebound). No real tracking data is used here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import rebound_frames as rf  # noqa: E402

GAME = "01.05.2015.AAA.at.BBB"
FPS = 25
N_FRAMES = 1300
PLAYERS = [  # (player_id, name, team_id, home, x, y): static positions, far from the ball paths
    (1001, "Bob Jones", 100, True, 84.0, 28.0),
    (1002, "Alan Smith", 100, True, 70.0, 10.0),
    (2001, "Carl Brown", 200, False, 20.0, 25.0),
    (2002, "Dan Jones", 200, False, 30.0, 40.0),
]
HOME_RIM, AWAY_RIM = (88.75, 25.0), (5.25, 25.0)

# script of miss 1 (Bob Jones rebounds): frame, (x, y, z) of the ball, linear between keyframes
MISS1 = [(0, (84.0, 28.0, 5.0)), (300, (84.0, 28.0, 5.0)), (310, (65.0, 25.0, 6.0)), (500, (65.0, 25.0, 6.0)),
         (525, (77.0, 25.0, 14.0)), (550, (88.75, 25.0, 10.0)), (556, (87.5, 25.5, 5.0)),
         (565, (84.0, 28.0, 5.0)), (790, (84.0, 28.0, 5.0))]
RIM1, REB1 = 551, 561            # by the rules: first frame under 10 ft at the rim; first frame within 2 ft of Bob
# script of miss 2 (team rebound): the ball ends in the corner with nobody near it
MISS2 = [(800, (20.0, 25.0, 6.0)), (1000, (20.0, 25.0, 6.0)), (1025, (12.0, 25.0, 14.0)),
         (1050, (5.25, 25.0, 10.0)), (1080, (2.0, 45.0, 1.0)), (N_FRAMES - 1, (2.0, 45.0, 1.0))]
RIM2 = 1051
MISS1_CLOCK, MISS2_CLOCK = 720.0 - 500 / FPS, 720.0 - 1000 / FPS     # 700.0 and 680.0


def ball_path():
    keys = MISS1 + MISS2
    kf = np.array([k for k, _ in keys], float)
    kv = np.array([v for _, v in keys], float)
    f = np.arange(N_FRAMES, dtype=float)
    return np.stack([np.interp(f, kf, kv[:, i]) for i in range(3)], axis=1)


@pytest.fixture(scope="module")
def frames() -> pd.DataFrame:
    ball = ball_path()
    rows = []
    for f in range(N_FRAMES):
        for pid, name, tid, home, x, y in PLAYERS:
            rows.append({"game": GAME, "period": 1, "frame": f, "game_clock_s": 720.0 - f / FPS,
                         "ball_x": ball[f, 0], "ball_y": ball[f, 1], "ball_z": ball[f, 2],
                         "team_id": tid, "home": home, "player_id": pid, "player": name, "x": x, "y": y})
    df = pd.DataFrame(rows)
    return df.astype({"period": "int8", "frame": "int32", "game_clock_s": "float32", "ball_x": "float32",
                      "ball_y": "float32", "ball_z": "float32", "team_id": "int32", "player_id": "int32",
                      "x": "float32", "y": "float32"})


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    ev = [
        (1, 710.0, "home", "A. Smith makes 2-pt jump shot from 15 ft", 2, "shot_made"),
        (1, MISS1_CLOCK, "away", "C. Brown misses 3-pt jump shot from 26 ft", 0, "shot_missed"),
        (1, MISS1_CLOCK - 1.0, "home", "Defensive rebound by B. Jones", 0, "rebound"),
        (1, 690.0, "home", "B. Jones misses free throw 1 of 2", 0, "ft_missed"),
        (1, 690.0, "home", "B. Jones makes free throw 2 of 2", 1, "ft_made"),
        (1, MISS2_CLOCK, "home", "A. Smith misses 2-pt layup from 2 ft", 0, "shot_missed"),
        (1, MISS2_CLOCK - 1.0, "away", "Offensive rebound by Team", 0, "rebound"),
    ]
    df = pd.DataFrame(ev, columns=["period", "clock_s", "team", "desc", "points", "kind"])
    df.insert(0, "event", np.arange(len(df), dtype="int32"))
    df["frame"] = ((720.0 - df["clock_s"]) * FPS).round().astype("int32")
    df["game"] = GAME
    return df


@pytest.fixture(scope="module")
def table(frames, events) -> pd.DataFrame:
    return rf.locate_rebounds(frames, events, GAME)


def test_schema_and_row_count(table):
    assert list(table.columns) == rf.COLUMNS
    assert len(table) == 2                      # two shot_missed; the 1 of 2 free throw miss is not a trip end
    assert set(table["confidence_flag"]) <= set(rf.FLAGS)
    assert ((table["confidence"] >= 0) & (table["confidence"] <= 1)).all()
    assert table["period"].dtype == np.int8 and table["event"].dtype == np.int32
    rf.to_arrow(table)                          # the arrow schema accepts it


def test_scripted_miss_is_located(table):
    r = table[table["event"] == 1].iloc[0]
    assert abs(int(r["rim_frame"]) - RIM1) <= 2
    assert abs(int(r["rebound_frame"]) - REB1) <= 2
    assert int(r["rebounder_id"]) == 1001
    assert int(r["pbp_rebounder_id"]) == 1001 and r["pbp_rebounder_name"] == "B. Jones"
    assert r["identity_match"] is True or bool(r["identity_match"]) is True
    assert r["confidence_flag"] == "high" and r["confidence"] >= 0.5
    assert int(r["rebound_event"]) == 2


def test_team_rebound_is_unmatched(table):
    r = table[table["event"] == 5].iloc[0]
    assert abs(int(r["rim_frame"]) - RIM2) <= 2
    assert int(r["rebound_frame"]) == -1 and pd.isna(r["rebounder_id"])
    assert pd.isna(r["pbp_rebounder_id"]) and pd.isna(r["pbp_rebounder_name"])
    assert pd.isna(r["identity_match"])
    assert r["confidence_flag"] == "unmatched" and float(r["confidence"]) == 0.0


def test_validate_counts(table):
    v = rf.validate(table)
    assert v == {"n_misses": 2, "n_rebound_found": 1, "share_rebound_found": 0.5,
                 "n_identity_known": 1, "identity_agreement": 1.0, "share_high": 0.5}


def test_name_matching():
    roster = [(1, "Draymond Green"), (2, "Klay Thompson"), (3, "Jason Thompson"), (4, "Mirza Teletovic"),
              (5, "James Michael McAdoo")]
    assert rf.match_player("D. Green", roster) == 1
    assert rf.match_player("K. Thompson", roster) == 2
    assert rf.match_player("J. Thompson", roster) == 3
    assert rf.match_player("M. TeletoviÄ\u0087", roster) == 4     # latin-1 mojibake of Teletović
    assert rf.match_player("M. Teletović", roster) == 4
    assert rf.match_player("J. McAdoo", roster) == 5
    assert rf.match_player("Z. Nobody", roster) is None
    assert rf.pbp_rebounder("Offensive rebound by Team") is None
    assert rf.pbp_rebounder("Defensive rebound by D. Green") == "D. Green"


def test_last_free_throw_of_trip():
    ev = pd.DataFrame({
        "event": range(5), "period": [1] * 5, "clock_s": [600.0, 600.0, 580.0, 580.0, 570.0],
        "desc": ["X misses free throw 1 of 2", "X makes free throw 2 of 2", "X makes free throw 1 of 2",
                 "X misses free throw 2 of 2", "X misses technical free throw"],
        "kind": ["ft_missed", "ft_made", "ft_made", "ft_missed", "ft_missed"]})
    assert rf.miss_events(ev) == [3, 4]


def test_run_writes_parquet_and_report(frames, events, tmp_path):
    fp = tmp_path / (GAME + ".parquet"); pp = tmp_path / (GAME + "_pbp.parquet")
    frames.to_parquet(fp, index=False); events.to_parquet(pp, index=False)
    report = tmp_path / "localisation_rebounds.md"
    res = rf.run(GAME, fp, pp, tmp_path / "rebound_frames", report)
    out = tmp_path / "rebound_frames" / "2014-15" / (GAME + ".parquet")     # not held out: no heldout/
    assert out.exists() and res["out"] == str(out)
    t = pq.read_table(out)
    assert t.schema.equals(rf.SCHEMA) and t.num_rows == 2
    text = report.read_text()
    assert GAME in text and "| 2 | 1 | 0.500 | 1 | 1.000 | 0.500 |" in text
    assert chr(0x2014) not in text               # the report never carries an em dash

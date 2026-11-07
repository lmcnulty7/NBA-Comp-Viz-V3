"""analytics.rosters: name matching on a synthetic roster, the roster read of a synthetic frames parquet, build,
unmatched, write_report and the output location. No real data: the only real file is the committed synthetic
basketball-reference page of tests/fixtures/bref."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import rosters as R  # noqa: E402

GSW, CLE = 1610612744, 1610612739


def synthetic_roster() -> pd.DataFrame:
    """Ten players with the tricky forms: an accent, a hyphen, a Jr., an apostrophe, a two-word particle surname,
    a dotted-initials first name and two players who share a last name on different teams."""
    rows = [
        (1, "Stephen Curry", GSW, "GSW", True, "30"),
        (2, "Klay Thompson", GSW, "GSW", True, "11"),
        (3, "Mirza Teletović", GSW, "GSW", True, "35"),
        (4, "Rondae Hollis-Jefferson", GSW, "GSW", True, "24"),
        (5, "Nick Van Exel", GSW, "GSW", True, "31"),
        (6, "Tristan Thompson", CLE, "CLE", False, "13"),
        (7, "Tim Hardaway Jr.", CLE, "CLE", False, "10"),
        (8, "Shaquille O'Neal", CLE, "CLE", False, "33"),
        (9, "PJ Tucker", CLE, "CLE", False, "17"),
        (10, "LeBron James", CLE, "CLE", False, None),
    ]
    df = pd.DataFrame(rows, columns=["player_id", "player", "team_id", "team", "home", "jersey"])
    df.insert(0, "sportvu_game", "01.01.2016.CLE.at.GSW")
    df["player_id"] = df["player_id"].astype("int32")
    df["team_id"] = df["team_id"].astype("int32")
    return df


# ── normalisation ──────────────────────────────────────────────────────────────────────────────────
def test_normalize_strips_accents_periods_and_suffixes():
    assert R.normalize("Mirza Teletović") == "mirza teletovic"
    assert R.normalize("T.J. Warren") == "tj warren"
    assert R.normalize("Tim Hardaway Jr.") == "tim hardaway"
    assert R.normalize("  Shaquille  O'Neal ") == "shaquille o'neal"
    assert R.fix_mojibake("Mirza TeletoviÄ\u0087") == "Mirza Teletović"
    assert R.fix_mojibake("Stephen Curry") == "Stephen Curry"
    assert R.split_name("nick van exel") == ("nick", "van exel")
    assert R.split_name("rondae hollis-jefferson") == ("rondae", "hollis-jefferson")
    assert R.is_initial_form("S. Curry") and not R.is_initial_form("T.J. Warren")


def test_missing_name_normalises_to_empty():
    """Real frames carry a player_id 0 row with no name (NaN after pandas); it must not crash the matcher."""
    for missing in (float("nan"), None):
        assert R.fix_mojibake(missing) == "" and R.normalize(missing) == ""


# ── bref-style full names: every match_method ──────────────────────────────────────────────────────
def test_match_names_bref_every_method():
    ros = synthetic_roster()
    names = ["Stephen Curry", "mirza teletovic", "P.J. Tucker", "Tim Hardaway",            # exact
             "Steph Curry", "Shaq O'Neal",                                                 # lastname (unique)
             "Stephen Curryy", "Mirza Teletovich",                                         # fuzzy (>= 0.85)
             "Thompson", "Michael Jordan", ""]                                             # none
    out = R.match_names(ros, names, "bref")
    assert list(out.columns) == ["name", "player_id", "team_id", "match_method"]
    assert str(out["player_id"].dtype) == "Int32" and str(out["team_id"].dtype) == "Int32"
    got = dict(zip(out["name"], out["match_method"]))
    pid = dict(zip(out["name"], out["player_id"]))
    assert [got[n] for n in names[:4]] == ["exact"] * 4
    assert [pid[n] for n in names[:4]] == [1, 3, 9, 7]
    assert got["Steph Curry"] == "lastname" and pid["Steph Curry"] == 1
    assert got["Shaq O'Neal"] == "lastname" and pid["Shaq O'Neal"] == 8
    assert got["Stephen Curryy"] == "fuzzy" and pid["Stephen Curryy"] == 1
    assert got["Mirza Teletovich"] == "fuzzy" and pid["Mirza Teletovich"] == 3
    assert got["Thompson"] == "none" and pd.isna(pid["Thompson"])    # two Thompsons, no hint
    assert got["Michael Jordan"] == "none" and got[""] == "none"
    assert set(out["match_method"]) == set(R.METHODS)


def test_match_names_team_hint_resolves_shared_last_name():
    ros = synthetic_roster()
    out = R.match_names(ros, ["Thompson", "Thompson", "Thompson", "Thompson"], "bref", teams=["GSW", "away", CLE, None])
    assert list(out["match_method"]) == ["lastname", "lastname", "lastname", "none"]
    assert list(out["player_id"].astype(object)[:3]) == [2, 6, 6]
    assert list(out["team_id"].astype(object)[:3]) == [GSW, CLE, CLE]
    # a basketball-reference code maps through BR_ABBR; an unknown hint allows the whole roster
    out = R.match_names(ros, ["Stephen Curry", "Thompson"], "bref", teams=["XXX", "ZZZ"])
    assert list(out["match_method"]) == ["exact", "none"]
    with pytest.raises(ValueError):
        R.match_names(ros, ["a"], "bref", teams=["x", "y"])
    with pytest.raises(ValueError):
        R.match_names(ros, ["a"], "shots")


# ── play-by-play "F. Lastname" forms ───────────────────────────────────────────────────────────────
def test_match_names_pbp_initial_forms_every_method():
    ros = synthetic_roster()
    names = ["S. Curry", "K. Thompson", "T. Thompson", "N. Van Exel", "S. O'Neal", "T. Hardaway",
             "R. Hollis-Jefferson", "P. Tucker", "M. Teletović",                                 # exact
             "J. Teletovic",                                                                     # lastname (wrong initial)
             "M. Teletovi",                                                                      # fuzzy (clipped accent)
             "X. Thompson", "Z. Nobody"]                                                         # none
    out = R.match_names(ros, names, "pbp")
    got = dict(zip(out["name"], out["match_method"]))
    pid = dict(zip(out["name"], out["player_id"]))
    assert all(got[n] == "exact" for n in names[:9]), got
    assert pid["K. Thompson"] == 2 and pid["T. Thompson"] == 6 and pid["P. Tucker"] == 9 and pid["M. Teletović"] == 3
    assert got["J. Teletovic"] == "lastname" and pid["J. Teletovic"] == 3
    assert got["M. Teletovi"] == "fuzzy" and pid["M. Teletovi"] == 3
    assert got["X. Thompson"] == "none" and got["Z. Nobody"] == "none"
    assert set(out["match_method"]) == set(R.METHODS)


def test_pbp_names_regex_forms():
    events = [
        {"desc": "S. Curry makes 3-pt jump shot from 26 ft (assist by D. Green)"},
        {"desc": "Turnover by N. Van Exel (bad pass; steal by K. O'Neal )"},
        {"desc": "Defensive rebound by R. Hollis-Jefferson"},
        {"desc": "Offensive rebound by Team"},
        {"desc": "T. Hardaway Jr. misses 2-pt layup from 1 ft (block by S. Curry)"},
        {"desc": "M. TeletoviÄ\u0087 enters the game for T. Chandler"},
        {"desc": "Phoenix full timeout"},
        {"desc": None},
    ]
    assert R.pbp_names({"meta": {}, "events": events}) == [
        "S. Curry", "D. Green", "N. Van Exel", "K. O'Neal", "R. Hollis-Jefferson", "T. Hardaway Jr.",
        "M. Teletović", "T. Chandler"]
    assert R.pbp_names(events[2:3]) == ["R. Hollis-Jefferson"]


# ── the roster from a synthetic frames parquet (DATA_CONTRACT section 4) ───────────────────────────
def frames_table(n_frames: int = 4) -> pa.Table:
    ros = synthetic_roster()
    rows = []
    for f in range(n_frames):
        for _, p in ros.iterrows():
            rows.append({"game": "01.01.2016.CLE.at.GSW", "gameid": "0021500001", "date": "2016-01-01", "period": 1,
                         "frame": f, "ts_ms": 1000 + 40 * f, "game_clock_s": 720.0 - f * 0.04, "shot_clock_s": 24.0,
                         "n_players": 10, "ball_x": 47.0, "ball_y": 25.0, "ball_z": 5.0,
                         "team_id": int(p["team_id"]), "team": p["team"], "home": bool(p["home"]),
                         "player_id": int(p["player_id"]), "player": p["player"],
                         "jersey": p["jersey"] if isinstance(p["jersey"], str) else None,
                         "x": float(f), "y": float(p["player_id"])})
    schema = pa.schema([("game", pa.string()), ("gameid", pa.string()), ("date", pa.string()), ("period", pa.int8()),
                        ("frame", pa.int32()), ("ts_ms", pa.int64()), ("game_clock_s", pa.float32()),
                        ("shot_clock_s", pa.float32()), ("n_players", pa.int8()), ("ball_x", pa.float32()),
                        ("ball_y", pa.float32()), ("ball_z", pa.float32()), ("team_id", pa.int32()), ("team", pa.string()),
                        ("home", pa.bool_()), ("player_id", pa.int32()), ("player", pa.string()), ("jersey", pa.string()),
                        ("x", pa.float32()), ("y", pa.float32())])
    return pa.Table.from_pylist(rows, schema=schema)


def test_roster_from_frames(tmp_path):
    path = tmp_path / "01.01.2016.CLE.at.GSW.parquet"
    pq.write_table(frames_table(), path, row_group_size=25)       # several row groups: players of later groups count
    assert pq.ParquetFile(path).metadata.num_row_groups > 1
    ros = R.roster_from_frames(path)
    assert list(ros.columns) == R.ROSTER_COLS
    assert len(ros) == 10 and ros["player_id"].is_unique
    assert str(ros["player_id"].dtype) == "int32" and str(ros["team_id"].dtype) == "int32" and ros["home"].dtype == bool
    assert set(ros["sportvu_game"]) == {"01.01.2016.CLE.at.GSW"}
    assert list(ros["home"][:5]) == [True] * 5 and list(ros["home"][5:]) == [False] * 5    # home first
    by = ros.set_index("player")
    assert by.loc["Stephen Curry", "jersey"] == "30" and by.loc["LeBron James", "jersey"] is None
    assert by.loc["Mirza Teletović", "team"] == "GSW" and by.loc["PJ Tucker", "team_id"] == CLE


# ── build, unmatched, report, output path ──────────────────────────────────────────────────────────
def test_build_unmatched_and_report(tmp_path):
    ros = synthetic_roster()
    pbp = {"meta": {"home": "GSW", "away": "CLE"}, "events": [
        {"desc": "S. Curry makes 3-pt jump shot from 26 ft (assist by K. Thompson)"},
        {"desc": "T. Thompson misses 2-pt layup from 1 ft (block by N. Van Exel)"},
        {"desc": "Personal foul by T. Hardaway Jr. (drawn by S. O'Neal)"},
        {"desc": "J. Teletovic enters the game for P. Tucker"},
        {"desc": "Turnover by Z. Nobody (bad pass)"},
    ]}
    # the committed synthetic shot chart page names real players in "F. Lastname" form; against this roster only
    # S. Curry, K. Thompson and P. Tucker exist, which also exercises initial forms coming from the bref side
    html = (ROOT / "tests" / "fixtures" / "bref" / "201512160GSW.synthetic.html").read_text(encoding="utf-8")
    out = R.build("01.01.2016.CLE.at.GSW", None, bref_html=html, pbp_json=json.dumps(pbp), roster=ros)
    assert list(out.columns) == R.OUT_COLS and len(out) == 10
    by = out.set_index("player")
    assert by.loc["Stephen Curry", ["bref_slug", "bref_name", "pbp_name", "match_method"]].tolist() == \
        ["currys01", "S. Curry", "S. Curry", "exact"]
    assert by.loc["Klay Thompson", ["bref_slug", "bref_name", "pbp_name", "match_method"]].tolist() == \
        ["thompk01", "K. Thompson", "K. Thompson", "exact"]
    assert by.loc["Tristan Thompson", "pbp_name"] == "T. Thompson" and by.loc["Tristan Thompson", "bref_name"] is None
    assert by.loc["PJ Tucker", ["bref_name", "pbp_name", "match_method"]].tolist() == ["P. Tucker", "P. Tucker", "exact"]
    assert by.loc["Mirza Teletović", "match_method"] == "lastname" and by.loc["Mirza Teletović", "pbp_name"] == "J. Teletovic"
    assert by.loc["Tim Hardaway Jr.", "pbp_name"] == "T. Hardaway Jr."
    assert by.loc["LeBron James", "match_method"] == "none" and by.loc["LeBron James", "pbp_name"] is None
    assert by.loc["Rondae Hollis-Jefferson", "match_method"] == "none"
    assert set(out["match_method"]) <= set(R.METHODS)

    bref_list = list(R.bref_names(html, "01.01.2016.CLE.at.GSW")["name"])
    pbp_list = R.pbp_names(pbp)
    assert len(bref_list) == 22 and "M. Teletović" in bref_list       # mojibake repaired at the source
    un = R.unmatched(out, bref_list, pbp_list)
    assert "Z. Nobody" in un and "S. Curry" not in un and "K. Thompson" not in un and "P. Tucker" not in un
    assert len(un) == 19 + 1 and len(set(un)) == len(un)

    s = R.summarize("01.01.2016.CLE.at.GSW", out, bref_list, pbp_list)
    assert (s["bref_matched"], s["bref_total"], s["pbp_matched"], s["pbp_total"]) == (3, 22, 8, 9)
    assert s["roster_size"] == 10 and sum(s["methods"].values()) == 10
    report = tmp_path / "reports" / "rosters_unmatched.md"
    R.write_report({s["game"]: s}, report)
    text = report.read_text(encoding="utf-8")
    assert "| 01.01.2016.CLE.at.GSW | 10 | 3 / 22 | 8 / 9 |" in text
    assert "Z. Nobody" in text and "LeBron James" in text and chr(0x2014) not in text

    # the table round-trips through parquet with the contract types
    path = R.write_table(out, tmp_path / "t.parquet")
    back = pq.read_table(path)
    assert back.schema.equals(R.SCHEMA)
    assert back.to_pandas().set_index("player").loc["Stephen Curry", "bref_slug"] == "currys01"


def test_out_path_for_heldout_and_train(tmp_path):
    sp = {"train": {"arenas": ["GSW"], "games": {"g1": {"sportvu": "11.14.2015.BKN.at.GSW"}}},
          "heldout_game": {"games": {"g2": {"sportvu": "12.16.2015.PHX.at.GSW"}}},
          "heldout_arena": {"arenas": ["CLE"], "games": {}},
          "heldout_era": {"games": {}}}
    assert R.out_path_for("11.14.2015.BKN.at.GSW", tmp_path, sp) == tmp_path / "2015-16" / "11.14.2015.BKN.at.GSW.parquet"
    assert R.out_path_for("12.16.2015.PHX.at.GSW", tmp_path, sp) == tmp_path / "heldout" / "2015-16" / "12.16.2015.PHX.at.GSW.parquet"
    assert R.out_path_for("06.19.2016.GSW.at.CLE", tmp_path, sp) == tmp_path / "heldout" / "2015-16" / "06.19.2016.GSW.at.CLE.parquet"
    assert R.out_path_for("10.30.2016.NOP.at.GSW", tmp_path, sp) == tmp_path / "2016-17" / "10.30.2016.NOP.at.GSW.parquet"


def test_cli_end_to_end(tmp_path):
    frames = tmp_path / "01.01.2016.CLE.at.GSW.parquet"
    pq.write_table(frames_table(2), frames)
    pbp = tmp_path / "pbp.json"
    pbp.write_text(json.dumps({"meta": {}, "events": [{"desc": "S. Curry makes 2-pt layup from 1 ft"}]}))
    report = tmp_path / "reports" / "rosters_unmatched.md"
    assert R.main(["--game", "01.01.2016.CLE.at.GSW", "--frames", str(frames), "--pbp-json", str(pbp),
                   "--out-dir", str(tmp_path / "rosters"), "--report", str(report)]) == 0
    table = pq.read_table(tmp_path / "rosters" / "2015-16" / "01.01.2016.CLE.at.GSW.parquet").to_pandas()
    assert len(table) == 10 and (table["match_method"] == "exact").sum() == 1
    assert "| 01.01.2016.CLE.at.GSW | 10 | 0 / 0 | 1 / 1 |" in report.read_text(encoding="utf-8")
    # a second run for another game adds a row instead of replacing the report
    assert R.main(["--game", "01.02.2016.CLE.at.GSW", "--frames", str(frames),
                   "--out-dir", str(tmp_path / "rosters"), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "| 01.01.2016.CLE.at.GSW |" in text and "| 01.02.2016.CLE.at.GSW | 10 | 0 / 0 | 0 / 0 |" in text

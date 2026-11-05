"""sportvu.colab_contract: the SportVU -> docs/DATA_CONTRACT.md converter, checked on two games built in the
source format (one training game, one held-out game of sportvu/splits.json). The Drive path is not tested;
the run loop writes to a temp directory with a local archive directory as the listing."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
py7zr = pytest.importorskip("py7zr")

from sportvu import colab_contract as cc, splits  # noqa: E402

TRAIN, HELD = "11.14.2015.BKN.at.GSW", "12.16.2015.PHX.at.GSW"
GSW, PHX, BKN = 1610612744, 1610612756, 1610612751
GAP_AT = 40   # a 3 s hole in the clock after this frame of quarter 1


def roster(team_id, abbr, name, first_id):
    return {"teamid": team_id, "abbreviation": abbr, "name": name,
            "players": [{"lastname": "P%d" % i, "firstname": abbr, "playerid": first_id + i, "jersey": str(i), "position": "G"}
                        for i in range(5)]}


def moment(q, i, clock, home_id, away_id, ball=True):
    ents = ([[-1, -1, 47.0 + i * 0.1, 25.0, 5.0 + (i % 3)]] if ball else []) + \
           [[home_id, 100 + k, 10.0 + k + i * 0.01, 20.0 + k, 0.0] for k in range(5)] + \
           [[away_id, 200 + k, 60.0 + k + i * 0.01, 30.0 + k, 0.0] for k in range(5)]
    return [q, 1450000000000 + i * 40, round(clock, 2), 24.0 if i % 7 else None, None, ents]


def make_game(game, gameid, date, away, home, away_id, home_id, n=120):
    """Two quarters of n frames at 25 Hz; events overlap (the second repeats the last 5 moments of the
    first), one clock gap > 2 s in quarter 1, three frames without a ball."""
    clock = 720.0
    q1 = []
    for i in range(n):
        q1.append(moment(1, i, clock, home_id, away_id, ball=i not in (3, 4, 5)))
        clock -= 3.0 if i == GAP_AT else 1 / cc.HZ
    q2 = [moment(2, n + i, 720.0 - i / cc.HZ, home_id, away_id) for i in range(n)]
    sides = {"visitor": roster(away_id, away, away + " Team", 200), "home": roster(home_id, home, home + " Team", 100)}
    events = [{"eventId": "1", **sides, "moments": q1[:70]}, {"eventId": "2", **sides, "moments": q1[65:]},
              {"eventId": "3", **sides, "moments": q2}]
    return {"gameid": gameid, "gamedate": date, "events": events}


def write_archive(d: Path, game: str, doc: dict) -> Path:
    p = d / (game + ".7z")
    with py7zr.SevenZipFile(p, "w") as z:
        z.writestr(json.dumps(doc).encode(), doc["gameid"] + ".json")
    return p


@pytest.fixture(scope="module")
def archives(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("sportvu_archives")
    write_archive(d, TRAIN, make_game(TRAIN, "0021500144", "2015-11-14", "BKN", "GSW", BKN, GSW))
    write_archive(d, HELD, make_game(HELD, "0021500381", "2015-12-16", "PHX", "GSW", PHX, GSW))
    return d


def fake_pbp(code: str):
    """basketball-reference stand-in: events for the training game, an HTTP failure for the held-out one."""
    if code != "201511140GSW":
        raise RuntimeError("HTTP 403 for %s" % code)
    return [{"period": 1, "clock_s": 720.0, "team": None, "desc": "Jump ball", "points": 0, "score": "0-0", "kind": "other"},
            {"period": 1, "clock_s": 719.95, "team": "home", "desc": "makes 2-pt", "points": 2, "score": "0-2", "kind": "shot_made"},
            {"period": 2, "clock_s": 719.0, "team": "away", "desc": "misses 3-pt", "points": 0, "score": "0-2", "kind": "shot_missed"},
            {"period": 3, "clock_s": 700.0, "team": "home", "desc": "turnover", "points": 0, "score": "0-2", "kind": "turnover"}]


def test_frames_follow_the_contract(archives, tmp_path):
    raw = cc.extract(archives / (TRAIN + ".7z"), tmp_path)
    parsed = cc.parse(raw)
    parsed["game"] = TRAIN
    t = cc.frames_table(parsed)
    assert t.schema.equals(cc.frames_schema())
    assert len(parsed["moments"]) == 240            # 5 repeated moments deduped as sportvu.fetch does
    assert t.num_rows == 240 * 10                   # one row per tracked player per frame
    fr = t.column("frame").to_numpy()
    assert fr.min() == 0 and fr.max() == 239 and np.all(np.diff(fr) >= 0)
    ball = t.to_pandas().drop_duplicates("frame").set_index("frame")
    assert ball.loc[0, ["ball_x", "ball_y", "ball_z"]].tolist() == pytest.approx([47.0, 25.0, 5.0])
    assert ball.loc[[3, 4, 5], "ball_x"].isna().all() and ball["ball_x"].notna().sum() == 237
    assert ball["shot_clock_s"].isna().sum() > 0
    assert set(t.column("team").to_pylist()) == {"GSW", "BKN"}
    df = t.slice(0, 10).to_pandas()
    assert df.loc[df.team == "GSW", "home"].all() and not df.loc[df.team == "BKN", "home"].any()
    assert df.loc[df.player_id == 100, "player"].item() == "GSW P0" and df.loc[df.player_id == 100, "jersey"].item() == "0"
    assert (t.column("n_players").to_numpy() == 10).all()
    assert sorted(set(t.column("period").to_pylist())) == [1, 2]


def test_gaps_over_two_seconds(archives, tmp_path):
    parsed = cc.parse(cc.extract(archives / (HELD + ".7z"), tmp_path))
    g = cc.gap_summary(parsed)
    assert g["count"] == 1 and g["seconds"] == pytest.approx(3.0, abs=0.1)
    assert list(g["by_quarter"]) == ["1", "2"] and g["by_quarter"]["2"] == []


def test_join_pbp_nearest_frame_within_tolerance():
    period = np.array([1, 1, 1, 2, 2])
    clock = np.array([720.0, 719.96, 700.0, 720.0, 719.96])
    ev = [{"period": 1, "clock_s": 719.9}, {"period": 1, "clock_s": 710.0}, {"period": 1, "clock_s": 701.5},
          {"period": 2, "clock_s": 720.0}, {"period": 4, "clock_s": 10.0}]
    t = cc.join_pbp("g", ev, period, clock)
    assert t.schema.equals(cc.pbp_schema())
    assert t.column("frame").to_pylist() == [1, -1, 2, 3, -1]
    assert t.column("frame_clock_diff_s").to_pylist()[2] == pytest.approx(1.5, abs=1e-3)
    assert t.column("event").to_pylist() == [0, 1, 2, 3, 4]


def test_order_and_split_tags():
    sp = splits.load()
    mk = lambda game, date, away, home: {"game": game, "date": date, "away": away, "home": home}
    games = [mk("01.01.2016.CHA.at.TOR", "2016-01-01", "CHA", "TOR"), mk(HELD, "2015-12-16", "PHX", "GSW"),
             mk("10.30.2015.GSW.at.HOU", "2015-10-30", "GSW", "HOU"), mk(TRAIN, "2015-11-14", "BKN", "GSW"),
             mk("01.22.2016.IND.at.GSW", "2016-01-22", "IND", "GSW"), mk("12.23.2015.NYK.at.CLE", "2015-12-23", "NYK", "CLE"),
             mk("11.04.2015.NYK.at.CLE", "2015-11-04", "NYK", "CLE")]
    order = [g["game"] for g in cc.order_games(games, sp)]
    assert order[:2] == [TRAIN, "01.22.2016.IND.at.GSW"]                      # training games, splits order
    assert order[2:4] == ["10.30.2015.GSW.at.HOU", HELD]                      # other GSW games by date
    assert order[4:] == ["11.04.2015.NYK.at.CLE", "12.23.2015.NYK.at.CLE", "01.01.2016.CHA.at.TOR"]
    tags = {g["game"]: cc.split_of(g, sp) for g in games}
    assert tags[TRAIN] == "train" and tags[HELD] == "heldout_game"
    assert tags["12.23.2015.NYK.at.CLE"] == "heldout_arena" and tags["11.04.2015.NYK.at.CLE"] == "heldout_arena"
    assert tags["01.01.2016.CHA.at.TOR"] is None and tags["10.30.2015.GSW.at.HOU"] is None


def test_br_code_uses_basketball_reference_abbreviations():
    assert cc.br_code({"date": "2015-11-14", "home": "GSW"}) == "201511140GSW"
    assert cc.br_code({"date": "2015-12-16", "home": "PHX"}) == "201512160PHO"
    assert cc.br_code({"date": "2016-01-01", "home": "BKN"}) == "201601010BRK"
    assert cc.br_code({"date": "2016-01-01", "home": "CHA"}) == "201601010CHO"


def test_run_two_games_is_resumable_and_tags_heldout(archives, tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"
    games = cc.list_local(archives)
    assert [g["game"] for g in games] == [TRAIN, HELD]
    man = cc.run(games, out, raw, pbp_fetch=fake_pbp)
    train, held = man["games"][TRAIN], man["games"][HELD]
    assert train["status"] == held["status"] == "ok"
    assert train["split"] == "train" and not train["heldout"] and train["parquet"] == TRAIN + ".parquet"
    assert held["split"] == "heldout_game" and held["heldout"] and held["parquet"] == "heldout/" + HELD + ".parquet"
    assert man["heldout"] == [HELD] and man["counts"]["ok"] == 2 and man["counts"]["heldout"] == 1
    assert train["moments"] == held["moments"] == 240 and train["gaps_over_2s"]["count"] == 1
    # play-by-play: joined for the training game, logged (not fatal) for the held-out one
    assert train["pbp_joined"] and train["pbp"]["events"] == 4 and train["pbp"]["matched"] == 3
    assert not held["pbp_joined"] and "403" in held["pbp"]["error"] and held["pbp"]["code"] == "201512160GSW"
    pbp = pq.read_table(out / (TRAIN + "_pbp.parquet"))
    assert pbp.column("frame").to_pylist() == [0, 1, 145, -1]   # 719.0 s of Q2 is frame 120 + 25
    # files on disk: frames read back, raw files gone, README and manifest written
    assert pq.read_metadata(out / "heldout" / (HELD + ".parquet")).num_rows == 2400
    assert not any(p.name.startswith(("11.", "12.", "_extract_")) for p in raw.iterdir())
    readme = (out / "README.md").read_text()
    assert "2015-10-27" in readme and "2016-01-23" in readme and "redistribut" in readme and "linouk23" in readme
    assert json.loads((out / "manifest.json").read_text())["heldout"] == [HELD]
    # the committed guard refuses the held-out parquet by its id, and passes the training one
    with pytest.raises(SystemExit):
        splits.refuse_heldout([str(out / held["parquet"])], "t")
    assert splits.refuse_heldout([str(out / train["parquet"])], "t") == 1
    # a second run converts nothing (resumable) and keeps the manifest entries
    written = {g: e["written"] for g, e in man["games"].items()}
    mtime = os.path.getmtime(out / (TRAIN + ".parquet"))
    man2 = cc.run(games, out, raw, pbp_fetch=lambda code: pytest.fail("pbp fetched on a resumed game"))
    assert {g: e["written"] for g, e in man2["games"].items()} == written
    assert os.path.getmtime(out / (TRAIN + ".parquet")) == mtime
    # --retry-pbp joins the game that failed before, from the frames on disk, without the archive
    ok_pbp = lambda code: [{"period": 2, "clock_s": 719.0, "team": "home", "desc": "x", "points": 0, "score": None, "kind": "other"}]
    man3 = cc.run(games, out, raw, pbp_fetch=ok_pbp, retry_pbp=True)
    assert man3["games"][HELD]["pbp_joined"] and man3["games"][HELD]["pbp"]["matched"] == 1
    assert pq.read_table(out / "heldout" / (HELD + "_pbp.parquet")).column("frame").to_pylist() == [145]
    assert cc.verdict(man3, games).startswith("RUN OK")


def test_broken_archive_is_recorded_not_fatal(tmp_path):
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    src.mkdir()
    (src / "01.14.2016.CHI.at.PHI.7z").write_bytes(b"not an archive at all, 32 bytes")
    games = cc.list_local(src)
    man = cc.run(games, out, raw, do_pbp=False)
    e = man["games"]["01.14.2016.CHI.at.PHI"]
    assert e["status"] == "source_broken" and "extract/parse" in e["error"] and e["split"] is None
    assert cc.verdict(man, games).startswith("RUN OK")
    man["games"]["01.14.2016.CHI.at.PHI"]["status"] = "failed"
    assert cc.verdict(man, games).startswith("RUN FAILED")


@pytest.mark.skipif(not os.environ.get("SPORTVU_ARCHIVES"), reason="set SPORTVU_ARCHIVES to a dir with real .7z game logs")
def test_real_archives(tmp_path):
    games = cc.list_local(os.environ["SPORTVU_ARCHIVES"])[:2]
    man = cc.run(games, tmp_path / "out", tmp_path / "raw", do_pbp=False)
    for g in games:
        e = man["games"][g["game"]]
        assert e["status"] in ("ok", "source_broken")
        if e["status"] == "ok":
            assert e["frames_rows"] >= e["moments"] * 9

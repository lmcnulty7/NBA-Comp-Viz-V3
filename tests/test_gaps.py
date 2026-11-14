"""analytics.gaps on a synthetic three-game manifest: one clean game, one with a 12 s gap in Q2, one with
two 3 s gaps. Clocks count down, so a gap span is [larger clock, smaller clock]."""
import json

import pandas as pd
import pytest

from analytics import gaps as G

CLEAN, BIG, SMALL = "11.01.2015.AAA.at.BBB", "11.02.2015.CCC.at.DDD", "11.03.2015.EEE.at.FFF"


def entry(game, split, by_quarter, status="ok"):
    spans = [s for v in by_quarter.values() for s in v]
    return {"game": game, "date": "2015-" + game[:5].replace(".", "-"), "away": game[11:14], "home": game[18:21],
            "split": split, "heldout": split != "train", "status": status, "error": None, "moments": 1000,
            "gaps_over_2s": {"count": len(spans), "seconds": round(sum(a - b for a, b in spans), 1),
                             "by_quarter": by_quarter}}


@pytest.fixture
def manifest():
    games = {
        CLEAN: entry(CLEAN, "train", {"1": [], "2": [], "3": [], "4": []}),
        BIG: entry(BIG, "heldout_game", {"1": [], "2": [[395.0, 383.0]], "3": [], "4": []}),
        SMALL: entry(SMALL, "train", {"1": [[700.0, 697.0]], "2": [], "3": [[100.5, 97.5]], "4": []}),
    }
    return {"contract_version": 1, "updated": "2026-11-05 00:00:00", "heldout": [BIG],
            "counts": {"games": 3, "ok": 3}, "games": games}


def test_game_gaps_rows(manifest):
    df = G.game_gaps(manifest)
    assert list(df.columns) == ["sportvu_game", "period", "gap_start_clock_s", "gap_end_clock_s", "gap_s"]
    assert len(df) == 3 and CLEAN not in set(df["sportvu_game"])
    assert str(df["period"].dtype) == "int8"
    big = df[df["sportvu_game"] == BIG].iloc[0]
    assert (big["period"], big["gap_start_clock_s"], big["gap_end_clock_s"], big["gap_s"]) == (2, 395.0, 383.0, 12.0)
    small = df[df["sportvu_game"] == SMALL]
    assert list(small["period"]) == [1, 3] and list(small["gap_s"]) == [3.0, 3.0]
    assert (df["gap_start_clock_s"] >= df["gap_end_clock_s"]).all()


def test_game_gaps_skips_games_without_gap_data(manifest):
    manifest["games"]["x"] = {"game": "x", "split": None, "status": "source_broken", "gaps_over_2s": None}
    manifest["games"]["y"] = {"game": "y", "split": None, "status": "ok"}
    assert len(G.game_gaps(manifest)) == 3
    empty = G.game_gaps({"games": {}})
    assert len(empty) == 0 and list(empty.columns) == G.GAP_COLUMNS


def test_games_over_at_10(manifest):
    over = G.games_over(manifest, 10.0)
    assert list(over.columns) == ["sportvu_game", "split", "n_gaps_over", "longest_gap_s", "seconds_over"]
    assert len(over) == 1
    r = over.iloc[0]
    assert (r["sportvu_game"], r["split"], r["n_gaps_over"], r["longest_gap_s"], r["seconds_over"]) == \
        (BIG, "heldout_game", 1, 12.0, 12.0)
    assert G.games_over(manifest, 20.0).empty


def test_games_over_at_2(manifest):
    over = G.games_over(manifest, 2.0).set_index("sportvu_game")
    assert set(over.index) == {BIG, SMALL}
    assert over.loc[SMALL, "n_gaps_over"] == 2
    assert over.loc[SMALL, "longest_gap_s"] == 3.0 and over.loc[SMALL, "seconds_over"] == 6.0
    assert over.loc[BIG, "n_gaps_over"] == 1
    assert list(over.index) == [BIG, SMALL]        # seconds_over descending


def test_flag_possessions(manifest):
    gaps = G.game_gaps(manifest)
    poss = pd.DataFrame({
        "sportvu_game": [BIG, BIG, BIG, BIG, CLEAN, SMALL],
        "possession": [1, 2, 3, 4, 1, 1],
        "period": [2, 2, 3, 2, 2, 1],
        "start_clock_s": [400.0, 300.0, 400.0, 420.0, 400.0, 702.0],
        "end_clock_s": [380.0, 280.0, 380.0, 395.0, 380.0, 699.0],
    }, index=[10, 11, 12, 13, 14, 15])
    flags = G.flag_possessions(poss, gaps)
    assert flags.dtype == bool and list(flags.index) == list(poss.index) and flags.name == "gap_overlap"
    # straddles the 12 s gap; clear of it; same clocks but another period; ends exactly where the gap starts
    # (touching, not overlapping); clean game; a 3 s gap partly inside the possession
    assert list(flags) == [True, False, False, False, False, True]


def test_flag_possessions_without_gaps(manifest):
    poss = pd.DataFrame({"sportvu_game": [BIG], "possession": [1], "period": [2],
                         "start_clock_s": [400.0], "end_clock_s": [380.0]})
    assert list(G.flag_possessions(poss, G.game_gaps({"games": {}}))) == [False]
    assert len(G.flag_possessions(poss.iloc[:0], G.game_gaps(manifest))) == 0
    with pytest.raises(KeyError):
        G.flag_possessions(poss.drop(columns=["period"]), G.game_gaps(manifest))


def test_summary(manifest):
    s = G.summary(manifest, 10.0)
    assert s["games"] == 3 and s["games_ok"] == 3
    assert s["games_with_gap_over_2s"] == 2 and s["games_with_gap_over_threshold"] == 1
    assert s["total_gap_seconds"] == 18.0 and s["gaps_over_2s"] == 3 and s["gaps_over_threshold"] == 1
    assert s["by_split"]["train"] == {"games": 2, "games_ok": 2, "games_with_gap_over_2s": 1,
                                      "games_with_gap_over_threshold": 0, "total_gap_seconds": 6.0}
    assert s["by_split"]["heldout_game"]["games_with_gap_over_threshold"] == 1
    assert G.summary(manifest, 2.0)["games_with_gap_over_threshold"] == 2


def test_write_report_and_cli(manifest, tmp_path):
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    out = tmp_path / "reports" / "sportvu_gaps.md"
    s = G.write_report(mp, out, 10.0, note="Synthetic manifest.")
    text = out.read_text()
    assert s["games"] == 3 and "3 games" in text and str(mp) in text
    assert "| %s | heldout_game | 1 | 12 | 12 |" % BIG in text
    assert SMALL not in text.split("## Games with a gap over 10 s")[1]
    assert "| games with any gap over 10 s | 1 |" in text and "Synthetic manifest." in text
    assert chr(0x2014) not in text               # no em dashes in any file this lane writes
    out2 = tmp_path / "t2.md"
    assert G.main(["--manifest", str(tmp_path), "--out", str(out2), "--threshold", "2"]) == 0
    assert "## Games with a gap over 2 s" in out2.read_text() and SMALL in out2.read_text()

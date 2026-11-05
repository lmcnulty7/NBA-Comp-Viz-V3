"""collect/bref_shots.py on saved basketball-reference shot chart pages (tests/fixtures/bref/*.html). No network.

Two kinds of page: <code>.synthetic.html (make_synthetic.py: the real attempts of data/pbp/<code>.json at generated
positions, so the recovered set must equal the play-by-play exactly) and any real <code>.html saved with
--save-pages, which gets the generic checks (pixel frame against the distances in the tips, count against the local
play-by-play when there is one)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")

import config  # noqa: E402
from collect import common, bref_shots as bs  # noqa: E402
from sportvu import colab_contract as cc, splits  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "bref"
PAGES = sorted(FIX.glob("*.html"))
SYNTHETIC = [p for p in PAGES if p.name.endswith(".synthetic.html")]
REAL = [p for p in PAGES if not p.name.endswith(".synthetic.html")]
GAME_OF = {"201512160GSW": "12.16.2015.PHX.at.GSW", "201602270OKC": "02.27.2016.GSW.at.OKC"}


def code_of(page: Path) -> str:
    return page.name.split(".")[0]


def pbp_events(code: str) -> list[dict] | None:
    p = config.PROJECT_ROOT / "data" / "pbp" / ("%s.json" % code)
    return json.loads(p.read_text())["events"] if p.exists() else None


def pbp_shots(code: str) -> list[dict] | None:
    ev = pbp_events(code)
    return [e for e in ev if e["kind"] in ("shot_made", "shot_missed")] if ev is not None else None


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)


# ── the court frame ────────────────────────────────────────────────────────────────────────────────
def test_pixel_frame_lands_on_the_contract_court():
    hx, hy = bs.HOOP_PX
    assert (hx, hy) == (239.5, 35.5) and bs.HOOP_IMG_PX == (250.0, 52.5)                     # the basket on the image, in marker (anchor) coordinates
    assert bs.to_court(hx, hy) == (5.25, 25.0) == bs.RIM_LEFT                                 # the hoop is the left rim
    assert bs.to_court(hx, hy + 23.75 * bs.PX_PER_FT) == (29.0, 25.0)                          # top-of-key three: court33 vertex 13 (29.01, 25)
    ox, oy = bs.MARKER_OFFSET_PX
    assert bs.to_court(0.0 - ox, 0.0 - oy) == (0.0, 0.0) and bs.to_court(500.0 - ox, 470.0 - oy) == (47.0, 50.0)   # the image spans a half court
    assert bs.to_court(30.0 - ox, hy) == (5.25, 3.0)                                          # corner three line sits 3 ft in (vertex 1: 2.99)
    assert bs.mirror_to_right(*bs.RIM_LEFT) == bs.RIM_RIGHT == (88.75, 25.0)
    assert bs.mirror_to_right(29.0, 3.0) == (65.0, 47.0)
    assert bs.nba_loc(hx, hy) == (0.0, 0.0) and bs.nba_loc(hx + 50.0, hy + 237.5) == (50.0, 237.5)
    assert bs.pixel_distance_ft(hx, hy + 237.5) == pytest.approx(23.75)


def test_games_codes_and_seasons():
    g = bs.game_from_id("12.16.2015.PHX.at.GSW")
    assert g == {"game": "12.16.2015.PHX.at.GSW", "date": "2015-12-16", "away": "PHX", "home": "GSW"}
    assert cc.br_code(g) == "201512160GSW" and cc.br_code(bs.game_from_id("11.14.2015.BKN.at.GSW")) == "201511140GSW"
    assert cc.br_code(bs.game_from_id("12.23.2015.NYK.at.CHA")) == "201512230CHO" and bs.BREF_TO_SPORTVU["BRK"] == "BKN"
    assert bs.season_of("2015-12-16") == "2015-16" == bs.season_of("2016-02-27") and bs.season_of("2016-10-25") == "2016-17"
    assert bs.shot_chart_url("201512160GSW").endswith("/boxscores/shot-chart/201512160GSW.html")
    with pytest.raises(ValueError):
        bs.game_from_id("0021500381")


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def test_marker_fields_from_the_bref_markup():
    html = ('<div class="shot-area" id="shots-BRK">'
            '<div style="top:36px;left:238px;" tip="1st Qtr, 11:46.0 remaining&lt;br&gt;Brook Lopez made 2-pointer from 2 ft&lt;br&gt;BRK now leads 2-0" class="tooltip make p-lopezbr01 q-1">●</div>'
            '<div style="left:412px;top:290px" tip="2nd OT, 0:04.5 remaining<br>Joe Johnson missed 3-pointer from 29 ft<br>BRK trails 110-112" class="tooltip miss p-johnsjo02 q-6">×</div>'
            '<div style="top:52px;left:250px;" tip="3rd Qtr, 5:00.0 remaining<br>Thaddeus Young made 2-pointer" class="tooltip make p-youngth01 q-3">●</div>'
            '</div>')
    s = bs.parse_shot_chart(html)
    assert [x["team_br"] for x in s] == ["BRK"] * 3 and [x["idx"] for x in s] == [0, 1, 2]
    assert s[0] == {"team_br": "BRK", "player_slug": "lopezbr01", "player": "Brook Lopez", "period": 1, "clock_s": 706.0, "clock_str": "11:46.0",
                    "made": True, "shot_type": "2PT", "shot_distance_ft": 2.0, "px_left": 238.0, "px_top": 36.0,
                    "desc": "1st Qtr, 11:46.0 remaining | Brook Lopez made 2-pointer from 2 ft | BRK now leads 2-0", "idx": 0}
    assert s[1]["period"] == 6 and s[1]["clock_s"] == 4.5 and s[1]["made"] is False and s[1]["shot_type"] == "3PT" and s[1]["px_left"] == 412.0
    assert s[2]["shot_distance_ft"] is None and s[2]["made"] is True                     # a tip without a distance still parses
    # period from the tip when the q- class is missing; shot areas inside an HTML comment are found too
    html2 = '<div><!-- <div class="shot-area" id="shots-PHO"><div style="top:300px;left:100px;" tip="1st OT, 1:00.0 remaining<br>A B missed 3-pointer from 29 ft" class="tooltip miss p-ab01">x</div></div> --></div>'
    s2 = bs.parse_shot_chart(html2)
    assert len(s2) == 1 and s2[0]["period"] == 5 and s2[0]["team_br"] == "PHO"
    assert bs.parse_shot_chart("<html><body>no chart</body></html>") == []


def test_calibration_flags_a_foreign_pixel_frame():
    hx, hy = bs.HOOP_PX
    good = [{"px_left": hx + 10 * d * np.sin(t), "px_top": hy + 10 * d * np.cos(t), "shot_distance_ft": float(d)}
            for d, t in zip(range(3, 28), np.linspace(-1.2, 1.2, 25))]
    cal = bs.calibrate(good)
    assert cal["ok"] is True and cal["hoop_shift_px"] == 0.0 and cal["resid_default_ft"] < 0.01 and cal["n"] == 25
    shifted = [{**s, "px_top": s["px_top"] + 20.0} for s in good]                             # the baseline 2 ft lower than assumed
    cal = bs.calibrate(shifted)
    assert cal["ok"] is False and cal["hoop_px"] == [hx, hy + 20.0] and cal["resid_fit_ft"] < 0.01 < cal["resid_default_ft"]
    assert bs.calibrate(good[:3])["ok"] is None                                               # too few far shots to say


@pytest.mark.parametrize("page", SYNTHETIC, ids=[p.name for p in SYNTHETIC])
def test_synthetic_page_recovers_every_pbp_shot(page):
    code = code_of(page)
    truth = pbp_shots(code)
    assert truth, "data/pbp/%s.json is the fixture's source" % code
    shots = bs.parse_shot_chart(page.read_text())
    game = bs.game_from_id(GAME_OF[code])
    df = bs.rows_for_game(game, shots, "2026-11-05T00:00:00Z", cc.split_of(game, splits.load()))
    assert len(shots) == len(df) == len(truth)
    assert df["made"].sum() == sum(e["kind"] == "shot_made" for e in truth)
    assert sorted(zip(df["period"], df["clock_s"])) == sorted((e["period"], e["clock_s"]) for e in truth)
    side = {"home": game["home"], "away": game["away"]}
    assert df.groupby("team").size().to_dict() == pd.Series([side[e["team"]] for e in truth]).value_counts().to_dict()
    assert set(df["source"]) == {"bref"} and set(df["endpoint"]) == {"boxscores/shot-chart/%s.html" % code} and set(df["season"]) == {"2015-16"}
    assert set(df["basket"]) == {"left"} and df["court_x"].between(0, 47).all() and df["court_y"].between(0, 50).all()
    assert df["dist_resid_ft"].abs().max() < 0.5                                              # whole-foot tips, rounded pixels
    assert list(df.columns) == list(bs.COLUMNS)
    if code == "201602270OKC":
        assert df["period"].max() == 5 and (df["period"] == 5).sum() == sum(e["period"] == 5 for e in truth)   # the overtime
    if code == "201512160GSW":
        assert set(df["split"]) == {"heldout_game"} and df["heldout"].all() and set(df["team_br"]) == {"PHO", "GSW"} and set(df["team"]) == {"PHX", "GSW"}


@pytest.mark.skipif(not REAL, reason="no real saved page in tests/fixtures/bref (save one with --save-pages on a machine that reaches basketball-reference)")
@pytest.mark.parametrize("page", REAL, ids=[p.name for p in REAL])
def test_real_page_pixel_frame_and_count(page):
    """A real page: at least a full game of markers, every marker placed and timed, the hoop where HOOP_PX says it is
    (the tips' distances agree with the pixel distances within a foot), and the count against the play-by-play."""
    code = code_of(page)
    shots = bs.parse_shot_chart(page.read_text())
    assert len(shots) >= 120, "a full NBA game has 150 to 200 attempts; got %d markers" % len(shots)
    placed = [s for s in shots if s["px_left"] is not None and s["px_top"] is not None and s["period"] and s["clock_s"] is not None]
    assert len(placed) == len(shots)
    cal = bs.calibrate(shots)
    assert cal["ok"] is True, cal
    assert cal["resid_default_ft"] < 1.0, cal
    truth = pbp_shots(code)
    if truth is not None:
        print("%s: bref %d markers vs pbp %d attempts (%+d)" % (code, len(shots), len(truth), len(shots) - len(truth)))
        assert abs(len(shots) - len(truth)) <= 0.05 * len(truth), "bref and the play-by-play disagree by more than 5%%: %d vs %d" % (len(shots), len(truth))


# ── the run loop: cached, resumable, throttled, source=bref, held-out rows apart ───────────────────
def fake_fetcher(pages: dict, calls: list, fail_after: int | None = None):
    def fetch_html(url, throttle, timeout=30.0, tries=3):
        code = re.search(r"/shot-chart/(\w+)\.html$", url).group(1)
        calls.append(code)
        throttle.wait()
        if fail_after is not None and len(calls) > fail_after:
            raise bs.PageError("%s HTTP 429" % url, 429, stop_run=True)
        if code not in pages:
            raise bs.PageError("%s HTTP 404" % url, 404)
        return pages[code]
    return fetch_html


@pytest.mark.skipif(len(SYNTHETIC) < 2, reason="needs both synthetic pages")
def test_collector_run_is_cached_resumable_and_tags_source(tmp_path, monkeypatch):
    pages = {code_of(p): p.read_text() for p in SYNTHETIC}
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(pages, calls))
    out = tmp_path / "collect"
    games = ["02.27.2016.GSW.at.OKC", "12.16.2015.PHX.at.GSW", "11.14.2015.BKN.at.GSW"]                  # the last has no page: 404
    final = bs.main(["--games", *games, "--out", str(out), "--rpm", "60"])
    assert final.startswith("RUN FAILED") and "2 ok" in final and "1 failed" in final
    assert calls == ["201511140GSW", "201512160GSW", "201602270OKC"]                                      # order_games: train game first, then GSW by date
    man = json.loads((out / "bref_shots" / "manifest.json").read_text())
    u = man["units"]["shotchart|2015-16|12.16.2015.PHX.at.GSW"]
    assert u["status"] == "ok" and u["bref_shots"] == u["pbp_shots"] == 179 and u["pbp_source"] == "data/pbp/201512160GSW.json"
    assert u["params"] == {"code": "201512160GSW"} and u["calibration"]["ok"] is True and u["split"] == "heldout_game" and u["heldout"]
    assert man["units"]["shotchart|2015-16|02.27.2016.GSW.at.OKC"]["bref_shots"] == 191 and man["units"]["shotchart|2015-16|02.27.2016.GSW.at.OKC"]["by_team"] == {"GSW": 95, "OKC": 96}
    failed = man["units"]["shotchart|2015-16|11.14.2015.BKN.at.GSW"]
    assert failed["status"] == "failed" and "404" in failed["error"]
    # the page is in the raw cache as received, the rows under parts/, the held-out game compacted apart
    assert (out / "raw" / bs.RAW_ENDPOINT).exists() and len(list((out / "raw" / bs.RAW_ENDPOINT).glob("*.json.gz"))) == 2
    assert (out / "bref_shots" / "parts" / "shotchart" / "2015-16" / "12.16.2015.PHX.at.GSW.parquet").exists()
    fit = pq.read_table(out / "bref_shots" / "shotchart" / "2015-16.parquet").to_pandas()
    held = pq.read_table(out / "bref_shots" / "heldout" / "shotchart" / "2015-16.parquet").to_pandas()
    assert set(fit["game"]) == {"02.27.2016.GSW.at.OKC"} and set(held["game"]) == {"12.16.2015.PHX.at.GSW"} and len(held) == 179
    assert set(fit["source"]) == set(held["source"]) == {"bref"} and set(fit["basket"]) == {"left"}
    assert fit["period"].dtype == np.int8 and fit["clock_s"].dtype == np.float32 and fit["made"].dtype == bool
    # a rerun requests nothing and leaves the failed game alone until --retry-failed
    n = len(calls)
    assert bs.main(["--games", *games, "--out", str(out)]).startswith("RUN FAILED") and len(calls) == n
    pages["201511140GSW"] = pages["201602270OKC"]
    assert bs.main(["--games", *games, "--out", str(out), "--retry-failed"]).startswith("RUN OK") and calls[-1] == "201511140GSW" and len(calls) == n + 1
    rep = bs.main(["--out", str(out), "--report"])
    assert "12.16.2015.PHX.at.GSW" in rep and "+0" in rep and "heldout_game" in rep
    # --save-pages keeps the page for the fixtures directory
    saved = tmp_path / "pages"
    bs.main(["--games", "12.16.2015.PHX.at.GSW", "--out", str(tmp_path / "again"), "--save-pages", str(saved)])
    assert (saved / "201512160GSW.html").read_text() == pages["201512160GSW"]


def test_first_429_stops_the_run_and_rpm_is_capped(tmp_path, monkeypatch, capsys):
    pages = {code_of(p): p.read_text() for p in SYNTHETIC}
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(pages, calls, fail_after=1))
    waits = []
    real_wait = common.Throttle.wait
    monkeypatch.setattr(common.Throttle, "wait", lambda self, extra=0.0: (waits.append(self.interval), real_wait(self, extra))[1])
    final = bs.main(["--games", "12.16.2015.PHX.at.GSW", "02.27.2016.GSW.at.OKC", "11.14.2015.BKN.at.GSW", "--out", str(tmp_path), "--rpm", "40"])
    assert final.startswith("RUN FAILED") and len(calls) == 2                                              # the 429 ends the run: no third request
    assert waits and all(w == pytest.approx(60.0 / 17.0) for w in waits)                                   # never more than 17 a minute
    assert "capped at 17" in capsys.readouterr().out
    man = json.loads((tmp_path / "bref_shots" / "manifest.json").read_text())
    st = {k.split("|")[-1]: v["status"] for k, v in man["units"].items()}
    # order_games puts the train game (no page: 404) first, then PHX at GSW whose 429 ends the run; GSW at OKC is never tried
    assert st == {"11.14.2015.BKN.at.GSW": "failed", "12.16.2015.PHX.at.GSW": "failed"}, st
    assert "429" in man["units"]["shotchart|2015-16|12.16.2015.PHX.at.GSW"]["error"] and "404" in man["units"]["shotchart|2015-16|11.14.2015.BKN.at.GSW"]["error"]


def test_pbp_count_sources(tmp_path, monkeypatch):
    sp = splits.load()
    fam = common.Family(tmp_path, "bref_shots")
    g = bs.game_from_id("12.16.2015.PHX.at.GSW")
    assert bs.pbp_shot_count(g, "201512160GSW", None, fam, common.Throttle(0), False, sp) == (179, "data/pbp/201512160GSW.json")
    g2 = bs.game_from_id("11.14.2015.BKN.at.GSW")
    assert bs.pbp_shot_count(g2, "201511140GSW", None, fam, common.Throttle(0), False, sp) == (None, None)
    # the Step 1 pbp parquet wins when the contract dir has it
    cdir = tmp_path / "contract"
    tbl = pa.table({"kind": pa.array(["shot_made", "rebound", "shot_missed", "ft_made"], pa.string())})
    (cdir / "heldout").mkdir(parents=True)
    pq.write_table(tbl, cdir / "heldout" / "12.16.2015.PHX.at.GSW_pbp.parquet")
    n, src = bs.pbp_shot_count(g, "201512160GSW", cdir, fam, common.Throttle(0), False, sp)
    assert n == 2 and src.endswith("heldout/12.16.2015.PHX.at.GSW_pbp.parquet")
    # the bref pbp page only with --fetch-pbp, cached afterwards
    import fetch_pbp
    calls = []
    monkeypatch.setattr(fetch_pbp, "fetch_game", lambda code: (calls.append(code), [{"kind": "shot_made"}, {"kind": "foul"}, {"kind": "shot_missed"}])[1])
    assert bs.pbp_shot_count(g2, "201511140GSW", None, fam, common.Throttle(0), True, sp) == (2, "boxscores/pbp/201511140GSW.html")
    assert bs.pbp_shot_count(g2, "201511140GSW", None, fam, common.Throttle(0), True, sp)[0] == 2 and calls == ["201511140GSW"]
    monkeypatch.setattr(fetch_pbp, "fetch_game", lambda code: (_ for _ in ()).throw(RuntimeError("HTTP 429 Too Many Requests")))
    with pytest.raises(bs.PageError) as ei:
        bs.pbp_shot_count(bs.game_from_id("12.28.2015.SAC.at.GSW"), "201512280GSW", None, fam, common.Throttle(0), True, sp)
    assert ei.value.stop_run

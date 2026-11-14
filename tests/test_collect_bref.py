"""The basketball-reference ingestion drivers (collect/bref_schedule, bref_seasons, bref_boxscores, bref_lineups,
positions, drive) on hand-built fixture pages under tests/fixtures/bref/pages/ (the pages' real markup, a few rows
each) and the synthetic shot chart pages of tests/fixtures/bref/. No network: an autouse fixture makes
collect.bref_shots.fetch_html and requests.get raise, and each test patches fetch_html to serve the fixtures by URL."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")

import fetch_pbp  # noqa: E402
from collect import common, bref_shots as bs, bref_schedule as sched, bref_seasons as seasons, pbp as collect_pbp  # noqa: E402
from collect import bref_boxscores as box, bref_lineups as lu, positions as pos, drive  # noqa: E402
from sportvu import splits  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "bref"
PAGES = FIX / "pages"
SYNTH = {p.name.split(".")[0]: p for p in FIX.glob("*.synthetic.html")}
PHX_GSW, GSW_OKC = "201512160GSW", "201602270OKC"
NO_TABLE = "<html><body><div id='content'>Site maintenance</div></body></html>"


def _no_network(*a, **k):
    raise AssertionError("network: a fetch reached the real client in a test")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No sleeping, and no path to the network: a forgotten patch raises instead of requesting."""
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    monkeypatch.setattr(bs, "fetch_html", _no_network)
    monkeypatch.setattr(fetch_pbp.requests, "get", _no_network)


def page(name: str) -> str:
    return (PAGES / name).read_text()


def boxscore_as(code: str) -> str:
    """The PHO at GSW box score page re-coded as GSW at OKC (same rows, the team codes swapped) for a second game."""
    html = page("boxscore_201512160GSW.html")
    if code == GSW_OKC:
        return html.replace("GSW", "OKC").replace("PHO", "GSW")
    return html


def fake_fetcher(calls: list, fail_after: int | None = None, missing=(), tableless=(), status_for=None):
    """bref_shots.fetch_html stand-in: fixture pages by URL, PageError 404 for the rest, a 429 after fail_after calls,
    a 200 page without any table for the codes in tableless, a given status (429/403) for the codes in status_for."""
    def fetch_html(url, throttle, timeout=30.0, tries=3):
        calls.append(url)
        throttle.wait()
        if fail_after is not None and len(calls) > fail_after:
            raise bs.PageError("%s HTTP 429" % url, 429, stop_run=True)
        code = (re.search(r"/(\d{8}0[A-Z]{3})\.html$", url) or [None, None])[1]
        if status_for and code in status_for:
            raise bs.PageError("%s HTTP %d" % (url, status_for[code]), status_for[code], stop_run=True)
        if any(t in url for t in tableless):
            return NO_TABLE
        m = re.search(r"/leagues/NBA_(\d{4})_games-([a-z0-9-]+)\.html$", url)
        if m and (PAGES / ("schedule_%s_%s.html" % m.groups())).exists():
            return page("schedule_%s_%s.html" % m.groups())
        m = re.search(r"/boxscores/shot-chart/(\w+)\.html$", url)
        if m and m.group(1) in SYNTH and m.group(1) not in missing:
            return SYNTH[m.group(1)].read_text()
        m = re.search(r"/boxscores/pbp/(\w+)\.html$", url)
        if m and (PAGES / ("pbp_%s.html" % m.group(1))).exists() and m.group(1) not in missing:
            return page("pbp_%s.html" % m.group(1))
        m = re.search(r"/boxscores/(\w+)\.html$", url)
        if m and m.group(1) in (PHX_GSW, GSW_OKC) and m.group(1) not in missing:
            return boxscore_as(m.group(1))
        m = re.search(r"/teams/([A-Z]{3})/(\d{4})/lineups/$", url)
        if m and (PAGES / ("lineups_%s_%s.html" % m.groups())).exists():
            return page("lineups_%s_%s.html" % m.groups())
        m = re.search(r"/leagues/NBA_(\d{4})_per_game\.html$", url)
        if m and (PAGES / ("per_game_%s.html" % m.group(1))).exists():
            return page("per_game_%s.html" % m.group(1))
        raise bs.PageError("%s HTTP 404" % url, 404)
    return fetch_html


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code != 200:
            import requests
            raise requests.HTTPError("%d Client Error" % self.status_code, response=self)


def fake_pbp_get(calls: list, status_for=None):
    """requests.get stand-in for fetch_pbp.fetch_game: the fixture pbp page for the codes that have one, 404 otherwise."""
    def get(url, headers=None, timeout=None):
        calls.append(url)
        code = re.search(r"/boxscores/pbp/(\w+)\.html$", url).group(1)
        if status_for and code in status_for:
            return FakeResponse(status_for[code])
        p = PAGES / ("pbp_%s.html" % code)
        return FakeResponse(200, p.read_text()) if p.exists() else FakeResponse(404)
    return get


def collect_schedule(out: Path, monkeypatch, calls=None) -> str:
    calls = [] if calls is None else calls
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls))
    return sched.main(["--seasons", "2015-16", "--out", str(out)])


def codes_of(calls, kind) -> list:
    rx = {"shots": r"/shot-chart/(\w+)\.html", "pbp": r"/pbp/(\w+)\.html", "box": r"/boxscores/(\d{8}0[A-Z]{3})\.html"}[kind]
    return [m.group(1) for m in (re.search(rx, u) for u in calls) if m]


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────
def test_shared_helpers(capsys):
    assert sched.minutes_of("35:23") == 35.38 and sched.minutes_of("1,065:30") == 1065.5 and sched.minutes_of("240") == 240.0
    assert sched.minutes_of("") is None and sched.minutes_of(None) is None and sched.minutes_of("Did Not Play") is None
    assert sched.parse_date("Tue, Oct 27, 2015") == "2015-10-27" and sched.parse_date("Sat, Feb 6, 2016") == "2016-02-06" and sched.parse_date("nope") is None
    assert sched.end_year("2015-16") == 2016 and sched.schedule_url("2019-20", "october-2019").endswith("/leagues/NBA_2020_games-october-2019.html")
    assert sched.months_of("2016-17") == sched.MONTHS_DEFAULT and len(sched.months_of("2019-20")) == 10 and sched.months_of("2020-21")[0] == "december"
    assert sched.sportvu_game_id("2015-12-16", "PHO", "GSW") == "12.16.2015.PHX.at.GSW" and sched.sportvu_game_id("2015-12-30", "ORL", "BRK") == "12.30.2015.ORL.at.BKN"
    assert sched.split_for("2015-12-16", "PHO", "GSW") == "heldout_game" and sched.split_for("2015-12-28", "SAC", "GSW") == "train"
    assert sched.split_for("2015-12-23", "NYK", "CLE") == "heldout_arena" and sched.split_for("2015-12-30", "ORL", "BRK") is None
    assert sched.split_for("2016-12-16", "PHO", "GSW") is None                                # only 2015-16 games have a split
    assert sched.is_heldout("heldout_arena") and not sched.is_heldout("train") and not sched.is_heldout(None)
    assert len(sched.TEAMS) == 30 and "BRK" in sched.TEAMS and "CHO" in sched.TEAMS and "PHO" in sched.TEAMS
    # the throttle: capped at 17 a minute, 0 or a negative --rpm is the maximum, never unthrottled
    assert sched.capped_throttle(40).interval == pytest.approx(60 / 17) and "capped at 17" in capsys.readouterr().out
    assert sched.capped_throttle(0).interval == pytest.approx(60 / 17) and sched.capped_throttle(-5).interval == pytest.approx(60 / 17)
    assert sched.capped_throttle(10).interval == pytest.approx(6.0)
    # the shared parser: --rpm help names basketball-reference, Play-In is a season type
    ap = sched.bref_parser("x", ["2015-16"])
    assert "basketball-reference requests per minute" in ap.format_help() and "stats.nba.com" not in ap.format_help()
    a = ap.parse_args(["--season-types", "Play-In", "Playoffs"])
    assert a.season_types == ["Play-In", "Playoffs"] and a.rpm == 17.0
    with pytest.raises(SystemExit):
        ap.parse_args(["--season-types", "Preseason"])


def test_pbp_fetch_errors_classify_by_status_code_not_by_substring(tmp_path, monkeypatch):
    """A URL carrying a game code like 202304290DEN must not read as a 429; a real 429 (response status) must."""
    import requests
    sp = splits.load()
    fam = common.Family(tmp_path, "bref_shots")
    g = bs.game_from_id("12.28.2015.SAC.at.GSW")
    monkeypatch.setattr(fetch_pbp, "fetch_game", lambda code: (_ for _ in ()).throw(RuntimeError("404 Client Error for url: https://www.basketball-reference.com/boxscores/pbp/202304290DEN.html")))
    with pytest.raises(bs.PageError) as ei:
        bs.pbp_shot_count(g, "202304290DEN", None, fam, common.Throttle(0), True, sp)
    assert not ei.value.stop_run
    with pytest.raises(Exception) as ei2:
        collect_pbp.br_events("202304290DEN", fam, common.Throttle(0), {})
    assert not isinstance(ei2.value, collect_pbp.BRJailed)
    err = requests.HTTPError("429 Client Error: Too Many Requests for url: .../pbp/202403010BOS.html", response=FakeResponse(429))
    monkeypatch.setattr(fetch_pbp, "fetch_game", lambda code: (_ for _ in ()).throw(err))
    with pytest.raises(bs.PageError) as ei:
        bs.pbp_shot_count(g, "202403010BOS", None, fam, common.Throttle(0), True, sp)
    assert ei.value.stop_run
    state = {}
    with pytest.raises(collect_pbp.BRJailed):
        collect_pbp.br_events("202403010BOS", fam, common.Throttle(0), state)
    assert state["br_jailed"]


# ── schedule ───────────────────────────────────────────────────────────────────────────────────────
def test_schedule_parser_on_the_month_pages():
    g = sched.parse_schedule(page("schedule_2016_december.html"))
    assert [x["game_id"] for x in g] == [PHX_GSW, "201512230CLE", "201512280GSW", "201512300BRK"]
    assert g[0] == {"game_id": PHX_GSW, "date": "2015-12-16", "home": "GSW", "away": "PHO", "home_score": 128, "away_score": 103,
                    "season_type": "Regular Season", "start_time": "10:30p", "overtimes": None, "attendance": 19596.0, "arena": "Oracle Arena", "notes": None}
    assert g[2]["overtimes"] == "OT" and g[3]["away"] == "ORL" and g[3]["home"] == "BRK"
    a = sched.parse_schedule(page("schedule_2016_april.html"))
    assert [(x["game_id"], x["season_type"]) for x in a] == [("201604130GSW", "Play-In"), ("201604160GSW", "Playoffs"), ("201604300GSW", "Playoffs")]
    assert a[2]["home_score"] is None and a[2]["away_score"] is None                            # not played yet: the code from csk
    with pytest.raises(ValueError, match="2015-16 october: no table id=schedule on the page"):
        sched.parse_schedule(NO_TABLE, "2015-16 october")
    df = sched.rows_for_month(g, "2015-16", "december", "2026-11-05T00:00:00Z")
    assert list(df.columns) == list(sched.COLUMNS) and list(df["split"]) == ["heldout_game", "heldout_arena", "train", pd.NA]
    assert "heldout" not in df.columns and str(df["home_score"].dtype) == "Int32" and set(df["source"]) == {"basketball-reference.com"}
    assert set(df["endpoint"]) == {"leagues/NBA_2016_games-december.html"}


def test_schedule_run_is_resumable_and_feeds_games_for_season(tmp_path, monkeypatch, capsys):
    out = tmp_path / "collect"
    calls = []
    final = collect_schedule(out, monkeypatch, calls)
    txt = capsys.readouterr().out
    assert "estimated requests: 9 (9 pending units, 0 pages already cached)" in txt
    assert final.startswith("RUN FAILED") and "3 ok" in final and "6 failed" in final             # three month pages exist, six 404
    assert [re.search(r"games-(\S+)\.html", u).group(1) for u in calls] == list(sched.MONTHS_DEFAULT)
    man = json.loads((out / "bref_schedule" / "manifest.json").read_text())
    dec = man["units"]["games|2015-16|december"]
    assert dec["status"] == "ok" and dec["rows"] == 4 and dec["played"] == 4 and dec["params"] == {"season": "2015-16", "month": "december"}
    assert man["units"]["games|2015-16|april"]["by_type"] == {"Play-In": 1, "Playoffs": 2}
    assert "404" in man["units"]["games|2015-16|october"]["error"]
    games = pq.read_table(out / "bref_schedule" / "games" / "2015-16.parquet").to_pandas()
    assert len(games) == 8 and list(games.columns) == list(sched.COLUMNS) and not (out / "bref_schedule" / "heldout").exists()
    assert len(list((out / "raw" / "bref_schedule").glob("*.json.gz"))) == 3
    # a rerun requests nothing; --retry-failed asks the six again; --dry-run lists without fetching
    n = len(calls)
    assert sched.main(["--seasons", "2015-16", "--out", str(out)]).startswith("RUN FAILED") and len(calls) == n
    assert "estimated requests: 0 (0 pending" in capsys.readouterr().out
    sched.main(["--seasons", "2015-16", "--out", str(out), "--retry-failed", "--limit", "2"])
    assert len(calls) == n + 2
    assert sched.main(["--seasons", "2016-17", "--out", str(out), "--dry-run"]).startswith("DRY RUN: 9 pending") and len(calls) == n + 2
    # a 200 page without the schedule table fails the unit (never "empty"), and --retry-failed redoes it
    out2 = tmp_path / "collect2"
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher([], tableless=("games-december.html",)))
    sched.main(["--seasons", "2015-16", "--out", str(out2)])
    u = json.loads((out2 / "bref_schedule" / "manifest.json").read_text())["units"]["games|2015-16|december"]
    assert u["status"] == "failed" and "2015-16 december: no table id=schedule on the page" in u["error"]
    assert not common.RawCache(out2 / "raw").path("bref_schedule", {"season": "2015-16", "month": "december"}).exists()   # the tableless page is not kept
    calls2 = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls2))
    sched.main(["--seasons", "2015-16", "--out", str(out2), "--retry-failed"])
    assert json.loads((out2 / "bref_schedule" / "manifest.json").read_text())["units"]["games|2015-16|december"]["status"] == "ok" and len(calls2) == 7
    # games_for_season: played regular-season games by default, Play-In and Playoffs on request, held-out tagging on the way out
    fam = common.Family(out, "bref_seasons")
    reg = sched.games_for_season(fam, "2015-16")
    assert [g["game_id"] for g in reg] == [PHX_GSW, "201512230CLE", "201512280GSW", "201512300BRK", GSW_OKC]
    assert reg[0] == {"game_id": PHX_GSW, "date": "2015-12-16", "home": "GSW", "away": "PHO", "home_score": 128, "away_score": 103, "season": "2015-16",
                      "season_type": "Regular Season", "split": "heldout_game", "heldout": True, "game": "12.16.2015.PHX.at.GSW"}
    assert [(g["split"], g["heldout"]) for g in reg[1:]] == [("heldout_arena", True), ("train", False), (None, False), (None, False)]
    both = sched.games_for_season(fam, "2015-16", ("Regular Season", "Playoffs"))
    assert [g["game_id"] for g in both][-1] == "201604160GSW" and len(both) == 6                 # the unplayed playoff game is left out
    assert [g["game_id"] for g in sched.games_for_season(fam, "2015-16", ("Play-In",))] == ["201604130GSW"]
    assert len(sched.games_for_season(fam, "2015-16", None, played_only=False)) == 8
    assert sched.games_for_season(fam, "2016-17") == [] and sched.teams_for_season(fam, "2016-17") == list(sched.TEAMS)
    assert sched.teams_for_season(fam, "2015-16") == ["BRK", "CLE", "GSW", "NYK", "OKC", "ORL", "PHO", "SAC"]
    # the parts feed games_for_season when the compacted file is absent
    (out / "bref_schedule" / "games" / "2015-16.parquet").unlink()
    assert len(sched.games_for_season(fam, "2015-16")) == 5


# ── seasons: shots and play-by-play per game ───────────────────────────────────────────────────────
def test_seasons_run_shots_and_pbp_with_heldout_tagging(tmp_path, monkeypatch, capsys):
    out = tmp_path / "collect"
    collect_schedule(out, monkeypatch)
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls))
    monkeypatch.setattr(seasons, "LOCAL_PBP_DIR", tmp_path / "no_local_pbp")
    capsys.readouterr()
    final = seasons.main(["--shots-seasons", "2015-16", "--pbp-seasons", "2015-16", "--out", str(out), "--season-types", "Regular Season", "Playoffs"])
    txt = capsys.readouterr().out
    assert "estimated requests: 12 (12 pending units, 0 pages already cached)" in txt and "about 1 min" in txt and "stints copies:" in txt
    assert final.startswith("RUN FAILED") and "3 ok" in final and "9 failed" in final
    order = [PHX_GSW, "201512230CLE", "201512280GSW", "201512300BRK", GSW_OKC, "201604160GSW"]
    assert codes_of(calls, "shots") == order and codes_of(calls, "pbp") == order and len(calls) == 12
    man = json.loads((out / "bref_seasons" / "manifest.json").read_text())
    u = man["units"]["shotchart|2015-16|" + PHX_GSW]
    assert u["status"] == "ok" and u["rows"] == 179 and u["split"] == "heldout_game" and u["heldout"] and u["calibration"]["ok"] and u["params"] == {"code": PHX_GSW}
    assert man["units"]["shotchart|2015-16|" + GSW_OKC]["rows"] == 191 and man["units"]["shotchart|2015-16|" + GSW_OKC]["heldout"] is False
    p = man["units"]["playbyplay|2015-16|" + PHX_GSW]
    assert p["status"] == "ok" and p["rows"] == 12 and p["from"] == "page" and p["periods"] == 2 and p["kinds"]["shot_made"] == 3
    assert "404" in man["units"]["playbyplay|2015-16|" + GSW_OKC]["error"] and "404" in man["units"]["shotchart|2015-16|201512230CLE"]["error"]
    # the page and its parsed events are both cached: the page for this driver, the events for bref_shots and collect.pbp
    assert len(list((out / "raw" / "bref_pbp_page").glob("*.json.gz"))) == 1 and len(list((out / "raw" / "bref_pbp").glob("*.json.gz"))) == 1
    assert len(common.RawCache(out / "raw").get("bref_pbp", {"code": PHX_GSW})["body"]) == 12
    # the shots: the held-out game apart from the fitting file, game_id the basketball-reference code, source bref
    fit = pq.read_table(out / "bref_seasons" / "shotchart" / "2015-16.parquet").to_pandas()
    held = pq.read_table(out / "bref_seasons" / "heldout" / "shotchart" / "2015-16.parquet").to_pandas()
    assert set(fit["game_id"]) == {GSW_OKC} and set(fit["game"]) == {"02.27.2016.GSW.at.OKC"} and len(fit) == 191 and set(fit["season_type"]) == {"Regular Season"}
    assert set(held["game_id"]) == {PHX_GSW} and len(held) == 179 and held["heldout"].all() and set(held["split"]) == {"heldout_game"}
    assert list(fit.columns) == list(bs.COLUMNS) and set(fit["source"]) == {"bref"} and set(fit["team"]) == {"GSW", "OKC"} and set(held["team"]) == {"PHX", "GSW"}
    # the play-by-play: contract section 13 columns, frame -1, NBA.com fields null, under heldout/ for the held-out game
    pbp = pq.read_table(out / "bref_seasons" / "heldout" / "playbyplay" / "2015-16.parquet").to_pandas()
    assert len(pq.read_table(out / "bref_seasons" / "playbyplay" / "2015-16.parquet")) == 0        # compact() leaves an empty fitting file
    assert list(pbp.columns) == list(seasons.PBP_COLUMNS) and len(pbp) == 12 and set(pbp["frame"]) == {-1} and pbp["frame_clock_diff_s"].isna().all()
    assert set(pbp["game"]) == {"12.16.2015.PHX.at.GSW"} and set(pbp["game_id"]) == {PHX_GSW} and set(pbp["source"]) == {"basketball-reference.com"}
    assert set(pbp["endpoint"]) == {"boxscores/pbp/%s.html" % PHX_GSW} and set(pbp["season"]) == {"2015-16"} and set(pbp["game_date"]) == {"2015-12-16"}
    assert list(pbp["event"]) == list(range(12)) and pbp["event_id"].isna().all() and pbp["player1_id"].isna().all() and pbp["team_id"].isna().all()
    assert pbp["period"].dtype == np.int8 and pbp["points"].dtype == np.int8 and pbp["clock_s"].dtype == np.float32 and pbp["heldout"].all()
    assert list(pbp["kind"][:3]) == ["shot_made", "shot_missed", "rebound"] and list(pbp["team"][:3]) == ["home", "away", "home"] and pbp["points"].sum() == 8
    assert list(pbp["score"][:2]) == ["0-2", "0-2"] and list(pbp["period"][-3:]) == [2, 2, 2] and pbp["clock_s"].iloc[0] == pytest.approx(700.0)
    # the copies the lineups lane reads
    assert (out / "stints" / "heldout" / "playbyplay" / "2015-16.parquet").read_bytes() == (out / "bref_seasons" / "heldout" / "playbyplay" / "2015-16.parquet").read_bytes()
    assert (out / "stints" / "playbyplay" / "2015-16.parquet").exists() and not list((out / "stints").rglob("*.part"))
    # the raw cache is shared with bref_shots: its run of the same game requests nothing, and a rerun here requests nothing
    n = len(calls)
    assert bs.main(["--games", "02.27.2016.GSW.at.OKC", "--out", str(out)]).startswith("RUN OK") and len(calls) == n
    assert seasons.main(["--shots-seasons", "2015-16", "--pbp-seasons", "2015-16", "--out", str(out), "--season-types", "Regular Season", "Playoffs"]).startswith("RUN FAILED")
    assert len(calls) == n and "estimated requests: 0 (0 pending units" in capsys.readouterr().out
    # events another collector cached under raw/bref_pbp (collect.pbp, bref_shots --fetch-pbp) spare the page request
    cache = common.RawCache(out / "raw")
    cache.put("bref_pbp", {"code": "201512230CLE"}, [{"period": 1, "clock_s": 600.0, "team": "home", "desc": "x", "points": 2, "score": "0-2", "kind": "shot_made"}], "x", 200)
    # play-by-play from data/pbp/<code>.json when present (no request), seeded into the cache
    local = tmp_path / "local_pbp"
    local.mkdir()
    (local / ("%s.json" % GSW_OKC)).write_text(json.dumps({"events": [{"period": 1, "clock_s": 700.0, "team": "away", "desc": "x", "points": 2, "score": "2-0", "kind": "shot_made"}]}))
    monkeypatch.setattr(seasons, "LOCAL_PBP_DIR", local)
    capsys.readouterr()
    seasons.main(["--shots-seasons", "--pbp-seasons", "2015-16", "--out", str(out), "--retry-failed", "--season-types", "Regular Season", "Playoffs"])
    assert "estimated requests: 4 (5 pending units, 1 pages already cached)" in capsys.readouterr().out   # the data/pbp file is not a cached page
    assert len(calls) == n + 3
    units = json.loads((out / "bref_seasons" / "manifest.json").read_text())["units"]
    assert units["playbyplay|2015-16|" + GSW_OKC]["from"] == "data/pbp/%s.json" % GSW_OKC and units["playbyplay|2015-16|201512230CLE"]["from"] == "cache"
    fitpbp = pq.read_table(out / "bref_seasons" / "playbyplay" / "2015-16.parquet").to_pandas()
    assert set(fitpbp["game_id"]) == {GSW_OKC} and len(fitpbp) == 1 and not fitpbp["heldout"].iloc[0]
    held2 = pq.read_table(out / "stints" / "heldout" / "playbyplay" / "2015-16.parquet").to_pandas()
    assert set(held2["game_id"]) == {PHX_GSW, "201512230CLE"} and set(held2["split"]) == {"heldout_game", "heldout_arena"}


def test_seasons_first_429_stops_the_run_and_rpm_is_capped(tmp_path, monkeypatch, capsys):
    out = tmp_path / "collect"
    collect_schedule(out, monkeypatch)
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls, fail_after=1))
    waits = []
    real_wait = common.Throttle.wait
    monkeypatch.setattr(common.Throttle, "wait", lambda self, extra=0.0: (waits.append(self.interval), real_wait(self, extra))[1])
    capsys.readouterr()
    final = seasons.main(["--shots-seasons", "2015-16", "--pbp-seasons", "--out", str(out), "--rpm", "40"])
    txt = capsys.readouterr().out
    assert "--rpm 40 capped at 17" in txt and "estimated requests: 5" in txt and "stopping the run" in txt
    assert final.startswith("RUN FAILED") and len(calls) == 2                                      # the 429 ends the run: no third request
    assert waits and all(w == pytest.approx(60.0 / 17.0) for w in waits)
    man = json.loads((out / "bref_seasons" / "manifest.json").read_text())
    st = {k.split("|")[-1]: v["status"] for k, v in man["units"].items()}
    assert st == {PHX_GSW: "ok", "201512230CLE": "failed"} and "429" in man["units"]["shotchart|2015-16|201512230CLE"]["error"]
    # bref_shots.main shares the fix: --rpm 0 is the maximum, not unthrottled
    waits.clear()
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher([]))
    assert bs.main(["--games", "02.27.2016.GSW.at.OKC", "--out", str(tmp_path / "bs0"), "--rpm", "0"]).startswith("RUN OK")
    assert waits and all(w == pytest.approx(60.0 / 17.0) for w in waits)
    # a 429 on a play-by-play page stops the run the same way (no second request); a 403 too; the later units never enter the manifest
    monkeypatch.setattr(seasons, "LOCAL_PBP_DIR", tmp_path / "none")
    pbp_calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(pbp_calls, status_for={PHX_GSW: 429}))
    final = seasons.main(["--shots-seasons", "--pbp-seasons", "2015-16", "--out", str(out)])
    assert len(pbp_calls) == 1 and final.startswith("RUN FAILED")
    units = json.loads((out / "bref_seasons" / "manifest.json").read_text())["units"]
    assert "429" in units["playbyplay|2015-16|" + PHX_GSW]["error"] and "playbyplay|2015-16|201512230CLE" not in units
    fam = common.Family(out, "bref_seasons")
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher([], status_for={"201512230CLE": 403}))
    with pytest.raises(bs.PageError) as ei:
        seasons.pbp_events(fam, "201512230CLE", common.Throttle(0))
    assert ei.value.stop_run and ei.value.status == 403
    # a 200 page without the pbp table, or a shot chart without markers, fails the unit so --retry-failed redoes it
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher([], tableless=("201512230CLE",)))
    with pytest.raises(ValueError, match="no table id=pbp"):
        seasons.pbp_events(fam, "201512230CLE", common.Throttle(0))
    g = sched.games_for_season(fam, "2015-16")[1]
    with pytest.raises(ValueError, match="201512230CLE: no shot markers on the page"):
        seasons.do_unit({"endpoint": "shotchart", "season": "2015-16", "game": g, "params": {"code": "201512230CLE"}}, fam, common.Throttle(0))


def test_pbp_fixture_page_parses_with_fetch_pbp(monkeypatch):
    """fetch_pbp.parse_events (the parser every play-by-play path reuses) on the fixture pbp page; fetch_game still
    fetches and parses in one go."""
    ev = fetch_pbp.parse_events(page("pbp_%s.html" % PHX_GSW))
    assert len(ev) == 12 and [e["period"] for e in ev] == [1] * 9 + [2] * 3
    assert ev[0] == {"period": 1, "clock_s": 700.0, "team": "home", "desc": "S. Curry makes 2-pt jump shot from 2 ft (assist by D. Green )", "points": 2, "score": "0-2", "kind": "shot_made"}
    assert [e["kind"] for e in ev] == ["shot_made", "shot_missed", "rebound", "shot_made", "foul", "ft_made", "ft_missed", "rebound", "turnover", "shot_made", "shot_missed", "rebound"]
    assert [e["team"] for e in ev[:3]] == ["home", "away", "home"] and ev[9]["team"] == "away" and ev[9]["points"] == 2 and ev[9]["score"] == "2-6"
    assert ev[3]["points"] == 3 and ev[5]["points"] == 1 and ev[6]["points"] == 0
    with pytest.raises(ValueError, match="no table id=pbp"):
        fetch_pbp.parse_events(NO_TABLE)
    calls = []
    monkeypatch.setattr(fetch_pbp.requests, "get", fake_pbp_get(calls))
    assert fetch_pbp.fetch_game(PHX_GSW) == ev and calls == ["https://www.basketball-reference.com/boxscores/pbp/%s.html" % PHX_GSW]
    with pytest.raises(Exception):
        fetch_pbp.fetch_game("201512230CLE")                                                   # 404: raise_for_status


# ── box scores ─────────────────────────────────────────────────────────────────────────────────────
def test_boxscore_parser_on_the_fixture_page():
    rows = box.parse_boxscore(page("boxscore_201512160GSW.html"))
    assert [r["team"] for r in rows] == ["PHO"] * 12 + ["GSW"] * 12                               # away first, then home; the advanced table in the comment ignored
    pho = [r for r in rows if r["team"] == "PHO"]
    assert [r["player_slug"] for r in pho[:5]] == ["blednar01", "knighbr03", "tuckepj01", "morrima03", "chandty01"]
    assert [r["starter"] for r in pho] == [True] * 5 + [False] * 7
    assert pho[0]["player"] == "Eric Bledsoe" and pho[0]["minutes"] == 35.38 and pho[0]["mp_str"] == "35:23" and pho[0]["pts"] == 20.0 and pho[0]["plus_minus"] == -19.0
    assert pho[2]["ft_pct"] is None and pho[2]["fg3_pct"] == 0.333                               # a blank cell is null
    assert pho[6]["player"] == "Mirza Teletović"
    dnp = [r for r in pho if r["reason"]]
    assert [(r["player_slug"], r["reason"], r["minutes"]) for r in dnp] == [("pricero01", "Did Not Play", None), ("weemsso01", "Did Not Dress", None)]
    gsw = [r for r in rows if r["team"] == "GSW"]
    assert [r["player_slug"] for r in gsw[:5]] == ["curryst01", "thompkl01", "barneha02", "greendr01", "bogutan01"] and gsw[0]["plus_minus"] == 23.0
    assert gsw[-1]["reason"] == "Inactive" and sum(r["starter"] for r in rows) == 10
    g = {"game_id": PHX_GSW, "date": "2015-12-16", "home": "GSW", "away": "PHO", "season": "2015-16", "season_type": "Regular Season", "split": "heldout_game", "heldout": True}
    starters, allp = box.rows_for_game(g, rows, "2026-11-05T00:00:00Z")
    assert list(starters.columns) == list(box.STARTER_COLUMNS) and list(allp.columns) == list(box.PLAYER_COLUMNS)
    assert len(starters) == 21 and len(allp) == 24 and starters["starter"].sum() == 10 and starters["home"].sum() == 11
    assert starters["minutes"].dtype == np.float32 and starters["heldout"].all() and set(starters["endpoint"]) == {"boxscores/%s.html" % PHX_GSW}
    assert allp["reason"].notna().sum() == 3 and allp[allp["reason"].notna()]["minutes"].isna().all()
    with pytest.raises(ValueError, match="%s: no table id=box-<TEAM>-game-basic on the page" % PHX_GSW):
        box.parse_boxscore(NO_TABLE, PHX_GSW)
    # older page ids
    old = page("boxscore_201512160GSW.html").replace('id="box-PHO-game-basic"', 'id="box_pho_basic"').replace('id="box-GSW-game-basic"', 'id="box_gsw_basic"')
    assert {r["team"] for r in box.parse_boxscore(old)} == {"PHO", "GSW"}
    # the opening-lineup table the lineups lane reads: starters only, side, SportVU-style game id, null NBA.com ids
    st = box.stints_starters(starters)
    assert list(st.columns) == list(box.STINTS_COLUMNS) and len(st) == 10 and set(st["game"]) == {"12.16.2015.PHX.at.GSW"}
    assert list(st["side"]) == ["away"] * 5 + ["home"] * 5 and list(st["team"]) == ["PHO"] * 5 + ["GSW"] * 5 and list(st["player_slug"][5:]) == ["curryst01", "thompkl01", "barneha02", "greendr01", "bogutan01"]
    assert str(st["team_id"].dtype) == "Int64" and str(st["player_id"].dtype) == "Int64" and st["team_id"].isna().all() and st["player_id"].isna().all()
    assert set(st["game_id"]) == {PHX_GSW} and set(st["game_date"]) == {"2015-12-16"} and set(st["season"]) == {"2015-16"} and st["player"].iloc[5] == "Stephen Curry"


def test_boxscore_run_writes_starters_players_and_the_stints_copy(tmp_path, monkeypatch, capsys):
    out = tmp_path / "collect"
    collect_schedule(out, monkeypatch)
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls, tableless=("201512280GSW",)))
    capsys.readouterr()
    final = box.main(["--seasons", "2015-16", "--out", str(out)])
    txt = capsys.readouterr().out
    assert "estimated requests: 5 (5 pending" in txt and "stints copies:" in txt
    assert final.startswith("RUN FAILED") and "2 ok" in final and "3 failed" in final
    assert codes_of(calls, "box") == [PHX_GSW, "201512230CLE", "201512280GSW", "201512300BRK", GSW_OKC]
    man = json.loads((out / "bref_boxscores" / "manifest.json").read_text())
    u = man["units"]["starters|2015-16|" + PHX_GSW]
    assert u["status"] == "ok" and u["rows"] == 21 and u["players"] == 24 and u["dnp"] == 3 and u["heldout"] and u["params"] == {"code": PHX_GSW}
    assert man["units"]["starters|2015-16|201512280GSW"]["status"] == "failed" and "201512280GSW: no table id=box-<TEAM>-game-basic" in man["units"]["starters|2015-16|201512280GSW"]["error"]
    fit = pq.read_table(out / "bref_boxscores" / "starters" / "2015-16.parquet").to_pandas()
    held = pq.read_table(out / "bref_boxscores" / "heldout" / "starters" / "2015-16.parquet").to_pandas()
    assert set(fit["game_id"]) == {GSW_OKC} and set(fit["team"]) == {"GSW", "OKC"} and fit["starter"].sum() == 10 and len(fit) == 21
    assert set(held["game_id"]) == {PHX_GSW} and held["heldout"].all() and set(held["split"]) == {"heldout_game"}
    assert list(fit.columns) == list(box.STARTER_COLUMNS)
    players = pq.read_table(out / "bref_boxscores" / "players" / "2015-16.parquet").to_pandas()
    assert len(players) == 24 and list(players.columns) == list(box.PLAYER_COLUMNS) and players["pts"].sum() == 231.0
    # the copy the lineups lane reads: STINTS_COLUMNS, starter rows only, held-out rows apart
    st = pq.read_table(out / "stints" / "starters" / "2015-16.parquet").to_pandas()
    sth = pq.read_table(out / "stints" / "heldout" / "starters" / "2015-16.parquet").to_pandas()
    assert list(st.columns) == list(sth.columns) == list(box.STINTS_COLUMNS) and len(st) == len(sth) == 10
    assert set(st["game_id"]) == {GSW_OKC} and set(st["game"]) == {"02.27.2016.GSW.at.OKC"} and list(st["side"]) == ["away"] * 5 + ["home"] * 5
    assert list(st["team"]) == ["GSW"] * 5 + ["OKC"] * 5 and st["team_id"].isna().all() and st["player_id"].isna().all()
    assert set(sth["game"]) == {"12.16.2015.PHX.at.GSW"} and not list((out / "stints").rglob("*.part"))
    n = len(calls)
    assert box.main(["--seasons", "2015-16", "--out", str(out)]).startswith("RUN FAILED") and len(calls) == n
    assert box.main(["--seasons", "2015-16", "--out", str(out), "--compact-only"]) == "compact only"
    assert box.main(["--seasons", "2015-16", "--out", str(out), "--season-types", "Play-In", "--dry-run"]).startswith("DRY RUN: 1 pending")
    # a page whose tables are not the schedule's teams, or not five starters, fails the unit
    fam = common.Family(out, "bref_boxscores")
    g = sched.games_for_season(fam, "2015-16")[0]
    unit = {"season": "2015-16", "game": {**g, "home": "LAL"}, "params": {"code": PHX_GSW}}
    with pytest.raises(ValueError, match="schedule says"):
        box.do_game(unit, fam, common.Throttle(0))
    four_starters = [r for r in box.parse_boxscore(boxscore_as(PHX_GSW)) if r["player_slug"] != "curryst01"]
    monkeypatch.setattr(box, "parse_boxscore", lambda html, where="": four_starters)
    with pytest.raises(ValueError, match="want five each"):
        box.do_game({"season": "2015-16", "game": g, "params": {"code": PHX_GSW}}, fam, common.Throttle(0))


# ── lineups ────────────────────────────────────────────────────────────────────────────────────────
def test_lineups_parser_and_run(tmp_path, monkeypatch):
    rows = lu.parse_lineups(page("lineups_GSW_2016.html"))
    assert [r["rank"] for r in rows] == [1, 2, 3]                                                   # from the comment; the 4-man table ignored
    assert rows[0]["slugs"] == ["curryst01", "thompkl01", "barneha02", "greendr01", "bogutan01"] and rows[0]["names"][0] == "S. Curry"
    assert rows[0]["minutes"] == 812.15 and rows[0]["pace"] == 100.2 and rows[0]["net_rating"] == 13.6 and rows[0]["stats"]["diff_fg_pct"] == 0.063
    assert rows[2]["net_rating"] == -4.2 and rows[1]["minutes"] == 172.5
    df = lu.rows_for_team("GSW", "2015-16", rows, "2026-11-05T00:00:00Z")
    assert list(df.columns) == list(lu.COLUMNS) and len(df) == 3 and set(df["n_players"]) == {5}
    assert df["players"].iloc[0] == "barneha02|bogutan01|curryst01|greendr01|thompkl01" and df["player_names"].iloc[0] == "H. Barnes|A. Bogut|S. Curry|D. Green|K. Thompson"
    assert list(df[["p1", "p5"]].iloc[0]) == ["barneha02", "thompkl01"] and json.loads(df["stats_json"].iloc[1])["diff_pts"] == 31.4
    assert df["net_rating"].dtype == np.float32 and set(df["endpoint"]) == {"teams/GSW/2016/lineups/"}
    # a lineup cell without links falls back to csk
    html = '<table id="lineups_5-man_"><tbody><tr><th data-stat="ranker">1</th><td data-stat="lineup" csk="a01;b01;c01;d01;e01">A | B | C | D | E</td><td data-stat="mp">10:30</td><td data-stat="net_rtg">+2.0</td></tr></tbody></table>'
    r = lu.parse_lineups(html)
    assert r[0]["slugs"] == ["a01", "b01", "c01", "d01", "e01"] and r[0]["net_rating"] == 2.0 and r[0]["minutes"] == 10.5
    with pytest.raises(ValueError, match="GSW 2015-16: no table id=lineups_5-man_ on the page"):
        lu.parse_lineups(NO_TABLE, "GSW 2015-16")
    # the run: one unit per team-season; teams from the schedule when there is one; a tableless page fails and is retried
    out = tmp_path / "collect"
    collect_schedule(out, monkeypatch)
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls, tableless=("PHO",)))
    assert lu.main(["--seasons", "2015-16", "--out", str(out), "--dry-run"]).startswith("DRY RUN: 8 pending")
    final = lu.main(["--seasons", "2015-16", "--teams", "GSW", "PHO", "--out", str(out)])
    assert final.startswith("RUN FAILED") and "1 ok" in final and "1 failed" in final
    assert calls == ["https://www.basketball-reference.com/teams/GSW/2016/lineups/", "https://www.basketball-reference.com/teams/PHO/2016/lineups/"]
    man = json.loads((out / "bref_lineups" / "manifest.json").read_text())
    assert man["units"]["lineups|2015-16|GSW"]["rows"] == 3 and man["units"]["lineups|2015-16|GSW"]["params"] == {"team": "GSW", "season": "2015-16"}
    assert man["units"]["lineups|2015-16|PHO"]["status"] == "failed" and "no table id=lineups_5-man_" in man["units"]["lineups|2015-16|PHO"]["error"]
    t = pq.read_table(out / "bref_lineups" / "lineups" / "2015-16.parquet").to_pandas()
    assert len(t) == 3 and set(t["team"]) == {"GSW"} and t["minutes"].sum() == pytest.approx(1045.65, abs=0.01)
    n = len(calls)
    lu.main(["--seasons", "2015-16", "--teams", "GSW", "PHO", "--out", str(out)])
    assert len(calls) == n
    assert not (out / "raw" / "bref_lineups").exists() or len(list((out / "raw" / "bref_lineups").glob("*.json.gz"))) == 1   # the tableless page is not kept
    lu.main(["--seasons", "2015-16", "--teams", "GSW", "PHO", "--out", str(out), "--retry-failed"])
    assert len(calls) == n + 1 and json.loads((out / "bref_lineups" / "manifest.json").read_text())["units"]["lineups|2015-16|PHO"]["status"] == "failed"


# ── positions ──────────────────────────────────────────────────────────────────────────────────────
def test_positions_parser_run_union_and_roster_join(tmp_path, monkeypatch):
    rows = pos.parse_per_game(page("per_game_2016.html"))
    assert len(rows) == 7 and [r["is_total"] for r in rows] == [False, True, False, False, False, False, False]
    assert rows[0] == {"player_slug": "curryst01", "player": "Stephen Curry", "position": "PG", "team": "GSW", "age": 27, "games": 79, "mp_per_g": 34.2, "is_total": False}
    assert rows[1]["team"] == "TOT" and [r["team"] for r in rows[2:4]] == ["PHO", "WAS"] and rows[5]["position"] == "SF-PF"
    df = pos.rows_for_season("2015-16", rows, "2026-11-05T00:00:00Z")
    assert list(df.columns) == list(pos.COLUMNS) and len(df) == 6 and "TOT" not in set(df["team"])
    assert str(df["age"].dtype) == "Int16" and set(df["season"]) == {"2015-16"} and set(df["endpoint"]) == {"leagues/NBA_2016_per_game.html"}
    by = pos.positions_by_player(df)
    assert len(by) == 5 and by.set_index("player_slug").loc["morrima03", "team"] == "PHO"                 # the team with more games
    # the 2024 redesign renames player, team, games and games started
    new = (page("per_game_2016.html").replace('data-stat="team_id"', 'data-stat="team_name_abbr"').replace(">TOT<", ">2TM<")
           .replace('data-stat="player"', 'data-stat="name_display"').replace('data-stat="g"', 'data-stat="games"').replace('data-stat="gs"', 'data-stat="games_started"'))
    assert pos.rows_for_season("2015-16", pos.parse_per_game(new), "x").drop(columns="fetched_at").equals(df.drop(columns="fetched_at"))
    with pytest.raises(ValueError, match="2015-16: no table id=per_game_stats on the page"):
        pos.parse_per_game(NO_TABLE, "2015-16")
    # the run: one unit per season, the union file after compaction
    out = tmp_path / "collect"
    calls = []
    monkeypatch.setattr(bs, "fetch_html", fake_fetcher(calls))
    final = pos.main(["--seasons", "2015-16", "2016-17", "--out", str(out)])
    assert final.startswith("RUN FAILED") and "1 ok" in final and "1 failed" in final
    assert calls == ["https://www.basketball-reference.com/leagues/NBA_2016_per_game.html", "https://www.basketball-reference.com/leagues/NBA_2017_per_game.html"]
    man = json.loads((out / "positions" / "manifest.json").read_text())
    assert man["units"]["positions|2015-16|2015-16"]["players"] == 5 and man["units"]["positions|2015-16|2015-16"]["total_rows_dropped"] == 1
    assert man["units"]["positions|2015-16|2015-16"]["positions"] == {"PF": 3, "PG": 1, "SF-PF": 1, "SG": 1}
    season_file = pq.read_table(out / "positions" / "positions" / "2015-16.parquet").to_pandas()
    union = pq.read_table(out / "positions.parquet").to_pandas()
    assert len(season_file) == len(union) == 6 and list(union.columns) == list(pos.COLUMNS)
    # the documented join: rosters (section 16) carry bref_slug per tracking player id; positions join on (slug, season), never by name
    rosters = pd.DataFrame({"sportvu_game": ["12.16.2015.PHX.at.GSW"] * 3, "player_id": [201939, 203110, 1626164], "player": ["Stephen Curry", "Draymond Green", "Devin Booker"],
                            "team_id": [1610612744, 1610612744, 1610612756], "team": ["GSW", "GSW", "PHX"], "home": [True, True, False], "jersey": ["30", "23", "1"],
                            "bref_slug": ["curryst01", "greendr01", None], "bref_name": ["Stephen Curry", "Draymond Green", None], "pbp_name": ["S. Curry", "D. Green", "D. Booker"],
                            "match_method": ["exact", "exact", "none"]})
    joined = pos.join_rosters(union, rosters)
    assert list(joined["position"]) == ["PG", "PF", pd.NA] and list(joined["player_id"]) == [201939, 203110, 1626164] and list(joined["season"]) == ["2015-16"] * 3


# ── drive ──────────────────────────────────────────────────────────────────────────────────────────
def test_drive_sync_copies_by_size_and_never_deletes(tmp_path, monkeypatch, capsys):
    src, dst = tmp_path / "data" / "collect", tmp_path / "drive" / "nba_harvest" / "collect"
    (src / "raw" / "bref_schedule").mkdir(parents=True)
    (src / "bref_schedule" / "games").mkdir(parents=True)
    (src / "raw" / "bref_schedule" / "a.json.gz").write_bytes(b"x" * 10)
    (src / "raw" / "bref_schedule" / "b.json.gz").write_bytes(b"y" * 20)
    (src / "raw" / "bref_schedule" / "c.json.gz.part").write_bytes(b"torn")
    (src / "bref_schedule" / "manifest.json").write_text("{}")
    (src / "bref_schedule" / "games" / "2015-16.parquet").write_bytes(b"p" * 30)
    (dst / "raw" / "bref_schedule").mkdir(parents=True)
    (dst / "raw" / "bref_schedule" / "a.json.gz").write_bytes(b"z" * 10)                        # same size: skipped, never compared by content
    (dst / "raw" / "bref_schedule" / "b.json.gz").write_bytes(b"y" * 5)                         # different size: copied
    (dst / "raw" / "bref_schedule" / "only_on_drive.json.gz").write_bytes(b"keep")              # never deleted
    todo = drive.plan(src, dst)
    assert [(str(f.relative_to(src)), why) for f, d, why in todo] == [("bref_schedule/games/2015-16.parquet", "missing"), ("bref_schedule/manifest.json", "missing"), ("raw/bref_schedule/b.json.gz", "size 20 != 5")]
    res = drive.sync(src, dst, dry_run=True)
    assert res["copied"] == 0 and res["skipped"] == 1 and res["files"] == ["bref_schedule/games/2015-16.parquet", "bref_schedule/manifest.json", "raw/bref_schedule/b.json.gz"]
    assert (dst / "raw" / "bref_schedule" / "b.json.gz").read_bytes() == b"y" * 5
    res = drive.sync(src, dst, only=["raw"])
    assert res["copied"] == 1 and res["skipped"] == 1 and (dst / "raw" / "bref_schedule" / "b.json.gz").read_bytes() == b"y" * 20 and not (dst / "bref_schedule").exists()
    assert "1 skipped" in capsys.readouterr().out
    res = drive.sync(src, dst)
    assert res["copied"] == 2 and res["skipped"] == 2 and (dst / "bref_schedule" / "games" / "2015-16.parquet").read_bytes() == b"p" * 30
    assert (dst / "raw" / "bref_schedule" / "a.json.gz").read_bytes() == b"z" * 10 and (dst / "raw" / "bref_schedule" / "only_on_drive.json.gz").exists()
    assert not (dst / "raw" / "bref_schedule" / "c.json.gz.part").exists() and not list(dst.rglob("*.part"))
    assert drive.sync(src, dst)["copied"] == 0 and drive.sync(src, dst, only=["nothing_here"]) == {"copied": 0, "bytes": 0, "skipped": 0, "files": []}
    # the CLI: --sync-to with DATA_ROOT given, --sync-from, the Drive root from the environment
    monkeypatch.setenv("NBA_DRIVE_ROOT", str(tmp_path / "drive" / "nba_harvest"))
    assert drive.drive_root() == tmp_path / "drive" / "nba_harvest"
    (src / "bref_schedule" / "manifest.json").write_text('{"units": {}}')
    assert drive.main(["--sync-to", "--local", str(src)])["files"] == ["bref_schedule/manifest.json"]
    (dst / "positions.parquet").write_bytes(b"q")
    assert drive.main(["--sync-from", str(tmp_path / "drive" / "nba_harvest"), "--local", str(src)])["files"] == ["positions.parquet", "raw/bref_schedule/only_on_drive.json.gz"]
    assert (src / "positions.parquet").read_bytes() == b"q" and (src / "raw" / "bref_schedule" / "only_on_drive.json.gz").read_bytes() == b"keep"
    drive.main(["--where"])
    assert str(tmp_path / "drive" / "nba_harvest") in capsys.readouterr().out
    monkeypatch.setenv("NBA_DRIVE_ROOT", str(tmp_path / "nowhere"))
    assert drive.drive_root() is None

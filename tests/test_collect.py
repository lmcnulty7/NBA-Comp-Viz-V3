"""collect/: the collection-plan collectors on tiny fixtures in the stats.nba.com response shapes. No network:
FakeStats answers from tests/fixtures/collect/<endpoint>.json (every endpoint, any parameters)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")

from collect import common, nba_stats, pbp, shots, tracking, context, stints, probe, extras  # noqa: E402
from sportvu import colab_contract as cc, splits  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "collect"
GSW, BKN = 1610612744, 1610612751
G1 = "0021500144"


class FakeStats:
    """collect.nba_stats.NBAStats stand-in: fixtures by endpoint; empty result sets for seasons in `empty`;
    a ClientError for endpoints in `fail`. Records every (endpoint, params) it was asked for."""
    def __init__(self, cache=None, throttle=None, empty=(), fail=()):
        self.cache = cache or common.RawCache(Path(os.environ.get("PYTEST_TMP", "/tmp")) / "unused_raw")
        self.calls, self.empty, self.fail, self.requests_made = [], set(empty), set(fail), 0

    def get_record(self, endpoint, overrides):
        params = nba_stats.params_for(endpoint, overrides)
        self.calls.append((endpoint, params))
        if endpoint in self.fail:
            raise nba_stats.ClientError("%s failed after 5 tries: HTTP 500" % endpoint)
        fx = json.loads((FIX / (endpoint + ".json")).read_text())
        body = fx["body"]
        season = params.get("Season") or params.get("SeasonYear")
        if (endpoint, season) in self.empty:
            body = {"resultSets": [{**rs, "rowSet": []} for rs in body.get("resultSets", [])]}
        self.requests_made += 1
        return {"body": body, "fetched_at": fx["fetched_at"], "params": params}

    def get(self, endpoint, overrides):
        return self.get_record(endpoint, overrides)["body"]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)


# ── common ─────────────────────────────────────────────────────────────────────────────────────────
def test_seasons_and_clocks():
    assert common.season_str(2015) == "2015-16" and common.season_str(1999) == "1999-00"
    assert common.seasons_from(2019, 2021) == ["2019-20", "2020-21", "2021-22"]
    import datetime as dt
    assert common.last_completed_season(dt.date(2026, 11, 5)) == 2025 and common.last_completed_season(dt.date(2026, 3, 1)) == 2024
    assert common.clock_to_s("11:23") == 683.0 and common.clock_to_s("0:04.5") == 4.5 and common.clock_to_s(None) is None
    assert common.elapsed_s(1, 720.0) == 0 and common.elapsed_s(2, 600.0) == 840 and common.elapsed_s(5, 300.0) == 2880
    assert nba_stats.game_id_season("0021500381") == "2015-16" and nba_stats.game_id_season("0029900001") == "1999-00"


def test_raw_cache_and_manifest(tmp_path):
    cache = common.RawCache(tmp_path / "raw")
    params = nba_stats.params_for("gamerotation", {"GameID": G1})
    assert cache.get("gamerotation", params) is None
    rec = cache.put("gamerotation", params, {"x": 1}, "https://stats.nba.com/stats/gamerotation", 200)
    assert cache.get("gamerotation", params)["body"] == {"x": 1} and rec["fetched_at"].endswith("Z")
    with pytest.raises(KeyError):
        nba_stats.params_for("gamerotation", {"Nope": 1})
    m = common.Manifest(tmp_path / "manifest.json", "t")
    keys = [common.Manifest.key("e", "2015-16", i) for i in range(4)]
    m.mark(keys[0], "ok", rows=3); m.mark(keys[1], "failed", error="HTTP 500"); m.mark(keys[2], "empty", rows=0)
    assert m.pending(keys, retry_failed=False) == [keys[3]]
    assert m.pending(keys, retry_failed=True) == [keys[1], keys[3]]
    m.save()
    again = common.Manifest(tmp_path / "manifest.json", "t")
    assert again.doc["counts"] == {"ok": 1, "empty": 1, "failed": 1, "rows": 3}


def test_throttle_spacing(monkeypatch):
    t = common.Throttle(rpm=60)
    clock = [100.0]
    slept = []
    monkeypatch.setattr(common.time, "time", lambda: clock[0])
    monkeypatch.setattr(common.time, "sleep", lambda s: slept.append(s))
    t.wait(); t.wait(); t.wait(extra=5)
    assert slept == [1.0, 6.0]


def test_heldout_split_from_splits_json():
    assert common.heldout_split("2015-12-16", "PHX", "GSW") == "heldout_game"
    assert common.heldout_split("2015-12-23", "NYK", "CLE") == "heldout_arena"
    assert common.heldout_split("2015-11-04", "NYK", "CLE") == "heldout_arena"      # every game at the held-out arena
    assert common.heldout_split("2015-11-14", "BKN", "GSW") is None                # a training game is free to fit on


# ── play-by-play ───────────────────────────────────────────────────────────────────────────────────
def test_nba_events_follow_the_contract():
    body = json.loads((FIX / "playbyplayv2.json").read_text())["body"]
    ev = pbp.nba_events(body, GSW)
    assert [e["kind"] for e in ev] == ["period_start", "jump_ball", "shot_made", "shot_missed", "rebound", "foul", "ft_made", "ft_missed",
                                       "substitution", "shot_made", "turnover", "period_end"]
    assert [e["points"] for e in ev] == [0, 0, 3, 0, 0, 0, 1, 0, 0, 2, 0, 0]
    assert [e["team"] for e in ev] == [None, "home", "home", "away", "home", "away", "home", "home", "away", "away", "home", None]
    assert ev[2]["score"] == "0-3" and ev[9]["score"] == "2-4" and ev[2]["clock_s"] == 702.0
    assert ev[8]["player1_id"] == 203900 and ev[8]["player2_id"] == 201960 and ev[8]["event_type"] == 8
    assert ev[0]["player1_id"] is None and ev[0]["team_id"] is None


def frames_parquet(path: Path, gameid: str, n: int = 500):
    """A tiny Step 1 frames parquet: n frames of period 1 from 720 s at 25 Hz (500 frames = 20 s), Curry (home) and Lopez (away)."""
    parsed = {"game": "x", "gameid": gameid, "date": "2015-11-14",
              "teams": {GSW: {"abbr": "GSW", "name": "W", "side": "home"}, BKN: {"abbr": "BKN", "name": "N", "side": "visitor"}},
              "players": {201939: {"name": "Stephen Curry", "jersey": "30", "team": GSW}, 201599: {"name": "Brook Lopez", "jersey": "11", "team": BKN}},
              "moments": [{"q": 1, "clock": round(720 - i * 0.04, 2), "shot": 24.0, "ts": 1447559011007 + 40 * i, "ball": [47.0, 25.0, 5.0],
                           "players": [[GSW, 201939, 40.0, 25.0], [BKN, 201599, 50.0, 25.0]]} for i in range(n)]}
    t = cc.frames_table(parsed)
    pq.write_table(t, path)
    return t


def step1_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sportvu_contract"
    (d / "heldout").mkdir(parents=True)
    frames_parquet(d / "11.14.2015.BKN.at.GSW.parquet", G1)
    frames_parquet(d / "heldout" / "12.16.2015.PHX.at.GSW.parquet", "0021500381")
    man = cc.load_manifest(d)
    for gid, date, away, home, split in (("11.14.2015.BKN.at.GSW", "2015-11-14", "BKN", "GSW", "train"), ("12.16.2015.PHX.at.GSW", "2015-12-16", "PHX", "GSW", "heldout_game")):
        man["games"][gid] = {"game": gid, "date": date, "away": away, "home": home, "split": split, "heldout": split != "train", "status": "ok",
                             "moments": 50, "frames_rows": 100, "pbp_joined": False, "pbp": None, "parquet": gid + ".parquet"}
    cc.write_manifest(d, man)
    return d


def test_pbp_collector_fills_step1_files(tmp_path, monkeypatch):
    d = step1_dir(tmp_path)
    fake = FakeStats()
    monkeypatch.setattr(pbp, "NBAStats", lambda cache, throttle: fake)
    final = pbp.main(["--contract-dir", str(d), "--out", str(tmp_path / "collect"), "--no-br"])
    assert final.startswith("RUN OK")
    t = pq.read_table(d / "11.14.2015.BKN.at.GSW_pbp.parquet")
    assert t.num_rows == 12 and t.column("source").to_pylist()[0] == "stats.nba.com" and t.column("endpoint").to_pylist()[0] == "playbyplayv2"
    assert t.column("season").to_pylist()[0] == "2015-16" and t.column("frame").to_pylist()[:3] == [0, 0, 450]    # 11:42 = 702 s = frame 450
    assert t.column("frame").to_pylist()[-1] == -1                                                             # 0:00 is outside the 500 frames
    assert set(t.schema.names) >= set(cc.pbp_schema().names) | set(pbp.EXTRA_COLS)
    man = json.loads((d / "manifest.json").read_text())
    assert man["games"]["11.14.2015.BKN.at.GSW"]["pbp_joined"] and man["games"]["11.14.2015.BKN.at.GSW"]["pbp"]["source"] == "stats.nba.com"
    assert (d / "heldout" / "12.16.2015.PHX.at.GSW_pbp.parquet").exists()                                      # held-out stays under heldout/
    own = json.loads((tmp_path / "collect" / "pbp" / "manifest.json").read_text())
    assert own["counts"]["ok"] == 2 and all(u["params"] for u in own["units"].values())
    # second run: nothing pending, no request
    n = len(fake.calls)
    assert pbp.main(["--contract-dir", str(d), "--out", str(tmp_path / "collect"), "--no-br"]).startswith("RUN OK") and len(fake.calls) == n


def test_pbp_falls_back_to_basketball_reference_and_stops_on_429(tmp_path, monkeypatch):
    d = step1_dir(tmp_path)
    monkeypatch.setattr(pbp, "NBAStats", lambda cache, throttle: FakeStats(fail={"playbyplayv2"}))
    import fetch_pbp
    calls = []

    def fake_br(code):
        calls.append(code)
        if len(calls) == 1:
            return [{"period": 1, "clock_s": 720.0, "team": None, "desc": "Jump ball", "points": 0, "score": None, "kind": "other"}]
        raise RuntimeError("HTTP 429 Too Many Requests")
    monkeypatch.setattr(fetch_pbp, "fetch_game", fake_br)
    final = pbp.main(["--contract-dir", str(d), "--out", str(tmp_path / "collect")])
    assert final.startswith("RUN FAILED") and calls == ["201511140GSW", "201512160GSW"]     # PHO only when Phoenix is home
    t = pq.read_table(d / "11.14.2015.BKN.at.GSW_pbp.parquet")
    assert t.column("source").to_pylist() == ["basketball-reference.com"] and t.column("frame").to_pylist() == [0]
    own = json.loads((tmp_path / "collect" / "pbp" / "manifest.json").read_text())
    failed = [u for u in own["units"].values() if u["status"] == "failed"]
    assert len(failed) == 1 and "429" in failed[0]["error"]
    # the failed game is redone only with --retry-failed
    monkeypatch.setattr(fetch_pbp, "fetch_game", lambda code: [{"period": 1, "clock_s": 719.0, "team": "home", "desc": "x", "points": 2, "score": "0-2", "kind": "shot_made"}])
    assert pbp.main(["--contract-dir", str(d), "--out", str(tmp_path / "collect")]).startswith("RUN FAILED")
    assert pbp.main(["--contract-dir", str(d), "--out", str(tmp_path / "collect"), "--retry-failed"]).startswith("RUN OK")


# ── shots ──────────────────────────────────────────────────────────────────────────────────────────
def test_shots_contract_and_heldout_split(tmp_path, monkeypatch):
    fake = FakeStats()
    monkeypatch.setattr(shots, "NBAStats", lambda cache, throttle: fake)
    out = tmp_path / "collect"
    final = shots.main(["--out", str(out), "--seasons", "2015-16", "--limit", "1"])
    assert final.startswith("RUN FAILED") and "29 units still pending" in final      # --limit stops early: honest verdict
    units = shots.units(["2015-16"], ["Regular Season"])
    assert len(units) == 30 and units[0][1]["params"]["TeamID"] == 1610612737
    df = common.read_parquet(out / "shots" / "shotchartdetail" / "2015-16.parquet")
    held = common.read_parquet(out / "shots" / "heldout" / "shotchartdetail" / "2015-16.parquet")
    assert len(df) == 1 and len(held) == 2
    assert df.iloc[0]["game_id"] == G1 and not df.iloc[0]["made"] and df.iloc[0]["shot_type"] == "2PT" and df.iloc[0]["clock_s"] == 680.0
    assert df.iloc[0]["sportvu_game"] == "11.14.2015.BKN.at.GSW" and pd.isna(df.iloc[0]["split"]) and not df.iloc[0]["heldout"]
    assert sorted(held["split"]) == ["heldout_arena", "heldout_game"] and held["made"].all()
    assert set(["source", "endpoint", "season", "fetched_at", "season_type", "loc_x", "loc_y", "release_frame", "closest_defender_id"]) <= set(df.columns)
    assert df["release_frame"].isna().all()
    with pytest.raises(SystemExit):
        common.guard_not_heldout(list(held["sportvu_game"]), "t")
    assert common.guard_not_heldout(list(df["sportvu_game"]), "t") == 1


# ── tracking ───────────────────────────────────────────────────────────────────────────────────────
def test_tracking_units_wide_and_long(tmp_path, monkeypatch):
    fake = FakeStats()
    monkeypatch.setattr(tracking, "NBAStats", lambda cache, throttle: fake)
    out = tmp_path / "collect"
    final = tracking.main(["--out", str(out), "--seasons", "2015-16"])
    assert final.startswith("RUN OK")
    eps = {e for e, _ in fake.calls}
    assert eps == {"leaguedashplayerstats", "leaguedashplayerptshot", "leaguedashptstats", "playerdashptshotdefend", "playerdashptreb"}
    assert sum(1 for e, p in fake.calls if e == "leaguedashplayerptshot") == 1 + 4 + 3
    assert sum(1 for e, p in fake.calls if e == "leaguedashptstats") == 12 * 2
    assert sum(1 for e, p in fake.calls if e == "playerdashptreb") == 2                   # one per rostered player
    ptshot = common.read_parquet(out / "tracking" / "leaguedashplayerptshot" / "2015-16.parquet")
    assert sorted(ptshot["split_name"].unique()) == ["CloseDefDistRange", "TouchTimeRange", "none"] and len(ptshot) == 16
    long = common.read_parquet(out / "tracking" / "aggregates" / "2015-16.parquet")
    assert set(["player_id", "season", "endpoint", "metric", "value", "source", "fetched_at", "split_name", "split_value"]) <= set(long.columns)
    curry_tight = long[(long.endpoint == "leaguedashplayerptshot") & (long.split_value == "0-2 Feet - Very Tight") & (long.player_id == 201939) & (long.metric == "fg_pct")]
    assert len(curry_tight) == 1 and curry_tight.iloc[0]["value"] == pytest.approx(0.504)
    assert not long["metric"].str.endswith("_rank").any()
    man = json.loads((out / "tracking" / "manifest.json").read_text())
    u = man["units"][common.Manifest.key("playerdashptshotdefend", "2015-16", "RegularSeason_201939")]
    assert u["params"]["PlayerID"] == 201939 and u["params"]["Season"] == "2015-16" and u["rows"] == 4


def test_tracking_per_game_and_v3_shapes(tmp_path, monkeypatch):
    fake = FakeStats()
    monkeypatch.setattr(tracking, "NBAStats", lambda cache, throttle: fake)
    out = tmp_path / "collect"
    assert tracking.main(["--out", str(out), "--seasons", "2015-16", "--skip-players", "--per-game"]).startswith("RUN OK")
    games = nba_stats.games_from_log(fake.get("leaguegamelog", {"Season": "2015-16"}))
    assert [g["game_id"] for g in games] == [G1, "0021500381"] and games[0]["home"] == "GSW" and games[0]["away_team_id"] == BKN
    box = common.read_parquet(out / "tracking" / "boxscoreplayertrackv3" / "2015-16.parquet")
    held = common.read_parquet(out / "tracking" / "heldout" / "boxscoreplayertrackv3" / "2015-16.parquet")
    assert len(box) == 2 and len(held) == 2 and set(box["game_id"]) == {G1} and held["split"].iloc[0] == "heldout_game"
    row = box[box.player_id == 201939].iloc[0]
    assert row["speed"] == 4.5 and row["distance"] == 2.7 and row["touches"] == 80 and row["home"] and row["team"] == "GSW"


# ── context ────────────────────────────────────────────────────────────────────────────────────────
def test_context_units_and_empty_seasons(tmp_path, monkeypatch):
    fake = FakeStats(empty={("synergyplaytypes", "2014-15"), ("leaguehustlestatsplayer", "2014-15"), ("leagueseasonmatchups", "2014-15")})
    monkeypatch.setattr(context, "NBAStats", lambda cache, throttle: fake)
    out = tmp_path / "collect"
    assert context.main(["--out", str(out), "--seasons", "2014-15"]).startswith("RUN OK")   # empty units are not failures
    man = json.loads((out / "context" / "manifest.json").read_text())
    assert man["counts"]["empty"] == 30 + 44 + 1 and man["counts"]["ok"] == 60 and man["counts"]["failed"] == 0
    assert context.main(["--out", str(out), "--seasons", "2015-16", "--per-game"]).startswith("RUN OK")
    syn = common.read_parquet(out / "context" / "synergyplaytypes" / "2015-16.parquet")
    assert len(syn) == 44 * 2 and set(syn["play_type"]) == set(nba_stats.SYNERGY_PLAY_TYPES) and set(syn["type_grouping"]) == {"offensive", "defensive"}
    m = common.read_parquet(out / "context" / "boxscorematchupsv3" / "2015-16.parquet")
    assert len(m) == 2 and set(m.columns) >= {"def_player_id", "off_player_id", "partial_possessions", "player_points", "matchup_field_goals_attempted", "game_id"}
    assert m[m.def_player_id == 203110].iloc[0]["player_points"] == 6
    rot = common.read_parquet(out / "context" / "gamerotation" / "2015-16.parquet")
    assert len(rot) == 12 and rot["home"].sum() == 6
    long = common.read_parquet(out / "context" / "aggregates" / "2015-16.parquet")
    assert "lineups" not in long["endpoint"].values and set(long["endpoint"]) >= {"synergyplaytypes", "leaguedashlineups", "leaguehustlestatsplayer", "boxscorematchupsv3"}


# ── stints ─────────────────────────────────────────────────────────────────────────────────────────
def test_stints_from_rotation_and_scores(tmp_path, monkeypatch):
    fake = FakeStats()
    monkeypatch.setattr(stints, "NBAStats", lambda cache, throttle: fake)
    out = tmp_path / "collect"
    assert stints.main(["--out", str(out), "--seasons", "2015-16"]).startswith("RUN OK")
    st = common.read_parquet(out / "stints" / "stints" / "2015-16.parquet")
    assert list(st["stint"]) == [0, 1] and list(st["start_s"]) == [0.0, 90.0] and list(st["end_s"]) == [90.0, 720.0]
    assert st["lineup_ok"].all() and st["n_home"].tolist() == [5, 5]
    assert st["home_ids"].iloc[0] == "2738|101106|201939|202691|203110" and st["home_ids"].iloc[1] == "2738|101106|201575|201939|203110"
    assert st["points_home"].tolist() == [4, 0] and st["points_away"].tolist() == [0, 2] and st["score_home"].tolist() == [4, 4]
    assert st["period_start"].tolist() == [1, 1] and st["endpoint"].iloc[0] == "gamerotation+playbyplayv2"
    pbp_rows = common.read_parquet(out / "stints" / "playbyplay" / "2015-16.parquet")
    held_rows = common.read_parquet(out / "stints" / "heldout" / "playbyplay" / "2015-16.parquet")
    assert len(pbp_rows) == 12 and (pbp_rows["frame"] == -1).all() and set(pbp_rows["game_id"]) == {G1}
    assert len(held_rows) == 12 and set(held_rows["game_id"]) == {"0021500381"} and held_rows["split"].iloc[0] == "heldout_game"
    assert (out / "stints" / "heldout" / "stints" / "2015-16.parquet").exists()
    assert stints.period_of(0) == 1 and stints.period_of(2879.9) == 4 and stints.period_of(2880) == 5 and stints.period_of(3180) == 6
    # seconds instead of tenths are detected
    rows = stints.build_stints("g", [{"person_id": 1, "home": True, "in_time_real": 0, "out_time_real": 720}, {"person_id": 2, "home": False, "in_time_real": 0, "out_time_real": 720}], [])
    assert rows[0]["end_s"] == 720.0 and not rows[0]["lineup_ok"]


# ── probe and fixtures ─────────────────────────────────────────────────────────────────────────────
def test_probe_earliest_season(tmp_path):
    fake = FakeStats(empty={("synergyplaytypes", "2013-14"), ("synergyplaytypes", "2014-15")})
    out = probe.probe(fake, ["synergy", "pbp"], ["2013-14", "2014-15", "2015-16"])
    assert out["families"]["synergy"]["earliest_season"] == "2015-16" and out["families"]["synergy"]["by_season"]["2014-15"] == 0
    assert out["families"]["pbp"]["earliest_season"] == "2013-14"
    assert probe.earliest({"2015-16": "error: HTTP 400", "2016-17": 5}) == "2016-17" and probe.earliest({}) is None
    assert "synergy" in probe.table(out)
    fail = FakeStats(fail={"leaguehustlestatsplayer"})
    out = probe.probe(fail, ["hustle"], ["2015-16"])
    assert out["families"]["hustle"]["by_season"]["2015-16"].startswith("error")


def test_dump_fixture_trims_a_cached_response(tmp_path):
    cache = common.RawCache(tmp_path / "raw")
    body = json.loads((FIX / "shotchartdetail.json").read_text())["body"]
    params = nba_stats.params_for("shotchartdetail", {"Season": "2015-16", "TeamID": GSW})
    cache.put("shotchartdetail", params, body, "u", 200)
    out = nba_stats.dump_fixture(cache, "shotchartdetail", params, tmp_path / "f.json", max_rows=2)
    assert len(out["body"]["resultSets"][0]["rowSet"]) == 2 and json.loads((tmp_path / "f.json").read_text())["trimmed_to"] == 2


def test_fixtures_carry_every_endpoint_parameter():
    for p in FIX.glob("*.json"):
        fx = json.loads(p.read_text())
        if fx["endpoint"] in nba_stats.ENDPOINTS:
            assert set(fx["params"]) == set(nba_stats.ENDPOINTS[fx["endpoint"]]), p.name


# ── extras ─────────────────────────────────────────────────────────────────────────────────────────
def test_odds_rows_and_injury_mapping(tmp_path):
    rec = json.loads((FIX / "oddsapi_historical.json").read_text())
    rows = extras.odds_rows(rec, is_close=False)
    assert len(rows) == 6 and {r["market"] for r in rows} == {"h2h", "spreads", "totals"}
    spread = [r for r in rows if r["market"] == "spreads" and r["side"] == "Brooklyn Nets"][0]
    assert spread["point"] == -7.5 and spread["price"] == -110 and spread["game_key"] == "2020-12-24_GoldenStateWarriors_BrooklynNets"
    assert all(r["source"] == extras.ODDS_SOURCE and r["snapshot_at"] == "2020-12-23T23:30:00Z" for r in rows)
    df, mapping = extras.injuries_frame(FIX / "injury_reports_sample.csv")
    assert mapping == {"player": "Player", "team": "Team", "date": "Date", "status": "Status", "reason": "Reason"}
    assert df["season"].tolist() == ["2021-22", "2021-22"] and df["status"].tolist() == ["Out", "Questionable"]
    assert {"source", "endpoint", "fetched_at"} <= set(df.columns)

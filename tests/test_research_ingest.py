"""tools/research_ingest.py: name normalisation, blob hash checks and pin loading. No network, no data."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools import research_ingest as I

GAME = "11.01.2015.AAA.at.BBB"            # made-up ids: never a real test game


def test_prefixed_names_normalise_to_the_game_the_frozen_regex_accepts():
    assert I.prefixed_name(GAME) == "2016.NBA.Raw.SportVU.Game.Logs11.01.2015.AAA.at.BBB.7z"
    assert I.normal_name(I.prefixed_name(GAME)) == GAME
    assert I.normal_name(GAME + ".7z") == GAME


def test_the_frozen_regex_still_rejects_the_prefixed_name():
    from sportvu.manifest import parse_name
    assert parse_name(I.prefixed_name(GAME)) is None


@pytest.mark.parametrize("name", ["2016.NBA.Raw.SportVU.Game.Logs", "readme.md", "x" + GAME + ".7z",
                                  I.PREFIX + GAME + ".zip", I.PREFIX + "11.01.2015.aaa.at.BBB.7z"])
def test_other_names_do_not_normalise(name):
    assert I.normal_name(name) is None


def test_git_blob_sha1_matches_git():
    assert I.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert I.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    p = tmp_path / "a.7z"
    data = bytes(range(256)) * 40
    p.write_bytes(data)
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True, text=True, check=True)
    assert I.git_blob_sha1(data) == out.stdout.strip() == I.git_hash_object(p)
    assert I.git_blob_sha1(data) != hashlib.sha1(data).hexdigest()      # the blob header is part of the hash


def test_check_blob_fails_loudly_on_a_mismatch():
    data = b"archive bytes"
    assert I.check_blob(GAME, data, I.git_blob_sha1(data)) == I.git_blob_sha1(data)
    with pytest.raises(SystemExit, match="does not match"):
        I.check_blob(GAME, data + b"!", I.git_blob_sha1(data))


def test_checked_pins_needs_a_blob_pin_for_every_game():
    other = "11.02.2015.CCC.at.DDD"
    pins = {GAME: "a" * 40, other: "b" * 40}
    assert I.checked_pins(pins) == pins
    with pytest.raises(SystemExit, match="without an archive blob pin"):
        I.checked_pins({GAME: None})                     # what the door returns for a source without pins
    with pytest.raises(SystemExit, match="without an archive blob pin"):
        I.checked_pins({GAME: "xyz"})
    with pytest.raises(SystemExit, match="not a game name"):
        I.checked_pins({"junk": "c" * 40})
    with pytest.raises(SystemExit, match="no pinned archive"):
        I.checked_pins({})


def test_check_listing_requires_exactly_the_pinned_prefixed_entries():
    pins = {GAME: "a" * 40}
    good = [{"name": I.prefixed_name(GAME), "sha": "a" * 40}, {"name": "11.03.2015.EEE.at.FFF.7z", "sha": "d" * 40}]
    assert I.check_listing(good, pins) == 1
    with pytest.raises(SystemExit, match="differs from the pin"):
        I.check_listing([{"name": I.prefixed_name(GAME), "sha": "e" * 40}], pins)
    with pytest.raises(SystemExit, match="not the pinned games"):
        I.check_listing(good + [{"name": I.prefixed_name("11.04.2015.GGG.at.HHH"), "sha": "f" * 40}], pins)


def test_the_output_root_may_not_sit_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    with pytest.raises(SystemExit):
        I.guard_root(tmp_path / "data" / "sportvu_untouched")
    assert I.guard_root(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()


def test_every_blocked_lane_step_is_described():
    steps = set(I.BLOCKED["steps"])
    for lane, needs in I.BLOCKED["lanes_needing_them"].items():
        assert set(needs) <= steps, lane
    assert set(I.BLOCKED["pages_per_game"]) == {"play_by_play", "shot_chart"}


def test_shot_counts_are_row_counts_only_and_never_read_an_outcome():
    import pandas as pd

    other = "11.02.2015.CCC.at.DDD"                      # no outcome column anywhere: reading one would fail
    shots = pd.DataFrame({"sportvu_game": [GAME, GAME, other]})
    sf = pd.DataFrame({"sportvu_game": [GAME, other, other], "confidence_flag": ["high", "low", "high"]})
    feats = pd.DataFrame({"sportvu_game": [GAME, GAME, other], "has_contest": [1, 0, 1]})
    out = I.shot_counts(shots, sf, feats, {GAME: 2})
    assert out[GAME] == {"shot_chart_rows": 2, "pbp_field_goal_attempts": 2, "shot_frames_rows": 1,
                         "shot_frames_high": 1, "assembled_rows": 2, "has_contest": 1}
    assert out[other]["pbp_field_goal_attempts"] is None and out[other]["shot_frames_rows"] == 2
    assert not any("made" in k for c in out.values() for k in c)


def test_the_bref_stage_stays_under_the_site_limit():
    assert I.BREF_RPM <= 15


# ── review fixes: record merge, reuse by archive, one hash per archive, play-by-play sources, stop on failure ──────
def test_update_record_merges_and_keeps_other_keys(tmp_path):
    I.update_record(tmp_path, {"games": {"a": 1}, "bref_stage": {"verdict": "OK"}})
    out = I.update_record(tmp_path, {"games": {"b": 2}})
    assert out["games"] == {"b": 2} and out["bref_stage"] == {"verdict": "OK"}
    assert json.loads((tmp_path / "ingest.json").read_text())["bref_stage"] == {"verdict": "OK"}


def test_reuse_needs_the_same_archive_and_the_same_rows():
    old = {"status": "ok", "frames_rows": 10, "archive_git_blob_sha1": "a" * 40}
    assert I.reusable(old, 10, "a" * 40)
    assert not I.reusable(old, 10, "b" * 40)                         # another archive
    assert not I.reusable(dict(old, archive_git_blob_sha1=None), 10, "a" * 40)
    assert not I.reusable(old, 9, "a" * 40) and not I.reusable(old, 0, "a" * 40)
    assert not I.reusable(dict(old, status="failed"), 10, "a" * 40) and not I.reusable(None, 10, "a" * 40)


def test_digests_come_from_one_pass():
    data = b"some archive"
    d = I.digests(data)
    assert d == {"archive_bytes": len(data), "archive_git_blob_sha1": I.git_blob_sha1(data),
                 "archive_sha256": hashlib.sha256(data).hexdigest()}


def test_download_keeps_a_matching_file_without_the_network(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "curl", lambda *a: pytest.fail("no network expected"))
    (tmp_path / f"{GAME}.7z").write_bytes(b"abc")
    path, d = I.download(GAME, I.git_blob_sha1(b"abc"), tmp_path)
    assert path == tmp_path / f"{GAME}.7z" and d == I.digests(b"abc")


def test_download_refuses_bytes_that_miss_the_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "curl", lambda url, dest: Path(dest).write_bytes(b"tampered"))
    with pytest.raises(SystemExit, match="does not match"):
        I.download(GAME, I.git_blob_sha1(b"original"), tmp_path)
    assert list(tmp_path.iterdir()) == []                             # no part file, no final file


def _pbp_parquet(path, kinds):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"kind": kinds}), path)


ENTRY = {"game": GAME, "date": "2015-11-01", "away": "AAA", "home": "BBB", "status": "ok"}
CODE = "201511010BBB"


def test_pbp_source_order_and_loud_failure(tmp_path):
    from collect import bref_shots, common
    pbp_path, collect_dir, project = tmp_path / "c" / f"{GAME}_pbp.parquet", tmp_path / "collect", tmp_path / "proj"
    with pytest.raises(SystemExit, match="no play-by-play"):
        I.pbp_source(ENTRY, pbp_path, collect_dir, project)
    events = [{"kind": "shot_made"}, {"kind": "rebound"}, {"kind": "shot_missed"}]
    common.RawCache(collect_dir / "raw").put(bref_shots.RAW_PBP_ENDPOINT, {"code": CODE}, events, "u", 200)
    assert I.pbp_source(ENTRY, pbp_path, collect_dir, project) == ("bref_pbp_cache", events)
    (project / "data" / "pbp").mkdir(parents=True)
    (project / "data" / "pbp" / f"{CODE}.json").write_text(json.dumps({"events": events[:1]}))
    assert I.pbp_source(ENTRY, pbp_path, collect_dir, project) == ("data_pbp_json", events[:1])
    _pbp_parquet(pbp_path, ["shot_made", "foul"])
    assert I.pbp_source(ENTRY, pbp_path, collect_dir, project)[0] == "data_pbp_json"   # parquet not marked joined
    assert I.pbp_source(dict(ENTRY, pbp_joined=True), pbp_path, collect_dir, project) == ("pbp_parquet", None)
    assert I.pbp_field_goal_attempts(None, pbp_path) == 1 and I.pbp_field_goal_attempts(events, pbp_path) == 2


def _stage_root(tmp_path, monkeypatch, record=True):
    from sportvu import colab_contract as cc
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    root = tmp_path / "root"
    out_dir = root / "sportvu_contract"
    cc.write_manifest(out_dir, {"games": {GAME: dict(ENTRY, pbp_joined=True)}})
    _pbp_parquet(out_dir / f"{GAME}_pbp.parquet", ["shot_missed", "shot_made"])
    if record:
        I.update_record(root, {"games": {GAME: {"status": "ok"}}})
    return root


def test_bref_stage_needs_the_record_before_any_request(tmp_path, monkeypatch):
    root = _stage_root(tmp_path, monkeypatch, record=False)
    with pytest.raises(SystemExit, match="no ingest.json"):
        I.bref_stage(root, collect=lambda argv: pytest.fail("fetched before the record was read"),
                     phase0=lambda *a, **k: 0, check=lambda *a: ({}, {}), project_root=tmp_path)


def test_bref_stage_stops_and_records_a_failed_phase0(tmp_path, monkeypatch):
    root = _stage_root(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="run_phase0 returned 1"):
        I.bref_stage(root, collect=lambda argv: "RUN OK: 1 ok", phase0=lambda *a, **k: 1,
                     check=lambda *a: pytest.fail("checked after a failed Phase 0"), project_root=tmp_path)
    rec = I.read_record(root)
    assert rec["bref_stage"]["verdict"] == "PHASE0 FAILED" and rec["games"] == {GAME: {"status": "ok"}}


def test_bref_stage_stops_on_a_failed_collector(tmp_path, monkeypatch):
    root = _stage_root(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="did not finish"):
        I.bref_stage(root, collect=lambda argv: "RUN FAILED: 0 ok", phase0=lambda *a, **k: pytest.fail("ran"),
                     check=lambda *a: ({}, {}), project_root=tmp_path)
    assert I.read_record(root)["bref_stage"]["verdict"] == "COLLECTOR FAILED"


def test_bref_stage_records_the_check_on_success(tmp_path, monkeypatch):
    root = _stage_root(tmp_path, monkeypatch)
    seen = {}
    st = I.bref_stage(root, collect=lambda argv: seen.setdefault("argv", argv) and "RUN OK: 1 ok",
                      phase0=lambda *a, **k: 0, check=lambda r, pbp: ({"shots": 2}, {GAME: pbp[GAME]}),
                      project_root=tmp_path)
    assert st["verdict"] == "OK" and st["per_game"] == {GAME: 2} and st["pbp_sources"] == {GAME: "pbp_parquet"}
    assert seen["argv"][seen["argv"].index("--rpm") + 1] == str(I.BREF_RPM)
    assert I.read_record(root)["games"] == {GAME: {"status": "ok"}}


def test_default_root_is_built_from_the_home_folder():
    assert I.DEFAULT_ROOT == Path.home() / "Developer" / "nba-v3-worktrees" / "_research-data" / "sportvu_untouched"

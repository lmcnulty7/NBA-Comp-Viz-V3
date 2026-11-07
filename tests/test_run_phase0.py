"""tests/test_run_phase0.py: analytics.run_phase0 and analytics.audit_shots on a synthetic DATA_ROOT.

The fixture root comes from tests/_fx_phase0.py (analytics.fixtures.write_fixture_root when lane I1's module
is in place, else the local fallback). Every expectation is read from the manifest the writer produced, not
hard-coded, so either writer satisfies the tests. No real tracking data is used.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import audit_shots, run_phase0 as rp, schema  # noqa: E402
from tests import _fx_phase0 as fx  # noqa: E402

EM_DASH = chr(0x2014)


def _call(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = rp.main(argv)
    return rc, buf.getvalue()


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """A fixture DATA_ROOT where the last training game has lost its play-by-play (pbp_joined false)."""
    base = tmp_path_factory.mktemp("phase0_root")
    fx.write_fixture_root(base, seed=0, n_games=6, heldout_every=3)
    r = fx.fixture_data_root(base)
    man = json.loads((r / "manifest.json").read_text())
    train = [g for g, e in man["games"].items() if not e["heldout"] and e["pbp_joined"]]
    assert len(train) >= 3 and any(e["heldout"] for e in man["games"].values())
    victim = train[-1]
    e = man["games"][victim]
    e["pbp_joined"] = False
    e["pbp"] = {"code": "x", "events": 0, "matched": 0, "error": "HTTPError 429"}
    e["pbp_parquet"] = None
    pbp = (r / "heldout" if e["heldout"] else r) / (victim + "_pbp.parquet")
    if pbp.exists():
        pbp.unlink()
    (r / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return r


@pytest.fixture(scope="module")
def step1(root) -> dict:
    return json.loads((root / "manifest.json").read_text())


@pytest.fixture(scope="module")
def eligible(step1) -> list:
    """Status ok, play-by-play joined, and the manifest's heldout flag agreeing with sportvu/splits.json."""
    return [g for g, e in step1["games"].items()
            if e["status"] == "ok" and e["pbp_joined"] and bool(e["heldout"]) == rp.routed_heldout(g, e)]


@pytest.fixture(scope="module")
def disagree(step1) -> list:
    """Games whose heldout flag disagrees with the routing (none in the fallback fixture)."""
    return [g for g, e in step1["games"].items()
            if e["status"] == "ok" and e["pbp_joined"] and bool(e["heldout"]) != rp.routed_heldout(g, e)]


@pytest.fixture(scope="module")
def no_pbp(step1) -> str:
    return [g for g, e in step1["games"].items() if not e["pbp_joined"]][0]


@pytest.fixture(scope="module")
def out(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("phase0_out")


def _manifest(out: Path) -> dict:
    return json.loads((out / "analytics" / rp.MANIFEST_NAME).read_text())


def _table(out: Path, step: str, game: str, step1: dict) -> Path:
    return schema.table_path(out / "analytics", step, rp.season_of_game(game), game, heldout=bool(step1["games"][game]["heldout"]))


# ── dry run ────────────────────────────────────────────────────────────────────────────────────────
def test_dry_run_lists_games_and_writes_nothing(root, out, eligible, no_pbp):
    rc, text = _call(["--data", str(root), "--out", str(out), "--dry-run"])
    assert rc == 0 and text.rstrip().splitlines()[-1] == "RUN OK"
    for g in eligible:
        assert g in text
    assert ("skip %s" % no_pbp)[:20] in text and "pbp_joined false" in text
    assert "nothing written" in text
    assert not (out / "analytics").exists() and not (out / "reports").exists()
    assert EM_DASH not in text


def test_games_and_limit_narrow_the_dry_run(root, out, eligible):
    rc, text = _call(["--data", str(root), "--out", str(out), "--dry-run", "--games", eligible[0], "nope.game"])
    assert rc == 0 and eligible[0] in text and "nope.game" in text and "not in the manifest" in text
    assert eligible[1] not in text
    rc, text = _call(["--data", str(root), "--out", str(out), "--dry-run", "--limit", "2"])
    assert "2 games would run" in text


# ── the full run ───────────────────────────────────────────────────────────────────────────────────
def test_full_run_writes_every_table(root, out, step1, eligible, no_pbp, disagree):
    rc, text = _call(["--data", str(root), "--out", str(out)])
    assert rc == 0, text
    assert text.rstrip().splitlines()[-1] == "RUN OK"
    man = _manifest(out)
    assert sorted(man["games"]) == sorted(eligible)
    assert no_pbp in man["skipped"] and "pbp_joined false" in man["skipped"][no_pbp]
    for g in disagree:
        assert g in man["skipped"] and "sportvu/splits.json" in man["skipped"][g]
        assert not any(p.stem == g for p in (out / "analytics").rglob("*.parquet"))
    for g in eligible:
        entry = man["games"][g]
        assert entry["status"] == "ok" and entry["heldout"] == bool(step1["games"][g]["heldout"])
        for step in rp.GAME_STEPS:
            rec = entry["steps"][step]
            assert rec["status"] == "ok" and rec["error"] is None and rec["seconds"] >= 0
            p = _table(out, step, g, step1)
            assert p.exists(), (step, g)
            assert (out / rec["path"]) == p
            assert rec["rows"] == pq.read_metadata(str(p)).num_rows
            assert rec["schema_problems"] == []
            assert ("heldout" in p.parts) == bool(step1["games"][g]["heldout"])
            df = pq.read_table(str(p)).to_pandas()
            assert schema.validate(df, step) == []
        assert entry["steps"]["shot_frames"]["validate"]["n_shots"] > 0
        assert entry["steps"]["possessions"]["validate"]["n"] > 0
        assert entry["points_check"] is not None and entry["points_check"]["ok"]
    # a held-out game never lands in a training folder and the reverse
    for step in rp.GAME_STEPS:
        train_dir = out / "analytics" / step / rp.season_of_game(eligible[0])
        held_dir = out / "analytics" / step / "heldout" / rp.season_of_game(eligible[0])
        train_files = {p.stem for p in train_dir.glob("*.parquet")} if train_dir.exists() else set()
        held_files = {p.stem for p in held_dir.glob("*.parquet")} if held_dir.exists() else set()
        assert train_files == {g for g in eligible if not step1["games"][g]["heldout"]}
        assert held_files == {g for g in eligible if step1["games"][g]["heldout"]}
    assert not ({no_pbp} | set(disagree)) & (train_files | held_files)
    assert man["gaps"]["games"] == len(step1["games"])
    assert (out / "reports" / "sportvu_gaps.md").exists()
    assert (out / "reports" / "localisation_shots.md").exists() and (out / "reports" / "localisation_rebounds.md").exists()


def test_summary_report_carries_the_totals(root, out, step1, eligible, no_pbp):
    text = (out / "reports" / rp.SUMMARY_NAME).read_text(encoding="utf-8")
    man = _manifest(out)
    n_shots = sum(e["steps"]["shot_frames"]["validate"]["n_shots"] for e in man["games"].values())
    n_found = sum(e["steps"]["shot_frames"]["validate"]["n_release_found"] for e in man["games"].values())
    n_misses = sum(e["steps"]["rebound_frames"]["validate"]["n_misses"] for e in man["games"].values())
    n_poss = sum(e["steps"]["possessions"]["validate"]["n"] for e in man["games"].values())
    assert "| games processed (status ok, pbp joined) | %d |" % len(eligible) in text
    assert "| n_shots | %d |" % n_shots in text and "| n_release_found | %d |" % n_found in text
    assert "| share_release_found | %.4f |" % (n_found / n_shots) in text
    assert "identity_agreement" in text and "| n_misses | %d |" % n_misses in text
    assert "| possessions | %d |" % n_poss in text
    assert "| points-sum check: possession points equal the final score | %d |" % len(eligible) in text
    assert "| games in manifest | %d |" % len(step1["games"]) in text      # the gaps summary
    for g in eligible:
        assert "| %s |" % g in text
    assert "| %s | no play-by-play (pbp_joined false" % no_pbp in text
    assert text.rstrip().endswith("RUN OK")
    assert EM_DASH not in text
    t = rp.totals(man)
    assert t["shot_frames"]["n_shots"] == n_shots and t["games"]["skipped"] == len(man["skipped"]) >= 1


def test_second_run_reuses_everything(root, out, eligible):
    before = {str(p): p.stat().st_mtime_ns for p in (out / "analytics").rglob("*.parquet")}
    rc, text = _call(["--data", str(root), "--out", str(out)])
    assert rc == 0 and text.rstrip().splitlines()[-1] == "RUN OK"
    assert "0 games to run, %d fully reused" % len(eligible) in text
    man = _manifest(out)
    for g in eligible:
        for step in rp.GAME_STEPS:
            assert man["games"][g]["steps"][step]["reused"] is True
            assert man["games"][g]["steps"][step]["status"] == "ok"
    after = {str(p): p.stat().st_mtime_ns for p in (out / "analytics").rglob("*.parquet")}
    assert after == before


def test_missing_output_is_rebuilt(root, out, step1, eligible):
    g = eligible[0]
    p = _table(out, "possessions", g, step1)
    p.unlink()
    rc, text = _call(["--data", str(root), "--out", str(out)])
    assert rc == 0 and p.exists()
    assert "1 games to run" in text
    rec = _manifest(out)["games"][g]["steps"]
    assert "reused" not in rec["possessions"] and rec["shot_frames"]["reused"] is True


def test_retry_failed_redoes_a_failed_game(root, out, eligible):
    g = eligible[1]
    mp = out / "analytics" / rp.MANIFEST_NAME
    man = json.loads(mp.read_text())
    man["games"][g]["steps"]["shot_frames"] = {"status": "failed", "error": "RuntimeError: injected", "path": None,
                                                "rows": None, "validate": None, "schema_problems": None, "seconds": 0.1}
    man["games"][g]["status"] = "failed"
    mp.write_text(json.dumps(man, indent=1))
    # without the flag the failure is kept and the run fails
    rc, text = _call(["--data", str(root), "--out", str(out)])
    assert rc == 1 and text.rstrip().splitlines()[-1] == "RUN FAILED"
    rec = _manifest(out)["games"][g]
    assert rec["status"] == "failed" and rec["steps"]["shot_frames"]["status"] == "failed"
    assert rec["steps"]["shot_frames"]["reused"] is True
    summary = (out / "reports" / rp.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "## Failed steps" in summary and "injected" in summary and summary.rstrip().endswith("RUN FAILED")
    # with it the step is redone and the run is ok again
    rc, text = _call(["--data", str(root), "--out", str(out), "--retry-failed"])
    assert rc == 0 and text.rstrip().splitlines()[-1] == "RUN OK"
    rec = _manifest(out)["games"][g]
    assert rec["status"] == "ok" and rec["steps"]["shot_frames"]["status"] == "ok"
    assert "reused" not in rec["steps"]["shot_frames"] and rec["steps"]["rosters"]["reused"] is True
    assert "1 games to run" in text


def test_steps_subset_and_workers(root, out, tmp_path, step1, eligible):
    out2 = tmp_path / "out2"
    rc, text = _call(["--data", str(root), "--out", str(out2), "--steps", "shot_frames,possessions", "--limit", "2",
                      "--workers", "2"])
    assert rc == 0 and text.rstrip().splitlines()[-1] == "RUN OK"
    assert "workers 2" in text and "Drive" in text
    man = _manifest(out2)
    assert len(man["games"]) == 2 and man["gaps"] is None
    for g, e in man["games"].items():
        assert set(e["steps"]) == {"shot_frames", "possessions"}
        assert all(r["status"] == "ok" for r in e["steps"].values())
        assert _table(out2, "shot_frames", g, step1).exists()
        assert not _table(out2, "rosters", g, step1).exists()
    assert not (out2 / "reports" / "sportvu_gaps.md").exists()


def test_fixtures_flag_writes_the_repo_report(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "FIXTURE_REPORT", tmp_path / "phase0_driver.md")
    monkeypatch.setattr(fx, "write_fixture_root", fx.write_fixture_root_fallback)   # the quick writer: plumbing only
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix="": str(tmp_path / "fxroot"))
    (tmp_path / "fxroot").mkdir()
    rc, text = _call(["--fixtures", "--out", str(tmp_path / "fxout"), "--limit", "1", "--steps", "possessions,gaps"])
    assert rc == 0 and text.rstrip().splitlines()[-1] == "RUN OK"
    rep = (tmp_path / "phase0_driver.md").read_text(encoding="utf-8")
    assert rep.startswith("# Phase 0 driver summary") and "Fixture output" in rep and EM_DASH not in rep


# ── the audit sample ───────────────────────────────────────────────────────────────────────────────
def test_audit_sample_is_equal_per_training_game(root, out, step1, eligible, tmp_path):
    train = [g for g in eligible if not step1["games"][g]["heldout"]]
    held = [g for g in eligible if step1["games"][g]["heldout"]]
    n = 2 * len(train)
    sample_path = tmp_path / "audit" / "shots_sample.parquet"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = audit_shots.main(["--data", str(out), "--n", str(n), "--seed", "0", "--out", str(sample_path)])
    assert rc == 0 and sample_path.exists()
    s = pd.read_parquet(sample_path)
    assert list(s.columns) == audit_shots.COLUMNS
    counts = s["sportvu_game"].value_counts()
    assert sorted(counts.index) == sorted(train)
    assert set(counts.tolist()) == {2}
    assert not set(s["sportvu_game"]) & set(held)
    assert (s["release_frame"] >= 0).all()
    assert s["audited_release_frame"].isna().all() and s["audited_ok"].isna().all() and s["auditor_note"].isna().all()
    assert str(s["audited_release_frame"].dtype) == "Int32" and str(s["audited_ok"].dtype) == "boolean"
    # the draw is deterministic in the seed
    s2, info = audit_shots.draw(audit_shots.training_tables(out), n, seed=0)
    assert s2["event"].tolist() == s["event"].tolist() and info["per_game"] == 2 and info["games_short"] == {}
    s3, _ = audit_shots.draw(audit_shots.training_tables(out), n, seed=1)
    assert s3.shape == s.shape


def test_audit_summarise_reports_agreement(out, tmp_path):
    sample_path = tmp_path / "shots_sample.parquet"
    s, _ = audit_shots.draw(audit_shots.training_tables(out), 8, seed=0)
    s["audited_release_frame"] = s["release_frame"].astype("Int32")
    s.loc[0, "audited_release_frame"] = int(s.loc[0, "release_frame"]) + 5      # one disagreement
    s.loc[1, "audited_release_frame"] = int(s.loc[1, "release_frame"]) - 2      # at the tolerance
    s.loc[2, "audited_release_frame"] = pd.NA                                     # not audited
    s["audited_ok"] = pd.array([True] * len(s), dtype="boolean")
    s.to_parquet(sample_path, index=False)
    res = audit_shots.summarise(pd.read_parquet(sample_path))
    assert res["audited"] == len(s) - 1 and res["within_tolerance"] == len(s) - 2
    assert res["agreement_within_tolerance"] == round((len(s) - 2) / (len(s) - 1), 4)
    assert res["share_audited_ok"] == 1.0 and "high" in res["by_confidence_flag"]
    report = tmp_path / "audit.md"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = audit_shots.main(["--summarise", str(sample_path), "--report", str(report)])
    assert rc == 0 and "agreement within 2 frames" in buf.getvalue()
    text = report.read_text(encoding="utf-8")
    assert "| agreement_within_tolerance |" in text and "## By game" in text and EM_DASH not in text


# ── review follow-ups: step records across --steps runs, refusals, captured errors, torn files, bref pages ──
def test_steps_subset_keeps_records_of_other_steps(root, out, tmp_path, eligible):
    out3 = tmp_path / "out3"
    g = eligible[0]
    rc, _ = _call(["--data", str(root), "--out", str(out3), "--steps", "shot_frames", "--games", g])
    assert rc == 0 and set(_manifest(out3)["games"][g]["steps"]) == {"shot_frames"}
    rc, text = _call(["--data", str(root), "--out", str(out3), "--steps", "possessions", "--games", g])
    assert rc == 0 and "1 games to run" in text
    steps = _manifest(out3)["games"][g]["steps"]
    assert set(steps) == {"shot_frames", "possessions"} and all(r["status"] == "ok" for r in steps.values())
    assert "reused" not in steps["possessions"]
    rc, text = _call(["--data", str(root), "--out", str(out3), "--games", g])
    assert rc == 0
    steps = _manifest(out3)["games"][g]["steps"]
    assert set(steps) == set(rp.GAME_STEPS)
    assert steps["shot_frames"]["reused"] is True and steps["possessions"]["reused"] is True
    assert "reused" not in steps["rosters"] and "reused" not in steps["rebound_frames"]
    # a step record failed in an earlier subset run still fails the game in a later run of other steps
    mp = out3 / "analytics" / rp.MANIFEST_NAME
    man = json.loads(mp.read_text())
    man["games"][g]["steps"]["rosters"]["status"] = "failed"
    mp.write_text(json.dumps(man))
    rc, text = _call(["--data", str(root), "--out", str(out3), "--steps", "possessions", "--games", g])
    assert rc == 1 and text.rstrip().splitlines()[-1] == "RUN FAILED"
    assert _manifest(out3)["games"][g]["status"] == "failed"


def test_heldout_flag_disagreeing_with_splits_is_refused(root, step1, eligible, tmp_path):
    import os
    g = next(x for x in eligible if not step1["games"][x]["heldout"])
    data2 = tmp_path / "data2"
    (data2 / "heldout").mkdir(parents=True)
    for p in root.glob("*.parquet"):
        os.symlink(p, data2 / p.name)
    for p in (root / "heldout").glob("*.parquet"):
        os.symlink(p, data2 / "heldout" / p.name)
    man = json.loads(json.dumps(step1))
    man["games"][g]["heldout"] = True
    man["games"][g]["split"] = "heldout_game"
    (data2 / "manifest.json").write_text(json.dumps(man))
    out4 = tmp_path / "out4"
    rc, text = _call(["--data", str(data2), "--out", str(out4), "--steps", "rosters", "--games", g])
    assert rc == 0 and text.rstrip().splitlines()[-1] == "RUN OK"
    reason = _manifest(out4)["skipped"][g]
    assert "sportvu/splits.json" in reason and "training folder" in reason
    assert not any(p.stem == g for p in (out4 / "analytics").rglob("*.parquet"))
    assert "| %s | manifest says heldout True" % g in (out4 / "reports" / rp.SUMMARY_NAME).read_text(encoding="utf-8")


def test_module_exception_is_captured_per_step(root, out, tmp_path, eligible, monkeypatch):
    from analytics import possessions
    def boom(*a, **k):
        raise RuntimeError("injected possessions failure")
    monkeypatch.setattr(possessions, "run", boom)
    out5 = tmp_path / "out5"
    g = eligible[0]
    rc, text = _call(["--data", str(root), "--out", str(out5), "--steps", "shot_frames,possessions", "--games", g])
    assert rc == 1 and text.rstrip().splitlines()[-1] == "RUN FAILED"
    rec = _manifest(out5)["games"][g]
    assert rec["status"] == "failed"
    assert rec["steps"]["shot_frames"]["status"] == "ok"
    assert rec["steps"]["possessions"]["status"] == "failed" and "injected possessions failure" in rec["steps"]["possessions"]["error"]
    assert "traceback" in rec["steps"]["possessions"]
    summary = (out5 / "reports" / rp.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "injected possessions failure" in summary and summary.rstrip().endswith("RUN FAILED")


def test_torn_parquet_is_rebuilt(root, out, step1, eligible):
    g = eligible[0]
    p = _table(out, "rosters", g, step1)
    data = p.read_bytes()
    p.write_bytes(data[:100])
    assert rp.reads_back(p) is None
    rc, text = _call(["--data", str(root), "--out", str(out), "--games", g])
    assert rc == 0 and "1 games to run" in text
    assert rp.reads_back(p) is not None and rp.reads_back(p) > 0
    assert "reused" not in _manifest(out)["games"][g]["steps"]["rosters"]


def test_bad_data_root_ends_with_run_failed(tmp_path):
    rc, text = _call(["--data", str(tmp_path / "nowhere"), "--out", str(tmp_path / "o")])
    assert rc == 1 and text.rstrip().splitlines()[-1] == "RUN FAILED" and "no Step 1 manifest" in text


def test_bref_dir_fills_slugs(root, step1, eligible, tmp_path):
    from analytics import rosters
    from sportvu.colab_contract import BR_ABBR
    g = next(x for x in eligible if not step1["games"][x]["heldout"])
    frames, _pbp = rp.input_paths(step1["games"][g], root)
    ros = rosters.roster_from_frames(frames)
    areas = []
    for team, grp in ros.groupby("team"):
        divs = []
        for k, name in enumerate(grp["player"]):
            first, last = str(name).split(" ", 1)
            slug = "".join(c for c in last.lower() if c.isalpha())[:5] + first[:2].lower() + "01"
            divs.append('<div style="top:%dpx;left:%dpx;" tip="1st Qtr, 11:%02d.0 remaining<br>%s. %s made 2-pointer from 15 ft<br>'
                        '%s now leads 2-0" class="tooltip make p-%s q-1">&#9679;</div>' % (60 + k, 80 + k, 59 - k, first[0], last, team, slug))
        br = BR_ABBR.get(team, team)
        areas.append('<div id="shooting-%s"><div class="shot-area" id="shots-%s">%s</div></div>' % (br, br, "".join(divs)))
    bref = tmp_path / "bref"
    bref.mkdir()
    code = rp.bref_code(g, step1["games"][g])
    (bref / (code + ".html")).write_text("<html><body>%s</body></html>" % "".join(areas), encoding="utf-8")
    assert rp.bref_page(g, step1["games"][g], bref) is not None
    assert rp.bref_page(eligible[-1], step1["games"][eligible[-1]], bref) is None or eligible[-1] == g
    out6 = tmp_path / "out6"
    rc, text = _call(["--data", str(root), "--out", str(out6), "--steps", "rosters", "--games", g, "--bref-dir", str(bref)])
    assert rc == 0, text
    table = pd.read_parquet(_table(out6, "rosters", g, step1))
    assert table["bref_slug"].notna().all() and table["bref_name"].notna().all()
    v = _manifest(out6)["games"][g]["steps"]["rosters"]["validate"]
    assert v["bref_page"] is True and v["bref_total"] == len(table) and v["bref_matched"] == len(table)
    assert v["bref_slug_filled"] == len(table)
    summary = (out6 / "reports" / rp.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "| games with a shot chart page (bref_slug, bref_name filled) | 1 |" in summary
    assert "| players with bref_slug filled | %d |" % len(table) in summary
    # without a page the columns stay null and the summary says so
    out7 = tmp_path / "out7"
    rc, _ = _call(["--data", str(root), "--out", str(out7), "--steps", "rosters", "--games", g])
    assert rc == 0
    table = pd.read_parquet(_table(out7, "rosters", g, step1))
    assert table["bref_slug"].isna().all()
    assert "| games without one (bref_slug, bref_name null; pass --bref-dir) | 1 |" in (out7 / "reports" / rp.SUMMARY_NAME).read_text(encoding="utf-8")


def test_audit_refuses_heldout_id_planted_in_training_folder(out, step1, eligible, tmp_path):
    import shutil
    held = next(x for x in eligible if step1["games"][x]["heldout"])
    train = next(x for x in eligible if not step1["games"][x]["heldout"])
    d = tmp_path / "planted" / "analytics" / "shot_frames" / rp.season_of_game(train)
    d.mkdir(parents=True)
    shutil.copy(_table(out, "shot_frames", train, step1), d / (train + ".parquet"))
    shutil.copy(_table(out, "shot_frames", held, step1), d / (held + ".parquet"))
    tables = audit_shots.training_tables(tmp_path / "planted")
    assert [p.stem for p in tables] == [train]
    sample, info = audit_shots.draw(tables, 2, seed=0)
    assert set(sample["sportvu_game"]) == {train} and info["games"] == 1


def test_shared_season_and_heldout_helpers(tmp_path):
    """run_phase0 uses analytics.schema.season_of_game (the July rule) and analytics.eval.is_heldout_game for its
    by-id held-out check; the expected output folder agrees with the modules' split_of routing."""
    from analytics import eval as ev
    assert rp.season_of_game is schema.season_of_game
    assert rp.season_of_game("07.01.2016.GSW.at.BOS") == "2016-17" and rp.season_of_game("06.30.2016.GSW.at.BOS") == "2015-16"
    for game in ("12.16.2015.PHX.at.GSW", "01.01.2016.GSW.at.CLE", "11.11.2015.LAL.at.BOS"):
        held = ev.is_heldout_game(game)
        assert held == rp.routed_heldout(game)
        p = rp.expected_output("shot_frames", game, tmp_path)
        assert ("heldout" in p.parts) == held and p.parent.name == rp.season_of_game(game)
    assert ev.is_heldout_game("12.16.2015.PHX.at.GSW") and not ev.is_heldout_game("11.11.2015.LAL.at.BOS")

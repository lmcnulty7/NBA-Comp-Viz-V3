"""Stale-join guard + credit gate: tier2_credit must refuse a join built from
different outcomes than the ones on disk (run-10 incident: a crashed join left
a stale file that credit aggregated silently), and the n>=MIN_BUCKET gate must
lift on its own the moment a bucket crosses — no manual override anywhere."""
import json

import pytest

from conftest import N_FRAMES  # noqa: F401  (repo-root sys.path side effect)
from tier2_credit import MIN_BUCKET, check_join_fresh, credit_rows
from tier2_join import sources_fingerprint


def write_outcomes(d, clip, payload):
    (d / f"{clip}_outcomes.json").write_text(json.dumps(payload))


@pytest.fixture()
def pbp_dir(tmp_path):
    write_outcomes(tmp_path, "game_a_s00", [{"set_start_frame": 1}])
    write_outcomes(tmp_path, "game_a_s01", [{"set_start_frame": 2}])
    return tmp_path


# ── sources_fingerprint ───────────────────────────────────────────────────────
def test_fingerprint_keys_are_clips(pbp_dir):
    assert set(sources_fingerprint(pbp_dir)) == {"game_a_s00", "game_a_s01"}


def test_fingerprint_tracks_content_not_mtime(pbp_dir):
    before = sources_fingerprint(pbp_dir)
    # rewrite identical bytes → identical fingerprint (mtime is irrelevant)
    write_outcomes(pbp_dir, "game_a_s00", [{"set_start_frame": 1}])
    assert sources_fingerprint(pbp_dir) == before
    write_outcomes(pbp_dir, "game_a_s00", [{"set_start_frame": 99}])
    after = sources_fingerprint(pbp_dir)
    assert after["game_a_s00"] != before["game_a_s00"]
    assert after["game_a_s01"] == before["game_a_s01"]


# ── check_join_fresh ──────────────────────────────────────────────────────────
def test_fresh_join_passes(pbp_dir):
    join = {"sources": sources_fingerprint(pbp_dir)}
    assert check_join_fresh(join, pbp_dir) == []


def test_new_outcomes_file_flags_stale(pbp_dir):
    """The run-10 shape: aligned outcomes landed AFTER the join was built."""
    join = {"sources": sources_fingerprint(pbp_dir)}
    write_outcomes(pbp_dir, "game_b_s00", [{"set_start_frame": 3}])
    problems = check_join_fresh(join, pbp_dir)
    assert problems and "game_b_s00" in problems[0]


def test_changed_outcomes_file_flags_stale(pbp_dir):
    join = {"sources": sources_fingerprint(pbp_dir)}
    write_outcomes(pbp_dir, "game_a_s01", [{"set_start_frame": 2, "status": "aligned"}])
    problems = check_join_fresh(join, pbp_dir)
    assert any("game_a_s01" in p and "changed" in p for p in problems)


def test_missing_outcomes_file_flags_stale(pbp_dir):
    join = {"sources": sources_fingerprint(pbp_dir)}
    (pbp_dir / "game_a_s01_outcomes.json").unlink()
    problems = check_join_fresh(join, pbp_dir)
    assert any("game_a_s01" in p and "missing" in p for p in problems)


def test_join_without_fingerprint_flags_stale(pbp_dir):
    """Pre-guard joins (no 'sources' key) must be rejected, not trusted."""
    assert check_join_fresh({"checks": {}}, pbp_dir) != []


# ── the n>=MIN_BUCKET gate ────────────────────────────────────────────────────
def bucket(n):
    return {"n": n, "pts": n, "exp": float(n), "games": {"g1", "g2"}}


def test_gate_lifts_exactly_at_min_bucket():
    rows = credit_rows({"GSW": bucket(MIN_BUCKET - 1)})
    assert rows[0]["meaningful"] is False
    rows = credit_rows({"GSW": bucket(MIN_BUCKET)})
    assert rows[0]["meaningful"] is True


def test_gate_is_per_bucket_not_global():
    rows = credit_rows({"GSW": bucket(MIN_BUCKET), "SAC": bucket(17)})
    by = {r["defense"]: r["meaningful"] for r in rows}
    assert by == {"GSW": True, "SAC": False}
    assert [r["defense"] for r in rows] == ["GSW", "SAC"]  # sorted by n desc

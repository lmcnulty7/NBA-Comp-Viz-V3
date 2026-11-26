"""tools.p8_truth_cpu_r23: the r23 detector guard (section 16 version 4.1 of docs/research/P8_truth_arenas.md; check C's
composition moved to tools.p8_resetfix compose-checkc). No video, no model."""

import pytest

from tools import p8_truth_cpu_r23 as L


def test_the_repository_detector_is_r23(tmp_path):
    assert L.R23_DETECTOR_SHA256.startswith("4c886d5f")
    assert L.detector_sha_ok()
    bad = tmp_path / "player_detector.pt"
    bad.write_bytes(b"not the weights")
    assert not L.detector_sha_ok(bad)
    assert not L.detector_sha_ok(tmp_path / "missing.pt")


def test_detector_check_exit_code():
    with pytest.raises(SystemExit) as e:
        L.main(["detector-check"])
    assert e.value.code == 0


def test_no_command_writes_builds_r23():
    assert not hasattr(L, "compose") and not hasattr(L, "link")

"""fps-awareness of the harvest pipeline: build stride (run-8 postmortem) and
pre-gate sampling (run-9 postmortem) must hold the same TEMPORAL rate at any
fps, and fps probing must fail safe."""
import pytest

import config
from harvest_driver import build_stride, video_fps


@pytest.mark.parametrize("fps,expected", [
    (29.97, 3), (30.0, 3),          # the validated baseline rate
    (59.94, 6), (60.0, 6),          # the 720p60 game that burned runs 8-9
    (24.0, 3), (25.0, 3),           # never below the validated stride 3
    (50.0, 5), (120.0, 12),
])
def test_build_stride_holds_10hz(fps, expected):
    assert build_stride(fps) == expected


def test_video_fps_reads_real_file(video_60fps):
    assert video_fps(video_60fps) == pytest.approx(60.0, abs=0.2)


def test_video_fps_missing_file_falls_back_to_30(tmp_path):
    # fail-safe contract: unprobeable video → 30fps assumption → stride 3
    assert video_fps(tmp_path / "nope.mp4") == 30.0


def test_pregate_params_match_validated_30fps_values():
    # 15/45 at 30fps is the accuracy-verified config (DEVLOG 07-08c) — the
    # time-based form must reproduce it exactly
    assert config.pregate_params(30.0) == (15, 45)


@pytest.mark.parametrize("fps", [59.94, 60.0])
def test_pregate_params_scale_with_fps(fps):
    assert config.pregate_params(fps) == (30, 90)


def test_pregate_params_never_zero():
    stride, pad = config.pregate_params(1.0)
    assert stride >= 1 and pad >= 1

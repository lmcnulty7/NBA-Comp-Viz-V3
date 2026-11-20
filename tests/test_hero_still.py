"""tools/hero_still.py (PLAN.md D116): the SSIM it measures with and the rule it chooses by, fixed before measuring."""

import numpy as np
import pytest

from tools import hero_still as H


def test_the_rule_is_the_one_fixed_in_advance():
    assert H.SSIM_FLOOR == 0.97
    assert H.WEBP_MARGIN == 0.15
    assert H.FFMPEG_Q[0] == 3 and H.FFMPEG_Q[-1] == 12
    assert H.PILLOW_Q[0] == 40 and H.PILLOW_Q[-1] == 90


def test_ssim_is_one_on_the_same_picture_and_falls_with_noise():
    rng = np.random.default_rng(0)
    picture = rng.uniform(0, 255, (72, 128))
    assert H.ssim(picture, picture) == pytest.approx(1.0)
    small = H.ssim(picture, np.clip(picture + rng.normal(0, 2, picture.shape), 0, 255))
    large = H.ssim(picture, np.clip(picture + rng.normal(0, 20, picture.shape), 0, 255))
    assert 1.0 > small > large


def test_ssim_refuses_pictures_of_two_sizes():
    with pytest.raises(ValueError):
        H.ssim(np.zeros((10, 10)), np.zeros((10, 12)))


def row(kind, quality, total, low):
    return {"kind": kind, "quality": quality, "total_bytes": total, "min_ssim_vs_committed": low}


def test_the_smallest_jpeg_over_the_floor_is_chosen():
    rows = [row("ffmpeg-jpeg", 3, 700, 0.99), row("ffmpeg-jpeg", 9, 300, 0.96), row("pillow-jpeg", 70, 400, 0.975)]
    assert H.choose(rows)["choice"]["quality"] == 70


def test_webp_needs_the_margin_over_the_best_jpeg():
    jpeg = row("pillow-jpeg", 70, 400, 0.98)
    assert H.choose([jpeg, row("pillow-webp", 70, 350, 0.98)])["choice"]["kind"] == "pillow-jpeg"
    assert H.choose([jpeg, row("pillow-webp", 70, 340, 0.98)])["choice"]["kind"] == "pillow-webp"
    assert H.choose([jpeg, row("pillow-webp", 40, 100, 0.90)])["choice"]["kind"] == "pillow-jpeg"


def test_nothing_is_chosen_when_no_candidate_clears_the_floor():
    assert H.choose([row("ffmpeg-jpeg", 12, 100, 0.9)])["choice"] is None

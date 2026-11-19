"""xfg-E2 (docs/research/xfg.md): the three release-frame features on a tiny synthetic game, and the doors."""

import math

import numpy as np
import pandas as pd
import pytest

from tools import research_xfg2 as X


def _frames():
    """Frames 0..20 of one period: shooter 1 (team 10) at (20, 25) still; defender 2 (team 20) moving from (24, 31)
    to (24, 26) over the last 10 frames; defender 3 (team 20) at (28, 25); teammate 4 (team 10) far away."""
    rows = []
    for fr in range(21):
        t = max(0, fr - 10) / 10
        for pid, team, x, y in ((1, 10, 20.0, 25.0), (2, 20, 24.0, 31.0 - 5.0 * t), (3, 20, 28.0, 25.0),
                                (4, 10, 60.0, 10.0)):
            rows.append({"frame": fr, "period": 1, "game_clock_s": 700 - fr / 25, "shot_clock_s": 20.0,
                         "ball_x": 20.0, "ball_y": 25.0, "ball_z": 5.0, "team_id": team, "player_id": pid,
                         "x": x, "y": y})
    return pd.DataFrame(rows)


def test_features_of_a_known_geometry():
    from analytics.xfg.features import GameFrames
    G = GameFrames(_frames())
    rim = (5.25, 25.0)                                     # the shooter attacks the left rim
    out = X.row_features(G, 20, 1, {(10, 1): rim})
    # at release defender 2 is at (24, 26): distance sqrt(17) = 4.12; defender 3 at 8 ft is the second
    assert out["second_defender_dist_ft"] == pytest.approx(8.0)
    # the rim is behind the shooter (x 5.25 < 20) and defender 2 is in front of him: angle near 180 - atan(1/4)
    assert out["contest_angle_deg"] == pytest.approx(180 - math.degrees(math.atan2(1, 4)), abs=1e-6)
    # the line shooter-rim is y = 25; the defender moved from 6 ft off it to 1 ft off it in 0.4 s: 12.5 ft/s toward
    assert out["defender_speed_to_line_ftps"] == pytest.approx(12.5)


def test_a_defender_on_the_line_to_the_rim_is_angle_zero_and_the_nearer_rim_is_the_fallback():
    assert X.angle_deg(20, 25, 5.25, 25, 15, 25) == pytest.approx(0.0)
    assert X.angle_deg(20, 25, 5.25, 25, 20, 30) == pytest.approx(90.0)
    assert X.nearer_rim(30.0) == (5.25, 25.0) and X.nearer_rim(60.0) == (88.75, 25.0)


def test_features_are_nan_when_not_measurable():
    from analytics.xfg.features import GameFrames
    G = GameFrames(_frames())
    assert all(math.isnan(v) for v in X.row_features(G, 999, 1, {}).values())      # no such frame
    out = X.row_features(G, 5, 1, {})                                              # 0.4 s window too early
    assert math.isnan(out["defender_speed_to_line_ftps"]) and not math.isnan(out["second_defender_dist_ft"])


def test_the_new_arm_adds_exactly_the_three_columns():
    arms = X.arm_columns()
    assert arms["new"][len(arms["baseline"]):] == X.NEW_FEATURES
    assert arms["ablation"][len(arms["baseline"]):] == ["second_defender_dist_ft"]
    assert X.THRESHOLD == -0.001


def test_the_module_has_no_test_set_scoring_yet():
    text = (X.__file__ and open(X.__file__, encoding="utf-8").read())
    assert "score_once(" not in text

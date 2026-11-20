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


DOOR_NAMES = ("research_data_root", "test_features")


def _tree():
    import ast
    return ast.parse(open(X.__file__, encoding="utf-8").read())


def test_test_data_is_reached_only_inside_the_scorer_given_to_score_once():
    import ast
    tree = _tree()
    once = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None)) == "score_once"]
    assert len(once) == 1 and [ast.unparse(a) for a in once[0].args] == ["EXPERIMENT", "TRACKING", "scorer"]
    scorer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "scorer")
    inside = {id(n) for n in ast.walk(scorer)}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                assert not (a.name in DOOR_NAMES and a.asname), "aliased import"
                if a.name in DOOR_NAMES:
                    seen.add(a.name)
                    continue
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in DOOR_NAMES:
            seen.add(name)
            assert id(node) in inside, f"{ast.unparse(node)} outside the scorer"
        if isinstance(node, ast.Constant) and node.value in DOOR_NAMES:
            raise AssertionError("named as a string")
    assert seen == set(DOOR_NAMES)
    assert "_SCORING" in ast.unparse(scorer)
    assert X.EXPERIMENT == "xfg-E2"


def test_final_checks_a_clean_tree_first():
    import ast
    final = next(n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "final")
    assert "clean_head()" in ast.unparse(final.body[1]) or "clean_head()" in ast.unparse(final.body[0])


@pytest.mark.parametrize("lo,hi,want", [(-0.004, -0.0001, "confirms"), (-0.003, 0.002, "inconclusive at 10 games"),
                                        (0.0001, 0.003, "contradiction"), (-0.001, 0.0, "inconclusive at 10 games")])
def test_the_d121_readings(lo, hi, want):
    assert X.reading(lo, hi) == want


def test_the_scorer_refuses_outside_score_once():
    with pytest.raises(SystemExit, match="only inside the scorer"):
        X.make_scorer({}, None)(["g"])


def _pair():
    rows = pd.DataFrame({"sportvu_game": ["g1", "g1", "g2"], "event": [1, 2, 1],
                         "player_id": pd.array([7, pd.NA, 9], dtype="Int64"),
                         "second_defender_dist_ft": [5.0, np.nan, 8.0], "contest_angle_deg": [10.0, np.nan, 90.0],
                         "defender_speed_to_line_ftps": [1.0, np.nan, np.nan]})
    return rows, rows.copy()


def test_the_rehearsal_check_passes_identical_rows_with_a_missing_shooter_id():
    built, cached = _pair()
    out = X.rehearsal_check(built, cached)
    assert out["rows_match_cache"] and all(out["new_features_match_cache"].values())


def test_the_rehearsal_check_fails_a_changed_value_and_a_moved_nan():
    built, cached = _pair()
    built.loc[0, "contest_angle_deg"] = 10.0 + 1e-12                 # no tolerance
    out = X.rehearsal_check(built, cached)
    assert out["rows_match_cache"] and out["new_features_match_cache"]["contest_angle_deg"] is False
    built, cached = _pair()
    built.loc[2, "defender_speed_to_line_ftps"] = 3.0                # a NaN replaced by a value
    assert X.rehearsal_check(built, cached)["new_features_match_cache"]["defender_speed_to_line_ftps"] is False


def test_the_rehearsal_check_fails_a_reordered_row():
    built, cached = _pair()
    built = built.iloc[[1, 0, 2]].reset_index(drop=True)
    out = X.rehearsal_check(built, cached)
    assert out["rows_match_cache"] is False and not any(out["new_features_match_cache"].values())

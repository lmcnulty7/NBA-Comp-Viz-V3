"""tools.p8_e1_m4: the sweep, threshold filtering, per-frame counts and the operating-point rule on synthetic data."""

import numpy as np
import pytest

from tools import p8_e1_m4 as M
from tools import p8_e1_postmortem as P
from tools import p8_teachers as T

H = np.diag([0.1, 0.1, 1.0])          # px -> ft, 10 px a foot


def test_sweep_and_settings_are_the_registered_ones():
    import config
    assert M.SWEEP == (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    assert M.FLOOR == min(M.SWEEP) and M.OPERATING in M.SWEEP
    assert (M.OPERATING, M.NMS_IOU) == (config.PLAYER_CONF, config.PLAYER_IOU)
    assert M.key_of(0.4) == "0.40"


def test_keep_at_is_inclusive():
    boxes = [[0, 0, 1, 1, 0.1], [0, 0, 1, 1, 0.4], [0, 0, 1, 1, 0.39999]]
    assert M.keep_at(boxes, 0.40) == [boxes[1]]
    assert len(M.keep_at(boxes, 0.10)) == 3


def test_frame_counts_fall_with_the_threshold():
    sv = np.array([[10.0, 5.0], [30.0, 5.0]])
    inside = np.array([True, True])
    box = lambda x_ft, s: [x_ft * 10 - 10, 0.0, x_ft * 10 + 10, 50.0, s]
    fc = M.frame_counts({"e1": [box(10, 0.9), box(30, 0.3), box(60, 0.2)], "r23": []}, H, sv, inside,
                        T.feet_ft, P.oracle_counts)
    assert fc["e1"]["0.10"] == {"in_frame": 2, "boxes": 3, "missed": 0, "ghosts": 1, "duplicates": 0}
    assert fc["e1"]["0.40"] == {"in_frame": 2, "boxes": 1, "missed": 1, "ghosts": 0, "duplicates": 0}
    assert fc["r23"]["0.70"]["missed"] == 2


def test_operating_point():
    ref = {"missed_rate": 0.25, "ghost_rate": 0.15}
    curve = {"0.10": {"missed_rate": 0.2, "ghost_rate": 0.2}, "0.20": {"missed_rate": 0.25, "ghost_rate": 0.15},
             "0.30": {"missed_rate": 0.3, "ghost_rate": 0.1}}
    assert M.operating_point(curve, ref) == ["0.20"]


def test_summarise_pairs_and_compares_at_the_operating_point():
    c = lambda m, g: {"in_frame": 10, "missed": m, "boxes": 10, "ghosts": g, "duplicates": 0}
    per_w = {w: {"e1": {M.key_of(t): c(4, 1) for t in M.SWEEP},
                 "r23": {M.key_of(t): c(3 if t <= 0.4 else 5, 2) for t in M.SWEEP}} for w in ("a", "b")}
    s = M.summarise(per_w)
    assert s["e1_minus_r23_same_threshold"]["0.40"]["missed_rate"]["diff"] == pytest.approx(0.1)
    assert s["e1_minus_r23_same_threshold"]["0.50"]["missed_rate"]["diff"] == pytest.approx(-0.1)
    assert s["e1_minus_r23_at_0.40"]["0.50"]["missed_rate"]["diff"] == pytest.approx(0.1)
    assert s["e1_thresholds_at_or_below_r23_at_0.40"] == []


def test_games_are_guarded_and_nothing_trains():
    assert M.check_games(["gsw_cha_2016"]) == ["gsw_cha_2016"]
    for bad in ("hou_cle_2016", "gsw_ind_2016"):
        with pytest.raises(SystemExit):
            M.check_games([bad])
    src = open(M.__file__).read()
    assert "check_provenance(game, game, 0" in src and ".train(" not in src and ".fit(" not in src

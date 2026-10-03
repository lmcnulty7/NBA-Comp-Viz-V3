"""sportvu/bench.py pure pieces: possession segments, id switches, line error, acceptance."""
import numpy as np

from sportvu.bench import accepted, id_switches, line_error_px, possession_ids


def _h_px2ft(scale=10.0, ox=150.0, oy=100.0):
    """A synthetic top-down camera: px = ft * scale + offset, returned as px -> ft."""
    ft2px = np.array([[scale, 0, ox], [0, scale, oy], [0, 0, 1]], np.float64)
    return np.linalg.inv(ft2px)


def test_possession_ids_split_on_shot_clock_reset():
    shot = [24.0, 20.0, 15.0, np.nan, 10.0, 24.0, 23.0, 14.0, 13.5]
    seg = possession_ids(shot)
    assert list(seg) == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_possession_ids_ignore_small_upticks():
    assert list(possession_ids([10.0, 10.5, 9.0])) == [0, 0, 0]


def test_id_switches():
    assert id_switches([3, 3, 3]) == 0
    assert id_switches([3, 7, 7, 3]) == 2
    assert id_switches([]) == 0


def test_line_error_zero_for_identical_h():
    H = _h_px2ft()
    assert line_error_px(H, H, 1280, 720) <= 0.5


def test_line_error_reads_a_diagonal_shift():
    Ht = _h_px2ft()
    Hb = _h_px2ft(ox=155.0, oy=105.0)        # template 5 px right and 5 px down
    e = line_error_px(Hb, Ht, 1280, 720)
    assert 4.0 <= e <= 6.0


def test_line_error_undefined_off_frame():
    Ht = _h_px2ft()
    Hb = _h_px2ft(ox=5000.0)                  # build template entirely off the frame
    assert line_error_px(Hb, Ht, 1280, 720) is None


def test_accepted_defaults_to_having_an_h():
    assert accepted({"H": [1] * 9})
    assert not accepted({"H": None})
    assert not accepted({"H": [1] * 9, "accepted": False})


def test_line_verdict_withheld_while_truth_is_coarse():
    from sportvu.bench import line_verdict
    assert line_verdict(0.0, 6.04)[0] is None and "reported" in line_verdict(0.0, 6.04)[1]
    assert line_verdict(0.97, 1.2) == (True, None)
    assert line_verdict(0.50, 1.5) == (False, None)
    assert line_verdict(None, 1.0) == (None, None)

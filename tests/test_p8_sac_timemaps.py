"""tools.p8_sac_timemaps: the content comparison."""

from tools import p8_sac_timemaps as T


def test_compare_timemaps():
    a = {"spans": [1], "frames": [[0, 1, 700.0], [6, 1, 699.9]], "ocr_seconds": 24.6}
    b = {"spans": [1], "frames": [[0, 1, 700.0], [6, 1, 699.9]], "ocr_seconds": 21.7}
    c = T.compare_timemaps(a, b)
    assert c["keys_differing"] == ["ocr_seconds"] and c["frames_equal_sets"] and c["spans_equal"]
    assert c["clock_abs_diff_max"] == 0.0 and c["period_differences"] == 0
    b2 = {"spans": [2], "frames": [[0, 2, 700.5], [12, 1, 699.0]], "ocr_seconds": 24.6}
    c = T.compare_timemaps(a, b2)
    assert c["keys_differing"] == ["frames", "spans"] and not c["frames_equal_sets"] and c["frames_common"] == 1
    assert abs(c["clock_abs_diff_max"] - 0.5) < 1e-12 and c["period_differences"] == 1

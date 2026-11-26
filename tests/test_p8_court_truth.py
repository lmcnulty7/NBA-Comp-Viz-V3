"""tools.p8_court_truth on synthetic homographies and boxes: the point gap, the frame-alignment check, the paired
bootstrap and its reading, the frame sets, and the design's summary (docs/research/P8_court_truth.md). No video."""

import json

import numpy as np
import pytest

from tools import p8_court_truth as C

I = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def shift(px):
    """An image -> court H whose inverse moves every court point by +px in x (court = image - px)."""
    return [1, 0, -px, 0, 1, 0, 0, 0, 1]


def test_point_gap():
    assert C.point_gap(I, I) == 0.0
    assert C.point_gap(shift(2.0), I) == pytest.approx(2.0)
    assert C.point_gap(None, I) is None and C.point_gap(I, None) is None
    assert C.point_gap([0] * 9, I) is None                             # singular


def _boxes(f, dx=0):
    return {"frame": f, "boxes": [{"bbox": [100 + 10 * f + dx, 50, 140 + 10 * f + dx, 150]},
                                   {"bbox": [600 + 3 * f + dx, 80, 650 + 3 * f + dx, 200]}]}


def test_alignment_check():
    build = {f: _boxes(f) for f in range(0, 120, 6)}
    truth = {f: _boxes(f) for f in (24, 30, 60, 90)}
    a = C.alignment(truth, build, 6)
    assert a["aligned"] and a["shares"]["0"] == 1.0 and a["shares"]["6"] < 0.9
    shifted = {f + 6: _boxes(f) for f in (24, 30, 60, 90)}             # truth indices one build step late
    assert not C.alignment(shifted, build, 6)["aligned"]


def test_reading_thresholds():
    assert C.reading(-1.0, [-1.5, -0.2], "a", "b") == "a closer"
    assert C.reading(-0.99, [-1.5, -0.2], "a", "b") == "no difference resolved"      # under 1.0 px
    assert C.reading(-2.0, [-3.0, 0.1], "a", "b") == "no difference resolved"        # interval reaches zero
    assert C.reading(1.5, [0.5, 2.5], "a", "b") == "b closer"


def test_pair_row_and_bootstraps():
    a = {f: 3.0 + 0.01 * f for f in range(40)}
    b = {f: 6.0 + 0.01 * f for f in range(40)}
    b[5] = None                                                       # a frame without H in b
    r = C.pair_row(a, b, "cuda_a", "mac_mps")
    assert r["frames"] == 39 and r["median_diff_px"] == pytest.approx(-3.0) and r["reading"] == "cuda_a closer"
    assert r["frame_ci"][1] < 0 and len(r["block_ci"]) == 2
    assert C.pair_row({0: 1.0}, {0: 2.0}, "x", "y")["reading"] == "not enough frames"


def test_measure_uses_only_the_frame_set_and_counts_missing_h():
    truth = {f: {"frame": f, "H_truth": I} for f in range(10)}
    builds = {"mac_mps": {f: {"H": shift(4.0)} for f in range(10)},
              "cuda_a": {f: {"H": shift(9.0) if f != 3 else None} for f in range(10)}}
    m = C.measure(truth, builds, [0, 1, 2, 3, 4], lambda hb, ht, w, h: 1.5)
    assert m["frames"] == 5
    assert m["builds"]["mac_mps"]["gap_median_px"] == pytest.approx(4.0)
    assert m["builds"]["cuda_a"]["missing_h"] == 1 and m["builds"]["cuda_a"]["frames"] == 4
    assert m["builds"]["cuda_a"]["line_error_median_px"] == 1.5
    assert m["pairs"]["cuda_a~mac_mps"]["reading"] == "mac_mps closer"


def _win(readings: dict, measured=True):
    return {"measured": measured, "primary": {"pairs": {k: {"reading": v} for k, v in readings.items()}}}


def test_summary_rules():
    cpu_wins = {C.W03: _win({"mac_cpu~mac_mps": "mac_cpu closer", "cuda_a~mac_mps": "cuda_a closer"}),
                C.W04: _win({"mac_cpu~mac_mps": "mac_cpu closer", "cuda_a~mac_mps": "cuda_a closer"}),
                C.W00: _win({"cuda_a~mac_mps": "mac_mps closer"})}
    s = C.summary(cpu_wins)
    assert s["court_device"].startswith("CPU court closer on both") and s["bias_signal"] is True
    mixed = {**cpu_wins, C.W04: _win({"mac_cpu~mac_mps": "no difference resolved", "cuda_a~mac_mps": "mac_mps closer"})}
    s = C.summary(mixed)
    assert s["court_device"].startswith("no change") and s["bias_signal"] is False
    one = {**cpu_wins, C.W04: {"measured": False}}
    assert C.summary(one)["court_device"] == "not enough aligned gsw_ind_2016 windows"
    mps = {C.W03: _win({"mac_cpu~mac_mps": "mac_mps closer"}), C.W04: _win({"mac_cpu~mac_mps": "mac_mps closer"})}
    assert C.summary(mps)["court_device"].startswith("MPS court closer on both")


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_window_result_end_to_end(tmp_path):
    w = C.W03
    truth_rows = [{"window": w, "frame": f, "H_truth": I, **_boxes(f)} for f in range(0, 60, 6)]
    _write(tmp_path / "truth.jsonl", "\n".join(json.dumps(r) for r in truth_rows))
    _write(tmp_path / "paint.jsonl", "\n".join(json.dumps({"window": w, "frame": r["frame"], "keep": r["frame"] != 12})
                                              for r in truth_rows))
    for name, px in (("mac_cpu", 2.0), ("mac_mps", 5.0), ("cuda_a", 9.0)):
        _write(tmp_path / name / f"{w}_frames.json",
               {"frames": [{**_boxes(f), "H": shift(px + 0.01 * f)} for f in range(0, 60, 6)]})
    spec = {"game": C.IND, "truth": tmp_path / "truth.jsonl", "paint": tmp_path / "paint.jsonl", "align": True,
            "builds": {n: tmp_path / n for n in ("mac_cpu", "mac_mps", "cuda_a")}}
    r = C.window_result(w, spec, lambda hb, ht, a, b: 1.0)
    assert r["alignment"]["aligned"] and r["measured"] and r["paint_kept"] == 9
    assert r["primary"]["frames"] == 9 and r["all_truth_frames"]["frames"] == 10
    assert r["primary"]["pairs"]["mac_cpu~mac_mps"]["reading"] == "mac_cpu closer"
    assert r["primary"]["builds"]["cuda_a"]["gap_median_px"] == pytest.approx(9.0 + 0.01 * 30, abs=0.1)
    from export.schemas import find_private
    assert find_private(json.dumps(r)) == []

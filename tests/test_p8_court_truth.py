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


def _win(readings: dict, measured=True, ready=True):
    return {"measured": measured,
            "primary": {"pairs": {k: {"reading": v, "d_line_ready": ready} for k, v in readings.items()}}}


def test_summary_rules_version_2():
    lean = {C.W03: _win({"mac_cpu~mac_mps": "mac_cpu closer", "cuda_a~mac_mps": "cuda_a closer",
                         "cuda_a~mac_cpu": "cuda_a closer"}),
            C.W04: _win({"mac_cpu~mac_mps": "mac_cpu closer", "cuda_a~mac_mps": "cuda_a closer",
                         "cuda_a~mac_cpu": "cuda_a closer"}),
            C.W00: _win({"cuda_a~mac_mps": "mac_mps closer"})}
    s = C.summary(lean)
    assert "propose a confirmation measurement" in s["court_device"] and "not a device change" in s["court_device"]
    assert s["bias_signal"].startswith("consistent with a lean")
    assert s["cuda"].startswith("CUDA closer on the gsw_ind_2016 windows only")       # not on s04_w00
    assert "not more accurate" in s["meaning"]
    # the block interval or the line error disagreeing stops a D-line outcome, not the reading
    weak = {w: _win({k: v["reading"] for k, v in lean[w]["primary"]["pairs"].items()}, ready=False) for w in lean}
    s = C.summary(weak)
    assert s["court_device"].startswith("no change") and s["bias_signal"].startswith("consistent with a lean")
    cuda_all = {**lean, C.W00: _win({"cuda_a~mac_mps": "cuda_a closer"})}
    assert C.summary(cuda_all)["cuda"].startswith("CUDA court closer to this truth on s04_w00 and both")
    one = {**lean, C.W04: {"measured": False}}
    s = C.summary(one)
    assert s["court_device"] == "not enough aligned gsw_ind_2016 windows"
    assert s["bias_signal"].startswith("does not show a lean")                          # needs both gsw_ind windows
    mps = {C.W03: _win({"mac_cpu~mac_mps": "mac_mps closer"}), C.W04: _win({"mac_cpu~mac_mps": "mac_mps closer"})}
    assert C.summary(mps)["court_device"].startswith("MPS court closer to this truth on both")


def test_on_frame_points():
    assert C.on_frame_points(shift(0.0)) == 3                         # (47,25), (20,10), (70,40) px all inside
    assert C.on_frame_points(shift(-1270.0)) == 0
    assert C.on_frame_points(None) is None


def test_pair_row_blocks_between_gap_and_line_error():
    n = 25
    a = {f: 3.0 for f in range(n)}
    b = {f: 6.0 for f in range(n)}
    between = {f: 3.2 for f in range(n)}
    la, lb = {f: 1.0 for f in range(n)}, {f: 2.0 for f in range(n)}
    r = C.pair_row(a, b, "cuda_a", "mac_mps", between, la, lb)
    assert r["blocks"] == 3 and r["median_build_to_build_gap_px"] == pytest.approx(3.2)
    assert r["line_error"]["median_diff_px"] == pytest.approx(-1.0) and r["line_error_sign_agrees"] is True
    assert r["d_line_ready"] is True
    r = C.pair_row(a, b, "cuda_a", "mac_mps", between, {f: 3.0 for f in range(n)}, lb)    # line error disagrees
    assert r["reading"] == "cuda_a closer" and r["line_error_sign_agrees"] is False and not r["d_line_ready"]
    assert C.side_of([-2.0, -0.1]) == -1 and C.side_of([0.1, 2.0]) == 1 and C.side_of([-1.0, 1.0]) == 0


def test_frame_size_is_asserted(tmp_path):
    with pytest.raises(SystemExit):
        C.window_result(C.W03, {}, lambda *a: 1.0, size=(1920, 1080))


def test_one_name_for_the_detector_cpu_build(tmp_path):
    sp = C.specs(tmp_path / "p8", tmp_path / "diag", tmp_path / "main", tmp_path / "cuda")
    names = {n for s in sp.values() for n in s["builds"]}
    assert "mac_detcpu" in names and "detector_cpu" not in names
    assert set(sp[C.W03]["builds"]) == {"mac_mps", "mac_cpu", "mac_detcpu", "cuda_a"}


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_window_result_end_to_end(tmp_path):
    w = C.W03
    truth_rows = [{"window": w, "frame": f, "H_truth": I, **_boxes(f)} for f in range(0, 60, 6)]
    _write(tmp_path / "truth.jsonl", "\n".join(json.dumps(r) for r in truth_rows))
    _write(tmp_path / "paint.jsonl", "\n".join(json.dumps({"window": w, "frame": r["frame"], "keep": r["frame"] != 12,
                                                           "edge_px": 4.0 if r["frame"] != 12 else 30.0})
                                              for r in truth_rows))
    for name, px in (("mac_cpu", 2.0), ("mac_mps", 5.0), ("cuda_a", 9.0)):
        _write(tmp_path / name / f"{w}_frames.json",
               {"frames": [{**_boxes(f), "H": shift(px + 0.01 * f)} for f in range(0, 60, 6)]})
    spec = {"game": C.IND, "truth": tmp_path / "truth.jsonl", "paint": tmp_path / "paint.jsonl", "align": True,
            "builds": {n: tmp_path / n for n in ("mac_cpu", "mac_mps", "cuda_a")}}
    r = C.window_result(w, spec, lambda hb, ht, a, b: 1.0, size=(1280, 720))
    assert r["alignment"]["aligned"] and r["measured"] and r["paint_kept"] == 9
    assert r["primary_paint_edge_px_median"] == pytest.approx(4.0)
    assert sum(r["primary"]["truth_on_frame_points"].values()) == 9
    assert r["primary"]["frames"] == 9 and r["all_truth_frames"]["frames"] == 10
    assert r["primary"]["pairs"]["mac_cpu~mac_mps"]["reading"] == "mac_cpu closer"
    assert r["primary"]["builds"]["cuda_a"]["gap_median_px"] == pytest.approx(9.0 + 0.01 * 30, abs=0.1)
    from export.schemas import find_private
    assert find_private(json.dumps(r)) == []

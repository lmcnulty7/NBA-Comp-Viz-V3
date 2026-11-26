"""tools.p8_truth_moved: the 'moved' bars of docs/research/P8_truth_arenas.md section 0 on synthetic fits. No video,
no model."""

import json

import pytest

from tools import p8_truth_moved as M


def fit(px, acc, partial=False):
    return {"partial": partial, "camera": {"status": "ok", "reproj_px_median_after": px}, "accepted": acc}


OFF = {f"gsw_cha_2016_s01_w{k:02d}": 0.1 * k for k in range(5)}


def test_the_bars_are_the_registered_ones():
    assert M.BARS == {"reproj_px_abs_min": 1.0, "accepted_rel_min": 0.10, "offset_median_s_min": 0.2}


def test_not_moved_when_every_measure_is_under_its_bar():
    cpu_off = {w: o + 0.04 for w, o in OFF.items()}                      # the D99 detector-only size of shift
    r = M.compare(fit(7.65, 1366), fit(7.0, 1450), OFF, cpu_off)
    assert not r["moved"] and r["measures"]["offsets"]["median_abs_diff_s"] == pytest.approx(0.04)
    assert r["measures"]["accepted"]["rel_diff"] == pytest.approx(84 / 1366)


def test_each_measure_alone_can_move_it():
    assert M.compare(fit(7.65, 1366), fit(6.65, 1366), OFF, OFF)["measures"]["reprojection"]["moved"]   # 1.0 px
    assert not M.compare(fit(7.65, 1366), fit(6.66, 1366), OFF, OFF)["moved"]                            # 0.99 px
    r = M.compare(fit(7.0, 1000), fit(7.0, 1100), OFF, OFF)                                              # +10 percent
    assert r["moved"] and r["measures"]["accepted"]["moved"]
    assert not M.compare(fit(7.0, 1000), fit(7.0, 1099), OFF, OFF)["moved"]
    shifted = {w: o + 0.2 for w, o in OFF.items()}
    r = M.compare(fit(7.0, 1000), fit(7.0, 1000), OFF, shifted)
    assert r["moved"] and r["measures"]["offsets"]["moved"]


def test_offsets_compare_only_windows_with_an_offset_in_both():
    cpu = {**{w: o for w, o in OFF.items()}, "gsw_cha_2016_s09_w00": 1.0}
    cpu["gsw_cha_2016_s01_w00"] = None
    d = M.offset_diffs(OFF, cpu)
    assert len(d) == 4 and all(x == pytest.approx(0) for x in d)
    with pytest.raises(SystemExit):
        M.compare(fit(7.0, 1000), fit(7.0, 1000), {"a": None}, {"a": 0.3})


def test_incomplete_or_failed_fits_are_refused():
    with pytest.raises(SystemExit):
        M.compare(fit(7.0, 1000, partial=True), fit(7.0, 1000), OFF, OFF)
    with pytest.raises(SystemExit):
        M.compare(fit(7.0, 1000), {**fit(7.0, 1000), "partial": None}, OFF, OFF)
    with pytest.raises(SystemExit):
        M.compare(fit(7.0, 1000), {**fit(7.0, 1000), "status": "camera_fit_failed"}, OFF, OFF)


def _root(tmp_path, name, result, offsets, device=None):
    root = tmp_path / name
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "gsw_cha_2016_result.json").write_text(json.dumps(result))
    (root / "labels" / "gsw_cha_2016_offsets.json").write_text(json.dumps(offsets))
    (root / "label_build").mkdir()
    for w in offsets:
        (root / "label_build" / f"{w}_frames.json").write_text("{}")
        if device:
            (root / "label_build" / f"{w}_device.json").write_text(json.dumps({"resolved": device}))
    return root


def test_run_end_to_end_and_its_refusals(tmp_path):
    cpu_dev = {"detector": "cpu", "court": "mps", "gate": "mps"}
    mps_root = _root(tmp_path, "work", fit(7.65, 1366), OFF)
    cpu_root = _root(tmp_path, "work_ref", fit(7.1, 1420), {w: o + 0.05 for w, o in OFF.items()}, device=cpu_dev)
    out = M.run("gsw_cha_2016", mps_root, cpu_root)
    assert out["moved"] is False and out["cpu_label_windows"] == 5
    assert json.loads((cpu_root / "truth_moved" / "gsw_cha_2016.json").read_text()) == out
    from export.schemas import find_private
    assert find_private(json.dumps(out)) == []
    with pytest.raises(SystemExit):
        M.run("gsw_ind_2016", mps_root, cpu_root)                     # stays apart (D98)
    with pytest.raises(SystemExit):
        M.run("gsw_cha_2016", cpu_root, cpu_root)                     # the same root twice
    (cpu_root / "label_build" / "gsw_cha_2016_s01_w03_device.json").write_text(
        json.dumps({"resolved": {"detector": "mps", "court": "mps", "gate": "mps"}}))
    with pytest.raises(SystemExit):                                   # not all CPU
        M.run("gsw_cha_2016", mps_root, cpu_root)

"""tools.p8_truth_cpu_r23: which workspace r23 builds may stand for the reference root's r23 builds (check C of
docs/research/P8_truth_arenas.md section 16). Synthetic folders only; no video, no model."""

import json

import pytest

from tools import p8_truth_cpu_r23 as L
from tools.p8_placement import BUILDS

REC = {"requested": {"detector": "cpu", "court": "auto", "gate": "auto"},
       "resolved": {"detector": "cpu", "court": "mps", "gate": "mps"}}


def clip(path, data=b"clip"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_verify_accepts_only_the_reference_setting_on_identical_clips(tmp_path):
    a, b = clip(tmp_path / "a.mp4"), clip(tmp_path / "b.mp4")
    assert L.verify(a, b, REC) is None
    assert L.verify(a, b, None) == "no device record"
    assert L.verify(a, b, {**REC, "requested": {**REC["requested"], "detector": "mps"}}).startswith("devices")
    assert L.verify(a, b, {**REC, "requested": {"detector": "auto", "court": "auto", "gate": "auto"}}).startswith("devices")
    assert L.verify(a, b, {**REC, "clip_fix": True}) == "clip fix"
    assert L.verify(a, clip(tmp_path / "c.mp4", b"other"), REC) == "clips differ"
    assert L.verify(a, tmp_path / "missing.mp4", REC) == "clip missing"


def make_roots(tmp_path, game="gsw_cha_2016"):
    mps, ref = tmp_path / "work", tmp_path / "work_ref"
    src = mps / "builds" / BUILDS[game]
    src.mkdir(parents=True)
    stems = [f"{game}_s01_w00", f"{game}_s01_w01", f"{game}_s02_w00"]
    for s in stems:
        clip(mps / "label_build" / f"{s}.mp4", s.encode())
        clip(ref / "label_build" / f"{s}.mp4", s.encode())
        (src / f"{s}_frames.json").write_text(json.dumps({"frames": []}))
        (src / f"{s}_device.json").write_text(json.dumps(REC))
        (src / f"{s}_trajectories.json").write_text("{}")
    return mps, ref, src, stems


def test_link_copies_verified_windows_and_leaves_the_rest_for_the_rebuild(tmp_path):
    mps, ref, src, stems = make_roots(tmp_path)
    clip(ref / "label_build" / f"{stems[1]}.mp4", b"a different cut")                  # clip differs
    (src / f"{stems[2]}_device.json").write_text(json.dumps({**REC, "clip_fix": True}))  # other setting
    out = L.link("gsw_cha_2016", mps, ref)
    dst = ref / "builds" / "r23"
    assert out["copied"] == 1 and set(out["refused"]) == {stems[1], stems[2]}
    assert (dst / f"{stems[0]}_frames.json").exists() and (dst / f"{stems[0]}_device.json").exists()
    assert (dst / f"{stems[0]}_trajectories.json").exists()
    assert not (dst / f"{stems[1]}_frames.json").exists() and not (dst / f"{stems[2]}_frames.json").exists()
    assert json.loads((ref / "builds" / "r23_source_gsw_cha_2016.json").read_text())["copied"] == 1
    assert (src / f"{stems[0]}_frames.json").exists()                                  # the workspace is untouched
    again = L.link("gsw_cha_2016", mps, ref)
    assert again["copied"] == 0 and again["already_present"] == 1


def test_link_refuses_the_apart_game_a_cv_test_game_and_one_root(tmp_path):
    with pytest.raises(SystemExit, match="apart"):
        L.link("gsw_ind_2016", tmp_path / "a", tmp_path / "b")
    with pytest.raises(SystemExit, match="CV test"):
        L.link("tor_gsw_2016", tmp_path / "a", tmp_path / "b")
    with pytest.raises(SystemExit, match="same folder"):
        L.link("gsw_cha_2016", tmp_path, tmp_path)

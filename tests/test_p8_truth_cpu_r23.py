"""tools.p8_truth_cpu_r23: composing check C's r23 side from the reset-fix arm's builds (reset windows) and the
workspace's baseline builds (other windows), section 16 version 4 of docs/research/P8_truth_arenas.md. Synthetic
folders only; no video, no model."""

import json

import pytest

from tools import p8_truth_cpu_r23 as L
from tools.p8_placement import BUILDS

G = "gsw_cha_2016"
REC = {"requested": {"detector": "cpu", "court": "auto", "gate": "auto"},
       "resolved": {"detector": "cpu", "court": "mps", "gate": "mps"}, "clip_fix": None}
FIX = {**REC, "reset_fix": True}
DET = {"weights": "player_detector.pt", "sha256": L.R23_DETECTOR_SHA256, "model": "r23", "seed": 0}
LOG = {"reset_fix": True, "resets": [{"frame": 120, "stripped": 2}], "tracker_reid_env": None,
       "tracker_config": "botsort.yaml", "versions": {"opencv": "4.13.0", "torch": "2.2.2", "ultralytics": "8.4.48"}}


def clip(path, data=b"clip"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def build(folder, stem, source, device, detector=DET):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_frames.json").write_text(json.dumps({"source": str(source), "frames": []}))
    (folder / f"{stem}_device.json").write_text(json.dumps(device))
    (folder / f"{stem}_detector.json").write_text(json.dumps(detector))
    (folder / f"{stem}_trajectories.json").write_text("{}")
    (folder / f"{stem}_identity.json").write_text(json.dumps({"identity": {"camera_cut_resets": 1}}))
    if device.get("reset_fix"):
        (folder / f"{stem}_resetfix.json").write_text(json.dumps(LOG))


def make_roots(tmp_path, reset=(1,), passes=True):
    """Three windows; the ones whose index is in `reset` had a reset in the baseline and have an arm build."""
    mps, ref = tmp_path / "work", tmp_path / "work_ref"
    base, arm = mps / "builds" / BUILDS[G], ref / "builds" / f"resetfix_r23_detcpu__{G}"
    stems = [f"{G}_s01_w00", f"{G}_s01_w01", f"{G}_s02_w00"]
    for i, s in enumerate(stems):
        clip(mps / "label_build" / f"{s}.mp4", s.encode())
        clip(ref / "label_build" / f"{s}.mp4", s.encode())
        (ref / "label_build" / f"{s}_frames.json").write_text("{}")            # the fixed V3 label build
        (ref / "label_build" / f"{s}_device.json").write_text(json.dumps(FIX))
        build(base, s, mps / "label_build" / f"{s}.mp4", REC)
        if i in reset:
            build(arm, s, ref / "label_build" / f"{s}.mp4", FIX)
    rf = ref / "resetfix"
    rf.mkdir(parents=True)
    (rf / f"plan__{G}.json").write_text(json.dumps({"reset_windows": [stems[i] for i in reset],
                                                     "equivalence_windows": [stems[0]]}))
    (rf / f"equiv__{G}.json").write_text(json.dumps({"passes": passes, "failing_windows": []}))
    return mps, ref, base, arm, stems


def test_verify_needs_the_reference_setting_the_r23_detector_and_identical_clips(tmp_path):
    a, b = clip(tmp_path / "a.mp4"), clip(tmp_path / "b.mp4")
    assert L.verify(a, b, REC, DET) is None
    assert L.verify(a, b, FIX, DET, want_fix=True) is None
    assert L.verify(a, b, FIX, DET).startswith("reset_fix")                         # a fixed build as a baseline
    assert L.verify(a, b, REC, DET, want_fix=True).startswith("reset_fix")          # an unfixed build as an arm build
    assert L.verify(a, b, None, DET) == "no device record"
    assert L.verify(a, b, {**REC, "requested": {**REC["requested"], "detector": "mps"}}, DET).startswith("devices")
    assert L.verify(a, b, {**FIX, "resolved": {**FIX["resolved"], "detector": "mps"}}, DET, want_fix=True).startswith("resolved")
    assert L.verify(a, b, {**REC, "clip_fix": True}, DET) == "clip fix"
    assert L.verify(a, b, REC) == "no detector record"
    assert L.verify(a, b, REC, {**DET, "sha256": "0" * 64}).startswith("detector")
    assert L.verify(a, b, REC, {**DET, "model": "e1"}).startswith("detector")
    assert L.verify(a, clip(tmp_path / "c.mp4", b"other"), REC, DET) == "clips differ"
    assert L.verify(a, tmp_path / "missing.mp4", REC, DET) == "clip missing"
    assert L.R23_DETECTOR_SHA256.startswith("4c886d5f")


def test_the_repository_detector_is_r23(tmp_path):
    assert L.detector_sha_ok()
    assert not L.detector_sha_ok(clip(tmp_path / "player_detector.pt", b"not the weights"))
    assert not L.detector_sha_ok(tmp_path / "missing.pt")


def test_compose_takes_reset_windows_from_the_arm_and_the_rest_from_the_baseline(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    out = L.compose(G, mps, ref)
    dst = ref / "builds" / "r23"
    assert out["complete"] and out["copied_from_arm"] == 1 and out["copied_from_baseline"] == 2
    assert json.loads((dst / f"{stems[1]}_device.json").read_text())["reset_fix"] is True
    assert "reset_fix" not in json.loads((dst / f"{stems[0]}_device.json").read_text())
    assert (dst / f"{stems[0]}_detector.json").exists() and (dst / f"{stems[0]}_trajectories.json").exists()
    assert json.loads((ref / "builds" / f"r23_checkc_{G}.json").read_text())["complete"] is True
    assert (base / f"{stems[0]}_frames.json").exists() and (arm / f"{stems[1]}_frames.json").exists()   # sources kept
    again = L.compose(G, mps, ref)
    assert again["already_present"] == 3 and again["copied_from_arm"] == again["copied_from_baseline"] == 0


def test_compose_waits_for_a_passing_equivalence(tmp_path):
    mps, ref, *_ = make_roots(tmp_path, passes=False)
    with pytest.raises(SystemExit, match="did not pass"):
        L.compose(G, mps, ref)
    (ref / "resetfix" / f"equiv__{G}.json").unlink()
    with pytest.raises(SystemExit, match="no reset-fix arm"):
        L.compose(G, mps, ref)


def test_a_refused_window_leaves_check_c_incomplete(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    clip(ref / "label_build" / f"{stems[0]}.mp4", b"a different cut")                    # clip differs
    build(arm, stems[1], mps / "label_build" / f"{stems[1]}.mp4", FIX)                   # arm build names the wrong clip
    eq = ref / "resetfix" / f"equiv__{G}.json"
    eq.write_text(eq.read_text())                                                          # equivalence rerun after it
    import os
    t = (arm / f"{stems[1]}_frames.json").stat().st_mtime
    os.utime(eq, (t + 1, t + 1))
    out = L.compose(G, mps, ref)
    assert set(out["refused"]) == {stems[0], stems[1]} and out["refused"][stems[0]] == "clips differ"
    assert out["refused"][stems[1]].startswith("built from")
    assert out["complete"] is False and set(out["v3_windows_without_r23"]) == {stems[0], stems[1]}


def test_a_reset_window_without_an_arm_build_is_refused(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    for p in arm.glob(f"{stems[1]}_*"):
        p.unlink()
    out = L.compose(G, mps, ref)
    assert stems[1] in out["refused"] and not out["complete"]


def test_a_copy_already_present_is_checked_again(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    L.compose(G, mps, ref)
    dst = ref / "builds" / "r23"
    (dst / f"{stems[1]}_device.json").write_text(json.dumps(REC))                       # an unfixed copy on a reset window
    with pytest.raises(SystemExit, match="already in"):
        L.compose(G, mps, ref)


def test_a_build_present_without_one_on_record_stops_the_step(tmp_path):
    mps, ref, *_ = make_roots(tmp_path)
    dst = ref / "builds" / "r23"
    dst.mkdir(parents=True)
    (dst / f"{G}_s09_w09_frames.json").write_text("{}")
    with pytest.raises(SystemExit, match="no build on record"):
        L.compose(G, mps, ref)


def test_compose_refuses_the_apart_game_a_cv_test_game_and_one_root(tmp_path):
    with pytest.raises(SystemExit, match="apart"):
        L.compose("gsw_ind_2016", tmp_path / "a", tmp_path / "b")
    with pytest.raises(SystemExit, match="CV test"):
        L.compose("tor_gsw_2016", tmp_path / "a", tmp_path / "b")
    with pytest.raises(SystemExit, match="same folder"):
        L.compose(G, tmp_path, tmp_path)


def test_arm_windows_outside_the_reset_list_are_ignored(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    build(arm, stems[0], ref / "label_build" / f"{stems[0]}.mp4", FIX)                  # an equivalence or smoke window
    out = L.compose(G, mps, ref)
    assert out["copied_from_arm"] == 1 and out["copied_from_baseline"] == 2 and out["complete"]
    assert "reset_fix" not in json.loads((ref / "builds" / "r23" / f"{stems[0]}_device.json").read_text())


def test_an_unfixed_copy_left_on_a_reset_window_stops_the_step_until_supersede_moves_it(tmp_path):
    from tools import p8_truth_fixprep as F
    mps, ref, base, arm, stems = make_roots(tmp_path)
    old = ref / "builds" / "r23"
    build(old, stems[1], mps / "label_build" / f"{stems[1]}.mp4", REC)                   # version 3's unfixed copy
    with pytest.raises(SystemExit, match="already in"):
        L.compose(G, mps, ref)
    F.supersede(G, ref)
    assert (ref / "superseded_nofix" / "builds" / "r23" / f"{stems[1]}_device.json").exists()
    assert (ref / "label_build" / f"{stems[1]}_frames.json").exists()                  # fixed label builds stay
    out = L.compose(G, mps, ref)
    assert out["complete"] and out["v3_label_windows"] == 3 and out["copied_from_arm"] == 1


def test_a_stale_equivalence_record_stops_the_step(tmp_path):
    import os
    mps, ref, base, arm, stems = make_roots(tmp_path)
    eq = ref / "resetfix" / f"equiv__{G}.json"
    t = (arm / f"{stems[1]}_frames.json").stat().st_mtime
    os.utime(eq, (t - 60, t - 60))                                                      # equivalence older than the build
    with pytest.raises(SystemExit, match="older than the arm build"):
        L.compose(G, mps, ref)


def test_a_present_file_that_differs_from_its_source_stops_the_step(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    L.compose(G, mps, ref)
    (ref / "builds" / "r23" / f"{stems[0]}_trajectories.json").write_text('{"changed": 1}')
    with pytest.raises(SystemExit, match="differs from its source"):
        L.compose(G, mps, ref)


def test_an_arm_build_needs_a_valid_reset_log(tmp_path):
    mps, ref, base, arm, stems = make_roots(tmp_path)
    (arm / f"{stems[1]}_resetfix.json").write_text(json.dumps({**LOG, "resets": [{"frame": 120, "stripped": 1}]}))
    out = L.compose(G, mps, ref)
    assert out["refused"][stems[1]].startswith("a reset did not strip") and not out["complete"]
    (arm / f"{stems[1]}_resetfix.json").unlink()
    (ref / "builds" / f"r23_checkc_{G}.json").unlink()
    assert L.compose(G, mps, ref)["refused"][stems[1]] == "no reset log"

"""tools.p8_colab_cpu: the isa environment, the start rule, the step order, the box matching, the reading rule
(docs/research/P8_colab_cpu_parity.md section 4) and the build checks, on synthetic inputs. No video, no model."""

import json

import pytest

from tools import p8_colab_cpu as C

MAC = {"torch_cpu_capability": "NO AVX", "onednn_isa": "Intel SSE4.1",
       "mkl": "MKL_VERBOSE oneMKL 2022.0 ... Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled processors",
       "cv2_cpu_features": "SSE SSE2 SSE3 SSSE3 SSE4.1 *SSE4.2 *AVX? *FP16? *AVX2? *AVX512-SKX?",
       "numpy_cpu_found": ["MMX", "POPCNT", "SSE", "SSE2", "SSE3", "SSE41", "SSE42", "SSSE3"]}
HERE = {"cv2_cpu_features": "SSE SSE2 SSE3 *SSE4.1 *SSE4.2 *FP16 *AVX *AVX2 *AVX512-SKX?",
        "numpy_cpu_found": ["MMX", "POPCNT", "SSE", "SSE2", "SSE3", "SSE41", "SSE42", "SSSE3", "AVX", "F16C", "FMA3",
                            "AVX2"],
        "numpy_dispatch": ["SSSE3", "SSE41", "POPCNT", "SSE42", "AVX", "F16C", "FMA3", "AVX2", "AVX512F"]}


def test_isa_env_caps_each_library_at_the_macs_level():
    env = C.isa_env(MAC, HERE)
    assert env["ATEN_CPU_CAPABILITY"] == "default"
    assert env["DNNL_MAX_CPU_ISA"] == env["ONEDNN_MAX_CPU_ISA"] == "SSE41"
    assert env["MKL_ENABLE_INSTRUCTIONS"] == "SSE4_2"
    assert env["OPENCV_CPU_DISABLE"].split(",") == ["AVX", "AVX2", "FP16"]      # available here, not on the Mac
    assert env["NPY_DISABLE_CPU_FEATURES"].split() == ["AVX", "AVX2", "F16C", "FMA3"]
    assert set(env) <= set(C.ISA_KEYS)


def test_isa_env_refuses_an_unknown_mac_level():
    with pytest.raises(SystemExit):
        C.isa_env({**MAC, "onednn_isa": "Intel AVX-512"}, HERE)
    with pytest.raises(SystemExit):
        C.isa_env({**MAC, "mkl": None}, HERE)


def test_parse_verbose_takes_the_first_isa_lines():
    text = "x\nonednn_verbose,info,cpu,isa:Intel SSE4.1\nonednn_verbose,info,cpu,isa:other\n" \
           "MKL_VERBOSE oneMKL 2022 Intel(R) AVX2 enabled processors, Lnx\n"
    v = C.parse_verbose(text)
    assert v["onednn_isa"] == "Intel SSE4.1" and "AVX2" in v["mkl"]
    assert C.parse_verbose("nothing") == {"onednn_isa": None, "mkl": None}


def test_start_rule():
    rates = {}
    assert C.predicted_s("plain", 100, rates) is None
    assert C.may_start(0, None, None) and C.may_start(0, 10, None) and not C.may_start(10, 10, None)
    rates["plain"] = 2.0
    assert C.predicted_s("plain_b", 100, rates) == 200
    assert C.predicted_s("isa", 100, rates) == 2.0 * C.ISA_FIRST_FACTOR * 100
    rates["isa"] = 5.0
    assert C.predicted_s("isa", 10, rates) == 50
    assert C.may_start(100, 300, 200) and not C.may_start(101, 300, 200)


def test_steps_follow_the_fixed_order():
    assert C.steps_for(["plain"], None) == [("plain", C.RESET_W), ("plain", C.NORESET_W), ("plain", C.TAIL[0]),
                                           ("plain", C.TAIL[1])]
    full = C.steps_for(list(C.COLAB_ARMS), None)
    assert full == list(C.ORDER) and full[:2] == [("plain", C.RESET_W), ("plain", C.NORESET_W)]
    assert C.steps_for(["mac"], [C.RESET_W, C.NORESET_W]) == [("mac", C.RESET_W), ("mac", C.NORESET_W)]
    with pytest.raises(SystemExit):
        C.steps_for(["plain", "isa"], [C.RESET_W])
    assert C.ALL_WINDOWS == C.CORE + C.TAIL and all(w.startswith(C.GAME + "_") for w in C.ALL_WINDOWS)


def _b(tid, x, y=100, foot=None):
    return {"tid": tid, "bbox": [x, y, x + 40, y + 120], "foot_stab": foot or [x + 20.0, y + 120.0]}


def test_match_boxes_is_one_to_one_with_a_gate():
    a = [_b(1, 0), _b(2, 100), _b(3, 500)]
    b = [_b(7, 101), _b(8, 1), _b(9, 900)]
    pairs, ua, ub = C.match_boxes(a, b)
    assert sorted((i, j, d) for i, j, d in pairs) == [(0, 1, 1.0), (1, 0, 1.0)]
    assert (ua, ub) == (1, 1)                               # 500 and 900 are farther apart than the gate
    assert C.match_boxes([], b) == ([], 0, 3)


def test_box_metrics_and_window_reading():
    x = {0: {"boxes": [_b(1, 0), _b(2, 100)], "H": None}, 6: {"boxes": [_b(1, 2), _b(2, 100)], "H": None}}
    r = {0: {"boxes": [_b(1, 0), _b(2, 101)], "H": None}, 6: {"boxes": [_b(5, 0), _b(2, 100)], "H": None}}
    m = C.box_metrics(x, r)
    assert m["frames_jaccard"] == 1.0 and m["common"] == 2
    assert m["frames_within_1px"] == 0.5                    # frame 6 has a box 2 px off
    assert m["offset_px_max"] == 2.0 and m["pairs"] == 4 and m["unmatched_boxes"] == 0
    assert m["pairs_within_1px"] == 3 and m["tid_equal_share"] == pytest.approx(3 / 3)
    eq_all = {k: True for k in C.SIDE}
    assert C.window_reading(eq_all, m, 1, 1) == "identical"
    eq = {**eq_all, "frames": False}
    assert C.window_reading(eq, m, 1, 1) == "different"     # 0.5 of frames within 1 px
    good = {**m, "frames_within_1px": 0.995, "h_gap_px": {"median": 0.2}}
    assert C.window_reading(eq, good, 1, 1) == "near"
    assert C.window_reading(eq, good, 1, 2) == "different"   # reset counts differ
    assert C.window_reading(eq, {**good, "h_gap_px": {"median": 1.5}}, 1, 1) == "different"
    assert C.window_reading(eq, {**good, "frames_jaccard": 0.99}, 1, 1) == "different"


def test_tid_agreement_counts_raw_and_mapped():
    x = {0: {"boxes": [_b(1, 0), _b(2, 100)]}, 6: {"boxes": [_b(1, 0), _b(2, 100)]}}
    r = {0: {"boxes": [_b(11, 0), _b(12, 100)]}, 6: {"boxes": [_b(11, 0), _b(12, 100)]}}
    m = C.box_metrics(x, r)
    assert m["tid_equal_share"] == 0.0 and m["tid_mapped_share"] == 1.0


def test_arm_reading_needs_a_reset_and_a_no_reset_window():
    resets = {C.RESET_W}
    assert C.arm_reading({C.RESET_W: "identical"}, resets) == "not read"
    assert C.arm_reading({C.NORESET_W: "identical"}, resets) == "not read"
    assert C.arm_reading({C.RESET_W: "identical", C.NORESET_W: "identical"}, resets) == "identical"
    assert C.arm_reading({C.RESET_W: "near", C.NORESET_W: "identical"}, resets) == "near"
    assert C.arm_reading({C.RESET_W: "near", C.NORESET_W: "different"}, resets) == "different"


def test_canonical_drops_only_the_frames_source():
    a = json.dumps({"source": "/a/x.mp4", "fps": 59.94, "frames": [{"frame": 0}]})
    b = json.dumps({"fps": 59.94, "frames": [{"frame": 0}], "source": "/b/x.mp4"})
    assert C.canonical("frames", a) == C.canonical("frames", b)
    assert C.canonical("identity", a) != C.canonical("identity", b)


def _write_side(d, w, dev, resets, log):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{w}_device.json").write_text(json.dumps(dev))
    (d / f"{w}_identity.json").write_text(json.dumps({"identity": {"camera_cut_resets": resets}}))
    if log is not None:
        (d / f"{w}_resetfix.json").write_text(json.dumps({"resets": log}))


def test_side_problems(tmp_path):
    cpu = {"requested": {"detector": "cpu", "court": "cpu", "gate": "cpu"},
           "resolved": {"detector": "cpu", "court": "cpu", "gate": "cpu"}, "reset_fix": True}
    _write_side(tmp_path / "ok", "w", cpu, 2, [{"stripped": 2}, {"stripped": 2}])
    assert C.side_problems(tmp_path / "ok", "w") == []
    _write_side(tmp_path / "mps", "w", {**cpu, "resolved": {**cpu["resolved"], "court": "mps"}}, 0, [])
    assert C.side_problems(tmp_path / "mps", "w")
    _write_side(tmp_path / "nofix", "w", {k: v for k, v in cpu.items() if k != "reset_fix"}, 0, [])
    assert C.side_problems(tmp_path / "nofix", "w")
    _write_side(tmp_path / "log", "w", cpu, 2, [{"stripped": 2}])
    assert C.side_problems(tmp_path / "log", "w")
    _write_side(tmp_path / "nolog", "w", cpu, 0, None)
    assert C.side_problems(tmp_path / "nolog", "w")


def test_guard_accepts_the_game_and_windows():
    C.guard_all()


def test_job_script_keeps_the_token_out_of_children_and_pins_everything():
    base = "6261476535188ed4cedf7d6d4c69a7d310e3889b"
    s = C.job_script(base, "abc_DEF-123")
    assert s.splitlines()[1] == 'T="$DRIVE_TOKEN"; unset DRIVE_TOKEN'
    assert "trap 'echo \"JOB FAILED" in s and "CHECK1 cv2" in s and "import cv2, numpy, torch, ultralytics" in s
    assert s.count('DRIVE_TOKEN="$T"') == 2                  # the fetch and the run only
    assert f"git checkout -q --detach {base}" in s
    assert "--no-deps opencv-python-headless==4.13.0.92 ultralytics==8.4.48" in s
    assert "--index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2" in s
    assert f"uv venv -q --python {C.PYTHON_PIN}" in s and "--arms plain,isa,plain_b" in s
    assert not any(p.startswith("opencv") for p in C.CORE_PINS)       # numpy>=2 metadata; installed without deps
    for pin in C.CORE_PINS + C.SOFT_PINS + C.NODEPS_PINS:
        assert pin in s
    with pytest.raises(SystemExit):
        C.job_script("main", "abc")
    with pytest.raises(SystemExit):
        C.job_script(base, "a/b")

"""tools/p8_device.py: device flags reach the frozen build without changing a frozen file (PLAN.md D99)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import p8_device as D

REPO = Path(__file__).resolve().parents[1]
NO_FLAGS = {k: "" for k in ("P8_DEVICE", "P8_DEVICE_DETECTOR", "P8_DEVICE_COURT", "P8_DEVICE_GATE")}


@pytest.fixture
def clean_env(monkeypatch):
    for k in NO_FLAGS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_flags_default_to_auto_and_a_per_model_flag_overrides_the_default(clean_env):
    assert D.requested() == {"detector": "auto", "court": "auto", "gate": "auto"} and not D.is_set()
    clean_env.setenv("P8_DEVICE", "cpu")
    clean_env.setenv("P8_DEVICE_COURT", "mps")
    assert D.requested() == {"detector": "cpu", "court": "mps", "gate": "cpu"} and D.is_set()
    clean_env.setenv("P8_DEVICE_GATE", "tpu")
    with pytest.raises(SystemExit):
        D.requested()


def test_only_the_build_command_is_rewritten_and_only_with_a_flag(clean_env):
    build = ["/py", "build_trajectories.py", "--source", "w.mp4"]
    assert D.rewrite(build) == build                                   # no flag: the frozen command as it is
    clean_env.setenv("P8_DEVICE_DETECTOR", "cpu")
    assert D.rewrite(build) == ["/py", "-m", "tools.p8_device", "build_trajectories.py", "--source", "w.mp4"]
    other = ["ffmpeg", "-i", "x.mp4"]
    assert D.rewrite(other) == other


def test_the_frozen_rebuild_subprocess_goes_through_the_proxy(clean_env, tmp_path):
    from tools import p8_frozen as F
    from sportvu import rebuild
    import config
    saved = {a: getattr(m, a) for m, a in ((rebuild, "subprocess"),)}
    keep = {(m, a): getattr(m, a) for m, a in ((config, "REPORTS_DIR"), (config, "HEAD_V2_PATH"))}
    try:
        F.install_paths(tmp_path / "ws")
        assert isinstance(rebuild.subprocess, D._SubprocessProxy)
        seen = []
        rebuild.subprocess._real = type("R", (), {"run": staticmethod(lambda cmd, *a, **k: seen.append(cmd) or
                                                  type("P", (), {"returncode": 0, "stderr": ""})())})()
        clean_env.setenv("P8_DEVICE", "cpu")
        w = {"window": "gsw_bkn_2015_s00_w00", "fps": 30.0, "f_start": 0, "f_end": 90}
        rebuild.build(w, tmp_path / "gsw_bkn_2015_s00_w00.mp4", out_dir=tmp_path)
        assert seen and seen[0][1:4] == ["-m", "tools.p8_device", "build_trajectories.py"]
    finally:
        rebuild.subprocess = saved["subprocess"]
        for (m, a), v in keep.items():
            setattr(m, a, v)


def test_apply_sets_each_model_device_the_build_sees(clean_env, monkeypatch):
    """The device each component of build_trajectories.py receives: the court through config.get_device, the
    detector through PlayerTracker(device=...), the gate through get_backbone(name, device)."""
    import config
    import detect.tracker as T
    import gate.backbones as GB
    got = {}
    monkeypatch.setattr(config, "get_device", lambda: "mps")
    monkeypatch.setattr(T.PlayerTracker, "__init__", lambda self, *a, **k: got.__setitem__("detector", k.get("device")))
    monkeypatch.setattr(GB, "get_backbone", lambda name, device=None, *a, **k: got.__setitem__("gate", device))
    resolved = D.apply({"detector": "cpu", "court": "mps", "gate": "cpu"})
    assert resolved == {"detector": "cpu", "court": "mps", "gate": "cpu"}
    T.PlayerTracker(device=config.get_device())           # what build_trajectories.py does
    GB.get_backbone("clip", config.get_device())
    assert got == {"detector": "cpu", "gate": "cpu"} and config.get_device() == "mps"


def test_apply_with_no_flag_patches_nothing(clean_env, monkeypatch):
    import config
    import detect.tracker as T
    before = (config.get_device, T.PlayerTracker.__init__)
    D.apply(D.requested())
    assert (config.get_device, T.PlayerTracker.__init__) == before


def test_the_child_records_its_devices_next_to_the_window_outputs(tmp_path):
    """python -m tools.p8_device in a child process, with a stand-in build script: the device record lands in
    config.TRACKING_DIR under the window's stem (sportvu.rebuild.build moves it with the outputs) and the script runs
    with config.get_device() patched."""
    (tmp_path / "build_trajectories.py").write_text(
        "import config, json\nopen(config.TRACKING_DIR / 'seen.json', 'w').write(json.dumps(config.get_device()))\n")
    code = ("import sys, config; from pathlib import Path; config.PROJECT_ROOT = Path(sys.argv[1]); "
            "config.TRACKING_DIR = Path(sys.argv[1]) / 'trk'; from tools import p8_device as D; "
            "D.main(['build_trajectories.py', '--source', 'gsw_bkn_2015_s00_w00.mp4'])")
    env = {**{k: v for k, v in __import__("os").environ.items() if k not in NO_FLAGS}, "P8_DEVICE_COURT": "cpu"}
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    rec = json.loads((tmp_path / "trk" / "gsw_bkn_2015_s00_w00_device.json").read_text())
    assert rec["requested"] == {"detector": "auto", "court": "cpu", "gate": "auto"} and rec["resolved"]["court"] == "cpu"
    assert json.loads((tmp_path / "trk" / "seen.json").read_text()) == "cpu"


def test_a_workspace_never_mixes_device_settings(clean_env, tmp_path):
    stem = "gsw_bkn_2015_s00_w00"
    D.check_cached(tmp_path, stem)                         # a pre-wrapper window, no flag now: same defaults
    clean_env.setenv("P8_DEVICE", "cpu")
    with pytest.raises(SystemExit):
        D.check_cached(tmp_path, stem)                     # built with the defaults, asked for cpu now
    (tmp_path / f"{stem}_device.json").write_text(json.dumps({"requested": D.requested(), "resolved": {}}))
    D.check_cached(tmp_path, stem)


def test_the_device_wrapper_guards_its_source(clean_env):
    with pytest.raises(SystemExit):
        D.main(["build_trajectories.py", "--source", "/x/12.25.2015.NOP.at.MIA_s00_w00.mp4"])

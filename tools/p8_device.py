"""tools/p8_device.py: the compute device of each model in a P8 window build, set by environment flags (PLAN.md D99).

    P8_DEVICE=cpu                   every model of the build on this device
    P8_DEVICE_DETECTOR=cpu          the player detector and tracker only (detect.tracker.PlayerTracker)
    P8_DEVICE_COURT=mps             the court models (court/: grid tracker, keypoint and line models) and every other
                                    model that asks config.get_device(), except the two below
    P8_DEVICE_GATE=cpu              the gate's CLIP backbone (gate.backbones.get_backbone)

A per-model flag overrides P8_DEVICE. With no flag set nothing is patched and the build picks its device as the frozen
code does (config.get_device(): CUDA, else MPS, else CPU).

    P8_MPS_CLIP_FIX=1               ultralytics takes its own out-of-place clamp branches (docs/research/P8_E1.md, M7
                                    and M8): on this Mac the in-place clamp_ of ops.clip_boxes on MPS sets the x
                                    coordinates of the most confident boxes to the image height. Off unless set; with
                                    it set, the window's device record carries "clip_fix": true.

Why: on this Mac MPS gives about one player box per frame fewer than CPU and a court H about 5 px away, which moves
the frozen per-window clock offset and shrinks the camera-truth candidates (docs/research/P8_PLATFORM.md).

How, with no frozen file changed: sportvu.rebuild.build runs `python build_trajectories.py ...` as a subprocess.
install_proxy() (called by tools.p8_frozen.install_paths) replaces the subprocess module that sportvu.rebuild holds
with a proxy whose run() rewrites exactly that command to `python -m tools.p8_device build_trajectories.py ...` when a
flag is set. In the child, main() patches config.get_device, PlayerTracker.__init__ and gate.backbones.get_backbone,
writes <window>_device.json into config.TRACKING_DIR (sportvu.rebuild.build moves it next to the window's outputs) and
runs build_trajectories.py unchanged with runpy. Every other subprocess call passes through untouched.
"""

import json
import os
import runpy
import sys
from pathlib import Path

from tools.research_testset import refuse_cv_test

WHO = "tools.p8_device"
FLAGS = {"detector": "P8_DEVICE_DETECTOR", "court": "P8_DEVICE_COURT", "gate": "P8_DEVICE_GATE"}
DEFAULT_FLAG = "P8_DEVICE"
DEVICES = ("cpu", "mps", "cuda")
CLIP_FIX_FLAG = "P8_MPS_CLIP_FIX"
BUILD_SCRIPT = "build_trajectories.py"


def requested(env=None) -> dict:
    """{detector, court, gate} -> the device asked for, or "auto" when no flag names it. SystemExit on a bad value."""
    env = os.environ if env is None else env
    default = env.get(DEFAULT_FLAG) or "auto"
    out = {k: env.get(v) or default for k, v in FLAGS.items()}
    bad = {k: v for k, v in out.items() if v not in DEVICES + ("auto",)}
    if bad:
        raise SystemExit(f"{WHO}: unknown device {bad}; use one of {DEVICES}")
    return out


def clip_fix_requested(env=None) -> bool:
    """P8_MPS_CLIP_FIX: "1" sets it, unset or empty leaves it off; anything else is refused."""
    env = os.environ if env is None else env
    v = env.get(CLIP_FIX_FLAG) or ""
    if v not in ("", "0", "1"):
        raise SystemExit(f"{WHO}: {CLIP_FIX_FLAG} must be 1 (or unset), not {v!r}")
    return v == "1"


def is_set(env=None) -> bool:
    return any(v != "auto" for v in requested(env).values()) or clip_fix_requested(env)


def apply_clip_fix() -> list:
    """In this process, before any model loads: every ultralytics 8.4.48 module that imported NOT_MACOS14 by name takes
    the out-of-place branch (ops.clip_boxes and ops.clip_coords; the pose heads' keypoint sigmoid) -> the names set."""
    import ultralytics.nn.modules.head as head
    import ultralytics.utils.ops as ops
    done = []
    for mod in (ops, head):
        if not hasattr(mod, "NOT_MACOS14"):
            raise SystemExit(f"{WHO}: {mod.__name__} has no NOT_MACOS14; the clip fix does not fit this ultralytics")
        mod.NOT_MACOS14 = False
        done.append(f"{mod.__name__}.NOT_MACOS14")
    return done


# ── driver side: route the frozen build subprocess through this module ─────────────────────────────────────────────
class _SubprocessProxy:
    """Stands in for the subprocess module inside sportvu.rebuild: run() of a build_trajectories.py command goes
    through `python -m tools.p8_device` when a device flag is set; everything else is the real subprocess module."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, cmd, *a, **k):
        return self._real.run(rewrite(cmd), *a, **k)


def rewrite(cmd):
    """The command a build subprocess runs: unchanged without a flag, else through this module."""
    if is_set() and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == BUILD_SCRIPT:
        return [cmd[0], "-m", "tools.p8_device", *cmd[1:]]
    return cmd


def install_proxy() -> None:
    import subprocess
    from sportvu import rebuild
    if not isinstance(rebuild.subprocess, _SubprocessProxy):
        rebuild.subprocess = _SubprocessProxy(subprocess)


def record_of(out_dir: Path, stem: str):
    """The device record a build wrote for a window, or None (a build made before this wrapper, or without a flag)."""
    p = Path(out_dir) / f"{stem}_device.json"
    return json.loads(p.read_text()) if p.exists() else None


def check_cached(out_dir: Path, stem: str) -> None:
    """Refuse to reuse a cached window built on other devices than the ones asked for now: a workspace never mixes
    devices. A window without a record was built with the frozen defaults ("auto")."""
    want = requested()
    rec = record_of(out_dir, stem)
    have = (rec or {}).get("requested") or {k: "auto" for k in FLAGS}
    if have != want:
        raise SystemExit(f"{WHO}: {stem} in {out_dir} was built with devices {have}, not {want}; use a fresh "
                         "V3_P8_ROOT for another device setting")
    if bool((rec or {}).get("clip_fix", False)) != clip_fix_requested():
        raise SystemExit(f"{WHO}: {stem} in {out_dir} was built with clip_fix {bool((rec or {}).get('clip_fix'))}, "
                         f"not {clip_fix_requested()}")


# ── child side: the patched build ───────────────────────────────────────────────────────────────────────────────────
def apply(req: dict) -> dict:
    """Patch the device choices of the build in this process -> the resolved devices."""
    import config
    frozen_default = config.get_device
    auto = frozen_default()
    resolved = {k: (auto if v == "auto" else v) for k, v in req.items()}
    if req["court"] != "auto":
        config.get_device = lambda: resolved["court"]
    if req["detector"] != "auto":
        import detect.tracker as T
        init = T.PlayerTracker.__init__

        def tracker_init(self, *a, **k):
            k["device"] = resolved["detector"]
            init(self, *a, **k)
        T.PlayerTracker.__init__ = tracker_init
    if req["gate"] != "auto":
        import gate.backbones as GB
        get = GB.get_backbone

        def get_backbone(name, device=None, *a, **k):
            return get(name, resolved["gate"], *a, **k)
        GB.get_backbone = get_backbone
    return resolved


def device_record(req: dict, resolved: dict) -> dict:
    """The window's device record; with P8_MPS_CLIP_FIX set the fix is applied here and recorded as clip_fix."""
    rec = {"requested": req, "resolved": resolved}
    if clip_fix_requested():
        rec["clip_fix"] = True
        rec["clip_fix_names"] = apply_clip_fix()
    return rec


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != BUILD_SCRIPT:
        raise SystemExit(f"{WHO}: usage: python -m tools.p8_device {BUILD_SCRIPT} <its arguments>")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    refuse_cv_test([src], WHO)
    req = requested()
    resolved = apply(req)
    import config
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    part = config.TRACKING_DIR / f"{stem}_device.json.part"
    part.write_text(json.dumps(device_record(req, resolved)))
    os.replace(part, config.TRACKING_DIR / f"{stem}_device.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())

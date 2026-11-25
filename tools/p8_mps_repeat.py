"""tools/p8_mps_repeat.py: M7 part 2, MPS repeatability of the registered build path (docs/research/P8_E1.md, "M7,
part 2", definitions committed before any number). Report only.

    <frozen CV python> -m tools.p8_mps_repeat rebuild --model {r23,e1}       inside the heavy-job lock
    <frozen CV python> -m tools.p8_mps_repeat compare                         inside the heavy-job lock
    (internal) <python> -m tools.p8_mps_repeat child build_trajectories.py <its arguments>

Rebuilds the fixed windows with no device flag (MPS on this Mac, as the registered builds) into builds/m7_r23_mps_repeat
and builds/m7_e1_gsw_bkn_2015_mps_repeat only (any other folder is refused), then compares them byte for byte with
builds/r23 and builds/e1_gsw_bkn_2015_seed0. The detector weights reach the frozen build through a child that sets
config.PLAYER_DETECTOR_WEIGHTS and records the weights' sha256, as tools.p8_e1_score's stub did (a worktree has no
models/ folder); no device patch is applied and the tool stops if any P8 device flag is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from tools.p8_common import atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_mps_repeat"
GAME = FOLD = "gsw_bkn_2015"
WINDOWS = {"r23": ("gsw_bkn_2015_s04_w00", "gsw_bkn_2015_s05_w00", "gsw_bkn_2015_s06_w00"),
           "e1": ("gsw_bkn_2015_s04_w00",)}
FOLDERS = {"r23": "m7_r23_mps_repeat", "e1": "m7_e1_gsw_bkn_2015_mps_repeat"}
REGISTERED = {"r23": "r23", "e1": "e1_gsw_bkn_2015_seed0"}
CPU = {"r23": "e1m6_r23_detcpu", "e1": "e1m6_e1_gsw_bkn_2015_detcpu"}
FILES = ("frames", "trajectories", "identity")
BUILD_SCRIPT = "build_trajectories.py"
ENV = "P8_M7_DETECTOR"


def check_guard() -> None:
    refuse_cv_test([GAME], WHO)
    guard([GAME], WHO)


def folder(model: str) -> str:
    if model not in FOLDERS:
        raise SystemExit(f"{WHO}: unknown model {model}")
    name = FOLDERS[model]
    if name in REGISTERED.values() or not name.startswith("m7_"):
        raise SystemExit(f"{WHO}: {name} is not an M7 folder")
    return name


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def sha(path: Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def box_counts(side: dict) -> dict:
    return {int(r["frame"]): len(r["boxes"]) for r in side["frames"]}


def compare_counts(a: dict, b: dict) -> dict:
    """Per-frame box counts of two sidecars -> processed frames, boxes per frame, frames with the same count."""
    shared = sorted(a.keys() & b.keys())
    return {"frames_a": len(a), "frames_b": len(b), "shared_frames": len(shared),
            "boxes_per_frame_a": round(sum(a.values()) / len(a), 4) if a else None,
            "boxes_per_frame_b": round(sum(b.values()) / len(b), 4) if b else None,
            "same_count_frames": sum(1 for f in shared if a[f] == b[f])}


# ── build ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class _Proxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, cmd, *a, **k):
        return self._real.run(rewrite(cmd), *a, **k)


def rewrite(cmd, env=None):
    env = os.environ if env is None else env
    if env.get(ENV) and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == BUILD_SCRIPT:
        return [cmd[0], "-m", "tools.p8_mps_repeat", "child", *cmd[1:]]
    return cmd


def child(argv: list) -> int:
    from tools import p8_device
    if p8_device.is_set():
        raise SystemExit(f"{WHO}: a P8 device flag is set; M7 repeats the registered (unflagged) path")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    guard([src], WHO)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    part = config.TRACKING_DIR / f"{stem}_detector.json.part"
    part.write_text(json.dumps({"weights": weights.name, "sha256": sha(weights), "model": os.environ.get("P8_M7_MODEL"),
                                "device": config.get_device()}))
    os.replace(part, config.TRACKING_DIR / f"{stem}_detector.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_rebuild(args) -> dict:
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_device
    from tools import p8_frozen as F
    from tools.p8_e1_m4 import weights_of
    from tools.p8_e1_score import check_provenance
    check_guard()
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    if p8_device.is_set():
        raise SystemExit(f"{WHO}: a P8 device flag is set; M7 repeats the registered (unflagged) path")
    if args.model == "e1":
        weights = weights_of(FOLD).resolve()
        check_provenance(FOLD, GAME, 0, weights)
    else:
        weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    P = F.install_paths()
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV], os.environ["P8_M7_MODEL"] = str(weights), args.model
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / folder(args.model)
    out.mkdir(parents=True, exist_ok=True)
    wins = {w["window"]: w for w in F.label_windows(P, GAME)}
    res = {"model": args.model, "weights_sha256": sha(weights), "windows": {}}
    for s in WINDOWS[args.model]:
        r = F.safe_build(P, wins[s], P["label_build"] / f"{s}.mp4", out, extra_args=gate_args())
        if not r["ok"]:
            raise SystemExit(f"{WHO}: build of {s} failed: {r.get('tail')}")
        rec = json.loads((out / f"{s}_detector.json").read_text())
        if rec["sha256"] != res["weights_sha256"]:
            raise SystemExit(f"{WHO}: {s} did not record the expected weights")
        res["windows"][s] = {"cached": bool(r.get("cached")), "device": rec.get("device")}
    print(json.dumps(res, indent=1))
    return res


def cmd_compare(args) -> dict:
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    check_guard()
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    g = F.game_of(GAME)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, GAME)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    wins = {w["window"]: w for w in F.label_windows(P, GAME)}
    builds = P["builds"]

    def score(bd, s):
        rec = S.score_window(P, GAME, wins[s], bd, "camera", mom, mirror, refs, events)[0]
        return {k: rec[k] for k in ("missed", "in_frame", "ghosts", "boxes", "ref_ghosts", "ref_boxes")}

    out = {"game": GAME, "repeat": {}, "s04_w00_by_build": {}}
    for m, stems in WINDOWS.items():
        for s in stems:
            rep, reg = builds / folder(m) / s, builds / REGISTERED[m] / s
            same = {f: sha(f"{rep}_{f}.json") == sha(f"{reg}_{f}.json") for f in FILES}
            row = {"byte_identical": same, "all_identical": all(same.values())}
            if not row["all_identical"]:
                a = box_counts(json.loads(Path(f"{reg}_frames.json").read_text()))
                b = box_counts(json.loads(Path(f"{rep}_frames.json").read_text()))
                row["counts_registered_vs_repeat"] = compare_counts(a, b)
                row["score"] = {"registered": score(builds / REGISTERED[m], s), "repeat": score(builds / folder(m), s)}
            out["repeat"][f"{m}/{s}"] = row
    s = "gsw_bkn_2015_s04_w00"
    counts = {}
    for label, bd in (("r23 MPS registered", REGISTERED["r23"]), ("r23 MPS repeat", folder("r23")),
                      ("r23 CPU (M6)", CPU["r23"]), ("E1 MPS registered", REGISTERED["e1"]),
                      ("E1 MPS repeat", folder("e1")), ("E1 CPU (M6)", CPU["e1"])):
        p = builds / bd / f"{s}_frames.json"
        if p.exists():
            counts[label] = box_counts(json.loads(p.read_text()))
    for label, c in counts.items():
        out["s04_w00_by_build"][label] = {"frames": len(c), "boxes_per_frame": round(sum(c.values()) / len(c), 4)}
    for m in ("r23", "E1"):
        a, b = counts.get(f"{m} MPS registered"), counts.get(f"{m} CPU (M6)")
        if a and b:
            out["s04_w00_by_build"][f"{m} MPS registered vs CPU"] = compare_counts(a, b)
    atomic_write(p8_root() / "parity" / "mps_repeat.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        return child(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m tools.p8_mps_repeat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--model", required=True, choices=tuple(FOLDERS))
    sub.add_parser("compare")
    a = ap.parse_args(argv)
    {"rebuild": cmd_rebuild, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

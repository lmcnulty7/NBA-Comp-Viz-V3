"""tools/p8_truth_cpu_r23.py: check C's r23 side for a reset-fixed CPU-detector truth in the reference root
(docs/research/P8_truth_arenas.md section 16 version 4; PLAN.md D192). No model runs here.

    python3 -m tools.p8_truth_cpu_r23 compose GAME --mps-root <P8 workspace> --ref-root <reference root>
    python3 -m tools.p8_truth_cpu_r23 detector-check        exit 1 unless models/player_detector.pt is r23's

Check C compares, per label window, the V3 label build with an r23 build of the same clip, both with the tracker reset
fix, so that it stays a court-model comparison (section 5). compose is the only writer of <ref root>/builds/r23
(version 4.1; the reset-fix arm writes its unfixed baseline to builds/r23_nofix and composes nothing). builds/r23 is
composed per game, only after the
reset-fix arm's equivalence record (resetfix/equiv__<game>.json, docs/research/P8_resetfix.md section 14) passes:
- the windows listed in resetfix/plan__<game>.json "reset_windows" (a reset in the baseline build): the arm's fixed
  build, builds/resetfix_r23_detcpu__<game> of the reference root (its other windows, the equivalence and smoke
  windows, are ignored);
- every other window: the workspace's reference-platform r23 build (tools.p8_placement.BUILDS: M9's, and M6's r23 arm
  for gsw_bkn_2015). With no reset, the fix changes nothing, which the arm's equivalence windows test byte for byte.
A window's build is copied only when:
- its sidecar's "source" is the window's label clip: the workspace clip for a baseline build, the reference root's
  clip for an arm build; and that clip is byte-identical (SHA-256) in the two roots;
- its device record asked for the reference setting (detector cpu, court and gate auto), no clip fix, and "reset_fix"
  true for an arm build or absent for a baseline build;
- its detector record names the r23 player detector (model "r23", weights sha256 R23_DETECTOR_SHA256, the
  repository's models/player_detector.pt).
A window already in builds/r23 passes the same checks on its own files or stops the step; a build there with no build
on record stops it too. The record (builds/r23_checkc_<game>.json) says whether every V3 label window now has an r23
build ("complete"); check C runs only then, because tools.p8_truth_v2 measures only windows with both builds and a
missing one would drop out unseen. Files are copied, not linked; nothing is built here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from tools.p8_common import REPO, atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_cpu_r23"
APART = ("gsw_ind_2016",)
REFERENCE = {"detector": "cpu", "court": "auto", "gate": "auto"}
R23_DETECTOR_SHA256 = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"   # models/player_detector.pt, r23


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(mps_clip: Path, ref_clip: Path, record, detector=None, want_sha: str = R23_DETECTOR_SHA256,
           want_fix: bool = False) -> str | None:
    """None when the build may stand for the reference root's r23 build of this window, else the reason.
    record: the window's _device.json; detector: its _detector.json; want_fix: an arm build (reset_fix true)."""
    if not detector:
        return "no detector record"
    if detector.get("sha256") != want_sha or detector.get("model") != "r23":
        return f"detector {detector.get('model')} {str(detector.get('sha256'))[:12]}"
    if not record:
        return "no device record"
    if record.get("requested") != REFERENCE:
        return f"devices {record.get('requested')}"
    if want_fix and (record.get("resolved") or {}).get("detector") != "cpu":
        return f"resolved {record.get('resolved')}"
    if record.get("clip_fix"):
        return "clip fix"
    if bool(record.get("reset_fix")) != want_fix:
        return f"reset_fix {bool(record.get('reset_fix'))}, not {want_fix}"
    if not mps_clip.exists() or not ref_clip.exists():
        return "clip missing"
    if sha256(mps_clip) != sha256(ref_clip):
        return "clips differ"
    return None


def check(stem: str, folder: Path, mps_root: Path, ref_root: Path, fixed: bool = False,
          want_sha: str = R23_DETECTOR_SHA256) -> str | None:
    """Every condition for one window's build in `folder` (a source build, or a copy already in builds/r23): its
    recorded source is the right clip (the reference root's for an arm build, the workspace's otherwise), then
    verify()."""
    def rec(name):
        p = folder / f"{stem}_{name}.json"
        return json.loads(p.read_text()) if p.exists() else None
    mclip, rclip = mps_root / "label_build" / f"{stem}.mp4", ref_root / "label_build" / f"{stem}.mp4"
    want_src = rclip if fixed else mclip
    built_from = (rec("frames") or {}).get("source")
    if not built_from or Path(built_from).expanduser().resolve() != want_src.resolve():
        return (f"built from {Path(built_from).name if built_from else 'an unrecorded clip'}, not the "
                f"{'reference root' if fixed else 'workspace'} clip")
    return verify(mclip, rclip, rec("device"), rec("detector"), want_sha, want_fix=fixed)


def detector_sha_ok(path: Path = REPO / "models" / "player_detector.pt") -> bool:
    return path.exists() and sha256(path) == R23_DETECTOR_SHA256


def arm_records(ref_root: Path, game: str) -> tuple:
    """(reset windows, equivalence record) of the reset-fix arm; SystemExit unless the equivalence passed."""
    eq_p, plan_p = ref_root / "resetfix" / f"equiv__{game}.json", ref_root / "resetfix" / f"plan__{game}.json"
    if not eq_p.exists() or not plan_p.exists():
        raise SystemExit(f"{WHO}: {game} has no reset-fix arm plan or equivalence record yet")
    eq = json.loads(eq_p.read_text())
    if eq.get("passes") is not True:
        raise SystemExit(f"{WHO}: {game}'s reset-fix equivalence did not pass ({eq.get('failing_windows')})")
    return set(json.loads(plan_p.read_text())["reset_windows"]), eq


def compose(game: str, mps_root: Path, ref_root: Path) -> dict:
    from tools.p8_placement import BUILDS
    if game in APART:
        raise SystemExit(f"{WHO}: {game} stays apart (D98)")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    if game not in BUILDS:
        raise SystemExit(f"{WHO}: {game} has no reference-platform r23 build on record")
    if not detector_sha_ok():
        raise SystemExit(f"{WHO}: the repository's models/player_detector.pt is not the r23 detector")
    mps_root, ref_root = check_outside(Path(mps_root).expanduser()), check_outside(Path(ref_root).expanduser())
    if mps_root == ref_root:
        raise SystemExit(f"{WHO}: the two roots are the same folder")
    reset, _ = arm_records(ref_root, game)
    base, arm = mps_root / "builds" / BUILDS[game], ref_root / "builds" / f"resetfix_r23_detcpu__{game}"
    dst = ref_root / "builds" / "r23"
    dst.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-len("_frames.json")] for p in base.glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if not reset <= set(stems):
        raise SystemExit(f"{WHO}: reset windows not among the builds on record: {sorted(reset - set(stems))[:3]}")
    unrecorded = sorted(p.name[:-len("_frames.json")] for p in dst.glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json")
                        if p.name[:-len("_frames.json")] not in stems)
    if unrecorded:
        raise SystemExit(f"{WHO}: {len(unrecorded)} r23 build(s) in {dst} have no build on record, for example "
                         f"{unrecorded[0]}; remove them before check C")
    copied, present, refused = {"arm": [], "baseline": []}, [], {}
    for s in stems:
        fixed = s in reset
        if (dst / f"{s}_frames.json").exists():
            why = check(s, dst, mps_root, ref_root, fixed)     # a copy already here passes the same checks, or stops
            if why:
                raise SystemExit(f"{WHO}: {s} is already in {dst} but fails the checks ({why}); remove it first")
            present.append(s)
            continue
        src = arm if fixed else base
        why = check(s, src, mps_root, ref_root, fixed)
        if why:
            refused[s] = why
            continue
        for p in sorted(q for q in src.glob(f"{s}_*") if q.name != f"{s}_frames.json"):
            shutil.copy2(p, dst / p.name)
        shutil.copy2(src / f"{s}_frames.json", dst / f"{s}_frames.json")   # last: its presence marks the window built
        copied["arm" if fixed else "baseline"].append(s)
    v3 = sorted(p.name[:-len("_frames.json")] for p in (ref_root / "label_build").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))
    gap = [s for s in v3 if not (dst / f"{s}_frames.json").exists()]
    out = {"game": game, "design": "docs/research/P8_truth_arenas.md section 16 version 4",
           "sources": {"arm": f"builds/resetfix_r23_detcpu__{game} of the reference root (reset windows)",
                       "baseline": f"builds/{BUILDS[game]} of the P8 workspace (other windows)"},
           "reference_devices": REFERENCE, "windows_on_record": len(stems), "reset_windows": len(reset),
           "copied_from_arm": len(copied["arm"]), "copied_from_baseline": len(copied["baseline"]),
           "already_present": len(present), "refused": refused, "v3_label_windows": len(v3),
           "v3_windows_without_r23": gap, "complete": not gap,
           "note": "check C runs only when every V3 label window has an r23 build here (complete); nothing is built"}
    atomic_write(ref_root / "builds" / f"r23_checkc_{game}.json", json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("game", "windows_on_record", "reset_windows", "copied_from_arm",
                                          "copied_from_baseline", "already_present", "refused", "v3_label_windows",
                                          "complete")}))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("compose")
    lk.add_argument("game")
    lk.add_argument("--mps-root", required=True)
    lk.add_argument("--ref-root", required=True)
    sub.add_parser("detector-check", help="exit 1 unless models/player_detector.pt is the r23 detector")
    a = ap.parse_args(argv)
    if a.cmd == "detector-check":
        ok = detector_sha_ok()
        print(json.dumps({"r23_detector": ok, "want_sha256": R23_DETECTOR_SHA256}))
        raise SystemExit(0 if ok else 1)
    compose(a.game, Path(a.mps_root), Path(a.ref_root))


if __name__ == "__main__":
    sys.exit(main())

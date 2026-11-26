"""tools/p8_truth_cpu_r23.py: the r23 side of check C for a CPU-detector truth in the reference root
(docs/research/P8_truth_arenas.md section 16). No model runs here.

    python3 -m tools.p8_truth_cpu_r23 link GAME --mps-root <P8 workspace> --ref-root <reference root>
    python3 -m tools.p8_truth_cpu_r23 detector-check        exit 1 unless models/player_detector.pt is r23's

Check C compares, per label window, the V3 label build with an r23 build of the same clip. The reference-platform r23
builds of the four in-use games already exist in the P8 workspace (tools.p8_placement.BUILDS: M9's for gsw_cha_2016,
gsw_sac_2015 and hou_cle_2016, M6's r23 arm for gsw_bkn_2015), built from the workspace's label clips with the
detector on CPU. A window's build is copied into <ref root>/builds/r23 only when:
- the build records the workspace's label clip as its source (the sidecar's "source"),
- its clip in the workspace (label_build/<window>.mp4) and the reference root's clip are byte-identical (SHA-256), and
- its device record asked for exactly the reference setting (detector cpu, court and gate auto) with no clip fix, and
- its detector record names the r23 player detector (model "r23", weights sha256 R23_DETECTOR_SHA256, the
  repository's models/player_detector.pt).
A window already in <ref root>/builds/r23 must pass the same checks on its own files, and a build there with no build
on record stops the copy.
No r23 build is made here or by the driver (section 16 version 3). The record says whether every V3 label window of
the reference root (label_build/<window>_frames.json) now has an r23 build ("complete"); check C runs only then,
because tools.p8_truth_v2 measures only windows with both builds and a missing one would drop out unseen. Files are
copied, not linked, so nothing in the workspace can change and nothing in the reference root points into it.
Writes <ref root>/builds/r23_source_<game>.json.
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


def verify(mps_clip: Path, ref_clip: Path, record, detector=None, want_sha: str = R23_DETECTOR_SHA256) -> str | None:
    """None when the workspace build may stand for the reference root's r23 build of this window, else the reason.
    record: the window's _device.json; detector: its _detector.json (weights sha256 and model name)."""
    if not detector:
        return "no detector record"
    if detector.get("sha256") != want_sha or detector.get("model") != "r23":
        return f"detector {detector.get('model')} {str(detector.get('sha256'))[:12]}"
    if not record:
        return "no device record"
    if record.get("requested") != REFERENCE:
        return f"devices {record.get('requested')}"
    if record.get("clip_fix"):
        return "clip fix"
    if not mps_clip.exists() or not ref_clip.exists():
        return "clip missing"
    if sha256(mps_clip) != sha256(ref_clip):
        return "clips differ"
    return None


def check(stem: str, folder: Path, mps_root: Path, ref_root: Path, want_sha: str = R23_DETECTOR_SHA256) -> str | None:
    """Every condition for one window's build in `folder` (the workspace build, or a copy already in the reference
    root): its recorded source is the workspace clip, then verify()."""
    def rec(name):
        p = folder / f"{stem}_{name}.json"
        return json.loads(p.read_text()) if p.exists() else None
    mclip = mps_root / "label_build" / f"{stem}.mp4"
    built_from = (rec("frames") or {}).get("source")
    if not built_from or Path(built_from).expanduser().resolve() != mclip.resolve():
        return f"built from {Path(built_from).name if built_from else 'an unrecorded clip'}, not the workspace clip"
    return verify(mclip, ref_root / "label_build" / f"{stem}.mp4", rec("device"), rec("detector"), want_sha)


def detector_sha_ok(path: Path = REPO / "models" / "player_detector.pt") -> bool:
    return path.exists() and sha256(path) == R23_DETECTOR_SHA256


def link(game: str, mps_root: Path, ref_root: Path) -> dict:
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
    src, dst = mps_root / "builds" / BUILDS[game], ref_root / "builds" / "r23"
    dst.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-len("_frames.json")] for p in src.glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    unrecorded = sorted(p.name[:-len("_frames.json")] for p in dst.glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json")
                        if p.name[:-len("_frames.json")] not in stems)
    if unrecorded:
        raise SystemExit(f"{WHO}: {len(unrecorded)} r23 build(s) in {dst} have no build on record, for example "
                         f"{unrecorded[0]}; remove them before check C")
    copied, present, refused = [], [], {}
    for s in stems:
        if (dst / f"{s}_frames.json").exists():
            why = check(s, dst, mps_root, ref_root)          # a copy already here passes the same checks, or stops
            if why:
                raise SystemExit(f"{WHO}: {s} is already in {dst} but fails the checks ({why}); remove it first")
            present.append(s)
            continue
        why = check(s, src, mps_root, ref_root)
        if why:
            refused[s] = why
            continue
        files = sorted(p for p in src.glob(f"{s}_*") if p.name != f"{s}_frames.json")
        for p in files:
            shutil.copy2(p, dst / p.name)
        shutil.copy2(src / f"{s}_frames.json", dst / f"{s}_frames.json")   # last: its presence marks the window built
        copied.append(s)
    v3 = sorted(p.name[:-len("_frames.json")] for p in (ref_root / "label_build").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))
    gap = [s for s in v3 if not (dst / f"{s}_frames.json").exists()]
    out = {"game": game, "source": f"builds/{BUILDS[game]} of the P8 workspace", "reference_devices": REFERENCE,
           "windows_on_record": len(stems), "copied": len(copied), "already_present": len(present),
           "refused": refused, "v3_label_windows": len(v3), "v3_windows_without_r23": gap,
           "complete": not gap,
           "note": "check C runs only when every V3 label window has an r23 build here (complete); no r23 is built"}
    atomic_write(ref_root / "builds" / f"r23_source_{game}.json", json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("game", "windows_on_record", "copied", "already_present", "refused",
                                          "v3_label_windows", "complete")}))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("link")
    lk.add_argument("game")
    lk.add_argument("--mps-root", required=True)
    lk.add_argument("--ref-root", required=True)
    sub.add_parser("detector-check", help="exit 1 unless models/player_detector.pt is the r23 detector")
    a = ap.parse_args(argv)
    if a.cmd == "detector-check":
        ok = detector_sha_ok()
        print(json.dumps({"r23_detector": ok, "want_sha256": R23_DETECTOR_SHA256}))
        raise SystemExit(0 if ok else 1)
    link(a.game, Path(a.mps_root), Path(a.ref_root))


if __name__ == "__main__":
    sys.exit(main())

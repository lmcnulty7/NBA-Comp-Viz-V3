"""tools/p8_truth_cpu_r23.py: the r23 side of check C for a CPU-detector truth in the reference root
(docs/research/P8_truth_arenas.md section 16). No model runs here.

    python3 -m tools.p8_truth_cpu_r23 link GAME --mps-root <P8 workspace> --ref-root <reference root>

Check C compares, per label window, the V3 label build with an r23 build of the same clip. The reference-platform r23
builds of the four in-use games already exist in the P8 workspace (tools.p8_placement.BUILDS: M9's for gsw_cha_2016,
gsw_sac_2015 and hou_cle_2016, M6's r23 arm for gsw_bkn_2015), built from the workspace's label clips with the
detector on CPU. A window's build is copied into <ref root>/builds/r23 only when:
- its clip in the workspace (label_build/<window>.mp4) and the reference root's clip are byte-identical (SHA-256), and
- its device record asked for exactly the reference setting (detector cpu, court and gate auto) with no clip fix.
Any other window is left for the frozen rebuild step (tools.p8_frozen rebuild --build r23, under the heavy-job lock),
which builds only windows without an r23 build and refuses a cached build made on other devices. Files are copied, not
linked, so nothing in the workspace can change. Writes <ref root>/builds/r23_source_<game>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from tools.p8_common import atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_cpu_r23"
APART = ("gsw_ind_2016",)
REFERENCE = {"detector": "cpu", "court": "auto", "gate": "auto"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(mps_clip: Path, ref_clip: Path, record) -> str | None:
    """None when the workspace build may stand for the reference root's r23 build of this window, else the reason."""
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


def link(game: str, mps_root: Path, ref_root: Path) -> dict:
    from tools.p8_placement import BUILDS
    if game in APART:
        raise SystemExit(f"{WHO}: {game} stays apart (D98)")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    if game not in BUILDS:
        raise SystemExit(f"{WHO}: {game} has no reference-platform r23 build on record")
    mps_root, ref_root = check_outside(Path(mps_root).expanduser()), check_outside(Path(ref_root).expanduser())
    if mps_root == ref_root:
        raise SystemExit(f"{WHO}: the two roots are the same folder")
    src, dst = mps_root / "builds" / BUILDS[game], ref_root / "builds" / "r23"
    dst.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-len("_frames.json")] for p in src.glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    copied, present, refused = [], [], {}
    for s in stems:
        if (dst / f"{s}_frames.json").exists():
            present.append(s)
            continue
        rp = src / f"{s}_device.json"
        why = verify(mps_root / "label_build" / f"{s}.mp4", ref_root / "label_build" / f"{s}.mp4",
                     json.loads(rp.read_text()) if rp.exists() else None)
        if why:
            refused[s] = why
            continue
        files = sorted(p for p in src.glob(f"{s}_*") if p.name != f"{s}_frames.json")
        for p in files:
            shutil.copy2(p, dst / p.name)
        shutil.copy2(src / f"{s}_frames.json", dst / f"{s}_frames.json")   # last: its presence marks the window built
        copied.append(s)
    out = {"game": game, "source": f"builds/{BUILDS[game]} of the P8 workspace", "reference_devices": REFERENCE,
           "windows_on_record": len(stems), "copied": len(copied), "already_present": len(present),
           "refused": refused, "note": "windows not copied are built by tools.p8_frozen rebuild under the lock"}
    atomic_write(ref_root / "builds" / f"r23_source_{game}.json", json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("game", "windows_on_record", "copied", "already_present", "refused")}))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("link")
    lk.add_argument("game")
    lk.add_argument("--mps-root", required=True)
    lk.add_argument("--ref-root", required=True)
    a = ap.parse_args(argv)
    link(a.game, Path(a.mps_root), Path(a.ref_root))


if __name__ == "__main__":
    sys.exit(main())

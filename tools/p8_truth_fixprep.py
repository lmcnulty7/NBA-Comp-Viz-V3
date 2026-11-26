"""tools/p8_truth_fixprep.py: the reference root's housekeeping for CPU truth built with the tracker reset fix
(docs/research/P8_truth_arenas.md section 16 version 4; PLAN.md D192). No model runs here.

    python3 -m tools.p8_truth_fixprep supersede GAME --ref-root <reference root>
    python3 -m tools.p8_truth_fixprep label-check GAME --ref-root <reference root>

supersede: moves a game's outputs built without the fix out of the way, kept, under <ref root>/superseded_nofix/ at
the same relative path (nothing is deleted, CLAUDE.md rule 10):
- every label window output (label_build/<window>_*, team_clusters_<window>.png) of a window whose device record does
  not carry "reset_fix": true; the window's clip (<window>.mp4) and clock time map (<window>_timemap.json) stay, so the
  label step reuses them and the clips stay byte-identical to the P8 workspace's;
- the truth fitted from them: labels/<game>/, labels/<game>_result.json, _offsets.json, _paint.jsonl;
- the records read from that truth: truth_v2/dev/<game>.json, truth_accept/<game>.json, truth_moved/<game>.json,
  truth_moved/<game>_dh_posthoc.json;
- the label and fit done markers (meta/done/label__<game>__*.json, fit__<game>.json);
- version 3's unfixed r23 copies for check C (builds/r23/<window>_*, builds/r23_source_<game>.json): under version 4,
  builds/r23 is the fixed-with-fixed composite of tools.p8_truth_cpu_r23 compose.
The link markers and every other builds/ folder are not touched. It runs once per game: it writes superseded_nofix/<game>.json and
is a no-op when that record exists. It refuses to overwrite a file already in superseded_nofix/.

label-check: exit 1 unless every label window of the game (a sidecar or a gate-empty marker in label_build) has a device
record with "reset_fix": true, the detector requested and resolved cpu, court and gate requested auto, and no clip fix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tools.p8_common import atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_fixprep"
APART = ("gsw_ind_2016",)
KEEP_SUFFIXES = (".mp4", "_timemap.json")
REFERENCE = {"detector": "cpu", "court": "auto", "gate": "auto"}


def _root(game: str, ref_root: Path) -> Path:
    if game in APART:
        raise SystemExit(f"{WHO}: {game} stays apart (D98)")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    return check_outside(Path(ref_root).expanduser())


def window_stems(label_build: Path, game: str) -> list:
    pat = re.compile(rf"^(?:team_clusters_)?({re.escape(game)}_s\d\d_w\d\d)(?:[_.]|$)")
    return sorted({m.group(1) for p in label_build.iterdir() if (m := pat.match(p.name))}) if label_build.exists() else []


def fixed(record) -> bool:
    return bool(record) and record.get("reset_fix") is True


def record(label_build: Path, stem: str):
    p = label_build / f"{stem}_device.json"
    return json.loads(p.read_text()) if p.exists() else None


def unfixed_outputs(label_build: Path, game: str) -> list:
    """The output files of the game's label windows built without the fix (clips and time maps excluded)."""
    out = []
    for s in window_stems(label_build, game):
        if fixed(record(label_build, s)):
            continue
        out += [p for p in sorted(label_build.glob(f"{s}_*")) if not p.name.endswith(KEEP_SUFFIXES)]
        out += [p for p in [label_build / f"team_clusters_{s}.png"] if p.exists()]
    return out


def truth_outputs(root: Path, game: str) -> list:
    cands = [root / "labels" / game, *(root / "labels" / f"{game}_{k}" for k in ("result.json", "offsets.json", "paint.jsonl")),
             root / "truth_v2" / "dev" / f"{game}.json", root / "truth_accept" / f"{game}.json",
             root / "truth_moved" / f"{game}.json", root / "truth_moved" / f"{game}_dh_posthoc.json",
             root / "meta" / "done" / f"fit__{game}.json",
             *sorted((root / "meta" / "done").glob(f"label__{game}__*.json")),
             root / "builds" / f"r23_source_{game}.json",
             *sorted((root / "builds" / "r23").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_*"))]
    return [p for p in cands if p.exists()]


def supersede(game: str, ref_root: Path) -> dict:
    root = _root(game, ref_root)
    dest = root / "superseded_nofix"
    rec_p = dest / f"{game}.json"
    if rec_p.exists():
        print(json.dumps({"game": game, "already_superseded": True}))
        return json.loads(rec_p.read_text())
    files = unfixed_outputs(root / "label_build", game) + truth_outputs(root, game)
    clash = [p for p in files if (dest / p.relative_to(root)).exists()]
    if clash:
        raise SystemExit(f"{WHO}: {len(clash)} file(s) already in {dest}, for example {clash[0].relative_to(root)}")
    for p in files:
        t = dest / p.relative_to(root)
        t.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, t)
    out = {"game": game, "why": "built without the tracker reset fix (PLAN.md D192); kept, not deleted",
           "moved": [str(p.relative_to(root)) for p in files],
           "label_windows_moved": len({p.name.split("_device")[0] for p in files if p.name.endswith("_device.json")}),
           "kept_in_place": "label clips (.mp4) and window time maps; builds/r23; link markers"}
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write(rec_p, json.dumps(out, indent=1))
    print(json.dumps({k: (len(v) if k == "moved" else v) for k, v in out.items() if k != "why"}))
    return out


def label_problems(label_build: Path, game: str) -> dict:
    """{window: reason} for label windows (sidecar or gate-empty marker) whose device record is not the fixed
    reference setting."""
    bad = {}
    for s in window_stems(label_build, game):
        if not ((label_build / f"{s}_frames.json").exists() or (label_build / f"{s}_gate_empty.json").exists()):
            continue
        r = record(label_build, s)
        if not r:
            bad[s] = "no device record"
        elif r.get("reset_fix") is not True:
            bad[s] = "no reset fix"
        elif r.get("requested") != REFERENCE or (r.get("resolved") or {}).get("detector") != "cpu":
            bad[s] = f"devices {r.get('requested')} resolved {r.get('resolved')}"
        elif r.get("clip_fix"):
            bad[s] = "clip fix"
    return bad


def label_check(game: str, ref_root: Path) -> dict:
    root = _root(game, ref_root)
    stems = window_stems(root / "label_build", game)
    bad = label_problems(root / "label_build", game)
    out = {"game": game, "windows": len(stems), "problems": bad, "ok": bool(stems) and not bad}
    print(json.dumps(out))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("supersede", "label-check"):
        p = sub.add_parser(name)
        p.add_argument("game")
        p.add_argument("--ref-root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "supersede":
        supersede(a.game, Path(a.ref_root))
    else:
        raise SystemExit(0 if label_check(a.game, Path(a.ref_root))["ok"] else 1)


if __name__ == "__main__":
    sys.exit(main())

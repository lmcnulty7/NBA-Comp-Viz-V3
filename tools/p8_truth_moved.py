"""tools/p8_truth_moved.py: does a game's camera truth move when it is rebuilt from CPU-detector builds? Report only
(docs/research/P8_truth_arenas.md section 0, "Platform mix"; bars fixed there before any rebuild).

    python3 -m tools.p8_truth_moved GAME --mps-root <P8 root of the existing truth> --cpu-root <reference root>

Reads, in each root, labels/<game>_result.json (the frozen fit's result) and labels/<game>_offsets.json (every built
window's frozen clock offset); in the CPU root also the device record of every label window. Runs no model and
changes no truth; writes <cpu root>/truth_moved/<game>.json.

"Moved" is any one of:
- the median reprojection after refinement (camera.reproj_px_median_after) differs by 1.0 px or more;
- the accepted camera-truth frames differ by 10 percent or more of the MPS count;
- the median absolute difference of the per-window frozen offsets, over windows with an offset in both roots, is
  0.2 s or more.
Both fits must be complete ("partial" False), and every label window of the CPU root must resolve the detector to cpu
(tools.p8_truth_accept.platform_ok), or the comparison is refused. Moved: the other in-use games' truth is rebuilt on
the CPU detector before any cross-arena tuning. Not moved: the MPS truth stays in use beside the CPU truth, with the
platform recorded per game. gsw_ind_2016 is refused here: it stays apart (D98).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from tools.p8_common import atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_moved"
BARS = {"reproj_px_abs_min": 1.0, "accepted_rel_min": 0.10, "offset_median_s_min": 0.2}
OVER_S = 0.2
APART = ("gsw_ind_2016",)


def offset_diffs(mps: dict, cpu: dict) -> list:
    """{window: offset} of each root -> |difference| over windows with an offset in both, sorted by window."""
    return [abs(cpu[w] - mps[w]) for w in sorted(set(mps) & set(cpu)) if mps[w] is not None and cpu[w] is not None]


def compare(mps_result: dict, cpu_result: dict, mps_off: dict, cpu_off: dict, bars: dict = BARS) -> dict:
    """The three measures, each against its bar, and the moved verdict (unrounded values decide)."""
    for name, r in (("MPS", mps_result), ("CPU", cpu_result)):
        if r.get("partial") is not False:
            raise SystemExit(f"{WHO}: the {name} fit is not complete (partial is not False)")
        if r.get("status") == "camera_fit_failed":
            raise SystemExit(f"{WHO}: the {name} camera fit failed")
    rp_m, rp_c = mps_result["camera"]["reproj_px_median_after"], cpu_result["camera"]["reproj_px_median_after"]
    acc_m, acc_c = mps_result["accepted"], cpu_result["accepted"]
    d = offset_diffs(mps_off, cpu_off)
    med = statistics.median(d) if d else None
    rows = {
        "reprojection": {"mps_px": rp_m, "cpu_px": rp_c, "abs_diff_px": abs(rp_c - rp_m),
                         "moved": abs(rp_c - rp_m) >= bars["reproj_px_abs_min"]},
        "accepted": {"mps": acc_m, "cpu": acc_c, "rel_diff": (acc_c - acc_m) / acc_m if acc_m else None,
                     "moved": bool(acc_m) and abs(acc_c - acc_m) >= bars["accepted_rel_min"] * acc_m},
        "offsets": {"windows_both": len(d), "windows_mps": len(mps_off), "windows_cpu": len(cpu_off),
                    "median_abs_diff_s": med, "share_over_0_2_s": (sum(x > OVER_S + 1e-9 for x in d) / len(d)) if d else None,
                    "max_abs_diff_s": max(d) if d else None,
                    "moved": med is not None and med >= bars["offset_median_s_min"]}}
    if not d:
        raise SystemExit(f"{WHO}: no window has an offset in both roots")
    moved = any(r["moved"] for r in rows.values())
    return {"bars": bars, "measures": rows, "moved": moved,
            "reading": ("moved: rebuild the other in-use games' truth on the CPU detector before any cross-arena tuning"
                        if moved else "not moved: the MPS truth stays in use beside CPU truth, platform recorded per game")}


def run(game: str, mps_root: Path, cpu_root: Path) -> dict:
    from tools.p8_device import record_of
    from tools.p8_truth_accept import platform_ok, platform_tally
    if game in APART:
        raise SystemExit(f"{WHO}: {game} stays apart (D98)")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    mps_root, cpu_root = check_outside(Path(mps_root).expanduser()), check_outside(Path(cpu_root).expanduser())
    if mps_root == cpu_root:
        raise SystemExit(f"{WHO}: the two roots are the same folder")

    def load(root, name):
        return json.loads((root / "labels" / f"{game}_{name}.json").read_text())
    stems = sorted({p.name[:-len("_frames.json")] for p in (cpu_root / "label_build").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_frames.json")})
    recs = [record_of(cpu_root / "label_build", s) for s in stems]
    if not platform_ok(recs):
        raise SystemExit(f"{WHO}: not every CPU-root label window resolves the detector to cpu")
    out = {"game": game, "design": "docs/research/P8_truth_arenas.md section 0 (platform mix)", "report_only": True,
           **compare(load(mps_root, "result"), load(cpu_root, "result"), load(mps_root, "offsets"), load(cpu_root, "offsets")),
           "cpu_platform": platform_tally(recs), "cpu_label_windows": len(stems)}
    d = cpu_root / "truth_moved"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{game}.json", json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("game", "moved", "reading")}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    ap.add_argument("game")
    ap.add_argument("--mps-root", required=True)
    ap.add_argument("--cpu-root", required=True)
    a = ap.parse_args(argv)
    run(a.game, Path(a.mps_root), Path(a.cpu_root))


if __name__ == "__main__":
    sys.exit(main())

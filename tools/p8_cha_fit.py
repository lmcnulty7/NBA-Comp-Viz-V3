"""tools/p8_cha_fit.py: why gsw_cha_2016's camera fit broke with the reset-fixed label builds
(docs/research/P8_cha_fit.md). Report only: nothing is retuned and no truth changes.

    python3 -m tools.p8_cha_fit stage1        stored files only (no fit, no model, no video decode, no lock)
        -> <reference root>/cha_fit/stage1.json and reports/research/p8_cha_fit.json

Stage 1 measurements (the design's section 4, fixed before any number):
- S1 arena consistency: C, C0 and reprojection of every recorded camera fit of the in-use games (P8 workspace, MPS;
  reference root, CPU without the fix where it exists, under superseded_nofix; reference root, CPU with the fix), and
  gsw_ind_2016's MPS fit for reference; each C's distance from the median C of the other GSW home fits.
- S2 court identity: for every label window and frame in both gsw_cha_2016 label builds, whether the per-frame court H
  is identical; where not, the change in the closed-form camera decomposition (sportvu.camera.pose_from_h) position.
- S3 where the change sits: per frame, boxes in the sidecar and frames with at least 6 boxes, before and after the
  window's first reset (the fixed build's reset log; before it both builds run the same code), in both builds.
- S4 what the truth kept: accepted rows of both truths (count, feet_resid_ft, line_matches) before and after the first
  reset, and the windows with the largest loss and gain.
- S5 C0 against C for both gsw_cha_2016 fits.
Stage 2 (refits) is not in this tool until the lead approves it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_cha_fit"
GAME = "gsw_cha_2016"
GSW = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
OTHERS = ("hou_cle_2016",)
REFERENCE_ONLY = ("gsw_ind_2016",)          # apart (D98); its MPS fit is read for reference only
MIN_BOXES = 6
W_PX, H_PX = 1280, 720


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def first_reset(log) -> int | None:
    rs = (log or {}).get("resets") or []
    frames = [r["frame"] for r in rs if r.get("frame") is not None]
    return min(frames) if frames else None


def side_of(frame: int, first: int | None) -> str:
    return "before" if first is None or frame < first else "after"


def stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p25": round(float(np.percentile(v, 25)), 3),
            "p75": round(float(np.percentile(v, 75)), 3), "mean": round(float(v.mean()), 3)}


def distance_to(c, others) -> float | None:
    if not others:
        return None
    return round(float(np.linalg.norm(np.asarray(c, float) - np.median(np.asarray(others, float), axis=0))), 2)


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def roots():
    base = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
    return check_outside(base / "work"), check_outside(base / "work_ref")


def fits(work: Path, ref: Path) -> dict:
    out = {}
    for g in GSW + OTHERS + REFERENCE_ONLY:
        for name, p in (("mps", work / "labels" / f"{g}_result.json"),
                        ("cpu_nofix", ref / "superseded_nofix" / "labels" / f"{g}_result.json"),
                        ("cpu_fix", ref / "labels" / f"{g}_result.json")):
            if g in REFERENCE_ONLY and name != "mps":
                continue
            if p.exists():
                r = json.loads(p.read_text())
                c = r.get("camera") or {}
                out[f"{g}:{name}"] = {"game": g, "fit": name, "C": c.get("C"), "C0": c.get("C0"),
                                      "reproj_px_after": c.get("reproj_px_median_after"),
                                      "reproj_px_before": c.get("reproj_px_median_before"),
                                      "inits": c.get("inits"), "pooled": c.get("pooled_frames"),
                                      "accepted": r.get("accepted"), "status": c.get("status")}
    for k, v in out.items():
        if v["game"] in GSW and v["C"]:
            peers = [o["C"] for kk, o in out.items() if kk != k and o["game"] in GSW and o["C"]
                     and not (o["game"] == GAME and o["fit"] == "cpu_fix")]
            v["dist_ft_to_median_C_of_other_gsw_fits_excluding_cha_fix"] = distance_to(v["C"], peers)
    return out


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def windows(lb: Path) -> list:
    return sorted(p.name[:-len("_frames.json")] for p in lb.glob(f"{GAME}_s[0-9][0-9]_w[0-9][0-9]_frames.json"))


def stage1() -> dict:
    from sportvu.camera import pose_from_h
    refuse_cv_test([GAME], WHO)
    guard([GAME], WHO)
    work, ref = roots()
    lb_fix, lb_old = ref / "label_build", ref / "superseded_nofix" / "label_build"
    stems = sorted(set(windows(lb_fix)) | set(windows(lb_old)))
    guard(stems, WHO)
    s2 = {"frames_both": 0, "H_identical": 0, "H_differs": 0, "H_none_both": 0, "H_missing_one_side": 0, "dC_ft": []}
    s3 = {b: {s: {"boxes": [], "frames_ge6": 0, "frames": 0} for s in ("before", "after")} for b in ("nofix", "fix")}
    s4 = {b: {s: {"feet_resid_ft": [], "line_matches": [], "n": 0} for s in ("before", "after")} for b in ("nofix", "fix")}
    per_window = {}
    for s in stems:
        log = load(lb_fix / f"{s}_resetfix.json")
        first = first_reset(log)
        fx, ol = load(lb_fix / f"{s}_frames.json"), load(lb_old / f"{s}_frames.json")
        rows_f = {int(r["frame"]): r for r in (fx or {}).get("frames", [])}
        rows_o = {int(r["frame"]): r for r in (ol or {}).get("frames", [])}
        for f in sorted(set(rows_f) & set(rows_o)):
            a, b = rows_o[f].get("H"), rows_f[f].get("H")
            s2["frames_both"] += 1
            if a is None or b is None:
                s2["H_missing_one_side" if (a is None) != (b is None) else "H_none_both"] += 1
                continue
            if np.allclose(a, b, rtol=0, atol=1e-9):
                s2["H_identical"] += 1
            else:
                s2["H_differs"] += 1
                pa, pb = pose_from_h(a, W_PX, H_PX), pose_from_h(b, W_PX, H_PX)
                if pa is not None and pb is not None:
                    s2["dC_ft"].append(float(np.linalg.norm(np.asarray(pa[2]) - np.asarray(pb[2]))))
        for build, rows in (("nofix", rows_o), ("fix", rows_f)):
            for f, r in rows.items():
                c = s3[build][side_of(f, first)]
                n = len(r.get("boxes") or [])
                c["boxes"].append(n); c["frames"] += 1; c["frames_ge6"] += int(n >= MIN_BOXES)
        acc = {}
        for build, root in (("nofix", ref / "superseded_nofix" / "labels" / GAME), ("fix", ref / "labels" / GAME)):
            p = root / f"{s}.jsonl"
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
            acc[build] = len(rows)
            for r in rows:
                c = s4[build][side_of(int(r["frame"]), first)]
                c["n"] += 1; c["feet_resid_ft"].append(r.get("feet_resid_ft")); c["line_matches"].append(r.get("line_matches"))
        per_window[s] = {"first_reset_frame": first, "resets_fix": len((log or {}).get("resets") or []),
                         "accepted_nofix": acc["nofix"], "accepted_fix": acc["fix"]}
    f = fits(work, ref)
    old, new = f.get(f"{GAME}:cpu_nofix", {}), f.get(f"{GAME}:cpu_fix", {})
    s5 = {"nofix": {"C0": old.get("C0"), "C": old.get("C")}, "fix": {"C0": new.get("C0"), "C": new.get("C")},
          "C0_shift_ft": distance_to(new.get("C0"), [old.get("C0")]) if old.get("C0") and new.get("C0") else None,
          "C_shift_ft": distance_to(new.get("C"), [old.get("C")]) if old.get("C") and new.get("C") else None}
    deltas = sorted(per_window.items(), key=lambda kv: kv[1]["accepted_fix"] - kv[1]["accepted_nofix"])
    out = {"what": "docs/research/P8_cha_fit.md stage 1; report only; stored files", "game": GAME,
           "S1_fits": f,
           "S2_court": {**{k: v for k, v in s2.items() if k != "dC_ft"}, "dC_ft": stats(s2["dC_ft"])},
           "S3_boxes": {b: {s: {"frames": c["frames"], "frames_ge6_boxes": c["frames_ge6"], "boxes_per_frame": stats(c["boxes"])}
                            for s, c in v.items()} for b, v in s3.items()},
           "S4_accepted": {b: {s: {"n": c["n"], "feet_resid_ft": stats(c["feet_resid_ft"]), "line_matches": stats(c["line_matches"])}
                               for s, c in v.items()} for b, v in s4.items()},
           "S4_windows_largest_loss": [{"window": k, **v} for k, v in deltas[:8]],
           "S4_windows_largest_gain": [{"window": k, **v} for k, v in deltas[-5:]],
           "S5_C0_vs_C": s5,
           "windows": len(stems), "windows_with_a_reset_fix": sum(1 for v in per_window.values() if v["resets_fix"])}
    text = json.dumps(out, indent=1) + "\n"
    d = ref / "cha_fit"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "stage1.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_cha_fit.json", text)
    print(text)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    ap.parse_args(argv)
    stage1()


if __name__ == "__main__":
    sys.exit(main())

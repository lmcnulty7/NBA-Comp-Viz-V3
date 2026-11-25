"""tools/p8_truth_accept.py: the acceptance checks of a new game's camera truth (docs/research/P8_truth_arenas.md
section 5, bars fixed before any build). Report only; it runs no model and changes no truth.

    python3 -m tools.p8_truth_accept TAG       -> <P8 root>/truth_accept/<tag>.json

Reads, in the P8 root of the truth build (V3_P8_ROOT): labels/<tag>_result.json (the frozen fit's result),
sync/<tag>_s*_local_timemap.json (mapped running clock), truth_v2/dev/<tag>.json (tools.p8_truth_v2 measure-dev, the
frozen offset stability between the V3 label build and the r23 rebuild) and the device record of every label window and
every r23 rebuild window (the truth platform).

A. Registration: a complete fit (result "partial" False), status ok, camera.reproj_px_median_after at most 10.0 px.
B. Coverage: a complete fit, at least 1,000 accepted camera-truth frames and at least 1,500 s of mapped running clock.
C. Offset stability (D166, frozen rule): over windows where both builds give an offset, median absolute offset
   difference at most 0.54 s (gsw_cha_2016) and share over 0.2 s at most 0.7945 (gsw_sac_2015), the worst in-use pooled
   game of P8_truth_v2.md (version 3: the apart game gsw_ind_2016 is not a bar), with those windows at least 0.8 of the
   windows measured (registered records: 87 percent or more). The frozen rule moves 73 to 80 percent of windows by more
   than 0.2 s even on good games, so C catches only gross instability.
Platform: every label window and every r23 rebuild window must resolve the detector to cpu (D175), else "unused".
Use: platform ok and all three pass -> "use"; platform ok, B passes, A or C fails -> "apart" (reported apart and
flagged, as gsw_ind_2016 is, D98; never used to tune); otherwise -> "unused". The bars never change after a game is
seen (D98).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.p8_common import atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_accept"
BARS = {"reproj_px_max": 10.0, "accepted_min": 1000, "mapped_s_min": 1500.0, "offset_median_s_max": 0.54,
        "offset_share_max": 0.7945, "offset_n_share_min": 0.8}


def check_a(result: dict, bars: dict = BARS) -> dict:
    cam = result.get("camera") or {}
    px = cam.get("reproj_px_median_after")
    ok = result.get("partial") is False and result.get("status") != "camera_fit_failed" \
        and cam.get("status", "ok") == "ok" and px is not None and px <= bars["reproj_px_max"]
    return {"pass": bool(ok), "partial": result.get("partial"), "reproj_px_median_after": px, "bar": bars["reproj_px_max"]}


def check_b(result: dict, mapped_s: float, bars: dict = BARS) -> dict:
    acc = result.get("accepted")
    ok = result.get("partial") is False and acc is not None and acc >= bars["accepted_min"] \
        and mapped_s >= bars["mapped_s_min"]
    return {"pass": bool(ok), "partial": result.get("partial"), "accepted": acc, "mapped_s": round(mapped_s, 1),
            "bars": {"accepted_min": bars["accepted_min"], "mapped_s_min": bars["mapped_s_min"]}}


def check_c(stability: dict, windows_measured: int, bars: dict = BARS) -> dict:
    """stability: the frozen rule's tools.p8_truth_v2.stability record (n, median_abs_s, share_over_0_2_s), unrounded
    bars; windows_measured: the windows in the measure-dev record (n must be at least 0.8 of them)."""
    med, share, n = stability.get("median_abs_s"), stability.get("share_over_0_2_s"), stability.get("n") or 0
    enough = windows_measured > 0 and n >= bars["offset_n_share_min"] * windows_measured
    ok = enough and med is not None and share is not None \
        and med <= bars["offset_median_s_max"] and share <= bars["offset_share_max"]
    return {"pass": bool(ok), "n": n, "windows_measured": windows_measured, "median_abs_s": med,
            "share_over_0_2_s": share,
            "bars": {k: bars[k] for k in ("offset_median_s_max", "offset_share_max", "offset_n_share_min")}}


def verdict(a: dict, b: dict, c: dict, platform_ok: bool = True) -> str:
    if not platform_ok or not b["pass"]:
        return "unused"
    return "use" if a["pass"] and c["pass"] else "apart"


def platform_ok(records: list) -> bool:
    """Every window has a device record whose resolved detector is cpu (D175); none may be missing."""
    return bool(records) and all(r and (r.get("resolved") or {}).get("detector") == "cpu" for r in records)


def platform_tally(records: list) -> dict:
    """Device records of the label windows -> {resolved devices as text: count}; a window without one counts as
    "no record" (built with the frozen defaults)."""
    out = {}
    for r in records:
        k = json.dumps((r or {}).get("resolved"), sort_keys=True) if r else "no record"
        out[k] = out.get(k, 0) + 1
    return out


def accept(tag: str, root: Path | None = None) -> dict:
    from tools.p8_device import record_of
    refuse_cv_test([tag], WHO)
    guard([tag], WHO)
    root = Path(root or p8_root())
    result = json.loads((root / "labels" / f"{tag}_result.json").read_text())
    mapped = sum(json.loads(p.read_text()).get("mapped_video_s", 0)
                 for p in sorted((root / "sync").glob(f"{tag}_s[0-9][0-9]_local_timemap.json")))
    dev = root / "truth_v2" / "dev" / f"{tag}.json"
    if not dev.exists():
        raise SystemExit(f"{WHO}: {tag} has no truth_v2/dev record; run tools.p8_truth_v2 measure-dev first")
    dev_rec = json.loads(dev.read_text())
    stability = dev_rec["summary"]["frozen"]

    def stems(folder):
        return sorted({p.name[:-len("_frames.json")] for p in folder.glob(f"{tag}_s[0-9][0-9]_w[0-9][0-9]_frames.json")})
    label = [record_of(root / "label_build", s) for s in stems(root / "label_build")]
    rebuild = [record_of(root / "builds" / "r23", s) for s in stems(root / "builds" / "r23")]
    plat = platform_ok(label) and platform_ok(rebuild)
    a, b, c = check_a(result), check_b(result, mapped), check_c(stability, len(dev_rec.get("windows", {})))
    out = {"tag": tag, "design": "docs/research/P8_truth_arenas.md sections 0 and 5", "checks": {"A": a, "B": b, "C": c},
           "platform_ok": plat, "use": verdict(a, b, c, plat),
           "truth_platform": {"label": platform_tally(label), "r23_rebuild": platform_tally(rebuild)},
           "label_windows": len(label), "rebuild_windows": len(rebuild)}
    d = root / "truth_accept"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{tag}.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    ap.add_argument("tag")
    a = ap.parse_args(argv)
    accept(a.tag)


if __name__ == "__main__":
    sys.exit(main())

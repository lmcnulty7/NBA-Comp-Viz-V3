"""tools/p8_recallcause_aligndry.py: the dropped windows of the recall-cause run, and a dry run of a proposed
alignment rule on stored records (docs/research/P8_recallcause.md section 9, post hoc). Report only: no model runs,
no frame is read, nothing is fitted; it reads two named records and writes one.

    python3 -m tools.p8_recallcause_aligndry   -> reports/research/p8_recallcause_aligndry.json

Inputs: reports/research/p8_recallcause.json (the run's alignment shares and dropped windows) and
reports/research/p8_a3align.json (per window, S_k for processed-frame offsets k and its own verdict).

Rules (the registered one and the proposal), on (before, at, after) levels:
- registered (8.10): aligned only when at > before, at > after and at >= 0.9; ties read not aligned;
- proposed: misaligned only when a neighbour beats at by more than MARGIN, or at is under FLOOR; ties aligned.
On p8_a3align.json the levels are a3align's own paired shares: for k = -1 and +1, S_k against S_0 of the same
comparison (frames where k falls outside the window are left out of both), and at = S_0 of offset 0. They are shares
of raw boxes matched at IoU 0.5, not the proposal's mean IoU, which neither record stores; so this dry run checks the
margin, floor and tie logic, not the statistic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WHO = "tools.p8_recallcause_aligndry"
GAMES = ("gsw_cha_2016", "gsw_bkn_2015")
BAR = 0.9
MARGIN = 0.05
FLOOR = 0.75


def strict_ok(before: float, at: float, after: float) -> bool:
    return at > before and at > after and at >= BAR


def proposed_ok(before: float, at: float, after: float, margin: float = MARGIN, floor: float = FLOOR) -> bool:
    return not (max(before, after) > at + margin or at < floor)


def fails_on(before: float, at: float, after: float) -> str | None:
    """Why the registered rule drops a window (its mean shares): "at_under_bar" (at under 0.9), "neighbour_above"
    (at 0.9 or more, a neighbour higher), "tie" (at 0.9 or more, a neighbour equal, none higher); None if kept."""
    if strict_ok(before, at, after):
        return None
    if at < BAR:
        return "at_under_bar"
    return "neighbour_above" if max(before, after) > at else "tie"


def a3align_levels(v: dict):
    """(before, at, after) pairs from one a3align window: [(S_-1, S_0 paired), (S_+1, S_0 paired)] and S_0 at 0."""
    b = v["by_offset"]
    if any(b.get(k) is None for k in ("-1", "0", "1")):
        return None
    return {"pairs": [(b["-1"]["Sk"], b["-1"]["S0"]), (b["1"]["Sk"], b["1"]["S0"])], "at": b["0"]["Sk"]}


def a3align_strict(lv: dict) -> bool:
    return all(s0 > sk for sk, s0 in lv["pairs"]) and lv["at"] >= BAR


def a3align_proposed(lv: dict, margin: float = MARGIN, floor: float = FLOOR) -> bool:
    return not (any(sk > s0 + margin for sk, s0 in lv["pairs"]) or lv["at"] < floor)


def a3align_tie(lv: dict) -> bool:
    return any(sk == s0 for sk, s0 in lv["pairs"])


def measure(rc: dict, a3: dict) -> dict:
    from tools.p8_common import guard
    from tools.research_testset import refuse_cv_test
    aw = {s: v for g in GAMES for s, v in a3["games"][g]["windows"].items()}
    dropped, align = rc["windows_dropped"], rc["alignment"]
    names = sorted(set(aw) | set(align)) + list(GAMES)
    refuse_cv_test(names, WHO)
    guard(names, WHO)
    out = {"games": {}, "a3align_dry_run": {}}
    why = {s: fails_on(*np.asarray(align[s]["shares"], float).mean(0)) for s in dropped}
    if any(v is None for v in why.values()):
        raise SystemExit(f"{WHO}: a dropped window passes the registered rule on its stored shares")
    tot = sum(dropped.values())
    out["dropped_by_reason"] = {r: {"windows": sum(1 for s in dropped if why[s] == r),
                                    "misses": int(sum(dropped[s] for s in dropped if why[s] == r)),
                                    "share_of_dropped_misses": (sum(dropped[s] for s in dropped if why[s] == r) / tot)
                                    if tot else None}
                                for r in ("tie", "at_under_bar", "neighbour_above")}
    nfr = [int(align[s]["frames"]) for s in dropped]
    out["dropped_alignment_frames"] = {str(k): nfr.count(k) for k in sorted(set(nfr))}
    units = rc.get("units") or {}
    if "both__pooled" in units:
        kept_n = units["both__pooled"]["misses"]
        c1 = units["both__pooled"]["groups"]["C1"]["count"]
        out["c1_worst_case_both"] = {"kept_misses": kept_n, "all_misses": kept_n + tot, "kept_c1": c1,
                                     "c1_if_no_dropped_miss_is_c1": c1 / (kept_n + tot)}
    for g in GAMES:
        ws = sorted(s for s in align if s.startswith(g + "_"))
        dr = [s for s in ws if s in dropped]
        kept = [s for s in ws if s not in dropped]
        cov = [s for s in dr if s in aw]
        kcov = [s for s in kept if s in aw]
        own_kept = [s for s in dr if proposed_ok(*np.asarray(align[s]["shares"], float).mean(0))]
        own_newdrop = [s for s in kept if not proposed_ok(*np.asarray(align[s]["shares"], float).mean(0))]
        kept_m = units[f"both__{g}"]["misses"] if f"both__{g}" in units else None
        drop_m = int(sum(dropped[s] for s in dr))
        out["games"][g] = {
            "windows_with_misses": len(ws), "dropped": len(dr), "dropped_misses": drop_m, "kept_misses": kept_m,
            "dropped_share_of_misses": (drop_m / (drop_m + kept_m)) if kept_m is not None else None,
            "dropped_in_a3align": len(cov), "dropped_a3align_aligned": sum(not aw[s]["misaligned"] for s in cov),
            "kept": len(kept), "kept_in_a3align": len(kcov),
            "kept_a3align_misaligned": sum(bool(aw[s]["misaligned"]) for s in kcov),
            "own_shares_proposed": {"dropped_now_kept": len(own_kept),
                                    "their_misses": int(sum(dropped[s] for s in own_kept)),
                                    "kept_now_dropped": len(own_newdrop)},
            "dropped_windows": {s: {"misses": dropped[s], "fails_on": why[s], "alignment_frames": align[s]["frames"],
                                    "mean_shares": [round(float(x), 4) for x in np.asarray(align[s]["shares"], float).mean(0)],
                                    "a3align": (None if s not in aw else ("misaligned" if aw[s]["misaligned"] else "aligned"))}
                                for s in dr}}
    for g in GAMES:                                     # the pool games only; sac stays apart
        G = a3["games"][g]
        lv = {s: a3align_levels(v) for s, v in G["windows"].items()}
        lv = {s: x for s, x in lv.items() if x is not None}
        al = [s for s in lv if not G["windows"][s]["misaligned"]]
        mis = [s for s in lv if G["windows"][s]["misaligned"]]
        out["a3align_dry_run"][g] = {
            "aligned_windows": len(al), "misaligned_windows": len(mis),
            "strict_fails_on_aligned": sum(not a3align_strict(lv[s]) for s in al),
            "ties_on_aligned": sum(a3align_tie(lv[s]) for s in al),
            "proposed_fails_on_aligned": sum(not a3align_proposed(lv[s]) for s in al),
            "strict_flags_misaligned": sum(not a3align_strict(lv[s]) for s in mis),
            "proposed_flags_misaligned": sum(not a3align_proposed(lv[s]) for s in mis)}
    out["rule"] = {"registered": {"bar": BAR, "ties": "not aligned"},
                   "proposed": {"margin": MARGIN, "floor": FLOOR, "ties": "aligned"},
                   "levels": "shares of boxes matched at IoU 0.5 (both records); not the proposal's mean IoU"}
    out["what"] = ("Post hoc: the recall-cause run's dropped windows against a3align's verdicts, and a dry run of a "
                   "proposed alignment rule on stored records (docs/research/P8_recallcause.md section 9); report only")
    return out


def main() -> None:
    from tools.p8_common import REPO, atomic_write
    from export.schemas import find_private
    rd = REPO / "reports" / "research"
    out = measure(json.loads((rd / "p8_recallcause.json").read_text()), json.loads((rd / "p8_a3align.json").read_text()))
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(rd / "p8_recallcause_aligndry.json", text)
    print(json.dumps({"games": {g: {k: v for k, v in x.items() if k != "dropped_windows"} for g, x in out["games"].items()},
                      "a3align_dry_run": out["a3align_dry_run"]}, indent=1))


if __name__ == "__main__":
    main()

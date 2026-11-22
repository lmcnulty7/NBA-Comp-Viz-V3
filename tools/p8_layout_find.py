"""tools/p8_layout_find.py: an automatic scorebug layout finder (PLAN.md D152; D128 rules out a hand layout).

    <frozen CV python> -m tools.p8_layout_find validate     the development videos whose layouts already exist
    <frozen CV python> -m tools.p8_layout_find find TAG     propose a layout for one development game and check it
    <frozen CV python> -m tools.p8_layout_find register TAG write an accepted proposal into the P8 registry

Run each under the heavy-job lock (python3 -m export.heavy --name ... -- ...): it runs OCR on video frames.

What it reads: only video frames of development games (tools.p8_frozen.game_of: refuse_cv_test and the spent
held-out guard on the tag, SportVU id and video id before any frame is read). No tracking truth, no scorecard row,
no 2025-26 game.

## The rule, fixed before any run (2026-11-22)

Proposal. FIND_FRAMES frames spread over the game's middle sections (the first and last sections skipped, as
sportvu.prepare.sample_frames does) at (k + 0.25) / per of each section, so they are not the frozen check's own frames
(which sit at (k + 0.5) / per). easyocr reads each whole frame. A clock token is text that parses as M:SS (minutes
at most 12, seconds at most 59) or SS.T; a period token starts with an ordinal (1ST, 2ND, 3RD, 4TH, OT, OCR's I or L
for 1 allowed) or an ordinal word (FIRST to FOURTH). Tokens are grouped by box centre across frames (within
CLUSTER_PX). The clock box comes from the clock group seen in the most frames (it must be seen in at least
MIN_FRAME_SHARE of them); the period box from the period group seen in the most frames among those on the clock's row
(centres within one text height vertically, within MAX_DX_PX horizontally), else none. A period token longer than its
ordinal ("1ST QTR") is cut to the ordinal's share of its width, because the frozen reader turns "QTR" into OT. Boxes:
the 10th to 90th percentile of the group's corners, padded by PAD_FRAC of the median text height on every side.

Acceptance. The proposal passes only through the frozen automatic check, tools.p8_frozen.step_layout (sportvu.prepare's
scoring: 40 frame pairs 1 s apart, plausible share and consistent share at least MIN_LAYOUT_SHARE 0.25), run with the
proposal as the only candidate layout. Its outputs go to the finder's own folder, never to meta/.

Validation (before any new game). On every development video whose layout already exists, the proposal must have
IoU at least 0.5 with the existing clock box and with the existing period box (against the primary position or an
alternate, whichever is higher) and must pass the frozen check. The existing layout is also run through the same
check as a control. If any video fails, NOP is dropped (D152).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tools.p8_common import REPO, atomic_write, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_layout_find"
FIND_FRAMES = 40
CLUSTER_PX = 14
MIN_FRAME_SHARE = 0.25
MAX_DX_PX = 300
PAD_FRAC = 0.25
IOU_BAR = 0.5
CHECK_NAME = "proposal"

# development videos with an existing layout: tag -> where that layout comes from
VALIDATION = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015", "nyk_bos_2016", "okc_sac_2015",
              "por_okc_2016", "hou_cle_2016")

CLOCK_MSS = re.compile(r"^(\d{1,2})[:.](\d{2})$")
CLOCK_SST = re.compile(r"^(\d{1,2})\.(\d)$")
PERIOD_ORD = re.compile(r"^([1-4IL])(ST|ND|RD|TH)\b|^(OT)\b", re.I)
PERIOD_WORD = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH)\b", re.I)


# ── pure parts (tested without OCR) ─────────────────────────────────────────────────────────────────────────────────
def clock_token(text: str) -> bool:
    t = text.replace(" ", "")
    m = CLOCK_MSS.match(t)
    if m:
        return int(m.group(1)) <= 12 and int(m.group(2)) <= 59
    m = CLOCK_SST.match(t)
    return bool(m) and int(m.group(1)) < 60


def period_token(text: str):
    """-> (kind, share of the text width the ordinal takes) or None. kind: ordinal or word."""
    t = text.strip()
    m = PERIOD_ORD.match(t)
    if m:
        return "ordinal", len(m.group(0)) / max(len(t), 1)
    m = PERIOD_WORD.match(t)
    if m:
        return "word", 1.0
    return None


def box_of(pts) -> list:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def clusters(tokens: list, px: float = CLUSTER_PX) -> list:
    """tokens: dicts with frame and box -> groups of tokens whose box centres lie within px of the group's first
    member, largest first by distinct frames."""
    groups = []
    for t in tokens:
        cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
        for g in groups:
            if abs(g["cx"] - cx) <= px and abs(g["cy"] - cy) <= px:
                g["members"].append(t)
                break
        else:
            groups.append({"cx": cx, "cy": cy, "members": [t]})
    for g in groups:
        g["frames"] = len({t["frame"] for t in g["members"]})
    return sorted(groups, key=lambda g: -g["frames"])


def _pct(vals, q):
    vals = sorted(vals)
    k = (len(vals) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def group_box(members: list, w: int, h: int) -> list:
    """10th to 90th percentile of the corners, padded by PAD_FRAC of the median text height, clipped to the frame,
    as integers."""
    x1 = _pct([m["box"][0] for m in members], 10)
    y1 = _pct([m["box"][1] for m in members], 10)
    x2 = _pct([m["box"][2] for m in members], 90)
    y2 = _pct([m["box"][3] for m in members], 90)
    th = _pct([m["box"][3] - m["box"][1] for m in members], 50)
    pad = PAD_FRAC * th
    return [max(0, int(round(x1 - pad))), max(0, int(round(y1 - pad))),
            min(w, int(round(x2 + pad))), min(h, int(round(y2 + pad)))]


def propose(reads: list, n_frames: int, w: int, h: int) -> dict:
    """reads: [(frame index, box, text)] from whole-frame OCR -> a layout proposal or a reason it has none."""
    clock, period = [], []
    for f, box, text in reads:
        if clock_token(text):
            clock.append({"frame": f, "box": box, "text": text})
            continue
        p = period_token(text)
        if p:
            kind, share = p
            b = list(box)
            if share < 1.0:
                b[2] = b[0] + (b[2] - b[0]) * share
            period.append({"frame": f, "box": b, "text": text, "kind": kind})
    cg = clusters(clock)
    if not cg or cg[0]["frames"] < MIN_FRAME_SHARE * n_frames:
        return {"status": "no_clock", "clock_groups": [(round(g["cx"]), round(g["cy"]), g["frames"]) for g in cg[:5]]}
    c = cg[0]
    th = _pct([m["box"][3] - m["box"][1] for m in c["members"]], 50)
    row = [g for g in clusters(period) if abs(g["cy"] - c["cy"]) <= th and abs(g["cx"] - c["cx"]) <= MAX_DX_PX]
    if not row:
        return {"status": "no_period", "clock_box": group_box(c["members"], w, h), "clock_frames": c["frames"]}
    p = row[0]
    words = sum(m["kind"] == "word" for m in p["members"]) > len(p["members"]) / 2
    return {"status": "ok", "clock_box": group_box(c["members"], w, h), "period_box": group_box(p["members"], w, h),
            "period_words": bool(words), "clock_frames": c["frames"], "period_frames": p["frames"],
            "frames": n_frames}


def overlap_with(proposal: dict, layout: dict) -> dict:
    """IoU of the proposal's clock and period boxes with an existing layout's, best over its primary position and its
    alternates."""
    positions = [(layout["period_box"], layout["clock_box"])] + [tuple(a) for a in layout.get("alternates", [])]
    best = max(positions, key=lambda pc: iou(proposal["clock_box"], pc[1]) + iou(proposal["period_box"], pc[0]))
    return {"clock_iou": round(iou(proposal["clock_box"], best[1]), 3),
            "period_iou": round(iou(proposal["period_box"], best[0]), 3)}


def validation_passes(ov: dict, check: dict) -> bool:
    return ov["clock_iou"] >= IOU_BAR and ov["period_iou"] >= IOU_BAR and check.get("chosen") == CHECK_NAME


# ── OCR and the frozen check (frozen CV python, inside the lock) ────────────────────────────────────────────────────
def find_frames(tag: str, n: int = FIND_FRAMES) -> list:
    import cv2
    from harvest_driver import HARVEST_VIDEO
    secs = sorted(HARVEST_VIDEO.glob(tag + "_s[0-9][0-9].mp4"))
    secs = secs[1:-1] or secs
    per = max(1, n // len(secs))
    out = []
    for vid in secs:
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(per):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.25) / per))
            ok, fr = cap.read()
            if ok:
                out.append(fr)
        cap.release()
    return out


def ocr_reads(frames: list) -> list:
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    reads = []
    for i, fr in enumerate(frames):
        for pts, text, _conf in reader.readtext(fr, detail=1):
            reads.append((i, box_of(pts), text))
    return reads


def frozen_check(P: dict, tag: str, layout: dict, out_dir: Path) -> dict:
    """tools.p8_frozen.step_layout with `layout` as the only candidate (named CHECK_NAME): the frozen scoring and bar,
    written to out_dir instead of meta/."""
    import clock_reader
    from tools import p8_frozen as F
    saved, saved_inject = clock_reader.LAYOUTS, F.inject_layouts
    clock_reader.LAYOUTS = {CHECK_NAME: {k: tuple(map(tuple, v)) if k == "alternates" else tuple(v)
                                         for k, v in layout.items() if k in ("period_box", "clock_box", "alternates")}}
    F.inject_layouts = lambda: None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        F.step_layout({**P, "meta": out_dir}, tag)
        res = json.loads((out_dir / f"{tag}_layout.json").read_text())
    finally:
        clock_reader.LAYOUTS, F.inject_layouts = saved, saved_inject
    return {"chosen": res["chosen"], "plausible_share": res["plausible_share"].get(CHECK_NAME),
            "consistent_share": res["consistent_share"].get(CHECK_NAME), "pairs": res["pairs"]}


def existing_layout(P: dict, tag: str) -> tuple:
    """(name, layout dict) the pipeline uses for a development game today."""
    import clock_reader
    from tools import p8_frozen as F
    F.inject_layouts()
    name = F.layout_of(P, tag)
    reg = json.loads((REPO / "reports" / "research" / "p8_registry.json").read_text())
    lay = dict(clock_reader.LAYOUTS[name])
    if (reg.get("layouts") or {}).get(name, {}).get("period_words"):
        lay["period_words"] = True
    return name, lay


def find(tag: str, P: dict) -> dict:
    from tools import p8_frozen as F
    refuse_cv_test([tag], WHO)
    F.game_of(tag)                                   # refuse_cv_test and the spent held-out guard on its ids
    frames = find_frames(tag)
    if not frames:
        raise SystemExit(f"{WHO}: {tag}: no frames read")
    h, w = frames[0].shape[:2]
    prop = propose(ocr_reads(frames), len(frames), w, h)
    out = {"tag": tag, "proposal": prop}
    if prop["status"] == "ok":
        out["check"] = frozen_check(P, tag, prop, P["root_find"] / "check" / tag)
    return out


def cmd_validate(args) -> dict:
    from tools import p8_frozen as F
    P = F.install_paths()
    P["root_find"] = p8_root() / "layout_find"
    rows = {}
    for tag in VALIDATION:
        r = find(tag, P)
        name, lay = existing_layout(P, tag)
        r["existing"] = {"name": name, "layout": {k: list(v) if k != "alternates" else [list(map(list, a)) for a in v]
                                                  for k, v in lay.items() if k != "period_words"}}
        r["existing_check"] = frozen_check(P, tag, lay, P["root_find"] / "control" / tag)
        if r["proposal"]["status"] == "ok":
            r["overlap"] = overlap_with(r["proposal"], lay)
            r["passes"] = validation_passes(r["overlap"], r["check"])
        else:
            r["passes"] = False
        rows[tag] = r
        print(json.dumps({tag: {k: r.get(k) for k in ("passes", "overlap", "check", "existing_check")}}), flush=True)
    out = {"rule": "D152, docstring of tools/p8_layout_find.py", "videos": rows,
           "all_pass": all(r["passes"] for r in rows.values())}
    atomic_write(P["root_find"] / "validation.json", json.dumps(out, indent=1, default=str))
    print(json.dumps({"all_pass": out["all_pass"], "per_video": {t: r["passes"] for t, r in rows.items()}}, indent=1))
    return out


def cmd_find(args) -> dict:
    from tools import p8_frozen as F
    P = F.install_paths()
    P["root_find"] = p8_root() / "layout_find"
    val = P["root_find"] / "validation.json"
    if not val.exists() or not json.loads(val.read_text()).get("all_pass"):
        raise SystemExit(f"{WHO}: validation has not passed; no new game is looked at (D152)")
    out = find(args.tag, P)
    atomic_write(P["root_find"] / f"{args.tag}.json", json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return out


def cmd_register(args) -> dict:
    """Write an accepted proposal into reports/research/p8_registry.json: layouts[auto_<tag>] and games[tag].layout."""
    root = p8_root() / "layout_find"
    rec = json.loads((root / f"{args.tag}.json").read_text())
    if rec["proposal"].get("status") != "ok" or rec.get("check", {}).get("chosen") != CHECK_NAME:
        raise SystemExit(f"{WHO}: {args.tag}: no accepted proposal")
    refuse_cv_test([args.tag], WHO)
    p = REPO / "reports" / "research" / "p8_registry.json"
    reg = json.loads(p.read_text())
    if args.tag not in reg["games"]:
        raise SystemExit(f"{WHO}: {args.tag} is not a registry game")
    name = f"auto_{args.tag}"
    prop = rec["proposal"]
    reg.setdefault("layouts", {})[name] = {
        "period_box": prop["period_box"], "clock_box": prop["clock_box"],
        **({"period_words": True} if prop.get("period_words") else {}),
        "measured_on": f"found automatically (D152) by tools/p8_layout_find.py on {prop['frames']} frames of {args.tag}; "
                       f"frozen check plausible {rec['check']['plausible_share']}, consistent {rec['check']['consistent_share']}"}
    reg["games"][args.tag]["layout"] = name
    reg["games"][args.tag]["layout_note"] = "found automatically (D152); no frame read by a person"
    p.write_text(json.dumps(reg, indent=1) + "\n")
    print(json.dumps(reg["layouts"][name], indent=1))
    return reg["layouts"][name]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.p8_layout_find")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    f = sub.add_parser("find")
    f.add_argument("tag")
    r = sub.add_parser("register")
    r.add_argument("tag")
    a = ap.parse_args(argv)
    {"validate": cmd_validate, "find": cmd_find, "register": cmd_register}[a.cmd](a)


if __name__ == "__main__":
    main()

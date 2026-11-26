"""tools/p8_a3align.py: frame alignment between the corpus decode (where M4's raw boxes ran) and the build's decode, per
pool window, and the share of the build loss's a3 that falls in misaligned windows (docs/research/P8_a3align.md).
Report only: no model runs, nothing is built or changed.

    python3 -m tools.p8_a3align measure  -> <P8 workspace>/placement/a3align.json and reports/research/p8_a3align.json

What each side records (read before registering): the corpus plan and manifest record, per frame, the window and the
clip frame index (plus the truth row's q and moment); the sidecar records, per processed frame, the clip frame index,
the source clip's file name, fps and stride. Neither side records a decoded timestamp or a content hash. Both indices
count frames of the same window clip (label_build/<window>.mp4): the corpus by sequential grab from frame 0
(tools/p8_e1data.py run), the build by videoseq.SeqReader with the pre-gate jumps (build_trajectories.py). So a direct
index comparison only shows that both name the same index; whether the decoded content at that index is the same
must be inferred from the boxes. The tool records, as a check, that each sidecar names the same clip file and whether
the clip file was modified after the corpus preds were written.

Rules, fixed before any number:
- Windows: every scored window of the pool games (registered score files) that holds corpus frames with M4 raw boxes.
- For each corpus frame f of a window that the sidecar processed, and each offset k in -2, -1, 0, +1, +2 processed
  frames (the sidecar frame k places away from f in the window's sorted processed frames): every M4 r23 raw box at
  0.40 gets its largest IoU with a sidecar box of that frame. S_k = the share of raw boxes whose largest IoU is at least
  0.5, pooled over the window's corpus frames (frames where offset k falls outside the window are left out of S_k and
  of S_0 for that comparison).
- Offset-misaligned window: some k other than 0 has S_k at least S_0 + 0.05.
- Sub-stride suspect window (an offset shorter than one stride, 6 video frames, cannot show as k != 0): among raw boxes
  whose largest same-frame IoU is at least 0.3, the median of that IoU is below 0.75. A tracked box is the detection
  box updated by the Kalman filter, so in an aligned window this median is expected near 0.9.
- Misaligned = offset-misaligned or sub-stride suspect. Aligned otherwise.
- a3: build loss version 2's per-player rows of class a3_never_tracked_nearby with raw foot within 2.5 ft
  (placement/buildloss_v2.json; the tool stops if it is missing).
- Reading rule, on the pool: (i) at least 50 percent of reachable a3 lies in misaligned windows: a3 is mostly a
  measurement artefact of the corpus-to-build frame mismatch; the detection-row upper bound shrinks by that share and
  no build arm follows; (ii) otherwise a3 is mostly in aligned windows and the model run of P8_buildloss.md
  (model.track on those windows' decoded frames, under the lock) is justified and needs its own registration.
  The split between offset-misaligned and sub-stride suspect is reported.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.p8_common import REPO, atomic_write, p8_root
from tools.p8_placement import BUILDS, POOL, SCORES, check_games, rnd
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_a3align"
OFFSETS = (-2, -1, 0, 1, 2)
DET_CONF, IOU_MATCH, IOU_ANY = 0.40, 0.5, 0.3
MARGIN, SUBSTRIDE_IOU, REACH_FT, READ_SHARE = 0.05, 0.75, 2.5, 0.50


def need(path):
    from pathlib import Path
    if not Path(path).exists():
        raise SystemExit(f"{WHO}: required file {Path(path).name} is missing")
    return path


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def best_ious(raw: list, side_boxes: list) -> list:
    return [max((iou(r[:4], b["bbox"]) for b in side_boxes), default=0.0) for r in raw]


def window_alignment(pairs: dict) -> dict:
    """pairs: {k: [(best IoU at offset 0, best IoU at offset k) per raw box, over frames where k exists]}."""
    s = {}
    for k, v in pairs.items():
        if not v:
            s[k] = None
            continue
        a = np.array(v)
        s[k] = {"S0": float(np.mean(a[:, 0] >= IOU_MATCH)), "Sk": float(np.mean(a[:, 1] >= IOU_MATCH)), "n": len(v)}
    offs = [k for k in s if k != 0 and s[k] is not None and s[k]["Sk"] >= s[k]["S0"] + MARGIN]
    same = [x[0] for x in pairs.get(0, []) if x[0] >= IOU_ANY]
    med = float(np.median(same)) if same else None
    sub = med is not None and med < SUBSTRIDE_IOU
    return {"by_offset": s, "offset_misaligned": bool(offs), "best_offsets": offs, "median_same_frame_iou": med,
            "sub_stride_suspect": bool(sub), "misaligned": bool(offs or sub)}


def reading(share_misaligned: float | None) -> str:
    if share_misaligned is None:
        return "no a3"
    return "artefact" if share_misaligned >= READ_SHARE else "model run justified"


# ── run ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_measure(args) -> dict:
    import os
    from tools.p8_e1data import corpus_root
    from tools.p8_frozen import layout
    refuse_cv_test(check_games(POOL), WHO)
    wr = p8_root()
    P = layout(wr)
    v2 = json.loads(need(wr / "placement" / "buildloss_v2.json").read_text())["per_player"]
    preds_mtime = max(os.path.getmtime(p) for p in need(corpus_root() / "preds").glob("chunk_*.json"))
    games, all_a3 = {}, {"misaligned": 0, "offset": 0, "sub_stride": 0, "aligned": 0, "unscored": 0}
    for g in POOL:
        sf = json.loads(need(wr / "e1" / SCORES[g]).read_text())
        m4 = json.loads(need(wr / "e1" / f"m4_preds__{g}.json").read_text())["preds"]
        by_w = {}
        for stem in m4:
            w, f = stem.rsplit("_f", 1)
            by_w.setdefault(w, []).append(int(f))
        res, checks = {}, {"same_clip_name": 0, "windows": 0, "clip_modified_after_corpus": 0}
        for w in sf["scored_windows"]:
            if w not in by_w:
                continue
            side = json.loads(need(P["builds"] / BUILDS[g] / f"{w}_frames.json").read_text())
            checks["windows"] += 1
            checks["same_clip_name"] += side["source"].replace("\\", "/").rsplit("/", 1)[-1] == f"{w}.mp4"
            clip = P["label_build"] / f"{w}.mp4"
            checks["clip_modified_after_corpus"] += bool(clip.exists() and os.path.getmtime(clip) > preds_mtime)
            rows = {int(r["frame"]): r for r in side["frames"]}
            order = sorted(rows)
            pos = {f: i for i, f in enumerate(order)}
            pairs = {k: [] for k in OFFSETS}
            for f in sorted(by_w[w]):
                if f not in pos:
                    continue
                raw = [b for b in m4[f"{w}_f{f:06d}"]["r23"] if b[4] >= DET_CONF]
                if not raw:
                    continue
                b0 = best_ious(raw, rows[f]["boxes"])
                for k in OFFSETS:
                    j = pos[f] + k
                    if 0 <= j < len(order):
                        bk = b0 if k == 0 else best_ious(raw, rows[order[j]]["boxes"])
                        pairs[k].extend(zip(b0, bk))
            res[w] = window_alignment(pairs)
        a3 = [r for r in v2[g] if r["cls"] == "a3_never_tracked_nearby" and r["d_raw"] <= REACH_FT]
        tally = {"misaligned": 0, "offset": 0, "sub_stride": 0, "aligned": 0, "unscored": 0}
        for r in a3:
            a = res.get(r["w"])
            if a is None:
                tally["unscored"] += 1
                continue
            if a["misaligned"]:
                tally["misaligned"] += 1
                tally["offset" if a["offset_misaligned"] else "sub_stride"] += 1
            else:
                tally["aligned"] += 1
        for k in all_a3:
            all_a3[k] += tally[k]
        games[g] = {"windows": res, "checks": checks, "a3_reachable": len(a3), "a3_by_alignment": tally,
                    "misaligned_windows": sorted(w for w, a in res.items() if a["misaligned"]),
                    "windows_measured": len(res)}
        print(g, json.dumps(rnd({"misaligned_windows": games[g]["misaligned_windows"], "a3": tally, "checks": checks})))
    n = all_a3["misaligned"] + all_a3["aligned"]
    share = all_a3["misaligned"] / n if n else None
    out = rnd({"what": "Corpus-to-build frame alignment per pool window and the share of reachable a3 in misaligned "
                       "windows (docs/research/P8_a3align.md); report only; stored files",
               "games": games, "pooled_a3": all_a3, "share_in_misaligned": share, "reading": reading(share),
               "rule": {"offsets": list(OFFSETS), "iou_match": IOU_MATCH, "margin": MARGIN,
                        "sub_stride_median_iou": SUBSTRIDE_IOU, "read_share": READ_SHARE}})
    atomic_write(wr / "placement" / "a3align.json", json.dumps(out, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_a3align.json", json.dumps(out, indent=1))
    print(json.dumps({"pooled_a3": all_a3, "share": share, "reading": out["reading"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_a3align")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    a = ap.parse_args(argv)
    {"measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()

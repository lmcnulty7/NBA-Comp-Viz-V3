#!/usr/bin/env python
"""
triage_sheet.py — visual triage of the pipeline on the harvest games (the "is it even visually right?" check).

Renders the side-by-side review video (broadcast with boxes / ids / court lines ‖ top-down minimap) for a
stratified sample of aligned possessions across the 14 harvest games, running the real pipeline on an h264
snippet of each, then plays each clip for a human who scores it per FAULT:
  1 court template off the painted lines (homography)     2 missed players / ghost boxes (detection)
  3 ids swap or fragment (tracking)                        4 jersey colours flip (teams)
  5 dots glide where nobody is (interpolation / phantoms)
  g / m / b = overall good / minor / broken   ·   r restart playback   ·   SPACE next   ·   q quit
Report: per fault and per game, broken rate by arena, the dominant fault. Nothing here changes the pipeline.

  --sample N   N possessions per game (seeded), from data/pbp/<section>_outcomes.json, status aligned
  --render     ffmpeg snippet (h264, native res) -> build_trajectories.py --pregate (production gate, v1 @ 0.70) -> data/triage/renders/<id>.mp4
  --label      playback reviewer     --report
"""
from __future__ import annotations
import argparse, json, random, shutil, subprocess, time
from collections import defaultdict, Counter
from pathlib import Path
import config
from fetch_pbp import game_for_clip, video_path
from harvest_driver import build_stride

TRI = config.PROJECT_ROOT / "data" / "triage"
SAMPLE, LABELS, SRC, REN = TRI / "sample.json", TRI / "labels.json", TRI / "src", TRI / "renders"
FF, FP = "/opt/homebrew/bin/ffmpeg", "/opt/homebrew/bin/ffprobe"
FAULTS = {"1": "court_off", "2": "boxes", "3": "id_swaps", "4": "team_flips", "5": "gliding"}
PAD_S, MAX_S = 1.5, 8.0      # the clock resolves the offset to ±1 s, so pad generously
# Reproduce PRODUCTION exactly: harvest_driver runs build_trajectories --pregate, which gates every frame with
# v1 at thresholds.json["trained"] = 0.70 (verified 2026-10-02 against the harvest commit b751d18; the "harvest
# runs at 0.35" line in DEVLOG 07-05 was about label_factory.py, the court LABEL factory, not the possession
# harvest). So: no --gate-thr override, --pregate, same stride rule as harvest_driver.build_stride.
GATE_ARGS = ["--pregate"]


def fps_of(path):
    out = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    n, d = out.split("/"); return float(n) / float(d)


def build_sample(per_game, seed):
    rng = random.Random(seed); by = defaultdict(list)
    for f in sorted((config.PROJECT_ROOT / "data" / "pbp").glob("*_outcomes.json")):
        sec = f.name.replace("_outcomes.json", "")
        if sec.startswith("clip_") or sec.startswith("curry"): continue       # harvest games only
        for r in json.loads(f.read_text()):
            if r.get("status") == "aligned":
                by[game_for_clip(sec)].append({"section": sec, "set_start_frame": r["set_start_frame"], "core_end_frame": r["core_end_frame"], "period": r["period"],
                                          "anchors": [{"frame": a["frame"], "period": a["period"], "clock_s": a["clock_s"]} for a in r.get("anchors", [])]})
    items = []
    for game in sorted(by):
        rng.shuffle(by[game])
        for it in by[game][:per_game]:
            it["game"] = game; it["id"] = "%s_f%d" % (it["section"], it["set_start_frame"]); items.append(it)
    rng.shuffle(items)
    return items

OFFSETS = TRI / "offsets.json"
SCAN_S = 45                # ± window scanned around the production frame, 1 read per second
MAX_DISAGREE_S = 2.0       # the two anchors must imply the same offset within this
CLOCK_TOL_S = 1.0          # 1 read/s against a clock that shows tenths under 1:00


def _rawframes(vid, t0, dur, w, h):
    """Decode 1 frame/s of [t0, t0+dur) through ffmpeg (local files are AV1; cv2 here can't)."""
    r = subprocess.run([FF, "-v", "error", "-ss", "%.3f" % max(0.0, t0), "-t", "%.3f" % dur, "-i", str(vid), "-vf", "fps=1",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    import numpy as np
    n = len(r.stdout) // (w * h * 3)
    return np.frombuffer(r.stdout[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)


def solve_offset(section, anchors, vid, fps, reader):
    """Local section files are a different download (AV1) split at different keyframes than the
    h264 copies the harvest ran on, so production frame F is NOT local frame F. Read the local game
    clock once per second around each production anchor and return the frame offset (local − production)
    that both anchors agree on, or None (and the triage skips the possession, counting it)."""
    anchors = [a for a in anchors if a.get("clock_s") is not None and a.get("period") is not None][:2]
    if len(anchors) < 2:
        return None
    out = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(vid)],
                         capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    cands = []
    for a in anchors:
        t0 = a["frame"] / fps - SCAN_S
        frames = _rawframes(vid, t0, 2 * SCAN_S, w, h)
        hits = []
        for i, fr in enumerate(frames):
            period, clock, _ = reader.read(fr)
            if period == a["period"] and clock is not None and abs(clock - a["clock_s"]) <= CLOCK_TOL_S:   # sub-minute clocks show tenths
                hits.append((max(0.0, t0) + i) * fps - a["frame"])      # local − production, frames
        cands.append(hits)
    best = None
    for o1 in cands[0]:
        for o2 in cands[1]:
            if abs(o1 - o2) <= MAX_DISAGREE_S * fps and (best is None or abs(o1 - o2) < abs(best[0] - best[1])):
                best = (o1, o2)
    return None if best is None else round((best[0] + best[1]) / 2)


def render(items):
    SRC.mkdir(parents=True, exist_ok=True); REN.mkdir(exist_ok=True)
    from clock_reader import ClockReader, layout_for_clip
    offsets = json.loads(OFFSETS.read_text()) if OFFSETS.exists() else {}
    readers = {}
    t0 = time.time(); done = 0
    for k, it in enumerate(items):
        out = REN / (it["id"] + ".mp4")
        if out.exists(): done += 1; continue
        vid = video_path(it["section"]); fps = fps_of(vid)
        if it["id"] not in offsets:
            lay = layout_for_clip(it["section"])
            readers.setdefault(lay, ClockReader(lay))
            offsets[it["id"]] = solve_offset(it["section"], it.get("anchors", []), vid, fps, readers[lay])
            OFFSETS.write_text(json.dumps(offsets, indent=1))
        off = offsets[it["id"]]
        if off is None:
            done += 1; print("  [%d/%d] %-34s OFFSET UNRESOLVED — skipped" % (done, len(items), it["id"]), flush=True); continue
        start = max(0.0, (it["set_start_frame"] + off) / fps - PAD_S); dur = min(MAX_S, (it["core_end_frame"] - it["set_start_frame"]) / fps + 2 * PAD_S)
        snip = SRC / ("triage_%s.mp4" % it["id"])
        if not snip.exists():
            subprocess.run([FF, "-y", "-v", "error", "-ss", "%.3f" % start, "-t", "%.3f" % dur, "-i", str(vid), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(snip)], check=True)
        stride = build_stride(fps)                                 # production's 10 Hz rule
        nproc = int(dur * fps / stride)
        env = {**__import__("os").environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        subprocess.run(["/opt/anaconda3/bin/python", "build_trajectories.py", "--source", str(snip), "--start", "0", "--max-frames", str(nproc), "--stride", str(stride)] + GATE_ARGS,
                       cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True)
        stem = snip.stem                                            # triage_<id>
        made = config.TRACKING_DIR / (stem + "_trajectories.mp4")
        if made.exists():
            shutil.move(str(made), str(out))
        for p in list(config.TRACKING_DIR.glob(stem + "*")) + list((config.PROJECT_ROOT / "reports" / "viz").glob("team_clusters_" + stem + "*")):
            shutil.move(str(p), str(TRI / "side" / p.name)) if (TRI / "side").mkdir(exist_ok=True) or True else None
        frames = subprocess.run([FP, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip() if out.exists() else "0"
        done += 1
        print("  [%d/%d] %-34s %s frames  offset %+d  (%.0f s elapsed)" % (done, len(items), it["id"], frames or "0", off, time.time() - t0), flush=True)
    print("renders: %d/%d present" % (sum(1 for it in items if (REN / (it["id"] + ".mp4")).exists()), len(items)), flush=True)


def label_loop(items, labels):
    import cv2
    todo = [it for it in items if it["id"] not in labels and (REN / (it["id"] + ".mp4")).exists()]
    pre = TRI / "prescreen.json"                     # model pre-screen (not a label): flagged clips first
    if pre.exists():
        f = json.loads(pre.read_text()); hot = set(f.get("court_off", []) + f.get("no_template_drawn", []) + f.get("borderline", []) + f.get("boxes_suspect", []))
        todo.sort(key=lambda it: it["id"] not in hot)
    print("%d clips to triage (%d done). 1-5 toggle a fault · g/m/b overall · r restart · SPACE next · q quit" % (len(todo), len(labels)))
    win = "triage — score the faults you SEE"
    for n, it in enumerate(todo):
        cap = cv2.VideoCapture(str(REN / (it["id"] + ".mp4"))); fps = cap.get(cv2.CAP_PROP_FPS) or 15
        faults, overall, frames = set(), None, []
        while True:
            ok, fr = cap.read()
            if ok: frames.append(fr)
            else: break
        cap.release()
        i = 0
        while True:
            fr = frames[i % len(frames)].copy(); i += 1
            h, w = fr.shape[:2]; sc = min(1.0, 1320 / w)
            if sc < 1: fr = cv2.resize(fr, None, fx=sc, fy=sc)
            fr = cv2.copyMakeBorder(fr, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            on = "  ".join(("[%s %s]" if k in faults else " %s %s ") % (k, v) for k, v in FAULTS.items())
            cv2.putText(fr, "[%d/%d] %s   %s   overall: %s" % (n + 1, len(todo), it["id"], it["game"], overall or "-"), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(fr, on + "   g/m/b overall   r restart   SPACE next   q quit", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, fr)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == 255: continue
            c = chr(key)
            if c in FAULTS: faults.symmetric_difference_update({c})
            elif c in "gmb": overall = {"g": "good", "m": "minor", "b": "broken"}[c]
            elif c == "r": i = 0
            elif c == "q": cv2.destroyAllWindows(); print("stopped — %d triaged" % len(labels)); return
            elif key in (32, 13) and overall: break
        labels[it["id"]] = {"faults": sorted(FAULTS[k] for k in faults), "overall": overall, "game": it["game"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        LABELS.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows(); print("triage complete (%d)." % len(labels))


def report(items, labels):
    done = [it for it in items if it["id"] in labels]
    fault_n = Counter(f for it in done for f in labels[it["id"]]["faults"])
    overall = Counter(labels[it["id"]]["overall"] for it in done)
    by_game = {}
    for g in sorted({it["game"] for it in done}):
        gi = [it for it in done if it["game"] == g]
        by_game[g] = {"n": len(gi), "broken": sum(1 for it in gi if labels[it["id"]]["overall"] == "broken"),
                      "faults": dict(Counter(f for it in gi for f in labels[it["id"]]["faults"]))}
    offsets = json.loads(OFFSETS.read_text()) if OFFSETS.exists() else {}
    # label-free QC (FIX_PLAN Phase A, qc/track_qc.py) next to the human labels, per clip
    qc_dir = config.REPORTS_DIR / "qc"; qc = {}
    for it in items:
        qp = qc_dir / ("triage_%s.json" % it["id"])
        if qp.exists():
            sm = json.loads(qp.read_text())["summary"]
            qc[it["id"]] = {"lines_px": sm.get("dist_px_median"), "fail_rate": round(sm["frames_failing_any"] / max(sm["frames"], 1), 3),
                            "second_disagreement": sm.get("second_disagreement_median"), "rules": sm.get("rule_counts", {})}
    rep = {"sampled": len(items), "rendered": sum(1 for it in items if (REN / (it["id"] + ".mp4")).exists()),
           "qc_note": "label-free indicators per clip (qc/track_qc.py), thresholds uncalibrated until FIX_PLAN B6; see reports/qc_summary.*",
           "qc": qc,
           "offset_unresolved": [it["id"] for it in items if offsets.get(it["id"]) is None],
           "offset_note": "snippets are cut at production frame + a per-possession offset solved from the game clock (±1-3 s; "
                          "stopped clocks are ambiguous) — faults are judged on the footage shown, not on possession boundaries",
           "triaged": len(done), "overall": dict(overall),
           "fault_counts": dict(fault_n), "fault_rates": {f: round(fault_n[f] / len(done), 3) for f in FAULTS.values()} if done else {},
           "dominant_fault": fault_n.most_common(1)[0][0] if fault_n else None,
           "games_by_broken_rate": sorted(((g, round(v["broken"] / v["n"], 2)) for g, v in by_game.items()), key=lambda x: -x[1]),
           "by_game": by_game}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "triage.json").write_text(json.dumps(rep, indent=1))
    L = ["VISUAL TRIAGE — %d/%d clips triaged (%d rendered, %d offset-unresolved, QC on %d)" % (rep["triaged"], rep["sampled"], rep["rendered"], len(rep["offset_unresolved"]), len(qc)), "  overall: %s" % rep["overall"], "  fault rates: %s" % rep["fault_rates"],
         "  dominant: %s" % rep["dominant_fault"], "  games by broken rate: %s" % rep["games_by_broken_rate"]]
    for g, v in by_game.items(): L.append("    %-14s n=%d broken=%d faults=%s" % (g, v["n"], v["broken"], v["faults"]))
    if qc:
        L.append("  label-free QC per clip (human label beside it when present):")
        for it in sorted(items, key=lambda it: -(qc.get(it["id"], {}).get("fail_rate") or 0)):
            q = qc.get(it["id"])
            if q is None: continue
            hl = labels.get(it["id"]); hs = ("%s %s" % (hl["overall"], ",".join(hl["faults"]))) if hl else "-"
            L.append("    %-32s lines %5s px  fail %.2f  2nd %5s  human: %s" % (it["id"], q["lines_px"], q["fail_rate"], q["second_disagreement"], hs))
    txt = "\n".join(L); (config.REPORTS_DIR / "triage.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", type=int, metavar="PER_GAME"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true"); ap.add_argument("--label", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(); TRI.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    if a.sample:
        if labels: raise SystemExit("labels exist — move data/triage aside for a fresh sample")
        items = build_sample(a.sample, a.seed); SAMPLE.write_text(json.dumps(items, indent=1))
        print("sample: %d possessions, %s" % (len(items), dict(Counter(it["game"] for it in items))))
    items = json.loads(SAMPLE.read_text()) if SAMPLE.exists() else []
    if a.render: render(items)
    if a.label: label_loop(items, labels)
    if a.report: report(items, labels)

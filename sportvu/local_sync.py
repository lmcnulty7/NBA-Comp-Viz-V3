"""
sportvu/local_sync.py: LOCAL time map per section by reading the game clock off the local video
(FIX_PLAN B4 prerequisite). Why: local section files are not frame-aligned with the production
ones (verified 2026-10-03 on gsw_phx_2016: anchors re-read 4..9 s off, drifting within a section),
so production frame numbers cannot index local frames and the production anchors are useless here.
Reads (period, clock) with clock_reader at READ_HZ, keeps RUNNING spans (consecutive reads whose
clock falls by the elapsed time within RUN_TOL_S), interpolates (period, clock) for every frame at
the harvest stride inside those spans. Output: data/sportvu/sync/<section>_local_timemap.json.

  python -m sportvu.local_sync gsw_phx_2016 [--sections s00,s01]
  python -m sportvu.local_sync gsw_phx_2016 --windows     # time maps for the rebuilt windows themselves
                                                          # (an ffmpeg -ss cut can start a keyframe early,
                                                          # measured +4.5..+7 s on phx, so never map a
                                                          # window frame through the section map)
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import config
from fetch_pbp import video_path
from sportvu.sync import SYNC_DIR

READ_HZ = 1.0
RUN_TOL_S = 1.0
MIN_SPAN_READS = 4


def read_clock_track(section: str, layout_reader, vid=None) -> tuple[list, float]:
    import cv2
    from harvest_driver import video_fps
    vid = vid or video_path(section); fps = video_fps(vid)
    cap = cv2.VideoCapture(str(vid)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / READ_HZ))
    reads, f = [], 0
    while f < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        p, c, raw = layout_reader.read(frame)
        reads.append([f, p, None if c is None else float(c)])
        f += step
    cap.release()
    return reads, fps


def running_spans(reads: list, fps: float) -> list[dict]:
    spans, cur = [], []
    for r in reads:
        f, p, c = r
        if p is None or c is None:
            if len(cur) >= MIN_SPAN_READS: spans.append(cur)
            cur = []; continue
        if cur:
            f0, p0, c0 = cur[-1]
            dt = (f - f0) / fps
            if p == p0 and abs((c0 - c) - dt) <= RUN_TOL_S and dt > 0:
                cur.append(r); continue
            if len(cur) >= MIN_SPAN_READS: spans.append(cur)
            cur = [r]
        else:
            cur = [r]
    if len(cur) >= MIN_SPAN_READS: spans.append(cur)
    out = []
    for s in spans:
        out.append({"f0": s[0][0], "f1": s[-1][0], "period": s[0][1], "c0": s[0][2], "c1": s[-1][2],
                    "reads": len(s), "video_s": round((s[-1][0] - s[0][0]) / fps, 1)})
    return out


def build_local_timemap(section: str, reader, vid=None) -> dict:
    from harvest_driver import build_stride
    t0 = time.time()
    reads, fps = read_clock_track(section, reader, vid)
    spans = running_spans(reads, fps)
    stride = build_stride(fps)
    frames = []
    for s in spans:
        for f in range(s["f0"], s["f1"] + 1, stride):
            u = (f - s["f0"]) / max(s["f1"] - s["f0"], 1)
            frames.append([f, s["period"], round(s["c0"] + u * (s["c1"] - s["c0"]), 3)])
    return {"section": section, "fps": fps, "stride": stride, "reads": len(reads),
            "reads_ok": sum(1 for r in reads if r[1] is not None and r[2] is not None),
            "running_spans": len(spans), "mapped_video_s": round(sum(s["video_s"] for s in spans), 1),
            "mapped_frames": len(frames), "spans": spans, "frames": frames, "ocr_seconds": round(time.time() - t0, 1)}


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "gsw_phx_2016"
    only = None
    if "--sections" in sys.argv:
        only = set(sys.argv[sys.argv.index("--sections") + 1].split(","))
    from clock_reader import ClockReader, layout_for_clip
    secs = sorted(p.name.replace("_outcomes.json", "") for p in (config.PROJECT_ROOT / "data" / "pbp").glob(game + "_s*_outcomes.json"))
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    reader = ClockReader(layout_for_clip(secs[0]))
    if "--windows" in sys.argv:
        from sportvu.rebuild import BUILD_DIR
        windows = json.loads((BUILD_DIR / (game + "_windows.json")).read_text())
        for w in windows:
            out = BUILD_DIR / (w["window"] + "_timemap.json")
            if out.exists():
                continue
            tm = build_local_timemap(w["section"], reader, vid=BUILD_DIR / (w["window"] + ".mp4"))
            out.write_text(json.dumps(tm))
            print("  %-24s reads %d (ok %d) spans %d mapped %.1f s" % (w["window"], tm["reads"], tm["reads_ok"], tm["running_spans"], tm["mapped_video_s"]), flush=True)
        return
    for s in secs:
        if only and s.split("_")[-1] not in only:
            continue
        out = SYNC_DIR / (s + "_local_timemap.json")
        if out.exists():
            print("  %-20s cached" % s, flush=True); continue
        tm = build_local_timemap(s, reader)
        out.write_text(json.dumps(tm))
        print("  %-20s reads %d (ok %d)  spans %d  mapped %.1f s / %d frames  (%.0f s)" % (
            s, tm["reads"], tm["reads_ok"], tm["running_spans"], tm["mapped_video_s"], tm["mapped_frames"], tm["ocr_seconds"]), flush=True)


if __name__ == "__main__":
    main()

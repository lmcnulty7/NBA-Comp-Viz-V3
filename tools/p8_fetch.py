"""tools/p8_fetch.py: P8 Phase A video downloads and the one-frame scorebug check (PLAN.md D83, D85).

    python3 -m tools.p8_fetch dev [--tags ind_chi,...]      download the channel development candidates
    python3 -m tools.p8_fetch test                          download the 4 broadcast CV test videos (sealed)
    python3 -m tools.p8_fetch frame TAG SECONDS             one jpg of a development candidate at SECONDS

Downloads are network and disk only (no model), so they run outside the heavy-job lock. Each file is fetched with
yt-dlp under harvest_driver's avc1 format pin and proof-of-origin token arguments, written to a .part file, checked
with ffprobe (video codec h264, height 720) and only then renamed into place. A cached file that passes the check is
kept.

Development candidates go to <P8 root>/video/<tag>.mp4. The CV test videos go to their own sealed folder,
<P8 root>/../p8_cv_test/video/<game>.mp4, through tools.research_testset.cv_test_videos. Nothing reads that folder
except this tool's ffprobe check and, later, the single score_once scorer of each experiment: no model, frame read
or scorebug check runs on a test video here (D85 (1)).

The scorebug check reads one frame of a candidate by eye: teams, period, game clock and score. That reading is
compared with the contract's play-by-play of each candidate SportVU game (tools.p8_scorebug) and written to
reports/research/p8_registry.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.p8_common import channel_games, check_outside, guard, p8_root
from tools.research_testset import cv_test_videos, refuse_cv_test

YTDLP = os.environ.get("V3_YTDLP") or shutil.which("yt-dlp") or "yt-dlp"   # the one with the token plugin
WHO = "tools.p8_fetch"


def cv_test_dir() -> Path:
    """The sealed folder of the CV test videos, a sibling of the P8 workspace."""
    return check_outside(p8_root().parent / "p8_cv_test" / "video")


def probe(path: Path) -> dict:
    """ffprobe facts of a video file: codec, size, frame rate, duration."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-200:]}
    d = json.loads(r.stdout)
    s = (d.get("streams") or [{}])[0]
    num, den = (s.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    out = {"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
           "fps": round(float(num) / float(den), 3) if float(den) else None,
           "duration_s": round(float(d.get("format", {}).get("duration", 0)), 1)}
    out["ok"] = out["codec"] == "h264" and out["height"] == 720
    return out


def download(video_id: str, out: Path, timeout_s: int = 3 * 3600) -> dict:
    """Fetch one video to out (atomic), checked by ffprobe -> the probe facts."""
    from harvest_driver import YTDLP_FMT, ytdlp_pot_args
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        p = probe(out)
        if p["ok"]:
            return {**p, "cached": True}
    part = out.with_suffix(".part.mp4")
    part.unlink(missing_ok=True)
    r = subprocess.run([YTDLP, "--no-update", "--js-runtimes", "node", "--no-warnings", "-q", *ytdlp_pot_args(),
                        "-f", YTDLP_FMT, "--merge-output-format", "mp4", "-o", str(part),
                        f"https://www.youtube.com/watch?v={video_id}"], capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0 or not part.exists():
        part.unlink(missing_ok=True)
        return {"ok": False, "error": (r.stderr or r.stdout or "")[-300:]}
    p = probe(part)
    if not p["ok"]:
        part.unlink(missing_ok=True)
        return {**p, "ok": False, "error": "not h264 720p after download"}
    part.replace(out)
    return {**p, "cached": False}


def fetch_dev(tags=None) -> dict:
    games = channel_games()
    root = p8_root()
    res = {}
    for tag, g in games.items():
        if tags and tag not in tags:
            continue
        guard([g["video_id"], *g.get("candidates", [])], WHO)
        res[tag] = download(g["video_id"], root / "video" / f"{tag}.mp4")
        print(f"  {tag:10s} {g['video_id']} {res[tag]}", flush=True)
    return res


def fetch_test() -> dict:
    """The CV test videos: download and ffprobe only."""
    out_dir = cv_test_dir()
    res = {}
    for game, vid in sorted(cv_test_videos().items()):
        p = download(vid, out_dir / f"{game}.mp4")
        res[game] = {k: p.get(k) for k in ("ok", "codec", "height", "fps", "duration_s", "cached", "error")}
        print(f"  test video {game}: {res[game]}", flush=True)
    return res


def frame(tag: str, seconds: float, out_dir: Path | None = None) -> Path:
    """One jpg of a development candidate at the given time (ffmpeg, no model)."""
    g = channel_games()[tag]
    guard([g["video_id"]], WHO)
    src = p8_root() / "video" / f"{tag}.mp4"
    out_dir = check_outside(out_dir or p8_root() / "scorebug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}_{int(round(seconds)):05d}s.jpg"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{seconds:.2f}", "-i", str(src), "-frames:v", "1",
                    "-q:v", "2", str(out)], check=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--tags", default=None)
    sub.add_parser("test")
    f = sub.add_parser("frame")
    f.add_argument("tag")
    f.add_argument("seconds", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "dev":
        res = fetch_dev(set(a.tags.split(",")) if a.tags else None)
        return 0 if all(r.get("ok") for r in res.values()) else 1
    if a.cmd == "test":
        res = fetch_test()
        return 0 if all(r.get("ok") for r in res.values()) else 1
    refuse_cv_test([a.tag], WHO)
    print(frame(a.tag, a.seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

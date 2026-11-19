"""tools/hero_still.py: a smaller encoding of the cold-open stills, chosen by a rule fixed before measuring (D116).

The polish loop found the hero still (stills/scene-1.jpg, about 200 KB of JPEG written by export/demo_bridge.py) to
be what bounds the home page's first view (docs/POLISH_LOOP.md). This tool measures smaller encodings of the four
stills against a visual-quality floor and writes the chosen one with the bridge's own cut_still.

The rule, fixed before any measurement and not changed after it:
- Reference: the source frame of each still, decoded losslessly from the window clip the bridge cuts it from. The
  frame is the one whose gray image is nearest the committed still among the seven frames around
  first + round(still_t x fps); the nearest must be clearly nearer than its neighbours, as the bridge's own check
  "the still is the frame of its row" requires.
- Floor: SSIM at least SSIM_FLOOR (0.97) against the committed still, on every one of the four stills. SSIM is
  that of Wang et al. 2004 on the luma (BT.601), Gaussian window sigma 1.5 (11 by 11), K1 0.01, K2 0.03, range 255.
  SSIM against the lossless source frame is recorded beside it, for reading.
- Candidates, one setting for all four stills: JPEG through the bridge's ffmpeg (mjpeg -q:v 3 to 12; 3 is today's);
  JPEG through Pillow (quality 40 to 90 by 5, optimised, progressive, 4:2:0); WebP through Pillow (quality 40 to 90
  by 5, method 6).
- Choice: the smallest total bytes over the four stills among candidates that clear the floor. A WebP candidate is
  taken only when its total is at least WEBP_MARGIN (15%) below the best JPEG that clears the floor, because WebP
  changes the still's path in demo/frames.json and needs a JPEG fallback in the page, while a JPEG keeps every
  name, path and component.
- The chosen stills must still pass the bridge's alignment check: against the source frames at shifts -2 to 2,
  shift 0 has the smallest mean absolute difference.

    python3 -m tools.hero_still --cv-root DIR --frames DATA_ROOT/demo/frames.json
        --media-check ARTIFACTS/demo/media_check.json --stills web/public/stills --work SCRATCH
        --out reports/hero_still.json [--write]

It reads the window clips and the stills, and writes only into --work, --out and (with --write) --stills. Nothing
here fits, tunes or scores a model; the floor and the candidates are fixed above.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from export import demo_bridge as B

SSIM_FLOOR = 0.97
WEBP_MARGIN = 0.15
SEARCH = range(-3, 4)
FFMPEG_Q = list(range(3, 13))
PILLOW_Q = list(range(40, 95, 5))


def luma(path) -> np.ndarray:
    """BT.601 luma of a picture file as float64."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM of two luma images (Wang et al. 2004): Gaussian window sigma 1.5, K1 0.01, K2 0.03, range 255."""
    if a.shape != b.shape:
        raise ValueError(f"the pictures differ in size: {a.shape} and {b.shape}")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    index = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(index.mean())


def source_frame(clip, number: int, target) -> Path:
    """Source frame `number` of a window clip as a lossless PNG, selected and scaled exactly as cut_still does."""
    chain = f"select=eq(n\\,{number}),scale='min({B.CLIP_MAX_WIDTH},iw)':-2"
    subprocess.run([B._tool("ffmpeg"), "-y", "-v", "error", "-i", str(clip), "-vf", chain, "-fps_mode", "passthrough",
                    "-frames:v", "1", "-update", "1", str(target)], check=True)
    return Path(target)


def encode(kind: str, quality: int, png: Path, target: Path) -> Path:
    """One candidate encoding of a lossless frame."""
    if kind == "ffmpeg-jpeg":
        subprocess.run([B._tool("ffmpeg"), "-y", "-v", "error", "-i", str(png), "-q:v", str(quality), "-frames:v", "1",
                        "-update", "1", str(target)], check=True)
    elif kind == "pillow-jpeg":
        Image.open(png).convert("RGB").save(target, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    elif kind == "pillow-webp":
        Image.open(png).convert("RGB").save(target, "WEBP", quality=quality, method=6)
    else:
        raise ValueError(kind)
    return target


def candidates():
    yield from (("ffmpeg-jpeg", q, ".jpg") for q in FFMPEG_Q)
    yield from (("pillow-jpeg", q, ".jpg") for q in PILLOW_Q)
    yield from (("pillow-webp", q, ".webp") for q in PILLOW_Q)


def choose(rows: list) -> dict:
    """The rule of the module docstring, on the candidate rows (each with kind, quality, total_bytes, min_ssim)."""
    passing = [r for r in rows if r["min_ssim_vs_committed"] >= SSIM_FLOOR]
    jpegs = sorted((r for r in passing if r["kind"] != "pillow-webp"), key=lambda r: r["total_bytes"])
    webps = sorted((r for r in passing if r["kind"] == "pillow-webp"), key=lambda r: r["total_bytes"])
    if not jpegs and not webps:
        return {"choice": None, "why": "no candidate clears the floor"}
    best_jpeg = jpegs[0] if jpegs else None
    best_webp = webps[0] if webps else None
    if best_webp and (best_jpeg is None or best_webp["total_bytes"] <= (1 - WEBP_MARGIN) * best_jpeg["total_bytes"]):
        return {"choice": best_webp, "why": "WebP clears the floor and beats the best JPEG by the margin"}
    why = "the smallest JPEG that clears the floor"
    if best_webp:
        saving = 1 - best_webp["total_bytes"] / best_jpeg["total_bytes"]
        why += f"; the best WebP is {saving:.1%} smaller, under the {WEBP_MARGIN:.0%} margin"
    return {"choice": best_jpeg, "why": why}


def measure(cv_root, frames_file, media_check, stills, work) -> dict:
    cv_root, stills, work = Path(cv_root), Path(stills), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    scenes = {s["id"]: s for s in json.loads(Path(frames_file).read_text())["scenes"]}
    checks = json.loads(Path(media_check).read_text())["scenes"]
    found = {}
    for sid, scene in scenes.items():
        check = checks[sid]
        clip = cv_root / "build" / f"{check['window']}.mp4"
        committed = stills / Path(scene["still"]).name
        # A copy of the committed still, so --write can compare against it after the bridge has replaced it.
        committed_copy = work / f"{sid}_committed{committed.suffix}"
        committed_copy.write_bytes(committed.read_bytes())
        guess = check["source_first_frame"] + int(round(scene["still_t"] * scene["fps"]))
        reference = luma(committed)
        diffs = {}
        for d in SEARCH:
            png = source_frame(clip, guess + d, work / f"{sid}_{guess + d}.png")
            diffs[guess + d] = float(np.abs(luma(png) - reference).mean())
        number = min(diffs, key=diffs.get)
        ordered = sorted(diffs.values())
        found[sid] = {"target": committed, "window": check["window"], "clip": clip, "committed": committed_copy, "frame": number,
                      "png": work / f"{sid}_{number}.png", "committed_bytes": committed.stat().st_size,
                      "gray_mean_abs_diff_by_frame": {str(k): round(v, 4) for k, v in sorted(diffs.items())},
                      "nearest_margin": round(ordered[1] - ordered[0], 4)}
    rows = []
    for kind, quality, suffix in candidates():
        per = {}
        for sid, f in found.items():
            out = encode(kind, quality, f["png"], work / f"{sid}_{kind}_{quality}{suffix}")
            per[sid] = {"bytes": out.stat().st_size,
                        "ssim_vs_committed": round(ssim(luma(out), luma(f["committed"])), 5),
                        "ssim_vs_source": round(ssim(luma(out), luma(f["png"])), 5)}
        rows.append({"kind": kind, "quality": quality, "total_bytes": sum(p["bytes"] for p in per.values()),
                     "hero_bytes": per["scene-1"]["bytes"],
                     "min_ssim_vs_committed": min(p["ssim_vs_committed"] for p in per.values()),
                     "min_ssim_vs_source": min(p["ssim_vs_source"] for p in per.values()), "stills": per})
    committed_vs_source = {sid: round(ssim(luma(f["committed"]), luma(f["png"])), 5) for sid, f in found.items()}
    return {"found": found, "rows": rows, "committed_vs_source": committed_vs_source}


def alignment(clip, still, frame: int, work: Path) -> dict:
    """The bridge's still check: mean absolute gray difference to the source frames at shifts -2 to 2."""
    picture = luma(still)
    out = {}
    for shift in (-2, -1, 0, 1, 2):
        png = source_frame(clip, frame + shift, work / f"align_{Path(still).stem}_{frame + shift}.png")
        out[str(shift)] = round(float(np.abs(luma(png) - picture).mean()), 4)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--frames", required=True, help="demo/frames.json of a data root")
    parser.add_argument("--media-check", required=True, help="demo/media_check.json of the bridge's artifacts")
    parser.add_argument("--stills", required=True, help="the folder of the committed stills")
    parser.add_argument("--work", required=True, help="a scratch folder outside the repository")
    parser.add_argument("--out", required=True, help="the JSON record to write")
    parser.add_argument("--write", action="store_true", help="write the chosen stills with the bridge's cut_still")
    args = parser.parse_args(argv)
    work = Path(args.work)
    result = measure(args.cv_root, args.frames, args.media_check, args.stills, work)
    decision = choose(result["rows"])
    record = {
        "what": "Smaller encodings of the cold-open stills, chosen by the rule of tools/hero_still.py (PLAN.md D116).",
        "rule": {"ssim_floor_vs_committed": SSIM_FLOOR, "webp_margin": WEBP_MARGIN, "ffmpeg_q": FFMPEG_Q,
                 "pillow_quality": PILLOW_Q, "ssim": "Wang et al. 2004, luma BT.601, Gaussian sigma 1.5, K1 0.01, "
                                                  "K2 0.03, range 255"},
        "stills": {sid: {k: v for k, v in f.items() if k not in ("clip", "committed", "png", "target")}
                   for sid, f in result["found"].items()},
        "committed_total_bytes": sum(f["committed_bytes"] for f in result["found"].values()),
        "committed_ssim_vs_source": result["committed_vs_source"],
        "candidates": [{k: v for k, v in r.items() if k != "stills"} for r in result["rows"]],
        "choice": decision["choice"], "why": decision["why"],
    }
    choice = decision["choice"]
    if args.write and choice is not None:
        if choice["kind"] == "pillow-webp":
            raise SystemExit("the rule chose WebP: the bridge, the data and the page must change first; nothing written")
        written = {}
        for sid, f in result["found"].items():
            target = f["target"]
            B.cut_still(f["clip"], target, f["frame"])
            written[sid] = {"bytes": target.stat().st_size,
                            "ssim_vs_committed": round(ssim(luma(target), luma(f["committed"])), 5),
                            "ssim_vs_source": round(ssim(luma(target), luma(f["png"])), 5),
                            "alignment_by_shift": alignment(f["clip"], target, f["frame"], work)}
            shifts = written[sid]["alignment_by_shift"]
            if min(shifts, key=shifts.get) != "0":
                raise SystemExit(f"{sid}: the written still is not nearest its own source frame: {shifts}")
            if written[sid]["ssim_vs_committed"] < SSIM_FLOOR:
                raise SystemExit(f"{sid}: the written still is under the floor: {written[sid]['ssim_vs_committed']}")
        record["written"] = written
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"choice": choice and {k: choice[k] for k in ("kind", "quality", "total_bytes", "hero_bytes",
                                                                    "min_ssim_vs_committed", "min_ssim_vs_source")},
                      "why": decision["why"], "committed_total_bytes": record["committed_total_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

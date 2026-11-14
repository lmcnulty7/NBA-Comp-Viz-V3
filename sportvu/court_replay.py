"""sportvu/court_replay.py: score a court model on the held-out games without a full rebuild.

Every build of a window has the same boxes, track ids and feet (byte-identical, checked at R2.3); only the
court H differs. So a candidate court model is scored by replaying just the court tracker
(court/snap_track.CourtTracker with the candidate weights) over the frames V3's sidecar lists, writing a
stand-in build (V3's sidecars with H and state replaced) and running sportvu.bench on it. Position error,
near-field bias and coverage come out exactly as a full rebuild would give them, at about 20 minutes per
model for the three held-out games instead of about an hour.

Decode alignment: build_trajectories reached its first court frame (and any jump > SEEK_MIN_GAP) with a
cap.set seek, and AVFoundation seeks land k frames late, k not reproducible across processes. calibrate()
finds k per seek segment from V3's own sidecar: on a TRACK frame (H depends on that frame only) a stateless
V3 tracker must reproduce the sidecar H bit for bit at sequential frame f + k. Offsets are cached in
data/sportvu/replay_offsets.json; they belong to the clips and V3's build, so any weights reuse them.

Team labels and identity in a replay keep V3's track linking (the identity json is V3's), so only the court
rows (position error, bias, coverage) are meaningful.

  python -m sportvu.court_replay --weights models/court_grid_r23.pt --name r23_replay [--card-dir DIR]
"""
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
import config

BUILD_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "build"
OFFSETS_P = config.PROJECT_ROOT / "data" / "sportvu" / "replay_offsets.json"
REPLAY_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "replay"
K_SEARCH = 40               # cle_gsw_2016 (29.97 fps) seeks land up to ~+30 frames late


def heldout_games() -> list:
    from sportvu import splits
    sp = splits.load()
    return [g for s in ("heldout_game", "heldout_arena") for g, v in sp[s]["games"].items()
            if v.get("sportvu") and (BUILD_DIR / (g + "_windows.json")).exists()]


def windows_of(game: str) -> list:
    ws = json.loads((BUILD_DIR / (game + "_windows.json")).read_text())
    return [w["window"] for w in ws if (BUILD_DIR / (w["window"] + "_frames.json")).exists()]


def seg_starts(frames: list) -> list:
    from videoseq import SEEK_MIN_GAP
    return [frames[0]] + [b for a, b in zip(frames[:-1], frames[1:]) if b - (a + 1) > SEEK_MIN_GAP]


def _read_sequential(clip: Path, wanted: set) -> dict:
    import cv2
    cap, out, i = cv2.VideoCapture(str(clip)), {}, 0
    last = max(wanted) if wanted else -1
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = f
        i += 1
    cap.release()
    return out


def calibrate(window: str) -> dict:
    """{"offsets": [[segment start, k], ...], "uncalibrated": [segment starts without a TRACK probe]}."""
    from court.snap_track import CourtTracker
    side = json.loads((BUILD_DIR / (window + "_frames.json")).read_text())["frames"]
    fr = [r["frame"] for r in side]
    starts = seg_starts(fr)
    bounds = starts[1:] + [fr[-1] + 1]
    probes = {s0: [r for r in side if s0 <= r["frame"] < s1 and r["state"] == "TRACK" and r.get("H")][:2]
              for s0, s1 in zip(starts, bounds)}
    frames = _read_sequential(BUILD_DIR / (window + ".mp4"),
                              {r["frame"] + k for ps in probes.values() for r in ps for k in range(-K_SEARCH, K_SEARCH + 1)
                               if r["frame"] + k >= 0})
    trk = CourtTracker(weights=config.PROJECT_ROOT / "models" / "court_grid_snapped.pt")

    def h_of(img):
        trk.state, trk._P, trk._held, trk._prev_small = "LOST", None, 0, None
        hom = trk.update(img)
        return None if hom is None or not hom.is_valid else np.asarray(hom._H, np.float64).ravel()

    offs, bad = [], []
    for s0, ps in probes.items():
        hit = None
        for r in ps:
            H0 = np.array(r["H"], np.float64)
            ks = [k for k in range(-K_SEARCH, K_SEARCH + 1)
                  if (r["frame"] + k) in frames and (lambda H: H is not None and np.array_equal(H, H0))(h_of(frames[r["frame"] + k]))]
            if len(ks) == 1:
                hit = ks[0]
                break
        if hit is None:
            bad.append(s0)
            hit = 0
        offs.append([s0, hit])
    return {"offsets": offs, "uncalibrated": bad}


def load_offsets(windows: list) -> dict:
    cache = json.loads(OFFSETS_P.read_text()) if OFFSETS_P.exists() else {}
    todo = [w for w in windows if w not in cache]
    for i, w in enumerate(todo):
        t0 = time.time()
        cache[w] = calibrate(w)
        print("  calibrate [%d/%d] %s %s (%.0f s)" % (i + 1, len(todo), w, cache[w], time.time() - t0), flush=True)
        OFFSETS_P.write_text(json.dumps(cache, indent=1))
    return cache


def replay_window(window: str, weights: Path, offs: list) -> list:
    """[{frame, H, state}] for the frames V3's sidecar lists, tracker fed in the sidecar's order."""
    from court.snap_track import CourtTracker
    side = json.loads((BUILD_DIR / (window + "_frames.json")).read_text())["frames"]
    want = [r["frame"] for r in side]

    def k_of(f):
        k = 0
        for s0, kk in offs:
            if f >= s0:
                k = kk
        return k
    src = {f + k_of(f): f for f in want}
    frames = _read_sequential(BUILD_DIR / (window + ".mp4"), set(src))
    trk = CourtTracker(weights=weights)
    out = []
    for seq_i in sorted(src):
        f = src[seq_i]
        img = frames.get(seq_i)
        hom = trk.update(img) if img is not None else None
        ok = hom is not None and hom.is_valid
        out.append({"frame": f, "state": trk.state if ok else "LOST",
                    "H": [float(v) for v in np.asarray(hom._H, np.float64).ravel()] if ok else None})
    return out


def build_replay(name: str, weights: Path, games: list | None = None) -> Path:
    games = games or heldout_games()
    out = REPLAY_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    wins = [w for g in games for w in windows_of(g)]
    offsets = load_offsets(wins)
    t0 = time.time()
    for g in games:
        shutil.copy(BUILD_DIR / (g + "_windows.json"), out / (g + "_windows.json"))
    for i, w in enumerate(wins):
        dst = out / (w + "_frames.json")
        if dst.exists():
            continue
        side = json.loads((BUILD_DIR / (w + "_frames.json")).read_text())
        rp = {r["frame"]: r for r in replay_window(w, weights, offsets[w]["offsets"])}
        for r in side["frames"]:
            r["H"], r["state"] = rp[r["frame"]]["H"], rp[r["frame"]]["state"]
            r.pop("accepted", None)
        side["replay"] = {"weights": str(weights), "offsets": offsets[w]}
        dst.write_text(json.dumps(side))
        ident = BUILD_DIR / (w + "_identity.json")
        if ident.exists():
            shutil.copy(ident, out / ident.name)
        print("  replay [%d/%d] %s (%.0f s)" % (i + 1, len(wins), w, time.time() - t0), flush=True)
    return out


def main() -> None:
    from sportvu import bench
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--card-dir", type=Path, default=bench.SCORECARD_DIR)
    ap.add_argument("--calibrate-only", action="store_true")
    a = ap.parse_args()
    games = a.games or heldout_games()
    if a.calibrate_only:
        load_offsets([w for g in games for w in windows_of(g)])
        return
    d = build_replay(a.name, a.weights.resolve(), games)
    for g in games:
        rep = bench.score_build(d, g, a.name, a.card_dir, verbose=False)
        pe, nf, cv = (rep["metrics"][k] for k in ("position_error", "near_field_bias", "coverage"))
        print("%s %s: position p50 %s p90 %s | near %s far %s | coverage %s" % (
            a.name, g, pe["p50_ft"], pe["p90_ft"], nf["near_third_ft"], nf["far_third_ft"], cv["coverage"]), flush=True)


if __name__ == "__main__":
    main()

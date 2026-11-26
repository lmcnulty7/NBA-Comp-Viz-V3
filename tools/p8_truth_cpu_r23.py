"""tools/p8_truth_cpu_r23.py: the r23 detector guard of the reset-fixed CPU truth driver (tools/p8_truth_cpu.sh;
docs/research/P8_truth_arenas.md section 16 version 4.1). No model runs here.

    python3 -m tools.p8_truth_cpu_r23 detector-check        exit 1 unless models/player_detector.pt is r23's

Version 4.1: check C's r23 side (builds/r23 of the reference root) is composed only by the reset-fix arm's
`tools.p8_resetfix compose-checkc` (one composer; the arm owns the reset-window list and the equivalence gate). The
copy and compose commands that versions 3 and 4 had here are removed; nothing here writes builds/r23.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.p8_common import REPO

WHO = "tools.p8_truth_cpu_r23"
R23_DETECTOR_SHA256 = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"   # models/player_detector.pt, r23


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detector_sha_ok(path: Path = REPO / "models" / "player_detector.pt") -> bool:
    return path.exists() and sha256(path) == R23_DETECTOR_SHA256


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detector-check", help="exit 1 unless models/player_detector.pt is the r23 detector")
    ap.parse_args(argv)
    ok = detector_sha_ok()
    print(json.dumps({"r23_detector": ok, "want_sha256": R23_DETECTOR_SHA256}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())

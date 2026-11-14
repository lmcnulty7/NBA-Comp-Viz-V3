"""tools/check_no_frame_data.py: prove that git tracks no SportVU frame data and no video clip (PLAN.md D60, D62).

    python3 -m tools.check_no_frame_data            # scans every file git tracks; exit 1 on any hit

What counts as a hit:
- a tracked video file, a parquet table, or an archive that holds one of those;
- a tracked text file (JSON, CSV, TSV, JS, TS, HTML, TXT, MD) that holds a position track: a JSON list of at least
  TRACK_MIN entries that each carry numeric x and y, a JSON list of at least TRACK_MIN numbers under a key named
  x, y, z, xs, ys, positions, players, ball, truth, ghosts or moments, or a CSV or TSV with x and y columns and at
  least TRACK_MIN rows;
- the word "moments" next to a numeric array (the SportVU raw format).
A summary file (percentiles, counts, scorecards) holds no list that long and passes.
"""

import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_MIN = 50
VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
TABLES = {".parquet", ".feather", ".arrow"}
TEXT = {".json", ".csv", ".tsv", ".js", ".mjs", ".ts", ".tsx", ".html", ".txt", ".md"}
TRACK_KEYS = {"x", "y", "z", "xs", "ys", "positions", "players", "ball", "truth", "ghosts", "moments"}
MOMENTS_RE = re.compile(r'"moments"\s*:\s*\[\s*\[')


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def json_hits(obj, path="$") -> list:
    """Position tracks inside a parsed JSON value."""
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in TRACK_KEYS and isinstance(value, list) and len(value) >= TRACK_MIN \
                    and sum(_numeric(v) for v in value) >= TRACK_MIN:
                hits.append(f"{path}.{key}: {len(value)} numbers")
            hits += json_hits(value, f"{path}.{key}")
    elif isinstance(obj, list):
        points = [v for v in obj if isinstance(v, dict) and _numeric(v.get("x")) and _numeric(v.get("y"))]
        if len(points) >= TRACK_MIN:
            hits.append(f"{path}: {len(points)} points with x and y")
        for i, value in enumerate(obj[:2000]):
            hits += json_hits(value, f"{path}[{i}]")
    return hits


def count_points(obj) -> int:
    """All dicts with numeric x and y anywhere in a parsed JSON value (a frames file spreads them over many lists)."""
    if isinstance(obj, dict):
        own = 1 if _numeric(obj.get("x")) and _numeric(obj.get("y")) else 0
        return own + sum(count_points(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_points(v) for v in obj)
    return 0


def text_hits(name: str, text: str) -> list:
    hits = []
    if MOMENTS_RE.search(text):
        hits.append("raw SportVU moments")
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            hits += json_hits(parsed)
            total = count_points(parsed)
            if total >= TRACK_MIN and not hits:
                hits.append(f"{total} points with x and y spread over the file")
    if suffix in (".csv", ".tsv"):
        try:
            rows = list(csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ","))
        except csv.Error:
            rows = []
        if rows and {"x", "y"} <= {c.strip().lower() for c in rows[0]} and len(rows) >= TRACK_MIN:
            hits.append(f"table with x and y columns, {len(rows)} rows")
    return hits


def scan(files, read) -> dict:
    """{file: [hits]} for the given repository-relative names; `read(name)` returns the bytes."""
    found = {}
    for name in files:
        suffix = Path(name).suffix.lower()
        hits = []
        if suffix in VIDEO:
            hits.append("video file")
        elif suffix in TABLES:
            hits.append("data table")
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(read(name))) as archive:
                    for member in archive.namelist():
                        inner = Path(member).suffix.lower()
                        if inner in VIDEO | TABLES:
                            hits.append(f"archive holds {member}")
                        elif inner in TEXT:
                            hits += [f"{member}: {h}" for h in text_hits(member, archive.read(member).decode("utf-8", "replace"))]
            except zipfile.BadZipFile:
                pass
        elif suffix in TEXT:
            hits += text_hits(name, read(name).decode("utf-8", "replace"))
        if hits:
            found[name] = hits
    return found


def tracked_files(ref=None) -> list:
    args = ["git", "-C", str(REPO_ROOT), "ls-files"] if ref is None else \
        ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", ref]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.splitlines()


def main(argv=None) -> int:
    files = tracked_files()
    found = scan(files, lambda name: (REPO_ROOT / name).read_bytes() if (REPO_ROOT / name).is_file() else b"")
    for name, hits in sorted(found.items()):
        print(f"{name}: {'; '.join(hits[:3])}")
    print(f"{len(files)} tracked files scanned; {len(found)} hold frame data or video")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

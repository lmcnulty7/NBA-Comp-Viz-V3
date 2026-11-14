"""tools/frozen_common.py: shared pieces of the frozen rebuilds (PLAN.md D45).

No fitted model was saved at freeze-v1, and two outputs were never written. A rebuild reruns the frozen code
unchanged, on training games only, with the frozen settings and seed, and is used only when it reproduces the
frozen record. Every rebuild module follows the same contract:

- it writes only under the artifacts folder (never under DATA_ROOT, never under reports/): build every output
  path with `assert_safe_output`;
- it calls `begin_rebuild` first, which marks the folder as not passed until a new check is written;
- it calls `training_games_only` on the game ids it fits on;
- it ends by writing `check.json` with `write_check`: what was compared with what, on how many values, the
  largest absolute difference and the tolerance. `passed` is derived here, never taken on trust;
- consumers call `require_passed` before using anything from it.

A check compares predictions with stored predictions, or a reported out-of-fold number with a recomputed one.
It never reads a held-out outcome and never computes a held-out metric.
"""

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from export.heavy import peak_rss_gb

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_NAME = "check.json"


def artifacts_dir(path=None) -> Path:
    """The folder rebuilt objects live in: the argument, else V3_ARTIFACTS, else artifacts/ in the repository."""
    chosen = Path(path) if path else Path(os.environ.get("V3_ARTIFACTS") or REPO_ROOT / "artifacts")
    return chosen.expanduser().resolve()


def assert_safe_output(path, data_root=None) -> Path:
    """Refuse an output path inside the data root or inside the repository's reports/ folder. The data root is the
    argument, else the DATA_ROOT variable; with neither, the guard cannot be applied and the call is refused."""
    target = Path(path).expanduser().resolve()
    root = data_root or os.environ.get("DATA_ROOT")
    if not root:
        raise SystemExit("DATA_ROOT is not set, so the output guard cannot tell what must not be overwritten")
    for forbidden in (REPO_ROOT / "reports", Path(root).expanduser().resolve()):
        if target == forbidden or forbidden in target.parents:
            raise SystemExit(f"refusing to write under {forbidden.name}: frozen outputs and reports are never overwritten")
    return target


def training_games_only(game_ids, who: str) -> int:
    """SystemExit when any game id is a held-out game, or when there is no id to check; otherwise the number of
    ids checked. A single id may be given as a string."""
    from analytics.eval import refuse_heldout
    ids = [game_ids] if isinstance(game_ids, (str, bytes)) else [str(g) for g in game_ids]
    if not ids:
        raise SystemExit(f"{who}: no game id to check, so the held-out guard cannot pass")
    return refuse_heldout(ids, who)


def rss_gb() -> float:
    """Peak resident memory of this process in GB."""
    return peak_rss_gb()


def current_rss_gb() -> float:
    """Resident memory of this process right now in GB (from ps), or the peak when ps is not available."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
        return int(out.stdout.strip()) / 1e6
    except Exception:
        return peak_rss_gb()


class MemoryGuard:
    """Stops a rebuild before it can exhaust the machine: `check()` raises MemoryError when the memory in use right
    now is above `limit_gb`. Call it between stages and inside long loops."""

    def __init__(self, limit_gb: float = 11.5):
        self.limit_gb = limit_gb

    def check(self, where: str = "") -> float:
        used = current_rss_gb()
        if used > self.limit_gb:
            raise MemoryError(f"memory in use {used:.1f} GB is above the {self.limit_gb:.1f} GB guard ({where})")
        return used


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    """numpy scalars and the like as plain JSON values; NaN and infinity become None."""
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def begin_rebuild(folder, name: str) -> Path:
    """Mark an artifact folder as not passed before a rebuild writes into it, so an earlier pass cannot vouch for
    files a new run is about to replace."""
    folder = assert_safe_output(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "passed": False, "status": "rebuild started, no check written yet",
              "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (folder / CHECK_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return folder


def write_check(folder, record: dict, files=()) -> Path:
    """Write check.json. `record` must hold: name, compared, n, max_abs_diff, tolerance. `passed` is computed here
    as n > 0 and max_abs_diff <= tolerance; a `passed` given by the caller can only make the result stricter.
    `files` are the artifact files this check vouches for: their hashes are stored and verified by require_passed."""
    missing = {"name", "compared", "n", "max_abs_diff", "tolerance"} - set(record)
    if missing:
        raise ValueError(f"check record lacks {sorted(missing)}")
    folder = assert_safe_output(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = _plain(dict(record))
    diff, tolerance, n = record["max_abs_diff"], record["tolerance"], record["n"]
    derived = bool(n) and diff is not None and tolerance is not None and diff <= tolerance
    record["passed"] = bool(derived and record.get("passed", True))
    record["files"] = {Path(f).name: _sha256(assert_safe_output(Path(folder) / Path(f).name)) for f in files}
    record["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["peak_memory_gb"] = round(peak_rss_gb(), 2)
    path = folder / CHECK_NAME
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_check(folder):
    """The check record of an artifact folder, or None when there is none."""
    path = Path(folder) / CHECK_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def require_passed(folder, name: str) -> dict:
    """The check record, or RuntimeError when the artifact is missing, did not reproduce the frozen record, belongs
    to another artifact, or changed after its check was written."""
    folder = Path(folder)
    record = read_check(folder)
    if record is None:
        raise RuntimeError(f"{name}: no rebuilt artifact (run the rebuild first)")
    if record.get("name") not in (None, name):
        raise RuntimeError(f"{name}: the check in this folder is for {record.get('name')!r}")
    if not record.get("passed"):
        raise RuntimeError(f"{name}: the rebuild did not reproduce the frozen record, so it is not used")
    for file_name, digest in (record.get("files") or {}).items():
        path = folder / file_name
        if not path.exists() or _sha256(path) != digest:
            raise RuntimeError(f"{name}: {file_name} changed after its check was written, so it is not used")
    check_time = (folder / CHECK_NAME).stat().st_mtime
    newer = [p.name for p in folder.iterdir() if p.is_file() and p.name != CHECK_NAME and p.stat().st_mtime > check_time + 1]
    if newer:
        raise RuntimeError(f"{name}: {sorted(newer)} are newer than the check, so the check does not cover them")
    return record

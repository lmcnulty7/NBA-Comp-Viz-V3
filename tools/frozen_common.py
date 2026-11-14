"""tools/frozen_common.py: shared pieces of the frozen rebuilds (PLAN.md D45).

No fitted model was saved at freeze-v1, and two outputs were never written. A rebuild reruns the frozen code
unchanged, on training games only, with the frozen settings and seed, and is used only when it reproduces the
frozen record. Every rebuild module follows the same contract:

- it writes only under the artifacts folder (never under DATA_ROOT, never under reports/);
- it calls `training_games_only` on the game ids it fits on;
- it ends by writing `check.json` with `write_check`: what was compared with what, on how many values, the
  largest absolute difference, the tolerance, and `passed`;
- consumers call `require_passed` before using anything from it.

A check compares predictions with stored predictions, or a reported out-of-fold number with a recomputed one.
It never reads a held-out outcome and never computes a held-out metric.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def artifacts_dir(path=None) -> Path:
    """The folder rebuilt objects live in: the argument, else V3_ARTIFACTS, else artifacts/ in the repository."""
    chosen = Path(path) if path else Path(os.environ.get("V3_ARTIFACTS") or REPO_ROOT / "artifacts")
    return chosen.expanduser().resolve()


def assert_safe_output(path) -> Path:
    """Refuse an output path inside DATA_ROOT or inside the repository's reports/ folder."""
    target = Path(path).expanduser().resolve()
    forbidden = [REPO_ROOT / "reports"]
    if os.environ.get("DATA_ROOT"):
        forbidden.append(Path(os.environ["DATA_ROOT"]).expanduser().resolve())
    for root in forbidden:
        if target == root or root in target.parents:
            raise SystemExit(f"refusing to write under {root.name}: frozen outputs and reports are never overwritten")
    return target


def training_games_only(game_ids, who: str) -> int:
    """SystemExit when any game id is a held-out game; otherwise the number of ids checked."""
    from analytics.eval import refuse_heldout
    return refuse_heldout(list(game_ids), who)


def rss_gb() -> float:
    """Peak resident memory of this process in GB (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


class MemoryGuard:
    """Stops a rebuild before it can exhaust the machine: `check()` raises MemoryError above `limit_gb`."""

    def __init__(self, limit_gb: float = 11.5):
        self.limit_gb = limit_gb

    def check(self, where: str = "") -> float:
        used = rss_gb()
        if used > self.limit_gb:
            raise MemoryError(f"peak memory {used:.1f} GB is above the {self.limit_gb:.1f} GB guard ({where})")
        return used


def write_check(folder, record: dict) -> Path:
    """Write check.json. `record` must hold: name, compared, n, max_abs_diff, tolerance, passed."""
    missing = {"name", "compared", "n", "max_abs_diff", "tolerance", "passed"} - set(record)
    if missing:
        raise ValueError(f"check record lacks {sorted(missing)}")
    folder = assert_safe_output(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(record, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), peak_memory_gb=round(rss_gb(), 2))
    path = folder / "check.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_check(folder):
    """The check record of an artifact folder, or None when the rebuild never finished."""
    path = Path(folder) / "check.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def require_passed(folder, name: str) -> dict:
    """The check record, or RuntimeError when the artifact is missing or did not reproduce the frozen record."""
    record = read_check(folder)
    if record is None:
        raise RuntimeError(f"{name}: no rebuilt artifact (run tools.frozen_models first)")
    if not record.get("passed"):
        raise RuntimeError(f"{name}: the rebuild did not reproduce the frozen record, so it is not used")
    return record

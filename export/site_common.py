"""export/site_common.py: shared pieces of the site data export (PLAN.md M1).

Constants of the export, the problem type every step raises, source records, rounding, the per-player game
bootstrap, and the one place a contract model is turned into a file. Nothing here reads a model output.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from export import schemas as S

REPO_ROOT = Path(__file__).resolve().parents[1]
SEASON = "2015-16"
FREEZE_TAG = S.FREEZE_TAG
FPS = 25              # frames a second in the tracking table
STRIDE = 5            # the ghosting grid: every fifth frame, counted from the possession's first frame
HZ = FPS / STRIDE     # frames a second in a possession file
HALF_X_FT = S.COURT_LENGTH_FT / 2.0

N_BOOT = 200          # resamples of a player's own games (PLAN.md D47, the gravity lane's convention)
BOOT_SEED = 0
BOOT_PERCENTILES = (2.5, 97.5)
FIXTURE_MARKER = "fixture_root.json"  # a data root that holds this file was written by a test, never by a lane
# What the site data depends on inside the repository: a change here that is not committed marks the manifest.
CODE_AND_INPUTS = ("export", "analytics", "tools", "sportvu", "collect", "reports", "docs/research", "config.py")
UNCOMMITTED = "+uncommitted"  # the end of a manifest's repo_commit when the code that ran is not that commit


class ExportProblem(Exception):
    """An input or an output is not what the export requires. The run reports it and exits non-zero."""


def interval_label(extra: str = "") -> str:
    """The method name every player interval carries."""
    level = BOOT_PERCENTILES[1] - BOOT_PERCENTILES[0]
    text = f"{level:g}% percentile bootstrap over the player's own games ({N_BOOT} resamples, seed {BOOT_SEED})"
    return f"{text}, {extra}" if extra else text


def data_path(*parts: str) -> str:
    """A table path as the site names it: relative to the data root, which is written as $DATA_ROOT."""
    return "$DATA_ROOT/" + "/".join(str(p).strip("/") for p in parts)


def source(path: str, what: str, rows: str) -> S.Source:
    """A source record. `rows` says which rows were read: heldout, out_of_fold, training or not_split."""
    return S.Source(path=path, what=what, rows=rows)


def merge_sources(*groups) -> list:
    """Sources of several groups without repeats, in a stable order."""
    seen, out = set(), []
    for group in groups:
        for item in group:
            key = (item.path, item.what, item.rows)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return sorted(out, key=lambda s: (s.path, s.what))


def rnd(value, digits: int):
    """A plain rounded float, or None for a missing value. Negative zero is written as zero."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits) + 0.0


def round_list(values, digits: int) -> list:
    """An array as a list of rounded floats with None where the value is missing."""
    arr = np.round(np.asarray(values, dtype=float), digits) + 0.0
    out = arr.tolist()
    bad = ~np.isfinite(arr)
    if bad.any():
        out = [None if b else v for v, b in zip(out, bad.tolist())]
    return out


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def read_table(path: Path, columns=None, what: str = "") -> pd.DataFrame:
    """One parquet table, or ExportProblem naming the table when it is absent or lacks a column."""
    path = Path(path)
    if not path.exists():
        raise ExportProblem(f"missing table: {what or path.name}")
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:  # a missing column or an unreadable file
        # the library's message may hold a local path, so it is not repeated
        raise ExportProblem(f"cannot read {what or path.name} ({type(exc).__name__}); "
                            f"columns asked: {columns or 'all'}") from None


def bootstrap_by_game(per_game: pd.DataFrame, num: str, den: str, scale: float = 1.0) -> pd.DataFrame:
    """Per player: the ratio of sums over his games, and a percentile interval from resampling his own games.

    `per_game` has one row per (player_id, sportvu_game) with the numerator and denominator sums of that game. The
    draw follows the gravity lane (analytics/gravity/train.py, gravity_players): one generator seeded once for the
    table, players in ascending id order, games in sorted order, N_BOOT resamples of the player's games with
    replacement, the 2.5 and 97.5 percentiles. A player with one game has no interval. Nothing is estimated here
    beyond that ratio: no model is involved."""
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    ordered = per_game.sort_values(["player_id", "sportvu_game"], kind="stable")
    for player_id, group in ordered.groupby("player_id", sort=True):
        s = group[num].to_numpy(dtype=float)
        n = group[den].to_numpy(dtype=float)
        point = scale * s.sum() / n.sum()
        lo = hi = float("nan")
        if len(group) >= 2:
            draws = rng.integers(0, len(group), size=(N_BOOT, len(group)))
            values = scale * s[draws].sum(axis=1) / n[draws].sum(axis=1)
            lo, hi = (float(v) for v in np.percentile(values, BOOT_PERCENTILES))
        rows.append({"player_id": int(player_id), "point": float(point), "lo": lo, "hi": hi,
                     "n": float(n.sum()), "n_games": int(len(group))})
    return pd.DataFrame(rows, columns=["player_id", "point", "lo", "hi", "n", "n_games"]).set_index("player_id")


def estimate(value, lo, hi, digits: int, n: int, n_unit: str, interval: str) -> S.Estimate:
    """A contract estimate: with its interval when both ends exist, a point estimate otherwise."""
    value = rnd(value, digits)
    if value is None:
        raise ExportProblem(f"an estimate over {n} {n_unit} has no finite value")
    lo, hi = rnd(lo, digits), rnd(hi, digits)
    if lo is None or hi is None:
        return S.Estimate(value=value, point_estimate=True, n=int(n), n_unit=n_unit)
    return S.Estimate(value=value, lo=lo, hi=hi, interval=interval, point_estimate=False, n=int(n), n_unit=n_unit)


def dump(model: S.Strict) -> str:
    """The file text of a contract model. A NaN or an infinity is an error, never written."""
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"), allow_nan=False) + "\n"


def write_model(out: Path, rel: str, model: S.Strict) -> int:
    """Write one contract file under the output root and return its size in bytes. A file that would carry a
    local path or an address is refused before it reaches the disk."""
    if S.model_for(rel) is not type(model):
        raise ExportProblem(f"{rel}: not the path of a {type(model).__name__} file")
    text = dump(model)
    found = S.find_private(text)
    if found:
        raise ExportProblem(f"{rel}: private string {found[0]!r} would be written")
    path = Path(out) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def family_sha256(hashes) -> str:
    """The hash of a file family as export.schemas checks it: the hash of the sorted member hashes."""
    return hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()


def repo_commit() -> str:
    """The commit of the code that runs, with a marker when the code or the reports it reads differ from that
    commit (changed or new files under CODE_AND_INPUTS)."""
    def git(*args) -> str:
        done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        if done.returncode != 0:
            raise ExportProblem("git is not available, so the manifest cannot name the commit")
        return done.stdout.strip()

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *CODE_AND_INPUTS)
    return head + (UNCOMMITTED if dirty else "")

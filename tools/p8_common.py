"""tools/p8_common.py: paths, the development registry and the guard shared by every P8 Phase A tool (PLAN.md D83 to
D85, docs/research/P8_FEASIBILITY.md sections 7 and 9).

Development games (D85 (3)) are the 4 R1.3 training games of sportvu/splits.json plus the channel games of
reports/research/p8_registry.json whose scorebug check confirmed the date. Every game id and video id a P8 tool
touches passes tools.research_testset.refuse_cv_test first: a broadcast CV test game, a test video or any game played
in a test arena stops the tool. A spent held-out game (sportvu/splits.json, analytics.eval.is_heldout_game) stops it
too.

Outputs live under the P8 workspace (default <home>/Developer/nba-v3-worktrees/_p8-data/work, or V3_P8_ROOT):
outside DATA_ROOT, outside reports/ and outside the iCloud folder (CLAUDE.md rules 6 and 7, D44). The test videos have
their own sealed folder, named only in tools/p8_fetch.py.
"""

import json
import os
import subprocess
from pathlib import Path

from tools.research_testset import refuse_cv_test

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "reports" / "research" / "p8_registry.json"
WHO = "tools.p8"


def p8_root() -> Path:
    """The P8 workspace; never under DATA_ROOT, reports/ or this repository."""
    root = Path(os.environ.get("V3_P8_ROOT") or Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data" / "work")
    check_outside(root)
    return root


def check_outside(path: Path) -> Path:
    """SystemExit when a path lies under DATA_ROOT, under this repository (reports/ included) or under the main
    checkout (the iCloud folder)."""
    p = Path(path).expanduser().resolve()
    bad = [REPO, main_checkout()]
    if os.environ.get("DATA_ROOT"):
        bad.append(Path(os.environ["DATA_ROOT"]).expanduser().resolve())
    for b in bad:
        if b is not None and (p == b or b in p.parents):
            raise SystemExit(f"{WHO}: {p} is under {b}; P8 outputs go outside DATA_ROOT, reports/ and the repository")
    return p


def main_checkout():
    """The main checkout (holds the gitignored models and the R1.3 data), found through git, never typed."""
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(common).resolve().parent


def load_registry(path=None) -> dict:
    return json.loads(Path(path or REGISTRY).read_text())


def guard(items, who: str = WHO, homes=()) -> list:
    """refuse_cv_test (substring, any id, tag, stem, path or URL; plus the given home codes) and the spent held-out
    rule with analytics.eval.is_heldout_game's substring semantics (held-out game tags, SportVU ids, video ids and the
    held-out arena suffix) on every item -> the items, as strings."""
    from analytics.eval import is_heldout_game
    items = [str(i) for i in items]
    refuse_cv_test(items, who, homes=homes)
    spent = [i for i in items if is_heldout_game(i)]
    if spent:
        raise SystemExit(f"{who}: spent held-out item {spent[0]} (sportvu/splits.json); it is never used again")
    return items


def atomic_write(path: Path, text: str) -> Path:
    """Write text to path through a .part file and os.replace, so a killed job never leaves a half file."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    part.write_text(text)
    os.replace(part, path)
    return path


def r13_games() -> dict:
    """The 4 R1.3 training games (sportvu/splits.json train): tag -> {sportvu, video_id, arena}."""
    sp = json.loads((REPO / "sportvu" / "splits.json").read_text())
    out = {t: {"sportvu": g["sportvu"], "video_id": g["video_id"], "home": g["arena"], "source": "R1.3"}
           for t, g in sp["train"]["games"].items()}
    guard([v for t, g in out.items() for v in (t, g["sportvu"], g["video_id"])], homes=[g["home"] for g in out.values()])
    return out


def channel_games(path=None, status=None) -> dict:
    """The channel games of the registry (optionally only one status), guarded."""
    reg = load_registry(path)
    every = reg["games"]
    out = {t: g for t, g in every.items() if status is None or g.get("status") == status}
    ids = list(every) + [g["video_id"] for g in every.values()]          # every entry is checked, whatever its status
    ids += [c for g in every.values() for c in g.get("candidates", [])]
    ids += [g["sportvu"] for g in every.values() if g.get("sportvu")]
    guard(ids, homes=[g.get("home") for g in every.values() if g.get("home")])
    return out


def dev_games(path=None) -> dict:
    """Development games of D85 (3): the R1.3 games plus the confirmed channel games."""
    out = dict(r13_games())
    for t, g in channel_games(path, status="confirmed").items():
        if not g.get("sportvu"):
            raise SystemExit(f"{WHO}: {t} is confirmed but names no SportVU game")
        out[t] = {**g, "source": "channel"}
    return out

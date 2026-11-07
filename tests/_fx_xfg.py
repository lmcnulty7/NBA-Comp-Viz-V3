"""tests/_fx_xfg.py: the test-only adapter of lane XFG over analytics.fixtures.

The games come from ``analytics.fixtures`` (no local generator): synth_season and synth_game return its dicts
with two top-level keys added for the tests, ``sportvu_game`` and ``heldout`` (copies of meta), and
write_fixture_root writes the shared DATA_ROOT layout and returns the root as a Path. Production code never
imports this module (analytics.xfg.validate reads analytics.fixtures directly). Nothing here is real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics import fixtures as _fixtures  # noqa: E402

SEASON = "2015-16"
FPS = _fixtures.HZ


def _flat(g: dict) -> dict:
    """The game dict with meta's sportvu_game and heldout copied to the top level."""
    g["sportvu_game"] = str(g["meta"]["sportvu_game"])
    g["heldout"] = bool(g["meta"]["heldout"])
    return g


def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
    """analytics.fixtures.synth_game with the top-level sportvu_game and heldout keys."""
    return _flat(_fixtures.synth_game(seed, n_possessions=n_possessions, heldout=heldout, sportvu_game=sportvu_game))


def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
    """analytics.fixtures.synth_season with the top-level sportvu_game and heldout keys on every game."""
    return [_flat(g) for g in _fixtures.synth_season(seed, n_games=n_games, heldout_every=heldout_every,
                                                     n_possessions=n_possessions)]


def write_fixture_root(root, seed: int = 0, n_games: int = 8, heldout_every: int = 4, n_possessions: int = 40) -> Path:
    """analytics.fixtures.write_fixture_root (the DATA_ROOT layout the xFG CLIs read); returns the root."""
    _fixtures.write_fixture_root(root, seed=seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions)
    return Path(root)


__all__ = ["synth_game", "synth_season", "write_fixture_root", "FPS", "SEASON"]

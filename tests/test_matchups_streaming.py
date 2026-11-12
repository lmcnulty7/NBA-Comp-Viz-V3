"""tests/test_matchups_streaming.py: the matchups lane holds one game's frames at a time (validate, features, train).

The real season has about 600 games of about 780,000 frame rows each; reading them all at once ran a 16 GB Mac
out of memory. Here a small DATA_ROOT from the local fixture generator is read through a wrapped pandas.read_parquet
that keeps a weak reference to every frames table it returns; at each frames read the test counts how many earlier
frames tables are still alive. No network, no real data."""
import gc
import sys
import weakref
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.matchups import features as F, train as T, validate as V  # noqa: E402
from tests import _fx_matchups as FX  # noqa: E402

N_GAMES = 5


def _is_frames_file(path) -> bool:
    p = Path(str(path))
    return p.parent.name in ("sportvu_contract", "heldout") and "sportvu_contract" in p.parts and not p.name.endswith("_pbp.parquet")


@pytest.fixture
def tracker(monkeypatch):
    """Wrap pandas.read_parquet: record a weakref per frames table read and the most alive at any read."""
    real = pd.read_parquet
    state = {"refs": [], "max_alive": 0, "reads": 0}

    def alive() -> int:
        gc.collect()
        return sum(r() is not None for r in state["refs"])

    def wrapped(path, *a, **k):
        df = real(path, *a, **k)
        if _is_frames_file(path):
            state["refs"].append(weakref.ref(df))
            state["reads"] += 1
            state["max_alive"] = max(state["max_alive"], alive())
        return df

    monkeypatch.setattr(pd, "read_parquet", wrapped)
    state["alive"] = alive
    return state


@pytest.fixture
def root(tmp_path):
    return FX.local_write_fixture_root(tmp_path / "data", seed=7, n_games=N_GAMES, heldout_every=N_GAMES, n_possessions=8)


def test_load_games_reads_no_frames(root, tracker):
    games = V.load_games(root)
    assert len(games) == N_GAMES and tracker["reads"] == 0
    assert all(callable(g["frames"]) for g in games)


def test_validate_streams_one_game(root, tracker):
    V.validate_games(V.load_games(root))
    assert tracker["reads"] == N_GAMES and tracker["max_alive"] <= 1 and tracker["alive"]() == 0


def test_features_streams_one_game(root, tracker, tmp_path):
    res = F.main(["--data", str(root), "--out", str(tmp_path / "out")])
    assert len(res["matchup_possessions"]) > 0
    assert tracker["reads"] == N_GAMES and tracker["max_alive"] <= 1


def test_train_streams_one_game(root, tracker, tmp_path):
    ctx = T.main(["--data", str(root), "--out", str(tmp_path / "out"), "--report", str(tmp_path / "matchups.md"),
                  "--bootstrap", "0", "--folds", "2", "--n-min", "5", "--skip-heldout"])
    assert len(ctx["tables"]["train"]["matchup_possessions"]) > 0 and len(ctx["held_ids"]) == 1
    # one read per game: validation and assignment share it, and the CV folds reuse the derived tables
    assert tracker["reads"] == N_GAMES and tracker["max_alive"] <= 1


def test_streaming_matches_in_memory(root):
    """Loader-backed games give the same tables as the same games with their frames read up front."""
    lazy = V.load_games(root)
    eager = [{**g, "frames": g["frames"]()} for g in V.load_games(root)]
    V.validate_games(lazy)
    V.validate_games(eager)
    a, b = F.run_games(lazy), F.run_games(eager)
    for k in ("matchup_possessions", "matchup_frames", "shots"):
        pd.testing.assert_frame_equal(a[k], b[k])

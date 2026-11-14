"""tests/test_gravity_streaming.py: the gravity lane reads frames one game at a time on a DATA_ROOT.

A real season is 621 games of about 780,000 frame rows each, far more than fits in memory, so validate, features
and train must hold at most one game's frames at once. The test writes a small fixture DATA_ROOT (analytics.fixtures,
synthetic, under tmp_path), wraps pandas.read_parquet so every frames DataFrame it returns is tracked with a weakref
finalizer, and runs the three CLIs. It checks the most frames tables alive at any load (after a gc pass), and that
each game's frames are read once per CLI (train's sign test must not read them again). No network, no real data.
"""
import gc
import sys
import weakref
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from analytics.fixtures import write_fixture_root  # noqa: E402
from analytics.gravity import features as F, train as T, validate as V  # noqa: E402

N_GAMES = 5


class FramesTracker:
    """Counts frames tables returned by read_parquet and how many are alive at once."""

    def __init__(self, read):
        self.read, self.loads, self.alive, self.max_alive = read, 0, 0, 0

    def _released(self):
        self.alive -= 1

    def __call__(self, path, *args, **kwargs):
        df = self.read(path, *args, **kwargs)
        p = Path(path)
        if "sportvu_contract" in p.parts and p.suffix == ".parquet" and not p.name.endswith("_pbp.parquet"):
            gc.collect()                           # count only frames that are still reachable
            self.loads += 1
            self.alive += 1
            self.max_alive = max(self.max_alive, self.alive)
            weakref.finalize(df, self._released)
        return df


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("gravity_root")
    write_fixture_root(root, seed=0, n_games=N_GAMES, heldout_every=4, n_possessions=16)
    return root


@pytest.fixture()
def tracker(monkeypatch):
    t = FramesTracker(pd.read_parquet)
    monkeypatch.setattr(pd, "read_parquet", t)
    return t


def _n_games(root) -> int:
    return len(V.load_data_root(root, log=lambda *_: None))


def test_validate_reads_one_game_at_a_time(data_root, tmp_path, tracker):
    out = V.main(["--data", str(data_root), "--out", str(tmp_path)])
    assert out["games"] == _n_games(data_root) == N_GAMES
    assert tracker.loads == N_GAMES and tracker.max_alive <= 1


def test_features_reads_one_game_at_a_time(data_root, tmp_path, tracker):
    out = F.main(["--data", str(data_root), "--out", str(tmp_path)])
    assert out["games"] == N_GAMES and out["player_frames"] > 0
    assert tracker.loads == N_GAMES and tracker.max_alive <= 1


@pytest.mark.parametrize("skip", [False, True])
def test_train_reads_one_game_at_a_time(data_root, tmp_path, tracker, skip):
    argv = ["--data", str(data_root), "--out", str(tmp_path), "--report", str(tmp_path / "gravity.md"),
            "--bootstrap", "10", "--min-possessions", "5"] + (["--skip-heldout"] if skip else [])
    ctx = T.main(argv)
    assert ctx["heldout_games"] >= 1 and ctx["train_rows"] > 0
    assert ctx["sign_test"]["candidates"]                  # the sign test ran its on-court lookups
    assert tracker.loads == N_GAMES and tracker.max_alive <= 1

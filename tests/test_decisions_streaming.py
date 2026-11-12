"""tests/test_decisions_streaming.py: the decisions lane reads frames one game at a time on a DATA_ROOT.

A real season is 621 games of about 780,000 frame rows each, far more than fits in memory, so validate, features
and train must hold at most one game's frames at once. The test writes a small fixture DATA_ROOT (analytics.fixtures,
synthetic, under tmp_path), wraps pandas.read_parquet so every frames DataFrame it returns is tracked with a weakref
finalizer, and runs the three CLIs. It checks the most frames tables alive at any read (after a gc pass), and that
each game's frames are read once per CLI. A second test checks that train without --out reads the xFG and EPV
outputs under DATA_ROOT/analytics (run_all passes no --out on real data). No network, no real data.
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

from analytics import schema  # noqa: E402
from analytics.decisions import features as F, train as T, validate as V  # noqa: E402
from analytics.fixtures import write_fixture_root  # noqa: E402

N_GAMES = 5
SEASON = "2015-16"


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
    root = tmp_path_factory.mktemp("decisions_root")
    write_fixture_root(root, seed=0, n_games=N_GAMES, heldout_every=4, n_possessions=16)
    return root


@pytest.fixture()
def tracker(monkeypatch):
    t = FramesTracker(pd.read_parquet)
    monkeypatch.setattr(pd, "read_parquet", t)
    return t


def test_validate_reads_one_game_at_a_time(data_root, tmp_path, tracker):
    reps = V.main(["--data", str(data_root), "--out", str(tmp_path), "--season", SEASON])
    assert len(reps) == N_GAMES and all(r["rows"]["frames"] > 0 for r in reps)
    assert tracker.loads == N_GAMES and tracker.max_alive <= 1


def test_features_reads_one_game_at_a_time(data_root, tmp_path, tracker):
    ctx = F.main(["--data", str(data_root), "--out", str(tmp_path), "--season", SEASON])
    assert ctx["sportvu_game"].nunique() == N_GAMES
    assert tracker.loads == N_GAMES and tracker.max_alive <= 1


@pytest.mark.parametrize("skip", [False, True])
def test_train_reads_one_game_at_a_time(data_root, tmp_path, tracker, skip):
    argv = ["--data", str(data_root), "--out", str(tmp_path), "--season", SEASON, "--report", str(tmp_path / "decisions.md"),
            "--folds", "2", "--bootstrap", "5"] + (["--skip-heldout"] if skip else [])
    res = T.main(argv)
    assert len(res["train_ids"]) >= 2 and res["heldout_ids"] and len(res["events"])
    assert bool(len(res["events_heldout"])) is not skip
    assert tracker.loads == N_GAMES and tracker.max_alive <= 1


def test_train_without_out_reads_upstream_from_data_root(tmp_path):
    """No --out: the xFG predictions and EPV states under DATA_ROOT/analytics are used, outputs go to DATA_ROOT."""
    root = tmp_path / "root"
    write_fixture_root(root, seed=1, n_games=3, heldout_every=0, n_possessions=12)
    games = sorted(p.stem for p in (root / "analytics" / "possessions" / SEASON).glob("*.parquet"))
    preds = []
    for g in games:
        shots = pd.read_parquet(schema.table_path(root / "analytics", "shot_frames", SEASON, g))
        preds.append(pd.DataFrame({"sportvu_game": g, "event": shots["event"], "player_id": shots["shooter_id"],
                                   "p_location": 0.4, "p_contest": 0.4, "p_selected": 0.45, "made": shots["made"],
                                   "fold": 0, "heldout": False}))
    xdir = root / "analytics" / "xfg" / SEASON
    xdir.mkdir(parents=True)
    pd.concat(preds, ignore_index=True).to_parquet(xdir / "xfg_predictions.parquet", index=False)
    edir = root / "analytics" / "epv" / SEASON
    edir.mkdir(parents=True)
    pd.DataFrame({"sportvu_game": [games[0]], "possession": [0], "frame": [0], "elapsed_in_possession_s": [0.0],
                  "epv_points": [1.0], "p_score": [0.5], "points_to_end": [0], "fold": [0], "heldout": [False]}
                 ).to_parquet(edir / (games[0] + ".parquet"), index=False)
    res = T.main(["--data", str(root), "--season", SEASON, "--report", str(tmp_path / "decisions.md"),
                  "--folds", "2", "--bootstrap", "5"])
    assert res["has_xfg"] and res["prior"] == "xfg_selected" and res["xfg_coverage"] >= T.XFG_MIN_COVERAGE
    assert all(r["rows"]["xfg"] > 0 for r in res["val_reports"])
    assert next(r for r in res["val_reports"] if r["sportvu_game"] == games[0])["rows"]["epv"] == 1
    assert (root / "analytics" / "decisions" / SEASON / "decision_players.parquet").exists()

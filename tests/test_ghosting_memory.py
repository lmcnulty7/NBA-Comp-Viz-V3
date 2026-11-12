"""analytics/ghosting holds one game's frames at a time (the 621-game season does not fit in RAM).

A synthetic DATA_ROOT is written to tmp_path. Every frames table read from it gets a probe column with a
token unique to that read; the probe column survives check_frames and the slicing in build_panel_game, so
counting the distinct tokens among live DataFrames (after gc.collect) counts the games whose frames are
alive. The count is taken before every frames read, at every panel build and at every fit, through the
validate, features and train CLIs. No network, no real data; every number here is fixture output.
"""
from __future__ import annotations

import gc
import itertools
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import schema  # noqa: E402
from analytics.fixtures import synth_season  # noqa: E402
from analytics.ghosting import features as F, train as T, validate as V  # noqa: E402

PROBE = "_probe_load"


def write_root(root: Path, n_games: int = 4) -> list[str]:
    """Frames, possessions and rosters of synthetic games in the DATA_ROOT layout; returns the game ids."""
    ids = []
    for g in synth_season(0, n_games=n_games, heldout_every=0, n_possessions=6):
        gid, season = g["meta"]["sportvu_game"], g["meta"]["season"]
        (root / "sportvu_contract").mkdir(parents=True, exist_ok=True)
        g["frames"].to_parquet(root / "sportvu_contract" / (gid + ".parquet"), index=False)
        for table in ("possessions", "rosters"):
            p = schema.table_path(root / "analytics", table, season, gid, False)
            p.parent.mkdir(parents=True, exist_ok=True)
            g[table].to_parquet(p, index=False)
        ids.append(gid)
    return ids


def alive_games() -> int:
    """Distinct frames reads still referenced by some live DataFrame."""
    gc.collect()
    tokens = set()
    for o in gc.get_objects():
        if isinstance(o, pd.DataFrame) and PROBE in o.columns and len(o):
            tokens.add(o[PROBE].iloc[0])
    return len(tokens)


@pytest.fixture()
def probe(monkeypatch):
    """Tag every frames read and record the alive-game count at each checkpoint."""
    seen = {"reads": 0, "max": 0, "at": []}
    counter = itertools.count()
    real_read = pd.read_parquet

    def check(where):
        n = alive_games()
        seen["at"].append((where, n))
        seen["max"] = max(seen["max"], n)

    def read(path, *a, **k):
        df = real_read(path, *a, **k)
        if "sportvu_contract" in Path(path).parts:
            check("before frames read")
            seen["reads"] += 1
            df[PROBE] = next(counter)
        return df

    real_panel, real_fit = F.build_panel_game, T.fit_candidates

    def panel(frames, *a, **k):
        check("panel build")
        return real_panel(frames, *a, **k)

    def fit(*a, **k):
        check("fit")
        return real_fit(*a, **k)

    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(F, "build_panel_game", panel)
    monkeypatch.setattr(T, "fit_candidates", fit)
    monkeypatch.setattr(T, "MAX_ITER", 30)
    return seen


def test_ghosting_lane_holds_one_games_frames_at_a_time(tmp_path, probe):
    root = tmp_path / "root"
    ids = write_root(root)
    out = tmp_path / "out"

    rep = V.main(["--data", str(root), "--out", str(out)])
    assert rep["n_games"] == len(ids) and probe["reads"] == len(ids)
    assert alive_games() == 0, "validate kept frames after it returned"

    res = F.main(["--data", str(root), "--out", str(out)])
    assert res["rows"] > 0 and set(res["counts"]) == set(ids)

    ctx = T.main(["--data", str(root), "--out", str(out), "--report", str(out / "ghosting.md"), "--folds", "2",
                  "--bootstrap", "5", "--skip-heldout"])
    assert ctx["train_games"] == sorted(ids) and ctx["ghost_frames_rows"] > 0
    # the real-data path keeps no error rows, no feature frame and no ghost_frames in memory
    assert not ctx["keep_rows"] and ctx["errors"] is None and ctx["frame"] is None and ctx["ghost_frames"] is None
    assert len(list((out / "analytics" / "ghosting" / V.SEASON).glob("ghost_frames_*.parquet"))) == len(ids)
    # every checkpoint ran, and no fit ever saw a game's frames alive
    wheres = {w for w, _ in probe["at"]}
    assert wheres == {"before frames read", "panel build", "fit"}
    assert all(n == 0 for w, n in probe["at"] if w in ("before frames read", "fit")), probe["at"]
    assert probe["max"] <= 1, probe["at"]
    # frames reads per game: validate 1, features 1, train 2 (the fit-row pass while loading, then the 5 Hz scoring)
    assert probe["reads"] == 4 * len(ids)


def test_lazy_and_eager_frames_give_the_same_panel(tmp_path):
    """A loader game and a table game build the same panel and feature frame."""
    root = tmp_path / "root"
    write_root(root, n_games=2)
    lazy = V.iter_data_root(root)
    assert all(callable(g["frames"]) for g in lazy)
    eager = [dict(g, frames=V.frames_of(g)) for g in lazy]
    for g in eager:
        g["validation"] = V.validate_game(g)
    for g in lazy:
        g["validation"] = V.validate_game(g, F.attach_panel)
        assert callable(g["frames"]) and "panel" in g
    pe, fe, ce = F.build(eager)
    pl, fl, cl = F.build(lazy)
    pd.testing.assert_frame_equal(pe, pl)
    pd.testing.assert_frame_equal(fe, fl)
    assert ce == cl
    # a panel is handed over once; a second build reads the frames again and agrees
    assert all("panel" not in g for g in lazy)
    pd.testing.assert_frame_equal(F.build(lazy)[0], pe)

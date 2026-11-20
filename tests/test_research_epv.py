"""epv-E1 machinery (docs/research/epv.md, the D117 amendment): the smoothed label, the final attempt of a possession,
the label fit against the frozen fit, and guards on the module's source. No real data, no held-out game, no test
game; the module has no Set A path at all."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from tools import research_epv as E

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_epv.py"


def _states():
    # possession 1 ends on a made three at clock 700 (2 states before, 1 after); possession 2 has no attempt row
    return pd.DataFrame({"sportvu_game": "G", "possession": [1, 1, 1, 2, 2],
                         "game_clock_s": [710.0, 705.0, 699.0, 650.0, 640.0],
                         "points_to_end": [4, 4, 1, 2, 2]})


def _attempts():
    return pd.DataFrame({"sportvu_game": ["G"], "possession": [1], "event": [7], "clock_s": [700.0], "value": [3],
                         "realized": [3], "is_three": [True]})


def test_the_label_replaces_the_attempts_points_by_its_expectation_before_the_attempt_only():
    y, c = E.smoothed_label(_states(), _attempts(), {("G", 7): 0.4})
    assert np.allclose(y, [4 - 3 + 1.2, 4 - 3 + 1.2, 1, 2, 2])          # free throws (the +1) stay realized
    assert c == {"states_smoothed": 2, "possessions_smoothed": 1}


def test_three_only_leaves_two_point_attempts_realized_and_a_missing_probability_is_not_smoothed():
    a = _attempts().assign(value=2, realized=2, is_three=False)
    y, c = E.smoothed_label(_states(), a, {("G", 7): 0.4}, three_only=True)
    assert np.allclose(y, _states()["points_to_end"]) and c["states_smoothed"] == 0
    y2, c2 = E.smoothed_label(_states(), _attempts(), {})
    assert np.allclose(y2, _states()["points_to_end"]) and c2["possessions_smoothed"] == 0


def test_final_attempts_take_the_last_shot_of_possessions_that_end_on_it(tmp_path, monkeypatch):
    root = tmp_path
    d = root / "analytics" / "possessions" / E.SEASON
    d.mkdir(parents=True)
    pd.DataFrame({"sportvu_game": "G", "possession": [1, 2, 3], "period": 1, "start_event": [1, 10, 20],
                  "end_event": [9, 19, 29], "end_kind": ["shot_missed", "free_throws", "shot_made"]}).to_parquet(d / "G.parquet")
    shots = pd.DataFrame({"sportvu_game": "G", "event": [3, 5, 12, 25], "period": 1, "clock_s": [700, 690, 600, 500],
                          "is_three": [False, True, False, False], "made": [False, False, True, True]})
    a = E.final_attempts(root, ["G"], shots)
    assert a["possession"].tolist() == [1, 3] and a["event"].tolist() == [5, 25]      # possession 2 ends on free throws
    assert a["value"].tolist() == [3, 2] and a["realized"].tolist() == [0, 2]


def test_the_label_fit_equals_the_frozen_fit_when_the_label_is_the_realized_points():
    from analytics.epv import train as T
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({c: rng.normal(size=n) for c in ("a", "b")})
    y = rng.integers(0, 4, size=n).astype(float)
    games = np.array([f"g{i % 12}" for i in range(n)])
    import analytics.schema as S
    orig = S.assert_time_forward
    S.assert_time_forward = lambda *a, **k: None
    try:
        frozen = T.Calibrated("hgb", ["a", "b"], 0, 100, "torch").fit(X, y, games)
        mine = E.fit_label("hgb", ["a", "b"], X, y, y, games)
    finally:
        S.assert_time_forward = orig
    fe, fp = frozen.predict(X)
    me, mp = mine.predict(X)
    assert np.allclose(fe, me) and np.allclose(fp, mp)


def test_the_module_has_no_set_a_path_and_reads_development_games_only():
    text = SOURCE.read_text()
    for banned in ("score_once", "scoring_data_root", "research_data_root", "test_features", "TRACKING",
                   "T.run(", "train.main", "heldout_once"):
        assert banned not in text, banned
    assert "dev_games(" in text and "refuse_test(" in text and "clean_head()" in text
    tree = ast.parse(text)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("scorer", "final")]

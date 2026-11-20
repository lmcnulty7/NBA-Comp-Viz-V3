"""decisions-E2 machinery (docs/research/decisions.md, the D122 amendment): the state builder against the frozen
states, each counterfactual row scored as the last state of its sequence against the frozen feature pipeline on the
stored states, the drive and pass geometry, the metric's sufficient statistics against direct least squares, the
decision rule, and guards on the module's source. Fixture data only: no real game, no held-out game, no test game."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.epv import features as EF
from analytics.epv import train as T
from analytics.epv import validate as V
from tools import research_decisions_e2 as E

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "research_decisions_e2.py"
POS = [f"{s}{i}_{a}" for s in ("off", "def") for i in range(1, 6) for a in ("x", "y")] + ["ball_x", "ball_y", "ball_z"]


@pytest.fixture(scope="module")
def season():
    return V.fixture_season(seed=0, n_games=2, heldout_every=99, n_possessions=6)


def one_game(season):
    g = season["games"][0][0]
    pick = lambda t: season[t][season[t]["sportvu_game" if "sportvu_game" in season[t] else "game"].astype(str) == g]
    frames = season["frames"]
    key = "game" if "game" in frames.columns else "sportvu_game"
    frames = frames[frames[key].astype(str) == g].reset_index(drop=True)
    return g, frames, pick("possessions").reset_index(drop=True), pick("states_1hz").reset_index(drop=True)


class SumModel:
    """A stand-in model: a fixed linear function of the lagged columns, so equal inputs give equal outputs."""

    def __init__(self, n):
        self.w = np.linspace(-1.0, 1.0, n)

    def predict(self, X):
        return np.nan_to_num(np.asarray(X, dtype=float)) @ self.w


def test_state_rows_reproduce_the_frozen_states(season):
    g, frames, poss, states = one_game(season)
    mine = E.state_rows(frames, poss, g, states["frame"], states["possession"])
    assert mine["ok"].all()
    assert np.allclose(mine[POS].to_numpy(float), states[POS].to_numpy(float), equal_nan=True)
    ids = [f"{s}{i}_id" for s in ("off", "def") for i in range(1, 6)] + ["handler_id"]
    assert (mine[ids].to_numpy(float) == states[ids].to_numpy(float)).all()
    assert np.allclose(mine["handler_nearest_def_dist_ft"], states["handler_nearest_def_dist_ft"])
    assert np.abs(mine["elapsed_in_possession_s"] - states["elapsed_in_possession_s"]).max() <= 0.5 + 1e-6


def test_each_row_scored_as_the_last_state_equals_the_frozen_pipeline(season):
    """Scoring a stored state as the last row of its own sequence (the earlier stored states, then itself) gives the
    value the frozen features give it inside the whole possession: the counterfactual pipeline is the frozen one."""
    g, frames, poss, states = one_game(season)
    priors = EF.Priors.from_training(season["states_1hz"], None, None)
    cols = EF.lagged_columns(T.N_LAGS)
    model = SumModel(len(cols))
    X = EF.lagged(EF.build_features(states, priors), T.N_LAGS)
    direct = model.predict(X[cols].to_numpy(float))
    cf = states.copy()
    got = E.score_states(model, priors, states, cf, max_rows=37)          # small chunks on purpose
    assert np.allclose(got, direct, rtol=0, atol=1e-9)


def test_sequences_put_the_row_last_and_raise_its_elapsed():
    stored = pd.DataFrame({"sportvu_game": "g", "possession": [1, 1, 1, 2], "frame": [0, 25, 50, 80],
                           "elapsed_in_possession_s": [0.0, 1.0, 2.0, 0.0]})
    cf = pd.DataFrame({"sportvu_game": "g", "possession": [1, 1, 2], "frame": [30, 0, 81],
                       "elapsed_in_possession_s": [0.9, 0.0, 0.2]})
    seq, last = E.sequences(stored, cf)
    assert list(seq.groupby("possession").size()) == [3, 1, 2]
    assert list(seq.iloc[last]["frame"]) == [30, 0, 81]
    assert list(seq.iloc[last]["elapsed_in_possession_s"]) == [1.0, 0.0, 0.2]


def test_drive_moves_the_handler_8_ft_and_never_nearer_than_3(season):
    g, frames, poss, states = one_game(season)
    rows = E.state_rows(frames, poss, g, states["frame"][:4], states["possession"][:4])
    slot = np.zeros(4, dtype=int)
    bx, by = np.full(4, 5.25), np.full(4, 25.0)
    d0 = np.hypot(rows["off1_x"] - bx, rows["off1_y"] - by).to_numpy()
    out = E.drive_rows(rows, slot, bx, by)
    d1 = np.hypot(out["off1_x"] - bx, out["off1_y"] - by).to_numpy()
    assert np.allclose(d1, np.maximum(d0 - 8.0, 3.0))
    assert (out["handler_id"] == rows["off1_id"]).all()
    assert np.allclose(out["ball_x"] - out["off1_x"], rows["ball_x"] - rows["off1_x"])
    assert np.allclose(out["elapsed_in_possession_s"], rows["elapsed_in_possession_s"] + 1.0)
    assert (out[[f"def{i}_x" for i in range(1, 6)]].to_numpy() == rows[[f"def{i}_x" for i in range(1, 6)]].to_numpy()).all()


def test_pass_option_moves_the_ball_and_the_handler(season):
    g, frames, poss, states = one_game(season)
    rows = E.state_rows(frames, poss, g, states["frame"][:3], states["possession"][:3])
    out = E.with_handler(rows, np.array([2, 2, 2]))
    assert (out["handler_id"] == rows["off3_id"]).all()
    assert np.allclose(out["ball_x"], rows["off3_x"]) and np.allclose(out["ball_y"], rows["off3_y"])


def test_lane_stats():
    d, n = E.lane_stats(np.array([0.0]), np.array([0.0]), np.array([10.0]), np.array([0.0]),
                        np.array([[5.0, 5.0, 20.0, np.nan, 5.0]]), np.array([[1.0, 2.5, 0.0, np.nan, -4.0]]))
    assert d[0] == pytest.approx(1.0) and n[0] == 2


def test_metric_moments_match_least_squares():
    rng = np.random.default_rng(1)
    n = 3000
    v = pd.DataFrame({"sportvu_game": rng.integers(0, 30, n).astype(str), "chosen": rng.normal(1, 0.5, n),
                      "v_hold": rng.normal(1, 0.3, n), "regret": -rng.exponential(0.3, n)})
    v["points_to_end"] = 0.2 + 0.9 * v["chosen"] + rng.normal(0, 1, n)
    v["frozen_chosen"], v["frozen_hold"], v["frozen_regret"] = v["chosen"], v["v_hold"], v["regret"] * 0.5
    v["delta_after"] = rng.normal(0, 0.2, n)
    games, XX, Xy = E._game_moments(v, ["v_hold", "regret"])
    b = E._coef(XX, Xy, np.ones(len(games)))
    X = np.column_stack([np.ones(n), v["v_hold"], v["regret"]])
    assert np.allclose(b, np.linalg.lstsq(X, v["points_to_end"], rcond=None)[0])
    m = E.metric_set(v, n=200)
    assert m["new"]["slope"]["point"] == pytest.approx(np.polyfit(v["chosen"], v["points_to_end"], 1)[0])
    assert m["paired_regret_coef_new_minus_frozen"]["point"] == pytest.approx(
        m["new"]["regret_coef"]["point"] - m["frozen"]["regret_coef"]["point"])
    W = E.boot_weights(30, 50)
    assert W.shape == (50, 30) and (W.sum(axis=1) == 30).all()


def test_decision_rule():
    iv = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}
    m = {"new": {"regret_coef": iv(0.3, 0.1, 0.5), "slope": iv(1.0, 0.9, 1.1)},
         "paired_regret_coef_new_minus_frozen": iv(0.1, 0.02, 0.2)}
    assert E.decide(m)["adopt_on_development"]
    m["new"]["slope"] = iv(1.25, 1.1, 1.4)
    assert not E.decide(m)["adopt_on_development"]
    m["new"]["slope"] = iv(1.0, 0.9, 1.1)
    m["paired_regret_coef_new_minus_frozen"] = iv(0.1, -0.01, 0.2)
    assert not E.decide(m)["adopt_on_development"]


def test_source_guards():
    """No Set A door before development decides (D122): the module never names score_once, research_data_root or
    test_features; it writes only to the results folder or its work folder; and it never runs a lane command line."""
    text = SOURCE.read_text()
    for name in ("score_once", "research_data_root", "test_features", "subprocess", "os.system"):
        assert name not in text, name
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("to_parquet", "write_text"):
            owner = ast.unparse(node.func.value)
            assert any(k in owner for k in ("RESULTS", "out", "target", "rec_path", "work_dir", "gdir", "w ", "folder",
                                            "box", "cf[", "vf[")), owner
    assert "DATA_ROOT\"]) /" not in text.replace("root = Path(os.environ[\"DATA_ROOT\"])", "")


def test_completion_label_d123():
    att = pd.DataFrame({"sportvu_game": ["a", "a", "a", "b", "b"], "completed": [1, 0, 0, 0, 1],
                        "offense_regains": [False, False, True, False, True],
                        "possession_turnover": [False, True, True, False, False]})
    lab = E.completion_label(att)["label"].to_numpy()
    assert lab[0] == 1 and lab[1] == 0 and np.isnan(lab[2]) and np.isnan(lab[3]) and lab[4] == 1
    c = E.attempt_counts(E.completion_label(att))
    assert c["excluded"] == 2 and c["excluded_by_game"] == {"a": 1, "b": 1}
    assert c["completion_rate"] == pytest.approx(2 / 3)


def test_join_events_ignores_the_index():
    """The frozen decisions of a later game keep their table index; the join must not align on it."""
    keys = E.EVENT_KEYS[1:]
    base = pd.DataFrame({"possession": [1, 1], "frame": [10, 20], "player_id": [7, 8], "decision_kind": ["drive", "pass"],
                         "event": [-1, -1], "receiver_id": [-1, 7]})
    ev = base.assign(sportvu_game="g", chosen_value=[0.5, 0.6]).set_index(pd.Index([500, 501]))
    ctx = base.assign(sportvu_game="g", heldout=False, x=[1.0, 2.0])
    out = E._join_events(ctx, ev)
    assert list(out["decision_kind"]) == ["drive", "pass"] and list(out["x"]) == [1.0, 2.0]


def test_outcomes_without_a_stored_state_are_nan():
    dec = pd.DataFrame({"possession": [3, 3, 9, 3], "frame": [5, 30, 40, 60]}, index=[10, 11, 12, 13])
    stored = pd.DataFrame({"possession": [3, 3], "frame": [10, 50], "points_to_end": [2.0, 0.0]})
    out = E.outcomes(dec, stored)
    assert np.isnan(out[0]) and out[1] == 2.0 and np.isnan(out[2]) and out[3] == 0.0


def test_state_rows_flag_frames_that_are_not_five_on_five(season):
    g, frames, poss, states = one_game(season)
    f0 = int(states["frame"].iloc[0])
    drop = frames.index[(frames["frame"] == f0)][:1]
    rows = E.state_rows(frames.drop(drop), poss, g, states["frame"][:2], states["possession"][:2])
    assert not rows["ok"].iloc[0] and rows["not_ok_reason"].iloc[0] == "not five on five"
    assert np.isnan(rows["handler_nearest_def_dist_ft"].iloc[0]) and rows["ok"].iloc[1]


def test_exclusions_by_fold_and_game():
    att = E.completion_label(pd.DataFrame({"sportvu_game": ["a", "a", "b"], "fold": [0, 0, 1], "completed": [1, 0, 0],
                                           "offense_regains": [False, True, False],
                                           "possession_turnover": [False, False, True]}))
    v = pd.DataFrame({"sportvu_game": ["a", "b"], "fold": [0, 1], "decision_kind": ["shot", "drive"],
                      "v_shot": [1.0, np.nan], "v_drive": [0.5, 0.4], "v_hold": [1.0, 1.0],
                      **{f"v_pass{k}": [0.3, np.nan] for k in range(1, 5)},
                      **{f"tm{k}_id": [5, 6 if k == 1 else -1] for k in range(1, 5)}})
    x = E.exclusions(v, att, np.array([True, False]))
    assert x["attempts_by_fold"]["by"]["0"] == {"excluded": 1, "of": 2, "share": 0.5}
    assert x["option_shot_by_game"]["by"]["b"]["excluded"] == 1 and x["option_pass_any_by_fold"]["by"]["1"]["excluded"] == 1
    assert x["primary_left_out_by_kind"]["by"]["drive"]["excluded"] == 1


def test_fold_model_saves_and_reloads_bit_for_bit(tmp_path):
    pytest.importorskip("torch")
    from analytics.epv import deep as D
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5 * 3))
    y = X[:, 0] * 0.3 + rng.normal(size=400)
    m = D.GRUSequenceRegressor(n_steps=5, hidden=4, seed=0, max_epochs=3, patience=3).fit(X, y)
    X[rng.random(X.shape) < 0.05] = np.nan                  # prediction rows with gaps go through the saved imputer
    h = E.save_model(m, tmp_path, X)
    again = E.load_model(tmp_path)
    assert np.array_equal(again.predict(X), m.predict(X)) and h == E.model_hash(tmp_path)

"""tools.p8_resetfix and the P8_TRACKER_RESET_FIX flag of tools.p8_device: callbacks register once after a reset.

The fake below reproduces the three ultralytics 8.4.48 behaviours the bug needs (engine/model.py): Model.track registers
the tracker callbacks when the predictor has no trackers, Model.add_callback appends, and a new predictor takes the
model's callback lists. PlayerTracker.reset's frozen behaviour is predictor = None."""

import sys
import types
from functools import partial

import numpy as np
import pytest

from tools import p8_resetfix as R


def _tracker_module():
    mod = types.ModuleType(R.TRACKER_MODULE)

    def on_predict_start(predictor, persist=False):
        predictor.trackers = ["bot"]

    def on_predict_postprocess_end(predictor, persist=False):
        predictor.updates += 1
    for f in (on_predict_start, on_predict_postprocess_end):
        f.__module__ = R.TRACKER_MODULE
        setattr(mod, f.__name__, f)
    return mod


class FakePredictor:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.updates = 0

    def run(self):
        for c in self.callbacks["on_predict_start"]:
            c(self)
        for c in self.callbacks["on_predict_postprocess_end"]:
            c(self)


class FakeModel:
    def __init__(self, mod):
        self.mod = mod
        self.callbacks = {"on_predict_start": [], "on_predict_postprocess_end": [lambda p: None]}
        self.predictor = None

    def add_callback(self, ev, f):
        self.callbacks[ev].append(f)

    def track(self):
        if not hasattr(self.predictor, "trackers"):
            self.add_callback("on_predict_start", partial(self.mod.on_predict_start, persist=True))
            self.add_callback("on_predict_postprocess_end", partial(self.mod.on_predict_postprocess_end, persist=True))
        if self.predictor is None:
            self.predictor = FakePredictor(self.callbacks)
        before = self.predictor.updates
        self.predictor.run()
        return self.predictor.updates - before


class FakeTracker:
    def __init__(self, mod):
        self.model = FakeModel(mod)

    def reset(self):                     # the frozen detect.tracker.PlayerTracker.reset, reduced to what matters
        self.model.predictor = None


def test_bug_doubles_and_fix_restores_one_update_per_frame(monkeypatch):
    mod = _tracker_module()
    bug = FakeTracker(mod)
    assert [bug.model.track() for _ in range(3)] == [1, 1, 1]
    bug.reset()
    assert [bug.model.track() for _ in range(3)] == [2, 2, 2]          # the double update
    bug.reset()
    assert bug.model.track() == 3                                       # each reset adds one more
    fixed = FakeTracker(mod)
    orig = FakeTracker.reset

    def reset(self):
        orig(self)
        R.strip_tracker_callbacks(self.model.callbacks)
    monkeypatch.setattr(FakeTracker, "reset", reset)
    assert [fixed.model.track() for _ in range(2)] == [1, 1]
    for _ in range(3):
        fixed.reset()
        assert [fixed.model.track() for _ in range(2)] == [1, 1]        # exactly once after every reset
    # the non-tracker default callback is kept
    assert sum(1 for c in fixed.model.callbacks["on_predict_postprocess_end"] if not R.is_tracker_callback(c)) == 1


def test_is_tracker_callback_and_strip():
    mod = _tracker_module()
    cbs = {"on_predict_start": [partial(mod.on_predict_start, persist=True)],
           "on_predict_postprocess_end": [print, partial(mod.on_predict_postprocess_end, persist=True),
                                          partial(mod.on_predict_postprocess_end, persist=True)]}
    assert not R.is_tracker_callback(print) and not R.is_tracker_callback(partial(print))
    assert R.strip_tracker_callbacks(cbs) == 3
    assert cbs == {"on_predict_start": [], "on_predict_postprocess_end": [print]}


def test_device_flag():
    from tools import p8_device as D
    assert D.reset_fix_requested({}) is False and D.reset_fix_requested({"P8_TRACKER_RESET_FIX": "1"}) is True
    assert D.is_set({"P8_TRACKER_RESET_FIX": "1"})
    with pytest.raises(SystemExit):
        D.reset_fix_requested({"P8_TRACKER_RESET_FIX": "yes"})


def test_apply_reset_fix_patches_player_tracker_and_logs(monkeypatch, tmp_path):
    """apply_reset_fix on a stand-in detect.tracker module: the frozen reset still runs, the tracker callbacks go, the
    log records 2 stripped per reset, and the log file is written."""
    mod = _tracker_module()
    fake_dt = types.ModuleType("detect.tracker")
    fake_dt.PlayerTracker = type("PlayerTracker", (FakeTracker,), {})
    pkg = types.ModuleType("detect")
    pkg.tracker = fake_dt
    monkeypatch.setitem(sys.modules, "detect", pkg)
    monkeypatch.setitem(sys.modules, "detect.tracker", fake_dt)
    monkeypatch.setattr(R, "RESET_LOG", {"path": None, "resets": []})
    monkeypatch.setattr("atexit.register", lambda f: None)
    assert R.apply_reset_fix() == "detect.tracker.PlayerTracker.reset"
    t = fake_dt.PlayerTracker(mod)
    assert t.model.track() == 1
    for _ in range(3):
        t.reset()
        assert t.model.predictor is None                                 # the frozen reset still ran
        assert [t.model.track() for _ in range(2)] == [1, 1]
    assert R.RESET_LOG["resets"] == [{"stripped": 2}] * 3
    R.start_reset_log(tmp_path / "w_resetfix.json")
    R.write_reset_log()
    import json
    assert json.loads((tmp_path / "w_resetfix.json").read_text()) == {"reset_fix": True, "resets": [{"stripped": 2}] * 3}


def test_real_ultralytics_callbacks_are_recognised():
    """With ultralytics importable (the frozen CV interpreter), its own tracker partials are the ones stripped."""
    TR = pytest.importorskip("ultralytics.trackers.track")
    assert R.is_tracker_callback(partial(TR.on_predict_start, persist=True))
    assert R.is_tracker_callback(partial(TR.on_predict_postprocess_end, persist=True))


# ── design v2: the rule and the helpers ─────────────────────────────────────────────────────────────────────────────
def _good():
    point = {"missed_rate": -0.01, "detection_rate": -0.005, "ghost_rate": 0.002, "position_p50_ft": 0.0,
             "position_p90_ft": 0.02, "nearfield_bias_ft": 0.0, "team_accuracy": 0.0, "missed_1to1_rate": -0.008}
    wiv = {"missed_rate": [-0.02, -0.001], "detection_rate": [-0.01, 0.001], "ghost_rate": [-0.001, 0.008],
           "position_p50_ft": [-0.03, 0.04], "position_p90_ft": [-0.05, 0.09], "nearfield_bias_ft": [-0.05, 0.05],
           "team_accuracy": [-0.02, 0.02]}
    return point, wiv


def test_v2_verdict_passes_and_each_condition_fails_alone():
    point, wiv = _good()
    assert R.v2_verdict(point, wiv, (0.8, 0.8))["pass"]
    for row, iv in (("ghost_rate", [0.0, 0.011]), ("position_p50_ft", [0.0, 0.051]), ("position_p90_ft", [0.0, 0.11]),
                    ("nearfield_bias_ft", [0.0, 0.101]), ("team_accuracy", [-0.031, 0.0])):
        w = dict(wiv); w[row] = iv
        v = R.v2_verdict(point, w, (0.8, 0.8))
        assert not v["pass"] and not v["rows"][row]["pass"], row
    w = dict(wiv); w["missed_rate"] = [0.001, 0.02]                          # worse by the window interval
    assert not R.v2_verdict(point, w, (0.8, 0.8))["pass"]
    assert not R.v2_verdict(point, wiv, (0.8, 0.8), {"detection_rate": [0.001, 0.01]})["pass"]   # worse by the game iv
    p = dict(point); p["missed_rate"] = 0.001                                # does not improve
    assert not R.v2_verdict(p, wiv, (0.8, 0.8))["pass"]
    assert R.v2_verdict(p, wiv, (0.8, 0.8), improve_rows=(), one_to_one=True)["pass"]
    p = dict(point); p["missed_1to1_rate"] = 0.0
    assert not R.v2_verdict(p, wiv, (0.8, 0.8))["pass"] and R.v2_verdict(p, wiv, (0.8, 0.8), one_to_one=False)["pass"]
    assert not R.v2_verdict(point, wiv, (0.8, 0.49))["pass"]                 # coverage floor
    w = dict(wiv); w["ghost_rate"] = None                                    # no interval fails a margin row
    assert not R.v2_verdict(point, w, (0.8, 0.8))["pass"]
    # a margin row is judged only by its margin: a ghost interval wholly above zero but under the margin passes
    w = dict(wiv); w["ghost_rate"] = [0.001, 0.009]
    assert R.v2_verdict(point, w, (0.8, 0.8))["pass"]


def test_pool_verdict_needs_every_game():
    point, wiv = _good()
    ok = R.v2_verdict(point, wiv, (0.8, 0.8))
    w = dict(wiv); w["ghost_rate"] = [0.0, 0.02]
    bad = R.v2_verdict(point, w, (0.8, 0.8), improve_rows=(), one_to_one=False)
    v = R.pool_verdict(ok, {"a": ok, "b": bad})
    assert not v["pass"] and v["pooled"] and "b:ghost_rate" in v["failing"]
    assert R.pool_verdict(ok, {"a": ok})["pass"]


def test_ratio_interval_matches_the_registered_boot():
    from tools.p8_e1_score import boot
    rng = np.random.default_rng(3)
    wins = []
    for _ in range(12):
        b = {"missed": int(rng.integers(5, 40)), "in_frame": 100, "nr_ghosts": int(rng.integers(0, 10)), "nr_boxes": 80,
             "cov_acc": 5.0, "cov_live": 8.0, "team": {"n": 10, "majority": 8},
             "matched": {f"{i}:0": [float(rng.random()), 0.1, "near"] for i in range(5)}}
        n = dict(b); n["missed"] = b["missed"] - int(rng.integers(0, 4))
        wins.append({"r23": b, "new": n})
    real = boot({i: [w] for i, w in enumerate(wins)}, n=300)
    pk = R.resample_picks(len(wins), 300, 0)
    for row, (num, den) in (("missed_rate", ("missed", "in_frame")), ("ghost_rate", ("nr_ghosts", "nr_boxes"))):
        got = R.ratio_interval(pk, [w["r23"][num] for w in wins], [w["r23"][den] for w in wins],
                               [w["new"][num] for w in wins], [w["new"][den] for w in wins])
        assert got == real[row]


def test_sim_arm_changes_only_reset_windows():
    base = {"missed": 50, "far": 20, "nr_boxes": 100, "nr_ghosts": 10, "in_frame": 200}
    after = {"reach": 10, "far_reach": 4, "ours_boxes": 40, "nr_boxes": 20}
    assert R.sim_arm(base, after, 0, 0.7, 0.2) == base
    new = R.sim_arm(base, after, 1, 0.7, 0.25)
    assert new == {"missed": 43, "far": 17, "nr_boxes": 114, "nr_ghosts": 14, "in_frame": 200}
    assert base["missed"] == 50


def test_truth_window_and_summary():
    fr = [{"frame": f, "boxes": [{"tid": t} for t in ((1, 2, 3, 4, 5) if f < 30 else (9,))]} for f in range(0, 60, 3)]
    w = R.truth_window(fr, {0, 3, 6, 9, 33, 100}, 1)
    assert w["first_reset_frame"] == 30 and w["proc_before"] == 10 and w["proc_after"] == 10
    assert w["acc_before"] == 4 and w["acc_after"] == 1 and w["acc_not_processed"] == 1
    s = R.truth_summary({"a": w, "b": dict(w)}, n_boot=50)
    assert s["rate_before_in_reset_windows"] == 0.4 and s["rate_after_in_reset_windows"] == 0.1
    assert abs(s["ratio_before_over_after"] - 4.0) < 1e-12 and abs(s["under_representation"] - 2.5) < 1e-12


def test_frame_counts():
    sv = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]] + [[100.0, 100.0]] * 7)
    inside = np.array([True, True, True] + [False] * 7)
    feet = np.array([[0.5, 0.0], [40.0, 40.0]])             # finds player 0; one ghost
    ours = np.array([[0.2, 0.0], [11.0, 0.0], [23.0, 0.0]])    # reaches player 1 within 2.5; player 2 within 5
    c = R.frame_counts(feet, sv, inside, [False, False], ours)
    assert c["in_frame"] == 3 and c["missed"] == 2 and c["nr_ghosts"] == 1 and c["nr_boxes"] == 2
    assert c["reach"] == 1 and c["far"] == 2 and c["far_reach"] == 2 and c["ours_ghosts"] == 0

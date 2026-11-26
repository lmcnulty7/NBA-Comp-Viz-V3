"""tools.p8_resetfix and the P8_TRACKER_RESET_FIX flag of tools.p8_device: callbacks register once after a reset.

The fake below reproduces the three ultralytics 8.4.48 behaviours the bug needs (engine/model.py): Model.track registers
the tracker callbacks when the predictor has no trackers, Model.add_callback appends, and a new predictor takes the
model's callback lists. PlayerTracker.reset's frozen behaviour is predictor = None."""

import sys
import types
from functools import partial

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

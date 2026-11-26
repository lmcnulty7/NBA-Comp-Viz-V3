"""tools.p8_a3trace: the pure reading rules of the instrumented replay on synthetic trace records."""

from tools import p8_a3trace as M

BOX = [0, 0, 10, 20, 0.9]


def frame(**k):
    base = {"frame_id": 5, "input": [{"xyxy": [0, 0, 10, 20], "conf": 0.9, "idx": 0}], "assoc": [], "output": [],
            "removed_dup": [], "tracked_after": []}
    base.update(k)
    return base


def test_classify_drop():
    assert M.classify_drop(BOX, frame(input=[])) == "G_not_in_input"
    assert M.classify_drop(BOX, frame(input=[{"xyxy": [5, 0, 15, 20], "conf": 0.9, "idx": 0}])) == "G_not_in_input"
    assert M.classify_drop(BOX, frame(assoc=[{"thresh": 0.8, "matches": [[7, 0]]}])) == "F_matched_not_output"
    assert M.classify_drop(BOX, frame(removed_dup=[{"id": 9, "idx": 0}])) == "E_removed_as_duplicate"
    new = [{"id": 9, "activated": False, "state": 1, "start_frame": 5, "idx": 0, "box": [0, 0, 10, 20]}]
    assert M.classify_drop(BOX, frame(tracked_after=new)) == "D_new_unconfirmed"
    assert M.classify_drop(BOX, frame()) == "U_unmatched_no_new_track"
    assert M.classify_drop(BOX, frame(assoc=[{"thresh": 0.8, "matches": [[7, 0]]}],
                                      output=[{"id": 7, "idx": 0}])) == "H_other"


def test_step_of():
    assert M.step_of({"D_new_unconfirmed": 0.6, "G_not_in_input": 0.4}) == "D_new_unconfirmed"
    assert M.step_of({"D_new_unconfirmed": 0.49, "G_not_in_input": 0.51}) == "G_not_in_input"
    assert M.step_of({"D_new_unconfirmed": 0.4, "G_not_in_input": 0.4}) is None

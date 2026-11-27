"""tools.p8_recallcause: the pure rules (matching, suppression clusters, classes, alignment, shares) and the writes."""

import ast
import inspect

import numpy as np

from tools import p8_recallcause as R


def test_iou_match_and_same_boxes():
    assert R.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and R.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert R.match_boxes([[0, 0, 10, 10], [50, 50, 60, 60]], [[51, 50, 61, 60], [0, 1, 10, 11]]) == {0: 1, 1: 0}
    assert R.same_boxes([[1.004, 2, 3, 4, 0.912341]], [[1.0, 2.0, 3.0, 4.0, 0.91234]])
    assert not R.same_boxes([[1.02, 2, 3, 4, 0.9]], [[1.0, 2.0, 3.0, 4.0, 0.9]])
    assert not R.same_boxes([], [[1, 2, 3, 4, 0.9]])


def test_suppressors_and_c2():
    post = [[0, 0, 10, 20, 0.9]]
    pre = [[0, 0, 10, 20, 0.9], [1, 0, 11, 20, 0.6], [100, 0, 110, 20, 0.5]]
    supp = R.suppressors(pre, post, [7, 8, 9], [7])
    assert supp == {1: 0, 2: None}                                       # 2 overlaps no kept box
    sv = np.array([[0.0, 0.0], [3.0, 0.0]] + [[50.0, 50.0]] * 8)
    pre_feet = np.array([[0.2, 0.0], [3.1, 0.0], [80.0, 80.0]])
    post_feet = np.array([[0.2, 0.0]])
    assert R.is_c2(1, sv, pre_feet, post_feet, supp) == (True, 1)        # j=1's candidate suppressed by player 0's box
    assert R.is_c2(0, sv, pre_feet, post_feet, supp) == (False, None)    # a duplicate of j's own kept box


def test_classify_order():
    sv = np.array([[0.0, 0.0]] + [[50.0, 50.0]] * 9)
    far = [np.array([9.0, 0.0])]
    assert R.classify(0, sv, (5, 5), None, [], [], [], True, None, None, None) == "C2"
    assert R.classify(0, sv, (5, 5), None, [[0, 0, 10, 10]], far, [], False, None, None, None) == "C1a"
    assert R.classify(0, sv, (5, 5), None, [], [], [np.array([1.0, 0.0])], False, None, None, None) == "C1b"
    assert R.classify(0, sv, (5, 5), None, [], [], [], False, None, None, None) == "C1c"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, None, None, True) == "C3_late"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, None, None, None) == "C3_undetermined"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, 1, np.array([3.0, 0.0]), None) == "C4a"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, 1, np.array([9.0, 0.0]), None) == "C4b"
    assert R.covers([0, 0, 10, 10], (5, 10.5)) and not R.covers([0, 0, 10, 10], (5, 12))


def test_alignment_ok():
    assert R.alignment_ok([(0.5, 1.0, 0.6), (0.4, 0.95, 0.5)])
    assert not R.alignment_ok([(0.95, 0.9, 0.5)]) and not R.alignment_ok([(0.2, 0.8, 0.3)]) and not R.alignment_ok([])


def test_share_table_groups_and_main_loss():
    rows = [(f"w{i:02d}", "C1c") for i in range(12) for _ in range(3)] + [(f"w{i:02d}", "C3_late") for i in range(12)]
    t = R.share_table(rows, n_boot=200)
    assert t["misses"] == 48 and abs(t["classes"]["C1c"]["share"] - 0.75) < 1e-12
    assert R.groups_of(t)["C1"]["count"] == 36 and R.groups_of(t)["C3"]["count"] == 12
    m = R.main_loss(t)
    assert m["class"] == "C1c" and m["reading"] == "main loss"
    t2 = R.share_table([("w0", "C1c"), ("w1", "C4a"), ("w2", "C4a"), ("w3", "C1c"), ("w4", "C2")], n_boot=200)
    assert R.main_loss(t2)["reading"] == "largest, not main"


def test_writes_only_its_two_destinations():
    src = inspect.getsource(R)
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    targets = sorted(ast.unparse(c.args[0]) for c in calls)
    assert targets == ["out_path()", "raw_d / f'{s}.json'"]
    assert "cmd_measure_dev" not in src and '"truth_accept"' not in src.replace('"truth_accept", "cache"', "")

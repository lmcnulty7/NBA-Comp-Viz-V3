"""tools.p8_recallcause: the pure rules (matching, suppression clusters, classes, alignment, shares) and the writes."""

import ast
import inspect

import numpy as np
import pytest

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
    # NMS suppresses only above iou_thres: IoU exactly 0.45 is not a suppressor, just above it is
    edge = [[0, 0, 20, 10, 0.9]]
    assert R.iou([0, 0, 9, 10], edge[0]) == R.IOU
    assert R.suppressors([[0, 0, 20, 10, 0.9], [0, 0, 9, 10, 0.5]], edge, [1, 2], [1]) == {1: None}
    assert R.suppressors([[0, 0, 20, 10, 0.9], [0, 0, 9.1, 10, 0.5]], edge, [1, 2], [1]) == {1: 0}


def test_classify_order():
    sv = np.array([[0.0, 0.0]] + [[50.0, 50.0]] * 9)
    far = [np.array([9.0, 0.0])]
    box = [[0, 0, 10, 10]]
    assert R.classify(0, sv, (5, 5), None, [], [], [], True, None, None, None) == "C2"
    assert R.classify(0, sv, (5, 5), None, box, far, [], False, None, None, None) == "C1a_own"
    assert R.classify(0, sv, (5, 5), None, box, far, [], False, None, None, None, [None]) == "C1a_own"
    assert R.classify(0, sv, (5, 5), None, box, far, [], False, None, None, None, [3]) == "C1a_other"
    assert R.classify(0, sv, (5, 5), None, box * 2, far * 2, [], False, None, None, None, [3, None]) == "C1a_own"
    ref = [np.array([1.0, 0.0])]                                          # referee class feet only (M1)
    assert R.classify(0, sv, (5, 5), None, box, far, ref, False, None, None, None, [3]) == "C1a_other"
    assert R.classify(0, sv, (5, 5), None, [], [], ref, False, None, None, None) == "C1b"
    assert R.classify(0, sv, (5, 5), None, [], [], [], False, None, None, None) == "C1c"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, None, None, True) == "C3_late"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, None, None, False) == "C3_not_confirmed"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, None, None, None) == "C3_undetermined"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, 1, np.array([3.0, 0.0]), None) == "C4a"
    assert R.classify(0, sv, (5, 5), 3, [], [], [], False, 1, np.array([9.0, 0.0]), None) == "C4b"
    assert R.covers([0, 0, 10, 10], (5, 10.5)) and not R.covers([0, 0, 10, 10], (5, 12))
    assert set(c for cs in R.GROUPS.values() for c in cs) == set(R.CLASSES)


def test_alignment_ok():
    assert R.alignment_ok([(0.5, 1.0, 0.6), (0.4, 0.95, 0.5)])
    assert not R.alignment_ok([(0.95, 0.9, 0.5)]) and not R.alignment_ok([(0.2, 0.8, 0.3)]) and not R.alignment_ok([])


def test_alignment_compares_across_frames():
    """Code check B2: R_post at f against the sidecar at f - stride, f and f + stride, not each frame with itself."""
    def boxes(dx):
        return [[100 * k + dx, 0, 100 * k + dx + 20, 50] for k in range(10)]
    prev, at, nxt = boxes(-12), boxes(0), boxes(12)                       # players move 12 px per stride
    post_at = [b + [0.9, 0] for b in at]
    sh = R.align_shares(post_at, prev, at, nxt)
    assert sh[1] == 1.0 and sh[0] < 0.5 and sh[2] < 0.5 and R.alignment_ok([sh] * 5)
    post_late = [b + [0.9, 0] for b in nxt]                               # decoded one stride late
    sh = R.align_shares(post_late, prev, at, nxt)
    assert sh[2] == 1.0 and not R.alignment_ok([sh] * 5)
    assert R.align_shares(post_at, [], at, []) == (1.0, 1.0, 1.0)


def test_nms_once_does_not_change_its_input():
    """Code check B1: non_max_suppression rewrites its input in place; two calls on one raw tensor must equal a call
    on a fresh copy."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(1, 2, 400, generator=g) * 600
    wh = torch.rand(1, 2, 400, generator=g) * 80 + 10
    cls = torch.rand(1, 5, 400, generator=g)
    raw = torch.cat([xy, wh, cls], 1)
    keep = raw.clone()
    a, ia = R.nms_once(raw, 0.40, 0.45, [0], 300)
    b, ib = R.nms_once(raw, 0.40, 1.0, [0], 30000)
    c, ic = R.nms_once(raw, 0.40, 0.45, [0], 300)
    fb, fib = R.nms_once(keep.clone(), 0.40, 1.0, [0], 30000)
    assert torch.equal(raw, keep)
    assert torch.equal(a[0], c[0]) and torch.equal(ia[0], ic[0])
    assert torch.equal(b[0], fb[0]) and torch.equal(ib[0], fib[0]) and len(a[0]) > 0


def test_share_table_groups_and_main_loss():
    rows = [(f"w{i:02d}", "C1c") for i in range(12) for _ in range(2)] + \
           [(f"w{i:02d}", "C1a_own") for i in range(12)] + [(f"w{i:02d}", "C3_late") for i in range(12)]
    t = R.share_table(rows, n_boot=200)
    assert t["misses"] == 48 and abs(t["classes"]["C1c"]["share"] - 0.5) < 1e-12
    g = R.groups_of(rows, n_boot=200)
    assert g["C1"]["count"] == 36 and g["C3"]["count"] == 12 and g["other"]["count"] == 0
    assert g["C1"]["subclasses"]["C1c"]["count"] == 24 and g["C1"]["subclasses"]["C1a_own"]["count"] == 12
    assert g["C1"]["interval"] is not None and abs(g["C1"]["share"] - 0.75) < 1e-12
    m = R.main_loss(g)                                    # read on the groups (version 2.1), not the largest class
    assert m["group"] == "C1" and m["reading"] == "main loss"
    rows2 = [("w0", "C1c"), ("w1", "C4a"), ("w2", "C4a"), ("w3", "C1c"), ("w4", "C2")]
    assert R.main_loss(R.groups_of(rows2, n_boot=200))["reading"] == "largest, not main"


def test_writes_only_its_two_destinations():
    src = inspect.getsource(R)
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    targets = sorted(ast.unparse(c.args[0]) for c in calls)
    assert targets == ["out_path()", "raw_d / f'{s}.json'"]
    assert "cmd_measure_dev" not in src and '"truth_accept"' not in src.replace('"truth_accept", "cache"', "")

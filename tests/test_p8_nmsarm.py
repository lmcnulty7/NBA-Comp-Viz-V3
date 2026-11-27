"""tools.p8_nmsarm (stages 0a and 0b): the allow-list, the replay check, the frame rows, the C2 candidates, the added
box split, the gates and the writes."""

import ast
import inspect

import numpy as np
import pytest

from tools import p8_nmsarm as N


def test_allow_list_refuses_hou_and_sac():
    assert N.allowed(["gsw_cha_2016", "gsw_bkn_2015_s04_w00"]) == ["gsw_cha_2016", "gsw_bkn_2015_s04_w00"]
    for bad in ("hou_cle_2016", "hou_cle_2016_s00_w00", "gsw_sac_2015", "gsw_sac_2015_s01_w02", "gsw_cha_2016x"):
        with pytest.raises(SystemExit):
            N.allowed(["gsw_cha_2016", bad])


def test_constants_are_the_scorers():
    from sportvu.eval import GHOST_FT, MISS_FT
    from tools import p8_recallcause as RC
    assert (MISS_FT, GHOST_FT) == (2.5, 3.0) and RC.MISS_FT == MISS_FT and N.BASE_IOU == RC.IOU
    nums = {n.value for n in ast.walk(ast.parse(inspect.getsource(N))) if isinstance(n, ast.Constant)
            and isinstance(n.value, float)}
    assert not nums & {2.5, 3.0, 5.0}                                     # radii are imported, never typed


def test_keep_equals_post_and_ids():
    pre = [[0, 0, 10, 20, 0.9, 0], [1, 0, 11, 20, 0.6, 0], [50, 0, 60, 20, 0.5, 0]]
    post = [[50, 0, 60, 20, 0.5, 0], [0, 0, 10, 20, 0.9, 0]]
    assert N.keep_equals_post(pre, [0, 2], post) and not N.keep_equals_post(pre, [0, 1], post)
    assert N.post_ids_of(pre, post) == [2, 0]
    with pytest.raises(SystemExit):
        N.post_ids_of(pre, [[1, 1, 1, 1, 0.1, 0]])
    assert N.touches_edge([[0, 5, 10, 20]], 1280, 720) and not N.touches_edge([[1, 5, 10, 20]], 1280, 720)


def test_nms_keep_matches_its_threshold():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pre = [[0, 0, 10, 20, 0.9, 0], [3, 0, 13, 20, 0.8, 0], [100, 0, 110, 20, 0.7, 0]]   # IoU of the first two 7/13
    assert N.nms_keep(pre, 0.45) == [0, 2] and N.nms_keep(pre, 0.55) == [0, 1, 2]
    assert N.nms_keep(pre, 0.55, max_det=2) == [0, 1] and N.nms_keep([], 0.45) == []


def test_frame_counts_see_duplicates():
    sv = np.array([[0.0, 0.0], [10.0, 0.0]] + [[60.0, 60.0]] * 8)
    inside = np.array([True, True] + [False] * 8)
    one = N.frame_counts(np.array([[0.2, 0.0], [10.1, 0.0]]), sv, inside)
    assert (one["u1"], one["mg"], one["dup"]) == (0, 0, 0)
    dup = N.frame_counts(np.array([[0.2, 0.0], [0.5, 0.3], [10.1, 0.0]]), sv, inside)
    assert (dup["u1"], dup["mg"], dup["dup"]) == (1, 0, 1)          # the nearest rows cannot see it; one-to-one can
    ghost = N.frame_counts(np.array([[0.2, 0.0], [10.1, 0.0], [30.0, 30.0]]), sv, inside)
    assert (ghost["u1"], ghost["mg"], ghost["ghosts"]) == (1, 1, 1)
    none = N.frame_counts(np.zeros((0, 2)), sv, inside)
    assert (none["u1"], none["missed"], none["boxes"]) == (2, 2, 0)


def test_c2_candidates_and_added_split():
    sv = np.array([[0.0, 0.0], [3.0, 0.0]] + [[50.0, 50.0]] * 8)
    inside = np.array([True, True] + [False] * 8)
    pre_feet = np.array([[0.2, 0.0], [3.1, 0.0], [3.0, 0.4], [80.0, 80.0]])
    post_feet = np.array([[0.2, 0.0]])
    supp = {1: 0, 2: 0, 3: None}
    assert N.c2_candidates(1, sv, pre_feet, post_feet, supp) == [1, 2]
    assert N.c2_candidates(0, sv, pre_feet, post_feet, supp) == []
    sp = N.added_split([2, 3], pre_feet, np.array([[0.2, 0.0], [3.1, 0.0]]), sv, inside)
    assert sp == {"duplicates": 1, "ghost_candidates": 1, "other": 0}


def test_verdict_gates():
    def dl(a, b):
        return {"0.55": {"u1": a}, "0.7": {"u1": b}}
    ok = {"0.55": 200, "0.7": 300}
    assert N.verdict(dl(-10, -20), ok, 421)["arm"] == "0.7"
    assert N.verdict(dl(-10, -5), ok, 421)["arm"] == "0.55"               # K2: 0.70 not below 0.55
    assert N.verdict(dl(5, 10), ok, 421)["arm"] is None                   # K1 kills both
    assert N.verdict(dl(-10, -20), {"0.55": 100, "0.7": 300}, 421)["arm"] == "0.7"
    assert N.verdict(dl(-10, -20), {"0.55": 100, "0.7": 100}, 421)["arm"] is None   # G0: under 0.25 of 421
    assert N.verdict(dl(0, 3), ok, 421)["gates"]["0.55"]["K1"] is False


def test_deltas_and_interval():
    pw = {f"w{i}": {"0.45": {k: 10 for k in N.ROWS}, "0.55": {k: 9 for k in N.ROWS}, "0.7": {k: 12 for k in N.ROWS}}
          for i in range(5)}
    d = N.deltas(pw, list(pw))
    assert d["0.55"]["u1"] == -5 and d["0.7"]["dup"] == 10
    assert N.delta_interval(pw, list(pw), "0.55", "u1", n=200) == [-5.0, -5.0]


def test_writes_only_its_records():
    src = inspect.getsource(N)
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    assert [ast.unparse(c.args[0]) for c in calls] == ["out_path(stage)"]
    writes = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_write"]
    assert sorted(ast.unparse(c.args[0]) for c in writes) == ["'replay'", "'screen'"]
    assert "write_text" not in src and ".mkdir(" not in src

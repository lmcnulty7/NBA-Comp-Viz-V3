"""tools.p8_recallsupp: the copied read functions equal posefoot's, D208's classification as one function, the gates'
comparisons, the allow-list and the writes."""

import ast
import inspect
import subprocess

import numpy as np
import pytest

from tools import p8_recallsupp as SU


def _defs(src: str) -> dict:
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in SU.COPIED:
            for m in ast.walk(n):                      # docstrings aside
                if isinstance(m, (ast.FunctionDef, ast.ClassDef)) and m.body and isinstance(m.body[0], ast.Expr) \
                        and isinstance(getattr(m.body[0], "value", None), ast.Constant) \
                        and isinstance(m.body[0].value.value, str):
                    m.body = m.body[1:] or [ast.Pass()]
            out[n.name] = ast.dump(n)
    return out


def test_read_functions_equal_posefoots():
    from tools.p8_common import REPO
    p = subprocess.run(["git", "show", f"{SU.POSEFOOT_COMMIT}:tools/p8_posefoot.py"], cwd=REPO, capture_output=True,
                       text=True)
    if p.returncode != 0:
        pytest.skip("posefoot's commit is not in this repository")
    theirs, mine = _defs(p.stdout), _defs(inspect.getsource(SU))
    assert set(mine) == set(SU.COPIED) == set(theirs)
    assert all(mine[k] == theirs[k] for k in SU.COPIED)


def test_main_reads_follow_intervals_and_max_frames():
    class R:
        def __init__(self):
            self.next = 0

        def read(self, i):
            self.next = i + 1
            return True, i
    out, reads = SU.main_reads(R(), 100, [[10, 20], [50, 60]], 5, {10, 15, 50}, 3, {15, 50}, lambda img: img * 2)
    assert reads == [10, 15, 20, 50] and out == {15: 30, 50: 100}


def test_ids_and_same_candidates():
    pre = [[0.0, 0.0, 1.0, 2.0, 0.9, 0], [5.0, 0.0, 6.0, 2.0, 0.5, 0]]
    post = [[5.0, 0.0, 6.0, 2.0, 0.5, 0]]
    assert SU.ids_of(pre, post) == ([0, 1], [1])
    with pytest.raises(SystemExit):
        SU.ids_of(pre, [[9.0, 9.0, 9.0, 9.0, 0.1, 0]])
    assert SU.same_candidates({"pre": [tuple(b) for b in pre], "post": post}, {"pre": pre, "post": post})
    assert not SU.same_candidates({"pre": pre, "post": []}, {"pre": pre, "post": post})


def _row(j, sv, cls="P-i"):
    return {"frame": 10, "player": j, "pid": 100 + j, "rec": {"class": cls}, "sv": sv, "Ht": np.eye(3), "px": sv,
            "side_feet": np.zeros((0, 2)), "side_boxes": []}


def test_classify_window_c2_and_c1c():
    sv = np.array([[0.0, 0.0], [3.0, 0.0]] + [[100.0 + 10 * k, 100.0] for k in range(8)])
    post = [[-1.0, -20.0, 3.0, 0.0, 0.9, 0]]                       # foot (1, 0): player 0's box
    cand = [0.5, -20.0, 4.5, 0.0, 0.8, 0]                         # foot (2.5, 0): player 1's, IoU 50/110 with post
    res = {10: {"post": post, "pre": [post[0], cand], "other": [], "low": []}}
    x = _row(1, sv)
    m = SU.classify_window("gsw_cha_2016", "gsw_cha_2016_s00_w00", [x], [x], res, 30.0)
    assert m[0]["class"] == "C2" and m[0]["c2_score"] == 0.8 and abs(m[0]["c2_iou"] - 50 / 110) < 1e-9
    res2 = {10: {"post": post, "pre": [post[0]], "other": [], "low": []}}
    far = _row(2, sv, "D-i")
    assert SU.classify_window("gsw_cha_2016", "w", [far], [far], res2, 30.0)[0]["class"] == "C1c"


def test_unit_tables_and_counts_differ():
    ms = [{"game": "gsw_cha_2016", "window": f"gsw_cha_2016_s00_w{i:02d}", "d205": "P-i", "class": "C1c",
           "low_conf_box": False, "other_classes_within_5ft": {"ball": 0, "rim": 0, "number": 0}} for i in range(4)]
    ms += [{**ms[0], "game": "gsw_bkn_2015", "window": "gsw_bkn_2015_s00_w00", "d205": "D-i", "class": "C2"}]
    u = SU.unit_tables(ms)
    assert u["both__pooled"]["misses"] == 5 and u["P-i__gsw_bkn_2015"]["misses"] == 0
    assert SU.counts_differ(u, u) == []
    import copy
    v = copy.deepcopy(u)
    v["D-i__pooled"]["classes"]["C2"]["count"] = 2
    assert SU.counts_differ(u, v) == ["D-i__pooled C2"]


def test_allow_list():
    for bad in ("hou_cle_2016_s00_w00", "gsw_sac_2015_s00_w00"):
        with pytest.raises(SystemExit):
            SU.allowed([bad])
    assert all(s.startswith(("gsw_cha_2016_", "gsw_bkn_2015_")) for s in SU.CHECK_WINDOWS)


def test_writes_only_its_two_destinations():
    src = inspect.getsource(SU)
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    assert sorted(ast.unparse(c.args[0]) for c in calls) == ["out_path()", "sup_d / f'{s}.json'"]
    assert "raw_old /" in src and "atomic_write(raw_old" not in src             # recallcause/ is read, never written

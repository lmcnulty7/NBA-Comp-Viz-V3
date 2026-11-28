"""tools.p8_recallsupp: the copied read functions equal posefoot's, D208's classification as one function, the gates'
comparisons, the allow-list and the writes."""

import ast
import inspect
import subprocess

import numpy as np
import pytest

from tools import p8_recallsupp as SU


def _posefoot():
    from tools.p8_common import REPO
    p = subprocess.run(["git", "show", f"{SU.POSEFOOT_COMMIT}:tools/p8_posefoot.py"], cwd=REPO, capture_output=True,
                       text=True)
    if p.returncode != 0:
        pytest.skip("posefoot's commit is not in this repository")
    return p.stdout


def _defs(src: str, names) -> dict:
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            for m in ast.walk(n):                      # docstrings and comments aside
                if isinstance(m, (ast.FunctionDef, ast.ClassDef)) and m.body and isinstance(m.body[0], ast.Expr) \
                        and isinstance(getattr(m.body[0], "value", None), ast.Constant) \
                        and isinstance(m.body[0].value.value, str):
                    m.body = m.body[1:] or [ast.Pass()]
            out[n.name] = ast.dump(n)
    return out


def test_copies_equal_posefoots_except_the_predict_call():
    src = _posefoot()
    same = [n for n in SU.COPIED if n != "main_reads"]
    theirs, mine = _defs(src, same), _defs(inspect.getsource(SU), same)
    assert set(mine) == set(theirs) == set(same) and all(mine[k] == theirs[k] for k in same)
    t_mr = _defs(src, ["main_reads"])["main_reads"]
    m_mr = _defs(inspect.getsource(SU).replace("predict(idx, frame)", "predict(frame)"), ["main_reads"])["main_reads"]
    assert t_mr == m_mr                                # the one change is the predict call (9.7)
    ns = {}
    exec(compile(src, "p8_posefoot_53c44192", "exec"), ns)
    for k in ("ALIGN_MIN_ELIGIBLE", "ALIGN_MIN_BOXES", "ALIGN_MARGIN", "ALIGN_LEVEL", "ALIGN_M0", "PAIR_FLOOR",
              "LEVEL_PAIR"):
        assert getattr(SU, k) == ns[k]


class _Reader:
    def __init__(self, total):
        self.next, self.total, self.log = 0, total, []

    def read(self, i):
        self.log.append(i)
        self.next = i + 1
        return (i < self.total), i


def test_read_behaviour_equals_posefoots_on_a_fake_reader():
    ns = {}
    exec(compile(_posefoot(), "p8_posefoot_53c44192", "exec"), ns)
    live = lambda img: (300 <= img <= 900) or (1500 <= img <= 1600)
    out = {}
    for name, mod, pred in (("theirs", ns, lambda img: img), ("mine", vars(SU), lambda idx, img: img)):
        rd = mod["SeekCount"](_Reader(2400))
        iv = mod["pregate_intervals"](rd, 2400, 30, 60, live)
        pre = list(rd.seeks)
        acc = set(range(240, 1000, 6)) | set(range(1440, 1660, 6))
        preds, reads = mod["main_reads"](rd, 2400, iv, 6, acc, 200, {300, 1500, 1506}, pred)
        out[name] = (iv, pre, rd.seeks[len(pre):], reads, sorted(preds), rd.reader.log)
    assert out["theirs"] == out["mine"] and out["mine"][2]          # same intervals, seeks, reads and predictions


def test_main_reads_passes_the_frame_index():
    class R:
        def __init__(self):
            self.next = 0

        def read(self, i):
            self.next = i + 1
            return True, i * 10
    out, reads = SU.main_reads(R(), 100, [[10, 20], [50, 60]], 5, {10, 15, 50}, 3, {15, 50}, lambda i, img: (i, img))
    assert reads == [10, 15, 20, 50] and out == {15: (15, 150), 50: (50, 500)}


def _side(frames, nbox=5, stride=6):
    return {"stride": stride, "frames": [{"frame": f, "boxes": [{"bbox": [100 * k + f % 7, 0, 100 * k + 20 + f % 7, 50]}
                                                               for k in range(nbox)]} for f in frames]}


def test_eligible_and_verdict_142():
    side = _side(range(0, 600, 6))
    fr = list(range(60, 180, 6))                                   # 20 universe frames, all eligible
    assert SU.eligible(side, fr) == fr and SU.eligible(_side(range(0, 600, 6), nbox=3), fr) == []
    srow = {x["frame"]: x for x in side["frames"]}
    good = {f: [b["bbox"] + [0.9, 0] for b in srow[f]["boxes"]] for f in fr}
    v = SU.verdict_142(side, good)
    assert v["verdict"] == "aligned" and v["M"]["0"] == 1.0
    late = {f: [b["bbox"] + [0.9, 0] for b in srow[f + 6]["boxes"]] for f in fr}
    assert SU.verdict_142(side, late)["verdict"] == "misaligned"
    assert SU.verdict_142(side, {f: good[f] for f in fr[:9]})["verdict"] == "undetermined"


def test_read_status():
    assert SU.read_status(True, True, "aligned") == "classified"
    assert SU.read_status(True, True, "undetermined") == "classified"          # too few frames only: kept (9.9)
    assert SU.read_status(True, True, "misaligned").startswith("excluded")
    assert SU.read_status(False, True, "aligned").startswith("undetermined")
    assert SU.read_status(True, False, None).startswith("undetermined")


def test_reread_sets():
    kept, dropped = ["a_k1", "a_k2", "a_k3"], ["a_d1", "a_d2"]
    seek = {"a_k1": True, "a_k2": False, "a_k3": False, "a_d1": True, "a_d2": False}
    g0 = {s: {"verdict": "aligned"} for s in kept + dropped}
    g0["a_k3"]["verdict"] = "undetermined"
    s = SU.reread_sets(kept, dropped, seek, g0)
    assert s["reread"] == ["a_d1", "a_d2", "a_k1", "a_k3"] and s["certified"] == ["a_k2"]
    assert s["kept_reread"] == ["a_k1", "a_k3"]
    assert SU.seeks_past_zero({"seeks_main": [0]}) is False and SU.seeks_past_zero({"seeks_main": [150]}) is True


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
    assert sorted(ast.unparse(c.args[0]) for c in calls) == ["path", "sup_d / f'{s}.json'"]
    writes = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_write"]
    assert sorted(ast.unparse(c.args[0]) for c in writes) == ["out_path('g0')", "out_path()"]
    assert "raw_old /" in src and "atomic_write(raw_old" not in src             # recallcause/ is read, never written

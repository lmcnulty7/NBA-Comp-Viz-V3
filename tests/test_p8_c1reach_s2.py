"""tools.p8_c1reach_s2: the zones of 10.6 under both gaps, the increment, its reach form, and the venv boundary."""

import ast
import inspect

import numpy as np

from tools import p8_c1reach_s2 as S2


def test_bars():
    assert S2.bars(0.0021) == {"kill": -0.0021, "bar": -0.0042}
    b = S2.bars(0.0069)
    assert abs(b["kill"] + 0.0069) < 1e-12 and abs(b["bar"] + 0.0090) < 1e-12


def test_outcome_zones():
    o = lambda p, iv, fiv, pooled=-0.001, ghost=True, gap=0.0021: S2.outcome(p, iv, fiv, pooled, ghost, gap)
    assert o(-0.001, [-0.003, 0.001], None).startswith("kill")
    assert o(-0.003, [-0.005, -0.001], None).startswith("short of the bar")
    assert o(-0.005, [-0.008, -0.001], [-0.006, -0.0015]).startswith("not resolvable")
    assert o(-0.005, [-0.008, -0.001], [-0.006, -0.004]).startswith("not shown")
    assert o(-0.005, [-0.008, -0.003], None, pooled=0.001).startswith("not shown (pooled)")
    assert o(-0.005, [-0.008, -0.003], None, ghost=False).startswith("ghost failure")
    assert o(-0.005, [-0.008, -0.003], None).startswith("a training-arm design is justified")
    assert o(-0.005, [-0.008, -0.003], None, gap=0.0069).startswith("kill")      # under the larger gap


def _box(x, y, s=0.9):
    return [x - 1.0, y - 6.0, x + 1.0, y, s, 0]


def test_increment_marks_union():
    sv = np.array([[10.0, 10.0], [12.0, 10.0], [20.0, 20.0]] + [[60.0 + 3 * k, 40.0] for k in range(7)])
    fr = {"sv": sv.tolist(), "Ht": np.eye(3).tolist(), "inside": [True] * 10,
          "r23_post": [_box(10.1, 10.0)], "r23_low": [],
          "c1": [{"player": 1, "pid": 7, "unit": "D-i", "subclass": "C1a_other"},
                 {"player": 2, "pid": 8, "unit": "P-i", "subclass": "C1c"}]}
    teacher = [_box(10.1, 10.0), _box(12.1, 10.0)]                      # the teacher reaches player 1
    crowd = [[*_box(10.1, 10.0)[:4], 0.9, "person"], [*_box(12.2, 10.0)[:4], 0.8, "person"],
             [*_box(20.1, 20.0)[:4], 0.6, "person"], [*_box(20.3, 20.1)[:4], 0.4, "person"]]
    st = S2.increment_marks(fr, teacher, crowd, 0.5)
    m = {x["player"]: x for x in st["misses"]}
    assert m[1]["union"] and not m[1]["increment"]                      # the held teacher reaches it already
    assert m[2]["union"] and m[2]["increment"]                          # only the merged CrowdDet box reaches it
    assert (st["hits"], st["hits_reached_crowd"], st["hits_reached_union"]) == (1, 1, 1)
    assert st["cost"]["crowd"]["boxes"] == 3 and st["cost"]["union"]["boxes"] == 3   # two CrowdDet boxes merged away
    masked = S2.increment_marks(fr, teacher, crowd, 0.5, referees=[[19.1, 14.0, 21.1, 20.0, 0.9, 1]])
    assert not {x["player"]: x for x in masked["misses"]}[2]["increment"]  # the referee mask drops it
    r = S2.as_reach(st["misses"], S2.HEADLINE)
    assert [x["teacher"] for x in r] == [False, True] and not any(x["floor"] for x in r)
    assert S2.as_reach(st["misses"], S2.APART) == []


def test_union_and_mask():
    t = [[0, 0, 10, 20, 0.9, "player"]]
    c = [[0.5, 0, 10.5, 20, 0.8, "person"], [30, 0, 40, 20, 0.7, "person"]]
    assert S2.union_boxes(t, c) == t + [c[1]]
    assert S2.referee_mask(c, [[30, 0, 40, 19, 0.5, 1]]) == [c[0]]


def test_reading_follows_the_registered_gap():
    inc = {"point": -0.005, "interval": [-0.008, -0.003], "f_only_interval": [-0.006, -0.004]}
    r = S2.reading(inc, -0.001, True, 0.0021)
    assert r["action"].startswith("passes, not robust to the transfer residual")
    assert r["under gap 0.0021"].startswith("a training-arm design") and r["sensitivity, gap 0.0069"].startswith("kill")
    strong = {"point": -0.011, "interval": [-0.014, -0.0095], "f_only_interval": [-0.012, -0.0093]}
    assert S2.reading(strong, -0.002, True, 0.0021)["action"].startswith("a training-arm design is justified")


def test_load_is_weights_only_and_strict():
    src = inspect.getsource(S2.load_detector)
    calls = [n for n in ast.walk(ast.parse(src.strip().replace("\n    ", "\n") if False else src))
             if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.load"]
    assert calls and all(any(k.arg == "weights_only" and ast.unparse(k.value) == "True" for k in c.keywords)
                         for c in calls)
    assert "strict=True" in src and "except Exception" in src


def test_venv_steps_import_nothing_from_the_repository():
    for fn in (S2.check_source, S2.load_detector, S2._venv_common, S2.cmd_run):
        src = inspect.getsource(fn)
        assert "from tools" not in src and "import tools" not in src, fn.__name__
    assert S2.WEIGHTS_SHA is None or len(S2.WEIGHTS_SHA) == 64


def test_writes():
    src = inspect.getsource(S2)
    tree = ast.parse(src)
    w = [ast.unparse(n.args[0]) for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "write_atomic"]
    a = [ast.unparse(n.args[0]) for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    assert w == ["out_d / ('crowddet_probe.json' if probe else 'crowddet.json')"]
    assert a == ["REPO / 'reports' / 'research' / name"]
    assert "gsw_sac" not in src and "hou_cle" not in src

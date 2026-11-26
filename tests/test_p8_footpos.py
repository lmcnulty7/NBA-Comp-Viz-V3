"""tools.p8_footpos: the pure rules of the K1 position-row measurement on synthetic data."""

from tools import p8_footpos as M


def test_same_matched():
    a = {"1:0": [1.0, 0.5, "near"], "1:1": [2.0, -0.1, "far"]}
    assert M.same_matched(a, {"1:0": [1.00005, 0.5, "near"], "1:1": [2.0, -0.1, "far"]})
    assert not M.same_matched(a, {"1:0": [1.0, 0.5, "near"]})
    assert not M.same_matched(a, {"1:0": [1.0, 0.5, "middle"], "1:1": [2.0, -0.1, "far"]})
    assert not M.same_matched(a, {"1:0": [1.01, 0.5, "near"], "1:1": [2.0, -0.1, "far"]})


def test_read_rule():
    def g(p50, piv, nb, niv, miv, giv):
        return {"diff": {"position_p50_ft": p50, "nearfield_bias_ft": nb},
                "iv": {"position_p50_ft": piv, "nearfield_bias_ft": niv, "missed_rate": miv, "ghost_rate": giv}}
    ok = g(-0.1, [-0.2, -0.05], -0.2, [-0.3, -0.1], [-0.01, 0.005], [-0.01, 0.009])
    assert M.read_rule({"a": ok, "b": ok}) == {"position_gain": True, "non_inferior": True, "pass": True}
    harm = g(-0.1, [-0.2, -0.05], -0.2, [-0.3, -0.1], [-0.01, 0.0165], [-0.01, 0.009])
    assert M.read_rule({"a": ok, "b": harm}) == {"position_gain": True, "non_inferior": False, "pass": False}
    cross = g(-0.1, [-0.2, 0.01], -0.2, [-0.3, -0.1], [-0.01, 0.0], [-0.01, 0.0])
    assert M.read_rule({"a": cross})["position_gain"] is False

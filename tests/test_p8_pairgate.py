"""tools.p8_pairgate: the capped pairing, the swap and its restoring, and the reading."""

import ast
import inspect

import numpy as np
import pytest

from tools import p8_pairgate as PG


def test_capped_equals_frozen_when_nothing_is_beyond_the_gate():
    from sportvu.truth import _match
    rng = np.random.default_rng(0)
    for _ in range(50):
        sv = rng.uniform(0, 40, (10, 2))
        court = sv[:7] + rng.normal(0, 0.5, (7, 2))
        a = sorted((i, j) for i, j, _ in _match(court, sv, 1e9))      # a gate no pair exceeds
        b = sorted((i, j) for i, j, _ in PG.capped_match(court, sv, 1e9))
        assert a == b


def test_beyond_gate_distances_steer_only_the_frozen_rule():
    # boxes A, B; players p0, p1, p2. Frozen (Hungarian on D): A-p1 (3, kept) and B-p0 (6, dropped), total 9.
    # Capped (min(D, 5)): A-p0 (1, kept) and B beyond, total 6. The within-gate pair kept differs.
    D = np.array([[1.0, 3.0, 40.0], [6.0, 30.0, 20.0]])
    frozen = {(i, j) for i, j, _ in PG._pairs(D, D, 5.0)}
    capped = {(i, j) for i, j, _ in PG._pairs(D, np.minimum(D, 5.0), 5.0)}
    assert frozen == {(0, 1)} and capped == {(0, 0)}
    assert {(i, j) for i, j, _ in PG._pairs(D, np.where(D <= 5.0, D, 1e6), 5.0)} == {(0, 0)}   # the 1e6 rule


def test_frozen_match_is_the_full_distance_rule():
    from sportvu.truth import _match
    rng = np.random.default_rng(1)
    for _ in range(30):
        sv = rng.uniform(0, 40, (10, 2)); court = rng.uniform(0, 40, (6, 2))
        D = np.sqrt(((court[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
        assert sorted((i, j) for i, j, _ in _match(court, sv, 5.0)) == sorted((i, j) for i, j, _ in PG._pairs(D, D, 5.0))


def test_swap_restores_and_counts():
    import sportvu.truth as TR
    orig = TR._match
    c = PG.Counter()
    court = np.array([[0.1, 0.0]]); sv = np.array([[0.0, 0.0]] + [[60.0 + 5 * k, 40.0] for k in range(9)])
    with PG.swapped("frozen", c):
        TR._match(court, sv, 5.0)
    assert TR._match is orig and (c.frozen_calls, c.capped_calls) == (1, 0)
    with pytest.raises(RuntimeError):
        with PG.swapped("capped", c):
            TR._match(court, sv, 5.0)
            raise RuntimeError("boom")
    assert TR._match is orig and c.capped_calls == 1 and c.pairs["frozen"] == c.pairs["capped"] == 1


def test_reading():
    ch = {"team": {"change": 0.004, "interval": [0.001, 0.007]}, "position_p90_ft": {"change": -0.2, "interval": None},
          "missed": {"change": 0.0, "interval": [0.0, 0.0]}}
    d = {"team": {"interval": [0.818, 0.849]}, "position_p90_ft": {"interval": [3.26, 3.49]}, "missed": {"interval": [0.2, 0.22]}}
    r = PG.reading(ch, d)
    assert r["team"]["label"] == "insensitive" and r["position_p90_ft"]["label"] == "scorer-sensitive"
    assert r["missed"]["label"] == "insensitive" and abs(r["team"]["d202_half_width"] - 0.0155) < 1e-9


def test_scope_and_writes():
    for bad in ("hou_cle_2016", "gsw_sac_2015_s00_w00"):
        with pytest.raises(SystemExit):
            PG.allowed([bad])
    src = inspect.getsource(PG)
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "atomic_write"]
    assert [ast.unparse(c.args[0]) for c in calls] == ["REPO / 'reports' / 'research' / 'p8_pairgate.json'"]
    assert "hou_cle" not in src and "score_run" not in src

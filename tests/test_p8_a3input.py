"""tools.p8_a3input: the pure reading rules of the doubled-callback check on synthetic records."""

from tools import p8_a3input as M


def fr(n_calls, reset_before, n_out=2, n_in=10):
    return {"calls": [{"n_input": n_in if i == 0 else n_out, "frame_id": i} for i in range(n_calls)],
            "reset_before": reset_before, "output": [[0, 0, 1, 1]] * n_out}


def test_doubling_and_yield():
    A = [fr(1, False, 9)] * 3 + [fr(2, True, 3)] * 20
    B = [fr(1, False, 9)] * 3 + [fr(1, True, 9)] * 20
    dA, dB = M.doubling(A), M.doubling(B)
    assert dA["share"] == 1.0 and dB["frames_with_two_or_more_calls"] == 0
    inputs = [10] * 23
    assert abs(M.yield_of(A, inputs) - (27 + 60) / 230) < 1e-12
    assert abs(M.yield_of(B, inputs) - 207 / 230) < 1e-12


def test_read():
    dA = {"share": 1.0}
    dB = {"frames_with_two_or_more_calls": 0}
    assert M.read(1.0, dA, dB, 0.3, 0.95) == "cause confirmed"
    assert M.read(1.0, dA, dB, 0.3, 0.8) == "doubling is part of the cause"
    assert M.read(1.0, {"share": 0.5}, dB, 0.3, 0.95) == "doubling not confirmed"
    assert M.read(1.0, dA, {"frames_with_two_or_more_calls": 3}, 0.3, 0.95) == "doubling not confirmed"
    assert M.read(0.9, dA, dB, 0.3, 0.95) == "not reproduced"


def test_same_boxes():
    assert M.same_boxes([[0, 0, 10, 10]], [[1, 1, 11, 10]])
    assert not M.same_boxes([[0, 0, 10, 10]], [])

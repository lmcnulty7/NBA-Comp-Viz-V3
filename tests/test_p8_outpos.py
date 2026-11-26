"""tools.p8_outpos: the pure rules of the output-position arm on synthetic data."""

import numpy as np

from tools import p8_outpos as M


def test_shift_point_moves_along_the_camera_ray():
    ground = [0.0, 100.0]
    q = np.array([0.0, 50.0])                      # 50 ft in front of the camera
    out = M.shift_point(q, ground, ("150-200", "far"), {("150-200", "far"): -1.0})
    assert np.allclose(out, [0.0, 49.0])            # a short foot (-1 ft) is pushed 1 ft beyond, away from the camera
    assert np.allclose(M.shift_point(q, None, None, {}), q)
    assert np.allclose(M.shift_point(q, ground, ("x", "y"), {}), q)


def test_table_round_trip():
    t = {("150-200", "far"): -1.2, ("<150", "near"): -0.3}
    assert M.table_from_json(M.table_to_json(t)) == t


def test_decide():
    diff = {"position_p50_ft": -0.2, "nearfield_bias_ft": -0.9, "position_p90_ft": 0.02}
    iv = {"position_p50_ft": [-0.3, -0.1], "nearfield_bias_ft": [-1.0, -0.8], "position_p90_ft": [-0.05, 0.09]}
    assert M.decide(diff, iv)["verdict"] == "PASS"
    assert M.decide(diff, dict(iv, position_p90_ft=[-0.05, 0.11]))["verdict"] == "FAIL"
    assert M.decide(diff, dict(iv, position_p50_ft=[-0.3, 0.01]))["verdict"] == "FAIL"
    assert M.decide(dict(diff, nearfield_bias_ft=0.1), iv)["verdict"] == "FAIL"


def test_score_hou_refuses_without_a_committed_table(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(M, "TABLE", tmp_path / "missing.json")
    monkeypatch.setattr(M, "REPO", tmp_path)
    with pytest.raises(SystemExit):
        M.table_provenance()


# ── version 2 ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_k0_and_form_decision():
    t = M.k0_table([{"depth": -1.0}, {"depth": -0.5}, {"depth": -0.9}])
    assert set(t) == set(M.CELLS) and all(v == -0.9 for v in t.values())
    k1 = {g: -0.30 for g in M.POOL}
    k0 = {g: -0.25 for g in M.POOL}
    assert M.form_decision({"K1": k1, "K0": k0}) == "K1"
    assert M.form_decision({"K1": dict(k1, **{M.POOL[2]: -0.20}), "K0": k0}) == "K0"
    assert M.form_decision({"K1": k0, "K0": k0}) == "K0"


def test_removed_bias_and_bias_forecast():
    counts = {"150-200|near": 100, "150-200|far": 300, "<150|far": 0}
    med = {"150-200|near": -0.4, "150-200|far": -1.2, "<150|far": None}
    table = {("150-200", "near"): -0.4, ("150-200", "far"): -1.0}
    r = M.removed_bias(counts, med, table)
    assert abs(r - (100 * 0.4 + 300 * 1.0) / 400) < 1e-12
    bf = M.bias_forecast(counts, med, table)
    assert bf["third_before"] == "far" and abs(bf["before"] - 1.2) < 1e-12 and abs(bf["after"] - 0.2) < 1e-12


def test_decide_v2():
    diff = {"position_p50_ft": -0.15}
    iv = {"position_p50_ft": [-0.2, -0.1], "position_p90_ft": [-0.1, 0.05]}
    bias = {"before": 1.1, "after": 0.3}
    assert M.decide_v2(diff, iv, bias)["verdict"] == "PASS"
    assert M.decide_v2({"position_p50_ft": -0.09}, iv, bias)["verdict"] == "FAIL"
    assert M.decide_v2(diff, dict(iv, position_p90_ft=[0.0, 0.11]), bias)["verdict"] == "FAIL"
    assert M.decide_v2(diff, iv, {"before": 0.3, "after": 0.4})["verdict"] == "FAIL"


def test_refusals_on_existing_record_and_marker(tmp_path):
    import pytest
    rec, mark = tmp_path / "rec.json", tmp_path / "started"
    M.check_once(rec, mark)
    rec.write_text("{}")
    with pytest.raises(SystemExit):
        M.check_once(rec, mark)
    rec.unlink()
    mark.write_text("x")
    with pytest.raises(SystemExit):
        M.check_once(rec, mark)


def test_refusal_on_mismatched_or_unpinned_table(tmp_path):
    import pytest
    f = tmp_path / "t.json"
    f.write_text('{"a": 1}')
    good = M.sha256_of(f)
    assert M.check_frozen(f, good) == good
    with pytest.raises(SystemExit):
        M.check_frozen(f, "0" * 64)
    with pytest.raises(SystemExit):
        M.check_frozen(f, None)


def test_refusal_on_dirty_table_and_dirty_code(tmp_path):
    import subprocess
    import pytest
    run = lambda *a: subprocess.run(list(a), cwd=tmp_path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "tester")
    run("git", "config", "user.name", "t")
    (tmp_path / "tools").mkdir()
    f, code = tmp_path / "t.json", tmp_path / "tools" / "x.py"
    f.write_text("1"); code.write_text("a = 1\n")
    run("git", "add", "-A"); run("git", "commit", "-qm", "init")
    assert M.tracked_clean(f, repo=tmp_path)
    assert M.check_clean(("tools/x.py",), repo=tmp_path)
    f.write_text("2"); code.write_text("a = 2\n")
    with pytest.raises(SystemExit):
        M.tracked_clean(f, repo=tmp_path)
    with pytest.raises(SystemExit):
        M.check_clean(("tools/x.py",), repo=tmp_path)

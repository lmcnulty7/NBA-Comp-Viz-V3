"""tools.p8_multistart stage 0 (docs/research/P8_multistart_fit.md version 2, section 3): the pure parts, on synthetic
frames. No video, no model, no workspace."""

import json

import numpy as np
import pytest

from tools import p8_multistart as M


def synthetic_frames(n=30, pairs=8, seed=1, C=(47.0, 125.0, -30.0), noise=0.0):
    from sportvu.camera import h_px2ft, project
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        rv = np.array([-1.2, 0.0, 0.0]) + rng.normal(0, 0.03, 3)     # a pose pose_from_h recovers (C below z=0)
        f = float(rng.uniform(1500, 2500))
        ft = np.column_stack([rng.uniform(5, 89, pairs), rng.uniform(5, 45, pairs)])
        px = project(C, rv, f, 1280, 720, ft) + rng.normal(0, noise, (pairs, 2))
        out.append({"window": f"gsw_cha_2016_s01_w{k // 10:02d}", "frame": k, "w": 1280, "h": 720,
                    "H_v3": h_px2ft(C, rv, f, 1280, 720).ravel().tolist(), "H_b4": None, "px": px, "ft": ft,
                    "boxes": [], "row": {}})
    return out


def test_soft_l1_cost_is_scipys():
    from scipy.optimize import least_squares
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 40)
    y = 3 * t + 1 + rng.normal(0, 2, 40)
    sol = least_squares(lambda p: p[0] * t + p[1] - y, [0.0, 0.0], loss="soft_l1", f_scale=6.0)
    assert M.soft_l1_cost(sol.fun, 6.0) == pytest.approx(sol.cost, rel=1e-9)
    assert M.soft_l1_cost([0.0, 0.0]) == 0.0


def test_reading_rule():
    assert M.reading(100.0, [100.0, 120.0]) == "STOP"            # broken at or below every good position
    assert M.reading(100.0, [101.0, 99.5]) == "INCONCLUSIVE"     # best good lower by under 1 percent
    assert M.reading(100.0, [99.0, 130.0]) == "CONTINUE"         # lower by exactly 1 percent
    assert M.reading(100.0, [50.0]) == "CONTINUE"


def test_optimality_share():
    ok = M.optimality([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ok["passes"] and ok["worse"] == 0
    bad = M.optimality([1.02] * 6 + [1.0] * 94, [1.0] * 100)      # 6 percent more than 1 percent worse
    assert bad["worse"] == 6 and not bad["passes"]
    edge = M.optimality([1.02] * 5 + [1.0] * 95, [1.0] * 100)     # exactly 5 percent passes
    assert edge["passes"]
    assert not M.optimality([np.inf], [1.0])["passes"]


def test_first_chunk_is_fit_game_and_chunks_continue():
    from sportvu.camera import fit_game
    from tools.p8_cha_fit import inits_of
    fr = synthetic_frames(noise=0.5)
    inits = inits_of(fr)
    pool = M.pool_of(inits, 0)
    C0 = np.median(np.array([p[2] for _, p in inits]), axis=0)
    first, x = M.run_chunks(pool, M.start_vector(pool, C0), 1)
    assert np.allclose(x[:3], fit_game(fr, seed=0)["C"], atol=1e-9)
    two, x2 = M.run_chunks(pool, M.start_vector(pool, C0), 2, chunk_nfev=5)
    assert len(two) == 2 and two[1]["cost"] <= two[0]["cost"] + 1e-9


def test_frame_cost_and_common_objective_at_the_true_position():
    C = (47.0, 125.0, -30.0)
    fr = synthetic_frames(n=5)
    c, rv, f = M.frame_cost(C, fr[0])
    assert c == pytest.approx(0.0, abs=1e-6)
    j_true = M.common_objective(C, fr)
    j_off = M.common_objective((47.0, 105.0, -34.0), fr)
    assert j_true["J"] < j_off["J"] and j_true["fit_frame_failed"] == 0 and j_true["frames"] == 5


def test_recorded_positions_match_the_design(tmp_path):
    rec = M.recorded_positions()
    assert set(rec["positions"]) == {"B", "G1", "G2", "G3", "G4"}
    d = json.loads(M.STAGE2_RECORD.read_text())
    d["F1"]["fix"][0]["C"] = [0.0, 0.0, 0.0]                        # seed 1 altered
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not the design's"):
        M.recorded_positions(p)


def test_manifest_sees_a_changed_protected_file(tmp_path):
    (tmp_path / "labels" / "g").mkdir(parents=True)
    (tmp_path / "labels" / "g" / "a.jsonl").write_text("1")
    (tmp_path / "builds" / "r23").mkdir(parents=True)
    (tmp_path / "builds" / "r23" / "w_frames.json").write_text("{}")
    a = M.manifest(tmp_path)
    assert set(a) == {"labels/g/a.jsonl", "builds/r23/w_frames.json"}
    (tmp_path / "labels" / "g" / "a.jsonl").write_text("2")
    assert M.manifest(tmp_path) != a


def test_constants_are_the_registered_ones():
    assert (M.CHUNK, M.STOP_MARGIN, M.OPT_SLACK, M.OPT_SHARE, M.SMOKE_CHUNKS, M.EXPECT_INITS, M.REPRO_FT) == \
        (200, 0.01, 0.01, 0.05, 10, 4943, 0.01)
    assert M.GAME == "gsw_cha_2016"

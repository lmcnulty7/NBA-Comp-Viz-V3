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


def test_j_validity_stops_on_non_finite_or_failed_frames():
    good = {"B": {"J": 10.0, "fit_frame_failed": 0}, "G1": {"J": 5.0, "fit_frame_failed": 0}}
    assert M.j_valid(good) == (True, [])
    ok, why = M.j_valid({**good, "G2": {"J": float("nan"), "fit_frame_failed": 0}})
    assert not ok and why == ["G2: J not finite"]
    ok, why = M.j_valid({**good, "G3": {"J": float("inf"), "fit_frame_failed": 0}})
    assert not ok and "G3: J not finite" in why
    ok, why = M.j_valid({**good, "G4": {"J": 7.0, "fit_frame_failed": 2}})
    assert not ok and why == ["G4: fit_frame failed on 2 inits"]


def test_g4_is_fixed_seed_10_in_the_record():
    import json
    d = json.loads(M.STAGE2_RECORD.read_text())
    seed10 = [r["C"] for r in d["F1"]["fix"] if r["seed"] == 10][0]
    assert np.allclose(M.recorded_positions()["positions"]["G4"], seed10)


# ── stage 1 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def chunk(C, cost, status=0):
    return {"C": list(C), "cost": cost, "status": status}


def test_converged_at():
    assert M.converged_at([chunk((0, 0, 0), 10), chunk((0, 1, 0), 9), chunk((0, 1, 0), 9, status=2)]) == 3
    assert M.converged_at([chunk((0, 0, 0), 10), chunk((0, 0.04, 0), 9.995)]) == 2           # 0.04 ft and 0.05 percent
    assert M.converged_at([chunk((0, 0, 0), 10), chunk((0, 0.06, 0), 9.999)]) is None        # moved 0.06 ft
    assert M.converged_at([chunk((0, 0, 0), 10), chunk((0, 0.01, 0), 9.98)]) is None         # cost fell 0.2 percent
    assert M.converged_at([chunk((0, 0, 0), 10, status=1)]) == 1


def test_select_lowest_j_ties_to_the_lower_seed_and_every_run_competes():
    runs = [{"seed": 3, "J": 5.0, "converged_at_chunk": None}, {"seed": 1, "J": 7.0, "converged_at_chunk": 2},
            {"seed": 0, "J": 5.0 * (1 + 1e-12), "converged_at_chunk": 1}]
    assert M.select(runs)["seed"] == 0                                    # equal within 1e-9: the lower seed
    runs[2]["J"] = 6.0
    assert M.select(runs)["seed"] == 3                                    # not converged, lowest J, still selected


def test_stability_in_truth_space():
    a = {("w", 1): np.array([[0.0, 0.0], [10.0, 0.0]]), ("w", 2): np.array([[5.0, 5.0]])}
    b = {k: v + [0.1, 0.0] for k, v in a.items()}
    c = {k: v + [0.0, 0.3] for k, v in a.items()}
    assert np.allclose(M.foot_distances(a, b), 0.1)
    s = M.stability({"truth": a, "stab_a": b})
    assert s["passes"] and s["pairs"]["stab_a|truth"]["feet"] == 3
    s = M.stability({"truth": a, "stab_a": b, "stab_b": c})
    assert not s["passes"]                                                # c is 0.3 ft from a
    assert not M.stability({"truth": a})["passes"]                        # no pair, no pass


def test_kill4():
    assert M.kill4((47.0, 110.0, -34.0), (47.0, 104.3, -34.5))["stage2_runs"]
    assert not M.kill4((47.0, 108.0, -34.5), (47.0, 104.3, -34.5))["stage2_runs"]


def test_replaced_fit_game_asserts_and_restores():
    from sportvu import camera as cam
    orig = cam.fit_game
    frames = [{"window": "w", "frame": 1}, {"window": "w", "frame": 2}]
    with M.replaced_fit_game({"status": "ok", "C": [1, 2, 3]}, M.keys_hash(frames)) as rp:
        assert cam.fit_game(frames)["C"] == [1, 2, 3]
    assert cam.fit_game is orig and rp.calls == 1
    with pytest.raises(SystemExit, match="not the frames"):
        with M.replaced_fit_game({"C": [0, 0, 0]}, M.keys_hash(frames)):
            cam.fit_game(frames[:1])
    assert cam.fit_game is orig
    with pytest.raises(SystemExit, match="2 times"):
        with M.replaced_fit_game({"C": [0, 0, 0]}, M.keys_hash(frames)):
            cam.fit_game(frames); cam.fit_game(frames)
    assert cam.fit_game is orig
    with pytest.raises(SystemExit, match="0 times"):
        with M.replaced_fit_game({"C": [0, 0, 0]}, M.keys_hash(frames)):
            pass
    assert cam.fit_game is orig


def test_three_distinct_roots_and_the_work_ms_links(tmp_path):
    work, ref, ms = tmp_path / "work", tmp_path / "work_ref", tmp_path / "work_ms"
    for p in (work, ref):
        p.mkdir()
    M.check_roots(work, ref, ms)
    with pytest.raises(SystemExit, match="three different"):
        M.check_roots(work, ref, ref)
    (tmp_path / "alias").symlink_to(ref)
    with pytest.raises(SystemExit, match="three different"):
        M.check_roots(work, ref, tmp_path / "alias")
    for d in ("label_build", "video", "sync", "sportvu", "meta/done", "reports"):
        (ref / d).mkdir(parents=True)
    (ref / "meta" / "done" / "label__gsw_cha_2016__gsw_cha_2016_s00.json").write_text("{}")
    (ref / "meta" / "done" / "label__gsw_bkn_2015__gsw_bkn_2015_s00.json").write_text("{}")
    P = M.setup_ms(ref, ms, "gsw_cha_2016")
    assert (ms / "label_build").is_symlink() and P["labels"].is_dir()
    assert (ms / "meta" / "done" / "label__gsw_cha_2016__gsw_cha_2016_s00.json").exists()
    assert not (ms / "meta" / "done" / "label__gsw_bkn_2015__gsw_bkn_2015_s00.json").exists()
    M.setup_ms(ref, ms, "gsw_cha_2016")                                  # idempotent
    (ms / "video").unlink(); (ms / "video").symlink_to(tmp_path)
    with pytest.raises(SystemExit, match="not the link"):
        M.setup_ms(ref, ms, "gsw_cha_2016")


def test_stage1_constants_are_the_registered_ones():
    assert M.SEED_SETS == {"truth": tuple(range(8)), "stab_a": tuple(range(16, 24)), "stab_b": tuple(range(24, 32))}
    assert (M.CHUNKS, M.CONV_FT, M.CONV_COST, M.STAB_FT, M.KILL4_FT) == (20, 0.05, 0.001, 0.25, 5.0)
    assert M.STAGE1_GAMES == ("gsw_cha_2016", "gsw_bkn_2015")

"""Scorecard v2 of P8 Phase A (docs/research/P8_FEASIBILITY.md section 7 (4)): the pin guard and the pure parts."""

import json

import numpy as np
import pytest

from tools import p8_pin
from tools import p8_scorecard as S


def test_the_scorer_its_constants_and_the_frozen_scorer_match_the_committed_pin():
    """Fails on a branch that changes tools/p8_scorecard.py, its CONSTANTS, a frozen file it scores with or a frozen
    constant it reads. Changing them is allowed only with a new pin (python3 -m tools.p8_pin --write) in the same
    reviewed commit, so the change is visible to the lead and the judge."""
    pinned = json.loads(p8_pin.PIN_FILE.read_text())
    now = p8_pin.current()
    assert now["scorer"] == pinned["scorer"], "tools/p8_scorecard.py changed since the pin"
    assert now["constants_sha256"] == pinned["constants_sha256"], "scorecard v2 CONSTANTS changed since the pin"
    assert now["frozen_files"] == pinned["frozen_files"], "a frozen file of the scorer changed"
    assert now["frozen_constants"] == pinned["frozen_constants"], "a frozen scorer constant changed"
    assert now["fixed_constants_file"] == pinned["fixed_constants_file"], "reports/research/p8_constants.json changed or appeared"
    for f in ("tools/p8_frozen.py", "tools/p8_common.py", "tools/p8_scorecard.py"):
        assert f in pinned["scorer"], f
    assert "clock_reader.py" in pinned["frozen_files"]


def test_the_pin_file_changes_only_with_plan_md_once_the_lead_has_adopted_it():
    """Every non-merge commit that changes the pin while the pin is adopted (before or after) also changes PLAN.md."""
    import subprocess
    rel = "reports/research/p8_scorecard_pin.json"
    run = lambda *a: subprocess.run(["git", *a], cwd=p8_pin.REPO, capture_output=True, text=True)
    shas = run("log", "--no-merges", "--format=%H", "--", rel).stdout.split()
    problems = []
    for sha in shas:
        new = run("show", f"{sha}:{rel}")
        old = run("show", f"{sha}^:{rel}")
        adopted = any(r.returncode == 0 and json.loads(r.stdout).get("status") == "adopted" for r in (new, old))
        files = run("show", "--name-only", "--format=", sha).stdout.split()
        if adopted and "PLAN.md" not in files:
            problems.append(sha[:8])
    assert problems == []
    pinned = json.loads(p8_pin.PIN_FILE.read_text())
    assert pinned["status"] in ("draft", "adopted")


def test_the_section_6_2_constants_are_the_documented_ones():
    C = S.CONSTANTS
    assert (C["holder_max_ft"], C["holder_ball_max_z_ft"], C["event_tol_s"]) == (3.0, 6.0, 1.0)
    assert (C["ball_f1_target"], C["carrier_target"], C["identity_target"]) == (0.75, 0.85, 0.80)
    assert (C["shots_found_target"], C["shots_false_max"], C["rebounds_found_target"]) == (0.95, 0.05, 0.90)
    assert (C["shooter_p50_ft"], C["shooter_p90_ft"], C["rebounder_p50_ft"]) == (3.0, 6.0, 4.0)
    assert C["jersey_min_crops"] == 8 and C["switches_goal_per_possession"] == 5.0


def test_bins_and_zones():
    e = [3.0, 6.0, 10.0]
    assert [S.bin_of(v, e) for v in (0.0, 3.0, 5.9, 10.0, 40.0)] == ["<3", "3-6", "3-6", ">=10", ">=10"]
    assert S.bin_of(None, e) is None and S.bin_of(float("nan"), e) is None
    assert S.zone_of(5.25, 25.0) == "<4" and S.zone_of(88.75, 35.0) == "4-16"
    assert S.zone_of(5.25 + 20, 25.0) == "16-23.75" and S.zone_of(47.0, 25.0) == ">=23.75"


def test_crowding_and_holder():
    sv = np.array([[10, 10], [12, 10], [40, 40], [80, 25]], np.float32)
    assert np.allclose(S.crowding(sv)[:2], [2, 2])
    assert S.holder(np.array([40.5, 40.0, 4.0]), sv) == 2
    assert S.holder(np.array([40.5, 40.0, 7.0]), sv) is None          # ball too high
    assert S.holder(np.array([11.0, 10.0, 4.0]), sv) is None          # two players within 3 ft
    assert S.holder(np.array([60.0, 10.0, 4.0]), sv) is None          # nobody near
    assert S.holder(np.array([np.nan, 1, 1]), sv) is None


def test_cuts_and_time_since_cut():
    frames = [0, 3, 6, 9, 12]
    corr = [None, 0.95, 0.2, 0.9, 0.59]
    cuts = S.cuts_from_corr(frames, corr)
    assert cuts == [6, 12]
    assert S.since_cut(9, cuts, 0, 30.0) == (0.1, False)
    assert S.since_cut(3, cuts, 0, 30.0) == (0.1, True)               # before the first cut: censored


def test_f1_and_event_matching_are_one_to_one():
    assert S.f1(0, 0, 5) == 0.0 and S.f1(8, 2, 2) == 0.8
    truth = [(1, 700.0), (1, 650.0), (2, 600.0)]
    pred = [(1, 699.5), (1, 699.0), (2, 598.5), (2, 300.0)]
    found, false, pairs = S.match_events(truth, pred, 1.0)
    assert found == 1 and false == 3 and pairs[0][:2] == (0, 0)
    found, false, _ = S.match_events(truth, [(1, 650.4), (2, 600.9), (1, 701.0)], 1.0)
    assert (found, false) == (3, 0)


def test_cluster_bootstrap_is_seeded_and_needs_two_groups():
    groups = {"a": [1.0, 2.0], "b": [3.0], "c": [10.0, 11.0]}
    stat = lambda p: float(np.median(p)) if p else None
    one = S.boot_windows(groups, stat, n=200)
    assert one == S.boot_windows(groups, stat, n=200) and one[0] <= one[1]
    assert S.boot_windows({"a": [1.0]}, stat) is None


# A camera-truth H of a real R1.3 label (gsw_bkn_2015, s00_w00, frame 120; px -> ft), 1280 x 720.
H_LABEL = [0.1715445705296883, 0.2643684746182778, -170.914688644882, -0.015034947590733308, 0.7122753653274643,
           -212.01520180800415, 0.0002780259742405138, 0.005364810186631225, 1.0]


def test_ball_projection_agrees_with_the_ground_plane_and_puts_height_up():
    cv2 = pytest.importorskip("cv2")
    from sportvu.camera import pose_from_h
    W, H = 1280, 720
    rvec, f, C = pose_from_h(H_LABEL, W, H)
    assert C[2] < 0                                                   # the camera sits at negative z: height is -z
    ground = np.array([[30.0, 20.0], [47.0, 25.0], [70.0, 30.0]])
    P = np.linalg.inv(np.array(H_LABEL).reshape(3, 3))
    want = cv2.perspectiveTransform(ground.reshape(-1, 1, 2), P).reshape(-1, 2)
    uv0, depth = S.project3d(C, rvec, f, W, H, np.column_stack([ground, np.zeros(3)]))
    assert (depth > 0).all() and np.allclose(uv0, want, atol=1.0)
    uv10, _ = S.project3d(C, rvec, f, W, H, np.column_stack([ground, np.full(3, 10.0)]))
    assert (uv10[:, 1] < uv0[:, 1]).all()                             # 10 ft up draws higher in the image


def test_third_of_clamps_points_just_off_the_court():
    assert S.third_of(-2.0, True) == "far" and S.third_of(52.0, True) == "near"
    assert S.third_of(-2.0, False) == "near" and S.third_of(25.0, False) == "middle"


def test_f1_is_zero_when_nothing_is_right():
    assert S.f1(0, 3, 2) == 0.0 and S.f1(0, 0, 4) == 0.0 and S.f1(0, 0, 0) is None


def test_the_universe_leaves_out_frames_the_build_never_processed():
    rows = [{"frame": f} for f in (3, 6, 9, 12)]
    kept, left_out = S.universe(rows, live_wide={3, 6, 9}, processed={3: {}, 9: {}})
    assert [r["frame"] for r in kept] == [3, 9] and left_out == 1          # 12 is not live wide; 6 was never processed


def test_events_count_only_near_a_live_wide_frame():
    lw = [(1, 700.2), (1, 699.9), (2, 300.0)]
    ev = [(1, 700.0), (1, 690.0), (2, 300.6), (3, 300.0)]
    assert S.live_wide_events(ev, lw) == [(1, 700.0)]
    assert S.CONSTANTS["event_live_wide_s"] == 0.5


def test_the_window_offset_comes_from_the_fit_also_without_truth_rows(tmp_path):
    P = {"labels": tmp_path}
    (tmp_path / "g_offsets.json").write_text(json.dumps({"g_s00_w00": 0.44}))
    rows, off = S.truth_rows(P, "g", {"window": "g_s00_w00"}, "camera", index=None, mirror="identity")
    assert rows == [] and off == 0.44
    rows, off = S.truth_rows(P, "g", {"window": "g_s00_w01"}, "camera", index=None, mirror="identity")
    assert rows == [] and off is None


def test_the_identity_reported_row_is_not_measurable_with_the_frozen_switches_beside_it():
    R = S.v2_rows([{"ball_pred_written": False}], events_pred_written=False)
    row = R["identity_reported"]
    assert row["value"] is None and "jersey" in row["not_measurable"] and "id_switches_per_possession_frozen" in row


def test_caveats_name_the_third_clamp_and_the_in_sample_build():
    assert any("clamp" in c and "region_of" in c for c in S.caveats(False))
    assert any(c.startswith("IN SAMPLE") for c in S.caveats(True)) and not any(c.startswith("IN SAMPLE") for c in S.caveats(False))
    assert S.build_in_sample("gsw_bkn_2015", "r23") and not S.build_in_sample("nyk_bos_2016", "r23")
    assert not S.build_in_sample("gsw_bkn_2015", "v3")


def _fake_card(root, tag, build, mode, errs):
    import gzip
    (root / "scorecards" / "obs").mkdir(parents=True, exist_ok=True)
    card = {"frozen_rows": {"position_error": {"p50_ft": 1.0, "p90_ft": 2.0}, "missed_players": {"rate": 0.2}},
            "universe": {"truth_share": 0.5}, "build_in_sample": S.build_in_sample(tag, build),
            "intervals": {"position_p50_ft": [0.5, 1.5], "position_p90_ft": [1.5, 2.5]}}
    (root / "scorecards" / f"{tag}__{build}__{mode}.json").write_text(json.dumps(card))
    with gzip.open(root / "scorecards" / "obs" / f"{tag}__{build}__{mode}.jsonl.gz", "wt") as fh:
        for k, e in enumerate(errs):
            fh.write(json.dumps({"w": f"{tag}_s00_w0{k % 2}", "err": e, "missed": False}) + "\n")
    with gzip.open(root / "scorecards" / "obs" / f"{tag}__{build}__{mode}.reproj.json.gz", "wt") as fh:
        fh.write(json.dumps([10.0, 20.0, 30.0]))


def test_pooled_rows_keep_r13_and_channel_games_apart(tmp_path):
    _fake_card(tmp_path, "gsw_bkn_2015", "r23", "camera", [1.0, 1.2, 1.4])
    _fake_card(tmp_path, "nyk_bos_2016", "r23", "camera", [3.0, 3.2])
    out = S.combine("r23", "camera", ["gsw_bkn_2015", "nyk_bos_2016"], root=tmp_path)
    assert set(out["groups"]) == {"r13", "channel"}
    assert out["groups"]["r13"]["build_in_sample"] and not out["groups"]["channel"]["build_in_sample"]
    assert out["groups"]["r13"]["games"] == ["gsw_bkn_2015"] and out["groups"]["channel"]["position_p50_ft"] == 3.1


def test_the_gate_refuses_without_the_b4_scorecards(tmp_path):
    for t in S.CONSTANTS["gate_games"]:
        _fake_card(tmp_path, t, "r23", "camera", [1.0, 1.1])
    with pytest.raises(SystemExit):
        S.gate(root=tmp_path)
    for t in S.CONSTANTS["gate_games"]:
        _fake_card(tmp_path, t, "r23", "b4", [1.0, 1.1])
    out = S.gate(root=tmp_path)
    assert out["pass"] and set(out["games"]) == set(S.CONSTANTS["gate_games"])
    assert {S.build_in_sample(t, "r23") for t in S.CONSTANTS["gate_games"]} == {True, False}   # one R1.3, one channel


def test_fix_ball_tolerance_needs_every_game_and_writes_once(tmp_path):
    out = tmp_path / "p8_constants.json"
    _fake_card(tmp_path, "gsw_bkn_2015", "r23", "camera", [1.0])
    with pytest.raises(SystemExit):
        S.fix_ball_tolerance(root=tmp_path, out_path=out)              # the other development games are missing
    from tools import p8_frozen as F
    for t in F.games():
        _fake_card(tmp_path, t, "r23", "camera", [1.0])
    rec = S.fix_ball_tolerance(root=tmp_path, out_path=out)
    assert rec["ball_tol_px"] == 40.0 and out.exists()
    with pytest.raises(SystemExit):
        S.fix_ball_tolerance(root=tmp_path, out_path=out)              # never rewritten

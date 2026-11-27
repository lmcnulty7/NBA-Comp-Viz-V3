"""tools.p8_multistart_stage2: the stage 2 gate, the game table, the workspace manifest and the refusals. No video, no
model; nothing under the real P8 roots is written."""

import json

import pytest

from tools import p8_multistart_stage2 as S


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def test_stage2_gate_needs_cha_kill4_and_stability_and_bkn_stability(tmp_path):
    ms = tmp_path / "work_ms"
    with pytest.raises(SystemExit, match="waits"):
        S.gate_stage2(ms)
    cha, bkn = (ms / "multistart" / g / "multipool.json" for g in ("gsw_cha_2016", "gsw_bkn_2015"))
    write(cha, {"kill4": {"stage2_runs": True}, "stability": {"passes": True}})
    write(bkn, {"stability": {"passes": False}})
    with pytest.raises(SystemExit, match="kill rule 3"):
        S.gate_stage2(ms)
    write(cha, {"kill4": {"stage2_runs": False}, "stability": {"passes": True}})
    write(bkn, {"stability": {"passes": True}})
    with pytest.raises(SystemExit, match="kill rule 4 or stability"):
        S.gate_stage2(ms)
    write(cha, {"kill4": {"stage2_runs": True}, "stability": {"passes": True}})
    S.gate_stage2(ms)


def test_the_stage2_games_and_their_steps():
    assert set(S.STAGE2) == {"hou_cle_2016", "gsw_sac_2015", "gsw_ind_2016"}
    assert [g for g, v in S.STAGE2.items() if v["truth"]] == ["hou_cle_2016"]
    assert S.STAGE2["gsw_ind_2016"]["label_root"] == "work" and S.STAGE2["gsw_sac_2015"]["label_root"] == "ref"


def test_workspace_manifest_sees_a_change_to_the_game_only(tmp_path):
    (tmp_path / "labels" / "gsw_ind_2016").mkdir(parents=True)
    (tmp_path / "label_build").mkdir()
    (tmp_path / "labels" / "gsw_ind_2016_result.json").write_text("{}")
    (tmp_path / "labels" / "gsw_ind_2016" / "w.jsonl").write_text("1")
    (tmp_path / "label_build" / "gsw_ind_2016_s01_w00_frames.json").write_text("{}")
    (tmp_path / "label_build" / "gsw_cha_2016_s01_w00_frames.json").write_text("{}")
    a = S.workspace_manifest(tmp_path, "gsw_ind_2016")
    assert len(a) == 3 and not any("gsw_cha" in k for k in a)
    (tmp_path / "labels" / "gsw_ind_2016" / "w.jsonl").write_text("2")
    assert S.workspace_manifest(tmp_path, "gsw_ind_2016") != a


def test_refusals():
    with pytest.raises(SystemExit, match="not a stage 2 game"):
        S._check("gsw_cha_2016")
    with pytest.raises(SystemExit, match="CV test|not a stage 2"):
        S._check("tor_gsw_2016")
    with pytest.raises(SystemExit, match="no truth step"):
        S.truth("gsw_ind_2016")
    with pytest.raises(SystemExit, match="no truth step"):
        S.accept("gsw_ind_2016")
    for g in ("hou_cle_2016", "gsw_sac_2015"):                     # deferred by section 17's narrowing
        for step in (S.multipool, S.truth, S.accept):
            with pytest.raises(SystemExit, match="deferred"):
                step(g)


def test_ind_reading_rule():
    F = (46.64, 106.04, -34.73)
    far = (47.2, 137.0, -30.7)
    assert S.ind_reading(far, 98.0, F, 100.0, True) == "same fragility"
    assert S.ind_reading(far, 99.0, F, 100.0, True) == "same fragility"          # exactly 0.99 J_F
    assert S.ind_reading((46.64, 109.0, -34.7), 50.0, F, 100.0, True) == "not the same"   # within 5 ft
    assert S.ind_reading(far, 99.5, F, 100.0, True) == "not the same"            # J above 0.99 J_F
    assert S.ind_reading(far, 50.0, F, 100.0, False) == "inconclusive"           # far and lower but not stable


def test_ind_anchor():
    rec = {"C": [46.639, 106.0425, -34.7329], "inits": 3900, "reproj_px_median_after": 18.35}
    fz = {"status": "ok", "C": [46.639, 106.0425, -34.7329], "inits": 3900, "reproj_px_median_after": 18.35}
    assert S.anchor_problem(fz, rec) is None
    assert S.anchor_problem({**fz, "C": [46.639, 106.06, -34.7329]}, rec).startswith("C ")
    assert S.anchor_problem({**fz, "inits": 3899}, rec).startswith("inits")
    assert S.anchor_problem({**fz, "reproj_px_median_after": 18.37}, rec).startswith("pooled median")
    assert S.anchor_problem({"status": "too_few_inits"}, rec).startswith("fit_game status")
    assert (S.IND_ESTIMATE_S, S.IND_CAP_S, S.ENABLED) == (4640, 6960, ("gsw_ind_2016",))


def loop_lines(src: str) -> list:
    body = src[src.index("    for set_name, seeds in"):src.index("    bad_j = ")]
    out = []
    for line in body.splitlines():
        if "stage 2 only" in line:
            continue
        code = line.split("#")[0].rstrip().replace("M.", "")
        if code.strip():
            out.append(code)
    return out


def test_stage2_loop_equals_stage1_loop():
    import inspect
    from tools import p8_multistart as M
    one, two = loop_lines(inspect.getsource(M.multipool)), loop_lines(inspect.getsource(S.multipool))
    assert len(one) > 20 and one == two

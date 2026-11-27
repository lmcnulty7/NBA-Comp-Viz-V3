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
    for g in ("gsw_sac_2015", "gsw_ind_2016"):
        with pytest.raises(SystemExit, match="no truth step"):
            S.truth(g)
        with pytest.raises(SystemExit, match="no truth step"):
            S.accept(g)

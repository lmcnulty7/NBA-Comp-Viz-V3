"""P8 Phase A tools: the workspace stays outside the repository and DATA_ROOT, the registry is guarded, the scorebug
check reads the score correctly, and the night run plans only development games."""

import json
from pathlib import Path

import pytest

from tools import p8_common as C
from tools import p8_night as N
from tools import p8_scorebug as SB
from tools import research_testset as R


def test_the_workspace_is_outside_the_repository_data_root_and_the_main_checkout(monkeypatch, tmp_path):
    root = C.p8_root()
    assert C.REPO not in root.parents and root != C.REPO
    with pytest.raises(SystemExit):
        C.check_outside(C.REPO / "reports" / "x")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    with pytest.raises(SystemExit):
        C.check_outside(tmp_path / "dr" / "p8")
    monkeypatch.setenv("V3_P8_ROOT", str(C.REPO / "data" / "p8"))
    with pytest.raises(SystemExit):
        C.p8_root()
    assert C.check_outside(tmp_path / "ok") == (tmp_path / "ok").resolve()


def test_the_registry_holds_no_cv_test_item_and_every_id_passes_the_guard():
    reg = C.load_registry()
    cv = json.loads(R.TESTSET_FILE.read_text())["sources"][R.BROADCAST_CV]
    text = C.REGISTRY.read_text()
    assert not [i for i in list(cv["games"]) + list(cv["video_ids"].values()) if i in text]
    assert C.channel_games() and set(C.channel_games()) == set(reg["games"])
    homes = {h for hs in cv["arena_homes"].values() for h in hs}
    for t, g in reg["games"].items():
        assert g["home"] not in homes, t
        assert g["status"] in ("pending", "confirmed", "rejected", "ambiguous"), t
        if g["status"] == "confirmed":
            assert g["sportvu"] in g["candidates"] and g.get("scorebug_check"), t


def test_the_guard_refuses_spent_and_test_items():
    with pytest.raises(SystemExit):
        C.guard(["12.16.2015.PHX.at.GSW"])                    # spent held-out game
    with pytest.raises(SystemExit):
        C.guard(["01.01.2016.DAL.at.MIA"])                    # a test arena
    assert C.guard(["11.27.2015.CHI.at.IND", "wtGuZJAkm08"]) == ["11.27.2015.CHI.at.IND", "wtGuZJAkm08"]


def test_development_games_are_the_r13_games_plus_confirmed_channel_games():
    dg = C.dev_games()
    assert {"gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015"} <= set(dg)
    confirmed = {t for t, g in C.load_registry()["games"].items() if g["status"] == "confirmed"}
    assert set(dg) == {"gsw_bkn_2015", "gsw_cha_2016", "gsw_ind_2016", "gsw_sac_2015"} | confirmed


def test_score_at_reads_the_play_by_play_score_at_a_running_clock():
    ev = [(1, 700.0, "2-0"), (1, 690.0, "2-2"), (1, 0.0, "20-18"), (2, 704.0, "22-18"), (2, 650.0, "22-21")]
    assert SB.score_at(ev, 1, 695.0) == (2, 0)
    assert SB.score_at(ev, 2, 717.0) == (20, 18)               # start of Q2 carries Q1's last score
    assert SB.score_at(ev, 2, 660.0) == (22, 18)
    assert SB.matches(ev, 2, 703.0, 22, 18) and not SB.matches(ev, 2, 703.0, 28, 27)


def test_the_night_plan_runs_one_game_at_a_time_and_locks_every_model_job(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_P8_ROOT", str(tmp_path / "p8"))
    pl = N.plan()
    assert list(pl["games"])[:4] == list(N.R13)
    for tag, g in pl["games"].items():
        names = [j["name"] for j in g["jobs"]]
        assert names.index("fit") > max(i for i, n in enumerate(names) if n.startswith("label "))
        assert names.index("fit") < min(i for i, n in enumerate(names) if n.startswith("rebuild "))
        assert names[-4:] == ["score r23 camera", "score v3 camera", "score r23 b4", "score r23 camera_paint"]
        for j in g["jobs"]:
            step = j["argv"][1]
            assert j["heavy"] == (step not in ("link", "split", "moments", "mirror")), (tag, j["name"])
    assert pl["lock_h_total"] > 0
    small = N.plan(["gsw_bkn_2015"], max_sections=1)
    assert list(small["games"]) == ["gsw_bkn_2015"]
    assert sum(j["name"].startswith("label ") for j in small["games"]["gsw_bkn_2015"]["jobs"]) == 1


def test_install_paths_points_every_frozen_output_at_the_workspace(tmp_path):
    pytest.importorskip("cv2")
    from tools import p8_frozen as F
    import config
    import harvest_driver
    from sportvu import autolabel, direction, fetch, local_sync, rebuild, sync, truth
    saved = {(m, a): getattr(m, a) for m, a in ((config, "REPORTS_DIR"), (config, "HEAD_V2_PATH"),
                                                 (harvest_driver, "HARVEST_VIDEO"), (fetch, "SV_DIR"),
                                                 (sync, "SYNC_DIR"), (rebuild, "SYNC_DIR"), (rebuild, "BUILD_DIR"),
                                                 (autolabel, "SYNC_DIR"), (autolabel, "OUT"), (autolabel, "WIN_DIR"),
                                                 (direction, "SYNC_DIR"), (direction, "DIR_BUILD"),
                                                 (local_sync, "SYNC_DIR"), (truth, "TRUTH_DIR"),
                                                 (rebuild, "video_path"), (local_sync, "video_path"), (sync, "video_path"))}
    try:
        root = (tmp_path / "ws").resolve()
        F.install_paths(root)
        for (m, a) in saved:
            v = getattr(m, a)
            if a == "video_path":
                v = v("x_s00")
            if a == "HEAD_V2_PATH":
                continue
            assert root in Path(v).parents or Path(v) == root, (m.__name__, a, v)
    finally:
        for (m, a), v in saved.items():
            setattr(m, a, v)


def test_period_words_map_to_periods():
    from tools.p8_frozen import period_from_words
    assert [period_from_words(w) for w in ("FIRST QTR", "SECONDQTR", "third", "FOURTH", "OT", "OVERTIME", "2ND", "")] \
        == [1, 2, 3, 4, 5, 5, None, None]


def test_registry_layouts_never_shadow_a_frozen_layout():
    import clock_reader
    reg = C.load_registry()
    assert not set(reg.get("layouts") or {}) & set(clock_reader.LAYOUTS)
    for t, g in reg["games"].items():
        if g.get("layout"):
            assert g["layout"] in clock_reader.LAYOUTS or g["layout"] in reg["layouts"], t

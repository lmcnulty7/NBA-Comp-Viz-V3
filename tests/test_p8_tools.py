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
        scores = ["score r23 camera", "score v3 camera", "score r23 camera_paint"]
        from tools.p8_scorecard import CONSTANTS
        assert names[-4:] == scores + ["score r23 b4"] if tag in CONSTANTS["gate_games"] else names[-3:] == scores
        assert names[0] == "sweep"
        for j in g["jobs"]:
            step = j["argv"][1]
            assert j["heavy"] == (step not in ("sweep", "link", "split", "moments", "mirror")), (tag, j["name"])
    assert sum(any(j["name"] == "score r23 b4" for j in g["jobs"]) for g in pl["games"].values()) == 2
    assert pl["lock_h_total"] > 0
    small = N.plan(["gsw_bkn_2015"], max_sections=1)
    assert list(small["games"]) == ["gsw_bkn_2015"]
    assert sum(j["name"].startswith("label ") for j in small["games"]["gsw_bkn_2015"]["jobs"]) == 1
    with_b4 = N.plan(["gsw_cha_2016"], b4="all")["games"]["gsw_cha_2016"]["jobs"]
    assert with_b4[-1]["name"] == "score r23 b4" and with_b4[-1]["heavy"]
    assert not any(j["name"] == "score r23 b4" for j in N.plan(["gsw_bkn_2015"], b4="none")["games"]["gsw_bkn_2015"]["jobs"])


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


# ── runner robustness (review items 14 to 17) ───────────────────────────────────────────────────────────────────────
def test_a_smoke_run_never_marks_a_full_job_done(tmp_path, monkeypatch):
    from tools import p8_frozen as F
    monkeypatch.setenv("V3_P8_ROOT", str(tmp_path / "p8"))
    P = F.layout(C.p8_root())
    assert N.config_key(1, "gate") != N.config_key(None, "gate") != N.config_key(None, "all")
    fit = {"name": "fit", "argv": ["tools.p8_frozen", "fit", "gsw_bkn_2015"], "heavy": True}
    assert not N.output_ok(P, "gsw_bkn_2015", fit, None)                     # no marker: not done
    marker = F.done_path(P, "fit", "gsw_bkn_2015")
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"result": {"partial": True}}))
    assert N.output_ok(P, "gsw_bkn_2015", fit, 1) and not N.output_ok(P, "gsw_bkn_2015", fit, None)
    marker.write_text(json.dumps({"result": {"partial": False}}))
    assert N.output_ok(P, "gsw_bkn_2015", fit, None)
    score = {"name": "score r23 camera", "argv": ["tools.p8_scorecard", "game", "gsw_bkn_2015", "--build", "r23",
                                                 "--truth", "camera"], "heavy": True}
    card = P["scorecards"] / "gsw_bkn_2015__r23__camera.json"
    card.parent.mkdir(parents=True)
    card.write_text("{}")
    import os
    os.utime(card, (1, 1))                                                    # older than the fit: stale
    assert not N.output_ok(P, "gsw_bkn_2015", score, None)
    os.utime(card, None)
    assert N.output_ok(P, "gsw_bkn_2015", score, None)
    jobs = N.jobs_for("gsw_bkn_2015", {}, P, 11, max_sections=1)
    assert [a for n, a, *_ in jobs if n == "fit"][0][-1] == "--partial"


def _fake_window(P, stem="gsw_bkn_2015_s00_w00"):
    return {"window": stem, "section": "gsw_bkn_2015_s00", "fps": 30.0, "f_start": 0, "f0": 45, "f_end": 300, "period": 1}


def test_a_failed_build_fails_the_step_and_moves_its_outputs_aside(tmp_path, monkeypatch):
    from tools import p8_frozen as F
    from sportvu import rebuild
    import config
    P = F.layout(tmp_path / "p8")
    for p in P.values():
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "TRACKING_DIR", tmp_path / "trk")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "repo")
    (tmp_path / "trk").mkdir()
    w = _fake_window(P)

    def bad_build(window, snip, out_dir, extra_args=()):
        (out_dir / (window["window"] + "_frames.json")).write_text("{}")   # a partial output of a crashed build
        return {"window": window["window"], "rc": 1, "moved": [], "tail": "boom"}
    monkeypatch.setattr(rebuild, "build", bad_build)
    r = F.safe_build(P, w, P["label_build"] / "x.mp4", P["label_build"])
    assert not r["ok"] and not (P["label_build"] / (w["window"] + "_frames.json")).exists()
    assert (P["failed"] / "label_build" / (w["window"] + "_frames.json")).exists()
    with pytest.raises(SystemExit):
        F.fail_if([w["window"]], "s00")


def test_the_checkout_is_swept_of_build_leftovers(tmp_path, monkeypatch):
    from tools import p8_frozen as F
    import config
    P = F.layout(tmp_path / "p8")
    monkeypatch.setattr(config, "TRACKING_DIR", tmp_path / "repo" / "data" / "tracking")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "repo")
    (tmp_path / "repo" / "data" / "tracking").mkdir(parents=True)
    (tmp_path / "repo" / "reports" / "viz").mkdir(parents=True)
    left = [tmp_path / "repo" / "data" / "tracking" / "gsw_bkn_2015_s03_w01_trajectories.json",
            tmp_path / "repo" / "reports" / "viz" / "team_clusters_gsw_bkn_2015_s03_w01.png"]
    keep = tmp_path / "repo" / "data" / "tracking" / "other_clip_trajectories.json"
    for f in left + [keep]:
        f.write_text("x")
    moved = F.step_sweep(P, "gsw_bkn_2015")["moved"]
    assert sorted(moved) == sorted(f.name for f in left) and keep.exists() and not any(f.exists() for f in left)
    assert all((P["failed"] / "checkout_leftovers" / f.name).exists() for f in left)


def test_cached_outputs_are_written_atomically_and_a_cut_lands_whole(tmp_path, monkeypatch):
    from tools import p8_frozen as F
    from sportvu import rebuild
    out = C.atomic_write(tmp_path / "a.json", "{}")
    assert out.read_text() == "{}" and not list(tmp_path.glob("*.part"))
    P = F.layout(tmp_path / "p8")
    for p in P.values():
        p.mkdir(parents=True, exist_ok=True)
    seen = {}

    def fake_cut(window, out_dir):
        seen["dir"] = out_dir
        p = out_dir / (window["window"] + ".mp4")
        p.write_text("clip")
        return p
    monkeypatch.setattr(rebuild, "cut", fake_cut)
    w = _fake_window(P)
    final = F.safe_cut(P, w, P["label_build"])
    assert seen["dir"] != P["label_build"] and final.read_text() == "clip" and final.parent == P["label_build"]


def test_the_guard_catches_held_out_ids_in_any_form_and_a_test_arena_home():
    for item in ("f8lAcHg6kk0", "bzmm0WiWYog", "gsw_phx_2016", "/v/cle_nyk_2015_s01.mp4",
                 "https://www.youtube.com/watch?v=f8lAcHg6kk0", "12.23.2015.NYK.at.CLE_s02"):
        with pytest.raises(SystemExit):
            C.guard([item])
    with pytest.raises(SystemExit):
        C.guard(["mia_xyz_2016"], homes=["MIA"])            # a registry entry's own home, even with no candidates
    assert C.guard(["nyk_bos_2016"], homes=["NYK"]) == ["nyk_bos_2016"]


def test_a_registry_entry_in_a_test_arena_is_refused_without_candidates(tmp_path):
    reg = C.load_registry()
    reg["games"]["mia_xyz_2016"] = {"video_id": "zzzzzzzzzzz", "home": "MIA", "away": "XYZ", "candidates": [],
                                    "status": "pending", "sportvu": None}
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(reg))
    with pytest.raises(SystemExit):
        C.channel_games(p)


def test_a_partial_clip_read_is_never_cached(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from tools import p8_scorecard as S
    vid = tmp_path / "w.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 36))
    for k in range(12):
        vw.write(np.full((36, 64, 3), k * 10, np.uint8))
    vw.release()

    class Gate:
        class backbone:
            @staticmethod
            def embed_images(batch, batch_size=32):
                return batch

        @staticmethod
        def score_embeddings(emb):
            return [1.0] * len(emb)

    class Refs:
        wide_thr = 0.5
        gate = staticmethod(lambda: Gate)
        referees = staticmethod(lambda img: [])
    P = {"cache": tmp_path / "cache"}
    full = S.frame_refs(P, "w_ok", vid, [0, 3, 6], set(), 3, Refs)
    assert full["complete"] and (tmp_path / "cache" / "scorecard" / "w_ok.json").exists()
    part = S.frame_refs(P, "w_bad", vid, [0, 3, 600], set(), 3, Refs)       # frame 600 is past the end
    assert not part["complete"] and not (tmp_path / "cache" / "scorecard" / "w_bad.json").exists()


ZERO_GATE = ("build processed 0 frames (successful reads: 120, total frames: 375). All read frames were gate-rejected "
             "or outside pre-gate intervals, gate threshold/domain issue, or the section is genuinely dead footage.")
ZERO_PREGATE = ("build processed 0 frames (successful reads: 0, total frames: 375). Video unreadable by cv2, unsupported "
                "codec (AV1 on Colab, run-7 postmortem: ffprobe the file), FUSE/network-mounted storage, or corruption.")


def _empty_build_env(tmp_path, monkeypatch, tail, decodable=True):
    from tools import p8_frozen as F
    from sportvu import rebuild
    import config
    P = F.layout(tmp_path / "p8")
    for p in P.values():
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "TRACKING_DIR", tmp_path / "trk")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "repo")
    (tmp_path / "trk").mkdir()
    calls = []

    def zero_build(window, snip, out_dir, extra_args=()):
        calls.append(window["window"])
        return {"window": window["window"], "rc": 1, "moved": [], "tail": tail}
    monkeypatch.setattr(rebuild, "build", zero_build)
    monkeypatch.setattr(F, "_decodable", lambda p: decodable)
    return F, P, calls


@pytest.mark.parametrize("tail", [ZERO_GATE, ZERO_PREGATE])
def test_a_window_the_gate_empties_is_marked_empty_not_failed_and_never_rebuilt(tmp_path, monkeypatch, tail):
    """night 1: gsw_cha_2016_s06_w07 is a picture-in-picture interview over running clock; gate v2 rejects every
    frame, the frozen build exits 1 with "processed 0 frames", and the step must go on, as the frozen autolabel did."""
    F, P, calls = _empty_build_env(tmp_path, monkeypatch, tail)
    w = _fake_window(P, "gsw_cha_2016_s06_w07")
    r = F.safe_build(P, w, P["label_build"] / "x.mp4", P["label_build"])
    assert r["ok"] and r["empty"] and F.gate_empty_path(P["label_build"], w["window"]).exists()
    again = F.safe_build(P, w, P["label_build"] / "x.mp4", P["label_build"])
    assert again["ok"] and again["empty"] and again["cached"] and calls == [w["window"]]


def test_a_zero_frame_exit_on_an_undecodable_clip_still_fails(tmp_path, monkeypatch):
    F, P, _ = _empty_build_env(tmp_path, monkeypatch, ZERO_PREGATE, decodable=False)
    w = _fake_window(P, "gsw_cha_2016_s06_w07")
    r = F.safe_build(P, w, P["label_build"] / "x.mp4", P["label_build"])
    assert not r["ok"] and not F.gate_empty_path(P["label_build"], w["window"]).exists()


def test_any_other_build_failure_still_fails(tmp_path, monkeypatch):
    F, P, _ = _empty_build_env(tmp_path, monkeypatch, "Traceback ... RuntimeError: MPS out of memory")
    w = _fake_window(P, "gsw_cha_2016_s06_w07")
    assert not F.safe_build(P, w, P["label_build"] / "x.mp4", P["label_build"])["ok"]

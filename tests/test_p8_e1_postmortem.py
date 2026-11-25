"""tools.p8_e1_postmortem: the pure rules of the E1 post-mortem (M1 label audit, M2 miss pairing, M3 label oracle) on
synthetic data, and that they reuse the corpus and scorer constants."""

import numpy as np
import pytest

from tools import p8_e1_postmortem as M
from tools import p8_teachers as T

H = np.diag([0.1, 0.1, 1.0])          # px -> ft, 10 px a foot


def test_constants_agree_with_the_frozen_rules():
    from sportvu import eval as E
    from sportvu.bench import MATCH_FT
    assert (M.MISS_FT, M.GHOST_FT, M.FRAME_MARGIN_PX) == (E.MISS_FT, E.GHOST_FT, E.FRAME_MARGIN_PX)
    assert M.NEAR_FT == MATCH_FT
    assert T.CONFIRM_FT == 2.5


def test_yolo_lines_round_trip_to_pixels():
    from tools.p8_e1data import yolo_line
    box = [100.0, 200.0, 180.0, 400.0]
    back = M.yolo_to_boxes(yolo_line(box, 1280, 720) + "\n\n", 1280, 720)
    assert len(back) == 1 and np.allclose(back[0], box, atol=0.01)
    assert M.yolo_to_boxes("", 1280, 720) == []


def test_assignment_is_the_found_targets_match():
    rng = np.random.default_rng(1)
    for _ in range(50):
        feet, tg = rng.uniform(0, 10, (rng.integers(0, 6), 2)), rng.uniform(0, 10, (rng.integers(1, 6), 2))
        assert set(M.assignment(feet, tg, 2.5)) == T.found_targets(feet, tg)


def test_in_image_mask_is_targets_in_frame():
    sv = np.array([[1.0, 1.0], [130.0, 10.0], [-1.0, 5.0], [127.9, 71.9]])
    mask = M.in_image_mask(sv, H, 1280, 720)
    assert mask.tolist() == [True, False, False, True]
    assert np.array_equal(sv[mask], T.targets_in_frame(sv, H, 1280, 720))


def test_box_class():
    assert M.box_class(True, False, None) == "r23 only"
    assert M.box_class(False, True, None) == "teacher only"
    assert M.box_class(True, True, 0.3) == "both, IoU < 0.5"
    assert M.box_class(True, True, 0.5) == "both, IoU >= 0.5"
    assert M.box_class(False, False, None) == "neither"


def test_edge_and_height_bins():
    assert M.is_edge(39.0, 300.0, 1280, 720) and M.is_edge(600.0, 690.0, 1280, 720)
    assert not M.is_edge(640.0, 360.0, 1280, 720)
    assert [M.height_bin(v) for v in (10, 50, 99.9, 120, 150, None, np.nan)] == \
        ["<50", "50-75", "75-100", "100-150", ">=150", "no pose", "no pose"]
    assert M.HEIGHT_BINS == ("<50", "50-75", "75-100", "100-150", ">=150", "no pose")
    assert M.height_bin(210, M.HEIGHT_EDGES_FINE) == "200-250"


def test_audit_frame_labels_and_classes():
    # four players at 10, 30, 50 and 70 ft (y 5 ft); feet at bottom centres, 10 px a foot
    sv = np.array([[10.0, 5.0], [30.0, 5.0], [50.0, 5.0], [70.0, 5.0], [500.0, 5.0]])
    box = lambda x_ft, w=20: [x_ft * 10 - w / 2, 0.0, x_ft * 10 + w / 2, 50.0]
    rec = {"size": [1280, 720],
           T.OURS: [box(10) + [0.9, "player"], box(30) + [0.9, "player"]],
           T.MISS_TEACHER: [box(10) + [0.9, "player"], box(50) + [0.9, "player"]]}
    from tools.p8_e1data import yolo_line
    labels = yolo_line(box(10), 1280, 720) + "\n"
    rows, orphans, n = M.audit_frame(rec, labels, sv, np.arange(5), H, T)
    by = {r["pid"]: r for r in rows}
    assert len(rows) == 4 and orphans == 0 and n == 1               # player 4 is off the image
    assert by[0]["labelled"] and by[0]["cls"] is None
    assert by[1]["cls"] == "r23 only" and by[2]["cls"] == "teacher only" and by[3]["cls"] == "neither"
    assert not by[3]["near_box_5ft"]


def test_pair_classes_and_net():
    a = [{"w": "w", "f": 1, "pid": p, "missed": m} for p, m in ((1, False), (2, False), (3, True), (4, True))]
    b = [{"w": "w", "f": 1, "pid": p, "missed": m} for p, m in ((1, False), (2, True), (3, False), (4, True))]
    c = M.pair_classes(a, b)
    assert c == {("w", 1, 1): "both found", ("w", 1, 2): "E1 extra miss", ("w", 1, 3): "E1 extra find",
                 ("w", 1, 4): "both missed"}
    with pytest.raises(SystemExit):
        M.pair_classes(a, b[:3])


def test_m2_status_and_share_diff():
    players = {"w:1:1": {"labelled": True, "cls": None}, "w:1:2": {"labelled": False, "cls": "r23 only",
                                                                   "label_foot_within_5ft": True}}
    rows = M.m2_status([{"w": "w", "f": 1, "pid": 1, "cls": "both found"},
                        {"w": "w", "f": 1, "pid": 2, "cls": "E1 extra miss"},
                        {"w": "w", "f": 2, "pid": 1, "cls": "E1 extra miss"}], players)
    assert [r["status"] for r in rows] == ["labelled", "r23 only", "not a corpus frame"]
    assert rows[1]["status_near"] == "unlabelled, a label foot within 5 ft"
    g = M.window_groups(rows)
    assert g == {"w": [2, 1, 3, 1]}
    sd = M.share_diff({"a": (2, 2, 10, 1), "b": (2, 1, 10, 1)}, n=200)
    assert sd["diff"] == pytest.approx(3 / 4 - 2 / 20)
    assert sd["window_interval"][0] <= sd["diff"] <= sd["window_interval"][1]


def test_oracle_counts_follow_eval_frame():
    sv = np.array([[10.0, 5.0], [30.0, 5.0], [50.0, 5.0], [90.0, 5.0]])
    inside = np.array([True, True, True, False])
    feet = np.array([[10.5, 5.0], [11.0, 5.0], [32.0, 5.0], [70.0, 5.0], [92.0, 5.0]])
    c = M.oracle_counts(feet, sv, inside)
    # player 0 has two feet within 2.5 ft (a duplicate), 1 found at 2 ft, 2 missed; player 3 is out of frame;
    # the box at 70 ft is a ghost; the box at 92 ft is near the out-of-frame player, so it is not a ghost
    assert c == {"in_frame": 3, "boxes": 5, "missed": 1, "ghosts": 1, "duplicates": 1}
    assert M.oracle_counts(np.zeros((0, 2)), sv, inside) == {"in_frame": 3, "boxes": 0, "missed": 3, "ghosts": 0,
                                                             "duplicates": 0}


def test_scorer_inside_uses_the_margin():
    sv = np.array([[0.5, 5.0], [1.5, 5.0], [127.5, 5.0], [10.0, 71.5]])
    assert M.scorer_inside(sv, H, 1280, 720).tolist() == [False, True, False, False]


def test_oracle_boot_point_and_interval():
    pw = {k: {"ours": {"in_frame": 10, "missed": 3, "boxes": 10, "ghosts": 1, "duplicates": 0},
              "labels": {"in_frame": 10, "missed": m, "boxes": 10, "ghosts": 2, "duplicates": 0}}
          for k, m in (("a", 2), ("b", 3), ("c", 1))}
    r = M.oracle_boot(pw, "ours", "labels", n=200)
    assert r["missed_rate"]["diff"] == pytest.approx(0.2 - 0.3)
    assert r["ghost_rate"]["diff"] == pytest.approx(0.1)
    assert r["ghost_rate"]["window_interval"] == [pytest.approx(0.1), pytest.approx(0.1)]


def test_games_are_guarded():
    assert M.check_games(["gsw_bkn_2015"]) == ["gsw_bkn_2015"]
    for bad in ("gsw_ind_2016", "gsw_phx_2016"):
        with pytest.raises(SystemExit):
            M.check_games([bad])
    src = open(M.__file__).read()
    assert "refuse_cv_test(" in src and "guard(" in src and ".train(" not in src and ".fit(" not in src


def test_guard_refuses_a_held_out_game_even_when_listed(monkeypatch):
    calls = []
    real = M.refuse_cv_test
    monkeypatch.setattr(M, "refuse_cv_test", lambda items, who, **k: calls.append(list(items)) or real(items, who, **k))
    monkeypatch.setattr(M, "POOL", M.POOL + ("gsw_phx_2016",))
    with pytest.raises(SystemExit, match="held-out"):
        M.check_games(["gsw_phx_2016"])
    assert calls == [["gsw_phx_2016"]]


def test_own_label_box_is_one_to_one_and_unclaimed():
    targets = np.array([[10.0, 5.0], [14.0, 5.0], [40.0, 5.0]])
    feet = np.array([[10.2, 5.0], [11.0, 5.0], [44.0, 5.0]])     # box 0 claims player 0; box 1 is free, 3 ft from 1
    lab = M.assignment(feet, targets, 2.5)
    assert lab == {0: 0}
    assert M.own_label_within(feet, targets, lab) == {1, 2}      # free boxes 1 and 2 go to players 1 and 2
    assert M.own_label_within(feet[:1], targets, lab) == set()   # a claimed box never counts for a neighbour

"""tools.p8_e1data: chunking, the source merge, YOLO lines, the source 2 rule and the corpus guards, on synthetic data."""

import numpy as np
import pytest

from tools import p8_e1data as D
from tools import p8_teachers as T


def test_chunks_keep_whole_windows_in_order():
    ws = [("a", [1] * 400), ("b", [1] * 300), ("c", [1] * 100), ("d", [1] * 700)]
    ch = D.chunks_of(ws, 600)
    assert [[w for w, _ in c] for c in ch] == [["a", "b"], ["c", "d"]]


def test_source2_boxes_overlapping_source1_are_written_once():
    s1 = [[0, 0, 10, 30, 0.9]]
    s2 = [[1, 0, 11, 30, 0.8], [50, 0, 60, 30, 0.7]]
    labels, both = D.merge_sources(s1, s2, T.iou)
    assert labels == [[0, 0, 10, 30], [50, 0, 60, 30]] and both == 1


def test_yolo_line_is_normalised_and_clipped():
    assert D.yolo_line([-10, 0, 640, 360], 1280, 720) == "0 0.250000 0.250000 0.500000 0.500000"


def test_teacher_boxes_count_only_at_players_our_detector_missed():
    H = np.diag([0.1, 0.1, 1.0])                                     # px -> ft, 10 px a foot
    targets = np.array([[1.0, 5.0], [10.0, 5.0], [20.0, 5.0]])
    t = [[0, 0, 20, 50, 0.9], [90, 0, 110, 50, 0.8], [300, 0, 320, 50, 0.7]]   # feet at 1, 10 and 31 ft
    assert D._teacher_at_missed(t, targets, {0}, H, T) == [t[1]]      # target 0 found by ours, target 2 too far
    assert D._teacher_at_missed([], targets, set(), H, T) == []


def test_corpus_games_and_guards():
    assert D.GAMES == ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
    assert not set(D.OUT_OF_CORPUS) & set(D.GAMES)
    src = open(D.__file__).read()
    assert "refuse_cv_test(" in src and "guard(" in src and ".train(" not in src


def test_the_corpus_lives_outside_the_repository(monkeypatch):
    monkeypatch.setenv("V3_P8_E1", str(D.REPO / "data" / "e1"))
    with pytest.raises(SystemExit):
        D.corpus_root()


def test_colab_split_keeps_whole_windows_and_takes_the_last_ones_per_game():
    from tools import p8_e1_colab as C
    stems = [f"g1/g1_s00_w{w:02d}_f{f:06d}" for w in range(10) for f in (1, 2)] + \
            [f"g2/g2_s01_w{w:02d}_f{f:06d}" for w in range(5) for f in (1,)]
    train, val = C.split_windows(stems, 0.1)
    assert val == ["g1/g1_s00_w09_f000001", "g1/g1_s00_w09_f000002", "g2/g2_s01_w04_f000001"]
    assert len(train) + len(val) == len(stems) and not set(train) & set(val)
    assert "refuse_cv_test(" in open(C.__file__).read()

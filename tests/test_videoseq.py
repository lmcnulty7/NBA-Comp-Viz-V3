"""SeqReader correctness: every access pattern must return the exact frame a
per-frame cap.set() would have — sequential decode is a speed change only."""
import cv2
import numpy as np
import pytest

from conftest import N_FRAMES, frame_value, open_video
from videoseq import SEEK_MIN_GAP, SeqReader


def assert_is_frame(frame, idx):
    v = float(np.mean(frame[..., 0] if frame.ndim == 3 else frame))
    expect = frame_value(idx)
    assert abs(v - expect) < 6, (
        f"read at index {idx} returned luma {v:.1f}, expected ~{expect} — "
        f"nearest frame values are 13 apart, so this is the WRONG frame")


@pytest.fixture()
def reader(video_60fps):
    cap = open_video(video_60fps)
    assert cap.isOpened()
    yield SeqReader(cap)
    cap.release()


def test_stride_walk_grab_path(reader):
    """The build loop's pattern: monotonic small strides, no seeks needed."""
    for idx in range(0, 300, 6):
        ok, frame = reader.read(idx)
        assert ok
        assert_is_frame(frame, idx)


def test_gap_below_threshold_crosses_gop(reader):
    """A dead-gap jump smaller than SEEK_MIN_GAP grabs forward, across a
    GOP boundary (GOP=300; 120 → 360)."""
    ok, frame = reader.read(120)
    assert ok
    ok, frame = reader.read(360)
    assert ok
    assert_is_frame(frame, 360)


def test_gap_above_threshold_seeks(reader):
    ok, _ = reader.read(0)
    assert ok
    idx = 1 + SEEK_MIN_GAP + 100          # forces the cap.set() branch
    ok, frame = reader.read(idx)
    assert ok
    assert_is_frame(frame, idx)


def test_backward_read_seeks(reader):
    ok, _ = reader.read(750)
    assert ok
    ok, frame = reader.read(100)
    assert ok
    assert_is_frame(frame, 100)


def test_exact_gop_boundaries(reader):
    for idx in (299, 300, 301, 600):
        ok, frame = reader.read(idx)
        assert ok
        assert_is_frame(frame, idx)


def test_eof_and_recovery(reader):
    ok, frame = reader.read(N_FRAMES - 1)
    assert ok
    assert_is_frame(frame, N_FRAMES - 1)
    ok, frame = reader.read(N_FRAMES + 300)   # past EOF → clean failure
    assert not ok
    ok, frame = reader.read(10)               # backward seek recovers
    assert ok
    assert_is_frame(frame, 10)


def test_matches_per_frame_seek_reference(video_60fps):
    """Bit-for-bit agreement with the old cap.set()-per-frame pattern on a
    mixed access sequence (stride walk + gap jumps)."""
    indices = list(range(0, 120, 6)) + [200, 360, 750, 751, 757]
    cap_a = open_video(video_60fps)
    cap_b = open_video(video_60fps)
    assert cap_a.isOpened() and cap_b.isOpened()
    reader = SeqReader(cap_a)
    for idx in indices:
        ok_new, f_new = reader.read(idx)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_old, f_old = cap_b.read()
        assert ok_new and ok_old, f"frame {idx} unreadable"
        assert np.array_equal(f_new, f_old), f"frame {idx} differs from cap.set reference"
    cap_a.release()
    cap_b.release()

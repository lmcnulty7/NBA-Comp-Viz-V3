"""sportvu.grid_dataset: an R1.3 row becomes a YOLO-pose line in the grid model's format."""
import numpy as np

from court.grid import GRID_FT
from sportvu.grid_dataset import r13_label

P = np.array([[14.0, 0.0, -300.0], [0.0, 12.0, 60.0], [0.0, 0.0, 1.0]])   # court ft -> px


def test_label_line_format_and_visibility():
    line, n_vis = r13_label({"H_truth": np.linalg.inv(P).ravel().tolist()}, 1280, 720)
    f = line.split()
    assert f[0] == "0" and len(f) == 5 + 91 * 3
    kp = np.array(f[5:], float).reshape(91, 3)
    px = np.c_[GRID_FT, np.ones(91)] @ P.T
    px = px[:, :2] / px[:, 2:]
    on = (px[:, 0] >= 0) & (px[:, 0] < 1280) & (px[:, 1] >= 0) & (px[:, 1] < 720)
    assert n_vis == on.sum() == (kp[:, 2] == 2).sum()
    assert np.allclose(kp[on, :2] * [1280, 720], px[on], atol=0.01)


def test_too_few_visible_points_is_no_label():
    far = np.array([[1.0, 0.0, 5000.0], [0.0, 1.0, 5000.0], [0.0, 0.0, 1.0]])   # court far off-frame
    line, n_vis = r13_label({"H_truth": np.linalg.inv(far).ravel().tolist()}, 1280, 720)
    assert line is None and n_vis == 0

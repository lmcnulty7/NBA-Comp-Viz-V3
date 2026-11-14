import numpy as np
from sportvu.camera import h_px2ft, project
from sportvu import camera_track as ct


def _look_at(C, target):
    """rvec of a camera at C looking at target, image x along court +x (court z is up = negative here)."""
    import cv2
    z = np.asarray(target, float) - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, -1.0]); x /= np.linalg.norm(x)
    if x[0] < 0:
        x = -x
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    if np.linalg.det(R) < 0:
        R[1] = -R[1]
    return cv2.Rodrigues(R)[0].ravel()


def _synthetic(C, n=40, w=1280, h=720, seed=0):
    """Frames of a fixed camera panning over the court, with exact line matches."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rv = _look_at(C, [rng.uniform(20, 74), rng.uniform(15, 35), 0.0])
        f = 1500 + rng.normal(0, 100)
        ft = np.column_stack([rng.uniform(0, 94, 200), rng.uniform(0, 50, 200)])
        px = project(C, rv, f, w, h, ft)
        ok = (px[:, 0] > 0) & (px[:, 0] < w) & (px[:, 1] > 0) & (px[:, 1] < h)
        H = h_px2ft(C, rv, f, w, h)
        rows.append({"frame": i, "state": "TRACK", "H": H.ravel().tolist(), "ft": ft[ok].tolist(), "px": px[ok].tolist(), "res": 1.0})
    return {"w0": {"w": w, "h": h, "rows": rows}}


def test_position_and_frames_recovered():
    C = np.array([47.0, 125.0, -29.0])
    tr = _synthetic(C)
    usable = [r for r in tr["w0"]["rows"] if len(r["ft"]) >= ct.POOL_MIN_MATCH]
    assert len(usable) >= 20
    pos = ct.fit_position(tr)
    assert pos["status"] == "ok"
    assert np.abs(np.array(pos["C"]) - C).max() < 0.5
    der = ct.derive(tr, pos["C"])
    for r, d in zip(tr["w0"]["rows"], der["w0"]):
        if d["H"] is None:
            continue
        H0, H1 = np.array(r["H"]).reshape(3, 3), np.array(d["H"]).reshape(3, 3)
        assert np.abs(H0 / H0[2, 2] - H1 / H1[2, 2]).max() < 1e-2 or d["how"] == "free_h"


def test_lost_frames_stay_lost():
    tr = {"w0": {"w": 1280, "h": 720, "rows": [{"frame": 0, "state": "LOST", "H": None, "ft": [], "px": [], "res": None}]}}
    assert ct.derive(tr, [47, 125, -29])["w0"][0]["H"] is None

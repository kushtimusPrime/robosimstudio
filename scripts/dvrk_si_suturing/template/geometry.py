# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Deterministic suturing geometry. All lengths in metres; no robot edits."""

import csv
import json
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def recorded(profile, episode, image_index=0):
    rows = list(csv.DictReader(open(ROOT / "ref/dataset" / episode / "ee_csv.csv")))
    x = float(np.clip(image_index - profile["lag_frames"], 0, len(rows) - 1))
    a = int(x)
    b = min(a + 1, len(rows) - 1)
    w = x - a
    row = {k: (1 - w) * float(rows[a][k]) + w * float(rows[b][k]) for k in rows[a] if rows[a][k]}
    q = {}
    for arm in ("PSM1", "PSM2", "PSM3", "ECM"):
        joints = ["yaw", "pitch", "insertion", "roll"] + ([] if arm == "ECM" else ["wrist_pitch", "wrist_yaw"])
        for i, j in enumerate(joints):
            q[arm + "_" + j] = row[arm.lower() + f"_js[{i}]"]
        q[arm + "_pitch_2"] = -q[arm + "_pitch"]
        q[arm + "_pitch_3"] = q[arm + "_pitch"]
        if arm != "ECM":
            v = row.get(arm.lower() + "_jaw", 0)
            g = profile["geometry"]
            v = g[arm + "_jaw_offset"] + g[arm + "_jaw_gain"] * (v + 0.35) if arm != "PSM3" else v / 2
            q[arm + "_jaw_1"] = v
            q[arm + "_jaw_2"] = -v
    return q


def camera_and_robot(profile, episode):
    u = yourdfpy.URDF.load(ROOT / profile["asset"], load_meshes=False)
    u.update_cfg(recorded(profile, episode))
    x = profile["camera_in_ecm_roll"]
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(x[3:]).as_matrix()
    m[:3, 3] = x[:3]
    return u.get_transform("ECM_roll_link") @ m, u


def orient_tets(v, t):
    t = np.asarray(t, dtype=np.int32)
    a = v[t]
    vol = np.linalg.det(np.stack((a[:, 1] - a[:, 0], a[:, 2] - a[:, 0], a[:, 3] - a[:, 0]), axis=-1))
    neg = vol < 0
    t[neg, 0], t[neg, 1] = t[neg, 1].copy(), t[neg, 0].copy()
    assert np.min(np.abs(vol)) > 1e-20
    return t


def needle_mesh(p, stations=65, sides=12):
    """Circular tapered solid; fan of triangular wedges, each split into 3 tets."""
    radius = p["arc_length_m"] / p["angle_rad"]
    theta = np.linspace(0, p["angle_rad"], stations)
    centre = np.column_stack((radius * (np.cos(theta) - 1), radius * np.sin(theta), theta * 0))
    verts = []
    for i, t in enumerate(theta):
        taper = np.clip((p["arc_length_m"] * (1 - i / (stations - 1))) / p["tip_taper_m"], 0.025, 1)
        verts.append(centre[i])
        for j in range(sides):
            a = 2 * np.pi * j / sides
            verts.append(
                centre[i] + p["radius_m"] * taper * np.array([np.cos(a) * np.cos(t), np.cos(a) * np.sin(t), np.sin(a)])
            )
    tets = []
    stride = sides + 1
    for i in range(stations - 1):
        for j in range(sides):
            a = i * stride
            b = a + 1 + j
            c = a + 1 + (j + 1) % sides
            d = a + stride
            e = b + stride
            f = c + stride
            # Consistent prism splitting by ordered lower-face vertex ids.
            a, b, c = sorted([a, b, c])
            d, e, f = a + stride, b + stride, c + stride
            tets.extend([[a, b, c, f], [a, b, e, f], [a, d, e, f]])
    v = np.array(verts)
    return v, orient_tets(v, tets), centre


def tissue_mesh(p):
    """Wounded pad, continuous below the slit; coarse far field, denser wound lips.

    No needle channel is cut in advance. The permanent central wound is present
    in the real pad. Base fixation is a separate, explicit boundary assumption.
    """
    h = p["size_m"][2]
    half = p["wound_length_m"] / 2
    xs = np.unique(np.r_[-0.075, -0.035, -0.022, -0.017, np.linspace(-half, half, 15), 0.017, 0.022, 0.035, 0.075])
    ys = np.array([0, 0.001, 0.0025, 0.0035, 0.005, 0.009, 0.016, 0.035, 0.075])
    zs = np.unique(np.array([0, p["rigid_backing_m"], h - p["wound_depth_m"], h - 0.0015, h]))
    vertices = []
    lookup = {}
    sides = {}
    for sign in [-1, 1]:
        grid = np.empty((len(xs), len(ys), len(zs)), dtype=int)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    window = max(0, 1 - (x / half) ** 8) if abs(x) < half else 0
                    gap = 0.0003 * window * max(0, (z - (h - p["wound_depth_m"])) / p["wound_depth_m"]) if j == 0 else 0
                    rim = np.sqrt(y * y + max(0, abs(x) - half) ** 2)
                    bump = 0.0018 * np.exp(-(((rim - 0.0035) / 0.0022) ** 2)) * (z / h) ** 2
                    pos = (float(x), float(sign * (y + gap)), float(z + bump))
                    key = tuple(round(c, 12) for c in pos)
                    if key not in lookup:
                        lookup[key] = len(vertices)
                        vertices.append(pos)
                    grid[i, j, k] = lookup[key]
        sides[sign] = grid
    tets = []
    # Freudenthal subdivision: common face triangulations throughout each grid.
    import itertools

    for grid in sides.values():
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                for k in range(len(zs) - 1):
                    for perm in itertools.permutations(range(3)):
                        q = np.array([i, j, k])
                        ids = [grid[tuple(q)]]
                        for d in perm:
                            q = q.copy()
                            q[d] += 1
                            ids.append(grid[tuple(q)])
                        tets.append(ids)
    v = np.array(vertices)
    return v, orient_tets(v, tets)


def build_assets(params, profile):
    out = ROOT / "assets/suturing"
    out.mkdir(parents=True, exist_ok=True)
    cam, u = camera_and_robot(profile, params["episode"])
    v, t, curve = needle_mesh(params["needle"])
    # Initial needle pose is an episode state estimate, not a robot calibration.
    # Swage sits between the driver's contact faces. Exact 3-D needle pose still
    # needs a multi-frame image fit; this is deliberately recorded as a prior.
    jaw = u.get_transform("PSM1_wrist_yaw_link")
    fit = json.loads((ROOT / "assets/suturing/needle_contact_initial.json").read_text())
    swage = (jaw @ np.r_[fit["needle_swage_in_wrist_m"], 1])[:3]
    rot = jaw[:3, :3] @ Rotation.from_quat(fit["needle_rotation_in_wrist_xyzw"]).as_matrix()
    v = v @ rot.T + swage
    curve = curve @ rot.T + swage
    np.savez(out / "needle.npz", vertices=v, tets=t)
    tv, tt = tissue_mesh(params["tissue"])
    # Pad surface initially 54 mm from the calibrated camera. Camera is frozen.
    # Local x follows image right, local y follows image down, local z faces lens.
    padR = cam[:3, :3] @ np.diag([1, -1, -1])
    padcentre = (cam @ np.array([-0.002, 0, 0.062, 1]))[:3]
    tv = tv @ padR.T + padcentre
    np.savez(out / "tissue.npz", vertices=tv, tets=tt)
    # 120 mm of visible/free strand, continued out of view. A finite-length
    # initial-state assumption, not a claimed manufacturer cut length.
    knots = np.array(
        [
            swage,
            swage + cam[:3, :3] @ np.array([0.004, -0.007, 0]),
            swage + cam[:3, :3] @ np.array([0.006, -0.025, 0.006]),
            swage + cam[:3, :3] @ np.array([-0.02, -0.035, 0.012]),
            swage + cam[:3, :3] @ np.array([-0.04, -0.015, 0.014]),
            swage + cam[:3, :3] @ np.array([-0.025, 0.008, 0.014]),
        ]
    )
    lens = np.r_[0, np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
    arclen = np.linspace(0, lens[-1], 121)
    from scipy.interpolate import CubicSpline

    dense = CubicSpline(lens, knots, axis=0)(np.linspace(0, lens[-1], 2000))
    dl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    arclen = np.linspace(0, dl[-1], 121)
    thread = np.column_stack([np.interp(arclen, dl, dense[:, i]) for i in range(3)])
    np.savez(
        out / "initial.npz", camera=cam, needle_centreline=curve, thread=thread, pad_rotation=padR, pad_centre=padcentre
    )
    return out, cam, padR, padcentre, thread

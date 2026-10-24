# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Local-space convex contact distances for initial-state feasibility checks."""

import json
from pathlib import Path

import numpy as np
import trimesh
from geometry import ROOT, camera_and_robot
from scipy.spatial import ConvexHull


def jaw_planes():
    profile = json.loads((ROOT / "assets/multisession/setup_017/profile.json").read_text())
    params = json.loads((ROOT / "parameters.json").read_text())
    _, u = camera_and_robot(profile, params["episode"])
    from appearance import visual_asset
    from contacts import asset
    from exterior import visual_asset as exterior_asset

    contact = Path(asset(exterior_asset(visual_asset(ROOT / profile["asset"]))))
    manifest = json.loads(contact.with_name("manifest.json").read_text())
    wrist = u.get_transform("PSM1_wrist_yaw_link")
    result = []
    for side in [1, 2]:
        name = f"PSM1_jaw_{side}_link"
        tr = np.linalg.inv(u.get_transform(name)) @ wrist
        planes = []
        for rec in manifest["changed_links"][name]:
            mesh = trimesh.load(contact.parent / rec["file"], force="mesh")
            eq = ConvexHull(mesh.vertices).equations
            planes.append((eq[:, :3], eq[:, 3]))
        result.append((tr, planes))
    return result


def distances(points, jaws):
    """Supporting-plane signed distance; exact inside, conservative near corners."""
    output = []
    for tr, parts in jaws:
        local = points @ tr[:3, :3].T + tr[:3, 3]
        values = np.array([np.max(local @ normal.T + bias, axis=1) for normal, bias in parts])
        output.append(values.min(axis=0))
    return np.array(output)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    f = json.loads((ROOT / "assets/suturing/needle_image_fit.json").read_text())
    theta = np.linspace(0, np.pi, 500)
    r = f["circle_radius_m"]
    points = (
        np.c_[r * (np.cos(theta) - 1), r * np.sin(theta), theta * 0]
        @ Rotation.from_quat(f["needle_rotation_in_wrist_xyzw"]).as_matrix().T
        + f["needle_swage_in_wrist_m"]
    )
    gaps = distances(points, jaw_planes()) - 0.00024
    print("minimum centreline-to-contact gap mm, minus wire radius:", gaps.min(axis=1) * 1000)
    print("curve arc positions at minima:", theta[gaps.argmin(axis=1)] * r * 1000)

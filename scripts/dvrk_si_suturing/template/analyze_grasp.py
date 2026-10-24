# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Read numerical gates from a completed contact-only diagnostic."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

path = Path(sys.argv[1])
d = json.loads(path.read_text())
h = d["history"]
t = np.array([r["time_s"] for r in h])
pos = np.array([r["relative_position_m"] for r in h])
rot = Rotation.from_quat([r["relative_rotation_xyzw"] for r in h])
curve = np.load(path.parent.parent / "assets/suturing/initial.npz")["needle_centreline"] - h[0]["needle_position_m"]
points = np.einsum("nij,kj->nki", rot.as_matrix(), curve) + pos[:, None, :]
ref = np.argmin(np.abs(t - min(0.5, t[-1] / 2)))
held = (t >= t[ref]) & (t <= 4.0)
drift = np.max(np.linalg.norm(points - points[ref], axis=-1), axis=1)
angle = (rot[ref].inv() * rot).magnitude() * 180 / np.pi
release = t >= 4.3
metrics = {
    "time_s": float(t[-1]),
    "reference_time_s": float(t[ref]),
    "max_held_surface_drift_mm": float(drift[held].max() * 1000),
    "max_held_rotation_deg": float(angle[held].max()),
    "max_initial_settling_surface_motion_mm": float(np.linalg.norm(points[ref] - points[0], axis=-1).max() * 1000),
    "maximum_penetration_um": float(max(0.0, -min(r["minimum_contact_gap_m"] for r in h if r["time_s"] <= 4.0)) * 1e6),
    "max_mimic_error_rad": float(max(abs(r["mimic_error_rad"]) for r in h)),
    "finite": all(r["finite"] for r in h),
    "wrist_travel_mm": float(
        np.linalg.norm(np.array([r["wrist_position_m"] for r in h]) - h[0]["wrist_position_m"], axis=1).max() * 1000
    ),
    "release_evaluated": bool(release.any()),
}
if release.any():
    metrics["max_release_jaw_contacts"] = max(
        r.get("jaw_contacts", r["needle_contacts"]) for r in h if r["time_s"] >= 4.3
    )
    metrics["release_relative_displacement_mm"] = float(drift[-1] * 1000)
metrics["pass"] = bool(
    metrics["finite"]
    and metrics["max_held_surface_drift_mm"] < 0.2
    and metrics["maximum_penetration_um"] < 20
    and metrics["max_mimic_error_rad"] < 1e-3
    and metrics["wrist_travel_mm"] > 4
    and metrics.get("max_release_jaw_contacts", 1) == 0
    and metrics.get("release_relative_displacement_mm", 0) > 5
)
path.with_name(path.stem + "_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

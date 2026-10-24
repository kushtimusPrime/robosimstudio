# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Make the image-fit initial needle pose feasible in the unchanged visual jaws.

Only the needle's initial translation is adjusted. This is not calibration,
not a gripper attachment, and is never applied after simulation starts.
"""

import json

import numpy as np
from contact_geometry import distances, jaw_planes
from geometry import ROOT
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def run():
    fit = json.loads((ROOT / "assets/suturing/needle_image_fit.json").read_text())
    params = json.loads((ROOT / "parameters.json").read_text())
    theta = np.linspace(0, np.pi, 600)
    r = fit["circle_radius_m"]
    wire = params["needle"]["radius_m"]
    points = (
        np.c_[r * (np.cos(theta) - 1), r * np.sin(theta), theta * 0]
        @ Rotation.from_quat(fit["needle_rotation_in_wrist_xyzw"]).as_matrix().T
        + fit["needle_swage_in_wrist_m"]
    )
    jaws = jaw_planes()

    def residual(shift):
        gaps = distances(points + shift, jaws) - wire
        # Tiny initial overlap preloads a real frictional grasp. It is 5 microns,
        # not a grip latch; release is entirely governed by contact forces.
        return np.r_[(gaps.min(axis=1) + 0.000005) * 1e6, shift * 1e3]

    solve = least_squares(
        residual, np.zeros(3), bounds=(-0.002, 0.002), diff_step=1e-4, xtol=1e-12, ftol=1e-12, gtol=1e-9
    )
    shift = solve.x
    fit["image_only_swage_in_wrist_m"] = fit["needle_swage_in_wrist_m"]
    fit["needle_swage_in_wrist_m"] = (np.array(fit["needle_swage_in_wrist_m"]) + shift).tolist()
    fit["contact_initialization"] = {
        "translation_from_image_fit_m": shift.tolist(),
        "minimum_gaps_m": (distances(points + shift, jaws) - wire).min(axis=1).tolist(),
        "note": "Initial condition only; image-only residuals elsewhere in this file do not describe this adjusted pose.",
    }
    (ROOT / "assets/suturing/needle_contact_initial.json").write_text(json.dumps(fit, indent=2))
    print(json.dumps(fit["contact_initialization"], indent=2))


if __name__ == "__main__":
    run()

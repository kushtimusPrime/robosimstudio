# SPDX-License-Identifier: Apache-2.0
"""Task-local rigid-contact accuracy settings, not calibrated material constants.

MuJoCo's generic millimetre-scale compliant constraints let the 0.48 mm needle
penetrate by 0.20 mm and the mimic follower separate by 0.06 rad. Resolve the
existing geometric constraints more closely; no frames or inertias change.
"""

import numpy as np


def configure(pipeline, contact_time_s=4e-6, mimic_time_s=1e-4, impedance=0.9999):
    solver = getattr(pipeline.solver, "newton_solver", pipeline.solver)
    if not solver.use_mujoco_cpu:
        raise ValueError("Surgical contact experiment requires CPU MuJoCo")
    m = solver.mj_model
    m.opt.iterations = 200
    m.opt.tolerance = 1e-12
    m.opt.impratio = 10.0
    m.opt.noslip_iterations = 0
    m.opt.cone = 1  # elliptic friction cone
    m.opt.noslip_tolerance = 1e-12
    # Existing URDF mimic equations retain the same coefficients and joint ids.
    m.eq_solref[:, 0] = mimic_time_s
    m.eq_solref[:, 1] = 1.0
    m.eq_solimp[:, :2] = impedance
    selected = []
    for i in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[i])).name
        if "jaw_" in body or "needle" in body:
            selected.append(i)
    selected = np.asarray(selected, dtype=int)
    m.geom_solref[selected] = [contact_time_s, 1.0]
    m.geom_solimp[selected, :2] = impedance
    m.geom_solimp[selected, 2] = 1e-5
    # A finite finger contact patch resists spin about the contact normal.
    # Torsional friction has units of length; 50 um is an explicit patch prior.
    m.geom_condim[selected] = 6
    m.geom_friction[selected, 1] = 5e-5
    m.geom_friction[selected, 2] = 1e-5
    return {
        "contact_time_s": contact_time_s,
        "mimic_time_s": mimic_time_s,
        "impedance": impedance,
        "geoms": len(selected),
        "torsional_friction_m": 5e-5,
        "condim": 6,
        "rolling_friction_m": 1e-5,
        "note": "Numerical constraint resolution, not fitted contact compliance.",
    }

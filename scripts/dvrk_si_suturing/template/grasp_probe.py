# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Contact-only needle grasp diagnostic in the robot's rigid solver.

No thread, tissue, attachment or prescribed needle transform. Robot joint
commands close, carry, return and release the freely integrated needle.
"""

import json
from pathlib import Path

import numpy as np
from multisession_preview import replay
from native_needle import specification
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def add_needle(scene, folder, parameters):
    from dataclasses import replace

    # The standard rigid entity has no swage coupling hook.
    return scene.add(replace(specification(folder, parameters), entity="rigid"))


def scalar_targets(sim, q):
    targets = sim.control.joint_target_q.numpy().copy()
    qs = sim.model.joint_q_start.numpy()
    ds = sim.model.joint_qd_start.numpy()
    for j, name in enumerate(sim.model.joint_label):
        if str(name).startswith("dvrk_si/") and qs[j + 1] - qs[j] == 1:
            targets[ds[j]] = q[qs[j]]
    return targets


def read(sim, needle):
    solver = getattr(sim.pipeline.solver, "newton_solver", sim.pipeline.solver)
    m, d = solver.mj_model, solver.mj_data
    labels = list(map(str, sim.model.body_label))
    wrist = next(i for i, name in enumerate(labels) if name.split("/")[-1] == "PSM1_wrist_yaw_link")
    poses = sim.state.body_q.numpy()
    a, b = poses[wrist], poses[needle.entity.body]
    Ra, Rb = Rotation.from_quat(a[3:]), Rotation.from_quat(b[3:])
    contacts = []
    jaw_contacts = 0
    for c in d.contact[: d.ncon]:
        if any("needle_collision" in m.geom(int(g)).name for g in c.geom):
            contacts.append(float(c.dist))
            jaw_contacts += int(any("jaw_" in m.body(int(m.geom_bodyid[int(g)])).name for g in c.geom))
    q = sim.state.joint_q.numpy()
    names = [str(n).split("/")[-1] for n in sim.model.joint_label]
    starts = sim.model.joint_q_start.numpy()
    jaws = [float(q[starts[names.index("PSM1_jaw_" + str(i))]]) for i in (1, 2)]
    return {
        "time_s": float(sim.pipeline.sim_time),
        "relative_position_m": Ra.inv().apply(b[:3] - a[:3]).tolist(),
        "relative_rotation_xyzw": (Ra.inv() * Rb).as_quat().tolist(),
        "joint_positions_rad_m": [
            float(q[starts[j]]) for j, name in enumerate(names) if name.startswith(("PSM", "ECM"))
        ],
        "needle_pose_xyzw": b.tolist(),
        "needle_position_m": b[:3].tolist(),
        "wrist_position_m": a[:3].tolist(),
        "needle_velocity": sim.state.body_qd.numpy()[needle.entity.body].tolist(),
        "jaws_rad": jaws,
        "mimic_error_rad": sum(jaws),
        "needle_contacts": len(contacts),
        "jaw_contacts": jaw_contacts,
        "minimum_contact_gap_m": min(contacts, default=0.0),
        "finite": bool(np.isfinite(poses).all()),
    }


def run(sim, needle, profile, params, *, tag="grasp_probe", preload=0.05, duration=None, ramp_s=0.02):
    if sim.step_count:
        raise RuntimeError("Reset the scene before starting the contact diagnostic.")
    replay(sim, None, profile, 0, params["episode"])
    import mujoco

    solver = getattr(sim.pipeline.solver, "newton_solver", sim.pipeline.solver)
    mujoco.mj_resetData(solver.mj_model, solver.mj_data)
    solver._update_mjc_data(solver.mj_data, sim.model, sim.state)
    q0 = sim.state.joint_q.numpy().copy()
    names = [str(n).split("/")[-1] for n in sim.model.joint_label]
    qs = sim.model.joint_q_start.numpy()
    idx = {name: int(qs[j]) for j, name in enumerate(names)}
    hz = params["numerics"]["control_hz"]
    history = [read(sim, needle)]
    total = 4.5 if duration is None else duration
    for step in range(round(total * hz)):
        t = (step + 1) / hz
        q = q0.copy()
        close = min(t / ramp_s, 1.0)
        if t > 4.0:
            close = max(0.0, 1 - (t - 4.0) / 0.1)
        jaw = q0[idx["PSM1_jaw_1"]] - preload * close
        if t > 4.1:
            jaw = q0[idx["PSM1_jaw_1"]] + 0.3 * min((t - 4.1) / 0.15, 1.0)
        q[idx["PSM1_jaw_1"]] = jaw
        q[idx["PSM1_jaw_2"]] = -jaw
        # Smooth 5 mm insertion withdrawal plus 0.15 rad roll, then return.
        travel = 0.0
        if 1.0 < t < 2.0:
            travel = 0.5 - 0.5 * np.cos(np.pi * (t - 1.0))
        elif 2.0 <= t <= 3.0:
            travel = 1.0
        elif 3.0 < t < 4.0:
            travel = 0.5 + 0.5 * np.cos(np.pi * (t - 3.0))
        q[idx["PSM1_insertion"]] -= 0.005 * travel
        q[idx["PSM1_roll"]] += 0.15 * travel
        sim.control.joint_target_q.assign(scalar_targets(sim, q))
        sim.run(1)
        history.append(read(sim, needle))
        yield
    from robosimstudio.robots.registry import get_embodiment

    m = solver.mj_model
    result = {
        "urdf": get_embodiment("dvrk_dataset_profile").urdf,
        "needle_mesh": needle.spec.asset.path,
        "joint_names": [name for name in names if name.startswith(("PSM", "ECM"))],
        "numerics": {
            "rigid_dt_s": m.opt.timestep,
            "contact_solref": m.geom_solref[-1].tolist(),
            "contact_solimp": m.geom_solimp[-1].tolist(),
            "friction": m.geom_friction[-1].tolist(),
            "impratio": m.opt.impratio,
            "noslip_iterations": m.opt.noslip_iterations,
            "persistent_double_state": solver.update_data_interval == 0,
        },
        "needle_mass_kg": float(sim.model.body_mass.numpy()[needle.entity.body]),
        "status": "diagnostic, not a suturing pass",
        "preload_rad": preload,
        "closure_ramp_s": ramp_s,
        "thread_and_tissue_present": False,
        "history": history,
    }
    (ROOT / "run" / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print("Saved", ROOT / "run" / f"{tag}.json")

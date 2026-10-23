# SPDX-License-Identifier: Apache-2.0
"""Basic IK, on the dVRK Si

PyRoKi's ``examples/01_basic_ik.py`` with the dVRK Si cart in place of the
Panda: the URDF RoboSimStudio steps (``assets/embodiments/dvrk_si/dvrk_si.urdf``),
solved by PyRoKi alone, so the URDF can be judged without the library between.

Two changes from the original, both from PyRoKi's own examples: a target per
arm, solved in one problem (``02_bimanual_ik.py``), and locks per joint and arm
(``13_ik_with_locked_joints.py``'s joint mask, which zeroes a locked joint's
Jacobian column, so locking never recompiles). A locked arm's target is pinned
to where it stands. The targets are the frames the dVRK itself reports: each
PSM's ``tool_tip_link`` and the ECM's ``ECM_camera_link``. The jaws are not
solved. Working arms start halfway between their limits; PSM3 stays parked
and the ECM keeps the selected recording's camera pose.

    uv run python scripts/dvrk_si_pyroki_ik.py            # http://localhost:8080 (or the next free port)
    uv run python scripts/dvrk_si_pyroki_ik.py --check    # scripted solves, printed, no page
    uv run python scripts/dvrk_si_pyroki_ik.py --start suturebot   # SutureBot's posture instead of mid-range
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk  # noqa: TID253 -- this script is PyRoKi alone; it has no use without it
import yourdfpy

from robosimstudio.robots.embodiments.dvrk_si import ECM_HOME, PSM1_HOME, PSM2_HOME, PSM3_HOME, PSM_JOINTS

URDF = Path(__file__).resolve().parents[1] / "assets" / "embodiments" / "dvrk_si" / "dvrk_si.urdf"
ECM_JOINTS = ("yaw", "pitch", "insertion", "roll")
ARMS = {
    # arm: (target link, joints, SutureBot's posture, locked at start)
    "PSM1": ("PSM1_tool_tip_link", PSM_JOINTS, PSM1_HOME, False),
    "PSM2": ("PSM2_tool_tip_link", PSM_JOINTS, PSM2_HOME, False),
    "PSM3": ("PSM3_tool_tip_link", PSM_JOINTS, PSM3_HOME, True),
    "ECM": ("ECM_camera_link", ECM_JOINTS, ECM_HOME, True),
}


@jaxls.Cost.factory(kind="constraint_leq_zero")
def _limits(vals, robot, joint_var, joint_mask, prev_cfg):
    # Locked coordinates are constants, including when a log exceeds a limit.
    cfg = jnp.where(joint_mask > 0, vals[joint_var], prev_cfg)
    full = robot.joints.get_full_config(cfg)
    return jnp.concatenate(
        [jnp.maximum(0.0, full - robot.joints.upper_limits_all), jnp.maximum(0.0, robot.joints.lower_limits_all - full)]
    )


@jdc.jit
def _solve_ik(
    robot: pk.Robot,
    target_link_indices: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    joint_mask: jax.Array,
    prev_cfg: jax.Array,
    active_targets: jax.Array,
) -> jax.Array:
    """One problem over every target; ``joint_mask`` is 1 to solve a joint and 0 to lock it."""
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position)
    batch = target_pose.get_batch_axes()
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            robot.joint_var_cls(jnp.zeros(batch, dtype=jnp.int32)),
            target_pose,
            target_link_indices,
            pos_weight=50.0 * active_targets,
            ori_weight=10.0 * active_targets,
            joint_mask=jnp.broadcast_to(joint_mask, (*batch, joint_mask.shape[-1])),
        ),
        _limits(robot, joint_var, joint_mask, prev_cfg),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return sol[joint_var]


def load(start: str = "middle"):
    """The URDF, its PyRoKi robot, the target link indices, and a starting config.

    ``start`` is ``"middle"`` -- working arms halfway between limits, with
    PSM3 parked and the ECM at its recorded pose -- or ``"suturebot"``, the
    recorded home posture (tissue 7 knot-tying recovery, row 372), selected
    for working-joint clearance from the stops.
    """
    urdf = yourdfpy.URDF.load(str(URDF))
    robot = pk.Robot.from_urdf(urdf)
    names = list(robot.joints.actuated_names)
    targets = np.array([robot.links.names.index(link) for link, *_ in ARMS.values()])
    cfg = np.asarray(robot.joint_var_cls.default_factory(), dtype=np.float64)
    for arm, (_, joints, home, initially_locked) in ARMS.items():
        if start == "suturebot" or initially_locked:
            for joint, value in zip(joints, home, strict=True):
                cfg[names.index(f"{arm}_{joint}")] = value
    if start == "suturebot":
        for arm in ("PSM1", "PSM2", "PSM3"):
            cfg[names.index(f"{arm}_jaw_1")] = 0.35  # half open: a 0.7 rad dVRK jaw angle
    return urdf, robot, names, targets, cfg


def mask_for(names: list[str], locked: dict[str, bool]) -> np.ndarray:
    """Joint-name locks override arm defaults; jaws remain outside tool-tip IK."""
    return np.array(
        [
            0.0 if name.endswith("_jaw_1") or locked.get(name, locked.get(name.split("_", 1)[0], False)) else 1.0
            for name in names
        ]
    )


def solve_ik(robot, targets, wxyz, position, cfg, names, locked) -> np.ndarray:
    """Solve movable arms together; fully locked arms contribute no pose cost."""
    mask = mask_for(names, locked)
    if not np.any(mask):
        return np.array(cfg, copy=True)
    active = np.array([any(mask[names.index(f"{arm}_{j}")] for j in joints) for arm, (_, joints, _, _) in ARMS.items()])
    here = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
    wxyz = np.where(active[:, None], wxyz, here[:, :4])
    position = np.where(active[:, None], position, here[:, 4:])
    solution = np.asarray(
        _solve_ik(
            robot,
            jnp.asarray(targets),
            jnp.asarray(wxyz),
            jnp.asarray(position),
            jnp.asarray(mask),
            jnp.asarray(cfg),
            jnp.asarray(active, dtype=jnp.float32),
        )
    )
    return np.where(mask > 0.0, solution, cfg)


def main(start: str, port: int = 8080) -> None:
    import viser
    from viser.extras import ViserUrdf

    urdf, robot, names, targets, cfg = load(start)

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=3, height=3)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(cfg)

    home = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
    ik_targets = {
        arm: server.scene.add_transform_controls(
            f"/ik_target/{arm}", scale=0.05, position=home[i, 4:], wxyz=home[i, :4]
        )
        for i, arm in enumerate(ARMS)
    }
    locks = {}
    for arm, (_, joints, _, initially_locked) in ARMS.items():
        with server.gui.add_folder(arm):
            arm_lock = server.gui.add_checkbox("Lock entire arm", initial_value=initially_locked)
            joint_locks = [server.gui.add_checkbox(f"Lock {j}", initial_value=initially_locked) for j in joints]
            locks.update({f"{arm}_{j}": box for j, box in zip(joints, joint_locks, strict=True)})

            @arm_lock.on_update
            def set_arm_lock(event, boxes=joint_locks):
                for box in boxes:
                    box.value = event.target.value

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        locked = {arm: box.value for arm, box in locks.items()}
        start_time = time.time()
        cfg = solve_ik(
            robot,
            targets,
            np.array([ik_targets[arm].wxyz for arm in ARMS]),
            np.array([ik_targets[arm].position for arm in ARMS]),
            cfg,
            names,
            locked,
        )
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        urdf_vis.update_cfg(cfg)


def check(start: str) -> None:
    """What the page does, scripted: move PSM1 and PSM2 targets, lock one, and print how close each lands."""
    _, robot, names, targets, cfg = load(start)
    home = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
    print(f"{URDF.name}: {robot.links.num_links} links, {len(names)} actuated joints")

    def run(label, locked, shifts, turn_deg=0.0):
        nonlocal cfg
        wxyz, position = home[:, :4].copy(), home[:, 4:].copy()
        for i, arm in enumerate(ARMS):
            if arm in shifts:
                position[i] += shifts[arm]
                turn = jaxlie.SO3(jnp.asarray(wxyz[i])) @ jaxlie.SO3.from_z_radians(np.radians(turn_deg))
                wxyz[i] = np.asarray(turn.wxyz)
        before = cfg.copy()
        spent = []
        for _ in range(10):
            start = time.time()
            cfg = solve_ik(robot, targets, wxyz, position, cfg, names, locked)
            spent.append(time.time() - start)
        fk = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
        print(f"\n{label}  ({1000 * float(np.median(spent[1:])):.1f} ms a solve)")
        for i, arm in enumerate(ARMS):
            have = jaxlie.SE3(jnp.asarray(fk[i]))
            want = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.asarray(wxyz[i])), jnp.asarray(position[i]))
            deg = float(np.degrees(jnp.linalg.norm((have.inverse() @ want).log()[3:])))
            mm = 1000.0 * float(np.linalg.norm(fk[i, 4:] - position[i]))
            idx = [names.index(f"{arm}_{j}") for j in ARMS[arm][1]]
            moved = float(np.abs(cfg[idx] - before[idx]).max())
            state = "locked" if locked[arm] else "solved"
            print(f"  {arm:4s} {state:6s}  {mm:7.3f} mm {deg:6.3f} deg from its target   joints moved {moved:.4f}")

    # FK supplies reachable targets for the ECM's four degrees of freedom too.
    for label, locks in (
        ("All four arms", {}),
        ("Individual locks", {"PSM1_pitch": True, "PSM2_insertion": True, "ECM_yaw": True}),
        ("Locked PSM3", {"PSM3": True}),
    ):
        mask = mask_for(names, locks)
        desired = cfg + mask * np.array([0.002 if "insertion" in n else 0.025 for n in names])
        target = np.asarray(robot.forward_kinematics(jnp.asarray(desired)))[targets]
        solved = solve_ik(robot, targets, target[:, :4], target[:, 4:], cfg, names, locks)
        np.testing.assert_array_equal(solved[mask == 0], cfg[mask == 0])
        actual = np.asarray(robot.forward_kinematics(jnp.asarray(solved)))[targets]
        error = np.linalg.norm(actual[:, 4:] - target[:, 4:], axis=1)
        assert np.max(error) < 0.0005, (label, error)
        rotations = jaxlie.SO3(jnp.asarray(actual[:, :4])).inverse() @ jaxlie.SO3(jnp.asarray(target[:, :4]))
        assert float(jnp.max(jnp.linalg.norm(rotations.log(), axis=-1))) < np.radians(0.1), label
        print(f"{label}: locks exact; maximum position error {1000 * max(error):.4f} mm")
    np.testing.assert_array_equal(
        solve_ik(robot, targets, home[:, :4], home[:, 4:] + 1, cfg, names, dict.fromkeys(ARMS, True)), cfg
    )
    locked = {arm: start for arm, (*_, start) in ARMS.items()}
    run(
        "PSM1 10 mm right and turned 20 deg about its jaws, PSM2 10 mm up",
        locked,
        {"PSM1": (0.0, 0.01, 0.0), "PSM2": (0.0, 0.0, 0.01)},
        turn_deg=20.0,
    )
    run("PSM2 locked: its target 20 mm away is ignored", locked | {"PSM2": True}, {"PSM2": (0.0, -0.02, 0.0)})
    run("Everything locked: nothing moves", dict.fromkeys(ARMS, True), {"PSM1": (0.0, 0.01, 0.0)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic IK, on the dVRK Si")
    parser.add_argument("--check", action="store_true", help="run scripted solves and print, no page")
    parser.add_argument("--start", choices=("middle", "suturebot"), default="middle", help="where the arms start")
    parser.add_argument("--port", type=int, default=8080, help="viser port (next free port if occupied)")
    args = parser.parse_args()
    check(args.start) if args.check else main(args.start, args.port)

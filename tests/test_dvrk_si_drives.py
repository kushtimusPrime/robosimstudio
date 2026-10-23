# SPDX-License-Identifier: Apache-2.0
"""The dVRK Si cart builds, holds still, follows a target and grips, like any robot here.

``test_dvrk_si`` holds the URDF to what SutureBot recorded; this holds the
*embodiment* to what the platform expects of a robot, on the physics a scene
steps it with:

* every arm is solved by PyRoKi over the cart's URDF. Two things kept it off:
  the solver took the URDF's first six actuated joints for every arm, so
  PSM2's IK drove PSM1's joints, and the forward kinematics it calibrates
  against left the parallelogram's mimic followers where the template had
  them, 100 mm from where the arm puts the tool. Either one fails calibration
  and the arm falls back to Newton's IK, which reads the same frozen
  parallelogram;
* at home nothing drifts, on a gravity-compensated arm whose pitch may only
  push 17 N*m;
* a PSM follows a Cartesian target 10 mm away to under a millimetre;
* the jaws close to the squeeze stop with the second jaw mirroring the first,
  and the hand reports itself geared, not as a parallel jaw whose "width" is
  an opening angle;
* a camera riding the ECM resolves to the ECM's roll link.

Not checked: the joint-position mode. It needs arms of one width, and the ECM
has four joints to a PSM's six, so this robot offers the Cartesian modes only
(``test_fix_core_action_and_pairing`` holds that rule).
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""Builds the cart on a GPU and steps it."""

ARMS = ("PSM1", "PSM2", "PSM3", "ECM")


class TestTheCartDrives(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.robots.grippers.si_endoscope import CAMERA_IN_ECM_ROLL

        scene = rss.Scene(cell="bench_075", name="dvrk_drives", physics=rss.MUJOCO_500HZ)
        scene.add(rss.robot("dvrk_si"))
        scene.add(rss.CameraSpec(name="endoscope", width=64, height=48, parent_frame="ECM", xform=CAMERA_IN_ECM_ROLL))
        cls.sim = scene.build(control_mode="ee_pose_abs")
        # FK at build time must already satisfy the mimic equations. Otherwise
        # the studio caches a wrong starting target before physics settles it.
        model = cls.sim.model
        q = cls.sim.model.joint_q.numpy()
        starts = model.joint_q_start.numpy()
        for follower, leader in enumerate(model.joint_mimic_joint.numpy()):
            if leader >= 0:
                offset, multiplier = model.joint_mimic_coeffs.numpy()[follower]
                np.testing.assert_allclose(q[starts[follower]], offset + multiplier * q[starts[leader]], atol=1e-7)
        cls.sim.run(250).sync()
        cls.space = cls.sim.use_control("ee_pose_abs")
        cls.home = cls.space.hold_action(cls.sim.state)

    def _drive(self, action: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            self.sim.step(action)
        self.sim.sync()

    def _tcp(self, arm: str) -> np.ndarray:
        return np.asarray(self.sim.arm(arm).ee_pose(self.sim.state)[0])

    def test_1_every_arm_is_solved_by_pyroki(self) -> None:
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertIsNotNone(self.sim.arm(arm)._pyroki_solver())
                self.assertEqual("pyroki", self.sim.arm(arm).solver_name)

    def test_2_it_holds_its_home(self) -> None:
        before = {arm: self._tcp(arm) for arm in ARMS}
        self._drive(self.home, 250)
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertLess(1000.0 * np.linalg.norm(self._tcp(arm) - before[arm]), 0.5)
        self.assertTrue(self.sim.state_is_finite())

    def test_3_a_psm_follows_a_target(self) -> None:
        """Sideways, so the jaws stay clear of the pad they start 8 mm above."""
        slot = ARMS.index("PSM2") * self.space.per_arm
        action = self.home.copy()
        action[slot : slot + 3] += (-0.01, 0.01, 0.0)
        self._drive(action, 500)
        self.assertLess(1000.0 * np.linalg.norm(self._tcp("PSM2") - action[slot : slot + 3]), 1.0)
        self._drive(self.home, 250)

    def test_4_the_jaws_close_to_the_stop_and_mirror(self) -> None:
        from robosimstudio.robots.grippers.endowrist import JAW_CLOSED

        grip = ARMS.index("PSM1") * self.space.per_arm + self.space.per_arm - 1
        action = self.home.copy()
        action[grip] = 1.0
        self._drive(action, 250)
        index = self.sim.scene.embodiment.arms["PSM1"]
        q = self.sim.state.joint_q.numpy()
        jaw_1, jaw_2 = float(q[index.gripper_coords[0]]), float(q[index.gripper_follower_coords[0]])
        self.assertAlmostEqual(JAW_CLOSED, jaw_1, delta=0.01)
        self.assertAlmostEqual(-jaw_1, jaw_2, delta=0.01)
        self.assertEqual("geared", self.sim.gripper("PSM1").kind)
        self._drive(self.home, 250)

    def test_5_the_endoscope_rides_the_ecm(self) -> None:
        rig = {camera.name: camera for camera in self.sim._resolve_cameras()}
        body = self.sim.arm("ECM").ee_body
        self.assertEqual(body, rig["endoscope"].parent_body)
        self.assertTrue(str(self.sim.model.body_label[body]).endswith("ECM_roll_link"))


if __name__ == "__main__":
    unittest.main()

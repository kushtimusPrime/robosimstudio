# SPDX-License-Identifier: Apache-2.0
"""The dVRK Si cart is SutureBot's rig, and replays what SutureBot recorded.

``dvrk_si.urdf`` is generated (``scripts/dvrk_si_urdf.py``) from JHU's
``dvrk_model`` and re-calibrated onto the dVRK controller's own kinematics,
because SutureBot's poses are the controller's and ``dvrk_model`` disagrees
with it: approximate RCM closure, a shaft outside its cannula bore, a tool tip
out along the jaws, and SUJ chains off against the ECM. So
this holds the claims the embodiment's docstring makes, against the file:

* every file it names is shipped, and nothing else is;
* each arm's joints are its own, prefixed, with the parallelogram and the jaws
  mimic followers, and every setup joint frozen;
* each working PSM's RCM sits exactly where the controller logged it in the
  camera frame, so the placement the generator solved is the one shipped;
* sampled frames of SutureBot's selected home session, replayed by joint value,
  put both tool tips where the robot recorded them in the camera frame.

Static: URDF forward kinematics in numpy, no scene, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path

import numpy as np

from robosimstudio import paths as asset_paths
from robosimstudio.robots.registry import get_embodiment, get_gripper

TIER = "fast"
"""No GPU, no scene. Gates every commit."""

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "assets" / "embodiments" / "dvrk_si"
URDF = VENDORED / "dvrk_si.urdf"

SETUP = json.loads((VENDORED / "setups.json").read_text())["home"]
FRAMES = tuple(
    (f["joints"]["PSM1"], f["joints"]["PSM2"], f["joints"]["ECM"], f["poses"]["PSM1"], f["poses"]["PSM2"])
    for f in SETUP["frames"]
)
REPLAY_MM = 0.2
"""Rounded controller DH and measured-vs-reported state allow 0.2 mm."""

REPLAY_DEG = 0.2


def _generator():
    spec = importlib.util.spec_from_file_location("dvrk_si_urdf", REPO / "scripts" / "dvrk_si_urdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix(pose) -> np.ndarray:
    from robosimstudio.core.math import quat_to_mat

    t = np.eye(4)
    t[:3, :3] = quat_to_mat(np.asarray(pose[3:7], dtype=np.float64))
    t[:3, 3] = pose[:3]
    return t


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


class TestTheCartIsShipped(unittest.TestCase):
    def test_the_spec_names_the_vendored_urdf_for_both_import_and_ik(self) -> None:
        spec = get_embodiment("dvrk_si")
        self.assertEqual("embodiments/dvrk_si/dvrk_si.urdf", spec.asset.path)
        self.assertEqual(spec.asset.path, spec.urdf)
        self.assertEqual(URDF, asset_paths.resolve_asset(spec.asset))

    def test_exactly_the_named_meshes_are_shipped(self) -> None:
        named = set(re.findall(r'filename="([^"]+)"', URDF.read_text()))
        shipped = {str(p.relative_to(VENDORED)) for p in (VENDORED / "meshes").rglob("*") if p.is_file()}
        self.assertEqual(sorted(named), sorted(shipped), "a mesh is missing, or one nothing names came along")
        size = sum((VENDORED / name).stat().st_size for name in named)
        self.assertLess(size, 10e6, "the decimation in scripts/dvrk_si_urdf.py stopped applying")

    def test_it_says_where_it_came_from(self) -> None:
        self.assertIn("MIT License", (VENDORED / "LICENSE").read_text())
        note = (VENDORED / "README.md").read_text()
        for word in ("dvrk_model", "sawIntuitiveResearchKit", "SutureBot", "scripts/dvrk_si_urdf.py"):
            self.assertIn(word, note)


class TestEachArmIsItsOwn(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import yourdfpy

        cls.urdf = yourdfpy.URDF.load(
            str(URDF), load_meshes=False, build_collision_scene_graph=False, load_collision_meshes=False
        )

    def test_the_actuated_joints_are_the_controllers(self) -> None:
        psm = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw", "jaw_1")
        expected = [f"PSM{i}_{j}" for i in (1, 2, 3) for j in psm] + [
            f"ECM_{j}" for j in ("yaw", "pitch", "insertion", "roll")
        ]
        self.assertEqual(sorted(expected), sorted(self.urdf.actuated_joint_names))

    def test_the_parallelogram_and_the_jaws_follow(self) -> None:
        for arm in ("PSM1", "PSM2", "PSM3", "ECM"):
            for follower, leader, multiplier in (("pitch_2", "pitch", -1.0), ("pitch_3", "pitch", 1.0)):
                with self.subTest(arm=arm, joint=follower):
                    mimic = self.urdf.joint_map[f"{arm}_{follower}"].mimic
                    self.assertEqual(
                        (f"{arm}_{leader}", multiplier, 0.0), (mimic.joint, mimic.multiplier, mimic.offset)
                    )
        for arm in ("PSM1", "PSM2", "PSM3"):
            mimic = self.urdf.joint_map[f"{arm}_jaw_2"].mimic
            self.assertEqual((f"{arm}_jaw_1", -1.0), (mimic.joint, mimic.multiplier))

    def test_every_setup_joint_is_frozen(self) -> None:
        suj = [name for name in self.urdf.joint_map if re.fullmatch(r"SUJ_(PSM[123]|ECM)_J\d", name)]
        self.assertEqual(17, len(suj), "PSM1, PSM2 and the ECM have four setup joints, PSM3 five")
        self.assertTrue(all(self.urdf.joint_map[name].type == "fixed" for name in suj))

    def test_the_limits_are_the_controllers_not_the_xacros(self) -> None:
        pitch = self.urdf.joint_map["PSM1_pitch"].limit
        self.assertEqual((-1.265, 1.301, 17.0845), (pitch.lower, pitch.upper, pitch.effort))
        for name, joint in self.urdf.joint_map.items():
            if joint.type in ("revolute", "prismatic"):
                with self.subTest(joint=name):
                    self.assertGreater(joint.limit.effort, 0.0, "MuJoCo refuses a zero force range")
                    self.assertGreater(joint.limit.velocity, 0.0, "PyRoKi refuses a joint without one")

    def test_every_moving_link_has_mass(self) -> None:
        moving = {j.child for j in self.urdf.joint_map.values() if j.type != "fixed"}
        for link in sorted(moving):
            with self.subTest(link=link):
                inertial = self.urdf.link_map[link].inertial
                self.assertIsNotNone(inertial, "MuJoCo refuses a moving body with no mass")
                self.assertGreater(inertial.mass, 0.0)

    def test_every_jaw_collides(self) -> None:
        for arm in ("PSM1", "PSM2", "PSM3"):
            for side in ("jaw_1", "jaw_2"):
                with self.subTest(arm=arm, jaw=side):
                    self.assertTrue(self.urdf.link_map[f"{arm}_{side}_link"].collisions)


class TestItIsSutureBotsRig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import yourdfpy

        cls.urdf = yourdfpy.URDF.load(
            str(URDF), load_meshes=False, build_collision_scene_graph=False, load_collision_meshes=False
        )
        cls.names = list(cls.urdf.actuated_joint_names)
        cls.generator = _generator()

    def _pose(self, psm1=None, psm2=None, ecm=None) -> None:
        from robosimstudio.robots.embodiments.dvrk_si import ECM_HOME, PSM_JOINTS

        q = np.zeros(len(self.names))
        for arm, values, joints in (
            ("PSM1", psm1, PSM_JOINTS),
            ("PSM2", psm2, PSM_JOINTS),
            ("ECM", ecm or ECM_HOME, ("yaw", "pitch", "insertion", "roll")),
        ):
            for joint, value in zip(joints, values or (), strict=False):
                q[self.names.index(f"{arm}_{joint}")] = value
        self.urdf.update_cfg(q)

    def _in_camera(self, link: str) -> np.ndarray:
        return np.linalg.inv(self.urdf.get_transform("ECM_camera_link")) @ self.urdf.get_transform(link)

    def test_each_rcm_is_where_the_controller_logged_it(self) -> None:
        self._pose()
        for arm, logged in self.generator.RCM_IN_CAMERA.items():
            with self.subTest(arm=arm):
                got, want = self._in_camera(f"{arm}_RCM_link"), _matrix(logged)
                self.assertLess(np.linalg.norm(got[:3, 3] - want[:3, 3]), 1.0e-5)
                self.assertLess(_angle_deg(got, want), 1.0e-3)

    def test_home_uses_a_recorded_posture_clear_of_working_joint_limits(self) -> None:
        from robosimstudio.robots.embodiments import dvrk_si

        for arm in ("PSM1", "PSM2", "ECM"):
            np.testing.assert_allclose(getattr(dvrk_si, f"{arm}_HOME"), SETUP["joints"][arm], atol=1e-9)
        for arm in ("PSM1", "PSM2"):
            for name, value in zip(dvrk_si.PSM_JOINTS, getattr(dvrk_si, f"{arm}_HOME"), strict=True):
                limit = self.urdf.joint_map[f"{arm}_{name}"].limit
                self.assertGreater(min(value - limit.lower, limit.upper - value) / (limit.upper - limit.lower), 0.20)

    def test_shaft_and_cannula_share_the_rcm_throughout_pitch_range(self) -> None:
        for arm in ("PSM1", "PSM2", "PSM3", "ECM"):
            limit = self.urdf.joint_map[f"{arm}_pitch"].limit
            for pitch in np.linspace(limit.lower, limit.upper, 9):
                self.urdf.update_cfg(
                    {f"{arm}_yaw": 0.4, f"{arm}_pitch": pitch, f"{arm}_insertion": 0.15, f"{arm}_roll": 0.7}
                )
                rcm = self.urdf.get_transform(f"{arm}_RCM_link")[:3, 3]
                shaft = self.urdf.get_transform(f"{arm}_roll_link")
                axis = shaft[:3, 2]
                self.assertLess(np.linalg.norm(np.cross(rcm - shaft[:3, 3], axis)), 1e-7)
                cannula = self.urdf.get_transform(f"{arm}_pitch_3_link")
                bore = cannula @ np.array([0.10649, 0.0, -0.15, 1.0])
                self.assertLess(np.linalg.norm(np.cross(bore[:3] - shaft[:3, 3], axis)), 1e-7)

    def test_a_recorded_session_replays_by_joint_value(self) -> None:
        for frame, (psm1, psm2, ecm, pose1, pose2) in enumerate(FRAMES):
            self._pose(psm1, psm2, ecm)
            for arm, logged in (("PSM1", pose1), ("PSM2", pose2)):
                with self.subTest(frame=frame, arm=arm):
                    got, want = self._in_camera(f"{arm}_tool_tip_link"), _matrix(logged)
                    self.assertLess(1000.0 * np.linalg.norm(got[:3, 3] - want[:3, 3]), REPLAY_MM)
                    self.assertLess(_angle_deg(got, want), REPLAY_DEG)

    def test_the_endoscope_hand_is_the_camera_link(self) -> None:
        """The ECM's tool centre, posed in its roll link, is ``ECM_camera_link`` -- what a riding camera sees from."""
        self._pose()
        hand = get_gripper("si_endoscope")
        got = np.linalg.inv(self.urdf.get_transform("ECM_roll_link")) @ self.urdf.get_transform("ECM_camera_link")
        want = _matrix((*hand.tcp_offset, *hand.tcp_offset_rot))
        self.assertLess(np.linalg.norm(got[:3, 3] - want[:3, 3]), 1.0e-6)
        self.assertLess(_angle_deg(got, want), 1.0e-4)


class TestTheInstruments(unittest.TestCase):
    def test_each_tool_centre_is_between_its_blades_on_the_dvrks_tool_frame(self) -> None:
        from robosimstudio.robots.grippers.endowrist import DVRK_TOOL_TIP_ROT

        for name in ("endowrist_large_needle_driver", "endowrist_debakey_forceps"):
            with self.subTest(hand=name):
                hand = get_gripper(name)
                self.assertTrue(hand.geared, "two jaws on a pivot close along an arc")
                self.assertEqual(("{arm}_jaw_1",), hand.finger_names)
                self.assertEqual(("{arm}_jaw_2",), hand.follower_names)
                self.assertEqual(DVRK_TOOL_TIP_ROT, hand.tcp_offset_rot)
                self.assertGreater(hand.tcp_offset[1], 0.0)
                self.assertGreater(hand.tip_reach, 0.0)
                self.assertGreater(hand.span, 0.0)

    def test_the_tool_frames_rotation_is_the_urdfs(self) -> None:
        import yourdfpy

        from robosimstudio.robots.grippers.endowrist import DVRK_TOOL_TIP_ROT

        urdf = yourdfpy.URDF.load(
            str(URDF), load_meshes=False, build_collision_scene_graph=False, load_collision_meshes=False
        )
        got = np.linalg.inv(urdf.get_transform("PSM1_wrist_yaw_link")) @ urdf.get_transform("PSM1_tool_tip_link")
        self.assertLess(np.linalg.norm(got[:3, 3]), 1.0e-9, "the controller's tool tip is on the yaw pivot")
        self.assertLess(_angle_deg(got, _matrix((0.0, 0.0, 0.0, *DVRK_TOOL_TIP_ROT))), 1.0e-4)


if __name__ == "__main__":
    unittest.main()

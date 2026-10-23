# SPDX-License-Identifier: Apache-2.0
"""The dVRK Si patient cart, set up as SutureBot's rig: three PSMs and the ECM.

The da Vinci Research Kit on Si hardware -- the cart, its setup joints, three
patient-side manipulators (PSMs) carrying EndoWrist instruments and the
endoscope camera manipulator (ECM) -- from JHU's ``dvrk_model``, made one URDF
by ``scripts/dvrk_si_urdf.py`` (``assets/embodiments/dvrk_si/README.md`` says
what it changed and why).

**Set up the way SutureBot was.** The setup joints are frozen where one
SutureBot session had them, PSM1 carries the large needle driver on the right,
PSM2 the forceps on the left, PSM3 is parked out of the way at the back of the
cart, and the ECM looks down on the work from where that session's camera
was. A PSM's remote centre of motion (RCM) -- the point its shaft pivots
through, where the trocar is -- is placed from the camera-frame pose the
controller logged, so a pose recorded in SutureBot's camera frame lands on the
same place relative to this ECM's camera.

**Joint coordinates are the dVRK's.** Each arm's joints are its controller's
``measured_js``, in the controller's order and zero: ``yaw, pitch, insertion,
roll, wrist_pitch, wrist_yaw`` on a PSM and ``yaw, pitch, insertion, roll`` on
the ECM. The parallelogram closes exactly on the controller's RCM; recorded camera-frame
tool tips are checked to 0.2 mm. Home is tissue 7 knot-tying recovery, row 372,
selected for at least 20.8% joint-range clearance on both working arms.
The setup joints and camera are from that same session.

**The arms are parallelograms.** ``pitch_2`` and ``pitch_3`` are mimic
followers of ``pitch``, which is what keeps the shaft through the RCM: no IK
and no servo has to hold it there. The arm is gravity-compensated, as the
real one is balanced; without it the 17 N*m the controller allows the pitch
joint could not hold the arm up.

The work surface is estimated: 460 mm above the cart floor, below the new
home's tools with clearance for their open jaws. The old 488 mm estimate put
PSM2's 12 mm blades into the table at the selected posture. On a 750 mm bench
the cart still stands 290 mm above the floor; the real table height is unknown.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec

from ..grippers.si_endoscope import CAMERA_IN_ECM_ROLL
from ..registry import register_embodiment

__all__ = [
    "CAMERA_IN_ECM_ROLL",
    "DVRK_SI",
    "ECM_HOME",
    "PSM1_HOME",
    "PSM2_HOME",
    "PSM3_HOME",
    "PSM_JOINTS",
    "WORK_SURFACE_IN_CART",
]

PSM_JOINTS = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
"""A PSM's driven joints in the controller's order, the label suffix after the arm's name."""
ECM_JOINTS = ("yaw", "pitch", "insertion", "roll")

PSM1_HOME = (0.584666887, -0.685049652, 0.211238997, 0.444768188, -0.390707617, -0.691871804)
"""``psm1_js`` at row 372 of tissue_7/3_knot_tying_recovery/20250502-204132-813971_recovery [rad, m]: a needle throw starting."""
PSM2_HOME = (-0.675466007, -0.731069076, 0.195405709, -1.150317724, -0.528781581, 0.063573174)
"""``psm2_js``, the same recorded home frame."""
PSM3_HOME = (-0.166780901, -0.655480819, 0, 0.007732871, 0.003184123, 0.026496456)
"""``psm3_js``: parked; negative insertion encoder noise is clamped to its zero stop."""
ECM_HOME = (-0.1075097, -0.493382084, 0.192577792, 0.051016869)
"""``ecm_js``: where the camera was while the PSM bases were measured from it."""

WORK_SURFACE_IN_CART = (0.702, 0.0094, 0.46)
"""Estimated work surface in cart coordinates [m], centred under the new home.

Home pivots are at z=0.4955 and 0.4896 m. A 0.460 m surface clears the open
DeBakey blades; the previous 0.488 m estimate intersected them. This is a
scene-mount assumption, not a measured SutureBot table height.
"""


def _psm(name: str, gripper: str, home: tuple[float, ...]) -> ArmSpec:
    return ArmSpec(
        name=name,
        joint_names=tuple(f"{name}_{joint}" for joint in PSM_JOINTS),
        ee_body_name=f"{name}_wrist_yaw_link",
        gripper=gripper,
        home_q=home,
        # Inspection starts with empty jaws half open. Recorded jaw values
        # are applied explicitly by the validation replay.
        home_gripper_q=(0.35,),
    )


_PSM_ACTUATION = ActuationSpec(
    # yaw and pitch swing a 9 kg arm; insertion is a 100 g carriage on a 0.2 m/s
    # stage; roll and the wrist turn milligrams, and their armature is what lets
    # a 2 ms step hold them at all (a 65 mg wrist link alone rings at kHz).
    target_ke=200.0,
    target_kd=10.0,
    target_ke_per_dof=(200.0, 200.0, 2000.0, 0.5, 0.5, 0.5),
    target_kd_per_dof=(10.0, 10.0, 60.0, 0.01, 0.01, 0.01),
    # Preserve each joint's URDF effort limit: ECM pitch/insertion/roll
    # limits differ from the PSM's, despite sharing the gain table.
    armature=1.0e-4,
)

DVRK_SI = register_embodiment(
    EmbodimentSpec(
        name="dvrk_si",
        asset=AssetSpec(path="embodiments/dvrk_si/dvrk_si.urdf"),
        urdf="embodiments/dvrk_si/dvrk_si.urdf",
        base_offset=(-WORK_SURFACE_IN_CART[0], -WORK_SURFACE_IN_CART[1], -WORK_SURFACE_IN_CART[2], 0.0, 0.0, 0.0, 1.0),
        # The frozen setup joints and every fixed frame fold into the cart and
        # the links they ride, which is what makes this 36 bodies rather than 69.
        collapse_fixed_joints=True,
        gravcomp=True,
        hull_colliders=True,
        # The source cart colliders overlap between arms by design. Until
        # there is a validated inter-arm collision model, those overlaps must
        # not push the ECM away from a stationary commanded pose.
        cross_arm_collisions=False,
        arms=(
            _psm("PSM1", "endowrist_large_needle_driver", PSM1_HOME),
            _psm("PSM2", "endowrist_debakey_forceps", PSM2_HOME),
            _psm("PSM3", "endowrist_large_needle_driver", PSM3_HOME),
            ArmSpec(
                name="ECM",
                joint_names=tuple(f"ECM_{joint}" for joint in ECM_JOINTS),
                ee_body_name="ECM_roll_link",
                gripper="si_endoscope",
                home_q=ECM_HOME,
            ),
        ),
        # One table for every arm, the ECM included, which reads its first four
        # entries: yaw, pitch and insertion are the same kind of stage, and its
        # roll turns the scope about its own axis, which gravity does not load.
        arm_actuation=_PSM_ACTUATION,
        ik_iterations=8,
    )
)

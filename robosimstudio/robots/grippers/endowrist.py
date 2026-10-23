# SPDX-License-Identifier: Apache-2.0
"""The da Vinci Si EndoWrist instruments a dVRK PSM carries: two revolute jaws on a wrist.

Not a parallel jaw. Each jaw is a blade a centimetre or two long, pivoting on the
wrist-yaw axis, so the gap between the tips is a chord of the opening angle
rather than a distance the joint travels -- which is why both hands here say
``linear_travel=False``. The dVRK reports and commands the *opening* angle,
the sum of the two jaws' angles, from -20 degrees (the jaws squeezed past
closed, which is how the instrument grips) to 80. ``scripts/dvrk_si_urdf.py``
drives ``jaw_1`` and mirrors ``jaw_2`` on it, so the opening is ``2 * jaw_1``
and full open is ``jaw_1 = 40`` degrees.

Closed is the joint's lower stop, -10 degrees a jaw, rather than 0: a servo
asked for 0 stops where the blades meet and presses on nothing, and one asked
for the stop presses an object between them with the servo's force, which is
what the instrument's controller does when the operator squeezes.

Indexed in ``dvrk_si.urdf``, labels prefixed by the arm (``PSM1_jaw_1``), so
``{arm}`` is the PSM's own name.
"""

from __future__ import annotations

import math

from robosimstudio.specs import ActuationSpec, GripperSpec

from ..registry import register_gripper

__all__ = [
    "CADIERE_FORCEPS",
    "DEBAKEY_FORCEPS",
    "DVRK_TOOL_TIP_ROT",
    "JAW_CLOSED",
    "JAW_OPEN",
    "LARGE_NEEDLE_DRIVER",
    "tcp_depth",
]

JAW_OPEN = math.radians(40.0)
"""One jaw's angle at full open [rad]: half the 80 degree opening the tool file allows."""
JAW_CLOSED = math.radians(-10.0)
"""One jaw's angle squeezed shut [rad]: half the tool file's -20 degree opening."""

DVRK_TOOL_TIP_ROT = (-0.5, 0.5, 0.5, 0.5)
"""The dVRK's tool frame in the wrist-yaw link, xyzw: +z out along the jaws, +x the yaw axis.

Its origin is the yaw pivot itself, which is where the controller's tool frame
sits and what every ``measured_cp`` and setpoint SutureBot logged refers to
(``tool_tip_link`` in ``dvrk_si.urdf``). The hand's tool centre is that frame
slid :func:`tcp_depth` out along +z, to the middle of the blades, so a pose
recorded on the robot is ``tcp = recorded @ translate(0, 0, depth)``.
"""

_ACTUATION = ActuationSpec(target_ke=2.0, target_kd=0.02, effort_limit=0.16)
"""The jaw servo: 0.16 N*m is the tool file's ``ftmax`` for the jaw, so a grip presses no harder than the robot's."""


def _endowrist(name: str, *, blade: tuple[float, float], thickness: float) -> GripperSpec:
    """One instrument, from its blades' extent along the jaw from the pivot [m] and their thickness.

    Measured off the jaw colliders in ``dvrk_si.urdf`` (the needle driver's
    are ``dvrk_model``'s own boxes, the Cadiere's are the generator's box over
    its mesh). The tool centre is the middle of the blade, the reach is the
    rest of it, and the span is the chord between the two tips at full open
    less one blade's thickness -- the gap something has to pass through.
    """
    near, far = blade
    depth = (near + far) / 2.0
    return GripperSpec(
        name=name,
        finger_names=("{arm}_jaw_1",),
        follower_names=("{arm}_jaw_2",),
        finger_open_q=(JAW_OPEN,),
        finger_closed_q=(JAW_CLOSED,),
        linear_travel=False,
        open_width=JAW_OPEN,
        span=2.0 * far * math.sin(JAW_OPEN) - thickness,
        tcp_body="{arm}_wrist_yaw_link",
        tcp_offset=(0.0, depth, 0.0),
        tcp_offset_rot=DVRK_TOOL_TIP_ROT,
        # In the wrist-yaw link: the blades run out along +y and turn about +z,
        # so they close along x.
        approach_axis=(0.0, 1.0, 0.0),
        close_axis=(1.0, 0.0, 0.0),
        tip_reach=far - depth,
        finger_bodies=(("{arm}_jaw_1_link",), ("{arm}_jaw_2_link",)),
        actuation=_ACTUATION,
    )


def tcp_depth(gripper: GripperSpec) -> float:
    """How far the hand's tool centre sits out along the jaws from the dVRK's tool tip [m]."""
    return float(gripper.tcp_offset[1])


LARGE_NEEDLE_DRIVER = register_gripper(
    _endowrist("endowrist_large_needle_driver", blade=(-0.0024130, 0.0096520), thickness=0.003)
)
"""Large needle driver, 420006: SutureBot's right hand, on PSM1."""

CADIERE_FORCEPS = register_gripper(
    _endowrist("endowrist_cadiere_forceps", blade=(-0.0021525, 0.0210368), thickness=0.005567)
)
"""Cadiere forceps, 420049, standing in for the DeBakey forceps (420036) SutureBot carries on PSM2.

``dvrk_model`` has no DeBakey tip; the Cadiere is the same Si instrument up to
the jaws (housing, 4670 roll, 0091 wrist), so the arm's kinematics are the
DeBakey's and only the blades' shape is not.
"""


DEBAKEY_FORCEPS = register_gripper(_endowrist("endowrist_debakey_forceps", blade=(0.003, 0.015), thickness=0.0012))
"""420036 on SutureBot's PSM2; generated 12 mm blade approximation, not OEM CAD."""

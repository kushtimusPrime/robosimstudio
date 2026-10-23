# SPDX-License-Identifier: Apache-2.0
"""The Si straight (0 degree) endoscope on a dVRK ECM, as a hand with no fingers.

What the ECM carries is a camera, so its tool centre is the camera's frame and
a gizmo dragged on the ECM moves the view. The frame is the controller's ECM
frame -- SutureBot's camera frame, ``ECM_camera_link`` in ``dvrk_si.urdf`` --
with +z out of the scope. This is a kinematic frame, not a calibrated optical
frame. A camera using it directly is an inspection view. Image replay needs
camera intrinsics and an optical registration: the published dVRK-to-OpenCV
conversion alone includes a half turn around z, and setup-joint errors can
require an additional arm-to-camera registration.
"""

from __future__ import annotations

from robosimstudio.specs import GripperSpec

from ..registry import register_gripper

__all__ = ["CAMERA_IN_ECM_ROLL", "SI_ENDOSCOPE"]

CAMERA_IN_ECM_ROLL = (0.0, 0.0, 0.3829, 0.0, 0.0, 0.70710678, 0.70710678)
"""``ECM_camera_link`` in the ECM's roll link [m], xyzw.

The roll link is the body the endoscope's fixed links collapse into on import,
so this is the pose a camera riding the ECM states: 382.9 mm down the scope,
the controller's tip, and a quarter turn about it.
"""

SI_ENDOSCOPE = register_gripper(
    GripperSpec(
        name="si_endoscope",
        tcp_body="{arm}_roll_link",
        tcp_offset=CAMERA_IN_ECM_ROLL[:3],
        tcp_offset_rot=CAMERA_IN_ECM_ROLL[3:],
        approach_axis=(0.0, 0.0, 1.0),
    )
)

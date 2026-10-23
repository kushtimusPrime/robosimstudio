# SPDX-License-Identifier: Apache-2.0
"""Write ``assets/embodiments/dvrk_si/``: the dVRK Si patient cart as one URDF.

JHU's ``dvrk_model`` describes the Si cart as separate robots -- the cart and its
four setup-joint (SUJ) chains, three PSMs, the ECM -- each its own
``robot_description`` with the same unprefixed joint names (``yaw``, ``pitch``,
``insertion`` ...). This expands them into one tree and makes it something a
simulator can step:

* every arm joint is prefixed with its arm (``PSM1_yaw``), mimic references too,
  so the three arms stop colliding by name;
* the SUJ joints are **frozen** at :data:`SUJ_Q`, folded into fixed joints. On
  the real cart they are braked during a task, and the IK solves the arm below
  the remote centre of motion (RCM) and nothing above it;
* the instrument's phantom ``jaw`` link (a 5 mm visual box on a massless body)
  is dropped: ``jaw_1`` is driven and ``jaw_2`` mirrors it, so the dVRK's jaw
  angle is ``2 * jaw_1``;
* joint limits and torque limits are the controller's own
  (``sawIntuitiveResearchKit`` ``share/kinematic/PSM_Si.json``, ``ECM_Si.json``,
  ``share/tool/*_4200xx.json``), replacing the xacro's +-pi placeholders and the
  zero efforts MuJoCo refuses;
* jaws that carry no collider get a box over their mesh's bounds;
* meshes are copied beside the URDF under ``meshes/``, any over
  :data:`DECIMATE_BYTES` decimated (the Si sterile-adapter housing is 30 MB).

Run from the repository root with a ``dvrk_model`` checkout::

    git clone https://github.com/jhu-dvrk/dvrk_model /tmp/dvrk_model
    git -C /tmp/dvrk_model checkout 90472a933995acaa88d84eb5c58aeee4cd17e9f3
    uv run --with xacro --with fast-simplification python scripts/dvrk_si_urdf.py /tmp/dvrk_model
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "embodiments" / "dvrk_si"
DVRK_MODEL_COMMIT = "90472a933995acaa88d84eb5c58aeee4cd17e9f3"

INSTRUMENTS = {"PSM1": "420006", "PSM2": "420036", "PSM3": "420006"}
"""Which instrument each PSM carries, by the last six digits of its model number.

SutureBot's rig: large needle driver (420006) on the right, DeBakey (420036)
on the left. The upstream DeBakey placeholder is replaced by a generated
12 mm straight-jaw approximation; see :func:`debakey_jaws`.
"""

SETUPS = json.loads((OUT / "setups.json").read_text())
"""Recorded source values; home is tissue 7 row 372, calibration is tissue 10 row 0."""
SESSION = SETUPS["home"]["session"]
SUJ_Q = {arm: tuple(q) for arm, q in SETUPS["home"]["suj"].items()}
ECM_Q = tuple(SETUPS["home"]["joints"]["ECM"])
RCM_IN_CAMERA = {arm: tuple(p) for arm, p in SETUPS["home"]["rcm_in_camera"].items()}


PITCH_ZERO = -0.1615
"""Where the controller's pitch zero sits in ``dvrk_model``'s pitch coordinate [rad], PSM and ECM alike.

Fitted: the URDF's arm at ``q + (0, PITCH_ZERO, 0, ...)`` matches the
controller's DH (``PSM_Si.json`` with the 420006 tool file, ``ECM_Si.json``) at
``q`` to 3.0 mm and 0.06 degrees over 60 random postures on the PSM, -0.1619 rad
fitted; 2.9 mm and 0.13 degrees on the ECM, -0.1608. Folded into the three
parallelogram joints' origins, so this URDF's joint coordinates *are* the
dVRK's and a recorded ``psm1_js`` replays as it stands.
"""

CAMERA_IN_ECM_TIP = (0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5)
"""The controller's ECM frame -- SutureBot's camera frame -- in ``dvrk_model``'s ``ECM_tip_link``, xyzw.

A permutation of axes, fitted alongside :data:`PITCH_ZERO`: the fitted rotation
is this one to within 0.1 degrees (the camera's x, y, z are ``ECM_tip_link``'s
y, z, x), so the exact permutation is stated rather than the fit. Written into
the URDF as ``ECM_camera_link``.
"""

LIMITS: dict[str, tuple[float, float, float, float]] = {
    # name: (lower, upper, effort [N*m or N], velocity [rad/s or m/s]). Range and
    # effort: sawIntuitiveResearchKit @ 7435458, share/kinematic/PSM_Si.json,
    # ECM_Si.json and share/tool/LARGE_NEEDLE_DRIVER_420006.json (the DeBakey's
    # file states the same). Velocity: the joint trajectory's v_max that
    # mtsIntuitiveResearchKitPSM.cpp and mtsIntuitiveResearchKitECM.cpp set.
    "PSM_yaw": (-2.964, 2.964, 17.0845, math.radians(90.0)),
    "PSM_pitch": (-1.265, 1.301, 17.0845, math.radians(90.0)),
    "PSM_insertion": (0.0, 0.29, 40.9063, 0.2),
    "PSM_roll": (-4.53786, 4.53786, 0.33, math.radians(1080.0)),
    "PSM_wrist_pitch": (-1.39626, 1.39626, 0.25, math.radians(1080.0)),
    "PSM_wrist_yaw": (-1.39626, 1.39626, 0.20, math.radians(1080.0)),
    # The tool file's jaw is the *opening* angle, -20 to 80 degrees; jaw_1 is half of it.
    "PSM_jaw_1": (-0.349066 / 2, 1.39626 / 2, 0.16, math.radians(540.0)),
    "ECM_yaw": (-2.964, 2.964, 17.0845, math.radians(30.0)),
    "ECM_pitch": (-1.10, 1.04, 34.0845, math.radians(30.0)),
    "ECM_insertion": (0.001, 0.259, 70.900, 0.06),
    "ECM_roll": (-1.54, 1.54, 6.81923, math.radians(30.0)),
}
"""Joint range, torque limit and speed per arm kind and joint, from the controller's own configs."""

FOLLOWER_EFFORT = 1000.0
"""Effort on a mimic follower, which an equality carries rather than a motor [N*m]."""

STEEL = 7900.0
"""Density for the instrument's distal links, stainless steel [kg/m^3]: none of them states a mass."""
HOUSING_MASS = 0.1
"""What rides the PSM's insertion stage, the sterile adapter and instrument housing [kg].

``PSM_Si.json``'s insertion joint: ``"mass": 0.1, // about 100 grams``, the
number the controller compensates gravity with.
"""
TOKEN_MASS = 1.0e-3
"""For a moving link that carries nothing at all: the ECM's insertion stage, a frame between two joints [kg]."""

DECIMATE_BYTES = 1_500_000
DECIMATE_FACES = 20_000

TOP = """<?xml version="1.0" ?>
<robot name="dvrk_si" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="endoscope" default="Si_straight"/>
  <xacro:include filename="{root}/urdf/Si/SUJ.urdf.xacro"/>
  <xacro:include filename="{root}/urdf/Si/PSM_base.urdf.xacro"/>
  <xacro:include filename="{root}/urdf/common/instrument.urdf.xacro"/>
  <xacro:include filename="{root}/urdf/Si/ECM_base.urdf.xacro"/>
  <xacro:include filename="{root}/urdf/common/endoscope.urdf.xacro"/>
  <xacro:dvrk_materials/>
{psms}
  <xacro:ecm_base prefix="ECM_" parent_link="SUJ_ECM_RCM"/>
  <xacro:endoscope prefix="ECM_" parent_link="ECM_adaptor_link"/>
</robot>
"""
PSM = """  <xacro:base prefix="{arm}_" parent_link="SUJ_{arm}_RCM"/>
  <xacro:instrument prefix="{arm}_" instrument="{instrument}" is_virtual="false"/>"""


def expand(dvrk_model: Path, work: Path) -> ET.Element:
    """Run xacro over a copy of ``dvrk_model`` whose ``$(find dvrk_model)`` is a path."""
    root = work / "dvrk_model"
    shutil.copytree(dvrk_model / "urdf", root / "urdf")
    for path in (root / "urdf").rglob("*.xacro"):
        path.write_text(path.read_text().replace("$(find dvrk_model)", str(root)))
    psms = "\n".join(PSM.format(arm=arm, instrument=inst) for arm, inst in INSTRUMENTS.items())
    top = root / "urdf" / "dvrk_si.urdf.xacro"
    top.write_text(TOP.format(root=root, psms=psms))
    done = subprocess.run(
        [sys.executable, "-c", "import xacro; xacro.main()", str(top)], capture_output=True, text=True, check=False
    )
    if done.returncode:
        raise SystemExit(f"xacro failed:\n{done.stderr}")
    return ET.fromstring(done.stdout)


def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _matrix_rpy(m: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(-max(-1.0, min(1.0, m[2, 0])))
    if abs(m[2, 0]) < 1.0 - 1.0e-9:
        return math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])
    return 0.0, pitch, math.atan2(-m[0, 1], m[1, 1])


def _axis_angle(axis, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _floats(text: str | None, default="0 0 0") -> list[float]:
    return [float(v) for v in (text or default).split()]


def freeze(joint: ET.Element, q: float) -> None:
    """Fold a joint's motion at ``q`` into its origin and make it fixed."""
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    t = np.eye(4)
    t[:3, :3] = _rpy_matrix(_floats(origin.get("rpy")))
    t[:3, 3] = _floats(origin.get("xyz"))
    axis = _floats(joint.find("axis").get("xyz"), "1 0 0")
    motion = np.eye(4)
    if joint.get("type") == "prismatic":
        motion[:3, 3] = np.asarray(axis) * q
    else:
        motion[:3, :3] = _axis_angle(axis, q)
    t = t @ motion
    origin.set("xyz", " ".join(f"{v:.9g}" for v in t[:3, 3]))
    origin.set("rpy", " ".join(f"{v:.9g}" for v in _matrix_rpy(t[:3, :3])))
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "mimic"):
        for child in joint.findall(tag):
            joint.remove(child)


def prefix_joints(robot: ET.Element) -> None:
    """``yaw`` under ``PSM1_*`` links becomes ``PSM1_yaw``, and so do mimic references to it."""
    renamed: dict[tuple[str, str], str] = {}
    owner: dict[int, str] = {}
    for joint in robot.findall("joint"):
        child = joint.find("child").get("link")
        name = joint.get("name")
        for arm in (*INSTRUMENTS, "ECM"):
            if child.startswith(f"{arm}_") and not name.startswith(("SUJ_", f"{arm}_")):
                renamed[(name, arm)] = f"{arm}_{name}"
                joint.set("name", f"{arm}_{name}")
                owner[id(joint)] = arm
    for joint in robot.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None and id(joint) in owner:
            mimic.set("joint", renamed[(mimic.get("joint"), owner[id(joint)])])


def drop_phantom_jaw(robot: ET.Element, arm: str) -> None:
    """Drive ``jaw_1`` directly and mirror ``jaw_2`` on it; remove the ``jaw`` joint and its box."""
    joints = {j.get("name"): j for j in robot.findall("joint")}
    links = {link.get("name"): link for link in robot.findall("link")}
    robot.remove(joints[f"{arm}_jaw"])
    robot.remove(links[f"{arm}_jaw_link"])
    jaw_1 = joints[f"{arm}_jaw_1"]
    jaw_1.remove(jaw_1.find("mimic"))
    mimic = joints[f"{arm}_jaw_2"].find("mimic")
    mimic.set("joint", f"{arm}_jaw_1")
    mimic.set("multiplier", "-1")


def set_limits(robot: ET.Element) -> None:
    for joint in robot.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        arm, _, short = name.partition("_")
        kind = "ECM" if arm == "ECM" else "PSM"
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        if f"{kind}_{short}" in LIMITS:
            lower, upper, effort, velocity = LIMITS[f"{kind}_{short}"]
            limit.set("lower", f"{lower:.9g}")
            limit.set("upper", f"{upper:.9g}")
        elif mimic is not None:
            # A follower moves as fast as its leader times the multiplier, and no
            # motor pushes it, so its effort is the constraint's and not a limit.
            leader = mimic.get("joint").partition("_")[2]
            velocity = LIMITS[f"{kind}_{leader}"][3] * abs(float(mimic.get("multiplier", "1")))
            effort = FOLLOWER_EFFORT
        else:
            raise ValueError(f"no limits stated for actuated joint {name}")
        limit.set("effort", f"{effort:.9g}")
        limit.set("velocity", f"{velocity:.9g}")


def jaw_colliders(robot: ET.Element, meshes: Path) -> None:
    """A box over each colliderless jaw's mesh, in the mesh's own frame."""
    import trimesh

    for link in robot.findall("link"):
        name = link.get("name")
        if not name.endswith(("_jaw_1_link", "_jaw_2_link")) or link.find("collision") is not None:
            continue
        visual = link.find("visual")
        mesh_el = visual.find("geometry/mesh")
        mesh = trimesh.load(meshes / mesh_el.get("filename"), force="mesh")
        mesh.apply_scale(_floats(mesh_el.get("scale"), "1 1 1"))
        lo, hi = mesh.bounds
        collision = ET.SubElement(link, "collision")
        origin = visual.find("origin")
        t = np.eye(4)
        if origin is not None:
            t[:3, :3] = _rpy_matrix(_floats(origin.get("rpy")))
            t[:3, 3] = _floats(origin.get("xyz"))
        centre = t[:3, :3] @ ((lo + hi) / 2.0) + t[:3, 3]
        ET.SubElement(
            collision,
            "origin",
            xyz=" ".join(f"{v:.9g}" for v in centre),
            rpy=origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
        )
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size=" ".join(f"{v:.9g}" for v in (hi - lo)))


def trim_instrument_shafts(robot: ET.Element, out: Path) -> None:
    """End the proximal shaft at its housing, 467 mm behind the wrist pivot.

    Upstream's scaled roll mesh reaches -556.5921 mm in the roll frame. With
    the controller's 467 mm roll offset, it protrudes 45.6 mm beyond even the
    housing's rear surface. Clamp only that hidden proximal segment; the
    distal collar and wrist attachment remain the original CAD.
    """
    import trimesh

    filename = "meshes/generated/roll_4670.stl"
    for arm in INSTRUMENTS:
        link = robot.find(f"link[@name='{arm}_roll_link']")
        visual = link.find("visual")
        mesh_el = visual.find("geometry/mesh")
        old = out / mesh_el.get("filename")
        mesh = trimesh.load(old, force="mesh")
        mesh.apply_scale(_floats(mesh_el.get("scale"), "1 1 1"))
        mesh.apply_transform(_origin(visual))
        mesh.vertices[:, 2] = np.maximum(mesh.vertices[:, 2], -0.467)
        mesh.update_faces(mesh.nondegenerate_faces())
        path = out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        mesh_el.set("filename", filename)
        mesh_el.attrib.pop("scale", None)
        _set_origin(visual, np.eye(4))
        collision = link.find("collision")
        t = np.eye(4)
        lo, hi = mesh.bounds[:, 2]
        t[2, 3] = (lo + hi) / 2
        _set_origin(collision, t)
        collision.find("geometry/cylinder").set("length", f"{hi - lo:.9g}")
    # The original shaft mesh was copied before replacing its references.
    referenced = {m.get("filename") for m in robot.iter("mesh")}
    if str(old.relative_to(out)) not in referenced:
        old.unlink()


def debakey_jaws(robot: ET.Element, out: Path) -> None:
    """A labelled approximation, not manufacturer CAD, of the missing 420036 tip.

    Intuitive's October 2011 Instrument & Accessory Catalog, page 27, specifies
    12 mm jaw length. The SutureBot endoscope/wrist images show straight solid
    blades, unlike the 20 mm fenestrated Cadiere. Width (2.2 mm), thickness
    (1.2 mm), neck and serrations are visual estimates. Kinematic pivots stay
    the controller's. Generated geometry is Apache-2.0, like this script.
    """
    import trimesh

    for side, sign in ((1, -1.0), (2, 1.0)):
        link = robot.find(f"link[@name='PSM2_jaw_{side}_link']")
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
        # 12 mm blade, starting 3 mm beyond the pivot; taper toward its tip.
        vertices = []
        for y, width, thickness in ((0.003, 0.0022, 0.0012), (0.015, 0.0015, 0.0008)):
            for x in (0.0, sign * thickness):
                for z in (-width / 2, width / 2):
                    vertices.append((x, y, z))
        blade = trimesh.convex.convex_hull(np.asarray(vertices))
        hub = trimesh.creation.cylinder(radius=0.0024, height=0.0009, sections=32)
        hub.apply_translation((0.0, 0.0, sign * 0.00065))
        neck = trimesh.creation.box(extents=(0.0012, 0.0035, 0.0013))
        neck.apply_translation((sign * 0.0006, 0.00175, 0.0))
        parts = [hub, neck, blade]
        for y in np.linspace(0.0035, 0.0145, 23):
            ridge = trimesh.creation.box(extents=(0.00015, 0.00012, 0.0014))
            ridge.apply_translation((-sign * 0.000075, y, 0.0))
            parts.append(ridge)
        mesh = trimesh.util.concatenate(parts)
        rel = f"meshes/generated/debakey_420036_jaw_{side}.stl"
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=rel)
        ET.SubElement(ET.SubElement(visual, "material", name=""), "color", rgba="0.7 0.7 0.72 1")


def copy_meshes(robot: ET.Element, dvrk_model: Path, out: Path) -> None:
    """Every mesh the tree names, beside the URDF, relative, the big ones decimated."""
    import trimesh

    for mesh_el in robot.iter("mesh"):
        rel = mesh_el.get("filename").removeprefix("package://dvrk_model/")
        src, dst = dvrk_model / rel, out / rel
        mesh_el.set("filename", rel)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.stat().st_size <= DECIMATE_BYTES:
            shutil.copyfile(src, dst)
            continue
        import fast_simplification

        mesh = trimesh.load(src, force="mesh")
        ratio = 1.0 - min(1.0, DECIMATE_FACES / len(mesh.faces))
        points, faces = fast_simplification.simplify(
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.int64),
            target_reduction=ratio,
        )
        trimesh.Trimesh(points, faces).export(dst)


def _inertial(link: ET.Element, mass: float, com, inertia: np.ndarray) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.9g}" for v in com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    names = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    cells = (inertia[0, 0], inertia[0, 1], inertia[0, 2], inertia[1, 1], inertia[1, 2], inertia[2, 2])
    ET.SubElement(inertial, "inertia", {n: f"{v:.9g}" for n, v in zip(names, cells, strict=True)})


def _collider(link: ET.Element, meshes: Path, tag: str = "collision"):
    """The link's colliders (or ``tag``) as one trimesh in the link frame, each mesh as its hull."""
    import trimesh

    parts = []
    for collision in link.findall(tag):
        geometry = collision.find("geometry")
        if (box := geometry.find("box")) is not None:
            part = trimesh.creation.box(extents=_floats(box.get("size")))
        elif (cylinder := geometry.find("cylinder")) is not None:
            part = trimesh.creation.cylinder(radius=float(cylinder.get("radius")), height=float(cylinder.get("length")))
        elif (sphere := geometry.find("sphere")) is not None:
            part = trimesh.creation.icosphere(radius=float(sphere.get("radius")))
        elif (mesh := geometry.find("mesh")) is not None:
            part = trimesh.load(meshes / mesh.get("filename"), force="mesh").convex_hull
            part.apply_scale(_floats(mesh.get("scale"), "1 1 1"))
        else:
            continue
        origin = collision.find("origin")
        t = np.eye(4)
        if origin is not None:
            t[:3, :3] = _rpy_matrix(_floats(origin.get("rpy")))
            t[:3, 3] = _floats(origin.get("xyz"))
        parts.append(part.apply_transform(t))
    return trimesh.util.concatenate(parts) if parts else None


def add_inertials(robot: ET.Element, meshes: Path) -> None:
    """Mass for every moving link the source leaves without one.

    Measured where there is a measurement (the housing), steel over the collider
    where the part is a steel part (shaft, wrist, jaws), and a token gram where
    the link is only a frame. Nothing that states an inertial is touched.
    """
    moving = {j.find("child").get("link") for j in robot.findall("joint") if j.get("type") != "fixed"}
    for link in robot.findall("link"):
        name = link.get("name")
        if link.find("inertial") is not None:
            continue
        arm, _, part = name.partition("_")
        if part == "adaptor_link" and arm != "ECM":
            housing = next(lk for lk in robot.findall("link") if lk.get("name") == f"{arm}_housing_link")
            body = _collider(housing, meshes)
            lo, hi = body.bounds
            x, y, z = hi - lo
            box = np.diag([y * y + z * z, x * x + z * z, x * x + y * y]) * HOUSING_MASS / 12.0
            _inertial(link, HOUSING_MASS, (lo + hi) / 2.0, box)
        elif part in ("roll_link", "wrist_pitch_link", "wrist_yaw_link", "jaw_1_link", "jaw_2_link") and arm != "ECM":
            # The visual where a link has no collider: the source draws the part
            # it does not collide, and the part still weighs what it weighs.
            body = _collider(link, meshes) or _collider(link, meshes, "visual")
            body.density = STEEL
            _inertial(link, float(body.mass), body.center_mass, np.asarray(body.moment_inertia))
        elif name in moving and _collider(link, meshes) is None:
            _inertial(link, TOKEN_MASS, (0.0, 0.0, 0.0), np.eye(3) * 1.0e-7)


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    t = np.eye(4)
    if origin is not None:
        t[:3, :3] = _rpy_matrix(_floats(origin.get("rpy")))
        t[:3, 3] = _floats(origin.get("xyz"))
    return t


def _set_origin(joint: ET.Element, t: np.ndarray) -> None:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.set("xyz", " ".join(f"{v:.9g}" for v in t[:3, 3]))
    origin.set("rpy", " ".join(f"{v:.9g}" for v in _matrix_rpy(t[:3, :3])))


def _motion(joint: ET.Element, q: float) -> np.ndarray:
    axis = _floats(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, "1 0 0")
    t = np.eye(4)
    if joint.get("type") == "prismatic":
        t[:3, 3] = np.asarray(axis) * q
    elif joint.get("type") != "fixed":
        t[:3, :3] = _axis_angle(axis, q)
    return t


def _pose(xyz_xyzw) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = xyz_xyzw
    t = np.eye(4)
    t[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    t[:3, 3] = (x, y, z)
    return t


def fk(robot: ET.Element, link: str, q: dict[str, float]) -> np.ndarray:
    """``link``'s pose in the root frame, joints at ``q`` (mimic followers following, the rest at 0)."""
    by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
    t = np.eye(4)
    while link in by_child:
        joint = by_child[link]
        name = joint.get("name")
        mimic = joint.find("mimic")
        if mimic is not None:
            value = float(mimic.get("offset", "0")) + float(mimic.get("multiplier", "1")) * q.get(
                mimic.get("joint"), 0.0
            )
        else:
            value = q.get(name, 0.0)
        t = _origin(joint) @ _motion(joint, value) @ t
        link = joint.find("parent").get("link")
    return t


def fold_pitch_zero(robot: ET.Element) -> None:
    """Re-zero each parallelogram on the controller's pitch: :data:`PITCH_ZERO`, into the joints' origins.

    A joint is ``origin @ motion(q)``; reading it at ``q + c`` is the same joint
    with ``origin @ motion(c)``. The two followers turn ``multiplier`` times
    the leader, so they take ``multiplier * c``, and their mimic stays offset-free.
    """
    joints = {j.get("name"): j for j in robot.findall("joint")}
    for arm in (*INSTRUMENTS, "ECM"):
        for name, multiplier in (("pitch", 1.0), ("pitch_2", -1.0), ("pitch_3", 1.0)):
            joint = joints[f"{arm}_{name}"]
            _set_origin(joint, _origin(joint) @ _motion(joint, multiplier * PITCH_ZERO))


def exact_rcm(robot: ET.Element) -> None:
    """Close the parallelogram on the controller's RCM and the CAD cannula bore.

    USM_pitch_4.stl sections at z=-0.15 and -0.10 m have concentric inner
    and outer rings at x=0.10649, y=0. The upstream insertion axis is
    x=0.11523752: 8.74752 mm outside that bore centre. Keep the meshes rigid
    and place insertion on the bore, then solve the two hinge translations
    so the shaft pivots through the RCM at every pitch. Their positions are
    affine in these translations; three pitches determine the solution.

    No additional DOFs or disconnected visual chain: pitch_2 and pitch_3
    retain their physical mimic joints. The controller's rounded pi/2 is
    expressed as exact pi/2 (less than 0.002 mm difference over this arm).
    """
    joints = {j.get("name"): j for j in robot.findall("joint")}
    for arm in (*INSTRUMENTS, "ECM"):
        # Parallel axes need exact half turns; the CAD exported -3.1416.
        for name, angle in (("pitch", -0.973), ("pitch_2", -0.74611), ("pitch_3", -0.22689)):
            joint = joints[f"{arm}_{name}"]
            t = _origin(joint)
            t[:3, :3] = _rpy_matrix((0.0, angle, -math.pi if name != "pitch_3" else 0.0))
            _set_origin(joint, t)
        insertion = joints[f"{arm}_insertion"]
        t = _origin(insertion)
        t[:3, 3] = (0.10649, 0.0, 0.21055531 if arm == "ECM" else 0.31095531)
        _set_origin(insertion, t)
        if arm == "ECM":
            # Put the scope's roll on insertion, rather than 64.63 mm beside it.
            t = _origin(joints["ECM_roll"])
            t[:3, 3] = 0.0
            _set_origin(joints["ECM_roll"], t)
        hinges = [joints[f"{arm}_pitch_2"], joints[f"{arm}_pitch_3"]]
        origins = [_origin(j) for j in hinges]
        base = np.linalg.inv(fk(robot, f"{arm}_RCM_link", {}))
        length = 0.3822 if arm == "ECM" else 0.4826
        child = insertion.find("child").get("link")

        def residual(delta, hinges=hinges, origins=origins, base=base, child=child, arm=arm, length=length):
            for joint, origin, offset in zip(hinges, origins, delta.reshape(2, 2), strict=True):
                t = origin.copy()
                t[[0, 2], 3] += offset
                _set_origin(joint, t)
            errors = []
            for pitch in (-1.0, 0.0, 1.0):
                got = base @ fk(robot, child, {f"{arm}_pitch": pitch})
                # Modified DH yaw=0: the shaft points (0, -sin(pitch), -cos(pitch)).
                want = np.array((0.0, length * math.sin(pitch), length * math.cos(pitch)))
                errors.extend(got[:3, 3] - want)
            return np.array(errors)

        error = residual(np.zeros(4))
        columns = [(residual(np.eye(4)[i] * 0.01) - error) / 0.01 for i in range(4)]
        delta = np.linalg.lstsq(np.array(columns).T, -error, rcond=None)[0]
        remaining = residual(delta)
        if np.max(np.abs(remaining)) > 1e-7:
            raise ValueError(f"{arm}: RCM closure failed: {remaining}")
        print(f"{arm}: exact RCM hinge corrections [mm] {np.round(delta * 1000, 4).tolist()}")


def tool_tip_on_the_pivot(robot: ET.Element) -> None:
    """``tool_tip_link`` where the controller puts it: on the wrist-yaw axis, not 10.2 mm out along the jaws.

    The controller's tool frame is the last DH frame turned by the tool file's
    ``tooltip_offset``, a pure rotation, so its origin is the yaw pivot. The
    fit that found :data:`PITCH_ZERO` put it 10.65 mm back from ``dvrk_model``'s
    along the tool's z, with the rotation agreeing to 0.06 degrees.
    """
    for joint in robot.findall("joint"):
        if joint.get("name").endswith("_tool_tip"):
            t = _origin(joint)
            t[:3, 3] = 0.0
            _set_origin(joint, t)


def add_camera_link(robot: ET.Element) -> None:
    """``ECM_camera_link``: SutureBot's camera frame, on ``ECM_tip_link`` by :data:`CAMERA_IN_ECM_TIP`."""
    joint = ET.SubElement(robot, "joint", name="ECM_camera", type="fixed")
    ET.SubElement(joint, "parent", link="ECM_tip_link")
    ET.SubElement(joint, "child", link="ECM_camera_link")
    _set_origin(joint, _pose(CAMERA_IN_ECM_TIP))
    ET.SubElement(robot, "link", name="ECM_camera_link")


def place_psms(robot: ET.Element, setup: dict | None = None) -> dict[str, tuple[float, ...]]:
    """Solve PSM1's and PSM2's SUJ so their RCMs sit at :data:`rcm_in_camera`; freeze every SUJ.

    Four joints -- a column and three turns about it -- reach a position and a
    heading, which is everything a SUJ can set, and the logged values seed the
    solve. What is left (the RCM frame's tilt, which the SUJ cannot change) is
    folded into the fixed joint under the RCM so the frame lands exactly, and
    printed so a regenerated cart says how far it had to go.
    """
    from scipy.optimize import least_squares

    setup = SETUPS["home"] if setup is None else setup
    suj_q = setup["suj"]
    ecm_q = setup["joints"]["ECM"]
    rcm_in_camera = setup["rcm_in_camera"]

    joints = {j.get("name"): j for j in robot.findall("joint")}
    ecm = {f"ECM_{n}": v for n, v in zip(("yaw", "pitch", "insertion", "roll"), ecm_q, strict=True)}
    base = {f"SUJ_ECM_J{i}": v for i, v in enumerate(suj_q["ECM"])}
    camera = fk(robot, "ECM_camera_link", base | ecm)
    solved = dict(suj_q)
    for arm, pose in rcm_in_camera.items():
        want = camera @ _pose(pose)
        names = [f"SUJ_{arm}_J{i}" for i in range(len(suj_q[arm]))]

        def error(q, names=names, want=want, arm=arm):
            got = fk(robot, f"{arm}_RCM_link", dict(zip(names, q, strict=True)))
            heading = got[:3, 0] - want[:3, 0]
            return np.concatenate([got[:3, 3] - want[:3, 3], 0.1 * heading])

        fit = least_squares(error, np.asarray(suj_q[arm]))
        solved[arm] = tuple(float(v) for v in fit.x)
        got = fk(robot, f"{arm}_RCM_link", dict(zip(names, fit.x, strict=True)))
        rcm = joints[f"{arm}_rcm_fixed"]
        _set_origin(rcm, _origin(rcm) @ np.linalg.inv(got) @ want)
        moved = np.linalg.norm(got[:3, 3] - want[:3, 3]) * 1000.0
        print(
            f"{arm}: SUJ {np.round(suj_q[arm], 4).tolist()} -> {np.round(fit.x, 4).tolist()}, {moved:.2f} mm folded under the RCM"
        )
    for arm, values in solved.items():
        for i, q in enumerate(values):
            freeze(joints[f"SUJ_{arm}_J{i}"], q)
    return solved


def drop_unreferenced_visual_markers(robot: ET.Element) -> None:
    """The RCM spheres ``RCM_visual`` hangs on each arm: a debugging marker, not hardware."""
    for link in robot.findall("link"):
        if link.get("name").endswith("_RCM_link"):
            for visual in link.findall("visual"):
                link.remove(visual)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dvrk_model", type=Path, help=f"a jhu-dvrk/dvrk_model checkout at {DVRK_MODEL_COMMIT[:8]}")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--setup", choices=tuple(SETUPS), default="home")
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "-C", str(args.dvrk_model), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if commit != DVRK_MODEL_COMMIT:
        print(f"warning: dvrk_model is at {commit or 'an unknown commit'}, not {DVRK_MODEL_COMMIT}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        robot = expand(args.dvrk_model, Path(tmp))
    prefix_joints(robot)
    fold_pitch_zero(robot)
    exact_rcm(robot)
    tool_tip_on_the_pivot(robot)
    add_camera_link(robot)
    place_psms(robot, SETUPS[args.setup])
    for arm in INSTRUMENTS:
        drop_phantom_jaw(robot, arm)
    set_limits(robot)
    drop_unreferenced_visual_markers(robot)
    if args.out.exists():
        shutil.rmtree(args.out / "meshes", ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    copy_meshes(robot, args.dvrk_model, args.out)
    trim_instrument_shafts(robot, args.out)
    debakey_jaws(robot, args.out)
    jaw_colliders(robot, args.out)
    add_inertials(robot, args.out)
    ET.indent(robot)
    (args.out / "dvrk_si.urdf").write_text(
        f'<?xml version="1.0"?>\n<!-- Generated by scripts/dvrk_si_urdf.py from jhu-dvrk/dvrk_model @ '
        f"{DVRK_MODEL_COMMIT[:8]}; edit the script, not this file. -->\n" + ET.tostring(robot, encoding="unicode")
    )
    shutil.copyfile(args.dvrk_model / "LICENSE", args.out / "LICENSE")
    size = sum(p.stat().st_size for p in (args.out / "meshes").rglob("*") if p.is_file())
    print(f"wrote {args.out / 'dvrk_si.urdf'} and {size / 1e6:.1f} MB of meshes")


if __name__ == "__main__":
    main()

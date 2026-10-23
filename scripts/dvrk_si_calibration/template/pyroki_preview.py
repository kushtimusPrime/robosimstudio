"""The existing four-arm PyRoKi solver with the calibrated Studio PBR visuals.

Run from this restored twin: uv run --no-sync python pyroki_preview.py --port 8091
No Newton simulation or Studio arm solver participates in this page's IK.
"""

import argparse
import csv
import importlib.util
import json
import threading
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yourdfpy

from robosimstudio.core.render.finish import finish_for
from robosimstudio.core.render.uv import box_projected_uvs
from robosimstudio_studio.finish_gltf import glb_for, unweld

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / "scripts/dvrk_si_pyroki_ik.py").exists())


def prepare(setup, frame):
    from appearance import physical_tree, visual_asset
    from exterior import visual_asset as exterior_asset

    profile = json.loads((ROOT / f"assets/multisession/{setup}/profile.json").read_text())
    original = ROOT / profile["asset"]
    asset = exterior_asset(visual_asset(original))
    assert physical_tree(original) == physical_tree(asset)
    spec = importlib.util.spec_from_file_location("dvrk_basic_ik", REPO / "scripts/dvrk_si_pyroki_ik.py")
    basic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic)
    basic.URDF = Path(asset)
    with (ROOT / "ref/dataset" / profile["episode"] / "ee_csv.csv").open() as f:
        rows = list(csv.DictReader(f))
    index = np.clip(frame - profile["lag_frames"], 0, len(rows) - 1)
    lo, hi = int(index), min(int(index) + 1, len(rows) - 1)
    w = index - lo
    row = {k: (1 - w) * float(rows[lo][k]) + w * float(rows[hi][k]) for k in rows[lo] if rows[lo][k]}
    for arm, (tip, joints, _, locked) in list(basic.ARMS.items()):
        home = tuple(row[f"{arm.lower()}_js[{i}]"] for i in range(len(joints)))
        basic.ARMS[arm] = tip, joints, home, locked
    return basic, profile, row


class PbrUrdf:
    """URDF hierarchy plus Studio's own GLB material/texture conversion."""

    def __init__(self, server, asset):
        import trimesh
        from scipy.spatial.transform import Rotation

        self.server = server
        self.urdf = yourdfpy.URDF.load(str(asset), load_meshes=False)
        self.paths = {self.urdf.base_link: "/robot"}
        parents = {j.child: j.parent for j in self.urdf.joint_map.values()}

        def path(link):
            if link not in self.paths:
                self.paths[link] = path(parents[link]) + "/" + link
            return self.paths[link]

        self.frames = {name: server.scene.add_frame(path(name), show_axes=False) for name in self.urdf.link_map}
        self.mesh_count = 0
        for name, link in self.urdf.link_map.items():
            for i, visual in enumerate(link.visuals):
                source = visual.geometry.mesh
                if source is None:
                    raise ValueError(f"Unexpected non-mesh visual: {name}")
                mesh = trimesh.load(Path(asset).parent / source.filename, force="mesh", process=False)
                if source.scale is not None:
                    mesh.apply_scale(source.scale)
                material = visual.material
                if material and material.color:
                    color = tuple(material.color.rgba[:3])
                else:
                    embedded = getattr(mesh.visual, "material", None)
                    rgba = embedded.main_color if embedded is not None else mesh.visual.main_color
                    color = tuple(np.asarray(rgba[:3], dtype=float) / 255)
                finish = finish_for(color, role="robot")
                points, faces = mesh.vertices, mesh.faces
                uvs = None
                if finish.texture:
                    points, faces = unweld(points, faces)
                    uvs = box_projected_uvs(points, faces)
                blob = glb_for(points, faces, color, finish, uvs=uvs, smooth=True)
                if blob is None:
                    raise RuntimeError(f"PBR export failed: {name}/{i}")
                transform = visual.origin if visual.origin is not None else np.eye(4)
                server.scene.add_glb(
                    path(name) + f"/visual_{i}",
                    blob,
                    position=transform[:3, 3],
                    wxyz=Rotation.from_matrix(transform[:3, :3]).as_quat(scalar_first=True),
                )
                self.mesh_count += 1

    def update(self, names, cfg):
        from scipy.spatial.transform import Rotation

        self.urdf.update_cfg(dict(zip(names, cfg, strict=True)))
        for joint in self.urdf.joint_map.values():
            t = self.urdf.get_transform(joint.child, joint.parent)
            self.frames[joint.child].position = t[:3, 3]
            self.frames[joint.child].wxyz = Rotation.from_matrix(t[:3, :3]).as_quat(scalar_first=True)


def main(args):
    import viser

    basic, profile, row = prepare(args.setup, args.frame)
    if args.check:
        basic.check(args.start)
        _, robot, names, _, cfg = basic.load(args.start)
        visual = yourdfpy.URDF.load(str(basic.URDF), load_meshes=False)
        for displacement in (0.0, -0.01, 0.01):
            q = cfg + displacement
            visual.update_cfg(dict(zip(names, q, strict=True)))
            fk = np.asarray(basic.jaxlie.SE3(robot.forward_kinematics(jnp.asarray(q))).as_matrix())
            matrices = np.array([visual.get_transform(name) for name in robot.links.names])
            np.testing.assert_allclose(matrices, fk, atol=1e-6)
        print("Visual hierarchy matches PyRoKi FK for every link at three configurations.")
        return
    _, robot, names, targets, cfg = basic.load(args.start)
    for arm in ("PSM1", "PSM2", "PSM3"):
        raw = row.get(arm.lower() + "_jaw", 0.0)
        geo = profile["geometry"]
        value = geo[arm + "_jaw_offset"] + geo[arm + "_jaw_gain"] * (raw + 0.35) if arm != "PSM3" else raw / 2
        cfg[names.index(arm + "_jaw_1")] = value
    initial = cfg.copy()
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    view = PbrUrdf(server, basic.URDF)
    view.update(names, cfg)
    server.scene.add_grid("/ground", width=3, height=3)
    server.initial_camera.position = (1.9, 2.2, 2.25)
    server.initial_camera.look_at = (0.25, 0.25, 1.1)
    server.initial_camera.up = (0, 0, 1)
    server.initial_camera.fov = np.deg2rad(45)
    poses = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
    handles = {
        arm: server.scene.add_transform_controls(
            f"/targets/{arm}", scale=0.035, position=poses[i, 4:], wxyz=poses[i, :4]
        )
        for i, arm in enumerate(basic.ARMS)
    }
    for arm in basic.ARMS:
        server.scene.add_label(f"/targets/{arm}/label", arm)
    server.gui.add_markdown(
        "**PyRoKi · calibrated dVRK Si**\n\nDrag a tool's arrows/rings to set its pose. "
        "All movable arms share one IK problem. PSM3 and ECM start locked. "
        "No collision avoidance. Studio materials under Viser lighting."
    )
    camera_view = server.gui.add_dropdown("View", ("Whole robot", "Tools", "Instrument carriage"))
    reset = server.gui.add_button("Reset targets and joints")
    reset_requested = threading.Event()
    reset.on_click(lambda _: reset_requested.set())

    @camera_view.on_update
    def change_view(_):
        if camera_view.value == "Whole robot":
            eye, target, up = (1.9, 2.2, 2.25), (0.25, 0.25, 1.1), (0, 0, 1)
        elif camera_view.value == "Tools":
            target = np.mean([handles[a].position for a in ("PSM1", "PSM2")], axis=0)
            eye, up = np.add(target, (0.18, 0.18, 0.1)), (0, 0, 1)
        else:
            t = view.urdf.get_transform("PSM1_housing_link") @ view.urdf.link_map["PSM1_housing_link"].visuals[0].origin
            eye = (t @ [-0.01, 0.035, 0.115, 1])[:3]
            target = (t @ [0, -0.025, 0, 1])[:3]
            up = t[:3, :3] @ (-1, 0, 0)
        for camera in [server.initial_camera, *(c.camera for c in server.get_clients().values())]:
            camera.position, camera.look_at = eye, target
            if camera is server.initial_camera:
                camera.up = up
            else:
                camera.up_direction = up
            camera.fov = np.deg2rad(65 if camera_view.value == "Instrument carriage" else 45)
            camera.near = 0.001

    locks = {}
    for arm, (_, joints, _, locked) in basic.ARMS.items():
        with server.gui.add_folder(arm):
            arm_lock = server.gui.add_checkbox("Lock entire arm", initial_value=locked)
            boxes = [server.gui.add_checkbox(f"Lock {j}", initial_value=locked) for j in joints]
            locks.update({f"{arm}_{j}": b for j, b in zip(joints, boxes, strict=True)})

            @arm_lock.on_update
            def lock_arm(event, boxes=boxes):
                for b in boxes:
                    b.value = event.target.value

    elapsed = server.gui.add_number("Solve time (ms)", 0.0, disabled=True)
    errors = server.gui.add_markdown("Compiling the first solve…")
    print(f"PBR visuals ready: {view.mesh_count} meshes; {basic.URDF}", flush=True)
    previous_request = None
    while True:
        if reset_requested.is_set():
            cfg = initial.copy()
            poses = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
            for i, h in enumerate(handles.values()):
                h.wxyz, h.position = poses[i, :4], poses[i, 4:]
            reset_requested.clear()
            previous_request = None
        locked = {n: b.value for n, b in locks.items()}
        target_wxyz = np.array([h.wxyz for h in handles.values()])
        target_position = np.array([h.position for h in handles.values()])
        request = (target_wxyz.tobytes(), target_position.tobytes(), tuple(locked.values()))
        if request == previous_request:
            time.sleep(0.02)
            continue
        previous_request = request
        started = time.perf_counter()
        cfg = basic.solve_ik(
            robot,
            targets,
            target_wxyz,
            target_position,
            cfg,
            names,
            locked,
        )
        elapsed.value = 1000 * (time.perf_counter() - started)
        poses = np.asarray(robot.forward_kinematics(jnp.asarray(cfg)))[targets]
        report = []
        with server.atomic():
            view.update(names, cfg)
            for i, (arm, h) in enumerate(handles.items()):
                all_locked = all(locked[f"{arm}_{j}"] for j in basic.ARMS[arm][1])
                h.visible = not all_locked
                if all_locked:
                    h.wxyz, h.position = poses[i, :4], poses[i, 4:]
                mm = 1000 * np.linalg.norm(poses[i, 4:] - h.position)
                report.append(f"{arm}: locked" if all_locked else f"{arm}: {mm:.2f} mm target error")
            errors.content = "\n\n".join(report)
        time.sleep(0.02)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--setup", default="setup_017")
    parser.add_argument("--frame", type=int, default=94)
    parser.add_argument("--start", choices=("suturebot", "middle"), default="suturebot")
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args())

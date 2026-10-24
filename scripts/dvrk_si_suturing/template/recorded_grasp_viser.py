# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Read-only Viser playback of a passed Studio contact-physics recording."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import yourdfpy

from robosimstudio.core.render.finish import Finish, finish_for
from robosimstudio.core.render.uv import box_projected_uvs
from robosimstudio_studio.finish_gltf import glb_for, unweld

ROOT = Path(__file__).resolve().parent


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
    take_path = Path(args.take).resolve()
    data = json.loads(take_path.read_text())
    metrics = json.loads(take_path.with_name(take_path.stem + "_metrics.json").read_text())
    if not metrics["pass"]:
        raise RuntimeError("This viewer requires a recording that passed the contact gates.")
    # Register the exact Studio finish recipe used to generate these visuals.
    from appearance import visual_asset
    from exterior import visual_asset as exterior_asset

    profile = json.loads((ROOT / "assets/multisession/setup_017/profile.json").read_text())
    exterior_asset(visual_asset(ROOT / profile["asset"]))
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    robot = PbrUrdf(server, data["urdf"])
    mesh = trimesh.load(data["needle_mesh"], force="mesh", process=False)
    blob = glb_for(mesh.vertices, mesh.faces, (0.66, 0.68, 0.70), Finish(metallic=1.0, roughness=0.23), smooth=True)
    needle = server.scene.add_glb("/needle", blob)
    records = data["history"]
    times = np.array([r["time_s"] for r in records])
    camera = np.load(ROOT / "assets/suturing/initial.npz")["camera"]
    target = np.asarray(records[0]["needle_position_m"])
    server.gui.add_markdown(
        "**Recorded contact physics — dVRK Si**\n\nHold, carry, return, release. "
        "No needle attachment to the gripper. Thread and tissue are excluded from this diagnostic. "
        "Playback displays saved simulation states; fresh physics runs in Studio on port 8082."
    )
    play = server.gui.add_checkbox("Play recorded physics", initial_value=False)
    loop = server.gui.add_checkbox("Loop", initial_value=True)
    speed = server.gui.add_dropdown("Playback speed", ("0.25x", "0.5x", "1x"), initial_value="1x")
    slider = server.gui.add_slider("Simulation time (s)", min=0.0, max=float(times[-1]), step=0.005, initial_value=0.0)
    view = server.gui.add_dropdown(
        "View", ("Endoscope viewpoint", "Tool close-up", "Whole robot"), initial_value="Tool close-up"
    )
    phase = server.gui.add_markdown("Hold")
    server.gui.add_markdown(
        f"**Measured gates**\n\nMaximum held needle-surface drift: {metrics['max_held_surface_drift_mm']:.3f} mm. "
        "The needle releases when the jaws open. "
        "The physical constants are assumptions, not a SutureBot material fit."
    )

    def set_view():
        if view.value == "Whole robot":
            eye, look, up = (1.9, 2.2, 2.25), (0.25, 0.25, 1.1), (0, 0, 1)
            fov = np.deg2rad(45)
        elif view.value == "Tool close-up":
            eye = camera[:3, 3]
            look = target
            up = -camera[:3, 1]
            fov = np.deg2rad(45)
        else:
            eye = camera[:3, 3]
            look = eye + camera[:3, 2] * 0.05
            up = -camera[:3, 1]
            fov = 2 * np.arctan(540 / (2 * profile["intrinsics"]["fy"]))
        for c in [server.initial_camera, *(x.camera for x in server.get_clients().values())]:
            c.position = eye
            c.look_at = look
            c.fov = float(fov)
            c.near = 0.015 if view.value == "Tool close-up" else 0.002
            if c is server.initial_camera:
                c.up = up
            else:
                c.up_direction = up

    set_view()
    view.on_update(lambda _: set_view())
    last_value = 0.0
    anchor = 0.0
    wall = time.monotonic()
    was_playing = False
    last_index = -1
    print(f"Recorded physics ready: http://localhost:{server.get_port()}", flush=True)
    while True:
        now = time.monotonic()
        if slider.value != last_value or play.value != was_playing:
            anchor = slider.value
            wall = now
        t = anchor + (now - wall) * float(speed.value[:-1]) if play.value else slider.value
        if play.value and t > times[-1]:
            if loop.value:
                anchor = 0.0
                wall = now
                t = 0.0
            else:
                play.value = False
                t = float(times[-1])
        index = min(int(np.searchsorted(times, t, side="right") - 1), len(records) - 1)
        index = max(index, 0)
        if index != last_index:
            record = records[index]
            with server.atomic():
                robot.update(data["joint_names"], record["joint_positions_rad_m"])
                pose = record["needle_pose_xyzw"]
                needle.position = pose[:3]
                needle.wxyz = np.asarray(pose[3:])[[3, 0, 1, 2]]
            slider.value = float(times[index])
            last_value = slider.value
            phase.content = (
                "**Hold**"
                if t < 1
                else "**Carry**"
                if t < 2
                else "**Hold after move**"
                if t < 3
                else "**Return**"
                if t < 4
                else "**Release**"
            )
            last_index = index
        was_playing = play.value
        time.sleep(1 / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take", default=str(ROOT / "run/grasp_probe.json"))
    parser.add_argument("--port", type=int, default=8092)
    main(parser.parse_args())

"""Repeatable exterior inspection views of the live, calibrated robot."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import yourdfpy

import robosimstudio as rss
from robosimstudio.core.render.room import VOID

ROOT = Path(__file__).resolve().parent


def camera_specs(sim, profile):
    urdf = yourdfpy.URDF.load(str(ROOT / profile["asset"]), load_meshes=False)
    names = [n.split("/")[-1] for n in sim.model.joint_label]
    starts = sim.model.joint_q_start.numpy()
    q = sim.state.joint_q.numpy()
    urdf.update_cfg({n: float(q[starts[i]]) for i, n in enumerate(names) if n in urdf.joint_map})
    specs = []
    for name, link, eye, target, up in [
        ("tool", "PSM1_housing_link", (-0.025, 0.18, 0.04), (0, 0, 0), (-1, 0, 0)),
        ("tool_back", "PSM1_housing_link", (-0.025, -0.18, 0.04), (0, 0, 0), (-1, 0, 0)),
        ("mount", "PSM1_housing_link", (-0.01, 0.035, 0.115), (0, -0.025, 0), (-1, 0, 0)),
        ("forceps", "PSM2_housing_link", (0, 0.095, 0), (0, 0, 0), (-1, 0, 0)),
        ("scope_collar", "ECM_endoscope_link", (0.13, 0.12, 0.065), (0, 0, 0.065), (0, 0, -1)),
        ("scope", "ECM_endoscope_link", (0.23, 0.22, -0.18), (0, 0, -0.01), (0, 0, -1)),
        ("arm", "PSM1_pitch_link", (0.25, -0.6, 0.35), (0.15, 0, 0.07), (0, 0, 1)),
    ]:
        transform = urdf.get_transform(link) @ urdf.link_map[link].visuals[0].origin

        def pt(v, transform=transform):
            return tuple((transform @ [*v, 1])[:3])

        specs.append(
            rss.CameraSpec(
                name=name,
                width=1000,
                height=800,
                fov_y=65 if name in ("forceps", "mount") else 40,
                eye=pt(eye),
                look_at=pt(target),
                up=tuple(transform[:3, :3] @ up),
                near=0.001,
                far=8,
            )
        )
    specs.append(
        rss.CameraSpec(
            name="robot",
            width=1000,
            height=1000,
            fov_y=45,
            eye=(1.9, 2.2, 2.25),
            look_at=(0.25, 0.25, 1.1),
            up=(0, 0, 1),
            near=0.01,
            far=8,
        )
    )
    return specs


def set_browser_view(sim, session, profile, view="robot"):
    """Move only the free inspection viewpoint; never the calibrated camera."""
    cam = next(c for c in camera_specs(sim, profile) if c.name == view)
    handles = [session.server.initial_camera, *(c.camera for c in session.server.get_clients().values())]
    for camera in handles:
        camera.position = cam.eye
        camera.look_at = cam.look_at
        if camera is session.server.initial_camera:
            camera.up = cam.up
        else:
            camera.up_direction = cam.up
        camera.fov = math.radians(cam.fov_y)
        camera.near = cam.near
        camera.far = cam.far


def capture(sim, session, profile, label, view="tool"):
    from multisession_preview import replay
    from PIL import Image

    replay(sim, session, profile, 94)
    specs = camera_specs(sim, profile)
    folder = ROOT / "run/exterior"
    folder.mkdir(exist_ok=True)
    (folder / (label + "_cameras.json")).write_text(json.dumps([asdict(c) for c in specs], indent=2))
    specs = [c for c in specs if c.name == view]
    renderer = sim.make_renderer("rtx", cameras=specs, render=VOID, channels=("rgb",))
    try:
        frames = renderer.render(sim.state)
        for cam in specs:
            Image.fromarray(frames[cam.name]["rgb"][..., :3]).save(folder / f"{label}_{cam.name}.png")
    finally:
        renderer.close()
    return [str(folder / f"{label}_{c.name}.png") for c in specs]

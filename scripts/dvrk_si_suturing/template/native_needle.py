# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Rigid needle in the robot's solver; a physical swage couples to the rod.

The grasp is exclusively MuJoCo contact. The swage is a compliant endpoint
constraint with an equal-and-opposite wrench, delayed by one coupling step.
"""

from pathlib import Path

import numpy as np
import trimesh

from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.render.finish import Finish
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import AssetSpec, ObjectSpec


def specification(folder, parameters):
    from needle import boundary

    folder = Path(folder)
    data = np.load(folder / "needle.npz")
    v = data["vertices"]
    t = data["tets"]
    mesh = trimesh.Trimesh(v, boundary(v, t), process=True)
    trimesh.repair.fix_normals(mesh)
    centre = mesh.center_mass.copy()
    mesh.vertices -= centre
    mesh.export(folder / "needle.obj")
    # Each ring has a centre plus 12 perimeter vertices. Hull only adjacent
    # stations: a single hull over the entire arc would seal its opening.
    rings = (v - centre).reshape(65, 13, 3)
    paths = []
    for i in range(64):
        p = folder / f"needle_contact_{i:02d}.stl"
        trimesh.convex.convex_hull(rings[i : i + 2].reshape(-1, 3)).export(p)
        paths.append(str(p))
    return ObjectSpec(
        name="needle",
        entity="native_suture_needle",
        asset=AssetSpec(path=str(folder / "needle.obj"), collision_paths=tuple(paths)),
        pose=tuple(centre),
        density=parameters["density_kg_m3"],
        friction=parameters["friction"],
        contact_gap=1e-5,
        obb_grasp=False,
        color=(0.66, 0.68, 0.70),
        finish=Finish(metallic=1.0, roughness=0.23),
    )


@register_entity("native_suture_needle")
class NativeNeedle(EntityRigid):
    def on_scene_ready(self, pipeline):
        from robosimstudio.domains.capabilities import host_bridge

        bridge = host_bridge(pipeline, "thread")
        self.bridge = bridge
        self.pin = None
        self.swage_local = np.load(Path(self.spec.asset.path).parent / "initial.npz")["thread"][0] - np.asarray(
            self.spec.pose
        )
        original_tick = bridge.tick
        original_feed = bridge.apply_wrenches

        def ensure_swage(*args, **kwargs):
            if self.pin is None or not bridge.has_pin(self.pin):
                entry = next(e for e in bridge.actors if e["entity"] == "thread")
                target = np.asarray(entry["read"]()).reshape(-1, 3)[:1]
                self.pin = bridge.pin_nodes("thread", [entry["start"]], target, stiffness=1e4, damping=0.01)
                bridge.ride(self.pin, self.body, self.swage_local[None])
                for constraint in bridge._pins[self.pin]:
                    constraint.register_query(bridge.physics.QueryType.CONSTRAINT_FORCE)
            return original_tick(*args, **kwargs)

        def reaction(wrenches):
            wrenches = wrenches.copy()
            if self.pin is not None and bridge.has_pin(self.pin):
                constraint = bridge._pins[self.pin][0]
                force = -np.asarray(constraint.get_force()).reshape(-1)[:3]
                pose = pipeline.state_0.body_q.numpy()[self.body]
                arm = quat_to_mat(pose[3:7]) @ self.swage_local
                wrenches[self.body, :3] += force
                wrenches[self.body, 3:] += np.cross(arm, force)
            if original_feed is not None:
                original_feed(wrenches)

        bridge.tick = ensure_swage
        bridge.apply_wrenches = reaction

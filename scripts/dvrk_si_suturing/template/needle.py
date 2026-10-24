# SPDX-License-Identifier: Apache-2.0
"""A true six-DoF needle sharing SuperDex's implicit solve with thread/tissue.

Newton particles carry its rendering mesh only. They are never independently
integrated; read-back is the rigid actor's computed transform, not replay data.
"""

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.engines.geometry import _read_tet, actor_geometry, register_actor_builder, register_actor_kind
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import HostActor
from robosimstudio_materials.soft import EntitySoftSuperdex


class Needle(EntitySoftSuperdex):
    def host_actor(self):
        p = dict(self.spec.material.actor_params())
        return HostActor(
            kind="suture_rigid",
            params={
                k: v
                for k, v in p.items()
                if k
                in (
                    "density",
                    "friction",
                    "contact_penalty",
                    "contact_padding",
                    "contact_threshold",
                    "contact_smoothing",
                    "friction_falloff",
                )
            },
        )

    def on_scene_ready(self, pipeline):
        from precision_contacts import refine

        from robosimstudio.domains.capabilities import host_bridge

        bridge = host_bridge(pipeline, self.name)
        refine(bridge)
        for weld in bridge.welds:
            weld.register_query(bridge.physics.QueryType.CONSTRAINT_FORCE)
        super().on_scene_ready(pipeline)


def boundary(v, t):
    faces = np.concatenate([t[:, [0, 2, 1]], t[:, [0, 1, 3]], t[:, [0, 3, 2]], t[:, [1, 2, 3]]])
    _, idx, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[idx[count == 1]]


def build(bridge, record, particle_q, tet_indices=None, tet_materials=None):
    p = bridge.physics
    g = actor_geometry(bridge.model, {**record, "kind": "tet"})
    v = g.rest.copy()
    centre = v.mean(axis=0)
    local = v - centre
    faces = boundary(v, g.tets)
    shape = p.create_tri_mesh_shape(coordinates=local.ravel(), connectivity=faces.astype(np.int32).ravel())
    sdf = p.GridSdfParams()
    sdf.resolution_mode = p.GridSdfResolutionMode.EXPLICIT
    sdf.resolution_delta = [0.0001] * 3
    sdf.boundary_padding_dist = 0.0003
    actor = bridge.scene_of(record.get("world", 0)).create_rigid_actor(
        name=record["name"],
        layer="needle",
        shape=shape,
        is_static=False,
        world_from_local=p.TransformRT(translation=centre.tolist()),
        density=g.params["density"],
        collider_type=p.ColliderType.SDF,
        sdf=sdf,
        contact=bridge.contact_params(g.params, penalty=1e10, friction=0.3, padding=0),
        boundary_element_type=p.ActorBoundaryElementType.P1Q3,
    )

    def read():
        tr = actor.get_root_transform()
        return local @ quat_to_mat(np.asarray(tr.rotation)).T + np.asarray(tr.translation)

    return {
        "name": record["name"],
        "actor": actor,
        "link": actor,
        "read": read,
        "start": g.start,
        "count": g.count,
        "prev": v,
        "local": local,
        "rest_centre": centre,
        "faces": faces,
    }


register_entity("suture_needle")(Needle)
register_entity("suture_needle", engine="superdex")(Needle)
register_actor_kind("suture_rigid", _read_tet)
register_actor_builder(
    "superdex",
    "suture_rigid",
    build,
    consumes=(
        "density",
        "friction",
        "contact_penalty",
        "contact_padding",
        "contact_threshold",
        "contact_smoothing",
        "friction_falloff",
    ),
)

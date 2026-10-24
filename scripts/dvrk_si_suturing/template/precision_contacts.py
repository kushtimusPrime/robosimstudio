# SPDX-License-Identifier: Apache-2.0
"""Resolve surgical jaw contacts at 25 microns inside this twin's host solver.

Uses the same compound meshes as the task URDF. No geometry is enlarged.
Only SDF sampling is refined; the standard default follows average mesh edge
length and is too coarse relative to a 0.48 mm needle.
"""

from robosimstudio.engines.geometry import iter_mirrors
from robosimstudio.engines.superdex.mirror import make_collider_shape


def refine(bridge):
    if getattr(bridge, "suturing_precision_contacts", False):
        return
    p = bridge.physics
    geometry = {g.shape: g for g in iter_mirrors(bridge.model, bridge.recipe.get("mirrors", ()))}
    count = 0
    for mirror in bridge.mirrors:
        if mirror.body < 0:
            continue
        name = str(bridge.model.body_label[mirror.body]).split("/")[-1]
        if name not in [f"{a}_jaw_{j}_link" for a in ("PSM1", "PSM2") for j in [1, 2]]:
            continue
        geom = geometry[mirror.shape]
        scene = bridge.scene_of(mirror.world)
        sdf = p.GridSdfParams()
        sdf.resolution_mode = p.GridSdfResolutionMode.EXPLICIT
        sdf.resolution_delta = [0.000025] * 3
        sdf.boundary_padding_dist = 0.0001
        old = mirror.actor
        contact = old.get_contact_params()
        contact.penalty_coefficient = 1e11
        actor = scene.create_rigid_actor(
            name=f"precision_jaw:{mirror.shape}",
            layer="mirror",
            shape=make_collider_shape(p, geom),
            is_static=True,
            collider_type=p.ColliderType.SDF,
            sdf=sdf,
            world_from_local=old.get_root_transform(),
            contact=contact,
        )
        scene.destroy_actor(old.get_handle())
        mirror.actor = actor
        count += 1
    bridge.suturing_precision_contacts = True
    bridge.suturing_precision_count = count
    bridge._mirror_bodies = None

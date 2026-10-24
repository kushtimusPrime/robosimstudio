# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""SutureBot needle pass development: frozen robot, physically coupled task assets.

Initial material values and asset poses are priors, not an identified fit.
No scripted object animation or gripper latch. Task-specific jaw contacts are authorized.
"""

import json
from dataclasses import replace
from pathlib import Path

import needle  # noqa: F401 -- registers the local rigid host actor
from geometry import build_assets
from multisession_preview import build

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.specs.asset import AssetSpec
from robosimstudio_materials.rod import RodSpec
from robosimstudio_materials.soft import SoftMaterialSpec, SoftSpec

ROOT = Path(__file__).resolve().parent
params = json.loads((ROOT / "parameters.json").read_text())
n = params["numerics"]
ISOLATED_GRASP = True
host = rss.backend(
    "superdex",
    substeps=n["deformable_substeps"],
    newton_iterations=n["newton_iterations"],
    linear_solver="ldlt",
    linear_iterations=1000,
    feedback_alpha=1.0,
    feedback_dv_max=20.0,
    mirror_threshold=n["contact_threshold_m"],
    mirror_smoothing=n["contact_smoothing_m"],
    mirror_friction_falloff=n["friction_falloff_m_per_s"],
)
physics = replace(rss.MUJOCO_500HZ, control_hz=n["control_hz"], substeps=n["rigid_substeps"])
physics = replace(physics, collision=replace(physics.collision, rigid_gap=1e-5))
physics = physics.with_backend(
    "*",
    rss.backend(
        "mujoco", nconmax=2048, njmax=4096, enable_multiccd=True, use_mujoco_cpu=True, use_mujoco_contacts=True
    ),
)
physics = physics.with_backend("soft", host).with_backend("rod", host)


def configure(scene, profile):
    global asset_dir, camera, pad_rotation, pad_centre, tissue, strand, needle_ref  # noqa: PLW0603 -- live console handles
    asset_dir, camera, pad_rotation, pad_centre, thread = build_assets(params, profile)
    if ISOLATED_GRASP:
        from grasp_probe import add_needle

        needle_ref = add_needle(scene, asset_dir, params["needle"])
        return
    contact = {
        "contact_penalty": n["contact_penalty_Pa_per_m"],
        "contact_padding": 0.0,
        "contact_threshold": n["contact_threshold_m"],
        "contact_smoothing": n["contact_smoothing_m"],
        "friction_falloff": n["friction_falloff_m_per_s"],
    }
    t = params["tissue"]
    s = params["thread"]
    m = params["needle"]
    tissue = scene.add(
        SoftSpec(
            name="tissue",
            asset=AssetSpec(path=str(asset_dir / "tissue.npz")),
            material=SoftMaterialSpec(
                young=t["young_Pa"],
                poisson=t["poisson"],
                density=t["density_kg_m3"],
                stiffness_damping=t["stiffness_damping_s"],
                friction=t["friction"],
                particle_radius=0.0002,
                collider="sdf",
                sdf_resolution=0.0006,
                **contact,
            ),
            anchored=lambda x: ((x - pad_centre) @ pad_rotation)[:, 2] <= t["rigid_backing_m"] + 1e-6,
            color=(0.72, 0.42, 0.32),
            finish=Finish(roughness=0.59),
            settle_frames=0,
        )
    )
    needle_ref = scene.add(
        SoftSpec(
            name="needle",
            entity="suture_needle",
            asset=AssetSpec(path=str(asset_dir / "needle.npz")),
            material=SoftMaterialSpec(
                young=2e11,
                poisson=0.3,
                density=m["density_kg_m3"],
                friction=m["friction"],
                particle_radius=0.0001,
                **contact,
            ),
            color=(0.66, 0.68, 0.70),
            finish=Finish(metallic=1.0, roughness=0.23),
            settle_frames=0,
        )
    )
    strand = scene.add(
        RodSpec(
            name="thread",
            nodes=tuple(map(tuple, thread)),
            radius=s["radius_m"],
            density=s["density_kg_m3"],
            young_modulus=s["bending_modulus_Pa"],
            axial_stiffness=s["axial_stiffness_N"],
            poisson_ratio=0.4,
            stiffness_damping=s["stiffness_damping_s"],
            stress_free="straight",
            mu_cable=s["friction"],
            contact_penalty=n["contact_penalty_Pa_per_m"],
            contact_threshold=n["contact_threshold_m"],
            contact_smoothing=n["contact_smoothing_m"],
            friction_falloff=n["friction_falloff_m_per_s"],
            contact_gap=0.0001,
            allow_grasp_latch=False,
            filter_fingertip_scenery=False,
            self_contact=True,
            joins=(("start", "needle", 0.00001),),
            skin_segments=8,
            color=(0.035, 0.22, 0.13),
            finish=Finish(roughness=0.54),
        )
    )


scene, sim, profile = build(
    params["setup"],
    appearance=True,
    exterior=True,
    configure=configure,
    physics=physics,
    episode=params["episode"],
    suturing_contacts=True,
)
from rigid_subcycling import install

install(sim.pipeline, max_step=1e-6)
if ISOLATED_GRASP:
    from surgical_contacts import configure as configure_contacts

    contact_accuracy = configure_contacts(sim.pipeline)
from grasp_probe import scalar_targets

sim.control.joint_target_q.assign(scalar_targets(sim, sim.state.joint_q.numpy()))


def inspect_physics(sim):
    """200 ms at the recorded grasp; does not prescribe needle pose."""
    command = sim.state.joint_q.numpy()
    names = [str(n).split("/")[-1] for n in sim.model.joint_label]
    starts = sim.model.joint_q_start.numpy()
    for joint, sign in [("PSM1_jaw_1", -1), ("PSM1_jaw_2", 1)]:
        command[starts[names.index(joint)]] += sign * params["control"]["grasp_preload_rad"]
    initial = sim.state.joint_q.numpy().copy()
    for step in range(40):
        alpha = min((step + 1) / params["numerics"]["control_hz"] / params["control"]["closure_ramp_s"], 1.0)
        sim.control.joint_target_q.assign(initial + alpha * (command - initial))
        sim.run(1)
        yield


def grasp_test(sim):
    from grasp_probe import run

    yield from run(sim, needle_ref, profile, params, preload=0.015, ramp_s=0.001)


routines = (
    {"Needle contact: hold, carry, release": grasp_test}
    if ISOLATED_GRASP
    else {"Contact hold (0.2 s)": inspect_physics}
)
BUILD = {"control_mode": "ee_pose_abs", "warmup": False}

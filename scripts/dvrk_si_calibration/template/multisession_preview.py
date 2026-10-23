"""Render fitted dataset profiles through RoboSimStudio, with recorded joint replay."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import newton
import numpy as np

import robosimstudio as rss
from robosimstudio.core.render.room import VOID
from robosimstudio.robots.embodiments.dvrk_si import DVRK_SI
from robosimstudio.robots.registry import get_gripper, register_embodiment, register_gripper

ROOT = Path(__file__).resolve().parent


def replay(sim, session, profile, image_index=0, episode=None):
    eid = episode or profile["episode"]
    rows = list(csv.DictReader(open(ROOT / "ref/dataset" / eid / "ee_csv.csv")))
    idx = np.clip(image_index - profile["lag_frames"], 0, len(rows) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(rows) - 1)
    w = idx - lo
    row = {k: (1 - w) * float(rows[lo][k]) + w * float(rows[hi][k]) for k in rows[lo] if rows[lo][k]}
    names = [str(n).split("/")[-1] for n in sim.model.joint_label]
    starts = sim.model.joint_q_start.numpy()
    q = sim.state.joint_q.numpy()
    geo = profile["geometry"]
    for a in ["PSM1", "PSM2", "PSM3", "ECM"]:
        js = ["yaw", "pitch", "insertion", "roll"] + ([] if a == "ECM" else ["wrist_pitch", "wrist_yaw"])
        vals = {j: row[f"{a.lower()}_js[{i}]"] for i, j in enumerate(js)}
        vals.update(pitch_2=-vals["pitch"], pitch_3=vals["pitch"])
        if a != "ECM":
            raw = row.get(a.lower() + "_jaw", 0)
            jaw = geo[a + "_jaw_offset"] + geo[a + "_jaw_gain"] * (raw + 0.35) if a in ["PSM1", "PSM2"] else raw / 2
            vals.update(jaw_1=jaw, jaw_2=-jaw)
        for j, v in vals.items():
            q[starts[names.index(a + "_" + j)]] = v
    for state in [sim.pipeline.state_0, sim.pipeline.state_1]:
        state.joint_q.assign(q)
        state.joint_qd.zero_()
        newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
    if session is not None:
        session.set_paused(True)
        # A replay changes the actual joints, so also reseat the IK targets.
        # Otherwise the handles stay at the previous frame and unpausing drives
        # the robot back toward stale targets.
        for panel in session.arms.values():
            panel.ik.hold_into(sim.state.joint_q, sim.control.joint_target_q)
            if panel.gripper is not None:
                hand = panel.gripper
                fraction = (q[hand.coords] - hand.open_q) / (hand.closed_q - hand.open_q)
                hand.command = float(np.clip(np.mean(fraction), 0.0, 1.0))
            panel.resume()
            panel.gizmo.scale = 0.035
        if session.live is not None:
            session.live._drain_frustums(force=True)
            for frustum in session.live._frusta.values():
                frustum.scale = 0.035
        # Replay changes FK without advancing step_count. The paused Studio
        # refresh otherwise keeps its old mesh transforms while handles and
        # endoscope renders already use the new pose.
        session.draw()


def build(profile_name, appearance=False, exterior=False):
    profile = json.load(open(ROOT / "assets/multisession" / profile_name / "profile.json"))
    asset = str(ROOT / profile["asset"])
    look = VOID
    if appearance:
        from appearance import render_look, visual_asset

        asset = visual_asset(asset)
        look = render_look()
    if exterior:
        from exterior import visual_asset as exterior_asset

        asset = exterior_asset(asset)
    arms = []
    for arm in DVRK_SI.arms:
        gripper = arm.gripper
        if arm.name.startswith("PSM"):
            # Inspection targets the controller's wrist-yaw pivot, exactly as
            # scripts/dvrk_si_pyroki_ik.py does. A nominal mid-blade grasp TCP
            # is a different frame, especially with fitted blade lengths.
            hand = get_gripper(gripper)
            gripper = "dvrk_preview_" + arm.name
            register_gripper(replace(hand, name=gripper, tcp_offset=(0.0, 0.0, 0.0)))
        arms.append(replace(arm, home_q=tuple(profile["joints"][arm.name]), gripper=gripper))
    arms = tuple(arms)
    register_embodiment(
        replace(
            DVRK_SI,
            name="dvrk_dataset_profile",
            asset=replace(DVRK_SI.asset, path=asset),
            urdf=asset,
            arms=arms,
            base_offset=(0, 0, 0, 0, 0, 0, 1),
            cross_arm_collisions=False,
        )
    )
    scene = rss.Scene(cell="void", name="dvrk_si_dataset_preview", physics=rss.MUJOCO_500HZ, render=look)
    scene.add(rss.robot("dvrk_dataset_profile"))
    scene.add(
        rss.CameraSpec(
            name="endoscope",
            width=960,
            height=540,
            **profile["intrinsics"],
            parent_frame="ECM",
            xform=tuple(profile["camera_in_ecm_roll"]),
            # The CAD lens cover extends 1.5 mm beyond the controller tip.
            near=0.002,
            far=3,
        )
    )
    sim = scene.build(control_mode="ee_pose_abs", warmup=False)
    replay(sim, None, profile)
    return scene, sim, profile

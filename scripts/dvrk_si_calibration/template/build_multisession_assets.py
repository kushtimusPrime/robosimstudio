"""Build reviewable per-setup URDFs with shared fitted jaw geometry.

These are experimental visualization profiles. The library URDF and controller
frames are unchanged. SUJ placement absorbs each setup's rigid image registration.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yourdfpy

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def matrix(v):
    from scipy.spatial.transform import Rotation

    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    t[:3, 3] = v[3:]
    return t


def pose(t):
    from scipy.spatial.transform import Rotation

    return [*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_quat()]


def main():
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="setup_017")
    parser.add_argument("--dvrk-model", type=Path, default=Path.home() / "dvrk_src/dvrk_model")
    args = parser.parse_args()
    fit = json.load(open(ROOT / "multisession_registration.json"))
    if args.session == "all":
        for session in sorted(fit["registrations"]):
            subprocess.run(
                [sys.executable, __file__, "--session", session, "--dvrk-model", str(args.dvrk_model)], check=True
            )
        return
    labels = json.load(open(ROOT / "ref/multisession_landmarks.json"))
    e = next(e for e in labels["episodes"] if e["session"] == args.session)
    row = next(csv.DictReader(open(ROOT / "ref/dataset" / e["id"] / "ee_csv.csv")))
    baseline = json.load(open(ROOT / "registration.json"))
    # The image fit identifies camera_from_RCM, not a physical camera offset
    # independently of that RCM. Fix this gauge at the scope's controller tip,
    # with the dVRK -> OpenCV half turn about z. The old effective hand-eye fit
    # put the camera 44 mm outside the scope; absorb it into arm registration.
    camera = np.diag([-1.0, -1.0, 1.0, 1.0])
    spec = importlib.util.spec_from_file_location("generator", REPO / "scripts/dvrk_si_urdf.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    setup = copy.deepcopy(g.SETUPS["home"])
    setup["session"] = e["id"]
    setup["row"] = 0
    for a in ["PSM1", "PSM2", "PSM3", "ECM"]:
        prefix = "suj_ecm_jp" if a == "ECM" else "suj" + a[-1] + "_jp"
        setup["suj"][a] = [float(row.get(f"{prefix}[{i}]", setup["suj"][a][i])) for i in range(len(setup["suj"][a]))]
        setup["joints"][a] = [float(row[f"{a.lower()}_js[{i}]"]) for i in range(4 if a == "ECM" else 6)]
    for a in ["PSM1", "PSM2"]:
        if a in fit["registrations"][args.session]:
            base = np.linalg.inv(camera) @ matrix(fit["registrations"][args.session][a])
        else:
            prefix = "suj" + a[-1] + "_pose"
            base = np.eye(4)
            base[:3, :3] = Rotation.from_quat([float(row[prefix + ".orientation." + c]) for c in "xyzw"]).as_matrix()
            base[:3, 3] = [float(row[prefix + ".position." + c]) for c in "xyz"]
            if a == "PSM2":
                base = np.array(baseline["PSM2_correction_in_logged_ecm"]) @ base
            base = np.linalg.inv(camera) @ np.array(baseline["camera_from_logged_ecm"]) @ base
        setup["rcm_in_camera"][a] = pose(base)
    out = ROOT / "assets/multisession" / args.session
    g.SETUPS["multi"] = setup
    sys.argv = [
        "dvrk_si_urdf.py",
        str(args.dvrk_model),
        "--setup",
        "multi",
        "--out",
        str(out),
    ]
    g.main()
    tree = ET.parse(out / "dvrk_si.urdf")
    robot = tree.getroot()
    geo = fit["geometry"]
    for a, original in [("PSM1", 0.009652), ("PSM2", 0.015)]:
        scale = geo[a + "_length"] / original
        for side in [1, 2]:
            link = robot.find(f"link[@name='{a}_jaw_{side}_link']")
            for mesh in link.findall("visual/geometry/mesh"):
                mesh.set("scale", f"1 {scale:.12g} 1")
            for collision in link.findall("collision"):
                origin = collision.find("origin")
                xyz = np.fromstring(origin.get("xyz"), sep=" ")
                xyz[1] *= scale
                origin.set("xyz", " ".join(map(str, xyz)))
                box = collision.find("geometry/box")
                size = np.fromstring(box.get("size"), sep=" ")
                size[1] *= scale
                box.set("size", " ".join(map(str, size)))
    # Use a content hash so Newton cannot reuse a cached model from an earlier fit.
    content = ET.tostring(robot, encoding="unicode")
    name = "profile_" + hashlib.sha256(content.encode()).hexdigest()[:10] + ".urdf"
    (out / name).write_text(content)
    u = yourdfpy.URDF.load(str(out / name), load_meshes=False)
    u.update_cfg({f"ECM_{j}": setup["joints"]["ECM"][i] for i, j in enumerate(["yaw", "pitch", "insertion", "roll"])})
    mount = np.linalg.inv(u.get_transform("ECM_roll_link")) @ u.get_transform("ECM_camera_link") @ np.linalg.inv(camera)
    profile = {
        "session": args.session,
        "episode": e["id"],
        "asset": str((out / name).relative_to(ROOT)),
        "intrinsics": {k: fit["intrinsics"][k] for k in ["fx", "fy", "cx", "cy"]},
        "camera_in_ecm_roll": pose(mount),
        "camera_anchor": "ECM_camera_link origin; OpenCV half turn about local z; nominal optical centre, not measured stereo calibration",
        "lag_frames": fit["lag_frames"][args.session],
        "geometry": geo,
        "joints": setup["joints"],
        "method": "shared image-fit geometry, per-setup rigid registration; experimental visualization profile",
        "fit_sha256": hashlib.sha256((ROOT / "multisession_registration.json").read_bytes()).hexdigest(),
    }
    (out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(out / "profile.json")


if __name__ == "__main__":
    main()

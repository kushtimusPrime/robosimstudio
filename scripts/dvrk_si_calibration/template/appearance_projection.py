"""Verify appearance preserves the frozen projection model across all fitted setups."""

import csv
import json
from pathlib import Path

import numpy as np
from appearance import visual_asset

root = Path(__file__).resolve().parent


def main():
    import yourdfpy
    from scipy.spatial.transform import Rotation

    fit = json.loads((root / "multisession_registration.json").read_text())
    errs = []
    original = []
    poses = 0
    profiles = sorted((root / "assets/multisession").glob("*/profile.json"))
    for path in profiles:
        p = json.loads(path.read_text())
        source = root / p["asset"]
        variant = visual_asset(source)
        a = yourdfpy.URDF.load(source, load_meshes=False)
        b = yourdfpy.URDF.load(variant, load_meshes=False)
        rows = list(csv.DictReader((root / "ref/dataset" / p["episode"] / "ee_csv.csv").open()))
        mount = np.eye(4)
        mount[:3, :3] = Rotation.from_quat(p["camera_in_ecm_roll"][3:]).as_matrix()
        mount[:3, 3] = p["camera_in_ecm_roll"][:3]
        for o in [o for o in fit["observations"] if o["session"] == p["session"]]:
            idx = np.clip(o["frame"] - p["lag_frames"], 0, len(rows) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(rows) - 1)
            w = idx - lo
            row = {k: float(rows[lo][k]) * (1 - w) + float(rows[hi][k]) * w for k in rows[lo] if rows[lo][k]}
            q = {}
            for arm in ["PSM1", "PSM2", "ECM"]:
                for i, j in enumerate(
                    ["yaw", "pitch", "insertion", "roll"] + ([] if arm == "ECM" else ["wrist_pitch", "wrist_yaw"])
                ):
                    q[arm + "_" + j] = row[f"{arm.lower()}_js[{i}]"]
                if arm != "ECM":
                    q[arm + "_jaw_1"] = p["geometry"][arm + "_jaw_offset"] + p["geometry"][arm + "_jaw_gain"] * (
                        row[arm.lower() + "_jaw"] + 0.35
                    )
            uv_all = []
            for u in [a, b]:
                u.update_cfg(q)
                c = np.linalg.inv(u.get_transform("ECM_roll_link") @ mount)
                arm = o["arm"]
                t = c @ u.get_transform(arm + "_wrist_yaw_link")
                face = -1 if t[2, 2] >= 0 else 1
                pts = [(t @ np.array([0, 0, face * 0.00265, 1]))[:3]]
                for side, sign in [(1, -1), (2, 1)]:
                    pt = np.array([sign * (0.0005 if arm == "PSM1" else 0.0004), p["geometry"][arm + "_length"], 0, 1])
                    pts.append((c @ u.get_transform(arm + f"_jaw_{side}_link") @ pt)[:3])
                pts = np.array(pts)
                uv_all.append(pts[:, :2] / pts[:, 2:] * [p["intrinsics"]["fx"], p["intrinsics"]["fy"]] + [480, 270])
            errs.extend(np.linalg.norm(uv_all[0] - uv_all[1], axis=1))
            original.extend(np.linalg.norm(uv_all[1] - o["projected"], axis=1))
            poses += 1
    result = {
        "profiles": len(profiles),
        "observations": poses,
        "landmarks": len(errs),
        "appearance_projection_change_max_px": float(max(errs)),
        "urdf_vs_optimizer_max_px": float(max(original)),
        "urdf_vs_optimizer_rms_px": float(np.sqrt(np.mean(np.square(original)))),
    }
    assert max(errs) == 0
    (root / "run/appearance/projection_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()

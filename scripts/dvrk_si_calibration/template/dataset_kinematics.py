"""Check the actual URDF's RCM-relative FK against every downloaded CSV row."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yourdfpy

ROOT = Path(__file__).resolve().parent
URDF = ROOT.parents[3] / "assets/embodiments/dvrk_si/dvrk_si.urdf"


def fk(u, data, arm):
    from scipy.spatial.transform import Rotation

    n = len(next(iter(data.values())))
    t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    chain = []
    cur = arm + "_tool_tip_link"
    while cur != arm + "_RCM_link":
        j = next(j for j in u.robot.joints if j.child == cur)
        chain.append(j)
        cur = j.parent
    qnames = {
        arm + "_" + j: f"{arm.lower()}_js[{i}]"
        for i, j in enumerate(["yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw"])
    }
    for j in reversed(chain):
        t = t @ j.origin
        if j.type == "fixed":
            continue
        if j.mimic:
            q = data[qnames[j.mimic.joint]] * j.mimic.multiplier + j.mimic.offset
        else:
            q = data[qnames[j.name]]
        m = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        if j.type == "prismatic":
            m[:, :3, 3] = q[:, None] * j.axis
        else:
            m[:, :3, :3] = Rotation.from_rotvec(q[:, None] * j.axis).as_matrix()
        t = t @ m
    return t


def main():
    from scipy.spatial.transform import Rotation

    u = yourdfpy.URDF.load(str(URDF), load_meshes=False)
    records = []
    failures = []
    paths = sorted((ROOT / "ref/dataset").glob("*/*/*/ee_csv.csv"))
    for i, p in enumerate(paths):
        try:
            rows = list(csv.DictReader(p.open()))
            keys = [k for k in rows[0] if k.startswith(("psm1", "psm2", "suj1_jp", "suj2_jp", "suj_ecm_jp", "ecm_js"))]
            data = {k: np.array([float(r[k]) for r in rows]) for k in keys}
            result = {"id": str(p.parent.relative_to(ROOT / "ref/dataset")), "rows": len(rows), "arms": {}}
            for arm in ["PSM1", "PSM2"]:
                t = fk(u, data, arm)
                prefix = arm.lower() + "_rcm_pose"
                xyz = np.array([data[prefix + ".position." + a] for a in "xyz"]).T
                quat = np.array([data[prefix + ".orientation." + a] for a in "xyzw"]).T
                pos = np.linalg.norm(t[:, :3, 3] - xyz, axis=1) * 1000
                angle = (
                    Rotation.from_matrix(
                        t[:, :3, :3].transpose(0, 2, 1) @ Rotation.from_quat(quat).as_matrix()
                    ).magnitude()
                    * 180
                    / np.pi
                )
                result["arms"][arm] = {
                    "rms_mm": float(np.sqrt(np.mean(pos**2))),
                    "max_mm": float(pos.max()),
                    "max_deg": float(angle.max()),
                    "worst_row": int(np.argmax(pos)),
                }
            result["setup"] = [
                float(data[f"{a}[{j}]"][0]) for a in ["suj1_jp", "suj2_jp", "suj_ecm_jp"] for j in range(4)
            ]
            result["ecm_motion"] = {str(j): float(np.ptp(data[f"ecm_js[{j}]"])) for j in range(4)}
            records.append(result)
        except Exception as ex:
            failures.append({"path": str(p), "error": str(ex)})
        if i % 100 == 0:
            print(i, len(paths), len(failures), flush=True)
    summary = {
        "episodes": len(records),
        "rows": sum(r["rows"] for r in records),
        "failed": len(failures),
        "max_position_mm": max(v["max_mm"] for r in records for v in r["arms"].values()),
        "max_rotation_deg": max(v["max_deg"] for r in records for v in r["arms"].values()),
        "episodes_over_1mm": sum(max(v["max_mm"] for v in r["arms"].values()) > 1 for r in records),
        "episodes_over_1deg": sum(max(v["max_deg"] for v in r["arms"].values()) > 1 for r in records),
    }
    result = {
        "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "scope": "RCM-relative FK only. Does not validate image calibration or setup registration.",
        "summary": summary,
        "episodes": records,
        "failures": failures,
    }
    (ROOT / "dataset_kinematics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

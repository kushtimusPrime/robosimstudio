"""Shared endoscope intrinsics and tool dimensions with per-setup rigid registration.

Fits only manually selected training pixels. Reports held-out frames separately;
image-only feature proposals are deliberately excluded from the optimizer.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yourdfpy

ROOT = Path(__file__).resolve().parent
URDF = ROOT.parents[3] / "assets/embodiments/dvrk_si/dvrk_si.urdf"
JOINTS = ["yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw"]


def load(label_file="multisession_landmarks.json"):
    labels = json.load(open(ROOT / "ref" / label_file))
    tracks = {}
    observations = []
    u = yourdfpy.URDF.load(str(URDF), load_meshes=False)
    for episode in labels["episodes"]:
        eid = episode["id"]
        p = ROOT / "ref/dataset" / eid / "ee_csv.csv"
        rows = list(csv.DictReader(p.open()))
        armtracks = {}
        for arm in ["PSM1", "PSM2"]:
            if not any(arm in frame for frame in episode["frames"].values()):
                continue
            base = np.linalg.inv(u.get_transform(arm + "_RCM_link"))
            ts = []
            jaws = []
            for row in rows:
                u.update_cfg({arm + "_" + j: float(row[f"{arm.lower()}_js[{i}]"]) for i, j in enumerate(JOINTS)})
                ts.append(base @ u.get_transform(arm + "_wrist_yaw_link"))
                jaws.append(float(row[arm.lower() + "_jaw"]))
            armtracks[arm] = (np.array(ts), np.array(jaws))
        tracks[eid] = armtracks
        for f, frame in episode["frames"].items():
            for arm in armtracks:
                if arm not in frame:
                    continue
                pixels = np.array([p if p else [np.nan, np.nan] for p in frame[arm]])
                if not np.any(np.isfinite(pixels)):
                    continue
                observations.append(
                    {
                        "episode": eid,
                        "session": episode["session"],
                        "arm": arm,
                        "frame": int(f),
                        "split": frame["split"],
                        "pixels": pixels,
                    }
                )
    return labels, tracks, observations


def points(track, frame, lag, geometry, arm, extrinsic=None):
    from scipy.spatial.transform import Rotation

    ts, jaws = track
    i = np.clip(frame - lag, 0, len(ts) - 1)
    lo = int(i)
    hi = min(lo + 1, len(ts) - 1)
    w = i - lo
    t = ts[lo] * (1 - w) + ts[hi] * w
    raw = jaws[lo] * (1 - w) + jaws[hi] * w
    aidx = 0 if arm == "PSM1" else 1
    length = geometry[aidx]
    offset, gain = geometry[2 + 2 * aidx : 4 + 2 * aidx]
    angle = offset + gain * (raw + 0.35)
    width = 0.0005 if arm == "PSM1" else 0.0004
    face = -1 if arm == "PSM1" else 1
    if extrinsic is not None:
        normal = (Rotation.from_rotvec(extrinsic[:3]).as_matrix() @ t[:3, :3])[2, 2]
        face = -1 if normal >= 0 else 1
    p = [[0, 0, face * 0.00265]]
    for sign in [1, -1]:
        c, s = np.cos(sign * angle), np.sin(sign * angle)
        p.append([c * (-sign * width) - s * length, s * (-sign * width) + c * length, 0])
    return np.array(p) @ t[:3, :3].T + t[:3, 3]


def project(p, extrinsic, intrinsics):
    from scipy.spatial.transform import Rotation

    p = p @ Rotation.from_rotvec(extrinsic[:3]).as_matrix().T + extrinsic[3:]
    fx, fy, cx, cy, k1, k2 = intrinsics
    xy = p[:, :2] / np.maximum(p[:, 2:], 0.001)
    r2 = np.sum(xy**2, axis=1)
    xy *= np.clip(1 + k1 * r2 + k2 * r2 * r2, 0.2, 5)[:, None]
    return xy * [fx, fy] + [cx, cy], p[:, 2]


def errors(uv, pix):
    e = uv - pix
    flipped = uv[[0, 2, 1]] - pix
    return flipped if np.nansum(flipped**2) < np.nansum(e**2) else e


def main():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fixed", "focal", "distortion", "geometry", "bundle"], default="focal")
    args = parser.parse_args()
    _labels, tracks, obs = load()
    keys = sorted({(o["session"], o["arm"]) for o in obs})
    sessions = sorted({k[0] for k in keys})
    train = [o for o in obs if o["split"] == "train"]
    intr0 = np.array([866.3743, 866.3743, 480, 270, 0, 0])
    geo0 = np.array([0.009652, 0.015, 0.0424, 0.4365, 0.02, 0.44])
    lag0 = 4.61
    # Independent PnP starts for each arm; all motion remains from URDF FK.
    extr = []
    for key in keys:
        selected = [o for o in train if (o["session"], o["arm"]) == key]
        ps = []
        uv = []
        for o in selected:
            p = points(tracks[o["episode"]][o["arm"]], o["frame"], lag0, geo0, o["arm"])
            valid = np.isfinite(o["pixels"][:, 0])
            ps.extend(p[valid])
            uv.extend(o["pixels"][valid])
        K = np.array([[intr0[0], 0, 480], [0, intr0[1], 270], [0, 0, 1.0]])
        ok, r, t = cv2.solvePnP(np.array(ps, dtype=float), np.array(uv, dtype=float), K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError(f"PnP failed {key}")
        e = np.r_[r.ravel(), t.ravel()]

        def initres(e, selected=selected):
            out = []
            for o in selected:
                p = points(tracks[o["episode"]][o["arm"]], o["frame"], lag0, geo0, o["arm"], e)
                uv, z = project(p, e, intr0)
                err = errors(uv, o["pixels"])
                out.extend(err[np.isfinite(err)])
                out.extend(np.minimum(z - 0.005, 0) * 1e5)
            return out

        seeds = [e]
        # The logged RCM pose provides a physically meaningful start, independent of pixel fitting.
        item = selected[0]
        eid = item["episode"]
        csvpath = ROOT / "ref/dataset" / eid / "ee_csv.csv"
        row = next(csv.DictReader(csvpath.open()))
        prefix = "suj" + key[1][-1] + "_pose"
        base = np.eye(4)
        base[:3, :3] = Rotation.from_quat([float(row[prefix + ".orientation." + c]) for c in "xyzw"]).as_matrix()
        base[:3, 3] = [float(row[prefix + ".position." + c]) for c in "xyz"]
        reg = json.load(open(ROOT / "registration.json"))
        cam = np.array(reg["camera_from_logged_ecm"])
        if key[1] == "PSM2":
            cam = cam @ np.array(reg["PSM2_correction_in_logged_ecm"])
        prior = cam @ base
        v = np.r_[Rotation.from_matrix(prior[:3, :3]).as_rotvec(), prior[:3, 3]]
        seeds.append(v)
        rng = np.random.default_rng(42)
        for _ in range(8):
            seeds.append(v + rng.normal(size=6) * [0.15, 0.15, 0.15, 0.005, 0.005, 0.005])
        fits = [least_squares(initres, seed, loss="soft_l1", f_scale=10, max_nfev=400) for seed in seeds]
        fit = min(fits, key=lambda f: f.cost)
        print("init", key, fit.cost, flush=True)
        extr.extend(fit.x)
    n_ext = len(extr)
    n_lag = len(sessions)
    x0 = np.r_[extr, [lag0] * n_lag, intr0, geo0]
    lower = np.full(len(x0), -np.inf)
    upper = -lower
    lower[n_ext : n_ext + n_lag] = 0
    upper[n_ext : n_ext + n_lag] = 12
    ii = n_ext + n_lag
    gi = ii + 6
    lower[ii:gi] = [400, 400, 400, 200, -0.8, -0.8]
    upper[ii:gi] = [1600, 1600, 560, 340, 0.8, 0.8]
    lower[gi:] = [0.008, 0.012, -0.1, 0.25, -0.1, 0.25]
    upper[gi:] = [0.012, 0.019, 0.2, 0.75, 0.3, 0.75]
    active = np.ones(len(x0), bool)
    if args.model == "fixed":
        active[ii:gi] = False
    elif args.model in ["focal", "geometry"]:
        active[ii + 2 : gi] = False
    if args.model not in ["geometry", "bundle"]:
        active[gi:] = False

    def unpack(v):
        x = x0.copy()
        x[active] = v
        return x

    def residual(v):
        x = unpack(v)
        out = []
        for o in train:
            ki = keys.index((o["session"], o["arm"]))
            lag = x[n_ext + sessions.index(o["session"])]
            p = points(tracks[o["episode"]][o["arm"]], o["frame"], lag, x[gi:], o["arm"], x[6 * ki : 6 * ki + 6])
            uv, z = project(p, x[6 * ki : 6 * ki + 6], x[ii:gi])
            err = errors(uv, o["pixels"])
            out.extend(err[np.isfinite(err)])
            out.extend(np.minimum(z - 0.005, 0) * 1e5)
        if args.model in ["distortion", "bundle"]:
            out.extend((x[ii + 2 : gi] - intr0[2:]) / [60, 40, 0.3, 0.3])
        if args.model in ["geometry", "bundle"]:
            out.extend((x[gi:] - geo0) / [0.001, 0.002, 0.1, 0.2, 0.15, 0.2])
        return np.array(out)

    fit = least_squares(
        residual,
        x0[active],
        bounds=(lower[active], upper[active]),
        loss="soft_l1",
        f_scale=8,
        x_scale="jac",
        max_nfev=700,
    )
    x = unpack(fit.x)
    records = []
    for o in obs:
        ki = keys.index((o["session"], o["arm"]))
        lag = x[n_ext + sessions.index(o["session"])]
        p = points(tracks[o["episode"]][o["arm"]], o["frame"], lag, x[gi:], o["arm"], x[6 * ki : 6 * ki + 6])
        uv, _z = project(p, x[6 * ki : 6 * ki + 6], x[ii:gi])
        err = errors(uv, o["pixels"])
        norms = np.linalg.norm(err, axis=1)
        record = {k: v for k, v in o.items() if k != "pixels"}
        record.update(
            projected=uv.tolist(),
            observed=[[float(z) for z in p] if np.all(np.isfinite(p)) else None for p in o["pixels"]],
            errors_px=[float(v) if np.isfinite(v) else None for v in norms],
        )
        records.append(record)
    metrics = {}
    for split in ["train", "validation"]:
        vals = [v for o in records if o["split"] == split for v in o["errors_px"] if v is not None]
        metrics[split] = {
            "landmarks": len(vals),
            "rms_px": float(np.sqrt(np.mean(np.square(vals)))),
            "median_px": float(np.median(vals)),
            "p95_px": float(np.percentile(vals, 95)),
            "max_px": float(max(vals)),
        }
    result = {
        "model": args.model,
        "success": bool(fit.success),
        "message": fit.message,
        "intrinsics": dict(zip(["fx", "fy", "cx", "cy", "k1", "k2"], x[ii:gi], strict=True)),
        "geometry": dict(
            zip(
                ["PSM1_length", "PSM2_length", "PSM1_jaw_offset", "PSM1_jaw_gain", "PSM2_jaw_offset", "PSM2_jaw_gain"],
                x[gi:],
                strict=True,
            )
        ),
        "lag_frames": dict(zip(sessions, x[n_ext:ii], strict=True)),
        "registrations": {
            s: {a: x[6 * keys.index((s, a)) : 6 * keys.index((s, a)) + 6].tolist() for ss, a in keys if ss == s}
            for s in sessions
        },
        "metrics": metrics,
        "observations": records,
    }
    (ROOT / "run" / f"multisession_{args.model}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["observations", "registrations"]}, indent=2))


if __name__ == "__main__":
    main()

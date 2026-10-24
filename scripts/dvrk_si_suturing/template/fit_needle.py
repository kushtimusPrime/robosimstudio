# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Fit needle initial geometry/state from manually traced pre-release images.

The camera, lag and robot registration are never optimization variables.
This is an image-fit hypothesis, not measured needle dimensions or physics.
Frame 141 is reserved for evaluation; the dataset's final episodes are unused.
"""

import json

import numpy as np
from geometry import ROOT, camera_and_robot, recorded
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

ANNOTATIONS = {
    0: {"curve": [[656, 336], [645, 383], [623, 434], [593, 479], [558, 518], [528, 539]], "swage": [637, 273]},
    70: {
        "curve": [[544, 76], [540, 105], [535, 130], [527, 157], [515, 190], [501, 218]],
        "tip": [501, 218],
        "swage": [476, 29],
    },
    141: {"curve": [[520, 190], [518, 212], [515, 235], [509, 258]], "tip": [509, 258], "swage": [413, 183]},
    211: {"curve": [[528, 146], [528, 176], [524, 211], [518, 246], [510, 275]], "swage": [480, 119]},
}


def run():
    params = json.loads((ROOT / "parameters.json").read_text())
    profile = json.loads((ROOT / "assets/multisession/setup_017/profile.json").read_text())
    _cam, u = camera_and_robot(profile, params["episode"])
    K = profile["intrinsics"]
    transforms = {}
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_quat(profile["camera_in_ecm_roll"][3:]).as_matrix()
    mount[:3, 3] = profile["camera_in_ecm_roll"][:3]
    for frame in ANNOTATIONS:
        u.update_cfg(recorded(profile, params["episode"], frame))
        transforms[frame] = np.linalg.inv(u.get_transform("ECM_roll_link") @ mount) @ u.get_transform(
            "PSM1_wrist_yaw_link"
        )

    def project(p, frame):
        c = p @ transforms[frame][:3, :3].T + transforms[frame][:3, 3]
        return c[:, :2] / c[:, 2:] * [K["fx"], K["fy"]] + [K["cx"], K["cy"]]

    def triangulate(kind, frames):
        A = []
        b = []
        for f in frames:
            if kind not in ANNOTATIONS[f]:
                continue
            uv = np.array(ANNOTATIONS[f][kind])
            xy = (uv - [K["cx"], K["cy"]]) / [K["fx"], K["fy"]]
            M = np.array([[1, 0, -xy[0]], [0, 1, -xy[1]]]) @ transforms[f][:3]
            A.extend(M[:, :3])
            b.extend(-M[:, 3])
        return np.linalg.lstsq(A, b, rcond=None)[0]

    train = [0, 70, 211]
    S = triangulate("swage", train)
    T = triangulate("tip", train)
    axis = (S - T) / np.linalg.norm(T - S)
    y = np.cross(axis, [0, 0, 1])
    y /= np.linalg.norm(y)
    z = np.cross(axis, y)
    base = np.column_stack([axis, y, z])
    curves = [(f, np.asarray(ANNOTATIONS[f]["curve"])) for f in train]

    def curve(x, theta):
        R = np.exp(x[6])
        p = np.column_stack((R * (np.cos(theta) - 1), R * np.sin(theta), theta * 0))
        return p @ Rotation.from_rotvec(x[3:6]).as_matrix().T + x[:3]

    def residual(x):
        out = []
        offset = 7
        for f, pts in curves:
            pred = project(curve(x, x[offset : offset + len(pts)]), f)
            out.extend((pred - pts).ravel())
            offset += len(pts)
            if "tip" in ANNOTATIONS[f]:
                out.extend((project(curve(x, np.array([np.pi])), f)[0] - ANNOTATIONS[f]["tip"]) * 2)
            # Swage is partly obscured; down-weight its rough annotation.
            out.extend((project(curve(x, np.array([0.0])), f)[0] - ANNOTATIONS[f]["swage"]) * 0.4)
        return np.array(out)

    fits = []
    lower = np.r_[[-0.03, -0.01, -0.03], [-10] * 3, np.log(0.004), np.zeros(sum(len(p) for _, p in curves))]
    upper = np.r_[[0.03, 0.03, 0.03], [10] * 3, np.log(0.025), np.full(sum(len(p) for _, p in curves), np.pi)]
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        rot = base @ Rotation.from_euler("x", angle).as_matrix()
        x = np.r_[S, Rotation.from_matrix(rot).as_rotvec(), np.log(np.clip(np.linalg.norm(T - S) / 2, 0.0041, 0.024))]
        theta = np.linspace(0, np.pi, 400)
        guesses = []
        for f, pts in curves:
            guesses.extend(theta[np.argmin(cdist(pts, project(curve(x, theta), f)), axis=1)])
        initial = np.clip(np.r_[x, guesses], lower + 1e-8, upper - 1e-8)
        fit = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=3, max_nfev=500)
        fits.append(fit)
    fit = min(fits, key=lambda f: f.cost)
    x = fit.x
    points = curve(x, np.linspace(0, np.pi, 1000))
    report = {
        "status": "image-fit hypothesis, not validated mechanics",
        "train_frames": train,
        "reserved_frames": [141],
        "needle_swage_in_wrist_m": x[:3].tolist(),
        "needle_rotation_in_wrist_xyzw": Rotation.from_rotvec(x[3:6]).as_quat().tolist(),
        "circle_radius_m": float(np.exp(x[6])),
        "arc_length_m": float(np.pi * np.exp(x[6])),
        "assumed_arc_angle_rad": float(np.pi),
        "annotations_px": ANNOTATIONS,
        "warning": "Manual traces and partly occluded swage. Fixed-registration image error can bias inferred geometry. No force or material identification.",
        "frames": {},
    }
    tiles = []
    for f, anno in ANNOTATIONS.items():
        uv = project(points, f)
        errors = np.min(cdist(np.asarray(anno["curve"]), uv), axis=1)
        report["frames"][f] = {"curve_RMS_px": float(np.sqrt(np.mean(errors**2)))}
        if "tip" in anno:
            report["frames"][f]["tip_error_px"] = float(np.linalg.norm(uv[-1] - anno["tip"]))
        im = Image.open(ROOT / "ref/dataset" / params["episode"] / f"left_img_dir/frame{f:06d}_left.jpg").convert("RGB")
        d = ImageDraw.Draw(im)
        # Traces are cyan, fitted projection yellow. No real pixels become textures.
        d.line([tuple(p) for p in anno["curve"]], fill=(0, 255, 255), width=3)
        valid = np.all((uv > [-200, -200]) & (uv < [1160, 740]), axis=1)
        d.line([tuple(p) for p in uv[valid]], fill=(255, 230, 0), width=2)
        d.text((20, 20), f"frame {f}: " + ("reserved" if f == 141 else "fit"), fill="white")
        tiles.append(im.resize((480, 270)))
    sheet = Image.new("RGB", (960, 540))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 2) * 480, (i // 2) * 270))
    sheet.save(ROOT / "run/needle_fit.png")
    (ROOT / "run/needle_fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "annotations_px"}, indent=2))


if __name__ == "__main__":
    run()

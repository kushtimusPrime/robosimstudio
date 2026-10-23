"""Capture renderer depth and frozen frame/physics state on the live page."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def capture(sim, session, profile, name):
    from PIL import Image

    folder = ROOT / "run/appearance"
    folder.mkdir(parents=True, exist_ok=True)
    frame = sim.render("endoscope", channels=("rgb", "depth"))
    Image.fromarray(frame["rgb"][..., :3]).save(folder / (name + ".png"))
    arrays = {"depth": frame["depth"], "joint_q": sim.state.joint_q.numpy(), "body_q": sim.state.body_q.numpy()}
    for key in (
        "body_mass",
        "body_inertia",
        "body_com",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_effort_limit",
        "joint_velocity_limit",
        "joint_mimic_joint",
        "joint_mimic_coeffs",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
    ):
        arrays[key] = getattr(sim.model, key).numpy()
    camera = sim.cameras[0]
    arrays["camera_pose"] = camera.pose_on_body(arrays["body_q"])
    for name_arm, panel in session.arms.items():
        arrays[name_arm + "_gizmo_position"] = np.array(panel.gizmo.position)
        arrays[name_arm + "_gizmo_wxyz"] = np.array(panel.gizmo.wxyz)
    for key, frustum in session.live._frusta.items():
        arrays["frustum_" + str(key) + "_position"] = np.array(frustum.position)
        arrays["frustum_" + str(key) + "_wxyz"] = np.array(frustum.wxyz)
    np.savez_compressed(folder / (name + ".npz"), **arrays)
    (folder / (name + ".json")).write_text(
        json.dumps(
            {
                "camera": asdict(camera),
                "profile": profile,
                "body_labels": sim.model.body_label,
                "joint_labels": sim.model.joint_label,
            },
            indent=2,
        )
    )
    return str(folder / (name + ".png"))


def compare():
    from appearance_capture import CASES
    from scipy.ndimage import distance_transform_edt

    results = []
    for setup, frame, split in CASES:
        stem = f"{setup}_{frame:03d}"
        base = ROOT / "run/appearance"
        a, b = [np.load(base / f"{v}_{stem}.npz") for v in ("baseline", "updated")]
        assert a.files == b.files
        for key in a.files:
            if key != "depth":
                assert np.array_equal(a[key], b[key]), (stem, key, np.max(np.abs(a[key] - b[key])))
        assert (base / f"baseline_{stem}.json").read_bytes() == (base / f"updated_{stem}.json").read_bytes()
        # The empty-scene backdrop returns depth too; isolate nearby tools.
        ma, mb = [(d > 0) & (d < 0.2) for d in (a["depth"], b["depth"])]
        assert not (mb & ~ma).any(), (stem, "Unexpected outward silhouette change")
        assert distance_transform_edt(~mb)[ma & ~mb].max(initial=0) <= 1.5
        common = ma & mb
        delta = np.abs(a["depth"][common] - b["depth"][common])
        results.append(
            {
                "setup": setup,
                "frame": frame,
                "split": split,
                "state_camera_physics_equal": True,
                "tool_depth_cutoff_m": 0.2,
                "silhouette_added_pixels": int((mb & ~ma).sum()),
                "silhouette_max_inward_px": float(distance_transform_edt(~mb)[ma & ~mb].max(initial=0)),
                "silhouette_iou": float((ma & mb).sum() / (ma | mb).sum()),
                "silhouette_changed_pixels": int((ma ^ mb).sum()),
                "common_pixels_depth_change_over_50um": int((delta > 0.00005).sum()),
                "common_depth_p99_m": float(np.quantile(delta, 0.99)),
                "common_depth_max_m": float(delta.max()),
            }
        )
    frozen = json.loads((base / "frozen_sources.json").read_text())
    for path, digest in frozen.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    result = {"frozen_source_files_unchanged": len(frozen), "cases": results}
    (base / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    compare()

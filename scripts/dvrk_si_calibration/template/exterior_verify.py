"""Check actual Viser mesh-instance attachments after paused recorded replay."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def check_attachments(sim, session, profile):
    from multisession_preview import replay
    from scipy.spatial.transform import Rotation

    records = []
    for frame in (0, 94, 188):
        step = sim.step_count
        replay(sim, session, profile, frame)
        assert session.paused and sim.step_count == step
        bq = sim.state.body_q.numpy()
        sx = sim.model.shape_transform.numpy()
        bodies = sim.model.shape_body.numpy()
        errors = []
        rotations = []
        for key, handles in session.viewer._pbr_handles.items():
            shapes = session.viewer._batch_shapes(key)
            if shapes is None:
                continue
            for node, handle in handles.items():
                rows = session.viewer._pbr_groups[key][int(node.rsplit("/m", 1)[1])][1]
                for j, row in enumerate(rows):
                    shape = int(shapes[row])
                    body = int(bodies[shape])
                    if body < 0:
                        continue
                    rotation = Rotation.from_quat(bq[body, 3:])
                    expected_pos = bq[body, :3] + rotation.apply(sx[shape, :3])
                    expected_rot = rotation * Rotation.from_quat(sx[shape, 3:])
                    got_rot = Rotation.from_quat(np.asarray(handle.batched_wxyzs[j])[[1, 2, 3, 0]])
                    errors.append(float(np.linalg.norm(np.asarray(handle.batched_positions[j]) - expected_pos)))
                    rotations.append(float(np.degrees((got_rot.inv() * expected_rot).magnitude())))
        assert len(errors) > 100 and max(errors) < 1e-6 and max(rotations) < 1e-4
        records.append(
            {
                "frame": frame,
                "moving_mesh_instances": len(errors),
                "max_position_error_m": max(errors),
                "max_rotation_error_deg": max(rotations),
                "paused_replay": True,
            }
        )
    replay(sim, session, profile, 94)
    (ROOT / "run/exterior/attachments.json").write_text(json.dumps(records, indent=2) + "\n")
    return records

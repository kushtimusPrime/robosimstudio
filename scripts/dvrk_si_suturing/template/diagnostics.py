# SPDX-License-Identifier: Apache-2.0
"""Measurements over actual host physics, never substitutes for object dynamics."""

import json
from pathlib import Path

import numpy as np


def measure(sim, needle_ref, strand, tissue):
    bridge = needle_ref.entity.bridge
    entries = {e["entity"]: e for e in bridge.actors}
    result = {
        "time_s": float(sim.pipeline.sim_time),
        "finite": bool(np.isfinite(sim.state.particle_q.numpy()).all()),
        "diverged_worlds": list(bridge.diverged_worlds),
        "actors": {},
    }
    for name, e in entries.items():
        points = np.asarray(e["read"]()).reshape(-1, 3)
        result["actors"][name] = {
            "centroid_m": points.mean(axis=0).tolist(),
            "mass_kg": float(e["actor"].get_mass()),
            "dofs": e["actor"].get_num_dofs(),
        }
    points = np.asarray(entries["thread"]["read"]())
    rest = strand.entity.nodes()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    initial = np.linalg.norm(np.diff(rest, axis=0), axis=1)
    result["thread_max_abs_strain"] = float(np.max(np.abs(lengths / initial - 1)))
    needle = entries["needle"]
    tr = needle["actor"].get_root_transform()
    from robosimstudio.core.math import quat_to_mat

    q = quat_to_mat(np.asarray(tr.rotation))
    # Swage is the initial thread start expressed in needle local coordinates.
    swage = (rest[0] - needle["rest_centre"]) @ q.T + np.asarray(tr.translation)
    result["swage_gap_m"] = float(np.linalg.norm(points[0] - swage))
    stats = bridge.solver_stats()
    result["solver"] = {
        k: str(getattr(stats, k)) for k in dir(stats) if not k.startswith("_") and not callable(getattr(stats, k))
    }
    return result


def save(sim, needle_ref, strand, tissue, path):
    result = measure(sim, needle_ref, strand, tissue)
    Path(path).write_text(json.dumps(result, indent=2))
    return result

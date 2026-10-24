# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Optional suturing contact profile; preserves every non-collision URDF field.

User authorized contact changes on 2026-10-23, conditional on IK regression
checks. Convex pieces follow the approved visible jaws and leave the grasp gap
open. Fine serration valleys below hull resolution remain frictional proxies.
"""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
LINKS = tuple(f"{a}_jaw_{j}_link" for a in ("PSM1", "PSM2") for j in (1, 2))
VERSION = 1


def without_contacts(root):
    root = copy.deepcopy(root)
    for link in root.findall("link"):
        if link.get("name") in LINKS:
            for c in link.findall("collision"):
                link.remove(c)
    # Paths may be made absolute to keep separate generated assets portable.
    for m in root.findall(".//mesh"):
        m.attrib.pop("filename", None)
    return ET.tostring(root)


def asset(source):
    source = Path(source).resolve()
    raw = source.read_bytes()
    key = hashlib.sha256(raw + str(VERSION).encode()).hexdigest()[:12]
    out = ROOT / "assets/suturing_contacts" / key
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "suturing.urdf"
    if dst.exists():
        return str(dst)
    tree = ET.fromstring(raw)
    baseline = without_contacts(tree)
    for m in tree.findall(".//mesh"):
        p = Path(m.attrib["filename"])
        m.set("filename", str((source.parent / p).resolve() if not p.is_absolute() else p))
    summary = {}
    for link in tree.findall("link"):
        name = link.get("name")
        if name not in LINKS:
            continue
        pieces = []
        for vis in link.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            mesh = trimesh.load(m.attrib["filename"], force="mesh", process=True)
            mesh.apply_scale(np.fromstring(m.get("scale", "1 1 1"), sep=" "))
            o = vis.find("origin")
            if o is not None:
                from scipy.spatial.transform import Rotation

                x = np.eye(4)
                x[:3, :3] = Rotation.from_euler("xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                x[:3, 3] = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
                mesh.apply_transform(x)
            pieces.append(mesh)
        mesh = trimesh.util.concatenate(pieces)
        bounds = mesh.bounds
        # 0.5 mm slices resolve the narrow blade separately from the broad pivot.
        bins = np.unique(np.r_[bounds[0, 1] - 1e-8, np.arange(-0.0015, bounds[1, 1], 0.0005), bounds[1, 1] + 1e-8])
        for c in link.findall("collision"):
            link.remove(c)
        records = []
        for k, (lo, hi) in enumerate(pairwise(bins)):
            part = mesh.slice_plane([0, lo, 0], [0, 1, 0]).slice_plane([0, hi, 0], [0, -1, 0])
            if len(part.vertices) < 4:
                continue
            try:
                hull = part.convex_hull
            except Exception as e:
                raise RuntimeError(f"{name} contact slice {k} has invalid geometry") from e
            if hull.volume < 1e-15:
                continue
            file = out / f"{name}_{k:02d}.stl"
            hull.export(file)
            c = ET.SubElement(link, "collision", {"name": f"suture_contact_{k:02d}"})
            g = ET.SubElement(c, "geometry")
            ET.SubElement(g, "mesh", {"filename": str(file)})
            records.append({"file": file.name, "volume_m3": float(hull.volume), "slice_m": [float(lo), float(hi)]})
        summary[name] = records
    assert without_contacts(tree) == baseline, "Non-collision URDF content changed"
    ET.indent(tree)
    ET.ElementTree(tree).write(dst, encoding="utf-8", xml_declaration=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "changed_links": summary,
                "unchanged": "visuals, inertials, joints, limits, mimic, frames",
                "approximation": "0.5 mm longitudinal convex slices; fine grooves are not fully resolved",
            },
            indent=2,
        )
    )
    return str(dst)

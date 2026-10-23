"""Optional Si exterior visuals from detached-instrument and whole-robot scans.

Never edits canonical meshes, physics, calibrated distal visuals or camera data.
The original JHU instrument housing is retained separately as welded source CAD.
"""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
from appearance import physical_tree

from robosimstudio.core.render.finish import Finish, register_color_finish

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets/exterior"
SOURCE = ROOT / "assets/exterior_source/housing_original.npz"
CARRIAGES = True  # Visual-only sliding supports; disable for the previous exterior.
# Material hypotheses under neutral illumination; no photographed highlights.
SURFACES = {
    "ivory": ((0.84, 0.837, 0.805), 0.0, 0.32),
    "blue": ((0.56, 0.76, 0.84), 0.0, 0.26),
    "base": ((0.26, 0.28, 0.31), 0.0, 0.43),
    "black": ((0.045, 0.050, 0.056), 0.0, 0.34),
    "seam": ((0.20, 0.21, 0.215), 0.0, 0.58),
    "steel": ((0.67, 0.68, 0.70), 1.0, 0.24),
    "satin": ((0.59, 0.60, 0.615), 1.0, 0.36),
    "drive": ((0.80, 0.81, 0.79), 0.0, 0.39),
}


def register_materials():
    for color, metal, rough in SURFACES.values():
        register_color_finish(color, Finish(metallic=metal, roughness=rough))
    for arm in ("PSM1", "PSM2", "PSM3"):
        # Separate RGB lookup keys; the PNG replaces this color in both backends.
        color = (0.56, 0.76, 0.86 + int(arm[-1]) * 0.01)
        raw = (ROOT / f"assets/exterior_source/{arm}_label.png").read_bytes()
        texture = OUT / "textures" / (hashlib.sha256(raw).hexdigest()[:16] + ".png")
        texture.parent.mkdir(parents=True, exist_ok=True)
        if not texture.exists():
            texture.write_bytes(raw)
        register_color_finish(
            color,
            Finish(
                roughness=0.26,
                texture=str(texture),
                uv_scale=(1 / 0.09524, 1 / 0.0648),
                uv_offset=(0.04404 / 0.09524, 0.5),
            ),
        )


def crease_normals(mesh, angle=35):
    """Weld curved patches, retaining hard boundaries; triangles stay unchanged."""
    import trimesh

    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=10)
    adjacency = mesh.face_adjacency[mesh.face_adjacency_angles < np.deg2rad(angle)]
    groups = trimesh.graph.connected_components(adjacency, nodes=np.arange(len(mesh.faces)))
    return trimesh.util.concatenate([mesh.submesh([g], append=True, repair=False) for g in groups])


def tube_path(points, radius, sides=8):
    """Small surface seam; visual-only geometry in its original link coordinates."""
    import trimesh

    points = np.asarray(points)
    pieces = []
    for a, b in pairwise(points):
        if np.linalg.norm(b - a) > 1e-7:
            pieces.append(trimesh.creation.cylinder(radius, segment=[a, b], sections=sides))
    return trimesh.util.concatenate(pieces) if pieces else None


def cage_plate(angle0, angle1, z0, z1):
    """A thin curved cage strip; black CAD beneath reads as an open slot."""
    import trimesh

    angles = np.linspace(angle0, angle1, 13)
    vertices = np.array(
        [(r * np.cos(a), r * np.sin(a), z) for z in (z0, z1) for r in (0.03145, 0.03205) for a in angles]
    )
    n = len(angles)
    faces = []
    for i in range(n - 1):
        for a, b, c, d in (
            (i, i + 1, n + i + 1, n + i),
            (2 * n + i, 3 * n + i, 3 * n + i + 1, 2 * n + i + 1),
            (i, 2 * n + i, 2 * n + i + 1, i + 1),
            (n + i, n + i + 1, 3 * n + i + 1, 3 * n + i),
        ):
            faces.extend(((a, b, c), (a, c, d)))
    for i in (0, n - 1):
        faces.extend(((i, n + i, 3 * n + i), (i, 3 * n + i, 2 * n + i)))
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.fix_normals()
    return mesh


def rounded_plate(length, width, y0, y1, radius, bevel, x=0.0):
    """Beveled molded plate in housing CAD coordinates: shaft x, face normal y."""
    import trimesh

    points = []
    for y, inset in ((y0, bevel), (y0 + bevel, 0), (y1 - bevel, 0), (y1, bevel)):
        for sx, sz, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
            angles = np.deg2rad(np.linspace(start, start + 90, 13))
            for angle in angles:
                points.append(
                    (
                        x + sx * (length / 2 - radius) + (radius - inset) * np.cos(angle),
                        y,
                        sz * (width / 2 - radius) + (radius - inset) * np.sin(angle),
                    )
                )
    return crease_normals(trimesh.convex.convex_hull(np.asarray(points)))


def carriage_meshes():
    """Visible support envelope, not an inferred model of hidden drive hardware.

    The existing rail face is y=-55.8805 mm in housing mesh coordinates;
    drive discs reach y=-18.161 mm. Neither attachment is moved. The scan
    supports a rounded cream carriage and thin mounting plate. Their outline
    dimensions and bevels are estimates; the 38 mm depth closes the CAD gap.
    """
    yield "body", "ivory", rounded_plate(0.108, 0.064, -0.0562, -0.0246, 0.008, 0.0012, x=0.002)
    yield "joint", "seam", rounded_plate(0.105, 0.066, -0.0250, -0.0238, 0.007, 0.0003, x=0.002)
    yield "plate", "ivory", rounded_plate(0.110, 0.072, -0.0242, -0.0179, 0.007, 0.0008, x=0.002)


def visual_asset(source):
    import trimesh

    register_materials()
    source = Path(source)
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes() + SOURCE.read_bytes())
    digest.update((ROOT / "assets/exterior_source/cover_bevels.npz").read_bytes())
    for arm in ("PSM1", "PSM2", "PSM3"):
        digest.update((ROOT / f"assets/exterior_source/{arm}_label.png").read_bytes())
    folder = OUT / digest.hexdigest()[:12]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "exterior.urdf"
    if dest.exists():
        assert physical_tree(source) == physical_tree(dest)
        return str(dest)
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    original_visuals = {
        link.get("name"): [ET.tostring(v) for v in link.findall("visual")] for link in root.findall("link")
    }
    records = []
    mesh_cache = {}

    def add(link, template, mesh, style, suffix):
        key = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()[:16]
        path = folder / f"{key}.obj"
        if key not in mesh_cache:
            mesh.export(path, digits=12, include_normals=True)
            mesh_cache[key] = path
        visual = copy.deepcopy(template)
        visual.find("geometry/mesh").set("filename", str(path))
        for material in visual.findall("material"):
            visual.remove(material)
        color = SURFACES[style][0] if not style.startswith("PSM") else (0.56, 0.76, 0.86 + int(style[-1]) * 0.01)
        material = ET.SubElement(visual, "material", name="exterior_" + style)
        ET.SubElement(material, "color", rgba=" ".join(map(str, (*color, 1))))
        visual.set("name", "exterior_" + suffix)
        link.append(visual)

    cad = np.load(SOURCE)
    covers = np.load(ROOT / "assets/exterior_source/cover_bevels.npz")
    for link in root.findall("link"):
        name = link.get("name")
        if name.endswith(("roll_link", "wrist_pitch_link", "wrist_yaw_link", "jaw_1_link", "jaw_2_link")):
            continue
        if CARRIAGES and name.endswith("adaptor_link") and name.startswith("PSM"):
            # The existing insertion child moves with the housing, but not with
            # shaft roll or jaw motion. Add visuals only to this empty link.
            arm = name[:4]
            joint = root.find(f"joint[@name='{arm}_housing_fixed']")
            assert joint.get("type") == "fixed"
            assert all(float(v) == 0 for key in ("xyz", "rpy") for v in joint.find("origin").get(key).split())
            template = root.find(f"link[@name='{arm}_housing_link']/visual")
            for part, style, mesh in carriage_meshes():
                add(link, template, mesh, style, "carriage_" + part)
            records.append(
                {
                    "link": name,
                    "change": "visual-only sliding carriage body, seam and mounting plate; no attachment moved",
                    "rail_face_y_m": -0.0558805,
                    "instrument_drive_face_y_m": -0.018161,
                    "outline": "estimated from full_robot scan; not measured hidden drive hardware",
                }
            )
        elif name.endswith("housing_link") and name.startswith("PSM"):
            template = link.find("visual")
            for group in ("030", "031", "032", "033", "034", "035", "037"):
                mesh = trimesh.Trimesh(cad[f"Shape.{group}_vertices"], cad[f"Shape.{group}_faces"], process=False)
                style = {"030": "base", "035": "blue", "037": "steel"}.get(group, "drive")
                if group == "035":
                    # Print only on the outward, near-planar cover. Sidewalls stay
                    # untextured, preventing repeated labels at box-UV seams.
                    mask = (mesh.face_normals[:, 1] > 0.97) & (mesh.triangles_center[:, 1] > 0.014)
                    add(link, template, mesh.submesh([np.flatnonzero(mask)], append=True), name[:4], "label")
                    mesh = mesh.submesh([np.flatnonzero(~mask)], append=True)
                add(link, template, crease_normals(mesh), style, group)
            link.remove(template)
            records.append(
                {
                    "link": name,
                    "change": "restore welded CAD cover, drive discs and release tabs; tool-specific printed label",
                }
            )
        elif name == "ECM_endoscope_link":
            # Keep camera head, lens, shaft geometry and optical mounting unchanged.
            # The video shows the detached insert; its rear ports are occupied
            # by the camera head in this assembled robot.
            for visual in list(link.findall("visual")):
                path = Path(visual.find("geometry/mesh").get("filename"))
                if path.name.endswith("_metal.dae"):
                    for old in visual.findall("material"):
                        visual.remove(old)
                    mat = ET.SubElement(visual, "material", name="exterior_scope_steel")
                    ET.SubElement(mat, "color", rgba=" ".join(map(str, (*SURFACES["steel"][0], 1))))
                    continue
                if not path.name.endswith("_black.dae"):
                    continue
                mesh = trimesh.load(path, force="mesh", process=False)
                add(link, visual, crease_normals(mesh), "black", "scope_collar")
                # Slotted exposed cage plates and retaining bands. Their dimensions
                # are estimates constrained by the existing 63 mm CAD collar.
                for j in range(8):
                    theta = j * np.pi / 4
                    plates = [cage_plate(theta + a, theta + b, 0.023, 0.097) for a, b in ((-0.18, -0.07), (0.07, 0.18))]
                    plates += [
                        cage_plate(theta - 0.07, theta + 0.07, a, b)
                        for a, b in ((0.023, 0.031), (0.060, 0.065), (0.092, 0.097))
                    ]
                    add(link, visual, trimesh.util.concatenate(plates), "satin", f"scope_slotted_plate_{j}")
                for z in (0.019, 0.101):
                    ring = trimesh.creation.annulus(r_min=0.0312, r_max=0.0321, height=0.0025, sections=96)
                    ring.apply_translation((0, 0, z))
                    add(link, visual, ring, "satin", "scope_retainer")
                link.remove(visual)
            records.append(
                {
                    "link": name,
                    "change": "proximal collar and cage rails; stainless shaft finish; head/lenses and all original geometry untouched",
                }
            )
        elif name.startswith("SUJ_") or any(
            name.endswith(s) for s in ("base_link", "yaw_link", "pitch_link", "pitch_2_link", "pitch_3_link")
        ):
            for visual in list(link.findall("visual")):
                element = visual.find("geometry/mesh")
                if element is None:
                    continue
                path = Path(element.get("filename"))
                mesh = trimesh.load(path, force="mesh", process=False)
                cover_key = hashlib.sha256(path.read_bytes()).hexdigest()
                if cover_key + "_v" in covers:
                    mesh = trimesh.Trimesh(covers[cover_key + "_v"], covers[cover_key + "_f"], process=False)
                if name == "SUJ_cart":
                    c = mesh.triangles_center
                    mask = (
                        np.all(np.abs(mesh.triangles[:, :, 0]) <= 0.02001, axis=1)
                        | np.all(np.abs(mesh.triangles[:, :, 1]) <= 0.02001, axis=1)
                    ) & (np.abs(mesh.face_normals[:, 2]) < 0.1)
                    for selection, style in [(mask, "black"), (~mask, "ivory")]:
                        add(
                            link,
                            visual,
                            crease_normals(mesh.submesh([np.flatnonzero(selection)], append=True)),
                            style,
                            "column_" + style,
                        )
                elif name.endswith("pitch_3_link"):
                    c = mesh.triangles_center
                    mask = (c[:, 2] > 0.02) & (mesh.face_normals[:, 0] > 0.9)
                    for selection, style in [(mask, "satin"), (~mask, "ivory")]:
                        if selection.any():
                            add(
                                link,
                                visual,
                                crease_normals(mesh.submesh([np.flatnonzero(selection)], append=True)),
                                style,
                                "rail_" + style,
                            )
                else:
                    add(link, visual, crease_normals(mesh), "ivory", "cover")
                # Thin mold split and flush fasteners on the arm's side covers,
                # seated on CAD surfaces, so they follow every joint pose.
                if name.endswith(("pitch_link", "pitch_2_link", "yaw_link")):
                    bounds = mesh.bounds
                    section = mesh.section(plane_origin=(0, bounds[0, 1] * 0.78, 0), plane_normal=(0, 1, 0))
                    if section is not None:
                        for j, contour in enumerate(section.discrete):
                            seam = tube_path(contour, 0.00025)
                            if seam is not None:
                                add(link, visual, seam, "seam", f"split_{j}")
                    for x in np.linspace(
                        bounds[0, 0] + 0.25 * mesh.extents[0], bounds[1, 0] - 0.25 * mesh.extents[0], 2
                    ):
                        z = float(np.mean(bounds[:, 2]))
                        hits, _, _ = mesh.ray.intersects_location([[x, bounds[0, 1] - 0.1, z]], [[0, 1, 0]])
                        if not len(hits):
                            continue
                        p = hits[np.argmin(hits[:, 1])]
                        screw = trimesh.creation.cylinder(
                            0.0023, segment=[p - [0, 0.0006, 0], np.add(p, (0, 0.0001, 0))], sections=32
                        )
                        add(link, visual, screw, "satin", "fastener")
                        slot = trimesh.creation.cylinder(
                            0.00085, segment=[p - [0, 0.00066, 0], p - [0, 0.00060, 0]], sections=6
                        )
                        add(link, visual, slot, "black", "socket")
                link.remove(visual)
                records.append(
                    {
                        "link": name,
                        "change": "satin cream finish, 1.2 mm clamped visual cover bevel; seated seams/fasteners where applicable",
                    }
                )
    pending = folder / "exterior.pending.urdf"
    tree.write(pending, encoding="unicode")
    assert physical_tree(source) == physical_tree(pending), "Physics or frames changed"
    protected = []
    for link in root.findall("link"):
        name = link.get("name")
        if name.startswith(("PSM1_", "PSM2_", "PSM3_")) and name.endswith(
            ("roll_link", "wrist_pitch_link", "wrist_yaw_link", "jaw_1_link", "jaw_2_link")
        ):
            assert original_visuals[name] == [ET.tostring(v) for v in link.findall("visual")], name
            protected.append(name)
    (folder / "verification.json").write_text(
        json.dumps(
            {"source": str(source), "physics_identical": True, "protected_visual_links": protected, "changes": records},
            indent=2,
        )
    )
    pending.replace(dest)
    return str(dest)

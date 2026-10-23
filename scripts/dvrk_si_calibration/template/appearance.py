"""Optional surface-only dVRK appearance; controller and physics data stay frozen."""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from robosimstudio.core.render.finish import Finish, register_color_finish
from robosimstudio.core.render.room import VOID
from robosimstudio.core.render.scenery import register_scenery

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets/appearance"
# Set False to reproduce the first approved appearance pass.
REFINE_SURFACES = True
# Display RGB, metalness, roughness. Surface hypotheses, not measured BRDFs.
SURFACES = {
    "shell": ((0.69, 0.70, 0.72), 1.0, 0.27),
    "polished": ((0.77, 0.78, 0.79), 1.0, 0.26),
    "mechanism": ((0.42, 0.44, 0.46), 1.0, 0.29),
    "coating": ((0.065, 0.072, 0.082), 0.0, 0.28),
    "grip": ((0.53, 0.52, 0.49), 0.9, 0.42),
}

if REFINE_SURFACES:
    SURFACES["shell"] = ((0.69, 0.70, 0.72), 1.0, 0.34)
    SURFACES["collar"] = ((0.36, 0.37, 0.38), 1.0, 0.42)


def materials():
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(423)
    # A 4 mm repeat, isotropic microvariation and fine machining striations.
    # This modulates roughness only; no photographed colour or illumination.
    _, x = np.mgrid[:256, :512]
    amplitude, noise = (0.03, 0.018) if REFINE_SURFACES else (0.015, 0.008)
    grain = amplitude * np.sin(2 * np.pi * x / 13) + rng.normal(0, noise, x.shape)
    for name, (color, metal, rough) in SURFACES.items():
        base = OUT / f"{name}_albedo.png"
        Image.fromarray(np.full((2, 2, 3), np.rint(np.array(color) * 255), dtype=np.uint8)).save(base)
        texture = ""
        if name in ("shell", "grip"):
            path = OUT / f"{name}_roughness.png"
            Image.fromarray(np.uint8(np.clip(rough + grain, 0, 1) * 255)).save(path)
            texture = str(path)
        register_color_finish(
            color,
            Finish(roughness=rough, metallic=metal, texture=str(base), roughness_texture=texture, uv_scale=(250, 250)),
        )


def physical_tree(path):
    """Normalized non-visual URDF, including resolved collision mesh paths."""
    r = ET.parse(path).getroot()
    for m in r.iter("mesh"):
        m.set("filename", str((Path(path).parent / m.get("filename")).resolve()))
    for link in r.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    for material in r.findall("material"):
        r.remove(material)
    for element in r.iter():
        element.text = (element.text or "").strip() or None
        element.tail = None
    return ET.tostring(r)


def bevel_blade(part, sign):
    """An inward 40 µm visual chamfer; retain root cap and fitted tip point."""
    import trimesh
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    original_planes = ConvexHull(part.vertices).equations
    planes = original_planes.tolist()
    for adjacent in part.face_adjacency:
        n1, n2 = part.face_normals[adjacent]
        if np.dot(n1, n2) > 0.999:
            continue
        normal = n1 + n2
        normal /= np.linalg.norm(normal)
        planes.append([*normal, -(np.max(part.vertices @ normal) - 0.00004)])
    vertices = HalfspaceIntersection(np.array(planes), part.centroid).intersections
    root = part.vertices[np.isclose(part.vertices[:, 1], part.bounds[0, 1], atol=1e-9)]
    result = trimesh.convex.convex_hull(np.concatenate([vertices, root]))
    assert np.allclose(result.bounds, part.bounds, rtol=0, atol=1e-12)
    assert np.max(result.vertices @ original_planes[:, :3].T + original_planes[:, 3]) < 1e-12
    tip = np.array([sign * 0.0004, 0.015, 0.0])
    planes = ConvexHull(result.vertices).equations
    assert np.max(planes[:, :3] @ tip + planes[:, 3]) < 1e-9
    return result


def visual_asset(source):
    """Partition existing surfaces; optionally chamfer only the DeBakey blades."""
    import trimesh
    from scipy.spatial import cKDTree

    materials()
    source = Path(source)
    digest = hashlib.sha256(source.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    folder = OUT / digest
    folder.mkdir(exist_ok=True)
    destination = folder / "appearance.urdf"
    if destination.exists():
        assert physical_tree(source) == physical_tree(destination)
        return str(destination)
    tree = ET.parse(source)
    root = tree.getroot()
    # Preserve every reference, including colliders, without copying/editing it.
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    records = []
    for link in root.findall("link"):
        name = link.get("name")
        if not name.startswith(("PSM1_", "PSM2_")):
            continue
        if not any(
            name.endswith(s) for s in ("roll_link", "wrist_pitch_link", "wrist_yaw_link", "jaw_1_link", "jaw_2_link")
        ):
            continue
        for vi, visual in enumerate(list(link.findall("visual"))):
            element = visual.find("geometry/mesh")
            if element is None:
                continue
            mesh = trimesh.load(element.get("filename"), force="mesh", process=False)
            original = mesh.triangles.copy()
            topology = mesh.copy()
            topology.merge_vertices()
            groups = trimesh.graph.connected_components(topology.face_adjacency, nodes=np.arange(len(mesh.faces)))
            groups = sorted(groups, key=len, reverse=True)
            classified = {}
            for i, faces in enumerate(groups):
                bounds = mesh.vertices[mesh.faces[faces]].reshape(-1, 3)
                lo, hi = bounds.min(axis=0), bounds.max(axis=0)
                style = "shell"
                if name.endswith("roll_link"):
                    style = "coating" if lo[2] < -0.1 else "mechanism"
                    if REFINE_SURFACES and i == 0:
                        style = "collar"
                elif name.endswith("wrist_pitch_link"):
                    style = "shell" if i == 0 else "mechanism"
                elif name.endswith("wrist_yaw_link"):
                    style = "polished"
                elif "jaw_" in name:
                    style = "polished"
                    if name.startswith("PSM1") and lo[1] > 0.003:
                        style = "grip"
                    if name.startswith("PSM2") and hi[1] - lo[1] < 0.0002:
                        style = "grip"
                    if name.startswith("PSM2") and lo[1] > 0.0029 and hi[1] > 0.0149:
                        style = "polished_bevel"
                classified.setdefault(style, []).extend(faces)
            written = []
            preserved = []
            bevel_bounds = None
            for style, faces in sorted(classified.items()):
                part = mesh.submesh([faces], append=True, repair=False)
                if style == "polished_bevel":
                    part.merge_vertices()
                    part = bevel_blade(part, -1 if name.endswith("1_link") else 1)
                    bevel_bounds = part.bounds.tolist()
                if REFINE_SURFACES and style == "coating":
                    # STL stores hard per-triangle normals on this circular shaft.
                    # Smooth only its axial walls; keep end-cap normals and all
                    # triangle surfaces unchanged. Do not smooth the wrist's edges.
                    side_faces = np.abs(part.face_normals[:, 2]) < 1e-4
                    wall = part.submesh([np.flatnonzero(side_faces)], append=True, repair=False)
                    caps = part.submesh([np.flatnonzero(~side_faces)], append=True, repair=False)
                    # The installed force-mesh loader drops OBJ normal records.
                    # Shared wall vertices let it rebuild smooth normals, while
                    # separate cap vertices retain the sharp end boundaries.
                    wall.merge_vertices(merge_norm=True, merge_tex=True, digits_vertex=14)
                    part = trimesh.util.concatenate([wall, caps])
                filename = folder / f"{name}_{vi}_{style}.obj"
                part.export(filename, digits=14, include_normals=True)
                loaded = trimesh.load(filename, force="mesh", process=False)
                if style != "polished_bevel":
                    written.append(loaded.triangles)
                    preserved.append(original[faces])
                new = copy.deepcopy(visual)
                new.find("geometry/mesh").set("filename", str(filename))
                for old in new.findall("material"):
                    new.remove(old)
                material = ET.SubElement(new, "material", name="appearance_" + style)
                ET.SubElement(
                    material, "color", rgba=" ".join(map(str, (*SURFACES[style.removesuffix("_bevel")][0], 1)))
                )
                link.append(new)
            link.remove(visual)
            # Every oriented triangle must survive; allow only OBJ decimal
            # serialization roundoff, far below Newton's float32 precision.
            got = np.concatenate(written).reshape(-1, 9)
            want = np.concatenate(preserved).reshape(-1, 9)
            error = max(cKDTree(got).query(want)[0].max(), cKDTree(want).query(got)[0].max())
            assert got.shape == want.shape and error < 1e-12, (name, error)
            records.append(
                {
                    "link": name,
                    "source_triangles": len(original),
                    "surfaces": sorted(classified),
                    "preserved_triangle_error_m": float(error),
                    "beveled_blade_bounds_m": bevel_bounds,
                }
            )
    tree.write(destination, encoding="unicode")
    assert physical_tree(source) == physical_tree(destination), "Non-visual URDF changed"
    (folder / "verification.json").write_text(
        json.dumps({"source": str(source), "physics_identical": True, "meshes": records}, indent=2)
    )
    return str(destination)


def scope_lights(kit):
    """Two small emitters on the scope visual, which follows the real ECM body."""
    stage = getattr(kit, "stage", None)
    if stage is None:
        # Viser supports PBR finishes but its backdrop API has no USD lights.
        return
    from pxr import Gf, Usd, UsdGeom, UsdLux

    anchors = []
    # The immutable scope-body CAD uniquely identifies the attachment instance.
    # Avoid hard-coded shape indices, which change when visual parts are split.
    target = np.array([[-0.0499229431, -0.0374221802, -0.109922409], [0.0499229431, 0.0374221802, -0.0000762939453]])
    for prim in stage.Traverse():
        if (
            prim.GetTypeName() != "Mesh"
            or "/model/shapes/" not in str(prim.GetPath())
            or "instance_" not in str(prim.GetPath())
        ):
            continue
        attr = UsdGeom.Mesh(prim).GetPointsAttr()
        times = attr.GetTimeSamples()
        points = np.asarray(attr.Get(Usd.TimeCode(times[0])) if times else attr.Get())
        if points.ndim == 2 and np.allclose([points.min(axis=0), points.max(axis=0)], target, atol=2e-6):
            anchors.append(prim)
    assert len(anchors) == 1, f"Expected one ECM scope-body visual, found {len(anchors)}"
    for side, x in enumerate((-0.0035, 0.0035)):
        light = UsdLux.RectLight.Define(stage, str(anchors[0].GetPath()) + f"/scope_light_{side}")
        xf = UsdGeom.Xformable(light)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0.760))
        xf.AddRotateXOp().Set(180.0)  # USD lights emit -z; scope looks +z.
        light.GetWidthAttr().Set(0.004)
        light.GetHeightAttr().Set(0.003)
        light.GetIntensityAttr().Set(120000.0)
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.94))


register_scenery("dvrk_scope_illumination", scope_lights)


def render_look():
    return replace(VOID, key_intensity=0.12, ambient=(0.22, 0.22, 0.22), scenery="dvrk_scope_illumination")

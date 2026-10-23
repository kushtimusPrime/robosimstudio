# SPDX-License-Identifier: Apache-2.0
"""Regenerate exterior source meshes/artwork; normal builds use bundled assets.

The covers command requires Blender's bpy==4.2.0 (Python 3.11). Housing uses
trimesh/numpy; labels use Pillow and Liberation Sans. See EXTERIOR.md.
"""

import argparse
from pathlib import Path

root = Path(__file__).resolve().parent


def prepare_housing(dvrk_model):
    import hashlib
    import json

    import numpy as np
    import trimesh

    out = root / "assets/exterior_source"
    out.mkdir(exist_ok=True)
    p = dvrk_model / "meshes/instruments/housing/Si/housing_Si.obj"
    s = trimesh.load(p, force="scene", process=False)
    a = {}
    for name, m in s.geometry.items():
        m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=9)
        a[name + "_vertices"] = m.vertices
        a[name + "_faces"] = m.faces
        print(name, len(m.vertices), len(m.faces), m.is_watertight)
    np.savez_compressed(out / "housing_original.npz", **a)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source": "jhu-dvrk/dvrk_model",
                "commit": "90472a933995acaa88d84eb5c58aeee4cd17e9f3",
                "path": "meshes/instruments/housing/Si/housing_Si.obj",
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "license": "MIT",
                "conversion": "Separate original material groups; weld only coincident vertices at 1 nm precision. No simplification.",
            },
            indent=2,
        )
    )


def bevel_covers():
    import hashlib
    import json

    import bpy
    import numpy as np
    import trimesh

    base = root / "assets/multisession/setup_017/meshes"
    paths = sorted((base / "SUJ").rglob("*.stl")) + sorted((base / "arms/Si/USM").glob("*.stl")) + [base / "tower.stl"]
    out = {}
    records = []
    for path in paths:
        m = trimesh.load(path, force="mesh")
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if not m.is_winding_consistent:
            m.fix_normals()
        data = bpy.data.meshes.new(path.stem)
        data.from_pydata(m.vertices.tolist(), [], m.faces.tolist())
        data.update()
        ob = bpy.data.objects.new(path.stem, data)
        bpy.context.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        mod = ob.modifiers.new("Small molded cover edge", "BEVEL")
        mod.width = 0.0012
        mod.segments = 4
        mod.limit_method = "ANGLE"
        mod.angle_limit = np.deg2rad(30)
        mod.use_clamp_overlap = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        ob.data.calc_loop_triangles()
        v = np.array([p.co[:] for p in ob.data.vertices])
        f = np.array([p.vertices[:] for p in ob.data.loop_triangles])
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        error = float(np.abs(np.array([v.min(0), v.max(0)]) - m.bounds).max())
        assert error < 0.00121, (path, error)
        out[key + "_v"] = v
        out[key + "_f"] = f
        records.append(
            {
                "mesh": str(path.relative_to(base)),
                "sha256": key,
                "source_faces": len(m.faces),
                "output_faces": len(f),
                "bounds_max_change_m": error,
                "bevel_m": 0.0012,
            }
        )
        bpy.data.objects.remove(ob, do_unlink=True)
    np.savez_compressed(root / "assets/exterior_source/cover_bevels.npz", **out)
    (root / "assets/exterior_source/cover_bevels.json").write_text(
        json.dumps(
            {
                "blender": bpy.app.version_string,
                "segments": 4,
                "angle_threshold_degrees": 30,
                "clamp_overlap": True,
                "records": records,
            },
            indent=2,
        )
    )
    print(len(records), "cover meshes")


def make_labels(fontdir):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    out = root / "assets/exterior_source"
    for arm, name, ref in [
        ("PSM1", "LARGE NEEDLE\nDRIVER", "420006"),
        ("PSM2", "DEBAKEY\nFORCEPS", "420036"),
        ("PSM3", "LARGE NEEDLE\nDRIVER", "420006"),
    ]:
        im = Image.new("RGB", (1504, 1024), tuple(round(v * 255) for v in (0.56, 0.76, 0.84)))
        d = ImageDraw.Draw(im)
        ink = (120, 88, 53)

        def text(y, s, size, bold=False, d=d, ink=ink):
            f = ImageFont.truetype(
                str(fontdir / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")), size
            )
            d.multiline_text((752, y), s, font=f, fill=ink, anchor="ma", align="center", spacing=5)

        d.text(
            (752, 210),
            "da Vinci S",
            font=ImageFont.truetype(str(fontdir / "LiberationSans-BoldItalic.ttf"), 92),
            fill=ink,
            anchor="ma",
        )
        text(315, "SURGICAL SYSTEM", 25)
        text(435, name, 84, True)
        text(710, "8MM  REF  " + ref, 32)
        # Box UV uses x horizontally and z vertically. The printed lettering
        # runs along the shaft direction, as in the detached-tool video.
        im = ImageOps.flip(im)
        im.save(out / f"{arm}_label.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("housing", "covers", "labels"))
    parser.add_argument("--dvrk-model", type=Path, default=Path.home() / "dvrk_src/dvrk_model")
    parser.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/liberation"))
    args = parser.parse_args()
    (root / "assets/exterior_source").mkdir(parents=True, exist_ok=True)
    if args.part == "housing":
        prepare_housing(args.dvrk_model)
    elif args.part == "covers":
        bevel_covers()
    else:
        make_labels(args.font_dir)


if __name__ == "__main__":
    main()

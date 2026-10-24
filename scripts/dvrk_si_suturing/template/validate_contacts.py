# SPDX-License-Identifier: Apache-2.0
"""Non-collision, calibration, FK, and four-arm IK regression for contact layer."""

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yourdfpy
from appearance import visual_asset
from contacts import LINKS, asset
from exterior import visual_asset as exterior_asset
from geometry import ROOT, recorded


def normalized(path):
    path = Path(path)
    root = ET.parse(path).getroot()
    for l in root.findall("link"):
        if l.get("name") in LINKS:
            for c in l.findall("collision"):
                l.remove(c)
    for m in root.findall(".//mesh"):
        f = Path(m.get("filename"))
        m.set("filename", str((path.parent / f).resolve()))

    # Ignore whitespace produced by XML pretty-printing, retain every attribute.
    def node(e):
        return (e.tag, tuple(sorted(e.attrib.items())), tuple(node(c) for c in e))

    return node(root)


def run():
    for name, sha in json.loads((ROOT / "frozen_calibration.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    profile = json.loads((ROOT / "assets/multisession/setup_017/profile.json").read_text())
    original = exterior_asset(visual_asset(ROOT / profile["asset"]))
    new = asset(original)
    assert normalized(original) == normalized(new), "Non-collision change"
    a = yourdfpy.URDF.load(original, load_meshes=False)
    b = yourdfpy.URDF.load(new, load_meshes=False)
    params = json.loads((ROOT / "parameters.json").read_text())
    for f in [0, 70, 141, 211, 282, 352, 422, 493, 563]:
        q = recorded(profile, params["episode"], f)
        a.update_cfg(q)
        b.update_cfg(q)
        for name in a.link_map:
            np.testing.assert_array_equal(a.get_transform(name), b.get_transform(name))
    print(f"PASS: frozen calibration hashes; all non-collision XML; {len(a.link_map)} link transforms at 9 frames.")
    repo = next(p for p in ROOT.parents if (p / "scripts/dvrk_si_pyroki_ik.py").exists())
    spec = importlib.util.spec_from_file_location("dvrk_basic", repo / "scripts/dvrk_si_pyroki_ik.py")
    basic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic)
    basic.URDF = Path(new)
    q = recorded(profile, params["episode"])
    for arm, (tip, joints, _, locked) in list(basic.ARMS.items()):
        basic.ARMS[arm] = (tip, joints, tuple(q[arm + "_" + j] for j in joints), locked)
    basic.check("suturebot")
    _, robot, names, _, cfg = basic.load("suturebot")
    import jax.numpy as jnp

    for displacement in [0.0, -0.01, 0.01]:
        q = cfg + displacement
        b.update_cfg(dict(zip(names, q, strict=True)))
        fk = np.asarray(basic.jaxlie.SE3(robot.forward_kinematics(jnp.asarray(q))).as_matrix())
        matrices = np.array([b.get_transform(name) for name in robot.links.names])
        np.testing.assert_allclose(matrices, fk, atol=1e-6)
    print("PASS: visual hierarchy and PyRoKi FK agree for every link at three configurations.")
    (ROOT / "run/contact_regression.json").write_text(
        json.dumps(
            {
                "passed": True,
                "asset": new,
                "frames": 9,
                "links": len(a.link_map),
                "checks": [
                    "frozen calibration",
                    "non-collision XML",
                    "recorded FK",
                    "4-arm IK",
                    "joint locks",
                    "visual vs IK FK",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()

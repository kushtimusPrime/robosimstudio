# SPDX-License-Identifier: Apache-2.0
"""Regenerate task assets and relocate recording metadata without changing states."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    from appearance import visual_asset
    from contacts import asset
    from exterior import visual_asset as exterior_asset
    from geometry import build_assets
    from native_needle import specification

    for name, sha in json.loads((ROOT / "frozen_calibration.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    params = json.loads((ROOT / "parameters.json").read_text())
    profile = json.loads((ROOT / "assets/multisession" / params["setup"] / "profile.json").read_text())
    folder, *_ = build_assets(params, profile)
    spec = specification(folder, params["needle"])
    urdf = asset(exterior_asset(visual_asset(ROOT / profile["asset"])))
    path = ROOT / "run/grasp_probe.json"
    if path.exists():
        data = json.loads(path.read_text())
        data.update(urdf=urdf, needle_mesh=spec.asset.path)
        path.write_text(json.dumps(data, indent=2))
    print("Prepared task assets; frozen calibration unchanged.")


if __name__ == "__main__":
    main()

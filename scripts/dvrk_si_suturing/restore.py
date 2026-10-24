# SPDX-License-Identifier: Apache-2.0
"""Restore the verified dVRK needle-contact milestone into a new live twin."""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="dvrk_si_needle_contact")
    args = parser.parse_args()
    subprocess.run([str(REPO / "examples/live/ctl"), "new", args.name], check=True)
    target = REPO / "examples/live/twins" / args.name
    base = ROOT.parent / "dvrk_si_calibration/template"
    # Reuse the approved appearance recipe and source assets, not live caches.
    for name in ("appearance.py", "exterior.py", "dataset_fetch.py", "survey.json"):
        shutil.copy2(base / name, target / name)
    shutil.copytree(base / "assets/exterior_source", target / "assets/exterior_source")
    shutil.copytree(
        ROOT / "template", target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(ROOT / "evidence", target / "run", dirs_exist_ok=True)
    params = json.loads((target / "parameters.json").read_text())
    csv = target / "ref/dataset" / params["episode"] / "ee_csv.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_bytes(gzip.decompress((target / "ref/ee_csv.csv.gz").read_bytes()))
    (target / "run/grasp_probe.json").write_bytes(gzip.decompress((target / "run/grasp_probe.json.gz").read_bytes()))
    subprocess.run([sys.executable, str(target / "prepare.py")], check=True, cwd=target)
    print(f"Restored {target}")
    print("Preview: python recorded_grasp_viser.py --port 8092")
    print("Fresh physics: ./ctl up, then the Needle contact routine")


if __name__ == "__main__":
    main()

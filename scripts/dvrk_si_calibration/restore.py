# SPDX-License-Identifier: Apache-2.0
"""Restore the versioned SutureBot calibration into its own live workspace."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="dvrk_si_suturebot")
    args = parser.parse_args()
    # ctl validates the name and refuses to overwrite an existing twin.
    subprocess.run([str(REPO / "examples/live/ctl"), "new", args.name], check=True)
    target = REPO / "examples/live/twins" / args.name
    shutil.copytree(
        ROOT / "template", target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(ROOT / "evidence", target / "run", dirs_exist_ok=True)
    shutil.copy2(target / "multisession_registration.json", target / "run/multisession_geometry.json")
    print(f"Restored {target}")
    print(f"Next: cd {target}")
    print(f"  {sys.executable} fetch_reference.py")
    print("  uv run --no-sync --with xacro --with fast-simplification python build_multisession_assets.py")
    print("  ./ctl up")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Fetch only annotated calibration/test CSVs and left images, using ZIP ranges."""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from dataset_fetch import RangeReader

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()
    entries = defaultdict(set)
    for filename in ["multisession_landmarks.json", "multisession_final_test.json", "multisession_test_episodes.json"]:
        for episode in json.loads((ROOT / "ref" / filename).read_text())["episodes"]:
            eid = episode["id"]
            tissue = eid.split("/")[0]
            entries[tissue].add(eid + "/ee_csv.csv")
            if not args.csv_only:
                for frame in episode["frames"]:
                    entries[tissue].add(f"{eid}/left_img_dir/frame{int(frame):06d}_left.jpg")
    for tissue, wanted in sorted(entries.items()):
        missing = [name for name in sorted(wanted) if not (ROOT / "ref/dataset" / name).exists()]
        if not missing:
            print(tissue, "cached", flush=True)
            continue
        with RangeReader(f"https://huggingface.co/datasets/jchen396/SutureBot/resolve/main/{tissue}.zip") as reader:
            with zipfile.ZipFile(reader) as archive:
                for name in missing:
                    path = ROOT / "ref/dataset" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(path.suffix + ".part")
                    partial.write_bytes(archive.read(name))
                    partial.replace(path)
            print(tissue, len(missing), "entries;", reader.transferred, "bytes transferred", flush=True)


if __name__ == "__main__":
    main()

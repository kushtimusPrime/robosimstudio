"""Capture identical recorded poses through the existing live Studio page."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = [("setup_017", f, "tune" if f in (0, 94, 188) else "evaluation") for f in (0, 94, 188, 219)] + [
    ("setup_015", f, "evaluation") for f in (32, 161)
]


def ctl(*args):
    p = subprocess.run([str(ROOT / "ctl"), *args, "--json"], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["baseline", "updated"])
    parser.add_argument("--tune-only", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "run/appearance"
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / "frozen_sources.json"
    if not frozen.exists():
        sources = list((ROOT / "assets/multisession").rglob("*")) + list(
            (ROOT.parents[3] / "assets/embodiments/dvrk_si").rglob("*")
        )
        sources += [
            ROOT / "multisession_registration.json",
            ROOT / "registration.json",
            ROOT / "ref/multisession_landmarks.json",
        ]
        frozen.write_text(
            json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}, indent=2)
        )
    records = []
    for setup, frame, split in CASES:
        if args.tune_only and split != "tune":
            continue
        path = ROOT / "scene.py"
        s = re.sub(r"ACTIVE_PROFILE = .*", f'ACTIVE_PROFILE = "{setup}"', path.read_text(), count=1)
        s = re.sub(r"APPEARANCE = .*", f"APPEARANCE = {args.variant == 'updated'}", s, count=1)
        if s != path.read_text():
            path.write_text(s)
        name = f"{args.variant}_{setup}_{frame:03d}"
        for _ in range(12):
            ctl("status", "--wait")
            try:
                reply = ctl(
                    "eval",
                    f"assert profile['session'] == '{setup}'; assert APPEARANCE is {args.variant == 'updated'}; show_frame(sim, session, {frame}); capture_gen = live.generation",
                    "--wait",
                )
                if reply.get("pending") or reply.get("stale"):
                    continue
                result = ctl(
                    "eval", f"from appearance_verify import capture; capture(sim, session, profile, '{name}')", "--wait"
                )
                if result.get("pending") or result.get("stale"):
                    continue
                checked = ctl("eval", "assert live.generation == capture_gen")
                if not checked.get("pending") and not checked.get("stale"):
                    break
            except subprocess.CalledProcessError as exc:
                last_error = exc.stdout + exc.stderr
                continue
        else:
            raise RuntimeError("Scene failed to load: " + locals().get("last_error", "pending rebuild"))
        record = {"setup": setup, "frame": frame, "split": split, "variant": args.variant, "image": result["value"]}
        records.append(record)
        print(name, flush=True)
    (ROOT / "run/appearance" / f"{args.variant}_captures.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

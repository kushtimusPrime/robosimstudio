"""Render one reserved frame per fitted setup through the existing live page."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ctl(*args):
    p = subprocess.run([str(ROOT / "ctl"), *args, "--json"], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def main():
    labels = json.load(open(ROOT / "ref/multisession_landmarks.json"))
    records = []
    for e in labels["episodes"]:
        choices = [
            (sum(p is not None for a in ["PSM1", "PSM2"] for p in v.get(a, [])), int(f))
            for f, v in e["frames"].items()
            if v["split"] == "validation"
        ]
        f = max(choices, key=lambda v: (v[0], -v[1]))[1]
        ref = ROOT / "ref/dataset" / e["id"] / "left_img_dir" / f"frame{f:06d}_left.jpg"
        if not ref.exists():
            ref = ROOT / "ref" / (e["id"].replace("/", "__") + f"__left_img_dir__frame{f:06d}_left.jpg")
        path = ROOT / "scene.py"
        s = path.read_text()
        s = re.sub(r"ACTIVE_PROFILE = .*", f'ACTIVE_PROFILE = "{e["session"]}"', s, count=1)
        path.write_text(s)
        for _attempt in range(12):
            ctl("status", "--wait")
            try:
                state = ctl(
                    "eval", f"assert profile['session'] == '{e['session']}'; show_frame(sim, session, {f})", "--wait"
                )
                if state.get("ok", True) and not state.get("pending") and not state.get("stale"):
                    break
            except subprocess.CalledProcessError:
                continue
        else:
            raise RuntimeError("Profile did not finish loading: " + e["session"])
        result = ctl("snapshot", "--camera", "endoscope", "--beside", str(ref))
        assert not result.get("pending") and not result.get("stale"), result
        records.append({"session": e["session"], "episode": e["id"], "frame": f, "result": result})
        print(e["session"], f, result, flush=True)
    (ROOT / "run/rendered_profiles.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

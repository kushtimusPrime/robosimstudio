"""Evaluate a frozen candidate on independently annotated entire episodes."""

import argparse
import json
from pathlib import Path

import numpy as np
from calibrate_multisession import errors, load, points, project

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="geometry")
    parser.add_argument("--labels", default="multisession_test_episodes.json")
    parser.add_argument("--suffix", default="test")
    args = parser.parse_args()
    fit = json.load(open(ROOT / "run" / f"multisession_{args.model}.json"))
    _, tracks, obs = load(args.labels)
    intr = [fit["intrinsics"][k] for k in ["fx", "fy", "cx", "cy", "k1", "k2"]]
    geo = [
        fit["geometry"][k]
        for k in ["PSM1_length", "PSM2_length", "PSM1_jaw_offset", "PSM1_jaw_gain", "PSM2_jaw_offset", "PSM2_jaw_gain"]
    ]
    records = []
    for o in obs:
        s, a = o["session"], o["arm"]
        p = points(tracks[o["episode"]][a], o["frame"], fit["lag_frames"][s], geo, a, fit["registrations"][s][a])
        uv, _z = project(p, fit["registrations"][s][a], intr)
        err = np.linalg.norm(errors(uv, o["pixels"]), axis=1)
        rec = {k: v for k, v in o.items() if k != "pixels"}
        rec.update(
            projected=uv.tolist(),
            observed=[v.tolist() if np.all(np.isfinite(v)) else None for v in o["pixels"]],
            errors_px=[float(v) if np.isfinite(v) else None for v in err],
        )
        records.append(rec)
    metrics = {}
    for name, rs in [("all", records)] + [(a, [r for r in records if r["arm"] == a]) for a in ["PSM1", "PSM2"]]:
        v = [v for r in rs for v in r["errors_px"] if v is not None]
        metrics[name] = {
            "landmarks": len(v),
            "rms_px": float(np.sqrt(np.mean(np.square(v)))),
            "median_px": float(np.median(v)),
            "p95_px": float(np.percentile(v, 95)),
            "max_px": max(v),
        }
    out = {
        "candidate": args.model,
        "scope": "Two whole episodes excluded from fitting; manual approximate pixels. This is tool-landmark reprojection, not full-mesh silhouette accuracy.",
        "metrics": metrics,
        "observations": records,
    }
    (ROOT / "run" / f"multisession_{args.model}_{args.suffix}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

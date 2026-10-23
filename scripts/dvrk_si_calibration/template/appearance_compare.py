"""Unwarped, identical pixel crops for real/baseline/appearance inspection."""

import json
from pathlib import Path

import numpy as np
from appearance_capture import CASES

root = Path(__file__).resolve().parent


def main():
    from PIL import Image, ImageDraw, ImageFont

    out = root / "run/appearance"
    labels = json.loads((root / "ref/multisession_landmarks.json").read_text())["episodes"]
    fit = json.loads((root / "multisession_registration.json").read_text())
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    overview = []
    crops = []
    for setup, frame, split in CASES:
        ep = next(e for e in labels if e["session"] == setup)
        paths = [root / "ref/dataset" / ep["id"] / f"left_img_dir/frame{frame:06d}_left.jpg"] + [
            out / f"{v}_{setup}_{frame:03d}.png" for v in ("baseline", "updated")
        ]
        images = [Image.open(p).convert("RGB") for p in paths]
        titles = [
            "Real: SutureBot left endoscope",
            "Baseline: frozen calibrated geometry",
            "Updated: optional appearance",
        ]
        sheet = Image.new("RGB", (2880, 595), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for i, (im, title) in enumerate(zip(images, titles, strict=True)):
            sheet.paste(im, (960 * i, 55))
            d.text((960 * i + 12, 8), title, font=font, fill="white")
            d.text((960 * i + 12, 34), f"{setup} / frame {frame} / {split} for appearance", font=small, fill="#cccccc")
        sheet.save(out / f"comparison_{setup}_{frame:03d}.png")
        overview.append(sheet.resize((1440, 298), Image.Resampling.LANCZOS))
        for arm in ["PSM1", "PSM2"]:
            real = [p for p in ep["frames"][str(frame)][arm] if p is not None]
            obs = next(
                (o for o in fit["observations"] if o["session"] == setup and o["frame"] == frame and o["arm"] == arm),
                None,
            )
            pts = real + ([] if obs is None else obs["projected"])
            if not pts:
                continue
            xy = np.array(pts)
            lo = np.maximum([0, 0], np.floor(xy.min(axis=0) - 100)).astype(int)
            hi = np.minimum([960, 540], np.ceil(xy.max(axis=0) + 100)).astype(int)
            if np.any(hi <= lo):
                continue
            box = (*lo, *hi)
            w, h = hi - lo
            crop = Image.new("RGB", (int(w) * 3, int(h) + 62), (24, 24, 24))
            d = ImageDraw.Draw(crop)
            for i, (im, title) in enumerate(zip(images, titles, strict=True)):
                crop.paste(im.crop(box), (int(w) * i, 62))
                d.text((int(w) * i + 8, 8), title.split(":")[0], font=font, fill="white")
                d.text((int(w) * i + 8, 36), f"{setup} f{frame}: {arm} ({split})", font=small, fill="#cccccc")
            name = f"closeup_{setup}_{frame:03d}_{arm}.png"
            crop.save(out / name)
            crops.append(
                {"file": name, "crop_xyxy": list(map(int, box)), "scale": 1, "real_landmarks_visible": len(real)}
            )
    canvas = Image.new("RGB", (1440, 298 * len(overview)))
    for i, im in enumerate(overview):
        canvas.paste(im, (0, i * 298))
    canvas.save(out / "overview.png")
    (out / "comparison_crops.json").write_text(json.dumps(crops, indent=2) + "\n")


if __name__ == "__main__":
    main()

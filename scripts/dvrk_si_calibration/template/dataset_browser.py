"""Browse sampled SutureBot images and frozen landmark comparisons in the live Viser page."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def install(session):
    from PIL import Image, ImageDraw

    old = getattr(session, "_suturebot_browser", None)
    if old is not None:
        old.remove()
    survey = json.load(open(ROOT / "survey.json"))
    episodes = {e["id"]: e for e in survey["episodes"]}
    observations = {}
    for filename in ["../multisession_registration.json", "multisession_geometry_final_test.json"]:
        path = ROOT / "run" / filename
        if path.exists():
            for o in json.load(path.open())["observations"]:
                observations.setdefault((o["episode"], o["frame"]), []).append(o)
    manifest = ROOT / "run/rendered_profiles.json"
    rendered = {
        (r["episode"], r["frame"]): r["result"]["beside"]
        for r in (json.loads(manifest.read_text()) if manifest.exists() else [])
    }
    gui = session.server.gui
    with gui.add_folder("SutureBot dataset comparison", expand_by_default=True) as folder:
        gui.add_markdown(
            "Recorded images across all 10 tissues. Yellow crosses: manual landmarks. Cyan circles: candidate projection. Overlays appear only on annotated left-camera frames; the 3D scene remains on its selected setup. Rendered comparisons are available at the ten reserved frames in the comparison boards."
        )
        episode = gui.add_dropdown(
            "Episode",
            options=tuple(episodes),
            initial_value=next(e for e in episodes if e.startswith("tissue_4/3_knot_tying/20250429-205423")),
        )
        frame = gui.add_dropdown(
            "Image frame", options=tuple(map(str, episodes[episode.value]["frames"])), initial_value="94"
        )
        side = gui.add_dropdown("Stereo camera", options=("left", "right"))
        mode = gui.add_dropdown(
            "Display", options=("Landmark overlay", "Rendered comparison"), initial_value="Rendered comparison"
        )
        overlay = gui.add_checkbox("Show landmark comparison", initial_value=True)
        info = gui.add_markdown("")
        view = gui.add_image(np.zeros((540, 960, 3), dtype=np.uint8), label="Recorded endoscope image")

    def update(_=None):
        eid = episode.value
        f = int(frame.value)
        path = ROOT / "ref/dataset" / eid / f"{side.value}_img_dir" / f"frame{f:06d}_{side.value}.jpg"
        if not path.exists():
            info.content = "This sampled image is not available."
            return
        im = Image.open(path).convert("RGB")
        obs = observations.get((eid, f), []) if side.value == "left" else []
        vals = []
        if overlay.value and obs:
            d = ImageDraw.Draw(im)
            for o in obs:
                for x, y in o["projected"]:
                    d.ellipse((x - 5, y - 5, x + 5, y + 5), outline="cyan", width=2)
                for p in o["observed"]:
                    if p:
                        x, y = p
                        d.line((x - 6, y, x + 6, y), fill="yellow", width=2)
                        d.line((x, y - 6, x, y + 6), fill="yellow", width=2)
                vals.extend(v for v in o["errors_px"] if v is not None)
        if mode.value == "Rendered comparison" and (eid, f) in rendered and side.value == "left":
            im = Image.open(rendered[eid, f]).convert("RGB")
        view.image = np.asarray(im)
        msg = f"Setup group: {episodes[eid]['setup_group']}. Frame {f}. "
        if vals:
            msg += f"{obs[0]['split']}: {np.sqrt(np.mean(np.square(vals))):.1f} px landmark RMS ({len(vals)} points)."
        else:
            msg += "Survey image; no measured calibration score for this frame."
        if mode.value == "Rendered comparison" and ((eid, f) not in rendered or side.value != "left"):
            msg += " No cached render for this frame; showing recorded image."
        info.content = msg

    @episode.on_update
    def change_episode(_):
        frame.options = tuple(map(str, episodes[episode.value]["frames"]))
        frame.value = frame.options[0]
        update()

    mode.on_update(update)
    frame.on_update(update)
    side.on_update(update)
    overlay.on_update(update)
    update()
    session._suturebot_browser = folder
    return {"episodes": len(episodes), "annotated_frames": len(observations)}

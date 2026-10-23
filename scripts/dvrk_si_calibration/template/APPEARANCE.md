# Optional instrument appearance

This adds a reversible appearance layer to the frozen calibrated SutureBot
inspection scene. It changes PSM1's large needle driver and PSM2's approximate
DeBakey forceps. It does not reconstruct the full exterior from endoscope images.
The canonical robot, calibrated profile assets, physics and calibration remain
unchanged. PSM3 and the ECM visual meshes retain their original materials.

A subsequent [wider survey](APPEARANCE_SURVEY.md) covers 122 cached episodes in
66 setup groups. It adds a small, separately switchable surface refinement:
duller existing collars, housing roughness .34 and smooth shaft-wall shading.
Set `REFINE_SURFACES = False` in `appearance.py` to retain the first approved
appearance recipe described below. The historical comparison images below are
unchanged; the new review has its own evidence and verification.

## Use in the existing Studio page

Set `APPEARANCE = True` in `scene.py`; its normal save/rebuild workflow loads the
variant. Set it to `False` to restore the baseline materials, meshes and lighting.
`ACTIVE_PROFILE` still selects the calibrated setup. No viewer restart is needed.
For the representative pose, use:

```bash
./ctl status --wait
./ctl eval 'show_frame(sim, session, 94)'
./ctl snapshot --camera endoscope
```

`appearance.py` generates partitioned OBJ meshes, constant albedo/roughness PNGs,
a separate URDF and `verification.json` under `assets/appearance/`. The calibrated
source URDF and meshes are referenced, never overwritten. Each variant is keyed
by the source URDF and generator SHA-256. Paths are resolved when generated in
the restored twin; copying generated URDFs alone to another machine is not a
portable installation. Keep the generator with the calibration bundle.

The entry point is `build(profile_name, appearance=True)` in
`multisession_preview.py`. Its default remains `False`. No library or canonical
embodiment changes are needed. Generated files can be regenerated from the
existing calibrated assets; deleting the appearance directory is optional cleanup
only after switching back to baseline.

## What changed, and what the images support

Across setup_017 frames 0, 94 and 188, the shafts remain dark while wrist housings
and blade faces carry moving highlights. The upgrade distinguishes coated shafts,
steel housings, exposed mechanisms, polished jaws and rougher gripping surfaces.
It reveals the CAD's existing pulley, seam, engraving and serration geometry.
The shaft/sleeve transition is differentiated by material; its geometry is frozen.
No tissue, thread, reflected highlight or background pixel is used in a texture.

The only shape change is a **40 micrometre inward chamfer on the generated
DeBakey blade edges**, before the existing 0.8 longitudinal scale. It keeps the
original root cap, outer bounds and fitted tip point. The blade remains inside
the original convex volume; length stays 12 mm. The radius is a visual assumption,
not a dimension recovered from the footage. The other surfaces retain every
oriented triangle to below 1e-12 m serialization error. Colliders are not chamfered.
The baseline 8.739 mm PSM1 jaw reach and all jaw mappings are preserved.

The DeBakey tip still looks too rectangular in several views, with coarse teeth
and an uncertain neck profile. The real needle-driver gripping faces also differ
from the CAD. Correcting these broad shapes would require better dimensional
reference; they were not stretched or shifted to hide existing alignment errors.
No detached parts or new visible intersections were found in the six inspected
poses. The fine procedural roughness stays on the moving parts without visible
texture stretching at this resolution; it is not evidence of real machining pitch.

## Materials and renderer limits

Studio's RTX renderer and Viser mesh loader support metallic/roughness PBR and
texture maps. The imported URDF's ordinary grey colours did not specify those
metal properties. The overlay registers explicit `Finish` materials and uses
constant RGB textures with a metric 4 mm UV repeat, plus weak procedural roughness
variation on housings and gripping surfaces. Roughness spans 0.26–0.42; metallic
is 1 for steel, 0 for the dark coating and 0.9 for gripping faces. These are
plausible appearance assumptions, not measured BRDFs or identified polymers.

RTX receives two 4×3 mm area lights, 7 mm apart, fixed to the scope visual body
near the lens, with intensity 120000 in renderer units and RGB (1, .98, .94).
The overhead key is reduced to .12 and dome ambient to .22. These parameters are
an artistic approximation of endoscope illumination, not measured illumination
or exposure. The light attachments follow the ECM; the optical frame does not move.
Viser's interactive viewport shows the same PBR surfaces but retains its own
lighting: custom USD scope lights are RTX-only. Use the named endoscope render
or the comparison board to judge the calibrated camera view.

There is no exposed anisotropy/normal-map/clearcoat control in `Finish`, and no
added lens blur, vignetting, bloom, distortion, white-balance or response curve.
The existing camera/render response remains in use. Simulated highlights can be
too broad, strong or in the wrong place. Wet films, tissue-coloured environmental
reflections, real exposure and shadows are missing. Tissue/needle/thread and room
reconstruction are outside this appearance pass. These comparisons do not show
pixel-perfect agreement or establish camera calibration accuracy.

## Comparison protocol and results

All images use the **calibrated left endoscope**, 960×540, unchanged intrinsics,
optical mount, per-setup registrations, time offsets and recorded replay poses.
Appearance tuning used setup_017 frames **0, 94, 188**. After freezing settings,
evaluation used setup_017 **219** and setup_015 **32, 161**. Neither existing final
test episode was used, opened, downloaded or re-evaluated for this appearance pass.
These appearance splits are separate from the older calibration train/validation
split (frame 94 was a calibration validation frame).

`run/appearance/overview.png` shows all six cases, real/baseline/updated.
`comparison_<setup>_<frame>.png` retains native-resolution panels.
`closeup_<setup>_<frame>_<arm>.png` uses identical, unwarped pixel crops in all
three panels. Crop boxes are recorded in `comparison_crops.json`. PSM1 is largely
outside the image at frame 219, so that case validates PSM2 only. No photo-wide
similarity metric is reported: missing tissue and occlusion would dominate it.

Verification established:

- All **420 protected files** retain their initial SHA-256, including canonical
  assets and calibrated profile assets. Non-visual URDF trees match exactly.
- Six baseline/updated replays have identical joint and body poses, mass, inertia,
  COM, limits, mimic parameters, joint frames/axes, camera specifications/world
  poses, IK handles and camera frusta.
- Across ten setups and **453 landmarks**, appearance changes projection by
  **0.0 px**. URDF-versus-optimizer consistency stays 0.02898 px maximum. This is
  internal consistency; the existing real-image errors remain 10.66 px training,
  19.07 px reserved, and 24.01 px on the previously reported final episodes.
- Rendered tool silhouettes overlap the baseline by **99.64–99.93% IoU**. The
  bevel removes 78–220 edge pixels per image, grows no silhouette outward and
  retreats at most **1.414 px**. Tool masks use RTX depth between 0 and 0.2 m to
  exclude the distant backdrop. This compares renders, not real-image masks.
- Scope-light transforms relative to the camera stay within 6.7e-7 across three
  recorded poses. They are attached to the ECM body, not the viewing camera.

The Studio gate passed build, finite-state, camera, penetration and rest checks
(0 mm reported overlap/drift over 60 steps); object-presence checks were skipped
without keys. This does not validate surgical contact physics. Repository Ruff
check and formatting passed (761 files); no library test tiers were required for
this twin-only visual overlay and its exported template.

Reproduce through the live page (the capture runner changes profile/appearance
and leaves the last captured pose paused):

```bash
uv run --no-sync python appearance_capture.py baseline
uv run --no-sync python appearance_capture.py updated
uv run --no-sync python appearance_verify.py
uv run --no-sync python appearance_projection.py
uv run --no-sync python appearance_compare.py
./ctl show run/appearance/overview.png --label 'Real / baseline / optional appearance'
```

The required images/CSVs are covered by `fetch_reference.py`'s existing bounded
ZIP-range requests. This pass reused cached references and downloaded nothing.
The full state/depth captures remain in the working twin; selected images and
numerical reports are exported with the reusable calibration bundle.

SutureBot reference panels: Haworth, Chen et al., jchen396/SutureBot, **CC BY 4.0**.
Comparison panels and manual annotations are derived modifications. Upstream
JHU CAD retains its MIT license; the new generator is Apache-2.0.

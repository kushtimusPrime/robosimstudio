# Wider SutureBot appearance review

This follow-up reviewed one middle cached **left-endoscope frame from each of
122 non-final-test episodes**, spanning all **66 provisional setup groups** and
all ten tissue archives. The two existing final-test episodes were excluded.
`run/appearance_survey/manifest.json` records every reviewed episode/frame;
`survey_01.jpg` through `survey_11.jpg` are the inspected contact sheets.
This is a broad sample of the existing image cache, **not an exhaustive review
of all 1,891 episodes or every video frame**. No downloads were needed.

## Findings and action

| Recurring observation | Evidence and confidence | Action |
| --- | --- | --- |
| A dull collar separates the dark shaft from the polished wrist mechanism. | Visible across January and April/May sessions, notably survey #17, #64, #89, #93 and #105. The CAD already has a separate collar/fork component. Its exact alloy/finish is unknown. | Give that existing component a neutral, darker steel finish with roughness .42. No added ring, altered diameter or shifted material boundary. |
| Wrist plates have a less uniform, less mirror-like response than our first appearance pass. | Repeats in setups 001, 010, 020 and many survey images. Compression and sensor noise contribute; the footage cannot uniquely identify a BRDF. | Modestly increase housing roughness from .27 to .34 and weak procedural roughness variation. No photographed pixels, scratches, tissue colours or highlights are baked in. |
| Flat shading makes the cylindrical shaft look faceted. | The existing STL wall normals deviate from radial by about 5.6 degrees. This is a mesh-shading issue, not an observed physical feature. | Share coincident wall vertices so the actual mesh loader produces smooth normals; leave cap boundaries separate. Loaded wall normals are within .007 degrees of radial. Every oriented triangle surface remains unchanged. |
| DeBakey blades look rounder and less rectangular, with a different neck/serration profile. | Consistent in multiple orientations; dimensions and hidden geometry remain ambiguous. | Record as the highest-value geometry follow-up. Do not reshape the calibrated jaws in this pass. Better instrument CAD or dimensional photographs are needed. |
| Needle-driver gripping surfaces have finer detail than the CAD. | Visible particularly in open-jaw views; inner faces, highlights and compression are hard to distinguish. | Preserve existing teeth. A supported normal-map/detail workflow or better CAD would be preferable to invented large teeth or image-derived textures. |
| Some setups render substantially darker, and highlights differ despite the same materials. | Especially apparent in setup_001; tissue reflections, illumination distribution and camera response are absent or approximate. | Keep lights/camera response fixed in this pass. This deserves a separate endoscope-rendering study; material changes alone do not resolve it. |

`collar_details.png` and `new_setup_details.png` show native-pixel reference crops
that informed these choices. The collar is not assigned a brown tissue tint;
its finish is neutral grey. The small roughness pattern is only an appearance
assumption, not a reconstruction of machining marks or measured wear.

## Implementation and comparison

`REFINE_SURFACES = True` in `appearance.py` enables this refinement inside the
existing optional appearance layer. Set it to `False` to reproduce the first
approved appearance recipe (commit `6445512`), or set `APPEARANCE = False` in
`scene.py` to restore the original grey calibrated baseline. All variants are
generated separately from the canonical/calibrated assets.

The installed force-mesh OBJ loading path discards explicit normal records, so
simply writing smooth `vn` values did not work. Sharing only the shaft-wall
vertices lets the loader reconstruct smooth normals. The generator's existing
oriented-triangle equality check verifies that this changes topology/shading,
not the surfaces. Caps, collar/fork geometry, jaws and the previously approved
40 µm DeBakey visual chamfer are otherwise unchanged.

Tuning comparisons use setup_001 frame 244, setup_010 frame 352 and setup_020
frame 126. Setup_017 frame 94 checks the previously approved view. Settings were
then frozen before inspecting setup_006 frame 91 and setup_027 frame 42.
Neither final-test episode was inspected or evaluated. These roles are separate
from the older calibration train/validation split.

All comparisons use identical recorded joints, frozen left-camera intrinsics,
optical mounting, per-session registrations, time offsets, resolution and
illumination. `review_overview.png` and the six `comparison_*.png` triptychs show
**real / first approved appearance / refinement**. Unlike the previous pass,
the middle panel is already metallic, not the original grey model. The changes
are intentionally subtle. Existing image misalignment remains visible.

## Verification and limits

`verification.json` records paired state, camera, physics, depth and silhouette
checks: all six pairs have identical physics/frame state, zero changed
silhouette pixels, and bit-identical depth. The projection audit still covers
453 landmarks in ten calibrated setups, with 0.0 px appearance-induced change. All 420 canonical/calibrated files retain their original checksums. These are internal consistency checks, not proof of real-image accuracy.
The Studio gate passed build, finite state, camera, reported overlap and rest
checks; object checks were skipped without keys. Ruff check and formatting passed
for all 761 Python files. No library test tiers were run for this visual-only pass.

In setup_006, a distant grey cart-base part appears at the top of the empty
render (about 0.89 m depth, identified as `tower.stl`). The real work scene would
occlude that region; it is left visible rather than hidden or moved for the
comparison. This is separate from the tool-region appearance evaluation.

The material refinements do not fix the approximate jaw profiles, missing tissue
reflection, dark setups, wetness, optical blur or vignetting. The endoscope does
not show the full robot exterior; this review adds no claims about unseen parts.
Custom scope lights remain RTX-only; Viser uses its own viewport lighting.

The comparison runner and contact-sheet script are preserved in
`run/appearance_survey/`; the working twin contains raw paired state/depth data.
Selected evidence and the generator are exported in the calibration bundle.
Source imagery: Haworth, Chen et al., SutureBot, **CC BY 4.0**. Crops/contact
sheets and annotations are derived modifications; no source-image pixels are
used in robot textures. Existing JHU CAD remains MIT; the generator is Apache-2.0.

# Optional exterior appearance from the detached-tool scans

Set `EXTERIOR = True` in `scene.py`, alongside the existing `APPEARANCE = True`.
The two switches are independent: `APPEARANCE` controls the approved distal
instrument visuals and scope lighting; `EXTERIOR` controls the proximal housings,
scope mounting collar/shaft finish, and larger robot covers. Disable only
`EXTERIOR` to return to the previous exterior. The canonical URDF and every
calibrated profile remain untouched.

The page includes **Inspect exterior**, **Needle driver housing**, **DeBakey
housing**, and **Scope collar** viewpoints. These buttons replay setup_017 frame
94 in the working scene and move only the free browser viewpoint. The calibrated
endoscope stays fixed. **Inspect IK** still enables interactive motion.

## What changed

- The old 20,000-triangle housing contained disconnected fragments after
  simplification. The optional asset restores all seven material groups from
  JHU's original housing: blue cover, dark base, four drive discs and metal
  release tabs. Coincident vertices are welded before loading; no housing
  simplification remains. Its original CAD attachment and dimensions are used.
- Pale-blue, opaque molded polymer and dark polymer are distinct from polished
  steel. The label is authored artwork, with `LARGE NEEDLE DRIVER / 420006` on
  PSM1/PSM3 and `DEBAKEY FORCEPS / 420036` on PSM2, in the scan's rotated print
  layout. Liberation Sans Bold/Italic approximates the type and wordmark; the
  exact proprietary lettering was not recovered. No lit video crop is used.
- Larger arm/setup-joint/cart covers use a satin cream dielectric finish.
  Nineteen source cover meshes receive overlap-clamped, 1.2 mm visual bevels
  with four segments, for softer edge highlights. The largest change in a
  cover bounding-box coordinate is 0.503 mm. Thin molded split seams and
  flush metal fasteners are seated on selected arm covers. Existing column
  channels and insertion guides receive separate dark/metal finishes.
- The scope's proximal collar receives black/metal separation, narrow cage
  rails and retaining bands. The existing shaft gets a stainless finish.
  The rectangular camera head, green controls, lens geometry, shaft dimensions
  and optical frame are retained. The detached scan lacks the attached camera
  head; replacing it with the scan's bare optical ports would misrepresent the
  assembled robot.

`exterior.visual_asset()` writes a separate, content-keyed URDF under
`assets/exterior/`. Materials work through Studio's existing Finish/PBR path in
Viser and RTX. Texture filenames include content hashes so edits cannot reuse
stale GPU texture data. `exterior_capture.py` renders the actual live simulation
under neutral VOID lighting for exterior comparisons. It does not change the
scene's endoscope lighting or calibrated camera.

## Evidence and preservation

The videos were reviewed across their durations: tool 2–74 s, scope 2–50 s,
and whole robot 2–132 s. Their HLG/BT.2020 frames were tone-mapped to BT.709 for
inspection; differing exposure, warm room light and reflections were not
copied into albedo. See `run/exterior/reference_manifest.json` for timestamps
and source hashes. The original MOV files are not part of the portable bundle.

The comparison sheets in `run/exterior/` show real scan / old exterior / updated
exterior. The scan and simulation intentionally differ in robot configuration,
viewpoint and lighting; the two rendered versions use identical recorded
poses, camera settings, resolution and neutral lighting. This is an appearance
comparison, not a fitted reconstruction of the scan camera.

The regression captures use calibrated left-endoscope setup_017 frames 0, 94
and 188, plus setup_006 frame 91. `verification.json` checks identical live
joint/body state, camera/frustum attachments, limits, mimic coefficients,
masses and inertias. Near-tool depth and silhouettes are unchanged. The far
column can appear in setup_006, so exterior changes there are reported
separately from surgical-tool silhouettes. All 421 protected original files,
including the approved distal appearance generator, retain their hashes.
The two final-test SutureBot episodes were not used.

All 15 PSM roll/wrist/jaw visual link definitions are asserted identical before
writing an exterior asset. Non-visual URDF trees must match exactly, including
resolved collision paths. Every moving PBR mesh instance is checked against FK
at three paused replay poses; see `attachments.json`. The free-view controls
are exercised in the existing live Studio process. RTX images are inspected;
no browser client was connected for a browser framebuffer screenshot.

Small RGB differences in the endoscope renders remain: altered remote surfaces
can change reflections and the path tracer has sampling variation. These are
reported separately from exact depth/projection consistency. No claim of
pixel-perfect real-image alignment is made.

## Assumptions and remaining differences

Color and roughness are plausible material assumptions, not measured BRDFs.
The blue polymer is treated as opaque; clearcoat, anisotropy, transmission and
normal maps are not exposed by the current Finish API. Fine surface texture
is therefore limited to geometry, scalar roughness and the printed label.
Viser uses different lighting from the RTX endoscope.

The scope's cage uses a 63 mm CAD collar as its dimensional reference: eight
roughly 11.5 mm-wide slotted plates, 74 mm long, and 2.5 mm retaining bands. The scan supports the
cage/slot construction but not these exact dimensions. Cover bevels (1.2 mm),
seam radius (0.25 mm) and fastener radius (2.3 mm) are explicit visual estimates.
They have no collision or inertial effect. The original coarse cover contours
and simplified insertion stages remain; cables, the full wheeled cart base,
individual drive connectors, laboratory stickers and hidden internal hardware
are not reconstructed. The unchanged DeBakey distal jaws remain approximate.

The live Studio gate passes: finite state, zero reported overlap and zero drift
over 60 steps. Attachment checks cover 130 moving PBR mesh instances at three
poses, with maximum position error below 0.1 micrometre. Source housing/cover
arrays and all three label PNGs were regenerated in a separate directory and
matched exactly; see `reproduction.json`.

## Reproduce or edit the assets

Normal use needs only the existing Studio environment. Bundled
`assets/exterior_source/` contains the welded source housing, beveled cover
arrays, generated PNG labels, provenance, and the JHU MIT license. The assets
are about 3.3 MB; generated URDFs/OBJ files are disposable and rebuilt locally.

To regenerate source assets, from the restored twin:

```bash
uv run --no-sync python exterior_sources.py housing --dvrk-model ~/dvrk_src/dvrk_model
uv run --no-sync python exterior_sources.py labels
# Blender is only an asset-generation dependency, not a Studio dependency.
uv run --no-project --python 3.11 --with bpy==4.2.0 --with numpy --with trimesh \
  python exterior_sources.py covers
```

Housing provenance is JHU `dvrk_model` commit
`90472a933995acaa88d84eb5c58aeee4cd17e9f3`. The covers command reads that revision's
meshes from the generated setup_017 profile. Labels use the system Liberation
Sans fonts; pass `--font-dir` if installed elsewhere. Use the second drive for
`TMPDIR`, `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` when generating Blender
assets. Normal replay never invokes Blender.

To capture or inspect:

```bash
./ctl eval 'from exterior_capture import capture; capture(sim, session, profile, "updated", "tool")' --wait
./ctl eval 'from exterior_verify import check_attachments; check_attachments(sim, session, profile)' --wait
./ctl play 'Inspect exterior' --wait
```

No control action is attached to a visual detail; replay is the existing
recorded-FK inspection workflow, not a manipulation demonstration.

## Sliding instrument carriage (2026-10-23 follow-up)

The PSM adaptor links had no visual geometry. In housing CAD coordinates the
rail face lies at y=-55.8805 mm, while the instrument drive discs reach
-18.161 mm: a 37.72 mm empty gap. The user's mounting screenshots and the
whole-robot scan at 12/82 seconds show the missing rounded cream carriage.

The exterior layer now adds a support body, small parting seam and mounting
plate to each existing `PSM*_adaptor_link`. They follow insertion and arm motion
but not shaft roll. The housing, shaft, rail, tool tips and joint origins are
unchanged. This is visual-only geometry, with no collision or inertial additions.
Set `CARRIAGES = False` in `exterior.py` to disable just this addition, or disable
`EXTERIOR` to restore the original exterior. **Instrument carriage** on the
live page opens a close side view; **Inspect IK** enables normal interaction.

The body is estimated at 108 x 64 mm, the plate at 110 x 72 mm, with 7–8 mm
corner radii and 0.8–1.2 mm edge bevels. Their depth fills the existing CAD gap,
with 0.32 mm rail seating and 0.26 mm drive-face overlap to avoid cracks.
The reference establishes the support assembly, not these precise dimensions.
Hidden drive connectors are not reconstructed; the sterile interface is shown
as opaque cream rather than claiming calibrated translucent material properties.
The existing rail and insertion stage remain simplified.

Studio RTX comparisons use identical recorded poses, camera and illumination:
`run/exterior/carriage_comparison.png`, `carriage_travel.png` and
`carriage_endoscope.png`. The carriage stays seated through insertion
0 / 0.145 / 0.29 m. Geometric checks cover all three PSMs at three insertion
values and three pitches (27 cases). The moving PBR-instance checks now cover
139 instances across three recorded poses, within 0.1 micrometre of FK.
The live gate passes with zero reported overlap and drift; it does not test
new visual geometry for physical collision because physics is unchanged.
Browser screenshot requests timed out; the reviewed images use Studio RTX.

Carriage regression checks also passed setup_006 frame 91: four captures across
two setups retain identical entire-image depth, joints/body state, physics and
camera data. All 421 protected source files retain their hashes. RGB can differ
with sampling/reflections. `carriage_preservation.json` records these checks.
The page actions now use Studio's generator callback signature; the carriage
inspection button is exercised through `ctl play`, not only direct evaluation.

## Standalone PyRoKi preview

`uv run --no-sync python pyroki_preview.py --port 8091` opens the same upgraded
asset with the original four-arm PyRoKi solver, independent of Newton/Studio
physics. PSM1/PSM2 start movable at setup_017 frame 94; ECM/PSM3 start locked.
Use the View selector, drag pose handles, or lock individual joints. Jaw
openings retain the recorded calibration mapping. `--start middle` uses the
working arms' mid-range configuration; `--check` exercises locks and reachable
targets and compares every visual link's FK against PyRoKi. The script reuses
Studio's GLB/PBR/UV conversion; lighting is Viser's, not the RTX endoscope.
The headless solver and FK checks pass. Automated browser screenshots could
not be verified because this host's headless browsers failed to initialize
WebGL reliably; no new screenshot is claimed as visual verification.

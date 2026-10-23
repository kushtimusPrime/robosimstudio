# SutureBot calibration and live inspection

This bundle preserves the fitted dVRK Si inspection scene without committing a
live workspace or the downloaded dataset. `restore.py` creates a new workspace
under `examples/live/twins/`; it refuses to overwrite an existing one. The main
robot asset and independent four-arm IK example are documented in
[the asset README](../../assets/embodiments/dvrk_si/README.md).

## Restore and run

From the repository root, in the configured RoboSimStudio environment:

```bash
uv run --no-sync python scripts/dvrk_si_calibration/restore.py dvrk_si_suturebot
cd examples/live/twins/dvrk_si_suturebot
uv run --no-sync python fetch_reference.py
uv run --no-sync --with xacro --with fast-simplification python build_multisession_assets.py \
  --session all --dvrk-model ~/dvrk_src/dvrk_model
./ctl up
```

The generator expects JHU's `dvrk_model` checkout at
`90472a933995acaa88d84eb5c58aeee4cd17e9f3`. Fetching uses bounded HTTP ranges:
only annotated calibration/test episode CSVs and selected left images are
downloaded, not entire tissue ZIPs. `--csv-only` skips images. The full survey's
124 episodes remain listed in the browser; unsampled downloads are explicitly
shown as unavailable. `dataset_fetch.py` can fetch the broader survey later.
Set `TMPDIR` to a directory on a drive with free space if needed for GPU builds.

The restored page starts on setup_017, in an empty scene without room walls or
a bench. **Inspect IK** seats the handles on the recorded frame and enables
interactive motion. **Browse SutureBot** opens the image comparison sidebar.
The image browser does not change the robot's setup or pose. To replay frame 94:

```bash
./ctl eval 'show_frame(sim, session, 94)'
./ctl snapshot --camera endoscope \
  --beside ref/dataset/tissue_4/3_knot_tying/20250429-205423-555549/left_img_dir/frame000094_left.jpg
```

Set `ACTIVE_PROFILE` in the restored `scene.py` to select another generated
setup. `render_multisession.py` renders one reserved frame per fitted setup.
Run it before selecting cached rendered comparisons in the image browser.
The original working twin and historical single-session wrist-camera experiments
are not needed by this bundle; its scope is the current ten-setup endoscope fit.

## Optional instrument appearance

The restored scene keeps the calibrated baseline by default. Set
`APPEARANCE = True` in its `scene.py` to generate and load the separate PSM1/PSM2
appearance layer; set it back to `False` to disable it. Materials, fine roughness,
a small inward DeBakey visual bevel and scope lighting improve the visible tools
without changing the calibrated model or physics.

See [usage, assumptions and checks](template/APPEARANCE.md), the
[six-frame real/baseline/updated overview](evidence/appearance/overview.png), and
[setup_017 frame 94 at native resolution](evidence/appearance/comparison_setup_017_094.png).
The [needle-driver close-up](evidence/appearance/closeup_setup_017_094_PSM1.png)
and [DeBakey close-up](evidence/appearance/closeup_setup_017_094_PSM2.png) use
identical image crops. This is an appearance improvement, not a new calibration
or a full exterior reconstruction. Custom scope lighting is RTX-only; Viser
shows the PBR surfaces under its own viewport lighting.

The [wider appearance survey](template/APPEARANCE_SURVEY.md) reviews 122 cached
episodes across 66 setup groups. Its [six-pose comparison](evidence/appearance_survey/review_overview.png)
shows the first approved appearance alongside modest collar/housing/shading
refinements. `REFINE_SURFACES = False` in `appearance.py` retains the first recipe.

## Optional exterior from the user scans

Set `EXTERIOR = True` to restore the complete instrument housings with pale-blue
polymer and tool-specific labels, add satin cover finishes and small visual
bevels, and detail the scope mounting cage. Keep `APPEARANCE = True` for the
previously approved endoscope-visible tools. Both layers can be disabled
independently; canonical meshes, physics, frames and calibration stay unchanged.

See [the exterior recipe and limitations](template/EXTERIOR.md),
[instrument housing comparison](evidence/exterior/tool_comparison.png),
[scope collar comparison](evidence/exterior/scope_comparison.png),
[robot covers](evidence/exterior/arm_comparison.png), and
[calibrated endoscope regression views](evidence/exterior/endoscope_preservation.png).
The page has dedicated free-view buttons for the exterior, each housing and the
scope collar. The 3.3 MB source assets are bundled; normal builds do not require
Blender or the original MOV files.

## PyRoKi with the upgraded visuals

From a restored twin with its setup assets and CSVs available:

```bash
uv run --no-sync python pyroki_preview.py --port 8091
uv run --no-sync python pyroki_preview.py --check
```

This separate Viser page imports the existing `scripts/dvrk_si_pyroki_ik.py`
solver unchanged and loads the same optional appearance/exterior asset as
Studio, including labels and sliding carriages. It starts at setup_017 frame
94; use `--start middle` for working-arm joint midpoints, or `--setup` and
`--frame` for another recorded configuration. PSM3 and ECM start locked.
Each joint can be locked, fully locked arms are excluded from the pose costs,
and jaws retain the selected recording's calibrated opening. Drag target arrows
and rings; use **View → Tools** for a closer look and **Reset targets and joints**
to return to the start. No collision avoidance is enabled. Viser lighting is
not the RTX endoscope lighting. The standalone page does not simulate physics.

Forward port 8091 when connecting over SSH. This page is independent of the
existing live Studio on 8081 and the original canonical-URDF IK example.

## Calibration and limits

The frozen candidate is `template/multisession_registration.json`. It fits shared
left-camera intrinsics and jaw dimensions/mappings, with separate rigid arm
registrations for ten setups. Intrinsics at 960×540 are fx=939.507, fy=930.059,
cx=480, cy=270, with no distortion. Fifty-six additional setup groups have survey
coverage but no fitted registration.

The camera origin is fixed at the physical scope's controller-tip frame, with
a half turn about z for OpenCV coordinates. The earlier effective camera offset
of 44 mm is absorbed by arm registration. This preserves the fitted image
projections; the generated URDFs agree with them within 0.029 px. This is a
nominal optical-centre assumption, not measured stereo/hand-eye calibration.
The 2 mm near plane clears the CAD lens cover. Inspection handles target the
PSM wrist-yaw pivots, matching the PyRoKi example; replay reseats handles and
updates grip targets and the camera frustum.

Against manually annotated real images, RMS errors are **10.66 px training**,
**19.07 px reserved frames**, and **24.01 px on two final held-out episodes**.
The 0.029 px consistency check does not describe accuracy against the photos.
The DeBakey blades are an approximation; jaw length/mapping are empirical, and
the visual profiles have not been validated for physical manipulation.

`template/REPORT.md` retains the full investigation report, including historical
artifact names from the original workspace. This bundle includes the frozen
fit, reviewed and pre-review landmarks, the dataset FK audit, and selected
results in `evidence/`; it does not include every historical image/log named
in that report. The scope-anchored comparison is
[simulation left, real image right](evidence/scope_anchored_comparison.png).

To evaluate the frozen candidate again from the restored workspace:

```bash
uv run --no-sync python validate_multisession.py \
  --labels multisession_final_test.json --suffix final_test
```

`calibrate_multisession.py --model geometry` fits a new candidate in `run/`;
it does not overwrite the frozen registration. Its training labels, selection
frames and final-test episodes are separate files. The original single-session
`registration.json` is retained only as initialization/prior data.

## Attribution

SutureBot: Haworth, Chen et al. (2025),
[jchen396/SutureBot](https://huggingface.co/datasets/jchen396/SutureBot),
**CC BY 4.0**. Episode identifiers, derived kinematic values, manual annotations,
and the reference half of the comparison image originate from that dataset.
The comparison and fitted parameters are modifications produced for this work.
Source images/CSVs otherwise remain downloadable from the original dataset.

The scripts and generated approximation geometry are Apache-2.0. JHU dVRK CAD
retains its MIT license; see the asset's provenance and license files.

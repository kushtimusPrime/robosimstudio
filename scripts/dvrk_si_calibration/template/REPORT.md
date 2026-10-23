# dVRK Si: SutureBot calibration across setups

## What still differs from the reference

The optional appearance pass is described in [APPEARANCE.md](APPEARANCE.md).
It adds steel/coating PBR surfaces, fine procedural roughness, existing CAD detail,
a 40 µm inward DeBakey visual bevel and scope-mounted RTX lighting. The six-frame
real/baseline/updated sheet is `run/appearance/overview.png`. Jaw profiles remain
approximate; highlights, tissue reflections and camera response still differ.
The older descriptions of missing polished steel below describe the baseline.

This is a reproducible **image-based calibration candidate**, not a factory
calibration or a pixel-exact match. It fits shared left-endoscope intrinsics and
shared jaw dimensions/mappings, with a separate rigid registration for each arm
in each fitted setup. Ten setups, one from each tissue archive, have fitted
profiles and RoboSimStudio renders. Fifty-six additional provisional setup groups
have survey images but do not yet have fitted registrations.

At 960×540, the selected model gives:

| Evaluation | Landmarks | RMS | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, ten setups | 230 | 10.66 px | 7.00 px | 18.35 px | 56.28 px |
| Reserved frames, used for model selection | 182 | 19.07 px | 12.72 px | 38.29 px | 68.17 px |
| Final two whole episodes, excluded from fitting/model selection | 34 | 24.01 px | 14.79 px | 42.01 px | 79.40 px |

The final test episodes are tissue 6 needle throw `20250501-194221-341946`
and tissue 7 needle throw `20250502-195952-837967`. PSM1 is 31.99 px RMS;
PSM2 is 13.38 px RMS. The largest error is PSM1 in tissue 7 at frame 298,
with its wrist turned toward the camera. No training CSV has the same SHA-256
as either final-test CSV. These are approximate manually observed screw/blade
landmarks, not a full silhouette metric or exact ground truth.

The reviewed renders are `run/multisession_render_sheet_1.png` through `_4.png`.
Tool tips and jaw directions agree at many poses. The housing outlines, sleeves,
blade thickness, serrations, engraved details and exposed wrist mechanisms still
differ. The simulation lacks polished-steel reflections, coaxial endoscope
illumination and vignetting. Tissue, needle and thread are absent, and the empty inspection background is not the real tissue background. Those differences are outside the
reported landmark scores.

Only the left endoscope was fitted across setups. Right-camera images are
available for inspection but its intrinsics/stereo baseline have not been fitted.
The original single-session wrist-camera fits remain available with
`ACTIVE_PROFILE = None`; their held-out RMS values were 24.0 and 36.8 px.
They have not been validated across this expanded dataset.

## Shortcuts and departures from the reference

Appearance is opt-in (`APPEARANCE` in `scene.py`), generated separately under
`assets/appearance/`. No camera/robot/tool frame, registration, timing, fitted jaw
length, mapping, collision, mass or inertia changes. The 40 µm chamfer is an
explicit visual assumption inside the original blade volume. Surface parameters,
roughness grain and scope illumination are plausible, not measured. Viser uses
its own lights; the paired scope emitters apply only to RTX endoscope renders.

- This is **kinematic replay**, not contact-driven suturing. The replay writes
  recorded joints, propagates mimics, runs FK and pauses. There is no physical
  manipulation routine, manipulated object, or motion take.
- Per-setup, per-arm rigid registration is explicitly allowed by the user. It
  changes the arm-to-optical-camera relationship. Its six parameters are an
  effective image registration, not a measured correction to SUJ encoders.
  Registered profiles should not be compared directly to unregistered logged
  camera-frame poses. Their RCM-relative kinematics remain the controller model.
- Shared jaw reach is fitted to 8.739 mm on PSM1 and 12.000 mm on PSM2. The
  generated profiles scale the existing jaw meshes and colliders along their
  length, by 0.9054 and 0.8000. Pivots, joint axes and tool-tip frames are unchanged.
  PSM2 reaches the fit's lower length bound; these dimensions are estimates,
  not independently measured CAD. Wrist housings are unchanged. This is not a
  complete reconstruction of the instrument surfaces.
- Each jaw leader uses `offset + gain * (logged_jaw + 0.35)`: PSM1 offset
  0.04394 rad/gain 0.62615; PSM2 offset −0.01369 rad/gain 0.31347. This empirical
  mapping is applied only in dataset replay. The inspection profiles now target
  the controller's wrist-yaw pivot, as the dedicated PyRoKi example does, rather
  than the library hands' nominal mid-blade grasp centres. Library hands retain
  their grasp TCPs. These preview profiles are not validated for manipulation
  or for jaw commands outside the observed range.
- Profile setup joints are re-solved from each recording's SUJ values, with any
  remaining rigid correction folded under its RCM. The optical camera is anchored
  at `ECM_camera_link`, with the dVRK-to-OpenCV half turn about z. The previous
  coordinate convention put it 44.37 mm outside the scope. Its effective correction
  is now absorbed into per-arm placement, preserving the fitted camera-to-arm
  transforms. The scope-tip centre is a nominal optical-centre assumption, not a
  measured left stereo entrance pupil. PSM3 is parked using recorded values.
  In setup_001, only PSM1 entered the new fit; PSM2 retains the previous estimate.
- The visible screw centre switches between the two surfaces of the wrist-yaw
  plate as the wrist rolls. Projection now uses the front-facing surface at
  ±2.65 mm, rather than treating every visible screw as one fixed face.
- One time offset is fitted per setup, ranging from approximately 0 to 7.79 CSV
  rows. Fractional rows are interpolated. This is an empirical image alignment,
  not independent timestamp calibration; the original approximately 233 ms
  reported delay does not explain every recording equally well.
- Shared optical/tool estimates are in `multisession_registration.json`; ten
  generated profiles are under `assets/multisession/`. The library URDF, default
  home and PyRoKi joint-lock behavior were not changed by this calibration pass.
  The original single-session fit remains in `registration.json` and its report
  is preserved in `run/REPORT_single_session.md`.

## Checks and comparisons

Appearance verification: six paired 960×540 renders (setup_017: 0/94/188 tuning,
219 evaluation; setup_015: 32/161 evaluation), identical joint/body/camera/handle
states and physics arrays, all 420 protected files unchanged. Projection change
is 0.0 px over 453 landmarks in ten setups. Render-to-render silhouette IoU is
99.64–99.93%, with no outward growth and at most 1.414 px inward edge retreat.
`run/appearance/verification.json`, `projection_check.json` and
`light_attachment_check.json` record checks; the close-up triptychs show both
tools wherever visible. The two final-test episodes were excluded from appearance
work. Studio gate passed builds, finite state, camera, zero reported overlap,
zero cart drift over 60 steps and finite state afterward; object-presence checks
were skipped without keys. This is not a physical manipulation test.
`run/appearance/studio_check.json` records the gate. Repository Ruff check and
format check passed (761 files). No broad library tiers apply to this live visual overlay.

The dataset inventory contains **1,891 CSV episodes, 662,298 rows and 1,885 unique
CSV checksums**. All CSVs were extracted with bounded HTTP Range requests; no
whole tissue ZIP was downloaded. Sampling now includes **2,232 stereo images
from 124 episodes**, spanning every one of the **66 provisional setup groups**
found by date/SUJ proximity. All requested sampled images are present. Setup
clustering is a survey heuristic, not authoritative session metadata.
`dataset_coverage.json`, `survey.json`, and `run/csv_checksums.json` record the
coverage; `run/full_setup_survey_1.png` through `_4.png` were visually inspected.
Coverage of images is not equivalent to calibration of all 66 groups.

`dataset_kinematics.py` evaluated the actual library URDF's RCM-to-tool-tip chain,
including pitch mimics, against every downloaded row for both PSMs. Maximum
position difference is **0.86764 mm**, maximum orientation difference **2.34477°**.
No episode exceeds 1 mm; 54 episodes have at least one orientation error above
1°. This establishes broad joint/FK consistency, not camera or mesh accuracy.
Full results and the tested URDF checksum are in `dataset_kinematics.json`.

Generated profile URDF projections agree with the optimizer across 453 projected
landmarks to **0.02898 px maximum**, **0.00429 px RMS**, including interpolation
and camera attachment differences (`run/profile_projection_check.json`).
This checks that the drawn robot uses the same geometry and registration that
were fitted, rather than displaying a separate fitted point cloud.

The frame-attachment correction is verified at recorded frames 0, 94 and 150
in `run/frame_attachment_check.json`: all four live IK tool centres agree with
their URDF controller frames within 0.00062 mm, and the scope camera origin agrees
within 0.00050 mm (float32 FK precision). Handles coincide with the tool centres,
and the frustum apex coincides with the rendered camera. The CAD lens cover
extends 1.5 mm beyond the controller tip, so the camera near plane is 2 mm.
Replay now reseats handles, updates their joint/grip targets, and refreshes the
frustum even while paused. Handles and the frustum are reduced to 35 mm for
inspection. This corrects stale targets after replay as well as the frame choice.
The updated endoscope comparison is `run/snap_058_vs_ref.png`; the remaining
image residuals reported above are unchanged at the reported precision.
Both working arms also solved 3 mm Cartesian moves with position errors below
0.00055 mm (`run/anchored_profile_ik_check.json`). The live scene gate passes,
and the page is left unpaused at frame 94 with a close oblique inspection view
of the tools and scope (`run/snap_060.png`). Changed helpers pass Ruff.

The verified renderer run is `run/rendered_profiles.json`: ten reserved-frame
comparisons, `snap_047_vs_ref.png` through `snap_056_vs_ref.png`. Each render was
guarded by an assertion that the intended profile had loaded, followed by replay
and pause in the same command. The earlier `render_multisession.log` contained a
save/rebuild race and is **not** the verified batch. Its pictures are retained as
debugging artifacts only.

The live gate passes builds, finite state, camera, zero reported penetration,
zero root drift and finite state after stepping. Object checks are skipped:
there is no manipulated object. These gate lines do not certify optical accuracy
or gripper control. New calibration helpers pass Ruff and format checks. Library
test tiers were not rerun for these twin-only changes.

The Viser page is **http://localhost:8081**. At the user's request, the live
inspection scene now uses the `void` cell and room-free `VOID` render preset.
There are no room walls or bench. The cart is placed at the world origin and
all camera-to-arm transforms are preserved. The page is unpaused for interactive
arm controls and framed around the whole robot. `run/snap_057.png` verifies the
unobstructed view. The independent four-arm PyRoKi page is at
**http://localhost:8090**, with tool-tip targets and per-joint/per-arm locks.
The CPU headless check passed all four-arm and individual-lock cases with exact
lock preservation (`run/ik_roomless_check.log`). The “Browse SutureBot” routine installs
a dataset browser with episode, sampled frame, stereo side, landmark overlay and
cached rendered-comparison controls. Yellow crosses are observed pixels; cyan
circles are the calibrated projection. Unannotated survey frames have no score.
Changing the image browser does not change the 3D robot's setup. The current 3D
preview is setup_017; edit `ACTIVE_PROFILE` in `scene.py` to select another built
profile, or set it to `None` for the original endoscope/wrist comparison.

Reproduction from this twin folder, with `TMPDIR` on the second drive:

```bash
export TMPDIR=/mnt/2nd_drive/home/justin/dvrk_src/tmp
# Already downloaded; these commands resume existing files.
uv run --no-sync python dataset_fetch.py csv
uv run --no-sync python dataset_kinematics.py
uv run --no-sync python expand_dataset_survey.py
uv run --no-sync python dataset_fetch.py images

# Writes a new candidate under run/, not the frozen registration.
OPENBLAS_NUM_THREADS=1 uv run --no-sync python calibrate_multisession.py --model geometry
uv run --no-sync python validate_multisession.py --labels multisession_final_test.json --suffix final_test

# Builds from the frozen multisession_registration.json.
uv run --no-sync --with xacro python build_multisession_assets.py --session setup_017
./ctl status --wait
./ctl eval 'assert profile["session"] == "setup_017"; show_frame(sim, session, 94)' --wait
./ctl snapshot --camera endoscope --beside ref/dataset/tissue_4/3_knot_tying/20250429-205423-555549/left_img_dir/frame000094_left.jpg
./ctl eval 'from dataset_browser import install; print(install(session)); session.paused = True' --wait
```

The fitting observations are in `ref/multisession_landmarks.json`. Reserved
frames were used to choose the optical model. Two earlier held-out episodes
were monitored during development (`ref/multisession_test_episodes.json`) and
are not the final blind test. The final test file is
`ref/multisession_final_test.json`, which records the frozen fit checksum.
Automatic SIFT proposals were too sparse to trust and were excluded from the fit.
Two training-label mistakes were corrected after full-resolution image review;
the original labels and review note are retained.

## Unsettled measurements and assumptions

The appearance layer does not resolve blade width/thickness, DeBakey tooth count,
neck shape, polymer chemistry, BRDF, wetness, real endoscope light/exposure or
lens response. It leaves existing image registration residuals visible. The
render-to-render silhouette score is not a real-image accuracy score.

The selected pinhole camera is **fx=939.507, fy=930.059, cx=480, cy=270**, at
960×540, with no distortion. Allowing principal-point and radial-distortion
parameters reduced training RMS only from 10.66 to 10.57 px and worsened reserved
frame RMS from 19.07 to 19.09 px; its second radial coefficient hit a bound.
That model was rejected. No SutureBot factory/hand-eye calibration file was found.
The earlier published JHU endoscope example was only an initialization reference,
not evidence of this camera's actual calibration.

The [Intuitive S/Si instrument catalog](https://dvrk.lcsr.jhu.edu/downloads/manuals/davinci-s-si-instrument-accessory-catalog-11-2015.pdf)
lists 10 mm needle-driver and 12 mm DeBakey jaws. The upstream meshes and the
previous generated DeBakey approximation use different pivot/blade extents.
Fitted reach, optical depth, arm registration and physical jaw calibration can
trade off. The current fit does not uniquely determine all those physical
quantities. Width, thickness, neck, serrations and sleeve geometry remain
approximate; no claim of metrological mesh accuracy follows from a low point error.

Remaining useful work is full-surface/shaft landmark validation, better alignment
of rapidly rotating wrists, checking whether time offsets vary by episode, and
fitting the other setup groups and stereo/wrist cameras. An exact image match is
not established by this pass, and the reported worst cases remain visible rather
than being removed from the evaluation.

# dvrk_si_suturing

## What still differs from the reference

Work in progress. A complete physical needle pass and thread pull-through has
**not** passed validation. The scene on port 8082 currently isolates a rigid
needle and the robot jaws. Contact-only hold, carry, return and release now pass
the diagnostic gates. Thread and tissue are excluded from this subgoal; coupled
thread/tissue solver convergence remains unresolved. None of the material constants
has been identified against the dataset yet. Tissue appearance, initial thread
configuration and endoscope response remain approximate.

The initial needle image fit uses setup_017, frames 0, 70 and 211 of
`tissue_4/2_needle_throw/20250429-205402-595720`. Frame 141 is reserved. The
half-circle hypothesis gives a 24.0 mm arc: traced-curve RMS is 1.9, 7.6 and
2.8 pixels on fit frames; the reserved curve RMS is 10.2 pixels and its tip
error is 21.7 pixels. Occlusion at frame 211 is **not** annotated as a visible
tip. These errors include fixed robot-registration error and manual annotation
uncertainty; they are not physical-parameter validation.

## Shortcuts and departures from the reference

No gripper welds, latches, fake grasp forces or prescribed needle/thread replay.
The earlier full-scene experiment connected the first thread node to the
needle through the host solver's two-way swage constraint; it is disabled in
the current isolated contact test. The current thread is a shortened, visible
120 mm-scale segment; the unobserved tail and its loading are not reconstructed.
The needle initial pose is adjusted from the image-only estimate by
(+0.379, -0.574, +0.176) mm in wrist coordinates to avoid starting inside a
finger. This is an initial-condition estimate, never a runtime attachment.
The diagnostic commands a 0.015 rad jaw-servo closure bias, separately from
the unchanged measured-jaw mapping. It is not calibrated actuator force.
The robot follows a synthetic 5 mm withdrawal and 0.15 rad roll, then returns
and opens its jaws. This tests contact mechanics; it is not dataset motion replay.

User authorized task-specific contact changes on 2026-10-23, conditional on
preserving IK and previous functionality. `contacts.py` creates a separate URDF
with 0.5 mm longitudinal convex slices of the approved PSM1/PSM2 visual jaws.
The old boxes enclosed the visible grasp gap: the prospective needle centre
was about 1 mm inside **both** boxes. Fine grooves remain approximated by the
convex slices. The canonical robot, visuals, joint/link frames, limits, mimic,
mass/inertia, fitted dimensions, camera intrinsics/mount, registration and lag
are unchanged. The disabled full-scene host solver uses 25 micron jaw SDF sampling; the
current isolated rigid test uses native MuJoCo convex contacts.
A local CPU MuJoCo adapter integrates the rigid robot in at most 1 microsecond
steps, retaining the same masses, inertias, actuators and mimic equations.
The previous 1.25 ms step allowed a jaw follower to move about 0.54 rad in
one control interval under closure. This numerical refinement is not a
calibration change. The passing diagnostic uses persistent double-precision
MuJoCo state, elliptic six-dimensional contacts, tangential friction inherited
from the shapes, torsional friction length 50 microns and rolling friction
length 10 microns. Contact solref is (4 microseconds, 1), solimp begins
(0.9999, 0.9999, 10 microns); mimic solref is (100 microseconds, 1).
These stiff finite-patch contacts are engineering assumptions, not fitted
SutureBot properties. Sub-nanometre numerical penetration is not a claim of
physical material accuracy.

The user explicitly suggested a rigid or mostly rigid phantom base. The disabled full-scene
8 mm-thick pad has an explicit fixed lower 2 mm backing and a deformable upper
layer, including a pre-existing slit. There is no pre-cut needle channel.
Puncture and thread-channel friction are not implemented yet.

## Readings judged

- **Contact-only milestone PASS:** `run/grasp_probe.json`,
  `run/grasp_probe_metrics.json`, `run/contact_carry_release.mp4` and
  `run/contact_carry_release_strip.png`. A 4.5 s simulation (about 349 s wall
  time) held, carried 5.10 mm, returned and released the freely integrated
  needle. Maximum held centreline-point drift relative to the wrist was
  0.111 mm, rotation 0.419 degrees, initial settling 0.023 mm and mimic
  error 8.65e-7 rad. There were zero jaw contacts after 4.3 s, and the released
  needle fell. Thresholds: drift <0.2 mm, penetration <20 microns, mimic error
  <1e-3 rad, wrist travel >4 mm, no late jaw contacts, release displacement
  >5 mm, all states finite. These are development gates, not measured real
  grip compliance. The video has one pre-routine frame with stale Studio
  panel jaw state; the numerical recording starts after recorded-pose reset.
- `run/viser_verified_start.png`, `run/viser_verified_held.png`, and
  `run/viser_verified_released.png`: inspected actual browser renders.
  Play and timeline scrubbing were exercised; final browser check recorded
  no JavaScript errors (`run/viser_browser_validation.json`). The headless
  browser needed its inherited SSH DISPLAY unset for WebGL.
- `run/grasp_half_timestep.json`, `run/grasp_timestep_comparison.json`:
  halving the rigid step from 1 to 0.5 microseconds over the first 0.3 s changed
  relative centreline positions by at most 0.231 microns and orientation by
  0.000864 degrees. This checks initial hold resolution, not the entire carry.
- `recorded_grasp_viser.py` serves the actual saved physics states on port
  8092, with Play, scrub and three viewpoints. It is read-only playback;
  the Studio routine on 8082 performs fresh physics. Viser viewport lighting
  and aspect differ from calibrated RTX endoscope renders. The close-up
  viewport clips the first 15 mm to avoid looking through the endoscope
  housing; no physical camera mount or calibration was edited.

- `run/jaw_contact_proposal.png`: inspected comparison of visual jaws and old
  boxes; contact change subsequently authorized.
- `run/contact_regression.json` and `.log`: PASS for frozen calibration hashes,
  every non-collision XML field, all 82 links at nine recorded frames, four-arm
  PyRoKi IK, individual/all-arm locks, and rendered hierarchy versus PyRoKi FK
  at three configurations.
- `run/needle_fit.png`: inspected multi-frame needle projection hypothesis.
- `run/snap_001_vs_ref.png` through `snap_005_vs_ref.png`: inspected iterations;
  early images exposed incorrect initial tissue depth and uncontrolled jaw
  opening on the page. The current development page is held on rebuild.
- `run/preload_hold.mp4`, `run/preload_hold_strip.png`,
  `run/preload_hold_before.json`, `run/preload_hold_after.json`: inspected failed
  hold. The needle rotated substantially; this is not a successful grasp.
  Swage error remained 0.11 microns and max axial thread strain 0.27%, but the
  solver stopped at its iteration limit. Finite state alone is not a pass.
- `run/backing_validation.json`: 1,981 nodes, 8,448 positively oriented tets;
  629 fixed nodes in the lower 2 mm and 1,352 free nodes above it.
- `run/refined_hold.mp4`, `run/refined_hold_strip.png` and
  `run/refined_hold_measurements.json`: inspected failed hold with 0.05 rad
  closure bias. Needle surface moved 17 mm and rotated. Swage gap was 1.36
  microns; solver still STOPPED at 32 nonlinear iterations. The fixed backing
  moved at most 0.038 microns, versus 4.58 microns in the free tissue under
  this light loading. This verifies the backing constraint, not tissue stiffness.
- A separate native rigid-needle experiment (`native_needle.py`, disabled in
  the current scene) exposed unstable explicit swage-force exchange. It is
  retained for investigation, not presented as a working alternative.
- SOFA 26.06 was installed outside the repository and a free-fall smoke test
  passed. Its barycentric/sliding constraints are a candidate for puncture;
  no SOFA suturing integration is claimed.
- Full-episode manipulation and final physical gates are pending.

## Assumptions

`parameters.json` contains explicit, unfitted engineering priors in SI.
`sources.json` records primary sources and the two final-test episodes excluded
from tuning. The SutureBot paper identifies a 3-DMed pad and green braided
polyester 3-0 Ethibond; it does not provide enough mechanical specifications to
uniquely identify the simulator. The Ethicon catalog offers multiple needle
variants. The image-fit circular arc does not establish a manufacturer SKU.
Needle wire radius 0.24 mm, tissue thickness/backing, elastic/damping/friction
values, and initial thread layout are assumptions. The base is fixed; no
phantom sliding on the support is modeled.

All calibration files copied into this twin are hashed in
`frozen_calibration.json`. The live result is self-contained in this folder;
external experimental solver installs are under `~/dvrk_src`, not the root
filesystem cache. Cached original SutureBot references are used. No complete
tissue ZIP was downloaded. The two pre-existing final-test episodes remain
untouched by fitting.

The finer-step contact experiment uses 10 host substeps and 64 Newton
iterations; at 15 ms it still reached the iteration limit (residual 0.0225),
and at 35 ms its residual was 0.310 (`run/finer_steps_measurements.json`).
The 40 ms test completed in 318 seconds; reducing timestep alone has not
validated this contact model. `run/snap_006_vs_ref.png` was inspected at the
restored recorded start: tool poses retained, tissue flat and untextured,
thread layout approximate, needle image fit still limited. That earlier failed full-scene experiment is retained for diagnosis. The live
scene now opens the isolated passing contact subgoal, paused at its start.
A complete needle pass, puncture and thread pull-through remain pending.

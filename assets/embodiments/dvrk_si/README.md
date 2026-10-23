# dVRK Si patient cart

One URDF for the cart, its frozen setup joints, PSM1/PSM2/PSM3 and ECM.
RoboSimStudio and `scripts/dvrk_si_pyroki_ik.py` use this same file.

## Sources and reproduction

- JHU [dvrk_model](https://github.com/jhu-dvrk/dvrk_model) at
  `90472a933995acaa88d84eb5c58aeee4cd17e9f3`: meshes, arm inertias and xacros,
  MIT (`LICENSE`).
- JHU [sawIntuitiveResearchKit](https://github.com/jhu-dvrk/sawIntuitiveResearchKit)
  at `7435458`: `PSM_Si.json`, `ECM_Si.json`, instrument DH, joint limits,
  efforts and trajectory velocities.
- [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot), Haworth,
  Chen et al., 2025, CC-BY-4.0: the recorded values in `setups.json`.
  `home` is `tissue_7/3_knot_tying_recovery/20250502-204132-813971_recovery`,
  CSV row 372 (zero-based). `calibration` retains
  `tissue_10/2_needle_throw/20250430-100535-002019`, row 0.
- Intuitive Surgical's [October 2011 instrument catalogue](https://dvrk.lcsr.jhu.edu/downloads/manuals/davinci-classic-s-si-instrument-accessory-catalog-08-2011.pdf),
  printed page 27: DeBakey 420036 jaw length 12 mm; Cadiere 420049 is 20 mm.

```bash
TMPDIR=/mnt/2nd_drive/home/justin/dvrk_src/tmp uv run --with xacro --with fast-simplification \
  python scripts/dvrk_si_urdf.py ~/dvrk_src/dvrk_model
# An independent calibration asset for replay; do not overwrite the default rig:
uv run --with xacro --with fast-simplification python scripts/dvrk_si_urdf.py \
  ~/dvrk_src/dvrk_model --setup calibration --out /path/to/calibration
uv run python scripts/dvrk_si_pyroki_ik.py --check
uv run python scripts/dvrk_si_pyroki_ik.py --start suturebot
```

Edit the generator, not `dvrk_si.urdf`. `setups.json` is source data for the
frozen setup and reproducible replay tests; regeneration preserves it.

## Geometry and kinematics corrections

| Change | Reason |
| --- | --- |
| Prefix each arm's joints and mimic references | All four arms share one IK problem without name collisions. |
| Freeze SUJs; solve working PSM setup against each session's logged camera-frame RCM poses | Upstream setup FK differs from controller DH by about 68 mm. The camera and both working PSM bases must come from the same session. |
| Re-zero pitch; make parallelogram axes exactly parallel; solve its two hinge translations against exact-RCM FK | Upstream misses the RCM across pitch. Hinge corrections are `(-1.3873, 0, 8.5832)` and `(0.8771, 0, -3.7991)` mm. Existing physical mimic joints remain connected; no extra actuated joints. |
| Put insertion on the CAD bore at local x=106.49 mm | Upstream put the PSM shaft at x=115.23752 mm, 8.74752 mm outside the bore centre. ECM roll's lateral offset is removed too. |
| Trim only the hidden proximal shaft to 467 mm behind the wrist | The source shaft extended 45.6 mm beyond the housing's rear. Distal CAD is preserved; collider and mass follow the corrected length. |
| PSM tool-tip frame on the wrist-yaw pivot; ECM camera frame on scope tip | These are the controller targets, distinct from the instrument's mid-blade grasp centre. |
| Remove phantom jaw; drive jaw_1, mirror jaw_2 | Recorded opening is twice jaw_1; negative openings represent squeezing. |
| PSM1/PSM3 use large needle driver 420006; PSM2 uses a generated DeBakey 420036 approximation | No DeBakey mesh ships upstream. Its 12 mm solid straight blades replace the visibly wrong 20 mm fenestrated Cadiere. Width, thickness, neck and serrations are estimates, not manufacturer CAD. |
| Controller ranges, efforts and speeds; missing inertias and jaw colliders; remove RCM marker spheres | Valid simulator input and hardware visuals. |

The generated DeBakey meshes are Apache-2.0, authored by this repository's
generator. The trimmed shaft remains a derivative of JHU's MIT mesh. Other
meshes retain their source license. Large meshes are decimated.

## Home and IK

Home was selected by maximizing the minimum normalized distance to a joint
limit across both working arms in the 87 downloaded CSVs. The chosen recorded
posture has at least **20.8% range clearance** in every working-arm joint;
the old calibration first frame had only 2.6%. Its own SUJs and ECM pose are
used. PSM3 remains parked; its slightly negative insertion encoder reading is
clamped to the controller's zero stop. Empty jaws start half open instead of
reproducing the recording's squeeze on thread.

The PyRoKi page defaults to mid-range working arms, while keeping PSM3 parked
and the ECM at the recorded pose. `--start suturebot` selects the new recorded
home. Each arm has individual joint locks and a whole-arm toggle. Fully locked
arms contribute no pose cost; locking all arms bypasses optimization. Jaws
remain outside tool-tip IK. No collision-avoidance costs are constructed.

`--check` asserts reachable four-arm targets, partial locks, whole-arm locks,
unchanged locked coordinates and pose error thresholds. Static tests check
sampled home-session replay to 0.2 mm and full-range shaft/bore/RCM alignment
to 0.1 µm. Across all 398 home-session rows, maximum recorded-pose errors are
0.120 mm / 0.320 degrees; across all 489 calibration rows, 0.244 mm / 0.782
degrees. These whole-session maxima include variation between logged joint
and Cartesian values; the previous 4.7 mm systematic geometry residual is gone.

## Visual validation and remaining limits

The versioned [SutureBot calibration bundle](../../../scripts/dvrk_si_calibration/README.md)
restores a live inspection workspace with its own generated URDF profiles,
manual image landmarks and calibration scripts. It preserves shared endoscope
intrinsics and jaw geometry, with per-arm registration for ten setups. Against
real 960×540 images, landmark RMS is 10.66 px on training frames, 19.07 px on
reserved frames, and 24.01 px on two whole episodes excluded from fitting and
model selection. These are approximate manual landmarks, not metrology accuracy.

The camera is anchored at the scope's controller tip, rotated into OpenCV axes;
the earlier effective optical offset is absorbed into arm registration. Generated
profiles match the fitted projections within 0.029 px, which is consistency
with the model, not accuracy against photos. Inspection handles use the dVRK
wrist pivots and are reseated when recorded frames change. The optical centre is
a nominal assumption, not factory stereo calibration. The library URDF retains
logged kinematics; only validation profiles apply empirical image corrections.
The reference views cannot show the external carriage or cannula; those are
checked against CAD and FK, not claimed as photographically verified.

The new home uses an estimated work surface at cart z=0.460 m, centred at
(0.702, 0.0094), so the open DeBakey jaws clear the table. The previous 0.488 m
estimate intersected the new posture. On bench_075 the cart remains 0.290 m
above the floor; the physical table height is unknown. Inter-arm physics
contacts are disabled because the source colliders overlap and displace the
ECM at rest; workpiece contacts remain available. Mixed 6-DOF PSM and 4-DOF ECM joint-position actions remain unsupported;
Cartesian modes work. Servo gains are simulation tuning, not measured controller
gains; effort limits remain per-joint URDF values, including the ECM's higher
limits. Collision avoidance is deliberately absent. This asset
is a simulation model, not a calibrated hand-eye model of the physical rig.

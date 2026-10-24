# dVRK Si needle contact milestone

This bundle restores the verified **contact-only hold, carry, return and release**
test. It does not yet implement a validated tissue puncture or thread pull-through.
The physical coefficients are engineering priors, not fitted SutureBot mechanics.

From the repository environment:

```bash
uv run --no-sync python scripts/dvrk_si_suturing/restore.py my_needle_contact
cd examples/live/twins/my_needle_contact
uv run --no-sync python recorded_grasp_viser.py --port 8092
```

Forward port 8092 over SSH. Select **Play recorded physics**, or scrub the
timeline. This is a saved simulation, with PBR materials under Viser lighting.
The close-up viewport clips the near scope housing; it does not alter calibration.

For fresh physics, use `./ctl up` in that twin, pause and reset as documented,
then run **Needle contact: hold, carry, release**. See [the routine instructions](template/CONTACT_PROBE.md).
The 4.5-second simulation took about 349 seconds on the development machine.
Keep `TMPDIR` on a disk with free space for GPU builds.

The robot held and carried the freely simulated needle 5.10 mm, then released
it without a gripper attachment. Maximum held needle centreline-point motion
relative to the wrist was 0.111 mm. Halving the rigid timestep changed positions
by at most 0.231 microns over the initial 0.3 s hold. See [metrics](evidence/grasp_probe_metrics.json),
[the recording](evidence/contact_carry_release.mp4), and the browser renders
[held](evidence/viser_verified_held.png) and [released](evidence/viser_verified_released.png).

Task-specific convex jaw contacts replace the oversized collision boxes in a
separate generated URDF. Calibration, joint/link frames, limits, mimic mappings,
visuals, masses and inertias are preserved. `validate_contacts.py` checks frozen
hashes, non-collision XML, recorded FK, all four arms' IK, and joint locks.
`analyze_grasp.py run/grasp_probe.json` recomputes the contact gates.

The bundle includes the frozen setup_017 assets and a compressed source CSV,
so restoring this milestone needs neither a dataset ZIP nor a JHU source clone.
Generated appearance/contact assets are rebuilt in the new twin. The recorded
physics states remain unchanged; only their local asset paths are relocated.

The [report](template/REPORT.md) also documents earlier failed coupled thread/tissue
experiments, whose code remains disabled. Their scratch runs are not included.
The needle fitting helpers require the corresponding cached left reference images;
the ready-to-run contact milestone does not. The two final-test episodes were
not used. Next milestones are a validated needle–thread swage, tissue indentation
with fixed backing, puncture, then thread pull-through.

## Attribution

SutureBot: Haworth, Chen et al. (2025), [jchen396/SutureBot](https://huggingface.co/datasets/jchen396/SutureBot),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled CSV is from
`tissue_4/2_needle_throw/20250429-205402-595720`; setup and needle initial conditions
are derived from that dataset. This work modifies the geometry and simulates a
synthetic diagnostic motion. The comparison strip shows simulation only.
JHU CAD retains its bundled MIT license. Scripts and generated approximations
use Apache-2.0. Approved exterior assets and their attribution are inherited
from `scripts/dvrk_si_calibration/template/assets/exterior_source`.

## Packaging validation

Restored into a new twin without the original workspace: regenerated assets,
matched all 38 calibration hashes and the saved contact metrics, passed the
82-link/nine-frame and four-arm IK checks, and built/stepped the restored Studio
scene. Fast tier: 2,436 tests passed (20 skipped). Repository Ruff checks passed.

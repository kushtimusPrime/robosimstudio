# Contact-only needle milestone

Open http://localhost:8092 and select **Play recorded physics**. The timeline
can be scrubbed while paused; **Tool close-up** shows the needle grasp.
**Whole robot** shows the actual arm motion. This viewer replays saved simulated
states; it does not run physics in the browser. Viser lighting and viewport
aspect are not the calibrated RTX endoscope image.

From this twin folder, start the viewer with the repository environment:

```bash
../../../../.venv/bin/python recorded_grasp_viser.py --port 8092
```

Fresh physics runs in Studio (port 8082):

```bash
./ctl eval 'live.hold=True; session.set_paused(True)'
./ctl reset
./ctl eval 'from multisession_preview import replay; replay(sim,session,profile,0,params["episode"])'
./ctl play 'Needle contact: hold, carry, release' --record contact_carry_release --wait
```

The 4.5-second diagnostic currently takes about six minutes to simulate. The
scene starts with `ISOLATED_GRASP = True`: rigid needle and robot only, with
no thread or tissue. The routine writes `run/grasp_probe.json`; the replay
requires passing `run/grasp_probe_metrics.json`, computed with:

```bash
../../../../.venv/bin/python analyze_grasp.py run/grasp_probe.json
```

The recorded synthetic motion holds, withdraws 5 mm while rolling 0.15 rad,
holds, returns, then opens the jaw. It is a contact diagnostic, not reproduction
of a SutureBot needle pass. Needle motion is freely simulated from contact and
gravity, with no gripper attachment. Contact coefficients remain assumptions.
See REPORT.md for evidence, numerical settings and the unfinished coupled
thread/tissue milestone.

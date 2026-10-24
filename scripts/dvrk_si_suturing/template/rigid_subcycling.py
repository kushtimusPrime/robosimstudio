# SPDX-License-Identifier: Apache-2.0
"""CPU MuJoCo microsteps without an extra host/device copy per microstep.

The 1.25 ms baseline step lets an implicitly damped driven jaw and its tiny,
unactuated mimic follower separate by ~0.54 rad under a closure command.
Subcycling refines integration without changing masses, joints or actuation.
"""

import math
from types import MethodType


def install(pipeline, max_step=5e-6):
    solver = getattr(pipeline.solver, "newton_solver", pipeline.solver)
    if not solver.use_mujoco_cpu:
        raise ValueError("Rigid subcycling requires the CPU MuJoCo backend")
    # Keep the double-precision native state between steps. Re-importing the
    # float32 display state rounds a 0.7 m world position by tens of nanometres,
    # comparable to the loaded steel contact penetration. Reset/run sync once.
    solver.update_data_interval = 0
    solver._update_mjc_data(solver.mj_data, solver.model, pipeline.state_0)
    pipeline.use_cuda_graph = False
    pipeline._graph = None

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        self.validate_observables(observables, contacts)
        self._apply_mjc_control(self.model, state_in, control, self.mj_data)
        if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
            self._update_mjc_data(self.mj_data, self.model, state_in)
        count = max(1, math.ceil(dt / max_step))
        self.mj_model.opt.timestep = dt / count
        self._mujoco.mj_step(self.mj_model, self.mj_data, nstep=count)
        self._update_newton_state(self.model, state_out, self.mj_data, state_prev=state_in, observables=observables)
        self._step += 1

    solver.step = MethodType(step, solver)

# Physical-transfer policy revision

The earlier long run was preserved after the user reported rough walking,
poor yaw and backward falls. Validation confirmed good speed tracking could
coexist with large backward yaw error. The desktop was also using an older
intermediate checkpoint. Neither was suitable evidence for physical deployment.

The replacement uses an identical motor-target controller in GPU training,
CPU MuJoCo and the physical sensor API. It ramps velocity/yaw requests, filters
complete joint targets, and limits requested target speed to 2.5/3/3 rad/s per
leg. Its reference cadence is 1.5 Hz with 8 mm lift. These remain provisional
settings. A target-rate bound does not establish a measured servo torque-speed
curve or prevent every back-driven joint-speed excursion.

The actor observes 41 fields: the original 29 plus applied target offsets and
target velocities. These twelve additional values are software-controller state,
not extra physical sensors. Earlier actor/critic input mappings are preserved by
zero-padding their input weights; the new stage uses fresh optimizer state and
exploration. Tests verify that transfer does not discard the learned mapping.

Training explicitly samples straight forward, straight backward, pivot turns,
and combined commands. Strong yaw-error costs, full-target speed/acceleration
costs, and overspeed costs supplement tracking. The randomized stage adds
RSL-RL mirror consistency with a tested lateral reflection of gyro, gravity,
joints, commands, phase and controller state. The first learning stage adapts to
the revised controller; the subsequent 20,000-update GPU stage introduces:

- Friction from 0.55 to 1.05 and pseudo-inertia density log perturbations ±0.08.
- PD gains scaled 0.85–1.15 and effort limits scaled 0.75–1.05.
- Actuation delay of 0–20 ms, resampled at command intervals with lag holding.
- Gyro and encoder bias ±0.01, plus the documented observation noise.

These are explicit uncertain simulation parameters. Actual servo specifications,
measured mass/COM, IMU alignment/bias, joint feedback and bus timing are still
needed. Self-collision and CAD mechanical clearance remain unvalidated. Domain
randomization does not replace those measurements.

Validation scores the weakest command group first, then aggregate stepping
success and worst yaw error. The frozen test set uses separate seeds and friction
0.5/0.8/1.1. CPU tests include starts, stops, reversals and pivot requests. Every
failed direction remains in results and videos. Software tests are not walking
acceptance results, and simulation acceptance is not hardware-transfer evidence.

## Reusing exactly the same outer policy

`docs/physical_policy_contract.json` defines the required sensor packet. Measured
joint angles/velocities must be in URDF coordinates; gyro is body-frame rad/s and
projected gravity is a unit vector. Joint commands cannot simply substitute for
unavailable joint-state measurements without a separately validated estimator.

Exports contain the actor with normalization, controller configuration and a
small CPU runtime source package. The ONNX path does not require MuJoCo or the
training stack. In a calibrated host program, running at 50 Hz:

```python
from biped_ppo.runtime import PositionPolicyRuntime
policy = PositionPolicyRuntime.from_export(export_directory)
targets_rad = policy.step(q, qd, gyro_body, gravity_body, requested_velocity)
```

The runtime returns six position targets and performs no hardware I/O. The actual
communication adapter and servo/joint mapping depend on the robot's hardware.
The checkpoint runtime and simulator targets match exactly in the 100-step sensor
handoff test. ONNX numerical parity is verified separately at export.

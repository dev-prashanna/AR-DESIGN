# Frozen initial protocol

Written 2026-11-01. The source URDF is provisional. The appearance CAD is not the
physical model. All research claims remain simulation-only and exploratory
until a matched matrix is completed.

The 2x2 matrix is configs/experiments/ablation-matrix.yaml: curriculum on/off
crossed with randomization on/off, seeds 0,1,2 sequentially. Fixed rewards,
architecture, timestep, transition budget, test suites. Without curriculum,
forward command U(0,.06) m/s and yaw U(-.15,.15) rad/s from the start; 20% standing.
With curriculum: zero commands, then U(0,.03) forward, then final distribution.
Promotion uses >=8 completed episodes per update, mean survival >90% and
velocity RMSE <.025; demotion when survival <50%. Each change logs its trigger.
Evaluation commands and seeds are fixed in separate validation/test YAMLs.
Never tune against test. Select best validation success rate, ties broken by
velocity RMSE then earlier update; report all evaluated checkpoints.

Success: no fall, >=95% episode duration, velocity RMSE <=.025 m/s, yaw RMSE <=.1
rad/s. Fall: torso below .075 m or tilt >.9 rad. Numerical invalidity is an error.
Cost of transport: integral(sum(abs(tau*qdot)))/(mass*9.81*forward_distance), excluding
non-forward commands and integrated body-frame forward distance <=.02 m. This is absolute mechanical work, not battery consumption.

Use independent trained seeds for mean/spread and bootstrap 95% intervals
(over run means, 10,000 bootstrap samples with seed 0) only when >=3 matched
training seeds exist. Episode repetitions do not increase the training-seed n.
Record failures and exclusions. Raw plots have no smoothing; any added smoothing
must show raw curves and disclose the exact filter/window.

Initial randomization (provisional, not measured uncertainty): per-environment
startup foot/body friction U(.7,.9); per-body log-density alpha U(-.025,.025)
scales both mass and inertia by exp(2*alpha); per-reset Kp/Kd U(.9,1.1)*default;
actor/critic sensor noise as configured; command delay U{0,1,2} physics steps.
Held-out test friction is .65/.95. Held-out mass/delay, COM offsets, pushes,
terrain and actuator-strength grids are pending implementation/calibration;
no robustness claim on those is allowed. Current randomization is an initial
subset of the full specification, not a completed sim-to-real model.

A diagnostic pilot has a wall-time cap including environment initialization.
Caps are checked at rollout boundaries; compilation and a current rollout can
extend the nominal deadline. Research sweeps are explicit commands, never setup
side effects. No paid compute. Parallel worlds share one policy, and seed runs
have separate weights and folders.

All conditions and evaluation use a common small reset distribution: each joint
receives U(-.005,.005) rad around nominal, clamped to limits; root roll/pitch
U(-.01,.01) rad; positive spawn lift U(0,.0005) m. Velocity starts zero. These
provisional initial disturbances provide distinct seeded evaluation episodes;
they are not a claimed broad robustness domain. They are separate from the
optional randomization factor in the study.

## Initial bounded study, frozen before held-out results

Use the pilot transition budget of 1,536,000 per independent policy at 256 worlds,
24-step horizons, 250 updates, and three training seeds per condition. This is a
short-budget initial study, not the 30,720,000-transition research profile.
Only the final checkpoint is eligible for the initial study's validation/test
comparison; periodic checkpoints remain diagnostics. Validation selection over
that eligible set retains best_validation.pt. No test result may tune or select
checkpoints. Reuse a completed pilot only if its task, seed, batch, episode length,
condition and transition budget exactly match; record reuse explicitly. Resume
smoke tests are excluded from the research matrix. Cap the sequential study sweep
at 1,200 seconds, saving any missing cells for a later explicit continuation.

Supplementary stopping diagnostic (defined before observing stopping outcomes):
seeds 1101--1103, 0.045 m/s forward for 5 s followed by zero command for 5 s,
friction .8. Report whether the stop phase was reached and post-stop planar-speed
RMS separately. This diagnostic is distinct from the frozen main test comparison
and cannot select/tune a checkpoint. Main test cases are unchanged.

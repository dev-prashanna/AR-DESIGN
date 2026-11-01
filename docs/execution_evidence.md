# Execution evidence and handoff — 2026-11-01

GitHub publication was requested after the initial local handoff. See
[publication.json](publication.json) for the server-recorded repository details;
the initial handoff status below predates that publication.

The local pipeline and short-budget experiment completed. **Reliable commanded
walking did not meet the declared acceptance criteria.** Forward motion occurs,
but heading control is poor. All results are simulation-only on a provisional
URDF; the appearance CAD is a separate design concept.

## Measured results

- 12 independent policies: four conditions x training seeds 0,1,2.
- 1,536,000 transitions each; 18,432,000 aggregate transitions total.
- 576 validation episodes, 1,440 held-out test episodes and 144 supplementary
  stopping episodes; these are within-policy repetitions, not independent seeds.
- Every condition's mean test success is 20.0%, accounted for by standing cases.
  Full moving-command success is 0% across all conditions at this budget.
- Validation-selected policy: baseline, seed 0.
  On the 0.045 m/s cases, mean integrated body-frame forward distance is about
  0.436 m and yaw-rate RMSE about 0.468 rad/s (limit 0.1 rad/s).
- Selected ONNX max absolute parity error: 3.58e-07;
  declared tolerance 1e-05. Normalization is embedded.
- All twelve policies completed 250 updates; measured training wall sum is about
  17.34 minutes. Initialization/runtime overhead differs between standalone pilot
  reuse and the sequential sweep, so this is an equal-sample study, not an
  isolated equal-time hardware contest.
- CPU and GPU checks pass. Final test output: docs/final_tests.log. CPU-only
  execution with CUDA_VISIBLE_DEVICES='' also passes (docs/cpu_only_tests.log).
- The report compiles to paper/main.pdf (eight pages at this milestone).
- Active authenticated GitHub account: dev-prashanna. No target repository was
  provided and no remote was created or pushed. URDF redistribution rights remain
  unspecified. The GitHub destination question is pending.

## Executed commands

Commands below ran in this directory; `scripts/run` clears inherited ROS Python
paths for that command only. Existing Python/CUDA/ROS installations are preserved.

```sh
uv lock
uv sync --locked
scripts/run biped doctor
scripts/run biped validate-model --robot configs/robot/custom.yaml
scripts/run pytest -m 'not gpu' -q
scripts/run pytest -m gpu -q
CUDA_VISIBLE_DEVICES='' scripts/run pytest -m 'not gpu' -q
scripts/run biped benchmark --num-envs 32 64 128 256 --steps 50
scripts/run biped train --config configs/experiments/smoke.yaml --seed 0 --max-wall-time 180
scripts/run biped train --resume runs/20261101T022550Z_train_61bf8501/last.pt --max-wall-time 60 --max-transitions 4608 --iterations 6
scripts/run biped train --config configs/experiments/pilot.yaml --seed 0 --max-wall-time 120
scripts/run biped train --config configs/experiments/randomized-smoke.yaml --seed 0 --max-wall-time 60
scripts/run biped sweep --config configs/experiments/initial-study.yaml --skip-completed --max-wall-time 1200
scripts/run python scripts/evaluate_study.py
scripts/run biped analyze --runs runs
scripts/run biped paper --results results/aggregates
scripts/run pytest -q
```

Individual evaluate/export commands were also executed for the pilot and smoke
policy; console logs are in docs/. Final study evaluation is batched over fixed
seed/command/friction cases. Validation selects eligible finals before test use.
Only the first completed matching curriculum seed-0 pilot was reused; backend
source equivalence is verified in docs/training_source_audit.json. Failed setup
attempts were corrected (ROS plugin leakage, initial constraint buffer/pose,
integer nominal-state dtype, and the explicit qfrc_actuator API). These are not
reported as successful runs.

## Actual study run IDs

| Condition | Training seed | Run ID | Transitions | Measured wall seconds |
|---|---:|---|---:|---:|
| baseline | 0 | 20261101T023310Z_train_fc523e6c | 1,536,000 | 83.5 |
| baseline | 1 | 20261101T023433Z_train_d690c49e | 1,536,000 | 80.7 |
| baseline | 2 | 20261101T023554Z_train_5a11c811 | 1,536,000 | 81.8 |
| curriculum | 0 | 20261101T022655Z_train_a424f977 | 1,536,000 | 92.3 |
| curriculum | 1 | 20261101T023716Z_train_bc2cbf0d | 1,536,000 | 79.0 |
| curriculum | 2 | 20261101T023835Z_train_adba2a09 | 1,536,000 | 82.5 |
| randomization | 0 | 20261101T023957Z_train_aecdb681 | 1,536,000 | 88.7 |
| randomization | 1 | 20261101T024126Z_train_4f4a86a7 | 1,536,000 | 88.9 |
| randomization | 2 | 20261101T024255Z_train_c8f04518 | 1,536,000 | 89.2 |
| both | 0 | 20261101T024424Z_train_11c8aa5d | 1,536,000 | 90.7 |
| both | 1 | 20261101T024555Z_train_1172c025 | 1,536,000 | 94.4 |
| both | 2 | 20261101T024730Z_train_f391b26e | 1,536,000 | 88.5 |

## Artifacts and reproduction

- Model audit: docs/model_audit.json and docs/model_audit.md.
- All installed packages/licenses: docs/dependency_inventory.json; exact lock
  uv.lock; upstream release-tag SHAs docs/upstream_versions.json.
- Independent-run aggregates and uncertainty method: results/aggregates/results.json.
- Claims-to-artifacts mapping: paper/result_manifest.json.
- Run manifests: experiments/manifests; generated reproduction configurations:
  experiments/resolved/<run-id>.yaml. Example:
  `scripts/run biped train --config experiments/resolved/<run-id>.yaml`.
- Raw metrics, episodes, checkpoints and trajectories: ignored runs/<run-id>/.
- Selected portable policy and actual recorded-state videos: artifacts/selected_policy/.
- Main paper sources and bibliography: paper/main.tex and paper/references.bib.

Resuming restores actor, critic, optimizer, normalization, RNG, counters and
curriculum state. Physics resets; continuation is not bitwise simulator replay.
The named network and PPO settings are fixed in the versioned backend source;
architecture and all outer run settings are serialized with each run. The GUI
viewer command is provided; offscreen renders were exercised, while interactive
GUI use was not part of the automated checks.

## Remaining work

1. Improve heading control and reach full moving-command acceptance. Longer
   runs and reward/curriculum diagnosis are needed; the initial study does not
   establish reliable walking or an advantage from randomization/curriculum.
2. The 30,720,000-transition full.yaml profile and its 12-policy matrix were not
   run. Linear extrapolation from the short runs suggests roughly 5--6 GPU hours
   for that matrix, with substantial uncertainty; this is an estimate, not a run.
3. Calibrate actual masses, inertias, servo models/gearing, torque-speed response,
   delay, contact geometry and CAD-to-link mapping. Self-collision remains disabled.
4. Extend and test held-out mass/delay/COM/actuator-strength/push/terrain grids.
   Current randomization covers density-consistent mass/inertia, friction, PD
   gains, sensor noise, small reset disturbances and short command delay.
5. Confirm repository destination and asset redistribution permission before
   GitHub publication. No public history was changed, and no upload is claimed.
6. Physical deployment requires the separate staged procedure in
   docs/hardware_validation.md; policy export does not authorize hardware motion.

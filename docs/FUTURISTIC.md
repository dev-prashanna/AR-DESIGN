# Futuristic MicroDuck in MuJoCo

The actual 122 surfaces in `assets/robot/source/cad/MicroDuck_Futuristic.FCStd`
are exported as individual STL meshes and attached to six articulated joints.
`assets/robot/meshes/futuristic/manifest.json` records the source, part mapping,
colors, triangle counts and hashes. Millimetres are converted to metres and the
CAD is rotated 180 degrees around Z so the camera/beak face the +X direction.
Left CAD parts map to positive Y. Hip, knee and ankle landmarks determine link
lengths and the nominal bent stance; this is not just the original box robot
with its head replaced.

This remains an explicitly assumed mechanical assembly: the CAD contains an
appearance design, rather than an engineering joint/actuator specification.
Neck/head and ankles are fixed. Cables are rigid visual parts. Contact uses
box proxies fitted to the chassis, head, legs and soles; detailed visual meshes
do not participate in collision. Self-collision is disabled. Total mass is an
estimated 0.500 kg, with provisional box inertias and servo limits. The reflected
rotor inertia of 0.0001 kg m² per joint is a simulation assumption, not a measured
motor specification. `docs/futuristic_pd_diagnostic.json` records why explicit
500 Hz PD requires this assumption: otherwise an airborne small-target test
oscillates and saturates. Hardware transfer has not been tested.

## Rebuild and train

Run these from the repository directory. FreeCAD is needed only for re-export;
exported meshes are already included.

```sh
flatpak run --command=FreeCADCmd org.freecad.FreeCAD "$PWD/scripts/export_futuristic_cad.py" > /tmp/microduck-cad-export.log 2>&1
scripts/run python scripts/build_futuristic_robot.py
scripts/run biped validate-model --robot configs/robot/futuristic_residual.yaml
scripts/run biped train --config configs/experiments/futuristic_long.yaml
```

GPU training uses MuJoCo Warp and PyTorch CUDA. A shared PPO policy controls
1024 independent worlds. Checkpoints, raw rewards, episode errors, falls,
resource usage and configurations are retained in `runs/`. New runs use a UUID;
older results and the initial twelve-policy study are retained separately.

The gait-guided task adds sine/cosine of a 2 Hz phase to the original 27 inputs.
It rewards alternating foot clearance and contact in addition to velocity,
heading and posture. The final residual variant uses a documented inverse-kinematics stepping prior
and PPO learns six corrections. This prior is a controller, not a learned
result; MuJoCo dynamics determine the resulting movement. Earlier unconstrained
and phase-reward runs are preserved as exploratory failures. Its commands include negative forward speed,
positive/negative turns, and combinations. This reward revision is separate from
the initial factorial study and must not be presented as a matched comparison.

## Keyboard control

On a graphical desktop:

```sh
scripts/run biped play --checkpoint artifacts/futuristic_policy/policy.pt
```

Hold **W** to request +0.05 m/s and **S** for -0.05 m/s. Hold **A** or **D** to
request ±0.25 rad/s turning; combine either with W/S. Release keys or press
**Space** to request zero velocity. **R** resets after a fall; **Esc** closes.
The viewer performs actual CPU MuJoCo simulation and inference from the trained
policy. Key commands request movement; actual tracking depends on the measured
policy results. A fall is displayed rather than hidden by automatic resets.

A repeatable transition diagnostic can run without a display:

```sh
scripts/run biped play --checkpoint artifacts/futuristic_policy/policy.pt --headless --seconds 30 --output results/futuristic_keyboard.json
```

Constant-command validation and test suites are frozen in
`configs/evaluation/futuristic_walking_{validation,test}.yaml`. Each suite has three
independent initial-condition seeds, four replicas, and seven commands: stand,
forward, backward, left, right, forward-left, forward-right. The step-evidence check additionally requires at least two contact-free lifts
above 3 mm per foot on translating commands. Training seeds are
not confused with these episode seeds. Acceptance uses 10-second episodes,
no fall, at least 95% survival, planar speed RMSE ≤0.025 m/s, and yaw RMSE
≤0.1 rad/s. Validation selects checkpoints; the test suite is used after
selection. A single training seed is not a population reliability estimate.

The newest measured result and checkpoint hashes are in
`results/futuristic_results.json`. The paper's futuristic section is generated
from these artifacts. Training completion alone does not mean all commands pass.

## Long GPU run and finalization

The long profile allows 20,000 updates and 491,520,000 transitions, with a
four-hour wall-time ceiling. The current run continues saved learning state;
its manifest and checkpoint record the exact budgets and parents. Physics resets
on resume, as in the original pipeline. The final report follows warm-start and
resume ancestry without double-counting transitions.

`results/futuristic_pipeline_status.json` records live GPU samples and progress.
After the full budget, the finalizer freezes validation selection, evaluates the
untouched test suite, exports the policy, records CPU keyboard playback, renders
all command cases and regenerates the paper. It writes a failure status if any
stage fails. `pipeline_completed` and `walking_gate_passed` are distinct outcomes.

Run GPU telemetry is retained in `runs/<id>/gpu_samples.jsonl`; device utilization
includes the desktop and concurrent validation, while compute-process listing
confirms the training worker on CUDA. Source snapshots and hashes are retained
with the long run. No service upload or hardware deployment is performed.

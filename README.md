# Microduck biped PPO

Provisional six-joint robot locomotion using **mjlab / MuJoCo Warp / RSL-RL**.
This is a simulation research project. The original CAD, appearance redesign,
URDF and learner draft remain intact in the parent directory. The training
robot uses the supplied URDF, not an invented mechanism extracted from CAD.
All physical parameters remain estimates until hardware specifications arrive.

## First successful run

Ubuntu 24.04, Python 3.12, recent NVIDIA driver. The project creates its own
`.venv`; it does not alter existing CUDA/PyTorch/ROS installations.

```sh
cd /home/prashanna/Documents/AR_DESIGN/biped-ppo
uv sync --locked
uv run biped doctor
uv run biped validate-model --robot configs/robot/custom.yaml
uv run pytest -m 'not gpu'
uv run pytest -m gpu
uv run biped benchmark --num-envs 32 64 128 256
uv run biped train --config configs/experiments/smoke.yaml --seed 0 --max-wall-time 180
```

If your shell sources ROS, use `scripts/run biped ...` and `scripts/run pytest ...`
in place of `uv run ...`. The wrapper clears inherited PYTHONPATH/VIRTUAL_ENV
for that command only, preventing unrelated ROS test plugins from loading.
Your ROS installation and shell configuration are unchanged.

The lock uses PyTorch 2.10.0 CUDA 12.8 wheels. PyTorch's bundled runtime does
not replace the system CUDA toolkit. CPU diagnostics require no NVIDIA GPU;
the default lock still installs CUDA-capable PyTorch. GPU tests report skips
when CUDA is unavailable. `MUJOCO_GL=egl` supports offscreen CPU-model renders
on the inspected laptop; rendering is optional infrastructure, not a training input.

## Commands

```sh
uv run biped view --robot configs/robot/custom.yaml
uv run biped train --config configs/experiments/pilot.yaml --seed 0 --max-wall-time 1200
uv run biped train --resume runs/<run-id>/last.pt --max-wall-time 120
uv run biped evaluate --checkpoint runs/<run-id>/final.pt --suite configs/evaluation/validation.yaml
uv run biped export --checkpoint runs/<run-id>/final.pt --format onnx
uv run biped analyze --runs runs
uv run biped paper --results results/aggregates
uv run biped sweep --config configs/experiments/ablation-matrix.yaml --max-wall-time 14400
uv run biped train --config configs/experiments/full.yaml --seed 0
```

Use the bounded pilot before research runs. Training checks wall-time at update
boundaries including initialization; compilation/current update can overrun the
boundary. Max aggregate transitions is also enforced. Ctrl-C saves learning
state at the next update boundary. An interrupted partial rollout is not an
exact simulator checkpoint. Resume requires identical batch/task/seed settings;
use a new experiment for changes. Runs default to ignored `runs/`, configurable
with `BIPED_RUNS`. View TensorBoard with `.venv/bin/tensorboard --logdir runs`.

Each timestamp/UUID run contains immutable manifest, lock, robot specification,
JSONL metrics/episodes, TensorBoard, architecture, periodic/last/final checkpoints
and status. Compact manifests are retained in `experiments/manifests`. Simulation
performance benchmarks exclude PPO; per-update training throughput includes it.
PPO uses one shared actor/critic across N independent worlds. Separate seeds
produce separate policies and are run sequentially.

27 observation fields: 3 body angular velocity, 3 projected gravity, 6 relative
joint positions, 6 joint velocities, 6 previous bounded actions, 3 commands.
Joint order and SI conventions are in `configs/robot/custom.yaml`. PD control
runs at 500 Hz, policy at 50 Hz. The Gaussian latent is transformed with tanh
inside the environment; likelihoods are computed for the original latent.
True falls do not bootstrap; time limits use the post-action terminal value.

The report is explicitly preliminary, with absent comparisons marked **NOT
MEASURED**. Numerical paper values come from artifacts and a result manifest.
This repository does not claim a full research matrix, hardware transfer,
calibrated servo dynamics, or fabrication-ready CAD.

See `docs/model_audit.json`, `docs/experiment_protocol.md`,
`docs/design_decisions.md`, `docs/ROADMAP.md`, and `docs/execution_evidence.md`
for current measured status and remaining work. Source CAD/URDF distribution
rights need confirmation before public release; see `THIRD_PARTY.md`.

"""Verify the physical sensor API emits the simulator's identical motor targets."""

import argparse
import numpy as np
import mujoco
from biped_ppo.config import ROOT
from biped_ppo.runtime import PositionPolicyRuntime
from biped_ppo.play import PolicySimulation, keyboard_command
from biped_ppo.tracking import write_json, sha

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()
sim = PolicySimulation(args.checkpoint)
runtime = PositionPolicyRuntime(args.checkpoint)
errors = []
for step in range(100):
    keys = {"W"} if step < 25 else {"S"} if step < 50 else {"A"} if step < 75 else {"D"}
    command = keyboard_command(keys)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, sim.root, velocity, 1)
    gravity = sim.data.xmat[sim.root].reshape(3, 3).T @ np.array([0, 0, -1])
    target = runtime.step(
        sim.data.qpos[sim.qids], sim.data.qvel[sim.vids], velocity[:3], gravity, command
    )
    sim.step(command)
    errors.append(float(np.max(np.abs(target - sim.target))))
result = dict(
    checkpoint=args.checkpoint,
    checkpoint_sha=sha(args.checkpoint),
    steps=100,
    maximum_target_error_rad=max(errors),
    passed=max(errors) < 1e-6,
    hardware_io_performed=False,
)
write_json(ROOT / "docs/futuristic_runtime_parity.json", result)
assert result["passed"], result
print(result)

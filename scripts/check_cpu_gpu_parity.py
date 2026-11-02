import argparse
import json
import torch
import numpy as np
from biped_ppo.play import PolicySimulation
from biped_ppo.config import RunConfig, ROOT
from biped_ppo.task import make_cfg
from pathlib import Path
from mjlab.envs import ManagerBasedRlEnv
from biped_ppo.tracking import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
checkpoint = args.checkpoint
s = PolicySimulation(checkpoint)
cfg = make_cfg(RunConfig(robot_path=s.cfg.robot_path, task_path=s.cfg.task_path, num_envs=4))
cfg.events.pop("reset_base")
cfg.events.pop("reset_joints")
e = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
e.reset()
command = [0.05, 0, 0]
term = e.command_manager.get_term("twist")
term.cfg.resampling_time_range = (100.0, 100.0)
term.vel_command_b[:] = torch.tensor(command, device="cuda:0")
term.is_standing_env[:] = False
errors = []
for i in range(50):
    obs = e.observation_manager.compute(update_history=(i == 0))["actor"][0].cpu().numpy()
    cpu = s.observation(command)
    errors.append(float(np.max(np.abs(obs - cpu))))
    with torch.no_grad():
        from tensordict import TensorDict

        o = torch.from_numpy(cpu).unsqueeze(0)
        latent = s.runner.alg.actor(TensorDict({"actor": o, "critic": o}, batch_size=[1]))
    e.step(latent.to("cuda:0").repeat(4, 1))
    s.step(command)
result = dict(
    steps=50,
    seconds=1.0,
    max_observation_error=max(errors),
    final_error=errors[-1],
    errors=errors,
    checkpoint=str(checkpoint),
    interpretation="Same policy and action targets; CPU/GPU contact solver numerical differences are reported, not hidden",
)
write_json(ROOT / "docs/futuristic_cpu_gpu_parity.json", result)
print("PARITY", json.dumps(result))
e.close()

"""Command sampling explicitly covers keyboard actions and reversals."""

from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class KeyboardVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        mode = torch.rand(len(env_ids), device=self.device)
        # 30% exact straight forward, 30% exact straight backward, 20% pivot,
        # 20% combined commands. Independent standing draws remain applicable.
        forward = env_ids[mode < 0.3]
        backward = env_ids[(mode >= 0.3) & (mode < 0.6)]
        pivot = env_ids[(mode >= 0.6) & (mode < 0.8)]
        self.vel_command_b[forward, 0] = self.vel_command_b[forward, 0].abs()
        self.vel_command_b[backward, 0] = -self.vel_command_b[backward, 0].abs()
        self.vel_command_b[torch.cat((forward, backward)), 2] = 0
        self.vel_command_b[pivot, 0] = 0


@dataclass(kw_only=True)
class KeyboardVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return KeyboardVelocityCommand(self, env)

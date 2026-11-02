"""Shared, stateful motor-target controller for GPU training and CPU/deployment.

Inputs and outputs are SI units. This emits joint position targets; the physical
servo dynamics still need calibration for the actual actuators.
"""

import math
import torch
from .gait import reference_targets


class MotorTargetController:
    def __init__(self, nominal, scale, lower, upper, task, dt):
        self.nominal = nominal.clone()
        self.scale = torch.as_tensor(scale, device=nominal.device, dtype=nominal.dtype)
        self.lower = torch.as_tensor(lower, device=nominal.device, dtype=nominal.dtype)
        self.upper = torch.as_tensor(upper, device=nominal.device, dtype=nominal.dtype)
        self.task = task
        self.dt = dt
        self.acceleration = torch.tensor(task["command_acceleration"], device=nominal.device)
        self.speed_limit = torch.tensor(task["target_speed_limits"], device=nominal.device)
        self.alpha = 1 - math.exp(-dt / task["target_filter_seconds"])
        self.command = torch.zeros(nominal.shape[0], 3, device=nominal.device)
        self.target = nominal.clone()
        self.velocity = torch.zeros_like(nominal)
        self.acceleration_target = torch.zeros_like(nominal)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.command[ids] = 0
        self.target[ids] = self.nominal[ids]
        self.velocity[ids] = 0
        self.acceleration_target[ids] = 0

    def advance(self, latent, requested_command, time):
        change = self.acceleration * self.dt
        self.command += torch.clamp(requested_command - self.command, min=-change, max=change)
        reference = reference_targets(
            time,
            self.command,
            self.nominal,
            frequency=self.task["gait_frequency_hz"],
            sway=self.task.get("reference_sway_m", 0.012),
            lift=self.task["foot_clearance_m"],
            soft_amplitude=True,
        )
        requested = torch.clamp(
            reference + self.scale * torch.tanh(latent), min=self.lower, max=self.upper
        )
        delta = torch.clamp(
            self.alpha * (requested - self.target),
            min=-self.speed_limit * self.dt,
            max=self.speed_limit * self.dt,
        )
        velocity = delta / self.dt
        self.acceleration_target[:] = (velocity - self.velocity) / self.dt
        self.velocity[:] = velocity
        self.target += delta
        return self.target

    def observation(self):
        return torch.cat((self.target - self.nominal, self.velocity * 0.1), dim=-1)

"""Optional explicit inverse-kinematics reference for residual PPO.

This is a control prior, not learned performance. PPO outputs residuals and the
normal MuJoCo dynamics determine whether the robot moves and stays upright.
"""

import math
import torch


def reference_targets(
    time, command, nominal, frequency=2.0, sway=0.012, lift=0.012, soft_amplitude=False
):
    phase = 2 * math.pi * frequency * time
    ramp = (time / 0.5).clamp(0, 1)
    moving = (command[:, 0].abs() > 0.005) | (command[:, 2].abs() > 0.025)
    amplitude = (
        (command[:, 0].abs() / 0.02 + command[:, 2].abs() / 0.15).clamp(0, 1)
        if soft_amplitude
        else moving
    )
    ramp = ramp * amplitude
    lateral = -sway * torch.sin(phase) * ramp
    length1 = math.hypot(0.017, 0.037)
    length2 = math.hypot(0.021, 0.044)
    targets = []
    for side in (1, -1):
        speed = command[:, 0] - side * command[:, 2] * 0.034
        x = 0.004 + side * (speed / (4 * frequency)) * torch.cos(phase) * ramp
        z = -0.081 + lift * torch.clamp(-side * torch.sin(phase), min=0) * ramp
        roll = torch.atan2(lateral, -z)
        zz = -torch.sqrt(z * z + lateral * lateral)
        cosine = ((x * x + zz * zz - length1**2 - length2**2) / (2 * length1 * length2)).clamp(
            -1, 1
        )
        knee = -torch.acos(cosine)
        hip = torch.atan2(-x, -zz) - torch.atan2(
            length2 * torch.sin(knee), length1 + length2 * torch.cos(knee)
        )
        targets.extend((roll, hip, knee))
    reference = torch.stack(targets, dim=-1)
    enabled = amplitude > 0 if soft_amplitude else moving
    return torch.where(enabled[:, None], reference, nominal)

"""Lateral reflection for the six-joint policy's documented input ordering."""

import torch


def mirror_joints(values):
    order = [3, 4, 5, 0, 1, 2]
    signs = values.new_tensor([-1, 1, 1, -1, 1, 1])
    return values[..., order] * signs


def mirror_observation(values):
    if values.shape[-1] not in (29, 41):
        raise ValueError("Mirror convention is defined for 29/41 field policies")
    result = values.clone()
    result[..., 0:3] = values[..., 0:3] * values.new_tensor([-1, 1, -1])
    result[..., 3:6] = values[..., 3:6] * values.new_tensor([1, -1, 1])
    for begin in (6, 12, 18):
        result[..., begin : begin + 6] = mirror_joints(values[..., begin : begin + 6])
    result[..., 24:27] = values[..., 24:27] * values.new_tensor([1, -1, -1])
    result[..., 27:29] = -values[..., 27:29]
    if values.shape[-1] == 41:
        for begin in (29, 35):
            result[..., begin : begin + 6] = mirror_joints(values[..., begin : begin + 6])
    return result


def mirror_batch(env=None, obs=None, actions=None):
    mirrored_obs = None
    if obs is not None:
        mirrored = obs.clone()
        for key in mirrored.keys():
            mirrored[key] = mirror_observation(obs[key])
        mirrored_obs = torch.cat((obs, mirrored), dim=0)
    mirrored_actions = (
        None if actions is None else torch.cat((actions, mirror_joints(actions)), dim=0)
    )
    return mirrored_obs, mirrored_actions

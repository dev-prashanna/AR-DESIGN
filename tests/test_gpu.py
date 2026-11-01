import pytest
import torch
from biped_ppo.config import RunConfig


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
def test_vector_reset_isolation_and_finite_observations():
    from biped_ppo.task import make_env

    env = make_env(RunConfig(num_envs=4))
    try:
        env.reset(seed=4)
        for _ in range(5):
            obs, _, _, _, _ = env.step(torch.zeros(4, 6, device="cuda:0"))
        before = env.scene["robot"].data.joint_pos[1:].clone()
        env.reset(env_ids=torch.tensor([0], device="cuda:0"))
        assert torch.equal(before, env.scene["robot"].data.joint_pos[1:])
        assert all(torch.isfinite(x).all() for x in obs.values())
        term = env.action_manager.get_term("joint_pos")
        assert term.target_names == [
            "l_hip_roll",
            "l_hip_pitch",
            "l_knee",
            "r_hip_roll",
            "r_hip_pitch",
            "r_knee",
        ]
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
def test_terminal_observation_bootstrap_before_reset():
    from biped_ppo.task import make_env
    from biped_ppo.training import TerminalAwareWrapper

    env = TerminalAwareWrapper(make_env(RunConfig(num_envs=4, episode_seconds=0.02)))
    try:
        env.critic = lambda obs: obs["critic"][:, 18:24].abs().sum(-1)
        action = torch.ones(4, 6, device="cuda:0") * 0.2
        obs, reward, done, extras = env.step(action)
        assert extras["truncated"].all() and not extras["terminated"].any()
        expected = env.last_task_reward + 0.99 * torch.tanh(action).abs().sum(-1)
        assert torch.allclose(reward, expected, atol=1e-5)
        assert done.all()
        assert torch.all(obs["actor"][:, 18:24] == 0)
        assert torch.all(extras["terminal_observation"]["actor"][:, 18:24] > 0)
    finally:
        env.close()

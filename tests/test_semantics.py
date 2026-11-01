import numpy as np
import pytest
import torch
from tensordict import TensorDict
from biped_ppo.config import RunConfig
from biped_ppo.training import timeout_reward, runner_config
from biped_ppo.tracking import rng_state, restore_rng


def test_timeout_bootstrap_uses_terminal_value_and_fall_precedence():
    r = torch.tensor([1.0, 1.0, 1.0, 1.0])
    v = torch.tensor([10.0, 20.0, 30.0, 40.0])
    trunc = torch.tensor([False, True, False, True])
    term = torch.tensor([False, False, True, True])
    assert torch.allclose(
        timeout_reward(r, v, trunc, term, 0.9), torch.tensor([1.0, 19.0, 1.0, 1.0])
    )


def test_checkpoint_restores_normalization_optimizer_and_prediction(tmp_path):
    from mjlab.rl.runner import MjlabOnPolicyRunner

    class FakeEnv:
        num_envs = 4
        num_actions = 6
        device = "cpu"
        cfg = {}

        def get_observations(self):
            return TensorDict(
                {"actor": torch.randn(4, 27), "critic": torch.randn(4, 27)}, batch_size=[4]
            )

    cfg = runner_config(RunConfig(num_envs=4))
    a = MjlabOnPolicyRunner(FakeEnv(), cfg, device="cpu")
    obs = FakeEnv().get_observations()
    a.alg.actor.update_normalization(obs)
    a.alg.critic.update_normalization(obs)
    a.alg.eval_mode()
    loss = a.alg.actor(obs).square().mean() + a.alg.critic(obs).square().mean()
    a.alg.optimizer.zero_grad()
    loss.backward()
    a.alg.optimizer.step()
    expected = a.alg.actor(obs).detach()
    p = tmp_path / "checkpoint.pt"
    torch.save(a.alg.save(), p)
    b = MjlabOnPolicyRunner(FakeEnv(), runner_config(RunConfig(num_envs=4)), device="cpu")
    b.alg.load(torch.load(p, weights_only=False), None, True)
    b.alg.eval_mode()
    assert torch.allclose(expected, b.alg.actor(obs), atol=1e-7)
    assert (
        b.alg.optimizer.state_dict()["param_groups"] == a.alg.optimizer.state_dict()["param_groups"]
    )


def test_rng_roundtrip():
    state = rng_state()
    expected = (np.random.rand(), torch.rand(3))
    restore_rng(state)
    assert np.random.rand() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])


def test_bad_batch_config_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("num_envs: 3\nhorizon: 3\n")
    with pytest.raises(ValueError, match="divisible"):
        RunConfig.load(p)


def test_tracking_reward_sign_and_units():
    from types import SimpleNamespace
    from biped_ppo.task import reward_terms

    command = torch.tensor([[0.06, 0.0, 0.0]])
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[0.06, 0.0, 0.0]]))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    assert reward_terms(env, "tracking").item() == pytest.approx(1.0)
    data.root_link_lin_vel_b.zero_()
    assert reward_terms(env, "tracking").item() < 0.01
    data.joint_vel = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
    assert reward_terms(env, "speed").item() == pytest.approx(5.0)


def test_doctor_without_nvidia_driver_tools(monkeypatch):
    from biped_ppo import diagnostics

    monkeypatch.setattr(diagnostics.shutil, "which", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = diagnostics.doctor()
    assert not result["cuda_available"]
    assert result["nvidia_smi"] is None

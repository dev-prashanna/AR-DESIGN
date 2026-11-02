import torch
import pytest
from tensordict import TensorDict
from biped_ppo.config import RunConfig, read, observation_dimension
from biped_ppo.control import MotorTargetController
from biped_ppo.gait import reference_targets
from biped_ppo.training import runner_config, transfer_policy_weights


def controller(batch=4):
    spec = read("configs/robot/futuristic_residual.yaml")
    return MotorTargetController(
        torch.tensor([spec["nominal"]] * batch),
        spec["action_scale"],
        [-0.52, -1.05, -1.57] * 2,
        [0.52, 1.05, 0] * 2,
        read("configs/task/futuristic_transfer.yaml"),
        0.02,
    )


def test_motor_command_and_target_limits_during_reversals():
    control = controller()
    for step in range(200):
        previous = control.target.clone()
        command_before = control.command.clone()
        requested = torch.tensor(
            [[0.05 if step % 40 < 20 else -0.05, 0, 0.25 if step % 30 < 15 else -0.25]] * 4
        )
        target = control.advance(torch.randn(4, 6) * 10, requested, torch.full((4,), step * 0.02))
        assert torch.isfinite(target).all()
        assert ((target - previous).abs() <= control.speed_limit * 0.02 + 1e-6).all()
        assert (
            (control.command - command_before).abs() <= control.acceleration * 0.02 + 1e-6
        ).all()
        assert (target >= control.lower).all() and (target <= control.upper).all()
    before = control.target[1:].clone()
    control.reset(torch.tensor([0]))
    assert torch.equal(before, control.target[1:])
    assert torch.equal(control.target[0], control.nominal[0])
    assert control.observation().shape == (4, 12)


def test_soft_reference_has_no_start_stop_threshold_jump():
    nominal = torch.tensor([read("configs/robot/futuristic_residual.yaml")["nominal"]])

    def target(v):
        return reference_targets(
            torch.tensor([1.125]),
            torch.tensor([[v, 0.0, 0.0]]),
            nominal,
            frequency=1.5,
            soft_amplitude=True,
        )

    assert torch.max(torch.abs(target(0.00499) - target(0.00501))) < 0.001
    assert torch.max(torch.abs(target(0.000001) - target(0.0))) < 0.0001


def test_policy_input_extension_preserves_existing_mapping():
    from mjlab.rl.runner import MjlabOnPolicyRunner

    class Env:
        num_envs = 4
        num_actions = 6
        device = "cpu"
        cfg = {}

        def __init__(self, n):
            self.n = n

        def get_observations(self):
            return TensorDict(
                {k: torch.zeros(4, self.n) for k in ("actor", "critic")}, batch_size=[4]
            )

    old = MjlabOnPolicyRunner(Env(29), runner_config(RunConfig(num_envs=4)), device="cpu")
    new = MjlabOnPolicyRunner(
        Env(41), runner_config(RunConfig(num_envs=4, init_std=0.08)), device="cpu"
    )
    old.alg.eval_mode()
    state = old.alg.save()
    transfer_policy_weights(state, new)
    new.alg.eval_mode()
    x = torch.randn(4, 29)
    extended = torch.cat((x, torch.randn(4, 12)), dim=-1)
    a = TensorDict({k: x for k in ("actor", "critic")}, batch_size=[4])
    b = TensorDict({k: extended for k in ("actor", "critic")}, batch_size=[4])
    assert torch.allclose(old.alg.actor(a), new.alg.actor(b), atol=1e-6)
    assert torch.allclose(old.alg.critic(a), new.alg.critic(b), atol=1e-6)
    assert not new.alg.optimizer.state
    assert observation_dimension("configs/task/futuristic_transfer.yaml") == 41


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_randomized_transfer_environment_and_reset_isolation():
    from biped_ppo.task import make_env

    cfg = RunConfig.load("configs/experiments/futuristic_transfer_robust.yaml")
    cfg.num_envs = 8
    env = make_env(cfg)
    try:
        env.reset(seed=71)
        obs, _, _, _, _ = env.step(torch.zeros(8, 6, device="cuda:0"))
        assert obs["actor"].shape == (8, 41)
        assert all(torch.isfinite(x).all() for x in obs.values())
        motor = env.action_manager.get_term("joint_pos").motor
        before = motor.target[1:].clone()
        env.reset(env_ids=torch.tensor([0], device="cuda:0"))
        assert torch.equal(before, motor.target[1:])
        assert (env.gyro_bias.abs() <= 0.01).all()
        assert (env.encoder_bias.abs() <= 0.01).all()
    finally:
        env.close()


def test_mirror_is_involution_and_matches_gait_reference():
    from biped_ppo.symmetry import mirror_observation, mirror_joints, mirror_batch

    x = torch.randn(7, 41)
    assert torch.equal(mirror_observation(mirror_observation(x)), x)
    a = torch.randn(7, 6)
    assert torch.equal(mirror_joints(mirror_joints(a)), a)
    obs = TensorDict({"actor": x, "critic": x}, batch_size=[7])
    mirrored, actions = mirror_batch(obs=obs, actions=a)
    assert mirrored.batch_size == torch.Size([14]) and actions.shape == (14, 6)
    spec = read("configs/robot/futuristic_residual.yaml")
    nominal = torch.tensor([spec["nominal"]] * 7)
    command = torch.randn(7, 3) * 0.03
    command[:, 1] = 0
    time = torch.linspace(1.0, 2.0, 7)
    original = reference_targets(time, command, nominal, frequency=1.5, soft_amplitude=True)
    reflected = command * torch.tensor([1, -1, -1])
    mirror = reference_targets(
        time + 1 / (2 * 1.5), reflected, nominal, frequency=1.5, soft_amplitude=True
    )
    assert torch.allclose(mirror_joints(original), mirror, atol=1e-6)

"""CAD reconstruction, keyboard semantics and per-model task isolation."""

import json
import numpy as np
import mujoco
import pytest
from biped_ppo.config import ROOT, RunConfig, read
from biped_ppo.diagnostics import initialize, physics_checks
from biped_ppo.play import keyboard_command
from biped_ppo.robot import generate
from biped_ppo.task import make_cfg


def test_futuristic_cad_surfaces_and_landmarks():
    xml, audit = generate("configs/robot/futuristic.yaml", ground=True, motors=True)
    model = mujoco.MjModel.from_xml_string(xml)
    cfg = audit["robot"]
    data = initialize(model, cfg)
    parts = json.loads((ROOT / "assets/robot/meshes/futuristic/manifest.json").read_text())["parts"]
    assert len(parts) == model.nmesh == 122
    assert model.nu == 6 and model.nq == 13
    assert model.body_mass.sum() == pytest.approx(0.5)
    # CAD coordinates rotated about Z, with 0.5 mm spawn clearance.
    assert data.xpos[model.body("l_knee_link").id] == pytest.approx(
        [-0.016, 0.034, 0.0655], abs=1e-8
    )
    assert data.xpos[model.body("r_foot").id] == pytest.approx([0.005, -0.034, 0.0215], abs=1e-8)
    assert data.site_xpos[model.site("l_foot_contact").id, 2] == pytest.approx(0.0005)
    for side in ("l", "r"):
        assert data.xmat[model.body(side + "_foot").id].reshape(3, 3) == pytest.approx(
            np.eye(3), abs=1e-8
        )
    assert physics_checks(model, cfg)["gates_passed"]


def test_futuristic_config_does_not_change_initial_study():
    baseline = make_cfg(RunConfig())
    futuristic = make_cfg(RunConfig.load("configs/experiments/futuristic.yaml"))
    assert baseline.commands["twist"].ranges.lin_vel_x == (0, 0.06)
    assert futuristic.commands["twist"].ranges.lin_vel_x == (-0.06, 0.06)
    assert baseline.robot_spec["nominal"] == [0] * 6
    assert futuristic.robot_spec["nominal"][2] < -0.8
    assert read("configs/robot/custom.yaml")["urdf"].endswith("microduck_minimal.urdf")


def test_keyboard_hold_release_stop_and_combinations():
    assert keyboard_command({"W"}) == pytest.approx([0.05, 0, 0])
    assert keyboard_command({"S"}) == pytest.approx([-0.05, 0, 0])
    assert keyboard_command({"A"}) == pytest.approx([0, 0, 0.25])
    assert keyboard_command({"D"}) == pytest.approx([0, 0, -0.25])
    assert keyboard_command({"W", "D"}) == pytest.approx([0.05, 0, -0.25])
    assert keyboard_command({"W", "S", "A", "D"}) == pytest.approx([0, 0, 0])
    assert keyboard_command({"W", "SPACE"}) == pytest.approx([0, 0, 0])
    assert keyboard_command(set()) == pytest.approx([0, 0, 0])


def test_airborne_pd_target_tracking_is_stable():
    cfg = read("configs/robot/futuristic_stable.yaml")
    xml, _ = generate("configs/robot/futuristic_stable.yaml", motors=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = initialize(model, cfg, height=0.8)
    qids = [model.joint(n).qposadr[0] for n in cfg["joint_names"]]
    vids = [model.joint(n).dofadr[0] for n in cfg["joint_names"]]
    target = data.qpos[qids].copy() + [0.02, 0.03, -0.03, -0.02, -0.03, 0.03]
    peak_speed = 0
    for _ in range(250):
        torque = (
            np.array(cfg["kp"]) * (target - data.qpos[qids]) - np.array(cfg["kd"]) * data.qvel[vids]
        )
        assert np.all(np.abs(torque) < model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        peak_speed = max(peak_speed, float(np.abs(data.qvel[vids]).max()))
    assert peak_speed < 1
    assert np.max(np.abs(data.qpos[qids] - target)) < 0.001


def test_directional_progress_rewards_requested_direction():
    import torch
    from types import SimpleNamespace
    from biped_ppo.task import reward_terms

    command = torch.tensor([[-0.05, 0.0, -0.25]])
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([[-0.05, 0.0, 0.0]]),
        root_link_ang_vel_b=torch.tensor([[0.0, 0.0, -0.25]]),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    assert reward_terms(env, "linear_progress").item() == pytest.approx(1)
    assert reward_terms(env, "yaw_progress").item() == pytest.approx(1)
    data.root_link_lin_vel_b.neg_()
    data.root_link_ang_vel_b.neg_()
    assert reward_terms(env, "linear_progress").item() == pytest.approx(-1)
    assert reward_terms(env, "yaw_progress").item() == pytest.approx(-1)


def test_reference_and_clearance_reward_use_same_swing_foot():
    import torch
    from types import SimpleNamespace
    from biped_ppo.gait import reference_targets
    from biped_ppo.task import reward_terms

    spec = read("configs/robot/futuristic_residual.yaml")
    task = read("configs/task/futuristic_walk.yaml")
    xml, _ = generate("configs/robot/futuristic_residual.yaml")
    model = mujoco.MjModel.from_xml_string(xml)
    qids = [model.joint(n).qposadr[0] for n in spec["joint_names"]]
    sites = [model.site(n).id for n in ("l_foot_contact", "r_foot_contact")]
    command = torch.tensor([[0.05, 0.0, 0.0]])
    for t in (0.625, 0.875):
        data = initialize(model, spec)
        data.qpos[qids] = reference_targets(
            torch.tensor([t]), command, torch.tensor([spec["nominal"]]), frequency=2
        )[0].numpy()
        mujoco.mj_forward(model, data)
        physics = SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(1, 3),
            site_pos_w=torch.tensor(data.site_xpos[sites][None], dtype=torch.float32),
        )
        env = SimpleNamespace(
            cfg=SimpleNamespace(task_spec=task, robot_spec=spec),
            episode_length_buf=torch.tensor([t / 0.02]),
            step_dt=0.02,
            command_manager=SimpleNamespace(get_command=lambda _: command),
            scene={"robot": SimpleNamespace(data=physics, find_sites=lambda _: ([0, 1], None))},
        )
        correct = reward_terms(env, "clearance").item()
        wrong = dict(task, left_swing_phase_sign=1)
        env.cfg.task_spec = wrong
        incorrect = reward_terms(env, "clearance").item()
        assert correct > incorrect + 0.3, (t, correct, incorrect)
